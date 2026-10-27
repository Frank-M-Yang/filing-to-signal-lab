"""Cache evidence and clocks are not a license to backdate availability or train."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_information_data_closure import build_closure
from sec_data.information_data_closure import (
    cached_industries,
    industry_closure,
    missing_market_plan,
    receipt_audit,
)


def receipt_fixture(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Synthetic raw bytes with a late lab receipt; no empirical vendor claim."""
    path = tmp_path / "raw.json"
    path.write_bytes(b'{"results": []}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "status_code": 200,
        "fetched_at": "2026-08-28T12:00:00Z",
    }
    required = [
        {"role": "issuer_close", "source_sha256": digest, "not_before": "2025-04-04T20:00:00Z"}
    ]
    return receipt, required


def test_receipt_is_not_historical_label_availability(tmp_path: Path) -> None:
    receipt, required = receipt_fixture(tmp_path)
    result = receipt_audit(required, [receipt], [tmp_path])
    assert result["complete"] and result["verified_roles"] == 1
    assert result["lab_inputs_received_by"] == "2026-08-28T12:00:00+00:00"
    assert result["label_available_at"] is None
    assert not result["historical_delivery_certified"] and not result["source_admitted"]


def test_missing_and_repeated_receipts_stay_conservative(tmp_path: Path) -> None:
    receipt, required = receipt_fixture(tmp_path)
    missing = receipt_audit(required, [], [tmp_path])
    assert missing["missing_roles"] == ["issuer_close"]
    assert missing["lab_inputs_received_by"] is None and not missing["complete"]
    later = {**receipt, "fetched_at": "2026-09-01T12:00:00Z"}
    result = receipt_audit(required, [receipt, later], [tmp_path])
    assert result["lab_inputs_received_by"] == later["fetched_at"].replace("Z", "+00:00")
    assert result["receipts"][0]["verified_receipts"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("bytes", 0),
        ("status_code", 403),
        ("fetched_at", "2025-01-01T00:00:00Z"),
        ("fetched_at", "2026-08-28T12:00:00"),
        ("fetched_at", None),
    ],
)
def test_bad_receipts_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    receipt, required = receipt_fixture(tmp_path)
    receipt[field] = value
    with pytest.raises(ValueError):
        receipt_audit(required, [receipt], [tmp_path])


def test_hash_corruption_and_path_escape_fail_closed(tmp_path: Path) -> None:
    receipt, required = receipt_fixture(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        receipt_audit(required, [receipt], [tmp_path / "other"])
    Path(receipt["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        receipt_audit(required, [receipt], [tmp_path])


def test_receipt_roles_and_hashes_cannot_be_ambiguous(tmp_path: Path) -> None:
    receipt, required = receipt_fixture(tmp_path)
    for roles in ([], required * 2, [{"role": "close", "source_sha256": "invalid"}]):
        with pytest.raises(ValueError):
            receipt_audit(roles, [receipt], [tmp_path])


def sec_fixture(tmp_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """A minimal SEC original FILER header, kept separate from market receipts."""
    path = tmp_path / "filing.txt"
    path.write_bytes(b"""<SEC-HEADER>
ACCESSION NUMBER: 0000000001-25-000001
<ACCEPTANCE-DATETIME>20250101100000
FILER:
 CENTRAL INDEX KEY: 0000000001
 STANDARD INDUSTRIAL CLASSIFICATION: TEST [3500]
</SEC-HEADER>""")
    return {
        ("0000000001", "0000000001-25-000001"): {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "fetched_at": "2026-08-28T12:00:00Z",
            "status_code": 200,
            "requested_url": "https://www.sec.gov/example",
        }
    }


def test_cached_original_header_does_not_certify_first_public_time(tmp_path: Path) -> None:
    cache = sec_fixture(tmp_path)
    rows, rejected = cached_industries(cache, {"0000000001"})
    assert not rejected and len(rows) == 1
    assert rows[0]["industry_available_at"] == "2025-01-01T15:00:00+00:00"
    assert rows[0]["lab_fetched_at"] == "2026-08-28T12:00:00+00:00"
    assert not rows[0]["first_public_time_certified"]
    assert cached_industries(cache, {"other"}) == ([], [])


def test_bad_header_is_rejected_not_replaced_with_current_profile(tmp_path: Path) -> None:
    cache = sec_fixture(tmp_path)
    key = next(iter(cache))
    meta = cache[key]
    path = Path(meta["path"])
    path.write_bytes(path.read_bytes().replace(b"[3500]", b"unknown"))
    meta.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    rows, rejected = cached_industries(cache, {key[0]})
    assert not rows and len(rejected) == 1
    assert rejected[0]["source_sha256"] == meta["sha256"]
    meta["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        cached_industries(cache, {key[0]})


def test_sec_receipt_cannot_precede_its_header(tmp_path: Path) -> None:
    cache = sec_fixture(tmp_path)
    cache[next(iter(cache))]["fetched_at"] = "2025-01-01T14:00:00Z"
    with pytest.raises(ValueError, match="precedes"):
        cached_industries(cache, {"0000000001"})


def industry_fixture() -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """One target and two peers. Quarantine must not be relaxed to make them fit."""
    events = [
        {
            "validation_event_id": "target",
            "ticker": "TTC",
            "cik": "1",
            "announcement_at": "2025-03-06T13:00:00Z",
            "period_start": "2024-11-01",
        }
    ]
    history = pd.DataFrame(
        [
            {
                "event_id": f"event-{cik}",
                "company_id": cik,
                "actual_available_at": "2024-10-15T13:00:00Z",
                "period_end": "2024-09-30",
                "analysis_ready_level": True,
                "scope_quarantined": cik == "3",
                "gm_level": 0.2 + int(cik) / 10,
                "filing_accession": f"accession-{cik}",
                "gross_profit_fact_id": f"fact-{cik}",
                "derivation_method": "direct_gross_profit"
                if cik != "3"
                else "revenue_minus_total_cost",
            }
            for cik in ("1", "2", "3")
        ]
    )
    observations = [
        {
            "cik": cik,
            "accession_number": f"accession-{cik}",
            "sic": "3500",
            "industry_available_at": "2024-10-15T13:00:00Z",
            "source_sha256": cik * 64,
        }
        for cik in ("1", "2", "3")
    ]
    return events, history, observations


def test_industry_gap_and_quarantine_are_explained_not_imputed() -> None:
    events, history, observations = industry_fixture()
    values, ledger = industry_closure(events, history, observations)
    assert not values.iloc[0].industry_state_ready and pd.isna(values.iloc[0].lagged_industry_gm)
    assert values.iloc[0].industry_peer_count == 1
    assert ledger[0]["candidate_peer_issuers_before_scope_filter"] == 2
    assert ledger[0]["scope_quarantined_candidate_issuers"] == ["3"]
    assert ledger[0]["selected_peers"][0]["financial_event_id"] == "event-2"
    assert ledger[0]["selected_peers"][0]["header_sha256"] == ["2" * 64]
    assert not ledger[0]["financial_peer_source_admission_certified"]


def test_industry_repair_never_uses_future_header_or_return_labels() -> None:
    events, history, observations = industry_fixture()
    history.loc[history.company_id.eq("3"), "scope_quarantined"] = False
    values, ledger = industry_closure(events, history, observations)
    assert values.iloc[0].industry_state_ready and values.iloc[0].industry_peer_count == 2
    assert values.iloc[0].lagged_industry_gm == pytest.approx(0.45)
    changed = [{**events[0], "market_excess_return": 1e9}]
    second, second_ledger = industry_closure(changed, history, observations)
    pd.testing.assert_frame_equal(values, second)
    assert ledger == second_ledger
    observations[-1]["industry_available_at"] = events[0]["announcement_at"]
    rejected, _ = industry_closure(events, history, observations)
    assert not rejected.iloc[0].industry_state_ready


def test_gap_plan_distinguishes_policy_from_account_entitlement() -> None:
    audit = [
        {
            "validation_event_id": "old",
            "ticker": "TTC",
            "requested_start": "2024-09-01",
            "requested_end": "2025-03-05",
            "missing_issuer_dates": [],
            "missing_benchmark_dates": ["2024-09-01"],
            "calendar_extension_required": True,
            "action_query_interval_covered": False,
        }
    ]
    plan = missing_market_plan(audit, "2026-10-02")[0]
    assert plan["start_precedes_advertised_basic_two_year_window"]
    assert plan["cache_gap_tasks"] == [
        "benchmark_daily_bars",
        "exchange_calendar_extension",
        "historical_identity_and_action_query_extension",
    ]
    assert not plan["redownload_cached_prices_needed"]
    assert plan["actual_account_entitlement"] == "unknown_not_probed"
    assert not plan["network_request_authorized_by_this_artifact"]


@pytest.mark.parametrize(
    "flag", ["network_allowed", "model_fitting_allowed", "overwrite_originals", "trading_admission"]
)
def test_runner_contract_cannot_enable_network_or_training(tmp_path: Path, flag: str) -> None:
    config = {
        "version": "information-data-closure-v1",
        "core_tickers": ["TTC", "USNA", "UFPT"],
        "network_allowed": False,
        "model_fitting_allowed": False,
        "overwrite_originals": False,
        "trading_admission": False,
        flag: True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contract"):
        build_closure(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()
