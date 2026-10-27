"""Preserve raw witnesses, exact share-class identity and the common research gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.audit_publication_market_capture import validate_contract
from scripts.build_sec_clock_layer import _encode
from sec_data.massive_security_actions import (
    _dividend_columns,
    _identity_columns,
    _split_columns,
    _ticker_event_columns,
)
from sec_data.next_publication import DISARM_FLAGS, PRICE_COLUMNS
from sec_data.publication_market_capture import (
    action_audit,
    history_access_sensitivity,
    overlay_price_dates,
    parse_query,
)


def request(kind: str) -> dict[str, Any]:
    """Synthetic source identity; no API access or real credentials."""
    return {
        "query_id": kind,
        "query_kind": kind,
        "ticker": "TEST",
        "cik": "0000000001",
        "price_target_id": "price",
        "window_start": "2025-03-05",
        "window_end": "2025-04-01",
        "selected_composite_figi": "BBG000000002",
        "listing_outcome_cutoff": "2026-02-01",
    }


def parse(kind: str, data: dict[str, Any]) -> tuple[Any, ...]:
    raw = _encode(data)
    return parse_query(request(kind), raw, hashlib.sha256(raw).hexdigest(), "actions")


def reference_data() -> dict[str, Any]:
    """One selected common share class with explicit active state and FIGIs."""
    return {
        "status": "OK",
        "results": [
            {
                "ticker": "TEST",
                "cik": "0000000001",
                "primary_exchange": "XNAS",
                "type": "CS",
                "share_class_figi": "BBG000000001",
                "composite_figi": "BBG000000002",
                "name": "Synthetic",
                "active": True,
            }
        ],
    }


@pytest.mark.parametrize("kind", ["reference_start", "reference_end", "listing_active"])
def test_reference_rows_keep_query_date_cik_and_original_source(kind: str) -> None:
    status, rows, rejected = parse(kind, reference_data())
    assert status["query_complete"] and len(rows) == 1 and not rejected
    assert rows[0]["action_target_id"] == "actions" and rows[0]["query_kind"] == kind
    assert (
        rows[0]["as_of_date"]
        == request(kind)[
            "listing_outcome_cutoff"
            if kind == "listing_active"
            else "window_start"
            if kind == "reference_start"
            else "window_end"
        ]
    )
    assert all(status[k] is False for k in DISARM_FLAGS)


@pytest.mark.parametrize("kind", ["splits", "dividends", "reference_start", "listing_active"])
def test_successful_empty_queries_are_observations_not_ready_securities(kind: str) -> None:
    status, rows, rejected = parse(kind, {"status": "OK", "results": []})
    assert status["query_complete"] and not rows and not rejected
    assert not status["return_admission"]


@pytest.mark.parametrize(
    ("mutation", "reason"), [("status", "bad_status"), ("pagination", "pagination_incomplete")]
)
def test_incomplete_envelopes_cannot_authorize_actions(mutation: str, reason: str) -> None:
    data = {"status": "OK", "results": []}
    data["status" if mutation == "status" else "next_url"] = (
        "ERROR" if mutation == "status" else "https://api.massive.com/next"
    )
    status, rows, _ = parse("splits", data)
    assert not status["query_complete"] and status["parser_error"] == reason and not rows


def test_timeline_requires_the_exact_original_composite_figi() -> None:
    data = {"status": "OK", "results": {"composite_figi": "BBG000000003", "events": []}}
    status, rows, _ = parse("ticker_events", data)
    assert status["parser_error"] == "timeline_identity_mismatch" and not rows
    data["results"]["composite_figi"] = "BBG000000002"  # type: ignore[index]
    assert parse("ticker_events", data)[0]["query_complete"]


def test_raw_hash_is_verified_before_parser_and_errors_do_not_echo_content() -> None:
    with pytest.raises(ValueError, match="hash changed"):
        parse_query(request("splits"), b"not-the-preserved-bytes", "a" * 64, "actions")
    raw = b"synthetic-invalid-response"
    status, _, _ = parse_query(request("splits"), raw, hashlib.sha256(raw).hexdigest(), "actions")
    assert status["parser_error"] == "malformed_json"
    assert "synthetic-invalid-response" not in json.dumps(status)


def test_daily_payload_validates_ohlc_but_only_date_metadata_leaves_parser() -> None:
    millis = int(pd.Timestamp("2025-03-05T05:00:00Z").timestamp() * 1000)
    data = {
        "status": "OK",
        "ticker": "TEST",
        "adjusted": False,
        "results": [{"t": millis, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}],
    }
    status, rows, bad = parse("daily_bars", data)
    assert status["query_complete"] and not bad and len(rows) == 1
    assert list(rows[0]) == PRICE_COLUMNS and rows[0]["session_date"] == "2025-03-05"
    data["adjusted"] = True
    assert parse("daily_bars", data)[0]["parser_error"] == "adjustment_mismatch"


@pytest.fixture()
def action_fixture() -> tuple[Any, ...]:
    """Both endpoints plus a fully captured timeline; empty split and dividend evidence."""
    target = {
        "action_target_id": "actions",
        "selected_ticker": "TEST",
        "cik": "0000000001",
        "selected_exchange_mic": "XNAS",
        "selected_provider_security_id": "BBG000000001",
        "selected_share_class_figi": "BBG000000001",
        "selected_composite_figi": "BBG000000002",
        "price_window_start": "2025-03-05",
        "price_window_end": "2025-04-01",
    }
    statuses, identities = [], []
    for kind in ("reference_start", "reference_end"):
        s, rows, _ = parse(kind, reference_data())
        statuses.append(s)
        identities.extend(rows)
    for kind in ("splits", "dividends"):
        statuses.append(parse(kind, {"status": "OK", "results": []})[0])
    statuses.append(
        parse(
            "ticker_events",
            {"status": "OK", "results": {"composite_figi": "BBG000000002", "events": []}},
        )[0]
    )
    return (
        pd.DataFrame([target]),
        pd.DataFrame(identities, columns=_identity_columns()),
        pd.DataFrame(columns=_ticker_event_columns()),
        pd.DataFrame(columns=_split_columns()),
        pd.DataFrame(columns=_dividend_columns()),
        statuses,
        pd.DataFrame(
            {
                "exchange_mic": "XNAS",
                "session_date": pd.bdate_range("2025-03-05", "2025-04-01").strftime("%Y-%m-%d"),
            }
        ),
    )


def test_complete_empty_actions_and_exact_timeline_close_only_metadata(
    action_fixture: tuple[Any, ...],
) -> None:
    result = action_audit(*action_fixture).iloc[0]
    assert result.historical_symbol_interval_ready and result.corporate_actions_ready
    assert result.observed_split_count == result.observed_dividend_count == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_composite",
        "duplicate_endpoint",
        "wrong_share",
        "wrong_cik",
        "wrong_date",
        "timeline_rejection",
        "in_window_symbol_change",
    ],
)
def test_identity_is_not_relaxed_to_force_otc_or_conflicting_cases_ready(
    action_fixture: tuple[Any, ...], mutation: str
) -> None:
    args = list(action_fixture)
    if mutation == "missing_composite":
        args[0]["selected_composite_figi"] = None
        args[1]["composite_figi"] = None
    elif mutation == "duplicate_endpoint":
        args[1] = pd.concat([args[1], args[1].iloc[:1]])
    elif mutation == "wrong_share":
        args[1]["share_class_figi"] = "BBG000000003"
    elif mutation == "wrong_cik":
        args[1]["cik"] = "0000000002"
    elif mutation == "wrong_date":
        args[1].loc[0, "as_of_date"] = "2025-03-06"
    elif mutation == "timeline_rejection":
        next(s for s in args[5] if s["query_kind"] == "ticker_events")["rejected_rows"] = 1
    else:
        args[2] = pd.DataFrame(
            [{"action_target_id": "actions", "event_date": "2025-03-10", "new_ticker": "OTHER"}]
        )
    assert not action_audit(*args).iloc[0].historical_symbol_interval_ready


def test_future_ticker_event_is_not_applied_to_an_earlier_window(
    action_fixture: tuple[Any, ...],
) -> None:
    args = list(action_fixture)
    args[2] = pd.DataFrame(
        [{"action_target_id": "actions", "event_date": "2026-09-08", "new_ticker": "TEST"}]
    )
    assert action_audit(*args).iloc[0].historical_symbol_interval_ready


@pytest.mark.parametrize(
    "mutation", ["missing_query", "action_rejection", "not_an_official_session"]
)
def test_actions_are_not_imputed_as_zero_without_complete_queries(
    action_fixture: tuple[Any, ...], mutation: str
) -> None:
    args = list(action_fixture)
    if mutation == "missing_query":
        args[5] = [s for s in args[5] if s["query_kind"] != "dividends"]
    elif mutation == "action_rejection":
        next(s for s in args[5] if s["query_kind"] == "splits")["rejected_rows"] = 1
    else:
        args[3] = pd.DataFrame([{"action_target_id": "actions", "execution_date": "2025-03-08"}])
    assert not action_audit(*args).iloc[0].corporate_actions_ready


def prices(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "price_target_id": "price",
                "ticker": "TEST",
                "session_date": d,
                "source_sha256": "a" * 64,
            }
            for d in dates
        ],
        columns=PRICE_COLUMNS,
    )


def overlay_status(complete: bool = True) -> dict[str, Any]:
    return {"query_complete": complete, "rejected_rows": 0, "source_sha256": "b" * 64}


def test_new_empty_query_quarantines_overlap_loss_and_preserves_old_bytes_by_hash() -> None:
    cached = prices(["2025-03-05", "2025-03-06"])
    original = cached.copy(deep=True)
    result, audit = overlay_price_dates(cached, prices([]), request("daily_bars"), overlay_status())
    pd.testing.assert_frame_equal(result, original)
    pd.testing.assert_frame_equal(cached, original)
    assert audit["cached_dates_absent_from_new_response"] == ["2025-03-05", "2025-03-06"]
    assert audit["overlap_coverage_discrepancy"] and not audit["new_query_reauthorized"]
    assert not audit["zero_return_or_no_trade_inferred"] and audit["queries_not_assumed_identical"]


def test_additive_dates_do_not_replace_original_overlapping_provenance() -> None:
    cached = prices(["2025-03-06"])
    new = prices(["2025-03-05", "2025-03-06"]).assign(source_sha256="b" * 64)
    result, audit = overlay_price_dates(cached, new, request("daily_bars"), overlay_status())
    assert audit["new_query_reauthorized"] and audit["new_dates_added"] == 1
    assert result.loc[result.session_date.eq("2025-03-06"), "source_sha256"].iloc[0] == "a" * 64


def test_history_expiry_is_a_hypothesis_not_a_reason_to_drop_or_impute_a_case() -> None:
    result = history_access_sensitivity(
        ["2024-09-17", "2024-09-27"], "2026-08-28T03:58:05Z", "2026-10-03T09:00:00Z"
    )
    assert result["all_discrepant_dates_fit_expiry_hypothesis"]
    assert result["new_assumed_lower_date"] == "2024-10-03"
    assert not result["expiry_cause_confirmed"] and not result["actual_key_plan_verified"]
    assert all(result[k] is False for k in DISARM_FLAGS)
    assert not history_access_sensitivity(
        ["2024-11-01"], "2026-08-28T03:58:05Z", "2026-10-03T09:00:00Z"
    )["all_discrepant_dates_fit_expiry_hypothesis"]


@pytest.mark.parametrize(
    ("old", "days"), [("2026-08-28", 730), ("2026-11-01T00:00Z", 730), ("2026-08-28T00:00Z", 0)]
)
def test_history_hypothesis_requires_explicit_capture_times_and_parameters(
    old: str, days: int
) -> None:
    with pytest.raises(ValueError):
        history_access_sensitivity(["2024-09-17"], old, "2026-10-03T00:00Z", history_days=days)


@pytest.mark.parametrize("mutation", ["numeric", "wrong_target", "duplicate_date", "incomplete"])
def test_metadata_only_overlay_rejects_ambiguity_and_does_not_authorize_incomplete_queries(
    mutation: str,
) -> None:
    cached, new, status = (
        prices(["2025-03-06"]),
        prices(["2025-03-05", "2025-03-06"]),
        overlay_status(),
    )
    if mutation == "numeric":
        new["close"] = 10
    elif mutation == "wrong_target":
        new["price_target_id"] = "other"
    elif mutation == "duplicate_date":
        new = pd.concat([new, new.iloc[:1]])
    else:
        status["query_complete"] = False
        result, audit = overlay_price_dates(cached, new, request("daily_bars"), status)
        pd.testing.assert_frame_equal(result, cached)
        assert not audit["new_query_reauthorized"]
        return
    with pytest.raises(ValueError):
        overlay_price_dates(cached, new, request("daily_bars"), status)


@pytest.mark.parametrize(
    "flag",
    [
        "network_allowed",
        "forecast_refit_allowed",
        "returns_allowed",
        "overwrite_originals",
        "sample_replacement_allowed",
        "partial_sample_returns_allowed",
        "return_admission",
    ],
)
def test_contract_keeps_all_six_cases_and_return_gate_closed(flag: str) -> None:
    config = json.loads(Path("docs/publication_market_capture_audit_v1.json").read_bytes())
    validate_contract(config)
    config = copy.deepcopy(config)
    config[flag] = True
    with pytest.raises(ValueError):
        validate_contract(config)
