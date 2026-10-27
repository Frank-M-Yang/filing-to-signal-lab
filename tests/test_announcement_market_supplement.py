"""Frozen two-case gaps, isolated manifests and source-preserving overlap reconciliation."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.build_sec_clock_layer import _artifact
from scripts.collect_announcement_market_supplement import (
    audit_supplement_prices,
    prepare_supplement,
)
from sec_data.announcement_market_supplement import (
    PRICE_TOLERANCE,
    collection_targets,
    reconcile_overlap,
    replace_target_rows,
    validate_contract,
    verify_control_returns,
    write_typed_bundle,
)
from sec_data.disclosure_audit import digest_file
from sec_data.filing_catalog import load_catalog_table


def target_fixture() -> tuple[Any, ...]:
    """Two frozen, synthetic identities and their explicitly changed proxy windows."""
    jobs, windows, prices, actions = [], [], [], []
    for order, ticker in enumerate(["EW", "EA"], 1):
        w = {"entry_session_date": "2024-10-28", "exit_session_date": "2024-11-22"}
        job = {
            "ticker": ticker,
            "cik": str(order),
            "exchange_mic": "XNYS",
            "security_id": str(order),
            "security_id_kind": "share_class_figi",
            "price_target_id": ticker,
            "action_target_id": ticker + "-action",
            "validation_event_id": ticker,
            "announcement_at": "2024-10-24T20:28:00Z",
            "window": w,
        }
        jobs.append(job)
        windows.append({"validation_event_id": ticker, "window": copy.deepcopy(w)})
        target = {
            "price_target_id": ticker,
            "cik": str(order),
            "selected_ticker": ticker,
            "selected_exchange_mic": "XNYS",
            "security_identity_key": str(order),
            "security_identity_key_kind": "share_class_figi",
            "selection_uses_return_values": False,
            "price_window_start": "2024-11-06",
            "price_window_end": "2025-12-09",
            "adjusted": False,
        }
        prices.append(copy.deepcopy(target))
        actions.append({**target, "action_target_id": ticker + "-action"})
    return jobs, windows, pd.DataFrame(prices), pd.DataFrame(actions)


def test_collection_targets_keep_parent_identity_and_query_only_frozen_windows() -> None:
    args = target_fixture()
    original = args[2].copy(deep=True)
    prices, actions = collection_targets(*args)
    assert prices.price_target_id.tolist() == ["EW", "EA"]
    assert prices.price_window_start.eq("2024-10-28").all()
    assert actions.price_window_end.eq("2024-11-22").all()
    assert actions.session_grid_fallback_frozen_before_collection.all()
    assert not prices.adjusted.any()
    pd.testing.assert_frame_equal(original, args[2])


@pytest.mark.parametrize(
    "violation",
    [
        "extra",
        "replacement",
        "window",
        "duplicate_parent",
        "ticker",
        "security_id",
        "security_kind",
        "outcomes",
    ],
)
def test_target_plan_cannot_change_selection_or_misbind_identities(violation: str) -> None:
    args = list(target_fixture())
    if violation == "extra":
        args[0].append(copy.deepcopy(args[0][0]))
    elif violation == "replacement":
        args[0][0]["ticker"] = "OTHER"
    elif violation == "window":
        args[0][0]["window"]["entry_session_date"] = "2024-10-29"
    elif violation == "duplicate_parent":
        args[2] = pd.concat([args[2], args[2].iloc[[0]]])
    elif violation == "ticker":
        args[2].loc[0, "selected_ticker"] = "OTHER"
    elif violation == "security_id":
        args[3].loc[0, "security_identity_key"] = "wrong"
    elif violation == "security_kind":
        args[3].loc[0, "security_identity_key_kind"] = "composite_figi"
    else:
        args[3].loc[0, "selection_uses_return_values"] = True
    with pytest.raises(ValueError):
        collection_targets(*args)


def test_typed_immutable_bundle_is_compatible_with_both_readers(tmp_path: Path) -> None:
    frame = pd.DataFrame({"ticker": ["TEST"]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    files = {"rows.parquet": buffer.getvalue(), "report.json": b'{"ready":false}'}
    root = write_typed_bundle(tmp_path, "test", files, report_name="report.json")
    assert root == write_typed_bundle(tmp_path, "test", files, report_name="report.json")
    pd.testing.assert_frame_equal(load_catalog_table(root, "rows.parquet"), frame)
    assert _artifact(root, digest_file(root / "output_manifest.json"), "rows.parquet").is_file()
    (root / "rows.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_typed_bundle(tmp_path, "test", files, report_name="report.json")


@pytest.mark.parametrize("name", ["../escape", "nested/file", "output_manifest.json"])
def test_bundle_names_cannot_escape_or_replace_a_manifest(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direct artifact"):
        write_typed_bundle(tmp_path, "test", {name: b"x"})


def bars_fixture() -> pd.DataFrame:
    """Finite unadjusted OHLCV and independent raw source hashes."""
    return pd.DataFrame(
        [
            {
                "price_target_id": "EW",
                "ticker": "EW",
                "session_date": "2024-11-06",
                "open_unadjusted": 10.0,
                "high_unadjusted": 12.0,
                "low_unadjusted": 9.0,
                "close_unadjusted": 11.0,
                "volume": 100.0,
                "source_sha256": "a" * 64,
            }
        ]
    )


def test_overlap_agreement_is_not_an_overwrite_and_preserves_both_hashes() -> None:
    original = bars_fixture()
    supplement = original.copy()
    supplement["source_sha256"] = "b" * 64
    result = reconcile_overlap(original, supplement)
    assert result.all_fields_agree.all()
    assert result.original_source_sha256.eq("a" * 64).all()
    assert result.supplement_source_sha256.eq("b" * 64).all()
    assert original.source_sha256.eq("a" * 64).all()


@pytest.mark.parametrize(
    "field", ["open_unadjusted", "high_unadjusted", "low_unadjusted", "close_unadjusted", "volume"]
)
def test_overlap_revision_never_silently_replaces_an_original_price(field: str) -> None:
    original = bars_fixture()
    supplement = original.copy()
    supplement[field] += max(PRICE_TOLERANCE * 2, 0.000001)
    with pytest.raises(ValueError, match="Overlapping raw prices differ"):
        reconcile_overlap(original, supplement)


def test_duplicate_and_nonfinite_overlap_rows_fail_closed() -> None:
    original = bars_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_overlap(original, pd.concat([original, original]))
    bad = original.copy()
    bad["close_unadjusted"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        reconcile_overlap(original, bad)


def test_derived_replacement_view_retains_unrelated_targets() -> None:
    original = pd.DataFrame({"target": ["a", "b"], "value": [1, 2]})
    derived = replace_target_rows(original, pd.DataFrame({"target": ["a"], "value": [3]}), "target")
    assert derived.set_index("target").value.to_dict() == {"a": 3, "b": 2}
    assert original.value.tolist() == [1, 2]


@pytest.mark.parametrize("complete", [False, True])
def test_source_adapter_does_not_estimate_entry_or_require_legacy_query_date(
    complete: bool,
    tmp_path: Path,
) -> None:
    targets = pd.DataFrame({"price_target_id": ["EW"], "selected_ticker": ["EW"]})
    bars = bars_fixture() if complete else bars_fixture().iloc[:0]
    rejected = pd.DataFrame(columns=["price_target_id", "reason"])
    completions = pd.DataFrame({"price_target_id": ["EW"] if complete else []})
    plan = {"child_manifest_sha256": {"prices": "a" * 64}, "raw_store_scope_sha256": "b" * 64}
    with patch(
        "scripts.collect_announcement_market_supplement.load_collected_massive_daily_bars",
        return_value=(bars, rejected, completions),
    ) as load:
        root = audit_supplement_prices(targets, tmp_path / "raw", tmp_path / "out", plan)
    load.assert_called_once()
    audit = load_catalog_table(root, "sample_massive_price_audit.parquet")
    assert bool(audit.provider_price_query_complete.iloc[0]) is complete
    assert audit.daily_bar_count.iloc[0] == int(complete)
    assert audit.rejected_bar_count.iloc[0] == 0
    contract = json.loads((root / "source_contract.json").read_bytes())
    assert not contract["entry_inferred_from_observed_bars"]
    assert contract["entry_defined_by_frozen_official_calendar"]
    assert not contract["returns_computed"]


def control_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Canonical ledger fields, not invented holding-return aliases."""
    results = [
        {
            "validation_event_id": str(i),
            "cik": str(i),
            "ticker": str(i),
            "security_id": str(i),
            "price_target_id": str(i),
            "announcement_at": "2024-10-24T20:28:00Z",
            "entry_at": "2024-10-28T13:30:00Z",
            "exit_at": "2024-11-22T21:00:00Z",
            "entry_price_type": "unadjusted_open",
            "exit_price_type": "unadjusted_close",
            "issuer_total_return": 0.02,
            "benchmark_total_return": 0.01,
            "market_excess_return": 0.01,
        }
        for i in range(14)
    ]
    return copy.deepcopy(results[:6]), results


def test_control_return_comparison_uses_canonical_cash_retained_ledger_fields() -> None:
    verify_control_returns(*control_fixture())


@pytest.mark.parametrize(
    "field",
    [
        "market_excess_return",
        "issuer_total_return",
        "benchmark_total_return",
        "entry_at",
        "security_id",
        "entry_price_type",
        "duplicate",
        "missing",
    ],
)
def test_supplement_cannot_change_an_old_control_identity_window_or_return(field: str) -> None:
    controls, results = control_fixture()
    if field.endswith("return"):
        results[0][field] += 0.001
    elif field == "entry_at":
        results[0][field] = "2024-10-29T13:30:00Z"
    elif field == "duplicate":
        results[-1] = copy.deepcopy(results[0])
    elif field == "missing":
        results[0]["validation_event_id"] = "unrelated"
    else:
        results[0][field] = "changed"
    with pytest.raises(ValueError):
        verify_control_returns(controls, results)


@pytest.mark.parametrize(
    "field",
    [
        "adjusted",
        "model_refitting_allowed",
        "all_fourteen_cases_required",
        "session_identity_fallback_frozen_before_collection",
        "sample_replacement_allowed",
        "first_public_certified",
        "trading_admission",
        "inferential_tests_allowed",
        "strategy_pnl_allowed",
        "overwrite_originals",
    ],
)
def test_weakened_contract_fails_before_reading_private_sources(field: str, tmp_path: Path) -> None:
    c = json.loads(Path("docs/announcement_market_supplement_v1.json").read_bytes())
    c[field] = not c[field]
    with pytest.raises(ValueError, match="weakened"):
        validate_contract(c)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="weakened"):
        prepare_supplement(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
