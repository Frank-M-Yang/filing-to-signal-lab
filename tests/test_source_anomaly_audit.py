"""Synthetic split-unit invariants and no-imputation/no-halt-inference controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.audit_source_anomalies import freeze_plan
from sec_data.source_anomaly_audit import (
    classify_execution_exceptions,
    split_boundary_changes,
    split_to_cutoff_units,
)


def _quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "TEST",
            "session_date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "quote_close": [100.0, 50.0, 51.0],
            "volume": [100.0, 200.0, 220.0],
            "source_sha256": "a" * 64,
        }
    )


def _action() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "execution_date": "2025-01-03",
                "split_from": 1.0,
                "split_to": 2.0,
                "source_sha256": "b" * 64,
                "historical_adjustment_factor": 0.000001,
            }
        ]
    )


def test_forward_split_uses_local_ratio_not_vendor_cumulative_factor() -> None:
    q, a = _quotes(), _action()
    out, ledger = split_to_cutoff_units(q, a, "2025-01-07T12:00Z")
    np.testing.assert_allclose(out.quote_close, [50, 50, 51])
    np.testing.assert_allclose(out.volume, [200, 200, 220])
    np.testing.assert_allclose(out.quote_close * out.volume, q.quote_close * q.volume)
    assert q.quote_close.tolist() == [100, 50, 51]
    assert ledger[0]["event_ratio_factor"] == 0.5
    assert ledger[0]["vendor_cumulative_factor_used"] is False
    boundary = split_boundary_changes({"raw": q, "normalized": out}, a, q.session_date.tolist())
    assert boundary[0]["close_change_pct"] == -50
    assert boundary[1]["close_change_pct"] == 0
    assert all(r["not_a_strategy_return"] for r in boundary)


def test_reverse_split_and_double_adjustment_are_distinct_hypotheses() -> None:
    q, a = _quotes(), _action()
    q["quote_close"] = [1.0, 10.0, 11.0]
    q["volume"] = [1000.0, 100.0, 110.0]
    a["split_from"], a["split_to"] = 10.0, 1.0
    out, _ = split_to_cutoff_units(q, a, "2025-01-07T12:00Z")
    double, _ = split_to_cutoff_units(out, a, "2025-01-07T12:00Z")
    np.testing.assert_allclose(out.quote_close, [10, 10, 11])
    boundary = split_boundary_changes(
        {"raw": q, "normalized": out, "double": double}, a, q.session_date.tolist()
    )
    assert [r["close_change_pct"] for r in boundary] == pytest.approx([900, 0, -90])


def test_future_and_cutoff_day_splits_cannot_adjust_history() -> None:
    q, a = _quotes(), _action()
    for day in ("2025-01-07", "2030-01-01"):
        a["execution_date"] = day
        out, ledger = split_to_cutoff_units(q, a, "2025-01-07T12:00Z")
        pd.testing.assert_frame_equal(out, q)
        assert not ledger


def test_multiple_local_split_factors_compose() -> None:
    q, a = _quotes(), _action()
    extra = a.copy()
    extra["execution_date"], extra["split_from"], extra["split_to"] = "2025-01-06", 1.0, 3.0
    out, ledger = split_to_cutoff_units(q, pd.concat([a, extra]), "2025-01-07T12:00Z")
    np.testing.assert_allclose(out.quote_close, [100 / 6, 50 / 3, 51])
    assert len(ledger) == 2


@pytest.mark.parametrize("bad", ["zero", "nan", "ticker", "duplicate", "hash", "date"])
def test_bad_actions_block(bad: str) -> None:
    a = _action()
    if bad == "zero":
        a["split_to"] = 0
    elif bad == "nan":
        a["split_from"] = np.nan
    elif bad == "ticker":
        a["ticker"] = "OTHER"
    elif bad == "duplicate":
        a = pd.concat([a, a])
    elif bad == "hash":
        a["source_sha256"] = "missing"
    else:
        a["execution_date"] = "2025-1-3"
    with pytest.raises(ValueError):
        split_to_cutoff_units(_quotes(), a, "2025-01-07T12:00Z")


@pytest.mark.parametrize("bad", ["future", "extra", "duplicate", "negative", "nan"])
def test_bad_quotes_cannot_enter_unit_conversion(bad: str) -> None:
    q = _quotes()
    if bad == "future":
        q.loc[2, "session_date"] = "2025-01-07"
    elif bad == "extra":
        q["future_return"] = 1
    elif bad == "duplicate":
        q = pd.concat([q, q])
    elif bad == "negative":
        q.loc[0, "quote_close"] = -1
    else:
        q.loc[0, "volume"] = np.nan
    with pytest.raises(ValueError):
        split_to_cutoff_units(q, _action(), "2025-01-07T12:00Z")


def test_missing_exact_split_endpoint_not_next_available_date() -> None:
    q = _quotes()
    rows = split_boundary_changes(
        {"missing_split_day": q.drop(index=1)}, _action(), q.session_date.tolist()
    )
    assert rows[0]["close_change_pct"] is None
    assert not rows[0]["exact_endpoints_present"]


def test_missing_prices_do_not_prove_halt_or_zero_payout() -> None:
    e = pd.DataFrame(
        [
            {
                "validation_event_id": str(n),
                "cik": str(n),
                "ticker": str(n),
                "formal_entry_session_date": "2025-01-02",
                "formal_exit_session_date": "2025-01-31",
            }
            for n in range(3)
        ]
    )
    listing = pd.DataFrame(
        [
            {
                "cik": "0",
                "ticker_at_event": "0",
                "delisting_date": "2025-01-15",
                "listing_outcome_status": "inactive_with_date",
                "terminal_value_observed": False,
            },
            {
                "cik": "1",
                "ticker_at_event": "1",
                "delisting_date": "2026-01-15",
                "listing_outcome_status": "later",
                "terminal_value_observed": False,
            },
        ]
    )
    result = classify_execution_exceptions(e, listing)
    assert len(result) == 3 and result.event_retained.all()
    assert result.dated_delisting_inside_window.tolist() == [True, False, False]
    assert result.terminal_value_required_but_unobserved.tolist() == [True, False, False]
    assert result.later_delisting_not_explanation_for_prior_gap.tolist() == [False, True, False]
    assert result.halt_status.eq("unknown_no_official_halt_evidence").all()
    assert not result.zero_return_imputed.any() and not result.price_imputed.any()
    assert not result.model_admission.any()
    listing.loc[0, "terminal_value_observed"] = True
    assert not classify_execution_exceptions(
        e, listing
    ).terminal_value_required_but_unobserved.any()


@pytest.mark.parametrize(
    "key,value",
    [
        ("scenario_winner_selection", True),
        ("forward_price_values_allowed", True),
        ("official_halt_evidence_available", True),
        ("sample_replacement", True),
        ("model_admission", True),
        ("required_sessions", 20),
    ],
)
def test_contract_rejects_weakening_before_data_reads(tmp_path: Path, key: str, value: Any) -> None:
    c = json.loads(Path("docs/source_anomaly_audit_v1.json").read_bytes())
    c[key] = value
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        freeze_plan(p, tmp_path / "output")
    assert not (tmp_path / "output").exists()
