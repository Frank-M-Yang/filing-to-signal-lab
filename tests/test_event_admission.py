"""A metadata flag, publisher label or old authorization cannot admit a return study."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_event_admission_audit import read_liquidity_bars
from sec_data.event_admission import (
    FORECAST_CLOCKS,
    HISTORY_CHECKS,
    LIQUIDITY_COLUMNS,
    REPRESENTATIONS,
    WINDOW_CHECKS,
    audit_label_liquidity,
    build_admission_ledger,
    prior_liquidity_sessions,
)


@pytest.fixture()
def liquidity_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    dates = pd.bdate_range("2025-09-01", "2025-11-10")
    event = {
        "validation_event_id": "e1",
        "cik": "1",
        "ticker": "TEST",
        "price_target_id": "p",
        "exchange_mic": "XNAS",
        "announcement_at": "2025-11-03T21:08:00Z",
    }
    calendar = pd.DataFrame(
        {
            "session_date": dates.strftime("%Y-%m-%d"),
            "exchange_mic": "XNAS",
            "regular_open_at": (dates + pd.Timedelta(hours=9, minutes=30))
            .tz_localize("America/New_York")
            .tz_convert("UTC"),
            "regular_close_at": (dates + pd.Timedelta(hours=16))
            .tz_localize("America/New_York")
            .tz_convert("UTC"),
            "source_snapshot_sha256": "a" * 64,
        }
    )
    sessions = prior_liquidity_sessions(event, calendar)
    bars = pd.DataFrame(
        {
            "price_target_id": "p",
            "ticker": "TEST",
            "session_date": sessions.session_date,
            "close_unadjusted": 100.0,
            "volume": 10000.0,
            "source_sha256": "b" * 64,
        }
    )
    return event, sessions, bars, dict.fromkeys(HISTORY_CHECKS, True)


def test_liquidity_uses_prior_dates_and_frozen_boundary(liquidity_inputs: Any) -> None:
    event, sessions, bars, checks = liquidity_inputs
    result, audit = audit_label_liquidity(event, sessions, bars, checks)
    assert len(sessions) == 20
    assert sessions.session_date.iloc[-1] == "2025-10-31"
    assert result["median_dollar_volume_usd"] == 1_000_000
    assert result["label_liquidity_eligible"]
    assert result["first_public_certified"] is False
    assert not result["model_admission"] and not result["trading_admission"]
    assert audit.price_row_observed.all()


def test_missing_rows_are_not_silent_zeroes(liquidity_inputs: Any) -> None:
    event, sessions, bars, checks = liquidity_inputs
    bars.loc[:3, "volume"] = 0
    result, _ = audit_label_liquidity(event, sessions, bars, checks)
    assert result["print_coverage"] == 0.8
    assert result["label_liquidity_eligible"]
    result, audit = audit_label_liquidity(event, sessions, bars.iloc[1:], checks)
    assert result["print_coverage"] == 0.8  # Same prints, different evidence completeness.
    assert not result["label_liquidity_eligible"]
    assert result["status"] == "liquidity_evidence_incomplete"
    assert pd.isna(audit.volume.iloc[0])
    bars.loc[4, "volume"] = 0
    result, _ = audit_label_liquidity(event, sessions, bars, checks)
    assert result["status"] == "insufficient_print_coverage"


@pytest.mark.parametrize(
    "field,value", [("close_unadjusted", np.inf), ("volume", -1), ("volume", np.nan)]
)
def test_invalid_price_evidence_blocks(liquidity_inputs: Any, field: str, value: float) -> None:
    event, sessions, bars, checks = liquidity_inputs
    bars.loc[0, field] = value
    result, _ = audit_label_liquidity(event, sessions, bars, checks)
    assert not result["label_liquidity_eligible"]
    assert result["status"] == "liquidity_evidence_incomplete"


@pytest.mark.parametrize(
    "change", ["low_volume", "short_calendar", "missing_query", "empty_checks"]
)
def test_capacity_calendar_and_source_controls(liquidity_inputs: Any, change: str) -> None:
    event, sessions, bars, checks = liquidity_inputs
    if change == "low_volume":
        bars["volume"] = 9999
    elif change == "short_calendar":
        sessions, bars = sessions.iloc[1:], bars.iloc[1:]
    elif change == "missing_query":
        checks["issuer_price_query_interval"] = False
    else:
        checks = {}
    result, _ = audit_label_liquidity(event, sessions, bars, checks)
    assert not result["label_liquidity_eligible"]


@pytest.mark.parametrize("change", ["duplicate", "target", "ticker", "forward", "hash", "columns"])
def test_unscoped_or_ambiguous_prices_rejected(liquidity_inputs: Any, change: str) -> None:
    event, sessions, bars, checks = liquidity_inputs
    if change == "duplicate":
        bars = pd.concat([bars, bars.iloc[:1]])
    elif change == "target":
        bars.loc[0, "price_target_id"] = "other"
    elif change == "ticker":
        bars.loc[0, "ticker"] = "OTHER"
    elif change == "forward":
        bars.loc[0, "session_date"] = "2025-11-03"
    elif change == "hash":
        bars.loc[0, "source_sha256"] = "unknown"
    else:
        bars["forward_return"] = 0.5
    with pytest.raises(ValueError):
        audit_label_liquidity(event, sessions, bars, checks)


def test_reader_projects_before_materializing(tmp_path: Path, liquidity_inputs: Any) -> None:
    _, sessions, bars, _ = liquidity_inputs
    future = bars.iloc[:1].assign(session_date="2025-11-03", close_unadjusted=-9999)
    other = bars.iloc[:1].assign(price_target_id="other", close_unadjusted=-9999)
    frame = pd.concat([bars, future, other]).assign(forward_return=0.9, open_unadjusted=-9999)
    path = tmp_path / "bars.parquet"
    frame.to_parquet(path, index=False)
    loaded = read_liquidity_bars(path, "p", list(sessions.session_date))
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), bars[LIQUIDITY_COLUMNS].reset_index(drop=True)
    )


@pytest.fixture()
def ledger_inputs(liquidity_inputs: Any) -> list[Any]:
    event, sessions, bars, checks = liquidity_inputs
    ids = ["e1", "e2", "e3", "e4"]
    readiness = pd.DataFrame(
        {
            "validation_event_id": ids,
            "cik": "1",
            "target_accession_number": ["a1", "a2", "a3", "a4"],
            "legacy_gate_status_not_new_authorization": ["authorized_complete_case"] * 2
            + ["excluded_pre_event_liquidity", "excluded_liquidity_evidence_incomplete"],
        }
    )
    identities = readiness[["validation_event_id", "cik"]].assign(ticker="TEST")
    gates = identities.drop(columns="ticker").assign(
        liquidity_eligible=[True, True, False, False],
        liquidity_status=[
            "eligible",
            "eligible",
            "insufficient_print_coverage",
            "liquidity_evidence_incomplete",
        ],
    )
    financial = (
        readiness[["cik", "target_accession_number"]]
        .rename(columns={"cik": "company_id", "target_accession_number": "filing_accession"})
        .assign(**dict.fromkeys(REPRESENTATIONS, True))
    )
    review = {
        "validation_event_id": "e1",
        "cik": "1",
        "filing_accession": "a1",
        "sources": [{"source_id": "wire"}],
        "status": "publisher_evidence_reviewed_not_first_public",
        "precise_label_replay_candidates": ["wire"],
        "chronology_scenarios": [
            {
                "source_id": "wire",
                "applies_to_exact_gross_margin": True,
                "lower_inclusive": "2025-11-03T21:08:00Z",
                "upper_exclusive": "2025-11-03T21:09:00Z",
            }
        ],
    }
    target = {
        "company_id": "1",
        "filing_accession": "a1",
        **dict.fromkeys(REPRESENTATIONS, True),
        **dict.fromkeys(FORECAST_CLOCKS, "2025-10-31T20:09:22Z"),
    }
    publisher = {
        "baseline_reproduced": True,
        "minute_endpoint_signals_equal": True,
        "hyperparameters_changed": False,
        "scenarios": {
            k: {"target": {**target, "actual_available_at": stamp}}
            for k, stamp in (
                ("minute_start", "2025-11-03T21:08:00Z"),
                ("minute_end", "2025-11-03T21:08:59.999Z"),
            )
        },
    }
    liquidity, _ = audit_label_liquidity(event, sessions, bars, checks)
    state = {
        **event,
        "market_state_ready": True,
        "market_feature_available_at": "2025-10-31T20:00:00Z",
        "history_evidence_checks": checks,
    }
    window = {
        "validation_event_id": "e1",
        "window": {"announcement_at": event["announcement_at"]},
        "window_evidence_ready": True,
        "checks": dict.fromkeys(WINDOW_CHECKS, True),
    }
    return [
        readiness,
        identities,
        gates,
        financial,
        [review],
        publisher,
        event,
        liquidity,
        state,
        window,
    ]


def test_full_cohort_and_other_quarters_preserved(ledger_inputs: Any) -> None:
    ledger = build_admission_ledger(*ledger_inputs)
    assert list(ledger.validation_event_id) == ["e1", "e2", "e3", "e4"]
    assert list(ledger.label_scenario_inputs_complete) == [True, False, False, False]
    assert ledger.first_public_certified.eq(False).all()
    assert ledger.return_admission.eq(False).all()
    assert ledger.model_admission.eq(False).all()
    assert ledger.trading_admission.eq(False).all()
    assert ledger.iloc[2].legacy_liquidity_status == "insufficient_print_coverage"
    assert ledger.iloc[3].legacy_liquidity_status == "liquidity_evidence_incomplete"
    assert ledger.iloc[0].blockers == [
        "first_public_clock_not_verified",
        "return_experiment_not_frozen",
    ]


@pytest.mark.parametrize("clock", FORECAST_CLOCKS)
def test_same_time_errors_or_forecasts_block_replay(ledger_inputs: Any, clock: str) -> None:
    ledger_inputs[5]["scenarios"]["minute_start"]["target"][clock] = "2025-11-03T21:08:00Z"
    result = build_admission_ledger(*ledger_inputs)
    assert not result.iloc[0].label_financial_replay_ready
    assert not result.iloc[0].label_scenario_inputs_complete


@pytest.mark.parametrize(
    "change",
    [
        "raw_innovation_missing",
        "window_check_missing",
        "market_check_missing",
        "liquidity_incomplete",
    ],
)
def test_all_dependencies_are_required(ledger_inputs: Any, change: str) -> None:
    if change == "raw_innovation_missing":
        ledger_inputs[3].loc[0, "analysis_ready_innovation_raw"] = False
    elif change == "window_check_missing":
        ledger_inputs[9]["checks"].pop("listing_outcome_evidence_covers_new_window")
    elif change == "market_check_missing":
        ledger_inputs[8]["history_evidence_checks"] = {}
    else:
        ledger_inputs[7]["evidence_complete"] = False
    assert not build_admission_ledger(*ledger_inputs).iloc[0].label_scenario_inputs_complete


@pytest.mark.parametrize(
    "change",
    [
        "missing_event",
        "wrong_cik",
        "wrong_review",
        "duplicate_review",
        "wrong_replay",
        "wrong_liquidity_clock",
    ],
)
def test_identity_and_clock_mismatches_rejected(ledger_inputs: Any, change: str) -> None:
    values = copy.deepcopy(ledger_inputs)
    if change == "missing_event":
        values[2] = values[2].iloc[:-1]
    elif change == "wrong_cik":
        values[1].loc[0, "cik"] = "2"
    elif change == "wrong_review":
        values[4][0]["filing_accession"] = "a2"
    elif change == "duplicate_review":
        values[4].append(values[4][0])
    elif change == "wrong_replay":
        values[5]["scenarios"]["minute_start"]["target"]["filing_accession"] = "a2"
    else:
        values[7]["announcement_at"] = "2025-11-04T21:08:00Z"
    with pytest.raises(ValueError):
        build_admission_ledger(*values)
