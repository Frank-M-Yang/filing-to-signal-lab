from __future__ import annotations

import pandas as pd
import pytest

from prospective_event_state_machine import (
    CONSENSUS_REVISION_READY,
    INNOVATION_READY,
    REJECTED,
    ProspectiveEventStateMachineError,
    calculate_prospective_event_state_machine,
)


def test_revision_ready_event_waits_for_future_filing() -> None:
    readiness = _readiness()
    snapshots = _consensus_snapshots()
    tables = calculate_prospective_event_state_machine(
        readiness,
        snapshots,
        _empty_actuals(),
        _empty_actual_snapshots(),
        _empty_innovations(),
        _quality(),
    )

    state = tables.event_states.iloc[0]
    assert state["current_state"] == CONSENSUS_REVISION_READY
    assert state["blocking_reason"] == "future_sec_filing_not_observed"
    assert state["next_expected_state"] == "filing_observed"
    assert tables.transitions["to_state"].tolist() == [
        "registered",
        "consensus_level_ready",
        "consensus_revision_ready",
    ]
    assert tables.report["live_trading_authorized"] is False


def test_filing_actual_and_innovation_advance_in_clock_order() -> None:
    accepted_at = pd.Timestamp("2026-09-02T20:00:00Z")
    readiness = _readiness(
        filing_observed=True,
        accepted_at=accepted_at,
        filing_observed_at=pd.Timestamp("2026-09-02T20:10:00Z"),
    )
    actuals = _empty_actuals()
    actuals.loc[0] = {
        "forward_actual_target_id": "actual-1",
        "cik": "0000000001",
        "ticker": "AAA",
        "target_period_end": pd.Timestamp("2026-06-30T00:00:00Z"),
        "accession_number": "0000000001-26-000001",
        "accepted_at": accepted_at,
        "actual_status": "direct_quarterly_fact_ready",
        "actual_revenue": 111.0,
        "actual_available_at": accepted_at,
        "source_snapshot_sha256": "actual-sha",
        "historical_backfill_admission": False,
        "return_values_loaded": False,
    }
    actual_snapshots = pd.DataFrame(
        [
            {
                "forward_actual_target_id": "actual-1",
                "fetched_at": pd.Timestamp("2026-09-02T20:11:00Z"),
                "sha256": "actual-sha",
            }
        ]
    )
    innovations = _empty_innovations()
    innovations.loc[0] = {
        "cik": "0000000001",
        "ticker": "AAA",
        "target_period_end": pd.Timestamp("2026-06-30T00:00:00Z"),
        "accession_number": "0000000001-26-000001",
        "actual_status": "direct_quarterly_fact_ready",
        "raw_innovation_ready": True,
        "revenue_innovation_pct": 0.11,
        "consensus_precedes_actual": True,
        "join_status": "raw_innovation_ready",
        "historical_backfill_admission": False,
        "return_values_loaded": False,
    }

    tables = calculate_prospective_event_state_machine(
        readiness,
        _consensus_snapshots(),
        actuals,
        actual_snapshots,
        innovations,
        _quality(),
    )

    state = tables.event_states.iloc[0]
    assert state["current_state"] == INNOVATION_READY
    assert state["next_expected_state"] == "price_window_ready"
    assert state["revenue_innovation_pct"] == pytest.approx(0.11)
    assert tables.transitions["to_state"].tolist()[-4:] == [
        "filing_observed",
        "consensus_frozen",
        "actual_ready",
        "innovation_ready",
    ]
    assert (tables.transitions["observed_at"] >= tables.transitions["effective_at"]).all()

    quarantined = calculate_prospective_event_state_machine(
        readiness,
        _consensus_snapshots(),
        actuals,
        actual_snapshots,
        innovations,
        _quality("quarantined"),
    ).event_states.iloc[0]
    assert quarantined["current_state"] == "actual_ready"
    assert quarantined["raw_innovation_computed"]
    assert not quarantined["innovation_ready"]
    assert quarantined["blocking_reason"] == ("consensus_quality_not_admitted:quarantined")


def test_missing_preacceptance_consensus_is_terminal_rejection() -> None:
    accepted_at = pd.Timestamp("2026-09-02T20:00:00Z")
    readiness = _readiness(
        filing_observed=True,
        accepted_at=accepted_at,
        filing_observed_at=pd.Timestamp("2026-09-02T20:10:00Z"),
    )
    readiness.loc[0, "consensus_snapshot_rows"] = 0
    readiness.loc[0, "consensus_capture_dates"] = 0
    readiness.loc[0, "earliest_consensus_captured_at"] = pd.NaT
    readiness.loc[0, "latest_consensus_captured_at"] = pd.NaT
    readiness.loc[0, "latest_consensus_revenue"] = float("nan")
    readiness.loc[0, "revision_ready"] = False

    tables = calculate_prospective_event_state_machine(
        readiness,
        _empty_consensus(),
        _empty_actuals(),
        _empty_actual_snapshots(),
        _empty_innovations(),
        _quality(),
    )

    state = tables.event_states.iloc[0]
    assert state["current_state"] == REJECTED
    assert state["terminal"]
    assert state["rejection_reason"] == "missing_strictly_preacceptance_consensus"


def test_previous_state_cannot_regress() -> None:
    previous = pd.DataFrame(
        [
            {
                "prospective_event_id": "event-1",
                "current_state": "innovation_ready",
                "state_rank": 7,
            }
        ]
    )
    with pytest.raises(ProspectiveEventStateMachineError, match="regressed"):
        calculate_prospective_event_state_machine(
            _readiness(),
            _consensus_snapshots(),
            _empty_actuals(),
            _empty_actual_snapshots(),
            _empty_innovations(),
            _quality(),
            previous,
        )


def _readiness(
    *,
    filing_observed: bool = False,
    accepted_at: pd.Timestamp | None = None,
    filing_observed_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prospective_event_id": "event-1",
                "registration_order": 1,
                "cik": "0000000001",
                "ticker": "AAA",
                "target_period_end": pd.Timestamp("2026-06-30T00:00:00Z"),
                "registered_at": pd.Timestamp("2026-05-01T12:00:00Z"),
                "consensus_snapshot_rows": 2,
                "consensus_capture_dates": 2,
                "earliest_consensus_captured_at": pd.Timestamp("2026-05-01T12:00:00Z"),
                "latest_consensus_captured_at": pd.Timestamp("2026-09-02T19:00:00Z"),
                "latest_consensus_revenue": 100.0,
                "revision_ready": True,
                "future_accession_number": ("0000000001-26-000001" if filing_observed else None),
                "accepted_at": accepted_at,
                "future_filing_first_observed_at": filing_observed_at,
                "future_filing_observed": filing_observed,
            }
        ]
    )


def _consensus_snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prospective_event_id": "event-1",
                "captured_at": pd.Timestamp("2026-05-01T12:00:00Z"),
                "capture_date": pd.Timestamp("2026-05-01T00:00:00Z"),
                "revenue_estimate_average": 99.0,
            },
            {
                "prospective_event_id": "event-1",
                "captured_at": pd.Timestamp("2026-09-02T19:00:00Z"),
                "capture_date": pd.Timestamp("2026-09-02T00:00:00Z"),
                "revenue_estimate_average": 100.0,
            },
        ]
    )


def _empty_consensus() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prospective_event_id",
            "captured_at",
            "capture_date",
            "revenue_estimate_average",
        ]
    )


def _empty_actuals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "forward_actual_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "accepted_at",
            "actual_status",
            "actual_revenue",
            "actual_available_at",
            "source_snapshot_sha256",
            "historical_backfill_admission",
            "return_values_loaded",
        ]
    )


def _empty_actual_snapshots() -> pd.DataFrame:
    return pd.DataFrame(columns=["forward_actual_target_id", "fetched_at", "sha256"])


def _empty_innovations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "actual_status",
            "raw_innovation_ready",
            "revenue_innovation_pct",
            "consensus_precedes_actual",
            "join_status",
            "historical_backfill_admission",
            "return_values_loaded",
        ]
    )


def _quality(status: str = "clean") -> pd.DataFrame:
    admitted = status == "clean"
    return pd.DataFrame(
        [
            {
                "prospective_event_id": "event-1",
                "quality_status": status,
                "quality_reasons": "none" if admitted else "test_quality_reason",
                "downstream_admission": admitted,
                "manual_review_required": not admitted,
                "quarantined": status == "quarantined",
            }
        ]
    )
