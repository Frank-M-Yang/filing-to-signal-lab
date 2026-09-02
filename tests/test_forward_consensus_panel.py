"""Tests for the forward-only analyst-consensus state and revision panel."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.forward_consensus_panel import (
    ForwardConsensusPanelConfig,
    ForwardConsensusPanelError,
    calculate_forward_consensus_panel,
)


def test_one_capture_builds_watchlist_but_keeps_revision_gate_closed() -> None:
    result = calculate_forward_consensus_panel(_observations(1), _coverage())

    assert result.report["forward_snapshot_rows"] == 2
    assert result.report["next_events"] == 2
    assert result.report["event_watchlist_gate_passed"] is True
    assert result.report["capture_dates"] == 1
    assert result.report["revision_rows"] == 0
    assert result.report["revision_research_gate_passed"] is False
    assert result.report["innovation_research_gate_passed"] is False
    assert result.snapshots["actual_revenue"].isna().all()
    assert (
        result.next_events["future_actual_join_rule"]
        .str.contains("strictly_before_sec_accepted_at")
        .all()
    )


def test_two_capture_dates_produce_strictly_prior_revision_deltas() -> None:
    result = calculate_forward_consensus_panel(
        _observations(2),
        _coverage(),
        ForwardConsensusPanelConfig(
            minimum_capture_dates=2,
            minimum_revision_issuers=2,
        ),
    )

    assert result.report["capture_dates"] == 2
    assert result.report["revision_rows"] == 2
    assert result.report["revision_issuers"] == 2
    assert result.report["revision_research_gate_passed"] is True
    assert result.revisions["strictly_prior_snapshot"].all()
    deltas = result.revisions.set_index("ticker")["consensus_revision_delta"]
    assert deltas["AAA"] == pytest.approx(5.0)
    assert deltas["BBB"] == pytest.approx(-5.0)
    assert result.report["return_values_loaded"] is False


def test_historical_backfill_admission_is_rejected() -> None:
    observations = _observations(1)
    observations.loc[0, "historical_backtest_admission"] = True

    with pytest.raises(ForwardConsensusPanelError, match="Historical backfill"):
        calculate_forward_consensus_panel(observations, _coverage())


def test_forward_flag_must_match_capture_clock() -> None:
    observations = _observations(1)
    observations.loc[0, "forward_event_candidate"] = False

    with pytest.raises(ForwardConsensusPanelError, match="Forward-period flag"):
        calculate_forward_consensus_panel(observations, _coverage())


def _observations(capture_dates: int) -> pd.DataFrame:
    rows = []
    values = {"AAA": [100.0, 105.0], "BBB": [200.0, 195.0]}
    for day_index in range(capture_dates):
        captured_at = f"2026-08-{29 + day_index:02d}T12:00:00+00:00"
        for ticker, estimates in values.items():
            average = estimates[day_index]
            rows.append(
                {
                    "ticker": ticker,
                    "target_period_end": "2026-09-30",
                    "revenue_estimate_average": average,
                    "revenue_estimate_high": average * 1.05,
                    "revenue_estimate_low": average * 0.95,
                    "revenue_estimate_analyst_count": 10.0,
                    "captured_at": captured_at,
                    "availability_clock": "collector_fetched_at",
                    "forward_only_admission": True,
                    "historical_backtest_admission": False,
                    "forward_event_candidate": True,
                    "source_sha256": f"hash-{ticker}-{day_index}",
                }
            )
    return pd.DataFrame(rows)


def _coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "live_consensus_target_id": "target-aaa",
                "cik": "0000000001",
                "ticker": "AAA",
                "strict_forward_revenue_consensus_observed": True,
            },
            {
                "live_consensus_target_id": "target-bbb",
                "cik": "0000000002",
                "ticker": "BBB",
                "strict_forward_revenue_consensus_observed": True,
            },
        ]
    )
