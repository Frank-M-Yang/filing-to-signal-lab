from __future__ import annotations

import pandas as pd
import pytest

from prospective_market_innovation import (
    ProspectiveMarketInnovationError,
    calculate_prospective_market_innovation_preregistration,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = [f"T{index:02d}" for index in range(15)]
    coverage = pd.DataFrame(
        {
            "live_consensus_target_id": [f"target-{ticker}" for ticker in tickers],
            "cik": [f"{index + 1:010d}" for index in range(15)],
            "ticker": tickers,
            "strict_forward_revenue_consensus_observed": True,
        }
    )
    rows: list[dict[str, object]] = []
    for sequence in range(30):
        ticker = tickers[sequence % len(tickers)]
        target = pd.Timestamp("2026-09-30", tz="UTC") + pd.DateOffset(
            months=3 * (sequence // len(tickers))
        )
        rows.append(
            {
                "ticker": ticker,
                "target_period_end": target,
                "captured_at": pd.Timestamp("2026-08-29 01:00:00", tz="UTC"),
                "capture_date": pd.Timestamp("2026-08-29", tz="UTC"),
                "consensus_revenue_level": float(1_000_000_000 + sequence),
                "consensus_range_pct": 0.05,
                "revenue_estimate_analyst_count": 12.0,
                "source_sha256": f"sha-{sequence}",
                "forward_only_admission": True,
                "historical_backtest_admission": False,
                "innovation_status": "pending_future_sec_actual",
                "actual_revenue": float("nan"),
                "revenue_innovation": float("nan"),
                "revenue_innovation_pct": float("nan"),
            }
        )
    holdout = pd.DataFrame(
        {
            "validation_event_id": [f"observed-{index}" for index in range(16)],
            "cik": [f"{index + 100:010d}" for index in range(16)],
            "ticker": [f"O{index:02d}" for index in range(16)],
            "accepted_at": pd.date_range("2025-09-01", periods=16, tz="UTC"),
        }
    )
    return pd.DataFrame(rows), coverage, holdout


def test_preregistration_freezes_new_hypothesis_without_outcomes() -> None:
    watchlist, coverage, holdout = _inputs()
    tables = calculate_prospective_market_innovation_preregistration(watchlist, coverage, holdout)

    assert len(tables.event_registry) == 30
    assert tables.event_registry["cik"].nunique() == 15
    assert tables.event_registry["prospective_event_id"].is_unique
    assert not tables.event_registry["replacement_allowed"].any()
    assert not tables.event_registry["actual_value_loaded_at_registration"].any()
    assert not tables.event_registry["return_value_loaded_at_registration"].any()
    assert tables.registration_gates["passed"].all()
    assert tables.report["primary_signal_frozen"] is True
    assert tables.report["return_values_loaded"] is False
    assert set(tables.hypotheses["representation"]) == {
        "State",
        "Delta",
        "Raw Innovation",
        "Standardized Innovation",
        "Reaction",
    }
    assert not tables.observed_event_firewall["eligible_for_future_validation"].any()
    assert not tables.retired_models["eligible_for_v3"].any()


def test_preregistration_rejects_a_future_actual() -> None:
    watchlist, coverage, holdout = _inputs()
    watchlist.loc[0, "actual_revenue"] = 1_100_000_000.0

    with pytest.raises(ProspectiveMarketInnovationError, match="future value"):
        calculate_prospective_market_innovation_preregistration(watchlist, coverage, holdout)
