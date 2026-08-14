"""Tests for the longitudinal event-return ledger."""

from __future__ import annotations

import pandas as pd

from sec_data.longitudinal_event_returns import (
    LongitudinalReturnConfig,
    calculate_longitudinal_event_returns,
)


def test_longitudinal_returns_keep_delisted_issuer_and_explicit_dividend() -> None:
    horizons = pd.DataFrame(
        {
            "longitudinal_event_id": ["event-1"],
            "cik": ["0000000001"],
            "accession_number": ["accession-1"],
            "accepted_at": ["2025-01-02T21:00:00+00:00"],
            "form": ["10-Q"],
            "horizon_sessions": [20],
            "is_primary_horizon": [True],
            "entry_session_date": ["2025-01-06"],
            "entry_open_at": ["2025-01-06T14:30:00+00:00"],
            "exit_session_date": ["2025-01-31"],
            "exit_close_at": ["2025-01-31T21:00:00+00:00"],
        }
    )
    event_map = pd.DataFrame(
        {
            "longitudinal_event_id": ["event-1"],
            "price_target_id": ["price-1"],
            "action_target_id": ["action-1"],
            "representation_features_complete": [True],
        }
    )
    registry = pd.DataFrame(
        {
            "longitudinal_event_id": ["event-1"],
            "gm_level": [0.40],
            "gm_delta_yoy": [0.02],
            "gm_innovation_raw": [0.01],
            "gm_innovation_std": [1.1],
            "analysis_sample": [True],
        }
    )
    price_targets = pd.DataFrame(
        {
            "price_target_id": ["price-1"],
            "selected_ticker": ["ONE"],
            "selected_provider_security_id": ["security-1"],
            "currency_name": ["usd"],
        }
    )
    issuer_bars = _bars("price-1", 100.0, 110.0)
    action_audit = pd.DataFrame(
        {
            "action_target_id": ["action-1"],
            "historical_symbol_interval_ready": [True],
            "corporate_actions_ready": [True],
            "historical_symbol_interval_status": ["ready"],
            "corporate_action_status": ["ready"],
        }
    )
    issuer_actions = pd.DataFrame(
        {
            "action_target_id": ["action-1"],
            "effective_at": ["2025-01-15T14:30:00+00:00"],
            "action_type": ["cash_dividend"],
            "cash_amount": [1.0],
            "currency": ["USD"],
            "split_from": [float("nan")],
            "split_to": [float("nan")],
            "provider_event_id": ["dividend-1"],
            "source_snapshot_sha256": ["action-source"],
        }
    )
    benchmark_bars = _bars("benchmark", 200.0, 210.0).drop(columns="price_target_id")
    benchmark_actions = pd.DataFrame(
        columns=[
            "effective_at",
            "action_type",
            "cash_amount",
            "currency",
            "split_from",
            "split_to",
            "provider_event_id",
            "source_snapshot_sha256",
        ]
    )
    delisting = pd.DataFrame(
        {
            "cik": ["0000000001"],
            "window_terminal_outcome_ready": [True],
            "later_delisted_after_formal_exit": [True],
            "missing_return_reason": ["delisted_after_exit"],
            "source_snapshot_sha256": ["delisting-source"],
        }
    )

    result = calculate_longitudinal_event_returns(
        horizons,
        event_map,
        registry,
        price_targets,
        issuer_bars,
        action_audit,
        issuer_actions,
        benchmark_bars,
        benchmark_actions,
        delisting,
        benchmark_ready=True,
        config=LongitudinalReturnConfig(minimum_complete_events=20),
    )

    row = result.ledger.iloc[0]
    assert bool(row["issuer_return_ready"])
    assert bool(row["market_excess_return_ready"])
    assert bool(row["later_delisted_after_formal_exit"])
    assert abs(row["issuer_total_return"] - 0.11) < 1e-12
    assert abs(row["market_total_return"] - 0.05) < 1e-12
    assert abs(row["market_excess_return_arithmetic"] - 0.06) < 1e-12
    assert len(result.cashflows) == 1
    assert result.report["primary_complete_feature_return_events"] == 1
    assert result.report["primary_analysis_gate"] is False


def _bars(target_id: str, entry: float, exit_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_target_id": [target_id, target_id],
            "session_date": ["2025-01-06", "2025-01-31"],
            "open_unadjusted": [entry, exit_value - 1.0],
            "close_unadjusted": [entry + 1.0, exit_value],
            "source_sha256": ["entry-source", "exit-source"],
        }
    )
