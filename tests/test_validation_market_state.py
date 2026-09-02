"""Tests for strict pre-filing market-state features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sec_data.validation_market_state import calculate_validation_market_state


def test_market_state_is_split_adjusted_and_future_invariant() -> None:
    dates = pd.bdate_range("2025-01-02", periods=70)
    underlying = np.linspace(100.0, 120.0, len(dates))
    raw_close = underlying.copy()
    raw_close[:40] *= 2.0
    raw_volume = np.full(len(dates), 1_000.0)
    raw_volume[:40] /= 2.0
    accepted_at = (dates[-1] + pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=20)
    issuer_bars = pd.DataFrame(
        {
            "price_target_id": "price-1",
            "session_date": dates,
            "close_unadjusted": raw_close,
            "volume": raw_volume,
        }
    )
    benchmark_bars = pd.DataFrame(
        {
            "benchmark_target_id": "spy",
            "session_date": dates,
            "close_unadjusted": np.linspace(400.0, 430.0, len(dates)),
            "volume": 10_000.0,
        }
    )
    tables = calculate_validation_market_state(
        _authorized(accepted_at),
        _endpoints(),
        issuer_bars,
        benchmark_bars,
        pd.DataFrame(
            {
                "action_target_id": ["action-1"],
                "execution_date": [dates[40]],
                "historical_adjustment_factor": [0.5],
            }
        ),
        pd.DataFrame({"action_target_id": ["action-1"], "price_target_id": ["price-1"]}),
        _filing_features(accepted_at),
        _sessions(dates),
    )
    row = tables.features.iloc[0]
    assert bool(row["market_state_ready"])
    assert row["split_adjustments_applied"] == 1
    assert row["pre_price_momentum_60d"] == pytest.approx(underlying[-1] / underlying[-61] - 1.0)
    assert pd.Timestamp(row["market_feature_available_at"]) < accepted_at

    future = issuer_bars.copy()
    future.loc[len(future)] = [
        "price-1",
        accepted_at.tz_convert(None).normalize() + pd.Timedelta(days=1),
        9_999.0,
        1.0,
    ]
    rerun = calculate_validation_market_state(
        _authorized(accepted_at),
        _endpoints(),
        future,
        benchmark_bars,
        pd.DataFrame(
            {
                "action_target_id": ["action-1"],
                "execution_date": [dates[40]],
                "historical_adjustment_factor": [0.5],
            }
        ),
        pd.DataFrame({"action_target_id": ["action-1"], "price_target_id": ["price-1"]}),
        _filing_features(accepted_at),
        _sessions(dates),
    )
    assert rerun.features.loc[0, "pre_price_momentum_60d"] == pytest.approx(
        row["pre_price_momentum_60d"]
    )


def _authorized(accepted_at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_event_component_id": ["component-1"],
            "accepted_at": [accepted_at],
            "form": ["10-Q"],
        }
    )


def _endpoints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_event_component_id": ["component-1"],
            "validation_event_id": ["event-1"],
            "cik": ["0000000001"],
            "ticker": ["TEST"],
            "exchange_mic": ["XNYS"],
            "price_target_id": ["price-1"],
        }
    )


def _filing_features(accepted_at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "form": ["10-Q"],
            "actual_available_at": [accepted_at],
            "gm_level": [0.40],
            "gm_yoy_lag": [0.39],
            "gm_delta_yoy": [0.01],
            "gm_forecast": [0.395],
            "gm_innovation_raw": [0.005],
            "gm_innovation_std": [0.20],
            "forecast_feature_cutoff": [accepted_at - pd.Timedelta(days=30)],
            "forecast_training_end": [accepted_at - pd.Timedelta(days=40)],
            "forecast_training_observations": [100],
        }
    )


def _sessions(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange_mic": "XNYS",
            "session_date": dates,
            "regular_close_at": dates.tz_localize("UTC") + pd.Timedelta(hours=21),
        }
    )
