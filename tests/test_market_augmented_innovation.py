"""Tests for the outcome-independent market-augmented Innovation tournament."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sec_data.market_augmented_innovation import (
    MarketAugmentedInnovationConfig,
    calculate_market_augmented_innovation,
    calculate_market_augmented_return_diagnostics,
)
from sec_data.validation_market_state import MARKET_STATE_FEATURES


def test_market_correction_is_oos_and_future_actual_invariant() -> None:
    source = _market_state_fixture()
    config = _config()
    first = calculate_market_augmented_innovation(source, config)
    assert first.report["return_values_loaded"] is False
    assert first.report["selected_model_id"] != "E0_financial_baseline"
    baseline = first.forecast_metrics.set_index("model_id").loc["E0_financial_baseline", "mae"]
    selected = first.forecast_metrics.set_index("model_id").loc[
        first.report["selected_model_id"], "mae"
    ]
    assert selected < baseline

    changed = source.copy()
    changed.loc[changed.index[-1], "gm_level"] += 0.50
    rerun = calculate_market_augmented_innovation(changed, config)
    model_id = "E1_market_correction"
    pd.testing.assert_series_equal(
        first.predictions.loc[:-2, f"{model_id}_forecast"].reset_index(drop=True),
        rerun.predictions.loc[:-2, f"{model_id}_forecast"].reset_index(drop=True),
        check_names=False,
    )


def test_return_diagnostic_preserves_baseline_when_no_upgrade_is_selected() -> None:
    source = _market_state_fixture()
    config = _config()
    model = calculate_market_augmented_innovation(source, config)
    returns = pd.DataFrame(
        {
            "validation_event_id": model.predictions["validation_event_id"],
            "cik": model.predictions["cik"],
            "market_excess_return_arithmetic": np.linspace(-0.05, 0.05, len(model.predictions)),
            "market_excess_return_ready": True,
        }
    )
    result = calculate_market_augmented_return_diagnostics(
        model.predictions,
        returns,
        "E0_financial_baseline",
        config,
    )
    comparison = result.incremental_tests.set_index("test_id").loc["selected_minus_original_std_ic"]
    assert comparison["point_estimate"] == 0.0
    assert comparison["bootstrap_p_value"] == 1.0


def _config() -> MarketAugmentedInnovationConfig:
    return MarketAugmentedInnovationConfig(
        minimum_training_events=20,
        minimum_training_issuers=3,
        ridge_alpha=0.1,
        minimum_prior_oos_errors=10,
        minimum_relative_mae_improvement=0.01,
        bootstrap_repetitions=500,
        bootstrap_seed=7,
    )


def _market_state_fixture(events: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    accepted = pd.date_range("2023-01-01", periods=events, freq="7D", tz="UTC")
    market = rng.normal(size=(events, len(MARKET_STATE_FEATURES)))
    baseline_forecast = 0.40 + 0.01 * np.sin(np.arange(events) / 9.0)
    error = 0.04 * market[:, 0] - 0.02 * market[:, 3]
    frame = pd.DataFrame(
        {
            "validation_event_id": [f"event-{index:03d}" for index in range(events)],
            "cik": [f"{index % 12:010d}" for index in range(events)],
            "accepted_at": accepted,
            "actual_available_at": accepted,
            "market_feature_available_at": accepted - pd.Timedelta(days=1),
            "forecast_feature_cutoff": accepted - pd.Timedelta(days=2),
            "market_last_session_date": accepted.tz_convert(None).normalize()
            - pd.Timedelta(days=1),
            "prior_common_sessions": 61,
            "split_adjustments_applied": 0,
            "market_state_ready": True,
            "market_state_status": "ready",
            "gm_level": baseline_forecast + error,
            "gm_yoy_lag": baseline_forecast - 0.01,
            "gm_delta_yoy": error + 0.01,
            "gm_forecast": baseline_forecast,
            "gm_innovation_raw": error,
            "gm_innovation_std": error / np.std(error, ddof=1),
        }
    )
    for position, feature in enumerate(MARKET_STATE_FEATURES):
        frame[feature] = market[:, position]
    return frame
