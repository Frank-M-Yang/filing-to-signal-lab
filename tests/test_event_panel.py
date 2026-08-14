"""Temporal-contract tests for Company x Filing Event x Information Set rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from event_panel import (
    EventPanelValidationError,
    ForecastConfig,
    derive_gross_margin_representations,
    generate_expanding_gm_forecasts,
    select_common_research_sample,
    validate_filing_event_panel,
)


def _forecast_source(n_events: int = 18) -> pd.DataFrame:
    event_number = np.arange(n_events, dtype=float)
    available_at = pd.date_range("2020-01-01 21:00", periods=n_events, freq="7D", tz="UTC")
    gm_yoy_lag = 0.28 + event_number * 0.001
    gm_delta_yoy_lag = np.sin(event_number) * 0.003
    revenue_growth = 0.04 + np.cos(event_number) * 0.01
    industry_lag = 0.30 + event_number * 0.0004
    gross_margin = (
        0.03
        + 0.72 * gm_yoy_lag
        + 0.35 * gm_delta_yoy_lag
        + 0.08 * revenue_growth
        + 0.20 * industry_lag
        + np.sin(event_number * 1.7) * 0.001
    )
    return pd.DataFrame(
        {
            "actual_available_at": available_at,
            "forecast_feature_cutoff": available_at - pd.Timedelta(days=30),
            "gross_margin": gross_margin,
            "gm_yoy_lag": gm_yoy_lag,
            "gm_delta_yoy_lag": gm_delta_yoy_lag,
            "revenue_growth_lag1": revenue_growth,
            "industry_gm_lag1": industry_lag,
        }
    )


def test_expanding_forecast_and_sigma_use_strictly_prior_information() -> None:
    source = _forecast_source()
    config = ForecastConfig(min_train_observations=6, min_oos_errors=3)

    result = generate_expanding_gm_forecasts(source, config)
    forecasted = result.loc[result["gm_forecast"].notna()]
    standardized = result.loc[result["gm_innovation_std"].notna()]

    assert len(forecasted) == 12
    assert len(standardized) == 9
    assert (forecasted["forecast_training_end"] < forecasted["actual_available_at"]).all()
    assert (forecasted["forecast_generated_at"] < forecasted["actual_available_at"]).all()
    assert (standardized["forecast_error_history_end"] < standardized["actual_available_at"]).all()
    assert standardized["forecast_sigma_observations"].min() == 3


def test_future_actuals_cannot_change_past_forecasts_or_sigma() -> None:
    source = _forecast_source()
    config = ForecastConfig(min_train_observations=6, min_oos_errors=3)
    baseline = generate_expanding_gm_forecasts(source, config)
    changed = source.copy()
    changed.loc[14:, "gross_margin"] += 0.20

    rerun = generate_expanding_gm_forecasts(changed, config)

    pd.testing.assert_series_equal(
        baseline.loc[:14, "gm_forecast"], rerun.loc[:14, "gm_forecast"], check_names=False
    )
    pd.testing.assert_series_equal(
        baseline.loc[:14, "forecast_sigma"], rerun.loc[:14, "forecast_sigma"], check_names=False
    )


def test_same_timestamp_peer_actual_cannot_leak_into_prediction() -> None:
    source = _forecast_source()
    source.loc[13, "actual_available_at"] = source.loc[12, "actual_available_at"]
    config = ForecastConfig(min_train_observations=6, min_oos_errors=3)
    baseline = generate_expanding_gm_forecasts(source, config)
    changed = source.copy()
    changed.loc[12, "gross_margin"] += 0.25

    rerun = generate_expanding_gm_forecasts(changed, config)

    assert rerun.loc[13, "gm_forecast"] == pytest.approx(baseline.loc[13, "gm_forecast"])
    assert rerun.loc[13, "forecast_sigma"] == pytest.approx(baseline.loc[13, "forecast_sigma"])


def test_panel_validation_rejects_future_trained_forecast() -> None:
    source = _forecast_source()
    config = ForecastConfig(min_train_observations=6, min_oos_errors=3)
    forecasted = generate_expanding_gm_forecasts(source, config)
    row = forecasted.loc[forecasted["gm_innovation_std"].notna()].iloc[-1]
    accepted_at = pd.Timestamp(row["actual_available_at"])
    revenue = 1_000.0
    panel = pd.DataFrame(
        {
            "event_id": ["event-1"],
            "company_id": ["company-1"],
            "ticker": ["ABC"],
            "sector": ["Technology"],
            "fiscal_year": [2021],
            "fiscal_quarter": [1],
            "period_end": [pd.Timestamp("2021-03-31")],
            "filing_accepted_at": [accepted_at],
            "actual_available_at": [accepted_at],
            "executable_at": [accepted_at + pd.Timedelta(hours=16)],
            "revenue": [revenue],
            "gross_profit": [float(row["gross_margin"]) * revenue],
            "gross_margin": [float(row["gross_margin"])],
            "gm_yoy_lag": [float(row["gm_yoy_lag"])],
            "gm_level": [float(row["gross_margin"])],
            "gm_delta_yoy": [float(row["gross_margin"] - row["gm_yoy_lag"])],
            "gm_forecast": [float(row["gm_forecast"])],
            "forecast_model_version": [row["forecast_model_version"]],
            "forecast_feature_cutoff": [row["forecast_feature_cutoff"]],
            "forecast_training_end": [row["forecast_training_end"]],
            "forecast_training_observations": [row["forecast_training_observations"]],
            "forecast_generated_at": [row["forecast_generated_at"]],
            "forecast_sigma": [float(row["forecast_sigma"])],
            "forecast_error_history_end": [row["forecast_error_history_end"]],
            "forecast_sigma_observations": [row["forecast_sigma_observations"]],
            "gm_innovation_raw": [float(row["gm_innovation_raw"])],
            "gm_innovation_std": [float(row["gm_innovation_std"])],
            "revenue_fact_id": ["fact-revenue"],
            "gross_profit_fact_id": ["fact-gross-profit"],
            "filing_accession": ["0000000001-21-000001"],
        }
    )
    validate_filing_event_panel(panel)

    panel.loc[0, "forecast_training_end"] = accepted_at
    with pytest.raises(EventPanelValidationError, match="same-time or future"):
        validate_filing_event_panel(panel)


def test_common_sample_is_identical_across_representations() -> None:
    panel = pd.DataFrame(
        {
            "gm_level": [0.30, 0.31, 0.32],
            "gm_delta_yoy": [0.01, np.nan, 0.03],
            "gm_innovation_raw": [0.005, 0.004, 0.003],
            "gm_innovation_std": [1.0, 0.8, np.nan],
            "abnormal_return_drift": [0.02, 0.01, 0.03],
        }
    )

    raw = select_common_research_sample(
        panel, innovation="raw", outcome_column="abnormal_return_drift"
    )
    standardized = select_common_research_sample(
        panel, innovation="standardized", outcome_column="abnormal_return_drift"
    )

    assert raw.index.tolist() == [0, 2]
    assert standardized.index.tolist() == [0]


def test_gross_margin_derivation_rejects_inconsistent_supplied_value() -> None:
    panel = pd.DataFrame(
        {"revenue": [100.0], "gross_profit": [40.0], "gross_margin": [0.30], "gm_yoy_lag": [0.35]}
    )

    with pytest.raises(EventPanelValidationError, match="does not tie"):
        derive_gross_margin_representations(panel)
