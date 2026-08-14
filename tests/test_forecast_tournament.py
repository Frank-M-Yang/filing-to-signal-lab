"""Tests for the preregistered point-in-time forecast tournament."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_tournament import (
    ForecastTournamentConfig,
    ForecastTournamentError,
    build_forecast_tournament,
    validate_forecast_tournament,
)


def test_tournament_locks_common_sample_and_reports_all_gates() -> None:
    result = build_forecast_tournament(_tournament_panel(), _config())

    assert not result.common_sample.empty
    assert set(result.metrics["model_id"]) == {"M0", "M1", "M2", "M3", "M4"}
    assert result.common_sample[[f"M{i}_forecast" for i in range(5)]].notna().all().all()
    assert result.feature_clock_audit["features_strictly_prior"].all()
    assert result.feature_clock_audit["training_strictly_prior"].all()
    assert not result.eligibility_gate["distinctness_used_for_selection"].any()
    assert not result.eligibility_gate["eligible_for_downstream_returns"].any()
    assert len(result.bootstrap_intervals) == 15
    assert result.report["evidence_boundary"].startswith("Financial forecast tournament")


def test_future_actuals_cannot_change_past_tournament_forecasts() -> None:
    source = _tournament_panel()
    config = _config()
    baseline = build_forecast_tournament(source, config).panel.set_index("event_id")
    changed = source.copy()
    future = pd.to_datetime(changed["actual_available_at"], utc=True).nlargest(4).index
    changed.loc[future, "gross_margin"] += 0.15
    changed.loc[future, "gm_delta_yoy"] = (
        changed.loc[future, "gross_margin"] - changed.loc[future, "gm_yoy_lag"]
    )
    rerun = build_forecast_tournament(changed, config).panel.set_index("event_id")
    cutoff = pd.to_datetime(source.loc[future, "actual_available_at"], utc=True).min()
    past_ids = baseline.index[baseline["actual_available_at"] < cutoff]

    for model_id in ("M2", "M3", "M4"):
        pd.testing.assert_series_equal(
            baseline.loc[past_ids, f"{model_id}_forecast"],
            rerun.loc[past_ids, f"{model_id}_forecast"],
            check_names=False,
        )


def test_tournament_validation_rejects_same_time_peer_feature() -> None:
    result = build_forecast_tournament(_tournament_panel(), _config())
    panel = result.panel.copy()
    index = panel.index[panel["M3_forecast"].notna()][0]
    panel.loc[index, "M3_feature_cutoff"] = panel.loc[index, "actual_available_at"]

    with pytest.raises(ForecastTournamentError, match="same-time or future feature"):
        validate_forecast_tournament(panel)


def test_tournament_is_deterministic_for_fixed_configuration() -> None:
    source = _tournament_panel()
    config = _config()
    first = build_forecast_tournament(source, config)
    second = build_forecast_tournament(source, config)

    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.bootstrap_intervals, second.bootstrap_intervals)
    assert (
        first.report["forecast_tournament_version"] == second.report["forecast_tournament_version"]
    )


def _config() -> ForecastTournamentConfig:
    return ForecastTournamentConfig(
        min_train_observations=12,
        min_eligible_year_events=5,
        bootstrap_repetitions=100,
    )


def _tournament_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    companies = [
        ("0000000001", "AAA"),
        ("0000000002", "BBB"),
        ("0000000003", "CCC"),
        ("0000000004", "DDD"),
    ]
    periods = pd.date_range("2016-03-31", periods=28, freq="QE")
    histories: dict[str, list[float]] = {ticker: [] for _, ticker in companies}
    revenues: dict[str, list[float]] = {ticker: [] for _, ticker in companies}
    for quarter_number, period_end in enumerate(periods):
        for company_number, (company_id, ticker) in enumerate(companies):
            margin = (
                0.24
                + company_number * 0.035
                + quarter_number * 0.0007
                + np.sin(quarter_number * 0.8 + company_number) * 0.008
            )
            revenue = 1_000.0 * (1.0 + company_number * 0.04) * (1.0 + quarter_number * 0.012)
            histories[ticker].append(margin)
            revenues[ticker].append(revenue)
            accepted = period_end + pd.Timedelta(days=30, minutes=company_number)
            lag4 = quarter_number - 4
            lag8 = quarter_number - 8
            prior = quarter_number - 1
            gm_yoy_lag = histories[ticker][lag4] if lag4 >= 0 else np.nan
            gm_delta = margin - gm_yoy_lag if lag4 >= 0 else np.nan
            delta_lag4 = histories[ticker][lag4] - histories[ticker][lag8] if lag8 >= 0 else np.nan
            prior_revenue_growth = (
                revenues[ticker][prior] / revenues[ticker][prior - 4] - 1.0
                if prior >= 4
                else np.nan
            )
            strict_history = histories[ticker][max(0, quarter_number - 8) : quarter_number]
            trailing_mean = float(np.mean(strict_history)) if strict_history else np.nan
            trailing_volatility = (
                float(np.std(strict_history, ddof=1)) if len(strict_history) >= 2 else np.nan
            )
            lag_time = (
                periods[lag4] + pd.Timedelta(days=30, minutes=company_number)
                if lag4 >= 0
                else pd.NaT
            )
            prior_time = (
                periods[prior] + pd.Timedelta(days=30, minutes=company_number)
                if prior >= 0
                else pd.NaT
            )
            feature_times = [value for value in (lag_time, prior_time) if not pd.isna(value)]
            feature_cutoff = max(feature_times) if feature_times else pd.NaT
            m1_forecast = (
                gm_yoy_lag + 0.15 * delta_lag4 + 0.02 * prior_revenue_growth
                if lag8 >= 0 and prior >= 4
                else np.nan
            )
            rows.append(
                {
                    "event_id": f"{ticker}-{quarter_number:02d}",
                    "company_id": company_id,
                    "ticker": ticker,
                    "sic1": "3",
                    "sic2": "35",
                    "period_end": period_end,
                    "actual_available_at": accepted,
                    "gross_margin": margin,
                    "gm_yoy_lag": gm_yoy_lag,
                    "gm_yoy_lag_available_at": lag_time,
                    "gm_delta_yoy": gm_delta,
                    "gm_delta_yoy_lag": delta_lag4,
                    "revenue_growth_lag1": prior_revenue_growth,
                    "revenue_growth_lag1_available_at": prior_time,
                    "industry_gm_lag1": 0.29 + np.sin(quarter_number * 0.4) * 0.003,
                    "industry_gm_lag1_available_at": prior_time,
                    "company_trailing_gm_mean": trailing_mean,
                    "company_trailing_gm_volatility": trailing_volatility,
                    "company_trailing_gm_history_end": prior_time,
                    "gm_forecast": m1_forecast,
                    "forecast_feature_cutoff": feature_cutoff,
                    "forecast_training_end": (
                        accepted - pd.Timedelta(days=1) if not np.isnan(m1_forecast) else pd.NaT
                    ),
                    "forecast_training_observations": (
                        quarter_number * len(companies) if not np.isnan(m1_forecast) else pd.NA
                    ),
                    "forecast_generated_at": (
                        accepted - pd.Timedelta(days=1) if not np.isnan(m1_forecast) else pd.NaT
                    ),
                }
            )
    return pd.DataFrame(rows)
