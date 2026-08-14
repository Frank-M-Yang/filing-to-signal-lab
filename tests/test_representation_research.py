"""Real-data adapter tests for Level, Delta, and Innovation research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from event_panel import ForecastConfig
from representation_research import (
    RepresentationResearchConfig,
    RepresentationResearchError,
    build_representation_research,
    validate_representation_research,
)


def test_build_representations_respects_all_feature_clocks() -> None:
    result = build_representation_research(_clean_events(), _small_config())
    forecasted = result.panel.loc[result.panel["gm_forecast"].notna()]
    standardized = result.panel.loc[result.panel["gm_innovation_std"].notna()]

    assert not forecasted.empty
    assert not standardized.empty
    assert (forecasted["forecast_feature_cutoff"] < forecasted["actual_available_at"]).all()
    assert (forecasted["forecast_training_end"] < forecasted["actual_available_at"]).all()
    assert (standardized["forecast_error_history_end"] < standardized["actual_available_at"]).all()
    assert set(result.common_sample["event_id"]) == set(standardized["event_id"])
    assert result.report["evidence_boundary"].startswith("Financial representation")
    assert np.allclose(
        forecasted["gm_innovation_raw"],
        forecasted["gm_delta_yoy"] + forecasted["expectation_adjustment"],
    )
    assert set(result.expectation_adjustment_diagnostics["rank_scope"]) == {
        "pooled_common_sample",
        "availability_quarter_minimum_10_events",
    }
    assert len(result.forecast_bootstrap_intervals) == 3
    assert not result.forecast_loss_events.empty
    assert not result.universe_timepoint_audit["eligible_for_final_return_study"].any()


def test_future_filing_cannot_change_past_innovations() -> None:
    events = _clean_events()
    config = _small_config()
    baseline = build_representation_research(events, config).panel.set_index("event_id")
    changed = events.copy()
    future = pd.to_datetime(changed["actual_available_at"], utc=True).nlargest(5).index
    changed.loc[future, "gross_profit"] += changed.loc[future, "revenue"] * 0.20
    changed.loc[future, "gross_margin"] = (
        changed.loc[future, "gross_profit"] / changed.loc[future, "revenue"]
    )
    changed.loc[future, "gm_level"] = changed.loc[future, "gross_margin"]
    changed.loc[future, "gm_delta_yoy"] = (
        changed.loc[future, "gross_margin"] - changed.loc[future, "gm_yoy_lag"]
    )
    rerun = build_representation_research(changed, config).panel.set_index("event_id")
    cutoff = pd.to_datetime(events.loc[future, "actual_available_at"], utc=True).min()
    past_ids = baseline.index[baseline["actual_available_at"] < cutoff]

    pd.testing.assert_series_equal(
        baseline.loc[past_ids, "gm_forecast"],
        rerun.loc[past_ids, "gm_forecast"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        baseline.loc[past_ids, "forecast_sigma"],
        rerun.loc[past_ids, "forecast_sigma"],
        check_names=False,
    )


def test_validation_rejects_same_time_feature_cutoff() -> None:
    result = build_representation_research(_clean_events(), _small_config())
    panel = result.panel.copy()
    forecast_index = panel.index[panel["gm_forecast"].notna()][0]
    panel.loc[forecast_index, "forecast_feature_cutoff"] = panel.loc[
        forecast_index, "actual_available_at"
    ]

    with pytest.raises(RepresentationResearchError, match="strictly point-in-time"):
        validate_representation_research(panel, _small_config())


def test_validation_rejects_broken_expectation_adjustment_identity() -> None:
    result = build_representation_research(_clean_events(), _small_config())
    panel = result.panel.copy()
    forecast_index = panel.index[panel["gm_forecast"].notna()][0]
    panel.loc[forecast_index, "expectation_adjustment"] += 0.01

    with pytest.raises(RepresentationResearchError, match="Expectation Adjustment"):
        validate_representation_research(panel, _small_config())


def _small_config() -> RepresentationResearchConfig:
    return RepresentationResearchConfig(
        forecast=ForecastConfig(
            model_version="fixture-ols-v1",
            min_train_observations=15,
            min_oos_errors=5,
        ),
        min_industry_peers=2,
        min_company_standardized_innovations=2,
        bootstrap_repetitions=100,
    )


def _clean_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    companies = [("0000000001", "AAA"), ("0000000002", "BBB"), ("0000000003", "CCC")]
    periods = pd.date_range("2017-03-31", periods=24, freq="QE")
    for company_number, (company_id, ticker) in enumerate(companies):
        margins: list[float] = []
        revenues: list[float] = []
        for quarter_number, period_end in enumerate(periods):
            margin = (
                0.25
                + company_number * 0.04
                + quarter_number * 0.0008
                + np.sin(quarter_number * 0.9 + company_number) * 0.006
            )
            revenue = 1_000.0 * (1.0 + 0.02 * company_number) * (1.0 + 0.015 * quarter_number)
            margins.append(margin)
            revenues.append(revenue)
            lag_index = quarter_number - 4
            event_id = f"{ticker}-{quarter_number:02d}"
            accepted = period_end + pd.Timedelta(days=30, minutes=company_number)
            lag_event_id = f"{ticker}-{lag_index:02d}" if lag_index >= 0 else None
            lag_margin = margins[lag_index] if lag_index >= 0 else np.nan
            delta = margin - lag_margin if lag_index >= 0 else np.nan
            rows.append(
                {
                    "event_id": event_id,
                    "core_asset_version": "fixture-core-v1",
                    "company_id": company_id,
                    "company_name": ticker,
                    "ticker": ticker,
                    "sic": "3571",
                    "sic_description": "Computers",
                    "fiscal_year": period_end.year,
                    "fiscal_quarter": f"Q{quarter_number % 4 + 1}",
                    "period_end": period_end,
                    "filing_accession": f"accession-{event_id}",
                    "filing_accepted_at": accepted,
                    "actual_available_at": accepted,
                    "revenue": revenue,
                    "gross_profit": revenue * margin,
                    "gross_margin": margin,
                    "gm_level": margin,
                    "gm_yoy_lag": lag_margin,
                    "gm_yoy_lag_event_id": lag_event_id,
                    "gm_yoy_lag_available_at": (
                        periods[lag_index] + pd.Timedelta(days=30, minutes=company_number)
                        if lag_index >= 0
                        else pd.NaT
                    ),
                    "gm_delta_yoy": delta,
                    "gross_margin_latest": margin,
                    "latest_available_at": accepted,
                    "was_revised": False,
                    "revision_size": 0.0,
                    "revenue_fact_id": f"revenue-{event_id}",
                    "gross_profit_fact_id": f"profit-{event_id}",
                    "cost_fact_id": None,
                    "analysis_ready_level": True,
                    "analysis_ready_delta": lag_index >= 0,
                }
            )
    return pd.DataFrame(rows)
