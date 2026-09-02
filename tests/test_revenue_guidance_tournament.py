"""Tests for the point-in-time revenue-guidance forecast tournament."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_tournament import (
    RevenueGuidanceTournamentConfig,
    calculate_revenue_guidance_tournament,
)


def test_guidance_beats_strictly_prior_seasonal_baseline_without_returns() -> None:
    event_metrics, links, filings, history = _fixtures()
    result = calculate_revenue_guidance_tournament(
        event_metrics,
        links,
        filings,
        history,
        RevenueGuidanceTournamentConfig(
            minimum_events=10,
            minimum_issuers=3,
            bootstrap_repetitions=1_000,
            bootstrap_seed=9,
        ),
    )

    assert result.report["forecast_gate_passed"] is True
    assert result.report["selected_revenue_expectation"] == "management_guidance"
    assert result.report["relative_mape_improvement"] > 0.80
    assert result.report["mean_ape_difference_ci_high"] < 0.0
    assert result.report["return_values_loaded"] is False
    assert result.forecasts["seasonal_information_precedes_guidance"].all()


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    link_rows = []
    filing_rows = []
    history_rows = []
    for index in range(12):
        issuer = index % 3
        year = 2022 + index // 3
        event_id = f"event-{index}"
        company_id = f"company-{issuer}"
        actual = 100.0 + index * 10.0
        prior_at = pd.Timestamp(f"{year}-02-01 20:00:00", tz="UTC")
        actual_at = pd.Timestamp(f"{year}-05-01 20:00:00", tz="UTC")
        metric_rows.append(
            {
                "validation_event_id": event_id,
                "cik": f"{issuer + 1:010d}",
                "metric": "revenue",
                "selected_guidance_low": actual * 0.98,
                "selected_guidance_high": actual,
                "selected_guidance_midpoint": actual * 0.99,
                "target_actual": actual,
                "model_feature_ready": True,
            }
        )
        link_rows.append(
            {
                "validation_event_id": event_id,
                "prior_accepted_at": prior_at,
                "target_accepted_at": actual_at,
                "guidance_age_days": 89.0,
            }
        )
        filing_rows.append(
            {
                "validation_event_id": event_id,
                "company_id": company_id,
                "fiscal_year": year,
                "fiscal_quarter": "Q1",
                "actual_available_at": actual_at,
            }
        )
        history_rows.append(
            {
                "event_id": f"source-{index}",
                "company_id": company_id,
                "fiscal_year": year - 1,
                "fiscal_quarter": "Q1",
                "revenue": actual * 0.70,
                "unit": "USD",
                "actual_available_at": prior_at - pd.Timedelta(days=30),
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(link_rows),
        pd.DataFrame(filing_rows),
        pd.DataFrame(history_rows),
    )
