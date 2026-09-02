"""Tests for strict quarterly PIT feature construction and target coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.pilot_filing_features import (
    PilotFilingFeatureConfig,
    _annual_minus_ytd_values,
    _validated_asof_tag_bridges,
    audit_target_feature_coverage,
)


def test_q4_uses_annual_minus_strictly_prior_original_ytd() -> None:
    annual = pd.DataFrame(
        [
            _flow_row(
                "annual-revenue", "RevenueFromContractWithCustomerExcludingAssessedTax", 100.0
            ),
            _flow_row("annual-profit", "GrossProfit", 40.0),
        ]
    )
    ytd = pd.DataFrame(
        [
            _flow_row(
                "ytd-revenue",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                70.0,
                period_end="2024-09-30",
                accepted_at="2024-11-01T21:00:00Z",
            ),
            _flow_row(
                "ytd-profit",
                "GrossProfit",
                25.0,
                period_end="2024-09-30",
                accepted_at="2024-11-01T21:00:00Z",
            ),
        ]
    )

    result = _annual_minus_ytd_values(annual, ytd)

    assert result is not None
    assert result["revenue"] == pytest.approx(30.0)
    assert result["gross_profit"] == pytest.approx(15.0)
    assert result["quarter_start"] == pd.Timestamp("2024-10-01")
    assert result["annual_revenue_fact_id"] == "annual-revenue"
    assert result["ytd_revenue_fact_id"] == "ytd-revenue"

    future_ytd = ytd.copy()
    future_ytd["accepted_at"] = pd.Timestamp("2025-03-01T21:00:00Z")
    assert _annual_minus_ytd_values(annual, future_ytd) is None


def test_target_audit_retains_every_exact_target_accession() -> None:
    targets = pd.DataFrame(
        {
            "target_id": ["ready", "missing"],
            "cik": ["1", "2"],
            "selected_ticker": ["ONE", "TWO"],
            "accession_number": ["acc-ready", "acc-missing"],
            "form": ["10-Q", "10-K"],
            "report_date": ["2024-09-30", "2024-12-31"],
            "accepted_at": ["2024-11-01T21:00:00Z", "2025-02-01T21:00:00Z"],
            "formal_entry_open_at": ["2024-11-05T14:30:00Z", "2025-02-05T14:30:00Z"],
            "formal_exit_close_at": ["2024-12-03T21:00:00Z", "2025-03-05T21:00:00Z"],
        }
    )
    history = pd.DataFrame([_history_row()])

    features, audit = audit_target_feature_coverage(targets, history)

    assert features["filing_accession"].tolist() == ["acc-ready", "acc-missing"]
    assert features["accession_match"].tolist() == [True, True]
    assert audit["quarterly_feature_event_match"].tolist() == [True, False]
    assert audit["feature_status"].tolist() == [
        "complete_case_ready",
        "no_quarterly_level_event",
    ]
    assert (
        pd.to_datetime(features["actual_available_at"], utc=True).tolist()
        == pd.to_datetime(targets["accepted_at"], utc=True).tolist()
    )


def test_q4_cross_tag_bridge_requires_strictly_prior_equal_fact_evidence() -> None:
    annual = pd.DataFrame(
        [
            _flow_row(
                "annual-revenue", "RevenueFromContractWithCustomerExcludingAssessedTax", 100.0
            ),
            _flow_row("annual-cost", "CostOfRevenue", 60.0),
        ]
    )
    ytd = pd.DataFrame(
        [
            _flow_row(
                "ytd-revenue",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                70.0,
                period_end="2024-09-30",
                accepted_at="2024-11-01T21:00:00Z",
            ),
            _flow_row(
                "ytd-cost",
                "CostOfGoodsAndServicesSold",
                42.0,
                period_end="2024-09-30",
                accepted_at="2024-11-01T21:00:00Z",
            ),
        ]
    )
    evidence = pd.DataFrame(
        [
            _flow_row(
                "bridge-cost-a",
                "CostOfRevenue",
                55.0,
                period_start="2023-01-01",
                period_end="2023-12-31",
                accepted_at="2024-02-01T21:00:00Z",
            ),
            _flow_row(
                "bridge-cost-b",
                "CostOfGoodsAndServicesSold",
                55.0,
                period_start="2023-01-01",
                period_end="2023-12-31",
                accepted_at="2024-02-01T21:00:00Z",
            ),
        ]
    )
    bridges = _validated_asof_tag_bridges(
        evidence,
        pd.Timestamp("2025-02-01T21:00:00Z"),
        ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    )

    assert _annual_minus_ytd_values(annual, ytd) is None
    result = _annual_minus_ytd_values(annual, ytd, cost_tag_bridges=bridges)

    assert result is not None
    assert result["gross_profit"] == pytest.approx(12.0)
    assert "bridge-cost-a" in str(result["tag_bridge_evidence"])
    future_bridges = _validated_asof_tag_bridges(
        evidence.assign(accepted_at=pd.Timestamp("2025-03-01T21:00:00Z")),
        pd.Timestamp("2025-02-01T21:00:00Z"),
        ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    )
    assert _annual_minus_ytd_values(annual, ytd, cost_tag_bridges=future_bridges) is None


def test_applicability_counts_only_strictly_prior_history() -> None:
    targets = pd.DataFrame(
        {
            "target_id": ["ready"],
            "cik": ["1"],
            "selected_ticker": ["ONE"],
            "accession_number": ["acc-ready"],
            "form": ["10-Q"],
            "report_date": ["2024-12-31"],
            "accepted_at": ["2025-02-01T21:00:00Z"],
            "formal_entry_open_at": ["2025-02-05T14:30:00Z"],
            "formal_exit_close_at": ["2025-03-05T21:00:00Z"],
        }
    )
    history = pd.DataFrame(
        [_history_row(available_at="2025-02-01T21:00:00Z")]
        + [
            _history_row(
                accession=f"prior-{index}",
                available_at=f"2024-0{index + 1}-01T21:00:00Z",
            )
            for index in range(4)
        ]
    )
    strict = PilotFilingFeatureConfig(
        minimum_prior_level_events_for_applicability=5,
        minimum_prior_delta_events_for_applicability=4,
        minimum_prior_standardized_innovations_for_applicability=4,
    )
    relaxed = PilotFilingFeatureConfig(
        minimum_prior_level_events_for_applicability=4,
        minimum_prior_delta_events_for_applicability=4,
        minimum_prior_standardized_innovations_for_applicability=4,
    )

    strict_features, _ = audit_target_feature_coverage(targets, history, strict)
    relaxed_features, _ = audit_target_feature_coverage(targets, history, relaxed)

    assert strict_features.iloc[0]["prior_level_events"] == 4
    assert not strict_features.iloc[0]["gross_margin_history_eligible"]
    assert relaxed_features.iloc[0]["gross_margin_history_eligible"]


def _flow_row(
    fact_id: str,
    tag: str,
    value: float,
    *,
    period_start: str = "2024-01-01",
    period_end: str = "2024-12-31",
    accepted_at: str = "2025-02-01T21:00:00Z",
) -> dict[str, object]:
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    return {
        "fact_id": fact_id,
        "taxonomy": "us-gaap",
        "tag": tag,
        "unit": "USD",
        "value_numeric": value,
        "period_start": start,
        "period_end": end,
        "duration_days": (end - start).days + 1,
        "fiscal_year": 2024,
        "accepted_at": pd.Timestamp(accepted_at),
        "source_snapshot_sha256": "source",
    }


def _history_row(
    *,
    accession: str = "acc-ready",
    available_at: str = "2024-11-01T21:00:00Z",
) -> dict[str, object]:
    return {
        "company_id": "0000000001",
        "ticker": "ONE",
        "fiscal_year": 2024,
        "fiscal_quarter": "Q3",
        "period_end": pd.Timestamp("2024-09-30"),
        "filing_accession": accession,
        "actual_available_at": pd.Timestamp(available_at),
        "gm_level": 0.40,
        "gm_yoy_lag": 0.38,
        "gm_delta_yoy": 0.02,
        "gm_forecast": 0.39,
        "forecast_sigma": 0.01,
        "gm_innovation_raw": 0.01,
        "gm_innovation_std": 1.0,
        "forecast_feature_cutoff": pd.Timestamp("2024-10-31T21:00:00Z"),
        "forecast_training_end": pd.Timestamp("2024-10-30T21:00:00Z"),
        "forecast_training_observations": 100,
        "forecast_error_history_end": pd.Timestamp("2024-10-29T21:00:00Z"),
        "forecast_sigma_observations": 50,
        "revenue_fact_id": "revenue",
        "gross_profit_fact_id": "profit",
        "cost_fact_id": None,
        "q4_source_method": None,
        "ytd_filing_accession": None,
        "tag_bridge_evidence": None,
        "analysis_ready_level": True,
        "analysis_ready_delta": True,
        "analysis_ready_innovation_raw": True,
        "analysis_ready_innovation_std": True,
        "pilot_feature_version": "test",
    }
