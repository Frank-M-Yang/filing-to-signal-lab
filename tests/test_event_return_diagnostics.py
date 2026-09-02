"""Tests for same-sample event-return diagnostics and feature gates."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.event_return_diagnostics import (
    EventReturnDiagnosticConfig,
    calculate_event_return_diagnostics,
)


def test_survivors_only_counterfactual_reports_future_status_inflation() -> None:
    ledger, readiness = _return_inputs(
        issuer_returns=[0.0, 0.1, -0.3, 0.2],
        later_delisted=[False, False, True, False],
    )
    representation = _representation_panel(["0009999999"], ["other-accession"])

    result = calculate_event_return_diagnostics(
        ledger,
        readiness,
        representation,
        EventReturnDiagnosticConfig(uncertainty_resamples=1000),
    )

    assert result.report["events"] == 4
    assert result.report["survivors_only_mean_inflation_pp"] == pytest.approx(10.0)
    assert result.report["later_delisted_mean_market_excess_return"] == pytest.approx(-0.3)
    assert result.report["representation_complete_events"] == 0
    assert result.report["exploratory_ic_allowed"] is False
    all_events = result.survivorship_counterfactual.loc[
        result.survivorship_counterfactual["sample_id"].eq("all_events")
    ].iloc[0]
    assert all_events["sample_label"] == "全部 4 个事件（无偏主样本）"
    assert set(result.feature_join_audit["feature_join_status"]) == {
        "issuer_not_in_representation_universe"
    }
    assert result.representation_ic["spearman_ic"].isna().all()


def test_exact_accession_and_clock_join_unlocks_exploratory_ic() -> None:
    observations = 20
    issuer_returns = [index / 100.0 for index in range(observations)]
    ledger, readiness = _return_inputs(
        issuer_returns=issuer_returns,
        later_delisted=[False] * observations,
    )
    ciks = [f"{index + 1:010d}" for index in range(observations)]
    accessions = [f"accession-{index}" for index in range(observations)]
    readiness["cik"] = ciks
    readiness["accession_number"] = accessions
    ledger["cik"] = ciks
    representation = _representation_panel(ciks, accessions)

    result = calculate_event_return_diagnostics(
        ledger,
        readiness,
        representation,
        EventReturnDiagnosticConfig(uncertainty_resamples=1000),
    )

    assert result.report["representation_complete_events"] == observations
    assert result.report["exploratory_ic_allowed"] is True
    assert set(result.feature_join_audit["feature_join_status"]) == {"complete_case_ready"}
    assert result.representation_ic["spearman_ic"].tolist() == pytest.approx([1.0] * 4)
    assert set(result.representation_ic["status"]) == {"exploratory_only"}


def test_event_count_alone_cannot_bypass_feature_coverage_gate() -> None:
    observations = 30
    ledger, readiness = _return_inputs(
        issuer_returns=[index / 100.0 for index in range(observations)],
        later_delisted=[False] * observations,
    )
    ciks = [f"{index + 1:010d}" for index in range(observations)]
    accessions = [f"accession-{index}" for index in range(observations)]
    readiness["cik"] = ciks
    readiness["accession_number"] = accessions
    ledger["cik"] = ciks
    representation = _representation_panel(ciks, accessions)
    representation.loc[20:, "gm_innovation_std"] = float("nan")

    result = calculate_event_return_diagnostics(
        ledger,
        readiness,
        representation,
        EventReturnDiagnosticConfig(uncertainty_resamples=1000),
    )

    assert result.report["representation_complete_events"] == 20
    assert result.report["representation_complete_coverage"] == pytest.approx(2.0 / 3.0)
    assert result.report["exploratory_ic_allowed"] is False
    assert result.representation_ic["spearman_ic"].isna().all()
    assert not result.representation_ic["coverage_ratio_pass"].any()


def test_pre_event_applicability_defines_the_feature_coverage_denominator() -> None:
    observations = 30
    ledger, readiness = _return_inputs(
        issuer_returns=[index / 100.0 for index in range(observations)],
        later_delisted=[False] * observations,
    )
    ciks = [f"{index + 1:010d}" for index in range(observations)]
    accessions = [f"accession-{index}" for index in range(observations)]
    readiness["cik"] = ciks
    readiness["accession_number"] = accessions
    ledger["cik"] = ciks
    representation = _representation_panel(ciks, accessions)
    representation["gross_margin_history_eligible"] = [True] * 25 + [False] * 5
    representation.loc[20:24, "gm_innovation_std"] = float("nan")

    result = calculate_event_return_diagnostics(
        ledger,
        readiness,
        representation,
        EventReturnDiagnosticConfig(uncertainty_resamples=1000),
    )

    assert result.report["representation_applicable_events"] == 25
    assert result.report["representation_complete_events"] == 20
    assert result.report["representation_complete_coverage"] == pytest.approx(0.8)
    assert result.report["exploratory_ic_allowed"] is True
    assert len(result.representation_return_sample) == 20
    assert result.representation_ic_uncertainty["permutation_p_two_sided"].notna().all()
    assert (
        result.feature_join_audit["feature_join_status"].eq("gross_margin_history_ineligible").sum()
        == 5
    )


def _return_inputs(
    *, issuer_returns: list[float], later_delisted: list[bool]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = len(issuer_returns)
    accepted = [f"2025-01-{index + 1:02d}T21:00:00+00:00" for index in range(rows)]
    target_ids = [f"target-{index}" for index in range(rows)]
    ciks = [f"{index + 1:010d}" for index in range(rows)]
    tickers = [f"T{index}" for index in range(rows)]
    ledger = pd.DataFrame(
        {
            "target_id": target_ids,
            "cik": ciks,
            "ticker": tickers,
            "form": ["10-K"] * rows,
            "accepted_at": accepted,
            "entry_at": ["2025-02-01T14:30:00+00:00"] * rows,
            "exit_at": ["2025-03-01T21:00:00+00:00"] * rows,
            "issuer_total_return": issuer_returns,
            "market_total_return": [0.0] * rows,
            "market_excess_return_arithmetic": issuer_returns,
            "issuer_return_ready": [True] * rows,
            "market_excess_return_ready": [True] * rows,
            "sector_excess_return_ready": [False] * rows,
            "later_delisted_after_formal_exit": later_delisted,
        }
    )
    readiness = pd.DataFrame(
        {
            "target_id": target_ids,
            "cik": ciks,
            "selected_ticker": tickers,
            "accession_number": [f"accession-{index}" for index in range(rows)],
            "report_date": ["2024-12-31"] * rows,
            "accepted_at": accepted,
            "return_study_eligible": [True] * rows,
            "delisting_outcome_ready": [True] * rows,
        }
    )
    return ledger, readiness


def _representation_panel(ciks: list[str], accessions: list[str]) -> pd.DataFrame:
    rows = len(ciks)
    feature = [index / 100.0 for index in range(rows)]
    return pd.DataFrame(
        {
            "company_id": ciks,
            "ticker": [f"T{index}" for index in range(rows)],
            "filing_accession": accessions,
            "actual_available_at": [
                f"2025-01-{index + 1:02d}T21:00:00+00:00" for index in range(rows)
            ],
            "gm_level": feature,
            "gm_delta_yoy": feature,
            "gm_innovation_raw": feature,
            "gm_innovation_std": feature,
        }
    )
