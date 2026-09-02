"""Tests for filing-time SIC fixed-effect return diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.sector_neutral_diagnostics import (
    SectorNeutralDiagnosticConfig,
    calculate_sector_neutral_diagnostics,
)


def test_exact_filing_time_sic_unlocks_fixed_effect_diagnostic() -> None:
    sample, industries = _inputs()

    result = calculate_sector_neutral_diagnostics(
        sample,
        industries,
        SectorNeutralDiagnosticConfig(
            minimum_events_for_ic=10,
            uncertainty_resamples=1000,
        ),
    )

    assert result.report["historical_industry_exact_matches"] == 12
    assert result.report["sector_fixed_effect_events"] == 12
    assert result.report["sic_divisions_fixed_effect_eligible"] == 2
    assert result.report["sector_fixed_effect_diagnostic_allowed"] is True
    assert result.report["tradable_sector_benchmark_gate"] is False
    assert result.ic_comparison["sic_fixed_effect_ic"].notna().all()
    assert result.ic_uncertainty["permutation_p_holm"].notna().all()


def test_singleton_division_is_audited_but_not_used_for_identification() -> None:
    sample, industries = _inputs()
    extra_sample = sample.iloc[[0]].copy()
    extra_sample["target_id"] = "target-singleton"
    extra_sample["cik"] = "0000000099"
    extra_sample["accession_number"] = "accession-singleton"
    extra_industry = industries.iloc[[0]].copy()
    extra_industry["cik"] = "0000000099"
    extra_industry["accession_number"] = "accession-singleton"
    extra_industry["sic"] = "1400"

    result = calculate_sector_neutral_diagnostics(
        pd.concat([sample, extra_sample], ignore_index=True),
        pd.concat([industries, extra_industry], ignore_index=True),
        SectorNeutralDiagnosticConfig(
            minimum_events_for_ic=10,
            uncertainty_resamples=1000,
        ),
    )

    singleton = result.sample_audit.loc[
        result.sample_audit["accession_number"].eq("accession-singleton")
    ].iloc[0]
    assert singleton["historical_industry_exact_match"]
    assert not singleton["sector_fixed_effect_eligible"]
    assert result.report["singleton_or_unmapped_events_excluded"] == 1


def test_industry_acceptance_timestamp_must_match_the_return_event() -> None:
    sample, industries = _inputs()
    industries.loc[0, "accepted_at"] = "2025-02-01T20:00:01Z"

    result = calculate_sector_neutral_diagnostics(
        sample,
        industries,
        SectorNeutralDiagnosticConfig(
            minimum_events_for_ic=10,
            uncertainty_resamples=1000,
        ),
    )

    mismatch = result.sample_audit.loc[
        result.sample_audit["accession_number"].eq("accession-00")
    ].iloc[0]
    assert mismatch["industry_join_status"] == "accepted_at_mismatch"
    assert not mismatch["historical_industry_exact_match"]
    assert result.report["historical_industry_exact_matches"] == 11


def test_duplicate_industry_evidence_is_rejected() -> None:
    sample, industries = _inputs()

    with pytest.raises(FilingCatalogError, match="not unique"):
        calculate_sector_neutral_diagnostics(
            sample,
            pd.concat([industries, industries.iloc[[0]]], ignore_index=True),
            SectorNeutralDiagnosticConfig(
                minimum_events_for_ic=10,
                uncertainty_resamples=1000,
            ),
        )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 12
    accepted_at = [f"2025-02-{index + 1:02d}T20:00:00Z" for index in range(rows)]
    sample = pd.DataFrame(
        {
            "target_id": [f"target-{index:02d}" for index in range(rows)],
            "cik": [f"{index + 1:010d}" for index in range(rows)],
            "accession_number": [f"accession-{index:02d}" for index in range(rows)],
            "accepted_at": accepted_at,
            "gm_level": [index / 20 for index in range(rows)],
            "gm_delta_yoy": [((index * 5) % 11) / 20 for index in range(rows)],
            "gm_innovation_raw": [((index * 7) % 13) / 20 for index in range(rows)],
            "gm_innovation_std": [((index * 7) % 13) / 10 for index in range(rows)],
            "market_excess_return_arithmetic": [index / 100 for index in range(rows)],
        }
    )
    industries = pd.DataFrame(
        {
            "cik": sample["cik"],
            "accession_number": sample["accession_number"],
            "accepted_at": accepted_at,
            "sic": ["2834"] * 6 + ["7372"] * 6,
            "sic_description": ["Pharmaceutical preparations"] * 6 + ["Prepackaged software"] * 6,
            "index_subject_cik_matched": [True] * rows,
            "historical_industry_ready": [True] * rows,
            "source_sha256": [f"sha-{index}" for index in range(rows)],
            "source_path": [f"source-{index}.html" for index in range(rows)],
        }
    )
    return sample, industries
