"""Tests for strict multi-metric prior-guidance coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sec_data.guidance_metric_coverage import (
    GuidanceMetricCoverageConfig,
    calculate_guidance_metric_coverage,
    extract_guidance_metric_observations,
)


def test_extracts_basis_values_and_explicit_quarter_horizon() -> None:
    observations, report = extract_guidance_metric_observations(_guidance_text())

    revenue = observations.loc[
        observations["metric"].eq("revenue") & observations["explicit_quarter_horizon_ready"]
    ].iloc[0]
    assert revenue["numeric_low"] == pytest.approx(91_000_000.0)
    assert revenue["numeric_high"] == pytest.approx(94_000_000.0)
    assert revenue["horizon_fiscal_year"] == 2025
    assert revenue["horizon_fiscal_quarter"] == "Q4"

    eps = observations.loc[
        observations["metric"].eq("diluted_eps")
        & observations["accounting_basis"].eq("gaap")
        & observations["explicit_quarter_horizon_ready"]
    ].iloc[0]
    assert eps["numeric_low"] == pytest.approx(-0.14)
    assert eps["numeric_high"] == pytest.approx(-0.08)

    margin = observations.loc[
        observations["metric"].eq("operating_margin")
        & observations["explicit_quarter_horizon_ready"]
    ].iloc[0]
    assert margin["accounting_basis"] == "gaap"
    assert margin["numeric_midpoint"] == pytest.approx(0.10)
    assert report["guidance_sections"] >= 1


def test_coverage_authorizes_only_metrics_with_actuals_and_issuer_diversity(
    tmp_path: Path,
) -> None:
    content = _submission_bytes()
    digest = hashlib.sha256(content).hexdigest()
    raw_path = tmp_path / "raw" / digest[:2] / f"{digest}.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(content)
    completion_dir = tmp_path / "completions"
    completion_dir.mkdir()
    for document_id in ("doc-1", "doc-2"):
        (completion_dir / f"{document_id}.json").write_text(
            json.dumps(
                {
                    "sha256": digest,
                    "raw_path": raw_path.relative_to(tmp_path).as_posix(),
                }
            ),
            encoding="utf-8",
        )
    event_links = pd.DataFrame(
        {
            "validation_event_id": ["event-1", "event-2"],
            "cik": ["0000000001", "0000000002"],
            "guidance_document_id": ["doc-1", "doc-2"],
            "prior_guidance_candidate_ready": [True, True],
        }
    )
    document_targets = pd.DataFrame(
        {
            "guidance_document_id": ["doc-1", "doc-2"],
            "cik": ["0000000001", "0000000002"],
            "prior_accession_number": [
                "0000000001-25-000001",
                "0000000002-25-000001",
            ],
        }
    )
    filing_targets = pd.DataFrame(
        {
            "validation_event_id": ["event-1", "event-2"],
            "filing_accession": ["filing-1", "filing-2"],
            "fiscal_year": [2025, 2025],
            "fiscal_quarter": ["Q4", "Q4"],
            "gm_level": [0.40, 0.45],
        }
    )
    quarterly_history = pd.DataFrame(
        {
            "filing_accession": ["filing-1", "filing-2"],
            "revenue": [92_000_000.0, 93_000_000.0],
            "unit": ["USD", "USD"],
        }
    )

    result = calculate_guidance_metric_coverage(
        event_links,
        document_targets,
        filing_targets,
        quarterly_history,
        tmp_path,
        GuidanceMetricCoverageConfig(
            minimum_pilot_coverage=0.50,
            minimum_pilot_issuers=2,
        ),
    )
    coverage = result.coverage.set_index("metric")

    assert bool(coverage.loc["revenue", "evidence_coverage_gate"])
    assert bool(coverage.loc["revenue", "model_feature_gate"])
    assert coverage.loc["revenue", "model_ready_events"] == 2
    assert bool(coverage.loc["diluted_eps", "evidence_coverage_gate"])
    assert not bool(coverage.loc["diluted_eps", "model_feature_gate"])
    revenue_sanity = result.sanity_checks.set_index("metric").loc["revenue"]
    assert revenue_sanity["events"] == 2
    assert not bool(revenue_sanity["used_for_selection"])
    assert result.report["return_values_loaded"] is False
    assert result.report["current_filing_text_loaded"] is False


def test_revenue_values_stay_with_their_sentence_and_quarter() -> None:
    text = "\n".join(
        [
            "2023 Guidance",
            (
                "Net Sales — Annual net sales guidance is between $78 million and $82 "
                "million. We expect third quarter 2023 net sales between $21 million "
                "and $23 million."
            ),
            (
                "Distribution revenue is expected to grow in the fourth quarter. "
                "Operating expenses will increase $0.5 million."
            ),
            "Conference Call Information",
        ]
    )

    observations, _ = extract_guidance_metric_observations(text)
    revenue = observations.loc[observations["metric"].eq("revenue")]
    quarter = revenue.loc[revenue["horizon_type"].eq("quarter")].iloc[0]

    assert quarter["numeric_low"] == pytest.approx(21_000_000.0)
    assert quarter["numeric_high"] == pytest.approx(23_000_000.0)
    assert quarter["horizon_fiscal_quarter"] == "Q3"
    assert not revenue["numeric_low"].eq(500_000.0).any()


def _guidance_text() -> str:
    return """Fourth Quarter 2025 Financial Outlook
The following table is in millions, except per share data:
Revenue
$91.0 to $94.0
GAAP Net loss per share — basic and diluted
($0.14) to ($0.08)
Non-GAAP Net income per share — diluted
$0.45 to $0.49
GAAP and non-GAAP operating margins are expected to be 10% and 12%, respectively.
Conference Call Information"""


def _submission_bytes() -> bytes:
    text = _guidance_text()
    return (
        "<SEC-DOCUMENT><DOCUMENT><TYPE>EX-99.1\n<FILENAME>release.htm\n"
        f"<TEXT><html><body>{text}</body></html></TEXT>"
        "</DOCUMENT></SEC-DOCUMENT>"
    ).encode()
