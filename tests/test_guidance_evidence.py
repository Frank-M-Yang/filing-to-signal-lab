"""Tests for point-in-time prior company-guidance evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sec_data.guidance_evidence import (
    GuidanceEvidenceConfig,
    calculate_guidance_evidence,
    calculate_guidance_plan,
    extract_guidance_sentences,
    parse_complete_submission,
)


def test_plan_prefers_prior_item_202_and_never_loads_current_filing() -> None:
    event_time = pd.Timestamp("2025-07-01 20:00:00", tz="UTC")
    market_state = pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "cik": ["0000000001"],
            "accepted_at": [event_time],
            "market_state_ready": [True],
        }
    )
    catalog = pd.DataFrame(
        [
            _filing("0000000001-25-000004", event_time, "2.02"),
            _filing("0000000001-25-000003", event_time - pd.Timedelta(days=42), "7.01"),
            _filing("0000000001-25-000002", event_time - pd.Timedelta(days=49), "2.02"),
            _filing("0000000001-24-000001", event_time - pd.Timedelta(days=250), "2.02"),
        ]
    )

    result = calculate_guidance_plan(market_state, catalog)
    row = result.event_links.iloc[0]

    assert row["prior_accession_number"] == "0000000001-25-000002"
    assert row["guidance_age_days"] == pytest.approx(49.0)
    assert result.report["current_filing_text_loaded"] is False
    assert result.report["outcome_values_loaded"] is False


def test_complete_submission_extracts_exhibit_and_numeric_gross_margin() -> None:
    documents = parse_complete_submission(_submission_bytes())

    assert [row["document_type"] for row in documents] == ["8-K", "EX-99.1"]
    exhibit = documents[1]
    evidence, summary = extract_guidance_sentences(exhibit["visible_text"], 0.02)

    assert len(evidence) == 1
    assert summary["numeric_gross_margin_guidance_ready"] is True
    assert summary["prior_guidance_gross_margin_low"] == pytest.approx(0.415)
    assert summary["prior_guidance_gross_margin_high"] == pytest.approx(0.425)
    assert summary["prior_guidance_gross_margin_midpoint"] == pytest.approx(0.42)

    _, plural = extract_guidance_sentences(
        "GAAP and non-GAAP gross margins are expected to be 73.0% and 73.5%.",
        0.02,
    )
    assert plural["numeric_gross_margin_guidance_ready"] is True
    assert plural["prior_guidance_gross_margin_midpoint"] == pytest.approx(0.73)

    _, adjusted = extract_guidance_sentences(
        "We expect adjusted gross margin to be between 78% and 79%.",
        0.02,
    )
    assert adjusted["numeric_gross_margin_guidance_ready"] is False
    assert adjusted["gross_margin_extraction_status"] == (
        "non_gaap_or_ambiguous_numeric_guidance_only"
    )


def test_evidence_audit_is_hash_tied_and_authorizes_complete_numeric_sample(
    tmp_path: Path,
) -> None:
    content = _submission_bytes()
    digest = hashlib.sha256(content).hexdigest()
    raw_path = tmp_path / "raw" / digest[:2] / f"{digest}.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(content)
    completion_path = tmp_path / "completions" / "document-1.json"
    completion_path.parent.mkdir()
    completion_path.write_text(
        json.dumps({"sha256": digest, "raw_path": raw_path.relative_to(tmp_path).as_posix()}),
        encoding="utf-8",
    )
    links = pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "cik": ["0000000001"],
            "guidance_document_id": ["document-1"],
            "prior_accession_number": ["0000000001-25-000002"],
            "prior_guidance_candidate_ready": [True],
        }
    )
    targets = pd.DataFrame(
        {
            "guidance_document_id": ["document-1"],
            "cik": ["0000000001"],
            "prior_accession_number": ["0000000001-25-000002"],
        }
    )

    result = calculate_guidance_evidence(
        links,
        targets,
        tmp_path,
        GuidanceEvidenceConfig(
            minimum_document_coverage=1.0,
            minimum_exhibit_text_coverage=1.0,
            minimum_numeric_gross_margin_coverage=1.0,
        ),
    )

    assert result.report["documents_collected"] == 1
    assert result.report["model_run_authorized"] is True
    assert result.document_audit.loc[0, "source_sha256"] == digest
    assert result.event_features.loc[0, "prior_guidance_gross_margin_midpoint"] == pytest.approx(
        0.42
    )


def _filing(accession: str, accepted_at: pd.Timestamp, items: object) -> dict[str, object]:
    return {
        "cik": "0000000001",
        "accession_number": accession,
        "form": "8-K",
        "accepted_at": accepted_at,
        "items": items,
        "primary_document": "filing.htm",
        "source_url": "https://data.sec.gov/submissions/CIK0000000001.json",
    }


def _submission_bytes() -> bytes:
    return b"""<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>8-K
<FILENAME>filing.htm
<DESCRIPTION>CURRENT REPORT
<TEXT><html><body>Current report.</body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>release.htm
<DESCRIPTION>EARNINGS RELEASE
<TEXT><html><body>
We expect gross margin to be between 41.5% and 42.5% next quarter.
</body></html></TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""
