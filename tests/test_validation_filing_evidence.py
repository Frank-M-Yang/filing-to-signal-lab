"""Tests for frozen validation filing-index evidence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.filing_cover import FilingArtifactSnapshot
from sec_data.validation_filing_evidence import calculate_validation_filing_evidence


def test_exact_subject_sic_is_ready_without_outcomes(tmp_path: Path) -> None:
    targets = _targets()
    snapshot = _snapshot(tmp_path, _index_document("0000000001", "3571"))

    evidence, rejections, gates, report = calculate_validation_filing_evidence(
        targets, [snapshot], raw_root=tmp_path
    )

    assert evidence.loc[0, "filing_time_industry_ready"]
    assert evidence.loc[0, "sic"] == "3571"
    assert evidence.loc[0, "source_path_relative_to_raw_root"].startswith("raw/")
    assert rejections.empty
    assert report["outcome_values_loaded"] is False
    assert not gates.loc[gates["gate"].eq("return_outcome_read"), "passed"].item()


def test_wrong_subject_is_rejected_even_when_sic_exists(tmp_path: Path) -> None:
    targets = _targets()
    snapshot = _snapshot(tmp_path, _index_document("0000000002", "3571"))

    evidence, rejections, _, report = calculate_validation_filing_evidence(
        targets, [snapshot], raw_root=tmp_path
    )

    assert not evidence.loc[0, "filing_time_industry_ready"]
    assert evidence.loc[0, "rejection_reason"] == "filing_index_subject_cik_mismatch"
    assert len(rejections) == 1
    assert report["filing_time_industry_ready"] == 0


def test_missing_snapshot_remains_missing_and_is_not_replaced(tmp_path: Path) -> None:
    evidence, rejections, _, report = calculate_validation_filing_evidence(
        _targets(), [], raw_root=tmp_path
    )

    assert evidence.loc[0, "evidence_status"] == "not_collected"
    assert evidence.loc[0, "rejection_reason"] == "missing_preserved_index_artifact"
    assert len(rejections) == 1
    assert report["index_artifacts_preserved"] == 0


def test_outcome_columns_are_rejected_before_parsing(tmp_path: Path) -> None:
    targets = _targets()
    targets["market_excess_return_arithmetic"] = 0.01

    with pytest.raises(FilingCatalogError, match="contain outcomes"):
        calculate_validation_filing_evidence(targets, [], raw_root=tmp_path)


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": "event-hash",
                "acquisition_order": 1,
                "market_event_id": "event-hash",
                "validation_event_id": "validation-1",
                "cik": "0000000001",
                "accession_number": "0000000001-20-000001",
                "accepted_at": "2020-05-01T20:30:00Z",
                "index_url": (
                    "https://www.sec.gov/Archives/edgar/data/1/000000000120000001/"
                    "0000000001-20-000001-index.html"
                ),
                "selection_uses_return_values": False,
            }
        ]
    )


def _snapshot(tmp_path: Path, content: bytes) -> FilingArtifactSnapshot:
    digest = hashlib.sha256(content).hexdigest()
    path = tmp_path / "raw" / "index" / digest[:2] / f"{digest}.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return FilingArtifactSnapshot(
        target_id="event-hash",
        artifact_kind="index",
        requested_url=(
            "https://www.sec.gov/Archives/edgar/data/1/000000000120000001/"
            "0000000001-20-000001-index.html"
        ),
        response_url=(
            "https://www.sec.gov/Archives/edgar/data/1/000000000120000001/"
            "0000000001-20-000001-index.html"
        ),
        sha256=digest,
        fetched_at=datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        path=str(path.resolve()),
        bytes=len(content),
        status_code=200,
        headers={"content-type": "text/html"},
    )


def _index_document(cik: str, sic: str) -> bytes:
    return f"""<html><body>
    <div class="companyInfo"><span class="companyName">DEMO CORPORATION (Filer)
    <acronym title="Central Index Key">CIK</acronym>:
    <a href="/cgi-bin/browse-edgar?CIK={cik}&amp;action=getcompany">{cik}</a>
    </span><p>Type: <strong>10-Q</strong><br />
    <acronym title="Standard Industrial Code">SIC</acronym>:
    <b><a href="/cgi-bin/browse-edgar?action=getcompany&amp;SIC={sic}">{sic}</a></b>
    Electronic Computers</p></div>
    </body></html>""".encode()
