"""Tests for accession-matched event-time SEC size features."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from sec_data.bulk import SecBulkSnapshot
from sec_data.size_features import build_event_time_size_features


def test_size_features_use_accession_acceptance_not_filed_date(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    filings = pd.DataFrame(
        [
            {
                "filing_id": "filing-1",
                "cik": "0000000001",
                "accession_number": "0000000001-20-000001",
                "form": "10-Q",
                "report_date": "2020-03-31",
                "accepted_at": "2020-05-01T20:30:00+00:00",
                "source_url": "https://www.sec.gov/filing-1",
            }
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "calendar_year": 2020,
                "representative_candidate": True,
            }
        ]
    )
    _write_catalog(catalog, filings, inventory)
    zip_path = tmp_path / "companyfacts.zip"
    companyfacts = {
        "cik": 1,
        "entityName": "Demo Corp",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2020-03-31",
                                "val": 125000000,
                                "accn": "0000000001-20-000001",
                                "form": "10-Q",
                                "filed": "2099-12-31",
                                "frame": "CY2020Q1I",
                            }
                        ]
                    }
                }
            }
        },
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("CIK0000000001.json", json.dumps(companyfacts))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    snapshot = SecBulkSnapshot(
        kind="companyfacts",
        sha256=digest,
        source_url="https://www.sec.gov/companyfacts.zip",
        final_url="https://www.sec.gov/companyfacts.zip",
        fetched_at="2026-08-13T00:00:00+00:00",
        path=str(zip_path),
        bytes=zip_path.stat().st_size,
        status_code=200,
        content_type="application/zip",
        content_length_header=zip_path.stat().st_size,
        etag=None,
        last_modified=None,
    )

    result = build_event_time_size_features(catalog, snapshot, output_root=tmp_path / "size")
    events = pd.read_parquet(result.output_root / "event_time_total_assets.parquet")
    anchors = pd.read_parquet(result.output_root / "issuer_year_size_anchors.parquet")

    assert str(events.loc[0, "accepted_at"]) == "2020-05-01 20:30:00+00:00"
    assert events.loc[0, "companyfacts_filed_date_not_used"] == "2099-12-31"
    assert events.loc[0, "total_assets_usd"] == 125000000
    assert anchors.loc[0, "asset_size_stratum"] == "asset_q4_largest"
    assert result.report["filed_date_used_as_availability"] is False
    assert result.report["issuer_year_size_coverage"] == 1.0


def _write_catalog(root: Path, filings: pd.DataFrame, inventory: pd.DataFrame) -> None:
    frames = {
        "filing_catalog.parquet": filings,
        "issuer_year_inventory.parquet": inventory,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for filename, frame in frames.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "catalog_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-catalog",
                "archive_coverage_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
