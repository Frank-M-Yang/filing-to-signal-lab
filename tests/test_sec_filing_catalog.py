"""Tests for SEC-wide filing inventory and exit/stress taxonomy."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from sec_data.bulk import SecBulkSnapshot
from sec_data.filing_catalog import (
    FilingCatalogConfig,
    build_filing_catalog,
    load_catalog_table,
    parse_submission_entity,
)


def test_parse_submission_entity_keeps_event_time_and_separates_signals() -> None:
    tables = parse_submission_entity(
        _entity_document(),
        source_member="CIK0000000001.json",
        source_zip_sha256="a" * 64,
        config=FilingCatalogConfig(start_year=2020, end_year=2022, batch_size=1_000),
    )

    assert [row["form"] for row in tables.filings] == ["10-Q", "8-K", "25"]
    assert tables.filings[0]["accepted_at"] == "2020-05-01T20:30:00+00:00"
    assert tables.filings[0]["accession_prefix"] == "0000000001"
    assert tables.issuer["metadata_timepoint_status"].endswith("do_not_backfill_history")
    signal_types = {row["signal_type"] for row in tables.filing_signals}
    assert signal_types == {
        "listing_compliance_or_delisting_notice",
        "financial_statement_non_reliance",
        "formal_delisting",
    }
    assert not tables.rejections


def test_catalog_builds_immutable_parquet_inventory(tmp_path: Path) -> None:
    zip_path = tmp_path / "submissions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("CIK0000000001.json", json.dumps(_entity_document()))
        archive.writestr("CIK0000000001-submissions-001.json", json.dumps(_archive_document()))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    snapshot = SecBulkSnapshot(
        kind="submissions",
        sha256=digest,
        source_url="https://www.sec.gov/submissions.zip",
        final_url="https://www.sec.gov/submissions.zip",
        fetched_at="2026-08-13T00:00:00+00:00",
        path=str(zip_path),
        bytes=zip_path.stat().st_size,
        status_code=200,
        content_type="application/zip",
        content_length_header=zip_path.stat().st_size,
        etag=None,
        last_modified=None,
    )

    result = build_filing_catalog(
        snapshot,
        output_root=tmp_path / "processed",
        config=FilingCatalogConfig(start_year=2020, end_year=2022, batch_size=1_000),
        progress_every=0,
    )

    filings = load_catalog_table(result.output_root, "filing_catalog.parquet")
    signals = load_catalog_table(result.output_root, "filing_signals.parquet")
    inventory = load_catalog_table(result.output_root, "issuer_year_inventory.parquet")
    archive_audit = load_catalog_table(result.output_root, "archive_coverage_audit.parquet")
    assert len(filings) == 4
    assert len(signals) == 3
    assert inventory["representative_candidate"].sum() == 2
    assert inventory["formal_delistings"].sum() == 1
    assert result.report["selected_filings"] == 4
    assert result.report["signal_counts"]["financial_statement_non_reliance"] == 1
    assert archive_audit["coverage_status"].tolist() == ["resolved_from_bulk_snapshot"]
    assert result.report["archive_shards_resolved"] == 1
    assert result.report["archive_coverage_complete"]


def _entity_document() -> dict[str, object]:
    return {
        "cik": 1,
        "name": "Historical Demo Corp",
        "entityType": "operating",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "fiscalYearEnd": "1231",
        "stateOfIncorporation": "DE",
        "tickers": ["NOW"],
        "exchanges": ["Nasdaq"],
        "formerNames": [{"name": "Old Demo Corp", "from": "2018-01-01", "to": "2020-01-01"}],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000000001-20-000001",
                    "0000000001-21-000002",
                    "0000000001-22-000003",
                    "0000000001-19-000004",
                ],
                "filingDate": ["2020-05-01", "2021-06-01", "2022-07-01", "2019-01-01"],
                "reportDate": ["2020-03-31", "2021-05-31", "2022-06-30", "2018-12-31"],
                "acceptanceDateTime": [
                    "20200501203000",
                    "20210601150000",
                    "20220701120000",
                    "20190101120000",
                ],
                "form": ["10-Q", "8-K", "25", "10-K"],
                "primaryDocument": ["q1.htm", "event.htm", "form25.htm", "old.htm"],
                "items": ["", "3.01;4.02", "", ""],
                "isXBRL": [1, 1, 0, 1],
                "isInlineXBRL": [1, 1, 0, 0],
            },
            "files": [
                {
                    "name": "CIK0000000001-submissions-001.json",
                    "filingCount": 10,
                    "filingFrom": "2019-01-01",
                    "filingTo": "2021-12-31",
                }
            ],
        },
    }


def _archive_document() -> dict[str, list[object]]:
    return {
        "accessionNumber": ["0000000001-21-000099"],
        "filingDate": ["2021-03-01"],
        "reportDate": ["2020-12-31"],
        "acceptanceDateTime": ["2021-03-01T20:00:00.000Z"],
        "form": ["10-K"],
        "primaryDocument": ["archive-10k.htm"],
        "items": [""],
        "isXBRL": [1],
        "isInlineXBRL": [1],
    }
