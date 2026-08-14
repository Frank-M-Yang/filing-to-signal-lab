"""End-to-end SEC download, raw preservation, cleaning, and dataset export."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.cleaning import (
    REJECTION_COLUMNS,
    compare_amendment_facts,
    parse_companyfacts,
    parse_submissions_documents,
)
from sec_data.client import SecClient, decode_json_object, normalize_cik
from sec_data.models import RawSnapshot, SecIngestionResult, SecResponse


class RawSecStore:
    """Preserve exact SEC payload bytes under content-addressed paths."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def preserve(
        self,
        response: SecResponse,
        *,
        cik: str,
        kind: str,
        suffix: str = ".json",
    ) -> RawSnapshot:
        """Write a source payload once and verify any existing file."""
        if re.fullmatch(r"\.[a-z0-9]+", suffix) is None:
            raise ValueError(f"Unsafe raw snapshot suffix: {suffix}")
        digest = hashlib.sha256(response.content).hexdigest()
        company_directory = self.root / normalize_cik(cik) / digest[:2]
        company_directory.mkdir(parents=True, exist_ok=True)
        target = company_directory / f"{kind}-{digest}{suffix}"
        if target.exists():
            if target.read_bytes() != response.content:
                raise RuntimeError(f"Content hash collision at {target}")
        else:
            with target.open("xb") as stream:
                stream.write(response.content)
        return RawSnapshot(
            sha256=digest,
            source_url=response.url,
            fetched_at=response.fetched_at,
            path=target.resolve(),
            kind=kind,
        )


class SecIngestionService:
    """Orchestrate a small-company SEC EDGAR research download."""

    def __init__(
        self,
        user_agent: str,
        *,
        data_root: Path = Path("data/sec"),
        client: SecClient | None = None,
    ) -> None:
        self.client = client or SecClient(user_agent)
        self.data_root = data_root
        self.raw_store = RawSecStore(data_root / "raw")

    def ingest(self, company: str, *, include_archives: bool = False) -> SecIngestionResult:
        """Download and clean one ticker or CIK into an auditable local snapshot."""
        query = company.strip()
        if not query:
            raise ValueError("Company ticker or CIK is required")
        snapshots: list[RawSnapshot] = []
        ticker_hint: str | None = None
        company_hint: str | None = None
        if query.isdigit():
            cik = normalize_cik(query)
        else:
            cik, ticker_hint, company_hint, ticker_response = self.client.resolve_ticker(query)
            snapshots.append(self.raw_store.preserve(ticker_response, cik=cik, kind="ticker-map"))

        submissions_response = self.client.get_submissions(cik)
        submissions_document = decode_json_object(
            submissions_response.content, source_url=submissions_response.url
        )
        submissions_snapshot = self.raw_store.preserve(
            submissions_response, cik=cik, kind="submissions"
        )
        snapshots.append(submissions_snapshot)
        company_name = str(submissions_document.get("name") or company_hint or cik)
        tickers = submissions_document.get("tickers")
        if ticker_hint is None and isinstance(tickers, list) and tickers:
            ticker_hint = str(tickers[0])
        submission_documents: list[tuple[dict[str, Any], str, str]] = [
            (
                submissions_document,
                "submissions.recent",
                submissions_snapshot.sha256,
            )
        ]

        archived_files = _archive_file_names(submissions_document)
        if include_archives:
            for archive_index, filename in enumerate(archived_files):
                archive_response = self.client.get_submission_archive(filename)
                archive_document = decode_json_object(
                    archive_response.content, source_url=archive_response.url
                )
                archive_snapshot = self.raw_store.preserve(
                    archive_response, cik=cik, kind=f"submissions-archive-{archive_index:03d}"
                )
                snapshots.append(archive_snapshot)
                submission_documents.append((archive_document, filename, archive_snapshot.sha256))

        parsed_at = datetime.now(timezone.utc)
        filings_result = parse_submissions_documents(
            submission_documents,
            cik=cik,
            ticker=ticker_hint,
            company_name=company_name,
            parsed_at=parsed_at,
        )
        companyfacts_response = self.client.get_companyfacts(cik)
        companyfacts_document = decode_json_object(
            companyfacts_response.content, source_url=companyfacts_response.url
        )
        companyfacts_snapshot = self.raw_store.preserve(
            companyfacts_response, cik=cik, kind="companyfacts"
        )
        snapshots.append(companyfacts_snapshot)
        facts_result = parse_companyfacts(
            companyfacts_document,
            filings_result.data,
            source_url=companyfacts_response.url,
            snapshot_hash=companyfacts_snapshot.sha256,
            parsed_at=parsed_at,
        )
        comparisons = compare_amendment_facts(facts_result.data)
        rejection_frames = [
            frame
            for frame in (filings_result.rejections, facts_result.rejections)
            if not frame.empty
        ]
        rejections = (
            pd.concat(rejection_frames, ignore_index=True)
            if rejection_frames
            else pd.DataFrame(columns=REJECTION_COLUMNS)
        ).reindex(columns=REJECTION_COLUMNS)
        dataset_version = _dataset_version(tuple(snapshots))
        metadata: dict[str, Any] = {
            "downloaded_at": parsed_at.isoformat(),
            "dataset_version": dataset_version,
            "include_archives": include_archives,
            "available_archive_files": len(archived_files),
            "downloaded_archive_files": len(archived_files) if include_archives else 0,
            "source": "SEC EDGAR public data.sec.gov APIs",
            "forms": ["10-K", "10-K/A", "10-Q", "10-Q/A"],
        }
        result = SecIngestionResult(
            cik=cik,
            ticker=ticker_hint,
            company_name=company_name,
            filings=filings_result.data,
            facts=facts_result.data,
            amendment_comparisons=comparisons,
            rejections=rejections,
            snapshots=tuple(snapshots),
            metadata=metadata,
        )
        normalized_directory = persist_normalized_result(result, self.data_root / "normalized")
        metadata["normalized_directory"] = str(normalized_directory.resolve())
        return SecIngestionResult(
            cik=result.cik,
            ticker=result.ticker,
            company_name=result.company_name,
            filings=result.filings,
            facts=result.facts,
            amendment_comparisons=result.amendment_comparisons,
            rejections=result.rejections,
            snapshots=result.snapshots,
            metadata=metadata,
        )


def persist_normalized_result(result: SecIngestionResult, root: Path) -> Path:
    """Persist deterministic cleaned CSVs and a compact manifest without overwriting versions."""
    dataset_version = str(result.metadata["dataset_version"])
    target = root / result.cik / dataset_version[:16]
    target.mkdir(parents=True, exist_ok=True)
    exports = {
        "filings.csv": result.filings,
        "facts.csv": result.facts,
        "amendment_comparisons.csv": result.amendment_comparisons,
        "rejections.csv": result.rejections,
    }
    for filename, frame in exports.items():
        path = target / filename
        if not path.exists():
            frame.to_csv(path, index=False)
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "cik": result.cik,
            "ticker": result.ticker,
            "company_name": result.company_name,
            "metadata": result.metadata,
            "snapshot_hashes": [snapshot.sha256 for snapshot in result.snapshots],
            "counts": {
                "filings": len(result.filings),
                "facts": len(result.facts),
                "amendment_comparisons": len(result.amendment_comparisons),
                "rejections": len(result.rejections),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return target


def _archive_file_names(document: dict[str, Any]) -> list[str]:
    filings = document.get("filings")
    if not isinstance(filings, dict) or not isinstance(filings.get("files"), list):
        return []
    names: list[str] = []
    for record in filings["files"]:
        if isinstance(record, dict) and isinstance(record.get("name"), str):
            names.append(record["name"])
    return names


def _dataset_version(snapshots: tuple[RawSnapshot, ...]) -> str:
    encoded = json.dumps(
        [(snapshot.kind, snapshot.sha256) for snapshot in snapshots],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
