"""Preserve SEC primary documents needed to audit revision-tail events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.client import SecClient, normalize_cik
from sec_data.pipeline import RawSecStore

ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def main() -> int:
    """Download unique original and later filings for the 95% revision tail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tail-events",
        type=Path,
        default=Path(
            "data/processed/global500_2026/revision_robustness_v1/revision_tail_events.csv"
        ),
    )
    parser.add_argument("--sec-root", type=Path, default=Path("data/sec"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/revision_tail_v1"))
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    args = parser.parse_args()

    tail_sha256 = _sha256_file(args.tail_events)
    tail = pd.read_csv(args.tail_events, dtype={"company_id": "string"})
    exceeded = _as_bool(tail["exceeded_primary_buffer"])
    events = tail.loc[exceeded].copy()
    if events.empty:
        raise ValueError("No primary-buffer tail events were found")

    filing_index = _filing_index(args.sec_root / "normalized", events["company_id"].unique())
    requests = _document_requests(events, filing_index)
    client = SecClient(args.user_agent, request_interval_seconds=0.20)
    store = RawSecStore(args.sec_root / "raw_filings")
    records: list[dict[str, Any]] = []
    for request in requests.itertuples(index=False):
        company_id = str(request.company_id)
        accession_number = str(request.accession_number)
        response = client.get(str(request.source_url))
        snapshot = store.preserve(
            response,
            cik=company_id,
            kind=f"filing-{accession_number.replace('-', '')}",
            suffix=".html",
        )
        records.append(
            {
                "company_id": company_id,
                "accession_number": accession_number,
                "form": request.form,
                "accepted_at": request.accepted_at,
                "primary_document": request.primary_document,
                "document_role": request.document_role,
                "source_url": snapshot.source_url,
                "fetched_at": snapshot.fetched_at.isoformat(),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "sha256": snapshot.sha256,
                "bytes": len(response.content),
                "raw_path": _portable_path(snapshot.path, fallback_root=args.sec_root),
            }
        )

    manifest = pd.DataFrame(records).sort_values(["company_id", "accepted_at"])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "filing_document_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    report = {
        "audit_version": "revision-tail-filing-snapshots-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tail_events_path": _portable_path(args.tail_events),
        "tail_events_sha256": tail_sha256,
        "tail_event_count": len(events),
        "unique_company_count": int(events["company_id"].nunique()),
        "unique_filing_documents": len(manifest),
        "manifest_sha256": _sha256_file(manifest_path),
        "source": "SEC EDGAR primary filing documents",
    }
    (args.output / "collection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"DONE tail_events={len(events)} filing_documents={len(manifest)} "
        f"companies={events['company_id'].nunique()}",
        flush=True,
    )
    return 0


def _portable_path(path: Path, fallback_root: Path | None = None) -> str:
    """Return a portable path without recording a user's absolute workspace path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        if fallback_root is not None:
            try:
                relative = resolved.relative_to(fallback_root.resolve())
                return (Path(fallback_root.name) / relative).as_posix()
            except ValueError:
                pass
        return path.name


def _filing_index(normalized_root: Path, company_ids: Any) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_cik in company_ids:
        cik = normalize_cik(str(raw_cik))
        paths = sorted((normalized_root / cik).glob("*/filings.csv"))
        if not paths:
            raise FileNotFoundError(f"No normalized SEC filings are available for CIK {cik}")
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        frame["company_id"] = cik
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["company_id", "accession_number"], keep="last"
    )


def _document_requests(events: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    roles = []
    for role, column in (
        ("original", "filing_accession"),
        ("later_comparative", "latest_filing_accession"),
    ):
        frame = events[["company_id", column]].rename(columns={column: "accession_number"})
        frame["document_role"] = role
        roles.append(frame)
    requests = pd.concat(roles, ignore_index=True).drop_duplicates(
        ["company_id", "accession_number"]
    )
    requests = requests.merge(
        filings[
            [
                "company_id",
                "accession_number",
                "form",
                "accepted_at",
                "primary_document",
            ]
        ],
        on=["company_id", "accession_number"],
        how="left",
        validate="one_to_one",
    )
    if requests[["form", "accepted_at", "primary_document"]].isna().any().any():
        missing = requests.loc[requests["primary_document"].isna(), "accession_number"].tolist()
        raise ValueError(f"Missing filing metadata for accessions: {missing}")
    requests["source_url"] = requests.apply(
        lambda row: _filing_url(
            str(row["company_id"]),
            str(row["accession_number"]),
            str(row["primary_document"]),
        ),
        axis="columns",
    )
    return requests.sort_values(["company_id", "accepted_at"])


def _filing_url(cik: str, accession: str, primary_document: str) -> str:
    normalized_cik = str(int(normalize_cik(cik)))
    if re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) is None:
        raise ValueError(f"Unsafe accession: {accession}")
    if re.fullmatch(r"[A-Za-z0-9._-]+\.(?:htm|html)", primary_document) is None:
        raise ValueError(f"Unsafe primary document: {primary_document}")
    return ARCHIVES_URL.format(
        cik=normalized_cik,
        accession=accession.replace("-", ""),
        document=primary_document,
    )


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Boolean column contains invalid values")
    return normalized.eq("true")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
