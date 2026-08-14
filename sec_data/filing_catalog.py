"""Historical SEC filing catalog and exit/stress-signal extraction."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from sec_data.bulk import SecBulkSnapshot
from sec_data.client import normalize_cik

ACCESSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{10}-\d{2}-\d{6}$")
CIK_MEMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:^|/)CIK(\d{10})\.json$")
ITEM_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(\d\.\d{2})(?!\d)")
PERIODIC_FORMS: Final[frozenset[str]] = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})
OTHER_SELECTED_FORMS: Final[frozenset[str]] = frozenset(
    {
        "8-K",
        "8-K/A",
        "NT 10-Q",
        "NT 10-K",
        "25",
        "25-NSE",
        "DEFM14A",
        "SC 13E3",
        "SC 13E3/A",
    }
)


class FilingCatalogError(ValueError):
    """Raised when a bulk submissions catalog violates its source contract."""


@dataclass(frozen=True)
class FilingCatalogConfig:
    """Frozen first-pass filing inventory window and output settings."""

    start_year: int = 2016
    end_year: int = 2025
    batch_size: int = 50_000
    include_overlapping_archives: bool = True
    version: str = "sec-filing-catalog-v2"

    def __post_init__(self) -> None:
        if not 1994 <= self.start_year <= self.end_year <= 2100:
            raise ValueError("Filing catalog year range is invalid")
        if self.batch_size < 1_000:
            raise ValueError("Filing catalog batch_size must be at least 1,000")


@dataclass(frozen=True)
class SubmissionEntityTables:
    """Normalized rows extracted from one CIK submissions document."""

    filings: list[dict[str, Any]]
    issuer: dict[str, Any]
    former_names: list[dict[str, Any]]
    archive_shards: list[dict[str, Any]]
    filing_signals: list[dict[str, Any]]
    rejections: list[dict[str, Any]]


@dataclass(frozen=True)
class FilingCatalogBuildResult:
    """Immutable catalog directory and its compact build report."""

    output_root: Path
    report: dict[str, Any]


FILING_SCHEMA = pa.schema(
    [
        ("filing_id", pa.string()),
        ("cik", pa.string()),
        ("accession_prefix", pa.string()),
        ("company_name_snapshot", pa.string()),
        ("entity_type_snapshot", pa.string()),
        ("sic_snapshot", pa.string()),
        ("accession_number", pa.string()),
        ("form", pa.string()),
        ("base_form", pa.string()),
        ("is_amendment", pa.bool_()),
        ("filing_date", pa.string()),
        ("report_date", pa.string()),
        ("accepted_at", pa.string()),
        ("primary_document", pa.string()),
        ("items", pa.string()),
        ("is_xbrl", pa.bool_()),
        ("is_inline_xbrl", pa.bool_()),
        ("source_url", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("source_member", pa.string()),
        ("source_row_index", pa.int64()),
    ]
)
ISSUER_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("company_name_snapshot", pa.string()),
        ("entity_type_snapshot", pa.string()),
        ("sic_snapshot", pa.string()),
        ("sic_description_snapshot", pa.string()),
        ("fiscal_year_end_snapshot", pa.string()),
        ("state_of_incorporation_snapshot", pa.string()),
        ("current_tickers_json", pa.string()),
        ("current_exchanges_json", pa.string()),
        ("metadata_timepoint_status", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("source_member", pa.string()),
    ]
)
FORMER_NAME_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("former_name", pa.string()),
        ("name_from", pa.string()),
        ("name_to", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("source_member", pa.string()),
    ]
)
ARCHIVE_SHARD_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("archive_name", pa.string()),
        ("filing_count", pa.int64()),
        ("filing_from", pa.string()),
        ("filing_to", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("source_member", pa.string()),
    ]
)
ARCHIVE_COVERAGE_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("archive_name", pa.string()),
        ("filing_count", pa.int64()),
        ("filing_from", pa.string()),
        ("filing_to", pa.string()),
        ("overlaps_catalog_window", pa.bool_()),
        ("archive_present_in_bulk_zip", pa.bool_()),
        ("archive_parsed", pa.bool_()),
        ("archive_member_sha256", pa.string()),
        ("selected_filings", pa.int64()),
        ("selected_signals", pa.int64()),
        ("duplicate_filings_skipped", pa.int64()),
        ("rejections", pa.int64()),
        ("coverage_status", pa.string()),
        ("source_zip_sha256", pa.string()),
    ]
)
SIGNAL_SCHEMA = pa.schema(
    [
        ("signal_id", pa.string()),
        ("cik", pa.string()),
        ("accession_number", pa.string()),
        ("form", pa.string()),
        ("accepted_at", pa.string()),
        ("event_family", pa.string()),
        ("signal_type", pa.string()),
        ("is_exit_candidate", pa.bool_()),
        ("listing_status_effect", pa.string()),
        ("reporting_status_effect", pa.string()),
        ("operating_status_effect", pa.string()),
        ("signal_confidence", pa.string()),
        ("source_items", pa.string()),
        ("source_url", pa.string()),
        ("source_zip_sha256", pa.string()),
    ]
)
REJECTION_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("stage", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
        ("source_member", pa.string()),
        ("source_row_index", pa.int64()),
        ("source_zip_sha256", pa.string()),
    ]
)
ISSUER_YEAR_SCHEMA = pa.schema(
    [
        ("cik", pa.string()),
        ("calendar_year", pa.int64()),
        ("company_name_snapshot", pa.string()),
        ("entity_type_snapshot", pa.string()),
        ("sic_snapshot", pa.string()),
        ("original_10q_filings", pa.int64()),
        ("original_10k_filings", pa.int64()),
        ("periodic_amendments", pa.int64()),
        ("late_filing_notices", pa.int64()),
        ("non_reliance_notices", pa.int64()),
        ("auditor_change_notices", pa.int64()),
        ("listing_notices", pa.int64()),
        ("formal_delistings", pa.int64()),
        ("reporting_deregistrations", pa.int64()),
        ("bankruptcy_notices", pa.int64()),
        ("acquisition_candidates", pa.int64()),
        ("representative_candidate", pa.bool_()),
        ("current_metadata_only", pa.bool_()),
    ]
)


def parse_submission_entity(
    document: dict[str, Any],
    *,
    source_member: str,
    source_zip_sha256: str,
    config: FilingCatalogConfig | None = None,
) -> SubmissionEntityTables:
    """Extract selected filings and status signals from one bulk CIK document."""
    cfg = config or FilingCatalogConfig()
    try:
        cik = normalize_cik(str(document.get("cik", "")))
    except ValueError as exc:
        raise FilingCatalogError(f"Invalid submissions CIK in {source_member}: {exc}") from exc
    company_name = _text(document.get("name"))
    entity_type = _text(document.get("entityType"))
    sic = _text(document.get("sic"))
    issuer = {
        "cik": cik,
        "company_name_snapshot": company_name,
        "entity_type_snapshot": entity_type,
        "sic_snapshot": sic,
        "sic_description_snapshot": _text(document.get("sicDescription")),
        "fiscal_year_end_snapshot": _text(document.get("fiscalYearEnd")),
        "state_of_incorporation_snapshot": _text(document.get("stateOfIncorporation")),
        "current_tickers_json": _json_array(document.get("tickers")),
        "current_exchanges_json": _json_array(document.get("exchanges")),
        "metadata_timepoint_status": "bulk_snapshot_current_metadata_do_not_backfill_history",
        "source_zip_sha256": source_zip_sha256,
        "source_member": source_member,
    }
    former_names = _former_name_rows(document, cik, source_member, source_zip_sha256)
    archive_shards = _archive_shard_rows(document, cik, source_member, source_zip_sha256)
    recent = (
        document.get("filings", {}).get("recent")
        if isinstance(document.get("filings"), dict)
        else None
    )
    if not isinstance(recent, dict):
        rejection = _rejection(
            cik,
            "submissions",
            "missing_recent_filings",
            "filings.recent is not an object",
            source_member,
            None,
            source_zip_sha256,
        )
        return SubmissionEntityTables([], issuer, former_names, archive_shards, [], [rejection])
    accessions = recent.get("accessionNumber")
    forms = recent.get("form")
    if (
        not isinstance(accessions, list)
        or not isinstance(forms, list)
        or len(accessions) != len(forms)
    ):
        rejection = _rejection(
            cik,
            "submissions",
            "invalid_required_arrays",
            "accessionNumber and form must be equal-length arrays",
            source_member,
            None,
            source_zip_sha256,
        )
        return SubmissionEntityTables([], issuer, former_names, archive_shards, [], [rejection])

    filings: list[dict[str, Any]] = []
    filing_signals: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index, (raw_accession, raw_form) in enumerate(zip(accessions, forms, strict=True)):
        form = _text(raw_form).upper()
        if not _selected_form(form):
            continue
        filing_date = _array_text(recent, "filingDate", index)
        quick_year = _leading_year(filing_date)
        if quick_year is not None and not cfg.start_year <= quick_year <= cfg.end_year:
            continue
        accession = _text(raw_accession)
        accepted_at = _acceptance_timestamp(_array_value(recent, "acceptanceDateTime", index))
        if not ACCESSION_PATTERN.fullmatch(accession) or accepted_at is None:
            rejections.append(
                _rejection(
                    cik,
                    "submissions",
                    "invalid_filing_identity_or_acceptance_time",
                    f"form={form} accession={accession}",
                    source_member,
                    index,
                    source_zip_sha256,
                )
            )
            continue
        accepted_year = int(accepted_at[:4])
        if not cfg.start_year <= accepted_year <= cfg.end_year:
            continue
        primary_document = _array_text(recent, "primaryDocument", index)
        items = _array_text(recent, "items", index)
        archive_base = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        )
        source_url = f"{archive_base}/{primary_document}" if primary_document else archive_base
        filing = {
            "filing_id": _stable_hash({"cik": cik, "accession": accession}),
            "cik": cik,
            "accession_prefix": accession[:10],
            "company_name_snapshot": company_name,
            "entity_type_snapshot": entity_type,
            "sic_snapshot": sic,
            "accession_number": accession,
            "form": form,
            "base_form": form.removesuffix("/A"),
            "is_amendment": form.endswith("/A"),
            "filing_date": filing_date,
            "report_date": _array_text(recent, "reportDate", index),
            "accepted_at": accepted_at,
            "primary_document": primary_document,
            "items": items,
            "is_xbrl": _array_bool(recent, "isXBRL", index),
            "is_inline_xbrl": _array_bool(recent, "isInlineXBRL", index),
            "source_url": source_url,
            "source_zip_sha256": source_zip_sha256,
            "source_member": source_member,
            "source_row_index": index,
        }
        filings.append(filing)
        filing_signals.extend(_signals_for_filing(filing))
    return SubmissionEntityTables(
        filings,
        issuer,
        former_names,
        archive_shards,
        filing_signals,
        rejections,
    )


def build_filing_catalog(
    snapshot: SecBulkSnapshot,
    *,
    output_root: Path = Path("data/processed/sec_universe"),
    config: FilingCatalogConfig | None = None,
    progress_every: int = 25_000,
) -> FilingCatalogBuildResult:
    """Stream primary and overlapping archived submissions into an immutable catalog."""
    cfg = config or FilingCatalogConfig()
    if snapshot.kind != "submissions":
        raise FilingCatalogError("Filing catalog requires a submissions bulk snapshot")
    source = Path(snapshot.path)
    if not source.is_file() or _sha256_file(source) != snapshot.sha256:
        raise FilingCatalogError("Submissions snapshot path does not tie to its SHA-256")
    version_payload = json.dumps(
        {"snapshot": snapshot.sha256, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = f"{cfg.version}-{hashlib.sha256(version_payload).hexdigest()[:16]}"
    target = output_root / dataset_version
    manifest_path = target / "output_manifest.json"
    if manifest_path.exists():
        existing_report = json.loads((target / "catalog_report.json").read_text(encoding="utf-8"))
        return FilingCatalogBuildResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete catalog directory already exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    writers = {
        "filing_catalog.parquet": _ParquetRowWriter(
            staging / "filing_catalog.parquet", FILING_SCHEMA, cfg.batch_size
        ),
        "issuer_registry_snapshot.parquet": _ParquetRowWriter(
            staging / "issuer_registry_snapshot.parquet", ISSUER_SCHEMA, cfg.batch_size
        ),
        "former_names.parquet": _ParquetRowWriter(
            staging / "former_names.parquet", FORMER_NAME_SCHEMA, cfg.batch_size
        ),
        "archive_shard_index.parquet": _ParquetRowWriter(
            staging / "archive_shard_index.parquet", ARCHIVE_SHARD_SCHEMA, cfg.batch_size
        ),
        "filing_signals.parquet": _ParquetRowWriter(
            staging / "filing_signals.parquet", SIGNAL_SCHEMA, cfg.batch_size
        ),
        "catalog_rejections.parquet": _ParquetRowWriter(
            staging / "catalog_rejections.parquet", REJECTION_SCHEMA, cfg.batch_size
        ),
    }
    issuer_year: dict[tuple[str, int], dict[str, Any]] = {}
    form_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    archive_window_rows: list[dict[str, Any]] = []
    archive_issuers: dict[str, dict[str, Any]] = {}
    archive_coverage_rows: list[dict[str, Any]] = []
    seen_filing_keys: set[tuple[str, str]] = set()
    member_count = 0
    parsed_entities = 0
    invalid_json_members = 0
    duplicate_filings_skipped = 0
    with zipfile.ZipFile(source) as archive:
        archive_names = set(archive.namelist())
        for member in archive.infolist():
            matched = CIK_MEMBER_PATTERN.search(member.filename.replace("\\", "/"))
            if matched is None or member.is_dir():
                continue
            member_count += 1
            try:
                with archive.open(member) as stream:
                    document = json.load(stream)
                if not isinstance(document, dict):
                    raise ValueError("top-level JSON is not an object")
                tables = parse_submission_entity(
                    document,
                    source_member=member.filename,
                    source_zip_sha256=snapshot.sha256,
                    config=cfg,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                invalid_json_members += 1
                writers["catalog_rejections.parquet"].append(
                    _rejection(
                        matched.group(1),
                        "bulk_zip",
                        "invalid_entity_json",
                        str(exc),
                        member.filename,
                        None,
                        snapshot.sha256,
                    )
                )
                continue
            parsed_entities += 1
            writers["issuer_registry_snapshot.parquet"].append(tables.issuer)
            writers["former_names.parquet"].extend(tables.former_names)
            writers["archive_shard_index.parquet"].extend(tables.archive_shards)
            for shard in tables.archive_shards:
                if _date_ranges_overlap(
                    str(shard["filing_from"]),
                    str(shard["filing_to"]),
                    f"{cfg.start_year}-01-01",
                    f"{cfg.end_year}-12-31",
                ):
                    archive_window_rows.append(shard)
                    archive_issuers[str(shard["cik"])] = tables.issuer
            writers["catalog_rejections.parquet"].extend(tables.rejections)
            _, _, duplicates = _append_unique_tables(
                tables,
                issuer=tables.issuer,
                writers=writers,
                issuer_year=issuer_year,
                form_counts=form_counts,
                signal_counts=signal_counts,
                seen_filing_keys=seen_filing_keys,
                source_zip_sha256=snapshot.sha256,
            )
            duplicate_filings_skipped += duplicates
            if progress_every > 0 and member_count % progress_every == 0:
                print(
                    f"CATALOG members={member_count:,} entities={parsed_entities:,} "
                    f"filings={writers['filing_catalog.parquet'].rows:,}",
                    flush=True,
                )
        unique_archive_rows = {
            (str(row["cik"]), str(row["archive_name"])): row for row in archive_window_rows
        }
        if cfg.include_overlapping_archives:
            for archive_index, ((cik, archive_name), shard) in enumerate(
                sorted(unique_archive_rows.items()), start=1
            ):
                present = archive_name in archive_names
                audit_row = {
                    "cik": cik,
                    "archive_name": archive_name,
                    "filing_count": int(shard["filing_count"]),
                    "filing_from": shard["filing_from"],
                    "filing_to": shard["filing_to"],
                    "overlaps_catalog_window": True,
                    "archive_present_in_bulk_zip": present,
                    "archive_parsed": False,
                    "archive_member_sha256": "",
                    "selected_filings": 0,
                    "selected_signals": 0,
                    "duplicate_filings_skipped": 0,
                    "rejections": 0,
                    "coverage_status": "missing_from_bulk_zip",
                    "source_zip_sha256": snapshot.sha256,
                }
                if present:
                    member_bytes = archive.read(archive_name)
                    audit_row["archive_member_sha256"] = hashlib.sha256(member_bytes).hexdigest()
                    try:
                        archive_document = json.loads(member_bytes)
                        if not isinstance(archive_document, dict):
                            raise ValueError("archive top-level JSON is not an object")
                        issuer = archive_issuers[cik]
                        archive_tables = parse_submission_entity(
                            _archive_entity_document(archive_document, issuer),
                            source_member=archive_name,
                            source_zip_sha256=snapshot.sha256,
                            config=cfg,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
                        writers["catalog_rejections.parquet"].append(
                            _rejection(
                                cik,
                                "submissions_archive",
                                "invalid_archive_json",
                                str(exc),
                                archive_name,
                                None,
                                snapshot.sha256,
                            )
                        )
                        audit_row["rejections"] = 1
                        audit_row["coverage_status"] = "archive_parse_failed"
                    else:
                        writers["catalog_rejections.parquet"].extend(archive_tables.rejections)
                        selected_filings, selected_signals, duplicates = _append_unique_tables(
                            archive_tables,
                            issuer=issuer,
                            writers=writers,
                            issuer_year=issuer_year,
                            form_counts=form_counts,
                            signal_counts=signal_counts,
                            seen_filing_keys=seen_filing_keys,
                            source_zip_sha256=snapshot.sha256,
                        )
                        duplicate_filings_skipped += duplicates
                        audit_row.update(
                            {
                                "archive_parsed": True,
                                "selected_filings": selected_filings,
                                "selected_signals": selected_signals,
                                "duplicate_filings_skipped": duplicates,
                                "rejections": len(archive_tables.rejections),
                                "coverage_status": "resolved_from_bulk_snapshot",
                            }
                        )
                archive_coverage_rows.append(audit_row)
                if progress_every > 0 and archive_index % 250 == 0:
                    print(
                        f"ARCHIVES parsed={archive_index:,}/{len(unique_archive_rows):,} "
                        f"filings={writers['filing_catalog.parquet'].rows:,}",
                        flush=True,
                    )
        else:
            for (cik, archive_name), shard in sorted(unique_archive_rows.items()):
                archive_coverage_rows.append(
                    {
                        "cik": cik,
                        "archive_name": archive_name,
                        "filing_count": int(shard["filing_count"]),
                        "filing_from": shard["filing_from"],
                        "filing_to": shard["filing_to"],
                        "overlaps_catalog_window": True,
                        "archive_present_in_bulk_zip": archive_name in archive_names,
                        "archive_parsed": False,
                        "archive_member_sha256": "",
                        "selected_filings": 0,
                        "selected_signals": 0,
                        "duplicate_filings_skipped": 0,
                        "rejections": 0,
                        "coverage_status": "parsing_disabled",
                        "source_zip_sha256": snapshot.sha256,
                    }
                )
    for writer in writers.values():
        writer.close()
    inventory_writer = _ParquetRowWriter(
        staging / "issuer_year_inventory.parquet", ISSUER_YEAR_SCHEMA, cfg.batch_size
    )
    inventory_writer.extend(issuer_year.values())
    inventory_writer.close()
    representative_ciks = {
        cik for (cik, _year), row in issuer_year.items() if row["representative_candidate"]
    }
    archive_coverage_writer = _ParquetRowWriter(
        staging / "archive_coverage_audit.parquet", ARCHIVE_COVERAGE_SCHEMA, cfg.batch_size
    )
    archive_coverage_writer.extend(archive_coverage_rows)
    archive_coverage_writer.close()
    artifacts = {
        path.name: {
            "rows": _parquet_rows(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in staging.glob("*.parquet")
    }
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "source_snapshot": asdict(snapshot),
        "zip_cik_members": member_count,
        "parsed_entities": parsed_entities,
        "invalid_json_members": invalid_json_members,
        "selected_filings": writers["filing_catalog.parquet"].rows,
        "filing_signals": writers["filing_signals.parquet"].rows,
        "issuer_years": len(issuer_year),
        "representative_candidate_ciks": len(representative_ciks),
        "archive_shards_overlapping_window": len(archive_coverage_rows),
        "archive_ciks_overlapping_window": len({str(row["cik"]) for row in archive_coverage_rows}),
        "archive_shards_resolved": sum(
            row["coverage_status"] == "resolved_from_bulk_snapshot" for row in archive_coverage_rows
        ),
        "archive_shards_unresolved": sum(
            row["coverage_status"] != "resolved_from_bulk_snapshot" for row in archive_coverage_rows
        ),
        "archive_coverage_complete": bool(archive_coverage_rows)
        and all(
            row["coverage_status"] == "resolved_from_bulk_snapshot" for row in archive_coverage_rows
        ),
        "duplicate_filings_skipped": duplicate_filings_skipped,
        "form_counts": dict(sorted(form_counts.items())),
        "signal_counts": dict(sorted(signal_counts.items())),
        "artifacts": artifacts,
        "evidence_boundary": (
            "Ticker, exchange, name, and SIC fields in issuer_registry_snapshot are current "
            "bulk-snapshot metadata and must not be backfilled into historical events."
        ),
    }
    (staging / "catalog_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_manifest = {
        "dataset_version": dataset_version,
        "source_sha256": snapshot.sha256,
        "artifacts": artifacts,
        "report_sha256": _sha256_file(staging / "catalog_report.json"),
    }
    (staging / "output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    staging.replace(target)
    return FilingCatalogBuildResult(target.resolve(), report)


def _append_unique_tables(
    tables: SubmissionEntityTables,
    *,
    issuer: dict[str, Any],
    writers: dict[str, _ParquetRowWriter],
    issuer_year: dict[tuple[str, int], dict[str, Any]],
    form_counts: Counter[str],
    signal_counts: Counter[str],
    seen_filing_keys: set[tuple[str, str]],
    source_zip_sha256: str,
) -> tuple[int, int, int]:
    """Append event-unique filing rows and only the signals tied to newly added filings."""
    accepted_accessions: set[str] = set()
    duplicates = 0
    for filing in tables.filings:
        key = (str(filing["cik"]), str(filing["accession_number"]))
        if key in seen_filing_keys:
            duplicates += 1
            writers["catalog_rejections.parquet"].append(
                _rejection(
                    key[0],
                    "catalog_merge",
                    "duplicate_subject_accession",
                    f"Duplicate subject CIK and accession: {key}",
                    str(filing["source_member"]),
                    int(filing["source_row_index"]),
                    source_zip_sha256,
                )
            )
            continue
        seen_filing_keys.add(key)
        accepted_accessions.add(key[1])
        writers["filing_catalog.parquet"].append(filing)
        form_counts[str(filing["form"])] += 1
        _update_issuer_year(issuer_year, issuer, filing, None)
    selected_signals = 0
    for signal in tables.filing_signals:
        if str(signal["accession_number"]) not in accepted_accessions:
            continue
        writers["filing_signals.parquet"].append(signal)
        signal_counts[str(signal["signal_type"])] += 1
        _update_issuer_year(issuer_year, issuer, None, signal)
        selected_signals += 1
    return len(accepted_accessions), selected_signals, duplicates


def _archive_entity_document(
    archive_document: dict[str, Any], issuer: dict[str, Any]
) -> dict[str, Any]:
    """Wrap one archived columnar shard with its subject entity metadata."""
    return {
        "cik": int(str(issuer["cik"])),
        "name": issuer["company_name_snapshot"],
        "entityType": issuer["entity_type_snapshot"],
        "sic": issuer["sic_snapshot"],
        "sicDescription": issuer["sic_description_snapshot"],
        "fiscalYearEnd": issuer["fiscal_year_end_snapshot"],
        "stateOfIncorporation": issuer["state_of_incorporation_snapshot"],
        "filings": {"recent": archive_document, "files": []},
    }


def load_catalog_table(root: Path, filename: str) -> pd.DataFrame:
    """Load one verified catalog Parquet table for downstream cohort construction."""
    manifest = json.loads((root / "output_manifest.json").read_text(encoding="utf-8"))
    artifact = manifest.get("artifacts", {}).get(filename)
    path = root / filename
    if not isinstance(artifact, dict) or not path.is_file():
        raise FilingCatalogError(f"Catalog artifact is not declared: {filename}")
    if _sha256_file(path) != artifact.get("sha256"):
        raise FilingCatalogError(f"Catalog artifact hash mismatch: {filename}")
    return pd.read_parquet(path)


class _ParquetRowWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        self.rows += 1
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def extend(self, rows: list[dict[str, Any]] | Any) -> None:
        for row in rows:
            self.append(row)

    def close(self) -> None:
        self._flush()
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path)
        else:
            self.writer.close()

    def _flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.buffer.clear()


def _signals_for_filing(filing: dict[str, Any]) -> list[dict[str, Any]]:
    form = str(filing["form"])
    items = set(ITEM_PATTERN.findall(str(filing["items"])))
    definitions: list[tuple[str, str, bool, str, str, str, str]] = []
    if form in {"10-Q/A", "10-K/A"}:
        definitions.append(
            (
                "stress",
                "periodic_amendment",
                False,
                "unchanged",
                "unchanged",
                "unchanged",
                "formal_filing",
            )
        )
    if form in {"NT 10-Q", "NT 10-K"}:
        definitions.append(
            (
                "stress",
                "late_filing_notice",
                False,
                "unchanged",
                "unchanged",
                "unchanged",
                "formal_filing",
            )
        )
    if form in {"25", "25-NSE"}:
        definitions.append(
            ("exit", "formal_delisting", True, "delisted", "unchanged", "unknown", "formal_filing")
        )
    if form.startswith("15-"):
        definitions.append(
            (
                "exit",
                "reporting_deregistration",
                True,
                "unknown",
                "deregistered",
                "unknown",
                "formal_filing",
            )
        )
    item_definitions = {
        "1.03": (
            "exit",
            "bankruptcy_or_receivership",
            True,
            "unknown",
            "unchanged",
            "bankruptcy_or_receivership",
        ),
        "2.01": (
            "exit",
            "acquisition_or_disposition_completed",
            True,
            "unknown",
            "unchanged",
            "possible_successor_event",
        ),
        "3.01": (
            "exit",
            "listing_compliance_or_delisting_notice",
            True,
            "pending_or_transfer",
            "unchanged",
            "unknown",
        ),
        "4.01": ("stress", "auditor_change", False, "unchanged", "unchanged", "unchanged"),
        "4.02": (
            "stress",
            "financial_statement_non_reliance",
            False,
            "unchanged",
            "unchanged",
            "unchanged",
        ),
        "5.01": (
            "exit",
            "change_in_control",
            True,
            "unknown",
            "unchanged",
            "possible_successor_event",
        ),
    }
    for item in sorted(items):
        if item not in item_definitions:
            continue
        family, signal_type, exit_candidate, listing, reporting, operating = item_definitions[item]
        definitions.append(
            (
                family,
                signal_type,
                exit_candidate,
                listing,
                reporting,
                operating,
                "disclosed_8k_item_candidate",
            )
        )
    rows: list[dict[str, Any]] = []
    for (
        family,
        signal_type,
        exit_candidate,
        listing,
        reporting,
        operating,
        confidence,
    ) in definitions:
        rows.append(
            {
                "signal_id": _stable_hash(
                    {"accession": filing["accession_number"], "signal_type": signal_type}
                ),
                "cik": filing["cik"],
                "accession_number": filing["accession_number"],
                "form": form,
                "accepted_at": filing["accepted_at"],
                "event_family": family,
                "signal_type": signal_type,
                "is_exit_candidate": exit_candidate,
                "listing_status_effect": listing,
                "reporting_status_effect": reporting,
                "operating_status_effect": operating,
                "signal_confidence": confidence,
                "source_items": filing["items"],
                "source_url": filing["source_url"],
                "source_zip_sha256": filing["source_zip_sha256"],
            }
        )
    return rows


def _update_issuer_year(
    inventory: dict[tuple[str, int], dict[str, Any]],
    issuer: dict[str, Any],
    filing: dict[str, Any] | None,
    signal: dict[str, Any] | None,
) -> None:
    source = filing if filing is not None else signal
    if source is None:
        return
    year = int(str(source["accepted_at"])[:4])
    key = (str(source["cik"]), year)
    row = inventory.setdefault(
        key,
        {
            "cik": key[0],
            "calendar_year": year,
            "company_name_snapshot": issuer["company_name_snapshot"],
            "entity_type_snapshot": issuer["entity_type_snapshot"],
            "sic_snapshot": issuer["sic_snapshot"],
            "original_10q_filings": 0,
            "original_10k_filings": 0,
            "periodic_amendments": 0,
            "late_filing_notices": 0,
            "non_reliance_notices": 0,
            "auditor_change_notices": 0,
            "listing_notices": 0,
            "formal_delistings": 0,
            "reporting_deregistrations": 0,
            "bankruptcy_notices": 0,
            "acquisition_candidates": 0,
            "representative_candidate": False,
            "current_metadata_only": True,
        },
    )
    if filing is not None:
        form = str(filing["form"])
        row["original_10q_filings"] += int(form == "10-Q")
        row["original_10k_filings"] += int(form == "10-K")
        row["representative_candidate"] = bool(
            row["original_10q_filings"] or row["original_10k_filings"]
        )
    if signal is not None:
        column_by_signal = {
            "periodic_amendment": "periodic_amendments",
            "late_filing_notice": "late_filing_notices",
            "financial_statement_non_reliance": "non_reliance_notices",
            "auditor_change": "auditor_change_notices",
            "listing_compliance_or_delisting_notice": "listing_notices",
            "formal_delisting": "formal_delistings",
            "reporting_deregistration": "reporting_deregistrations",
            "bankruptcy_or_receivership": "bankruptcy_notices",
            "acquisition_or_disposition_completed": "acquisition_candidates",
        }
        column = column_by_signal.get(str(signal["signal_type"]))
        if column is not None:
            row[column] += 1


def _selected_form(form: str) -> bool:
    return form in PERIODIC_FORMS or form in OTHER_SELECTED_FORMS or form.startswith("15-")


def _former_name_rows(
    document: dict[str, Any], cik: str, source_member: str, source_hash: str
) -> list[dict[str, Any]]:
    raw = document.get("formerNames")
    if not isinstance(raw, list):
        return []
    return [
        {
            "cik": cik,
            "former_name": _text(item.get("name")),
            "name_from": _text(item.get("from")),
            "name_to": _text(item.get("to")),
            "source_zip_sha256": source_hash,
            "source_member": source_member,
        }
        for item in raw
        if isinstance(item, dict)
    ]


def _archive_shard_rows(
    document: dict[str, Any], cik: str, source_member: str, source_hash: str
) -> list[dict[str, Any]]:
    filings = document.get("filings")
    raw = filings.get("files") if isinstance(filings, dict) else None
    if not isinstance(raw, list):
        return []
    return [
        {
            "cik": cik,
            "archive_name": _text(item.get("name")),
            "filing_count": _int_or_zero(item.get("filingCount")),
            "filing_from": _text(item.get("filingFrom")),
            "filing_to": _text(item.get("filingTo")),
            "source_zip_sha256": source_hash,
            "source_member": source_member,
        }
        for item in raw
        if isinstance(item, dict)
    ]


def _rejection(
    cik: str,
    stage: str,
    reason: str,
    detail: str,
    source_member: str,
    source_row_index: int | None,
    source_hash: str,
) -> dict[str, Any]:
    return {
        "cik": cik.zfill(10),
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "source_member": source_member,
        "source_row_index": source_row_index,
        "source_zip_sha256": source_hash,
    }


def _acceptance_timestamp(value: Any) -> str | None:
    text = _text(value)
    for pattern in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            parsed = datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return parsed.isoformat()
    return None


def _array_value(document: dict[str, Any], field: str, index: int) -> Any:
    values = document.get(field)
    return values[index] if isinstance(values, list) and index < len(values) else None


def _array_text(document: dict[str, Any], field: str, index: int) -> str:
    return _text(_array_value(document, field, index))


def _array_bool(document: dict[str, Any], field: str, index: int) -> bool | None:
    value = _array_value(document, field, index)
    if isinstance(value, bool):
        return value
    if value in {0, 1, "0", "1"}:
        return bool(int(value))
    return None


def _json_array(value: Any) -> str:
    return json.dumps(
        value if isinstance(value, list) else [], ensure_ascii=False, separators=(",", ":")
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _leading_year(value: str) -> int | None:
    return int(value[:4]) if len(value) >= 4 and value[:4].isdigit() else None


def _date_ranges_overlap(start: str, end: str, window_start: str, window_end: str) -> bool:
    return bool(start and end and start <= window_end and end >= window_start)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stable_hash(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)
