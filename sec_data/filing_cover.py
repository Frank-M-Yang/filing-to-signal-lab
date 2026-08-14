"""Preserve and parse filing-time identity, security, and SIC cover evidence."""

from __future__ import annotations

import hashlib
import html
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from sec_data.client import SecClient, SecRequestError, normalize_cik
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.legacy_cover import (
    merge_visible_registration_evidence,
    parse_visible_registration_tables,
)
from sec_data.models import SecResponse

COVER_CONCEPTS: Final[frozenset[str]] = frozenset(
    {
        "documenttype",
        "documentfiscalyearfocus",
        "documentfiscalperiodfocus",
        "entitycentralindexkey",
        "entityregistrantname",
        "entityfilenumber",
        "tradingsymbol",
        "securityexchangename",
        "security12btitle",
        "security12gtitle",
        "notradingsymbolflag",
    }
)
CONCEPT_LABELS: Final[dict[str, str]] = {
    concept.lower(): concept
    for concept in (
        "DocumentType",
        "DocumentFiscalYearFocus",
        "DocumentFiscalPeriodFocus",
        "EntityCentralIndexKey",
        "EntityRegistrantName",
        "EntityFileNumber",
        "TradingSymbol",
        "SecurityExchangeName",
        "Security12bTitle",
        "Security12gTitle",
        "NoTradingSymbolFlag",
    )
}
PARSER_VERSION: Final[str] = "filing-cover-parser-v4-visible-registration-tables"
ArtifactPolicy = Literal["index-only", "index-and-cover"]


@dataclass(frozen=True)
class FilingArtifactSnapshot:
    """Exact persisted bytes and request metadata for one filing artifact."""

    target_id: str
    artifact_kind: str
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


@dataclass(frozen=True)
class CoverTargetPlanResult:
    """Immutable issuer-year filing-cover target plan."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class CoverCollectionResult:
    """Counts from one resumable acquisition batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int
    snapshots_reused: int


@dataclass(frozen=True)
class FilingCoverHistoryResult:
    """Processed identity, security, SIC, interval, and coverage assets."""

    output_root: Path
    report: dict[str, Any]


def build_filing_cover_target_plan(
    catalog_root: Path,
    *,
    output_root: Path = Path("data/sec/filing_covers/targets"),
) -> CoverTargetPlanResult:
    """Select one first-accepted original periodic filing per candidate issuer-year."""
    catalog_report = json.loads((catalog_root / "catalog_report.json").read_text(encoding="utf-8"))
    if not bool(catalog_report.get("archive_coverage_complete")):
        raise FilingCatalogError("Filing-cover targets require complete archive coverage")
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    inventory = load_catalog_table(catalog_root, "issuer_year_inventory.parquet")
    filings = load_catalog_table(catalog_root, "filing_catalog.parquet")
    representative = inventory.loc[
        inventory["representative_candidate"].astype(bool), ["cik", "calendar_year"]
    ].copy()
    periodic = filings.loc[filings["form"].isin(["10-Q", "10-K"])].copy()
    periodic["accepted_at"] = pd.to_datetime(periodic["accepted_at"], utc=True)
    periodic["calendar_year"] = periodic["accepted_at"].dt.year
    periodic = periodic.merge(
        representative,
        on=["cik", "calendar_year"],
        how="inner",
        validate="many_to_one",
    )
    anchors = (
        periodic.sort_values(["accepted_at", "cik", "accession_number"])
        .groupby(["cik", "calendar_year"], as_index=False)
        .first()
    )
    if len(anchors) != len(representative):
        raise FilingCatalogError("Filing-cover anchors do not cover the candidate issuer-years")
    anchors["target_id"] = anchors.apply(
        lambda row: _stable_hash(
            {
                "cik": str(row["cik"]),
                "calendar_year": int(row["calendar_year"]),
                "accession_number": str(row["accession_number"]),
            }
        ),
        axis=1,
    )
    anchors["index_url"] = anchors.apply(
        lambda row: _filing_index_url(str(row["cik"]), str(row["accession_number"])),
        axis=1,
    )
    anchors["primary_document_url"] = anchors["source_url"]
    anchors["anchor_rule"] = "first_original_periodic_filing_accepted_in_calendar_year"
    anchors["selection_uses_future_exit_information"] = False
    anchors["sampling_hash"] = anchors.apply(
        lambda row: hashlib.sha256(
            f"{row['cik']}|{row['calendar_year']}|filing-cover-bootstrap-v1".encode()
        ).hexdigest(),
        axis=1,
    )
    anchors["within_year_order"] = (
        anchors.groupby("calendar_year")["sampling_hash"].rank(method="first").astype(int)
    )
    anchors = anchors.sort_values(
        ["within_year_order", "calendar_year", "sampling_hash"]
    ).reset_index(drop=True)
    anchors["acquisition_order"] = np.arange(1, len(anchors) + 1)
    target_columns = [
        "target_id",
        "acquisition_order",
        "within_year_order",
        "sampling_hash",
        "cik",
        "calendar_year",
        "filing_id",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "is_xbrl",
        "is_inline_xbrl",
        "primary_document",
        "primary_document_url",
        "index_url",
        "anchor_rule",
        "selection_uses_future_exit_information",
    ]
    targets = anchors[target_columns].copy()
    payload = json.dumps(
        {
            "catalog_version": catalog_report["dataset_version"],
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "anchor_rule": "first_original_periodic_filing_accepted_in_calendar_year",
            "order_rule": "round_robin_year_then_sha256_cik_year",
        },
        sort_keys=True,
    ).encode()
    dataset_version = "filing-cover-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads((target / "target_report.json").read_text(encoding="utf-8"))
        return CoverTargetPlanResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete filing-cover target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    targets.to_parquet(staging / "filing_cover_targets.parquet", index=False)
    artifact = _artifact_record(staging / "filing_cover_targets.parquet", len(targets))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": catalog_report["dataset_version"],
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "candidate_issuer_year_targets": len(targets),
        "candidate_ciks": targets["cik"].nunique(),
        "years": sorted(targets["calendar_year"].astype(int).unique().tolist()),
        "inline_xbrl_targets": int(targets["is_inline_xbrl"].astype(bool).sum()),
        "anchor_rule": "first_original_periodic_filing_accepted_in_calendar_year",
        "acquisition_order_rule": "balanced_by_year_then_deterministic_hash",
        "future_exit_information_used": False,
        "artifacts": {"filing_cover_targets.parquet": artifact},
    }
    (staging / "target_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "artifacts": {"filing_cover_targets.parquet": artifact},
                "report_sha256": _sha256_file(staging / "target_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return CoverTargetPlanResult(target.resolve(), report)


def load_filing_cover_targets(plan_root: Path) -> pd.DataFrame:
    """Load and verify a filing-cover target plan."""
    return load_catalog_table(plan_root, "filing_cover_targets.parquet")


def collect_filing_cover_artifacts(
    targets: pd.DataFrame,
    client: SecClient,
    *,
    raw_root: Path = Path("data/sec/filing_covers"),
    limit: int | None = None,
    artifact_policy: ArtifactPolicy = "index-only",
    progress: Callable[[int, int, str], None] | None = None,
) -> CoverCollectionResult:
    """Collect a resumable target batch, preserving bytes before any parsing."""
    required = {
        "target_id",
        "cik",
        "primary_document_url",
        "index_url",
        "acquisition_order",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise FilingCatalogError(f"Filing-cover targets lack columns: {sorted(missing)}")
    if artifact_policy not in {"index-only", "index-and-cover"}:
        raise ValueError(f"Unsupported artifact policy: {artifact_policy}")
    ordered = targets.sort_values("acquisition_order")
    pending_rows = [
        row
        for row in ordered.itertuples(index=False)
        if not _target_policy_complete(
            raw_root,
            str(row.target_id),
            str(row.cik),
            artifact_policy,
        )
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("Collection limit must be positive")
        pending_rows = pending_rows[:limit]
    requests_made = 0
    snapshots_reused = 0
    completed = 0
    failed = 0
    total = len(pending_rows)
    for index, row in enumerate(pending_rows, start=1):
        target_id = str(row.target_id)
        try:
            filing_index, made = _load_or_fetch(
                raw_root,
                target_id,
                "index",
                str(row.index_url),
                client,
            )
            requests_made += int(made)
            snapshots_reused += int(not made)
            if artifact_policy == "index-and-cover":
                primary, made = _load_or_fetch(
                    raw_root,
                    target_id,
                    "primary",
                    str(row.primary_document_url),
                    client,
                )
                requests_made += int(made)
                snapshots_reused += int(not made)
                primary_facts = parse_cover_facts(Path(primary.path).read_bytes())
                index_metadata = parse_filing_index(
                    Path(filing_index.path).read_bytes(), index_url=filing_index.requested_url
                )
                instance_url = index_metadata.get("instance_document_url")
                if (
                    (
                        not _has_core_identity_facts(primary_facts)
                        or not _has_subject_security_evidence(primary_facts, str(row.cik))
                    )
                    and isinstance(instance_url, str)
                    and instance_url
                ):
                    _, made = _load_or_fetch(
                        raw_root,
                        target_id,
                        "instance",
                        instance_url,
                        client,
                    )
                    requests_made += int(made)
                    snapshots_reused += int(not made)
            completed += 1
        except (OSError, SecRequestError, FilingCatalogError) as exc:
            failed += 1
            _write_acquisition_failure(raw_root, target_id, str(exc))
        if progress is not None:
            progress(index, total, target_id)
    return CoverCollectionResult(total, completed, failed, requests_made, snapshots_reused)


def build_filing_cover_history(
    catalog_root: Path,
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/filing_covers"),
    output_root: Path = Path("data/processed/sec_filing_cover_history"),
) -> FilingCoverHistoryResult:
    """Parse preserved artifacts into filing-time observations and validity intervals."""
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    plan_report = json.loads((plan_root / "target_report.json").read_text(encoding="utf-8"))
    if plan_report.get("catalog_manifest_sha256") != catalog_manifest_sha256:
        raise FilingCatalogError("Filing-cover target plan does not tie to the catalog")
    targets = load_filing_cover_targets(plan_root)
    snapshots = _load_all_valid_snapshots(raw_root)
    if not snapshots:
        raise FilingCatalogError("No preserved filing-cover artifacts are available")
    snapshots_by_target_kind = {
        (snapshot.target_id, snapshot.artifact_kind): snapshot for snapshot in snapshots
    }
    fact_rows: list[dict[str, Any]] = []
    visible_security_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    security_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    completed_targets: list[str] = []
    for row in targets.itertuples(index=False):
        target_id = str(row.target_id)
        filing_index = snapshots_by_target_kind.get((target_id, "index"))
        if filing_index is None:
            continue
        primary = snapshots_by_target_kind.get((target_id, "primary"))
        completed_targets.append(target_id)
        facts: list[dict[str, str]] = []
        fact_source: FilingArtifactSnapshot | None = None
        fact_artifact_kind: str | None = None
        if primary is not None:
            facts = parse_cover_facts(Path(primary.path).read_bytes())
            fact_source = primary
            fact_artifact_kind = "primary"
        if not facts:
            instance = snapshots_by_target_kind.get((target_id, "instance"))
            if instance is not None:
                facts = parse_cover_facts(Path(instance.path).read_bytes())
                fact_source = instance
                fact_artifact_kind = "instance"
        metadata = parse_filing_index(
            Path(filing_index.path).read_bytes(),
            index_url=filing_index.requested_url,
            subject_cik=str(row.cik),
        )
        common = {
            "target_id": target_id,
            "cik": str(row.cik),
            "calendar_year": int(str(row.calendar_year)),
            "accession_number": str(row.accession_number),
            "form": str(row.form),
            "report_date": str(row.report_date),
            "accepted_at": str(row.accepted_at),
        }
        for occurrence, fact in enumerate(facts):
            fact_rows.append(
                {
                    **common,
                    "concept": fact["concept"],
                    "value": fact["value"],
                    "context_ref": fact["context_ref"],
                    "context_entity_cik": fact["context_entity_cik"],
                    "format": fact["format"],
                    "source_occurrence": occurrence,
                    "source_artifact_kind": fact_artifact_kind,
                    "source_sha256": fact_source.sha256 if fact_source else None,
                    "source_path": fact_source.path if fact_source else None,
                }
            )
        fact_lookup = _fact_lookup(facts)
        reported_cik, registrant_name, entity_context_ref = _subject_cover_identity(
            fact_lookup, str(row.cik)
        )
        document_type = _first_fact(fact_lookup, "DocumentType")
        cik_match = reported_cik == str(row.cik)
        form_match = document_type == str(row.form)
        cover_identity_ready = bool(registrant_name and cik_match and form_match)
        index_cik = _normalized_reported_cik(metadata.get("subject_cik"))
        index_name = _clean_cover_value(metadata.get("subject_name"))
        index_form = _clean_cover_value(metadata.get("subject_form"))
        index_identity_ready = bool(
            index_cik == str(row.cik) and index_name and index_form == str(row.form)
        )
        identity_ready = cover_identity_ready or index_identity_ready
        if not cover_identity_ready and index_identity_ready:
            reported_cik = index_cik
            registrant_name = index_name
            document_type = index_form
            cik_match = True
            form_match = True
            entity_context_ref = None
            identity_source = filing_index
            identity_artifact_kind = "index"
            identity_status = "verified_from_filing_index_subject_block"
        else:
            identity_source = fact_source or filing_index
            identity_artifact_kind = fact_artifact_kind or "index"
            identity_status = (
                "verified_from_cover_facts"
                if cover_identity_ready
                else "incomplete_or_mismatched_cover_and_index"
            )
        identity_rows.append(
            {
                **common,
                "registrant_name": registrant_name,
                "reported_cik": reported_cik,
                "document_type": document_type,
                "cik_match": cik_match,
                "document_type_match": form_match,
                "entity_context_ref": entity_context_ref,
                "filing_identity_ready": identity_ready,
                "identity_status": identity_status,
                "source_artifact_kind": identity_artifact_kind,
                "source_sha256": identity_source.sha256,
                "source_path": identity_source.path,
            }
        )
        visible_securities = (
            parse_visible_registration_tables(Path(primary.path).read_bytes())
            if primary is not None
            else []
        )
        single_filer_subject = bool(
            metadata.get("filer_entity_count") == 1 and metadata.get("subject_cik") == str(row.cik)
        )
        for visible_security in visible_securities:
            visible_security_rows.append(
                {
                    **common,
                    **visible_security,
                    "single_filer_subject": single_filer_subject,
                    "automatic_subject_attribution_eligible": single_filer_subject,
                    "source_artifact_kind": "primary",
                    "source_sha256": primary.sha256 if primary else None,
                    "source_path": primary.path if primary else None,
                }
            )
        filing_securities = merge_visible_registration_evidence(
            _security_observations(fact_lookup, subject_cik=str(row.cik)),
            visible_securities,
            subject_cik=str(row.cik),
            single_filer_subject=single_filer_subject,
        )
        for security in filing_securities:
            security_rows.append(
                {
                    **common,
                    **security,
                    "source_artifact_kind": fact_artifact_kind,
                    "source_sha256": fact_source.sha256 if fact_source else None,
                    "source_path": fact_source.path if fact_source else None,
                }
            )
        sic = metadata.get("sic")
        industry_rows.append(
            {
                **common,
                "sic": sic,
                "sic_description": metadata.get("sic_description"),
                "index_subject_cik_matched": metadata.get("subject_cik") == str(row.cik),
                "historical_industry_ready": bool(
                    isinstance(sic, str) and re.fullmatch(r"\d{4}", sic)
                ),
                "industry_evidence_status": (
                    "filing_index_sic" if sic else "missing_filing_index_sic"
                ),
                "source_sha256": filing_index.sha256,
                "source_path": filing_index.path,
            }
        )
        if primary is not None and not facts:
            rejection_rows.append(_cover_rejection(common, "no_supported_cover_facts", primary))
        if not sic:
            rejection_rows.append(
                _cover_rejection(common, "missing_filing_index_sic", filing_index)
            )

    identities = pd.DataFrame(identity_rows)
    industries = pd.DataFrame(industry_rows)
    securities_frame = pd.DataFrame(security_rows)
    facts_frame = pd.DataFrame(fact_rows)
    visible_securities_frame = pd.DataFrame(visible_security_rows)
    if identities.empty or industries.empty:
        raise FilingCatalogError("Preserved filing covers produced no identity/industry anchors")
    identity_columns = [
        "target_id",
        "filing_identity_ready",
        "registrant_name",
        "reported_cik",
        "document_type",
        "identity_status",
    ]
    industry_columns = [
        "target_id",
        "historical_industry_ready",
        "sic",
        "sic_description",
        "industry_evidence_status",
    ]
    anchors = targets.merge(
        identities[identity_columns], on="target_id", how="left", validate="one_to_one"
    ).merge(industries[industry_columns], on="target_id", how="left", validate="one_to_one")
    anchors["index_artifact_preserved"] = anchors["target_id"].isin(completed_targets)
    cover_target_ids = {
        target_id
        for target_id, artifact_kind in snapshots_by_target_kind
        if artifact_kind in {"primary", "instance"}
    }
    anchors["cover_artifact_preserved"] = anchors["target_id"].isin(cover_target_ids)
    anchors["artifact_pair_preserved"] = (
        anchors["index_artifact_preserved"] & anchors["cover_artifact_preserved"]
    )
    anchors["filing_identity_ready"] = anchors["filing_identity_ready"].fillna(False)
    anchors["historical_industry_ready"] = anchors["historical_industry_ready"].fillna(False)
    if securities_frame.empty:
        security_summary = pd.DataFrame(
            columns=[
                "target_id",
                "security_identity_observed",
                "tradable_security_mapping_ready",
                "explicit_no_trading_symbol",
                "security_observation_count",
            ]
        )
    else:
        security_summary = securities_frame.groupby("target_id", as_index=False).agg(
            security_identity_observed=("security_identity_observed", "max"),
            tradable_security_mapping_ready=("tradable_security_mapping_ready", "max"),
            explicit_no_trading_symbol=("explicit_no_trading_symbol", "max"),
            security_observation_count=("context_ref", "size"),
        )
    anchors = anchors.merge(security_summary, on="target_id", how="left", validate="one_to_one")
    anchors["security_identity_observed"] = anchors["security_identity_observed"].fillna(False)
    anchors["tradable_security_mapping_ready"] = anchors["tradable_security_mapping_ready"].fillna(
        False
    )
    anchors["explicit_no_trading_symbol"] = anchors["explicit_no_trading_symbol"].fillna(False)
    anchors["security_observation_count"] = (
        anchors["security_observation_count"].fillna(0).astype(int)
    )
    anchors["historical_identity_ready"] = (
        anchors["filing_identity_ready"] & anchors["security_identity_observed"]
    )
    anchors["identity_industry_ready"] = (
        anchors["historical_identity_ready"] & anchors["historical_industry_ready"]
    )
    anchors["sample_freeze_eligible_from_this_asset"] = False
    anchors["unacquired_is_missing_not_negative"] = ~anchors["index_artifact_preserved"]
    coverage = (
        anchors.groupby("calendar_year", as_index=False)
        .agg(
            target_issuer_years=("target_id", "size"),
            index_artifacts_preserved=("index_artifact_preserved", "sum"),
            cover_artifacts_preserved=("cover_artifact_preserved", "sum"),
            artifact_pairs_preserved=("artifact_pair_preserved", "sum"),
            filing_identity_ready=("filing_identity_ready", "sum"),
            security_identity_observed=("security_identity_observed", "sum"),
            tradable_security_mapping_ready=("tradable_security_mapping_ready", "sum"),
            historical_industry_ready=("historical_industry_ready", "sum"),
            identity_industry_ready=("identity_industry_ready", "sum"),
        )
        .sort_values("calendar_year")
    )
    for column in (
        "index_artifacts_preserved",
        "cover_artifacts_preserved",
        "artifact_pairs_preserved",
        "filing_identity_ready",
        "security_identity_observed",
        "tradable_security_mapping_ready",
        "historical_industry_ready",
        "identity_industry_ready",
    ):
        coverage[f"{column}_coverage"] = coverage[column] / coverage["target_issuer_years"]
        coverage[f"{column}_among_acquired"] = coverage[column] / coverage[
            "index_artifacts_preserved"
        ].replace(0, np.nan)
    identity_intervals = _observation_intervals(
        identities.loc[identities["filing_identity_ready"].astype(bool)],
        ["registrant_name", "reported_cik"],
    )
    industry_intervals = _observation_intervals(
        industries.loc[industries["historical_industry_ready"].astype(bool)],
        ["sic", "sic_description"],
    )
    security_intervals = _observation_intervals(
        securities_frame,
        [
            "context_ref",
            "context_entity_cik",
            "security_subject_cik_match",
            "trading_symbol",
            "security_exchange_name",
            "security_title",
            "explicit_no_trading_symbol",
            "security_evidence_kind",
            "security_evidence_confidence",
        ],
    )

    snapshot_payload = [
        {
            "target_id": snapshot.target_id,
            "kind": snapshot.artifact_kind,
            "sha256": snapshot.sha256,
        }
        for snapshot in sorted(snapshots, key=lambda item: (item.target_id, item.artifact_kind))
    ]
    dataset_payload = json.dumps(
        {
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "plan_manifest_sha256": plan_manifest_sha256,
            "snapshots": snapshot_payload,
            "parser_version": PARSER_VERSION,
            "interval_rule": "observed_at_inclusive_to_next_observation_exclusive_no_backfill",
        },
        sort_keys=True,
    ).encode()
    dataset_version = "filing-cover-history-v1-" + hashlib.sha256(dataset_payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads(
            (target / "filing_cover_report.json").read_text(encoding="utf-8")
        )
        return FilingCoverHistoryResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete filing-cover history directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    snapshot_frame = pd.DataFrame(asdict(snapshot) for snapshot in snapshots)
    snapshot_frame["headers_json"] = snapshot_frame["headers"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    snapshot_frame = snapshot_frame.drop(columns="headers")
    rejection_frame = pd.DataFrame(
        rejection_rows,
        columns=[
            "target_id",
            "cik",
            "calendar_year",
            "accession_number",
            "form",
            "report_date",
            "accepted_at",
            "reason",
            "source_artifact_kind",
            "source_sha256",
            "source_path",
        ],
    )
    outputs: dict[str, pd.DataFrame] = {
        "filing_cover_targets.parquet": targets,
        "artifact_snapshots.parquet": snapshot_frame,
        "filing_cover_facts.parquet": facts_frame,
        "filing_visible_security_evidence.parquet": visible_securities_frame,
        "filing_identity_observations.parquet": identities,
        "filing_security_observations.parquet": securities_frame,
        "filing_industry_observations.parquet": industries,
        "issuer_year_identity_industry_anchors.parquet": anchors,
        "identity_intervals.parquet": identity_intervals,
        "security_intervals.parquet": security_intervals,
        "industry_intervals.parquet": industry_intervals,
        "identity_industry_coverage_by_year.parquet": coverage,
        "filing_cover_rejections.parquet": rejection_frame,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
    artifacts = {
        filename: _artifact_record(staging / filename, len(frame))
        for filename, frame in outputs.items()
    }
    acquired = anchors["index_artifact_preserved"].astype(bool)
    covers_acquired = anchors["cover_artifact_preserved"].astype(bool)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "parser_version": PARSER_VERSION,
        "target_issuer_years": len(targets),
        "artifact_snapshots": len(snapshots),
        "acquired_issuer_years": int(acquired.sum()),
        "acquisition_coverage": float(acquired.mean()),
        "cover_acquired_issuer_years": int(covers_acquired.sum()),
        "cover_acquisition_coverage": float(covers_acquired.mean()),
        "filing_identity_ready_acquired": int(anchors.loc[acquired, "filing_identity_ready"].sum()),
        "security_identity_observed_acquired": int(
            anchors.loc[acquired, "security_identity_observed"].sum()
        ),
        "visible_registration_rows_parsed": len(visible_securities_frame),
        "visible_registration_rows_subject_attributable": int(
            visible_securities_frame.get(
                "automatic_subject_attribution_eligible", pd.Series(dtype=bool)
            )
            .astype(bool)
            .sum()
        ),
        "security_observations_using_visible_evidence": int(
            securities_frame.get("security_evidence_kind", pd.Series(dtype=str))
            .astype(str)
            .str.contains("visible", na=False)
            .sum()
        ),
        "tradable_security_mapping_ready_acquired": int(
            anchors.loc[acquired, "tradable_security_mapping_ready"].sum()
        ),
        "historical_industry_ready_acquired": int(
            anchors.loc[acquired, "historical_industry_ready"].sum()
        ),
        "identity_industry_ready_acquired": int(
            anchors.loc[acquired, "identity_industry_ready"].sum()
        ),
        "representative_identity_gate": bool(anchors["historical_identity_ready"].all()),
        "representative_industry_gate": bool(anchors["historical_industry_ready"].all()),
        "representative_sample_frozen": False,
        "unacquired_rows_are_missing_not_negative": True,
        "current_registry_metadata_used_for_history": False,
        "companyfacts_filed_date_used_for_availability": False,
        "interval_rule": "observed_at_inclusive_to_next_observation_exclusive_no_backfill",
        "artifacts": artifacts,
    }
    (staging / "filing_cover_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(staging / "filing_cover_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return FilingCoverHistoryResult(target.resolve(), report)


def parse_cover_facts(content: bytes) -> list[dict[str, str]]:
    """Extract supported DEI cover facts from inline HTML or an XBRL instance."""
    parser = _CoverFactParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    parser.close()
    for fact in parser.facts:
        fact["context_entity_cik"] = parser.context_entities.get(fact["context_ref"], "")
    return parser.facts


def parse_filing_index(
    content: bytes,
    *,
    index_url: str,
    subject_cik: str | None = None,
) -> dict[str, Any]:
    """Extract the filing-specific SIC and optional XBRL instance document link."""
    source = content.decode("utf-8", errors="replace")
    entities = _index_entity_blocks(source)
    selected_entity = next(
        (entity for entity in entities if entity["cik"] == subject_cik),
        entities[0] if entities else {},
    )
    instance_url: str | None = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", source, flags=re.IGNORECASE | re.DOTALL):
        row_text = _strip_html(row)
        if "EX-101.INS" not in row_text.upper():
            continue
        link = re.search(r"<a[^>]+href=[\"']([^\"']+)[\"']", row, flags=re.IGNORECASE)
        if link:
            instance_url = urljoin(index_url, html.unescape(link.group(1)))
            break
    return {
        "sic": selected_entity.get("sic"),
        "sic_description": selected_entity.get("sic_description"),
        "subject_cik": selected_entity.get("cik"),
        "subject_name": selected_entity.get("name"),
        "subject_form": selected_entity.get("form"),
        "filer_entity_count": len([entity for entity in entities if entity.get("cik")]),
        "instance_document_url": instance_url,
    }


class _CoverFactParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.facts: list[dict[str, str]] = []
        self._capture_depth = 0
        self._concept: str | None = None
        self._context_ref = ""
        self._format = ""
        self._fragments: list[str] = []
        self.context_entities: dict[str, str] = {}
        self._context_id: str | None = None
        self._identifier_fragments: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        lowered = tag.lower()
        if _local_concept(lowered) == "context":
            self._context_id = attributes.get("id") or None
        elif self._context_id is not None and _local_concept(lowered) == "identifier":
            self._identifier_fragments = []
        concept = ""
        if lowered == "ix:nonnumeric":
            concept = _local_concept(attributes.get("name", ""))
        elif ":" in lowered:
            concept = _local_concept(lowered)
        if self._capture_depth == 0 and concept.lower() in COVER_CONCEPTS:
            self._capture_depth = 1
            self._concept = CONCEPT_LABELS[concept.lower()]
            self._context_ref = attributes.get("contextref", "")
            self._format = attributes.get("format", "")
            self._fragments = []
        elif self._capture_depth > 0:
            self._capture_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if _local_concept(lowered) == "identifier" and self._identifier_fragments is not None:
            identifier = _normalized_reported_cik("".join(self._identifier_fragments))
            if identifier is not None and self._context_id is not None:
                self.context_entities[self._context_id] = identifier
            self._identifier_fragments = None
        elif _local_concept(lowered) == "context":
            self._context_id = None
        if self._capture_depth == 0:
            return
        self._capture_depth -= 1
        if self._capture_depth != 0 or self._concept is None:
            return
        value = re.sub(r"\s+", " ", "".join(self._fragments)).strip()
        if self._concept == "NoTradingSymbolFlag" and self._format.lower().endswith("fixed-true"):
            value = "true"
        self.facts.append(
            {
                "concept": self._concept,
                "value": value,
                "context_ref": self._context_ref,
                "format": self._format,
            }
        )
        self._concept = None
        self._context_ref = ""
        self._format = ""
        self._fragments = []

    def handle_data(self, data: str) -> None:
        if self._capture_depth > 0:
            self._fragments.append(data)
        if self._identifier_fragments is not None:
            self._identifier_fragments.append(data)


def _load_or_fetch(
    raw_root: Path,
    target_id: str,
    artifact_kind: str,
    url: str,
    client: SecClient,
) -> tuple[FilingArtifactSnapshot, bool]:
    existing = _existing_snapshot(raw_root, target_id, artifact_kind, url)
    if existing is not None:
        return existing, False
    response = client.get_filing_artifact(url)
    return _preserve_response(raw_root, target_id, artifact_kind, url, response), True


def _preserve_response(
    raw_root: Path,
    target_id: str,
    artifact_kind: str,
    requested_url: str,
    response: SecResponse,
) -> FilingArtifactSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    raw_directory = raw_root / "raw" / artifact_kind / digest[:2]
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_directory / f"{digest}.bin"
    if raw_path.exists():
        if _sha256_file(raw_path) != digest:
            raise FilingCatalogError(f"Existing filing artifact hash mismatch: {raw_path}")
    else:
        temporary = raw_directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Filing artifact changed before preservation")
        temporary.replace(raw_path)
    snapshot = FilingArtifactSnapshot(
        target_id=target_id,
        artifact_kind=artifact_kind,
        requested_url=requested_url,
        response_url=response.url,
        sha256=digest,
        fetched_at=response.fetched_at.isoformat(),
        path=str(raw_path.resolve()),
        bytes=len(response.content),
        status_code=response.status_code,
        headers=response.headers,
    )
    manifest_directory = raw_root / "snapshots" / target_id
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / f"{artifact_kind}-{digest}.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8"
        )
    return snapshot


def _existing_snapshot(
    raw_root: Path, target_id: str, artifact_kind: str, url: str
) -> FilingArtifactSnapshot | None:
    directory = raw_root / "snapshots" / target_id
    for manifest_path in sorted(directory.glob(f"{artifact_kind}-*.json")):
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = FilingArtifactSnapshot(**document)
        if snapshot.requested_url != url:
            continue
        path = Path(snapshot.path)
        if (
            path.is_file()
            and path.stat().st_size == snapshot.bytes
            and _sha256_file(path) == snapshot.sha256
        ):
            return snapshot
    return None


def _load_all_valid_snapshots(raw_root: Path) -> list[FilingArtifactSnapshot]:
    selected: dict[tuple[str, str], FilingArtifactSnapshot] = {}
    for manifest_path in sorted((raw_root / "snapshots").glob("*/*.json")):
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = FilingArtifactSnapshot(**document)
        path = Path(snapshot.path)
        if not path.is_file() or path.stat().st_size != snapshot.bytes:
            raise FilingCatalogError(f"Filing artifact is missing or truncated: {path}")
        if _sha256_file(path) != snapshot.sha256:
            raise FilingCatalogError(f"Filing artifact hash mismatch: {path}")
        selected.setdefault((snapshot.target_id, snapshot.artifact_kind), snapshot)
    return list(selected.values())


def _target_policy_complete(
    raw_root: Path,
    target_id: str,
    subject_cik: str,
    artifact_policy: ArtifactPolicy,
) -> bool:
    directory = raw_root / "snapshots" / target_id
    index_snapshot = _valid_snapshot_for_kind(directory, "index")
    has_index = index_snapshot is not None
    if artifact_policy == "index-only":
        return has_index
    primary_snapshot = _valid_snapshot_for_kind(directory, "primary")
    if index_snapshot is None or primary_snapshot is None:
        return False
    primary_facts = parse_cover_facts(Path(primary_snapshot.path).read_bytes())
    if _has_core_identity_facts(primary_facts) and _has_subject_security_evidence(
        primary_facts, subject_cik
    ):
        return True
    index_metadata = parse_filing_index(
        Path(index_snapshot.path).read_bytes(), index_url=index_snapshot.requested_url
    )
    if not index_metadata.get("instance_document_url"):
        return True
    return _valid_snapshot_for_kind(directory, "instance") is not None


def _has_valid_artifact_manifest(directory: Path, artifact_kind: str) -> bool:
    return _valid_snapshot_for_kind(directory, artifact_kind) is not None


def _valid_snapshot_for_kind(directory: Path, artifact_kind: str) -> FilingArtifactSnapshot | None:
    for manifest_path in directory.glob(f"{artifact_kind}-*.json"):
        try:
            snapshot = FilingArtifactSnapshot(
                **json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        path = Path(snapshot.path)
        if (
            path.is_file()
            and path.stat().st_size == snapshot.bytes
            and _sha256_file(path) == snapshot.sha256
        ):
            return snapshot
    return None


def _write_acquisition_failure(raw_root: Path, target_id: str, detail: str) -> None:
    directory = raw_root / "failures" / target_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "target_id": target_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "detail": detail,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _fact_lookup(facts: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    for fact in facts:
        lookup.setdefault(fact["concept"], []).append(fact)
    return lookup


def _first_fact(lookup: dict[str, list[dict[str, str]]], concept: str) -> str | None:
    for fact in lookup.get(concept, []):
        value = _clean_cover_value(fact["value"])
        if value:
            return value
    return None


def _has_core_identity_facts(facts: list[dict[str, str]]) -> bool:
    concepts = {fact["concept"] for fact in facts if _clean_cover_value(fact["value"])}
    return {
        "DocumentType",
        "EntityRegistrantName",
        "EntityCentralIndexKey",
    }.issubset(concepts)


def _has_subject_security_evidence(facts: list[dict[str, str]], subject_cik: str) -> bool:
    return any(
        observation["security_identity_observed"]
        for observation in _security_observations(
            _fact_lookup(facts),
            subject_cik=subject_cik,
        )
    )


def _subject_cover_identity(
    lookup: dict[str, list[dict[str, str]]], subject_cik: str
) -> tuple[str | None, str | None, str | None]:
    for cik_fact in lookup.get("EntityCentralIndexKey", []):
        reported_cik = _normalized_reported_cik(cik_fact["value"])
        if reported_cik != subject_cik:
            continue
        context = cik_fact["context_ref"]
        matching_names = [
            _clean_cover_value(fact["value"])
            for fact in lookup.get("EntityRegistrantName", [])
            if fact["context_ref"] == context
        ]
        registrant_name = next((name for name in matching_names if name), None)
        if registrant_name is None:
            registrant_name = _first_fact(lookup, "EntityRegistrantName")
        return reported_cik, registrant_name, context or None
    return (
        _normalized_reported_cik(_first_fact(lookup, "EntityCentralIndexKey")),
        _first_fact(lookup, "EntityRegistrantName"),
        None,
    )


def _security_observations(
    lookup: dict[str, list[dict[str, str]]],
    *,
    subject_cik: str,
) -> list[dict[str, Any]]:
    by_context: dict[str, dict[str, str]] = {}
    security_concepts = (
        "TradingSymbol",
        "SecurityExchangeName",
        "Security12bTitle",
        "Security12gTitle",
        "NoTradingSymbolFlag",
    )
    for concept in security_concepts:
        for fact in lookup.get(concept, []):
            context = fact["context_ref"] or "missing_context"
            by_context.setdefault(context, {})[concept] = _clean_cover_value(fact["value"]) or ""
            raw_context_entity_cik = fact.get("context_entity_cik") or ""
            if raw_context_entity_cik:
                by_context[context]["context_entity_cik"] = raw_context_entity_cik
    observations: list[dict[str, Any]] = []
    for context, values in sorted(by_context.items()):
        ticker = _valid_trading_symbol(values.get("TradingSymbol"))
        exchange = _clean_cover_value(values.get("SecurityExchangeName"))
        title = _clean_cover_value(values.get("Security12bTitle") or values.get("Security12gTitle"))
        explicit_no_symbol = _truthy(values.get("NoTradingSymbolFlag"))
        context_entity_cik = _normalized_reported_cik(values.get("context_entity_cik"))
        subject_cik_match = context_entity_cik == subject_cik
        if not ticker and not explicit_no_symbol:
            continue
        observations.append(
            {
                "context_ref": context,
                "context_entity_cik": context_entity_cik,
                "security_subject_cik_match": subject_cik_match,
                "trading_symbol": ticker,
                "security_exchange_name": exchange,
                "security_title": title,
                "explicit_no_trading_symbol": explicit_no_symbol,
                "security_identity_observed": bool(
                    subject_cik_match and (ticker or explicit_no_symbol)
                ),
                "tradable_security_mapping_ready": bool(subject_cik_match and ticker and exchange),
            }
        )
    return observations


def _index_entity_blocks(source: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    blocks = re.findall(
        r"<div[^>]+class=[\"']companyInfo[\"'][^>]*>(.*?)</div>",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        cik_match = re.search(r"[?&](?:amp;)?CIK=(\d{1,10})", block, flags=re.IGNORECASE)
        name_match = re.search(
            r"<span[^>]+class=[\"']companyName[\"'][^>]*>(.*?)<acronym",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        form_match = re.search(r"Type:\s*<strong>([^<]+)</strong>", block, flags=re.IGNORECASE)
        sic_match = re.search(
            r"SIC</acronym>\s*:\s*<b>\s*<a[^>]*(?:[?&]|&amp;)SIC=(\d{4})"
            r"[^>]*>\d{4}</a>\s*</b>\s*([^<]*?)(?:<br|</p>)",
            block,
            flags=re.IGNORECASE,
        )
        if cik_match is None:
            continue
        name = _strip_html(name_match.group(1)) if name_match else ""
        name = re.sub(r"\s*\((?:Filer|Issuer)\)\s*$", "", name, flags=re.IGNORECASE)
        entities.append(
            {
                "cik": normalize_cik(cik_match.group(1)),
                "name": name,
                "form": html.unescape(form_match.group(1)).strip() if form_match else "",
                "sic": sic_match.group(1) if sic_match else "",
                "sic_description": (html.unescape(sic_match.group(2)).strip() if sic_match else ""),
            }
        )
    if not entities:
        sic_match = re.search(
            r"SIC</acronym>\s*:\s*<b>\s*<a[^>]*(?:[?&]|&amp;)SIC=(\d{4})"
            r"[^>]*>\d{4}</a>\s*</b>\s*([^<]*?)(?:<br|</p>)",
            source,
            flags=re.IGNORECASE,
        )
        if sic_match:
            entities.append(
                {
                    "cik": "",
                    "name": "",
                    "form": "",
                    "sic": sic_match.group(1),
                    "sic_description": html.unescape(sic_match.group(2)).strip(),
                }
            )
    return entities


def _observation_intervals(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    columns = [
        "cik",
        "accession_number",
        "observed_at",
        "valid_from_inclusive",
        "valid_to_exclusive",
        "interval_backfilled_before_observation",
        *value_columns,
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    source = frame.copy()
    source["accepted_at"] = pd.to_datetime(source["accepted_at"], utc=True)
    source = source.sort_values(["cik", "accepted_at", "accession_number"])
    event_times = source[["cik", "accepted_at"]].drop_duplicates()
    event_times["valid_to_exclusive"] = event_times.groupby("cik")["accepted_at"].shift(-1)
    source = source.merge(
        event_times,
        on=["cik", "accepted_at"],
        how="left",
        validate="many_to_one",
    )
    result = source[["cik", "accession_number", "accepted_at", *value_columns]].copy()
    result = result.rename(columns={"accepted_at": "observed_at"})
    result["valid_from_inclusive"] = result["observed_at"]
    result["valid_to_exclusive"] = source["valid_to_exclusive"]
    result["interval_backfilled_before_observation"] = False
    return result[columns]


def _cover_rejection(
    common: dict[str, Any], reason: str, snapshot: FilingArtifactSnapshot
) -> dict[str, Any]:
    return {
        **common,
        "reason": reason,
        "source_artifact_kind": snapshot.artifact_kind,
        "source_sha256": snapshot.sha256,
        "source_path": snapshot.path,
    }


def _normalized_reported_cik(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    if not digits or len(digits) > 10:
        return None
    return normalize_cik(digits)


def _clean_cover_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", html.unescape(value)).strip()
    return normalized or None


def _valid_trading_symbol(value: str | None) -> str | None:
    normalized = _clean_cover_value(value)
    if normalized is None or normalized in {"—", "–", "-", "N/A", "NA", "NONE"}:
        return None
    return normalized.upper()


def _truthy(value: str | None) -> bool:
    return (_clean_cover_value(value) or "").lower() in {"true", "1", "yes"}


def _local_concept(value: str) -> str:
    return value.split(":")[-1]


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _filing_index_url(cik: str, accession_number: str) -> str:
    cik_unpadded = str(int(cik))
    accession_compact = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_compact}/"
        f"{accession_number}-index.html"
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
