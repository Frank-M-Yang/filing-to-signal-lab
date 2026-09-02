"""Collect and audit filing-time SIC evidence for a frozen validation cohort.

This layer is deliberately outcome blind.  It binds each request to an exact
CIK, accession, and SEC acceptance timestamp from the frozen validation market
plan, preserves the response bytes before parsing, and records missing evidence
as missing rather than replacing the event.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.filing_cover import (
    FilingArtifactSnapshot,
    load_filing_artifact_snapshots,
    parse_filing_index,
)

VALIDATION_FILING_EVIDENCE_TARGET_VERSION: Final[str] = "validation-filing-evidence-targets-v1"
VALIDATION_FILING_EVIDENCE_AUDIT_VERSION: Final[str] = "validation-filing-evidence-audit-v1"
VALIDATION_FILING_EVIDENCE_PARSER_VERSION: Final[str] = "validation-filing-index-sic-v1"


class ValidationFilingEvidenceError(FilingCatalogError):
    """Raised when validation evidence violates its frozen-data contract."""


class ValidationFilingEvidencePlanResult:
    """Immutable SEC filing-index target plan."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


class ValidationFilingEvidenceAuditResult:
    """Immutable audit of preserved filing-index evidence."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


def build_validation_filing_evidence_plan(
    market_plan_root: Path,
    *,
    output_root: Path = Path("data/sec/validation_filing_evidence/targets"),
) -> ValidationFilingEvidencePlanResult:
    """Persist exact SEC index requests from one outcome-blind market plan."""
    market_report = _read_json(market_plan_root / "validation_market_plan_report.json")
    if bool(market_report.get("outcome_values_loaded")):
        raise ValidationFilingEvidenceError("Market plan has already opened outcome values")
    if bool(market_report.get("selection_uses_return_values")):
        raise ValidationFilingEvidenceError("Market plan selection used return values")
    market_manifest_sha256 = _sha256_file(market_plan_root / "output_manifest.json")
    source = load_catalog_table(market_plan_root, "validation_filing_evidence_targets.parquet")
    _require_columns(
        source,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "target_accession_number",
            "accepted_at",
            "catalog_accepted_at",
            "catalog_clock_match",
            "index_url",
            "primary_document_url",
            "sec_index_collection_required",
        },
        "validation filing evidence targets",
    )
    if _has_outcome_columns(source):
        raise ValidationFilingEvidenceError("Filing evidence targets contain outcome columns")
    targets = source.loc[source["sec_index_collection_required"].astype(bool)].copy()
    if targets.empty:
        raise ValidationFilingEvidenceError("No SEC filing-index targets are authorized")
    targets["cik"] = targets["cik"].astype(str).str.zfill(10)
    targets["accepted_at"] = pd.to_datetime(targets["accepted_at"], utc=True, errors="coerce")
    targets["catalog_accepted_at"] = pd.to_datetime(
        targets["catalog_accepted_at"], utc=True, errors="coerce"
    )
    if targets[["accepted_at", "catalog_accepted_at"]].isna().any().any():
        raise ValidationFilingEvidenceError("SEC targets contain invalid acceptance clocks")
    if not targets["accepted_at"].eq(targets["catalog_accepted_at"]).all():
        raise ValidationFilingEvidenceError("SEC targets are not tied to the catalog clock")
    if targets["market_event_id"].isna().any() or targets["market_event_id"].duplicated().any():
        raise ValidationFilingEvidenceError("Market event IDs must be non-null and unique")
    targets["target_id"] = targets["market_event_id"].astype(str)
    targets["accession_number"] = targets["target_accession_number"].astype(str)
    targets["artifact_policy"] = "index-only"
    targets["selection_uses_return_values"] = False
    targets["replacement_after_collection_failure_allowed"] = False
    targets = targets.sort_values(
        ["accepted_at", "cik", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)
    targets["acquisition_order"] = targets.index + 1
    columns = [
        "target_id",
        "acquisition_order",
        "market_event_id",
        "validation_event_id",
        "cik",
        "accession_number",
        "accepted_at",
        "catalog_accepted_at",
        "catalog_clock_match",
        "primary_document_url",
        "index_url",
        "artifact_policy",
        "selection_uses_return_values",
        "replacement_after_collection_failure_allowed",
    ]
    targets = targets[columns].copy()
    payload = json.dumps(
        {
            "market_plan_manifest_sha256": market_manifest_sha256,
            "target_version": VALIDATION_FILING_EVIDENCE_TARGET_VERSION,
            "artifact_policy": "index-only",
            "order_rule": "accepted_at_then_cik_then_accession",
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_FILING_EVIDENCE_TARGET_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationFilingEvidencePlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationFilingEvidenceError(f"Incomplete target plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "validation_filing_evidence_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifacts = {target_path.name: _artifact_record(target_path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_plan_version": market_report["dataset_version"],
        "market_plan_manifest_sha256": market_manifest_sha256,
        "target_events": len(targets),
        "target_issuers": int(targets["cik"].nunique()),
        "artifact_policy": "index-only",
        "acquisition_order_rule": "accepted_at_then_cik_then_accession",
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "outcome_values_loaded": False,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "market_plan_manifest_sha256": market_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationFilingEvidencePlanResult(target.resolve(), report)


def load_validation_filing_evidence_targets(plan_root: Path) -> pd.DataFrame:
    """Load and verify one validation filing-evidence target plan."""
    return load_catalog_table(plan_root, "validation_filing_evidence_targets.parquet")


def calculate_validation_filing_evidence(
    targets: pd.DataFrame,
    snapshots: list[FilingArtifactSnapshot],
    *,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse preserved index pages and return evidence, rejections, gates, and metrics."""
    _require_columns(
        targets,
        {
            "target_id",
            "market_event_id",
            "validation_event_id",
            "cik",
            "accession_number",
            "accepted_at",
            "index_url",
            "selection_uses_return_values",
        },
        "validation filing targets",
    )
    if _has_outcome_columns(targets):
        raise ValidationFilingEvidenceError("Validation filing targets contain outcomes")
    if targets["selection_uses_return_values"].astype(bool).any():
        raise ValidationFilingEvidenceError("Validation filing targets used return values")
    snapshot_lookup = {
        (snapshot.target_id, snapshot.artifact_kind): snapshot for snapshot in snapshots
    }
    rows: list[dict[str, Any]] = []
    for target in targets.sort_values("acquisition_order").itertuples(index=False):
        target_id = str(target.target_id)
        expected_cik = str(target.cik).zfill(10)
        snapshot = snapshot_lookup.get((target_id, "index"))
        common: dict[str, Any] = {
            "target_id": target_id,
            "market_event_id": str(target.market_event_id),
            "validation_event_id": str(target.validation_event_id),
            "cik": expected_cik,
            "accession_number": str(target.accession_number),
            "accepted_at": target.accepted_at,
            "index_url": str(target.index_url),
            "selection_uses_return_values": False,
            "outcome_values_loaded": False,
        }
        if snapshot is None:
            rows.append(
                {
                    **common,
                    "artifact_preserved": False,
                    "source_sha256": None,
                    "source_bytes": None,
                    "source_fetched_at": None,
                    "source_path_relative_to_raw_root": None,
                    "response_url": None,
                    "index_subject_cik": None,
                    "index_subject_cik_matched": False,
                    "index_subject_name": None,
                    "index_subject_form": None,
                    "sic": None,
                    "sic_description": None,
                    "filing_time_industry_ready": False,
                    "evidence_status": "not_collected",
                    "rejection_reason": "missing_preserved_index_artifact",
                }
            )
            continue
        metadata = parse_filing_index(
            Path(snapshot.path).read_bytes(),
            index_url=snapshot.requested_url,
            subject_cik=expected_cik,
        )
        subject_cik = _normalized_cik(metadata.get("subject_cik"))
        subject_match = subject_cik == expected_cik
        sic = str(metadata["sic"]).strip() if metadata.get("sic") is not None else None
        sic_ready = bool(subject_match and sic and re.fullmatch(r"\d{4}", sic))
        if not subject_match:
            evidence_status = "rejected"
            rejection_reason = "filing_index_subject_cik_mismatch"
        elif not sic_ready:
            evidence_status = "rejected"
            rejection_reason = "missing_or_invalid_four_digit_sic"
        else:
            evidence_status = "ready"
            rejection_reason = None
        rows.append(
            {
                **common,
                "artifact_preserved": True,
                "source_sha256": snapshot.sha256,
                "source_bytes": snapshot.bytes,
                "source_fetched_at": snapshot.fetched_at,
                "source_path_relative_to_raw_root": _relative_snapshot_path(snapshot, raw_root),
                "response_url": snapshot.response_url,
                "index_subject_cik": subject_cik,
                "index_subject_cik_matched": subject_match,
                "index_subject_name": metadata.get("subject_name"),
                "index_subject_form": metadata.get("subject_form"),
                "sic": sic,
                "sic_description": metadata.get("sic_description"),
                "filing_time_industry_ready": sic_ready,
                "evidence_status": evidence_status,
                "rejection_reason": rejection_reason,
            }
        )
    evidence = pd.DataFrame(rows)
    rejections = evidence.loc[
        evidence["rejection_reason"].notna(),
        [
            "target_id",
            "market_event_id",
            "validation_event_id",
            "cik",
            "accession_number",
            "accepted_at",
            "evidence_status",
            "rejection_reason",
            "source_sha256",
        ],
    ].copy()
    acquired = int(evidence["artifact_preserved"].sum())
    ready = int(evidence["filing_time_industry_ready"].sum())
    gates = pd.DataFrame(
        [
            _gate(
                "filing_index_acquisition",
                acquired,
                len(evidence),
                acquired == len(evidence),
                "Exact filing-index bytes exist for every frozen validation event",
            ),
            _gate(
                "filing_time_industry",
                ready,
                len(evidence),
                ready == len(evidence),
                "Exact-subject four-digit SIC is parsed from each event filing index",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(evidence),
                False,
                "Locked until all market-data evidence gates pass",
            ),
        ]
    )
    report: dict[str, Any] = {
        "target_events": len(evidence),
        "target_issuers": int(evidence["cik"].nunique()),
        "index_artifacts_preserved": acquired,
        "acquisition_coverage": acquired / len(evidence) if len(evidence) else 0.0,
        "filing_time_industry_ready": ready,
        "industry_coverage": ready / len(evidence) if len(evidence) else 0.0,
        "subject_cik_matches": int(evidence["index_subject_cik_matched"].sum()),
        "rejected_or_missing": len(rejections),
        "outcome_values_loaded": False,
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "return_outcome_read_authorized": False,
    }
    return evidence, rejections, gates, report


def build_validation_filing_evidence_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/validation_filing_evidence"),
    output_root: Path = Path("data/processed/validation_filing_evidence"),
) -> ValidationFilingEvidenceAuditResult:
    """Build an immutable audit from verified raw filing-index snapshots."""
    plan_report = _read_json(plan_root / "target_report.json")
    if bool(plan_report.get("outcome_values_loaded")):
        raise ValidationFilingEvidenceError("Target plan has opened outcome data")
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    targets = load_validation_filing_evidence_targets(plan_root)
    target_ids = set(targets["target_id"].astype(str))
    snapshots = [
        snapshot
        for snapshot in load_filing_artifact_snapshots(raw_root)
        if snapshot.target_id in target_ids and snapshot.artifact_kind == "index"
    ]
    evidence, rejections, gates, metrics = calculate_validation_filing_evidence(
        targets, snapshots, raw_root=raw_root
    )
    snapshot_payload = [
        {
            "target_id": snapshot.target_id,
            "kind": snapshot.artifact_kind,
            "sha256": snapshot.sha256,
        }
        for snapshot in sorted(snapshots, key=lambda item: (item.target_id, item.sha256))
    ]
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "parser_version": VALIDATION_FILING_EVIDENCE_PARSER_VERSION,
            "snapshots": snapshot_payload,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_FILING_EVIDENCE_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationFilingEvidenceAuditResult(
            target.resolve(), _read_json(target / "validation_filing_evidence_report.json")
        )
    if target.exists():
        raise ValidationFilingEvidenceError(f"Incomplete filing evidence audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_filing_industry_evidence.parquet": evidence,
        "validation_filing_evidence_rejections.parquet": rejections,
        "validation_filing_evidence_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        **metrics,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "parser_version": VALIDATION_FILING_EVIDENCE_PARSER_VERSION,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_filing_evidence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationFilingEvidenceAuditResult(target.resolve(), report)


def _has_outcome_columns(frame: pd.DataFrame) -> bool:
    return any(
        column in frame.columns
        for column in (
            "issuer_total_return",
            "market_total_return",
            "market_excess_return_arithmetic",
            "sector_excess_return",
            "future_return",
            "delisting_return",
        )
    )


def _normalized_cik(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits or len(digits) > 10:
        return None
    return digits.zfill(10)


def _relative_snapshot_path(snapshot: FilingArtifactSnapshot, raw_root: Path) -> str:
    try:
        return Path(snapshot.path).resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationFilingEvidenceError(
            "Preserved SEC artifact is outside the configured raw root"
        ) from exc


def _gate(
    gate: str,
    numerator: int,
    denominator: int,
    passed: bool,
    meaning: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "numerator": numerator,
        "denominator": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
        "passed": passed,
        "meaning": meaning,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationFilingEvidenceError(f"{label} lacks required columns: {missing}")


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
