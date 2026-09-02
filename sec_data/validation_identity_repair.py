"""Outcome-blind identity repair evidence for unresolved validation issuers."""

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
    load_filing_artifact_snapshots,
    parse_filing_security_evidence,
)

VALIDATION_IDENTITY_REPAIR_TARGET_VERSION: Final[str] = "validation-identity-repair-targets-v1"
VALIDATION_IDENTITY_COVER_AUDIT_VERSION: Final[str] = (
    "validation-identity-cover-audit-v2-security-classification"
)
VALIDATION_SYMBOL_FALLBACK_TARGET_VERSION: Final[str] = "validation-symbol-fallback-targets-v1"
VALIDATION_IDENTITY_REPAIR_RESOLUTION_VERSION: Final[str] = (
    "validation-identity-repair-resolution-v2-null-safe"
)


class ValidationIdentityRepairError(FilingCatalogError):
    """Raised when identity repair violates its frozen evidence contract."""


class ValidationIdentityRepairResult:
    """Immutable identity-repair asset."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


def build_validation_identity_repair_plan(
    market_plan_root: Path,
    identity_audit_root: Path,
    reference_audit_root: Path,
    *,
    output_root: Path = Path("data/sec/validation_identity_repair/targets"),
) -> ValidationIdentityRepairResult:
    """Freeze SEC primary-document targets for every unresolved identity endpoint."""
    market_manifest = _sha256_file(market_plan_root / "output_manifest.json")
    identity_manifest = _sha256_file(identity_audit_root / "output_manifest.json")
    reference_manifest = _sha256_file(reference_audit_root / "output_manifest.json")
    identity_report = _read_json(identity_audit_root / "validation_security_identity_report.json")
    if bool(identity_report.get("outcome_values_loaded")):
        raise ValidationIdentityRepairError("Identity audit has opened outcome data")
    intervals = load_catalog_table(
        identity_audit_root, "validation_issuer_identity_intervals.parquet"
    )
    unresolved = intervals.loc[
        intervals["identity_interval_status"].eq("endpoint_reference_unresolved"), "cik"
    ].astype(str)
    if unresolved.empty:
        raise ValidationIdentityRepairError("No unresolved identity issuers require repair")
    anchors = load_catalog_table(market_plan_root, "validation_identity_anchor_targets.parquet")
    filings = load_catalog_table(market_plan_root, "validation_filing_evidence_targets.parquet")
    reference = load_catalog_table(reference_audit_root, "sample_massive_reference_audit.parquet")
    for frame, label in (
        (anchors, "identity anchors"),
        (filings, "filing evidence targets"),
        (reference, "reference audit"),
    ):
        if _has_outcome_columns(frame):
            raise ValidationIdentityRepairError(f"{label} contains outcome columns")
    anchors["cik"] = anchors["cik"].astype(str).str.zfill(10)
    repair = anchors.loc[anchors["cik"].isin(set(unresolved.str.zfill(10)))].copy()
    filings["cik"] = filings["cik"].astype(str).str.zfill(10)
    repair = repair.merge(
        filings[
            [
                "market_event_id",
                "validation_event_id",
                "cik",
                "target_accession_number",
                "index_url",
                "primary_document_url",
                "catalog_clock_match",
            ]
        ],
        on=["validation_event_id", "cik"],
        how="left",
        validate="one_to_one",
    )
    reference_columns = [
        "target_id",
        "provider_query_status",
        "event_date_reference_status",
        "reference_candidate_count",
    ]
    repair = repair.merge(
        reference[reference_columns].rename(columns={"target_id": "identity_anchor_id"}),
        on="identity_anchor_id",
        how="left",
        validate="one_to_one",
    )
    if repair["market_event_id"].isna().any() or not repair["catalog_clock_match"].all():
        raise ValidationIdentityRepairError("Repair anchors do not tie to exact filing clocks")
    if len(repair) != len(unresolved) * 2:
        raise ValidationIdentityRepairError("Unresolved issuers do not have two endpoint anchors")
    repair["target_id"] = repair["market_event_id"].astype(str)
    repair["accession_number"] = repair["target_accession_number"].astype(str)
    repair["artifact_policy"] = "index-and-cover"
    repair["selection_uses_return_values"] = False
    repair["replacement_after_collection_failure_allowed"] = False
    repair["repair_policy"] = repair["event_date_reference_status"].map(
        {
            "provider_no_cik_match": "sec_symbol_then_ticker_date_provider_fallback",
            "ambiguous_common_stock": "sec_registered_symbols_filter_or_multiclass_basket",
            "missing_provider_security_id": "sec_cover_plus_provider_ticker_exchange_composite",
        }
    )
    if repair["repair_policy"].isna().any():
        raise ValidationIdentityRepairError("An unresolved endpoint lacks a frozen repair policy")
    repair = repair.sort_values(
        ["accepted_at", "cik", "anchor_role"], kind="mergesort"
    ).reset_index(drop=True)
    repair["acquisition_order"] = repair.index + 1
    columns = [
        "target_id",
        "acquisition_order",
        "market_event_id",
        "identity_anchor_id",
        "validation_event_id",
        "cik",
        "anchor_role",
        "accession_number",
        "accepted_at",
        "query_date",
        "index_url",
        "primary_document_url",
        "provider_query_status",
        "event_date_reference_status",
        "reference_candidate_count",
        "artifact_policy",
        "repair_policy",
        "selection_uses_return_values",
        "replacement_after_collection_failure_allowed",
    ]
    repair = repair[columns].copy()
    payload = json.dumps(
        {
            "market_manifest": market_manifest,
            "identity_manifest": identity_manifest,
            "reference_manifest": reference_manifest,
            "version": VALIDATION_IDENTITY_REPAIR_TARGET_VERSION,
            "policies": sorted(repair["repair_policy"].unique().tolist()),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_IDENTITY_REPAIR_TARGET_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationIdentityRepairResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationIdentityRepairError(f"Incomplete repair target plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "validation_identity_repair_filing_targets.parquet"
    repair.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(repair))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_plan_manifest_sha256": market_manifest,
        "identity_audit_manifest_sha256": identity_manifest,
        "reference_audit_manifest_sha256": reference_manifest,
        "repair_issuers": int(repair["cik"].nunique()),
        "repair_endpoint_targets": len(repair),
        "sec_primary_documents_required": len(repair),
        "repair_policy_counts": repair["repair_policy"].value_counts().to_dict(),
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
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationIdentityRepairResult(target.resolve(), report)


def load_validation_identity_repair_targets(plan_root: Path) -> pd.DataFrame:
    """Load and verify frozen SEC identity-repair targets."""
    return load_catalog_table(plan_root, "validation_identity_repair_filing_targets.parquet")


def build_validation_identity_cover_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/validation_filing_evidence"),
    output_root: Path = Path("data/processed/validation_identity_cover_repair"),
) -> ValidationIdentityRepairResult:
    """Parse preserved SEC primary documents and prepare ticker fallback targets."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    targets = load_validation_identity_repair_targets(plan_root)
    target_ids = set(targets["target_id"].astype(str))
    snapshots = [
        snapshot
        for snapshot in load_filing_artifact_snapshots(raw_root)
        if snapshot.target_id in target_ids
    ]
    lookup = {(item.target_id, item.artifact_kind): item for item in snapshots}
    security_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for target in targets.sort_values("acquisition_order").itertuples(index=False):
        target_id = str(target.target_id)
        index = lookup.get((target_id, "index"))
        primary = lookup.get((target_id, "primary"))
        common = {
            "target_id": target_id,
            "market_event_id": str(target.market_event_id),
            "identity_anchor_id": str(target.identity_anchor_id),
            "validation_event_id": str(target.validation_event_id),
            "cik": str(target.cik).zfill(10),
            "anchor_role": str(target.anchor_role),
            "accession_number": str(target.accession_number),
            "accepted_at": target.accepted_at,
            "query_date": str(target.query_date),
            "prior_reference_status": str(target.event_date_reference_status),
            "repair_policy": str(target.repair_policy),
            "selection_uses_return_values": False,
            "outcome_values_loaded": False,
        }
        observations: list[dict[str, Any]] = []
        if index is not None and primary is not None:
            observations = parse_filing_security_evidence(
                Path(primary.path).read_bytes(),
                Path(index.path).read_bytes(),
                index_url=index.requested_url,
                subject_cik=str(target.cik),
            )
        subject_tickers = [
            observation
            for observation in observations
            if observation.get("security_subject_cik_match") and observation.get("trading_symbol")
        ]
        equity_subject_tickers = [
            observation
            for observation in subject_tickers
            if _security_kind(observation.get("security_title")) == "common_equity"
        ]
        tradable_equity = [
            observation
            for observation in equity_subject_tickers
            if observation.get("tradable_security_mapping_ready")
        ]
        for occurrence, observation in enumerate(observations):
            ticker = _normalized_ticker(observation.get("trading_symbol"))
            exchange = _normalized_exchange_mic(observation.get("security_exchange_name"))
            security_kind = _security_kind(observation.get("security_title"))
            security_rows.append(
                {
                    **common,
                    "source_occurrence": occurrence,
                    "trading_symbol": ticker,
                    "reported_exchange": observation.get("security_exchange_name"),
                    "normalized_exchange_mic": exchange,
                    "security_title": observation.get("security_title"),
                    "security_kind": security_kind,
                    "common_equity_candidate": security_kind == "common_equity",
                    "security_subject_cik_match": bool(
                        observation.get("security_subject_cik_match")
                    ),
                    "tradable_security_mapping_ready": bool(
                        observation.get("tradable_security_mapping_ready")
                    ),
                    "security_evidence_kind": observation.get("security_evidence_kind"),
                    "security_evidence_confidence": observation.get("security_evidence_confidence"),
                    "primary_source_sha256": primary.sha256 if primary else None,
                    "index_source_sha256": index.sha256 if index else None,
                }
            )
        subject_equity_tickers = sorted(
            {
                str(observation["trading_symbol"]).strip().upper()
                for observation in equity_subject_tickers
                if observation.get("trading_symbol")
            }
        )
        tradable_equity_tickers = sorted(
            {
                str(observation["trading_symbol"]).strip().upper()
                for observation in tradable_equity
                if observation.get("trading_symbol")
            }
        )
        symbol_ready = bool(index is not None and primary is not None and subject_equity_tickers)
        tradable_ready = bool(index is not None and primary is not None and tradable_equity_tickers)
        endpoint_rows.append(
            {
                **common,
                "index_artifact_preserved": index is not None,
                "primary_artifact_preserved": primary is not None,
                "sec_subject_equity_ticker_count": len(subject_equity_tickers),
                "sec_subject_equity_tickers": "|".join(subject_equity_tickers),
                "sec_tradable_equity_count": len(tradable_equity_tickers),
                "sec_tradable_equity_tickers": "|".join(tradable_equity_tickers),
                "sec_cover_symbol_ready": symbol_ready,
                "sec_cover_tradable_mapping_ready": tradable_ready,
                "ticker_fallback_required": (
                    symbol_ready
                    and str(target.event_date_reference_status) == "provider_no_cik_match"
                ),
                "repair_status": (
                    "sec_cover_tradable_ready"
                    if tradable_ready
                    else "sec_cover_symbol_ready_exchange_missing"
                    if symbol_ready
                    else "sec_cover_missing_or_unresolved"
                ),
                "primary_source_sha256": primary.sha256 if primary else None,
                "index_source_sha256": index.sha256 if index else None,
            }
        )
        if symbol_ready and str(target.event_date_reference_status) == "provider_no_cik_match":
            for ticker in subject_equity_tickers:
                fallback_rows.append(
                    {
                        "target_id": _stable_hash(
                            {
                                "identity_anchor_id": str(target.identity_anchor_id),
                                "ticker": ticker,
                                "query_date": str(target.query_date),
                                "version": VALIDATION_SYMBOL_FALLBACK_TARGET_VERSION,
                            }
                        ),
                        "identity_anchor_id": str(target.identity_anchor_id),
                        "validation_event_id": str(target.validation_event_id),
                        "cik": str(target.cik).zfill(10),
                        "query_ticker": ticker,
                        "equity_trading_symbols": ticker,
                        "query_date": str(target.query_date),
                        "anchor_role": str(target.anchor_role),
                        "selection_uses_return_values": False,
                    }
                )
    securities = pd.DataFrame(security_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    fallback = pd.DataFrame(
        fallback_rows,
        columns=[
            "target_id",
            "identity_anchor_id",
            "validation_event_id",
            "cik",
            "query_ticker",
            "equity_trading_symbols",
            "query_date",
            "anchor_role",
            "selection_uses_return_values",
        ],
    )
    if not fallback.empty:
        fallback = fallback.sort_values(
            ["query_date", "cik", "query_ticker"], kind="mergesort"
        ).reset_index(drop=True)
        fallback["massive_acquisition_order"] = fallback.index + 1
    else:
        fallback["massive_acquisition_order"] = pd.Series(dtype="int64")
    snapshot_payload = [
        {"target_id": item.target_id, "kind": item.artifact_kind, "sha256": item.sha256}
        for item in sorted(snapshots, key=lambda value: (value.target_id, value.artifact_kind))
    ]
    payload = json.dumps(
        {
            "plan_manifest": plan_manifest,
            "snapshots": snapshot_payload,
            "version": VALIDATION_IDENTITY_COVER_AUDIT_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_IDENTITY_COVER_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return ValidationIdentityRepairResult(
            target_root.resolve(), _read_json(target_root / "cover_repair_report.json")
        )
    if target_root.exists():
        raise ValidationIdentityRepairError(f"Incomplete cover audit exists: {target_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_identity_repair_security_evidence.parquet": securities,
        "validation_identity_repair_endpoint_summary.parquet": endpoints,
        "validation_symbol_fallback_candidates.parquet": fallback,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    acquired = int(endpoints["primary_artifact_preserved"].sum())
    symbol_ready_count = int(endpoints["sec_cover_symbol_ready"].sum())
    tradable_ready_count = int(endpoints["sec_cover_tradable_mapping_ready"].sum())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "repair_endpoint_targets": len(endpoints),
        "primary_artifacts_preserved": acquired,
        "sec_cover_symbol_ready": symbol_ready_count,
        "sec_cover_tradable_mapping_ready": tradable_ready_count,
        "security_evidence_rows": len(securities),
        "ticker_fallback_queries": len(fallback),
        "ticker_fallback_issuers": int(fallback["cik"].nunique()) if not fallback.empty else 0,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_collection_authorized": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "cover_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target_root)
    return ValidationIdentityRepairResult(target_root.resolve(), report)


def build_validation_symbol_fallback_plan(
    cover_audit_root: Path,
    *,
    output_root: Path = Path("data/market/validation_identity_symbol_fallback/targets"),
) -> ValidationIdentityRepairResult:
    """Persist Massive-compatible ticker/date targets derived only from SEC evidence."""
    cover_report = _read_json(cover_audit_root / "cover_repair_report.json")
    if bool(cover_report.get("outcome_values_loaded")):
        raise ValidationIdentityRepairError("Cover repair audit has opened outcome data")
    cover_manifest = _sha256_file(cover_audit_root / "output_manifest.json")
    targets = load_catalog_table(cover_audit_root, "validation_symbol_fallback_candidates.parquet")
    if targets.empty:
        raise ValidationIdentityRepairError("No SEC-grounded ticker fallback targets exist")
    if _has_outcome_columns(targets) or targets["selection_uses_return_values"].astype(bool).any():
        raise ValidationIdentityRepairError("Ticker fallback targets use outcome data")
    targets = targets.copy()
    targets["calendar_year"] = pd.to_datetime(targets["query_date"]).dt.year
    targets["accepted_at"] = pd.to_datetime(targets["query_date"], utc=True)
    targets["selection_uses_provider_coverage"] = False
    targets["automatic_interval_propagation_allowed"] = False
    targets["price_collection_authorized"] = False
    targets["outcome_values_loaded"] = False
    payload = json.dumps(
        {
            "cover_manifest": cover_manifest,
            "version": VALIDATION_SYMBOL_FALLBACK_TARGET_VERSION,
            "query_method": "sec_symbol_and_date_then_expected_cik_validation",
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_SYMBOL_FALLBACK_TARGET_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return ValidationIdentityRepairResult(
            target_root.resolve(), _read_json(target_root / "target_report.json")
        )
    if target_root.exists():
        raise ValidationIdentityRepairError(f"Incomplete symbol plan exists: {target_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "massive_reference_targets.parquet"
    targets.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_version": cover_report["dataset_version"],
        "scope_manifest_sha256": cover_manifest,
        "cover_audit_version": cover_report["dataset_version"],
        "cover_audit_manifest_sha256": cover_manifest,
        "price_mapping_targets": len(targets),
        "symbol_fallback_targets": len(targets),
        "distinct_ciks": int(targets["cik"].nunique()),
        "query_method": "sec_symbol_and_date_then_expected_cik_validation",
        "selection_uses_provider_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_collection_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "scope_manifest_sha256": cover_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target_root)
    return ValidationIdentityRepairResult(target_root.resolve(), report)


def build_validation_identity_repair_resolution(
    market_plan_root: Path,
    base_identity_audit_root: Path,
    original_reference_audit_root: Path,
    repair_plan_root: Path,
    cover_audit_root: Path,
    fallback_plan_root: Path,
    fallback_reference_audit_root: Path,
    filing_evidence_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_identity_repair_resolution"),
) -> ValidationIdentityRepairResult:
    """Combine exact SEC and provider evidence into final outcome-blind identities."""
    input_roots = {
        "market_plan": market_plan_root,
        "base_identity_audit": base_identity_audit_root,
        "original_reference_audit": original_reference_audit_root,
        "repair_plan": repair_plan_root,
        "cover_audit": cover_audit_root,
        "fallback_plan": fallback_plan_root,
        "fallback_reference_audit": fallback_reference_audit_root,
        "filing_evidence": filing_evidence_root,
    }
    manifests = {
        label: _sha256_file(root / "output_manifest.json") for label, root in input_roots.items()
    }
    repair_targets = load_validation_identity_repair_targets(repair_plan_root)
    cover_securities = load_catalog_table(
        cover_audit_root, "validation_identity_repair_security_evidence.parquet"
    )
    original_observations = load_catalog_table(
        original_reference_audit_root, "massive_reference_observations.parquet"
    )
    fallback_targets = load_catalog_table(fallback_plan_root, "massive_reference_targets.parquet")
    fallback_snapshots = load_catalog_table(
        fallback_reference_audit_root, "massive_reference_snapshots.parquet"
    )
    generic_fallback_rejections = load_catalog_table(
        fallback_reference_audit_root, "massive_reference_rejections.parquet"
    )
    filing_evidence = load_catalog_table(
        filing_evidence_root, "validation_filing_industry_evidence.parquet"
    )
    base_events = load_catalog_table(
        base_identity_audit_root, "validation_event_identity_readiness.parquet"
    )
    for frame, label in (
        (repair_targets, "repair targets"),
        (cover_securities, "cover securities"),
        (original_observations, "original reference observations"),
        (fallback_targets, "fallback targets"),
        (base_events, "base event readiness"),
    ):
        if _has_outcome_columns(frame):
            raise ValidationIdentityRepairError(f"{label} contains outcome columns")
    issuer_name_values = (
        filing_evidence[["cik", "index_subject_name"]]
        .dropna()
        .assign(cik=lambda frame: frame["cik"].astype(str).str.zfill(10))
        .drop_duplicates("cik")
        .set_index("cik")["index_subject_name"]
        .astype(str)
        .to_dict()
    )
    issuer_names: dict[str, str] = {str(cik): str(name) for cik, name in issuer_name_values.items()}
    fallback_candidates, fallback_rejections = calculate_validation_fallback_candidates(
        fallback_targets,
        fallback_snapshots,
        issuer_names,
    )
    components = calculate_validation_identity_repair_components(
        repair_targets,
        cover_securities,
        original_observations,
        fallback_candidates,
    )
    endpoints = _repair_endpoint_summary(repair_targets, cover_securities, components)
    issuers = _repair_issuer_summary(endpoints, components, issuer_names)
    final_events = _apply_repairs_to_events(base_events, issuers, components)
    ready_events = int(final_events["event_security_identity_ready"].sum())
    ready_issuers = int(issuers["repair_ready"].sum())
    gates = pd.DataFrame(
        [
            _gate(
                "repair_endpoint_evidence",
                int(endpoints["endpoint_repair_ready"].sum()),
                len(endpoints),
                bool(endpoints["endpoint_repair_ready"].all()),
                "Every unresolved endpoint has complete SEC/provider component identity",
            ),
            _gate(
                "repair_issuer_consistency",
                ready_issuers,
                len(issuers),
                bool(issuers["repair_ready"].all()),
                "First and last endpoints retain identical security component identities",
            ),
            _gate(
                "price_candidate_event_identity",
                ready_events,
                len(final_events),
                ready_events == len(final_events),
                "Every frozen price-candidate event has an evidence-backed identity",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(final_events),
                False,
                "Locked pending calendar, actions, prices, and delisting evidence",
            ),
        ]
    )
    payload = json.dumps(
        {
            "manifests": manifests,
            "version": VALIDATION_IDENTITY_REPAIR_RESOLUTION_VERSION,
            "missing_provider_cik_rule": (
                "exact_sec_ticker_and_normalized_sec_name_and_common_stock_and_stable_figi"
            ),
            "multiclass_rule": "equal_weighted_share_class_basket_at_issuer_event_level",
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_IDENTITY_REPAIR_RESOLUTION_VERSION
        + "-"
        + hashlib.sha256(payload).hexdigest()[:16]
    )
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return ValidationIdentityRepairResult(
            target_root.resolve(), _read_json(target_root / "identity_repair_report.json")
        )
    if target_root.exists():
        raise ValidationIdentityRepairError(f"Incomplete repair resolution exists: {target_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_identity_fallback_candidates.parquet": fallback_candidates,
        "validation_identity_fallback_rejections.parquet": fallback_rejections,
        "validation_identity_generic_fallback_rejections.parquet": (generic_fallback_rejections),
        "validation_identity_repair_components.parquet": components,
        "validation_identity_repair_endpoints.parquet": endpoints,
        "validation_identity_repair_issuers.parquet": issuers,
        "validation_event_identity_readiness_final.parquet": final_events,
        "validation_identity_repair_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": manifests,
        "repair_issuers": len(issuers),
        "repair_issuers_ready": ready_issuers,
        "repair_endpoints": len(endpoints),
        "repair_endpoints_ready": int(endpoints["endpoint_repair_ready"].sum()),
        "repair_security_components": len(components),
        "fallback_candidates_accepted": len(fallback_candidates),
        "fallback_rows_rejected": len(fallback_rejections),
        "generic_fallback_rows_rejected": len(generic_fallback_rejections),
        "price_candidate_events": len(final_events),
        "price_candidate_events_identity_ready": ready_events,
        "price_candidate_event_identity_coverage": (
            ready_events / len(final_events) if len(final_events) else 0.0
        ),
        "price_candidate_events_requiring_identity_repair": int(
            (~final_events["event_security_identity_ready"].astype(bool)).sum()
        ),
        "missing_provider_cik_acceptance_rule": (
            "exact SEC ticker + normalized SEC issuer name + active common stock + "
            "stable FIGI at both endpoints"
        ),
        "multiclass_return_rule": "equal_weighted_share_class_basket_at_issuer_event_level",
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_collection_authorized": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "identity_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target_root)
    return ValidationIdentityRepairResult(target_root.resolve(), report)


def calculate_validation_fallback_candidates(
    targets: pd.DataFrame,
    snapshots: pd.DataFrame,
    issuer_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept missing-CIK provider rows only with exact SEC ticker/name evidence."""
    candidate_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_id = str(target.target_id)
        expected_cik = str(target.cik).zfill(10)
        expected_ticker = str(target.query_ticker).strip().upper()
        expected_name = issuer_names.get(expected_cik, "")
        active_snapshots = snapshots.loc[
            snapshots["target_id"].eq(target_id) & snapshots["query_active"].astype(bool)
        ]
        for snapshot in active_snapshots.itertuples(index=False):
            document = json.loads(Path(str(snapshot.path)).read_text(encoding="utf-8"))
            for occurrence, item in enumerate(document.get("results", [])):
                if not isinstance(item, dict):
                    continue
                raw_cik = item.get("cik")
                observed_cik = _normalized_cik(raw_cik)
                ticker_match = str(item.get("ticker", "")).strip().upper() == expected_ticker
                name_match = _normalized_name(item.get("name")) == _normalized_name(expected_name)
                common_stock = item.get("type") == "CS"
                cik_acceptable = observed_cik in {None, expected_cik}
                provider_id = _valid_figi(item.get("share_class_figi")) or _valid_figi(
                    item.get("composite_figi")
                )
                accepted = bool(
                    ticker_match and name_match and common_stock and cik_acceptable and provider_id
                )
                common = {
                    "target_id": target_id,
                    "identity_anchor_id": str(target.identity_anchor_id),
                    "validation_event_id": str(target.validation_event_id),
                    "cik": expected_cik,
                    "query_date": str(target.query_date),
                    "anchor_role": str(target.anchor_role),
                    "query_ticker": expected_ticker,
                    "source_occurrence": occurrence,
                    "source_sha256": str(snapshot.sha256),
                }
                if not accepted:
                    reason = (
                        "provider_cik_conflict"
                        if observed_cik not in {None, expected_cik}
                        else "ticker_name_type_or_figi_mismatch"
                    )
                    rejection_rows.append({**common, "reason": reason})
                    continue
                candidate_rows.append(
                    {
                        **common,
                        "provider_cik": observed_cik,
                        "provider_cik_status": (
                            "exact" if observed_cik == expected_cik else "missing_name_exact"
                        ),
                        "ticker": expected_ticker,
                        "name": item.get("name"),
                        "security_type": item.get("type"),
                        "exchange_mic": _normalized_exchange_mic(item.get("primary_exchange")),
                        "provider_security_id": provider_id,
                        "provider_security_id_kind": (
                            "share_class_figi"
                            if _valid_figi(item.get("share_class_figi"))
                            else "composite_figi"
                        ),
                        "evidence_source": "sec_name_ticker_plus_massive_active_record",
                        "selection_uses_return_values": False,
                    }
                )
    candidate_columns = [
        "target_id",
        "identity_anchor_id",
        "validation_event_id",
        "cik",
        "query_date",
        "anchor_role",
        "query_ticker",
        "source_occurrence",
        "source_sha256",
        "provider_cik",
        "provider_cik_status",
        "ticker",
        "name",
        "security_type",
        "exchange_mic",
        "provider_security_id",
        "provider_security_id_kind",
        "evidence_source",
        "selection_uses_return_values",
    ]
    rejection_columns = [
        "target_id",
        "identity_anchor_id",
        "validation_event_id",
        "cik",
        "query_date",
        "anchor_role",
        "query_ticker",
        "source_occurrence",
        "source_sha256",
        "reason",
    ]
    candidates = pd.DataFrame(candidate_rows, columns=candidate_columns)
    rejections = pd.DataFrame(rejection_rows, columns=rejection_columns)
    return candidates, rejections


def calculate_validation_identity_repair_components(
    targets: pd.DataFrame,
    cover_securities: pd.DataFrame,
    original_observations: pd.DataFrame,
    fallback_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve endpoint security components without treating NaN as an identifier."""
    rows: list[dict[str, Any]] = []
    equities = cover_securities.loc[
        cover_securities["common_equity_candidate"].astype(bool)
        & cover_securities["security_subject_cik_match"].astype(bool)
    ].copy()
    for target in targets.itertuples(index=False):
        endpoint_equities = equities.loc[
            equities["identity_anchor_id"].eq(str(target.identity_anchor_id))
        ]
        original = original_observations.loc[
            original_observations["target_id"].eq(str(target.identity_anchor_id))
            & original_observations["query_active"].astype(bool)
            & original_observations["security_type"].eq("CS")
        ]
        fallback = fallback_candidates.loc[
            fallback_candidates["identity_anchor_id"].eq(str(target.identity_anchor_id))
        ]
        for security in endpoint_equities.itertuples(index=False):
            ticker = str(security.trading_symbol)
            original_matches = original.loc[original["ticker"].eq(ticker)]
            fallback_matches = fallback.loc[fallback["ticker"].eq(ticker)]
            provider_rows: list[dict[str, Any]] = []
            for provider in original_matches.itertuples(index=False):
                provider_rows.append(
                    {
                        "exchange_mic": provider.exchange_mic,
                        "provider_security_id": provider.provider_security_id,
                        "provider_security_id_kind": provider.provider_security_id_kind,
                        "provider_cik_status": "exact",
                        "evidence_source": "massive_cik_date_query",
                    }
                )
            for provider in fallback_matches.itertuples(index=False):
                provider_rows.append(
                    {
                        "exchange_mic": provider.exchange_mic,
                        "provider_security_id": provider.provider_security_id,
                        "provider_security_id_kind": provider.provider_security_id_kind,
                        "provider_cik_status": provider.provider_cik_status,
                        "evidence_source": provider.evidence_source,
                    }
                )
            unique_provider_rows = {
                (
                    str(item.get("exchange_mic") or ""),
                    str(item.get("provider_security_id") or ""),
                    str(item.get("provider_security_id_kind") or ""),
                    str(item.get("provider_cik_status") or ""),
                    str(item.get("evidence_source") or ""),
                ): item
                for item in provider_rows
            }
            selected = (
                next(iter(unique_provider_rows.values()))
                if len(unique_provider_rows) == 1
                else None
            )
            exchange = (
                _clean_optional_text(selected.get("exchange_mic")) if selected is not None else None
            ) or _clean_optional_text(security.normalized_exchange_mic)
            provider_id = (
                _clean_optional_text(selected.get("provider_security_id"))
                if selected is not None
                else None
            )
            provider_id_kind = (
                _clean_optional_text(selected.get("provider_security_id_kind"))
                if selected is not None
                else None
            )
            if selected is not None and provider_id:
                identity_key = str(provider_id)
                identity_key_kind = provider_id_kind
            elif selected is not None and exchange:
                identity_key = _stable_hash(
                    {
                        "cik": str(target.cik).zfill(10),
                        "ticker": ticker,
                        "exchange": exchange,
                        "security_kind": "common_equity",
                        "basis": "dual_sec_cover_plus_exact_cik_provider_record",
                    }
                )
                identity_key_kind = "sec_cik_ticker_exchange_composite"
            else:
                identity_key = None
                identity_key_kind = None
            rows.append(
                {
                    "identity_anchor_id": str(target.identity_anchor_id),
                    "validation_event_id": str(target.validation_event_id),
                    "cik": str(target.cik).zfill(10),
                    "anchor_role": str(target.anchor_role),
                    "query_date": str(target.query_date),
                    "ticker": ticker,
                    "exchange_mic": exchange or None,
                    "security_title": security.security_title,
                    "provider_security_id": provider_id,
                    "provider_security_id_kind": (provider_id_kind),
                    "provider_cik_status": (
                        _clean_optional_text(selected.get("provider_cik_status"))
                        if selected
                        else None
                    ),
                    "identity_key": identity_key,
                    "identity_key_kind": identity_key_kind,
                    "component_identity_ready": identity_key is not None,
                    "provider_candidate_count": len(unique_provider_rows),
                    "evidence_source": (
                        _clean_optional_text(selected.get("evidence_source")) if selected else None
                    ),
                    "selection_uses_return_values": False,
                    "outcome_values_loaded": False,
                }
            )
    return pd.DataFrame(rows)


def _repair_endpoint_summary(
    targets: pd.DataFrame,
    cover_securities: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    equities = cover_securities.loc[
        cover_securities["common_equity_candidate"].astype(bool)
        & cover_securities["security_subject_cik_match"].astype(bool)
    ]
    for target in targets.itertuples(index=False):
        anchor_id = str(target.identity_anchor_id)
        expected = equities.loc[equities["identity_anchor_id"].eq(anchor_id)]
        observed = components.loc[components["identity_anchor_id"].eq(anchor_id)]
        ready = bool(
            len(expected) > 0
            and len(observed) == len(expected)
            and observed["component_identity_ready"].astype(bool).all()
        )
        rows.append(
            {
                "identity_anchor_id": anchor_id,
                "validation_event_id": str(target.validation_event_id),
                "cik": str(target.cik).zfill(10),
                "anchor_role": str(target.anchor_role),
                "query_date": str(target.query_date),
                "expected_common_equity_components": len(expected),
                "resolved_common_equity_components": int(
                    observed["component_identity_ready"].sum()
                ),
                "endpoint_repair_ready": ready,
                "endpoint_repair_status": (
                    "all_sec_equity_components_resolved"
                    if ready
                    else "provider_component_evidence_incomplete"
                ),
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    return pd.DataFrame(rows)


def _repair_issuer_summary(
    endpoints: pd.DataFrame,
    components: pd.DataFrame,
    issuer_names: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cik, endpoint_group in endpoints.groupby("cik", sort=True):
        component_group = components.loc[components["cik"].eq(cik)]
        component_maps: list[dict[str, str]] = []
        for _, anchor_components in component_group.groupby("identity_anchor_id", sort=True):
            component_maps.append(
                {
                    str(row.ticker): str(row.identity_key)
                    for row in anchor_components.itertuples(index=False)
                    if row.component_identity_ready
                }
            )
        consistent = bool(
            len(component_maps) == 2
            and component_maps[0]
            and component_maps[0] == component_maps[1]
            and endpoint_group["endpoint_repair_ready"].astype(bool).all()
        )
        component_count = len(component_maps[0]) if consistent else 0
        construction = (
            "single_security"
            if consistent and component_count == 1
            else "equal_weighted_share_class_basket"
            if consistent and component_count > 1
            else "unresolved"
        )
        rows.append(
            {
                "cik": str(cik),
                "issuer_name": issuer_names.get(str(cik)),
                "repair_ready": consistent,
                "resolved_security_component_count": component_count,
                "resolved_tickers": ("|".join(sorted(component_maps[0])) if consistent else ""),
                "resolved_identity_keys": (
                    "|".join(component_maps[0][ticker] for ticker in sorted(component_maps[0]))
                    if consistent
                    else ""
                ),
                "issuer_return_construction": construction,
                "repair_status": (
                    "stable_components_across_both_endpoints"
                    if consistent
                    else "endpoint_components_unresolved_or_changed"
                ),
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    return pd.DataFrame(rows)


def _apply_repairs_to_events(
    base_events: pd.DataFrame,
    issuers: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    result = base_events.copy()
    result["security_component_count"] = result["event_security_identity_ready"].astype(int)
    result["assigned_security_identity_keys"] = result["assigned_provider_security_id"].astype(
        "string"
    )
    result["issuer_return_construction"] = result["event_security_identity_ready"].map(
        {True: "single_security", False: "unresolved"}
    )
    result["identity_repair_status"] = "not_required"
    for issuer in issuers.itertuples(index=False):
        mask = result["cik"].astype(str).eq(str(issuer.cik))
        result.loc[mask, "identity_repair_status"] = str(issuer.repair_status)
        if not issuer.repair_ready:
            continue
        issuer_components = components.loc[
            components["cik"].eq(str(issuer.cik)) & components["anchor_role"].eq("first_candidate")
        ].sort_values("ticker")
        tickers = "|".join(issuer_components["ticker"].astype(str))
        exchanges = "|".join(
            sorted(issuer_components["exchange_mic"].dropna().astype(str).unique())
        )
        provider_ids = "|".join(issuer_components["provider_security_id"].dropna().astype(str))
        identity_keys = "|".join(issuer_components["identity_key"].astype(str))
        result.loc[mask, "event_security_identity_ready"] = True
        result.loc[mask, "identity_repair_required"] = False
        result.loc[mask, "identity_assignment_method"] = (
            "identity_repair_equal_weighted_share_class_basket"
            if issuer.issuer_return_construction == "equal_weighted_share_class_basket"
            else "identity_repair_single_security"
        )
        result.loc[mask, "assigned_ticker"] = tickers
        result.loc[mask, "assigned_exchange_mic"] = exchanges or pd.NA
        result.loc[mask, "assigned_provider_security_id"] = provider_ids or pd.NA
        resolved_component_count = int(str(issuer.resolved_security_component_count))
        result.loc[mask, "security_component_count"] = resolved_component_count
        result.loc[mask, "assigned_security_identity_keys"] = identity_keys
        result.loc[mask, "issuer_return_construction"] = str(issuer.issuer_return_construction)
    result["price_collection_authorized"] = False
    result["return_outcome_read_authorized"] = False
    result["selection_uses_return_values"] = False
    result["outcome_values_loaded"] = False
    return result


def _normalized_ticker(value: Any) -> str | None:
    if value is None:
        return None
    ticker = str(value).strip().upper()
    return ticker if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-/]{0,31}", ticker) else None


def _clean_optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalized_cik(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(10) if digits and len(digits) <= 10 else None


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _valid_figi(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if re.fullmatch(r"BBG[A-Z0-9]{9}", normalized) else None


def _security_kind(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    if re.search(r"\b(?:note|notes|bond|bonds|debenture|debentures)\b", title):
        return "debt"
    if re.search(r"\b(?:common stock|ordinary shares|common shares)\b", title):
        return "common_equity"
    return "other_or_unresolved"


def _normalized_exchange_mic(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip().casefold()
    mapping = {
        "nasdaq": "XNAS",
        "the nasdaq stock market llc": "XNAS",
        "nasdaq global select market": "XNAS",
        "new york stock exchange": "XNYS",
        "nyse": "XNYS",
        "nyse american": "XASE",
        "otc": "OTCM",
        "otc markets": "OTCM",
        "otc link": "OTCM",
    }
    return mapping.get(normalized)


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


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
