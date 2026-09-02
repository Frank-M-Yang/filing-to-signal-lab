"""Exact event-date repair for revenue-guidance security-identity transitions."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

REVENUE_GUIDANCE_TRANSITION_TARGET_VERSION: Final[str] = (
    "revenue-guidance-identity-transition-targets-v1"
)
REVENUE_GUIDANCE_IDENTITY_RESOLUTION_VERSION: Final[str] = "revenue-guidance-identity-resolution-v1"
FORBIDDEN_RETURN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "future_return",
        "delisting_return",
    }
)


class RevenueGuidanceIdentityTransitionError(FilingCatalogError):
    """Raised when exact event-date identity repair is not outcome blind."""


@dataclass(frozen=True)
class RevenueGuidanceIdentityTransitionPlanResult:
    """Immutable exact-event transition target plan."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueGuidanceIdentityResolutionResult:
    """Immutable final identity resolution for guidance market events."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_transition_targets(
    market_events: pd.DataFrame,
    base_identity_events: pd.DataFrame,
    base_reference_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Target only transition events not already queried at their exact event date."""
    _validate_no_returns(market_events, "market events")
    _validate_no_returns(base_identity_events, "base identity events")
    _validate_no_returns(base_reference_audit, "base reference audit")
    _require_columns(
        market_events,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "accepted_at",
            "form",
            "target_accession_number",
            "price_collection_candidate",
            "selection_uses_return_values",
        },
        "market events",
    )
    _require_columns(
        base_identity_events,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "event_security_identity_ready",
            "identity_interval_status",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "base identity events",
    )
    _require_columns(
        base_reference_audit,
        {
            "validation_event_id",
            "event_date_reference_candidate",
            "selection_uses_return_values",
        },
        "base reference audit",
    )
    for frame, label in (
        (market_events, "market"),
        (base_identity_events, "identity"),
    ):
        if frame["selection_uses_return_values"].astype(bool).any():
            raise RevenueGuidanceIdentityTransitionError(f"{label} selection used returns")
    if base_identity_events["outcome_values_loaded"].astype(bool).any():
        raise RevenueGuidanceIdentityTransitionError("Base identity opened outcomes")
    candidates = market_events.loc[market_events["price_collection_candidate"].astype(bool)].copy()
    base = base_identity_events.copy()
    if len(candidates) != len(base):
        raise RevenueGuidanceIdentityTransitionError("Market and base identity event sets differ")
    unresolved = base.loc[~base["event_security_identity_ready"].astype(bool)].copy()
    if unresolved.empty:
        raise RevenueGuidanceIdentityTransitionError("No identity transition requires repair")
    if (
        not unresolved["identity_interval_status"]
        .eq("stable_security_with_symbol_or_exchange_transition")
        .all()
    ):
        raise RevenueGuidanceIdentityTransitionError("An unresolved event is not a transition")
    exact_base_ids = set(
        base_reference_audit.loc[
            base_reference_audit["event_date_reference_candidate"].astype(bool),
            "validation_event_id",
        ].astype(str)
    )
    unresolved["exact_endpoint_reference_already_available"] = (
        unresolved["validation_event_id"].astype(str).isin(exact_base_ids)
    )
    pending_ids = set(
        unresolved.loc[
            ~unresolved["exact_endpoint_reference_already_available"], "validation_event_id"
        ].astype(str)
    )
    pending = candidates.loc[candidates["validation_event_id"].astype(str).isin(pending_ids)].copy()
    if pending.empty:
        raise RevenueGuidanceIdentityTransitionError("No missing exact-event query remains")
    pending["accepted_at"] = _utc(pending["accepted_at"], "transition accepted_at")
    pending["target_id"] = pending.apply(
        lambda row: _stable_hash(
            {
                "validation_event_id": str(row["validation_event_id"]),
                "query_date": str(row["accepted_at"].date()),
                "version": REVENUE_GUIDANCE_TRANSITION_TARGET_VERSION,
            }
        ),
        axis="columns",
    )
    pending["calendar_year"] = pending["accepted_at"].dt.year
    pending["query_date"] = pending["accepted_at"].dt.date.astype(str)
    pending["accession_number"] = pending["target_accession_number"].astype(str)
    pending["provider_query_method"] = "cik_and_exact_event_date"
    pending["query_active_first"] = True
    pending["query_inactive_only_after_no_active_match"] = True
    pending["equity_trading_symbols"] = ""
    pending["selection_uses_provider_coverage"] = False
    pending["selection_uses_return_values"] = False
    pending["outcome_values_loaded"] = False
    pending["price_collection_authorized"] = False
    pending = pending.sort_values(
        ["query_date", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    pending["massive_acquisition_order"] = pending.index + 1
    targets = pending[
        [
            "target_id",
            "massive_acquisition_order",
            "market_event_id",
            "validation_event_id",
            "cik",
            "calendar_year",
            "accepted_at",
            "query_date",
            "form",
            "accession_number",
            "provider_query_method",
            "query_active_first",
            "query_inactive_only_after_no_active_match",
            "equity_trading_symbols",
            "selection_uses_provider_coverage",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "price_collection_authorized",
        ]
    ].copy()
    audit = unresolved[
        [
            "market_event_id",
            "validation_event_id",
            "cik",
            "identity_interval_status",
            "exact_endpoint_reference_already_available",
        ]
    ].copy()
    audit["new_exact_event_query_required"] = ~audit["exact_endpoint_reference_already_available"]
    audit["selection_uses_return_values"] = False
    report: dict[str, Any] = {
        "transition_events": len(unresolved),
        "transition_issuers": int(unresolved["cik"].nunique()),
        "existing_exact_endpoint_references": int(
            unresolved["exact_endpoint_reference_already_available"].sum()
        ),
        "new_exact_event_queries": len(targets),
        "selection_uses_provider_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "return_outcome_read_authorized": False,
    }
    return targets, audit, report


def build_revenue_guidance_transition_plan(
    market_plan_root: Path,
    base_identity_root: Path,
    base_reference_root: Path,
    *,
    output_root: Path = Path("data/market/revenue_guidance_identity_transition/targets"),
) -> RevenueGuidanceIdentityTransitionPlanResult:
    """Persist exact-event Massive queries for unresolved exchange transitions."""
    input_hashes = {
        "market_plan": _sha256_file(market_plan_root / "output_manifest.json"),
        "base_identity": _sha256_file(base_identity_root / "output_manifest.json"),
        "base_reference": _sha256_file(base_reference_root / "output_manifest.json"),
    }
    targets, audit, metrics = calculate_revenue_guidance_transition_targets(
        load_catalog_table(market_plan_root, "validation_market_event_targets.parquet"),
        load_catalog_table(base_identity_root, "validation_event_identity_readiness.parquet"),
        load_catalog_table(base_reference_root, "sample_massive_reference_audit.parquet"),
    )
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "version": REVENUE_GUIDANCE_TRANSITION_TARGET_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        REVENUE_GUIDANCE_TRANSITION_TARGET_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceIdentityTransitionPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise RevenueGuidanceIdentityTransitionError(f"Incomplete transition plan: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_reference_targets.parquet": targets,
        "transition_event_query_audit.parquet": audit,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    market_report = _read_json(market_plan_root / "validation_market_plan_report.json")
    report = {
        **metrics,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_version": market_report["dataset_version"],
        "scope_manifest_sha256": input_hashes["market_plan"],
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return RevenueGuidanceIdentityTransitionPlanResult(target.resolve(), report)


def calculate_revenue_guidance_identity_resolution(
    base_identity_events: pd.DataFrame,
    base_reference_audit: pd.DataFrame,
    transition_reference_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve every transition event with an exact-date provider observation."""
    for frame, label in (
        (base_identity_events, "base identity"),
        (base_reference_audit, "base reference"),
        (transition_reference_audit, "transition reference"),
    ):
        _validate_no_returns(frame, label)
    result = base_identity_events.copy()
    if (
        result["selection_uses_return_values"].astype(bool).any()
        or result["outcome_values_loaded"].astype(bool).any()
    ):
        raise RevenueGuidanceIdentityTransitionError("Base identity is not outcome blind")
    reference_columns = [
        "validation_event_id",
        "event_date_reference_candidate",
        "selected_ticker",
        "selected_exchange_mic",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
    ]
    references = pd.concat(
        [base_reference_audit[reference_columns], transition_reference_audit[reference_columns]],
        ignore_index=True,
    )
    references = references.loc[references["event_date_reference_candidate"].astype(bool)].copy()
    for event_id, group in references.groupby("validation_event_id", sort=False):
        identity_values = group[
            [
                "selected_ticker",
                "selected_exchange_mic",
                "selected_provider_security_id",
                "selected_provider_security_id_kind",
            ]
        ].astype(str)
        if len(identity_values.drop_duplicates()) != 1:
            raise RevenueGuidanceIdentityTransitionError(
                f"Conflicting exact identity observations for {event_id}"
            )
    references = references.drop_duplicates("validation_event_id", keep="first")
    exact = references.set_index("validation_event_id")
    unresolved = ~result["event_security_identity_ready"].astype(bool)
    missing = set(result.loc[unresolved, "validation_event_id"].astype(str)).difference(exact.index)
    if missing:
        raise RevenueGuidanceIdentityTransitionError(
            f"Transition events still lack exact references: {sorted(missing)}"
        )
    for index in result.index[unresolved]:
        event_id = str(result.at[index, "validation_event_id"])
        evidence = cast(pd.Series, exact.loc[event_id])
        result.at[index, "event_security_identity_ready"] = True
        result.at[index, "identity_assignment_method"] = (
            "event_date_reference_transition_resolution"
        )
        result.at[index, "assigned_ticker"] = evidence["selected_ticker"]
        result.at[index, "assigned_exchange_mic"] = evidence["selected_exchange_mic"]
        result.at[index, "assigned_provider_security_id"] = evidence[
            "selected_provider_security_id"
        ]
        result.at[index, "assigned_provider_security_id_kind"] = evidence[
            "selected_provider_security_id_kind"
        ]
        result.at[index, "identity_repair_required"] = False
    result["security_component_count"] = 1
    result["assigned_security_identity_keys"] = result["assigned_provider_security_id"]
    result["issuer_return_construction"] = "single_security"
    result["identity_repair_status"] = "not_required"
    result.loc[unresolved, "identity_repair_status"] = "event_date_transition_resolved"
    result["price_collection_authorized"] = False
    result["return_outcome_read_authorized"] = False
    result["selection_uses_return_values"] = False
    result["outcome_values_loaded"] = False
    if not result["event_security_identity_ready"].astype(bool).all():
        raise RevenueGuidanceIdentityTransitionError("Final identity coverage is incomplete")
    if result["assigned_provider_security_id"].isna().any():
        raise RevenueGuidanceIdentityTransitionError("A final identity lacks a provider key")
    component_columns = [
        "identity_anchor_id",
        "validation_event_id",
        "cik",
        "anchor_role",
        "query_date",
        "ticker",
        "exchange_mic",
        "security_title",
        "provider_security_id",
        "provider_security_id_kind",
        "provider_cik_status",
        "identity_key",
        "identity_key_kind",
        "component_identity_ready",
        "provider_candidate_count",
        "evidence_source",
        "selection_uses_return_values",
        "outcome_values_loaded",
    ]
    components = pd.DataFrame(columns=component_columns)
    report: dict[str, Any] = {
        "price_candidate_events": len(result),
        "price_candidate_events_identity_ready": int(result["event_security_identity_ready"].sum()),
        "price_candidate_event_identity_coverage": float(
            result["event_security_identity_ready"].mean()
        ),
        "transition_events_resolved": int(unresolved.sum()),
        "security_components": len(result),
        "price_collection_authorized": True,
        "return_outcome_read_authorized": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }
    return result, components, report


def build_revenue_guidance_identity_resolution(
    market_plan_root: Path,
    base_identity_root: Path,
    base_reference_root: Path,
    transition_reference_root: Path,
    *,
    output_root: Path = Path("data/processed/revenue_guidance_identity_resolution"),
) -> RevenueGuidanceIdentityResolutionResult:
    """Persist final exact-event identity assignments for execution planning."""
    input_hashes = {
        "market_plan": _sha256_file(market_plan_root / "output_manifest.json"),
        "base_identity": _sha256_file(base_identity_root / "output_manifest.json"),
        "base_reference": _sha256_file(base_reference_root / "output_manifest.json"),
        "transition_reference": _sha256_file(transition_reference_root / "output_manifest.json"),
    }
    result, components, metrics = calculate_revenue_guidance_identity_resolution(
        load_catalog_table(base_identity_root, "validation_event_identity_readiness.parquet"),
        load_catalog_table(base_reference_root, "sample_massive_reference_audit.parquet"),
        load_catalog_table(transition_reference_root, "sample_massive_reference_audit.parquet"),
    )
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "version": REVENUE_GUIDANCE_IDENTITY_RESOLUTION_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        REVENUE_GUIDANCE_IDENTITY_RESOLUTION_VERSION
        + "-"
        + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceIdentityResolutionResult(
            target.resolve(), _read_json(target / "identity_repair_report.json")
        )
    if target.exists():
        raise RevenueGuidanceIdentityTransitionError(f"Incomplete identity resolution: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_event_identity_readiness_final.parquet": result,
        "validation_identity_repair_components.parquet": components,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report = {
        **metrics,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "identity_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return RevenueGuidanceIdentityResolutionResult(target.resolve(), report)


def _validate_no_returns(frame: pd.DataFrame, label: str) -> None:
    forbidden = sorted(FORBIDDEN_RETURN_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise RevenueGuidanceIdentityTransitionError(
            f"{label} contains return columns: {forbidden}"
        )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RevenueGuidanceIdentityTransitionError(f"{label} lacks columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise RevenueGuidanceIdentityTransitionError(f"{label} contains invalid timestamps")
    return result


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
