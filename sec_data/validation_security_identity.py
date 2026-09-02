"""Outcome-blind historical security identity for the validation cohort."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_SECURITY_IDENTITY_TARGET_VERSION: Final[str] = "validation-security-identity-targets-v1"
VALIDATION_SECURITY_IDENTITY_AUDIT_VERSION: Final[str] = (
    "validation-security-identity-audit-v2-event-readiness"
)


class ValidationSecurityIdentityError(FilingCatalogError):
    """Raised when the frozen security-identity contract is violated."""


class ValidationSecurityIdentityPlanResult:
    """Immutable Massive identity-anchor target plan."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


class ValidationSecurityIdentityAuditResult:
    """Immutable endpoint-consistency audit."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


def build_validation_security_identity_plan(
    market_plan_root: Path,
    *,
    output_root: Path = Path("data/market/validation_security_identity/targets"),
) -> ValidationSecurityIdentityPlanResult:
    """Persist Massive-compatible CIK/date queries for frozen identity anchors."""
    market_report = _read_json(market_plan_root / "validation_market_plan_report.json")
    if bool(market_report.get("outcome_values_loaded")):
        raise ValidationSecurityIdentityError("Market plan has opened outcome data")
    market_manifest_sha256 = _sha256_file(market_plan_root / "output_manifest.json")
    anchors = load_catalog_table(market_plan_root, "validation_identity_anchor_targets.parquet")
    _require_columns(
        anchors,
        {
            "identity_anchor_id",
            "cik",
            "anchor_role",
            "validation_event_id",
            "accession_number",
            "accepted_at",
            "query_date",
            "provider_query_method",
            "query_active_first",
            "query_inactive_only_after_no_active_match",
            "automatic_interval_propagation_allowed",
            "selection_uses_return_values",
        },
        "validation identity anchors",
    )
    if _has_outcome_columns(anchors):
        raise ValidationSecurityIdentityError("Identity anchors contain outcome columns")
    if anchors["selection_uses_return_values"].astype(bool).any():
        raise ValidationSecurityIdentityError("Identity anchors used return values")
    if anchors["automatic_interval_propagation_allowed"].astype(bool).any():
        raise ValidationSecurityIdentityError("Identity intervals were pre-authorized")
    targets = anchors.copy()
    targets["cik"] = targets["cik"].astype(str).str.zfill(10)
    targets["accepted_at"] = pd.to_datetime(targets["accepted_at"], utc=True, errors="coerce")
    if targets["accepted_at"].isna().any():
        raise ValidationSecurityIdentityError("Identity anchors contain invalid clocks")
    if not targets["query_date"].astype(str).eq(targets["accepted_at"].dt.date.astype(str)).all():
        raise ValidationSecurityIdentityError("Identity query dates do not match acceptance dates")
    if (
        targets["identity_anchor_id"].isna().any()
        or targets["identity_anchor_id"].duplicated().any()
    ):
        raise ValidationSecurityIdentityError("Identity anchor IDs must be unique")
    targets["target_id"] = targets["identity_anchor_id"].astype(str)
    targets["calendar_year"] = targets["accepted_at"].dt.year
    targets["equity_trading_symbols"] = ""
    targets["selection_uses_provider_coverage"] = False
    targets["outcome_values_loaded"] = False
    targets["price_collection_authorized"] = False
    targets = targets.sort_values(
        ["query_date", "cik", "anchor_role", "target_id"], kind="mergesort"
    ).reset_index(drop=True)
    targets["massive_acquisition_order"] = targets.index + 1
    columns = [
        "target_id",
        "massive_acquisition_order",
        "identity_anchor_id",
        "cik",
        "calendar_year",
        "anchor_role",
        "validation_event_id",
        "accession_number",
        "accepted_at",
        "query_date",
        "provider_query_method",
        "query_active_first",
        "query_inactive_only_after_no_active_match",
        "equity_trading_symbols",
        "automatic_interval_propagation_allowed",
        "selection_uses_provider_coverage",
        "selection_uses_return_values",
        "outcome_values_loaded",
        "price_collection_authorized",
    ]
    targets = targets[columns].copy()
    payload = json.dumps(
        {
            "market_plan_manifest_sha256": market_manifest_sha256,
            "target_version": VALIDATION_SECURITY_IDENTITY_TARGET_VERSION,
            "query_method": "active_true_then_active_false_only_if_empty",
            "order_rule": "query_date_then_cik_then_anchor_role",
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_SECURITY_IDENTITY_TARGET_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationSecurityIdentityPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationSecurityIdentityError(f"Incomplete identity plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    targets_path = staging / "massive_reference_targets.parquet"
    targets.to_parquet(targets_path, index=False)
    artifacts = {targets_path.name: _artifact_record(targets_path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_version": market_report["dataset_version"],
        "scope_manifest_sha256": market_manifest_sha256,
        "market_plan_version": market_report["dataset_version"],
        "market_plan_manifest_sha256": market_manifest_sha256,
        "price_mapping_targets": len(targets),
        "identity_anchor_targets": len(targets),
        "distinct_ciks": int(targets["cik"].nunique()),
        "single_anchor_issuers": int(targets.groupby("cik").size().eq(1).sum()),
        "endpoint_pair_issuers": int(targets.groupby("cik").size().eq(2).sum()),
        "query_method": "active_true_then_active_false_only_if_empty",
        "automatic_interval_propagation_allowed": False,
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
                "scope_manifest_sha256": market_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationSecurityIdentityPlanResult(target.resolve(), report)


def calculate_validation_identity_intervals(
    reference_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare frozen first/last endpoint identities without authorizing prices."""
    _require_columns(
        reference_audit,
        {
            "target_id",
            "cik",
            "anchor_role",
            "query_date",
            "provider_query_complete",
            "event_date_reference_status",
            "event_date_reference_candidate",
            "selected_ticker",
            "selected_exchange_mic",
            "selected_provider_security_id",
            "selected_provider_security_id_kind",
            "selection_uses_return_values",
        },
        "validation reference audit",
    )
    if _has_outcome_columns(reference_audit):
        raise ValidationSecurityIdentityError("Reference audit contains outcome columns")
    if reference_audit["selection_uses_return_values"].astype(bool).any():
        raise ValidationSecurityIdentityError("Reference audit used return values")
    rows: list[dict[str, Any]] = []
    for cik, group in reference_audit.groupby("cik", sort=True):
        ordered = group.sort_values(["query_date", "anchor_role"], kind="mergesort")
        candidates = ordered["event_date_reference_candidate"].fillna(False).astype(bool)
        identifiers = ordered.loc[candidates, "selected_provider_security_id"].dropna().astype(str)
        tickers = ordered.loc[candidates, "selected_ticker"].dropna().astype(str)
        exchanges = ordered.loc[candidates, "selected_exchange_mic"].dropna().astype(str)
        all_endpoints_collected = bool(ordered["provider_query_complete"].astype(bool).all())
        all_endpoints_candidates = bool(candidates.all())
        anchor_count = len(ordered)
        if not all_endpoints_collected:
            status = "endpoint_collection_incomplete"
        elif not all_endpoints_candidates:
            status = "endpoint_reference_unresolved"
        elif identifiers.nunique() != 1:
            status = "provider_security_id_changed_or_ambiguous"
        elif anchor_count == 1:
            status = "single_event_exact_identity"
        elif tickers.nunique() != 1 or exchanges.nunique() != 1:
            status = "stable_security_with_symbol_or_exchange_transition"
        else:
            status = "stable_security_and_symbol_at_endpoints"
        exact_event_identity_ready = all_endpoints_candidates
        stable_security_endpoint_evidence = bool(
            all_endpoints_candidates and identifiers.nunique() == 1
        )
        automatic_interval_propagation_ready = bool(
            anchor_count == 2
            and stable_security_endpoint_evidence
            and tickers.nunique() == 1
            and exchanges.nunique() == 1
        )
        rows.append(
            {
                "cik": str(cik).zfill(10),
                "identity_anchor_count": anchor_count,
                "first_query_date": str(ordered["query_date"].min()),
                "last_query_date": str(ordered["query_date"].max()),
                "provider_queries_complete": all_endpoints_collected,
                "exact_event_identity_ready": exact_event_identity_ready,
                "stable_security_endpoint_evidence": stable_security_endpoint_evidence,
                "endpoint_provider_security_id": (
                    identifiers.iloc[0] if stable_security_endpoint_evidence else None
                ),
                "endpoint_provider_security_id_kind": (
                    ordered.loc[candidates, "selected_provider_security_id_kind"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                    if stable_security_endpoint_evidence
                    else None
                ),
                "first_ticker": _first_non_null(ordered["selected_ticker"]),
                "last_ticker": _last_non_null(ordered["selected_ticker"]),
                "first_exchange_mic": _first_non_null(ordered["selected_exchange_mic"]),
                "last_exchange_mic": _last_non_null(ordered["selected_exchange_mic"]),
                "automatic_interval_propagation_ready": (automatic_interval_propagation_ready),
                "price_collection_authorized": False,
                "return_outcome_read_authorized": False,
                "identity_interval_status": status,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    intervals = pd.DataFrame(rows)
    gates = pd.DataFrame(
        [
            _gate(
                "identity_anchor_collection",
                int(reference_audit["provider_query_complete"].sum()),
                len(reference_audit),
                bool(reference_audit["provider_query_complete"].astype(bool).all()),
                "Every frozen endpoint has a completed CIK/date reference query",
            ),
            _gate(
                "exact_endpoint_identity",
                int(reference_audit["event_date_reference_candidate"].sum()),
                len(reference_audit),
                bool(reference_audit["event_date_reference_candidate"].astype(bool).all()),
                "Every endpoint resolves to one common-stock security with stable ID",
            ),
            _gate(
                "automatic_interval_propagation",
                int(intervals["automatic_interval_propagation_ready"].sum()),
                len(intervals),
                bool(intervals["automatic_interval_propagation_ready"].astype(bool).all()),
                "Paired endpoints retain the same security, ticker, and exchange",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(reference_audit),
                False,
                "Locked pending calendar, actions, prices, and delisting evidence",
            ),
        ]
    )
    report: dict[str, Any] = {
        "identity_anchor_targets": len(reference_audit),
        "identity_anchor_issuers": int(reference_audit["cik"].nunique()),
        "provider_queries_complete": int(reference_audit["provider_query_complete"].sum()),
        "exact_endpoint_identity_ready": int(
            reference_audit["event_date_reference_candidate"].sum()
        ),
        "issuers_with_exact_event_identity": int(intervals["exact_event_identity_ready"].sum()),
        "issuers_with_stable_security_endpoint_evidence": int(
            intervals["stable_security_endpoint_evidence"].sum()
        ),
        "issuers_ready_for_automatic_interval_propagation": int(
            intervals["automatic_interval_propagation_ready"].sum()
        ),
        "issuers_requiring_symbol_or_identity_repair": int(
            (~intervals["automatic_interval_propagation_ready"].astype(bool)).sum()
        ),
        "price_collection_authorized": False,
        "return_outcome_read_authorized": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }
    return intervals, gates, report


def calculate_validation_event_identity_readiness(
    market_events: pd.DataFrame,
    identity_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Assign only endpoint-supported identities to frozen price-candidate events."""
    _require_columns(
        market_events,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "price_collection_candidate",
            "selection_uses_return_values",
        },
        "validation market events",
    )
    _require_columns(
        identity_intervals,
        {
            "cik",
            "identity_anchor_count",
            "exact_event_identity_ready",
            "endpoint_provider_security_id",
            "endpoint_provider_security_id_kind",
            "first_ticker",
            "last_ticker",
            "first_exchange_mic",
            "last_exchange_mic",
            "automatic_interval_propagation_ready",
            "identity_interval_status",
        },
        "validation identity intervals",
    )
    if _has_outcome_columns(market_events):
        raise ValidationSecurityIdentityError("Market events contain outcome columns")
    candidates = market_events.loc[market_events["price_collection_candidate"].astype(bool)].copy()
    candidates["cik"] = candidates["cik"].astype(str).str.zfill(10)
    intervals = identity_intervals.copy()
    intervals["cik"] = intervals["cik"].astype(str).str.zfill(10)
    if intervals["cik"].duplicated().any():
        raise ValidationSecurityIdentityError("Identity intervals must be unique by CIK")
    candidates["issuer_price_candidate_events"] = candidates.groupby("cik")[
        "market_event_id"
    ].transform("size")
    columns = [
        "cik",
        "identity_anchor_count",
        "exact_event_identity_ready",
        "endpoint_provider_security_id",
        "endpoint_provider_security_id_kind",
        "first_ticker",
        "last_ticker",
        "first_exchange_mic",
        "last_exchange_mic",
        "automatic_interval_propagation_ready",
        "identity_interval_status",
    ]
    result = candidates.merge(intervals[columns], on="cik", how="left", validate="many_to_one")
    propagated = result["automatic_interval_propagation_ready"].fillna(False).astype(bool)
    exact_single = (
        result["identity_interval_status"].eq("single_event_exact_identity")
        & result["exact_event_identity_ready"].fillna(False).astype(bool)
        & result["issuer_price_candidate_events"].eq(1)
    )
    result["event_security_identity_ready"] = propagated | exact_single
    result["identity_assignment_method"] = "unresolved_identity_repair_required"
    result.loc[propagated, "identity_assignment_method"] = "endpoint_interval_propagation"
    result.loc[exact_single, "identity_assignment_method"] = "single_event_exact_anchor"
    result["assigned_ticker"] = result["first_ticker"].where(
        result["event_security_identity_ready"]
    )
    result["assigned_exchange_mic"] = result["first_exchange_mic"].where(
        result["event_security_identity_ready"]
    )
    result["assigned_provider_security_id"] = result["endpoint_provider_security_id"].where(
        result["event_security_identity_ready"]
    )
    result["assigned_provider_security_id_kind"] = result[
        "endpoint_provider_security_id_kind"
    ].where(result["event_security_identity_ready"])
    result["identity_repair_required"] = ~result["event_security_identity_ready"]
    result["price_collection_authorized"] = False
    result["return_outcome_read_authorized"] = False
    result["selection_uses_return_values"] = False
    result["outcome_values_loaded"] = False
    return result[
        [
            "market_event_id",
            "validation_event_id",
            "cik",
            "issuer_price_candidate_events",
            "identity_anchor_count",
            "identity_interval_status",
            "event_security_identity_ready",
            "identity_assignment_method",
            "assigned_ticker",
            "assigned_exchange_mic",
            "assigned_provider_security_id",
            "assigned_provider_security_id_kind",
            "identity_repair_required",
            "price_collection_authorized",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
        ]
    ].copy()


def build_validation_security_identity_audit(
    market_plan_root: Path,
    reference_audit_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_security_identity"),
) -> ValidationSecurityIdentityAuditResult:
    """Persist endpoint consistency tied to market and reference manifests."""
    market_manifest_sha256 = _sha256_file(market_plan_root / "output_manifest.json")
    reference_manifest_sha256 = _sha256_file(reference_audit_root / "output_manifest.json")
    reference_report = _read_json(reference_audit_root / "massive_reference_report.json")
    reference_audit = load_catalog_table(
        reference_audit_root, "sample_massive_reference_audit.parquet"
    )
    intervals, gates, metrics = calculate_validation_identity_intervals(reference_audit)
    market_events = load_catalog_table(market_plan_root, "validation_market_event_targets.parquet")
    event_readiness = calculate_validation_event_identity_readiness(market_events, intervals)
    ready_events = int(event_readiness["event_security_identity_ready"].sum())
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    _gate(
                        "price_candidate_event_identity",
                        ready_events,
                        len(event_readiness),
                        ready_events == len(event_readiness),
                        "Each price-candidate event has exact or endpoint-propagated identity",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    payload = json.dumps(
        {
            "market_plan_manifest_sha256": market_manifest_sha256,
            "reference_audit_manifest_sha256": reference_manifest_sha256,
            "audit_version": VALIDATION_SECURITY_IDENTITY_AUDIT_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_SECURITY_IDENTITY_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationSecurityIdentityAuditResult(
            target.resolve(), _read_json(target / "validation_security_identity_report.json")
        )
    if target.exists():
        raise ValidationSecurityIdentityError(f"Incomplete identity audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_issuer_identity_intervals.parquet": intervals,
        "validation_event_identity_readiness.parquet": event_readiness,
        "validation_security_identity_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        **metrics,
        "price_candidate_events": len(event_readiness),
        "price_candidate_events_identity_ready": ready_events,
        "price_candidate_event_identity_coverage": (
            ready_events / len(event_readiness) if len(event_readiness) else 0.0
        ),
        "price_candidate_events_requiring_identity_repair": int(
            event_readiness["identity_repair_required"].sum()
        ),
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_plan_manifest_sha256": market_manifest_sha256,
        "reference_audit_version": reference_report["dataset_version"],
        "reference_audit_manifest_sha256": reference_manifest_sha256,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_security_identity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "market_plan_manifest_sha256": market_manifest_sha256,
                "reference_audit_manifest_sha256": reference_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationSecurityIdentityAuditResult(target.resolve(), report)


def _first_non_null(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    return values.iloc[0] if not values.empty else None


def _last_non_null(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    return values.iloc[-1] if not values.empty else None


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
        raise ValidationSecurityIdentityError(f"{label} lacks required columns: {missing}")


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
