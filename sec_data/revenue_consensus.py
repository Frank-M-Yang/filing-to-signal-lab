"""Point-in-time analyst-consensus target planning and Massive evidence audit."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

import pandas as pd

from sec_data.client import normalize_cik
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_reference import (
    MassiveCollectionResult,
    MassiveReferenceClient,
    MassiveRequestError,
)

MASSIVE_EARNINGS_URL: Final[str] = "https://api.massive.com/benzinga/v1/earnings"
REVENUE_CONSENSUS_PLAN_VERSION: Final[str] = "revenue-consensus-plan-v1-frozen-pit-clock"
REVENUE_CONSENSUS_PARSER_VERSION: Final[str] = (
    "massive-benzinga-revenue-consensus-v2-strict-record-version-failure-aware"
)


class RevenueConsensusError(FilingCatalogError):
    """Raised when consensus evidence violates the frozen point-in-time contract."""


@dataclass(frozen=True)
class RevenueConsensusConfig:
    """Frozen matching and minimum-coverage requirements for consensus evidence."""

    maximum_event_date_distance_days: int = 3
    minimum_ready_events: int = 20
    minimum_ready_issuers: int = 10
    version: str = REVENUE_CONSENSUS_PLAN_VERSION

    def validate(self) -> None:
        """Reject weak matching or coverage assumptions."""
        if not 0 <= self.maximum_event_date_distance_days <= 7:
            raise ValueError("Consensus event-date tolerance must be between zero and seven days")
        if self.minimum_ready_events < 10 or self.minimum_ready_issuers < 3:
            raise ValueError("Consensus coverage gates are too small")


@dataclass(frozen=True)
class RevenueConsensusPlanTables:
    """Frozen provider queries and their identity-source audit."""

    targets: pd.DataFrame
    identity_audit: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueConsensusAuditTables:
    """Parsed consensus observations, rejections, coverage, and report."""

    snapshots: pd.DataFrame
    observations: pd.DataFrame
    rejections: pd.DataFrame
    failures: pd.DataFrame
    event_audit: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueConsensusResult:
    """Immutable target-plan or evidence-audit result."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_consensus_plan(
    forecasts: pd.DataFrame,
    market_targets: pd.DataFrame,
    event_identities: pd.DataFrame,
    config: RevenueConsensusConfig | None = None,
) -> RevenueConsensusPlanTables:
    """Freeze one historical consensus query per selected revenue-guidance event."""
    cfg = config or RevenueConsensusConfig()
    cfg.validate()
    _require_columns(
        forecasts,
        {
            "validation_event_id",
            "cik",
            "target_fiscal_year",
            "target_fiscal_quarter",
            "target_actual_available_at",
            "target_revenue",
            "management_guidance_forecast",
            "selection_uses_return_values",
        },
        "revenue-guidance forecasts",
    )
    _require_columns(
        market_targets,
        {"validation_event_id", "cik", "report_date", "accepted_at"},
        "market event targets",
    )
    _require_columns(
        event_identities,
        {
            "validation_event_id",
            "cik",
            "ticker",
            "exchange_mic",
            "identity_source",
        },
        "event identities",
    )
    for frame, label in (
        (forecasts, "forecast"),
        (market_targets, "market target"),
        (event_identities, "event identity"),
    ):
        if (
            frame["validation_event_id"].isna().any()
            or frame["validation_event_id"].duplicated().any()
        ):
            raise RevenueConsensusError(f"{label} event IDs must be non-null and unique")
    if forecasts["selection_uses_return_values"].astype(bool).any():
        raise RevenueConsensusError(
            "Consensus targets cannot originate from return-selected events"
        )

    target_context = market_targets[
        ["validation_event_id", "cik", "report_date", "accepted_at"]
    ].rename(columns={"cik": "market_target_cik"})
    identity = event_identities.rename(columns={"cik": "identity_cik"})
    targets = forecasts.merge(
        target_context, on="validation_event_id", how="left", validate="one_to_one"
    ).merge(identity, on="validation_event_id", how="left", validate="one_to_one")
    targets["cik"] = targets["cik"].map(normalize_cik)
    targets["market_target_cik"] = targets["market_target_cik"].map(normalize_cik)
    targets["identity_cik"] = targets["identity_cik"].map(normalize_cik)
    if not (
        targets["cik"].eq(targets["market_target_cik"]) & targets["cik"].eq(targets["identity_cik"])
    ).all():
        raise RevenueConsensusError("Consensus target CIKs disagree across frozen inputs")
    targets["target_actual_available_at"] = _utc(
        targets["target_actual_available_at"], "target actual availability"
    )
    targets["accepted_at"] = _utc(targets["accepted_at"], "market target acceptance")
    if not targets["target_actual_available_at"].eq(targets["accepted_at"]).all():
        raise RevenueConsensusError("Consensus cutoff differs from the target filing acceptance")
    targets["ticker"] = targets["ticker"].astype(str).str.strip().str.upper()
    targets["target_fiscal_quarter"] = (
        targets["target_fiscal_quarter"].astype(str).str.strip().str.upper()
    )
    if not targets["ticker"].str.fullmatch(r"[A-Z0-9][A-Z0-9.\-/]{0,31}").all():
        raise RevenueConsensusError("Consensus target contains an invalid historical ticker")
    if not targets["target_fiscal_quarter"].isin({"Q1", "Q2", "Q3", "Q4"}).all():
        raise RevenueConsensusError("Consensus target contains a non-quarterly fiscal period")
    targets["target_fiscal_year"] = pd.to_numeric(
        targets["target_fiscal_year"], errors="raise"
    ).astype(int)
    targets["consensus_cutoff_at"] = targets["target_actual_available_at"]
    targets["provider"] = "massive_benzinga_earnings"
    targets["query_url"] = targets.apply(_massive_earnings_query_url, axis="columns")
    targets["query_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "provider": row["provider"],
                "ticker": row["ticker"],
                "fiscal_year": int(row["target_fiscal_year"]),
                "fiscal_period": row["target_fiscal_quarter"],
                "query_url": row["query_url"],
            }
        ),
        axis="columns",
    )
    targets["consensus_target_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "validation_event_id": row["validation_event_id"],
                "query_id": row["query_id"],
                "consensus_cutoff_at": row["consensus_cutoff_at"].isoformat(),
            }
        ),
        axis="columns",
    )
    targets["record_version_must_precede_cutoff"] = True
    targets["post_event_vendor_snapshot_allowed"] = False
    targets["retrospective_record_allowed_for_formal_model"] = False
    targets["selection_uses_provider_coverage"] = False
    targets["selection_uses_return_values"] = False
    targets["return_values_loaded"] = False
    targets = targets.sort_values(
        ["consensus_cutoff_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    targets["collection_order"] = range(1, len(targets) + 1)
    output_columns = [
        "consensus_target_id",
        "collection_order",
        "query_id",
        "query_url",
        "provider",
        "validation_event_id",
        "cik",
        "ticker",
        "exchange_mic",
        "identity_source",
        "report_date",
        "target_fiscal_year",
        "target_fiscal_quarter",
        "target_actual_available_at",
        "consensus_cutoff_at",
        "target_revenue",
        "management_guidance_forecast",
        "record_version_must_precede_cutoff",
        "post_event_vendor_snapshot_allowed",
        "retrospective_record_allowed_for_formal_model",
        "selection_uses_provider_coverage",
        "selection_uses_return_values",
        "return_values_loaded",
    ]
    frozen = targets[output_columns].copy()
    identity_audit = (
        frozen.groupby("identity_source", dropna=False)
        .agg(events=("validation_event_id", "size"), issuers=("cik", "nunique"))
        .reset_index()
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "consensus_targets": len(frozen),
        "consensus_target_issuers": int(frozen["cik"].nunique()),
        "identity_complete_events": int(frozen["ticker"].notna().sum()),
        "provider": "massive_benzinga_earnings",
        "provider_estimate_field": "estimated_revenue",
        "provider_record_version_field": "last_updated",
        "strict_point_in_time_requirement": "last_updated < consensus_cutoff_at",
        "analyst_count_required": False,
        "dispersion_available_from_endpoint": False,
        "selection_uses_return_values": False,
        "return_values_loaded": False,
        "config": asdict(cfg),
    }
    return RevenueConsensusPlanTables(frozen, identity_audit, report)


def build_revenue_consensus_plan(
    tournament_root: Path,
    market_plan_root: Path,
    primary_return_root: Path,
    supplemental_identity_root: Path,
    *,
    output_root: Path = Path("data/market/revenue_consensus/targets"),
    config: RevenueConsensusConfig | None = None,
) -> RevenueConsensusResult:
    """Persist the frozen, outcome-independent consensus collection plan."""
    cfg = config or RevenueConsensusConfig()
    cfg.validate()
    input_hashes = {
        "tournament": _sha256_file(tournament_root / "output_manifest.json"),
        "market_plan": _sha256_file(market_plan_root / "output_manifest.json"),
        "primary_returns": _sha256_file(primary_return_root / "output_manifest.json"),
        "supplemental_identity": _sha256_file(supplemental_identity_root / "output_manifest.json"),
    }
    identities = assemble_revenue_consensus_identities(
        load_catalog_table(primary_return_root, "validation_event_return_ledger.parquet"),
        load_catalog_table(
            supplemental_identity_root, "validation_event_identity_readiness_final.parquet"
        ),
    )
    tables = calculate_revenue_consensus_plan(
        load_catalog_table(tournament_root, "revenue_guidance_forecasts.parquet"),
        load_catalog_table(market_plan_root, "validation_market_event_targets.parquet"),
        identities,
        cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueConsensusResult(
            target.resolve(), _read_json(target / "revenue_consensus_plan_report.json")
        )
    if target.exists():
        raise RevenueConsensusError(f"Incomplete revenue-consensus plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "revenue_consensus_targets.parquet": tables.targets,
        "revenue_consensus_identity_audit.parquet": tables.identity_audit,
    }
    artifacts = _write_frames(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "revenue_consensus_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_output_manifest(staging, dataset_version, input_hashes, artifacts, report_path)
    staging.replace(target)
    return RevenueConsensusResult(target.resolve(), report)


def assemble_revenue_consensus_identities(
    primary_return_ledger: pd.DataFrame,
    supplemental_identity: pd.DataFrame,
) -> pd.DataFrame:
    """Union disjoint historical event identities without reading any return value."""
    _require_columns(
        primary_return_ledger,
        {"validation_event_id", "cik", "ticker", "exchange_mic"},
        "primary return identity ledger",
    )
    _require_columns(
        supplemental_identity,
        {
            "validation_event_id",
            "cik",
            "assigned_ticker",
            "assigned_exchange_mic",
            "event_security_identity_ready",
        },
        "supplemental identity audit",
    )
    primary = primary_return_ledger[["validation_event_id", "cik", "ticker", "exchange_mic"]].copy()
    primary["identity_source"] = "prior_authorized_market_ledger"
    supplemental = supplemental_identity.loc[
        supplemental_identity["event_security_identity_ready"].astype(bool),
        [
            "validation_event_id",
            "cik",
            "assigned_ticker",
            "assigned_exchange_mic",
        ],
    ].rename(
        columns={
            "assigned_ticker": "ticker",
            "assigned_exchange_mic": "exchange_mic",
        }
    )
    supplemental["identity_source"] = "outcome_blind_identity_resolution"
    identities = pd.concat([primary, supplemental], ignore_index=True)
    identities["cik"] = identities["cik"].map(normalize_cik)
    identities["ticker"] = identities["ticker"].astype(str).str.strip().str.upper()
    duplicate = identities["validation_event_id"].duplicated(keep=False)
    if duplicate.any():
        conflict = (
            identities.loc[duplicate]
            .groupby("validation_event_id")[["cik", "ticker", "exchange_mic"]]
            .nunique(dropna=False)
            .gt(1)
            .any(axis="columns")
        )
        if conflict.any():
            raise RevenueConsensusError("Consensus event identities conflict across sources")
        identities = identities.drop_duplicates("validation_event_id", keep="first")
    return identities.reset_index(drop=True)


def collect_revenue_consensus_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path,
    limit: int = 5,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveCollectionResult:
    """Collect exact Massive earnings bytes for pending frozen consensus targets."""
    _require_columns(
        targets,
        {"consensus_target_id", "collection_order", "query_id", "query_url"},
        "revenue-consensus targets",
    )
    if limit < 1:
        raise ValueError("Consensus collection limit must be positive")
    pending = [
        row
        for row in targets.sort_values("collection_order").itertuples(index=False)
        if not _valid_completion(raw_root, str(row.consensus_target_id))
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        target_id = str(row.consensus_target_id)
        try:
            snapshots, made, reused_for_target, records = _collect_consensus_query(
                client,
                raw_root=raw_root,
                target_id=target_id,
                query_id=str(row.query_id),
                query_url=str(row.query_url),
            )
            requests_made += made
            reused += reused_for_target
            completion = {
                "consensus_target_id": target_id,
                "query_id": str(row.query_id),
                "query_url": str(row.query_url),
                "record_count": records,
                "snapshots": snapshots,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_immutable(_completion_path(raw_root, target_id), completion)
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, target_id, exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), target_id)
    return MassiveCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def calculate_revenue_consensus_audit(
    targets: pd.DataFrame,
    raw_root: Path,
    config: RevenueConsensusConfig | None = None,
) -> RevenueConsensusAuditTables:
    """Parse preserved provider bytes and enforce the record-version availability clock."""
    cfg = config or RevenueConsensusConfig()
    cfg.validate()
    _require_columns(
        targets,
        {
            "consensus_target_id",
            "validation_event_id",
            "cik",
            "ticker",
            "target_fiscal_year",
            "target_fiscal_quarter",
            "target_actual_available_at",
            "consensus_cutoff_at",
        },
        "revenue-consensus targets",
    )
    snapshot_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    completion_ids: set[str] = set()
    failure_by_target: dict[str, dict[str, Any]] = {}
    for target in targets.itertuples(index=False):
        target_id = str(target.consensus_target_id)
        failure = _load_failure(raw_root, target_id)
        if failure is not None:
            failure_by_target[target_id] = failure
            failure_rows.append(
                {
                    "consensus_target_id": target_id,
                    "reason": str(failure.get("reason", "unknown_failure")),
                    "failed_at": failure.get("failed_at"),
                    "failure_manifest_sha256": _sha256_file(_failure_path(raw_root, target_id)),
                }
            )
        completion = _load_completion(raw_root, target_id)
        if completion is None:
            continue
        completion_ids.add(target_id)
        for manifest_value in completion["snapshots"]:
            manifest_path = Path(str(manifest_value))
            manifest = _load_snapshot_manifest(manifest_path)
            snapshot_rows.append(manifest)
            content = Path(str(manifest["path"])).read_bytes()
            observations, rejections = parse_massive_revenue_consensus_response(
                content,
                target={
                    "consensus_target_id": target_id,
                    "validation_event_id": str(target.validation_event_id),
                    "cik": str(target.cik),
                    "ticker": str(target.ticker),
                    "target_fiscal_year": int(str(target.target_fiscal_year)),
                    "target_fiscal_quarter": str(target.target_fiscal_quarter),
                    "target_actual_available_at": target.target_actual_available_at,
                    "consensus_cutoff_at": target.consensus_cutoff_at,
                },
                source_sha256=str(manifest["sha256"]),
                maximum_event_date_distance_days=cfg.maximum_event_date_distance_days,
            )
            observation_rows.extend(observations)
            rejection_rows.extend(rejections)
    observation_frame = pd.DataFrame(observation_rows, columns=_observation_columns())
    rejection_frame = pd.DataFrame(rejection_rows, columns=_rejection_columns())
    failure_frame = pd.DataFrame(failure_rows, columns=_failure_columns())
    snapshot_frame = pd.DataFrame(snapshot_rows, columns=_snapshot_columns())
    event_rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_id = str(target.consensus_target_id)
        candidates = observation_frame.loc[
            observation_frame["consensus_target_id"].eq(target_id)
        ].copy()
        ready = candidates.loc[candidates["point_in_time_consensus_ready"].astype(bool)].copy()
        selected: pd.Series | None = None
        if not ready.empty:
            ready["record_last_updated_at"] = pd.to_datetime(
                ready["record_last_updated_at"], utc=True
            )
            selected = ready.sort_values(
                ["record_last_updated_at", "source_occurrence"], kind="mergesort"
            ).iloc[-1]
        status = _event_consensus_status(
            collected=target_id in completion_ids,
            failure_reason=(
                str(failure_by_target[target_id].get("reason"))
                if target_id in failure_by_target
                else None
            ),
            candidates=candidates,
            selected=selected,
            rejections=rejection_frame.loc[rejection_frame["consensus_target_id"].eq(target_id)],
        )
        event_rows.append(
            {
                "consensus_target_id": target_id,
                "validation_event_id": str(target.validation_event_id),
                "cik": str(target.cik),
                "ticker": str(target.ticker),
                "target_fiscal_year": int(str(target.target_fiscal_year)),
                "target_fiscal_quarter": str(target.target_fiscal_quarter),
                "consensus_cutoff_at": target.consensus_cutoff_at,
                "provider_query_complete": target_id in completion_ids,
                "provider_failure_reason": (
                    str(failure_by_target[target_id].get("reason"))
                    if target_id in failure_by_target
                    else None
                ),
                "matched_observations": len(candidates),
                "retrospective_consensus_candidate": bool(
                    candidates["retrospective_consensus_candidate"].any()
                )
                if not candidates.empty
                else False,
                "point_in_time_consensus_ready": selected is not None,
                "selected_revenue_consensus": (
                    float(selected["estimated_revenue"]) if selected is not None else math.nan
                ),
                "selected_record_last_updated_at": (
                    selected["record_last_updated_at"] if selected is not None else pd.NaT
                ),
                "selected_source_sha256": (
                    str(selected["source_sha256"]) if selected is not None else None
                ),
                "consensus_audit_status": status,
                "formal_model_admission": selected is not None,
                "selection_uses_return_values": False,
                "return_values_loaded": False,
            }
        )
    event_audit = pd.DataFrame(event_rows)
    ready_events = int(event_audit["point_in_time_consensus_ready"].sum())
    ready_issuers = int(
        event_audit.loc[event_audit["point_in_time_consensus_ready"], "cik"].nunique()
    )
    report: dict[str, Any] = {
        "parser_version": REVENUE_CONSENSUS_PARSER_VERSION,
        "consensus_targets": len(targets),
        "consensus_target_issuers": int(targets["cik"].nunique()),
        "provider_queries_complete": len(completion_ids),
        "provider_query_failures": len(failure_frame),
        "provider_entitlement_denied": int(failure_frame["reason"].eq("entitlement_denied").sum()),
        "raw_snapshots": len(snapshot_frame),
        "matched_consensus_observations": len(observation_frame),
        "retrospective_consensus_candidates": int(
            event_audit["retrospective_consensus_candidate"].sum()
        ),
        "point_in_time_consensus_ready_events": ready_events,
        "point_in_time_consensus_ready_issuers": ready_issuers,
        "minimum_ready_events_gate": ready_events >= cfg.minimum_ready_events,
        "minimum_ready_issuers_gate": ready_issuers >= cfg.minimum_ready_issuers,
        "formal_consensus_gate": (
            ready_events >= cfg.minimum_ready_events and ready_issuers >= cfg.minimum_ready_issuers
        ),
        "post_event_record_versions_admitted": False,
        "selection_uses_return_values": False,
        "return_values_loaded": False,
        "config": asdict(cfg),
    }
    return RevenueConsensusAuditTables(
        snapshot_frame,
        observation_frame,
        rejection_frame,
        failure_frame,
        event_audit,
        report,
    )


def build_revenue_consensus_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/revenue_consensus"),
    output_root: Path = Path("data/processed/revenue_consensus"),
    config: RevenueConsensusConfig | None = None,
) -> RevenueConsensusResult:
    """Persist a manifest-tied point-in-time consensus evidence audit."""
    cfg = config or RevenueConsensusConfig()
    cfg.validate()
    plan_hash = _sha256_file(plan_root / "output_manifest.json")
    targets = load_catalog_table(plan_root, "revenue_consensus_targets.parquet")
    tables = calculate_revenue_consensus_audit(targets, raw_root, cfg)
    snapshot_payload = [
        {
            "consensus_target_id": row["consensus_target_id"],
            "query_id": row["query_id"],
            "sha256": row["sha256"],
        }
        for row in tables.snapshots.to_dict("records")
    ]
    failure_payload = [
        {
            "consensus_target_id": row["consensus_target_id"],
            "reason": row["reason"],
            "failure_manifest_sha256": row["failure_manifest_sha256"],
        }
        for row in tables.failures.to_dict("records")
    ]
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_hash,
            "parser_version": REVENUE_CONSENSUS_PARSER_VERSION,
            "snapshots": snapshot_payload,
            "failures": failure_payload,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "revenue-consensus-audit-v2-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueConsensusResult(
            target.resolve(), _read_json(target / "revenue_consensus_audit_report.json")
        )
    if target.exists():
        raise RevenueConsensusError(f"Incomplete revenue-consensus audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "revenue_consensus_snapshots.parquet": tables.snapshots,
        "revenue_consensus_observations.parquet": tables.observations,
        "revenue_consensus_rejections.parquet": tables.rejections,
        "revenue_consensus_failures.parquet": tables.failures,
        "revenue_consensus_event_audit.parquet": tables.event_audit,
    }
    artifacts = _write_frames(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_hash,
        "artifacts": artifacts,
    }
    report_path = staging / "revenue_consensus_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_output_manifest(
        staging,
        dataset_version,
        {"plan": plan_hash},
        artifacts,
        report_path,
    )
    staging.replace(target)
    return RevenueConsensusResult(target.resolve(), report)


def parse_massive_revenue_consensus_response(
    content: bytes,
    *,
    target: dict[str, Any],
    source_sha256: str,
    maximum_event_date_distance_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse one preserved Massive earnings page without admitting post-event versions."""
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRequestError(
            f"Malformed Massive earnings JSON: {exc}", reason="malformed_json"
        ) from exc
    if not isinstance(document, dict) or document.get("status") != "OK":
        raise MassiveRequestError("Massive earnings response status is not OK", reason="bad_status")
    results = document.get("results", [])
    if not isinstance(results, list):
        raise MassiveRequestError("Massive earnings results must be a list", reason="bad_schema")
    cutoff = pd.Timestamp(target["consensus_cutoff_at"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    target_date = pd.Timestamp(target["target_actual_available_at"]).date()
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for occurrence, item in enumerate(results):
        if not isinstance(item, dict):
            rejections.append(_rejection(target, occurrence, "result_not_object", source_sha256))
            continue
        if str(item.get("ticker", "")).strip().upper() != str(target["ticker"]).upper():
            rejections.append(_rejection(target, occurrence, "ticker_mismatch", source_sha256))
            continue
        if _safe_int(item.get("fiscal_year")) != int(target["target_fiscal_year"]):
            rejections.append(_rejection(target, occurrence, "fiscal_year_mismatch", source_sha256))
            continue
        if (
            str(item.get("fiscal_period", "")).strip().upper()
            != str(target["target_fiscal_quarter"]).upper()
        ):
            rejections.append(
                _rejection(target, occurrence, "fiscal_period_mismatch", source_sha256)
            )
            continue
        estimate = _positive_float(item.get("estimated_revenue"))
        currency = str(item.get("currency", "")).strip().upper()
        method = str(item.get("revenue_method", "")).strip().lower()
        provider_date = pd.to_datetime(str(item.get("date") or ""), errors="coerce")
        last_updated = pd.to_datetime(
            str(item.get("last_updated") or ""), utc=True, errors="coerce"
        )
        event_date_distance = (
            abs((provider_date.date() - target_date).days) if pd.notna(provider_date) else None
        )
        exact_metric_ready = estimate is not None and currency == "USD" and method == "gaap"
        event_date_ready = (
            event_date_distance is not None
            and event_date_distance <= maximum_event_date_distance_days
        )
        version_precedes_cutoff = bool(pd.notna(last_updated) and last_updated < cutoff)
        point_in_time_ready = exact_metric_ready and event_date_ready and version_precedes_cutoff
        observations.append(
            {
                "consensus_target_id": target["consensus_target_id"],
                "validation_event_id": target["validation_event_id"],
                "cik": target["cik"],
                "ticker": str(target["ticker"]).upper(),
                "provider_record_id": _optional_text(item.get("benzinga_id")),
                "provider_event_date": (
                    provider_date.date().isoformat() if pd.notna(provider_date) else None
                ),
                "provider_event_time": _optional_text(item.get("time")),
                "fiscal_year": int(target["target_fiscal_year"]),
                "fiscal_period": str(target["target_fiscal_quarter"]),
                "estimated_revenue": estimate,
                "currency": currency or None,
                "revenue_method": method or None,
                "record_last_updated_at": (
                    last_updated.isoformat() if pd.notna(last_updated) else None
                ),
                "consensus_cutoff_at": cutoff.isoformat(),
                "event_date_distance_days": event_date_distance,
                "exact_metric_ready": exact_metric_ready,
                "event_date_ready": event_date_ready,
                "record_version_precedes_cutoff": version_precedes_cutoff,
                "retrospective_consensus_candidate": exact_metric_ready and event_date_ready,
                "point_in_time_consensus_ready": point_in_time_ready,
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    return observations, rejections


def _massive_earnings_query_url(row: pd.Series) -> str:
    query = urlencode(
        {
            "ticker": str(row["ticker"]),
            "fiscal_year": str(int(row["target_fiscal_year"])),
            "fiscal_period": str(row["target_fiscal_quarter"]),
            "limit": "100",
            "sort": "last_updated.asc",
        }
    )
    return f"{MASSIVE_EARNINGS_URL}?{query}"


def _collect_consensus_query(
    client: MassiveReferenceClient,
    *,
    raw_root: Path,
    target_id: str,
    query_id: str,
    query_url: str,
) -> tuple[list[str], int, int, int]:
    snapshots: list[str] = []
    requests_made = 0
    reused = 0
    record_count = 0
    url: str | None = query_url
    page_number = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError("Consensus pagination exceeded 20 pages", reason="pagination")
        existing = _existing_snapshot(raw_root, query_id, url)
        if existing is None:
            response = client.get(url)
            requests_made += response.requests_made
            manifest_path = _preserve_response(
                raw_root,
                target_id=target_id,
                query_id=query_id,
                page_number=page_number,
                requested_url=response.requested_url,
                response_url=response.response_url,
                content=response.content,
                fetched_at=response.fetched_at,
                status_code=response.status_code,
                headers=response.headers,
            )
        else:
            manifest_path = existing
            reused += 1
        manifest = _load_snapshot_manifest(manifest_path)
        content = Path(str(manifest["path"])).read_bytes()
        snapshots.append(str(manifest_path.resolve()))
        document = json.loads(content)
        if not isinstance(document, dict) or not isinstance(document.get("results", []), list):
            raise MassiveRequestError("Consensus response schema is invalid", reason="bad_schema")
        record_count += len(document.get("results", []))
        next_url = document.get("next_url")
        url = str(next_url) if isinstance(next_url, str) and next_url else None
    return snapshots, requests_made, reused, record_count


def _preserve_response(
    raw_root: Path,
    *,
    target_id: str,
    query_id: str,
    page_number: int,
    requested_url: str,
    response_url: str,
    content: bytes,
    fetched_at: datetime,
    status_code: int,
    headers: dict[str, str],
) -> Path:
    sha256 = hashlib.sha256(content).hexdigest()
    directory = raw_root / "snapshots" / query_id
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"{sha256}.bin"
    if raw_path.exists() and _sha256_file(raw_path) != sha256:
        raise RevenueConsensusError(f"Consensus raw snapshot hash mismatch: {raw_path}")
    if not raw_path.exists():
        raw_path.write_bytes(content)
    manifest = {
        "consensus_target_id": target_id,
        "query_id": query_id,
        "page_number": page_number,
        "requested_url": requested_url,
        "response_url": response_url,
        "sha256": sha256,
        "fetched_at": fetched_at.isoformat(),
        "path": str(raw_path.resolve()),
        "bytes": len(content),
        "status_code": status_code,
        "headers": headers,
    }
    manifest_path = directory / f"{sha256}.json"
    _write_json_immutable(manifest_path, manifest)
    return manifest_path


def _existing_snapshot(raw_root: Path, query_id: str, requested_url: str) -> Path | None:
    for path in (raw_root / "snapshots" / query_id).glob("*.json"):
        manifest = _load_snapshot_manifest(path)
        if str(manifest["requested_url"]) == requested_url:
            return path
    return None


def _load_snapshot_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    raw_path = Path(str(manifest.get("path", "")))
    expected_hash = str(manifest.get("sha256", ""))
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != int(manifest.get("bytes", -1))
        or _sha256_file(raw_path) != expected_hash
    ):
        raise RevenueConsensusError(f"Consensus snapshot is missing or invalid: {raw_path}")
    return manifest


def _completion_path(raw_root: Path, target_id: str) -> Path:
    return raw_root / "completions" / f"{target_id}.json"


def _load_completion(raw_root: Path, target_id: str) -> dict[str, Any] | None:
    path = _completion_path(raw_root, target_id)
    if not path.is_file():
        return None
    document = _read_json(path)
    for manifest in document.get("snapshots", []):
        _load_snapshot_manifest(Path(str(manifest)))
    return document


def _valid_completion(raw_root: Path, target_id: str) -> bool:
    try:
        return _load_completion(raw_root, target_id) is not None
    except (OSError, ValueError, KeyError, RevenueConsensusError):
        return False


def _write_failure(raw_root: Path, target_id: str, reason: str, detail: str) -> None:
    path = _failure_path(raw_root, target_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "consensus_target_id": target_id,
        "reason": reason,
        "detail": detail,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _failure_path(raw_root: Path, target_id: str) -> Path:
    return raw_root / "failures" / f"{target_id}.json"


def _load_failure(raw_root: Path, target_id: str) -> dict[str, Any] | None:
    path = _failure_path(raw_root, target_id)
    return _read_json(path) if path.is_file() else None


def _write_json_immutable(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RevenueConsensusError(f"Immutable consensus metadata differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _event_consensus_status(
    *,
    collected: bool,
    failure_reason: str | None,
    candidates: pd.DataFrame,
    selected: pd.Series | None,
    rejections: pd.DataFrame,
) -> str:
    if not collected:
        if failure_reason:
            return f"provider_{failure_reason}"
        return "provider_query_pending"
    if selected is not None:
        return "point_in_time_consensus_ready"
    if not candidates.empty:
        if candidates["retrospective_consensus_candidate"].astype(bool).any():
            return "retrospective_only_post_event_or_unverifiable_version"
        if not candidates["exact_metric_ready"].astype(bool).any():
            return "consensus_metric_not_gaap_usd_revenue"
        if not candidates["event_date_ready"].astype(bool).any():
            return "provider_event_date_mismatch"
    if not rejections.empty:
        return "provider_records_failed_identity_or_period_match"
    return "provider_returned_no_matching_record"


def _rejection(
    target: dict[str, Any], occurrence: int, reason: str, source_sha256: str
) -> dict[str, Any]:
    return {
        "consensus_target_id": target["consensus_target_id"],
        "validation_event_id": target["validation_event_id"],
        "source_occurrence": occurrence,
        "reason": reason,
        "source_sha256": source_sha256,
    }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        raise RevenueConsensusError(f"{label} contains invalid timestamps")
    return parsed


def _stable_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RevenueConsensusError(f"{label} lacks required columns: {missing}")


def _observation_columns() -> list[str]:
    return [
        "consensus_target_id",
        "validation_event_id",
        "cik",
        "ticker",
        "provider_record_id",
        "provider_event_date",
        "provider_event_time",
        "fiscal_year",
        "fiscal_period",
        "estimated_revenue",
        "currency",
        "revenue_method",
        "record_last_updated_at",
        "consensus_cutoff_at",
        "event_date_distance_days",
        "exact_metric_ready",
        "event_date_ready",
        "record_version_precedes_cutoff",
        "retrospective_consensus_candidate",
        "point_in_time_consensus_ready",
        "source_occurrence",
        "source_sha256",
    ]


def _rejection_columns() -> list[str]:
    return [
        "consensus_target_id",
        "validation_event_id",
        "source_occurrence",
        "reason",
        "source_sha256",
    ]


def _failure_columns() -> list[str]:
    return [
        "consensus_target_id",
        "reason",
        "failed_at",
        "failure_manifest_sha256",
    ]


def _snapshot_columns() -> list[str]:
    return [
        "consensus_target_id",
        "query_id",
        "page_number",
        "requested_url",
        "response_url",
        "sha256",
        "fetched_at",
        "path",
        "bytes",
        "status_code",
        "headers",
    ]


def _write_frames(staging: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return artifacts


def _write_output_manifest(
    staging: Path,
    dataset_version: str,
    input_hashes: dict[str, str],
    artifacts: dict[str, dict[str, Any]],
    report_path: Path,
) -> None:
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


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
