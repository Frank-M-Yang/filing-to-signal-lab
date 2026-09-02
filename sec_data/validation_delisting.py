"""Formal listing-outcome evidence and event-level delisting classification."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_delisting_outcomes import load_collected_massive_listing_status

VALIDATION_DELISTING_VERSION: Final[str] = "validation-delisting-v3-price-supersedes-status"


class ValidationDelistingError(FilingCatalogError):
    """Raised when formal listing-outcome evidence violates frozen assumptions."""


@dataclass(frozen=True)
class ValidationDelistingResult:
    """Immutable formal listing-outcome target plan or audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_delisting_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Translate execution-plan targets into the shared Massive collector schema."""
    _require_columns(
        targets,
        {
            "delisting_target_id",
            "delisting_acquisition_order",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "security_identity_key_kind",
            "outcome_cutoff_date",
            "delisting_collection_authorized",
            "selection_uses_later_listing_status",
            "outcome_values_loaded",
        },
        "validation delisting targets",
    )
    if targets.empty or targets["delisting_target_id"].duplicated().any():
        raise ValidationDelistingError("Formal delisting targets are empty or duplicated")
    if not targets["delisting_collection_authorized"].astype(bool).all():
        raise ValidationDelistingError("A formal delisting query is unauthorized")
    if targets["selection_uses_later_listing_status"].astype(bool).any():
        raise ValidationDelistingError("Delisting target selection used later status")
    if targets["outcome_values_loaded"].astype(bool).any():
        raise ValidationDelistingError("Delisting target selection opened outcomes")
    result = targets.sort_values("delisting_acquisition_order").copy()
    result["outcome_target_id"] = result["delisting_target_id"]
    result["outcome_acquisition_order"] = result["delisting_acquisition_order"]
    result["target_id"] = result["delisting_target_id"]
    result["security_id"] = result["security_identity_key"]
    result["security_id_kind"] = result["security_identity_key_kind"]
    result["selection_uses_return_values"] = False
    result["returns_computed"] = False
    return result


def calculate_validation_listing_outcomes(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    rejections: pd.DataFrame,
    completions: pd.DataFrame,
    components: pd.DataFrame,
    reconciled_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify one listing outcome per security and apply it to every event."""
    _require_columns(
        targets,
        {
            "outcome_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_id",
            "security_id_kind",
            "outcome_cutoff_date",
        },
        "formal listing targets",
    )
    _require_columns(
        observations,
        {
            "outcome_target_id",
            "query_active",
            "provider_active",
            "ticker",
            "security_type",
            "exchange_mic",
            "provider_security_id",
            "delisted_utc",
            "last_updated_utc",
            "source_sha256",
        },
        "listing observations",
    )
    _require_columns(
        components,
        {
            "security_event_component_id",
            "cik",
            "ticker",
            "exchange_mic",
            "identity_key",
            "formal_exit_session_date",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "execution components",
    )
    _require_columns(
        reconciled_prices,
        {"security_event_component_id", "reconciled_formal_price_ready"},
        "reconciled price status",
    )
    if components["selection_uses_return_values"].astype(bool).any():
        raise ValidationDelistingError("Event construction used return values")
    if components["outcome_values_loaded"].astype(bool).any():
        raise ValidationDelistingError("Event construction opened outcomes")
    completion_ids = set(completions["outcome_target_id"].astype(str))
    rejection_ids = set(rejections["outcome_target_id"].astype(str))
    security_rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        outcome_target_id = str(target.outcome_target_id)
        records = observations.loc[observations["outcome_target_id"].eq(outcome_target_id)].copy()
        if str(target.security_id_kind) == "share_class_figi":
            exact = records.loc[
                records["provider_security_id"].eq(str(target.security_id))
                & records["security_type"].eq("CS")
            ].copy()
        else:
            exact = records.loc[
                records["ticker"].eq(str(target.selected_ticker))
                & records["exchange_mic"].eq(str(target.selected_exchange_mic))
                & records["security_type"].eq("CS")
            ].copy()
        active = exact.loc[exact["query_active"].astype(bool)].copy()
        inactive = exact.loc[~exact["query_active"].astype(bool)].copy()
        status = "not_collected"
        delisting_date: str | None = None
        outcome_ticker: str | None = None
        source_sha256: str | None = None
        ready = False
        if outcome_target_id in rejection_ids:
            status = "parse_rejection"
        elif outcome_target_id not in completion_ids:
            status = "not_collected"
        elif not active.empty:
            preferred = _preferred_record(active, str(target.selected_ticker))
            if bool(preferred["provider_active"]) and pd.isna(preferred["delisted_utc"]):
                status = "active_at_outcome_cutoff"
                ready = True
                outcome_ticker = str(preferred["ticker"])
                source_sha256 = str(preferred["source_sha256"])
            else:
                status = "active_record_semantic_conflict"
        elif not inactive.empty:
            dated = inactive.loc[inactive["delisted_utc"].notna()].copy()
            if dated.empty:
                status = "inactive_record_missing_delisting_date"
            else:
                dated["_delisted_at"] = pd.to_datetime(
                    dated["delisted_utc"], utc=True, errors="coerce"
                )
                dated = dated.loc[dated["_delisted_at"].notna()].copy()
                if dated.empty:
                    status = "inactive_record_invalid_delisting_date"
                else:
                    latest = dated["_delisted_at"].max()
                    preferred = _preferred_record(
                        dated.loc[dated["_delisted_at"].eq(latest)],
                        str(target.selected_ticker),
                    )
                    status = "inactive_with_delisting_date"
                    ready = True
                    delisting_date = latest.date().isoformat()
                    outcome_ticker = str(preferred["ticker"])
                    source_sha256 = str(preferred["source_sha256"])
        else:
            status = "security_not_resolved_at_cutoff"
        security_rows.append(
            {
                "outcome_target_id": outcome_target_id,
                "cik": str(target.cik).zfill(10),
                "ticker_at_event": str(target.selected_ticker),
                "exchange_mic": str(target.selected_exchange_mic),
                "security_id": str(target.security_id),
                "security_id_kind": str(target.security_id_kind),
                "outcome_cutoff_date": str(target.outcome_cutoff_date),
                "listing_outcome_status": status,
                "ticker_at_outcome": outcome_ticker,
                "delisting_date": delisting_date,
                "listing_outcome_ready": ready,
                "source_snapshot_sha256": source_sha256,
                "terminal_value_observed": False,
                "delisting_return_observed": False,
                "returns_computed": False,
            }
        )
    security = pd.DataFrame(security_rows)
    mapping = targets[["outcome_target_id", "cik", "security_id"]].rename(
        columns={"security_id": "identity_key"}
    )
    events = components.merge(
        mapping,
        on=["cik", "identity_key"],
        how="left",
        validate="many_to_one",
    )
    if events["outcome_target_id"].isna().any():
        raise ValidationDelistingError("An execution component has no listing target")
    events = events.merge(
        security[
            [
                "outcome_target_id",
                "listing_outcome_status",
                "ticker_at_outcome",
                "delisting_date",
                "listing_outcome_ready",
                "source_snapshot_sha256",
            ]
        ],
        on="outcome_target_id",
        how="left",
        validate="many_to_one",
    ).merge(
        reconciled_prices[["security_event_component_id", "reconciled_formal_price_ready"]],
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    )
    exit_dates = pd.to_datetime(events["formal_exit_session_date"], errors="coerce")
    delisting_dates = pd.to_datetime(events["delisting_date"], errors="coerce")
    events["delisted_on_or_before_formal_exit"] = (
        delisting_dates.notna() & exit_dates.notna() & delisting_dates.le(exit_dates)
    )
    events["later_delisted_after_formal_exit"] = (
        delisting_dates.notna() & exit_dates.notna() & delisting_dates.gt(exit_dates)
    )
    formal_price_ready = events["reconciled_formal_price_ready"].astype(bool)
    events["terminal_value_required"] = (
        events["delisted_on_or_before_formal_exit"] & ~formal_price_ready
    )
    events["delisting_window_ready"] = (
        formal_price_ready
        | events["listing_outcome_status"].eq("active_at_outcome_cutoff")
        | events["later_delisted_after_formal_exit"]
    )
    events["market_outcome_ready"] = formal_price_ready
    events["outcome_exception_status"] = "ready"
    events.loc[
        ~events["listing_outcome_ready"].astype(bool) & formal_price_ready,
        "outcome_exception_status",
    ] = "listing_status_advisory_nonblocking"
    events.loc[
        ~formal_price_ready,
        "outcome_exception_status",
    ] = "price_execution_unresolved"
    events.loc[
        events["terminal_value_required"].astype(bool),
        "outcome_exception_status",
    ] = "terminal_value_required"
    events["terminal_value_observed"] = False
    events["delisting_return_observed"] = False
    events["returns_computed"] = False
    return security, events


def calculate_validation_outcome_repair_queue(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-collection evidence gaps without changing the frozen sample."""
    _require_columns(
        events,
        {
            "outcome_target_id",
            "cik",
            "ticker",
            "exchange_mic",
            "listing_outcome_status",
            "listing_outcome_ready",
            "terminal_value_required",
            "reconciled_formal_price_ready",
            "market_outcome_ready",
            "outcome_exception_status",
        },
        "event listing outcomes",
    )
    exceptions = events.loc[~events["market_outcome_ready"].astype(bool)].copy()
    columns = [
        "outcome_target_id",
        "cik",
        "ticker",
        "exchange_mic",
        "listing_outcome_status",
        "affected_event_count",
        "terminal_value_required_events",
        "listing_status_unresolved_events",
        "price_execution_unresolved_events",
        "repair_priority",
        "repair_action",
        "repair_selection_uses_return_values",
    ]
    if exceptions.empty:
        return pd.DataFrame(columns=columns)
    exceptions["_terminal"] = exceptions["terminal_value_required"].astype(int)
    exceptions["_listing_unresolved"] = (~exceptions["listing_outcome_ready"].astype(bool)).astype(
        int
    )
    exceptions["_price_unresolved"] = (
        ~exceptions["reconciled_formal_price_ready"].astype(bool)
    ).astype(int)
    queue = (
        exceptions.groupby(
            [
                "outcome_target_id",
                "cik",
                "ticker",
                "exchange_mic",
                "listing_outcome_status",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            affected_event_count=("outcome_exception_status", "size"),
            terminal_value_required_events=("_terminal", "sum"),
            listing_status_unresolved_events=("_listing_unresolved", "sum"),
            price_execution_unresolved_events=("_price_unresolved", "sum"),
        )
        .sort_values(["ticker", "outcome_target_id"], kind="stable")
        .reset_index(drop=True)
    )
    queue["repair_priority"] = 3
    queue["repair_action"] = "review_authorized_price_execution_evidence"
    listing_mask = queue["listing_status_unresolved_events"].gt(0)
    queue.loc[listing_mask, "repair_priority"] = 2
    queue.loc[listing_mask, "repair_action"] = (
        "collect_secondary_identity_and_listing_status_evidence"
    )
    combined_mask = listing_mask & queue["price_execution_unresolved_events"].gt(0)
    queue.loc[combined_mask, "repair_action"] = (
        "review_execution_gap_and_secondary_listing_evidence"
    )
    missing_date = queue["listing_outcome_status"].eq("inactive_record_missing_delisting_date")
    queue.loc[missing_date, "repair_action"] = "collect_primary_delisting_date_evidence"
    terminal_mask = queue["terminal_value_required_events"].gt(0)
    queue.loc[terminal_mask, "repair_priority"] = 1
    queue.loc[terminal_mask, "repair_action"] = (
        "collect_primary_terminal_consideration_and_settlement_evidence"
    )
    queue["repair_selection_uses_return_values"] = False
    return (
        queue[columns]
        .sort_values(["repair_priority", "ticker"], kind="stable")
        .reset_index(drop=True)
    )


def build_validation_delisting_plan(
    execution_plan_root: Path,
    *,
    output_root: Path = Path("data/market/validation_delisting/targets"),
) -> ValidationDelistingResult:
    """Persist formal listing-status targets before any outcome query."""
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    targets = calculate_validation_delisting_targets(
        load_catalog_table(execution_plan_root, "validation_security_delisting_targets.parquet")
    )
    payload = json.dumps(
        {
            "version": VALIDATION_DELISTING_VERSION,
            "execution_manifest_sha256": execution_manifest,
            "targets": targets["outcome_target_id"].tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_DELISTING_VERSION + "-targets-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationDelistingResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationDelistingError(f"Incomplete formal delisting plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "massive_delisting_targets.parquet"
    targets.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "targets": len(targets),
        "outcome_cutoff_date": targets["outcome_cutoff_date"].iloc[0],
        "query_method": "cik_cutoff_active_true_then_inactive_if_security_absent",
        "selection_uses_later_listing_status": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "execution_plan_manifest_sha256": execution_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationDelistingResult(target.resolve(), report)


def build_validation_delisting_audit(
    plan_root: Path,
    execution_plan_root: Path,
    reconciliation_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_delisting"),
    output_root: Path = Path("data/processed/validation_delisting"),
) -> ValidationDelistingResult:
    """Persist security and event-level listing outcomes without terminal imputation."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    reconciliation_manifest = _sha256_file(reconciliation_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    if plan_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationDelistingError("Delisting plan does not tie to execution plan")
    targets = load_catalog_table(plan_root, "massive_delisting_targets.parquet")
    observations, rejections, completions = load_collected_massive_listing_status(
        targets, raw_root=raw_root
    )
    security, events = calculate_validation_listing_outcomes(
        targets,
        observations,
        rejections,
        completions,
        load_catalog_table(execution_plan_root, "validation_security_event_components.parquet"),
        load_catalog_table(reconciliation_root, "validation_reconciled_event_price_status.parquet"),
    )
    repair_queue = calculate_validation_outcome_repair_queue(events)
    event_exceptions = events.loc[~events["market_outcome_ready"].astype(bool)].copy()
    listing_advisories = events.loc[
        ~events["listing_outcome_ready"].astype(bool)
        & events["reconciled_formal_price_ready"].astype(bool)
    ].copy()
    collected = int(security["listing_outcome_status"].ne("not_collected").sum())
    security_ready = int(security["listing_outcome_ready"].sum())
    event_window_ready = int(events["delisting_window_ready"].sum())
    terminal_required = int(events["terminal_value_required"].sum())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "execution_manifest_sha256": execution_manifest,
            "reconciliation_manifest_sha256": reconciliation_manifest,
            "source_hashes": sorted(
                observations.get("source_sha256", pd.Series(dtype=str)).unique().tolist()
            ),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_DELISTING_VERSION + "-audit-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationDelistingResult(
            target.resolve(), _read_json(target / "validation_delisting_report.json")
        )
    if target.exists():
        raise ValidationDelistingError(f"Incomplete formal delisting audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "listing_status_collection",
                collected,
                len(targets),
                collected == len(targets),
                "Every frozen security has a completed active/inactive status query",
            ),
            _gate(
                "security_listing_outcome",
                security_ready,
                len(targets),
                security_ready == len(targets),
                "Each security is active or has an evidenced delisting date",
            ),
            _gate(
                "event_delisting_window",
                event_window_ready,
                len(events),
                event_window_ready == len(events),
                "Each event is active through exit or delists strictly after exit",
            ),
            _gate(
                "formal_market_outcome",
                int(events["market_outcome_ready"].sum()),
                len(events),
                bool(events["market_outcome_ready"].all()),
                "Each event has executable prices and a resolved listing outcome",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(events),
                False,
                "No terminal value or return is imputed during listing-status audit",
            ),
        ]
    )
    outputs = {
        "validation_delisting_targets.parquet": targets,
        "validation_listing_status_observations.parquet": observations,
        "validation_listing_status_rejections.parquet": rejections,
        "validation_listing_status_completions.parquet": completions,
        "validation_security_listing_outcomes.parquet": security,
        "validation_event_listing_outcomes.parquet": events,
        "validation_event_outcome_exceptions.parquet": event_exceptions,
        "validation_nonblocking_listing_advisories.parquet": listing_advisories,
        "validation_outcome_repair_queue.parquet": repair_queue,
        "validation_delisting_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "execution_plan_manifest_sha256": execution_manifest,
        "reconciliation_manifest_sha256": reconciliation_manifest,
        "targets": len(targets),
        "targets_collected": collected,
        "security_listing_outcomes_ready": security_ready,
        "active_at_outcome_cutoff": int(
            security["listing_outcome_status"].eq("active_at_outcome_cutoff").sum()
        ),
        "inactive_with_delisting_date": int(
            security["listing_outcome_status"].eq("inactive_with_delisting_date").sum()
        ),
        "event_components": len(events),
        "event_delisting_windows_ready": event_window_ready,
        "market_outcome_ready_events": int(events["market_outcome_ready"].sum()),
        "event_outcome_exceptions": len(event_exceptions),
        "nonblocking_listing_advisories": len(listing_advisories),
        "securities_in_repair_queue": len(repair_queue),
        "terminal_value_required_events": terminal_required,
        "terminal_values_observed": 0,
        "delisting_returns_observed": 0,
        "zero_delisting_return_imputed": False,
        "selection_uses_later_listing_status": False,
        "selection_uses_return_values": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_delisting_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest,
                "execution_plan_manifest_sha256": execution_manifest,
                "reconciliation_manifest_sha256": reconciliation_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationDelistingResult(target.resolve(), report)


def _preferred_record(records: pd.DataFrame, preferred_ticker: str) -> pd.Series[Any]:
    preferred = records.loc[records["ticker"].eq(preferred_ticker)]
    candidates = preferred if not preferred.empty else records
    return candidates.sort_values(
        ["last_updated_utc", "ticker"], na_position="first", kind="stable"
    ).iloc[-1]


def _gate(
    gate: str, numerator: int, denominator: int, passed: bool, meaning: str
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
        raise ValidationDelistingError(f"{label} lacks required columns: {missing}")


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
