"""Outcome-blind full price collection and endpoint audit for validation events."""

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
from sec_data.massive_prices import load_collected_massive_daily_bars

VALIDATION_PRICE_BATCH_VERSION: Final[str] = "validation-price-batch-v1-outcome-blind"


class ValidationPriceBatchError(FilingCatalogError):
    """Raised when the formal validation-price batch violates research gates."""


@dataclass(frozen=True)
class ValidationPriceBatchResult:
    """Immutable formal price target plan or structural audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_price_batch_targets(price_targets: pd.DataFrame) -> pd.DataFrame:
    """Validate and freeze every execution-plan price span without outcomes."""
    required = {
        "price_target_id",
        "cik",
        "selected_ticker",
        "selected_exchange_mic",
        "security_identity_key",
        "security_identity_key_kind",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
        "price_window_start",
        "price_window_end",
        "adjusted",
        "price_collection_authorized",
        "selection_uses_return_values",
        "outcome_values_loaded",
        "price_acquisition_order",
    }
    _require_columns(price_targets, required, "validation price targets")
    if price_targets.empty or price_targets["price_target_id"].duplicated().any():
        raise ValidationPriceBatchError("Formal price targets are empty or duplicated")
    if price_targets["adjusted"].astype(bool).any():
        raise ValidationPriceBatchError("Formal prices must be requested unadjusted")
    if not price_targets["price_collection_authorized"].astype(bool).all():
        raise ValidationPriceBatchError("Formal batch includes an unauthorized price span")
    if price_targets["selection_uses_return_values"].astype(bool).any():
        raise ValidationPriceBatchError("Formal target construction used return values")
    if price_targets["outcome_values_loaded"].astype(bool).any():
        raise ValidationPriceBatchError("Formal target construction opened outcomes")
    targets = price_targets.sort_values(
        ["price_acquisition_order", "price_target_id"], kind="mergesort"
    ).reset_index(drop=True)
    if targets["price_acquisition_order"].duplicated().any():
        raise ValidationPriceBatchError("Formal price acquisition order is duplicated")
    targets["target_id"] = targets["price_target_id"]
    targets["batch_selection_uses_price_coverage"] = False
    targets["batch_selection_uses_return_values"] = False
    targets["batch_replacement_after_failure_allowed"] = False
    return targets


def build_validation_price_batch_plan(
    execution_plan_root: Path,
    *,
    output_root: Path = Path("data/market/validation_prices/targets"),
) -> ValidationPriceBatchResult:
    """Persist the complete frozen price-query universe before collection."""
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    if not bool(execution_report.get("price_collection_authorized")):
        raise ValidationPriceBatchError("Execution plan has not authorized price collection")
    source = load_catalog_table(execution_plan_root, "validation_security_price_targets.parquet")
    targets = calculate_validation_price_batch_targets(source)
    payload = json.dumps(
        {
            "execution_manifest_sha256": execution_manifest,
            "version": VALIDATION_PRICE_BATCH_VERSION,
            "price_target_ids": targets["price_target_id"].astype(str).tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_PRICE_BATCH_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPriceBatchResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationPriceBatchError(f"Incomplete formal price plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "massive_daily_price_targets.parquet"
    targets.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "price_targets": len(targets),
        "query_adjusted": False,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "price_values_loaded": False,
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
    return ValidationPriceBatchResult(target.resolve(), report)


def build_validation_price_batch_audit(
    plan_root: Path,
    execution_plan_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_price_pilot"),
    output_root: Path = Path("data/processed/validation_prices"),
) -> ValidationPriceBatchResult:
    """Audit full-batch bars and formal endpoint presence without returns."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    if plan_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationPriceBatchError("Formal price plan does not tie to execution plan")
    targets = load_catalog_table(plan_root, "massive_daily_price_targets.parquet")
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    if components["selection_uses_return_values"].astype(bool).any():
        raise ValidationPriceBatchError("Execution components used return values")
    if components["outcome_values_loaded"].astype(bool).any():
        raise ValidationPriceBatchError("Execution components opened outcomes")
    bars, rejections, completions = load_collected_massive_daily_bars(targets, raw_root=raw_root)
    mapping = targets[
        [
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "price_target_id",
        ]
    ].rename(
        columns={
            "selected_ticker": "ticker",
            "selected_exchange_mic": "exchange_mic",
            "security_identity_key": "identity_key",
        }
    )
    endpoints = components.merge(
        mapping,
        on=["cik", "ticker", "exchange_mic", "identity_key"],
        how="left",
        validate="many_to_one",
    )
    if endpoints["price_target_id"].isna().any():
        raise ValidationPriceBatchError("An execution component has no formal price target")
    endpoint_columns = [
        "security_event_component_id",
        "market_event_id",
        "validation_event_id",
        "cik",
        "ticker",
        "exchange_mic",
        "identity_key",
        "component_weight",
        "formal_entry_session_date",
        "formal_entry_open_at",
        "formal_entry_price_type",
        "formal_exit_session_date",
        "formal_exit_close_at",
        "formal_exit_price_type",
        "price_target_id",
    ]
    endpoints = endpoints[endpoint_columns].copy()
    observed_dates = {
        str(target_id): set(group["session_date"].astype(str))
        for target_id, group in bars.groupby("price_target_id")
    }
    endpoints["entry_price_date_observed"] = endpoints.apply(
        lambda row: (
            str(row["formal_entry_session_date"])
            in observed_dates.get(str(row["price_target_id"]), set())
        ),
        axis=1,
    )
    endpoints["exit_price_date_observed"] = endpoints.apply(
        lambda row: (
            str(row["formal_exit_session_date"])
            in observed_dates.get(str(row["price_target_id"]), set())
        ),
        axis=1,
    )
    endpoints["required_endpoint_dates_ready"] = (
        endpoints["entry_price_date_observed"] & endpoints["exit_price_date_observed"]
    )
    endpoints["returns_computed"] = False
    target_audit = targets[
        [
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "price_window_start",
            "price_window_end",
        ]
    ].merge(completions, on="price_target_id", how="left", validate="one_to_one")
    target_audit["provider_query_complete"] = target_audit["completed_at"].notna()
    target_audit["validated_bar_rows"] = (
        target_audit["price_target_id"]
        .map(bars.groupby("price_target_id").size())
        .fillna(0)
        .astype(int)
    )
    target_audit["parser_rejection_rows"] = (
        target_audit["price_target_id"]
        .map(rejections.groupby("price_target_id").size())
        .fillna(0)
        .astype(int)
    )
    endpoint_ready = endpoints.groupby("price_target_id")["required_endpoint_dates_ready"].all()
    target_audit["required_endpoint_dates_ready"] = (
        target_audit["price_target_id"].map(endpoint_ready).fillna(False).astype(bool)
    )
    target_audit["formal_price_ready"] = (
        target_audit["provider_query_complete"]
        & target_audit["validated_bar_rows"].gt(0)
        & target_audit["parser_rejection_rows"].eq(0)
        & target_audit["required_endpoint_dates_ready"]
    )
    target_audit["returns_computed"] = False
    collected = int(target_audit["provider_query_complete"].sum())
    ready_targets = int(target_audit["formal_price_ready"].sum())
    ready_endpoints = int(endpoints["required_endpoint_dates_ready"].sum())
    source_hashes = sorted(
        bars.get("source_sha256", pd.Series(dtype=str)).astype(str).unique().tolist()
    )
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "execution_manifest_sha256": execution_manifest,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "validation-price-batch-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPriceBatchResult(
            target.resolve(), _read_json(target / "validation_price_batch_report.json")
        )
    if target.exists():
        raise ValidationPriceBatchError(f"Incomplete formal price audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "formal_price_collection",
                collected,
                len(targets),
                collected == len(targets),
                "Every frozen security span has a preserved provider response",
            ),
            _gate(
                "formal_required_endpoint_dates",
                ready_endpoints,
                len(endpoints),
                ready_endpoints == len(endpoints),
                "Every event component contains its declared entry open and exit close date",
            ),
            _gate(
                "formal_price_readiness",
                ready_targets,
                len(targets),
                ready_targets == len(targets),
                "Every target is collected, parser-clean, and endpoint-complete",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(endpoints),
                False,
                "No return is computed during the formal price acquisition phase",
            ),
        ]
    )
    outputs = {
        "validation_price_batch_targets.parquet": targets,
        "validation_price_batch_bars.parquet": bars,
        "validation_price_batch_rejections.parquet": rejections,
        "validation_price_batch_completions.parquet": completions,
        "validation_price_batch_event_endpoints.parquet": endpoints,
        "validation_price_batch_target_audit.parquet": target_audit,
        "validation_price_batch_gates.parquet": gates,
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
        "price_targets": len(targets),
        "targets_collected": collected,
        "targets_remaining": len(targets) - collected,
        "validated_daily_bars": len(bars),
        "parser_rejections": len(rejections),
        "event_component_endpoints": len(endpoints),
        "event_component_endpoints_ready": ready_endpoints,
        "formal_price_targets_ready": ready_targets,
        "formal_price_gate": ready_targets == len(targets),
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "price_values_loaded": not bars.empty,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "raw_store": str(raw_root),
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_price_batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest,
                "execution_plan_manifest_sha256": execution_manifest,
                "source_hashes": source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPriceBatchResult(target.resolve(), report)


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
        raise ValidationPriceBatchError(f"{label} lacks required columns: {missing}")


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
