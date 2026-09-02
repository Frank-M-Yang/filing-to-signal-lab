"""Reconcile primary and secondary price readiness without calculating returns."""

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

VALIDATION_PRICE_RECONCILIATION_VERSION: Final[str] = (
    "validation-price-reconciliation-v2-dual-endpoint-source-tiered"
)


class ValidationPriceReconciliationError(FilingCatalogError):
    """Raised when primary and secondary price evidence cannot be reconciled."""


@dataclass(frozen=True)
class ValidationPriceReconciliationResult:
    """Immutable source-tiered price readiness result."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_price_reconciliation(
    event_status: pd.DataFrame,
    fallback_targets: pd.DataFrame,
    fallback_prices: pd.DataFrame,
    benchmark_alignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exact frozen fallback dates to their declared consumers only."""
    _require_columns(
        event_status,
        {
            "security_event_component_id",
            "entry_price_date_observed",
            "exit_price_date_observed",
            "required_endpoint_dates_ready",
            "formal_price_status",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "returns_computed",
        },
        "event execution status",
    )
    _require_columns(
        fallback_targets,
        {"fallback_target_id", "consumer_kind", "consumer_ids", "ticker"},
        "fallback targets",
    )
    _require_columns(
        fallback_prices,
        {
            "fallback_target_id",
            "session_date",
            "open_unadjusted",
            "close_unadjusted",
            "source_provider",
            "source_sha256",
            "adjusted_close_used",
        },
        "fallback prices",
    )
    _require_columns(
        benchmark_alignment,
        {"target_id", "required_session_date", "benchmark_price_ready"},
        "benchmark alignment",
    )
    if event_status["selection_uses_return_values"].astype(bool).any():
        raise ValidationPriceReconciliationError("Event readiness used return values")
    if event_status["outcome_values_loaded"].astype(bool).any():
        raise ValidationPriceReconciliationError("Event readiness opened outcomes")
    if event_status["returns_computed"].astype(bool).any():
        raise ValidationPriceReconciliationError("Event returns were already computed")
    if fallback_prices["adjusted_close_used"].astype(bool).any():
        raise ValidationPriceReconciliationError("Fallback evidence used adjusted close")
    price_evidence = fallback_prices.merge(
        fallback_targets[["fallback_target_id", "consumer_kind", "consumer_ids", "ticker"]],
        on="fallback_target_id",
        how="left",
        validate="one_to_one",
    )
    issuer = price_evidence.loc[price_evidence["consumer_kind"].eq("issuer_formal_entry")].copy()
    if issuer["consumer_ids"].duplicated().any():
        raise ValidationPriceReconciliationError("Issuer fallback consumers are duplicated")
    issuer_lookup = issuer.set_index("consumer_ids")
    events = event_status.copy()
    events["fallback_entry_target_id"] = events["security_event_component_id"].map(
        issuer_lookup["fallback_target_id"]
    )
    events["fallback_entry_source_provider"] = events["security_event_component_id"].map(
        issuer_lookup["source_provider"]
    )
    events["fallback_entry_source_sha256"] = events["security_event_component_id"].map(
        issuer_lookup["source_sha256"]
    )
    events["fallback_entry_price_ready"] = events["fallback_entry_target_id"].notna()
    issuer_exit = price_evidence.loc[
        price_evidence["consumer_kind"].eq("issuer_formal_exit")
    ].copy()
    if issuer_exit["consumer_ids"].duplicated().any():
        raise ValidationPriceReconciliationError("Issuer exit fallback consumers are duplicated")
    issuer_exit_lookup = issuer_exit.set_index("consumer_ids")
    events["fallback_exit_target_id"] = events["security_event_component_id"].map(
        issuer_exit_lookup["fallback_target_id"]
    )
    events["fallback_exit_source_provider"] = events["security_event_component_id"].map(
        issuer_exit_lookup["source_provider"]
    )
    events["fallback_exit_source_sha256"] = events["security_event_component_id"].map(
        issuer_exit_lookup["source_sha256"]
    )
    events["fallback_exit_price_ready"] = events["fallback_exit_target_id"].notna()
    events["reconciled_entry_price_ready"] = (
        events["entry_price_date_observed"].astype(bool) | events["fallback_entry_price_ready"]
    )
    events["reconciled_exit_price_ready"] = (
        events["exit_price_date_observed"].astype(bool) | events["fallback_exit_price_ready"]
    )
    events["reconciled_formal_price_ready"] = (
        events["reconciled_entry_price_ready"] & events["reconciled_exit_price_ready"]
    )
    events["reconciled_price_source_tier"] = "unresolved"
    events.loc[
        events["required_endpoint_dates_ready"].astype(bool),
        "reconciled_price_source_tier",
    ] = "primary_massive"
    events.loc[
        ~events["required_endpoint_dates_ready"].astype(bool)
        & events["reconciled_formal_price_ready"],
        "reconciled_price_source_tier",
    ] = "primary_plus_secondary_yahoo"
    secondary_only = (
        ~events["entry_price_date_observed"].astype(bool)
        & ~events["exit_price_date_observed"].astype(bool)
        & events["reconciled_formal_price_ready"]
    )
    events.loc[secondary_only, "reconciled_price_source_tier"] = "secondary_yahoo"
    events["reconciled_formal_price_status"] = events["formal_price_status"]
    events.loc[
        events["required_endpoint_dates_ready"].astype(bool),
        "reconciled_formal_price_status",
    ] = "price_ready_primary"
    events.loc[
        ~events["required_endpoint_dates_ready"].astype(bool)
        & events["reconciled_formal_price_ready"],
        "reconciled_formal_price_status",
    ] = "price_ready_secondary_history"
    events["prices_imputed"] = False
    events["returns_computed"] = False
    benchmark_rows: list[dict[str, Any]] = []
    for row in price_evidence.loc[
        price_evidence["consumer_kind"].eq("benchmark_required_session")
    ].itertuples(index=False):
        for consumer_id in str(row.consumer_ids).split("|"):
            benchmark_rows.append(
                {
                    "target_id": consumer_id,
                    "required_session_date": str(row.session_date),
                    "fallback_target_id": str(row.fallback_target_id),
                    "fallback_source_provider": str(row.source_provider),
                    "fallback_source_sha256": str(row.source_sha256),
                    "fallback_open_unadjusted": float(str(row.open_unadjusted)),
                    "fallback_close_unadjusted": float(str(row.close_unadjusted)),
                }
            )
    benchmark_fallback = pd.DataFrame(
        benchmark_rows,
        columns=[
            "target_id",
            "required_session_date",
            "fallback_target_id",
            "fallback_source_provider",
            "fallback_source_sha256",
            "fallback_open_unadjusted",
            "fallback_close_unadjusted",
        ],
    )
    if benchmark_fallback.duplicated(["target_id", "required_session_date"]).any():
        raise ValidationPriceReconciliationError("Benchmark fallback consumers are duplicated")
    benchmark = benchmark_alignment.merge(
        benchmark_fallback,
        on=["target_id", "required_session_date"],
        how="left",
        validate="one_to_one",
    )
    benchmark["fallback_benchmark_price_ready"] = benchmark["fallback_target_id"].notna()
    benchmark["primary_open_unadjusted"] = benchmark["open_unadjusted"]
    benchmark["primary_close_unadjusted"] = benchmark["close_unadjusted"]
    benchmark["open_unadjusted"] = benchmark["open_unadjusted"].combine_first(
        benchmark["fallback_open_unadjusted"]
    )
    benchmark["close_unadjusted"] = benchmark["close_unadjusted"].combine_first(
        benchmark["fallback_close_unadjusted"]
    )
    benchmark["reconciled_benchmark_price_ready"] = (
        benchmark["benchmark_price_ready"].astype(bool)
        | benchmark["fallback_benchmark_price_ready"]
    )
    benchmark["reconciled_price_source_tier"] = "unresolved"
    benchmark.loc[
        benchmark["benchmark_price_ready"].astype(bool),
        "reconciled_price_source_tier",
    ] = "primary_massive"
    benchmark.loc[
        ~benchmark["benchmark_price_ready"].astype(bool)
        & benchmark["reconciled_benchmark_price_ready"],
        "reconciled_price_source_tier",
    ] = "secondary_yahoo"
    benchmark["returns_computed"] = False
    return events, benchmark


def build_validation_price_reconciliation(
    exception_root: Path,
    benchmark_audit_root: Path,
    fallback_audit_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_price_reconciliation"),
) -> ValidationPriceReconciliationResult:
    """Persist reconciled readiness while leaving return values closed."""
    exception_manifest = _sha256_file(exception_root / "output_manifest.json")
    benchmark_manifest = _sha256_file(benchmark_audit_root / "output_manifest.json")
    fallback_manifest = _sha256_file(fallback_audit_root / "output_manifest.json")
    fallback_report = _read_json(fallback_audit_root / "validation_price_fallback_report.json")
    if bool(fallback_report.get("outcome_values_loaded")) or bool(
        fallback_report.get("returns_computed")
    ):
        raise ValidationPriceReconciliationError("Fallback price audit opened outcomes")
    if int(fallback_report.get("fallback_prices_ready", 0)) < 1:
        raise ValidationPriceReconciliationError("Fallback price audit has no usable evidence")
    events, benchmark = calculate_validation_price_reconciliation(
        load_catalog_table(exception_root, "validation_event_execution_status.parquet"),
        load_catalog_table(fallback_audit_root, "validation_price_fallback_targets.parquet"),
        load_catalog_table(fallback_audit_root, "validation_price_fallback_observations.parquet"),
        load_catalog_table(benchmark_audit_root, "benchmark_event_window_alignment.parquet"),
    )
    event_ready = int(events["reconciled_formal_price_ready"].sum())
    benchmark_ready = int(benchmark["reconciled_benchmark_price_ready"].sum())
    payload = json.dumps(
        {
            "version": VALIDATION_PRICE_RECONCILIATION_VERSION,
            "exception_manifest_sha256": exception_manifest,
            "benchmark_manifest_sha256": benchmark_manifest,
            "fallback_manifest_sha256": fallback_manifest,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_PRICE_RECONCILIATION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPriceReconciliationResult(
            target.resolve(), _read_json(target / "validation_price_reconciliation_report.json")
        )
    if target.exists():
        raise ValidationPriceReconciliationError(
            f"Incomplete price reconciliation exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "reconciled_event_prices",
                event_ready,
                len(events),
                event_ready == len(events),
                "Primary or frozen secondary evidence provides both formal issuer endpoints",
            ),
            _gate(
                "reconciled_benchmark_prices",
                benchmark_ready,
                len(benchmark),
                benchmark_ready == len(benchmark),
                "Primary or frozen secondary evidence provides every benchmark date",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(events),
                False,
                "Reconciliation does not calculate returns",
            ),
        ]
    )
    outputs = {
        "validation_reconciled_event_price_status.parquet": events,
        "validation_reconciled_benchmark_price_status.parquet": benchmark,
        "validation_price_reconciliation_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception_manifest_sha256": exception_manifest,
        "benchmark_manifest_sha256": benchmark_manifest,
        "fallback_manifest_sha256": fallback_manifest,
        "event_components": len(events),
        "event_components_price_ready": event_ready,
        "event_components_price_pending": len(events) - event_ready,
        "event_price_source_tier_counts": events["reconciled_price_source_tier"]
        .value_counts()
        .to_dict(),
        "event_price_status_counts": events["reconciled_formal_price_status"]
        .value_counts()
        .to_dict(),
        "benchmark_required_dates": len(benchmark),
        "benchmark_required_dates_ready": benchmark_ready,
        "benchmark_price_gate": benchmark_ready == len(benchmark),
        "prices_imputed": 0,
        "events_removed": 0,
        "replacement_securities_used": 0,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_price_reconciliation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "exception_manifest_sha256": exception_manifest,
                "benchmark_manifest_sha256": benchmark_manifest,
                "fallback_manifest_sha256": fallback_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPriceReconciliationResult(target.resolve(), report)


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
        raise ValidationPriceReconciliationError(f"{label} lacks required columns: {missing}")


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
