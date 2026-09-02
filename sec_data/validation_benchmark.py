"""Formal SPY benchmark plan tied to validation execution sessions."""

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

VALIDATION_BENCHMARK_PLAN_VERSION: Final[str] = "validation-benchmark-plan-v1-spy"


class ValidationBenchmarkError(FilingCatalogError):
    """Raised when the formal SPY benchmark plan violates frozen assumptions."""


@dataclass(frozen=True)
class ValidationBenchmarkPlanResult:
    """Immutable benchmark plan plus compatible official-calendar adapter."""

    output_root: Path
    calendar_root: Path
    report: dict[str, Any]


def calculate_validation_benchmark_inputs(
    benchmark_targets: pd.DataFrame,
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate the frozen execution benchmark into auditable query inputs."""
    _require_columns(
        benchmark_targets,
        {
            "benchmark_target_id",
            "benchmark_role",
            "benchmark_ticker",
            "price_window_start",
            "price_window_end",
            "adjusted",
            "benchmark_collection_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "validation benchmark target",
    )
    _require_columns(
        components,
        {
            "security_event_component_id",
            "formal_entry_session_date",
            "formal_exit_session_date",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "validation execution components",
    )
    if len(benchmark_targets) != 1:
        raise ValidationBenchmarkError("Formal benchmark requires exactly one target")
    target = benchmark_targets.iloc[0]
    if str(target["benchmark_ticker"]) != "SPY" or bool(target["adjusted"]):
        raise ValidationBenchmarkError("Formal benchmark must be unadjusted SPY")
    if not bool(target["benchmark_collection_authorized"]):
        raise ValidationBenchmarkError("Formal benchmark collection is not authorized")
    for frame in (benchmark_targets, components):
        if frame["selection_uses_return_values"].astype(bool).any():
            raise ValidationBenchmarkError("Benchmark construction used return values")
        if frame["outcome_values_loaded"].astype(bool).any():
            raise ValidationBenchmarkError("Benchmark construction opened outcomes")
    query_target = pd.DataFrame(
        [
            {
                "benchmark_target_id": str(target["benchmark_target_id"]),
                "ticker": "SPY",
                "cik": "0000884394",
                "currency": "USD",
                "benchmark_role": str(target["benchmark_role"]),
                "window_start": str(target["price_window_start"]),
                "window_end": str(target["price_window_end"]),
                "adjusted": False,
            }
        ]
    )
    entry = components[["security_event_component_id", "formal_entry_session_date"]].rename(
        columns={
            "security_event_component_id": "target_id",
            "formal_entry_session_date": "required_session_date",
        }
    )
    exit_dates = components[["security_event_component_id", "formal_exit_session_date"]].rename(
        columns={
            "security_event_component_id": "target_id",
            "formal_exit_session_date": "required_session_date",
        }
    )
    required = pd.concat([entry, exit_dates], ignore_index=True).drop_duplicates()
    if required["required_session_date"].isna().any():
        raise ValidationBenchmarkError("A benchmark-required session date is missing")
    return query_target, required


def build_validation_benchmark_plan(
    execution_plan_root: Path,
    *,
    output_root: Path = Path("data/market/validation_benchmark/targets"),
    calendar_output_root: Path = Path("data/processed/validation_calendar_adapter"),
) -> ValidationBenchmarkPlanResult:
    """Persist SPY queries and an exact adapter for official execution sessions."""
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    benchmark_targets = load_catalog_table(
        execution_plan_root, "validation_benchmark_price_targets.parquet"
    )
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    sessions = load_catalog_table(execution_plan_root, "validation_exchange_sessions.parquet")
    query_target, required = calculate_validation_benchmark_inputs(benchmark_targets, components)
    calendar = _build_calendar_adapter(
        sessions,
        execution_manifest,
        execution_report,
        output_root=calendar_output_root,
    )
    calendar_manifest = _sha256_file(calendar / "output_manifest.json")
    payload = json.dumps(
        {
            "version": VALIDATION_BENCHMARK_PLAN_VERSION,
            "execution_manifest_sha256": execution_manifest,
            "calendar_manifest_sha256": calendar_manifest,
            "target": query_target.to_dict(orient="records"),
            "required_dates": required.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_BENCHMARK_PLAN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationBenchmarkPlanResult(
            target.resolve(), calendar.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationBenchmarkError(f"Incomplete validation benchmark plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_benchmark_target.parquet": query_target,
        "benchmark_required_session_dates.parquet": required,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "calendar_version": _read_json(calendar / "exchange_calendar_report.json")[
            "dataset_version"
        ],
        "calendar_manifest_sha256": calendar_manifest,
        "benchmark_target_id": query_target["benchmark_target_id"].iloc[0],
        "benchmark_ticker": "SPY",
        "window_start": query_target["window_start"].iloc[0],
        "window_end": query_target["window_end"].iloc[0],
        "required_event_dates": len(required),
        "queries": ["daily_bars", "splits", "dividends"],
        "query_adjusted": False,
        "benchmark_identity_method": "explicit_versioned_instrument_specification",
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
                "calendar_manifest_sha256": calendar_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationBenchmarkPlanResult(target.resolve(), calendar.resolve(), report)


def _build_calendar_adapter(
    sessions: pd.DataFrame,
    execution_manifest: str,
    execution_report: dict[str, Any],
    *,
    output_root: Path,
) -> Path:
    payload = json.dumps(
        {
            "execution_manifest_sha256": execution_manifest,
            "session_rows": len(sessions),
            "session_hash": int(pd.util.hash_pandas_object(sessions, index=False).sum()),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "validation-calendar-adapter-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return target.resolve()
    if target.exists():
        raise ValidationBenchmarkError(f"Incomplete calendar adapter exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "historical_exchange_sessions.parquet"
    sessions.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(sessions))}
    report = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "calendar_start": execution_report["calendar_start"],
        "calendar_end": execution_report["calendar_end"],
        "artifacts": artifacts,
    }
    report_path = staging / "exchange_calendar_report.json"
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
    return target.resolve()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationBenchmarkError(f"{label} lacks required columns: {missing}")


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
