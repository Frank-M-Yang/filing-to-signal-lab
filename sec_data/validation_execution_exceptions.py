"""Classify formal execution-price gaps without imputing or dropping events."""

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

VALIDATION_EXECUTION_EXCEPTION_VERSION: Final[str] = (
    "validation-execution-exceptions-v1-no-imputation"
)


class ValidationExecutionExceptionError(FilingCatalogError):
    """Raised when price exceptions cannot be classified without outcomes."""


@dataclass(frozen=True)
class ValidationExecutionExceptionResult:
    """Immutable event-level price exception ledger."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_execution_exceptions(
    endpoints: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Classify missing formal dates from observed support, never price returns."""
    _require_columns(
        endpoints,
        {
            "security_event_component_id",
            "market_event_id",
            "validation_event_id",
            "cik",
            "ticker",
            "exchange_mic",
            "price_target_id",
            "formal_entry_session_date",
            "formal_exit_session_date",
            "entry_price_date_observed",
            "exit_price_date_observed",
            "required_endpoint_dates_ready",
        },
        "formal price endpoints",
    )
    _require_columns(bars, {"price_target_id", "session_date"}, "validated daily bars")
    if endpoints["security_event_component_id"].duplicated().any():
        raise ValidationExecutionExceptionError("Execution endpoints are duplicated")
    support = (
        bars.assign(session_date_value=pd.to_datetime(bars["session_date"], errors="coerce"))
        .groupby("price_target_id")["session_date_value"]
        .agg(observed_price_start="min", observed_price_end="max", observed_bar_rows="size")
    )
    if support[["observed_price_start", "observed_price_end"]].isna().any().any():
        raise ValidationExecutionExceptionError("Validated bars contain invalid dates")
    result = endpoints.merge(
        support,
        on="price_target_id",
        how="left",
        validate="many_to_one",
    )
    result["formal_entry_date"] = pd.to_datetime(
        result["formal_entry_session_date"], errors="coerce"
    )
    result["formal_exit_date"] = pd.to_datetime(result["formal_exit_session_date"], errors="coerce")
    if result[["formal_entry_date", "formal_exit_date"]].isna().any().any():
        raise ValidationExecutionExceptionError("Formal endpoints contain invalid dates")
    result["entry_exception_code"] = result.apply(
        lambda row: _endpoint_code(row, endpoint="entry"), axis=1
    )
    result["exit_exception_code"] = result.apply(
        lambda row: _endpoint_code(row, endpoint="exit"), axis=1
    )
    result["execution_exception_route"] = result.apply(_exception_route, axis=1)
    result["formal_price_status"] = result.apply(_formal_price_status, axis=1)
    result["event_removed"] = False
    result["price_imputed"] = False
    result["replacement_security_used"] = False
    result["selection_uses_return_values"] = False
    result["outcome_values_loaded"] = False
    result["returns_computed"] = False
    result["observed_price_start"] = result["observed_price_start"].dt.strftime("%Y-%m-%d")
    result["observed_price_end"] = result["observed_price_end"].dt.strftime("%Y-%m-%d")
    return result.drop(columns=["formal_entry_date", "formal_exit_date"])


def build_validation_execution_exception_ledger(
    price_audit_root: Path,
    execution_plan_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_execution_exceptions"),
) -> ValidationExecutionExceptionResult:
    """Persist complete event status and unresolved exception routes."""
    price_manifest = _sha256_file(price_audit_root / "output_manifest.json")
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    price_report = _read_json(price_audit_root / "validation_price_batch_report.json")
    if price_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationExecutionExceptionError(
            "Price audit does not tie to the selected execution plan"
        )
    endpoints = load_catalog_table(
        price_audit_root, "validation_price_batch_event_endpoints.parquet"
    )
    bars = load_catalog_table(price_audit_root, "validation_price_batch_bars.parquet")
    statuses = calculate_validation_execution_exceptions(endpoints, bars)
    exceptions = statuses.loc[~statuses["required_endpoint_dates_ready"].astype(bool)].copy()
    if len(statuses) != int(price_report.get("event_component_endpoints", -1)):
        raise ValidationExecutionExceptionError("Exception ledger lost execution components")
    payload = json.dumps(
        {
            "version": VALIDATION_EXECUTION_EXCEPTION_VERSION,
            "price_manifest_sha256": price_manifest,
            "execution_manifest_sha256": execution_manifest,
            "status_counts": statuses["formal_price_status"].value_counts().to_dict(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_EXECUTION_EXCEPTION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationExecutionExceptionResult(
            target.resolve(), _read_json(target / "validation_execution_exception_report.json")
        )
    if target.exists():
        raise ValidationExecutionExceptionError(
            f"Incomplete execution exception ledger exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    ready = int(statuses["required_endpoint_dates_ready"].astype(bool).sum())
    non_executable = int(
        statuses["formal_price_status"]
        .isin(["non_executable_entry_under_frozen_rule", "non_executable_exit_under_frozen_rule"])
        .sum()
    )
    unresolved = len(statuses) - ready - non_executable
    gates = pd.DataFrame(
        [
            _gate(
                "price_ready_components",
                ready,
                len(statuses),
                ready == len(statuses),
                "Formal entry and exit dates are both observed",
            ),
            _gate(
                "exceptions_explicitly_routed",
                len(exceptions),
                len(exceptions),
                bool(not exceptions.empty and exceptions["execution_exception_route"].ne("").all()),
                "Every missing endpoint is assigned a non-imputing evidence route",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(statuses),
                False,
                "No return is computed while execution exceptions remain unresolved",
            ),
        ]
    )
    outputs = {
        "validation_event_execution_status.parquet": statuses,
        "validation_execution_exceptions.parquet": exceptions,
        "validation_execution_exception_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "price_audit_version": price_report["dataset_version"],
        "price_audit_manifest_sha256": price_manifest,
        "execution_plan_manifest_sha256": execution_manifest,
        "event_components": len(statuses),
        "price_ready_components": ready,
        "execution_exceptions": len(exceptions),
        "non_executable_under_frozen_rule": non_executable,
        "unresolved_evidence_routes": unresolved,
        "formal_price_status_counts": statuses["formal_price_status"].value_counts().to_dict(),
        "exception_route_counts": exceptions["execution_exception_route"].value_counts().to_dict(),
        "events_removed": 0,
        "prices_imputed": 0,
        "replacement_securities_used": 0,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_execution_exception_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "price_audit_manifest_sha256": price_manifest,
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
    return ValidationExecutionExceptionResult(target.resolve(), report)


def _endpoint_code(row: pd.Series, *, endpoint: str) -> str:
    observed = bool(row[f"{endpoint}_price_date_observed"])
    if observed:
        return "ready"
    start = row["observed_price_start"]
    end = row["observed_price_end"]
    if pd.isna(start) or pd.isna(end):
        return "no_price_history_observed"
    required = row[f"formal_{endpoint}_date"]
    if start > required:
        return f"history_begins_after_required_{endpoint}"
    if end < required:
        return f"history_ends_before_required_{endpoint}"
    return f"no_print_on_required_{endpoint}_session"


def _exception_route(row: pd.Series) -> str:
    entry = str(row["entry_exception_code"])
    exit_code = str(row["exit_exception_code"])
    if entry == "ready" and exit_code == "ready":
        return "none"
    if entry == "history_begins_after_required_entry":
        return "alternate_history_or_listing_start_evidence_required"
    if entry == "no_print_on_required_entry_session":
        return "non_executable_entry_under_frozen_rule"
    if entry != "ready":
        return "entry_identity_or_history_evidence_required"
    if exit_code == "history_ends_before_required_exit":
        return "delisting_or_sparse_tail_evidence_required"
    if exit_code == "no_print_on_required_exit_session":
        return "non_executable_exit_under_frozen_rule"
    return "exit_identity_or_history_evidence_required"


def _formal_price_status(row: pd.Series) -> str:
    route = str(row["execution_exception_route"])
    mapping = {
        "none": "price_ready",
        "alternate_history_or_listing_start_evidence_required": (
            "unresolved_external_history_or_listing_start"
        ),
        "non_executable_entry_under_frozen_rule": ("non_executable_entry_under_frozen_rule"),
        "non_executable_exit_under_frozen_rule": "non_executable_exit_under_frozen_rule",
        "delisting_or_sparse_tail_evidence_required": ("unresolved_terminal_or_sparse_exit"),
        "entry_identity_or_history_evidence_required": "unresolved_entry_evidence",
        "exit_identity_or_history_evidence_required": "unresolved_exit_evidence",
    }
    return mapping[route]


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
        raise ValidationExecutionExceptionError(f"{label} lacks required columns: {missing}")


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
