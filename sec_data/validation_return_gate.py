"""Outcome-blind authorization gate for the liquidity-eligible return cohort."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

VALIDATION_RETURN_GATE_VERSION: Final[str] = "validation-return-gate-v1-complete-case"


class ValidationReturnGateError(FilingCatalogError):
    """Raised when return authorization would violate the frozen design."""


@dataclass(frozen=True)
class ValidationReturnGateResult:
    """Immutable authorization cohort and exclusion ledger."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_return_authorization(
    liquidity: pd.DataFrame,
    outcomes: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Authorize only pre-event eligible cases with complete market evidence."""
    _require_columns(
        liquidity,
        {
            "security_event_component_id",
            "market_event_id",
            "validation_event_id",
            "cik",
            "accepted_at",
            "ticker",
            "liquidity_eligible",
            "liquidity_status",
            "pre_event_liquidity_evidence_complete",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "returns_computed",
        },
        "liquidity audit",
    )
    _require_columns(
        outcomes,
        {
            "security_event_component_id",
            "resolved_market_outcome_ready",
            "resolved_outcome_status",
            "returns_computed",
        },
        "resolved outcomes",
    )
    _require_columns(
        benchmark,
        {
            "target_id",
            "required_session_date",
            "reconciled_benchmark_price_ready",
            "returns_computed",
        },
        "benchmark readiness",
    )
    if liquidity["security_event_component_id"].duplicated().any():
        raise ValidationReturnGateError("Liquidity event components are duplicated")
    if outcomes["security_event_component_id"].duplicated().any():
        raise ValidationReturnGateError("Resolved market outcomes are duplicated")
    if liquidity["selection_uses_return_values"].astype(bool).any():
        raise ValidationReturnGateError("Liquidity selection used return values")
    if liquidity["outcome_values_loaded"].astype(bool).any():
        raise ValidationReturnGateError("Liquidity selection opened outcome values")
    if liquidity["returns_computed"].astype(bool).any():
        raise ValidationReturnGateError("Liquidity selection followed return computation")
    if (
        outcomes["returns_computed"].astype(bool).any()
        or benchmark["returns_computed"].astype(bool).any()
    ):
        raise ValidationReturnGateError("A readiness source already computed returns")

    benchmark_summary = (
        benchmark.groupby("target_id", as_index=False)
        .agg(
            benchmark_required_dates=("required_session_date", "size"),
            benchmark_required_dates_ready=("reconciled_benchmark_price_ready", "sum"),
            benchmark_window_ready=("reconciled_benchmark_price_ready", "all"),
        )
        .rename(columns={"target_id": "security_event_component_id"})
    )
    if not benchmark_summary["benchmark_required_dates"].eq(2).all():
        raise ValidationReturnGateError("Each event component must have two benchmark endpoints")
    universe = liquidity.merge(
        outcomes[
            [
                "security_event_component_id",
                "resolved_market_outcome_ready",
                "resolved_outcome_status",
            ]
        ],
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    ).merge(
        benchmark_summary,
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    )
    if universe["resolved_market_outcome_ready"].isna().any():
        raise ValidationReturnGateError("An event component has no market-outcome readiness row")
    if universe["benchmark_window_ready"].isna().any():
        raise ValidationReturnGateError("An event component has no benchmark readiness rows")
    universe["return_gate_status"] = universe.apply(_gate_status, axis=1)
    universe["return_outcome_read_authorized"] = universe["return_gate_status"].eq(
        "authorized_complete_case"
    )
    universe["authorization_uses_return_values"] = False
    universe["outcome_values_loaded"] = False
    universe["returns_computed"] = False
    authorized = universe.loc[universe["return_outcome_read_authorized"]].copy()
    excluded = universe.loc[~universe["return_outcome_read_authorized"]].copy()
    if authorized.empty:
        raise ValidationReturnGateError("No complete case is eligible for return authorization")
    if not authorized["resolved_market_outcome_ready"].astype(bool).all():
        raise ValidationReturnGateError("An authorized case lacks a resolved market outcome")
    if not authorized["benchmark_window_ready"].astype(bool).all():
        raise ValidationReturnGateError("An authorized case lacks its benchmark window")
    return (
        universe.sort_values(
            ["accepted_at", "security_event_component_id"], kind="mergesort"
        ).reset_index(drop=True),
        authorized.sort_values(
            ["accepted_at", "security_event_component_id"], kind="mergesort"
        ).reset_index(drop=True),
        excluded.sort_values(
            ["accepted_at", "security_event_component_id"], kind="mergesort"
        ).reset_index(drop=True),
    )


def build_validation_return_gate(
    liquidity_audit_root: Path,
    terminal_evidence_root: Path,
    price_reconciliation_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_return_gate"),
) -> ValidationReturnGateResult:
    """Persist authorization metadata without loading any price or return values."""
    liquidity_manifest = _sha256_file(liquidity_audit_root / "output_manifest.json")
    terminal_manifest = _sha256_file(terminal_evidence_root / "output_manifest.json")
    reconciliation_manifest = _sha256_file(price_reconciliation_root / "output_manifest.json")
    liquidity_report = _read_json(liquidity_audit_root / "validation_liquidity_report.json")
    terminal_report = _read_json(
        terminal_evidence_root / "validation_terminal_evidence_report.json"
    )
    reconciliation_report = _read_json(
        price_reconciliation_root / "validation_price_reconciliation_report.json"
    )
    if not bool(liquidity_report.get("complete_case_liquidity_gate")):
        raise ValidationReturnGateError("The complete-case liquidity gate is closed")
    for label, readiness_report in (
        ("liquidity", liquidity_report),
        ("terminal", terminal_report),
        ("reconciliation", reconciliation_report),
    ):
        if bool(readiness_report.get("returns_computed")):
            raise ValidationReturnGateError(f"The {label} readiness phase computed returns")
    liquidity_columns = [
        "security_event_component_id",
        "market_event_id",
        "validation_event_id",
        "cik",
        "accepted_at",
        "ticker",
        "liquidity_eligible",
        "liquidity_status",
        "pre_event_liquidity_evidence_complete",
        "selection_uses_return_values",
        "outcome_values_loaded",
        "returns_computed",
    ]
    outcome_columns = [
        "security_event_component_id",
        "resolved_market_outcome_ready",
        "resolved_outcome_status",
        "returns_computed",
    ]
    benchmark_columns = [
        "target_id",
        "required_session_date",
        "reconciled_benchmark_price_ready",
        "returns_computed",
    ]
    liquidity = pd.read_parquet(
        liquidity_audit_root / "validation_liquidity_event_audit.parquet",
        columns=liquidity_columns,
    )
    outcomes = pd.read_parquet(
        terminal_evidence_root / "validation_resolved_event_outcomes.parquet",
        columns=outcome_columns,
    )
    benchmark = pd.read_parquet(
        price_reconciliation_root / "validation_reconciled_benchmark_price_status.parquet",
        columns=benchmark_columns,
    )
    universe, authorized, excluded = calculate_validation_return_authorization(
        liquidity, outcomes, benchmark
    )
    authorized_count = len(authorized)
    authorization_ready = (
        authorized_count > 0
        and authorized["resolved_market_outcome_ready"].astype(bool).all()
        and authorized["benchmark_window_ready"].astype(bool).all()
    )
    payload = json.dumps(
        {
            "version": VALIDATION_RETURN_GATE_VERSION,
            "liquidity_manifest_sha256": liquidity_manifest,
            "terminal_manifest_sha256": terminal_manifest,
            "reconciliation_manifest_sha256": reconciliation_manifest,
            "authorized_ids": authorized["security_event_component_id"].astype(str).tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_RETURN_GATE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationReturnGateResult(
            target.resolve(), _read_json(target / "validation_return_gate_report.json")
        )
    if target.exists():
        raise ValidationReturnGateError(f"Incomplete validation return gate exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "outcome_blind_liquidity_selection",
                authorized_count,
                1,
                authorized_count > 0,
                "The cohort was selected only from strictly pre-event liquidity evidence",
            ),
            _gate(
                "authorized_market_outcomes",
                int(authorized["resolved_market_outcome_ready"].sum()),
                authorized_count,
                bool(authorized["resolved_market_outcome_ready"].all()),
                "Every authorized event has a price or documented terminal outcome",
            ),
            _gate(
                "authorized_benchmark_windows",
                int(authorized["benchmark_window_ready"].sum()),
                authorized_count,
                bool(authorized["benchmark_window_ready"].all()),
                "Every authorized event has both benchmark endpoints",
            ),
            _gate(
                "return_outcome_read",
                authorized_count if authorization_ready else 0,
                authorized_count,
                bool(authorization_ready),
                "Only listed complete-case component IDs may now load price outcomes",
            ),
        ]
    )
    tables = {
        "validation_return_gate_universe.parquet": universe,
        "validation_authorized_return_cohort.parquet": authorized,
        "validation_return_gate_exclusions.parquet": excluded,
        "validation_return_gate_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "liquidity_manifest_sha256": liquidity_manifest,
        "terminal_manifest_sha256": terminal_manifest,
        "price_reconciliation_manifest_sha256": reconciliation_manifest,
        "universe_event_components": len(universe),
        "authorized_event_components": authorized_count,
        "excluded_event_components": len(excluded),
        "authorization_status_counts": {
            str(key): int(value)
            for key, value in universe["return_gate_status"].value_counts().items()
        },
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": bool(authorization_ready),
        "authorization_scope": "validation_authorized_return_cohort.parquet only",
        "artifacts": artifacts,
    }
    report_path = staging / "validation_return_gate_report.json"
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
    return ValidationReturnGateResult(target.resolve(), report)


def _gate_status(row: pd.Series[Any]) -> str:
    if not bool(row["pre_event_liquidity_evidence_complete"]):
        return "excluded_liquidity_evidence_incomplete"
    if not bool(row["liquidity_eligible"]):
        return "excluded_pre_event_liquidity"
    if not bool(row["resolved_market_outcome_ready"]):
        return "blocked_market_outcome"
    if not bool(row["benchmark_window_ready"]):
        return "blocked_benchmark_window"
    return "authorized_complete_case"


def _gate(name: str, observed: int, required: int, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "gate": name,
        "observed": observed,
        "required": required,
        "passed": passed,
        "detail": detail,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationReturnGateError(f"{label} lacks columns: {missing}")


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
