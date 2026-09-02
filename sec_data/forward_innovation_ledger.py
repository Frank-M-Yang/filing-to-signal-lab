"""Strictly point-in-time Level, Revision Delta, and revenue Innovation ledger."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

FORWARD_INNOVATION_LEDGER_VERSION: Final[str] = "forward-innovation-ledger-v1"


class ForwardInnovationLedgerError(FilingCatalogError):
    """Raised when the forward Innovation clock or identity is invalid."""


@dataclass(frozen=True)
class ForwardInnovationLedgerConfig:
    """Frozen feature-readiness thresholds for the forward study."""

    minimum_raw_innovation_events: int = 10
    minimum_revision_events: int = 10
    minimum_prior_errors_for_sigma: int = 10
    version: str = FORWARD_INNOVATION_LEDGER_VERSION

    def validate(self) -> None:
        """Reject thresholds too small for a useful feature comparison."""
        if self.minimum_raw_innovation_events < 2:
            raise ValueError("Raw Innovation research requires at least two events")
        if self.minimum_revision_events < 2:
            raise ValueError("Revision research requires at least two events")
        if self.minimum_prior_errors_for_sigma < 2:
            raise ValueError("Innovation sigma requires at least two prior errors")


@dataclass(frozen=True)
class ForwardInnovationLedgerTables:
    """Event ledger, complete cases, gates, and summary report."""

    events: pd.DataFrame
    complete_cases: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ForwardInnovationLedgerResult:
    """Immutable persisted ledger result."""

    output_root: Path
    report: dict[str, Any]


def calculate_forward_innovation_ledger(
    consensus_snapshots: pd.DataFrame,
    actual_observations: pd.DataFrame,
    config: ForwardInnovationLedgerConfig | None = None,
) -> ForwardInnovationLedgerTables:
    """Join actuals to the last consensus strictly before SEC acceptance."""
    cfg = config or ForwardInnovationLedgerConfig()
    cfg.validate()
    _require_columns(
        consensus_snapshots,
        {
            "ticker",
            "target_period_end",
            "captured_at",
            "consensus_revenue_level",
            "source_sha256",
            "historical_backtest_admission",
        },
        "forward consensus snapshots",
    )
    _require_columns(
        actual_observations,
        {
            "forward_actual_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "accepted_at",
            "actual_available_at",
            "actual_status",
            "actual_revenue",
            "source_snapshot_sha256",
            "historical_backfill_admission",
            "return_values_loaded",
        },
        "forward actual observations",
    )
    consensus = consensus_snapshots.copy()
    if not consensus.empty:
        consensus["ticker"] = consensus["ticker"].astype(str).str.strip().str.upper()
        consensus["target_period_end"] = _utc(
            consensus["target_period_end"], "consensus target_period_end"
        ).dt.normalize()
        consensus["captured_at"] = _utc(consensus["captured_at"], "captured_at")
        consensus["capture_date"] = consensus["captured_at"].dt.normalize()
        consensus["consensus_revenue_level"] = pd.to_numeric(
            consensus["consensus_revenue_level"], errors="coerce"
        )
        if (
            consensus["historical_backtest_admission"].astype(bool).any()
            or consensus["consensus_revenue_level"].isna().any()
            or consensus["consensus_revenue_level"].le(0.0).any()
        ):
            raise ForwardInnovationLedgerError("Consensus inputs violate forward-only quality")
    actuals = actual_observations.copy()
    if actuals.empty:
        events = pd.DataFrame(columns=_event_columns())
    else:
        actuals["ticker"] = actuals["ticker"].astype(str).str.strip().str.upper()
        actuals["target_period_end"] = _utc(
            actuals["target_period_end"], "actual target_period_end"
        ).dt.normalize()
        actuals["accepted_at"] = _utc(actuals["accepted_at"], "accepted_at")
        actuals["actual_available_at"] = pd.to_datetime(
            actuals["actual_available_at"], utc=True, errors="coerce"
        )
        if actuals["historical_backfill_admission"].astype(bool).any():
            raise ForwardInnovationLedgerError("Historical actual backfill is forbidden")
        if actuals["return_values_loaded"].astype(bool).any():
            raise ForwardInnovationLedgerError("Return values cannot enter feature construction")
        sorted_records = actuals.sort_values(
            ["accepted_at", "ticker", "accession_number"], kind="mergesort"
        ).to_dict("records")
        rows = [
            _build_event_row(cast(dict[str, Any], actual), consensus) for actual in sorted_records
        ]
        events = pd.DataFrame(rows, columns=_event_columns())
        events = _attach_strictly_prior_sigma(events, cfg.minimum_prior_errors_for_sigma)
    complete_cases = events.loc[
        events["level_ready"].astype(bool)
        & events["revision_delta_ready"].astype(bool)
        & events["raw_innovation_ready"].astype(bool)
    ].copy()
    raw_count = int(events["raw_innovation_ready"].astype(bool).sum())
    revision_count = int(events["revision_delta_ready"].astype(bool).sum())
    standardized_count = int(events["standardized_innovation_ready"].astype(bool).sum())
    strict_clock = bool(
        raw_count > 0
        and events.loc[
            events["raw_innovation_ready"].astype(bool), "consensus_precedes_actual"
        ].all()
    )
    gates = pd.DataFrame(
        [
            _gate(
                "minimum_raw_innovation_events",
                raw_count,
                cfg.minimum_raw_innovation_events,
                raw_count >= cfg.minimum_raw_innovation_events,
            ),
            _gate(
                "minimum_revision_events",
                revision_count,
                cfg.minimum_revision_events,
                revision_count >= cfg.minimum_revision_events,
            ),
            _gate("strict_consensus_before_acceptance", raw_count, 1, strict_clock),
            _gate("historical_backfill_rows", 0, 0, True),
            _gate("return_values_loaded", 0, 0, True),
        ]
    )
    feature_gate = bool(gates["passed"].all())
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "actual_observations": len(actuals),
        "event_rows": len(events),
        "event_issuers": int(events["ticker"].nunique()),
        "level_ready_events": int(events["level_ready"].astype(bool).sum()),
        "revision_delta_ready_events": revision_count,
        "raw_innovation_ready_events": raw_count,
        "standardized_innovation_ready_events": standardized_count,
        "complete_case_events": len(complete_cases),
        "feature_research_gate_passed": feature_gate,
        "feature_research_blocking_reason": (
            None if feature_gate else "insufficient_forward_actual_or_revision_depth"
        ),
        "return_research_gate_passed": False,
        "return_research_blocking_reason": "forward_event_returns_not_loaded",
        "historical_backfill_rows": 0,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_fitting_performed": False,
        "config": asdict(cfg),
    }
    return ForwardInnovationLedgerTables(events, complete_cases, gates, report)


def build_forward_innovation_ledger(
    consensus_root: Path,
    actual_root: Path,
    *,
    output_root: Path = Path("data/processed/forward_innovation_ledger"),
    config: ForwardInnovationLedgerConfig | None = None,
) -> ForwardInnovationLedgerResult:
    """Persist the forward Level/Revision Delta/Innovation feature ledger."""
    cfg = config or ForwardInnovationLedgerConfig()
    cfg.validate()
    consensus_hash = _sha256_file(consensus_root / "output_manifest.json")
    actual_hash = _sha256_file(actual_root / "output_manifest.json")
    tables = calculate_forward_innovation_ledger(
        pd.read_parquet(consensus_root / "forward_consensus_snapshots.parquet"),
        pd.read_parquet(actual_root / "forward_actual_revenue_observations.parquet"),
        cfg,
    )
    payload = json.dumps(
        {
            "consensus_manifest_sha256": consensus_hash,
            "actual_manifest_sha256": actual_hash,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ForwardInnovationLedgerResult(
            target.resolve(), _read_json(target / "forward_innovation_ledger_report.json")
        )
    if target.exists():
        raise ForwardInnovationLedgerError(f"Incomplete Innovation ledger exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "forward_innovation_events.parquet": tables.events,
        "forward_innovation_complete_cases.parquet": tables.complete_cases,
        "forward_innovation_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "consensus_manifest_sha256": consensus_hash,
        "actual_manifest_sha256": actual_hash,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_innovation_ledger_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": {
                    "forward_consensus": consensus_hash,
                    "forward_actual_revenue": actual_hash,
                },
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ForwardInnovationLedgerResult(target.resolve(), report)


def _build_event_row(actual: dict[str, Any], consensus: pd.DataFrame) -> dict[str, Any]:
    accepted_at = _timestamp(actual["accepted_at"], "accepted_at")
    available_at = pd.to_datetime(actual["actual_available_at"], utc=True, errors="coerce")
    actual_status = str(actual["actual_status"])
    actual_ready = actual_status in {
        "direct_quarterly_fact_ready",
        "derived_q4_fy_minus_prior_9m_ready",
    }
    if actual_ready and (pd.isna(available_at) or pd.Timestamp(available_at) != accepted_at):
        raise ForwardInnovationLedgerError("Actual availability must equal SEC acceptance")
    ticker = str(actual["ticker"]).strip().upper()
    period_end = _timestamp(actual["target_period_end"], "target_period_end").normalize()
    eligible = consensus.loc[
        consensus["ticker"].astype(str).eq(ticker)
        & consensus["target_period_end"].eq(period_end)
        & consensus["captured_at"].lt(accepted_at)
    ].copy()
    selected: pd.Series | None = None
    previous: pd.Series | None = None
    if not eligible.empty:
        eligible = eligible.sort_values("captured_at", kind="mergesort")
        selected = eligible.iloc[-1]
        prior_dates = eligible.loc[eligible["capture_date"].lt(selected["capture_date"])].copy()
        if not prior_dates.empty:
            previous = prior_dates.iloc[-1]
    consensus_level = float(selected["consensus_revenue_level"]) if selected is not None else None
    actual_revenue = (
        float(actual["actual_revenue"])
        if actual_ready and pd.notna(actual["actual_revenue"])
        else None
    )
    raw_ready = actual_revenue is not None and consensus_level is not None
    if actual_revenue is not None and consensus_level is not None:
        innovation_value = actual_revenue - consensus_level
        innovation: float | None = innovation_value
        innovation_pct: float | None = innovation_value / consensus_level
    else:
        innovation = None
        innovation_pct = None
    previous_level = float(previous["consensus_revenue_level"]) if previous is not None else None
    revision_delta = (
        consensus_level - previous_level
        if consensus_level is not None and previous_level is not None
        else None
    )
    revision_pct = (
        revision_delta / previous_level
        if revision_delta is not None and previous_level is not None and previous_level != 0.0
        else None
    )
    return {
        "forward_actual_target_id": str(actual["forward_actual_target_id"]),
        "cik": str(actual["cik"]),
        "ticker": ticker,
        "target_period_end": period_end,
        "accession_number": str(actual["accession_number"]),
        "sec_accepted_at": accepted_at,
        "actual_available_at": available_at,
        "actual_status": actual_status,
        "actual_revenue": actual_revenue,
        "consensus_captured_at": selected["captured_at"] if selected is not None else None,
        "consensus_capture_date": selected["capture_date"] if selected is not None else None,
        "consensus_revenue_level": consensus_level,
        "consensus_log_level": math.log(consensus_level) if consensus_level else None,
        "previous_consensus_captured_at": (
            previous["captured_at"] if previous is not None else None
        ),
        "previous_consensus_revenue": previous_level,
        "consensus_revision_delta": revision_delta,
        "consensus_revision_pct": revision_pct,
        "revenue_innovation": innovation,
        "revenue_innovation_pct": innovation_pct,
        "prior_innovation_error_count": 0,
        "prior_innovation_error_sigma": None,
        "standardized_revenue_innovation": None,
        "level_ready": consensus_level is not None,
        "revision_delta_ready": revision_delta is not None,
        "raw_innovation_ready": raw_ready,
        "standardized_innovation_ready": False,
        "consensus_precedes_actual": bool(
            selected is not None and selected["captured_at"] < accepted_at
        ),
        "consensus_source_sha256": (
            str(selected["source_sha256"]) if selected is not None else None
        ),
        "actual_source_sha256": actual["source_snapshot_sha256"],
        "join_status": (
            "raw_innovation_ready"
            if raw_ready
            else "actual_not_ready"
            if not actual_ready
            else "no_pre_acceptance_consensus"
        ),
        "historical_backfill_admission": False,
        "return_values_loaded": False,
    }


def _attach_strictly_prior_sigma(events: pd.DataFrame, minimum_prior: int) -> pd.DataFrame:
    if events.empty:
        return events
    output = events.sort_values(
        ["sec_accepted_at", "ticker", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)
    history: list[float] = []
    for accepted_at, indexes in output.groupby("sec_accepted_at", sort=True).groups.items():
        del accepted_at
        finite_history = [value for value in history if math.isfinite(value)]
        sigma = (
            float(np.std(finite_history, ddof=1)) if len(finite_history) >= minimum_prior else None
        )
        for index in indexes:
            output.at[index, "prior_innovation_error_count"] = len(finite_history)
            innovation_pct: Any = output.at[index, "revenue_innovation_pct"]
            if (
                sigma is not None
                and sigma > 0.0
                and pd.notna(innovation_pct)
                and math.isfinite(float(innovation_pct))
            ):
                output.at[index, "prior_innovation_error_sigma"] = sigma
                output.at[index, "standardized_revenue_innovation"] = float(innovation_pct) / sigma
                output.at[index, "standardized_innovation_ready"] = True
        for index in indexes:
            value: Any = output.at[index, "revenue_innovation_pct"]
            if pd.notna(value) and math.isfinite(float(value)):
                history.append(float(value))
    return output


def _event_columns() -> list[str]:
    return [
        "forward_actual_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "accession_number",
        "sec_accepted_at",
        "actual_available_at",
        "actual_status",
        "actual_revenue",
        "consensus_captured_at",
        "consensus_capture_date",
        "consensus_revenue_level",
        "consensus_log_level",
        "previous_consensus_captured_at",
        "previous_consensus_revenue",
        "consensus_revision_delta",
        "consensus_revision_pct",
        "revenue_innovation",
        "revenue_innovation_pct",
        "prior_innovation_error_count",
        "prior_innovation_error_sigma",
        "standardized_revenue_innovation",
        "level_ready",
        "revision_delta_ready",
        "raw_innovation_ready",
        "standardized_innovation_ready",
        "consensus_precedes_actual",
        "consensus_source_sha256",
        "actual_source_sha256",
        "join_status",
        "historical_backfill_admission",
        "return_values_loaded",
    ]


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ForwardInnovationLedgerError(f"Invalid {label}")
    return pd.Timestamp(result)


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ForwardInnovationLedgerError(f"Invalid {label}")
    return parsed


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForwardInnovationLedgerError(f"{label} lacks required columns: {missing}")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
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
