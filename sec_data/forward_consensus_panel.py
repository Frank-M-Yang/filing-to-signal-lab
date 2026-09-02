"""Forward-only analyst-consensus Level, Revision Delta, and Innovation readiness."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

FORWARD_CONSENSUS_PANEL_VERSION: Final[str] = "forward-consensus-panel-v1"


class ForwardConsensusPanelError(FilingCatalogError):
    """Raised when a consensus snapshot violates the frozen forward-only contract."""


@dataclass(frozen=True)
class ForwardConsensusPanelConfig:
    """Frozen breadth and time-depth gates for consensus-revision research."""

    minimum_capture_dates: int = 2
    minimum_revision_issuers: int = 10
    version: str = FORWARD_CONSENSUS_PANEL_VERSION

    def validate(self) -> None:
        """Reject settings that cannot establish a cross-date revision."""
        if self.minimum_capture_dates < 2:
            raise ValueError("Consensus revisions require at least two capture dates")
        if self.minimum_revision_issuers < 2:
            raise ValueError("Consensus revision breadth must cover at least two issuers")


@dataclass(frozen=True)
class ForwardConsensusPanelTables:
    """Forward snapshots, current watchlist, revisions, gates, and report."""

    snapshots: pd.DataFrame
    watchlist: pd.DataFrame
    next_events: pd.DataFrame
    revisions: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ForwardConsensusPanelResult:
    """Immutable persisted consensus-panel result."""

    output_root: Path
    report: dict[str, Any]


def calculate_forward_consensus_panel(
    observations: pd.DataFrame,
    coverage: pd.DataFrame,
    config: ForwardConsensusPanelConfig | None = None,
) -> ForwardConsensusPanelTables:
    """Build Level and strictly cross-date Revision Delta without loading returns."""
    cfg = config or ForwardConsensusPanelConfig()
    cfg.validate()
    _require_columns(
        observations,
        {
            "ticker",
            "target_period_end",
            "revenue_estimate_average",
            "revenue_estimate_high",
            "revenue_estimate_low",
            "revenue_estimate_analyst_count",
            "captured_at",
            "availability_clock",
            "forward_only_admission",
            "historical_backtest_admission",
            "forward_event_candidate",
            "source_sha256",
        },
        "consensus observations",
    )
    _require_columns(
        coverage,
        {
            "live_consensus_target_id",
            "cik",
            "ticker",
            "strict_forward_revenue_consensus_observed",
        },
        "consensus coverage",
    )
    if observations.empty or coverage.empty:
        raise ForwardConsensusPanelError("Consensus panel inputs cannot be empty")
    panel = observations.copy()
    panel["ticker"] = panel["ticker"].astype(str).str.strip().str.upper()
    panel["captured_at"] = _utc(panel["captured_at"], "captured_at")
    panel["capture_date"] = panel["captured_at"].dt.normalize()
    panel["target_period_end"] = _utc(panel["target_period_end"], "target_period_end")
    if not panel["availability_clock"].astype(str).eq("collector_fetched_at").all():
        raise ForwardConsensusPanelError("Consensus availability clock is not collector time")
    if not panel["forward_only_admission"].astype(bool).all():
        raise ForwardConsensusPanelError("A consensus row is not forward-only")
    if panel["historical_backtest_admission"].astype(bool).any():
        raise ForwardConsensusPanelError("Historical backfill admission is forbidden")
    recomputed_candidate = panel["target_period_end"].ge(panel["capture_date"])
    provided_candidate = panel["forward_event_candidate"].astype(bool)
    if not recomputed_candidate.eq(provided_candidate).all():
        raise ForwardConsensusPanelError("Forward-period flag does not match the capture clock")
    panel = panel.loc[recomputed_candidate].copy()
    numeric_columns = [
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if (
        panel[numeric_columns].isna().any().any()
        or panel["revenue_estimate_average"].le(0.0).any()
        or panel["revenue_estimate_analyst_count"].lt(1.0).any()
        or panel["revenue_estimate_high"].lt(panel["revenue_estimate_low"]).any()
    ):
        raise ForwardConsensusPanelError("Forward consensus values fail basic quality checks")
    atomic_key = ["ticker", "target_period_end", "captured_at"]
    if panel.duplicated(atomic_key).any():
        raise ForwardConsensusPanelError("Duplicate ticker-period-capture observations exist")
    panel = panel.sort_values(atomic_key, kind="mergesort").reset_index(drop=True)
    panel["consensus_revenue_level"] = panel["revenue_estimate_average"]
    panel["consensus_revenue_log_level"] = np.log(panel["revenue_estimate_average"])
    panel["consensus_range"] = panel["revenue_estimate_high"] - panel["revenue_estimate_low"]
    panel["consensus_range_pct"] = panel["consensus_range"] / panel["revenue_estimate_average"]
    group = panel.groupby(["ticker", "target_period_end"], sort=False)
    panel["previous_captured_at"] = group["captured_at"].shift(1)
    panel["previous_consensus_revenue"] = group["revenue_estimate_average"].shift(1)
    panel["previous_analyst_count"] = group["revenue_estimate_analyst_count"].shift(1)
    panel["consensus_revision_delta"] = (
        panel["revenue_estimate_average"] - panel["previous_consensus_revenue"]
    )
    panel["consensus_revision_pct"] = (
        panel["consensus_revision_delta"] / panel["previous_consensus_revenue"]
    )
    panel["analyst_count_delta"] = (
        panel["revenue_estimate_analyst_count"] - panel["previous_analyst_count"]
    )
    panel["strictly_prior_snapshot"] = panel["previous_captured_at"].lt(panel["captured_at"])
    panel["revision_ready"] = (
        panel["previous_consensus_revenue"].notna() & panel["strictly_prior_snapshot"]
    )
    panel["innovation_status"] = "pending_future_sec_actual"
    panel["actual_revenue"] = np.nan
    panel["revenue_innovation"] = np.nan
    panel["revenue_innovation_pct"] = np.nan
    watchlist = (
        panel.sort_values("captured_at", kind="mergesort")
        .groupby(["ticker", "target_period_end"], as_index=False, sort=True)
        .tail(1)
        .sort_values(["target_period_end", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    watchlist["target_sequence"] = (
        watchlist.groupby("ticker")["target_period_end"].rank(method="first").astype(int)
    )
    next_events = watchlist.loc[watchlist["target_sequence"].eq(1)].copy().reset_index(drop=True)
    target_identity = coverage[["live_consensus_target_id", "cik", "ticker"]].drop_duplicates(
        "ticker"
    )
    next_events = next_events.merge(target_identity, on="ticker", how="left", validate="one_to_one")
    if next_events["cik"].isna().any():
        raise ForwardConsensusPanelError("A next event lacks frozen issuer identity")
    next_events["future_actual_join_rule"] = (
        "same_cik_and_fiscal_period_latest_capture_strictly_before_sec_accepted_at"
    )
    next_events["future_price_join_status"] = "blocked_until_actual_sec_acceptance"
    revisions = panel.loc[panel["revision_ready"]].copy().reset_index(drop=True)
    target_count = int(coverage["ticker"].astype(str).nunique())
    snapshot_issuers = int(panel["ticker"].nunique())
    capture_dates = int(panel["capture_date"].nunique())
    revision_issuers = int(revisions["ticker"].nunique()) if not revisions.empty else 0
    strictly_prior_revision_rows = bool(
        (not revisions.empty) and revisions["strictly_prior_snapshot"].all()
    )
    event_watchlist_gate = bool(
        snapshot_issuers == target_count
        and len(next_events) == target_count
        and coverage["strict_forward_revenue_consensus_observed"].astype(bool).all()
    )
    gate_rows = [
        _gate("forward_issuer_coverage", snapshot_issuers, target_count, event_watchlist_gate),
        _gate(
            "minimum_capture_dates",
            capture_dates,
            cfg.minimum_capture_dates,
            capture_dates >= cfg.minimum_capture_dates,
        ),
        _gate(
            "minimum_revision_issuers",
            revision_issuers,
            cfg.minimum_revision_issuers,
            revision_issuers >= cfg.minimum_revision_issuers,
        ),
        _gate(
            "strictly_prior_revision_rows",
            len(revisions),
            1,
            strictly_prior_revision_rows,
        ),
        _gate("historical_backfill_rows", 0, 0, True),
        _gate("return_values_loaded", 0, 0, True),
    ]
    gates = pd.DataFrame(gate_rows)
    revision_gate = bool(gates["passed"].all())
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "forward_snapshot_rows": len(panel),
        "forward_snapshot_issuers": snapshot_issuers,
        "target_issuers": target_count,
        "capture_dates": capture_dates,
        "watchlist_periods": len(watchlist),
        "next_events": len(next_events),
        "revision_rows": len(revisions),
        "revision_issuers": revision_issuers,
        "event_watchlist_gate_passed": event_watchlist_gate,
        "revision_research_gate_passed": revision_gate,
        "innovation_research_gate_passed": False,
        "innovation_blocking_reason": "future_sec_actual_not_yet_available",
        "historical_backfill_rows": 0,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_fitting_performed": False,
        "config": asdict(cfg),
    }
    return ForwardConsensusPanelTables(panel, watchlist, next_events, revisions, gates, report)


def build_forward_consensus_panel(
    consensus_audit_root: Path,
    *,
    output_root: Path = Path("data/processed/forward_consensus_panel"),
    config: ForwardConsensusPanelConfig | None = None,
) -> ForwardConsensusPanelResult:
    """Persist the forward consensus panel and its explicit readiness gates."""
    cfg = config or ForwardConsensusPanelConfig()
    cfg.validate()
    input_hash = _sha256_file(consensus_audit_root / "output_manifest.json")
    tables = calculate_forward_consensus_panel(
        pd.read_parquet(consensus_audit_root / "alpha_vantage_live_revenue_consensus.parquet"),
        pd.read_parquet(consensus_audit_root / "alpha_vantage_live_coverage.parquet"),
        cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hash, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ForwardConsensusPanelResult(
            target.resolve(), _read_json(target / "forward_consensus_panel_report.json")
        )
    if target.exists():
        raise ForwardConsensusPanelError(f"Incomplete consensus panel exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "forward_consensus_snapshots.parquet": tables.snapshots,
        "forward_consensus_watchlist.parquet": tables.watchlist,
        "forward_consensus_next_events.parquet": tables.next_events,
        "forward_consensus_revisions.parquet": tables.revisions,
        "forward_consensus_gates.parquet": tables.gates,
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
        "input_manifest_sha256": input_hash,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_consensus_panel_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": {"alpha_vantage_consensus": input_hash},
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ForwardConsensusPanelResult(target.resolve(), report)


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {
        "gate": name,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ForwardConsensusPanelError(f"Invalid {label}")
    return parsed


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForwardConsensusPanelError(f"{label} lacks required columns: {missing}")


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
