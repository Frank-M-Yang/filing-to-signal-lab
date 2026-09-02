"""Audit structural contamination in prospective analyst-consensus snapshots.

The layer separates a numeric consensus revision from changes in the provider's
row structure or analyst coverage.  Rules are explicit, outcome-free, and do not
discard evidence: review and quarantined events remain observable but are not
admitted to downstream signal or paper-decision states.
"""

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

PROSPECTIVE_CONSENSUS_QUALITY_VERSION: Final[str] = "prospective-consensus-quality-v1"

CLEAN: Final[str] = "clean"
REVIEW_REQUIRED: Final[str] = "review_required"
QUARANTINED: Final[str] = "quarantined"
INSUFFICIENT_HISTORY: Final[str] = "insufficient_history"


class ProspectiveConsensusQualityError(FilingCatalogError):
    """Raised when quality evidence or an availability clock is invalid."""


@dataclass(frozen=True)
class ProspectiveConsensusQualityConfig:
    """Frozen outcome-independent structural-change thresholds."""

    analyst_count_relative_quarantine_threshold: float = 0.25
    range_width_ratio_review_lower: float = 0.50
    range_width_ratio_review_upper: float = 2.00
    large_consensus_revision_abs_threshold: float = 0.10
    stale_preacceptance_calendar_days: int = 7

    def validate(self) -> None:
        """Prevent silent relaxation or outcome-driven retuning of v1 rules."""
        frozen = (
            self.analyst_count_relative_quarantine_threshold,
            self.range_width_ratio_review_lower,
            self.range_width_ratio_review_upper,
            self.large_consensus_revision_abs_threshold,
            self.stale_preacceptance_calendar_days,
        )
        if frozen != (0.25, 0.50, 2.00, 0.10, 7):
            raise ValueError("The prospective consensus quality thresholds are frozen")


@dataclass(frozen=True)
class ProspectiveConsensusQualityTables:
    """Event assessments, consecutive-change evidence, rules, gates, and report."""

    event_quality: pd.DataFrame
    change_evidence: pd.DataFrame
    rule_catalog: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveConsensusQualityResult:
    """One immutable quality audit over the frozen prospective registry."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_consensus_quality(
    event_readiness: pd.DataFrame,
    consensus_snapshots: pd.DataFrame,
    config: ProspectiveConsensusQualityConfig | None = None,
) -> ProspectiveConsensusQualityTables:
    """Classify structural quality without reading actuals, prices, or returns."""
    cfg = config or ProspectiveConsensusQualityConfig()
    cfg.validate()
    _require_columns(
        event_readiness,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "accepted_at",
            "future_filing_observed",
        },
        "prospective event readiness",
    )
    _require_columns(
        consensus_snapshots,
        {
            "prospective_event_id",
            "ticker",
            "target_period_end",
            "horizon",
            "revenue_estimate_average",
            "revenue_estimate_high",
            "revenue_estimate_low",
            "revenue_estimate_analyst_count",
            "captured_at",
            "capture_date",
            "source_occurrence",
            "source_sha256",
            "historical_backtest_admission",
        },
        "prospective consensus snapshots",
    )
    events = _prepare_events(event_readiness)
    snapshots = _prepare_snapshots(consensus_snapshots)
    if set(snapshots["prospective_event_id"]) - set(events["prospective_event_id"]):
        raise ProspectiveConsensusQualityError("Consensus evidence contains an unregistered event")
    changes = _build_change_evidence(snapshots, cfg)
    event_quality = _aggregate_event_quality(events, snapshots, changes, cfg)
    rules = _rule_catalog(cfg)
    gates = pd.DataFrame(
        [
            _gate(
                "frozen_event_coverage",
                int(event_quality["prospective_event_id"].nunique()),
                len(events),
                len(event_quality) == len(events)
                and not event_quality.duplicated("prospective_event_id").any(),
            ),
            _gate(
                "all_snapshots_forward_only",
                int(snapshots["historical_backtest_admission"].astype(bool).sum()),
                0,
                bool(not snapshots["historical_backtest_admission"].astype(bool).any()),
            ),
            _gate(
                "all_events_assessed",
                int(event_quality["quality_status"].notna().sum()),
                len(events),
                bool(event_quality["quality_status"].notna().all()),
            ),
            _gate("return_values_not_loaded", 0, 0, True),
            _gate("quality_rules_not_return_fitted", 0, 0, True),
        ]
    )
    counts = {
        str(key): int(value)
        for key, value in event_quality["quality_status"].value_counts().sort_index().items()
    }
    report: dict[str, Any] = {
        "contract_version": PROSPECTIVE_CONSENSUS_QUALITY_VERSION,
        "events": len(event_quality),
        "issuers": int(event_quality["cik"].nunique()),
        "snapshot_rows": len(snapshots),
        "independent_capture_dates": int(snapshots["capture_date"].nunique()),
        "change_rows": len(changes),
        "quality_status_counts": counts,
        "downstream_admitted_events": int(event_quality["downstream_admission"].sum()),
        "manual_review_events": int(event_quality["manual_review_required"].sum()),
        "quarantined_events": int(event_quality["quarantined"].sum()),
        "source_occurrence_change_events": int(
            event_quality["source_occurrence_change_count"].gt(0).sum()
        ),
        "analyst_count_change_events": int(event_quality["analyst_count_change_count"].gt(0).sum()),
        "stale_at_filing_events": int(event_quality["stale_at_filing"].sum()),
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "quality_audit_gate_passed": bool(gates["passed"].all()),
        "config": asdict(cfg),
    }
    return ProspectiveConsensusQualityTables(event_quality, changes, rules, gates, report)


def build_prospective_consensus_quality(
    readiness_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_consensus_quality"),
    config: ProspectiveConsensusQualityConfig | None = None,
) -> ProspectiveConsensusQualityResult:
    """Persist an immutable event-level structural quality audit."""
    cfg = config or ProspectiveConsensusQualityConfig()
    cfg.validate()
    manifests = {
        "readiness_manifest_sha256": _sha256_file(readiness_root / "output_manifest.json"),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    payload = json.dumps(
        {
            "version": PROSPECTIVE_CONSENSUS_QUALITY_VERSION,
            **manifests,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        PROSPECTIVE_CONSENSUS_QUALITY_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ProspectiveConsensusQualityResult(
            target.resolve(), _read_json(target / "prospective_consensus_quality_report.json")
        )
    if target.exists():
        raise ProspectiveConsensusQualityError(
            f"Incomplete consensus quality audit exists: {target}"
        )
    tables = calculate_prospective_consensus_quality(
        pd.read_parquet(readiness_root / "prospective_market_innovation_event_readiness.parquet"),
        pd.read_parquet(
            readiness_root / "prospective_market_innovation_consensus_snapshots.parquet"
        ),
        cfg,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "prospective_consensus_event_quality.parquet": tables.event_quality,
        "prospective_consensus_change_evidence.parquet": tables.change_evidence,
        "prospective_consensus_quality_rules.parquet": tables.rule_catalog,
        "prospective_consensus_quality_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "prospective_consensus_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ProspectiveConsensusQualityResult(target.resolve(), report)


def _prepare_events(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame.copy()
    if events.empty or events.duplicated("prospective_event_id").any():
        raise ProspectiveConsensusQualityError("The frozen event registry is empty or non-atomic")
    events["cik"] = events["cik"].astype(str).str.zfill(10)
    events["ticker"] = events["ticker"].astype(str).str.strip().str.upper()
    events["target_period_end"] = _utc(
        events["target_period_end"], "target_period_end"
    ).dt.normalize()
    events["accepted_at"] = pd.to_datetime(events["accepted_at"], utc=True, errors="coerce")
    return events


def _prepare_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    snapshots = frame.copy()
    snapshots["ticker"] = snapshots["ticker"].astype(str).str.strip().str.upper()
    snapshots["target_period_end"] = _utc(
        snapshots["target_period_end"], "snapshot target_period_end"
    ).dt.normalize()
    snapshots["captured_at"] = _utc(snapshots["captured_at"], "snapshot captured_at")
    snapshots["capture_date"] = _utc(
        snapshots["capture_date"], "snapshot capture_date"
    ).dt.normalize()
    numeric = [
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
        "source_occurrence",
    ]
    for column in numeric:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")
    if snapshots[numeric].isna().any().any():
        raise ProspectiveConsensusQualityError(
            "A quality snapshot contains missing numeric evidence"
        )
    if (
        snapshots["revenue_estimate_average"].le(0).any()
        or snapshots["revenue_estimate_analyst_count"].lt(1).any()
        or snapshots["revenue_estimate_high"].lt(snapshots["revenue_estimate_low"]).any()
    ):
        raise ProspectiveConsensusQualityError("A quality snapshot violates level or bound checks")
    if snapshots["historical_backtest_admission"].astype(bool).any():
        raise ProspectiveConsensusQualityError(
            "Historical consensus backfill entered quality audit"
        )
    snapshots = (
        snapshots.sort_values(
            ["prospective_event_id", "capture_date", "captured_at", "source_sha256"],
            kind="mergesort",
        )
        .drop_duplicates(["prospective_event_id", "capture_date"], keep="last")
        .reset_index(drop=True)
    )
    snapshots["range_width_pct"] = (
        snapshots["revenue_estimate_high"] - snapshots["revenue_estimate_low"]
    ) / snapshots["revenue_estimate_average"]
    return snapshots


def _build_change_evidence(
    snapshots: pd.DataFrame, config: ProspectiveConsensusQualityConfig
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=_change_columns())
    ordered = snapshots.sort_values(
        ["prospective_event_id", "captured_at"], kind="mergesort"
    ).copy()
    groups = ordered.groupby("prospective_event_id", sort=False)
    previous_columns = [
        "captured_at",
        "capture_date",
        "horizon",
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
        "source_occurrence",
        "source_sha256",
        "range_width_pct",
    ]
    for column in previous_columns:
        ordered[f"previous_{column}"] = groups[column].shift(1)
    changes = ordered.loc[ordered["previous_captured_at"].notna()].copy()
    if changes.empty:
        return pd.DataFrame(columns=_change_columns())
    changes["consensus_revision_pct"] = (
        changes["revenue_estimate_average"] / changes["previous_revenue_estimate_average"] - 1.0
    )
    changes["analyst_count_delta"] = (
        changes["revenue_estimate_analyst_count"]
        - changes["previous_revenue_estimate_analyst_count"]
    )
    changes["analyst_count_relative_change"] = (
        changes["analyst_count_delta"].abs() / changes["previous_revenue_estimate_analyst_count"]
    )
    changes["range_width_ratio"] = np.where(
        changes["previous_range_width_pct"].gt(0),
        changes["range_width_pct"] / changes["previous_range_width_pct"],
        np.where(changes["range_width_pct"].eq(0), 1.0, np.inf),
    )
    changes["consensus_level_changed"] = ~np.isclose(
        changes["revenue_estimate_average"],
        changes["previous_revenue_estimate_average"],
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
    )
    changes["analyst_count_changed"] = ~np.isclose(
        changes["revenue_estimate_analyst_count"],
        changes["previous_revenue_estimate_analyst_count"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    changes["source_occurrence_changed"] = changes["source_occurrence"].ne(
        changes["previous_source_occurrence"]
    )
    changes["source_payload_changed"] = changes["source_sha256"].ne(
        changes["previous_source_sha256"]
    )
    changes["horizon_changed"] = (
        changes["horizon"].astype(str).ne(changes["previous_horizon"].astype(str))
    )
    changes["range_width_shock"] = changes["range_width_ratio"].lt(
        config.range_width_ratio_review_lower
    ) | changes["range_width_ratio"].gt(config.range_width_ratio_review_upper)
    changes["large_consensus_level_change"] = (
        changes["consensus_revision_pct"].abs().ge(config.large_consensus_revision_abs_threshold)
    )
    statuses: list[str] = []
    reasons: list[str] = []
    for row in changes.to_dict("records"):
        status, reason = _classify_change({str(key): value for key, value in row.items()}, config)
        statuses.append(status)
        reasons.append(reason)
    changes["change_quality_status"] = statuses
    changes["quality_reasons"] = reasons
    changes["downstream_admission"] = changes["change_quality_status"].eq(CLEAN)
    return changes[_change_columns()].reset_index(drop=True)


def _classify_change(
    row: dict[str, Any], config: ProspectiveConsensusQualityConfig
) -> tuple[str, str]:
    quarantine_reasons: list[str] = []
    review_reasons: list[str] = []
    if bool(row["horizon_changed"]):
        quarantine_reasons.append("provider_horizon_changed")
    if bool(row["source_occurrence_changed"]):
        quarantine_reasons.append("provider_source_occurrence_changed")
    if float(row["analyst_count_relative_change"]) >= (
        config.analyst_count_relative_quarantine_threshold
    ):
        quarantine_reasons.append("large_analyst_coverage_change")
    if bool(row["range_width_shock"]) and bool(row["analyst_count_changed"]):
        quarantine_reasons.append("range_shock_with_coverage_change")
    if quarantine_reasons:
        return QUARANTINED, "|".join(sorted(set(quarantine_reasons)))
    if bool(row["analyst_count_changed"]):
        review_reasons.append("analyst_coverage_changed")
    if bool(row["range_width_shock"]):
        review_reasons.append("consensus_range_width_shock")
    if bool(row["large_consensus_level_change"]):
        review_reasons.append("large_consensus_level_change")
    if review_reasons:
        return REVIEW_REQUIRED, "|".join(sorted(set(review_reasons)))
    return CLEAN, "none"


def _aggregate_event_quality(
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    changes: pd.DataFrame,
    config: ProspectiveConsensusQualityConfig,
) -> pd.DataFrame:
    snapshot_summary = (
        snapshots.groupby("prospective_event_id", as_index=False).agg(
            snapshot_rows=("captured_at", "size"),
            independent_capture_dates=("capture_date", "nunique"),
            first_snapshot_at=("captured_at", "min"),
            last_snapshot_at=("captured_at", "max"),
        )
        if not snapshots.empty
        else pd.DataFrame(
            columns=[
                "prospective_event_id",
                "snapshot_rows",
                "independent_capture_dates",
                "first_snapshot_at",
                "last_snapshot_at",
            ]
        )
    )
    output = events[
        [
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "accepted_at",
            "future_filing_observed",
        ]
    ].merge(snapshot_summary, on="prospective_event_id", how="left", validate="one_to_one")
    output["snapshot_rows"] = output["snapshot_rows"].fillna(0).astype(int)
    output["independent_capture_dates"] = output["independent_capture_dates"].fillna(0).astype(int)
    change_rows: list[dict[str, Any]] = []
    for event_id in output["prospective_event_id"].astype(str):
        event_changes = changes.loc[changes["prospective_event_id"].astype(str).eq(event_id)]
        statuses = event_changes["change_quality_status"].astype(str).tolist()
        reasons = sorted(
            {
                reason
                for text in event_changes["quality_reasons"].astype(str)
                for reason in text.split("|")
                if reason and reason != "none"
            }
        )
        quality_status = (
            QUARANTINED
            if QUARANTINED in statuses
            else REVIEW_REQUIRED
            if REVIEW_REQUIRED in statuses
            else CLEAN
            if statuses
            else INSUFFICIENT_HISTORY
        )
        change_rows.append(
            {
                "prospective_event_id": event_id,
                "change_rows": len(event_changes),
                "clean_change_rows": statuses.count(CLEAN),
                "review_change_rows": statuses.count(REVIEW_REQUIRED),
                "quarantined_change_rows": statuses.count(QUARANTINED),
                "source_occurrence_change_count": int(
                    event_changes["source_occurrence_changed"].sum()
                ),
                "analyst_count_change_count": int(event_changes["analyst_count_changed"].sum()),
                "range_width_shock_count": int(event_changes["range_width_shock"].sum()),
                "large_consensus_level_change_count": int(
                    event_changes["large_consensus_level_change"].sum()
                ),
                "max_analyst_count_relative_change": (
                    float(event_changes["analyst_count_relative_change"].max())
                    if not event_changes.empty
                    else None
                ),
                "quality_status": quality_status,
                "quality_reasons": "|".join(reasons) if reasons else "none",
            }
        )
    output = output.merge(
        pd.DataFrame(change_rows), on="prospective_event_id", validate="one_to_one"
    )
    output["staleness_calendar_days_at_filing"] = np.where(
        output["future_filing_observed"].astype(bool)
        & output["accepted_at"].notna()
        & output["last_snapshot_at"].notna(),
        (output["accepted_at"].dt.normalize() - output["last_snapshot_at"].dt.normalize()).dt.days,
        np.nan,
    )
    output["stale_at_filing"] = output["staleness_calendar_days_at_filing"].gt(
        config.stale_preacceptance_calendar_days
    )
    stale = output["stale_at_filing"]
    output.loc[stale, "quality_status"] = QUARANTINED
    output.loc[stale, "quality_reasons"] = output.loc[stale, "quality_reasons"].map(
        lambda text: (
            "stale_preacceptance_consensus"
            if text == "none"
            else f"{text}|stale_preacceptance_consensus"
        )
    )
    output["downstream_admission"] = output["quality_status"].eq(CLEAN)
    output["manual_review_required"] = output["quality_status"].isin([REVIEW_REQUIRED, QUARANTINED])
    output["quarantined"] = output["quality_status"].eq(QUARANTINED)
    output["actual_values_loaded"] = False
    output["return_values_loaded"] = False
    output["selection_uses_return_values"] = False
    return output.sort_values("registration_order", kind="mergesort").reset_index(drop=True)


def _rule_catalog(config: ProspectiveConsensusQualityConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule": "provider_horizon_changed",
                "severity": QUARANTINED,
                "threshold": "any change",
                "reason": "A company-period no longer maps to the same provider horizon.",
            },
            {
                "rule": "provider_source_occurrence_changed",
                "severity": QUARANTINED,
                "threshold": "any change",
                "reason": "Provider row ordering changed and may indicate remapping.",
            },
            {
                "rule": "large_analyst_coverage_change",
                "severity": QUARANTINED,
                "threshold": f">={config.analyst_count_relative_quarantine_threshold:.0%}",
                "reason": "Consensus composition changed too much for a pure revision label.",
            },
            {
                "rule": "range_shock_with_coverage_change",
                "severity": QUARANTINED,
                "threshold": (
                    f"range ratio outside [{config.range_width_ratio_review_lower:.2f}, "
                    f"{config.range_width_ratio_review_upper:.2f}] plus analyst-count change"
                ),
                "reason": "Dispersion and coverage changed together.",
            },
            {
                "rule": "analyst_coverage_changed",
                "severity": REVIEW_REQUIRED,
                "threshold": "any non-quarantined change",
                "reason": "The numeric revision may partly reflect a different analyst set.",
            },
            {
                "rule": "consensus_range_width_shock",
                "severity": REVIEW_REQUIRED,
                "threshold": (
                    f"ratio outside [{config.range_width_ratio_review_lower:.2f}, "
                    f"{config.range_width_ratio_review_upper:.2f}]"
                ),
                "reason": "Consensus dispersion changed abruptly.",
            },
            {
                "rule": "large_consensus_level_change",
                "severity": REVIEW_REQUIRED,
                "threshold": f">={config.large_consensus_revision_abs_threshold:.0%}",
                "reason": "A large level change is preserved but requires human review.",
            },
            {
                "rule": "stale_preacceptance_consensus",
                "severity": QUARANTINED,
                "threshold": f">{config.stale_preacceptance_calendar_days} calendar days",
                "reason": "The last captured consensus is too stale at SEC acceptance.",
            },
        ]
    ).assign(uses_return_values=False)


def _change_columns() -> list[str]:
    return [
        "prospective_event_id",
        "ticker",
        "target_period_end",
        "previous_captured_at",
        "captured_at",
        "previous_capture_date",
        "capture_date",
        "previous_revenue_estimate_average",
        "revenue_estimate_average",
        "consensus_revision_pct",
        "previous_revenue_estimate_analyst_count",
        "revenue_estimate_analyst_count",
        "analyst_count_delta",
        "analyst_count_relative_change",
        "previous_range_width_pct",
        "range_width_pct",
        "range_width_ratio",
        "previous_source_occurrence",
        "source_occurrence",
        "previous_source_sha256",
        "source_sha256",
        "consensus_level_changed",
        "analyst_count_changed",
        "source_occurrence_changed",
        "source_payload_changed",
        "horizon_changed",
        "range_width_shock",
        "large_consensus_level_change",
        "change_quality_status",
        "quality_reasons",
        "downstream_admission",
    ]


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ProspectiveConsensusQualityError(f"{label} lacks required columns: {missing}")


def _utc(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ProspectiveConsensusQualityError(f"Invalid {label}")
    return result


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
