"""Track the frozen v3 cohort without changing its events or opening returns.

Unlike the generic forward-consensus panel, this registry-aware layer continues
to admit live snapshots for a registered fiscal period after the period end and
stops only at the eventual SEC acceptance timestamp.  That distinction preserves
the economically relevant immediately-pre-filing analyst expectation.
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
from sec_data.forward_filing_monitor import parse_forward_filing_snapshot

PROSPECTIVE_MARKET_INNOVATION_READINESS_VERSION: Final[str] = (
    "prospective-market-innovation-v3-readiness-v1"
)


class ProspectiveMarketInnovationReadinessError(FilingCatalogError):
    """Raised when the frozen registry or a forward clock is violated."""


@dataclass(frozen=True)
class ProspectiveMarketInnovationReadinessConfig:
    """Frozen collection-depth gates; these do not change the v3 hypothesis."""

    minimum_registered_events: int = 30
    minimum_registered_issuers: int = 15
    minimum_capture_dates: int = 2
    minimum_revision_events: int = 30
    minimum_submission_issuers: int = 15

    def validate(self) -> None:
        """Prevent silent relaxation of the prospective readiness rules."""
        if (
            self.minimum_registered_events,
            self.minimum_registered_issuers,
            self.minimum_capture_dates,
            self.minimum_revision_events,
            self.minimum_submission_issuers,
        ) != (30, 15, 2, 30, 15):
            raise ValueError("The prospective v3 readiness thresholds are frozen")


@dataclass(frozen=True)
class ProspectiveMarketInnovationReadinessTables:
    """Registered snapshots, revision state, SEC observations, and gates."""

    event_readiness: pd.DataFrame
    consensus_snapshots: pd.DataFrame
    revision_state: pd.DataFrame
    filing_observations: pd.DataFrame
    filing_candidates: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveMarketInnovationReadinessResult:
    """One immutable as-of readiness audit for the frozen v3 registry."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_market_innovation_readiness(
    registered_events: pd.DataFrame,
    live_consensus_observations: pd.DataFrame,
    filing_observations: pd.DataFrame,
    config: ProspectiveMarketInnovationReadinessConfig | None = None,
) -> ProspectiveMarketInnovationReadinessTables:
    """Measure v3 evidence depth while keeping actuals and returns closed."""
    cfg = config or ProspectiveMarketInnovationReadinessConfig()
    cfg.validate()
    _require_columns(
        registered_events,
        {
            "prospective_event_id",
            "live_consensus_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "captured_at",
            "actual_value_loaded_at_registration",
            "return_value_loaded_at_registration",
            "replacement_allowed",
        },
        "v3 registered events",
    )
    _require_columns(
        live_consensus_observations,
        {
            "ticker",
            "target_period_end",
            "captured_at",
            "availability_clock",
            "forward_only_admission",
            "historical_backtest_admission",
            "revenue_estimate_average",
            "revenue_estimate_high",
            "revenue_estimate_low",
            "revenue_estimate_analyst_count",
            "source_sha256",
        },
        "live consensus observations",
    )
    _require_columns(
        filing_observations,
        {
            "prospective_event_id",
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "form",
            "accepted_at",
            "observed_at",
            "future_filing_candidate",
            "amendment_form",
            "source_sha256",
        },
        "registered filing observations",
    )
    registry = registered_events.copy()
    if (
        registry["actual_value_loaded_at_registration"].astype(bool).any()
        or registry["return_value_loaded_at_registration"].astype(bool).any()
        or registry["replacement_allowed"].astype(bool).any()
    ):
        raise ProspectiveMarketInnovationReadinessError(
            "The frozen v3 registry violates its outcome firewall"
        )
    registry["cik"] = registry["cik"].astype(str).str.zfill(10)
    registry["ticker"] = registry["ticker"].astype(str).str.strip().str.upper()
    registry["target_period_end"] = _utc(
        registry["target_period_end"], "registered target_period_end"
    ).dt.normalize()
    registry["registered_at"] = _utc(registry["captured_at"], "registered captured_at")
    if (
        registry.duplicated("prospective_event_id").any()
        or registry.duplicated(["cik", "target_period_end"]).any()
    ):
        raise ProspectiveMarketInnovationReadinessError("The v3 event registry is not atomic")

    filings = filing_observations.copy()
    if not filings.empty:
        filings["cik"] = filings["cik"].astype(str).str.zfill(10)
        filings["ticker"] = filings["ticker"].astype(str).str.strip().str.upper()
        filings["target_period_end"] = _utc(
            filings["target_period_end"], "filing target_period_end"
        ).dt.normalize()
        filings["accepted_at"] = pd.to_datetime(filings["accepted_at"], utc=True, errors="coerce")
        filings["observed_at"] = _utc(filings["observed_at"], "filing observed_at")
    original_candidates = filings.loc[
        filings["future_filing_candidate"].astype(bool)
        & ~filings["amendment_form"].astype(bool)
        & filings["form"].astype(str).isin(["10-Q", "10-K"])
        & filings["accepted_at"].notna()
    ].copy()
    if not original_candidates.empty:
        original_candidates = (
            original_candidates.sort_values(
                ["prospective_event_id", "accepted_at", "observed_at", "accession_number"],
                kind="mergesort",
            )
            .drop_duplicates("prospective_event_id", keep="first")
            .reset_index(drop=True)
        )
    acceptance = original_candidates[
        ["prospective_event_id", "accession_number", "form", "accepted_at", "observed_at"]
    ].rename(
        columns={
            "accession_number": "future_accession_number",
            "form": "future_form",
            "observed_at": "future_filing_first_observed_at",
        }
    )
    events_with_acceptance = registry.merge(
        acceptance, on="prospective_event_id", how="left", validate="one_to_one"
    )

    observations = live_consensus_observations.copy()
    observations["ticker"] = observations["ticker"].astype(str).str.strip().str.upper()
    observations["target_period_end"] = _utc(
        observations["target_period_end"], "consensus target_period_end"
    ).dt.normalize()
    observations["captured_at"] = _utc(observations["captured_at"], "consensus captured_at")
    observations["capture_date"] = observations["captured_at"].dt.normalize()
    if not observations["availability_clock"].astype(str).eq("collector_fetched_at").all():
        raise ProspectiveMarketInnovationReadinessError(
            "A consensus observation lacks the collector availability clock"
        )
    if (
        not observations["forward_only_admission"].astype(bool).all()
        or observations["historical_backtest_admission"].astype(bool).any()
    ):
        raise ProspectiveMarketInnovationReadinessError(
            "A consensus observation violates forward-only admission"
        )
    snapshots = observations.merge(
        events_with_acceptance[
            [
                "prospective_event_id",
                "cik",
                "ticker",
                "target_period_end",
                "registered_at",
                "accepted_at",
            ]
        ],
        on=["ticker", "target_period_end"],
        how="inner",
        validate="many_to_one",
    )
    snapshots = snapshots.loc[snapshots["captured_at"].ge(snapshots["registered_at"])].copy()
    snapshots = snapshots.loc[
        snapshots["accepted_at"].isna() | snapshots["captured_at"].lt(snapshots["accepted_at"])
    ].copy()
    numeric = [
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
    ]
    for column in numeric:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")
    if (
        snapshots[numeric].isna().any().any()
        or snapshots["revenue_estimate_average"].le(0.0).any()
        or snapshots["revenue_estimate_high"].lt(snapshots["revenue_estimate_low"]).any()
        or snapshots["revenue_estimate_analyst_count"].lt(1.0).any()
    ):
        raise ProspectiveMarketInnovationReadinessError(
            "A registered consensus snapshot fails quality checks"
        )
    snapshots = (
        snapshots.sort_values(
            ["prospective_event_id", "captured_at", "source_sha256"], kind="mergesort"
        )
        .drop_duplicates(["prospective_event_id", "captured_at"], keep="last")
        .reset_index(drop=True)
    )
    snapshots["consensus_range_pct"] = (
        snapshots["revenue_estimate_high"] - snapshots["revenue_estimate_low"]
    ) / snapshots["revenue_estimate_average"]
    revision_state = _revision_state(events_with_acceptance, snapshots)
    event_readiness = events_with_acceptance.merge(
        revision_state,
        on="prospective_event_id",
        how="left",
        validate="one_to_one",
    )
    event_readiness["consensus_snapshot_rows"] = (
        event_readiness["consensus_snapshot_rows"].fillna(0).astype(int)
    )
    event_readiness["consensus_capture_dates"] = (
        event_readiness["consensus_capture_dates"].fillna(0).astype(int)
    )
    event_readiness["revision_ready"] = event_readiness["revision_ready"].fillna(False)
    event_readiness["future_filing_observed"] = event_readiness["accepted_at"].notna()
    event_readiness["actual_value_loaded"] = False
    event_readiness["return_value_loaded"] = False
    event_readiness["readiness_status"] = np.select(
        [
            event_readiness["future_filing_observed"],
            event_readiness["revision_ready"],
            event_readiness["consensus_snapshot_rows"].gt(0),
        ],
        [
            "future_filing_observed_actual_not_loaded",
            "revision_tracking_ready_awaiting_filing",
            "level_tracking_ready_awaiting_revision_and_filing",
        ],
        default="registered_consensus_snapshot_missing",
    )
    capture_dates = int(snapshots["capture_date"].nunique()) if not snapshots.empty else 0
    snapshot_events = int(snapshots["prospective_event_id"].nunique()) if not snapshots.empty else 0
    revision_events = int(event_readiness["revision_ready"].sum())
    submission_issuers = int(filings["cik"].nunique()) if not filings.empty else 0
    future_candidates = len(original_candidates)
    gates = pd.DataFrame(
        [
            _gate(
                "frozen_registered_events",
                len(registry),
                cfg.minimum_registered_events,
                len(registry) >= cfg.minimum_registered_events,
            ),
            _gate(
                "frozen_registered_issuers",
                int(registry["cik"].nunique()),
                cfg.minimum_registered_issuers,
                int(registry["cik"].nunique()) >= cfg.minimum_registered_issuers,
            ),
            _gate(
                "registered_consensus_event_coverage",
                snapshot_events,
                len(registry),
                snapshot_events == len(registry),
            ),
            _gate(
                "minimum_independent_capture_dates",
                capture_dates,
                cfg.minimum_capture_dates,
                capture_dates >= cfg.minimum_capture_dates,
            ),
            _gate(
                "minimum_revision_events",
                revision_events,
                cfg.minimum_revision_events,
                revision_events >= cfg.minimum_revision_events,
            ),
            _gate(
                "submission_snapshot_issuer_coverage",
                submission_issuers,
                cfg.minimum_submission_issuers,
                submission_issuers >= cfg.minimum_submission_issuers,
            ),
            _gate(
                "future_original_filing_candidates", future_candidates, 1, future_candidates >= 1
            ),
            _gate("actual_values_loaded", 0, 0, True),
            _gate("return_values_loaded", 0, 0, True),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": PROSPECTIVE_MARKET_INNOVATION_READINESS_VERSION,
        "registered_events": len(registry),
        "registered_issuers": int(registry["cik"].nunique()),
        "registered_consensus_events": snapshot_events,
        "registered_consensus_rows": len(snapshots),
        "capture_dates": capture_dates,
        "revision_ready_events": revision_events,
        "submission_snapshot_issuers": submission_issuers,
        "future_filing_candidates": future_candidates,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "feature_model_fitted": False,
        "period_end_is_collection_stop": False,
        "consensus_collection_stop": "sec_acceptance_timestamp_per_registered_event",
        "collection_foundation_gate_passed": bool(gates.iloc[[0, 1, 2, 5, 7, 8]]["passed"].all()),
        "revision_gate_passed": bool(gates.iloc[[3, 4]]["passed"].all()),
        "future_filing_gate_passed": bool(gates.iloc[6]["passed"]),
        "return_research_gate_passed": False,
        "config": asdict(cfg),
    }
    return ProspectiveMarketInnovationReadinessTables(
        event_readiness,
        snapshots,
        revision_state,
        filings,
        original_candidates,
        gates,
        report,
    )


def build_prospective_market_innovation_readiness(
    preregistration_root: Path,
    consensus_audit_root: Path,
    filing_monitor_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_market_innovation_readiness"),
    config: ProspectiveMarketInnovationReadinessConfig | None = None,
) -> ProspectiveMarketInnovationReadinessResult:
    """Persist one registry-aware, outcome-free readiness audit."""
    cfg = config or ProspectiveMarketInnovationReadinessConfig()
    cfg.validate()
    manifests = {
        "preregistration_manifest_sha256": _sha256_file(
            preregistration_root / "output_manifest.json"
        ),
        "consensus_audit_manifest_sha256": _sha256_file(
            consensus_audit_root / "output_manifest.json"
        ),
        "filing_monitor_manifest_sha256": _sha256_file(
            filing_monitor_root / "output_manifest.json"
        ),
    }
    registry = pd.read_parquet(
        preregistration_root / "prospective_market_innovation_events.parquet"
    )
    observations = pd.read_parquet(
        consensus_audit_root / "alpha_vantage_live_revenue_consensus.parquet"
    )
    filing_manifests = pd.read_parquet(
        filing_monitor_root / "forward_filing_submission_snapshots.parquet"
    )
    filings = parse_registered_filing_snapshots(registry, filing_manifests)
    tables = calculate_prospective_market_innovation_readiness(registry, observations, filings, cfg)
    payload = json.dumps(
        {
            "version": PROSPECTIVE_MARKET_INNOVATION_READINESS_VERSION,
            **manifests,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        PROSPECTIVE_MARKET_INNOVATION_READINESS_VERSION
        + "-"
        + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ProspectiveMarketInnovationReadinessResult(
            target.resolve(),
            _read_json(target / "prospective_market_innovation_readiness_report.json"),
        )
    if target.exists():
        raise ProspectiveMarketInnovationReadinessError(
            f"Incomplete prospective readiness audit exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "prospective_market_innovation_event_readiness.parquet": tables.event_readiness,
        "prospective_market_innovation_consensus_snapshots.parquet": tables.consensus_snapshots,
        "prospective_market_innovation_revision_state.parquet": tables.revision_state,
        "prospective_market_innovation_filing_observations.parquet": tables.filing_observations,
        "prospective_market_innovation_filing_candidates.parquet": tables.filing_candidates,
        "prospective_market_innovation_readiness_gates.parquet": tables.gates,
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
    report_path = staging / "prospective_market_innovation_readiness_report.json"
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
    return ProspectiveMarketInnovationReadinessResult(target.resolve(), report)


def parse_registered_filing_snapshots(
    registered_events: pd.DataFrame, filing_snapshot_manifests: pd.DataFrame
) -> pd.DataFrame:
    """Reuse one audited SEC submissions payload across all registered issuer periods."""
    required_registry = {
        "prospective_event_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "captured_at",
        "consensus_revenue_level",
    }
    _require_columns(registered_events, required_registry, "registered events")
    _require_columns(
        filing_snapshot_manifests,
        {"cik", "ticker", "fetched_at", "sha256", "path"},
        "filing snapshot manifests",
    )
    rows: list[dict[str, Any]] = []
    registry = registered_events.copy()
    registry["cik"] = registry["cik"].astype(str).str.zfill(10)
    manifests = filing_snapshot_manifests.copy()
    manifests["cik"] = manifests["cik"].astype(str).str.zfill(10)
    for manifest in manifests.sort_values(["fetched_at", "cik", "sha256"]).itertuples(index=False):
        raw_path = Path(str(manifest.path))
        if not raw_path.is_file() or _sha256_file(raw_path) != str(manifest.sha256):
            raise ProspectiveMarketInnovationReadinessError(
                f"Invalid registered SEC submissions snapshot: {raw_path}"
            )
        issuer_events = registry.loc[registry["cik"].eq(str(manifest.cik))]
        for event in issuer_events.to_dict("records"):
            target = {
                "sec_monitor_target_id": event["prospective_event_id"],
                "live_consensus_target_id": event["live_consensus_target_id"],
                "cik": event["cik"],
                "ticker": event["ticker"],
                "target_period_end": event["target_period_end"],
                "consensus_captured_at": event["captured_at"],
                "consensus_revenue_level": event["consensus_revenue_level"],
            }
            parsed = parse_forward_filing_snapshot(
                raw_path.read_bytes(),
                target=target,
                observed_at=str(manifest.fetched_at),
                source_sha256=str(manifest.sha256),
            )
            for row in parsed:
                rows.append({"prospective_event_id": event["prospective_event_id"], **row})
    if not rows:
        return pd.DataFrame(columns=_filing_columns())
    filings = pd.DataFrame(rows)
    filings = (
        filings.sort_values(
            ["prospective_event_id", "accession_number", "observed_at", "source_sha256"],
            kind="mergesort",
        )
        .drop_duplicates(["prospective_event_id", "accession_number"], keep="first")
        .reset_index(drop=True)
    )
    return filings


def _revision_state(registry: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in registry.itertuples(index=False):
        event_snapshots = snapshots.loc[
            snapshots["prospective_event_id"].astype(str).eq(str(event.prospective_event_id))
        ].copy()
        if event_snapshots.empty:
            rows.append(
                {
                    "prospective_event_id": event.prospective_event_id,
                    "consensus_snapshot_rows": 0,
                    "consensus_capture_dates": 0,
                    "earliest_consensus_captured_at": pd.NaT,
                    "earliest_consensus_revenue": np.nan,
                    "latest_consensus_captured_at": pd.NaT,
                    "latest_consensus_revenue": np.nan,
                    "latest_consensus_range_pct": np.nan,
                    "latest_analyst_count": np.nan,
                    "consensus_revision_delta": np.nan,
                    "consensus_revision_pct": np.nan,
                    "revision_ready": False,
                }
            )
            continue
        daily = (
            event_snapshots.sort_values(["capture_date", "captured_at"], kind="mergesort")
            .groupby("capture_date", as_index=False, sort=True)
            .tail(1)
            .sort_values("captured_at", kind="mergesort")
        )
        earliest = daily.iloc[0]
        latest = daily.iloc[-1]
        capture_dates = len(daily)
        earliest_level = float(earliest["revenue_estimate_average"])
        latest_level = float(latest["revenue_estimate_average"])
        revision_delta = latest_level - earliest_level if capture_dates >= 2 else np.nan
        revision_pct = revision_delta / earliest_level if capture_dates >= 2 else np.nan
        rows.append(
            {
                "prospective_event_id": event.prospective_event_id,
                "consensus_snapshot_rows": len(event_snapshots),
                "consensus_capture_dates": capture_dates,
                "earliest_consensus_captured_at": earliest["captured_at"],
                "earliest_consensus_revenue": earliest_level,
                "latest_consensus_captured_at": latest["captured_at"],
                "latest_consensus_revenue": latest_level,
                "latest_consensus_range_pct": float(latest["consensus_range_pct"]),
                "latest_analyst_count": float(latest["revenue_estimate_analyst_count"]),
                "consensus_revision_delta": revision_delta,
                "consensus_revision_pct": revision_pct,
                "revision_ready": capture_dates >= 2,
            }
        )
    return pd.DataFrame(rows)


def _filing_columns() -> list[str]:
    return [
        "prospective_event_id",
        "sec_monitor_target_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "consensus_captured_at",
        "consensus_revenue_level",
        "accession_number",
        "form",
        "filing_date",
        "report_date",
        "accepted_at",
        "primary_document",
        "period_end_distance_days",
        "period_match_status",
        "accepted_after_consensus",
        "observed_after_acceptance",
        "future_filing_candidate",
        "amendment_form",
        "observed_at",
        "availability_clock",
        "historical_backtest_admission",
        "source_sha256",
    ]


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ProspectiveMarketInnovationReadinessError(f"Invalid {label}")
    return parsed


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ProspectiveMarketInnovationReadinessError(
            f"{label} lacks required columns: {missing}"
        )


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
