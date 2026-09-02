"""Evidence-backed capability audit for candidate analyst-consensus providers."""

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

CONSENSUS_PROVIDER_AUDIT_VERSION: Final[str] = "consensus-provider-audit-v1-evidence-backed"


class ConsensusProviderAuditError(FilingCatalogError):
    """Raised when provider capability evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ConsensusProviderAuditTables:
    """Verified probe observations, provider decisions, and report."""

    eulerpool_snapshots: pd.DataFrame
    eulerpool_observations: pd.DataFrame
    provider_decisions: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConsensusProviderAuditResult:
    """Immutable persisted provider audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_consensus_provider_audit(
    consensus_targets: pd.DataFrame,
    massive_event_audit: pd.DataFrame,
    eulerpool_probe_root: Path,
) -> ConsensusProviderAuditTables:
    """Evaluate provider evidence without reading or selecting on return values."""
    _require_columns(
        consensus_targets,
        {
            "validation_event_id",
            "cik",
            "ticker",
            "consensus_cutoff_at",
            "selection_uses_return_values",
        },
        "consensus targets",
    )
    _require_columns(
        massive_event_audit,
        {
            "validation_event_id",
            "provider_query_complete",
            "provider_failure_reason",
            "point_in_time_consensus_ready",
            "selection_uses_return_values",
        },
        "Massive consensus audit",
    )
    if (
        consensus_targets["selection_uses_return_values"].astype(bool).any()
        or massive_event_audit["selection_uses_return_values"].astype(bool).any()
    ):
        raise ConsensusProviderAuditError("Provider audit inputs cannot be return-selected")
    snapshot_frame, observation_frame = load_eulerpool_probe_evidence(eulerpool_probe_root)
    cutoff = pd.to_datetime(consensus_targets["consensus_cutoff_at"], utc=True)
    target_cutoff_max = cutoff.max()
    fields = set(observation_frame["field"].dropna().astype(str).str.lower())
    revenue_field_present = any("revenue" in field for field in fields)
    estimate_rows = observation_frame["field"].astype(str).str.lower().str.contains("estimate")
    estimate_as_of = pd.to_datetime(
        observation_frame.loc[estimate_rows, "as_of_date"], utc=True, errors="coerce"
    )
    historical_asof_precedes_any_target = bool(
        estimate_as_of.notna().any() and estimate_as_of.min() < target_cutoff_max
    )
    euler_tickers = int(observation_frame["ticker"].nunique())
    euler_rows = len(observation_frame)
    massive_entitlement_failures = int(
        massive_event_audit["provider_failure_reason"].eq("entitlement_denied").sum()
    )
    massive_ready = int(massive_event_audit["point_in_time_consensus_ready"].sum())
    decisions = pd.DataFrame(
        [
            {
                "provider": "massive_benzinga_earnings",
                "authenticated_probe_succeeded": massive_entitlement_failures > 0,
                "provider_access_ready": False,
                "revenue_consensus_field_observed": False,
                "historical_record_version_observed": False,
                "point_in_time_ready_events": massive_ready,
                "decision": "rejected_current_entitlement",
                "formal_model_admission": False,
                "reason": "authenticated request returned entitlement_denied",
            },
            {
                "provider": "eulerpool_pit_estimates",
                "authenticated_probe_succeeded": euler_rows > 0,
                "provider_access_ready": euler_rows > 0,
                "revenue_consensus_field_observed": revenue_field_present,
                "historical_record_version_observed": historical_asof_precedes_any_target,
                "point_in_time_ready_events": 0,
                "decision": "rejected_for_revenue_consensus_v1",
                "formal_model_admission": False,
                "reason": (
                    "probe contained only EPS fields and 2026 as-of versions, not historical "
                    "revenue consensus"
                ),
            },
        ]
    )
    decisions["selection_uses_return_values"] = False
    report: dict[str, Any] = {
        "contract_version": CONSENSUS_PROVIDER_AUDIT_VERSION,
        "consensus_targets": len(consensus_targets),
        "consensus_target_issuers": int(consensus_targets["cik"].nunique()),
        "massive_entitlement_failures": massive_entitlement_failures,
        "massive_point_in_time_ready_events": massive_ready,
        "eulerpool_probe_tickers": euler_tickers,
        "eulerpool_probe_rows": euler_rows,
        "eulerpool_fields": sorted(fields),
        "eulerpool_revenue_field_present": revenue_field_present,
        "eulerpool_as_of_min": _timestamp_string(estimate_as_of.min()),
        "eulerpool_as_of_max": _timestamp_string(estimate_as_of.max()),
        "eulerpool_historical_asof_precedes_any_target": historical_asof_precedes_any_target,
        "providers_formally_admitted": int(decisions["formal_model_admission"].sum()),
        "provider_gate": bool(decisions["formal_model_admission"].any()),
        "return_values_loaded": False,
        "selection_uses_return_values": False,
    }
    return ConsensusProviderAuditTables(snapshot_frame, observation_frame, decisions, report)


def load_eulerpool_probe_evidence(probe_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the widest immutable probe for each ticker and verify its exact bytes."""
    candidates: list[dict[str, Any]] = []
    for manifest_path in probe_root.glob("*/*/manifest.json"):
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_path = Path(str(manifest.get("path", "")))
        expected_hash = str(manifest.get("sha256", ""))
        if (
            not raw_path.is_file()
            or raw_path.stat().st_size != int(manifest.get("bytes", -1))
            or _sha256_file(raw_path) != expected_hash
        ):
            raise ConsensusProviderAuditError(f"Invalid Eulerpool probe snapshot: {raw_path}")
        candidates.append({**manifest, "manifest_path": str(manifest_path.resolve())})
    if not candidates:
        raise ConsensusProviderAuditError(f"No Eulerpool probes exist under {probe_root}")
    manifests = pd.DataFrame(candidates)
    manifests["limit"] = pd.to_numeric(manifests["limit"], errors="raise").astype(int)
    manifests["fetched_at"] = pd.to_datetime(manifests["fetched_at"], utc=True)
    selected = (
        manifests.sort_values(["ticker", "limit", "fetched_at"], kind="mergesort")
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for snapshot in selected.itertuples(index=False):
        document: Any = json.loads(Path(str(snapshot.path)).read_bytes())
        if not isinstance(document, list):
            raise ConsensusProviderAuditError("Eulerpool probe response must be a list")
        for occurrence, item in enumerate(document):
            if not isinstance(item, dict):
                raise ConsensusProviderAuditError("Eulerpool probe record must be an object")
            rows.append(
                {
                    "ticker": str(item.get("ticker", "")).strip().upper(),
                    "provider_record_id": str(item.get("id", "")),
                    "field": str(item.get("field", "")).strip().lower(),
                    "period": item.get("period"),
                    "value": item.get("value"),
                    "as_of_date": item.get("as_of_date"),
                    "source": item.get("source"),
                    "source_occurrence": occurrence,
                    "source_sha256": str(snapshot.sha256),
                }
            )
    observations = pd.DataFrame(rows)
    return selected, observations


def build_consensus_provider_audit(
    consensus_plan_root: Path,
    massive_audit_root: Path,
    *,
    eulerpool_probe_root: Path = Path("data/market/eulerpool_consensus/probes"),
    output_root: Path = Path("data/processed/consensus_provider_audit"),
) -> ConsensusProviderAuditResult:
    """Persist provider decisions tied to immutable target and evidence manifests."""
    input_hashes = {
        "consensus_plan": _sha256_file(consensus_plan_root / "output_manifest.json"),
        "massive_consensus_audit": _sha256_file(massive_audit_root / "output_manifest.json"),
    }
    tables = calculate_consensus_provider_audit(
        load_catalog_table(consensus_plan_root, "revenue_consensus_targets.parquet"),
        load_catalog_table(massive_audit_root, "revenue_consensus_event_audit.parquet"),
        eulerpool_probe_root,
    )
    euler_hashes = sorted(tables.eulerpool_snapshots["sha256"].astype(str))
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "eulerpool_probe_sha256": euler_hashes,
            "contract_version": CONSENSUS_PROVIDER_AUDIT_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        CONSENSUS_PROVIDER_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConsensusProviderAuditResult(
            target.resolve(), _read_json(target / "consensus_provider_audit_report.json")
        )
    if target.exists():
        raise ConsensusProviderAuditError(f"Incomplete provider audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "eulerpool_probe_snapshots.parquet": tables.eulerpool_snapshots,
        "eulerpool_probe_observations.parquet": tables.eulerpool_observations,
        "consensus_provider_decisions.parquet": tables.provider_decisions,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "eulerpool_probe_sha256": euler_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "consensus_provider_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "eulerpool_probe_sha256": euler_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ConsensusProviderAuditResult(target.resolve(), report)


def _timestamp_string(value: Any) -> str | None:
    return value.isoformat() if pd.notna(value) else None


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConsensusProviderAuditError(f"{label} lacks required columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
