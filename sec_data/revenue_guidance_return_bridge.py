"""Outcome-blind union of primary and supplemental guidance return evidence."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import GuidanceEvidenceError

REVENUE_GUIDANCE_RETURN_BRIDGE_VERSION: Final[str] = (
    "revenue-guidance-return-bridge-v1-frozen-complete-cases"
)


@dataclass(frozen=True)
class RevenueGuidanceReturnBridgeResult:
    """Immutable union of authorized guidance events and their return ledgers."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_return_bridge(
    forecasts: pd.DataFrame,
    authorization_batches: list[pd.DataFrame],
    return_batches: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Union pre-authorized batches and restrict them to the frozen forecast IDs."""
    _require_columns(forecasts, {"validation_event_id", "cik"}, "guidance forecasts")
    if (
        forecasts["validation_event_id"].isna().any()
        or forecasts["validation_event_id"].duplicated().any()
    ):
        raise GuidanceEvidenceError("Guidance forecast event IDs must be unique")
    if not authorization_batches or len(authorization_batches) != len(return_batches):
        raise GuidanceEvidenceError("Authorization and return batches must be paired")
    forecast_ids = set(forecasts["validation_event_id"].astype(str))
    authorized_parts: list[pd.DataFrame] = []
    return_parts: list[pd.DataFrame] = []
    for batch_index, (authorization, ledger) in enumerate(
        zip(authorization_batches, return_batches, strict=True), start=1
    ):
        _validate_authorization(authorization)
        _validate_ledger(ledger)
        auth = authorization.loc[
            authorization["validation_event_id"].astype(str).isin(forecast_ids)
        ].copy()
        returns = ledger.loc[ledger["validation_event_id"].astype(str).isin(forecast_ids)].copy()
        if set(auth["validation_event_id"].astype(str)) != set(
            returns["validation_event_id"].astype(str)
        ):
            raise GuidanceEvidenceError(
                f"Return batch {batch_index} does not match its authorized guidance events"
            )
        auth["return_evidence_batch"] = batch_index
        returns["return_evidence_batch"] = batch_index
        authorized_parts.append(auth)
        return_parts.append(returns)
    authorized = pd.concat(authorized_parts, ignore_index=True, sort=False)
    ledger = pd.concat(return_parts, ignore_index=True, sort=False)
    for frame, label in ((authorized, "authorization"), (ledger, "return ledger")):
        if frame["validation_event_id"].duplicated().any():
            raise GuidanceEvidenceError(f"Combined {label} contains duplicate guidance events")
    if set(authorized["validation_event_id"].astype(str)) != set(
        ledger["validation_event_id"].astype(str)
    ):
        raise GuidanceEvidenceError("Combined authorization and return IDs differ")
    authorized = authorized.sort_values(
        ["validation_event_id", "return_evidence_batch"], kind="mergesort"
    ).reset_index(drop=True)
    ledger = ledger.sort_values(
        ["validation_event_id", "return_evidence_batch"], kind="mergesort"
    ).reset_index(drop=True)
    coverage = forecasts[["validation_event_id", "cik"]].copy()
    ready_ids = set(ledger["validation_event_id"].astype(str))
    coverage["authorized_return_ready"] = (
        coverage["validation_event_id"].astype(str).isin(ready_ids)
    )
    coverage["coverage_status"] = coverage["authorized_return_ready"].map(
        {True: "authorized_return_ready", False: "market_evidence_missing"}
    )
    coverage["selection_uses_return_values"] = False
    return authorized, ledger, coverage


def build_revenue_guidance_return_bridge(
    tournament_root: Path,
    authorization_roots: list[Path],
    return_ledger_roots: list[Path],
    *,
    output_root: Path = Path("data/processed/revenue_guidance_return_bridge"),
) -> RevenueGuidanceReturnBridgeResult:
    """Persist a manifest-tied union of independent authorized return batches."""
    if len(authorization_roots) != len(return_ledger_roots):
        raise GuidanceEvidenceError("Authorization and return root counts differ")
    input_hashes = {
        "tournament": _sha256_file(tournament_root / "output_manifest.json"),
        "authorization_batches": [
            _sha256_file(root / "output_manifest.json") for root in authorization_roots
        ],
        "return_batches": [
            _sha256_file(root / "output_manifest.json") for root in return_ledger_roots
        ],
    }
    forecasts = load_catalog_table(tournament_root, "revenue_guidance_forecasts.parquet")
    authorized, ledger, coverage = calculate_revenue_guidance_return_bridge(
        forecasts,
        [
            load_catalog_table(root, "validation_authorized_return_cohort.parquet")
            for root in authorization_roots
        ],
        [
            load_catalog_table(root, "validation_event_return_ledger.parquet")
            for root in return_ledger_roots
        ],
    )
    payload = json.dumps(
        {
            "version": REVENUE_GUIDANCE_RETURN_BRIDGE_VERSION,
            "input_manifest_sha256": input_hashes,
            "authorized_event_ids": authorized["validation_event_id"].astype(str).tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        REVENUE_GUIDANCE_RETURN_BRIDGE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceReturnBridgeResult(
            target.resolve(), _read_json(target / "revenue_guidance_return_bridge_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete revenue-guidance return bridge: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_authorized_return_cohort.parquet": authorized,
        "validation_event_return_ledger.parquet": ledger,
        "revenue_guidance_return_coverage.parquet": coverage,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_events": len(forecasts),
        "authorized_return_events": len(ledger),
        "authorized_return_issuers": int(ledger["cik"].astype(str).nunique()),
        "market_evidence_missing_events": len(forecasts) - len(ledger),
        "market_evidence_coverage": len(ledger) / len(forecasts),
        "evidence_batches": len(return_ledger_roots),
        "selection_uses_return_values": False,
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "revenue_guidance_return_bridge_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return RevenueGuidanceReturnBridgeResult(target.resolve(), report)


def _validate_authorization(frame: pd.DataFrame) -> None:
    _require_columns(
        frame,
        {
            "validation_event_id",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "authorization_uses_return_values",
            "returns_computed",
        },
        "authorized return cohort",
    )
    if not frame["return_outcome_read_authorized"].astype(bool).all():
        raise GuidanceEvidenceError("An authorization batch contains an unauthorized event")
    if (
        frame[["selection_uses_return_values", "authorization_uses_return_values"]]
        .astype(bool)
        .any()
        .any()
    ):
        raise GuidanceEvidenceError("Authorization selection used return values")
    if frame["returns_computed"].astype(bool).any():
        raise GuidanceEvidenceError("Authorization batch was created after returns")


def _validate_ledger(frame: pd.DataFrame) -> None:
    _require_columns(
        frame,
        {
            "validation_event_id",
            "cik",
            "market_excess_return_ready",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "returns_computed",
        },
        "authorized return ledger",
    )
    if (
        not frame[["market_excess_return_ready", "return_outcome_read_authorized"]]
        .astype(bool)
        .all()
        .all()
    ):
        raise GuidanceEvidenceError("A return batch contains an unresolved event")
    if frame["selection_uses_return_values"].astype(bool).any():
        raise GuidanceEvidenceError("Return ledger selection used return values")
    if not frame["returns_computed"].astype(bool).all():
        raise GuidanceEvidenceError("A return batch contains an uncomputed row")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GuidanceEvidenceError(f"{label} lacks required columns: {missing}")


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
