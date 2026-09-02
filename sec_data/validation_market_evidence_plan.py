"""Freeze outcome-blind corporate-action, delisting, and benchmark evidence routes."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_MARKET_EVIDENCE_PLAN_VERSION: Final[str] = (
    "validation-market-evidence-plan-v1-outcome-blind"
)


class ValidationMarketEvidencePlanError(FilingCatalogError):
    """Raised when formal market-evidence routes are incomplete or outcome-informed."""


@dataclass(frozen=True)
class ValidationMarketEvidencePlanResult:
    """Immutable market-evidence acquisition plan."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationMarketEvidenceTables:
    """Frozen evidence targets and explicit readiness gates."""

    action_targets: pd.DataFrame
    delisting_targets: pd.DataFrame
    benchmark_targets: pd.DataFrame
    gates: pd.DataFrame
    metrics: dict[str, Any]


def calculate_validation_market_evidence_plan(
    action_targets: pd.DataFrame,
    delisting_targets: pd.DataFrame,
    benchmark_targets: pd.DataFrame,
    reference_observations: pd.DataFrame,
) -> ValidationMarketEvidenceTables:
    """Assign direct-FIGI or session-grid action routes before collection."""
    _require_columns(
        action_targets,
        {
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "security_identity_key_kind",
            "price_window_start",
            "price_window_end",
            "action_target_id",
            "required_query_kinds",
            "action_collection_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "action_acquisition_order",
        },
        "validation action targets",
    )
    _require_columns(
        delisting_targets,
        {
            "delisting_target_id",
            "cik",
            "security_identity_key",
            "delisting_collection_authorized",
            "selection_uses_later_listing_status",
            "outcome_values_loaded",
        },
        "validation delisting targets",
    )
    _require_columns(
        benchmark_targets,
        {
            "benchmark_target_id",
            "benchmark_ticker",
            "adjusted",
            "benchmark_collection_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "validation benchmark targets",
    )
    _require_columns(
        reference_observations,
        {
            "cik",
            "ticker",
            "exchange_mic",
            "currency_name",
            "share_class_figi",
            "composite_figi",
            "source_sha256",
        },
        "point-in-time reference observations",
    )
    _validate_outcome_gates(action_targets, delisting_targets, benchmark_targets)
    if action_targets["action_target_id"].duplicated().any():
        raise ValidationMarketEvidencePlanError("Action evidence targets are duplicated")
    if delisting_targets["delisting_target_id"].duplicated().any():
        raise ValidationMarketEvidencePlanError("Delisting evidence targets are duplicated")
    if len(benchmark_targets) != 1 or benchmark_targets["benchmark_ticker"].iloc[0] != "SPY":
        raise ValidationMarketEvidencePlanError("The broad-market benchmark must be one SPY target")
    if benchmark_targets["adjusted"].astype(bool).any():
        raise ValidationMarketEvidencePlanError("Benchmark prices must be unadjusted")
    observations = reference_observations.copy()
    observations["cik"] = observations["cik"].astype(str).str.zfill(10)
    enriched_rows: list[dict[str, Any]] = []
    raw_targets = action_targets.sort_values("action_acquisition_order").to_dict(orient="records")
    for raw_target in raw_targets:
        target = cast(dict[str, Any], raw_target)
        cik = str(target["cik"]).zfill(10)
        candidates = observations.loc[
            observations["cik"].eq(cik)
            & observations["ticker"].eq(str(target["selected_ticker"]))
            & observations["exchange_mic"].eq(str(target["selected_exchange_mic"]))
        ].copy()
        if str(target["security_identity_key_kind"]) == "share_class_figi":
            candidates = candidates.loc[
                candidates["share_class_figi"].eq(str(target["security_identity_key"]))
            ]
        composite_values = sorted(
            candidates["composite_figi"].dropna().astype(str).unique().tolist()
        )
        currency_values = sorted(
            candidates["currency_name"].dropna().astype(str).str.lower().unique().tolist()
        )
        share_class_values = sorted(
            candidates["share_class_figi"].dropna().astype(str).unique().tolist()
        )
        source_hashes = sorted(candidates["source_sha256"].dropna().astype(str).unique())
        if len(composite_values) > 1 or len(currency_values) > 1:
            raise ValidationMarketEvidencePlanError(
                f"Ambiguous provider action evidence for {target['selected_ticker']}"
            )
        direct = len(composite_values) == 1
        row = dict(target)
        row["selected_composite_figi"] = composite_values[0] if direct else None
        row["selected_share_class_figi"] = (
            share_class_values[0]
            if len(share_class_values) == 1
            else (
                str(target["security_identity_key"])
                if str(target["security_identity_key_kind"]) == "share_class_figi"
                else None
            )
        )
        row["currency_name"] = currency_values[0] if currency_values else "usd"
        row["currency_evidence_method"] = (
            "point_in_time_provider_reference"
            if currency_values
            else "explicit_us_market_data_configuration"
        )
        row["action_identity_evidence_source_hashes"] = "|".join(source_hashes)
        row["ticker_change_query_mode"] = (
            "composite_figi_ticker_events" if direct else "official_session_grid_reference_fallback"
        )
        row["ticker_change_direct_identifier_ready"] = direct
        row["session_grid_fallback_frozen_before_collection"] = not direct
        row["action_evidence_route_ready"] = True
        row["selection_uses_action_coverage"] = False
        row["returns_computed"] = False
        enriched_rows.append(row)
    actions = pd.DataFrame(enriched_rows)
    delistings = delisting_targets.sort_values("delisting_acquisition_order").copy()
    delistings["replacement_after_outcome_failure_allowed"] = False
    delistings["selection_uses_return_values"] = False
    delistings["returns_computed"] = False
    benchmarks = benchmark_targets.copy()
    benchmarks["benchmark_identity_method"] = "explicit_versioned_spy_specification"
    benchmarks["replacement_benchmark_allowed"] = False
    benchmarks["returns_computed"] = False
    direct_count = int(actions["ticker_change_direct_identifier_ready"].sum())
    fallback_count = len(actions) - direct_count
    gates = pd.DataFrame(
        [
            _gate(
                "action_evidence_routes",
                int(actions["action_evidence_route_ready"].sum()),
                len(actions),
                bool(actions["action_evidence_route_ready"].all()),
                "Each security has a direct FIGI or frozen session-grid continuity route",
            ),
            _gate(
                "delisting_evidence_targets",
                int(delistings["delisting_collection_authorized"].astype(bool).sum()),
                len(delistings),
                bool(delistings["delisting_collection_authorized"].astype(bool).all()),
                "Every security has a fixed post-window listing-status target",
            ),
            _gate(
                "benchmark_evidence_target",
                int(benchmarks["benchmark_collection_authorized"].astype(bool).sum()),
                1,
                bool(benchmarks["benchmark_collection_authorized"].astype(bool).all()),
                "The SPY price, split, and dividend target is explicitly frozen",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(actions),
                False,
                "Evidence routes are frozen before returns are read",
            ),
        ]
    )
    metrics: dict[str, Any] = {
        "action_targets": len(actions),
        "ticker_events_direct_targets": direct_count,
        "session_grid_fallback_targets": fallback_count,
        "delisting_targets": len(delistings),
        "benchmark_targets": len(benchmarks),
        "minimum_action_queries": direct_count * 5 + fallback_count * 4,
        "minimum_delisting_queries": len(delistings),
        "benchmark_queries": 3,
        "selection_uses_action_coverage": False,
        "selection_uses_later_listing_status": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
    }
    return ValidationMarketEvidenceTables(actions, delistings, benchmarks, gates, metrics)


def build_validation_market_evidence_plan(
    execution_plan_root: Path,
    reference_audit_root: Path,
    *,
    output_root: Path = Path("data/market/validation_market_evidence/targets"),
) -> ValidationMarketEvidencePlanResult:
    """Persist all downstream market-evidence targets and acquisition routes."""
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    reference_manifest = _sha256_file(reference_audit_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    reference_report = _read_json(reference_audit_root / "massive_reference_report.json")
    tables = calculate_validation_market_evidence_plan(
        load_catalog_table(execution_plan_root, "validation_security_action_targets.parquet"),
        load_catalog_table(execution_plan_root, "validation_security_delisting_targets.parquet"),
        load_catalog_table(execution_plan_root, "validation_benchmark_price_targets.parquet"),
        load_catalog_table(reference_audit_root, "massive_reference_observations.parquet"),
    )
    payload = json.dumps(
        {
            "version": VALIDATION_MARKET_EVIDENCE_PLAN_VERSION,
            "execution_manifest_sha256": execution_manifest,
            "reference_manifest_sha256": reference_manifest,
            "action_target_ids": tables.action_targets["action_target_id"].tolist(),
            "delisting_target_ids": tables.delisting_targets["delisting_target_id"].tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_MARKET_EVIDENCE_PLAN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationMarketEvidencePlanResult(
            target.resolve(), _read_json(target / "validation_market_evidence_plan_report.json")
        )
    if target.exists():
        raise ValidationMarketEvidencePlanError(f"Incomplete market-evidence plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_action_evidence_targets.parquet": tables.action_targets,
        "validation_delisting_evidence_targets.parquet": tables.delisting_targets,
        "validation_benchmark_evidence_targets.parquet": tables.benchmark_targets,
        "validation_market_evidence_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        **tables.metrics,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "reference_audit_version": reference_report["dataset_version"],
        "reference_audit_manifest_sha256": reference_manifest,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_market_evidence_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "execution_plan_manifest_sha256": execution_manifest,
                "reference_audit_manifest_sha256": reference_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationMarketEvidencePlanResult(target.resolve(), report)


def _validate_outcome_gates(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if (
            "selection_uses_return_values" in frame
            and frame["selection_uses_return_values"].astype(bool).any()
        ):
            raise ValidationMarketEvidencePlanError("A market target used return values")
        if "outcome_values_loaded" in frame and frame["outcome_values_loaded"].astype(bool).any():
            raise ValidationMarketEvidencePlanError("A market target opened outcomes")


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
        raise ValidationMarketEvidencePlanError(f"{label} lacks required columns: {missing}")


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
