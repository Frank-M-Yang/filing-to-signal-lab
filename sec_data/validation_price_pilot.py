"""Outcome-independent structural price pilot for validation execution spans."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_prices import load_collected_massive_daily_bars

VALIDATION_PRICE_PILOT_VERSION: Final[str] = "validation-price-pilot-v2-primary-share-class"


class ValidationPricePilotError(FilingCatalogError):
    """Raised when the structural validation-price pilot is invalid."""


class ValidationPricePilotResult:
    """Immutable target-plan or audit result."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


def calculate_validation_price_pilot_targets(
    price_targets: pd.DataFrame,
    security_components: pd.DataFrame,
) -> pd.DataFrame:
    """Select listed, OTC, and all multiclass strata without price outcomes."""
    _require_columns(
        price_targets,
        {
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "security_identity_key_kind",
            "price_window_start",
            "price_window_end",
            "adjusted",
            "price_collection_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "validation price targets",
    )
    _require_columns(
        security_components,
        {
            "cik",
            "ticker",
            "identity_key",
            "component_count",
            "identity_assignment_method",
            "outcome_values_loaded",
        },
        "validation security components",
    )
    if price_targets["selection_uses_return_values"].astype(bool).any():
        raise ValidationPricePilotError("Price-target construction used return values")
    if price_targets["outcome_values_loaded"].astype(bool).any():
        raise ValidationPricePilotError("Price-target plan opened outcome values")
    if security_components["outcome_values_loaded"].astype(bool).any():
        raise ValidationPricePilotError("Component plan opened outcome values")
    if price_targets["adjusted"].astype(bool).any():
        raise ValidationPricePilotError("Pilot prices must be unadjusted")
    if not price_targets["price_collection_authorized"].astype(bool).all():
        raise ValidationPricePilotError("Pilot includes an unauthorized price span")
    targets = price_targets.copy()
    targets["selection_sort_key"] = targets["price_target_id"].astype(str)
    multiclass_keys = set(
        security_components.loc[
            security_components["component_count"].gt(1), "identity_key"
        ].astype(str)
    )
    multiclass = targets.loc[
        targets["security_identity_key"].astype(str).isin(multiclass_keys)
    ].sort_values("selection_sort_key", kind="mergesort")
    resolved_primary_keys = set(
        security_components.loc[
            security_components["identity_assignment_method"].eq(
                "primary_share_class_by_largest_disclosed_shares"
            ),
            "identity_key",
        ].astype(str)
    )
    resolved_primary = targets.loc[
        targets["security_identity_key"].astype(str).isin(resolved_primary_keys)
    ].sort_values("selection_sort_key", kind="mergesort")
    if len(multiclass) >= 2:
        share_class_piece = multiclass.assign(pilot_stratum="complete_multiclass_basket")
    elif not resolved_primary.empty:
        share_class_piece = resolved_primary.assign(pilot_stratum="resolved_multiclass_primary")
    else:
        raise ValidationPricePilotError(
            "Pilot requires a complete basket or a pre-event primary-class resolution"
        )
    share_class_ids = set(share_class_piece["price_target_id"].astype(str))
    otc = targets.loc[
        targets["selected_exchange_mic"].eq("OTCM")
        & ~targets["price_target_id"].astype(str).isin(share_class_ids)
    ].sort_values("selection_sort_key", kind="mergesort")
    listed = targets.loc[
        ~targets["selected_exchange_mic"].eq("OTCM")
        & ~targets["price_target_id"].astype(str).isin(share_class_ids)
    ].sort_values("selection_sort_key", kind="mergesort")
    if otc.empty or listed.empty:
        raise ValidationPricePilotError("Pilot requires both OTC and listed single-security spans")
    pieces = [
        listed.head(1).assign(pilot_stratum="listed_single_security"),
        otc.head(1).assign(pilot_stratum="otc_single_security"),
        share_class_piece,
    ]
    selected = pd.concat(pieces, ignore_index=True)
    if selected["price_target_id"].duplicated().any():
        raise ValidationPricePilotError("Pilot strata selected duplicate price targets")
    selected["target_id"] = selected["price_target_id"]
    selected["pilot_selection_uses_price_coverage"] = False
    selected["pilot_selection_uses_return_values"] = False
    selected["pilot_replacement_after_failure_allowed"] = False
    selected = selected.sort_values(
        ["pilot_stratum", "selection_sort_key"], kind="mergesort"
    ).reset_index(drop=True)
    selected["price_acquisition_order"] = selected.index + 1
    return selected.drop(columns=["selection_sort_key"])


def build_validation_price_pilot_plan(
    execution_plan_root: Path,
    *,
    output_root: Path = Path("data/market/validation_price_pilot/targets"),
) -> ValidationPricePilotResult:
    """Persist the frozen structural price-pilot sample before collection."""
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    if not bool(execution_report.get("price_collection_authorized")):
        raise ValidationPricePilotError("Execution plan has not authorized price collection")
    price_targets = load_catalog_table(
        execution_plan_root, "validation_security_price_targets.parquet"
    )
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    selected = calculate_validation_price_pilot_targets(price_targets, components)
    payload = json.dumps(
        {
            "execution_manifest_sha256": execution_manifest,
            "version": VALIDATION_PRICE_PILOT_VERSION,
            "selected_price_target_ids": sorted(selected["price_target_id"].astype(str)),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_PRICE_PILOT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPricePilotResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationPricePilotError(f"Incomplete validation price pilot exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "validation_price_pilot_targets.parquet"
    selected.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(selected))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "pilot_targets": len(selected),
        "pilot_stratum_counts": selected["pilot_stratum"].value_counts().to_dict(),
        "query_adjusted": False,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "price_values_loaded": False,
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
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPricePilotResult(target.resolve(), report)


def build_validation_price_pilot_audit(
    pilot_plan_root: Path,
    execution_plan_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_price_pilot"),
    output_root: Path = Path("data/processed/validation_price_pilot"),
) -> ValidationPricePilotResult:
    """Audit raw price structure and required endpoint dates without returns."""
    plan_manifest = _sha256_file(pilot_plan_root / "output_manifest.json")
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    plan_report = _read_json(pilot_plan_root / "target_report.json")
    if plan_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationPricePilotError("Pilot plan does not tie to execution plan")
    targets = load_catalog_table(pilot_plan_root, "validation_price_pilot_targets.parquet")
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    bars, rejections, completions = load_collected_massive_daily_bars(targets, raw_root=raw_root)
    key_columns = [
        "cik",
        "selected_ticker",
        "selected_exchange_mic",
        "security_identity_key",
        "price_target_id",
        "pilot_stratum",
    ]
    mapping = targets[key_columns].rename(
        columns={
            "selected_ticker": "ticker",
            "selected_exchange_mic": "exchange_mic",
            "security_identity_key": "identity_key",
        }
    )
    endpoints = components.merge(
        mapping,
        on=["cik", "ticker", "exchange_mic", "identity_key"],
        how="inner",
        validate="many_to_one",
    )[
        [
            "security_event_component_id",
            "market_event_id",
            "cik",
            "ticker",
            "exchange_mic",
            "identity_key",
            "component_weight",
            "formal_entry_session_date",
            "formal_exit_session_date",
            "price_target_id",
            "pilot_stratum",
        ]
    ].copy()
    observed_dates = {
        str(target_id): set(group["session_date"].astype(str))
        for target_id, group in bars.groupby("price_target_id")
    }
    endpoints["entry_price_date_observed"] = endpoints.apply(
        lambda row: (
            str(row["formal_entry_session_date"])
            in observed_dates.get(str(row["price_target_id"]), set())
        ),
        axis=1,
    )
    endpoints["exit_price_date_observed"] = endpoints.apply(
        lambda row: (
            str(row["formal_exit_session_date"])
            in observed_dates.get(str(row["price_target_id"]), set())
        ),
        axis=1,
    )
    endpoints["required_endpoint_dates_ready"] = (
        endpoints["entry_price_date_observed"] & endpoints["exit_price_date_observed"]
    )
    target_audit = targets[
        [
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "pilot_stratum",
        ]
    ].merge(completions, on="price_target_id", how="left", validate="one_to_one")
    target_audit["provider_query_complete"] = target_audit["completed_at"].notna()
    target_audit["validated_bar_rows"] = (
        target_audit["price_target_id"]
        .map(bars.groupby("price_target_id").size())
        .fillna(0)
        .astype(int)
    )
    target_audit["parser_rejection_rows"] = (
        target_audit["price_target_id"]
        .map(rejections.groupby("price_target_id").size())
        .fillna(0)
        .astype(int)
    )
    endpoint_ready = endpoints.groupby("price_target_id")["required_endpoint_dates_ready"].all()
    target_audit["required_endpoint_dates_ready"] = (
        target_audit["price_target_id"].map(endpoint_ready).fillna(False).astype(bool)
    )
    target_audit["structural_price_ready"] = (
        target_audit["provider_query_complete"]
        & target_audit["validated_bar_rows"].gt(0)
        & target_audit["parser_rejection_rows"].eq(0)
        & target_audit["required_endpoint_dates_ready"]
    )
    target_audit["returns_computed"] = False
    complete = int(target_audit["provider_query_complete"].sum())
    ready = int(target_audit["structural_price_ready"].sum())
    endpoint_count = int(endpoints["required_endpoint_dates_ready"].sum())
    source_hashes = sorted(bars.get("source_sha256", pd.Series(dtype=str)).unique().tolist())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "execution_manifest_sha256": execution_manifest,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "validation-price-pilot-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPricePilotResult(
            target.resolve(), _read_json(target / "validation_price_pilot_report.json")
        )
    if target.exists():
        raise ValidationPricePilotError(f"Incomplete validation price audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "pilot_price_collection",
                complete,
                len(targets),
                complete == len(targets),
                "Every frozen structural pilot span has a preserved response",
            ),
            _gate(
                "pilot_required_endpoint_dates",
                endpoint_count,
                len(endpoints),
                endpoint_count == len(endpoints),
                "Every pilot event component contains its formal entry and exit dates",
            ),
            _gate(
                "pilot_structural_price",
                ready,
                len(targets),
                ready == len(targets),
                "Listed, OTC, and multiclass pilot strata all pass structural checks",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(endpoints),
                False,
                "No return is computed during structural price validation",
            ),
        ]
    )
    outputs = {
        "validation_price_pilot_targets.parquet": targets,
        "validation_price_pilot_bars.parquet": bars,
        "validation_price_pilot_rejections.parquet": rejections,
        "validation_price_pilot_completions.parquet": completions,
        "validation_price_pilot_event_endpoints.parquet": endpoints,
        "validation_price_pilot_target_audit.parquet": target_audit,
        "validation_price_pilot_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_plan_version": plan_report["dataset_version"],
        "pilot_plan_manifest_sha256": plan_manifest,
        "execution_plan_manifest_sha256": execution_manifest,
        "pilot_targets": len(targets),
        "targets_collected": complete,
        "validated_daily_bars": len(bars),
        "parser_rejections": len(rejections),
        "event_component_endpoints": len(endpoints),
        "event_component_endpoints_ready": endpoint_count,
        "structural_price_targets_ready": ready,
        "structural_price_gate": ready == len(targets),
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "replacement_after_collection_failure_allowed": False,
        "price_values_loaded": not bars.empty,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_price_pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "pilot_plan_manifest_sha256": plan_manifest,
                "execution_plan_manifest_sha256": execution_manifest,
                "source_hashes": source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPricePilotResult(target.resolve(), report)


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
        raise ValidationPricePilotError(f"{label} lacks required columns: {missing}")


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
