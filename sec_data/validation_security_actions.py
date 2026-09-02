"""Adapter from the frozen validation evidence plan to action collection."""

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

VALIDATION_SECURITY_ACTION_PLAN_VERSION: Final[str] = (
    "validation-security-actions-plan-v1-frozen-routes"
)


class ValidationSecurityActionsError(FilingCatalogError):
    """Raised when validation action targets cannot be adapted without drift."""


@dataclass(frozen=True)
class ValidationSecurityActionsPlanResult:
    """Immutable generic-collector action target plan."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_security_action_targets(
    frozen_actions: pd.DataFrame,
    components: pd.DataFrame,
    price_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Attach traceable event-span metadata without changing frozen query routes."""
    _require_columns(
        frozen_actions,
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
            "action_acquisition_order",
            "selected_composite_figi",
            "selected_share_class_figi",
            "currency_name",
            "action_collection_authorized",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "returns_computed",
        },
        "frozen validation action targets",
    )
    _require_columns(
        components,
        {
            "security_event_component_id",
            "cik",
            "ticker",
            "exchange_mic",
            "identity_key",
            "accepted_at",
            "form",
            "accession_number",
            "formal_entry_session_date",
            "formal_exit_session_date",
        },
        "execution components",
    )
    _require_columns(
        price_targets,
        {
            "price_target_id",
            "selected_provider_security_id",
            "selected_provider_security_id_kind",
        },
        "execution price targets",
    )
    if not frozen_actions["action_collection_authorized"].astype(bool).all():
        raise ValidationSecurityActionsError("An action route is not collection-authorized")
    if frozen_actions["selection_uses_return_values"].astype(bool).any():
        raise ValidationSecurityActionsError("Action target selection used return values")
    if frozen_actions["outcome_values_loaded"].astype(bool).any():
        raise ValidationSecurityActionsError("Action target selection opened outcomes")
    if frozen_actions["returns_computed"].astype(bool).any():
        raise ValidationSecurityActionsError("Action routes were frozen after returns")
    if frozen_actions["action_target_id"].duplicated().any():
        raise ValidationSecurityActionsError("Frozen action target IDs are duplicated")

    identity_columns = ["cik", "ticker", "exchange_mic", "identity_key"]
    component_frame = components.copy()
    component_frame["accepted_value"] = pd.to_datetime(
        component_frame["accepted_at"], utc=True, errors="coerce"
    )
    if component_frame["accepted_value"].isna().any():
        raise ValidationSecurityActionsError("Execution components have invalid acceptances")
    component_frame = component_frame.sort_values(
        ["accepted_value", "security_event_component_id"], kind="mergesort"
    )
    representative = component_frame.groupby(identity_columns, as_index=False).first()
    spans = (
        component_frame.groupby(identity_columns, as_index=False)
        .agg(
            formal_entry_session_date=("formal_entry_session_date", "min"),
            formal_exit_session_date=("formal_exit_session_date", "max"),
            event_component_rows=("security_event_component_id", "size"),
        )
        .reset_index(drop=True)
    )
    metadata = representative[[*identity_columns, "accepted_at", "form", "accession_number"]].merge(
        spans, on=identity_columns, how="inner", validate="one_to_one"
    )
    targets = frozen_actions.rename(
        columns={
            "selected_ticker": "ticker",
            "selected_exchange_mic": "exchange_mic",
            "security_identity_key": "identity_key",
        }
    ).merge(metadata, on=identity_columns, how="left", validate="one_to_one")
    if targets["accepted_at"].isna().any():
        raise ValidationSecurityActionsError("A frozen action span lacks execution metadata")
    targets = targets.merge(
        price_targets[
            [
                "price_target_id",
                "selected_provider_security_id",
                "selected_provider_security_id_kind",
            ]
        ],
        on="price_target_id",
        how="left",
        validate="one_to_one",
    )
    targets["selected_provider_security_id_kind"] = targets[
        "selected_provider_security_id_kind"
    ].fillna(targets["security_identity_key_kind"])
    targets["selected_reference_source_sha256"] = targets["action_identity_evidence_source_hashes"]
    targets["target_id"] = targets["price_target_id"]
    targets["query_date"] = pd.to_datetime(targets["accepted_at"], utc=True).dt.strftime("%Y-%m-%d")
    targets = targets.rename(
        columns={
            "ticker": "selected_ticker",
            "exchange_mic": "selected_exchange_mic",
            "identity_key": "security_identity_key",
        }
    )
    columns = [
        "action_target_id",
        "action_acquisition_order",
        "price_target_id",
        "target_id",
        "cik",
        "accepted_at",
        "query_date",
        "form",
        "accession_number",
        "selected_ticker",
        "selected_exchange_mic",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
        "selected_share_class_figi",
        "selected_composite_figi",
        "selected_reference_source_sha256",
        "currency_name",
        "price_window_start",
        "price_window_end",
        "formal_entry_session_date",
        "formal_exit_session_date",
        "event_component_rows",
        "security_identity_key",
        "security_identity_key_kind",
        "ticker_change_query_mode",
        "session_grid_fallback_frozen_before_collection",
        "selection_uses_return_values",
        "outcome_values_loaded",
        "returns_computed",
    ]
    targets = targets[columns].sort_values(
        ["action_acquisition_order", "action_target_id"], kind="mergesort"
    )
    if targets["action_acquisition_order"].duplicated().any():
        raise ValidationSecurityActionsError("Action acquisition order is duplicated")
    return targets.reset_index(drop=True)


def filter_authorized_security_action_targets(
    targets: pd.DataFrame,
    authorized: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict collection to the pre-event authorized cohort without replacement."""
    _require_columns(
        authorized,
        {
            "security_event_component_id",
            "return_outcome_read_authorized",
            "authorization_uses_return_values",
            "returns_computed",
        },
        "authorized return cohort",
    )
    if not authorized["return_outcome_read_authorized"].astype(bool).all():
        raise ValidationSecurityActionsError("The action subset contains an unauthorized event")
    if authorized["authorization_uses_return_values"].astype(bool).any():
        raise ValidationSecurityActionsError("The action subset authorization used returns")
    if authorized["returns_computed"].astype(bool).any():
        raise ValidationSecurityActionsError("The action subset followed return computation")
    selected_components = components.merge(
        authorized[["security_event_component_id"]],
        on="security_event_component_id",
        how="inner",
        validate="one_to_one",
    )
    keys = selected_components[["cik", "ticker", "exchange_mic", "identity_key"]].drop_duplicates()
    selected = targets.merge(
        keys,
        left_on=[
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
        ],
        right_on=["cik", "ticker", "exchange_mic", "identity_key"],
        how="inner",
        validate="one_to_one",
    ).drop(columns=["ticker", "exchange_mic", "identity_key"])
    if selected.empty or selected["action_target_id"].duplicated().any():
        raise ValidationSecurityActionsError("Authorized action targets are empty or duplicated")
    selected = selected.sort_values(
        ["action_acquisition_order", "action_target_id"], kind="mergesort"
    ).reset_index(drop=True)
    selected["full_plan_action_acquisition_order"] = selected["action_acquisition_order"]
    selected["action_acquisition_order"] = selected.index + 1
    counts = (
        selected_components.groupby(["cik", "ticker", "exchange_mic", "identity_key"])
        .size()
        .to_dict()
    )
    selected["authorized_event_components"] = selected.apply(
        lambda row: int(
            counts[
                (
                    str(row["cik"]),
                    str(row["selected_ticker"]),
                    str(row["selected_exchange_mic"]),
                    str(row["security_identity_key"]),
                )
            ]
        ),
        axis=1,
    )
    selected["selection_uses_return_values"] = False
    selected["returns_computed"] = False
    return selected


def build_validation_security_actions_plan(
    market_evidence_plan_root: Path,
    execution_plan_root: Path,
    calendar_adapter_root: Path,
    *,
    output_root: Path = Path("data/market/validation_security_actions/targets"),
    return_gate_root: Path | None = None,
) -> ValidationSecurityActionsPlanResult:
    """Persist an adapter plan accepted by the existing audited collector."""
    evidence_manifest = _sha256_file(market_evidence_plan_root / "output_manifest.json")
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    calendar_manifest = _sha256_file(calendar_adapter_root / "output_manifest.json")
    return_gate_manifest = (
        _sha256_file(return_gate_root / "output_manifest.json")
        if return_gate_root is not None
        else None
    )
    evidence_report = _read_json(
        market_evidence_plan_root / "validation_market_evidence_plan_report.json"
    )
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    calendar_report = _read_json(calendar_adapter_root / "exchange_calendar_report.json")
    if evidence_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationSecurityActionsError("Market evidence plan does not tie to execution")
    if calendar_report.get("execution_plan_manifest_sha256") != execution_manifest:
        raise ValidationSecurityActionsError("Calendar adapter does not tie to execution")
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    targets = calculate_validation_security_action_targets(
        load_catalog_table(market_evidence_plan_root, "validation_action_evidence_targets.parquet"),
        components,
        load_catalog_table(execution_plan_root, "validation_security_price_targets.parquet"),
    )
    full_plan_targets = len(targets)
    authorized_events: int | None = None
    if return_gate_root is not None:
        return_gate_report = _read_json(return_gate_root / "validation_return_gate_report.json")
        if not bool(return_gate_report.get("return_outcome_read_authorized")):
            raise ValidationSecurityActionsError("The validation return gate is closed")
        authorized = load_catalog_table(
            return_gate_root, "validation_authorized_return_cohort.parquet"
        )
        authorized_events = len(authorized)
        targets = filter_authorized_security_action_targets(
            targets,
            authorized,
            components,
        )
    payload = json.dumps(
        {
            "version": VALIDATION_SECURITY_ACTION_PLAN_VERSION,
            "evidence_manifest_sha256": evidence_manifest,
            "execution_manifest_sha256": execution_manifest,
            "calendar_manifest_sha256": calendar_manifest,
            "return_gate_manifest_sha256": return_gate_manifest,
            "target_ids": targets["action_target_id"].astype(str).tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_SECURITY_ACTION_PLAN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationSecurityActionsPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationSecurityActionsError(
            f"Incomplete validation security-action plan exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "massive_security_actions_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifacts = {target_path.name: _artifact_record(target_path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": evidence_report["dataset_version"],
        "reference_manifest_sha256": evidence_manifest,
        "price_version": execution_report["dataset_version"],
        "price_manifest_sha256": execution_manifest,
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest,
        "validation_market_evidence_manifest_sha256": evidence_manifest,
        "execution_plan_manifest_sha256": execution_manifest,
        "return_gate_manifest_sha256": return_gate_manifest,
        "targets": len(targets),
        "full_plan_targets": full_plan_targets,
        "authorized_event_components": authorized_events,
        "queries_per_target": 5,
        "planned_queries": len(targets) * 5,
        "selection_uses_action_coverage": False,
        "selection_uses_return_values": False,
        "selection_uses_pre_event_liquidity": return_gate_root is not None,
        "outcome_values_loaded": False,
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
                "validation_market_evidence_manifest_sha256": evidence_manifest,
                "execution_plan_manifest_sha256": execution_manifest,
                "calendar_manifest_sha256": calendar_manifest,
                "return_gate_manifest_sha256": return_gate_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationSecurityActionsPlanResult(target.resolve(), report)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationSecurityActionsError(f"{label} lacks columns: {missing}")


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
