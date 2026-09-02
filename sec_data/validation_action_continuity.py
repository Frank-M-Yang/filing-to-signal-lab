"""Outcome-blind identity continuity repair for formal validation actions."""

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

VALIDATION_ACTION_CONTINUITY_VERSION: Final[str] = (
    "validation-action-continuity-v2-endpoints-daily-bars-same-ticker-events"
)
MIN_DAILY_BAR_COVERAGE: Final[float] = 0.95


class ValidationActionContinuityError(FilingCatalogError):
    """Raised when action identity continuity cannot be proven without outcomes."""


@dataclass(frozen=True)
class ValidationActionContinuityResult:
    """Immutable identity-continuity reconciliation bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_action_continuity(
    authorized: pd.DataFrame,
    components: pd.DataFrame,
    action_audit: pd.DataFrame,
    identities: pd.DataFrame,
    ticker_events: pd.DataFrame,
    price_dates: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    minimum_daily_bar_coverage: float = MIN_DAILY_BAR_COVERAGE,
) -> pd.DataFrame:
    """Reconcile failed ticker-event routes using endpoint identity and daily bars.

    Daily price values are not inputs. The fallback asks only whether the same
    ticker produced observations on the official sessions spanning authorized
    events. It never selects securities using returns or performance.
    """
    _validate_inputs(
        authorized,
        components,
        action_audit,
        identities,
        ticker_events,
        price_dates,
        sessions,
        minimum_daily_bar_coverage,
    )
    selected = components.merge(
        authorized[["security_event_component_id"]],
        on="security_event_component_id",
        how="inner",
        validate="one_to_one",
    )
    keys = ["cik", "ticker", "exchange_mic", "identity_key"]
    spans = (
        selected.groupby(keys, as_index=False)
        .agg(
            authorized_entry_start=("formal_entry_session_date", "min"),
            authorized_exit_end=("formal_exit_session_date", "max"),
            authorized_event_components=("security_event_component_id", "size"),
        )
        .reset_index(drop=True)
    )
    audit_base = action_audit.drop(
        columns=[
            "authorized_entry_start",
            "authorized_exit_end",
            "authorized_event_components",
        ],
        errors="ignore",
    )
    audit = audit_base.merge(
        spans,
        left_on=[
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
        ],
        right_on=keys,
        how="inner",
        validate="one_to_one",
    ).drop(columns=["ticker", "exchange_mic", "identity_key"])
    if audit.empty or len(audit) != len(action_audit):
        raise ValidationActionContinuityError("Authorized action targets do not tie to components")

    rows: list[dict[str, Any]] = []
    for target in audit.sort_values("action_acquisition_order", kind="mergesort").itertuples(
        index=False
    ):
        endpoint = _endpoint_identity(target, identities)
        ticker_event = _ticker_event_continuity(target, ticker_events)
        continuity = _daily_bar_continuity(
            target, price_dates, sessions, minimum_daily_bar_coverage
        )
        generic_ready = bool(target.historical_symbol_interval_ready)
        alternative_route_allowed = (
            bool(target.reference_start_query_complete)
            and bool(target.reference_end_query_complete)
            and (
                not bool(target.ticker_events_query_complete)
                or bool(ticker_event["ticker_events_preserve_selected_ticker"])
            )
        )
        alternative_ready = (
            alternative_route_allowed
            and bool(endpoint["endpoint_identity_stable"])
            and bool(continuity["daily_bar_continuity_ready"])
        )
        resolved_identity_ready = generic_ready or alternative_ready
        if generic_ready:
            route = "massive_ticker_events_or_session_grid"
        elif alternative_ready:
            route = (
                "same_ticker_event_stable_endpoints_plus_daily_bars"
                if bool(target.ticker_events_query_complete)
                else "stable_reference_endpoints_plus_official_session_daily_bars"
            )
        elif not bool(target.corporate_actions_ready):
            route = "blocked_incomplete_action_queries"
        elif not bool(endpoint["endpoint_identity_stable"]):
            route = "blocked_endpoint_identity_mismatch"
        else:
            route = "blocked_daily_bar_continuity"
        rows.append(
            {
                "action_target_id": str(target.action_target_id),
                "price_target_id": str(target.price_target_id),
                "cik": str(target.cik),
                "ticker": str(target.selected_ticker),
                "exchange_mic": str(target.selected_exchange_mic),
                "authorized_entry_start": str(target.authorized_entry_start),
                "authorized_exit_end": str(target.authorized_exit_end),
                "authorized_event_components": int(str(target.authorized_event_components)),
                "generic_identity_ready": generic_ready,
                "ticker_events_query_complete": bool(target.ticker_events_query_complete),
                **endpoint,
                **ticker_event,
                **continuity,
                "identity_continuity_route": route,
                "resolved_identity_ready": resolved_identity_ready,
                "historical_symbol_interval_ready": resolved_identity_ready,
                "corporate_actions_ready": bool(target.corporate_actions_ready),
                "formal_action_evidence_ready": (
                    resolved_identity_ready and bool(target.corporate_actions_ready)
                ),
                "price_values_loaded": False,
                "selection_uses_price_coverage": False,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
                "returns_computed": False,
            }
        )
    return pd.DataFrame(rows)


def build_validation_action_continuity(
    return_gate_root: Path,
    execution_plan_root: Path,
    security_actions_root: Path,
    price_audit_root: Path,
    calendar_adapter_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_action_continuity"),
    minimum_daily_bar_coverage: float = MIN_DAILY_BAR_COVERAGE,
) -> ValidationActionContinuityResult:
    """Persist the outcome-blind action identity reconciliation."""
    manifests = {
        "return_gate_manifest_sha256": _sha256_file(return_gate_root / "output_manifest.json"),
        "execution_plan_manifest_sha256": _sha256_file(
            execution_plan_root / "output_manifest.json"
        ),
        "security_actions_manifest_sha256": _sha256_file(
            security_actions_root / "output_manifest.json"
        ),
        "price_audit_manifest_sha256": _sha256_file(price_audit_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_adapter_root / "output_manifest.json"),
    }
    gate_report = _read_json(return_gate_root / "validation_return_gate_report.json")
    action_report = _read_json(security_actions_root / "massive_security_actions_report.json")
    price_report = _read_json(price_audit_root / "validation_price_batch_report.json")
    calendar_report = _read_json(calendar_adapter_root / "exchange_calendar_report.json")
    if not bool(gate_report.get("return_outcome_read_authorized")):
        raise ValidationActionContinuityError("The validation return gate is closed")
    if int(action_report.get("targets", -1)) <= 0:
        raise ValidationActionContinuityError("The action audit has no targets")
    if (
        price_report.get("execution_plan_manifest_sha256")
        != manifests["execution_plan_manifest_sha256"]
    ):
        raise ValidationActionContinuityError("Price dates and execution plan do not tie")
    if (
        calendar_report.get("execution_plan_manifest_sha256")
        != manifests["execution_plan_manifest_sha256"]
    ):
        raise ValidationActionContinuityError("Calendar and execution plan do not tie")

    status = calculate_validation_action_continuity(
        load_catalog_table(return_gate_root, "validation_authorized_return_cohort.parquet"),
        load_catalog_table(execution_plan_root, "validation_security_event_components.parquet"),
        load_catalog_table(security_actions_root, "sample_security_actions_audit.parquet"),
        load_catalog_table(
            security_actions_root, "security_reference_interval_observations.parquet"
        ),
        load_catalog_table(security_actions_root, "ticker_change_observations.parquet"),
        pd.read_parquet(
            price_audit_root / "validation_price_batch_bars.parquet",
            columns=["price_target_id", "session_date"],
        ),
        pd.read_parquet(
            calendar_adapter_root / "historical_exchange_sessions.parquet",
            columns=["exchange_mic", "session_date"],
        ),
        minimum_daily_bar_coverage=minimum_daily_bar_coverage,
    )
    payload = json.dumps(
        {
            "version": VALIDATION_ACTION_CONTINUITY_VERSION,
            **manifests,
            "minimum_daily_bar_coverage": minimum_daily_bar_coverage,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_ACTION_CONTINUITY_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationActionContinuityResult(
            target.resolve(), _read_json(target / "validation_action_continuity_report.json")
        )
    if target.exists():
        raise ValidationActionContinuityError(f"Incomplete action continuity exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    status_path = staging / "validation_action_evidence_status.parquet"
    status.to_parquet(status_path, index=False)
    gates = pd.DataFrame(
        [
            {
                "gate": "formal_action_evidence",
                "numerator": int(status["formal_action_evidence_ready"].sum()),
                "denominator": len(status),
                "coverage": float(status["formal_action_evidence_ready"].mean()),
                "passed": bool(status["formal_action_evidence_ready"].all()),
                "meaning": "Every authorized security has identity and action evidence",
            },
            {
                "gate": "return_computation",
                "numerator": 0,
                "denominator": len(status),
                "coverage": 0.0,
                "passed": False,
                "meaning": "Continuity reconciliation does not calculate returns",
            },
        ]
    )
    gates_path = staging / "validation_action_continuity_gates.parquet"
    gates.to_parquet(gates_path, index=False)
    artifacts = {
        status_path.name: _artifact_record(status_path, len(status)),
        gates_path.name: _artifact_record(gates_path, len(gates)),
    }
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "targets": len(status),
        "generic_identity_ready": int(status["generic_identity_ready"].sum()),
        "alternative_identity_ready": int(
            status["identity_continuity_route"]
            .isin(
                [
                    "stable_reference_endpoints_plus_official_session_daily_bars",
                    "same_ticker_event_stable_endpoints_plus_daily_bars",
                ]
            )
            .sum()
        ),
        "formal_action_evidence_ready": int(status["formal_action_evidence_ready"].sum()),
        "formal_action_evidence_gate": bool(status["formal_action_evidence_ready"].all()),
        "minimum_daily_bar_coverage": minimum_daily_bar_coverage,
        "price_values_loaded": False,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_action_continuity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "dataset_version": dataset_version,
        **manifests,
        "artifacts": artifacts,
        "report_sha256": _sha256_file(report_path),
    }
    (staging / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    staging.replace(target)
    return ValidationActionContinuityResult(target.resolve(), report)


def _validate_inputs(
    authorized: pd.DataFrame,
    components: pd.DataFrame,
    action_audit: pd.DataFrame,
    identities: pd.DataFrame,
    ticker_events: pd.DataFrame,
    price_dates: pd.DataFrame,
    sessions: pd.DataFrame,
    minimum_daily_bar_coverage: float,
) -> None:
    requirements = [
        (
            authorized,
            {
                "security_event_component_id",
                "return_outcome_read_authorized",
                "authorization_uses_return_values",
                "returns_computed",
            },
            "authorized cohort",
        ),
        (
            components,
            {
                "security_event_component_id",
                "cik",
                "ticker",
                "exchange_mic",
                "identity_key",
                "formal_entry_session_date",
                "formal_exit_session_date",
            },
            "execution components",
        ),
        (
            action_audit,
            {
                "action_target_id",
                "action_acquisition_order",
                "price_target_id",
                "cik",
                "selected_ticker",
                "selected_exchange_mic",
                "selected_provider_security_id",
                "selected_share_class_figi",
                "selected_composite_figi",
                "security_identity_key",
                "reference_start_query_complete",
                "reference_end_query_complete",
                "ticker_events_query_complete",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
                "selection_uses_return_values",
                "returns_computed",
            },
            "action audit",
        ),
        (
            identities,
            {
                "action_target_id",
                "query_kind",
                "ticker",
                "exchange_mic",
                "security_type",
                "provider_security_id",
                "share_class_figi",
                "composite_figi",
                "name",
                "source_sha256",
            },
            "reference identities",
        ),
        (
            ticker_events,
            {"action_target_id", "event_date", "new_ticker"},
            "ticker events",
        ),
        (price_dates, {"price_target_id", "session_date"}, "price dates"),
        (sessions, {"exchange_mic", "session_date"}, "official sessions"),
    ]
    for frame, required, label in requirements:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValidationActionContinuityError(f"{label} lacks required columns: {missing}")
    if not 0.0 < minimum_daily_bar_coverage <= 1.0:
        raise ValidationActionContinuityError("Daily-bar coverage threshold must be in (0, 1]")
    if not authorized["return_outcome_read_authorized"].astype(bool).all():
        raise ValidationActionContinuityError("The continuity sample is not authorized")
    if (
        authorized["authorization_uses_return_values"].astype(bool).any()
        or authorized["returns_computed"].astype(bool).any()
    ):
        raise ValidationActionContinuityError("Continuity authorization used returns")
    if (
        action_audit["selection_uses_return_values"].astype(bool).any()
        or action_audit["returns_computed"].astype(bool).any()
    ):
        raise ValidationActionContinuityError("Action collection used returns")


def _endpoint_identity(target: Any, identities: pd.DataFrame) -> dict[str, Any]:
    rows = identities[
        identities["action_target_id"].astype(str).eq(str(target.action_target_id))
        & identities["ticker"].astype(str).eq(str(target.selected_ticker))
        & identities["exchange_mic"].astype(str).eq(str(target.selected_exchange_mic))
        & identities["security_type"].astype(str).eq("CS")
    ]
    start = rows[rows["query_kind"].eq("reference_start")]
    end = rows[rows["query_kind"].eq("reference_end")]
    stable = len(start) == 1 and len(end) == 1
    observed_provider = observed_share = observed_composite = None
    identity_basis = "unresolved"
    source_hashes: list[str] = []
    if stable:
        left, right = start.iloc[0], end.iloc[0]
        fields = ["provider_security_id", "share_class_figi", "composite_figi"]
        left_values = {field: _clean_identifier(left[field]) for field in fields}
        right_values = {field: _clean_identifier(right[field]) for field in fields}
        expected_values = {
            field: _clean_identifier(getattr(target, f"selected_{field}")) for field in fields
        }
        identifiers_absent = all(value is None for value in left_values.values()) and all(
            value is None for value in right_values.values()
        )
        if identifiers_absent and all(value is None for value in expected_values.values()):
            left_name = _clean_identifier(left["name"])
            right_name = _clean_identifier(right["name"])
            stable = left_name is not None and left_name == right_name
            identity_basis = "stable_ticker_exchange_security_type_and_name_without_provider_ids"
        else:
            stable = all(
                left_values[field] is not None
                and left_values[field] == right_values[field]
                and (expected_values[field] is None or left_values[field] == expected_values[field])
                for field in fields
            )
            if stable:
                observed_provider = left_values["provider_security_id"]
                observed_share = left_values["share_class_figi"]
                observed_composite = left_values["composite_figi"]
                identity_basis = "stable_provider_share_class_and_composite_ids"
        if stable:
            source_hashes = sorted({str(left["source_sha256"]), str(right["source_sha256"])})
    return {
        "endpoint_identity_stable": stable,
        "resolved_provider_security_id": observed_provider,
        "resolved_share_class_figi": observed_share,
        "resolved_composite_figi": observed_composite,
        "endpoint_identity_basis": identity_basis,
        "endpoint_identity_source_hashes": "|".join(source_hashes),
    }


def _ticker_event_continuity(target: Any, ticker_events: pd.DataFrame) -> dict[str, Any]:
    events = ticker_events[
        ticker_events["action_target_id"].astype(str).eq(str(target.action_target_id))
    ].copy()
    if not events.empty:
        dates = pd.to_datetime(events["event_date"], errors="coerce")
        event_dates = dates.dt.strftime("%Y-%m-%d")
        events = events[
            event_dates.between(str(target.authorized_entry_start), str(target.authorized_exit_end))
        ]
    same_ticker = (
        bool(len(events)) and events["new_ticker"].astype(str).eq(str(target.selected_ticker)).all()
    )
    return {
        "ticker_change_events_in_authorized_span": len(events),
        "ticker_events_preserve_selected_ticker": same_ticker,
    }


def _daily_bar_continuity(
    target: Any,
    price_dates: pd.DataFrame,
    sessions: pd.DataFrame,
    minimum_daily_bar_coverage: float,
) -> dict[str, Any]:
    expected = (
        sessions[
            sessions["exchange_mic"].astype(str).eq(str(target.selected_exchange_mic))
            & sessions["session_date"]
            .astype(str)
            .between(str(target.authorized_entry_start), str(target.authorized_exit_end))
        ]["session_date"]
        .astype(str)
        .drop_duplicates()
    )
    observed = (
        price_dates[
            price_dates["price_target_id"].astype(str).eq(str(target.price_target_id))
            & price_dates["session_date"]
            .astype(str)
            .between(str(target.authorized_entry_start), str(target.authorized_exit_end))
        ]["session_date"]
        .astype(str)
        .drop_duplicates()
    )
    expected_set, observed_set = set(expected), set(observed)
    matched = len(expected_set.intersection(observed_set))
    coverage = matched / len(expected_set) if expected_set else 0.0
    boundary_ready = (
        bool(expected_set)
        and min(expected_set) in observed_set
        and max(expected_set) in observed_set
    )
    return {
        "official_sessions_expected": len(expected_set),
        "daily_bar_sessions_observed": len(observed_set),
        "daily_bar_sessions_matched": matched,
        "daily_bar_session_coverage": coverage,
        "daily_bar_boundaries_ready": boundary_ready,
        "daily_bar_continuity_ready": (coverage >= minimum_daily_bar_coverage and boundary_ready),
    }


def _clean_identifier(value: Any) -> str | None:
    result = str(value).strip()
    if not result or result.lower() in {"nan", "none", "<na>"}:
        return None
    return result


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
