"""Build an outcome-free shadow decision and order ledger.

The first ledger version is intentionally policy-closed.  It creates one
auditable decision row for every frozen prospective event, records the exact
reason no shadow order can yet exist, and reserves strict order and position
schemas for a separately preregistered execution policy.  It never submits an
order, opens a position, loads returns, or authorizes live trading.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError
from shadow_trading_policy import (
    SHADOW_TRADING_POLICY_VERSION,
    ShadowTradingPolicyConfig,
)

SHADOW_ORDER_LEDGER_VERSION: Final[str] = "shadow-order-ledger-v2-policy-aware"

AWAITING_INNOVATION: Final[str] = "awaiting_innovation"
QUALITY_REVIEW_REQUIRED: Final[str] = "quality_review_required"
QUALITY_QUARANTINED: Final[str] = "quality_quarantined"
RESEARCH_SIGNAL_READY_POLICY_CLOSED: Final[str] = "research_signal_ready_policy_closed"
TERMINAL_REJECTED: Final[str] = "terminal_rejected"
SHADOW_BUY_INTENT_READY: Final[str] = "shadow_buy_intent_ready"
SHADOW_EXIT_OR_AVOID_INTENT_READY: Final[str] = "shadow_exit_or_avoid_intent_ready"
SIGNAL_DEAD_ZONE: Final[str] = "signal_dead_zone"


class ShadowOrderLedgerError(FilingCatalogError):
    """Raised when a shadow ledger violates its outcome or execution firewall."""


@dataclass(frozen=True)
class ShadowOrderLedgerTables:
    """Decision, block, reserved execution, gate, and report outputs."""

    decisions: pd.DataFrame
    blocks: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ShadowOrderLedgerResult:
    """One immutable shadow-ledger artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_shadow_order_ledger(
    event_states: pd.DataFrame,
    decision_as_of: str | pd.Timestamp,
    policy_report: dict[str, Any] | None = None,
    policy_events: pd.DataFrame | None = None,
) -> ShadowOrderLedgerTables:
    """Classify every event without inventing an unfrozen trading policy.

    Parameters
    ----------
    event_states:
        Atomic current states emitted by the prospective event state machine.
    decision_as_of:
        The immutable evidence cutoff reported by that state snapshot.
    """
    _require_columns(
        event_states,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "current_state",
            "state_entered_at",
            "state_effective_at",
            "rejected",
            "rejection_reason",
            "blocking_reason",
            "innovation_ready",
            "raw_innovation_computed",
            "revenue_innovation_pct",
            "consensus_quality_status",
            "consensus_quality_reasons",
            "downstream_admission",
            "consensus_quality_review_required",
            "consensus_quality_quarantined",
            "price_window_ready",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "prospective event states",
    )
    states = event_states.copy()
    if states.empty or states.duplicated("prospective_event_id").any():
        raise ShadowOrderLedgerError("Event states are empty or non-atomic")
    if states["return_values_loaded"].fillna(False).astype(bool).any():
        raise ShadowOrderLedgerError("Return values entered the shadow decision ledger")
    if states["live_trading_authorized"].fillna(False).astype(bool).any():
        raise ShadowOrderLedgerError("Live authorization entered the shadow decision ledger")
    policy = _validated_policy(states, policy_report, policy_events)

    as_of = _timestamp(decision_as_of, "decision_as_of")
    for column in ("state_entered_at", "state_effective_at"):
        states[column] = pd.to_datetime(states[column], utc=True, errors="coerce")
    if states[["state_entered_at", "state_effective_at"]].isna().any().any():
        raise ShadowOrderLedgerError("An event state lacks a valid two-clock timestamp")
    if states["state_entered_at"].gt(as_of).any():
        raise ShadowOrderLedgerError("A state was observed after the ledger cutoff")
    if states["state_effective_at"].gt(states["state_entered_at"]).any():
        raise ShadowOrderLedgerError("A state was observed before it became effective")

    decision_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    ordered = states.sort_values("registration_order", kind="mergesort")
    for raw_state in ordered.to_dict("records"):
        state = {str(key): value for key, value in raw_state.items()}
        status, block_type, reason, signal_action, target_weight = _classify_decision(state, policy)
        event_id = str(state["prospective_event_id"])
        decision_id = _stable_id("shadow-decision", event_id, as_of.isoformat())
        innovation_value = (
            float(state["revenue_innovation_pct"])
            if bool(state["innovation_ready"]) and pd.notna(state["revenue_innovation_pct"])
            else None
        )
        decision_rows.append(
            {
                "shadow_decision_id": decision_id,
                "prospective_event_id": event_id,
                "registration_order": int(state["registration_order"]),
                "cik": str(state["cik"]).zfill(10),
                "ticker": str(state["ticker"]).strip().upper(),
                "target_period_end": state["target_period_end"],
                "decision_recorded_at": as_of,
                "current_state": str(state["current_state"]),
                "state_entered_at": state["state_entered_at"],
                "state_effective_at": state["state_effective_at"],
                "consensus_quality_status": str(state["consensus_quality_status"]),
                "consensus_quality_reasons": str(state["consensus_quality_reasons"]),
                "downstream_admission": bool(state["downstream_admission"]),
                "raw_innovation_computed": bool(state["raw_innovation_computed"]),
                "innovation_ready": bool(state["innovation_ready"]),
                "revenue_innovation_pct": innovation_value,
                "decision_status": status,
                "decision_reason": reason,
                "block_type": block_type,
                "trading_policy_version": (
                    SHADOW_TRADING_POLICY_VERSION if policy is not None else None
                ),
                "trading_policy_preregistered": policy is not None,
                "signal_action": signal_action,
                "eligible_for_shadow_intent": signal_action in {"buy", "exit_or_avoid"},
                "eligible_for_shadow_order": False,
                "target_side": signal_action if signal_action != "no_trade" else None,
                "target_weight": target_weight,
                "requested_quantity": 0.0,
                "expected_entry_at": pd.NaT,
                "entry_price_type": policy.entry_price_type if policy is not None else None,
                "holding_sessions": policy.holding_sessions if policy is not None else pd.NA,
                "maximum_issuer_weight": (
                    policy.maximum_issuer_weight if policy is not None else None
                ),
                "maximum_gross_weight": (
                    policy.maximum_gross_weight if policy is not None else None
                ),
                "price_window_ready": bool(state["price_window_ready"]),
                "round_trip_cost_bps": None,
                "broker_submission_attempted": False,
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        )
        block_rows.append(
            {
                "shadow_block_id": _stable_id("shadow-block", decision_id),
                "shadow_decision_id": decision_id,
                "prospective_event_id": event_id,
                "decision_recorded_at": as_of,
                "block_type": block_type,
                "blocking_status": status,
                "blocking_reason": reason,
                "temporary": block_type in {"evidence_pending", "quality_review"},
                "manual_review_required": block_type == "quality_review",
                "policy_change_required": block_type == "policy_not_preregistered",
                "eligible_for_shadow_intent": signal_action in {"buy", "exit_or_avoid"},
                "broker_submission_attempted": False,
                "return_values_loaded": False,
            }
        )

    decisions = pd.DataFrame(decision_rows)
    blocks = pd.DataFrame(block_rows)
    orders = _empty_orders()
    positions = _empty_positions()
    all_accounted = (
        len(decisions) == len(states) and not decisions.duplicated("prospective_event_id").any()
    )
    quality_mask = decisions["consensus_quality_status"].isin(["review_required", "quarantined"])
    quality_enforced = bool((~decisions.loc[quality_mask, "eligible_for_shadow_order"]).all())
    policy_active = policy is not None
    gates = pd.DataFrame(
        [
            _gate("all_frozen_events_accounted", len(decisions), len(states), all_accounted),
            _gate(
                "quality_blocks_enforced",
                int(quality_mask.sum()),
                int(quality_mask.sum()),
                quality_enforced,
            ),
            _gate(
                "trading_policy_preregistered",
                int(policy_active),
                1,
                policy_active,
            ),
            _gate("shadow_orders_created", len(orders), 0, orders.empty),
            _gate("shadow_positions_open", len(positions), 0, positions.empty),
            _gate(
                "broker_submission_not_attempted",
                int(decisions["broker_submission_attempted"].sum()),
                0,
                not decisions["broker_submission_attempted"].any(),
            ),
            _gate(
                "no_return_values_loaded",
                int(decisions["return_values_loaded"].sum()),
                0,
                not decisions["return_values_loaded"].any(),
            ),
            _gate("live_trading_not_authorized", 0, 0, True),
        ]
    )
    status_counts = {
        str(key): int(value)
        for key, value in decisions["decision_status"].value_counts().sort_index().items()
    }
    report: dict[str, Any] = {
        "contract_version": SHADOW_ORDER_LEDGER_VERSION,
        "decision_as_of": as_of.isoformat(),
        "events": len(decisions),
        "issuers": int(decisions["cik"].nunique()),
        "decision_status_counts": status_counts,
        "blocked_events": len(blocks),
        "quality_review_events": int(
            decisions["decision_status"].eq(QUALITY_REVIEW_REQUIRED).sum()
        ),
        "quality_quarantined_events": int(
            decisions["decision_status"].eq(QUALITY_QUARANTINED).sum()
        ),
        "research_signal_ready_events": int(decisions["innovation_ready"].sum()),
        "shadow_intent_events": int(decisions["eligible_for_shadow_intent"].sum()),
        "shadow_buy_intents": int(decisions["signal_action"].eq("buy").sum()),
        "shadow_exit_or_avoid_intents": int(decisions["signal_action"].eq("exit_or_avoid").sum()),
        "signal_dead_zone_events": int(decisions["decision_status"].eq(SIGNAL_DEAD_ZONE).sum()),
        "eligible_shadow_order_events": 0,
        "shadow_orders": 0,
        "shadow_positions": 0,
        "trading_policy_preregistered": policy_active,
        "trading_policy_version": SHADOW_TRADING_POLICY_VERSION if policy_active else None,
        "broker_submission_attempted": False,
        "return_values_loaded": False,
        "live_trading_authorized": False,
        "operational_gates_passed": bool(
            gates.loc[
                ~gates["gate"].eq("trading_policy_preregistered") | policy_active,
                "passed",
            ].all()
        ),
        "paper_order_gate_passed": False,
    }
    return ShadowOrderLedgerTables(decisions, blocks, orders, positions, gates, report)


def build_shadow_order_ledger(
    event_state_root: Path,
    *,
    trading_policy_root: Path | None = None,
    output_root: Path = Path("data/processed/shadow_order_ledger"),
) -> ShadowOrderLedgerResult:
    """Persist one immutable policy-closed shadow order ledger."""
    state_manifest = _sha256_file(event_state_root / "output_manifest.json")
    state_report = _read_json(event_state_root / "prospective_event_state_machine_report.json")
    decision_as_of = state_report.get("state_as_of")
    if not decision_as_of:
        raise ShadowOrderLedgerError("State report lacks a decision cutoff")
    lineage = {
        "event_state_manifest_sha256": state_manifest,
        "trading_policy_manifest_sha256": (
            _sha256_file(trading_policy_root / "output_manifest.json")
            if trading_policy_root is not None
            else None
        ),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    payload = json.dumps(
        {
            "version": SHADOW_ORDER_LEDGER_VERSION,
            "decision_as_of": decision_as_of,
            **lineage,
        },
        sort_keys=True,
    ).encode()
    dataset_version = SHADOW_ORDER_LEDGER_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ShadowOrderLedgerResult(
            target.resolve(), _read_json(target / "shadow_order_ledger_report.json")
        )
    if target.exists():
        raise ShadowOrderLedgerError(f"Incomplete shadow ledger exists: {target}")

    tables = calculate_shadow_order_ledger(
        pd.read_parquet(event_state_root / "prospective_event_states.parquet"),
        str(decision_as_of),
        (
            _read_json(trading_policy_root / "shadow_trading_policy_report.json")
            if trading_policy_root is not None
            else None
        ),
        (
            pd.read_parquet(trading_policy_root / "shadow_trading_policy_events.parquet")
            if trading_policy_root is not None
            else None
        ),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "shadow_order_decisions.parquet": tables.decisions,
        "shadow_order_blocks.parquet": tables.blocks,
        "shadow_orders.parquet": tables.orders,
        "shadow_positions.parquet": tables.positions,
        "shadow_order_gates.parquet": tables.gates,
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
        **lineage,
        "artifacts": artifacts,
    }
    report_path = staging / "shadow_order_ledger_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **lineage,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ShadowOrderLedgerResult(target.resolve(), report)


def _classify_decision(
    state: dict[str, Any], policy: ShadowTradingPolicyConfig | None
) -> tuple[str, str, str, str, float | None]:
    if bool(state["rejected"]):
        reason = state.get("rejection_reason") or state.get("blocking_reason")
        return TERMINAL_REJECTED, "terminal", str(reason or "event_rejected"), "no_trade", None
    if bool(state["consensus_quality_quarantined"]):
        return (
            QUALITY_QUARANTINED,
            "quality_quarantine",
            f"consensus_quality:{state['consensus_quality_reasons']}",
            "no_trade",
            None,
        )
    if bool(state["consensus_quality_review_required"]):
        return (
            QUALITY_REVIEW_REQUIRED,
            "quality_review",
            f"consensus_quality:{state['consensus_quality_reasons']}",
            "no_trade",
            None,
        )
    if not bool(state["innovation_ready"]):
        reason = state.get("blocking_reason") or "innovation_not_observed"
        return AWAITING_INNOVATION, "evidence_pending", str(reason), "no_trade", None
    if policy is None:
        return (
            RESEARCH_SIGNAL_READY_POLICY_CLOSED,
            "policy_not_preregistered",
            "direction_threshold_sizing_cost_and_execution_policy_not_preregistered",
            "no_trade",
            None,
        )
    innovation = state.get("revenue_innovation_pct")
    if pd.isna(innovation):
        raise ShadowOrderLedgerError("An Innovation-ready event lacks a signal value")
    value = float(innovation)
    if value >= policy.buy_threshold:
        return (
            SHADOW_BUY_INTENT_READY,
            "market_evidence_pending",
            "official_calendar_price_cost_and_portfolio_state_required_before_order",
            "buy",
            policy.target_event_weight,
        )
    if value <= policy.exit_or_avoid_threshold:
        return (
            SHADOW_EXIT_OR_AVOID_INTENT_READY,
            "cash_only_exit_or_avoid",
            "negative_signal_never_opens_a_short_and_requires_portfolio_state_for_exit",
            "exit_or_avoid",
            0.0,
        )
    return (
        SIGNAL_DEAD_ZONE,
        "signal_dead_zone",
        "absolute_raw_revenue_innovation_below_one_percent",
        "no_trade",
        0.0,
    )


def _validated_policy(
    states: pd.DataFrame,
    report: dict[str, Any] | None,
    events: pd.DataFrame | None,
) -> ShadowTradingPolicyConfig | None:
    if report is None and events is None:
        return None
    if report is None or events is None:
        raise ShadowOrderLedgerError("Policy report and event registry must be supplied together")
    forbidden_flags = (
        "actual_values_loaded",
        "return_values_loaded",
        "selection_uses_return_values",
        "model_fitting_performed",
        "shadow_order_execution_authorized",
        "broker_submission_authorized",
        "live_trading_authorized",
    )
    if any(bool(report.get(flag)) for flag in forbidden_flags):
        raise ShadowOrderLedgerError("The shadow policy violates the execution firewall")
    if not bool(report.get("policy_gate_passed")) or not bool(
        report.get("shadow_intent_generation_authorized")
    ):
        raise ShadowOrderLedgerError("The shadow policy gate is closed")
    config_data = report.get("config")
    if not isinstance(config_data, dict):
        raise ShadowOrderLedgerError("The shadow policy lacks frozen configuration")
    config = ShadowTradingPolicyConfig(**config_data)
    config.validate()
    _require_columns(
        events,
        {
            "prospective_event_id",
            "shadow_policy_admitted",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "shadow policy events",
    )
    if events.duplicated("prospective_event_id").any():
        raise ShadowOrderLedgerError("The shadow policy event registry is non-atomic")
    if set(events["prospective_event_id"].astype(str)) != set(
        states["prospective_event_id"].astype(str)
    ):
        raise ShadowOrderLedgerError("The shadow policy and state-machine event sets differ")
    if not events["shadow_policy_admitted"].astype(bool).all():
        raise ShadowOrderLedgerError("A frozen event is not admitted by the shadow policy")
    if (
        events["return_values_loaded"].astype(bool).any()
        or events["live_trading_authorized"].astype(bool).any()
    ):
        raise ShadowOrderLedgerError("Policy event scope violates the outcome firewall")
    return config


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shadow_order_id": pd.Series(dtype="string"),
            "shadow_decision_id": pd.Series(dtype="string"),
            "prospective_event_id": pd.Series(dtype="string"),
            "created_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "side": pd.Series(dtype="string"),
            "order_type": pd.Series(dtype="string"),
            "quantity": pd.Series(dtype="float64"),
            "planned_entry_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_price_type": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "broker_submission_attempted": pd.Series(dtype="bool"),
            "live_trading_authorized": pd.Series(dtype="bool"),
        }
    )


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shadow_position_id": pd.Series(dtype="string"),
            "shadow_order_id": pd.Series(dtype="string"),
            "prospective_event_id": pd.Series(dtype="string"),
            "side": pd.Series(dtype="string"),
            "quantity": pd.Series(dtype="float64"),
            "opened_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "planned_close_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "closed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "status": pd.Series(dtype="string"),
            "return_values_loaded": pd.Series(dtype="bool"),
            "live_trading_authorized": pd.Series(dtype="bool"),
        }
    )


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _stable_id(namespace: str, *parts: str) -> str:
    payload = "|".join((SHADOW_ORDER_LEDGER_VERSION, namespace, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _timestamp(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ShadowOrderLedgerError(f"Invalid {label}")
    return pd.Timestamp(result)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ShadowOrderLedgerError(f"{label} lacks required columns: {missing}")


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
