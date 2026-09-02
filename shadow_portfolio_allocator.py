"""Allocate cash-constrained shadow order candidates without placing orders."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError
from shadow_trading_policy import ShadowTradingPolicyConfig

SHADOW_PORTFOLIO_ALLOCATOR_VERSION: Final[str] = "shadow-portfolio-allocator-v1"


class ShadowPortfolioAllocatorError(FilingCatalogError):
    """Raised when an allocation violates cash, clock, scope, or outcome controls."""


@dataclass(frozen=True)
class ShadowPortfolioAllocatorConfig:
    """Frozen paper-account accounting assumptions."""

    initial_nav_usd: float = 100_000.0
    whole_shares_only: bool = True
    reserve_half_round_trip_cost_at_entry: bool = True
    allow_short_positions: bool = False
    allow_leverage: bool = False

    def validate(self) -> None:
        """Reject changes to the first paper-account convention."""
        if tuple(asdict(self).values()) != (100_000.0, True, True, False, False):
            raise ValueError("The first shadow portfolio accounting convention is frozen")


@dataclass(frozen=True)
class ShadowPortfolioAllocatorTables:
    """Per-event allocations, order candidates, account snapshot, gates, and report."""

    allocations: pd.DataFrame
    order_candidates: pd.DataFrame
    account_snapshot: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ShadowPortfolioAllocatorResult:
    """One immutable shadow allocation artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_shadow_portfolio_allocations(
    decisions: pd.DataFrame,
    market_evidence: pd.DataFrame,
    policy_report: dict[str, Any],
    allocation_as_of: str | pd.Timestamp,
    config: ShadowPortfolioAllocatorConfig | None = None,
) -> ShadowPortfolioAllocatorTables:
    """Reserve paper cash for fully evidenced buy intentions in stable order."""
    cfg = config or ShadowPortfolioAllocatorConfig()
    cfg.validate()
    policy = _validated_policy(policy_report)
    _require_columns(
        decisions,
        {
            "shadow_decision_id",
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "decision_recorded_at",
            "decision_status",
            "signal_action",
            "eligible_for_shadow_intent",
            "target_weight",
            "broker_submission_attempted",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "shadow decisions",
    )
    _require_columns(
        market_evidence,
        {
            "prospective_event_id",
            "market_evidence_status",
            "market_evidence_blockers",
            "order_evidence_ready",
            "planned_entry_at",
            "planned_exit_at",
            "sizing_reference_price",
            "sizing_reference_observed_at",
            "round_trip_cost_bps",
            "cost_evidence_observed_at",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "shadow market evidence",
    )
    if decisions.empty or decisions.duplicated("prospective_event_id").any():
        raise ShadowPortfolioAllocatorError("Shadow decisions are empty or non-atomic")
    if market_evidence.duplicated("prospective_event_id").any():
        raise ShadowPortfolioAllocatorError("Market evidence is non-atomic")
    if set(decisions["prospective_event_id"].astype(str)) != set(
        market_evidence["prospective_event_id"].astype(str)
    ):
        raise ShadowPortfolioAllocatorError("Decision and market-evidence event sets differ")
    for frame in (decisions, market_evidence):
        if frame["return_values_loaded"].fillna(False).astype(bool).any():
            raise ShadowPortfolioAllocatorError("Return values entered allocation")
        if frame["live_trading_authorized"].fillna(False).astype(bool).any():
            raise ShadowPortfolioAllocatorError("Live authorization entered allocation")
    if decisions["broker_submission_attempted"].fillna(False).astype(bool).any():
        raise ShadowPortfolioAllocatorError("A broker submission entered allocation")

    as_of = _timestamp(allocation_as_of, "allocation_as_of")
    joined = decisions.merge(
        market_evidence,
        on="prospective_event_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_market"),
    )
    joined["planned_entry_at"] = pd.to_datetime(
        joined["planned_entry_at"], utc=True, errors="coerce"
    )
    joined = joined.sort_values(
        ["planned_entry_at", "registration_order", "prospective_event_id"],
        na_position="last",
        kind="mergesort",
    )

    reserved_cash = 0.0
    proposed_gross = 0.0
    proposed_issuers: set[str] = set()
    order_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for raw in joined.to_dict("records"):
        row = {str(key): value for key, value in raw.items()}
        action = str(row["signal_action"])
        status = "no_shadow_intent"
        reason = str(row["decision_status"])
        candidate_id: str | None = None
        requested_quantity = 0
        target_notional = 0.0
        estimated_cash = 0.0
        if bool(row["eligible_for_shadow_intent"]):
            if action == "exit_or_avoid":
                status = "exit_or_avoid_no_open_position"
                reason = "v1_account_starts_without_positions_and_shorting_is_forbidden"
            elif action != "buy":
                raise ShadowPortfolioAllocatorError(f"Unsupported shadow intent: {action}")
            elif not bool(row["order_evidence_ready"]):
                status = "market_evidence_blocked"
                reason = str(row["market_evidence_blockers"])
            else:
                (
                    status,
                    reason,
                    candidate_id,
                    requested_quantity,
                    target_notional,
                    estimated_cash,
                ) = _allocate_buy(
                    row,
                    as_of,
                    cfg,
                    policy,
                    reserved_cash,
                    proposed_gross,
                    proposed_issuers,
                )
                if candidate_id is not None:
                    reserved_cash += estimated_cash
                    proposed_gross += target_notional
                    proposed_issuers.add(str(row["cik"]))
                    order_rows.append(
                        _order_candidate(
                            row,
                            candidate_id,
                            requested_quantity,
                            target_notional,
                            estimated_cash,
                            as_of,
                        )
                    )
        allocation_rows.append(
            {
                "shadow_allocation_id": _stable_id(
                    "allocation", str(row["shadow_decision_id"]), as_of.isoformat()
                ),
                "shadow_decision_id": str(row["shadow_decision_id"]),
                "prospective_event_id": str(row["prospective_event_id"]),
                "registration_order": int(row["registration_order"]),
                "cik": str(row["cik"]),
                "ticker": str(row["ticker"]),
                "allocation_as_of": as_of,
                "signal_action": action,
                "allocation_status": status,
                "allocation_reason": reason,
                "shadow_order_candidate_id": candidate_id,
                "target_weight": _optional_float(row.get("target_weight")),
                "target_notional_usd": target_notional,
                "requested_quantity": requested_quantity,
                "estimated_entry_cash_usd": estimated_cash,
                "broker_submission_attempted": False,
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        )
    allocations = pd.DataFrame(allocation_rows)
    order_candidates = pd.DataFrame(order_rows, columns=_order_columns())
    account_id = _stable_id("paper-account", as_of.isoformat())
    account = pd.DataFrame(
        [
            {
                "paper_account_snapshot_id": account_id,
                "snapshot_at": as_of,
                "initial_nav_usd": cfg.initial_nav_usd,
                "cash_balance_usd": cfg.initial_nav_usd,
                "reserved_cash_usd": reserved_cash,
                "available_cash_usd": cfg.initial_nav_usd - reserved_cash,
                "open_position_market_value_usd": 0.0,
                "proposed_gross_notional_usd": proposed_gross,
                "proposed_gross_weight": proposed_gross / cfg.initial_nav_usd,
                "open_positions": 0,
                "order_candidates": len(order_candidates),
                "realized_pnl_usd": 0.0,
                "unrealized_pnl_usd": 0.0,
                "return_values_loaded": False,
                "broker_connected": False,
                "live_trading_authorized": False,
            }
        ]
    )
    issuer_unique = (
        not order_candidates.duplicated("cik").any() if not order_candidates.empty else True
    )
    gates = pd.DataFrame(
        [
            _gate(
                "all_decisions_accounted",
                len(allocations),
                len(decisions),
                len(allocations) == len(decisions),
            ),
            _gate(
                "market_evidence_coverage",
                len(joined),
                len(decisions),
                len(joined) == len(decisions),
            ),
            _gate(
                "cash_reservation_limit",
                round(reserved_cash),
                int(cfg.initial_nav_usd),
                reserved_cash <= cfg.initial_nav_usd,
            ),
            _gate(
                "gross_weight_limit",
                round(proposed_gross),
                round(cfg.initial_nav_usd * policy.maximum_gross_weight),
                proposed_gross <= cfg.initial_nav_usd * policy.maximum_gross_weight,
            ),
            _gate(
                "one_candidate_per_issuer",
                len(proposed_issuers),
                len(order_candidates),
                issuer_unique,
            ),
            _gate(
                "concurrent_position_limit",
                len(order_candidates),
                policy.maximum_concurrent_positions,
                len(order_candidates) <= policy.maximum_concurrent_positions,
            ),
            _gate("broker_submission_disabled", 0, 0, True),
            _gate("no_return_values_loaded", 0, 0, True),
            _gate("live_trading_not_authorized", 0, 0, True),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": SHADOW_PORTFOLIO_ALLOCATOR_VERSION,
        "allocation_as_of": as_of.isoformat(),
        "events": len(allocations),
        "shadow_intents": int(decisions["eligible_for_shadow_intent"].sum()),
        "order_candidates": len(order_candidates),
        "reserved_cash_usd": reserved_cash,
        "available_cash_usd": cfg.initial_nav_usd - reserved_cash,
        "proposed_gross_weight": proposed_gross / cfg.initial_nav_usd,
        "allocation_status_counts": {
            str(key): int(value)
            for key, value in allocations["allocation_status"].value_counts().items()
        },
        "allocation_gates_passed": bool(gates["passed"].all()),
        "orders_submitted": 0,
        "positions_opened": 0,
        "return_values_loaded": False,
        "broker_submission_authorized": False,
        "live_trading_authorized": False,
        "config": asdict(cfg),
    }
    return ShadowPortfolioAllocatorTables(allocations, order_candidates, account, gates, report)


def build_shadow_portfolio_allocations(
    shadow_ledger_root: Path,
    market_evidence_root: Path,
    trading_policy_root: Path,
    *,
    output_root: Path = Path("data/processed/shadow_portfolio_allocator"),
    config: ShadowPortfolioAllocatorConfig | None = None,
) -> ShadowPortfolioAllocatorResult:
    """Persist an immutable cash-constrained shadow allocation snapshot."""
    cfg = config or ShadowPortfolioAllocatorConfig()
    cfg.validate()
    ledger_report = _read_json(shadow_ledger_root / "shadow_order_ledger_report.json")
    allocation_as_of = ledger_report.get("decision_as_of")
    if not allocation_as_of:
        raise ShadowPortfolioAllocatorError("Shadow ledger lacks a decision cutoff")
    policy_manifest = _sha256_file(trading_policy_root / "output_manifest.json")
    if ledger_report.get("trading_policy_manifest_sha256") != policy_manifest:
        raise ShadowPortfolioAllocatorError("Shadow ledger is not tied to the supplied policy")
    lineage = {
        "shadow_ledger_manifest_sha256": _sha256_file(shadow_ledger_root / "output_manifest.json"),
        "market_evidence_manifest_sha256": _sha256_file(
            market_evidence_root / "output_manifest.json"
        ),
        "trading_policy_manifest_sha256": policy_manifest,
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    audited_market_path = market_evidence_root / "shadow_market_event_audit.parquet"
    planned_market_path = market_evidence_root / "shadow_market_evidence_events.parquet"
    market_path = audited_market_path if audited_market_path.is_file() else planned_market_path
    if not market_path.is_file():
        raise ShadowPortfolioAllocatorError("Market evidence root lacks an event audit")
    tables = calculate_shadow_portfolio_allocations(
        pd.read_parquet(shadow_ledger_root / "shadow_order_decisions.parquet"),
        pd.read_parquet(market_path),
        _read_json(trading_policy_root / "shadow_trading_policy_report.json"),
        str(allocation_as_of),
        cfg,
    )
    payload = json.dumps(
        {
            "version": SHADOW_PORTFOLIO_ALLOCATOR_VERSION,
            **lineage,
            "allocation_as_of": allocation_as_of,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        SHADOW_PORTFOLIO_ALLOCATOR_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ShadowPortfolioAllocatorResult(
            target.resolve(), _read_json(target / "shadow_portfolio_allocator_report.json")
        )
    if target.exists():
        raise ShadowPortfolioAllocatorError(f"Incomplete allocation snapshot exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "shadow_portfolio_allocations.parquet": tables.allocations,
        "shadow_order_candidates.parquet": tables.order_candidates,
        "shadow_paper_account_snapshot.parquet": tables.account_snapshot,
        "shadow_portfolio_allocation_gates.parquet": tables.gates,
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
    report_path = staging / "shadow_portfolio_allocator_report.json"
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
    return ShadowPortfolioAllocatorResult(target.resolve(), report)


def _allocate_buy(
    row: dict[str, Any],
    as_of: pd.Timestamp,
    config: ShadowPortfolioAllocatorConfig,
    policy: ShadowTradingPolicyConfig,
    reserved_cash: float,
    proposed_gross: float,
    proposed_issuers: set[str],
) -> tuple[str, str, str | None, int, float, float]:
    entry_at = _timestamp(row.get("planned_entry_at"), "planned_entry_at")
    exit_at = _timestamp(row.get("planned_exit_at"), "planned_exit_at")
    sizing_at = _timestamp(row.get("sizing_reference_observed_at"), "sizing_reference_observed_at")
    cost_at = _timestamp(row.get("cost_evidence_observed_at"), "cost_evidence_observed_at")
    if not (as_of < entry_at < exit_at and sizing_at <= as_of and cost_at <= as_of):
        raise ShadowPortfolioAllocatorError("A buy intention violates its pre-entry clocks")
    price = _positive_float(row.get("sizing_reference_price"), "sizing reference price")
    cost_bps = _positive_float(row.get("round_trip_cost_bps"), "round-trip cost")
    if not policy.minimum_round_trip_cost_bps <= cost_bps <= policy.maximum_round_trip_cost_bps:
        raise ShadowPortfolioAllocatorError("Round-trip cost lies outside the frozen policy")
    weight = _positive_float(row.get("target_weight"), "target weight")
    if weight > policy.target_event_weight or weight > policy.maximum_issuer_weight:
        raise ShadowPortfolioAllocatorError("Target weight exceeds the frozen policy")
    target_notional = config.initial_nav_usd * weight
    quantity = math.floor(target_notional / price)
    if quantity < 1:
        return "notional_too_small_for_one_share", "whole_share_quantity_is_zero", None, 0, 0.0, 0.0
    half_cost = cost_bps / 20_000.0 if config.reserve_half_round_trip_cost_at_entry else 0.0
    estimated_cash = quantity * price * (1.0 + half_cost)
    if str(row["cik"]) in proposed_issuers:
        return (
            "issuer_cap_blocked",
            "another_candidate_for_same_issuer_already_reserved",
            None,
            0,
            0.0,
            0.0,
        )
    if len(proposed_issuers) >= policy.maximum_concurrent_positions:
        return "concurrency_cap_blocked", "maximum_concurrent_positions_reached", None, 0, 0.0, 0.0
    if proposed_gross + target_notional > config.initial_nav_usd * policy.maximum_gross_weight:
        return "gross_cap_blocked", "maximum_gross_weight_reached", None, 0, 0.0, 0.0
    if reserved_cash + estimated_cash > config.initial_nav_usd:
        return "cash_blocked", "insufficient_unreserved_paper_cash", None, 0, 0.0, 0.0
    candidate_id = _stable_id("order-candidate", str(row["shadow_decision_id"]), as_of.isoformat())
    return (
        "shadow_order_candidate_ready",
        "all_pre_entry_allocation_gates_passed",
        candidate_id,
        quantity,
        target_notional,
        estimated_cash,
    )


def _order_candidate(
    row: dict[str, Any],
    candidate_id: str,
    quantity: int,
    target_notional: float,
    estimated_cash: float,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "shadow_order_candidate_id": candidate_id,
        "shadow_decision_id": str(row["shadow_decision_id"]),
        "prospective_event_id": str(row["prospective_event_id"]),
        "cik": str(row["cik"]),
        "ticker": str(row["ticker"]),
        "created_at": as_of,
        "side": "buy",
        "order_simulation_type": "market_on_open",
        "requested_quantity": quantity,
        "target_weight": float(row["target_weight"]),
        "target_notional_usd": target_notional,
        "sizing_reference_price": float(row["sizing_reference_price"]),
        "estimated_entry_cash_usd": estimated_cash,
        "planned_entry_at": row["planned_entry_at"],
        "planned_exit_at": row["planned_exit_at"],
        "round_trip_cost_bps": float(row["round_trip_cost_bps"]),
        "candidate_status": "ready_not_submitted",
        "broker_submission_attempted": False,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _validated_policy(report: dict[str, Any]) -> ShadowTradingPolicyConfig:
    if not bool(report.get("policy_gate_passed")):
        raise ShadowPortfolioAllocatorError("The shadow trading policy gate is closed")
    config_data = report.get("config")
    if not isinstance(config_data, dict):
        raise ShadowPortfolioAllocatorError("The shadow policy lacks configuration")
    config = ShadowTradingPolicyConfig(**config_data)
    config.validate()
    if config.leverage_allowed or config.short_selling_allowed or not config.cash_only:
        raise ShadowPortfolioAllocatorError("The allocation policy is not cash-only")
    return config


def _order_columns() -> list[str]:
    return [
        "shadow_order_candidate_id",
        "shadow_decision_id",
        "prospective_event_id",
        "cik",
        "ticker",
        "created_at",
        "side",
        "order_simulation_type",
        "requested_quantity",
        "target_weight",
        "target_notional_usd",
        "sizing_reference_price",
        "estimated_entry_cash_usd",
        "planned_entry_at",
        "planned_exit_at",
        "round_trip_cost_bps",
        "candidate_status",
        "broker_submission_attempted",
        "return_values_loaded",
        "live_trading_authorized",
    ]


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ShadowPortfolioAllocatorError(f"Invalid {label}")
    return pd.Timestamp(result)


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ShadowPortfolioAllocatorError(f"Invalid {label}")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stable_id(namespace: str, *parts: str) -> str:
    payload = "|".join((SHADOW_PORTFOLIO_ALLOCATOR_VERSION, namespace, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ShadowPortfolioAllocatorError(f"{label} lacks required columns: {missing}")


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
