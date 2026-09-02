"""Preregister a cash-only shadow trading policy before forward outcomes exist."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

SHADOW_TRADING_POLICY_VERSION: Final[str] = "shadow-trading-policy-v1-preregistration"


class ShadowTradingPolicyError(FilingCatalogError):
    """Raised when the policy is not frozen on an outcome-blind event registry."""


@dataclass(frozen=True)
class ShadowTradingPolicyConfig:
    """Frozen signal, capital, timing, and cost rules for shadow intentions."""

    signal_name: str = "raw_revenue_market_innovation_pct"
    buy_threshold: float = 0.01
    exit_or_avoid_threshold: float = -0.01
    target_event_weight: float = 0.02
    maximum_issuer_weight: float = 0.02
    maximum_gross_weight: float = 0.20
    maximum_concurrent_positions: int = 10
    cash_only: bool = True
    short_selling_allowed: bool = False
    leverage_allowed: bool = False
    entry_session_offset: int = 2
    holding_sessions: int = 20
    entry_price_type: str = "unadjusted_open"
    exit_price_type: str = "unadjusted_close"
    order_simulation_type: str = "market_on_open"
    cost_model: str = "preacceptance_corwin_schultz_plus_fixed_slippage"
    fixed_slippage_bps: float = 5.0
    minimum_round_trip_cost_bps: float = 10.0
    maximum_round_trip_cost_bps: float = 100.0

    def validate(self) -> None:
        """Reject any silent alteration of the preregistered first policy."""
        frozen = tuple(asdict(self).values())
        expected = (
            "raw_revenue_market_innovation_pct",
            0.01,
            -0.01,
            0.02,
            0.02,
            0.20,
            10,
            True,
            False,
            False,
            2,
            20,
            "unadjusted_open",
            "unadjusted_close",
            "market_on_open",
            "preacceptance_corwin_schultz_plus_fixed_slippage",
            5.0,
            10.0,
            100.0,
        )
        if frozen != expected:
            raise ValueError("The first shadow trading policy is frozen")


@dataclass(frozen=True)
class ShadowTradingPolicyTables:
    """Frozen event scope, rule catalog, gates, protocol, and report."""

    event_registry: pd.DataFrame
    rules: pd.DataFrame
    gates: pd.DataFrame
    protocol: dict[str, Any]
    report: dict[str, Any]


@dataclass(frozen=True)
class ShadowTradingPolicyResult:
    """One immutable shadow-policy preregistration artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_shadow_trading_policy(
    prospective_events: pd.DataFrame,
    upstream_report: dict[str, Any],
    config: ShadowTradingPolicyConfig | None = None,
) -> ShadowTradingPolicyTables:
    """Freeze policy scope and rules without reading actuals or returns."""
    cfg = config or ShadowTradingPolicyConfig()
    cfg.validate()
    _require_columns(
        prospective_events,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "replacement_allowed",
            "actual_value_loaded_at_registration",
            "return_value_loaded_at_registration",
            "selection_uses_return_values",
        },
        "prospective market-Innovation events",
    )
    events = prospective_events.copy()
    if events.empty or events.duplicated("prospective_event_id").any():
        raise ShadowTradingPolicyError("The prospective event registry is empty or non-atomic")
    forbidden_report_flags = (
        "actual_values_loaded",
        "return_values_loaded",
        "selection_uses_return_values",
        "external_validation_metrics_computed",
        "portfolio_backtest_computed",
        "live_trading_authorized",
    )
    if any(bool(upstream_report.get(flag)) for flag in forbidden_report_flags):
        raise ShadowTradingPolicyError("The upstream preregistration is no longer outcome-blind")
    if not bool(upstream_report.get("preregistration_gate_passed")):
        raise ShadowTradingPolicyError("The upstream v3 preregistration gate did not pass")
    forbidden_event_flags = (
        "replacement_allowed",
        "actual_value_loaded_at_registration",
        "return_value_loaded_at_registration",
        "selection_uses_return_values",
    )
    if events[list(forbidden_event_flags)].fillna(False).astype(bool).any().any():
        raise ShadowTradingPolicyError("A registered event violates the outcome firewall")
    if int(upstream_report.get("registered_events", -1)) != len(events):
        raise ShadowTradingPolicyError("Upstream report and event registry counts differ")

    registry = events[
        [
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
        ]
    ].copy()
    registry["cik"] = registry["cik"].astype(str).str.zfill(10)
    registry["ticker"] = registry["ticker"].astype(str).str.strip().str.upper()
    registry["target_period_end"] = pd.to_datetime(
        registry["target_period_end"], utc=True, errors="coerce"
    )
    if registry["target_period_end"].isna().any():
        raise ShadowTradingPolicyError("The policy registry contains an invalid period")
    registry["shadow_policy_admitted"] = True
    registry["policy_replacement_allowed"] = False
    registry["signal_name"] = cfg.signal_name
    registry["buy_threshold"] = cfg.buy_threshold
    registry["exit_or_avoid_threshold"] = cfg.exit_or_avoid_threshold
    registry["return_values_loaded"] = False
    registry["live_trading_authorized"] = False
    registry = registry.sort_values("registration_order", kind="mergesort").reset_index(drop=True)

    rules = _rule_catalog(cfg)
    gates = pd.DataFrame(
        [
            _gate("upstream_v3_preregistration", 1, 1, True),
            _gate(
                "frozen_event_coverage", len(registry), len(events), len(registry) == len(events)
            ),
            _gate("no_event_replacement", 0, 0, not registry["policy_replacement_allowed"].any()),
            _gate("no_actual_values_loaded", 0, 0, True),
            _gate("no_return_values_loaded", 0, 0, True),
            _gate("cash_only_without_leverage", 1, 1, cfg.cash_only and not cfg.leverage_allowed),
            _gate("short_selling_disabled", 0, 0, not cfg.short_selling_allowed),
            _gate("broker_submission_disabled", 0, 0, True),
            _gate("live_trading_not_authorized", 0, 0, True),
        ]
    )
    protocol = _protocol(cfg)
    report: dict[str, Any] = {
        "contract_version": SHADOW_TRADING_POLICY_VERSION,
        "registered_events": len(registry),
        "registered_issuers": int(registry["cik"].nunique()),
        "policy_gate_passed": bool(gates["passed"].all()),
        "shadow_intent_generation_authorized": bool(gates["passed"].all()),
        "shadow_order_execution_authorized": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_fitting_performed": False,
        "broker_submission_authorized": False,
        "live_trading_authorized": False,
        "config": asdict(cfg),
    }
    return ShadowTradingPolicyTables(registry, rules, gates, protocol, report)


def build_shadow_trading_policy(
    prospective_preregistration_root: Path,
    *,
    output_root: Path = Path("data/processed/shadow_trading_policy"),
    config: ShadowTradingPolicyConfig | None = None,
) -> ShadowTradingPolicyResult:
    """Persist the immutable first shadow policy before any forward outcome."""
    cfg = config or ShadowTradingPolicyConfig()
    cfg.validate()
    upstream_manifest = _sha256_file(prospective_preregistration_root / "output_manifest.json")
    upstream_report = _read_json(
        prospective_preregistration_root / "prospective_market_innovation_report.json"
    )
    tables = calculate_shadow_trading_policy(
        pd.read_parquet(
            prospective_preregistration_root / "prospective_market_innovation_events.parquet"
        ),
        upstream_report,
        cfg,
    )
    lineage = {
        "prospective_preregistration_manifest_sha256": upstream_manifest,
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    payload = json.dumps(
        {
            "version": SHADOW_TRADING_POLICY_VERSION,
            **lineage,
            "event_ids": tables.event_registry["prospective_event_id"].tolist(),
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = SHADOW_TRADING_POLICY_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ShadowTradingPolicyResult(
            target.resolve(), _read_json(target / "shadow_trading_policy_report.json")
        )
    if target.exists():
        raise ShadowTradingPolicyError(f"Incomplete shadow policy exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "shadow_trading_policy_events.parquet": tables.event_registry,
        "shadow_trading_policy_rules.parquet": tables.rules,
        "shadow_trading_policy_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    protocol_path = staging / "shadow_trading_policy_protocol.json"
    protocol_path.write_text(
        json.dumps(tables.protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifacts[protocol_path.name] = _artifact(protocol_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **lineage,
        "artifacts": artifacts,
    }
    report_path = staging / "shadow_trading_policy_report.json"
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
    return ShadowTradingPolicyResult(target.resolve(), report)


def _rule_catalog(config: ShadowTradingPolicyConfig) -> pd.DataFrame:
    rows = [
        ("signal", config.signal_name, "No return-fitted score is allowed."),
        ("buy_threshold", config.buy_threshold, "Buy intention at or above +1%."),
        (
            "exit_or_avoid_threshold",
            config.exit_or_avoid_threshold,
            "Exit an existing long or stay in cash at or below -1%; never short.",
        ),
        ("dead_zone", "(-1%, +1%)", "No position intention inside the materiality band."),
        ("target_event_weight", config.target_event_weight, "Two percent of paper NAV."),
        ("maximum_issuer_weight", config.maximum_issuer_weight, "One event weight per issuer."),
        ("maximum_gross_weight", config.maximum_gross_weight, "At least 80% cash remains."),
        (
            "maximum_concurrent_positions",
            config.maximum_concurrent_positions,
            "No more than ten open shadow positions.",
        ),
        (
            "entry",
            "second_official_session_open_strictly_after_sec_acceptance",
            "Announcement-close execution is forbidden.",
        ),
        (
            "exit",
            "twentieth_official_session_close_including_entry_session",
            "The primary v3 holding window is retained.",
        ),
        (
            "cost",
            "preacceptance_corwin_schultz_plus_5bps_clipped_10_to_100bps",
            "Missing preacceptance cost evidence blocks the intention.",
        ),
        ("portfolio", "cash_only_long_no_leverage_no_short", "Broker submission is forbidden."),
    ]
    result = pd.DataFrame(rows, columns=["rule", "frozen_value", "rationale"])
    result["frozen_value"] = result["frozen_value"].astype(str)
    return result.assign(
        uses_actual_values=False,
        uses_return_values=False,
        live_trading_authorized=False,
    )


def _protocol(config: ShadowTradingPolicyConfig) -> dict[str, Any]:
    return {
        "protocol_version": SHADOW_TRADING_POLICY_VERSION,
        "purpose": "forward_shadow_intentions_for_research_only",
        "signal_actions": {
            "innovation_at_or_above_positive_one_percent": "buy_intent",
            "innovation_at_or_below_negative_one_percent": "exit_or_avoid_intent",
            "inside_open_interval_negative_one_to_positive_one_percent": "no_trade",
        },
        "portfolio_state_required_before_order": True,
        "official_exchange_calendar_required_before_order": True,
        "point_in_time_cost_evidence_required_before_order": True,
        "corporate_action_and_delisting_evidence_required_for_evaluation": True,
        "missing_evidence_action": "block_without_imputation",
        "broker_submission_allowed": False,
        "live_trading_allowed": False,
        "investment_recommendation_allowed": False,
        "config": asdict(config),
    }


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ShadowTradingPolicyError(f"{label} lacks required columns: {missing}")


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
