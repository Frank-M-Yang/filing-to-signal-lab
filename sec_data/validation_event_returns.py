"""Authorized point-in-time event returns for the formal validation cohort."""

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

from sec_data.event_returns import EventReturnConfig, calculate_cash_retained_holding_return
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_EVENT_RETURN_VERSION: Final[str] = (
    "validation-event-return-ledger-v2-source-tiered-authorized"
)


class ValidationEventReturnError(FilingCatalogError):
    """Raised when a formal return would violate its authorization contract."""


@dataclass(frozen=True)
class ValidationEventReturnResult:
    """Immutable validation event-return ledger and evidence bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_event_return_tables(
    authorized: pd.DataFrame,
    components: pd.DataFrame,
    price_status: pd.DataFrame,
    outcomes: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    action_audit: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_status: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    *,
    action_manifest_sha256: str,
    benchmark_manifest_sha256: str,
    config: EventReturnConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate only the event components opened by the frozen return gate.

    Issuer and benchmark values may come from the primary provider or a frozen
    secondary raw-OHLC query. The reconciliation tier and exact source hash remain
    attached to every endpoint. No event is selected using a return.
    """
    cfg = config or EventReturnConfig()
    cfg.validate()
    _validate_inputs(
        authorized,
        components,
        price_status,
        outcomes,
        issuer_bars,
        action_audit,
        benchmark_status,
    )
    sample = authorized[["security_event_component_id"]].merge(
        components,
        on="security_event_component_id",
        how="inner",
        validate="one_to_one",
    )
    sample = sample.merge(
        price_status[
            [
                "security_event_component_id",
                "price_target_id",
                "reconciled_formal_price_ready",
                "reconciled_price_source_tier",
                "returns_computed",
            ]
        ].rename(columns={"returns_computed": "price_status_returns_computed"}),
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    )
    sample = sample.merge(
        outcomes[
            [
                "security_event_component_id",
                "resolved_market_outcome_ready",
                "resolved_outcome_status",
                "delisted_on_or_before_formal_exit",
                "later_delisted_after_formal_exit",
                "terminal_value_required",
                "source_snapshot_sha256",
                "returns_computed",
            ]
        ].rename(
            columns={
                "source_snapshot_sha256": "outcome_source_sha256",
                "returns_computed": "outcome_returns_computed",
            }
        ),
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    )
    sample = sample.merge(
        action_audit[
            [
                "price_target_id",
                "action_target_id",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
            ]
        ],
        on="price_target_id",
        how="left",
        validate="many_to_one",
    )
    if sample.isna().all(axis=1).any() or len(sample) != len(authorized):
        raise ValidationEventReturnError("The authorized cohort did not join one-to-one")
    if not sample["reconciled_formal_price_ready"].astype(bool).all():
        raise ValidationEventReturnError("An authorized event lacks formal issuer prices")
    supported_source_tiers = {
        "primary_massive",
        "primary_plus_secondary_yahoo",
        "secondary_yahoo",
    }
    if not sample["reconciled_price_source_tier"].isin(supported_source_tiers).all():
        raise ValidationEventReturnError(
            "An authorized issuer endpoint uses an unsupported reconciled source tier"
        )
    if (
        sample[["price_status_returns_computed", "outcome_returns_computed"]]
        .astype(bool)
        .any()
        .any()
    ):
        raise ValidationEventReturnError("An upstream readiness table already contains returns")
    if not sample["resolved_market_outcome_ready"].astype(bool).all():
        raise ValidationEventReturnError("An authorized event lacks a resolved market outcome")
    if sample["terminal_value_required"].astype(bool).any():
        raise ValidationEventReturnError(
            "Terminal-value events require an explicit terminal cash-flow implementation"
        )
    action_ready = sample[["historical_symbol_interval_ready", "corporate_actions_ready"]].astype(
        bool
    )
    if not action_ready.all().all():
        raise ValidationEventReturnError("An authorized security lacks complete action evidence")

    ledger_rows: list[dict[str, Any]] = []
    cashflow_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    ordered = sample.sort_values(
        ["formal_entry_session_date", "security_event_component_id"], kind="mergesort"
    )
    for event in ordered.itertuples(index=False):
        event_row, event_cashflows, event_evidence = _calculate_event(
            event,
            issuer_bars,
            issuer_actions,
            benchmark_status,
            benchmark_actions,
            action_manifest_sha256,
            benchmark_manifest_sha256,
            cfg,
        )
        ledger_rows.append(event_row)
        cashflow_rows.extend(event_cashflows)
        evidence_rows.extend(event_evidence)
    ledger = pd.DataFrame(ledger_rows)
    cashflows = pd.DataFrame(cashflow_rows)
    evidence = pd.DataFrame(evidence_rows)
    if len(ledger) != len(authorized) or not ledger["market_excess_return_ready"].all():
        raise ValidationEventReturnError("Not every authorized component produced a return")
    return ledger, cashflows, evidence


def build_validation_event_return_ledger(
    return_gate_root: Path,
    execution_plan_root: Path,
    price_reconciliation_root: Path,
    price_audit_root: Path,
    terminal_evidence_root: Path,
    security_actions_root: Path,
    action_continuity_root: Path,
    benchmark_root: Path,
    *,
    price_fallback_root: Path | None = None,
    output_root: Path = Path("data/processed/validation_event_returns"),
    config: EventReturnConfig | None = None,
) -> ValidationEventReturnResult:
    """Persist the first formal validation ledger with full upstream hashes."""
    cfg = config or EventReturnConfig()
    cfg.validate()
    manifests = {
        "return_gate_manifest_sha256": _sha256_file(return_gate_root / "output_manifest.json"),
        "execution_plan_manifest_sha256": _sha256_file(
            execution_plan_root / "output_manifest.json"
        ),
        "price_reconciliation_manifest_sha256": _sha256_file(
            price_reconciliation_root / "output_manifest.json"
        ),
        "price_audit_manifest_sha256": _sha256_file(price_audit_root / "output_manifest.json"),
        "terminal_manifest_sha256": _sha256_file(terminal_evidence_root / "output_manifest.json"),
        "security_actions_manifest_sha256": _sha256_file(
            security_actions_root / "output_manifest.json"
        ),
        "action_continuity_manifest_sha256": _sha256_file(
            action_continuity_root / "output_manifest.json"
        ),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
        "price_fallback_manifest_sha256": (
            _sha256_file(price_fallback_root / "output_manifest.json")
            if price_fallback_root is not None
            else None
        ),
    }
    gate_report = _read_json(return_gate_root / "validation_return_gate_report.json")
    reconciliation_report = _read_json(
        price_reconciliation_root / "validation_price_reconciliation_report.json"
    )
    price_report = _read_json(price_audit_root / "validation_price_batch_report.json")
    action_report = _read_json(security_actions_root / "massive_security_actions_report.json")
    continuity_report = _read_json(
        action_continuity_root / "validation_action_continuity_report.json"
    )
    if not bool(gate_report.get("return_outcome_read_authorized")):
        raise ValidationEventReturnError("The formal return gate is closed")
    if bool(gate_report.get("selection_uses_return_values")) or bool(
        gate_report.get("returns_computed")
    ):
        raise ValidationEventReturnError("The authorization gate is not outcome-blind")
    if (
        gate_report.get("price_reconciliation_manifest_sha256")
        != manifests["price_reconciliation_manifest_sha256"]
    ):
        raise ValidationEventReturnError("Return gate and price reconciliation do not tie")
    if gate_report.get("terminal_manifest_sha256") != manifests["terminal_manifest_sha256"]:
        raise ValidationEventReturnError("Return gate and terminal evidence do not tie")
    if (
        reconciliation_report.get("benchmark_manifest_sha256")
        != manifests["benchmark_manifest_sha256"]
    ):
        raise ValidationEventReturnError("Price reconciliation and benchmark do not tie")
    if (
        reconciliation_report.get("fallback_manifest_sha256")
        != manifests["price_fallback_manifest_sha256"]
    ):
        raise ValidationEventReturnError("Price reconciliation and issuer fallback do not tie")
    if (
        price_report.get("execution_plan_manifest_sha256")
        != manifests["execution_plan_manifest_sha256"]
    ):
        raise ValidationEventReturnError("Issuer prices and execution plan do not tie")
    if continuity_report.get("security_actions_manifest_sha256") != manifests[
        "security_actions_manifest_sha256"
    ] or not bool(continuity_report.get("formal_action_evidence_gate")):
        raise ValidationEventReturnError("Authorized security-action evidence is incomplete")
    if bool(action_report.get("returns_computed")) or bool(
        action_report.get("selection_uses_action_coverage")
    ):
        raise ValidationEventReturnError("Security-action selection violates the frozen plan")

    ledger, cashflows, evidence = calculate_validation_event_return_tables(
        load_catalog_table(return_gate_root, "validation_authorized_return_cohort.parquet"),
        load_catalog_table(execution_plan_root, "validation_security_event_components.parquet"),
        load_catalog_table(
            price_reconciliation_root, "validation_reconciled_event_price_status.parquet"
        ),
        load_catalog_table(terminal_evidence_root, "validation_resolved_event_outcomes.parquet"),
        _load_reconciled_issuer_bars(price_audit_root, price_fallback_root),
        load_catalog_table(action_continuity_root, "validation_action_evidence_status.parquet"),
        load_catalog_table(security_actions_root, "normalized_corporate_actions.parquet"),
        load_catalog_table(
            price_reconciliation_root, "validation_reconciled_benchmark_price_status.parquet"
        ),
        load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet"),
        action_manifest_sha256=str(manifests["security_actions_manifest_sha256"]),
        benchmark_manifest_sha256=str(manifests["benchmark_manifest_sha256"]),
        config=cfg,
    )
    payload = json.dumps(
        {"version": VALIDATION_EVENT_RETURN_VERSION, **manifests, "config": asdict(cfg)},
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_EVENT_RETURN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationEventReturnResult(
            target.resolve(), _read_json(target / "validation_event_return_report.json")
        )
    if target.exists():
        raise ValidationEventReturnError(f"Incomplete validation return ledger exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_event_return_ledger.parquet": ledger,
        "validation_event_return_cashflows.parquet": cashflows,
        "validation_event_return_evidence.parquet": evidence,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "validation_event_return_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "authorized_event_components": len(ledger),
        "issuer_returns_ready": int(ledger["issuer_return_ready"].sum()),
        "market_excess_returns_ready": int(ledger["market_excess_return_ready"].sum()),
        "issuers": int(ledger["cik"].nunique()),
        "later_delisted_after_exit": int(ledger["later_delisted_after_formal_exit"].sum()),
        "issuer_action_cashflows": int(
            (
                cashflows["instrument_role"].eq("issuer")
                & cashflows["cashflow_type"].isin(["cash_dividend", "split"])
            ).sum()
        ),
        "benchmark_action_cashflows": int(
            (
                cashflows["instrument_role"].eq("market_benchmark")
                & cashflows["cashflow_type"].isin(["cash_dividend", "split"])
            ).sum()
        ),
        "formal_return_gate": bool(ledger["market_excess_return_ready"].all()),
        "entry_rule": "second_official_trading_session_open_strictly_after_acceptance",
        "exit_rule": "twentieth_official_trading_session_close_including_entry_session",
        "benchmark": "SPY cash-retained total return",
        "annualized_performance_computed": False,
        "strategy_pnl_computed": False,
        "signal_effectiveness_conclusion_allowed": False,
        "selection_uses_return_values": False,
        "returns_computed": True,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_event_return_report.json"
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
    return ValidationEventReturnResult(target.resolve(), report)


def _load_reconciled_issuer_bars(
    price_audit_root: Path,
    price_fallback_root: Path | None,
) -> pd.DataFrame:
    """Load exact issuer endpoints from the frozen primary and fallback tiers."""
    primary = load_catalog_table(price_audit_root, "validation_price_batch_bars.parquet").copy()
    primary["source_tier"] = "primary_massive"
    if price_fallback_root is None:
        return primary
    fallback_report = _read_json(price_fallback_root / "validation_price_fallback_report.json")
    if bool(fallback_report.get("outcome_values_loaded")) or bool(
        fallback_report.get("returns_computed")
    ):
        raise ValidationEventReturnError("Issuer fallback evidence violated the return gate")
    targets = load_catalog_table(price_fallback_root, "validation_price_fallback_targets.parquet")
    observations = load_catalog_table(
        price_fallback_root, "validation_price_fallback_observations.parquet"
    )
    issuer_targets = targets.loc[
        targets["consumer_kind"].isin(["issuer_formal_entry", "issuer_formal_exit"]),
        ["fallback_target_id", "primary_price_target_ids"],
    ].copy()
    if issuer_targets["primary_price_target_ids"].astype(str).str.contains("|", regex=False).any():
        raise ValidationEventReturnError("One issuer fallback endpoint maps to multiple spans")
    fallback = observations.merge(
        issuer_targets,
        on="fallback_target_id",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"primary_price_target_ids": "price_target_id"})
    if fallback["adjusted_close_used"].astype(bool).any():
        raise ValidationEventReturnError("Issuer fallback used adjusted prices")
    fallback["source_tier"] = "secondary_yahoo"
    combined = pd.concat([primary, fallback], ignore_index=True, sort=False)
    if combined.empty:
        raise ValidationEventReturnError("No issuer endpoint prices are available")
    combined["source_priority"] = combined["source_tier"].map(
        {"primary_massive": 0, "secondary_yahoo": 1}
    )
    combined = (
        combined.sort_values(
            ["price_target_id", "session_date", "source_priority"], kind="mergesort"
        )
        .drop_duplicates(["price_target_id", "session_date"], keep="first")
        .drop(columns="source_priority")
        .reset_index(drop=True)
    )
    return combined


def _calculate_event(
    event: Any,
    issuer_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_status: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    action_manifest_sha256: str,
    benchmark_manifest_sha256: str,
    config: EventReturnConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    component_id = str(event.security_event_component_id)
    price_target_id = str(event.price_target_id)
    entry_at = _aware_timestamp(event.formal_entry_open_at, "entry")
    exit_at = _aware_timestamp(event.formal_exit_close_at, "exit")
    if entry_at >= exit_at:
        raise ValidationEventReturnError("Entry must precede exit")
    entry_bar = _unique_issuer_bar(
        issuer_bars, price_target_id, str(event.formal_entry_session_date), "entry"
    )
    exit_bar = _unique_issuer_bar(
        issuer_bars, price_target_id, str(event.formal_exit_session_date), "exit"
    )
    entry_price = _positive_number(entry_bar["open_unadjusted"], "issuer entry")
    exit_price = _positive_number(exit_bar["close_unadjusted"], "issuer exit")
    security_actions = issuer_actions[
        issuer_actions["price_target_id"].astype(str).eq(price_target_id)
    ]
    holding_actions = _actions_inside_window(security_actions, entry_at, exit_at)
    issuer, issuer_details = calculate_cash_retained_holding_return(
        entry_price, exit_price, holding_actions, currency="USD"
    )

    benchmark_rows = benchmark_status[benchmark_status["target_id"].astype(str).eq(component_id)]
    entry_benchmark = _unique_benchmark_endpoint(
        benchmark_rows, str(event.formal_entry_session_date), "entry"
    )
    exit_benchmark = _unique_benchmark_endpoint(
        benchmark_rows, str(event.formal_exit_session_date), "exit"
    )
    if not bool(entry_benchmark["reconciled_benchmark_price_ready"]) or not bool(
        exit_benchmark["reconciled_benchmark_price_ready"]
    ):
        raise ValidationEventReturnError("An authorized benchmark endpoint is unresolved")
    benchmark_entry_price = _positive_number(entry_benchmark["open_unadjusted"], "benchmark entry")
    benchmark_exit_price = _positive_number(exit_benchmark["close_unadjusted"], "benchmark exit")
    market_actions = _actions_inside_window(benchmark_actions, entry_at, exit_at)
    market, market_details = calculate_cash_retained_holding_return(
        benchmark_entry_price, benchmark_exit_price, market_actions, currency="USD"
    )
    arithmetic_excess = issuer.total_return - market.total_return
    relative_wealth = (1.0 + issuer.total_return) / (1.0 + market.total_return) - 1.0
    row = {
        "security_event_component_id": component_id,
        "market_event_id": str(event.market_event_id),
        "validation_event_id": str(event.validation_event_id),
        "cik": str(event.cik),
        "ticker": str(event.ticker),
        "exchange_mic": str(event.exchange_mic),
        "accepted_at": str(event.accepted_at),
        "form": str(event.form),
        "accession_number": str(event.accession_number),
        "price_target_id": price_target_id,
        "action_target_id": str(event.action_target_id),
        "entry_session_date": str(event.formal_entry_session_date),
        "entry_at": entry_at.isoformat(),
        "entry_price_type": str(event.formal_entry_price_type),
        "entry_price_unadjusted": entry_price,
        "exit_session_date": str(event.formal_exit_session_date),
        "exit_at": exit_at.isoformat(),
        "exit_price_type": str(event.formal_exit_price_type),
        "exit_price_unadjusted": exit_price,
        "holding_sessions": config.holding_sessions,
        "issuer_actions_inside_window": len(issuer_details),
        "issuer_cumulative_split_factor": issuer.cumulative_split_factor,
        "issuer_cash_dividend_value_per_initial_share": issuer.cash_dividend_value,
        "issuer_terminal_wealth_per_initial_share": issuer.terminal_wealth,
        "issuer_total_return": issuer.total_return,
        "issuer_return_ready": True,
        "resolved_outcome_status": str(event.resolved_outcome_status),
        "later_delisted_after_formal_exit": bool(event.later_delisted_after_formal_exit),
        "market_benchmark_ticker": "SPY",
        "market_entry_price_unadjusted": benchmark_entry_price,
        "market_exit_price_unadjusted": benchmark_exit_price,
        "market_actions_inside_window": len(market_details),
        "market_total_return": market.total_return,
        "market_excess_return_arithmetic": arithmetic_excess,
        "market_relative_wealth_return": relative_wealth,
        "market_excess_return_ready": True,
        "return_outcome_read_authorized": True,
        "selection_uses_return_values": False,
        "returns_computed": True,
    }
    cashflows = _instrument_cashflows(
        component_id,
        "issuer",
        str(event.ticker),
        entry_at,
        exit_at,
        entry_price,
        exit_price,
        issuer.shares_at_exit,
        issuer_details,
    ) + _instrument_cashflows(
        component_id,
        "market_benchmark",
        "SPY",
        entry_at,
        exit_at,
        benchmark_entry_price,
        benchmark_exit_price,
        market.shares_at_exit,
        market_details,
    )
    evidence = [
        _evidence(component_id, "issuer_entry_price", str(entry_bar["source_sha256"])),
        _evidence(component_id, "issuer_exit_price", str(exit_bar["source_sha256"])),
        _evidence(
            component_id,
            "market_entry_price",
            _benchmark_source_hash(entry_benchmark),
        ),
        _evidence(
            component_id,
            "market_exit_price",
            _benchmark_source_hash(exit_benchmark),
        ),
        _evidence(component_id, "issuer_action_bundle", action_manifest_sha256),
        _evidence(component_id, "market_action_bundle", benchmark_manifest_sha256),
        _evidence(component_id, "listing_outcome", str(event.outcome_source_sha256)),
    ]
    evidence.extend(
        _evidence(component_id, "issuer_corporate_action", str(item["source_snapshot_sha256"]))
        for item in issuer_details
    )
    evidence.extend(
        _evidence(component_id, "market_corporate_action", str(item["source_snapshot_sha256"]))
        for item in market_details
    )
    return row, cashflows, evidence


def _validate_inputs(
    authorized: pd.DataFrame,
    components: pd.DataFrame,
    price_status: pd.DataFrame,
    outcomes: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    action_audit: pd.DataFrame,
    benchmark_status: pd.DataFrame,
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
                "market_event_id",
                "validation_event_id",
                "cik",
                "ticker",
                "exchange_mic",
                "accepted_at",
                "form",
                "accession_number",
                "formal_entry_session_date",
                "formal_entry_open_at",
                "formal_entry_price_type",
                "formal_exit_session_date",
                "formal_exit_close_at",
                "formal_exit_price_type",
            },
            "execution components",
        ),
        (
            price_status,
            {
                "security_event_component_id",
                "price_target_id",
                "reconciled_formal_price_ready",
                "reconciled_price_source_tier",
                "returns_computed",
            },
            "price reconciliation",
        ),
        (
            outcomes,
            {
                "security_event_component_id",
                "resolved_market_outcome_ready",
                "resolved_outcome_status",
                "delisted_on_or_before_formal_exit",
                "later_delisted_after_formal_exit",
                "terminal_value_required",
                "source_snapshot_sha256",
                "returns_computed",
            },
            "resolved outcomes",
        ),
        (
            issuer_bars,
            {
                "price_target_id",
                "session_date",
                "open_unadjusted",
                "close_unadjusted",
                "source_sha256",
            },
            "issuer bars",
        ),
        (
            action_audit,
            {
                "price_target_id",
                "action_target_id",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
            },
            "security-action audit",
        ),
        (
            benchmark_status,
            {
                "target_id",
                "required_session_date",
                "open_unadjusted",
                "close_unadjusted",
                "source_sha256",
                "fallback_source_sha256",
                "reconciled_benchmark_price_ready",
                "reconciled_price_source_tier",
                "returns_computed",
            },
            "benchmark reconciliation",
        ),
    ]
    for frame, required, label in requirements:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValidationEventReturnError(f"{label} lacks required columns: {missing}")
    if authorized.empty or authorized["security_event_component_id"].duplicated().any():
        raise ValidationEventReturnError("Authorized cohort is empty or duplicated")
    if not authorized["return_outcome_read_authorized"].astype(bool).all():
        raise ValidationEventReturnError("An event is not authorized for outcome reading")
    if (
        authorized["authorization_uses_return_values"].astype(bool).any()
        or authorized["returns_computed"].astype(bool).any()
    ):
        raise ValidationEventReturnError("Authorization is not outcome-blind")
    if benchmark_status["returns_computed"].astype(bool).any():
        raise ValidationEventReturnError("Benchmark readiness already contains returns")


def _actions_inside_window(
    actions: pd.DataFrame, entry_at: pd.Timestamp, exit_at: pd.Timestamp
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    frame = actions.copy()
    effective = pd.to_datetime(frame["effective_at"], utc=True, errors="coerce")
    if effective.isna().any():
        raise ValidationEventReturnError("A corporate action has an invalid timestamp")
    return frame[effective.gt(entry_at) & effective.le(exit_at)].copy()


def _unique_issuer_bar(
    bars: pd.DataFrame, price_target_id: str, session_date: str, label: str
) -> pd.Series[Any]:
    selected = bars[
        bars["price_target_id"].astype(str).eq(price_target_id)
        & bars["session_date"].astype(str).eq(session_date)
    ]
    if len(selected) != 1:
        raise ValidationEventReturnError(f"Issuer {label} requires one raw bar")
    return selected.iloc[0]


def _unique_benchmark_endpoint(rows: pd.DataFrame, session_date: str, label: str) -> pd.Series[Any]:
    selected = rows[rows["required_session_date"].astype(str).eq(session_date)]
    if len(selected) != 1:
        raise ValidationEventReturnError(f"Benchmark {label} requires one endpoint")
    return selected.iloc[0]


def _benchmark_source_hash(row: pd.Series[Any]) -> str:
    if str(row["reconciled_price_source_tier"]) == "primary_massive":
        value = row["source_sha256"]
    else:
        value = row["fallback_source_sha256"]
    result = str(value)
    if not result or result.lower() in {"nan", "none", "<na>"}:
        raise ValidationEventReturnError("Benchmark endpoint lacks a selected source hash")
    return result


def _instrument_cashflows(
    component_id: str,
    instrument_role: str,
    ticker: str,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    shares_at_exit: float,
    action_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "security_event_component_id": component_id,
            "instrument_role": instrument_role,
            "ticker": ticker,
            "effective_at": entry_at.isoformat(),
            "cashflow_type": "entry_purchase",
            "shares_before": 0.0,
            "shares_after": 1.0,
            "cash_flow": -entry_price,
            "source_snapshot_sha256": None,
        }
    ]
    for detail in action_details:
        rows.append(
            {
                "security_event_component_id": component_id,
                "instrument_role": instrument_role,
                "ticker": ticker,
                "effective_at": str(detail["effective_at"]),
                "cashflow_type": str(detail["action_type"]),
                "shares_before": float(detail["shares_before"]),
                "shares_after": float(detail["shares_after"]),
                "cash_flow": float(detail["cash_flow"]),
                "source_snapshot_sha256": str(detail["source_snapshot_sha256"]),
            }
        )
    rows.append(
        {
            "security_event_component_id": component_id,
            "instrument_role": instrument_role,
            "ticker": ticker,
            "effective_at": exit_at.isoformat(),
            "cashflow_type": "exit_sale",
            "shares_before": shares_at_exit,
            "shares_after": 0.0,
            "cash_flow": shares_at_exit * exit_price,
            "source_snapshot_sha256": None,
        }
    )
    return rows


def _evidence(component_id: str, evidence_type: str, source_sha256: str) -> dict[str, str]:
    return {
        "security_event_component_id": component_id,
        "evidence_type": evidence_type,
        "source_sha256": source_sha256,
    }


def _aware_timestamp(value: Any, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        raise ValidationEventReturnError(f"Formal {label} timestamp is timezone-naive")
    return timestamp


def _positive_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValidationEventReturnError(f"{label} price must be finite and positive")
    return number


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
