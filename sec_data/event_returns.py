"""Auditable cash-retained event-window returns for the Massive pilot."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

EVENT_RETURN_CONTRACT_VERSION: Final[str] = "event-return-ledger-v1"


@dataclass(frozen=True)
class EventReturnConfig:
    """Explicit execution, cash-flow, and benchmark conventions."""

    entry_session_offset: int = 2
    holding_sessions: int = 20
    entry_price_type: str = "unadjusted_regular_session_open"
    exit_price_type: str = "unadjusted_regular_session_close"
    dividend_treatment: str = "cash_retained_not_reinvested"
    split_treatment: str = "share_count_adjusted_on_effective_session_open"
    market_excess_method: str = "arithmetic_and_relative_wealth"
    sector_benchmark_status: str = "awaiting_historical_point_in_time_sector_mapping"
    version: str = EVENT_RETURN_CONTRACT_VERSION

    def validate(self) -> None:
        """Reject silent changes to the frozen v1 return convention."""
        if self.entry_price_type != "unadjusted_regular_session_open":
            raise FilingCatalogError("Event-return entry must use the unadjusted session open")
        if self.exit_price_type != "unadjusted_regular_session_close":
            raise FilingCatalogError("Event-return exit must use the unadjusted session close")
        if self.dividend_treatment != "cash_retained_not_reinvested":
            raise FilingCatalogError("Event-return v1 requires cash-retained dividends")
        if self.entry_session_offset != 2 or self.holding_sessions != 20:
            raise FilingCatalogError("Event-return v1 freezes a +2 open and 20-session hold")


@dataclass(frozen=True)
class HoldingReturnResult:
    """One-share terminal wealth after explicit corporate actions."""

    initial_shares: float
    shares_at_exit: float
    cumulative_split_factor: float
    exit_share_value: float
    cash_dividend_value: float
    terminal_wealth: float
    split_adjusted_price_return: float
    cash_dividend_return: float
    total_return: float


@dataclass(frozen=True)
class EventReturnLedgerResult:
    """Immutable event-return ledger and evidence bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_cash_retained_holding_return(
    entry_price: float,
    exit_price: float,
    actions: pd.DataFrame,
    *,
    currency: str,
) -> tuple[HoldingReturnResult, list[dict[str, Any]]]:
    """Apply ordered splits and dividends to one initial share without reinvestment.

    ``actions`` must already be restricted to events strictly after the entry open
    and no later than the exit close. Splits on the same effective timestamp are
    applied before cash dividends because quoted ex-date cash amounts are per
    post-split share.
    """
    if not _positive_finite(entry_price) or not _positive_finite(exit_price):
        raise FilingCatalogError("Entry and exit prices must be finite and positive")
    normalized_currency = str(currency).strip().upper()
    if not normalized_currency:
        raise FilingCatalogError("Holding return requires an explicit currency")
    frame = actions.copy()
    required = {
        "action_type",
        "effective_at",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
        "provider_event_id",
        "source_snapshot_sha256",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise FilingCatalogError(f"Holding actions lack required fields: {missing}")
    if not frame.empty:
        frame["effective_at_value"] = pd.to_datetime(
            frame["effective_at"], utc=True, errors="coerce"
        )
        if frame["effective_at_value"].isna().any():
            raise FilingCatalogError("Holding action has an invalid effective timestamp")
        frame["action_order"] = frame["action_type"].eq("cash_dividend").astype(int)
        frame["provider_event_sort"] = frame["provider_event_id"].fillna("").astype(str)
        frame = frame.sort_values(
            ["effective_at_value", "action_order", "provider_event_sort"],
            kind="mergesort",
        )
    shares = 1.0
    cash = 0.0
    details: list[dict[str, Any]] = []
    for action in frame.itertuples(index=False):
        shares_before = shares
        cash_flow = 0.0
        action_type = str(action.action_type)
        if action_type == "cash_dividend":
            action_currency = str(action.currency).strip().upper()
            amount = _finite_float(action.cash_amount, "cash dividend")
            if action_currency != normalized_currency or amount <= 0.0:
                raise FilingCatalogError("Dividend currency or amount is incompatible")
            cash_flow = shares * amount
            cash += cash_flow
        else:
            split_from = _finite_float(action.split_from, "split_from")
            split_to = _finite_float(action.split_to, "split_to")
            if split_from <= 0.0 or split_to <= 0.0:
                raise FilingCatalogError("Split factors must be positive")
            shares *= split_to / split_from
        details.append(
            {
                "provider_event_id": action.provider_event_id,
                "effective_at": str(action.effective_at),
                "action_type": action_type,
                "shares_before": shares_before,
                "shares_after": shares,
                "cash_flow": cash_flow,
                "currency": normalized_currency,
                "source_snapshot_sha256": str(action.source_snapshot_sha256),
            }
        )
    exit_value = shares * float(exit_price)
    terminal_wealth = exit_value + cash
    result = HoldingReturnResult(
        initial_shares=1.0,
        shares_at_exit=shares,
        cumulative_split_factor=shares,
        exit_share_value=exit_value,
        cash_dividend_value=cash,
        terminal_wealth=terminal_wealth,
        split_adjusted_price_return=exit_value / float(entry_price) - 1.0,
        cash_dividend_return=cash / float(entry_price),
        total_return=terminal_wealth / float(entry_price) - 1.0,
    )
    return result, details


def build_event_return_ledger(
    readiness_root: Path,
    price_root: Path,
    calendar_root: Path,
    security_actions_root: Path,
    delisting_root: Path,
    *,
    benchmark_root: Path | None = None,
    output_root: Path = Path("data/processed/event_return_ledger"),
    config: EventReturnConfig | None = None,
) -> EventReturnLedgerResult:
    """Build issuer and optional SPY excess returns from tied immutable evidence."""
    cfg = config or EventReturnConfig()
    cfg.validate()
    readiness_report = _read_json(readiness_root / "price_readiness_report.json")
    readiness_manifest = _sha256_file(readiness_root / "output_manifest.json")
    upstream = {
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
        "security_actions_manifest_sha256": _sha256_file(
            security_actions_root / "output_manifest.json"
        ),
        "delisting_manifest_sha256": _sha256_file(delisting_root / "output_manifest.json"),
    }
    for key, observed in upstream.items():
        if readiness_report.get(key) != observed:
            raise FilingCatalogError(f"Return ledger {key} does not match readiness")
    if not readiness_report.get("pilot_event_return_gate"):
        raise FilingCatalogError("Return ledger requires a passed pilot event-return gate")
    readiness = load_catalog_table(readiness_root, "sample_price_readiness.parquet")
    sample = readiness[readiness["return_study_eligible"].astype(bool)].copy()
    if sample.empty or len(sample) != int(readiness_report.get("return_study_eligible", -1)):
        raise FilingCatalogError("Return-ready sample is empty or inconsistent with its report")
    if sample["target_id"].duplicated().any():
        raise FilingCatalogError("Return-ready events are duplicated")
    bars = load_catalog_table(price_root, "massive_daily_price_observations.parquet")
    alignments = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
    action_audit = load_catalog_table(
        security_actions_root, "sample_security_actions_audit.parquet"
    )
    actions = load_catalog_table(security_actions_root, "normalized_corporate_actions.parquet")
    delisting = load_catalog_table(delisting_root, "event_window_delisting_outcomes.parquet")
    benchmark_report: dict[str, Any] | None = None
    benchmark_manifest: str | None = None
    benchmark_bars = pd.DataFrame()
    benchmark_actions = pd.DataFrame()
    if benchmark_root is not None:
        benchmark_report = _read_json(benchmark_root / "massive_benchmark_report.json")
        if benchmark_report.get("calendar_manifest_sha256") != upstream["calendar_manifest_sha256"]:
            raise FilingCatalogError("Market benchmark does not tie to the selected calendar")
        benchmark_manifest = _sha256_file(benchmark_root / "output_manifest.json")
        benchmark_bars = load_catalog_table(
            benchmark_root, "benchmark_daily_price_observations.parquet"
        )
        benchmark_actions = load_catalog_table(
            benchmark_root, "normalized_benchmark_actions.parquet"
        )
    align_columns = [
        "target_id",
        "price_target_id",
        "entry_calendar_source_sha256",
        "exit_calendar_source_sha256",
    ]
    sample = sample.merge(
        alignments[align_columns], on="target_id", how="left", validate="one_to_one"
    )
    sample = sample.merge(
        action_audit[
            [
                "target_id",
                "action_target_id",
                "all_security_action_queries_complete",
                "corporate_actions_ready",
            ]
        ],
        on="target_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_action"),
    )
    sample = sample.merge(
        delisting[
            [
                "target_id",
                "window_terminal_outcome_ready",
                "later_delisted_after_formal_exit",
                "missing_return_reason",
                "source_snapshot_sha256",
            ]
        ].rename(
            columns={
                "later_delisted_after_formal_exit": "outcome_later_delisted_after_formal_exit",
                "source_snapshot_sha256": "delisting_source_sha256",
            }
        ),
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    if (
        sample[
            [
                "entry_calendar_source_sha256",
                "exit_calendar_source_sha256",
                "action_target_id",
                "window_terminal_outcome_ready",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise FilingCatalogError(
            "A return-ready event lacks tied calendar, action, or outcome data"
        )
    ledger_rows: list[dict[str, Any]] = []
    cashflow_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for event in sample.sort_values(["formal_entry_session_date", "target_id"]).itertuples(
        index=False
    ):
        row, flows, event_evidence = _build_event_row(
            event,
            bars,
            actions,
            benchmark_bars,
            benchmark_actions,
            bool(benchmark_report and benchmark_report.get("benchmark_data_ready")),
            str(benchmark_report.get("benchmark_ticker", "SPY")) if benchmark_report else "SPY",
            cfg,
        )
        ledger_rows.append(row)
        cashflow_rows.extend(flows)
        evidence_rows.extend(event_evidence)
    ledger = pd.DataFrame(ledger_rows, columns=_ledger_columns())
    cashflows = pd.DataFrame(cashflow_rows, columns=_cashflow_columns())
    evidence_frame = pd.DataFrame(evidence_rows, columns=_evidence_columns())
    if len(ledger) != len(sample) or not ledger["issuer_return_ready"].astype(bool).all():
        raise FilingCatalogError("Not every eligible event produced an issuer return")
    payload = json.dumps(
        {
            "readiness_manifest_sha256": readiness_manifest,
            **upstream,
            "benchmark_manifest_sha256": benchmark_manifest,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "event-return-ledger-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return EventReturnLedgerResult(
            target.resolve(), _read_json(target / "event_return_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete event-return directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs: dict[str, pd.DataFrame] = {
        "event_return_ledger.parquet": ledger,
        "event_return_cashflows.parquet": cashflows,
        "event_return_evidence.parquet": evidence_frame,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    contract_path = staging / "event_return_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    issuer_ready = int(ledger["issuer_return_ready"].astype(bool).sum())
    market_ready = int(ledger["market_excess_return_ready"].astype(bool).sum())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_version": cfg.version,
        "readiness_version": readiness_report["dataset_version"],
        "readiness_manifest_sha256": readiness_manifest,
        **upstream,
        "benchmark_version": benchmark_report.get("dataset_version") if benchmark_report else None,
        "benchmark_manifest_sha256": benchmark_manifest,
        "events": len(ledger),
        "issuer_returns_ready": issuer_ready,
        "market_excess_returns_ready": market_ready,
        "sector_excess_returns_ready": 0,
        "cashflow_rows": len(cashflows),
        "corporate_action_cashflows": int(
            cashflows["cashflow_type"].isin(["cash_dividend", "split"]).sum()
        ),
        "pilot_issuer_return_gate": issuer_ready == len(ledger),
        "pilot_market_excess_return_gate": market_ready == len(ledger),
        "historical_sector_benchmark_gate": False,
        "historical_constituent_gate": False,
        "full_return_study_gate": False,
        "signal_effectiveness_conclusion_allowed": False,
        "annualized_performance_computed": False,
        "zero_delisting_return_imputed": False,
        "dividends_reinvested": False,
        "artifacts": artifacts,
    }
    report_path = staging / "event_return_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "readiness_manifest_sha256": readiness_manifest,
                **upstream,
                "benchmark_manifest_sha256": benchmark_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return EventReturnLedgerResult(target.resolve(), report)


def _build_event_row(
    event: Any,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    benchmark_ready: bool,
    benchmark_ticker: str,
    config: EventReturnConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_id = str(event.target_id)
    price_target_id = str(event.price_target_id)
    entry_at = pd.Timestamp(str(event.formal_entry_open_at))
    exit_at = pd.Timestamp(str(event.formal_exit_close_at))
    if entry_at.tzinfo is None or exit_at.tzinfo is None or entry_at >= exit_at:
        raise FilingCatalogError("Event return requires ordered timezone-aware execution times")
    entry_bar = _unique_bar(
        bars, price_target_id, str(event.formal_entry_session_date), "issuer entry"
    )
    exit_bar = _unique_bar(
        bars, price_target_id, str(event.formal_exit_session_date), "issuer exit"
    )
    entry_price = _matching_price(
        event.formal_entry_open_unadjusted, entry_bar["open_unadjusted"], "entry"
    )
    exit_price = _matching_price(
        event.formal_exit_close_unadjusted, exit_bar["close_unadjusted"], "exit"
    )
    event_actions = actions[actions["target_id"].eq(target_id)].copy()
    holding_actions = _actions_inside_window(event_actions, entry_at, exit_at)
    issuer, action_details = calculate_cash_retained_holding_return(
        entry_price, exit_price, holding_actions, currency="USD"
    )
    flows = _cashflow_rows(
        target_id,
        str(event.selected_ticker),
        entry_at,
        exit_at,
        entry_price,
        exit_price,
        issuer,
        action_details,
        str(entry_bar["source_sha256"]),
        str(exit_bar["source_sha256"]),
    )
    evidence = [
        _evidence(target_id, "issuer_entry_price", str(entry_bar["source_sha256"])),
        _evidence(target_id, "issuer_exit_price", str(exit_bar["source_sha256"])),
        _evidence(target_id, "entry_calendar", str(event.entry_calendar_source_sha256)),
        _evidence(target_id, "exit_calendar", str(event.exit_calendar_source_sha256)),
        _evidence(target_id, "delisting_outcome", str(event.delisting_source_sha256)),
    ]
    evidence.extend(
        _evidence(target_id, "issuer_corporate_action", item["source_snapshot_sha256"])
        for item in action_details
    )
    market_total = market_arithmetic = market_relative = np.nan
    market_status = "awaiting_market_benchmark_data"
    market_entry = market_exit = np.nan
    market_dividends = 0.0
    if benchmark_ready:
        benchmark_entry_bar = _unique_benchmark_bar(
            benchmark_bars, str(event.formal_entry_session_date), "benchmark entry"
        )
        benchmark_exit_bar = _unique_benchmark_bar(
            benchmark_bars, str(event.formal_exit_session_date), "benchmark exit"
        )
        market_entry = _finite_float(benchmark_entry_bar["open_unadjusted"], "market entry")
        market_exit = _finite_float(benchmark_exit_bar["close_unadjusted"], "market exit")
        market_actions = _actions_inside_window(benchmark_actions, entry_at, exit_at)
        market, market_action_details = calculate_cash_retained_holding_return(
            market_entry, market_exit, market_actions, currency="USD"
        )
        market_total = market.total_return
        market_dividends = market.cash_dividend_value
        market_arithmetic = issuer.total_return - market.total_return
        market_relative = (1.0 + issuer.total_return) / (1.0 + market.total_return) - 1.0
        market_status = "ready_spy_cash_retained_total_return"
        evidence.extend(
            [
                _evidence(
                    target_id,
                    "market_entry_price",
                    str(benchmark_entry_bar["source_sha256"]),
                ),
                _evidence(
                    target_id,
                    "market_exit_price",
                    str(benchmark_exit_bar["source_sha256"]),
                ),
            ]
        )
        evidence.extend(
            _evidence(target_id, "market_corporate_action", item["source_snapshot_sha256"])
            for item in market_action_details
        )
    row = {
        "target_id": target_id,
        "price_target_id": price_target_id,
        "action_target_id": str(event.action_target_id),
        "cik": str(event.cik),
        "ticker": str(event.selected_ticker),
        "security_id": str(event.stable_security_id),
        "form": str(event.form),
        "accepted_at": str(event.accepted_at),
        "entry_at": entry_at.isoformat(),
        "entry_price_type": config.entry_price_type,
        "entry_price_unadjusted": entry_price,
        "exit_at": exit_at.isoformat(),
        "exit_price_type": config.exit_price_type,
        "exit_price_unadjusted": exit_price,
        "holding_sessions": config.holding_sessions,
        "corporate_actions_inside_window": len(action_details),
        "cumulative_split_factor": issuer.cumulative_split_factor,
        "cash_dividend_value_per_initial_share": issuer.cash_dividend_value,
        "split_adjusted_exit_value_per_initial_share": issuer.exit_share_value,
        "terminal_wealth_per_initial_share": issuer.terminal_wealth,
        "split_adjusted_price_return": issuer.split_adjusted_price_return,
        "cash_dividend_return": issuer.cash_dividend_return,
        "issuer_total_return": issuer.total_return,
        "issuer_return_ready": True,
        "delisting_outcome_status": str(event.missing_return_reason),
        "later_delisted_after_formal_exit": bool(event.outcome_later_delisted_after_formal_exit),
        "market_benchmark_ticker": benchmark_ticker,
        "market_entry_price_unadjusted": market_entry,
        "market_exit_price_unadjusted": market_exit,
        "market_cash_dividend_value_per_initial_share": market_dividends,
        "market_total_return": market_total,
        "market_excess_return_arithmetic": market_arithmetic,
        "market_relative_wealth_return": market_relative,
        "market_benchmark_status": market_status,
        "market_excess_return_ready": benchmark_ready,
        "sector_benchmark_status": config.sector_benchmark_status,
        "sector_excess_return": np.nan,
        "sector_excess_return_ready": False,
    }
    return row, flows, evidence


def _actions_inside_window(
    actions: pd.DataFrame, entry_at: pd.Timestamp, exit_at: pd.Timestamp
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    frame = actions.copy()
    frame["effective_at_value"] = pd.to_datetime(frame["effective_at"], utc=True, errors="coerce")
    if frame["effective_at_value"].isna().any():
        raise FilingCatalogError("Corporate action has invalid effective_at")
    return frame[
        frame["effective_at_value"].gt(entry_at) & frame["effective_at_value"].le(exit_at)
    ].drop(columns="effective_at_value")


def _unique_bar(
    bars: pd.DataFrame, price_target_id: str, session_date: str, label: str
) -> pd.Series[Any]:
    selected = bars[
        bars["price_target_id"].eq(price_target_id) & bars["session_date"].eq(session_date)
    ]
    if len(selected) != 1:
        raise FilingCatalogError(f"{label} requires exactly one raw price bar")
    return selected.iloc[0]


def _unique_benchmark_bar(bars: pd.DataFrame, session_date: str, label: str) -> pd.Series[Any]:
    selected = bars[bars["session_date"].eq(session_date)]
    if len(selected) != 1:
        raise FilingCatalogError(f"{label} requires exactly one raw benchmark bar")
    return selected.iloc[0]


def _matching_price(expected: Any, observed: Any, label: str) -> float:
    expected_value = _finite_float(expected, f"formal {label} price")
    observed_value = _finite_float(observed, f"raw {label} price")
    if not math.isclose(expected_value, observed_value, rel_tol=0.0, abs_tol=1e-10):
        raise FilingCatalogError(f"Formal and raw {label} prices disagree")
    return observed_value


def _cashflow_rows(
    target_id: str,
    ticker: str,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    result: HoldingReturnResult,
    action_details: list[dict[str, Any]],
    entry_source: str,
    exit_source: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "target_id": target_id,
            "ticker": ticker,
            "cashflow_at": entry_at.isoformat(),
            "cashflow_type": "entry_purchase",
            "provider_event_id": None,
            "shares_before": 0.0,
            "shares_after": 1.0,
            "cash_flow_per_initial_share": -entry_price,
            "source_snapshot_sha256": entry_source,
        }
    ]
    for item in action_details:
        rows.append(
            {
                "target_id": target_id,
                "ticker": ticker,
                "cashflow_at": item["effective_at"],
                "cashflow_type": (
                    "cash_dividend" if item["action_type"] == "cash_dividend" else "split"
                ),
                "provider_event_id": item["provider_event_id"],
                "shares_before": item["shares_before"],
                "shares_after": item["shares_after"],
                "cash_flow_per_initial_share": item["cash_flow"],
                "source_snapshot_sha256": item["source_snapshot_sha256"],
            }
        )
    rows.append(
        {
            "target_id": target_id,
            "ticker": ticker,
            "cashflow_at": exit_at.isoformat(),
            "cashflow_type": "exit_sale",
            "provider_event_id": None,
            "shares_before": result.shares_at_exit,
            "shares_after": 0.0,
            "cash_flow_per_initial_share": result.shares_at_exit * exit_price,
            "source_snapshot_sha256": exit_source,
        }
    )
    return rows


def _evidence(target_id: str, role: str, source_sha256: str) -> dict[str, Any]:
    return {"target_id": target_id, "evidence_role": role, "source_sha256": source_sha256}


def _ledger_columns() -> list[str]:
    return [
        "target_id",
        "price_target_id",
        "action_target_id",
        "cik",
        "ticker",
        "security_id",
        "form",
        "accepted_at",
        "entry_at",
        "entry_price_type",
        "entry_price_unadjusted",
        "exit_at",
        "exit_price_type",
        "exit_price_unadjusted",
        "holding_sessions",
        "corporate_actions_inside_window",
        "cumulative_split_factor",
        "cash_dividend_value_per_initial_share",
        "split_adjusted_exit_value_per_initial_share",
        "terminal_wealth_per_initial_share",
        "split_adjusted_price_return",
        "cash_dividend_return",
        "issuer_total_return",
        "issuer_return_ready",
        "delisting_outcome_status",
        "later_delisted_after_formal_exit",
        "market_benchmark_ticker",
        "market_entry_price_unadjusted",
        "market_exit_price_unadjusted",
        "market_cash_dividend_value_per_initial_share",
        "market_total_return",
        "market_excess_return_arithmetic",
        "market_relative_wealth_return",
        "market_benchmark_status",
        "market_excess_return_ready",
        "sector_benchmark_status",
        "sector_excess_return",
        "sector_excess_return_ready",
    ]


def _cashflow_columns() -> list[str]:
    return [
        "target_id",
        "ticker",
        "cashflow_at",
        "cashflow_type",
        "provider_event_id",
        "shares_before",
        "shares_after",
        "cash_flow_per_initial_share",
        "source_snapshot_sha256",
    ]


def _evidence_columns() -> list[str]:
    return ["target_id", "evidence_role", "source_sha256"]


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FilingCatalogError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise FilingCatalogError(f"{label} is not finite")
    return result


def _positive_finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0.0


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
