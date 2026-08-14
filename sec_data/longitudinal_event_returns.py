"""Build multi-horizon longitudinal event returns from frozen market evidence."""

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

from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

LONGITUDINAL_RETURN_VERSION: Final[str] = "longitudinal-event-return-ledger-v1"


@dataclass(frozen=True)
class LongitudinalReturnConfig:
    """Frozen price, action, and benchmark accounting conventions."""

    entry_price_type: str = "unadjusted_regular_session_open"
    exit_price_type: str = "unadjusted_regular_session_close"
    dividend_treatment: str = "cash_retained_not_reinvested"
    split_treatment: str = "share_count_adjusted_on_effective_session_open"
    primary_horizon_sessions: int = 20
    minimum_complete_events: int = 47
    version: str = LONGITUDINAL_RETURN_VERSION

    def validate(self) -> None:
        """Reject a return definition that diverges from the pre-registration."""
        if self.entry_price_type != "unadjusted_regular_session_open":
            raise ValueError("Entry must use the unadjusted regular-session open")
        if self.exit_price_type != "unadjusted_regular_session_close":
            raise ValueError("Exit must use the unadjusted regular-session close")
        if self.dividend_treatment != "cash_retained_not_reinvested":
            raise ValueError("Dividends must remain cash-retained")
        if self.primary_horizon_sessions != 20:
            raise ValueError("The primary horizon is frozen at 20 sessions")
        if self.minimum_complete_events < 20:
            raise ValueError("The return gate cannot use fewer than 20 events")


@dataclass(frozen=True)
class LongitudinalReturnTables:
    """Pure event ledger and readiness tables before persistence."""

    ledger: pd.DataFrame
    cashflows: pd.DataFrame
    readiness: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class LongitudinalReturnResult:
    """Immutable longitudinal event-return bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_longitudinal_event_returns(
    event_horizons: pd.DataFrame,
    event_market_map: pd.DataFrame,
    event_registry: pd.DataFrame,
    price_targets: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    action_audit: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    delisting_outcomes: pd.DataFrame,
    *,
    benchmark_ready: bool,
    config: LongitudinalReturnConfig | None = None,
) -> LongitudinalReturnTables:
    """Calculate auditable 1-, 5-, and 20-session returns without dropping failures."""
    cfg = config or LongitudinalReturnConfig()
    cfg.validate()
    _validate_unique(event_horizons, ["longitudinal_event_id", "horizon_sessions"], "horizons")
    _validate_unique(event_market_map, ["longitudinal_event_id"], "event market map")
    _validate_unique(event_registry, ["longitudinal_event_id"], "event registry")
    _validate_unique(price_targets, ["price_target_id"], "price targets")
    _validate_unique(action_audit, ["action_target_id"], "action audit")
    _validate_unique(delisting_outcomes, ["cik"], "delisting outcomes")

    registry_columns = [
        "longitudinal_event_id",
        "gm_level",
        "gm_delta_yoy",
        "gm_innovation_raw",
        "gm_innovation_std",
        "analysis_sample",
    ]
    frame = event_horizons.merge(
        event_market_map[
            [
                "longitudinal_event_id",
                "price_target_id",
                "action_target_id",
                "representation_features_complete",
            ]
        ],
        on="longitudinal_event_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        event_registry[registry_columns],
        on="longitudinal_event_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        price_targets[
            [
                "price_target_id",
                "selected_ticker",
                "selected_provider_security_id",
                "currency_name",
            ]
        ],
        on="price_target_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        action_audit[
            [
                "action_target_id",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
                "historical_symbol_interval_status",
                "corporate_action_status",
            ]
        ],
        on="action_target_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        delisting_outcomes,
        on="cik",
        how="left",
        validate="many_to_one",
    )
    if frame[["price_target_id", "action_target_id", "selected_ticker"]].isna().any().any():
        raise FilingCatalogError("A frozen event lacks a market target mapping")

    ledger_rows: list[dict[str, Any]] = []
    cashflow_rows: list[dict[str, Any]] = []
    for event in frame.sort_values(
        ["accepted_at", "cik", "horizon_sessions"], kind="mergesort"
    ).itertuples(index=False):
        row, flows = _event_return_row(
            event,
            issuer_bars,
            issuer_actions,
            benchmark_bars,
            benchmark_actions,
            benchmark_ready,
            cfg,
        )
        ledger_rows.append(row)
        cashflow_rows.extend(flows)
    ledger = pd.DataFrame(ledger_rows)
    cashflows = pd.DataFrame(cashflow_rows, columns=_cashflow_columns())
    readiness = _readiness_table(ledger)
    primary = ledger.loc[ledger["horizon_sessions"].eq(cfg.primary_horizon_sessions)]
    combined = primary["analysis_sample"].astype(bool) & primary[
        "market_excess_return_ready"
    ].astype(bool)
    combined_count = int(combined.sum())
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "events": int(ledger["longitudinal_event_id"].nunique()),
        "event_horizon_rows": len(ledger),
        "issuer_return_rows_ready": int(ledger["issuer_return_ready"].sum()),
        "market_excess_rows_ready": int(ledger["market_excess_return_ready"].sum()),
        "primary_horizon_rows": len(primary),
        "primary_horizon_market_excess_ready": int(primary["market_excess_return_ready"].sum()),
        "primary_complete_feature_return_events": combined_count,
        "minimum_complete_feature_return_events": cfg.minimum_complete_events,
        "primary_analysis_gate": combined_count >= cfg.minimum_complete_events,
        "benchmark_ready": benchmark_ready,
        "later_delisted_events_retained": int(
            primary.loc[primary["analysis_sample"], "later_delisted_after_formal_exit"].sum()
        ),
        "zero_delisting_return_imputed": False,
        "dividends_reinvested": False,
        "annualized_performance_computed": False,
        "strategy_pnl_computed": False,
        "config": asdict(cfg),
    }
    return LongitudinalReturnTables(ledger, cashflows, readiness, report)


def build_longitudinal_event_returns(
    market_plan_root: Path,
    preregistration_root: Path,
    price_root: Path,
    action_root: Path,
    benchmark_root: Path,
    delisting_root: Path,
    *,
    transition_repair_root: Path | None = None,
    output_root: Path = Path("data/processed/longitudinal_event_returns"),
    config: LongitudinalReturnConfig | None = None,
) -> LongitudinalReturnResult:
    """Build an immutable ledger tied to every collection and pre-registration manifest."""
    cfg = config or LongitudinalReturnConfig()
    cfg.validate()
    child_roots = {
        "price": market_plan_root / "prices",
        "action": market_plan_root / "actions",
        "benchmark": market_plan_root / "benchmark",
    }
    manifests: dict[str, str | None] = {
        "market_plan_manifest_sha256": _sha256_file(market_plan_root / "output_manifest.json"),
        "preregistration_manifest_sha256": _sha256_file(
            preregistration_root / "output_manifest.json"
        ),
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "action_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
        "delisting_manifest_sha256": _sha256_file(delisting_root / "output_manifest.json"),
        "transition_repair_manifest_sha256": (
            _sha256_file(transition_repair_root / "output_manifest.json")
            if transition_repair_root is not None
            else None
        ),
    }
    price_report = _read_json(price_root / "massive_price_report.json")
    action_report = _read_json(action_root / "massive_security_actions_report.json")
    benchmark_report = _read_json(benchmark_root / "massive_benchmark_report.json")
    if price_report["target_plan_manifest_sha256"] != _sha256_file(
        child_roots["price"] / "output_manifest.json"
    ):
        raise FilingCatalogError("Price audit does not match the longitudinal market plan")
    if action_report["target_plan_manifest_sha256"] != _sha256_file(
        child_roots["action"] / "output_manifest.json"
    ):
        raise FilingCatalogError("Action audit does not match the longitudinal market plan")
    if benchmark_report["target_plan_manifest_sha256"] != _sha256_file(
        child_roots["benchmark"] / "output_manifest.json"
    ):
        raise FilingCatalogError("Benchmark audit does not match the market plan")

    event_horizons = load_catalog_table(market_plan_root, "event_horizon_registry.parquet")
    event_map = load_catalog_table(market_plan_root, "event_market_map.parquet")
    event_registry = load_catalog_table(preregistration_root, "event_registry.parquet")
    price_targets = load_catalog_table(child_roots["price"], "massive_daily_price_targets.parquet")
    bars = load_catalog_table(price_root, "massive_daily_price_observations.parquet")
    action_audit = load_catalog_table(action_root, "sample_security_actions_audit.parquet")
    actions = load_catalog_table(action_root, "normalized_corporate_actions.parquet")
    if transition_repair_root is not None:
        bars, action_audit, actions = _apply_transition_repair(
            bars,
            action_audit,
            actions,
            transition_repair_root,
            expected_original_price_manifest=str(manifests["price_manifest_sha256"]),
            expected_original_action_manifest=str(manifests["action_manifest_sha256"]),
        )
    benchmark_bars = load_catalog_table(
        benchmark_root, "benchmark_daily_price_observations.parquet"
    )
    benchmark_actions = load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet")
    delisting = _issuer_delisting_outcomes(delisting_root)
    tables = calculate_longitudinal_event_returns(
        event_horizons,
        event_map,
        event_registry,
        price_targets,
        bars,
        action_audit,
        actions,
        benchmark_bars,
        benchmark_actions,
        delisting,
        benchmark_ready=bool(benchmark_report["benchmark_data_ready"]),
        config=cfg,
    )
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return LongitudinalReturnResult(
            target.resolve(), _read_json(target / "longitudinal_return_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete longitudinal return directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "longitudinal_event_return_ledger.parquet": tables.ledger,
        "longitudinal_return_cashflows.parquet": tables.cashflows,
        "longitudinal_return_readiness.parquet": tables.readiness,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "longitudinal_return_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker_transition_repair_applied": transition_repair_root is not None,
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "longitudinal_return_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return LongitudinalReturnResult(target.resolve(), report)


def _event_return_row(
    event: Any,
    issuer_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    benchmark_ready: bool,
    config: LongitudinalReturnConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reasons: list[str] = []
    if not bool(event.historical_symbol_interval_ready):
        reasons.append("security_identity_not_ready")
    if not bool(event.corporate_actions_ready):
        reasons.append("corporate_actions_not_ready")
    if not bool(event.window_terminal_outcome_ready):
        reasons.append("delisting_outcome_not_ready")
    entry_bar = _unique_optional_bar(
        issuer_bars,
        "price_target_id",
        str(event.price_target_id),
        str(event.entry_session_date),
    )
    exit_bar = _unique_optional_bar(
        issuer_bars,
        "price_target_id",
        str(event.price_target_id),
        str(event.exit_session_date),
    )
    if entry_bar is None or exit_bar is None:
        reasons.append("issuer_price_not_ready")
    benchmark_entry = _unique_optional_bar(
        benchmark_bars,
        None,
        None,
        str(event.entry_session_date),
    )
    benchmark_exit = _unique_optional_bar(
        benchmark_bars,
        None,
        None,
        str(event.exit_session_date),
    )
    if not benchmark_ready or benchmark_entry is None or benchmark_exit is None:
        reasons.append("benchmark_not_ready")
    entry_at = pd.Timestamp(str(event.entry_open_at))
    exit_at = pd.Timestamp(str(event.exit_close_at))
    issuer_ready = not any(
        reason in reasons
        for reason in (
            "security_identity_not_ready",
            "corporate_actions_not_ready",
            "delisting_outcome_not_ready",
            "issuer_price_not_ready",
        )
    )
    issuer_total = issuer_price = issuer_dividend = np.nan
    issuer_entry = issuer_exit = np.nan
    issuer_entry_hash = issuer_exit_hash = None
    cashflows: list[dict[str, Any]] = []
    if issuer_ready:
        assert entry_bar is not None and exit_bar is not None
        issuer_entry = _positive_float(entry_bar["open_unadjusted"], "issuer entry")
        issuer_exit = _positive_float(exit_bar["close_unadjusted"], "issuer exit")
        issuer_entry_hash = str(entry_bar["source_sha256"])
        issuer_exit_hash = str(exit_bar["source_sha256"])
        actions = _actions_inside_window(
            issuer_actions.loc[issuer_actions["action_target_id"].eq(str(event.action_target_id))],
            entry_at,
            exit_at,
        )
        holding, details = calculate_cash_retained_holding_return(
            issuer_entry,
            issuer_exit,
            actions,
            currency=str(event.currency_name).upper(),
        )
        issuer_total = holding.total_return
        issuer_price = holding.split_adjusted_price_return
        issuer_dividend = holding.cash_dividend_return
        cashflows.extend(
            _action_cashflows(
                str(event.longitudinal_event_id),
                int(event.horizon_sessions),
                "issuer",
                details,
            )
        )
    market_ready = issuer_ready and "benchmark_not_ready" not in reasons
    market_total = market_excess = market_relative = np.nan
    market_entry_price = market_exit_price = np.nan
    market_entry_hash = market_exit_hash = None
    if market_ready:
        assert benchmark_entry is not None and benchmark_exit is not None
        market_entry_price = _positive_float(benchmark_entry["open_unadjusted"], "market entry")
        market_exit_price = _positive_float(benchmark_exit["close_unadjusted"], "market exit")
        market_entry_hash = str(benchmark_entry["source_sha256"])
        market_exit_hash = str(benchmark_exit["source_sha256"])
        actions = _actions_inside_window(benchmark_actions, entry_at, exit_at)
        market, details = calculate_cash_retained_holding_return(
            market_entry_price,
            market_exit_price,
            actions,
            currency="USD",
        )
        market_total = market.total_return
        market_excess = issuer_total - market_total
        market_relative = (1.0 + issuer_total) / (1.0 + market_total) - 1.0
        cashflows.extend(
            _action_cashflows(
                str(event.longitudinal_event_id),
                int(event.horizon_sessions),
                "benchmark",
                details,
            )
        )
    row = {
        "longitudinal_event_id": str(event.longitudinal_event_id),
        "cik": str(event.cik),
        "accession_number": str(event.accession_number),
        "accepted_at": str(event.accepted_at),
        "form": str(event.form),
        "horizon_sessions": int(event.horizon_sessions),
        "is_primary_horizon": bool(event.is_primary_horizon),
        "price_target_id": str(event.price_target_id),
        "action_target_id": str(event.action_target_id),
        "ticker": str(event.selected_ticker),
        "security_id": str(event.selected_provider_security_id),
        "entry_at": entry_at.isoformat(),
        "entry_price_type": config.entry_price_type,
        "entry_price_unadjusted": issuer_entry,
        "exit_at": exit_at.isoformat(),
        "exit_price_type": config.exit_price_type,
        "exit_price_unadjusted": issuer_exit,
        "issuer_split_adjusted_price_return": issuer_price,
        "issuer_cash_dividend_return": issuer_dividend,
        "issuer_total_return": issuer_total,
        "issuer_return_ready": issuer_ready,
        "market_benchmark_ticker": "SPY",
        "market_entry_price_unadjusted": market_entry_price,
        "market_exit_price_unadjusted": market_exit_price,
        "market_total_return": market_total,
        "market_excess_return_arithmetic": market_excess,
        "market_relative_wealth_return": market_relative,
        "market_excess_return_ready": market_ready,
        "later_delisted_after_formal_exit": bool(event.later_delisted_after_formal_exit),
        "delisting_outcome_status": str(event.missing_return_reason),
        "representation_features_complete": bool(event.representation_features_complete),
        "analysis_sample": bool(event.analysis_sample),
        "gm_level": event.gm_level,
        "gm_delta_yoy": event.gm_delta_yoy,
        "gm_innovation_raw": event.gm_innovation_raw,
        "gm_innovation_std": event.gm_innovation_std,
        "readiness_status": "ready" if market_ready else "|".join(sorted(set(reasons))),
        "issuer_entry_source_sha256": issuer_entry_hash,
        "issuer_exit_source_sha256": issuer_exit_hash,
        "market_entry_source_sha256": market_entry_hash,
        "market_exit_source_sha256": market_exit_hash,
        "delisting_source_sha256": str(event.source_snapshot_sha256),
    }
    return row, cashflows


def _issuer_delisting_outcomes(root: Path) -> pd.DataFrame:
    targets = load_catalog_table(root, "massive_delisting_targets.parquet")
    outcomes = load_catalog_table(root, "event_window_delisting_outcomes.parquet")
    frame = outcomes.merge(
        targets[["outcome_target_id", "cik", "security_id"]],
        on="outcome_target_id",
        how="left",
        validate="one_to_one",
    )
    frame["cik"] = frame["cik"].astype(str).str.zfill(10)
    if frame["cik"].duplicated().any():
        raise FilingCatalogError("Delisting outcomes must be unique by issuer for v1")
    return frame[
        [
            "cik",
            "window_terminal_outcome_ready",
            "later_delisted_after_formal_exit",
            "missing_return_reason",
            "source_snapshot_sha256",
        ]
    ]


def _apply_transition_repair(
    bars: pd.DataFrame,
    action_audit: pd.DataFrame,
    actions: pd.DataFrame,
    repair_root: Path,
    *,
    expected_original_price_manifest: str,
    expected_original_action_manifest: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = _read_json(repair_root / "ticker_transition_repair_report.json")
    if not report.get("transition_gate"):
        raise FilingCatalogError("Ticker-transition repair has not passed its evidence gate")
    if report.get("original_price_manifest_sha256") != expected_original_price_manifest:
        raise FilingCatalogError("Ticker-transition repair uses a different original price audit")
    if report.get("original_action_manifest_sha256") != expected_original_action_manifest:
        raise FilingCatalogError("Ticker-transition repair uses a different original action audit")
    additions = load_catalog_table(repair_root, "reconciled_daily_price_additions.parquet")
    action_additions = load_catalog_table(
        repair_root, "reconciled_corporate_action_additions.parquet"
    )
    overrides = load_catalog_table(repair_root, "identity_readiness_overrides.parquet")
    combined_bars = pd.concat([bars, additions[bars.columns]], ignore_index=True)
    if combined_bars.duplicated(["price_target_id", "session_date"]).any():
        raise FilingCatalogError("Ticker-transition repair creates duplicate daily bars")
    combined_actions = pd.concat([actions, action_additions[actions.columns]], ignore_index=True)
    observed_actions = combined_actions.loc[combined_actions["provider_event_id"].notna()]
    if observed_actions.duplicated(["action_target_id", "provider_event_id"]).any():
        raise FilingCatalogError("Ticker-transition repair duplicates a corporate action")
    updated_audit = action_audit.copy()
    for override in overrides.itertuples(index=False):
        mask = updated_audit["action_target_id"].eq(str(override.action_target_id))
        if int(mask.sum()) != 1:
            raise FilingCatalogError("Ticker-transition override target is missing or duplicated")
        updated_audit.loc[mask, "historical_symbol_interval_ready"] = bool(
            override.historical_symbol_interval_ready
        )
        updated_audit.loc[mask, "historical_symbol_interval_status"] = str(
            override.historical_symbol_interval_status
        )
        updated_audit.loc[mask, "corporate_actions_ready"] = bool(override.corporate_actions_ready)
        updated_audit.loc[mask, "stable_security_id"] = str(override.stable_security_id)
    return combined_bars, updated_audit, combined_actions


def _unique_optional_bar(
    bars: pd.DataFrame,
    id_column: str | None,
    identifier: str | None,
    session_date: str,
) -> pd.Series[Any] | None:
    selected = bars.loc[bars["session_date"].eq(session_date)]
    if id_column is not None:
        if identifier is None:
            raise FilingCatalogError("A bar id column requires an identifier")
        selected = selected.loc[selected[id_column].eq(identifier)]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise FilingCatalogError("A market target has duplicate daily bars")
    return selected.iloc[0]


def _actions_inside_window(
    actions: pd.DataFrame, entry_at: pd.Timestamp, exit_at: pd.Timestamp
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    frame = actions.copy()
    frame["effective_at_value"] = pd.to_datetime(frame["effective_at"], utc=True, errors="coerce")
    if frame["effective_at_value"].isna().any():
        raise FilingCatalogError("A corporate action has an invalid effective timestamp")
    return frame.loc[
        frame["effective_at_value"].gt(entry_at) & frame["effective_at_value"].le(exit_at)
    ].drop(columns="effective_at_value")


def _action_cashflows(
    event_id: str,
    horizon: int,
    asset_role: str,
    details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "longitudinal_event_id": event_id,
            "horizon_sessions": horizon,
            "asset_role": asset_role,
            "provider_event_id": detail["provider_event_id"],
            "effective_at": detail["effective_at"],
            "action_type": detail["action_type"],
            "shares_before": detail["shares_before"],
            "shares_after": detail["shares_after"],
            "cash_flow": detail["cash_flow"],
            "currency": detail["currency"],
            "source_snapshot_sha256": detail["source_snapshot_sha256"],
        }
        for detail in details
    ]


def _readiness_table(ledger: pd.DataFrame) -> pd.DataFrame:
    return (
        ledger.groupby(["horizon_sessions", "readiness_status"], dropna=False)
        .agg(events=("longitudinal_event_id", "size"), issuers=("cik", "nunique"))
        .reset_index()
        .sort_values(["horizon_sessions", "readiness_status"], kind="mergesort")
    )


def _validate_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.empty or frame.duplicated(columns).any():
        raise FilingCatalogError(f"{label} is empty or duplicated")


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FilingCatalogError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FilingCatalogError(f"{label} must be finite and positive")
    return result


def _cashflow_columns() -> list[str]:
    return [
        "longitudinal_event_id",
        "horizon_sessions",
        "asset_role",
        "provider_event_id",
        "effective_at",
        "action_type",
        "shares_before",
        "shares_after",
        "cash_flow",
        "currency",
        "source_snapshot_sha256",
    ]


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
