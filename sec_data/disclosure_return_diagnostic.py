"""Same-event 20-session returns under the filing clock and the earliest-disclosure clock.

The frozen validation return ledger times every event at its 10-Q/10-K
acceptance.  The first-public clock audit showed that most of those results
were already public days earlier through an Item 2.02 earnings 8-K.  This
diagnostic re-schedules the *same* observed events under the date-based
execution contract twice -- once from the filing acceptance, once from the
earliest SEC-evidenced disclosure -- and recomputes the cash-retained
issuer and SPY benchmark returns from the same preserved bars and corporate
actions, so that the only difference between the two ledgers is the clock.

It then runs the frozen issuer-cluster inference on the same complete-case
sample under both clocks.  Three things must be read into the result:

- These outcomes were observed before this diagnostic was written.  It is a
  diagnostic of the clock, never validation evidence, and it cannot be used
  to select a model.
- Gross margin is assumed public at the release (``release_value_assumption``).
  That assumption was checked on a small reviewed sample, not certified for
  every event.
- The Innovation features were constructed with pooled forecasts whose
  training and error-history clocks run on filing acceptances.  For events
  where any of those clocks falls at or after the earliest disclosure, the
  Innovation was not knowable at the release.  Those events are flagged and
  the inference is repeated on the knowable subset; a full re-timed replay of
  the forecasts is a separate step.

No model is fitted and no trading rule is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.execution_contract import ExecutionContract, execution_window
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.preliminary_effectiveness_diagnostics import (
    FEATURES,
    EffectivenessDiagnosticConfig,
    calculate_primary_effectiveness_tables,
)

DISCLOSURE_RETURN_DIAGNOSTIC_VERSION: Final[str] = "disclosure-return-diagnostic-v1"
CLOCKS: Final[tuple[str, ...]] = ("filing", "disclosure")
ANALYSES: Final[tuple[str, ...]] = ("same_sample", "innovation_knowable", "window_changed")
FORECAST_CLOCKS: Final[tuple[str, ...]] = (
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_error_history_end",
)
LEDGER_COLUMNS: Final[tuple[str, ...]] = (
    "validation_event_id",
    "cik",
    "accession_number",
    "ticker",
    "exchange_mic",
    "accepted_at",
    "price_target_id",
    "entry_session_date",
    "exit_session_date",
    "market_excess_return_arithmetic",
)
CLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "cik",
    "accession_number",
    "report_accepted_at",
    "earliest_disclosure_at",
    "clock_source",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = ("validation_event_id", *FEATURES, *FORECAST_CLOCKS)


class DisclosureReturnDiagnosticError(FilingCatalogError):
    """Raised when the two-clock diagnostic cannot be built faithfully."""


@dataclass(frozen=True)
class DisclosureReturnDiagnosticConfig:
    """Inference settings copied from the frozen effectiveness contract."""

    minimum_complete_events: int = 47
    bootstrap_replications: int = 10_000
    bootstrap_seed: int = 20260814
    ledger_match_tolerance: float = 1e-10
    version: str = DISCLOSURE_RETURN_DIAGNOSTIC_VERSION

    def validate(self) -> None:
        """Keep the inference at least as strict as the preregistered contract."""
        if self.minimum_complete_events < 20:
            raise DisclosureReturnDiagnosticError("minimum_complete_events cannot be below 20")
        if self.bootstrap_replications < 1_000:
            raise DisclosureReturnDiagnosticError("At least 1,000 bootstrap draws are required")
        if not 0 < self.ledger_match_tolerance <= 1e-6:
            raise DisclosureReturnDiagnosticError("ledger_match_tolerance must lie in (0, 1e-6]")

    def inference(self) -> EffectivenessDiagnosticConfig:
        """The frozen inference contract with only sample-size and draw counts overridden."""
        return EffectivenessDiagnosticConfig(
            minimum_complete_events=self.minimum_complete_events,
            bootstrap_replications=self.bootstrap_replications,
            bootstrap_seed=self.bootstrap_seed,
        )


@dataclass(frozen=True)
class DisclosureReturnDiagnosticTables:
    """Event-level two-clock ledger, inference tables, and the report."""

    events: pd.DataFrame
    representation_ic: pd.DataFrame
    incremental_tests: pd.DataFrame
    leave_one_issuer_out: pd.DataFrame
    quantile_returns: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class DisclosureReturnDiagnosticResult:
    """Immutable bundle location and report."""

    output_root: Path
    report: dict[str, Any]


def calculate_disclosure_return_diagnostic(
    ledger: pd.DataFrame,
    clocks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    config: DisclosureReturnDiagnosticConfig | None = None,
    contract: ExecutionContract | None = None,
) -> DisclosureReturnDiagnosticTables:
    """Recompute every observed event under both clocks and compare the inference."""
    cfg = config or DisclosureReturnDiagnosticConfig()
    cfg.validate()
    execution = contract or ExecutionContract()
    execution.validate()
    base = _prepare_ledger(ledger)
    base = _attach_clocks(base, clocks)
    base = _attach_features(base, features)
    bars = _prepare_bars(issuer_bars, "price_target_id")
    benchmark = _prepare_bars(benchmark_bars, None)
    calendars = {
        str(mic): group.sort_values("session_date")["session_date"].astype(str).tolist()
        for mic, group in sessions.groupby("exchange_mic")
    }

    rows: list[dict[str, Any]] = []
    for event in base.to_dict("records"):
        row: dict[str, Any] = {
            key: event[key]
            for key in (
                "validation_event_id",
                "cik",
                "accession_number",
                "ticker",
                "exchange_mic",
                "price_target_id",
                "accepted_at",
                "earliest_disclosure_at",
                "clock_source",
                "disclosure_lead_days",
                "innovation_knowable_at_disclosure",
                "ledger_entry_session_date",
                "ledger_exit_session_date",
                "ledger_market_excess_return_arithmetic",
                *FEATURES,
            )
        }
        for clock in CLOCKS:
            stamp = event["accepted_at"] if clock == "filing" else event["earliest_disclosure_at"]
            row.update(
                _window_return(
                    clock,
                    stamp,
                    str(event["exchange_mic"]),
                    str(event["price_target_id"]),
                    sessions,
                    execution,
                    bars,
                    issuer_actions,
                    benchmark,
                    benchmark_actions,
                )
            )
        dates = calendars.get(str(event["exchange_mic"]), [])
        row["window_changed"] = bool(
            row["filing_entry_session_date"] is not None
            and row["disclosure_entry_session_date"] is not None
            and row["filing_entry_session_date"] != row["disclosure_entry_session_date"]
        )
        row["entry_sessions_earlier"] = (
            dates.index(row["filing_entry_session_date"])
            - dates.index(row["disclosure_entry_session_date"])
            if row["window_changed"]
            and row["filing_entry_session_date"] in dates
            and row["disclosure_entry_session_date"] in dates
            else (0 if row["filing_entry_session_date"] is not None else None)
        )
        row["same_sample_ready"] = bool(
            row["filing_status"] == "ready"
            and row["disclosure_status"] == "ready"
            and all(pd.notna(event[feature]) for feature in FEATURES)
        )
        rows.append(row)
    events = pd.DataFrame(rows)
    ledger_check = _check_against_ledger(events, cfg)
    inference = _run_inference(events, cfg)
    report = _report(events, inference, ledger_check, cfg, execution)
    return DisclosureReturnDiagnosticTables(
        events,
        inference["representation_ic"],
        inference["incremental_tests"],
        inference["leave_one_issuer_out"],
        inference["quantile_returns"],
        report,
    )


def build_disclosure_return_diagnostic(
    return_ledger_root: Path,
    clock_root: Path,
    feature_root: Path,
    execution_plan_root: Path,
    price_audit_root: Path,
    security_actions_root: Path,
    benchmark_root: Path,
    *,
    output_root: Path = Path("data/processed/disclosure_return_diagnostic"),
    config: DisclosureReturnDiagnosticConfig | None = None,
) -> DisclosureReturnDiagnosticResult:
    """Persist a content-addressed two-clock diagnostic from hash-verified inputs."""
    cfg = config or DisclosureReturnDiagnosticConfig()
    cfg.validate()
    roots = {
        "return_ledger": return_ledger_root,
        "first_public_clock": clock_root,
        "filing_features": feature_root,
        "execution_plan": execution_plan_root,
        "price_audit": price_audit_root,
        "security_actions": security_actions_root,
        "benchmark": benchmark_root,
    }
    input_hashes = {
        label: _sha256_file(root / "output_manifest.json") for label, root in roots.items()
    }
    implementation_sha256 = _sha256_file(Path(__file__))
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "implementation_sha256": implementation_sha256,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = f"{cfg.version}-{hashlib.sha256(payload).hexdigest()[:16]}"
    target = output_root / dataset_version
    report_name = "disclosure_return_diagnostic_report.json"
    if (target / "output_manifest.json").is_file():
        stored: dict[str, Any] = json.loads((target / report_name).read_text(encoding="utf-8"))
        return DisclosureReturnDiagnosticResult(target.resolve(), stored)
    if target.exists():
        raise DisclosureReturnDiagnosticError(f"Incomplete diagnostic bundle: {target}")

    tables = calculate_disclosure_return_diagnostic(
        load_catalog_table(return_ledger_root, "validation_event_return_ledger.parquet"),
        load_catalog_table(clock_root, "first_public_clock.parquet"),
        load_catalog_table(feature_root, "target_filing_features.parquet"),
        load_catalog_table(execution_plan_root, "validation_exchange_sessions.parquet"),
        load_catalog_table(price_audit_root, "validation_price_batch_bars.parquet"),
        load_catalog_table(security_actions_root, "normalized_corporate_actions.parquet"),
        load_catalog_table(benchmark_root, "benchmark_daily_price_observations.parquet"),
        load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet"),
        cfg,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "disclosure_return_events.parquet": tables.events,
        "disclosure_return_representation_ic.parquet": tables.representation_ic,
        "disclosure_return_incremental_tests.parquet": tables.incremental_tests,
        "disclosure_return_leave_one_issuer_out.parquet": tables.leave_one_issuer_out,
        "disclosure_return_quantile_returns.parquet": tables.quantile_returns,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"rows": len(frame), "sha256": _sha256_file(path)}
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "implementation_sha256": implementation_sha256,
        "artifacts": artifacts,
    }
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return DisclosureReturnDiagnosticResult(target.resolve(), report)


def _window_return(
    clock: str,
    stamp: Any,
    exchange_mic: str,
    price_target_id: str,
    sessions: pd.DataFrame,
    contract: ExecutionContract,
    bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
) -> dict[str, Any]:
    prefix = f"{clock}_"
    empty: dict[str, Any] = {
        f"{prefix}entry_session_date": None,
        f"{prefix}exit_session_date": None,
        f"{prefix}issuer_total_return": np.nan,
        f"{prefix}market_total_return": np.nan,
        f"{prefix}market_excess_return_arithmetic": np.nan,
        f"{prefix}market_relative_wealth_return": np.nan,
        f"{prefix}status": "clock_missing",
    }
    if pd.isna(stamp):
        return empty
    try:
        window = execution_window(pd.Timestamp(stamp).isoformat(), exchange_mic, sessions, contract)
    except ValueError as error:
        empty[f"{prefix}status"] = f"window_unresolved: {error}"
        return empty
    entry_date = str(window["entry_session_date"])
    exit_date = str(window["exit_session_date"])
    empty[f"{prefix}entry_session_date"] = entry_date
    empty[f"{prefix}exit_session_date"] = exit_date
    entry_at = utc_timestamp(window["entry_at"])
    exit_at = utc_timestamp(window["exit_at"])
    issuer_rows = bars.loc[bars["price_target_id"].eq(price_target_id)]
    entry_bar = issuer_rows.loc[issuer_rows["session_date"].eq(entry_date)]
    exit_bar = issuer_rows.loc[issuer_rows["session_date"].eq(exit_date)]
    bench_entry = benchmark.loc[benchmark["session_date"].eq(entry_date)]
    bench_exit = benchmark.loc[benchmark["session_date"].eq(exit_date)]
    for label, frame in (
        ("issuer_entry_bar", entry_bar),
        ("issuer_exit_bar", exit_bar),
        ("benchmark_entry_bar", bench_entry),
        ("benchmark_exit_bar", bench_exit),
    ):
        if len(frame) != 1:
            empty[f"{prefix}status"] = f"{label}_missing_or_duplicated"
            return empty
    entry_price = float(entry_bar["open_unadjusted"].iloc[0])
    exit_price = float(exit_bar["close_unadjusted"].iloc[0])
    bench_entry_price = float(bench_entry["open_unadjusted"].iloc[0])
    bench_exit_price = float(bench_exit["close_unadjusted"].iloc[0])
    if min(entry_price, exit_price, bench_entry_price, bench_exit_price) <= 0:
        empty[f"{prefix}status"] = "non_positive_endpoint_price"
        return empty
    security_actions = (
        issuer_actions.loc[issuer_actions["price_target_id"].astype(str).eq(price_target_id)]
        if not issuer_actions.empty
        else issuer_actions
    )
    issuer, _ = calculate_cash_retained_holding_return(
        entry_price,
        exit_price,
        _actions_inside(security_actions, entry_at, exit_at),
        currency="USD",
    )
    market, _ = calculate_cash_retained_holding_return(
        bench_entry_price,
        bench_exit_price,
        _actions_inside(benchmark_actions, entry_at, exit_at),
        currency="USD",
    )
    return {
        f"{prefix}entry_session_date": entry_date,
        f"{prefix}exit_session_date": exit_date,
        f"{prefix}issuer_total_return": issuer.total_return,
        f"{prefix}market_total_return": market.total_return,
        f"{prefix}market_excess_return_arithmetic": issuer.total_return - market.total_return,
        f"{prefix}market_relative_wealth_return": (1.0 + issuer.total_return)
        / (1.0 + market.total_return)
        - 1.0,
        f"{prefix}status": "ready",
    }


def _actions_inside(
    actions: pd.DataFrame, entry_at: pd.Timestamp, exit_at: pd.Timestamp
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    effective = pd.to_datetime(actions["effective_at"], utc=True, errors="coerce")
    if effective.isna().any():
        raise DisclosureReturnDiagnosticError("A corporate action has an invalid timestamp")
    return actions.loc[effective.gt(entry_at) & effective.le(exit_at)].copy()


def _check_against_ledger(
    events: pd.DataFrame, config: DisclosureReturnDiagnosticConfig
) -> dict[str, Any]:
    """Where the date-based filing window equals the frozen ledger's window, returns must agree."""
    same_window = (
        events["filing_status"].eq("ready")
        & events["filing_entry_session_date"].eq(events["ledger_entry_session_date"])
        & events["filing_exit_session_date"].eq(events["ledger_exit_session_date"])
        & events["ledger_market_excess_return_arithmetic"].notna()
    )
    gap = (
        events.loc[same_window, "filing_market_excess_return_arithmetic"]
        - events.loc[same_window, "ledger_market_excess_return_arithmetic"]
    ).abs()
    mismatched = int(gap.gt(config.ledger_match_tolerance).sum())
    if mismatched:
        raise DisclosureReturnDiagnosticError(
            f"{mismatched} recomputed filing-window returns differ from the frozen ledger"
        )
    return {
        "events_with_ledger_window": int(same_window.sum()),
        "events_matching_ledger_return": int(same_window.sum()) - mismatched,
        "max_abs_gap": float(gap.max()) if len(gap) else 0.0,
        "tolerance": config.ledger_match_tolerance,
    }


def _run_inference(
    events: pd.DataFrame, config: DisclosureReturnDiagnosticConfig
) -> dict[str, pd.DataFrame]:
    inference = config.inference()
    ic_tables: list[pd.DataFrame] = []
    incremental: list[pd.DataFrame] = []
    loio: list[pd.DataFrame] = []
    quantiles: list[pd.DataFrame] = []
    same_sample = events.loc[events["same_sample_ready"]]
    subsets = {
        "same_sample": same_sample,
        "innovation_knowable": same_sample.loc[
            same_sample["innovation_knowable_at_disclosure"].astype(bool)
        ],
        "window_changed": same_sample.loc[same_sample["window_changed"].astype(bool)],
    }
    for analysis, subset in subsets.items():
        if len(subset) < config.minimum_complete_events or subset["cik"].nunique() < 3:
            continue
        for clock in CLOCKS:
            frame = subset.loc[:, ["validation_event_id", "cik", *FEATURES]].copy()
            frame["market_excess_return_arithmetic"] = subset[
                f"{clock}_market_excess_return_arithmetic"
            ].to_numpy(dtype=float)
            primary = calculate_primary_effectiveness_tables(frame, inference)
            for table, sink in (
                (primary.representation_ic, ic_tables),
                (primary.incremental_test, incremental),
                (primary.leave_one_issuer_out, loio),
                (primary.quantile_returns, quantiles),
            ):
                tagged = table.copy()
                tagged.insert(0, "clock", clock)
                tagged.insert(0, "analysis", analysis)
                sink.append(tagged)
    return {
        "representation_ic": _concat(ic_tables),
        "incremental_tests": _concat(incremental),
        "leave_one_issuer_out": _concat(loio),
        "quantile_returns": _concat(quantiles),
    }


def _report(
    events: pd.DataFrame,
    inference: dict[str, pd.DataFrame],
    ledger_check: dict[str, Any],
    config: DisclosureReturnDiagnosticConfig,
    contract: ExecutionContract,
) -> dict[str, Any]:
    ic = inference["representation_ic"]
    by_analysis: dict[str, Any] = {}
    for analysis in ANALYSES:
        block = ic.loc[ic["analysis"].eq(analysis)]
        if block.empty:
            by_analysis[analysis] = {"status": "below_minimum_sample"}
            continue
        entry: dict[str, Any] = {
            "events": int(block["events"].iloc[0]),
            "issuers": int(block["issuers"].iloc[0]),
        }
        for feature in FEATURES:
            per_clock: dict[str, Any] = {}
            for clock in CLOCKS:
                row = block.loc[block["clock"].eq(clock) & block["feature"].eq(feature)].iloc[0]
                per_clock[clock] = {
                    "spearman_ic": float(row["spearman_ic"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            per_clock["disclosure_minus_filing_ic"] = (
                per_clock["disclosure"]["spearman_ic"] - per_clock["filing"]["spearman_ic"]
            )
            entry[feature] = per_clock
        by_analysis[analysis] = entry
    ready_both = events["same_sample_ready"].astype(bool)
    return {
        "contract_version": config.version,
        "execution_contract": asdict(contract),
        "events": len(events),
        "issuers": int(events["cik"].nunique()),
        "clock_source_counts": {
            str(k): int(v) for k, v in events["clock_source"].value_counts().items()
        },
        "filing_window_ready": int(events["filing_status"].eq("ready").sum()),
        "disclosure_window_ready": int(events["disclosure_status"].eq("ready").sum()),
        "disclosure_status_counts": {
            str(k): int(v) for k, v in events["disclosure_status"].value_counts().items()
        },
        "same_sample_events": int(ready_both.sum()),
        "same_sample_issuers": int(events.loc[ready_both, "cik"].nunique()),
        "window_changed_events": int(events["window_changed"].astype(bool).sum()),
        "window_changed_in_same_sample": int(
            (ready_both & events["window_changed"].astype(bool)).sum()
        ),
        "entry_sessions_earlier_counts": {
            str(k): int(v)
            for k, v in events.loc[events["window_changed"].astype(bool), "entry_sessions_earlier"]
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        },
        "innovation_knowable_at_disclosure": int(
            events["innovation_knowable_at_disclosure"].astype(bool).sum()
        ),
        "innovation_not_knowable_at_disclosure": int(
            (~events["innovation_knowable_at_disclosure"].astype(bool)).sum()
        ),
        "ledger_consistency": ledger_check,
        "by_analysis": by_analysis,
        "diagnostic_on_previously_observed_outcomes": True,
        "validation_evidence": False,
        "model_selection_allowed": False,
        "release_value_assumption": True,
        "forecast_replay_at_disclosure_clock": False,
        "strategy_pnl_computed": False,
        "investment_recommendation_allowed": False,
        "config": asdict(config),
    }


def _prepare_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ledger, LEDGER_COLUMNS, "return ledger")
    base = ledger.loc[:, list(LEDGER_COLUMNS)].copy()
    if base.empty or base["validation_event_id"].duplicated().any():
        raise DisclosureReturnDiagnosticError("The return ledger is empty or non-atomic")
    base["cik"] = base["cik"].astype(str).str.zfill(10)
    base["accepted_at"] = pd.to_datetime(base["accepted_at"], utc=True, errors="coerce")
    if base["accepted_at"].isna().any():
        raise DisclosureReturnDiagnosticError("A ledger event lacks an acceptance clock")
    return base.rename(
        columns={
            "entry_session_date": "ledger_entry_session_date",
            "exit_session_date": "ledger_exit_session_date",
            "market_excess_return_arithmetic": "ledger_market_excess_return_arithmetic",
        }
    ).assign(
        ledger_entry_session_date=lambda f: f["ledger_entry_session_date"].astype(str),
        ledger_exit_session_date=lambda f: f["ledger_exit_session_date"].astype(str),
    )


def _attach_clocks(base: pd.DataFrame, clocks: pd.DataFrame) -> pd.DataFrame:
    _require_columns(clocks, CLOCK_COLUMNS, "first-public clocks")
    table = clocks.loc[:, list(CLOCK_COLUMNS)].copy()
    table["cik"] = table["cik"].astype(str).str.zfill(10)
    if table.duplicated(["cik", "accession_number"]).any():
        raise DisclosureReturnDiagnosticError("Clock rows must be unique per (cik, accession)")
    joined = base.merge(table, on=["cik", "accession_number"], how="left", validate="one_to_one")
    if joined["earliest_disclosure_at"].isna().any():
        raise DisclosureReturnDiagnosticError("A ledger event has no first-public clock")
    for column in ("report_accepted_at", "earliest_disclosure_at"):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="coerce")
    if not joined["report_accepted_at"].eq(joined["accepted_at"]).all():
        raise DisclosureReturnDiagnosticError("Clock table disagrees with the ledger filing clock")
    if joined["earliest_disclosure_at"].gt(joined["accepted_at"]).any():
        raise DisclosureReturnDiagnosticError("An earliest disclosure follows its filing")
    joined["disclosure_lead_days"] = (
        joined["accepted_at"] - joined["earliest_disclosure_at"]
    ).dt.total_seconds() / 86_400.0
    return joined.drop(columns=["report_accepted_at"])


def _attach_features(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    _require_columns(features, FEATURE_COLUMNS, "filing features")
    table = features.loc[:, list(FEATURE_COLUMNS)].copy()
    if table["validation_event_id"].duplicated().any():
        raise DisclosureReturnDiagnosticError("Feature rows are duplicated")
    joined = base.merge(table, on="validation_event_id", how="left", validate="one_to_one")
    for feature in FEATURES:
        joined[feature] = pd.to_numeric(joined[feature], errors="coerce")
    stamps = pd.concat(
        [pd.to_datetime(joined[column], utc=True, errors="coerce") for column in FORECAST_CLOCKS],
        axis=1,
    )
    latest_dependency = stamps.max(axis=1)
    joined["innovation_knowable_at_disclosure"] = (
        latest_dependency.notna() & latest_dependency.lt(joined["earliest_disclosure_at"])
    ).to_numpy(dtype=bool)
    return joined


def _prepare_bars(bars: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    required = ("session_date", "open_unadjusted", "close_unadjusted")
    _require_columns(bars, required, "bars")
    frame = bars.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
    if target_column is None:
        frame["price_target_id"] = "benchmark"
    return frame


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DisclosureReturnDiagnosticError(f"{label} lack columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
