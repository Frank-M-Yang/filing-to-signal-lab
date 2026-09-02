"""Point-in-time State/Delta/Innovation/Reaction drift diagnostics.

This module deliberately produces a development diagnostic, not a trading rule.
Every row is one company x filing event x information set.  Reaction features
stop at the last completed close before the formal second-session-open entry,
and every expanding-window forecast uses only labels that had already matured.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

CONDITIONAL_DRIFT_VERSION: Final[str] = "conditional-drift-study-v1"

STATE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "gm_level",
    "form_is_10k",
    "issuer_momentum_5d",
    "issuer_momentum_20d",
    "issuer_volatility_20d",
    "issuer_volume_ratio_5_20",
    "market_momentum_5d",
    "market_momentum_20d",
    "relative_momentum_20d",
    "market_beta_20d",
)
DELTA_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("gm_delta_yoy",)
INNOVATION_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "gm_innovation_raw",
    "gm_innovation_std",
)
REACTION_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_market_excess_return",
    "reaction_volume_ratio_20d",
    "reaction_session_count",
)
ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *STATE_FEATURE_COLUMNS,
    *DELTA_FEATURE_COLUMNS,
    *INNOVATION_FEATURE_COLUMNS,
    *REACTION_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class ConditionalDriftConfig:
    """Frozen feature clocks, cost scenarios, horizons, and training gates."""

    horizons: tuple[int, ...] = (5, 20, 60)
    primary_horizon: int = 20
    cost_scenarios_bps: tuple[tuple[str, float], ...] = (
        ("low", 10.0),
        ("base", 25.0),
        ("stress", 50.0),
    )
    reaction_volume_history_sessions: int = 20
    minimum_training_events: int = 15
    minimum_training_issuers: int = 8
    ridge_alpha: float = 10.0
    random_seed: int = 42
    version: str = CONDITIONAL_DRIFT_VERSION

    def validate(self) -> None:
        """Reject changes that weaken the frozen v1 protocol."""
        if self.horizons != (5, 20, 60) or self.primary_horizon != 20:
            raise ValueError("Conditional Drift v1 freezes 5/20/60-session labels")
        if dict(self.cost_scenarios_bps) != {"low": 10.0, "base": 25.0, "stress": 50.0}:
            raise ValueError("Conditional Drift v1 freezes 10/25/50 bps cost scenarios")
        if self.reaction_volume_history_sessions != 20:
            raise ValueError("Reaction volume requires 20 strictly prior sessions")
        if self.minimum_training_events < 10 or self.minimum_training_issuers < 3:
            raise ValueError("Training gates are too weak")
        if self.ridge_alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive")


@dataclass(frozen=True)
class DriftModelSpec:
    """One fixed linear baseline in the development-only experiment."""

    model_id: str
    feature_set: str
    features: tuple[str, ...]
    algorithm: Literal["zero", "ridge"] = "ridge"


MODEL_SPECS: Final[tuple[DriftModelSpec, ...]] = (
    DriftModelSpec("D0_zero", "none", (), "zero"),
    DriftModelSpec("D1_state", "state", STATE_FEATURE_COLUMNS),
    DriftModelSpec("D2_delta", "delta", DELTA_FEATURE_COLUMNS),
    DriftModelSpec("D3_innovation", "innovation", INNOVATION_FEATURE_COLUMNS),
    DriftModelSpec("D4_reaction", "reaction", REACTION_FEATURE_COLUMNS),
    DriftModelSpec(
        "D5_state_innovation",
        "state_plus_delta_plus_innovation",
        (*STATE_FEATURE_COLUMNS, *DELTA_FEATURE_COLUMNS, *INNOVATION_FEATURE_COLUMNS),
    ),
    DriftModelSpec(
        "D6_full",
        "state_plus_delta_plus_innovation_plus_reaction",
        ALL_FEATURE_COLUMNS,
    ),
)


@dataclass(frozen=True)
class ConditionalDriftTables:
    """Pure study outputs before immutable persistence."""

    event_dataset: pd.DataFrame
    predictions: pd.DataFrame
    fold_audit: pd.DataFrame
    model_metrics: pd.DataFrame
    issuer_influence: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftResult:
    """Immutable Conditional Drift study bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_conditional_drift_study(
    feature_store: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    calendar_sessions: pd.DataFrame,
    config: ConditionalDriftConfig | None = None,
) -> ConditionalDriftTables:
    """Build same-event features, cost-aware labels, and PIT development forecasts."""
    cfg = config or ConditionalDriftConfig()
    cfg.validate()
    _validate_inputs(
        feature_store,
        issuer_bars,
        benchmark_bars,
        issuer_actions,
        benchmark_actions,
        calendar_sessions,
    )
    data = feature_store.copy()
    data["accepted_at"] = _utc(data["accepted_at"], "accepted_at")
    data["execution_entry_at"] = _utc(data["execution_entry_at"], "execution_entry_at")
    stock = _prepare_bars(issuer_bars, "price_target_id")
    market = _prepare_bars(benchmark_bars, None)
    sessions = _prepare_calendar(calendar_sessions)

    reaction_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for event in data.sort_values(
        ["accepted_at", "cik", "longitudinal_event_id"], kind="mergesort"
    ).itertuples(index=False):
        event_stock = stock.loc[stock["price_target_id"].eq(str(event.price_target_id))]
        event_actions = issuer_actions.loc[
            issuer_actions["price_target_id"].eq(str(event.price_target_id))
        ]
        reaction_rows.append(
            _reaction_row(
                event, event_stock, market, event_actions, benchmark_actions, sessions, cfg
            )
        )
        label_rows.append(
            _label_row(event, event_stock, market, event_actions, benchmark_actions, sessions, cfg)
        )

    labels = pd.DataFrame(label_rows)
    _verify_existing_labels(data, labels)
    overlapping_labels = [
        column
        for column in labels.columns
        if column != "longitudinal_event_id" and column in data.columns
    ]
    data = data.drop(columns=overlapping_labels)
    data = data.merge(
        pd.DataFrame(reaction_rows),
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    ).merge(
        labels,
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    data["state_features_complete"] = data.loc[:, list(STATE_FEATURE_COLUMNS)].notna().all(axis=1)
    data["delta_features_complete"] = data.loc[:, list(DELTA_FEATURE_COLUMNS)].notna().all(axis=1)
    data["innovation_features_complete"] = (
        data.loc[:, list(INNOVATION_FEATURE_COLUMNS)].notna().all(axis=1)
    )
    data["reaction_features_complete"] = data.loc[:, list(REACTION_FEATURE_COLUMNS)].notna().all(
        axis=1
    ) & data["reaction_clock_valid"].astype(bool)
    data["same_event_features_complete"] = data.loc[:, list(ALL_FEATURE_COLUMNS)].notna().all(
        axis=1
    ) & data["reaction_clock_valid"].astype(bool)
    data["development_training_only"] = data["evidence_role"].eq(
        "legacy_observed_training_only"
    ) | data["prior_outcome_observed"].astype(bool)
    data["conditional_drift_development_candidate"] = (
        data["analysis_sample"].astype(bool)
        & data["same_event_features_complete"]
        & data[f"target_ready_{cfg.primary_horizon}d"].astype(bool)
        & data["development_training_only"]
    )
    data = data.sort_values(
        ["reaction_feature_available_at", "cik", "longitudinal_event_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    validate_conditional_drift_dataset(data, cfg)

    predictions, folds = _walk_forward_predictions(data, cfg)
    metrics = _model_metrics(predictions)
    influence = _issuer_influence(predictions)
    sensitivity = _cost_sensitivity(data, cfg)
    complete = data["same_event_features_complete"].astype(bool)
    primary_ready = data[f"target_ready_{cfg.primary_horizon}d"].astype(bool)
    development = data["conditional_drift_development_candidate"].astype(bool)
    prediction_events = (
        int(predictions["longitudinal_event_id"].nunique()) if not predictions.empty else 0
    )
    sixty_ready = int(data.loc[complete, "target_ready_60d"].astype(bool).sum())
    gates = pd.DataFrame(
        [
            _gate("same_event_dataset", int((complete & primary_ready).sum()) >= 30),
            _gate("issuer_breadth", int(data.loc[development, "cik"].nunique()) >= 10),
            _gate("sixty_session_label_coverage", sixty_ready >= 20),
            _gate("pit_development_predictions", prediction_events >= 10),
            _gate("unobserved_external_validation", False),
            _gate("live_trading_authorization", False),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "events": len(data),
        "issuers": int(data["cik"].nunique()),
        "state_complete_events": int(data["state_features_complete"].sum()),
        "innovation_complete_events": int(data["innovation_features_complete"].sum()),
        "reaction_complete_events": int(data["reaction_features_complete"].sum()),
        "same_event_complete_events": int(data["same_event_features_complete"].sum()),
        "development_candidates": int(development.sum()),
        "development_prediction_events": prediction_events,
        "target_ready_events": {
            f"{h}d": int(data[f"target_ready_{h}d"].astype(bool).sum()) for h in cfg.horizons
        },
        "same_event_target_ready_events": {
            f"{h}d": int(data.loc[complete, f"target_ready_{h}d"].astype(bool).sum())
            for h in cfg.horizons
        },
        "research_verdict": "development_diagnostic_only_external_validation_required",
        "external_validation_metrics_computed": False,
        "sealed_holdout_opened": False,
        "annualized_performance_computed": False,
        "portfolio_backtest_computed": False,
        "cost_model_status": "explicit_fixed_round_trip_scenarios_not_broker_estimates",
        "feature_families": {
            "state": list(STATE_FEATURE_COLUMNS),
            "delta_comparator": list(DELTA_FEATURE_COLUMNS),
            "innovation": list(INNOVATION_FEATURE_COLUMNS),
            "reaction": list(REACTION_FEATURE_COLUMNS),
        },
        "models": [asdict(spec) for spec in MODEL_SPECS],
        "config": asdict(cfg),
    }
    return ConditionalDriftTables(
        data,
        predictions,
        folds,
        metrics,
        influence,
        sensitivity,
        gates,
        report,
    )


def _verify_existing_labels(existing: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """Tie the rebuilt 5/20-day gross labels to the already audited ledger."""
    comparable = existing[
        [
            "longitudinal_event_id",
            "target_market_excess_return_5d",
            "target_ready_5d",
            "target_market_excess_return_20d",
            "target_ready_20d",
        ]
    ].merge(rebuilt, on="longitudinal_event_id", how="left", validate="one_to_one")
    for horizon in (5, 20):
        old_ready = comparable[f"target_ready_{horizon}d_x"].astype(bool)
        new_ready = comparable[f"target_ready_{horizon}d_y"].astype(bool)
        if (old_ready & ~new_ready).any():
            raise FilingCatalogError(f"A previously ready {horizon}-day label was not rebuilt")
        ready = old_ready & new_ready
        old = comparable.loc[ready, f"target_market_excess_return_{horizon}d_x"].astype(float)
        new = comparable.loc[ready, f"target_market_excess_return_{horizon}d_y"].astype(float)
        if not np.allclose(old, new, rtol=0.0, atol=1e-12):
            raise FilingCatalogError(
                f"Rebuilt {horizon}-day labels do not tie to the audited return ledger"
            )


def validate_conditional_drift_dataset(
    data: pd.DataFrame, config: ConditionalDriftConfig | None = None
) -> None:
    """Enforce feature and target clocks plus same-event completeness."""
    cfg = config or ConditionalDriftConfig()
    cfg.validate()
    required = {
        "longitudinal_event_id",
        "accepted_at",
        "execution_entry_at",
        "reaction_feature_available_at",
        "reaction_clock_valid",
        *ALL_FEATURE_COLUMNS,
        *(f"target_available_at_{h}d" for h in cfg.horizons),
        *(f"target_ready_{h}d" for h in cfg.horizons),
    }
    _require_columns(data, required, "conditional drift dataset")
    if data["longitudinal_event_id"].duplicated().any():
        raise FilingCatalogError("Conditional Drift events are duplicated")
    accepted = _utc(data["accepted_at"], "accepted_at")
    entry = _utc(data["execution_entry_at"], "execution_entry_at")
    reaction_rows = data["reaction_feature_available_at"].notna()
    reaction_at = _utc(
        data.loc[reaction_rows, "reaction_feature_available_at"],
        "reaction_feature_available_at",
    )
    if (reaction_at < accepted.loc[reaction_rows]).any() or (
        reaction_at >= entry.loc[reaction_rows]
    ).any():
        raise FilingCatalogError("Reaction features do not sit between acceptance and entry")
    for horizon in cfg.horizons:
        ready = data[f"target_ready_{horizon}d"].astype(bool)
        available = _utc(
            data.loc[ready, f"target_available_at_{horizon}d"],
            f"target_available_at_{horizon}d",
        )
        if (available <= entry.loc[ready]).any():
            raise FilingCatalogError("A return label matures before the formal entry")
    complete = data.get("same_event_features_complete", pd.Series(False, index=data.index)).astype(
        bool
    )
    if complete.any() and data.loc[complete, list(ALL_FEATURE_COLUMNS)].isna().any().any():
        raise FilingCatalogError("A same-event complete row contains a missing feature")


def build_conditional_drift_study(
    feature_store_root: Path,
    price_root: Path,
    benchmark_root: Path,
    action_root: Path,
    calendar_root: Path,
    *,
    transition_repair_root: Path | None = None,
    output_root: Path = Path("data/processed/conditional_drift_study"),
    config: ConditionalDriftConfig | None = None,
) -> ConditionalDriftResult:
    """Persist the v1 protocol, dataset, labels, diagnostics, and gates."""
    cfg = config or ConditionalDriftConfig()
    cfg.validate()
    manifests = {
        "implementation_sha256": _sha256_file(Path(__file__)),
        "feature_store_manifest_sha256": _sha256_file(feature_store_root / "output_manifest.json"),
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
        "action_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
        "transition_repair_manifest_sha256": (
            _sha256_file(transition_repair_root / "output_manifest.json")
            if transition_repair_root is not None
            else None
        ),
    }
    feature_store = load_catalog_table(feature_store_root, "event_feature_store.parquet")
    issuer_bars = load_catalog_table(price_root, "massive_daily_price_observations.parquet")
    issuer_actions = load_catalog_table(action_root, "normalized_corporate_actions.parquet")
    if transition_repair_root is not None:
        additions = load_catalog_table(
            transition_repair_root, "reconciled_daily_price_additions.parquet"
        )
        action_additions = load_catalog_table(
            transition_repair_root, "reconciled_corporate_action_additions.parquet"
        )
        issuer_bars = pd.concat([issuer_bars, additions[issuer_bars.columns]], ignore_index=True)
        issuer_actions = pd.concat(
            [issuer_actions, action_additions[issuer_actions.columns]], ignore_index=True
        )
    if issuer_bars.duplicated(["price_target_id", "session_date"]).any():
        raise FilingCatalogError("Transition repair creates duplicate issuer bars")
    benchmark_bars = load_catalog_table(
        benchmark_root, "benchmark_daily_price_observations.parquet"
    )
    benchmark_actions = load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet")
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    tables = calculate_conditional_drift_study(
        feature_store,
        issuer_bars,
        benchmark_bars,
        issuer_actions,
        benchmark_actions,
        sessions,
        cfg,
    )
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConditionalDriftResult(
            target.resolve(), _read_json(target / "conditional_drift_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete Conditional Drift directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "conditional_drift_event_dataset.parquet": tables.event_dataset,
        "conditional_drift_predictions.parquet": tables.predictions,
        "conditional_drift_fold_audit.parquet": tables.fold_audit,
        "conditional_drift_model_metrics.parquet": tables.model_metrics,
        "conditional_drift_issuer_influence.parquet": tables.issuer_influence,
        "conditional_drift_cost_sensitivity.parquet": tables.cost_sensitivity,
        "conditional_drift_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    protocol = {
        "protocol_version": cfg.version,
        "frozen_before_new_external_validation": True,
        "atomic_unit": "company_x_filing_event_x_information_set",
        "prediction_clock": "last_completed_regular_close_before_second_session_open_entry",
        "training_rule": "target_available_at_strictly_before_prediction_clock",
        "outcome_role": "all_existing_events_are_historical_development_only",
        "cost_scenarios_are_calibrations_not_observed_broker_costs": True,
        "external_validation_required_for_effectiveness_claim": True,
        "config": asdict(cfg),
        "models": [asdict(spec) for spec in MODEL_SPECS],
    }
    protocol_path = staging / "conditional_drift_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    artifacts[protocol_path.name] = _artifact_record(protocol_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "conditional_drift_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
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
    return ConditionalDriftResult(target.resolve(), report)


def _reaction_row(
    event: Any,
    stock: pd.DataFrame,
    market: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    sessions: pd.DataFrame,
    config: ConditionalDriftConfig,
) -> dict[str, Any]:
    base: dict[str, Any] = {"longitudinal_event_id": str(event.longitudinal_event_id)}
    accepted = pd.Timestamp(event.accepted_at)
    entry = pd.Timestamp(event.execution_entry_at)
    stock_clock = stock.merge(sessions, on="session_date", how="left", validate="many_to_one")
    stock_clock = stock_clock.loc[stock_clock["regular_close_at"].notna()].copy()
    if stock_clock.empty:
        return {**base, **_empty_reaction("calendar_session_missing")}
    before_acceptance = stock_clock.loc[stock_clock["regular_close_at"].le(accepted)]
    before_entry = stock_clock.loc[stock_clock["regular_close_at"].lt(entry)]
    if before_acceptance.empty or before_entry.empty:
        return {**base, **_empty_reaction("reaction_boundary_missing")}
    start = before_acceptance.iloc[-1]
    end = before_entry.iloc[-1]
    if pd.Timestamp(end["regular_close_at"]) <= pd.Timestamp(start["regular_close_at"]):
        return {**base, **_empty_reaction("no_completed_reaction_session")}
    market_start = _bar_on_date(market, start["session_date"])
    market_end = _bar_on_date(market, end["session_date"])
    if market_start is None or market_end is None:
        return {**base, **_empty_reaction("benchmark_reaction_boundary_missing")}
    start_at = pd.Timestamp(start["regular_close_at"])
    end_at = pd.Timestamp(end["regular_close_at"])
    issuer_holding, _ = calculate_cash_retained_holding_return(
        float(start["close_unadjusted"]),
        float(end["close_unadjusted"]),
        _actions_between(issuer_actions, start_at, end_at),
        currency="USD",
    )
    market_holding, _ = calculate_cash_retained_holding_return(
        float(market_start["close_unadjusted"]),
        float(market_end["close_unadjusted"]),
        _actions_between(benchmark_actions, start_at, end_at),
        currency="USD",
    )
    reaction_bars = stock_clock.loc[
        stock_clock["session_date"].gt(start["session_date"])
        & stock_clock["session_date"].le(end["session_date"])
    ]
    prior = before_acceptance.tail(config.reaction_volume_history_sessions)
    volume_ratio = np.nan
    if len(prior) == config.reaction_volume_history_sessions and not reaction_bars.empty:
        prior_volume = float(pd.to_numeric(prior["volume"], errors="coerce").mean())
        reaction_volume = float(pd.to_numeric(reaction_bars["volume"], errors="coerce").mean())
        if math.isfinite(prior_volume) and prior_volume > 0.0:
            volume_ratio = reaction_volume / prior_volume
    clock_valid = bool(accepted <= end_at < entry)
    return {
        **base,
        "reaction_reference_close_at": start_at,
        "reaction_feature_available_at": end_at,
        "reaction_session_count": float(len(reaction_bars)),
        "reaction_issuer_total_return": issuer_holding.total_return,
        "reaction_market_total_return": market_holding.total_return,
        "reaction_market_excess_return": (
            issuer_holding.total_return - market_holding.total_return
        ),
        "reaction_relative_wealth_return": (
            (1.0 + issuer_holding.total_return) / (1.0 + market_holding.total_return) - 1.0
        ),
        "reaction_volume_ratio_20d": volume_ratio,
        "reaction_clock_valid": clock_valid,
        "reaction_status": "ready"
        if clock_valid and pd.notna(volume_ratio)
        else "volume_history_short",
    }


def _empty_reaction(status: str) -> dict[str, Any]:
    return {
        "reaction_reference_close_at": pd.NaT,
        "reaction_feature_available_at": pd.NaT,
        "reaction_session_count": np.nan,
        "reaction_issuer_total_return": np.nan,
        "reaction_market_total_return": np.nan,
        "reaction_market_excess_return": np.nan,
        "reaction_relative_wealth_return": np.nan,
        "reaction_volume_ratio_20d": np.nan,
        "reaction_clock_valid": False,
        "reaction_status": status,
    }


def _label_row(
    event: Any,
    stock: pd.DataFrame,
    market: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    sessions: pd.DataFrame,
    config: ConditionalDriftConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {"longitudinal_event_id": str(event.longitudinal_event_id)}
    entry_at = pd.Timestamp(event.execution_entry_at)
    entry_date = entry_at.tz_convert(None).normalize()
    stock_after = stock.loc[stock["session_date"].ge(entry_date)].reset_index(drop=True)
    market_entry = _bar_on_date(market, entry_date)
    stock_entry = _bar_on_date(stock, entry_date)
    for horizon in config.horizons:
        prefix = f"{horizon}d"
        if stock_entry is None or market_entry is None or len(stock_after) < horizon:
            _set_empty_label(row, prefix, "price_window_censored")
            continue
        exit_bar = stock_after.iloc[horizon - 1]
        exit_date = pd.Timestamp(exit_bar["session_date"])
        market_exit = _bar_on_date(market, exit_date)
        session = _bar_on_date(sessions, exit_date)
        if market_exit is None or session is None:
            _set_empty_label(row, prefix, "benchmark_or_calendar_exit_missing")
            continue
        exit_at = pd.Timestamp(session["regular_close_at"])
        issuer_holding, _ = calculate_cash_retained_holding_return(
            float(stock_entry["open_unadjusted"]),
            float(exit_bar["close_unadjusted"]),
            _actions_between(issuer_actions, entry_at, exit_at),
            currency="USD",
        )
        market_holding, _ = calculate_cash_retained_holding_return(
            float(market_entry["open_unadjusted"]),
            float(market_exit["close_unadjusted"]),
            _actions_between(benchmark_actions, entry_at, exit_at),
            currency="USD",
        )
        gross = issuer_holding.total_return - market_holding.total_return
        row[f"target_issuer_total_return_{prefix}"] = issuer_holding.total_return
        row[f"target_market_total_return_{prefix}"] = market_holding.total_return
        row[f"target_market_excess_return_{prefix}"] = gross
        for name, bps in config.cost_scenarios_bps:
            row[f"target_net_market_excess_return_{prefix}_{name}"] = gross - bps / 10_000.0
        row[f"target_available_at_{prefix}"] = exit_at
        row[f"target_ready_{prefix}"] = True
        row[f"target_status_{prefix}"] = "ready"
    return row


def _set_empty_label(row: dict[str, Any], prefix: str, status: str) -> None:
    row[f"target_issuer_total_return_{prefix}"] = np.nan
    row[f"target_market_total_return_{prefix}"] = np.nan
    row[f"target_market_excess_return_{prefix}"] = np.nan
    for name in ("low", "base", "stress"):
        row[f"target_net_market_excess_return_{prefix}_{name}"] = np.nan
    row[f"target_available_at_{prefix}"] = pd.NaT
    row[f"target_ready_{prefix}"] = False
    row[f"target_status_{prefix}"] = status


def _walk_forward_predictions(
    data: pd.DataFrame, config: ConditionalDriftConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for horizon in config.horizons:
        target = f"target_net_market_excess_return_{horizon}d_base"
        target_at = f"target_available_at_{horizon}d"
        ready = (
            data["conditional_drift_development_candidate"].astype(bool)
            & data[f"target_ready_{horizon}d"].astype(bool)
            & data[target].notna()
        )
        sample = data.loc[ready].copy()
        sample[target_at] = _utc(sample[target_at], target_at)
        sample = sample.sort_values(
            ["reaction_feature_available_at", "cik", "longitudinal_event_id"], kind="mergesort"
        )
        for test in sample.itertuples(index=False):
            prediction_at = pd.Timestamp(str(test.reaction_feature_available_at))
            train = sample.loc[sample[target_at].lt(prediction_at)].copy()
            training_events = len(train)
            training_issuers = int(train["cik"].nunique())
            gate = (
                training_events >= config.minimum_training_events
                and training_issuers >= config.minimum_training_issuers
            )
            training_end = train[target_at].max() if not train.empty else pd.NaT
            folds.append(
                {
                    "longitudinal_event_id": str(test.longitudinal_event_id),
                    "horizon_sessions": horizon,
                    "prediction_generated_at": prediction_at,
                    "training_events": training_events,
                    "training_issuers": training_issuers,
                    "training_label_end": training_end,
                    "training_gate": gate,
                    "fold_status": "ready" if gate else "insufficient_matured_training_history",
                }
            )
            if not gate:
                continue
            for spec in MODEL_SPECS:
                prediction = _fit_predict(train, test, target, spec, config)
                predictions.append(
                    {
                        "longitudinal_event_id": str(test.longitudinal_event_id),
                        "cik": str(test.cik),
                        "ticker": str(test.ticker),
                        "horizon_sessions": horizon,
                        "model_id": spec.model_id,
                        "feature_set": spec.feature_set,
                        "prediction": prediction,
                        "actual_net_market_excess_return": float(getattr(test, target)),
                        "prediction_generated_at": prediction_at,
                        "target_available_at": pd.Timestamp(getattr(test, target_at)),
                        "training_events": training_events,
                        "training_issuers": training_issuers,
                        "training_label_end": training_end,
                        "evidence_role": "historical_development_only",
                    }
                )
    return pd.DataFrame(predictions, columns=_prediction_columns()), pd.DataFrame(folds)


def _fit_predict(
    train: pd.DataFrame,
    test: Any,
    target: str,
    spec: DriftModelSpec,
    config: ConditionalDriftConfig,
) -> float:
    if spec.algorithm == "zero":
        return 0.0
    x_train = train.loc[:, list(spec.features)].astype(float)
    y_train = train[target].astype(float)
    if x_train.isna().any().any() or y_train.isna().any():
        raise FilingCatalogError("A development fold contains missing model data")
    model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
    model.fit(x_train, y_train)
    x_test = pd.DataFrame(
        [{feature: float(getattr(test, feature)) for feature in spec.features}],
        columns=list(spec.features),
    )
    return float(model.predict(x_test)[0])


def _model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_sessions",
        "model_id",
        "feature_set",
        "prediction_events",
        "prediction_issuers",
        "spearman_ic",
        "mae",
        "rmse",
        "directional_accuracy",
        "top_minus_bottom_tertile_net_return",
        "evidence_role",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (horizon, model_id), group in predictions.groupby(
        ["horizon_sessions", "model_id"], sort=True
    ):
        actual = group["actual_net_market_excess_return"].astype(float)
        predicted = group["prediction"].astype(float)
        rows.append(
            {
                "horizon_sessions": int(cast(Any, horizon)),
                "model_id": str(model_id),
                "feature_set": str(group["feature_set"].iloc[0]),
                "prediction_events": len(group),
                "prediction_issuers": int(group["cik"].nunique()),
                "spearman_ic": _spearman(predicted, actual),
                "mae": float((predicted - actual).abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "directional_accuracy": float((np.sign(predicted) == np.sign(actual)).mean()),
                "top_minus_bottom_tertile_net_return": _tertile_spread(predicted, actual),
                "evidence_role": "historical_development_only",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _issuer_influence(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_sessions",
        "excluded_cik",
        "remaining_events",
        "remaining_issuers",
        "full_model_spearman_ic",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    full = predictions.loc[predictions["model_id"].eq("D6_full")]
    rows: list[dict[str, Any]] = []
    for horizon, horizon_data in full.groupby("horizon_sessions", sort=True):
        for cik in sorted(horizon_data["cik"].astype(str).unique()):
            remaining = horizon_data.loc[~horizon_data["cik"].eq(cik)]
            rows.append(
                {
                    "horizon_sessions": int(cast(Any, horizon)),
                    "excluded_cik": cik,
                    "remaining_events": len(remaining),
                    "remaining_issuers": int(remaining["cik"].nunique()),
                    "full_model_spearman_ic": _spearman(
                        remaining["prediction"],
                        remaining["actual_net_market_excess_return"],
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _cost_sensitivity(data: pd.DataFrame, config: ConditionalDriftConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    same_event = data["same_event_features_complete"].astype(bool)
    for horizon in config.horizons:
        ready = same_event & data[f"target_ready_{horizon}d"].astype(bool)
        sample = data.loc[ready]
        gross = sample[f"target_market_excess_return_{horizon}d"].astype(float)
        for scenario, bps in config.cost_scenarios_bps:
            net = sample[f"target_net_market_excess_return_{horizon}d_{scenario}"].astype(float)
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "cost_scenario": scenario,
                    "round_trip_cost_bps": bps,
                    "events": len(sample),
                    "issuers": int(sample["cik"].nunique()),
                    "mean_gross_market_excess_return": float(gross.mean()),
                    "mean_net_market_excess_return": float(net.mean()),
                    "positive_net_return_share": float(net.gt(0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _prepare_bars(frame: pd.DataFrame, id_column: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.normalize()
    if result["session_date"].isna().any():
        raise FilingCatalogError("Market bars contain invalid session dates")
    keys = ["session_date"] if id_column is None else [id_column, "session_date"]
    if result.duplicated(keys).any():
        raise FilingCatalogError("Market bars are duplicated")
    return result.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _prepare_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    sessions = frame.loc[frame["exchange_mic"].eq("XNYS")].copy()
    sessions["session_date"] = pd.to_datetime(
        sessions["session_date"], errors="coerce"
    ).dt.normalize()
    sessions["regular_close_at"] = _utc(sessions["regular_close_at"], "regular_close_at")
    if sessions.empty or sessions["session_date"].duplicated().any():
        raise FilingCatalogError("XNYS calendar sessions are empty or duplicated")
    return sessions[["session_date", "regular_close_at", "is_early_close"]].sort_values(
        "session_date", kind="mergesort"
    )


def _bar_on_date(frame: pd.DataFrame, session_date: Any) -> pd.Series[Any] | None:
    date_value = pd.Timestamp(session_date).tz_localize(None).normalize()
    selected = frame.loc[frame["session_date"].eq(date_value)]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise FilingCatalogError("A market date has duplicate bars")
    return selected.iloc[0]


def _actions_between(
    actions: pd.DataFrame, start_at: pd.Timestamp, end_at: pd.Timestamp
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    result = actions.copy()
    result["effective_at_value"] = _utc(result["effective_at"], "effective_at")
    return result.loc[
        result["effective_at_value"].gt(start_at) & result["effective_at_value"].le(end_at)
    ].drop(columns="effective_at_value")


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _tertile_spread(predicted: pd.Series, actual: pd.Series) -> float:
    if len(predicted) < 9 or predicted.nunique(dropna=True) < 3:
        return float("nan")
    ranks = predicted.rank(method="first", pct=True)
    return float(actual.loc[ranks.gt(2.0 / 3.0)].mean() - actual.loc[ranks.le(1.0 / 3.0)].mean())


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed)}


def _validate_inputs(
    feature_store: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    calendar_sessions: pd.DataFrame,
) -> None:
    _require_columns(
        feature_store,
        {
            "longitudinal_event_id",
            "cik",
            "accepted_at",
            "execution_entry_at",
            "price_target_id",
            "analysis_sample",
            "evidence_role",
            "prior_outcome_observed",
            *STATE_FEATURE_COLUMNS,
            *DELTA_FEATURE_COLUMNS,
            *INNOVATION_FEATURE_COLUMNS,
        },
        "feature store",
    )
    bar_columns = {
        "session_date",
        "open_unadjusted",
        "close_unadjusted",
        "volume",
    }
    _require_columns(issuer_bars, {"price_target_id", *bar_columns}, "issuer bars")
    _require_columns(benchmark_bars, bar_columns, "benchmark bars")
    action_columns = {
        "effective_at",
        "action_type",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
        "provider_event_id",
        "source_snapshot_sha256",
    }
    _require_columns(issuer_actions, {"price_target_id", *action_columns}, "issuer actions")
    _require_columns(benchmark_actions, action_columns, "benchmark actions")
    _require_columns(
        calendar_sessions,
        {"exchange_mic", "session_date", "regular_close_at", "is_early_close"},
        "calendar sessions",
    )


def _prediction_columns() -> list[str]:
    return [
        "longitudinal_event_id",
        "cik",
        "ticker",
        "horizon_sessions",
        "model_id",
        "feature_set",
        "prediction",
        "actual_net_market_excess_return",
        "prediction_generated_at",
        "target_available_at",
        "training_events",
        "training_issuers",
        "training_label_end",
        "evidence_role",
    ]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise FilingCatalogError(f"{name} is missing columns: {sorted(missing)}")


def _utc(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise FilingCatalogError(f"{name} contains invalid timestamps")
    return converted


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FilingCatalogError(f"Expected a JSON object in {path}")
    return document
