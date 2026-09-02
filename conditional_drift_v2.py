"""Exploratory v2 optimization for the frozen Conditional Drift study.

V2 preserves every v1 clock and evidence role.  It decomposes the observable
pre-entry reaction, adds explicitly signed Innovation/Reaction alignment terms,
and replaces the fixed base deduction with a pre-event liquidity cost proxy.
All current outputs remain historical-development diagnostics.
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

from conditional_drift import (
    ALL_FEATURE_COLUMNS,
    DELTA_FEATURE_COLUMNS,
    INNOVATION_FEATURE_COLUMNS,
    REACTION_FEATURE_COLUMNS,
    STATE_FEATURE_COLUMNS,
    ConditionalDriftConfig,
    calculate_conditional_drift_study,
)
from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

CONDITIONAL_DRIFT_V2_VERSION: Final[str] = "conditional-drift-optimization-v2"

REACTION_DECOMPOSITION_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_gap_market_excess_return",
    "reaction_residual_market_excess_return",
    "reaction_range_ratio_20d",
    "log_median_dollar_volume_20d",
)
ALIGNMENT_COLUMNS: Final[tuple[str, ...]] = (
    "innovation_aligned_reaction",
    "innovation_reaction_disagreement",
    "innovation_reaction_interaction",
)
V2_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *ALL_FEATURE_COLUMNS,
    *REACTION_DECOMPOSITION_COLUMNS,
    *ALIGNMENT_COLUMNS,
)


@dataclass(frozen=True)
class ConditionalDriftV2Config:
    """Frozen exploratory feature, cost-proxy, and training conventions."""

    horizons: tuple[int, ...] = (5, 20, 60)
    primary_horizon: int = 20
    liquidity_history_sessions: int = 20
    fixed_slippage_bps: float = 5.0
    minimum_base_cost_bps: float = 10.0
    maximum_base_cost_bps: float = 100.0
    low_cost_multiplier: float = 0.5
    stress_cost_multiplier: float = 2.0
    maximum_stress_cost_bps: float = 200.0
    minimum_training_events: int = 15
    minimum_training_issuers: int = 8
    ridge_alpha: float = 10.0
    version: str = CONDITIONAL_DRIFT_V2_VERSION

    def validate(self) -> None:
        """Reject post-result flexibility in the v2 development experiment."""
        if self.horizons != (5, 20, 60) or self.primary_horizon != 20:
            raise ValueError("Conditional Drift v2 freezes 5/20/60-session labels")
        if self.liquidity_history_sessions != 20:
            raise ValueError("V2 liquidity features require 20 pre-event sessions")
        expected = (5.0, 10.0, 100.0, 0.5, 2.0, 200.0)
        observed = (
            self.fixed_slippage_bps,
            self.minimum_base_cost_bps,
            self.maximum_base_cost_bps,
            self.low_cost_multiplier,
            self.stress_cost_multiplier,
            self.maximum_stress_cost_bps,
        )
        if observed != expected:
            raise ValueError("V2 liquidity-cost calibration is frozen")
        if self.minimum_training_events < 10 or self.minimum_training_issuers < 3:
            raise ValueError("V2 training gates are too weak")
        if self.ridge_alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive")


@dataclass(frozen=True)
class V2ModelSpec:
    """One fixed v2 development baseline."""

    model_id: str
    feature_set: str
    features: tuple[str, ...]
    algorithm: Literal["zero", "ridge"] = "ridge"


V2_MODEL_SPECS: Final[tuple[V2ModelSpec, ...]] = (
    V2ModelSpec("O0_zero", "none", (), "zero"),
    V2ModelSpec("O1_v1_full", "v1_full", ALL_FEATURE_COLUMNS),
    V2ModelSpec(
        "O2_reaction_decomposed",
        "state_delta_innovation_reaction_decomposed",
        (
            *STATE_FEATURE_COLUMNS,
            *DELTA_FEATURE_COLUMNS,
            *INNOVATION_FEATURE_COLUMNS,
            *REACTION_FEATURE_COLUMNS,
            *REACTION_DECOMPOSITION_COLUMNS,
        ),
    ),
    V2ModelSpec(
        "O3_alignment",
        "v1_full_plus_alignment",
        (*ALL_FEATURE_COLUMNS, *ALIGNMENT_COLUMNS),
    ),
    V2ModelSpec("O4_v2_full", "v2_full", V2_FEATURE_COLUMNS),
)


@dataclass(frozen=True)
class ConditionalDriftV2Tables:
    """Pure v2 optimization artifacts."""

    event_dataset: pd.DataFrame
    predictions: pd.DataFrame
    fold_audit: pd.DataFrame
    model_metrics: pd.DataFrame
    optimization_comparison: pd.DataFrame
    issuer_influence: pd.DataFrame
    cost_diagnostics: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftV2Result:
    """Immutable v2 optimization bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_conditional_drift_v2(
    feature_store: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    calendar_sessions: pd.DataFrame,
    config: ConditionalDriftV2Config | None = None,
) -> ConditionalDriftV2Tables:
    """Add v2 reaction/cost features and run label-maturity-aware diagnostics."""
    cfg = config or ConditionalDriftV2Config()
    cfg.validate()
    v1 = calculate_conditional_drift_study(
        feature_store,
        issuer_bars,
        benchmark_bars,
        issuer_actions,
        benchmark_actions,
        calendar_sessions,
        ConditionalDriftConfig(
            minimum_training_events=cfg.minimum_training_events,
            minimum_training_issuers=cfg.minimum_training_issuers,
            ridge_alpha=cfg.ridge_alpha,
        ),
    )
    stock = _prepare_bars(issuer_bars, "price_target_id")
    market = _prepare_bars(benchmark_bars, None)
    sessions = _prepare_calendar(calendar_sessions)
    rows: list[dict[str, Any]] = []
    for event in v1.event_dataset.itertuples(index=False):
        event_stock = stock.loc[stock["price_target_id"].eq(str(event.price_target_id))]
        event_actions = issuer_actions.loc[
            issuer_actions["price_target_id"].eq(str(event.price_target_id))
        ]
        rows.append(
            _v2_feature_row(
                event,
                event_stock,
                market,
                event_actions,
                benchmark_actions,
                sessions,
                cfg,
            )
        )
    data = v1.event_dataset.merge(
        pd.DataFrame(rows),
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    data["reaction_decomposition_complete"] = data.loc[
        :, list(REACTION_DECOMPOSITION_COLUMNS)
    ].notna().all(axis=1) & data["reaction_decomposition_clock_valid"].astype(bool)
    data["alignment_features_complete"] = data.loc[:, list(ALIGNMENT_COLUMNS)].notna().all(axis=1)
    data["liquidity_cost_complete"] = (
        data[["liquidity_cost_low_bps", "liquidity_cost_base_bps", "liquidity_cost_stress_bps"]]
        .notna()
        .all(axis=1)
    )
    data["v2_features_complete"] = data.loc[:, list(V2_FEATURE_COLUMNS)].notna().all(axis=1) & data[
        "reaction_decomposition_clock_valid"
    ].astype(bool)
    for horizon in cfg.horizons:
        gross = data[f"target_market_excess_return_{horizon}d"]
        ready = data[f"target_ready_{horizon}d"].astype(bool) & data[
            "liquidity_cost_complete"
        ].astype(bool)
        for scenario in ("low", "base", "stress"):
            data[f"target_net_market_excess_return_{horizon}d_liquidity_{scenario}"] = np.where(
                ready,
                gross - data[f"liquidity_cost_{scenario}_bps"] / 10_000.0,
                np.nan,
            )
        data[f"target_liquidity_cost_ready_{horizon}d"] = ready
    data["v2_development_candidate"] = (
        data["analysis_sample"].astype(bool)
        & data["development_training_only"].astype(bool)
        & data["v2_features_complete"].astype(bool)
        & data[f"target_liquidity_cost_ready_{cfg.primary_horizon}d"].astype(bool)
    )
    data = data.sort_values(
        ["reaction_feature_available_at", "cik", "longitudinal_event_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    validate_conditional_drift_v2(data, cfg)
    predictions, folds = _walk_forward_predictions(data, cfg)
    metrics = _model_metrics(predictions)
    comparison = _optimization_comparison(metrics)
    influence = _issuer_influence(predictions)
    cost = _cost_diagnostics(data, cfg)
    development = data["v2_development_candidate"].astype(bool)
    prediction_events = (
        int(predictions["longitudinal_event_id"].nunique()) if not predictions.empty else 0
    )
    twenty = comparison.loc[comparison["horizon_sessions"].eq(cfg.primary_horizon)]
    full_improvement = (
        float(twenty["v2_minus_v1_spearman"].iloc[0]) if not twenty.empty else float("nan")
    )
    alignment_improvements = {
        horizon: _model_metric_difference(metrics, horizon, "O3_alignment", "O1_v1_full")
        for horizon in (5, 20)
    }
    alignment_improved_both = all(
        math.isfinite(value) and value > 0.0 for value in alignment_improvements.values()
    )
    twenty_influence = influence.loc[influence["horizon_sessions"].eq(cfg.primary_horizon)]
    min_full_leave_one_out = (
        float(twenty_influence["v2_full_spearman_ic"].min())
        if not twenty_influence.empty
        else float("nan")
    )
    min_alignment_leave_one_out = (
        float(twenty_influence["alignment_spearman_ic"].min())
        if not twenty_influence.empty
        else float("nan")
    )
    gates = pd.DataFrame(
        [
            _gate("v2_same_event_coverage", int(development.sum()) >= 30),
            _gate(
                "reaction_decomposition_clock",
                bool(data.loc[development, "reaction_decomposition_clock_valid"].all()),
            ),
            _gate(
                "liquidity_cost_proxy_coverage",
                int(data.loc[development, "liquidity_cost_complete"].sum()) >= 30,
            ),
            _gate("pit_development_predictions", prediction_events >= 10),
            _gate(
                "full_v2_historical_rank_improvement",
                math.isfinite(full_improvement) and full_improvement > 0.0,
            ),
            _gate("alignment_rank_improvement_5d", alignment_improvements[5] > 0.0),
            _gate("alignment_rank_improvement_20d", alignment_improvements[20] > 0.0),
            _gate(
                "alignment_leave_one_issuer_out_sign_stability",
                math.isfinite(min_alignment_leave_one_out) and min_alignment_leave_one_out > 0.0,
            ),
            _gate("unobserved_external_validation", False),
            _gate("live_trading_authorization", False),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "events": len(data),
        "issuers": int(data["cik"].nunique()),
        "v1_same_event_complete_events": int(data["same_event_features_complete"].sum()),
        "v2_same_event_complete_events": int(data["v2_features_complete"].sum()),
        "v2_development_candidates": int(development.sum()),
        "development_prediction_events": prediction_events,
        "target_ready_events": {
            f"{horizon}d": int(
                data.loc[development, f"target_liquidity_cost_ready_{horizon}d"].sum()
            )
            for horizon in cfg.horizons
        },
        "primary_horizon_v2_minus_v1_spearman": full_improvement,
        "alignment_minus_v1_spearman": {
            f"{horizon}d": value for horizon, value in alignment_improvements.items()
        },
        "primary_horizon_min_full_leave_one_issuer_out_spearman": min_full_leave_one_out,
        "primary_horizon_min_alignment_leave_one_issuer_out_spearman": (
            min_alignment_leave_one_out
        ),
        "forward_candidate_status": (
            "retain_alignment_only_as_low_confidence_external_validation_candidate"
            if alignment_improved_both
            else "do_not_promote_any_v2_feature_block"
        ),
        "full_reaction_decomposition_decision": "discard_from_forward_candidate",
        "research_verdict": "historical_development_optimization_not_validation",
        "external_validation_metrics_computed": False,
        "sealed_holdout_opened": False,
        "portfolio_backtest_computed": False,
        "cost_model_status": "corwin_schultz_spread_proxy_plus_frozen_slippage_not_broker_cost",
        "models": [asdict(spec) for spec in V2_MODEL_SPECS],
        "config": asdict(cfg),
    }
    return ConditionalDriftV2Tables(
        data,
        predictions,
        folds,
        metrics,
        comparison,
        influence,
        cost,
        gates,
        report,
    )


def validate_conditional_drift_v2(
    data: pd.DataFrame, config: ConditionalDriftV2Config | None = None
) -> None:
    """Enforce v2 feature, cost, and target availability clocks."""
    cfg = config or ConditionalDriftV2Config()
    cfg.validate()
    required = {
        "accepted_at",
        "execution_entry_at",
        "reaction_feature_available_at",
        "reaction_gap_available_at",
        "reaction_decomposition_clock_valid",
        "liquidity_cost_available_at",
        *V2_FEATURE_COLUMNS,
        *(f"target_available_at_{horizon}d" for horizon in cfg.horizons),
    }
    _require_columns(data, required, "Conditional Drift v2 dataset")
    accepted = _utc(data["accepted_at"], "accepted_at")
    entry = _utc(data["execution_entry_at"], "execution_entry_at")
    complete = data["v2_features_complete"].astype(bool)
    reaction_at = _utc(
        data.loc[complete, "reaction_feature_available_at"], "reaction_feature_available_at"
    )
    gap_at = _utc(data.loc[complete, "reaction_gap_available_at"], "reaction_gap_available_at")
    cost_at = _utc(data.loc[complete, "liquidity_cost_available_at"], "liquidity_cost_available_at")
    if (gap_at <= accepted.loc[complete]).any() or (gap_at >= entry.loc[complete]).any():
        raise FilingCatalogError("V2 gap reaction is not strictly post-acceptance and pre-entry")
    if (reaction_at >= entry.loc[complete]).any() or (cost_at > accepted.loc[complete]).any():
        raise FilingCatalogError("V2 reaction or liquidity feature leaks beyond its clock")
    if data.loc[complete, list(V2_FEATURE_COLUMNS)].isna().any().any():
        raise FilingCatalogError("A v2 complete event has a missing feature")
    costs = data.loc[
        complete,
        ["liquidity_cost_low_bps", "liquidity_cost_base_bps", "liquidity_cost_stress_bps"],
    ]
    if not (
        costs["liquidity_cost_low_bps"].le(costs["liquidity_cost_base_bps"]).all()
        and costs["liquidity_cost_base_bps"].le(costs["liquidity_cost_stress_bps"]).all()
    ):
        raise FilingCatalogError("V2 liquidity cost scenarios are not ordered")


def build_conditional_drift_v2(
    feature_store_root: Path,
    price_root: Path,
    benchmark_root: Path,
    action_root: Path,
    calendar_root: Path,
    *,
    transition_repair_root: Path | None = None,
    output_root: Path = Path("data/processed/conditional_drift_optimization"),
    config: ConditionalDriftV2Config | None = None,
) -> ConditionalDriftV2Result:
    """Persist the frozen v2 optimization protocol and diagnostics."""
    cfg = config or ConditionalDriftV2Config()
    cfg.validate()
    manifests = {
        "implementation_sha256": _sha256_file(Path(__file__)),
        "v1_implementation_sha256": _sha256_file(Path(__file__).with_name("conditional_drift.py")),
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
        raise FilingCatalogError("V2 transition repair creates duplicate issuer bars")
    benchmark_bars = load_catalog_table(
        benchmark_root, "benchmark_daily_price_observations.parquet"
    )
    benchmark_actions = load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet")
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    tables = calculate_conditional_drift_v2(
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
        return ConditionalDriftV2Result(
            target.resolve(), _read_json(target / "conditional_drift_v2_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete Conditional Drift v2 directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "conditional_drift_v2_event_dataset.parquet": tables.event_dataset,
        "conditional_drift_v2_predictions.parquet": tables.predictions,
        "conditional_drift_v2_fold_audit.parquet": tables.fold_audit,
        "conditional_drift_v2_model_metrics.parquet": tables.model_metrics,
        "conditional_drift_v2_optimization_comparison.parquet": tables.optimization_comparison,
        "conditional_drift_v2_issuer_influence.parquet": tables.issuer_influence,
        "conditional_drift_v2_cost_diagnostics.parquet": tables.cost_diagnostics,
        "conditional_drift_v2_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    protocol = {
        "protocol_version": cfg.version,
        "parent_protocol": "conditional-drift-study-v1",
        "historical_outcomes_already_observed": True,
        "optimization_results_are_not_validation": True,
        "reaction_gap_clock": "first_regular_open_strictly_after_sec_acceptance",
        "reaction_end_clock": "last_completed_regular_close_before_formal_entry",
        "cost_clock": "twenty_sessions_ending_at_last_close_before_acceptance",
        "cost_proxy": "median_corwin_schultz_spread_plus_five_bps_frozen_slippage",
        "training_rule": "target_available_at_strictly_before_prediction_clock",
        "config": asdict(cfg),
        "models": [asdict(spec) for spec in V2_MODEL_SPECS],
    }
    protocol_path = staging / "conditional_drift_v2_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    artifacts[protocol_path.name] = _artifact_record(protocol_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "conditional_drift_v2_report.json"
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
    return ConditionalDriftV2Result(target.resolve(), report)


def _v2_feature_row(
    event: Any,
    stock: pd.DataFrame,
    market: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    sessions: pd.DataFrame,
    config: ConditionalDriftV2Config,
) -> dict[str, Any]:
    base: dict[str, Any] = {"longitudinal_event_id": str(event.longitudinal_event_id)}
    if not bool(event.same_event_features_complete):
        return {**base, **_empty_v2_features("v1_same_event_incomplete")}
    accepted = pd.Timestamp(str(event.accepted_at))
    entry = pd.Timestamp(str(event.execution_entry_at))
    reference_at = pd.Timestamp(str(event.reaction_reference_close_at))
    reaction_end = pd.Timestamp(str(event.reaction_feature_available_at))
    reference_date = reference_at.tz_convert(None).normalize()
    reference_stock = _bar_on_date(stock, reference_date)
    reference_market = _bar_on_date(market, reference_date)
    future_sessions = sessions.loc[
        sessions["regular_open_at"].gt(accepted) & sessions["regular_open_at"].lt(entry)
    ]
    if reference_stock is None or reference_market is None or future_sessions.empty:
        return {**base, **_empty_v2_features("reaction_gap_boundary_missing")}
    gap_session = future_sessions.iloc[0]
    gap_stock = _bar_on_date(stock, gap_session["session_date"])
    gap_market = _bar_on_date(market, gap_session["session_date"])
    if gap_stock is None or gap_market is None:
        return {**base, **_empty_v2_features("reaction_gap_price_missing")}
    gap_at = pd.Timestamp(gap_session["regular_open_at"])
    issuer_gap, _ = calculate_cash_retained_holding_return(
        float(reference_stock["close_unadjusted"]),
        float(gap_stock["open_unadjusted"]),
        _actions_between(issuer_actions, reference_at, gap_at),
        currency="USD",
    )
    market_gap, _ = calculate_cash_retained_holding_return(
        float(reference_market["close_unadjusted"]),
        float(gap_market["open_unadjusted"]),
        _actions_between(benchmark_actions, reference_at, gap_at),
        currency="USD",
    )
    full_issuer = float(event.reaction_issuer_total_return)
    full_market = float(event.reaction_market_total_return)
    issuer_residual = (1.0 + full_issuer) / (1.0 + issuer_gap.total_return) - 1.0
    market_residual = (1.0 + full_market) / (1.0 + market_gap.total_return) - 1.0
    before_acceptance = stock.loc[stock["session_date"].le(reference_date)].tail(
        config.liquidity_history_sessions
    )
    reaction = stock.loc[
        stock["session_date"].gt(reference_date)
        & stock["session_date"].le(reaction_end.tz_convert(None).normalize())
    ]
    if len(before_acceptance) != config.liquidity_history_sessions or reaction.empty:
        return {**base, **_empty_v2_features("liquidity_history_short")}
    prior_range = _mean_range_fraction(before_acceptance)
    reaction_range = _mean_range_fraction(reaction)
    spread = _corwin_schultz_spread(before_acceptance)
    dollar_volume = pd.to_numeric(
        before_acceptance["close_unadjusted"], errors="coerce"
    ) * pd.to_numeric(before_acceptance["volume"], errors="coerce")
    median_dollar_volume = float(dollar_volume.median())
    if (
        not math.isfinite(prior_range)
        or prior_range <= 0.0
        or not math.isfinite(reaction_range)
        or not math.isfinite(spread)
        or not math.isfinite(median_dollar_volume)
        or median_dollar_volume <= 0.0
    ):
        return {**base, **_empty_v2_features("liquidity_feature_invalid")}
    spread_bps = spread * 10_000.0
    base_cost = float(
        np.clip(
            spread_bps + config.fixed_slippage_bps,
            config.minimum_base_cost_bps,
            config.maximum_base_cost_bps,
        )
    )
    low_cost = base_cost * config.low_cost_multiplier
    stress_cost = min(base_cost * config.stress_cost_multiplier, config.maximum_stress_cost_bps)
    reaction_excess = float(event.reaction_market_excess_return)
    innovation = float(event.gm_innovation_std)
    aligned = float(np.sign(innovation) * reaction_excess)
    clock_valid = bool(accepted < gap_at <= reaction_end < entry and reference_at <= accepted)
    return {
        **base,
        "reaction_gap_available_at": gap_at,
        "reaction_gap_issuer_total_return": issuer_gap.total_return,
        "reaction_gap_market_total_return": market_gap.total_return,
        "reaction_gap_market_excess_return": issuer_gap.total_return - market_gap.total_return,
        "reaction_residual_issuer_return": issuer_residual,
        "reaction_residual_market_return": market_residual,
        "reaction_residual_market_excess_return": issuer_residual - market_residual,
        "reaction_range_ratio_20d": reaction_range / prior_range,
        "log_median_dollar_volume_20d": math.log1p(median_dollar_volume),
        "innovation_aligned_reaction": aligned,
        "innovation_reaction_disagreement": float(aligned < 0.0),
        "innovation_reaction_interaction": innovation * reaction_excess,
        "corwin_schultz_spread_fraction_20d": spread,
        "liquidity_cost_low_bps": low_cost,
        "liquidity_cost_base_bps": base_cost,
        "liquidity_cost_stress_bps": stress_cost,
        "liquidity_cost_available_at": reference_at,
        "reaction_decomposition_clock_valid": clock_valid,
        "v2_feature_status": "ready" if clock_valid else "reaction_clock_invalid",
    }


def _empty_v2_features(status: str) -> dict[str, Any]:
    return {
        "reaction_gap_available_at": pd.NaT,
        "reaction_gap_issuer_total_return": np.nan,
        "reaction_gap_market_total_return": np.nan,
        "reaction_gap_market_excess_return": np.nan,
        "reaction_residual_issuer_return": np.nan,
        "reaction_residual_market_return": np.nan,
        "reaction_residual_market_excess_return": np.nan,
        "reaction_range_ratio_20d": np.nan,
        "log_median_dollar_volume_20d": np.nan,
        "innovation_aligned_reaction": np.nan,
        "innovation_reaction_disagreement": np.nan,
        "innovation_reaction_interaction": np.nan,
        "corwin_schultz_spread_fraction_20d": np.nan,
        "liquidity_cost_low_bps": np.nan,
        "liquidity_cost_base_bps": np.nan,
        "liquidity_cost_stress_bps": np.nan,
        "liquidity_cost_available_at": pd.NaT,
        "reaction_decomposition_clock_valid": False,
        "v2_feature_status": status,
    }


def _corwin_schultz_spread(history: pd.DataFrame) -> float:
    """Return the median overnight-adjusted Corwin-Schultz spread estimate."""
    high = pd.to_numeric(history["high_unadjusted"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(history["low_unadjusted"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(history["close_unadjusted"], errors="coerce").to_numpy(float)
    adjusted_high = high.copy()
    adjusted_low = low.copy()
    for index in range(1, len(history)):
        previous_close = close[index - 1]
        if not math.isfinite(previous_close) or previous_close <= 0.0:
            continue
        if previous_close < low[index]:
            overnight_gap = low[index] - previous_close
            adjusted_high[index] = high[index] - overnight_gap
            adjusted_low[index] = previous_close
        elif previous_close > high[index]:
            overnight_gap = previous_close - high[index]
            adjusted_high[index] = previous_close
            adjusted_low[index] = low[index] + overnight_gap
    estimates: list[float] = []
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    for index in range(1, len(history)):
        if (
            min(
                adjusted_high[index - 1],
                adjusted_high[index],
                adjusted_low[index - 1],
                adjusted_low[index],
            )
            <= 0.0
        ):
            continue
        beta = (
            math.log(adjusted_high[index - 1] / adjusted_low[index - 1]) ** 2
            + math.log(adjusted_high[index] / adjusted_low[index]) ** 2
        )
        gamma = math.log(
            max(adjusted_high[index - 1], adjusted_high[index])
            / min(adjusted_low[index - 1], adjusted_low[index])
        )
        gamma **= 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denominator - math.sqrt(
            gamma / denominator
        )
        alpha = max(alpha, 0.0)
        spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        if math.isfinite(spread):
            estimates.append(spread)
    return float(np.median(estimates)) if estimates else float("nan")


def _mean_range_fraction(frame: pd.DataFrame) -> float:
    high = pd.to_numeric(frame["high_unadjusted"], errors="coerce")
    low = pd.to_numeric(frame["low_unadjusted"], errors="coerce")
    close = pd.to_numeric(frame["close_unadjusted"], errors="coerce")
    values = (high - low) / close
    return float(values.mean())


def _walk_forward_predictions(
    data: pd.DataFrame, config: ConditionalDriftV2Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for horizon in config.horizons:
        target = f"target_net_market_excess_return_{horizon}d_liquidity_base"
        target_at = f"target_available_at_{horizon}d"
        ready = (
            data["v2_development_candidate"].astype(bool)
            & data[f"target_liquidity_cost_ready_{horizon}d"].astype(bool)
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
            fold_rows.append(
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
            for spec in V2_MODEL_SPECS:
                prediction_rows.append(
                    {
                        "longitudinal_event_id": str(test.longitudinal_event_id),
                        "cik": str(test.cik),
                        "ticker": str(test.ticker),
                        "horizon_sessions": horizon,
                        "model_id": spec.model_id,
                        "feature_set": spec.feature_set,
                        "prediction": _fit_predict(train, test, target, spec, config),
                        "actual_net_market_excess_return": float(getattr(test, target)),
                        "prediction_generated_at": prediction_at,
                        "target_available_at": pd.Timestamp(str(getattr(test, target_at))),
                        "training_events": training_events,
                        "training_issuers": training_issuers,
                        "training_label_end": training_end,
                        "evidence_role": "historical_development_only",
                    }
                )
    return pd.DataFrame(prediction_rows, columns=_prediction_columns()), pd.DataFrame(fold_rows)


def _fit_predict(
    train: pd.DataFrame,
    test: Any,
    target: str,
    spec: V2ModelSpec,
    config: ConditionalDriftV2Config,
) -> float:
    if spec.algorithm == "zero":
        return 0.0
    x_train = train.loc[:, list(spec.features)].astype(float)
    y_train = train[target].astype(float)
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


def _optimization_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_sessions",
        "prediction_events",
        "v1_full_spearman_ic",
        "v2_full_spearman_ic",
        "v2_minus_v1_spearman",
        "v1_full_mae",
        "v2_full_mae",
        "v2_minus_v1_mae",
        "evidence_role",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for horizon in sorted(metrics["horizon_sessions"].unique()):
        current = metrics.loc[metrics["horizon_sessions"].eq(horizon)].set_index("model_id")
        if not {"O1_v1_full", "O4_v2_full"}.issubset(current.index):
            continue
        v1 = current.loc["O1_v1_full"]
        v2 = current.loc["O4_v2_full"]
        v1_ic = float(cast(Any, v1["spearman_ic"]))
        v2_ic = float(cast(Any, v2["spearman_ic"]))
        v1_mae = float(cast(Any, v1["mae"]))
        v2_mae = float(cast(Any, v2["mae"]))
        rows.append(
            {
                "horizon_sessions": int(cast(Any, horizon)),
                "prediction_events": int(cast(Any, v2["prediction_events"])),
                "v1_full_spearman_ic": v1_ic,
                "v2_full_spearman_ic": v2_ic,
                "v2_minus_v1_spearman": v2_ic - v1_ic,
                "v1_full_mae": v1_mae,
                "v2_full_mae": v2_mae,
                "v2_minus_v1_mae": v2_mae - v1_mae,
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
        "v1_full_spearman_ic",
        "alignment_spearman_ic",
        "alignment_minus_v1_spearman",
        "v2_full_spearman_ic",
    ]
    rows: list[dict[str, Any]] = []
    for horizon, horizon_data in predictions.groupby("horizon_sessions", sort=True):
        full = horizon_data.loc[horizon_data["model_id"].eq("O4_v2_full")]
        v1 = horizon_data.loc[horizon_data["model_id"].eq("O1_v1_full")]
        alignment = horizon_data.loc[horizon_data["model_id"].eq("O3_alignment")]
        for group in (full, v1, alignment):
            if group["longitudinal_event_id"].duplicated().any():
                raise FilingCatalogError("V2 issuer influence has duplicate model predictions")
        for cik in sorted(full["cik"].astype(str).unique()):
            remaining_full = full.loc[~full["cik"].eq(cik)]
            remaining_v1 = v1.loc[~v1["cik"].eq(cik)]
            remaining_alignment = alignment.loc[~alignment["cik"].eq(cik)]
            v1_ic = _spearman(
                remaining_v1["prediction"], remaining_v1["actual_net_market_excess_return"]
            )
            alignment_ic = _spearman(
                remaining_alignment["prediction"],
                remaining_alignment["actual_net_market_excess_return"],
            )
            rows.append(
                {
                    "horizon_sessions": int(cast(Any, horizon)),
                    "excluded_cik": cik,
                    "remaining_events": len(remaining_full),
                    "remaining_issuers": int(remaining_full["cik"].nunique()),
                    "v1_full_spearman_ic": v1_ic,
                    "alignment_spearman_ic": alignment_ic,
                    "alignment_minus_v1_spearman": alignment_ic - v1_ic,
                    "v2_full_spearman_ic": _spearman(
                        remaining_full["prediction"],
                        remaining_full["actual_net_market_excess_return"],
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _model_metric_difference(
    metrics: pd.DataFrame, horizon: int, left_model: str, right_model: str
) -> float:
    selected = metrics.loc[metrics["horizon_sessions"].eq(horizon)].set_index("model_id")
    if not {left_model, right_model}.issubset(selected.index):
        return float("nan")
    left_value = float(cast(Any, selected.loc[left_model, "spearman_ic"]))
    right_value = float(cast(Any, selected.loc[right_model, "spearman_ic"]))
    return left_value - right_value


def _cost_diagnostics(data: pd.DataFrame, config: ConditionalDriftV2Config) -> pd.DataFrame:
    development = data["v2_development_candidate"].astype(bool)
    rows: list[dict[str, Any]] = []
    for horizon in config.horizons:
        ready = development & data[f"target_liquidity_cost_ready_{horizon}d"].astype(bool)
        sample = data.loc[ready]
        for scenario in ("low", "base", "stress"):
            costs = sample[f"liquidity_cost_{scenario}_bps"].astype(float)
            net = sample[f"target_net_market_excess_return_{horizon}d_liquidity_{scenario}"].astype(
                float
            )
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "cost_scenario": scenario,
                    "events": len(sample),
                    "issuers": int(sample["cik"].nunique()),
                    "median_cost_bps": float(costs.median()),
                    "p90_cost_bps": float(costs.quantile(0.9)),
                    "mean_net_market_excess_return": float(net.mean()),
                    "positive_net_return_share": float(net.gt(0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _prepare_bars(frame: pd.DataFrame, id_column: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.normalize()
    if result["session_date"].isna().any():
        raise FilingCatalogError("V2 bars contain invalid session dates")
    keys = ["session_date"] if id_column is None else [id_column, "session_date"]
    if result.duplicated(keys).any():
        raise FilingCatalogError("V2 market bars are duplicated")
    return result.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _prepare_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    sessions = frame.loc[frame["exchange_mic"].eq("XNYS")].copy()
    sessions["session_date"] = pd.to_datetime(
        sessions["session_date"], errors="coerce"
    ).dt.normalize()
    sessions["regular_open_at"] = _utc(sessions["regular_open_at"], "regular_open_at")
    sessions["regular_close_at"] = _utc(sessions["regular_close_at"], "regular_close_at")
    if sessions.empty or sessions["session_date"].duplicated().any():
        raise FilingCatalogError("V2 XNYS calendar is empty or duplicated")
    return sessions[
        ["session_date", "regular_open_at", "regular_close_at", "is_early_close"]
    ].sort_values("session_date", kind="mergesort")


def _bar_on_date(frame: pd.DataFrame, session_date: Any) -> pd.Series[Any] | None:
    value = pd.Timestamp(session_date).tz_localize(None).normalize()
    selected = frame.loc[frame["session_date"].eq(value)]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise FilingCatalogError("V2 market date has duplicate bars")
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


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed)}


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
