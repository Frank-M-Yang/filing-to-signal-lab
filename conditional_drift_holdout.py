"""Preregister and evaluate the frozen Conditional Drift temporal holdout.

The preregistration stage reads frozen predictions but no outcome asset.  The
evaluation stage refuses to run without that immutable preregistration and then
computes the exact 20-session, second-session-open-to-close outcome once.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from conditional_drift_v2 import _corwin_schultz_spread
from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

HOLDOUT_PREREGISTRATION_VERSION: Final[str] = "conditional-drift-holdout-preregistration-v1"
HOLDOUT_EVALUATION_VERSION: Final[str] = "conditional-drift-holdout-evaluation-v1"
CHALLENGER_SCORE: Final[str] = "F1_v1_challenger_research_score"
CANDIDATE_SCORE: Final[str] = "F2_alignment_candidate_research_score"
CLIPPED_CANDIDATE_SCORE: Final[str] = "F2_alignment_candidate_support_clipped_score"


class ConditionalDriftHoldoutError(FilingCatalogError):
    """Raised when the holdout protocol or outcome firewall is violated."""


@dataclass(frozen=True)
class ConditionalDriftHoldoutConfig:
    """Frozen outcome, inference, and verdict rules."""

    primary_horizon_sessions: int = 20
    minimum_events: int = 16
    minimum_issuers: int = 14
    minimum_in_support_events: int = 8
    bootstrap_replications: int = 5_000
    bootstrap_seed: int = 20_260_829
    leave_one_issuer_out_positive_share: float = 0.80
    fixed_slippage_bps: float = 5.0
    minimum_base_cost_bps: float = 10.0
    maximum_base_cost_bps: float = 100.0

    def validate(self) -> None:
        """Reject changes that would alter the post-score test."""
        if self.primary_horizon_sessions != 20:
            raise ValueError("The holdout freezes the 20-session horizon")
        if self.minimum_events != 16 or self.minimum_issuers != 14:
            raise ValueError("The holdout event and issuer minimums are frozen")
        if self.minimum_in_support_events != 8:
            raise ValueError("The covariate-overlap minimum is frozen at eight events")
        if self.bootstrap_replications != 5_000 or self.bootstrap_seed != 20_260_829:
            raise ValueError("The cluster bootstrap settings are frozen")
        if self.leave_one_issuer_out_positive_share != 0.80:
            raise ValueError("The issuer-stability threshold is frozen")
        if (
            self.fixed_slippage_bps,
            self.minimum_base_cost_bps,
            self.maximum_base_cost_bps,
        ) != (5.0, 10.0, 100.0):
            raise ValueError("The liquidity-cost calibration is frozen")


@dataclass(frozen=True)
class ConditionalDriftHoldoutPreregistrationResult:
    """Immutable event set and evaluation contract."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftHoldoutTables:
    """Pure outcome and diagnostic tables before persistence."""

    outcomes: pd.DataFrame
    analysis_sample: pd.DataFrame
    model_metrics: pd.DataFrame
    bootstrap_inference: pd.DataFrame
    leave_one_issuer_out: pd.DataFrame
    support_sensitivity: pd.DataFrame
    quantile_returns: pd.DataFrame
    verdict_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftHoldoutResult:
    """Immutable temporal-holdout evaluation bundle."""

    output_root: Path
    report: dict[str, Any]


def build_conditional_drift_holdout_preregistration(
    forward_candidate_root: Path,
    *,
    output_root: Path = Path("data/processed/conditional_drift_holdout_preregistration"),
    config: ConditionalDriftHoldoutConfig | None = None,
) -> ConditionalDriftHoldoutPreregistrationResult:
    """Freeze exact scored events and hypotheses without opening outcome data."""
    cfg = config or ConditionalDriftHoldoutConfig()
    cfg.validate()
    forward_manifest_sha = _sha256_file(forward_candidate_root / "output_manifest.json")
    forward_report = _read_json(forward_candidate_root / "conditional_drift_forward_report.json")
    scores = load_catalog_table(forward_candidate_root, "forward_candidate_scores.parquet")
    _validate_preregistration_scores(scores, forward_report, cfg)
    event_registry = scores.loc[
        :,
        [
            "validation_event_id",
            "cik",
            "ticker",
            "accepted_at",
            "reaction_feature_available_at",
            "execution_entry_at",
            "prediction_generated_at",
            "model_training_label_end",
            "evidence_role",
            CHALLENGER_SCORE,
            CANDIDATE_SCORE,
            CLIPPED_CANDIDATE_SCORE,
            "alignment_candidate_within_training_support",
            "alignment_candidate_severe_extrapolation",
        ],
    ].copy()
    event_registry["holdout_event_order"] = np.arange(1, len(event_registry) + 1)
    event_registry["outcome_read_authorized_after_preregistration"] = True
    event_registry["outcome_value_loaded_at_preregistration"] = False
    event_registry = event_registry.sort_values(
        ["prediction_generated_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    hypotheses = _hypothesis_table(cfg)
    payload = json.dumps(
        {
            "version": HOLDOUT_PREREGISTRATION_VERSION,
            "implementation_sha256": _sha256_file(Path(__file__)),
            "forward_manifest_sha256": forward_manifest_sha,
            "ordered_event_ids": event_registry["validation_event_id"].astype(str).tolist(),
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        HOLDOUT_PREREGISTRATION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConditionalDriftHoldoutPreregistrationResult(
            target.resolve(), _read_json(target / "holdout_preregistration_report.json")
        )
    if target.exists():
        raise ConditionalDriftHoldoutError(f"Incomplete holdout preregistration exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "preregistered_holdout_events.parquet": event_registry,
        "preregistered_holdout_hypotheses.parquet": hypotheses,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    protocol = {
        "protocol_version": HOLDOUT_PREREGISTRATION_VERSION,
        "primary_estimand": "spearman_f2_score_vs_20_session_net_market_excess_return",
        "primary_contrast": "f2_spearman_minus_f1_spearman",
        "primary_sample": "all_16_preregistered_temporal_holdout_events",
        "outcome_window": "formal_second_trading_day_open_to_twentieth_session_close",
        "cost": "pre_acceptance_corwin_schultz_spread_plus_5bps_clipped_10_to_100bps",
        "inference": "issuer_cluster_bootstrap_percentile_interval",
        "support_rule": "all_features_within_development_min_max",
        "support_subset_role": "sensitivity_only_not_primary_sample_selection",
        "verdict": "all_preregistered_checks_must_pass",
        "annualized_performance_in_scope": False,
        "portfolio_pnl_in_scope": False,
        "investment_recommendation_allowed": False,
        "config": asdict(cfg),
    }
    protocol_path = staging / "holdout_evaluation_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    artifacts[protocol_path.name] = _artifact_record(protocol_path, 1)
    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "contract_version": HOLDOUT_PREREGISTRATION_VERSION,
        "dataset_version": dataset_version,
        "generated_at": generated_at,
        "events": len(event_registry),
        "issuers": int(event_registry["cik"].nunique()),
        "within_training_support_events": int(
            event_registry["alignment_candidate_within_training_support"].sum()
        ),
        "severe_extrapolation_events": int(
            event_registry["alignment_candidate_severe_extrapolation"].sum()
        ),
        "outcome_values_loaded": False,
        "return_assets_opened": False,
        "hypotheses_frozen_before_outcome_read": True,
        "event_set_frozen_before_outcome_read": True,
        "forward_candidate_version": forward_report["dataset_version"],
        "forward_candidate_manifest_sha256": forward_manifest_sha,
        "implementation_sha256": _sha256_file(Path(__file__)),
        "config": asdict(cfg),
        "artifacts": artifacts,
    }
    report_path = staging / "holdout_preregistration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "implementation_sha256": report["implementation_sha256"],
                "forward_candidate_manifest_sha256": forward_manifest_sha,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ConditionalDriftHoldoutPreregistrationResult(target.resolve(), report)


def calculate_conditional_drift_holdout_outcomes(
    preregistered_events: pd.DataFrame,
    execution_events: pd.DataFrame,
    event_price_endpoints: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    calendar_sessions: pd.DataFrame,
    config: ConditionalDriftHoldoutConfig | None = None,
) -> pd.DataFrame:
    """Compute the preregistered 20-session net outcome exactly once."""
    cfg = config or ConditionalDriftHoldoutConfig()
    cfg.validate()
    _require_columns(
        preregistered_events,
        {
            "validation_event_id",
            "cik",
            "accepted_at",
            "outcome_read_authorized_after_preregistration",
            "outcome_value_loaded_at_preregistration",
        },
        "preregistered holdout events",
    )
    if not preregistered_events["outcome_read_authorized_after_preregistration"].astype(bool).all():
        raise ConditionalDriftHoldoutError(
            "A holdout event lacks post-preregistration authorization"
        )
    if preregistered_events["outcome_value_loaded_at_preregistration"].astype(bool).any():
        raise ConditionalDriftHoldoutError("A holdout outcome was loaded before preregistration")
    execution_columns = [
        "validation_event_id",
        "formal_entry_session_date",
        "formal_entry_open_at",
        "formal_exit_session_date",
        "formal_exit_close_at",
        "formal_execution_window_ready",
    ]
    endpoint_columns = ["validation_event_id", "price_target_id", "required_endpoint_dates_ready"]
    execution = execution_events.loc[:, execution_columns].drop_duplicates("validation_event_id")
    endpoints = event_price_endpoints.loc[:, endpoint_columns].copy()
    if endpoints["validation_event_id"].duplicated().any():
        raise ConditionalDriftHoldoutError("Holdout evaluation requires one security component")
    events = (
        preregistered_events.loc[:, ["validation_event_id", "cik", "ticker", "accepted_at"]]
        .merge(execution, on="validation_event_id", how="left", validate="one_to_one")
        .merge(endpoints, on="validation_event_id", how="left", validate="one_to_one")
    )
    if (
        events[
            [
                "formal_entry_open_at",
                "formal_exit_close_at",
                "price_target_id",
                "required_endpoint_dates_ready",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ConditionalDriftHoldoutError(
            "A preregistered holdout event lacks an outcome endpoint"
        )
    if (
        not events["formal_execution_window_ready"].astype(bool).all()
        or not events["required_endpoint_dates_ready"].astype(bool).all()
    ):
        raise ConditionalDriftHoldoutError("A preregistered formal price window is incomplete")
    bars = issuer_bars.copy()
    bars["session_date"] = pd.to_datetime(bars["session_date"], errors="coerce").dt.normalize()
    market = benchmark_bars.copy()
    market["session_date"] = pd.to_datetime(market["session_date"], errors="coerce").dt.normalize()
    sessions = calendar_sessions.loc[calendar_sessions["exchange_mic"].eq("XNYS")].copy()
    sessions["session_date"] = pd.to_datetime(
        sessions["session_date"], errors="coerce"
    ).dt.normalize()
    sessions["regular_close_at"] = _utc(sessions["regular_close_at"], "regular_close_at")
    if sessions["session_date"].duplicated().any():
        raise ConditionalDriftHoldoutError("Holdout XNYS calendar contains duplicate sessions")
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        target_id = str(event.price_target_id)
        stock = bars.loc[bars["price_target_id"].astype(str).eq(target_id)].copy()
        stock_actions = issuer_actions.loc[
            issuer_actions["price_target_id"].astype(str).eq(target_id)
        ]
        rows.append(
            _outcome_row(
                event,
                stock,
                market,
                stock_actions,
                benchmark_actions,
                sessions,
                cfg,
            )
        )
    outcomes = pd.DataFrame(rows).sort_values(
        ["accepted_at", "cik", "validation_event_id"], kind="mergesort"
    )
    if len(outcomes) != len(preregistered_events) or not outcomes["outcome_ready"].all():
        raise ConditionalDriftHoldoutError("The preregistered holdout outcome ledger is incomplete")
    return outcomes.reset_index(drop=True)


def calculate_conditional_drift_holdout_diagnostics(
    preregistered_events: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: ConditionalDriftHoldoutConfig | None = None,
) -> ConditionalDriftHoldoutTables:
    """Evaluate frozen scores on the exact preregistered same-event sample."""
    cfg = config or ConditionalDriftHoldoutConfig()
    cfg.validate()
    _require_columns(
        preregistered_events,
        {
            "validation_event_id",
            "cik",
            CHALLENGER_SCORE,
            CANDIDATE_SCORE,
            CLIPPED_CANDIDATE_SCORE,
            "alignment_candidate_within_training_support",
            "alignment_candidate_severe_extrapolation",
        },
        "preregistered holdout events",
    )
    _require_columns(
        outcomes,
        {"validation_event_id", "net_market_excess_return_20d", "outcome_ready"},
        "holdout outcomes",
    )
    analysis = preregistered_events.merge(
        outcomes,
        on="validation_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )
    if len(analysis) != cfg.minimum_events or analysis["cik"].nunique() < cfg.minimum_issuers:
        raise ConditionalDriftHoldoutError("Holdout analysis does not match frozen sample size")
    if not analysis["outcome_ready"].astype(bool).all():
        raise ConditionalDriftHoldoutError("Holdout analysis contains a missing outcome")
    target = "net_market_excess_return_20d"
    metrics = _model_metrics(analysis, target)
    bootstrap = _cluster_bootstrap(analysis, target, cfg)
    leave_one_out = _leave_one_issuer_out(analysis, target)
    support = _support_sensitivity(analysis, target)
    quantiles = _quantile_returns(analysis, target)
    f1 = metrics.loc[metrics["model_id"].eq("F1_v1_challenger")].iloc[0]
    f2 = metrics.loc[metrics["model_id"].eq("F2_alignment_candidate")].iloc[0]
    contrast = bootstrap.loc[bootstrap["statistic"].eq("f2_minus_f1_spearman")].iloc[0]
    loo_positive_share = float(leave_one_out["f2_spearman_ic"].gt(0.0).mean())
    spread = float(f2["top_minus_bottom_tertile_net_return"])
    within_support = int(analysis["alignment_candidate_within_training_support"].sum())
    checks = pd.DataFrame(
        [
            _check("complete_preregistered_outcomes", len(analysis) >= cfg.minimum_events),
            _check("candidate_positive_rank_ic", float(f2["spearman_ic"]) > 0.0),
            _check(
                "candidate_beats_challenger_rank_ic",
                float(f2["spearman_ic"]) - float(f1["spearman_ic"]) > 0.0,
            ),
            _check("incremental_cluster_ci_above_zero", float(contrast["ci_low"]) > 0.0),
            _check(
                "leave_one_issuer_out_positive_share",
                loo_positive_share >= cfg.leave_one_issuer_out_positive_share,
            ),
            _check("candidate_top_bottom_spread_positive", math.isfinite(spread) and spread > 0.0),
            _check("minimum_covariate_overlap", within_support >= cfg.minimum_in_support_events),
        ]
    )
    verdict = (
        "candidate_effectiveness_established"
        if checks["passed"].all()
        else "candidate_effectiveness_not_established"
    )
    report: dict[str, Any] = {
        "contract_version": HOLDOUT_EVALUATION_VERSION,
        "events": len(analysis),
        "issuers": int(analysis["cik"].nunique()),
        "candidate_spearman_ic": float(f2["spearman_ic"]),
        "challenger_spearman_ic": float(f1["spearman_ic"]),
        "candidate_minus_challenger_spearman": float(f2["spearman_ic"]) - float(f1["spearman_ic"]),
        "candidate_minus_challenger_ci_low": float(contrast["ci_low"]),
        "candidate_minus_challenger_ci_high": float(contrast["ci_high"]),
        "candidate_minus_challenger_bootstrap_p": float(contrast["one_sided_p_value"]),
        "candidate_mae": float(f2["mae"]),
        "challenger_mae": float(f1["mae"]),
        "candidate_top_minus_bottom_tertile_net_return": spread,
        "leave_one_issuer_out_positive_share": loo_positive_share,
        "within_training_support_events": within_support,
        "severe_extrapolation_events": int(
            analysis["alignment_candidate_severe_extrapolation"].sum()
        ),
        "verdict": verdict,
        "verdict_checks_passed": int(checks["passed"].sum()),
        "verdict_checks_total": len(checks),
        "external_validation_metrics_computed": True,
        "portfolio_backtest_computed": False,
        "annualized_performance_computed": False,
        "live_trading_authorized": False,
        "investment_recommendation_allowed": False,
        "config": asdict(cfg),
    }
    return ConditionalDriftHoldoutTables(
        outcomes,
        analysis,
        metrics,
        bootstrap,
        leave_one_out,
        support,
        quantiles,
        checks,
        report,
    )


def build_conditional_drift_holdout_evaluation(
    preregistration_root: Path,
    forward_candidate_root: Path,
    execution_root: Path,
    price_root: Path,
    action_root: Path,
    benchmark_root: Path,
    *,
    output_root: Path = Path("data/processed/conditional_drift_holdout_evaluation"),
    config: ConditionalDriftHoldoutConfig | None = None,
) -> ConditionalDriftHoldoutResult:
    """Open outcomes only after validating the immutable preregistration."""
    cfg = config or ConditionalDriftHoldoutConfig()
    cfg.validate()
    prereg_report = _read_json(preregistration_root / "holdout_preregistration_report.json")
    if bool(prereg_report.get("outcome_values_loaded")) or not bool(
        prereg_report.get("hypotheses_frozen_before_outcome_read")
    ):
        raise ConditionalDriftHoldoutError("The holdout preregistration is not outcome-blind")
    forward_sha = _sha256_file(forward_candidate_root / "output_manifest.json")
    if forward_sha != prereg_report.get("forward_candidate_manifest_sha256"):
        raise ConditionalDriftHoldoutError("The scored candidate changed after preregistration")
    manifests = {
        "implementation_sha256": _sha256_file(Path(__file__)),
        "preregistration_manifest_sha256": _sha256_file(
            preregistration_root / "output_manifest.json"
        ),
        "forward_candidate_manifest_sha256": forward_sha,
        "execution_manifest_sha256": _sha256_file(execution_root / "output_manifest.json"),
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "action_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
    }
    preregistered = load_catalog_table(preregistration_root, "preregistered_holdout_events.parquet")
    execution = load_catalog_table(execution_root, "validation_execution_events.parquet")
    endpoints = pd.read_parquet(
        price_root / "validation_price_batch_event_endpoints.parquet",
        columns=["validation_event_id", "price_target_id", "required_endpoint_dates_ready"],
    )
    bars = load_catalog_table(price_root, "validation_price_batch_bars.parquet")
    actions = load_catalog_table(action_root, "normalized_corporate_actions.parquet")
    benchmark = load_catalog_table(benchmark_root, "benchmark_daily_price_observations.parquet")
    benchmark_actions = load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet")
    sessions = load_catalog_table(execution_root, "validation_exchange_sessions.parquet")
    outcomes = calculate_conditional_drift_holdout_outcomes(
        preregistered,
        execution,
        endpoints,
        bars,
        actions,
        benchmark,
        benchmark_actions,
        sessions,
        cfg,
    )
    tables = calculate_conditional_drift_holdout_diagnostics(preregistered, outcomes, cfg)
    payload = json.dumps(
        {"version": HOLDOUT_EVALUATION_VERSION, **manifests, "config": asdict(cfg)},
        sort_keys=True,
    ).encode()
    dataset_version = HOLDOUT_EVALUATION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConditionalDriftHoldoutResult(
            target.resolve(), _read_json(target / "conditional_drift_holdout_report.json")
        )
    if target.exists():
        raise ConditionalDriftHoldoutError(f"Incomplete holdout evaluation exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "conditional_drift_holdout_outcomes.parquet": tables.outcomes,
        "conditional_drift_holdout_analysis_sample.parquet": tables.analysis_sample,
        "conditional_drift_holdout_model_metrics.parquet": tables.model_metrics,
        "conditional_drift_holdout_bootstrap.parquet": tables.bootstrap_inference,
        "conditional_drift_holdout_leave_one_issuer_out.parquet": tables.leave_one_issuer_out,
        "conditional_drift_holdout_support_sensitivity.parquet": tables.support_sensitivity,
        "conditional_drift_holdout_quantile_returns.parquet": tables.quantile_returns,
        "conditional_drift_holdout_verdict_checks.parquet": tables.verdict_checks,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_version": prereg_report["dataset_version"],
        "preregistration_generated_at": prereg_report["generated_at"],
        "outcomes_computed_after_preregistration": True,
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "conditional_drift_holdout_report.json"
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
    return ConditionalDriftHoldoutResult(target.resolve(), report)


def _outcome_row(
    event: Any,
    stock: pd.DataFrame,
    market: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    sessions: pd.DataFrame,
    config: ConditionalDriftHoldoutConfig,
) -> dict[str, Any]:
    entry_date = pd.Timestamp(event.formal_entry_session_date).normalize()
    exit_date = pd.Timestamp(event.formal_exit_session_date).normalize()
    accepted = pd.Timestamp(event.accepted_at)
    entry_at = pd.Timestamp(event.formal_entry_open_at)
    exit_at = pd.Timestamp(event.formal_exit_close_at)
    stock_entry = _bar(stock, entry_date)
    stock_exit = _bar(stock, exit_date)
    market_entry = _bar(market, entry_date)
    market_exit = _bar(market, exit_date)
    if any(value is None for value in (stock_entry, stock_exit, market_entry, market_exit)):
        raise ConditionalDriftHoldoutError("A formal holdout entry or exit bar is missing")
    assert stock_entry is not None
    assert stock_exit is not None
    assert market_entry is not None
    assert market_exit is not None
    issuer_holding, issuer_cashflows = calculate_cash_retained_holding_return(
        float(stock_entry["open_unadjusted"]),
        float(stock_exit["close_unadjusted"]),
        _actions_between(issuer_actions, entry_at, exit_at),
        currency="USD",
    )
    market_holding, market_cashflows = calculate_cash_retained_holding_return(
        float(market_entry["open_unadjusted"]),
        float(market_exit["close_unadjusted"]),
        _actions_between(benchmark_actions, entry_at, exit_at),
        currency="USD",
    )
    stock_clock = stock.merge(
        sessions[["session_date", "regular_close_at"]],
        on="session_date",
        how="left",
        validate="many_to_one",
    )
    prior = stock_clock.loc[stock_clock["regular_close_at"].le(accepted)].tail(20)
    if len(prior) != 20:
        raise ConditionalDriftHoldoutError("A holdout event lacks 20 pre-acceptance cost sessions")
    spread = _corwin_schultz_spread(prior)
    if not math.isfinite(spread):
        raise ConditionalDriftHoldoutError("A holdout event has an invalid spread estimate")
    base_cost_bps = float(
        np.clip(
            spread * 10_000.0 + config.fixed_slippage_bps,
            config.minimum_base_cost_bps,
            config.maximum_base_cost_bps,
        )
    )
    gross = issuer_holding.total_return - market_holding.total_return
    return {
        "validation_event_id": str(event.validation_event_id),
        "cik": str(event.cik),
        "ticker": str(event.ticker),
        "accepted_at": accepted,
        "entry_at": entry_at,
        "entry_price_type": "unadjusted_regular_open",
        "exit_at": exit_at,
        "exit_price_type": "unadjusted_regular_close",
        "holding_sessions": config.primary_horizon_sessions,
        "issuer_total_return_20d": issuer_holding.total_return,
        "market_total_return_20d": market_holding.total_return,
        "gross_market_excess_return_20d": gross,
        "liquidity_cost_base_bps": base_cost_bps,
        "net_market_excess_return_20d": gross - base_cost_bps / 10_000.0,
        "issuer_actions_inside_window": len(issuer_cashflows),
        "market_actions_inside_window": len(market_cashflows),
        "outcome_available_at": exit_at,
        "outcome_ready": True,
        "outcome_computed_after_preregistration": True,
    }


def _model_metrics(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    specifications = (
        ("F1_v1_challenger", CHALLENGER_SCORE, "primary_challenger"),
        ("F2_alignment_candidate", CANDIDATE_SCORE, "primary_candidate"),
        ("F2_alignment_support_clipped", CLIPPED_CANDIDATE_SCORE, "sensitivity_only"),
    )
    rows: list[dict[str, Any]] = []
    actual = frame[target].astype(float)
    for model_id, score_column, evidence_role in specifications:
        score = frame[score_column].astype(float)
        rows.append(
            {
                "model_id": model_id,
                "score_column": score_column,
                "events": len(frame),
                "issuers": int(frame["cik"].nunique()),
                "spearman_ic": _spearman(score, actual),
                "mae": float((score - actual).abs().mean()),
                "rmse": float(np.sqrt(np.square(score - actual).mean())),
                "directional_accuracy": float(
                    (np.sign(score.to_numpy()) == np.sign(actual.to_numpy())).mean()
                ),
                "top_minus_bottom_tertile_net_return": _tertile_spread(score, actual),
                "evidence_role": evidence_role,
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    frame: pd.DataFrame,
    target: str,
    config: ConditionalDriftHoldoutConfig,
) -> pd.DataFrame:
    issuers = sorted(frame["cik"].astype(str).unique())
    groups = {issuer: frame.loc[frame["cik"].astype(str).eq(issuer)] for issuer in issuers}
    rng = np.random.default_rng(config.bootstrap_seed)
    f1_values: list[float] = []
    f2_values: list[float] = []
    differences: list[float] = []
    for _ in range(config.bootstrap_replications):
        sampled = rng.choice(issuers, size=len(issuers), replace=True)
        resample = pd.concat([groups[str(issuer)] for issuer in sampled], ignore_index=True)
        f1 = _spearman(resample[CHALLENGER_SCORE], resample[target])
        f2 = _spearman(resample[CANDIDATE_SCORE], resample[target])
        if math.isfinite(f1) and math.isfinite(f2):
            f1_values.append(f1)
            f2_values.append(f2)
            differences.append(f2 - f1)
    if len(differences) < config.bootstrap_replications * 0.95:
        raise ConditionalDriftHoldoutError("Too many invalid issuer-cluster bootstrap draws")
    rows = []
    for statistic, values in (
        ("f1_spearman", f1_values),
        ("f2_spearman", f2_values),
        ("f2_minus_f1_spearman", differences),
    ):
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "statistic": statistic,
                "replications": len(array),
                "bootstrap_mean": float(array.mean()),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "one_sided_p_value": float((1 + np.count_nonzero(array <= 0.0)) / (1 + len(array))),
                "cluster_unit": "issuer_cik",
            }
        )
    return pd.DataFrame(rows)


def _leave_one_issuer_out(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for cik in sorted(frame["cik"].astype(str).unique()):
        sample = frame.loc[~frame["cik"].astype(str).eq(cik)]
        f1 = _spearman(sample[CHALLENGER_SCORE], sample[target])
        f2 = _spearman(sample[CANDIDATE_SCORE], sample[target])
        rows.append(
            {
                "excluded_cik": cik,
                "remaining_events": len(sample),
                "remaining_issuers": int(sample["cik"].nunique()),
                "f1_spearman_ic": f1,
                "f2_spearman_ic": f2,
                "f2_minus_f1_spearman": f2 - f1,
            }
        )
    return pd.DataFrame(rows)


def _support_sensitivity(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    groups = (
        ("all_preregistered", pd.Series(True, index=frame.index)),
        ("within_training_support", frame["alignment_candidate_within_training_support"]),
        ("excluding_severe_extrapolation", ~frame["alignment_candidate_severe_extrapolation"]),
        ("severe_extrapolation_only", frame["alignment_candidate_severe_extrapolation"]),
    )
    rows = []
    for sample_id, mask in groups:
        sample = frame.loc[mask.astype(bool)]
        rows.append(
            {
                "sample_id": sample_id,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()) if not sample.empty else 0,
                "f1_spearman_ic": _spearman(sample[CHALLENGER_SCORE], sample[target]),
                "f2_spearman_ic": _spearman(sample[CANDIDATE_SCORE], sample[target]),
                "f2_clipped_spearman_ic": _spearman(
                    sample[CLIPPED_CANDIDATE_SCORE], sample[target]
                ),
                "mean_net_market_excess_return": float(sample[target].mean())
                if not sample.empty
                else np.nan,
                "evidence_role": "primary" if sample_id == "all_preregistered" else "sensitivity",
            }
        )
    return pd.DataFrame(rows)


def _quantile_returns(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    ranks = frame[CANDIDATE_SCORE].rank(method="first", pct=True)
    buckets = pd.cut(
        ranks,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["bottom", "middle", "top"],
        include_lowest=True,
    )
    rows = []
    for bucket in ("bottom", "middle", "top"):
        sample = frame.loc[buckets.eq(bucket)]
        rows.append(
            {
                "quantile": bucket,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()),
                "mean_net_market_excess_return": float(sample[target].mean()),
                "median_net_market_excess_return": float(sample[target].median()),
                "positive_share": float(sample[target].gt(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _hypothesis_table(config: ConditionalDriftHoldoutConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": 1,
                "check": "complete_preregistered_outcomes",
                "operator": ">=",
                "threshold": float(config.minimum_events),
            },
            {"order": 2, "check": "candidate_positive_rank_ic", "operator": ">", "threshold": 0.0},
            {
                "order": 3,
                "check": "candidate_beats_challenger_rank_ic",
                "operator": ">",
                "threshold": 0.0,
            },
            {
                "order": 4,
                "check": "incremental_cluster_ci_above_zero",
                "operator": ">",
                "threshold": 0.0,
            },
            {
                "order": 5,
                "check": "leave_one_issuer_out_positive_share",
                "operator": ">=",
                "threshold": config.leave_one_issuer_out_positive_share,
            },
            {
                "order": 6,
                "check": "candidate_top_bottom_spread_positive",
                "operator": ">",
                "threshold": 0.0,
            },
            {
                "order": 7,
                "check": "minimum_covariate_overlap",
                "operator": ">=",
                "threshold": float(config.minimum_in_support_events),
            },
        ]
    )


def _validate_preregistration_scores(
    scores: pd.DataFrame,
    report: dict[str, Any],
    config: ConditionalDriftHoldoutConfig,
) -> None:
    forbidden = [
        column
        for column in scores.columns
        if column.startswith("target_")
        or column.startswith("actual_")
        or column in {"net_market_excess_return_20d", "issuer_total_return_20d"}
    ]
    if forbidden:
        raise ConditionalDriftHoldoutError(
            "Preregistration score asset contains outcome columns: " + ", ".join(forbidden)
        )
    required = {
        "validation_event_id",
        "cik",
        "ticker",
        "accepted_at",
        "reaction_feature_available_at",
        "execution_entry_at",
        "prediction_generated_at",
        "model_training_label_end",
        "evidence_role",
        CHALLENGER_SCORE,
        CANDIDATE_SCORE,
        CLIPPED_CANDIDATE_SCORE,
        "alignment_candidate_within_training_support",
        "alignment_candidate_severe_extrapolation",
        "outcome_columns_loaded",
    }
    _require_columns(scores, required, "frozen holdout scores")
    if len(scores) != config.minimum_events or scores["cik"].nunique() < config.minimum_issuers:
        raise ConditionalDriftHoldoutError("Frozen holdout score count does not match the protocol")
    if scores["validation_event_id"].duplicated().any() or scores["outcome_columns_loaded"].any():
        raise ConditionalDriftHoldoutError("Frozen holdout scores violate the outcome firewall")
    if not bool(report.get("external_validation_metrics_computed") is False):
        raise ConditionalDriftHoldoutError(
            "Forward candidate metrics were opened before preregistration"
        )
    if int(report.get("scored_candidate_events", -1)) != len(scores):
        raise ConditionalDriftHoldoutError(
            "Forward candidate report does not match its score asset"
        )


def _bar(frame: pd.DataFrame, session_date: pd.Timestamp) -> pd.Series[Any] | None:
    selected = frame.loc[frame["session_date"].eq(session_date.tz_localize(None).normalize())]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise ConditionalDriftHoldoutError("A holdout price date contains duplicate bars")
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


def _spearman(left: pd.Series[Any], right: pd.Series[Any]) -> float:
    if len(left) < 3 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _tertile_spread(predicted: pd.Series[Any], actual: pd.Series[Any]) -> float:
    if len(predicted) < 9 or predicted.nunique(dropna=True) < 3:
        return float("nan")
    ranks = predicted.rank(method="first", pct=True)
    return float(actual.loc[ranks.gt(2.0 / 3.0)].mean() - actual.loc[ranks.le(1.0 / 3.0)].mean())


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConditionalDriftHoldoutError(f"{label} is missing columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise ConditionalDriftHoldoutError(f"{label} contains invalid timestamps")
    return result


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
