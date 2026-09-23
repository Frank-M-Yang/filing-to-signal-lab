"""Isolated, chronological risk-forecast diagnostic; no orders, exposures or alpha.

The input is a reconstructed academic return panel, NOT a historical real-time
feed. Chronological fitting removes one kind of leakage, not vintage/release bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from market_stress_sources import INDUSTRIES

BASE_FEATURES = ("log_rv5", "log_rv20", "log_rv60")
STATE_FEATURES = ("corr20", "corr60")
MODELS = ("rv20_persistence", "volatility_only_ridge", "volatility_plus_correlation_ridge")
FEATURES = (*BASE_FEATURES, *STATE_FEATURES)


@dataclass(frozen=True)
class StressConfig:
    """Fixed numerical choices; changing them after a replay is new exploration."""

    train_start: str = "2010-01-01"
    evaluation_start: str = "2019-01-01"
    evaluation_end: str = "2021-12-31"
    horizon_sessions: int = 5
    annualization: int = 252
    variance_floor: float = 1e-12
    ridge_alpha: float = 10.0
    minimum_train_rows: int = 1000
    alarm_quantile: float = 0.9
    downside_return_threshold: float = -0.05
    bootstrap_block: int = 20
    bootstrap_repetitions: int = 2000
    seed: int = 202009
    minimum_relative_improvement: float = 0.05

    def __post_init__(self) -> None:
        if (
            not pd.Timestamp(self.train_start)
            < pd.Timestamp(self.evaluation_start)
            <= pd.Timestamp(self.evaluation_end)
        ):
            raise ValueError("Training must begin before chronological evaluation")
        if (
            min(
                self.horizon_sessions,
                self.annualization,
                self.minimum_train_rows,
                self.bootstrap_block,
                self.bootstrap_repetitions,
            )
            < 1
        ):
            raise ValueError("Window sizes must be positive")
        if self.ridge_alpha <= 0 or self.variance_floor <= 0 or not 0 < self.alarm_quantile < 1:
            raise ValueError("Invalid numerical configuration")


def make_panel(returns: pd.DataFrame, config: StressConfig) -> pd.DataFrame:
    """Use returns through origin t to forecast sessions t+1 through t+h inclusive.

    No interior row is dropped before rolling or shifting. A missing observation
    invalidates every affected window. All 45 industry pairs must be available.
    """
    names = ["market_return", *[f"industry_{x}" for x in INDUSTRIES]]
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.tz is not None:
        raise ValueError("Expected timezone-naive source session dates, not availability times")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Sessions must be unique and ordered")
    values = returns[names].astype(float)
    if np.isinf(values.to_numpy()).any() or (values < -1).any().any():
        raise ValueError("Invalid simple returns")
    panel = pd.DataFrame(index=returns.index.copy())
    panel.index.name = "origin_date"
    market = values.market_return
    panel["market_return"] = market
    for window in (5, 20, 60):
        variance = market.pow(2).rolling(window, min_periods=window).mean() * config.annualization
        panel[f"log_rv{window}"] = np.log(variance.clip(lower=config.variance_floor))
    sectors = values.iloc[:, 1:]
    for window in (20, 60):
        pairs = pd.concat(
            [
                sectors[a].rolling(window, min_periods=window).corr(sectors[b])
                for a, b in combinations(sectors.columns, 2)
            ],
            axis=1,
        )
        panel[f"corr{window}"] = pairs.mean(axis=1).where(pairs.notna().all(axis=1))
    future = pd.concat([market.shift(-i) for i in range(1, config.horizon_sessions + 1)], axis=1)
    complete = future.notna().all(axis=1)
    panel["target_variance"] = (future.pow(2).mean(axis=1) * config.annualization).where(complete)
    panel["future_return"] = (
        (1 + future).prod(axis=1, min_count=config.horizon_sessions) - 1
    ).where(complete)
    panel["future_downside_variance"] = (
        future.clip(upper=0).pow(2).mean(axis=1) * config.annualization
    ).where(complete)
    sessions = pd.Series(returns.index, index=returns.index)
    panel["target_start"] = sessions.shift(-1)
    panel["target_end"] = sessions.shift(-config.horizon_sessions)
    panel["feature_complete"] = np.isfinite(panel[list(FEATURES)]).all(axis=1)
    panel["target_complete"] = complete & panel.target_variance.gt(config.variance_floor)
    return panel


def _ridge_prediction(
    train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...], config: StressConfig
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    x = train[list(features)].to_numpy(dtype=float)
    mu, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (x - mu) / scale
    y = np.log(train.target_variance.to_numpy(dtype=float))
    intercept = float(y.mean())
    beta = np.linalg.solve(
        standardized.T @ standardized + config.ridge_alpha * np.eye(len(features)),
        standardized.T @ (y - intercept),
    )
    prediction = np.exp(
        intercept + ((test[list(features)].to_numpy(dtype=float) - mu) / scale) @ beta
    )
    if not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError("Nonfinite forecast; do not silently clip or remove failures")
    return prediction, {
        "features": list(features),
        "training_mean": mu.tolist(),
        "training_std": scale.tolist(),
        "coefficients": beta.tolist(),
        "intercept": intercept,
    }


def forecast_panel(
    panel: pd.DataFrame, config: StressConfig
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate monthly-refitted predictions without reading test labels to fit/select.

    All arms use the same complete-feature origins. Test-target availability is
    deliberately not an eligibility condition here; it is applied only at scoring.
    """
    evaluation = panel.loc[
        (panel.index >= config.evaluation_start) & (panel.index <= config.evaluation_end)
    ]
    evaluation = evaluation.loc[evaluation.feature_complete]
    predictions, fits = [], []
    for month, block in evaluation.groupby(
        pd.DatetimeIndex(evaluation.index).to_period("M"), sort=True
    ):
        first_origin = block.index.min()
        train = panel.loc[
            (panel.index >= pd.Timestamp(config.train_start))
            & panel.feature_complete
            & panel.target_complete
            & (panel.target_end < first_origin)
        ]
        if len(train) < config.minimum_train_rows:
            raise ValueError(f"Insufficient matured training sample for {month}: {len(train)}")
        if not (train.index.max() < first_origin and train.target_end.max() < first_origin):
            raise ValueError("Training outcome not strictly earlier than forecast origin")
        result = pd.DataFrame(index=block.index.copy())
        result["model_fit_origin"] = first_origin
        result["training_origins"] = len(train)
        result["training_label_end"] = train.target_end.max()
        result["synthetic_forecast_at_utc"] = pd.DatetimeIndex(result.index).tz_localize(
            "UTC"
        ) + pd.Timedelta(hours=23, minutes=59)
        result["alarm_threshold_variance"] = float(
            train.target_variance.quantile(config.alarm_quantile)
        )
        result[MODELS[0]] = np.exp(block.log_rv20)
        fit: dict[str, Any] = {
            "month": str(month),
            "origins": len(train),
            "training_label_end": str(train.target_end.max().date()),
            "first_forecast_origin": str(first_origin.date()),
            "models": {},
        }
        for name, features in ((MODELS[1], BASE_FEATURES), (MODELS[2], FEATURES)):
            result[name], fit["models"][name] = _ridge_prediction(train, block, features, config)
        predictions.append(result)
        fits.append(fit)
    if not predictions:
        raise ValueError("No forecast origins")
    return pd.concat(predictions), fits


def paired_block_interval(
    candidate: NDArray[np.float64], baseline: NDArray[np.float64], config: StressConfig
) -> dict[str, Any]:
    """Descriptive paired moving-block interval; overlapping horizons are not IID."""
    n = len(candidate)
    if n < 2 * config.bootstrap_block:
        return {"available": False, "reason": "fewer_than_two_blocks"}
    rng = np.random.default_rng(config.seed)
    blocks = int(np.ceil(n / config.bootstrap_block))
    starts = rng.integers(
        0, n - config.bootstrap_block + 1, size=(config.bootstrap_repetitions, blocks)
    )
    indices = (starts[:, :, None] + np.arange(config.bootstrap_block)).reshape(
        config.bootstrap_repetitions, -1
    )[:, :n]
    deltas = (candidate[indices] - baseline[indices]).mean(axis=1)
    relative = candidate[indices].mean(axis=1) / baseline[indices].mean(axis=1) - 1
    return {
        "available": True,
        "candidate_minus_baseline_95pct": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "relative_loss_change_95pct": np.quantile(relative, [0.025, 0.975]).tolist(),
        "block_origins": config.bootstrap_block,
        "independent_crisis_inference": False,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_forecasts(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    config: StressConfig,
    *,
    descriptive_periods: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score identical origins; optional date slices affect reporting, never fitting.

    Pass predeclared slices for another episode. Default slices preserve the
    original 2020 pilot verbatim. Calendar-year screening follows the evaluation
    interval, not hardcoded 2019-2021 years.
    """
    ledger = predictions.join(
        panel[
            [
                "target_variance",
                "future_return",
                "future_downside_variance",
                "target_start",
                "target_end",
                "target_complete",
                *FEATURES,
            ]
        ]
    )
    ledger = ledger.loc[ledger.target_complete].copy()
    if ledger.empty:
        raise ValueError("No complete five-session outcomes")
    if not (ledger.target_start > ledger.index).all():
        raise ValueError("Forecast horizon includes contemporaneous or earlier return")
    high = ledger.target_variance > ledger.alarm_threshold_variance
    downside = ledger.future_return <= config.downside_return_threshold
    ledger["high_realized_risk"] = high
    ledger["downside_event"] = downside
    for name in MODELS:
        ratio = ledger.target_variance / ledger[name]
        ledger[f"qlike_{name}"] = ratio - np.log(ratio) - 1
        ledger[f"ae_vol_{name}"] = (
            np.sqrt(ledger.target_variance) - np.sqrt(ledger[name])
        ).abs() * 100
        ledger[f"alarm_{name}"] = ledger[name] > ledger.alarm_threshold_variance
    slices: dict[str, Any] = {}
    dates = pd.DatetimeIndex(ledger.index)
    years = range(
        pd.Timestamp(config.evaluation_start).year, pd.Timestamp(config.evaluation_end).year + 1
    )
    masks = {
        "all": np.ones(len(ledger), dtype=bool),
        **{str(y): dates.year == y for y in years},
        "2020_Q1": (dates.year == 2020) & (dates.month <= 3),
        "2020_Apr_Dec": (dates.year == 2020) & (dates.month >= 4),
    }
    if descriptive_periods is not None:
        masks = {"all": np.ones(len(ledger), dtype=bool)}
        for label, (start, end) in descriptive_periods.items():
            if label == "all" or not pd.Timestamp(config.evaluation_start) <= pd.Timestamp(
                start
            ) <= pd.Timestamp(end) <= pd.Timestamp(config.evaluation_end):
                raise ValueError("Descriptive period is outside evaluation or overwrites all")
            masks[label] = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        for year in years:
            label = str(year)
            expected = dates.year == year
            if label not in masks or not np.array_equal(masks[label], expected):
                raise ValueError("Every evaluation calendar year must have a complete slice")
    for label, mask in masks.items():
        subset = ledger.loc[mask]
        if subset.empty:
            continue
        stats: dict[str, Any] = {"scored_origins": len(subset), "models": {}}
        for name in MODELS:
            alarm = subset[f"alarm_{name}"]
            truth = subset.high_realized_risk
            stats["models"][name] = {
                "qlike": float(subset[f"qlike_{name}"].mean()),
                "volatility_mae_pp": float(subset[f"ae_vol_{name}"].mean()),
                "alarm_origins": int(alarm.sum()),
                "high_risk_origins": int(truth.sum()),
                "recall_high_risk": _rate(int((alarm & truth).sum()), int(truth.sum())),
                "false_positive_rate": _rate(int((alarm & ~truth).sum()), int((~truth).sum())),
                "false_discovery_rate": _rate(int((alarm & ~truth).sum()), int(alarm.sum())),
                "downside_origins": int(subset.downside_event.sum()),
                "recall_downside": _rate(
                    int((alarm & subset.downside_event).sum()), int(subset.downside_event.sum())
                ),
                "alarm_followed_by_positive_return_origins": int(
                    (alarm & subset.future_return.gt(0)).sum()
                ),
            }
        baseline, candidate = (subset[f"qlike_{x}"].to_numpy(dtype=float) for x in MODELS[1:])
        stats["relative_qlike_change"] = float(candidate.mean() / baseline.mean() - 1)
        stats["paired_uncertainty"] = paired_block_interval(candidate, baseline, config)
        slices[label] = stats
    all_stats = slices["all"]
    improved_years = sum(
        slices[str(y)]["relative_qlike_change"] < 0 for y in years if str(y) in slices
    )
    interval = all_stats["paired_uncertainty"]
    passes = (
        all_stats["relative_qlike_change"] <= -config.minimum_relative_improvement
        and interval["available"]
        and interval["candidate_minus_baseline_95pct"][1] < 0
        and improved_years >= 2
    )
    summary = {
        "scored_origins": len(ledger),
        "forecast_origins": len(predictions),
        "forecasts_without_complete_outcomes": len(predictions) - len(ledger),
        "evaluation_sessions_without_complete_features": int(
            (
                ~panel.loc[
                    (panel.index >= config.evaluation_start)
                    & (panel.index <= config.evaluation_end)
                ].feature_complete
            ).sum()
        ),
        "monthly_fits": int(ledger.model_fit_origin.nunique()),
        "slices": slices,
        "years_with_lower_qlike": improved_years,
        "historical_candidate_screen_passed": bool(passes),
        "verdict": "historical_followup_candidate_only"
        if passes
        else "incremental_risk_value_not_established",
        "data_vintage": "retrospectively_reconstructed",
        "model_fit_is_chronological": True,
        "live_availability_certified": False,
        "liquidity_tested": False,
        "independent_crises": 1,
        "market_impact_or_dynamical_instability_proven": False,
        "manifold_model_used": False,
        "deep_learning_used": False,
        "untouched_holdout": False,
        "trading_admission": False,
        "strategy_returns_computed": False,
    }
    return ledger, summary
