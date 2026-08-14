"""Strict point-in-time M0-M4 gross-margin forecast tournament.

The tournament evaluates financial forecasts only.  It deliberately stops before
market prices or returns, and it never selects a model from representation
distinctness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

MODEL_NAMES: Final[dict[str, str]] = {
    "M0": "Seasonal naive",
    "M1": "Frozen pooled OLS v1",
    "M2": "Company-aware OLS",
    "M3": "Richer PIT ridge",
    "M4": "Nonlinear ridge robustness",
}

Estimator = Literal["ols", "ridge", "nonlinear_ridge"]

M1_FEATURES: Final[tuple[str, ...]] = (
    "gm_yoy_lag",
    "gm_delta_yoy_lag",
    "revenue_growth_lag1",
    "industry_gm_lag1",
)

M2_FEATURES: Final[tuple[str, ...]] = (
    "company_gm_lag1",
    "gm_yoy_lag",
    "company_gm_delta_yoy_lag1",
    "company_trailing_gm_mean",
    "company_trailing_gm_volatility",
    "revenue_growth_lag1",
    "industry_gm_lag1",
)

M3_FEATURES: Final[tuple[str, ...]] = (
    *M2_FEATURES,
    "peer_log_report_count",
    "peer_mean_gm",
    "peer_mean_delta",
    "peer_days_since_latest_report",
    "peer_information_missing",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "company_id",
    "ticker",
    "sic1",
    "sic2",
    "period_end",
    "actual_available_at",
    "gross_margin",
    "gm_yoy_lag",
    "gm_yoy_lag_available_at",
    "gm_delta_yoy",
    "gm_delta_yoy_lag",
    "revenue_growth_lag1",
    "revenue_growth_lag1_available_at",
    "industry_gm_lag1",
    "industry_gm_lag1_available_at",
    "company_trailing_gm_mean",
    "company_trailing_gm_volatility",
    "company_trailing_gm_history_end",
    "gm_forecast",
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_training_observations",
    "forecast_generated_at",
)


class ForecastTournamentError(ValueError):
    """Raised when tournament inputs or outputs violate the research contract."""


@dataclass(frozen=True)
class ForecastTournamentConfig:
    """Frozen v2 candidate, scoring, calibration, and stability settings."""

    min_train_observations: int = 200
    ridge_alpha: float = 10.0
    nonlinear_ridge_alpha: float = 25.0
    min_eligible_year_events: int = 25
    min_group_events: int = 5
    bootstrap_repetitions: int = 1_000
    bootstrap_seed: int = 20_260_813
    calibration_absolute_bias_limit: float = 0.005
    calibration_bias_to_naive_mae_limit: float = 0.25
    calibration_slope_lower: float = 0.50
    calibration_slope_upper: float = 1.50
    min_prior_quarter_gap_days: int = 45
    max_prior_quarter_gap_days: int = 150

    def __post_init__(self) -> None:
        if self.min_train_observations <= len(M2_FEATURES):
            raise ValueError("min_train_observations must exceed M2 feature count")
        if self.ridge_alpha <= 0.0 or self.nonlinear_ridge_alpha <= 0.0:
            raise ValueError("Ridge penalties must be positive")
        if self.min_eligible_year_events < 5:
            raise ValueError("Eligible years require at least five events")
        if self.min_group_events < 2:
            raise ValueError("Diagnostic groups require at least two events")
        if self.bootstrap_repetitions < 100:
            raise ValueError("Bootstrap diagnostics require at least 100 repetitions")
        if self.min_prior_quarter_gap_days >= self.max_prior_quarter_gap_days:
            raise ValueError("Prior-quarter gap bounds are invalid")


@dataclass(frozen=True)
class ForecastTournamentResult:
    """Auditable tournament predictions, diagnostics, gates, and report."""

    panel: pd.DataFrame
    common_sample: pd.DataFrame
    metrics: pd.DataFrame
    metrics_by_year: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    calibration: pd.DataFrame
    eligibility_gate: pd.DataFrame
    representation_diagnostics: pd.DataFrame
    loss_events: pd.DataFrame
    subgroup_diagnostics: pd.DataFrame
    feature_clock_audit: pd.DataFrame
    report: dict[str, Any]


def build_forecast_tournament(
    representation_panel: pd.DataFrame,
    config: ForecastTournamentConfig | None = None,
) -> ForecastTournamentResult:
    """Run the preregistered M0-M4 tournament on one locked complete-case sample."""
    cfg = config or ForecastTournamentConfig()
    panel = _prepare_panel(representation_panel)
    panel = panel.merge(
        _prior_company_quarter_features(panel, cfg),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        _same_period_peer_features(panel),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    panel["peer_report_count"] = panel["peer_report_count"].fillna(0).astype(int)
    panel["peer_log_report_count"] = np.log1p(panel["peer_report_count"])
    panel["peer_mean_gm"] = panel["peer_mean_gm"].fillna(0.0)
    panel["peer_mean_delta"] = panel["peer_mean_delta"].fillna(0.0)
    panel["peer_days_since_latest_report"] = panel["peer_days_since_latest_report"].fillna(0.0)
    panel["peer_information_missing"] = panel["peer_information_available_at"].isna().astype(float)

    panel["M0_forecast"] = panel["gm_yoy_lag"]
    panel["M0_feature_cutoff"] = panel["gm_yoy_lag_available_at"]
    panel["M0_training_end"] = pd.NaT
    panel["M0_training_observations"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    panel["M0_generated_at"] = panel["M0_feature_cutoff"]

    panel["M1_forecast"] = panel["gm_forecast"]
    panel["M1_feature_cutoff"] = panel["forecast_feature_cutoff"]
    panel["M1_training_end"] = panel["forecast_training_end"]
    panel["M1_training_observations"] = panel["forecast_training_observations"]
    panel["M1_generated_at"] = panel["forecast_generated_at"]

    m2_cutoffs = (
        "company_gm_lag1_available_at",
        "gm_yoy_lag_available_at",
        "company_gm_lag1_available_at",
        "company_trailing_gm_history_end",
        "revenue_growth_lag1_available_at",
        "industry_gm_lag1_available_at",
    )
    panel["M2_feature_cutoff"] = panel.loc[:, list(m2_cutoffs)].max(axis="columns")
    panel["M3_feature_cutoff"] = panel.loc[
        :, ["M2_feature_cutoff", "peer_information_available_at"]
    ].max(axis="columns")
    panel["M4_feature_cutoff"] = panel["M3_feature_cutoff"]

    candidates: tuple[tuple[str, tuple[str, ...], Estimator, float], ...] = (
        ("M2", M2_FEATURES, "ols", 0.0),
        ("M3", M3_FEATURES, "ridge", cfg.ridge_alpha),
        ("M4", M3_FEATURES, "nonlinear_ridge", cfg.nonlinear_ridge_alpha),
    )
    for model_id, features, estimator, alpha in candidates:
        forecasts = _expanding_model_forecast(
            panel,
            feature_columns=features,
            cutoff_column=f"{model_id}_feature_cutoff",
            estimator=estimator,
            alpha=alpha,
            min_train_observations=cfg.min_train_observations,
        )
        for column in (
            "forecast",
            "training_end",
            "training_observations",
            "generated_at",
        ):
            panel[f"{model_id}_{column}"] = forecasts[column]

    for model_id in MODEL_NAMES:
        panel[f"{model_id}_error"] = panel["gross_margin"] - panel[f"{model_id}_forecast"]
        panel[f"{model_id}_innovation_raw"] = panel[f"{model_id}_error"]
        panel[f"{model_id}_expectation_adjustment"] = (
            panel["gm_yoy_lag"] - panel[f"{model_id}_forecast"]
        )

    validate_forecast_tournament(panel)
    forecast_columns = [f"{model_id}_forecast" for model_id in MODEL_NAMES]
    common = panel.dropna(subset=["gross_margin", "gm_delta_yoy", *forecast_columns]).copy()
    if common.empty:
        raise ForecastTournamentError("No locked common M0-M4 forecast sample is available")
    common["availability_year"] = common["actual_available_at"].dt.year
    common["availability_quarter"] = (
        common["actual_available_at"].dt.tz_localize(None).dt.to_period("Q").astype("string")
    )

    metrics = _model_metrics_table(common)
    metrics_by_year = _metrics_by_year(common)
    bootstrap = _bootstrap_intervals(common, cfg)
    calibration = _calibration_table(common, metrics, cfg)
    gates = _eligibility_gate(metrics, metrics_by_year, calibration, bootstrap, cfg)
    representation = _representation_diagnostics(common)
    losses = _loss_events(common)
    subgroups = _subgroup_diagnostics(losses, cfg)
    clock_audit = _feature_clock_audit(panel)
    report = _tournament_report(
        panel,
        common,
        metrics,
        metrics_by_year,
        calibration,
        gates,
        bootstrap,
        cfg,
    )
    panel["forecast_tournament_version"] = report["forecast_tournament_version"]
    common["forecast_tournament_version"] = report["forecast_tournament_version"]
    return ForecastTournamentResult(
        panel=panel,
        common_sample=common,
        metrics=metrics,
        metrics_by_year=metrics_by_year,
        bootstrap_intervals=bootstrap,
        calibration=calibration,
        eligibility_gate=gates,
        representation_diagnostics=representation,
        loss_events=losses,
        subgroup_diagnostics=subgroups,
        feature_clock_audit=clock_audit,
        report=report,
    )


def validate_forecast_tournament(panel: pd.DataFrame) -> None:
    """Validate point-in-time clocks and forecast-error accounting identities."""
    _require_columns(panel, ("actual_available_at", "gross_margin", "gm_yoy_lag"))
    actual_time = _utc(panel["actual_available_at"], "actual_available_at")
    actual = pd.to_numeric(panel["gross_margin"], errors="coerce")
    seasonal = pd.to_numeric(panel["gm_yoy_lag"], errors="coerce")
    for model_id in MODEL_NAMES:
        forecast_column = f"{model_id}_forecast"
        rows = panel[forecast_column].notna()
        if not rows.any():
            if model_id != "M0":
                raise ForecastTournamentError(f"{model_id} produced no forecasts")
            continue
        cutoff = _utc(panel.loc[rows, f"{model_id}_feature_cutoff"], f"{model_id} cutoff")
        if (cutoff >= actual_time.loc[rows]).any():
            raise ForecastTournamentError(f"{model_id} uses a same-time or future feature")
        if model_id != "M0":
            training_end = _utc(
                panel.loc[rows, f"{model_id}_training_end"], f"{model_id} training end"
            )
            generated_at = _utc(
                panel.loc[rows, f"{model_id}_generated_at"], f"{model_id} generated at"
            )
            if (training_end >= actual_time.loc[rows]).any():
                raise ForecastTournamentError(f"{model_id} trains on same-time or future actuals")
            if (generated_at >= actual_time.loc[rows]).any():
                raise ForecastTournamentError(f"{model_id} is generated after the target actual")
            if (training_end > generated_at).any() or (cutoff > generated_at).any():
                raise ForecastTournamentError(f"{model_id} has inconsistent forecast clocks")
        expected_error = actual.loc[rows] - pd.to_numeric(
            panel.loc[rows, forecast_column], errors="coerce"
        )
        error = pd.to_numeric(panel.loc[rows, f"{model_id}_error"], errors="coerce")
        if not np.allclose(expected_error, error, rtol=1e-12, atol=1e-12):
            raise ForecastTournamentError(f"{model_id} forecast error does not tie")
        adjustment = seasonal.loc[rows] - pd.to_numeric(
            panel.loc[rows, forecast_column], errors="coerce"
        )
        innovation = pd.to_numeric(panel.loc[rows, f"{model_id}_innovation_raw"], errors="coerce")
        delta = actual.loc[rows] - seasonal.loc[rows]
        if not np.allclose(innovation, delta + adjustment, rtol=1e-12, atol=1e-12):
            raise ForecastTournamentError(f"{model_id} Innovation identity does not tie")


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel, REQUIRED_COLUMNS)
    result = panel.copy()
    if result.empty or result["event_id"].duplicated().any():
        raise ForecastTournamentError("Representation panel must be non-empty and unique")
    for column in (
        "actual_available_at",
        "gm_yoy_lag_available_at",
        "revenue_growth_lag1_available_at",
        "industry_gm_lag1_available_at",
        "company_trailing_gm_history_end",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_generated_at",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    numeric_columns = {
        "gross_margin",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gm_delta_yoy_lag",
        "revenue_growth_lag1",
        "industry_gm_lag1",
        "company_trailing_gm_mean",
        "company_trailing_gm_volatility",
        "gm_forecast",
    }
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["forecast_training_observations"] = pd.to_numeric(
        result["forecast_training_observations"], errors="coerce"
    ).astype("Int64")
    result["company_id"] = result["company_id"].astype("string").str.zfill(10)
    result["sic1"] = result["sic1"].astype("string")
    result["sic2"] = result["sic2"].astype("string").str.zfill(2)
    result = result.sort_values(["actual_available_at", "company_id", "event_id"]).reset_index(
        drop=True
    )
    if result[["period_end", "actual_available_at"]].isna().any().any():
        raise ForecastTournamentError("Tournament panel contains invalid event dates")
    return result


def _prior_company_quarter_features(
    panel: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    company_panels = {
        str(company_id): group.sort_values(["period_end", "actual_available_at", "event_id"])
        for company_id, group in panel.groupby("company_id", sort=False)
    }
    for target in panel.itertuples(index=False):
        history = company_panels[str(target.company_id)]
        gap_days = (pd.Timestamp(str(target.period_end)) - history["period_end"]).dt.days
        eligible = history.loc[
            gap_days.between(
                config.min_prior_quarter_gap_days,
                config.max_prior_quarter_gap_days,
                inclusive="both",
            )
            & (history["actual_available_at"] < pd.Timestamp(str(target.actual_available_at)))
        ]
        if eligible.empty:
            rows.append({"event_id": target.event_id})
            continue
        prior = eligible.sort_values(["period_end", "actual_available_at", "event_id"]).iloc[-1]
        rows.append(
            {
                "event_id": target.event_id,
                "company_gm_lag1": prior["gross_margin"],
                "company_gm_delta_yoy_lag1": prior["gm_delta_yoy"],
                "company_gm_lag1_event_id": prior["event_id"],
                "company_gm_lag1_available_at": prior["actual_available_at"],
            }
        )
    return pd.DataFrame(rows)


def _same_period_peer_features(panel: pd.DataFrame) -> pd.DataFrame:
    source = panel.copy()
    source["period_bucket"] = source["period_end"].dt.to_period("Q").astype("string")
    rows: list[dict[str, Any]] = []
    for target in source.itertuples(index=False):
        eligible = source.loc[
            source["period_bucket"].eq(target.period_bucket)
            & source["company_id"].ne(target.company_id)
            & (source["actual_available_at"] < pd.Timestamp(str(target.actual_available_at)))
            & source["gross_margin"].notna()
            & source["gm_delta_yoy"].notna()
        ]
        sic2 = eligible.loc[eligible["sic2"].eq(target.sic2)]
        selected = sic2 if not sic2.empty else eligible.loc[eligible["sic1"].eq(target.sic1)]
        if selected.empty:
            rows.append({"event_id": target.event_id})
            continue
        latest = selected["actual_available_at"].max()
        target_time = pd.Timestamp(str(target.actual_available_at))
        rows.append(
            {
                "event_id": target.event_id,
                "peer_report_count": int(selected["company_id"].nunique()),
                "peer_mean_gm": float(selected["gross_margin"].mean()),
                "peer_mean_delta": float(selected["gm_delta_yoy"].mean()),
                "peer_days_since_latest_report": float(
                    (target_time - pd.Timestamp(str(latest))).total_seconds() / 86_400.0
                ),
                "peer_information_available_at": latest,
                "peer_group_level": "sic2" if not sic2.empty else "sic1_fallback",
            }
        )
    return pd.DataFrame(rows)


def _expanding_model_forecast(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    cutoff_column: str,
    estimator: Estimator,
    alpha: float,
    min_train_observations: int,
) -> pd.DataFrame:
    result = pd.DataFrame(index=panel.index)
    result["forecast"] = np.nan
    result["training_end"] = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns, UTC]")
    result["training_observations"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    result["generated_at"] = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns, UTC]")
    complete_columns = ["gross_margin", cutoff_column, *feature_columns]
    for _, current in panel.groupby("actual_available_at", sort=True):
        timestamp = cast(pd.Timestamp, current["actual_available_at"].iloc[0])
        training = panel.loc[panel["actual_available_at"] < timestamp].dropna(
            subset=complete_columns
        )
        current_complete = current.loc[:, [cutoff_column, *feature_columns]].notna().all(axis=1)
        current_index = current.index[current_complete]
        if len(training) < min_train_observations or current_index.empty:
            continue
        training_x = training.loc[:, feature_columns].to_numpy(dtype=float)
        current_x = panel.loc[current_index, feature_columns].to_numpy(dtype=float)
        training_y = training["gross_margin"].to_numpy(dtype=float)
        if estimator == "ols":
            design = np.column_stack([np.ones(len(training_x)), training_x])
            coefficients, *_ = np.linalg.lstsq(design, training_y, rcond=None)
            forecasts = np.column_stack([np.ones(len(current_x)), current_x]) @ coefficients
        else:
            mean = training_x.mean(axis=0)
            scale = training_x.std(axis=0, ddof=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
            train_z = (training_x - mean) / scale
            current_z = (current_x - mean) / scale
            if estimator == "nonlinear_ridge":
                train_z = _nonlinear_design(train_z)
                current_z = _nonlinear_design(current_z)
            centered_y = training_y - training_y.mean()
            penalty = alpha * np.eye(train_z.shape[1])
            coefficients = np.linalg.solve(train_z.T @ train_z + penalty, train_z.T @ centered_y)
            forecasts = training_y.mean() + current_z @ coefficients
        training_end = cast(pd.Timestamp, training["actual_available_at"].max())
        cutoffs = panel.loc[current_index, cutoff_column]
        generated_at = cutoffs.where(cutoffs > training_end, training_end)
        result.loc[current_index, "forecast"] = forecasts
        result.loc[current_index, "training_end"] = training_end
        result.loc[current_index, "training_observations"] = len(training)
        result.loc[current_index, "generated_at"] = generated_at
    return result


def _nonlinear_design(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -4.0, 4.0)
    interaction_pairs = ((0, 1), (0, 2), (1, 3), (5, 6), (8, 9))
    interactions = np.column_stack(
        [clipped[:, left] * clipped[:, right] for left, right in interaction_pairs]
    )
    return np.column_stack([clipped, np.square(clipped), interactions])


def _model_metrics_table(common: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_metric_row(common, model_id) for model_id in MODEL_NAMES])


def _metrics_by_year(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in common.groupby("availability_year", sort=True):
        for model_id in MODEL_NAMES:
            rows.append(_metric_row(group, model_id, availability_year=int(str(year))))
    return pd.DataFrame(rows)


def _metric_row(
    group: pd.DataFrame,
    model_id: str,
    *,
    availability_year: int | None = None,
) -> dict[str, Any]:
    error = pd.to_numeric(group[f"{model_id}_error"], errors="coerce")
    naive = pd.to_numeric(group["M0_error"], errors="coerce")
    model_mae = float(error.abs().mean())
    naive_mae = float(naive.abs().mean())
    model_rmse = float(np.sqrt(np.square(error).mean()))
    naive_rmse = float(np.sqrt(np.square(naive).mean()))
    naive_sse = float(np.square(naive).sum())
    return {
        "availability_year": availability_year,
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "forecast_events": len(group),
        "model_bias": float(error.mean()),
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "naive_seasonal_mae": naive_mae,
        "naive_seasonal_rmse": naive_rmse,
        "mae_improvement_vs_naive": 1.0 - model_mae / naive_mae if naive_mae else math.nan,
        "rmse_improvement_vs_naive": (1.0 - model_rmse / naive_rmse if naive_rmse else math.nan),
        "oos_r2_vs_naive": (
            1.0 - float(np.square(error).sum()) / naive_sse if naive_sse else math.nan
        ),
    }


def _bootstrap_intervals(
    common: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> pd.DataFrame:
    companies = sorted(common["company_id"].astype(str).unique())
    years = sorted(common["availability_year"].astype(int).unique())
    company_codes = pd.Categorical(common["company_id"].astype(str), categories=companies).codes
    year_codes = pd.Categorical(common["availability_year"].astype(int), categories=years).codes
    naive = common["M0_error"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model_number, model_id in enumerate(MODEL_NAMES):
        error = common[f"{model_id}_error"].to_numpy(dtype=float)
        estimates: dict[str, list[float]] = {
            "mae_improvement_vs_naive": [],
            "rmse_improvement_vs_naive": [],
            "oos_r2_vs_naive": [],
        }
        rng = np.random.default_rng(config.bootstrap_seed + model_number)
        for _ in range(config.bootstrap_repetitions):
            company_counts = np.bincount(
                rng.integers(0, len(companies), size=len(companies)), minlength=len(companies)
            )
            year_counts = np.bincount(
                rng.integers(0, len(years), size=len(years)), minlength=len(years)
            )
            weights = company_counts[company_codes] * year_counts[year_codes]
            if not np.any(weights):
                continue
            model_mae = float(np.average(np.abs(error), weights=weights))
            naive_mae = float(np.average(np.abs(naive), weights=weights))
            model_mse = float(np.average(np.square(error), weights=weights))
            naive_mse = float(np.average(np.square(naive), weights=weights))
            estimates["mae_improvement_vs_naive"].append(1.0 - model_mae / naive_mae)
            estimates["rmse_improvement_vs_naive"].append(
                1.0 - math.sqrt(model_mse) / math.sqrt(naive_mse)
            )
            estimates["oos_r2_vs_naive"].append(1.0 - model_mse / naive_mse)
        point = _metric_row(common, model_id)
        for metric, values in estimates.items():
            sample = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    "metric": metric,
                    "point_estimate": float(point[metric]),
                    "bootstrap_median": float(np.median(sample)),
                    "ci_lower_2_5": float(np.quantile(sample, 0.025)),
                    "ci_upper_97_5": float(np.quantile(sample, 0.975)),
                    "bootstrap_repetitions": len(sample),
                    "cluster_dimensions": "company_id x availability_year",
                    "bootstrap_seed": config.bootstrap_seed + model_number,
                }
            )
    return pd.DataFrame(rows)


def _calibration_table(
    common: pd.DataFrame,
    metrics: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual = common["gross_margin"].to_numpy(dtype=float)
    naive_mae = float(metrics.loc[metrics["model_id"].eq("M0"), "model_mae"].iloc[0])
    bias_limit = max(
        config.calibration_absolute_bias_limit,
        config.calibration_bias_to_naive_mae_limit * naive_mae,
    )
    for model_id in MODEL_NAMES:
        forecast = common[f"{model_id}_forecast"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(forecast)), forecast])
        intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
        error = actual - forecast
        bias = float(error.mean())
        bias_pass = abs(bias) <= bias_limit
        slope_pass = config.calibration_slope_lower <= slope <= config.calibration_slope_upper
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "forecast_events": len(common),
                "bias": bias,
                "absolute_bias_limit": bias_limit,
                "bias_pass": bias_pass,
                "calibration_intercept": float(intercept),
                "calibration_slope": float(slope),
                "slope_lower": config.calibration_slope_lower,
                "slope_upper": config.calibration_slope_upper,
                "slope_pass": slope_pass,
                "calibration_pass": bool(bias_pass and slope_pass),
            }
        )
    return pd.DataFrame(rows)


def _eligibility_gate(
    metrics: pd.DataFrame,
    by_year: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_NAMES:
        aggregate = metrics.loc[metrics["model_id"].eq(model_id)].iloc[0]
        eligible_years = by_year.loc[
            by_year["model_id"].eq(model_id)
            & by_year["forecast_events"].ge(config.min_eligible_year_events)
        ]
        positive_years = int(eligible_years["mae_improvement_vs_naive"].gt(0.0).sum())
        majority = bool(len(eligible_years) > 0 and positive_years > len(eligible_years) / 2)
        calibration_pass = bool(
            calibration.loc[calibration["model_id"].eq(model_id), "calibration_pass"].iloc[0]
        )
        uncertainty_reported = bool(
            set(bootstrap.loc[bootstrap["model_id"].eq(model_id), "metric"].astype(str))
            == {
                "mae_improvement_vs_naive",
                "rmse_improvement_vs_naive",
                "oos_r2_vs_naive",
            }
        )
        primary_improves = bool(float(aggregate["mae_improvement_vs_naive"]) > 0.0)
        forecast_gate = bool(
            model_id != "M0"
            and primary_improves
            and majority
            and calibration_pass
            and uncertainty_reported
        )
        primary_selection_candidate = model_id in {"M1", "M2", "M3"}
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "strict_pit_and_oos": True,
                "aggregate_mae_skill": float(aggregate["mae_improvement_vs_naive"]),
                "primary_mae_improves": primary_improves,
                "eligible_years": len(eligible_years),
                "positive_mae_skill_years": positive_years,
                "positive_in_majority_of_years": majority,
                "calibration_pass": calibration_pass,
                "clustered_uncertainty_reported": uncertainty_reported,
                "distinctness_used_for_selection": False,
                "forecast_gate_pass": forecast_gate,
                "primary_selection_candidate": primary_selection_candidate,
                "primary_selection_eligible": bool(forecast_gate and primary_selection_candidate),
                "historical_universe_and_sic_gate_pass": False,
                "eligible_for_downstream_returns": False,
            }
        )
    return pd.DataFrame(rows)


def _representation_diagnostics(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    delta = common["gm_delta_yoy"].astype(float)
    delta_rank = delta.rank(method="average", pct=True)
    delta_quintile = _quintiles(delta_rank)
    std_delta = float(delta.std(ddof=1))
    for model_id in MODEL_NAMES:
        innovation = common[f"{model_id}_innovation_raw"].astype(float)
        adjustment = common[f"{model_id}_expectation_adjustment"].astype(float)
        innovation_rank = innovation.rank(method="average", pct=True)
        innovation_quintile = _quintiles(innovation_rank)
        movement = innovation_quintile - delta_quintile
        top_delta = delta_quintile.eq(5)
        bottom_delta = delta_quintile.eq(1)
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "observations": len(common),
                "std_expectation_adjustment": float(adjustment.std(ddof=1)),
                "std_delta": std_delta,
                "std_adjustment_to_delta_ratio": (
                    float(adjustment.std(ddof=1) / std_delta) if std_delta else math.nan
                ),
                "pearson_adjustment_delta": _correlation(adjustment, delta),
                "spearman_adjustment_delta": _spearman(adjustment, delta),
                "spearman_delta_innovation": _spearman(delta, innovation),
                "median_abs_percentile_rank_movement": float(
                    (innovation_rank - delta_rank).abs().median()
                ),
                "top_quintile_overlap": float(innovation_quintile.loc[top_delta].eq(5).mean()),
                "bottom_quintile_overlap": float(
                    innovation_quintile.loc[bottom_delta].eq(1).mean()
                ),
                "observations_changing_quintile": float(movement.ne(0).mean()),
                "observations_moving_two_or_more_quintiles": float(movement.abs().ge(2).mean()),
            }
        )
    return pd.DataFrame(rows)


def _loss_events(common: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "event_id",
        "company_id",
        "ticker",
        "sic2",
        "availability_year",
        "availability_quarter",
        "actual_available_at",
        "gross_margin",
    ]
    rows: list[pd.DataFrame] = []
    for model_id in MODEL_NAMES:
        frame = common.loc[:, identity].copy()
        frame["model_id"] = model_id
        frame["model_name"] = MODEL_NAMES[model_id]
        frame["forecast"] = common[f"{model_id}_forecast"]
        frame["model_error"] = common[f"{model_id}_error"]
        frame["naive_error"] = common["M0_error"]
        frame["absolute_loss_difference_model_minus_naive"] = (
            frame["model_error"].abs() - frame["naive_error"].abs()
        )
        frame["squared_loss_difference_model_minus_naive"] = np.square(
            frame["model_error"]
        ) - np.square(frame["naive_error"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["model_id", "actual_available_at", "company_id"]
    )


def _subgroup_diagnostics(
    losses: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_NAMES:
        model_losses = losses.loc[losses["model_id"].eq(model_id)]
        rows.append(_subgroup_metric_row(model_losses, "all", "all"))
        for group_type, column in (
            ("availability_year", "availability_year"),
            ("provisional_sic2", "sic2"),
            ("company", "ticker"),
        ):
            for value, group in model_losses.groupby(column, observed=True, sort=True):
                if len(group) < config.min_group_events:
                    continue
                rows.append(_subgroup_metric_row(group, group_type, str(value)))
    return pd.DataFrame(rows)


def _subgroup_metric_row(
    group: pd.DataFrame,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    model_id = str(group["model_id"].iloc[0])
    error = group["model_error"].astype(float)
    naive = group["naive_error"].astype(float)
    model_mae = float(error.abs().mean())
    naive_mae = float(naive.abs().mean())
    model_rmse = float(np.sqrt(np.square(error).mean()))
    naive_rmse = float(np.sqrt(np.square(naive).mean()))
    return {
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "group_type": group_type,
        "group_value": group_value,
        "forecast_events": len(group),
        "model_bias": float(error.mean()),
        "model_mae": model_mae,
        "naive_seasonal_mae": naive_mae,
        "mae_improvement_vs_naive": 1.0 - model_mae / naive_mae if naive_mae else math.nan,
        "model_rmse": model_rmse,
        "naive_seasonal_rmse": naive_rmse,
        "rmse_improvement_vs_naive": (1.0 - model_rmse / naive_rmse if naive_rmse else math.nan),
        "oos_r2_vs_naive": (
            1.0 - float(np.square(error).sum()) / float(np.square(naive).sum())
            if float(np.square(naive).sum())
            else math.nan
        ),
    }


def _feature_clock_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_NAMES:
        forecasted = panel.loc[panel[f"{model_id}_forecast"].notna()]
        cutoff = pd.to_datetime(forecasted[f"{model_id}_feature_cutoff"], utc=True)
        actual = pd.to_datetime(forecasted["actual_available_at"], utc=True)
        if model_id == "M0":
            training_end_max: Any = pd.NaT
            training_strict = True
        else:
            training_end = pd.to_datetime(
                forecasted[f"{model_id}_training_end"], utc=True, errors="coerce"
            )
            training_end_max = training_end.max()
            training_strict = bool((training_end < actual).all())
        rows.append(
            {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "forecast_events_before_common_lock": len(forecasted),
                "features_strictly_prior": bool((cutoff < actual).all()),
                "training_strictly_prior": training_strict,
                "latest_feature_cutoff": cutoff.max(),
                "latest_training_end": training_end_max,
                "same_time_events_batched": True,
                "current_or_future_actual_used": False,
            }
        )
    return pd.DataFrame(rows)


def _tournament_report(
    panel: pd.DataFrame,
    common: pd.DataFrame,
    metrics: pd.DataFrame,
    by_year: pd.DataFrame,
    calibration: pd.DataFrame,
    gates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: ForecastTournamentConfig,
) -> dict[str, Any]:
    candidates = metrics.loc[metrics["model_id"].isin(["M1", "M2", "M3"])].sort_values(
        "mae_improvement_vs_naive", ascending=False
    )
    diagnostic_leader = str(candidates.iloc[0]["model_id"])
    passed = gates.loc[gates["primary_selection_eligible"]].merge(
        metrics[["model_id", "mae_improvement_vs_naive"]], on="model_id", how="left"
    )
    selected = (
        str(passed.sort_values("mae_improvement_vs_naive", ascending=False).iloc[0]["model_id"])
        if not passed.empty
        else None
    )
    payload = {
        "config": asdict(config),
        "models": MODEL_NAMES,
        "model_roles": {
            "M0": "benchmark",
            "M1": "primary selection candidate",
            "M2": "primary selection candidate",
            "M3": "primary selection candidate",
            "M4": "deferred nonlinear robustness only",
        },
        "features": {
            "M1": M1_FEATURES,
            "M2": M2_FEATURES,
            "M3": M3_FEATURES,
            "M4_base": M3_FEATURES,
            "M4_transform": "winsorized z-scores + squares + five frozen interactions",
        },
        "panel_events": len(panel),
        "common_sample_events": len(common),
        "common_sample_companies": int(common["company_id"].nunique()),
        "diagnostic_mae_leader": diagnostic_leader,
        "selected_model_id": selected,
        "forecast_gate_passed_models": gates.loc[gates["forecast_gate_pass"], "model_id"].tolist(),
        "primary_selection_eligible_models": gates.loc[
            gates["primary_selection_eligible"], "model_id"
        ].tolist(),
        "historical_return_gate_pass": False,
        "evidence_boundary": (
            "Financial forecast tournament only. No market prices, returns, execution timestamps, "
            "historical constituents, historical SIC membership, or delisting returns are joined."
        ),
        "selection_rule": (
            "Highest aggregate common-sample MAE skill among candidates passing PIT, "
            "aggregate MAE, "
            "majority-year stability, calibration, and reporting gates."
        ),
        "distinctness_used_for_model_selection": False,
        "metrics": metrics.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "eligibility_gate": gates.to_dict(orient="records"),
        "eligible_year_metrics": by_year.loc[
            by_year["forecast_events"].ge(config.min_eligible_year_events)
        ].to_dict(orient="records"),
        "mae_bootstrap": bootstrap.loc[bootstrap["metric"].eq("mae_improvement_vs_naive")].to_dict(
            orient="records"
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return {"forecast_tournament_version": f"gm-forecast-v2-{digest}", **payload}


def _quintiles(rank: pd.Series) -> pd.Series:
    values = np.ceil(pd.to_numeric(rank, errors="coerce") * 5.0).clip(1, 5)
    return pd.Series(values, index=rank.index, dtype="Int64")


def _correlation(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).dropna()
    if len(frame) < 2 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(np.corrcoef(frame.iloc[:, 0], frame.iloc[:, 1])[0, 1])


def _spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).dropna()
    if len(frame) < 2:
        return math.nan
    return _correlation(
        frame.iloc[:, 0].rank(method="average"),
        frame.iloc[:, 1].rank(method="average"),
    )


def _utc(values: pd.Series, name: str) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ForecastTournamentError(f"{name} contains an invalid UTC timestamp")
    return pd.Series(timestamps, index=values.index, name=name)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ForecastTournamentError(f"Missing tournament columns: {sorted(missing)}")
