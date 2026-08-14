"""Real-data Level, Delta, and Innovation research over cleaned SEC events.

The module stops at financial representations and forecast diagnostics. It does
not invent execution timestamps or returns when point-in-time market data are not
yet available.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from event_panel import ForecastConfig, generate_expanding_gm_forecasts

REPRESENTATION_COLUMNS: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
)

FORECAST_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "gm_forecast",
    "forecast_model_version",
    "forecast_training_end",
    "forecast_training_observations",
    "forecast_generated_at",
    "forecast_sigma",
    "forecast_error_history_end",
    "forecast_sigma_observations",
    "gm_innovation_raw",
    "gm_innovation_std",
)

REQUIRED_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "core_asset_version",
    "company_id",
    "company_name",
    "ticker",
    "sic",
    "sic_description",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    "filing_accession",
    "filing_accepted_at",
    "actual_available_at",
    "revenue",
    "gross_profit",
    "gross_margin",
    "gm_yoy_lag",
    "gm_yoy_lag_event_id",
    "gm_yoy_lag_available_at",
    "gm_delta_yoy",
    "gross_margin_latest",
    "latest_available_at",
    "was_revised",
    "revision_size",
    "revenue_fact_id",
    "gross_profit_fact_id",
    "cost_fact_id",
    "analysis_ready_level",
    "analysis_ready_delta",
)


class RepresentationResearchError(ValueError):
    """Raised when a representation panel violates its point-in-time contract."""


@dataclass(frozen=True)
class RepresentationResearchConfig:
    """Precommitted first-pass feature and expanding-forecast rules."""

    forecast: ForecastConfig = field(
        default_factory=lambda: ForecastConfig(
            model_version="gm-pooled-ols-pit-sic-snapshot-v1",
            min_train_observations=200,
            min_oos_errors=100,
        )
    )
    min_industry_peers: int = 3
    industry_period_lag_days: int = 45
    max_industry_staleness_days: int = 550
    min_prior_quarter_gap_days: int = 45
    max_prior_quarter_gap_days: int = 150
    min_company_standardized_innovations: int = 8
    min_rank_cohort_events: int = 10
    min_diagnostic_group_events: int = 5
    bootstrap_repetitions: int = 1_000
    bootstrap_seed: int = 20_260_813

    def __post_init__(self) -> None:
        if self.min_industry_peers < 2:
            raise ValueError("min_industry_peers must be at least two")
        if self.industry_period_lag_days <= 0:
            raise ValueError("industry_period_lag_days must be positive")
        if self.max_industry_staleness_days <= self.industry_period_lag_days:
            raise ValueError("Industry staleness bound must exceed the period lag")
        if self.min_prior_quarter_gap_days >= self.max_prior_quarter_gap_days:
            raise ValueError("Prior-quarter gap bounds are invalid")
        if self.min_company_standardized_innovations <= 0:
            raise ValueError("Company Innovation coverage threshold must be positive")
        if self.min_rank_cohort_events < 5:
            raise ValueError("Rank cohorts require at least five events")
        if self.min_diagnostic_group_events < 2:
            raise ValueError("Diagnostic groups require at least two events")
        if self.bootstrap_repetitions < 100:
            raise ValueError("Bootstrap diagnostics require at least 100 repetitions")


@dataclass(frozen=True)
class RepresentationResearchResult:
    """Research panel and its auditable diagnostics."""

    panel: pd.DataFrame
    common_sample: pd.DataFrame
    coverage_by_year: pd.DataFrame
    forecast_by_year: pd.DataFrame
    correlations: pd.DataFrame
    distributions: pd.DataFrame
    company_coverage: pd.DataFrame
    revision_diagnostics: pd.DataFrame
    expectation_adjustment_diagnostics: pd.DataFrame
    rank_movements: pd.DataFrame
    forecast_loss_events: pd.DataFrame
    forecast_diagnostics_by_group: pd.DataFrame
    forecast_bootstrap_intervals: pd.DataFrame
    sigma_diagnostics: pd.DataFrame
    universe_timepoint_audit: pd.DataFrame
    report: dict[str, Any]


def build_representation_research(
    events: pd.DataFrame,
    config: RepresentationResearchConfig | None = None,
) -> RepresentationResearchResult:
    """Build strict PIT Level, Delta, and expanding-OOS Innovation representations."""
    cfg = config or RepresentationResearchConfig()
    prepared = _prepare_events(events)
    level_source = prepared.loc[prepared["analysis_ready_level"]].copy()
    targets = prepared.loc[prepared["analysis_ready_delta"]].copy()
    if targets.empty:
        raise RepresentationResearchError("No strict Delta events are available")

    lookup = prepared.set_index("event_id", drop=False)
    lag_ids = targets["gm_yoy_lag_event_id"].astype("string")
    targets["gm_delta_yoy_lag"] = lag_ids.map(lookup["gm_delta_yoy"])
    targets["gm_delta_yoy_lag_available_at"] = lag_ids.map(lookup["actual_available_at"])
    targets["gm_yoy_lag_revenue"] = lag_ids.map(lookup["revenue"])

    prepared["revenue_yoy_growth"] = _safe_growth(
        prepared["revenue"],
        prepared["gm_yoy_lag_event_id"].astype("string").map(lookup["revenue"]),
    )
    prior_features = _previous_company_features(targets, prepared, cfg)
    industry_features = _asof_industry_features(targets, level_source, cfg)
    targets = targets.merge(prior_features, on="event_id", how="left", validate="one_to_one")
    targets = targets.merge(industry_features, on="event_id", how="left", validate="one_to_one")

    targets["sector"] = "SIC " + targets["sic2"]
    targets["gm_level"] = targets["gross_margin"]
    targets["naive_seasonal_forecast"] = targets["gm_yoy_lag"]
    targets["naive_seasonal_error"] = targets["gm_level"] - targets["naive_seasonal_forecast"]
    cutoff_columns = (
        "gm_yoy_lag_available_at",
        "gm_delta_yoy_lag_available_at",
        "revenue_growth_lag1_available_at",
        "industry_gm_lag1_available_at",
    )
    targets["forecast_feature_cutoff"] = targets.loc[:, list(cutoff_columns)].max(axis="columns")
    feature_complete = targets.loc[
        targets.loc[:, list(cfg.forecast.feature_columns)].notna().all(axis="columns")
        & targets["forecast_feature_cutoff"].notna()
    ].copy()
    forecasted = generate_expanding_gm_forecasts(feature_complete, cfg.forecast)
    forecast_lookup = forecasted.set_index("event_id")
    for column in FORECAST_OUTPUT_COLUMNS:
        targets[column] = targets["event_id"].map(forecast_lookup[column])

    targets["expectation_adjustment"] = targets["gm_yoy_lag"] - targets["gm_forecast"]

    targets["analysis_ready_innovation_raw"] = targets["gm_innovation_raw"].notna()
    targets["analysis_ready_innovation_std"] = targets["gm_innovation_std"].notna()
    targets["gm_level_latest"] = pd.to_numeric(targets["gross_margin_latest"], errors="coerce")
    targets["gm_yoy_lag_latest"] = lag_ids.map(lookup["gross_margin_latest"])
    targets["gm_delta_yoy_latest"] = targets["gm_level_latest"] - targets["gm_yoy_lag_latest"]
    targets["gm_innovation_raw_latest_hindsight"] = (
        targets["gm_level_latest"] - targets["gm_forecast"]
    )
    targets["innovation_revision_effect"] = (
        targets["gm_innovation_raw_latest_hindsight"] - targets["gm_innovation_raw"]
    )
    targets["gm_innovation_std_latest_hindsight"] = (
        targets["gm_innovation_raw_latest_hindsight"] / targets["forecast_sigma"]
    )
    targets["representation_research_version"] = _research_version(prepared, cfg)
    targets["industry_mapping_status"] = "provisional_2026_sic_snapshot"
    targets = targets.sort_values(["actual_available_at", "company_id", "event_id"]).reset_index(
        drop=True
    )
    validate_representation_research(targets, cfg)

    common_sample = targets.dropna(subset=list(REPRESENTATION_COLUMNS)).copy()
    coverage_by_year = _coverage_by_year(prepared, targets)
    forecast_by_year = _forecast_by_year(targets)
    correlations = _representation_correlations(common_sample)
    distributions = _representation_distributions(common_sample)
    company_coverage = _company_coverage(prepared, targets, cfg)
    revision_diagnostics = _revision_diagnostics(targets)
    rank_movements = _representation_rank_movements(common_sample, cfg)
    expectation_adjustment_diagnostics = _expectation_adjustment_diagnostics(
        common_sample,
        rank_movements,
    )
    forecast_loss_events = _forecast_loss_events(targets)
    forecast_diagnostics_by_group = _forecast_diagnostics_by_group(forecast_loss_events, cfg)
    forecast_bootstrap_intervals = _forecast_bootstrap_intervals(forecast_loss_events, cfg)
    sigma_diagnostics = _sigma_diagnostics(common_sample)
    universe_timepoint_audit = _universe_timepoint_audit(prepared)
    report = _research_report(
        prepared,
        targets,
        common_sample,
        company_coverage,
        expectation_adjustment_diagnostics,
        forecast_by_year,
        forecast_bootstrap_intervals,
        sigma_diagnostics,
        cfg,
    )
    return RepresentationResearchResult(
        panel=targets,
        common_sample=common_sample,
        coverage_by_year=coverage_by_year,
        forecast_by_year=forecast_by_year,
        correlations=correlations,
        distributions=distributions,
        company_coverage=company_coverage,
        revision_diagnostics=revision_diagnostics,
        expectation_adjustment_diagnostics=expectation_adjustment_diagnostics,
        rank_movements=rank_movements,
        forecast_loss_events=forecast_loss_events,
        forecast_diagnostics_by_group=forecast_diagnostics_by_group,
        forecast_bootstrap_intervals=forecast_bootstrap_intervals,
        sigma_diagnostics=sigma_diagnostics,
        universe_timepoint_audit=universe_timepoint_audit,
        report=report,
    )


def validate_representation_research(
    panel: pd.DataFrame,
    config: RepresentationResearchConfig | None = None,
) -> None:
    """Validate accounting identities and every representation feature clock."""
    cfg = config or RepresentationResearchConfig()
    required = (
        "event_id",
        "actual_available_at",
        "revenue",
        "gross_profit",
        "gross_margin",
        "gm_yoy_lag",
        "gm_level",
        "gm_delta_yoy",
        "forecast_feature_cutoff",
        "gm_forecast",
        "forecast_training_end",
        "forecast_generated_at",
        "forecast_sigma",
        "forecast_error_history_end",
        "gm_innovation_raw",
        "gm_innovation_std",
        "expectation_adjustment",
        "industry_peer_count",
        "analysis_ready_delta",
    )
    _require_columns(panel, required)
    if panel.empty or panel["event_id"].isna().any() or panel["event_id"].duplicated().any():
        raise RepresentationResearchError("Representation event IDs must be non-null and unique")
    if not _as_bool(panel["analysis_ready_delta"]).all():
        raise RepresentationResearchError("The representation panel contains a non-strict Delta")

    revenue = pd.to_numeric(panel["revenue"], errors="coerce")
    gross_profit = pd.to_numeric(panel["gross_profit"], errors="coerce")
    level = pd.to_numeric(panel["gm_level"], errors="coerce")
    delta = pd.to_numeric(panel["gm_delta_yoy"], errors="coerce")
    lag = pd.to_numeric(panel["gm_yoy_lag"], errors="coerce")
    if revenue.isna().any() or gross_profit.isna().any() or (revenue <= 0.0).any():
        raise RepresentationResearchError("Representation accounting inputs are invalid")
    if not np.allclose(level, gross_profit / revenue, rtol=1e-12, atol=1e-12):
        raise RepresentationResearchError("Level does not tie to PIT gross profit/revenue")
    if not np.allclose(delta, level - lag, rtol=1e-12, atol=1e-12):
        raise RepresentationResearchError("Delta does not tie to PIT Level and year lag")

    actual = _utc(panel["actual_available_at"], "actual_available_at")
    feature_rows = panel.loc[:, list(cfg.forecast.feature_columns)].notna().all(axis="columns")
    feature_cutoff = _utc(
        panel.loc[feature_rows, "forecast_feature_cutoff"], "forecast_feature_cutoff"
    )
    if feature_cutoff.isna().any() or (feature_cutoff >= actual.loc[feature_rows]).any():
        raise RepresentationResearchError("A forecast feature is not strictly point-in-time")
    peer_count = pd.to_numeric(
        panel.loc[panel["industry_gm_lag1"].notna(), "industry_peer_count"],
        errors="coerce",
    )
    if peer_count.isna().any() or (peer_count < cfg.min_industry_peers).any():
        raise RepresentationResearchError("An industry feature lacks enough prior peers")

    forecast_rows = panel["gm_forecast"].notna()
    if forecast_rows.any():
        training_end = _utc(
            panel.loc[forecast_rows, "forecast_training_end"], "forecast_training_end"
        )
        generated_at = _utc(
            panel.loc[forecast_rows, "forecast_generated_at"], "forecast_generated_at"
        )
        forecast_actual = actual.loc[forecast_rows]
        if (training_end >= forecast_actual).any() or (generated_at >= forecast_actual).any():
            raise RepresentationResearchError("A forecast uses same-time or future information")
        expected_raw = level.loc[forecast_rows] - pd.to_numeric(
            panel.loc[forecast_rows, "gm_forecast"], errors="coerce"
        )
        if not np.allclose(
            expected_raw,
            pd.to_numeric(panel.loc[forecast_rows, "gm_innovation_raw"], errors="coerce"),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RepresentationResearchError("Raw Innovation does not tie to Level minus forecast")
        expected_adjustment = lag.loc[forecast_rows] - pd.to_numeric(
            panel.loc[forecast_rows, "gm_forecast"], errors="coerce"
        )
        adjustment = pd.to_numeric(
            panel.loc[forecast_rows, "expectation_adjustment"], errors="coerce"
        )
        if not np.allclose(expected_adjustment, adjustment, rtol=1e-12, atol=1e-12):
            raise RepresentationResearchError("Expectation Adjustment does not tie")
        innovation_identity = delta.loc[forecast_rows] + adjustment
        if not np.allclose(expected_raw, innovation_identity, rtol=1e-12, atol=1e-12):
            raise RepresentationResearchError(
                "Innovation does not equal Delta plus Expectation Adjustment"
            )

    standardized = panel["gm_innovation_std"].notna()
    if standardized.any():
        sigma = pd.to_numeric(panel.loc[standardized, "forecast_sigma"], errors="coerce")
        sigma_end = _utc(
            panel.loc[standardized, "forecast_error_history_end"],
            "forecast_error_history_end",
        )
        if sigma.isna().any() or (sigma <= 0.0).any():
            raise RepresentationResearchError("Standardized Innovation has invalid sigma")
        if (sigma_end >= actual.loc[standardized]).any():
            raise RepresentationResearchError("Innovation sigma contains non-prior errors")
        expected_std = (
            pd.to_numeric(panel.loc[standardized, "gm_innovation_raw"], errors="coerce") / sigma
        )
        if not np.allclose(
            expected_std,
            pd.to_numeric(panel.loc[standardized, "gm_innovation_std"], errors="coerce"),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RepresentationResearchError("Standardized Innovation does not tie")


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    _require_columns(events, REQUIRED_EVENT_COLUMNS)
    result = events.copy()
    if result.empty or result["event_id"].duplicated().any():
        raise RepresentationResearchError("Cleaned events must be non-empty and unique")
    for column in (
        "period_end",
        "filing_accepted_at",
        "actual_available_at",
        "gm_yoy_lag_available_at",
        "latest_available_at",
    ):
        if column == "period_end":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        else:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[["period_end", "actual_available_at"]].isna().any().any():
        raise RepresentationResearchError("Cleaned events contain invalid dates")
    for column in (
        "revenue",
        "gross_profit",
        "gross_margin",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gross_margin_latest",
        "revision_size",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in (
        "analysis_ready_level",
        "analysis_ready_delta",
        "was_revised",
    ):
        result[column] = _as_bool(result[column])
    result["company_id"] = result["company_id"].astype("string").str.zfill(10)
    result["sic"] = result["sic"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
    result["sic2"] = result["sic"].str[:2]
    result["sic1"] = result["sic"].str[:1]
    return result


def _previous_company_features(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = {
        str(company_id): group.sort_values(["period_end", "actual_available_at", "event_id"])
        for company_id, group in events.groupby("company_id", sort=False)
    }
    for target in targets.itertuples(index=False):
        target_period_end = pd.Timestamp(str(target.period_end))
        target_available_at = pd.Timestamp(str(target.actual_available_at))
        company_history = sources[str(target.company_id)]
        strict_history = company_history.loc[
            (company_history["period_end"] < target_period_end)
            & (company_history["actual_available_at"] < target_available_at)
            & company_history["analysis_ready_level"]
        ].sort_values(["period_end", "actual_available_at", "event_id"])
        trailing = strict_history.tail(8)
        gap_days = (target_period_end - company_history["period_end"]).dt.days
        candidates = company_history.loc[
            gap_days.between(
                config.min_prior_quarter_gap_days,
                config.max_prior_quarter_gap_days,
                inclusive="both",
            )
            & (company_history["actual_available_at"] < target_available_at)
            & company_history["analysis_ready_level"]
        ].copy()
        base = {
            "event_id": target.event_id,
            "company_trailing_gm_mean": (
                float(trailing["gross_margin"].mean()) if not trailing.empty else math.nan
            ),
            "company_trailing_gm_volatility": (
                float(trailing["gross_margin"].std(ddof=1)) if len(trailing) >= 2 else math.nan
            ),
            "company_trailing_gm_observations": len(trailing),
            "company_trailing_gm_history_end": (
                trailing["actual_available_at"].max() if not trailing.empty else pd.NaT
            ),
        }
        if candidates.empty:
            rows.append(base)
            continue
        prior = candidates.sort_values(["period_end", "actual_available_at", "event_id"]).iloc[-1]
        rows.append(
            {
                **base,
                "revenue_growth_lag1": prior["revenue_yoy_growth"],
                "revenue_growth_lag1_event_id": prior["event_id"],
                "revenue_growth_lag1_available_at": prior["actual_available_at"],
                "revenue_growth_lag1_period_end": prior["period_end"],
            }
        )
    return pd.DataFrame(rows)


def _asof_industry_features(
    targets: pd.DataFrame,
    level_source: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_period_end = pd.Timestamp(str(target.period_end))
        target_available_at = pd.Timestamp(str(target.actual_available_at))
        period_latest = target_period_end - pd.Timedelta(days=config.industry_period_lag_days)
        period_earliest = target_period_end - pd.Timedelta(days=config.max_industry_staleness_days)
        eligible = level_source.loc[
            (level_source["company_id"] != target.company_id)
            & (level_source["actual_available_at"] < target_available_at)
            & level_source["period_end"].between(period_earliest, period_latest, inclusive="both")
        ].copy()
        selected, group_level, group_code = _industry_group(
            eligible,
            sic2=str(target.sic2),
            sic1=str(target.sic1),
            min_peers=config.min_industry_peers,
        )
        if selected.empty:
            rows.append({"event_id": target.event_id})
            continue
        latest_peer = (
            selected.sort_values(["company_id", "period_end", "actual_available_at", "event_id"])
            .groupby("company_id", as_index=False, sort=False)
            .tail(1)
        )
        if latest_peer["company_id"].nunique() < config.min_industry_peers:
            rows.append({"event_id": target.event_id})
            continue
        rows.append(
            {
                "event_id": target.event_id,
                "industry_gm_lag1": float(latest_peer["gross_margin"].median()),
                "industry_gm_lag1_available_at": latest_peer["actual_available_at"].max(),
                "industry_peer_count": int(latest_peer["company_id"].nunique()),
                "industry_group_level": group_level,
                "industry_group_code": group_code,
                "industry_information_period_end": latest_peer["period_end"].max(),
            }
        )
    return pd.DataFrame(rows)


def _industry_group(
    eligible: pd.DataFrame,
    *,
    sic2: str,
    sic1: str,
    min_peers: int,
) -> tuple[pd.DataFrame, str | None, str | None]:
    sic2_rows = eligible.loc[eligible["sic2"] == sic2]
    if sic2_rows["company_id"].nunique() >= min_peers:
        return sic2_rows, "sic2", sic2
    sic1_rows = eligible.loc[eligible["sic1"] == sic1]
    if sic1_rows["company_id"].nunique() >= min_peers:
        return sic1_rows, "sic1_fallback", sic1
    return eligible.iloc[0:0].copy(), None, None


def _coverage_by_year(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(
        pd.to_numeric(events["fiscal_year"], errors="coerce").dropna().astype(int).unique()
    )
    rows: list[dict[str, Any]] = []
    for year in years:
        source_year = pd.to_numeric(events["fiscal_year"], errors="coerce").eq(year)
        panel_year = pd.to_numeric(panel["fiscal_year"], errors="coerce").eq(year)
        rows.append(
            {
                "fiscal_year": year,
                "level_events": int((source_year & events["analysis_ready_level"]).sum()),
                "delta_events": int((source_year & events["analysis_ready_delta"]).sum()),
                "forecast_feature_complete_events": int(
                    (panel_year & panel["forecast_feature_cutoff"].notna()).sum()
                ),
                "raw_innovation_events": int(
                    (panel_year & panel["gm_innovation_raw"].notna()).sum()
                ),
                "standardized_innovation_events": int(
                    (panel_year & panel["gm_innovation_std"].notna()).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _forecast_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    forecasted = panel.loc[panel["gm_innovation_raw"].notna()].copy()
    forecasted["availability_year"] = forecasted["actual_available_at"].dt.year
    rows = [
        _forecast_metrics(group, availability_year=int(str(year)))
        for year, group in forecasted.groupby("availability_year", sort=True)
    ]
    rows.append(_forecast_metrics(forecasted, availability_year="all"))
    return pd.DataFrame(rows)


def _forecast_metrics(group: pd.DataFrame, *, availability_year: int | str) -> dict[str, Any]:
    model_error = pd.to_numeric(group["gm_innovation_raw"], errors="coerce").dropna()
    naive_error = pd.to_numeric(
        group.loc[model_error.index, "naive_seasonal_error"], errors="coerce"
    )
    model_rmse = _rmse(model_error)
    naive_rmse = _rmse(naive_error)
    model_mae = float(model_error.abs().mean())
    naive_mae = float(naive_error.abs().mean())
    model_sse = float(np.square(model_error).sum())
    naive_sse = float(np.square(naive_error).sum())
    return {
        "availability_year": availability_year,
        "forecast_events": len(model_error),
        "model_bias": float(model_error.mean()),
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "naive_seasonal_mae": naive_mae,
        "naive_seasonal_rmse": naive_rmse,
        "rmse_improvement_vs_naive": (
            1.0 - model_rmse / naive_rmse if naive_rmse > 0.0 else math.nan
        ),
        "mae_improvement_vs_naive": (1.0 - model_mae / naive_mae if naive_mae > 0.0 else math.nan),
        "oos_r2_vs_naive": 1.0 - model_sse / naive_sse if naive_sse > 0.0 else math.nan,
    }


def _representation_correlations(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in ("pearson", "spearman"):
        matrix: pd.DataFrame = common.loc[:, list(REPRESENTATION_COLUMNS)].corr(method=method)
        for left_index, left in enumerate(REPRESENTATION_COLUMNS):
            for right in REPRESENTATION_COLUMNS[left_index + 1 :]:
                rows.append(
                    {
                        "method": method,
                        "representation_1": left,
                        "representation_2": right,
                        "correlation": float(str(matrix.at[left, right])),
                        "observations": len(common),
                    }
                )
    return pd.DataFrame(rows)


def _representation_distributions(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in REPRESENTATION_COLUMNS:
        values = pd.to_numeric(common[column], errors="coerce").dropna()
        quantiles = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append(
            {
                "representation": column,
                "observations": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
            }
        )
    return pd.DataFrame(rows)


def _representation_rank_movements(
    common: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    """Measure whether Innovation materially reorders Delta on pooled and local ranks."""
    columns = [
        "event_id",
        "company_id",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "actual_available_at",
        "gm_delta_yoy",
        "gm_innovation_raw",
        "expectation_adjustment",
    ]
    result = common.loc[:, columns].copy()
    result["actual_available_at"] = pd.to_datetime(
        result["actual_available_at"], utc=True, errors="coerce"
    )
    result["availability_quarter"] = (
        result["actual_available_at"].dt.tz_localize(None).dt.to_period("Q").astype("string")
    )
    result["pooled_delta_percentile"] = _percentile_rank(result["gm_delta_yoy"])
    result["pooled_innovation_percentile"] = _percentile_rank(result["gm_innovation_raw"])
    result["pooled_rank_movement"] = (
        result["pooled_innovation_percentile"] - result["pooled_delta_percentile"]
    )
    result["pooled_abs_rank_movement"] = result["pooled_rank_movement"].abs()
    result["pooled_delta_quintile"] = _quintiles(result["pooled_delta_percentile"])
    result["pooled_innovation_quintile"] = _quintiles(result["pooled_innovation_percentile"])
    result["pooled_quintile_movement"] = (
        result["pooled_innovation_quintile"] - result["pooled_delta_quintile"]
    )
    result["pooled_changed_quintile"] = result["pooled_quintile_movement"].ne(0)

    cohort_sizes = result.groupby("availability_quarter")["event_id"].transform("size")
    result["local_rank_eligible"] = cohort_sizes.ge(config.min_rank_cohort_events)
    eligible = result["local_rank_eligible"]
    result["local_delta_percentile"] = pd.Series(np.nan, index=result.index, dtype=float)
    result["local_innovation_percentile"] = pd.Series(np.nan, index=result.index, dtype=float)
    result.loc[eligible, "local_delta_percentile"] = (
        result.loc[eligible]
        .groupby("availability_quarter")["gm_delta_yoy"]
        .rank(method="average", pct=True)
    )
    result.loc[eligible, "local_innovation_percentile"] = (
        result.loc[eligible]
        .groupby("availability_quarter")["gm_innovation_raw"]
        .rank(method="average", pct=True)
    )
    result["local_rank_movement"] = (
        result["local_innovation_percentile"] - result["local_delta_percentile"]
    )
    result["local_abs_rank_movement"] = result["local_rank_movement"].abs()
    result["local_delta_quintile"] = _quintiles(result["local_delta_percentile"])
    result["local_innovation_quintile"] = _quintiles(result["local_innovation_percentile"])
    result["local_quintile_movement"] = (
        result["local_innovation_quintile"] - result["local_delta_quintile"]
    )
    result["local_changed_quintile"] = result["local_quintile_movement"].ne(0).where(eligible)
    return result.sort_values(["actual_available_at", "company_id", "event_id"])


def _expectation_adjustment_diagnostics(
    common: pd.DataFrame,
    rank_movements: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Innovation = Delta + Expectation Adjustment and rank changes."""
    adjustment = pd.to_numeric(common["expectation_adjustment"], errors="coerce")
    delta = pd.to_numeric(common["gm_delta_yoy"], errors="coerce")
    base = {
        "std_expectation_adjustment": float(adjustment.std(ddof=1)),
        "std_delta": float(delta.std(ddof=1)),
        "std_adjustment_to_delta_ratio": float(adjustment.std(ddof=1) / delta.std(ddof=1)),
        "pearson_adjustment_delta": float(adjustment.corr(delta)),
        "spearman_adjustment_delta": _spearman(adjustment, delta),
    }
    pooled = _rank_diagnostic_row(
        rank_movements,
        prefix="pooled",
        scope="pooled_common_sample",
    )
    local = _rank_diagnostic_row(
        rank_movements.loc[rank_movements["local_rank_eligible"]],
        prefix="local",
        scope="availability_quarter_minimum_10_events",
    )
    return pd.DataFrame([{**pooled, **base}, {**local, **base}])


def _rank_diagnostic_row(
    frame: pd.DataFrame,
    *,
    prefix: str,
    scope: str,
) -> dict[str, Any]:
    delta_quintile = pd.to_numeric(frame[f"{prefix}_delta_quintile"], errors="coerce")
    innovation_quintile = pd.to_numeric(frame[f"{prefix}_innovation_quintile"], errors="coerce")
    valid = delta_quintile.notna() & innovation_quintile.notna()
    eligible = frame.loc[valid]
    delta_quintile = delta_quintile.loc[valid]
    innovation_quintile = innovation_quintile.loc[valid]

    def overlap(quintile: int) -> tuple[float, float]:
        delta_set = delta_quintile.eq(quintile)
        innovation_set = innovation_quintile.eq(quintile)
        intersection = int((delta_set & innovation_set).sum())
        denominator = int(delta_set.sum())
        union = int((delta_set | innovation_set).sum())
        return (
            intersection / denominator if denominator else math.nan,
            intersection / union if union else math.nan,
        )

    bottom_overlap, bottom_jaccard = overlap(1)
    top_overlap, top_jaccard = overlap(5)
    quintile_movement = pd.to_numeric(
        eligible[f"{prefix}_quintile_movement"], errors="coerce"
    ).abs()
    changed_share = float(quintile_movement.gt(0).mean()) if len(eligible) else math.nan
    large_move_share = float(quintile_movement.ge(2).mean()) if len(eligible) else math.nan
    return {
        "rank_scope": scope,
        "observations": len(eligible),
        "cohorts": (int(eligible["availability_quarter"].nunique()) if prefix == "local" else 1),
        "median_abs_percentile_rank_movement": float(
            pd.to_numeric(eligible[f"{prefix}_abs_rank_movement"], errors="coerce").median()
        ),
        "top_quintile_overlap": top_overlap,
        "bottom_quintile_overlap": bottom_overlap,
        "top_quintile_jaccard": top_jaccard,
        "bottom_quintile_jaccard": bottom_jaccard,
        "observations_changing_quintile": changed_share,
        "observations_moving_two_or_more_quintiles": large_move_share,
    }


def _forecast_loss_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Preserve paired model-versus-naive loss differences for stability audits."""
    forecasted = panel.loc[panel["gm_innovation_raw"].notna()].copy()
    available = pd.to_datetime(forecasted["actual_available_at"], utc=True, errors="coerce")
    period_end = pd.to_datetime(forecasted["period_end"], errors="coerce")
    forecasted["availability_year"] = available.dt.year
    forecasted["availability_quarter"] = (
        available.dt.tz_localize(None).dt.to_period("Q").astype("string")
    )
    forecasted["filing_lag_days"] = (
        available.dt.tz_localize(None).dt.normalize() - period_end.dt.normalize()
    ).dt.days
    forecasted["filing_lag_bucket"] = pd.cut(
        forecasted["filing_lag_days"],
        bins=[-np.inf, 35, 45, 60, np.inf],
        labels=["<=35 days", "36-45 days", "46-60 days", ">60 days"],
    ).astype("string")
    forecasted["gm_volatility_bucket"] = _quantile_bucket(
        forecasted["company_trailing_gm_volatility"],
        labels=("Q1 low", "Q2", "Q3", "Q4 high"),
    )
    forecasted["revenue_size_bucket"] = _quantile_bucket(
        forecasted["revenue"],
        labels=("Q1 small", "Q2", "Q3", "Q4 large"),
    )
    forecasted["model_error"] = forecasted["gm_innovation_raw"]
    forecasted["naive_error"] = forecasted["naive_seasonal_error"]
    forecasted["model_absolute_loss"] = forecasted["model_error"].abs()
    forecasted["naive_absolute_loss"] = forecasted["naive_error"].abs()
    forecasted["absolute_loss_difference_model_minus_naive"] = (
        forecasted["model_absolute_loss"] - forecasted["naive_absolute_loss"]
    )
    forecasted["model_squared_loss"] = np.square(forecasted["model_error"])
    forecasted["naive_squared_loss"] = np.square(forecasted["naive_error"])
    forecasted["squared_loss_difference_model_minus_naive"] = (
        forecasted["model_squared_loss"] - forecasted["naive_squared_loss"]
    )
    columns = [
        "event_id",
        "company_id",
        "ticker",
        "sic2",
        "actual_available_at",
        "availability_year",
        "availability_quarter",
        "filing_lag_days",
        "filing_lag_bucket",
        "company_trailing_gm_volatility",
        "gm_volatility_bucket",
        "revenue",
        "revenue_size_bucket",
        "model_error",
        "naive_error",
        "model_absolute_loss",
        "naive_absolute_loss",
        "absolute_loss_difference_model_minus_naive",
        "model_squared_loss",
        "naive_squared_loss",
        "squared_loss_difference_model_minus_naive",
    ]
    return forecasted.loc[:, columns].sort_values(["actual_available_at", "company_id"])


def _forecast_diagnostics_by_group(
    losses: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    """Report paired forecast skill by preregistered descriptive subgroups."""
    group_specs = (
        ("availability_year", "availability_year"),
        ("provisional_sic2", "sic2"),
        ("company", "ticker"),
        ("prior_gm_volatility", "gm_volatility_bucket"),
        ("revenue_size_proxy", "revenue_size_bucket"),
        ("filing_lag", "filing_lag_bucket"),
    )
    rows = [_loss_metric_row(losses, group_type="all", group_value="all")]
    for group_type, column in group_specs:
        for value, group in losses.dropna(subset=[column]).groupby(
            column, observed=True, sort=True
        ):
            if len(group) < config.min_diagnostic_group_events:
                continue
            rows.append(_loss_metric_row(group, group_type=group_type, group_value=str(value)))
    return pd.DataFrame(rows)


def _loss_metric_row(
    group: pd.DataFrame,
    *,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    model = pd.to_numeric(group["model_error"], errors="coerce").dropna()
    naive = pd.to_numeric(group.loc[model.index, "naive_error"], errors="coerce")
    model_mae = float(model.abs().mean())
    naive_mae = float(naive.abs().mean())
    model_rmse = _rmse(model)
    naive_rmse = _rmse(naive)
    naive_sse = float(np.square(naive).sum())
    return {
        "group_type": group_type,
        "group_value": group_value,
        "forecast_events": len(model),
        "model_bias": float(model.mean()),
        "model_mae": model_mae,
        "naive_seasonal_mae": naive_mae,
        "mae_improvement_vs_naive": 1.0 - model_mae / naive_mae if naive_mae else math.nan,
        "model_rmse": model_rmse,
        "naive_seasonal_rmse": naive_rmse,
        "rmse_improvement_vs_naive": (1.0 - model_rmse / naive_rmse if naive_rmse else math.nan),
        "oos_r2_vs_naive": (
            1.0 - float(np.square(model).sum()) / naive_sse if naive_sse else math.nan
        ),
        "mean_absolute_loss_difference_model_minus_naive": float(
            model.abs().mean() - naive.abs().mean()
        ),
        "mean_squared_loss_difference_model_minus_naive": float(
            np.square(model).mean() - np.square(naive).mean()
        ),
    }


def _forecast_bootstrap_intervals(
    losses: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    """Two-way company/year pigeonhole bootstrap for paired forecast-skill estimates."""
    frame = losses.dropna(subset=["model_error", "naive_error", "company_id", "availability_year"])
    companies = sorted(frame["company_id"].astype(str).unique())
    years = sorted(pd.to_numeric(frame["availability_year"], errors="coerce").dropna().astype(int))
    years = sorted(set(years))
    rng = np.random.default_rng(config.bootstrap_seed)
    company_codes = pd.Categorical(frame["company_id"].astype(str), categories=companies).codes
    year_codes = pd.Categorical(
        pd.to_numeric(frame["availability_year"], errors="coerce").astype(int), categories=years
    ).codes
    model = frame["model_error"].to_numpy(dtype=float)
    naive = frame["naive_error"].to_numpy(dtype=float)
    estimates: dict[str, list[float]] = {
        "mae_improvement_vs_naive": [],
        "rmse_improvement_vs_naive": [],
        "oos_r2_vs_naive": [],
    }
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
        estimates["mae_improvement_vs_naive"].append(
            1.0
            - np.average(np.abs(model), weights=weights)
            / np.average(np.abs(naive), weights=weights)
        )
        model_mse = float(np.average(np.square(model), weights=weights))
        naive_mse = float(np.average(np.square(naive), weights=weights))
        estimates["rmse_improvement_vs_naive"].append(
            1.0 - math.sqrt(model_mse) / math.sqrt(naive_mse)
        )
        estimates["oos_r2_vs_naive"].append(1.0 - model_mse / naive_mse)

    point = _loss_metric_row(frame, group_type="all", group_value="all")
    rows: list[dict[str, Any]] = []
    for metric, values in estimates.items():
        sample = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "bootstrap_median": float(np.median(sample)),
                "ci_lower_2_5": float(np.quantile(sample, 0.025)),
                "ci_upper_97_5": float(np.quantile(sample, 0.975)),
                "bootstrap_repetitions": len(sample),
                "cluster_dimensions": "company_id x availability_year",
                "bootstrap_seed": config.bootstrap_seed,
            }
        )
    return pd.DataFrame(rows)


def _sigma_diagnostics(common: pd.DataFrame) -> pd.DataFrame:
    """Explain why global expanding sigma barely changes cross-sectional ranks."""
    sigma = pd.to_numeric(common["forecast_sigma"], errors="coerce").dropna()
    available = pd.to_datetime(common.loc[sigma.index, "actual_available_at"], utc=True)
    same_time_sizes = available.groupby(available).size()
    raw = pd.to_numeric(common.loc[sigma.index, "gm_innovation_raw"], errors="coerce")
    standardized = pd.to_numeric(common.loc[sigma.index, "gm_innovation_std"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "sigma_scope": "expanding_global_prior_oos_errors",
                "observations": len(sigma),
                "distinct_sigma_values": int(sigma.nunique()),
                "sigma_mean": float(sigma.mean()),
                "sigma_std": float(sigma.std(ddof=1)),
                "sigma_cv": float(sigma.std(ddof=1) / sigma.mean()),
                "sigma_min": float(sigma.min()),
                "sigma_max": float(sigma.max()),
                "spearman_raw_standardized": _spearman(raw, standardized),
                "largest_same_timestamp_batch": int(same_time_sizes.max()),
                "company_conditioned": False,
                "industry_conditioned": False,
                "event_conditioned": False,
                "interpretation": (
                    "The denominator is a global expanding error scale, not conditional "
                    "company or event uncertainty; it changes magnitude more than ordering."
                ),
            }
        ]
    )


def _universe_timepoint_audit(events: pd.DataFrame) -> pd.DataFrame:
    """Make the current-universe and current-SIC blockers explicit before returns."""
    identity = (
        events.sort_values(["company_id", "actual_available_at"])
        .groupby("company_id", as_index=False)
        .last()
    )
    columns = ["company_id", "ticker", "company_name", "sic", "sic_description"]
    optional = [column for column in ("fortune_rank", "country") if column in identity]
    result = identity.loc[:, [*columns, *optional]].copy()
    result["universe_selection_basis"] = "Fortune Global 500 2026 current snapshot"
    result["selection_known_at_historical_event"] = False
    result["sic_membership_basis"] = "provisional 2026 SEC SIC snapshot"
    result["historical_sic_series_available"] = False
    result["historical_security_master_available"] = False
    result["eligible_for_final_return_study"] = False
    result["return_study_blocker"] = (
        "Current-survivor universe and current SIC can support representation diagnostics "
        "but not historical alpha claims."
    )
    return result.sort_values(["company_id"])


def _company_coverage(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> pd.DataFrame:
    identity = (
        events.sort_values(["company_id", "actual_available_at"])
        .groupby("company_id", as_index=False)
        .last()[["company_id", "ticker", "company_name", "sic", "sic_description"]]
    )
    rows: list[dict[str, Any]] = []
    for record in identity.itertuples(index=False):
        company_events = events.loc[events["company_id"] == record.company_id]
        company_panel = panel.loc[panel["company_id"] == record.company_id]
        std_count = int(company_panel["gm_innovation_std"].notna().sum())
        rows.append(
            {
                "company_id": record.company_id,
                "ticker": record.ticker,
                "company_name": record.company_name,
                "sic": record.sic,
                "sic_description": record.sic_description,
                "level_events": int(company_events["analysis_ready_level"].sum()),
                "delta_events": int(company_events["analysis_ready_delta"].sum()),
                "forecast_feature_complete_events": int(
                    company_panel.loc[:, list(config.forecast.feature_columns)]
                    .notna()
                    .all(axis="columns")
                    .sum()
                ),
                "raw_innovation_events": int(company_panel["gm_innovation_raw"].notna().sum()),
                "standardized_innovation_events": std_count,
                "revised_target_events": int(company_panel["was_revised"].sum()),
                "innovation_coverage_status": (
                    "innovation_ready"
                    if std_count >= config.min_company_standardized_innovations
                    else "limited_innovation_history"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["innovation_coverage_status", "ticker"])


def _revision_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    revised = panel.loc[panel["was_revised"]].copy()
    revised["abs_level_revision"] = (revised["gm_level_latest"] - revised["gm_level"]).abs()
    columns = [
        "event_id",
        "company_id",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "period_end",
        "actual_available_at",
        "gm_level",
        "gm_level_latest",
        "gm_delta_yoy",
        "gm_delta_yoy_latest",
        "gm_innovation_raw",
        "gm_innovation_raw_latest_hindsight",
        "innovation_revision_effect",
        "abs_level_revision",
        "filing_accession",
        "latest_available_at",
    ]
    return revised.loc[:, columns].sort_values("abs_level_revision", ascending=False)


def _research_report(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    common: pd.DataFrame,
    company_coverage: pd.DataFrame,
    expectation_adjustment_diagnostics: pd.DataFrame,
    forecast_by_year: pd.DataFrame,
    forecast_bootstrap_intervals: pd.DataFrame,
    sigma_diagnostics: pd.DataFrame,
    config: RepresentationResearchConfig,
) -> dict[str, Any]:
    forecasted = panel.loc[panel["gm_innovation_raw"].notna()]
    all_metrics = _forecast_metrics(forecasted, availability_year="all")
    common_correlations = (
        panel.dropna(subset=list(REPRESENTATION_COLUMNS))
        .loc[:, list(REPRESENTATION_COLUMNS)]
        .corr(method="spearman")
    )
    delta_raw_correlation = float(str(common_correlations.at["gm_delta_yoy", "gm_innovation_raw"]))
    delta_std_correlation = float(str(common_correlations.at["gm_delta_yoy", "gm_innovation_std"]))
    pooled_adjustment = expectation_adjustment_diagnostics.loc[
        expectation_adjustment_diagnostics["rank_scope"].eq("pooled_common_sample")
    ].iloc[0]
    bootstrap = forecast_bootstrap_intervals.set_index("metric")
    yearly = forecast_by_year.loc[
        forecast_by_year["availability_year"].astype(str).ne("all")
        & pd.to_numeric(forecast_by_year["forecast_events"], errors="coerce").ge(25)
    ]
    stable_primary_years = int(
        pd.to_numeric(yearly["mae_improvement_vs_naive"], errors="coerce").gt(0).sum()
    )
    sigma = sigma_diagnostics.iloc[0]
    return {
        "representation_research_version": str(panel["representation_research_version"].iloc[0]),
        "core_asset_version": str(events["core_asset_version"].iloc[0]),
        "core_companies": int(events["company_id"].nunique()),
        "config": asdict(config),
        "level_events": int(events["analysis_ready_level"].sum()),
        "delta_events": len(panel),
        "forecast_feature_complete_events": int(
            panel.loc[:, list(config.forecast.feature_columns)].notna().all(axis="columns").sum()
        ),
        "raw_innovation_events": int(panel["gm_innovation_raw"].notna().sum()),
        "standardized_innovation_events": int(panel["gm_innovation_std"].notna().sum()),
        "common_representation_events": len(common),
        "common_representation_companies": int(common["company_id"].nunique()),
        "companies_with_minimum_innovation_history": int(
            company_coverage["innovation_coverage_status"].eq("innovation_ready").sum()
        ),
        "forecast_diagnostics": all_metrics,
        "representation_distinctness": {
            "spearman_delta_raw_innovation": delta_raw_correlation,
            "spearman_delta_standardized_innovation": delta_std_correlation,
            "status": (
                "innovation_not_yet_distinct_from_delta"
                if abs(delta_raw_correlation) >= 0.80
                else "innovation_structurally_distinct_from_delta"
            ),
            "expectation_adjustment_identity": ("Innovation = Delta + (GM_t-4 - Forecast_t-)"),
            "std_expectation_adjustment": float(pooled_adjustment["std_expectation_adjustment"]),
            "std_delta": float(pooled_adjustment["std_delta"]),
            "std_adjustment_to_delta_ratio": float(
                pooled_adjustment["std_adjustment_to_delta_ratio"]
            ),
            "spearman_adjustment_delta": float(pooled_adjustment["spearman_adjustment_delta"]),
            "median_abs_percentile_rank_movement": float(
                pooled_adjustment["median_abs_percentile_rank_movement"]
            ),
            "top_quintile_overlap": float(pooled_adjustment["top_quintile_overlap"]),
            "bottom_quintile_overlap": float(pooled_adjustment["bottom_quintile_overlap"]),
            "observations_changing_quintile": float(
                pooled_adjustment["observations_changing_quintile"]
            ),
            "observations_moving_two_or_more_quintiles": float(
                pooled_adjustment["observations_moving_two_or_more_quintiles"]
            ),
        },
        "forecast_stability": {
            "primary_metric": "MAE improvement versus seasonal naive",
            "secondary_metrics": ["RMSE improvement versus seasonal naive", "OOS R2"],
            "eligible_years_minimum_25_events": len(yearly),
            "years_with_positive_primary_skill": stable_primary_years,
            "aggregate_oos_r2_vs_naive": float(all_metrics["oos_r2_vs_naive"]),
            "bootstrap_95pct_intervals": {
                metric: {
                    "point_estimate": float(str(bootstrap.at[metric, "point_estimate"])),
                    "lower": float(str(bootstrap.at[metric, "ci_lower_2_5"])),
                    "upper": float(str(bootstrap.at[metric, "ci_upper_97_5"])),
                }
                for metric in bootstrap.index
            },
        },
        "sigma_diagnostic": {
            "scope": str(sigma["sigma_scope"]),
            "spearman_raw_standardized": float(sigma["spearman_raw_standardized"]),
            "sigma_cv": float(sigma["sigma_cv"]),
            "conditional_uncertainty": False,
            "interpretation": str(sigma["interpretation"]),
        },
        "v2_eligibility_gate": {
            "point_in_time_and_oos": True,
            "primary_mae_improves_vs_naive": bool(
                float(all_metrics["mae_improvement_vs_naive"]) > 0.0
            ),
            "primary_skill_positive_in_majority_of_eligible_years": bool(
                len(yearly) > 0 and stable_primary_years > len(yearly) / 2
            ),
            "distinctness_used_for_model_selection": False,
            "eligible_for_downstream_return_testing": False,
            "status": "v1_fails_forecast_accuracy_and_stability_gate",
        },
        "revised_target_events": int(panel["was_revised"].sum()),
        "industry_mapping_status": "provisional_2026_sic_snapshot",
        "universe_selection_status": "current_2026_survivor_selected_snapshot",
        "evidence_boundary": (
            "Financial representation and forecast diagnostics only. No market prices, "
            "execution timestamps, abnormal returns, delisting returns, or trading-cost claims."
        ),
        "invariants": [
            "Level and Delta use only original point-in-time filing values",
            "Every forecast feature is available strictly before the target filing",
            "Same-time filings are predicted as one non-leaking batch",
            "Forecast sigma uses only strictly earlier out-of-sample errors",
            "Latest values are retained only as explicitly labelled hindsight diagnostics",
            "Industry facts are PIT but industry membership uses the provisional 2026 SIC snapshot",
            "Representation distinctness is reported as an outcome, never optimized",
            "Current 2026 universe selection and SIC membership block historical return claims",
        ],
    }


def _research_version(events: pd.DataFrame, config: RepresentationResearchConfig) -> str:
    payload = {
        "core_asset_version": str(events["core_asset_version"].iloc[0]),
        "event_ids_sha256": hashlib.sha256(
            "\n".join(sorted(events["event_id"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"ldi-pit-v1-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _safe_growth(current: pd.Series, prior: pd.Series) -> pd.Series:
    current_numeric = pd.to_numeric(current, errors="coerce")
    prior_numeric = pd.to_numeric(prior, errors="coerce")
    safe_prior = prior_numeric.where(prior_numeric.abs() > 1e-12)
    result = current_numeric / safe_prior - 1.0
    return result.replace([np.inf, -np.inf], np.nan)


def _percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True)


def _quintiles(percentile_ranks: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(percentile_ranks, errors="coerce")
    quintiles = np.ceil(numeric * 5.0).clip(1, 5)
    return pd.Series(quintiles, index=percentile_ranks.index, dtype="Float64")


def _quantile_bucket(values: pd.Series, *, labels: tuple[str, str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(method="average", pct=True)
    bucket = pd.Series(np.ceil(ranks * 4.0).clip(1, 4), index=values.index)
    mapping = {index + 1: label for index, label in enumerate(labels)}
    return bucket.map(mapping).astype("string")


def _spearman(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).dropna()
    if len(paired) < 2:
        return math.nan
    return float(
        paired.iloc[:, 0].rank(method="average").corr(paired.iloc[:, 1].rank(method="average"))
    )


def _rmse(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(numeric)))) if len(numeric) else math.nan


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().eq("true")


def _utc(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise RepresentationResearchError(f"{name} contains an invalid UTC timestamp")
    return pd.Series(result, index=values.index, name=name)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise RepresentationResearchError(f"Missing representation inputs: {sorted(missing)}")
