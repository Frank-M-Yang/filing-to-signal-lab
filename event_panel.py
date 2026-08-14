"""Point-in-time filing-event panel and expanding forecast infrastructure.

The atomic observation is Company x Filing Event x Information Set.  This module
does not fetch data or estimate returns.  It makes the temporal contract explicit
before real fundamentals and prices are joined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
import pandas as pd

FORECAST_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "gm_yoy_lag",
    "gm_delta_yoy_lag",
    "revenue_growth_lag1",
    "industry_gm_lag1",
)

FILING_EVENT_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    # Identity and fiscal period.
    "event_id",
    "company_id",
    "ticker",
    "sector",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    # Information and execution clocks.
    "filing_accepted_at",
    "actual_available_at",
    "executable_at",
    # Point-in-time facts.
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "gross_margin",
    "gm_yoy_lag",
    # State representations.
    "gm_level",
    "gm_delta_yoy",
    # Expectation and strictly prior uncertainty estimate.
    "gm_forecast",
    "forecast_model_version",
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_training_observations",
    "forecast_generated_at",
    "forecast_sigma",
    "forecast_error_history_end",
    "forecast_sigma_observations",
    # Innovations.
    "gm_innovation_raw",
    "gm_innovation_std",
    # Provenance.
    "revenue_fact_id",
    "gross_profit_fact_id",
    "filing_accession",
    # Ex-post comparison. Never used by a PIT feature.
    "revenue_latest",
    "gross_margin_latest",
    "latest_available_at",
    "was_revised",
    "revision_size",
    # Explicit outcome windows.
    "reaction_start",
    "reaction_end",
    "tradable_start",
    "tradable_end",
    "abnormal_return_reaction",
    "abnormal_return_drift",
    "realized_vol_20",
    "max_drawdown_20",
)

BASE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "company_id",
    "ticker",
    "sector",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    "filing_accepted_at",
    "actual_available_at",
    "executable_at",
    "revenue",
    "gross_profit",
    "gross_margin",
    "gm_yoy_lag",
    "gm_level",
    "gm_delta_yoy",
    "gm_forecast",
    "forecast_model_version",
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_training_observations",
    "forecast_generated_at",
    "forecast_sigma",
    "forecast_error_history_end",
    "forecast_sigma_observations",
    "gm_innovation_raw",
    "gm_innovation_std",
    "revenue_fact_id",
    "gross_profit_fact_id",
    "filing_accession",
)

OUTCOME_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_start",
    "reaction_end",
    "tradable_start",
    "tradable_end",
)


class EventPanelValidationError(ValueError):
    """Raised when a filing-event panel violates its point-in-time contract."""


@dataclass(frozen=True)
class ForecastConfig:
    """Fixed first-pass specification for an expanding pooled OLS forecast."""

    feature_columns: tuple[str, ...] = FORECAST_FEATURE_COLUMNS
    model_version: str = "gm-pooled-ols-v1"
    min_train_observations: int = 200
    min_oos_errors: int = 100

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        if self.min_train_observations <= len(self.feature_columns):
            raise ValueError("min_train_observations must exceed the number of regressors")
        if self.min_oos_errors < 2:
            raise ValueError("min_oos_errors must be at least two")
        if not self.model_version.strip():
            raise ValueError("model_version cannot be blank")


def derive_gross_margin_representations(panel: pd.DataFrame) -> pd.DataFrame:
    """Derive Level and Delta from original point-in-time filing facts.

    ``gm_yoy_lag`` must already be resolved from a filing that was available before
    the current event.  The function intentionally does not obtain it through a
    simple row shift because missing filings and later restatements make that unsafe.
    """
    _require_columns(panel, ("revenue", "gross_profit", "gm_yoy_lag"))
    result = panel.copy()
    revenue = pd.to_numeric(result["revenue"], errors="coerce")
    gross_profit = pd.to_numeric(result["gross_profit"], errors="coerce")
    gm_yoy_lag = pd.to_numeric(result["gm_yoy_lag"], errors="coerce")
    if revenue.isna().any() or gross_profit.isna().any() or gm_yoy_lag.isna().any():
        raise EventPanelValidationError("Gross-margin inputs must be finite numbers")
    if (revenue <= 0.0).any():
        raise EventPanelValidationError("Revenue must be positive to derive gross margin")
    calculated_margin = gross_profit / revenue
    if "gross_margin" in result:
        supplied_margin = pd.to_numeric(result["gross_margin"], errors="coerce")
        if supplied_margin.isna().any() or not np.allclose(
            supplied_margin.to_numpy(dtype=float),
            calculated_margin.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise EventPanelValidationError("gross_margin does not tie to gross_profit/revenue")
    result["gross_margin"] = calculated_margin
    result["gm_level"] = calculated_margin
    result["gm_delta_yoy"] = calculated_margin - gm_yoy_lag
    return result


def generate_expanding_gm_forecasts(
    panel: pd.DataFrame,
    config: ForecastConfig | None = None,
) -> pd.DataFrame:
    """Generate out-of-sample gross-margin forecasts in strict event-time order.

    Events sharing an availability timestamp are predicted as one batch.  They
    cannot train on one another.  Standardization uses only residuals from earlier
    out-of-sample forecasts, never in-sample or future residuals.
    """
    cfg = config or ForecastConfig()
    required = (
        "actual_available_at",
        "forecast_feature_cutoff",
        "gross_margin",
        *cfg.feature_columns,
    )
    _require_columns(panel, required)
    result = panel.copy().reset_index(drop=False).rename(columns={"index": "_original_index"})
    if result["_original_index"].duplicated().any():
        raise EventPanelValidationError("The panel index must be unique")
    result["actual_available_at"] = _utc_timestamps(
        result["actual_available_at"], "actual_available_at"
    )
    result["forecast_feature_cutoff"] = _utc_timestamps(
        result["forecast_feature_cutoff"], "forecast_feature_cutoff"
    )
    if (result["forecast_feature_cutoff"] >= result["actual_available_at"]).any():
        raise EventPanelValidationError(
            "Forecast features must be available strictly before the target actual"
        )
    result["_panel_order"] = np.arange(len(result))
    result = result.sort_values(["actual_available_at", "_panel_order"]).reset_index(drop=True)

    numeric_columns = ("gross_margin", *cfg.feature_columns)
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["gm_forecast"] = np.nan
    result["forecast_model_version"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["forecast_training_end"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["forecast_training_observations"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["forecast_generated_at"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["forecast_sigma"] = np.nan
    result["forecast_error_history_end"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["forecast_sigma_observations"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    historical_oos_errors: list[tuple[pd.Timestamp, float]] = []
    complete_training_columns = ["gross_margin", *cfg.feature_columns]
    for _, event_rows in result.groupby("actual_available_at", sort=True):
        timestamp = cast(pd.Timestamp, event_rows["actual_available_at"].iloc[0])
        training = result.loc[result["actual_available_at"] < timestamp].dropna(
            subset=complete_training_columns
        )
        current_complete = event_rows[list(cfg.feature_columns)].notna().all(axis=1)
        current_index = event_rows.index[current_complete]
        if len(training) < cfg.min_train_observations or current_index.empty:
            continue

        training_x = training.loc[:, cfg.feature_columns].to_numpy(dtype=float)
        training_y = training["gross_margin"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(training_x)), training_x])
        coefficients, *_ = np.linalg.lstsq(design, training_y, rcond=None)
        current_x = result.loc[current_index, cfg.feature_columns].to_numpy(dtype=float)
        current_design = np.column_stack([np.ones(len(current_x)), current_x])
        forecasts = current_design @ coefficients

        training_end = pd.Timestamp(training["actual_available_at"].max())
        feature_cutoffs = result.loc[current_index, "forecast_feature_cutoff"]
        generated_at = feature_cutoffs.where(feature_cutoffs > training_end, training_end)
        result.loc[current_index, "gm_forecast"] = forecasts
        result.loc[current_index, "forecast_model_version"] = cfg.model_version
        result.loc[current_index, "forecast_training_end"] = training_end
        result.loc[current_index, "forecast_training_observations"] = len(training)
        result.loc[current_index, "forecast_generated_at"] = generated_at

        prior_errors = [error for _, error in historical_oos_errors]
        result.loc[current_index, "forecast_sigma_observations"] = len(prior_errors)
        if historical_oos_errors:
            error_history_end = max(error_time for error_time, _ in historical_oos_errors)
            result.loc[current_index, "forecast_error_history_end"] = error_history_end
        if len(prior_errors) >= cfg.min_oos_errors:
            sigma = float(np.std(prior_errors, ddof=1))
            if np.isfinite(sigma) and sigma > 0.0:
                result.loc[current_index, "forecast_sigma"] = sigma

        actuals = result.loc[current_index, "gross_margin"].to_numpy(dtype=float)
        historical_oos_errors.extend(
            (timestamp, float(actual - forecast))
            for actual, forecast in zip(actuals, forecasts, strict=True)
        )

    result["gm_innovation_raw"] = result["gross_margin"] - result["gm_forecast"]
    result["gm_innovation_std"] = result["gm_innovation_raw"] / result["forecast_sigma"]
    result = result.sort_values("_panel_order")
    original_index = result.pop("_original_index")
    result = result.drop(columns="_panel_order")
    result.index = original_index
    result.index.name = panel.index.name
    return result


def validate_filing_event_panel(panel: pd.DataFrame, *, require_outcomes: bool = False) -> None:
    """Validate accounting identities and all point-in-time ordering constraints."""
    _require_columns(panel, BASE_REQUIRED_COLUMNS)
    if panel.empty:
        raise EventPanelValidationError("The filing-event panel cannot be empty")
    if panel["event_id"].isna().any() or panel["event_id"].duplicated().any():
        raise EventPanelValidationError("event_id must be present and unique")

    timestamps = {
        column: _utc_timestamps(panel[column], column)
        for column in ("filing_accepted_at", "actual_available_at", "executable_at")
    }
    if (timestamps["actual_available_at"] < timestamps["filing_accepted_at"]).any():
        raise EventPanelValidationError("actual_available_at precedes filing_accepted_at")
    if (timestamps["executable_at"] <= timestamps["actual_available_at"]).any():
        raise EventPanelValidationError("executable_at must be after actual_available_at")

    derived = derive_gross_margin_representations(panel)
    for column in ("gm_level", "gm_delta_yoy"):
        supplied = pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
        expected = derived[column].to_numpy(dtype=float)
        if not np.allclose(supplied, expected, rtol=1e-10, atol=1e-12):
            raise EventPanelValidationError(f"{column} does not match its PIT inputs")

    forecast_rows = panel["gm_forecast"].notna()
    if forecast_rows.any():
        feature_cutoff = _utc_timestamps(
            panel.loc[forecast_rows, "forecast_feature_cutoff"], "forecast_feature_cutoff"
        )
        training_end = _utc_timestamps(
            panel.loc[forecast_rows, "forecast_training_end"], "forecast_training_end"
        )
        generated_at = _utc_timestamps(
            panel.loc[forecast_rows, "forecast_generated_at"], "forecast_generated_at"
        )
        actual_at = timestamps["actual_available_at"].loc[forecast_rows]
        if (feature_cutoff >= actual_at).any():
            raise EventPanelValidationError("A forecast feature is not known before the actual")
        if (training_end >= actual_at).any():
            raise EventPanelValidationError("A forecast trains on same-time or future actuals")
        if (generated_at >= actual_at).any():
            raise EventPanelValidationError("A forecast is not timestamped before the actual")
        if (training_end > generated_at).any():
            raise EventPanelValidationError("forecast_training_end follows forecast generation")
        if (feature_cutoff > generated_at).any():
            raise EventPanelValidationError("forecast_feature_cutoff follows forecast generation")
        expected_raw = (
            panel.loc[forecast_rows, "gm_level"] - panel.loc[forecast_rows, "gm_forecast"]
        )
        if not np.allclose(
            panel.loc[forecast_rows, "gm_innovation_raw"].to_numpy(dtype=float),
            expected_raw.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise EventPanelValidationError("gm_innovation_raw is inconsistent")

    standardized_rows = panel["gm_innovation_std"].notna()
    if standardized_rows.any():
        sigma = pd.to_numeric(panel.loc[standardized_rows, "forecast_sigma"], errors="coerce")
        if sigma.isna().any() or (sigma <= 0.0).any():
            raise EventPanelValidationError("Standardized innovation requires positive PIT sigma")
        sigma_end = _utc_timestamps(
            panel.loc[standardized_rows, "forecast_error_history_end"],
            "forecast_error_history_end",
        )
        actual_at = timestamps["actual_available_at"].loc[standardized_rows]
        if (sigma_end >= actual_at).any():
            raise EventPanelValidationError("forecast_sigma contains same-time or future errors")
        expected_std = panel.loc[standardized_rows, "gm_innovation_raw"] / sigma
        if not np.allclose(
            panel.loc[standardized_rows, "gm_innovation_std"].to_numpy(dtype=float),
            expected_std.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise EventPanelValidationError("gm_innovation_std is inconsistent")

    present_outcome_times = [column for column in OUTCOME_TIME_COLUMNS if column in panel]
    if require_outcomes and len(present_outcome_times) != len(OUTCOME_TIME_COLUMNS):
        raise EventPanelValidationError("All explicit outcome-window timestamps are required")
    if present_outcome_times:
        if len(present_outcome_times) != len(OUTCOME_TIME_COLUMNS):
            raise EventPanelValidationError("Outcome-window timestamps must be supplied together")
        outcome_times = {
            column: _utc_timestamps(panel[column], column) for column in OUTCOME_TIME_COLUMNS
        }
        actual_at = timestamps["actual_available_at"]
        executable_at = timestamps["executable_at"]
        if not (
            (outcome_times["reaction_start"] < actual_at)
            & (actual_at <= outcome_times["reaction_end"])
        ).all():
            raise EventPanelValidationError("Reaction window does not contain the filing event")
        if (outcome_times["tradable_start"] < executable_at).any():
            raise EventPanelValidationError("Tradable window begins before executable_at")
        if (outcome_times["tradable_start"] < outcome_times["reaction_end"]).any():
            raise EventPanelValidationError("Tradable drift overlaps the reaction window")
        if (outcome_times["tradable_end"] <= outcome_times["tradable_start"]).any():
            raise EventPanelValidationError("Tradable window has non-positive length")


def select_common_research_sample(
    panel: pd.DataFrame,
    *,
    innovation: Literal["raw", "standardized"] = "standardized",
    outcome_column: str,
) -> pd.DataFrame:
    """Return one complete-case sample shared by Level, Delta, and Innovation."""
    innovation_column = "gm_innovation_raw" if innovation == "raw" else "gm_innovation_std"
    columns = ("gm_level", "gm_delta_yoy", innovation_column, outcome_column)
    _require_columns(panel, columns)
    return panel.dropna(subset=list(columns)).copy()


def empty_filing_event_panel() -> pd.DataFrame:
    """Return the canonical column order for adapters and export jobs."""
    return pd.DataFrame(columns=FILING_EVENT_PANEL_COLUMNS)


def _require_columns(panel: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = set(columns).difference(panel.columns)
    if missing:
        raise EventPanelValidationError(f"Missing filing-event columns: {sorted(missing)}")


def _utc_timestamps(values: pd.Series, name: str) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise EventPanelValidationError(f"{name} contains an invalid UTC timestamp")
    return pd.Series(timestamps, index=values.index, name=name)
