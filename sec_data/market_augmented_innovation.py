"""Outcome-independent market-augmented gross-margin Innovation tournament."""

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
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.validation_market_state import (
    MARKET_STATE_FEATURES,
    validate_validation_market_state,
)

MARKET_AUGMENTED_INNOVATION_VERSION: Final[str] = "market-augmented-innovation-v1"
MARKET_AUGMENTED_RETURN_VERSION: Final[str] = "market-augmented-return-diagnostic-v1"
PRIOR_ACCOUNTING_FEATURES: Final[tuple[str, ...]] = ("gm_forecast", "gm_yoy_lag")


class MarketAugmentedInnovationError(FilingCatalogError):
    """Raised when the augmented forecast or return diagnostic violates its contract."""


@dataclass(frozen=True)
class ExpectationModelSpec:
    """One frozen residual-correction model."""

    model_id: str
    algorithm: Literal["baseline", "ridge"]
    feature_set: Literal["financial_baseline", "market", "combined"]
    features: tuple[str, ...]
    complexity_rank: int


MODEL_SPECS: Final[tuple[ExpectationModelSpec, ...]] = (
    ExpectationModelSpec("E0_financial_baseline", "baseline", "financial_baseline", (), 0),
    ExpectationModelSpec("E1_market_correction", "ridge", "market", MARKET_STATE_FEATURES, 1),
    ExpectationModelSpec(
        "E2_combined_correction",
        "ridge",
        "combined",
        (*PRIOR_ACCOUNTING_FEATURES, *MARKET_STATE_FEATURES),
        2,
    ),
)


@dataclass(frozen=True)
class MarketAugmentedInnovationConfig:
    """Frozen training, scaling, selection, and inference settings."""

    minimum_training_events: int = 40
    minimum_training_issuers: int = 10
    ridge_alpha: float = 10.0
    minimum_prior_oos_errors: int = 20
    minimum_relative_mae_improvement: float = 0.01
    bootstrap_repetitions: int = 2_000
    bootstrap_seed: int = 20_260_828
    version: str = MARKET_AUGMENTED_INNOVATION_VERSION

    def validate(self) -> None:
        """Reject weak point-in-time or deterministic-model settings."""
        if self.minimum_training_events < 20:
            raise ValueError("At least 20 prior events are required")
        if self.minimum_training_issuers < 3:
            raise ValueError("At least three prior issuers are required")
        if self.ridge_alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive")
        if self.minimum_prior_oos_errors < 10:
            raise ValueError("At least ten prior OOS errors are required")
        if not 0.0 <= self.minimum_relative_mae_improvement <= 0.10:
            raise ValueError("The financial-forecast improvement gate is invalid")
        if self.bootstrap_repetitions < 500:
            raise ValueError("Issuer bootstrap requires at least 500 repetitions")


@dataclass(frozen=True)
class MarketAugmentedInnovationTables:
    """Pure forecast-tournament outputs before persistence."""

    predictions: pd.DataFrame
    common_sample: pd.DataFrame
    forecast_metrics: pd.DataFrame
    selection_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class MarketAugmentedReturnTables:
    """Exploratory return diagnostics evaluated after forecast selection."""

    analysis_sample: pd.DataFrame
    representation_ic: pd.DataFrame
    incremental_tests: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class MarketAugmentedResult:
    """Immutable market-augmented model or return bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_market_augmented_innovation(
    market_state: pd.DataFrame,
    config: MarketAugmentedInnovationConfig | None = None,
) -> MarketAugmentedInnovationTables:
    """Fit event-by-event residual corrections without loading market returns."""
    cfg = config or MarketAugmentedInnovationConfig()
    cfg.validate()
    validate_validation_market_state(market_state)
    required = {
        "validation_event_id",
        "cik",
        "actual_available_at",
        "market_feature_available_at",
        "market_state_ready",
        "gm_level",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gm_forecast",
        "gm_innovation_raw",
        "gm_innovation_std",
        "forecast_feature_cutoff",
        *MARKET_STATE_FEATURES,
    }
    _require_columns(market_state, required, "market-state feature store")
    data = market_state.loc[market_state["market_state_ready"].astype(bool)].copy()
    data["actual_available_at"] = _utc(data["actual_available_at"], "actual_available_at")
    data["forecast_feature_cutoff"] = _utc(
        data["forecast_feature_cutoff"], "forecast_feature_cutoff"
    )
    data["market_feature_available_at"] = _utc(
        data["market_feature_available_at"], "market_feature_available_at"
    )
    data["model_feature_cutoff"] = data[
        ["forecast_feature_cutoff", "market_feature_available_at"]
    ].max(axis="columns")
    if not (data["model_feature_cutoff"] < data["actual_available_at"]).all():
        raise MarketAugmentedInnovationError("An augmented forecast uses current filing data")
    numeric = [
        "gm_level",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gm_forecast",
        "gm_innovation_raw",
        "gm_innovation_std",
        *MARKET_STATE_FEATURES,
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(
        subset=["gm_level", "gm_yoy_lag", "gm_delta_yoy", "gm_forecast", *MARKET_STATE_FEATURES]
    ).sort_values(["actual_available_at", "cik", "validation_event_id"], kind="mergesort")
    if data.empty:
        raise MarketAugmentedInnovationError("No complete market-augmented forecast events exist")
    data["baseline_error"] = data["gm_level"] - data["gm_forecast"]
    predictions = data[
        [
            "validation_event_id",
            "cik",
            "actual_available_at",
            "model_feature_cutoff",
            "gm_level",
            "gm_delta_yoy",
            "gm_innovation_raw",
            "gm_innovation_std",
        ]
    ].copy()
    for spec in MODEL_SPECS:
        if spec.algorithm == "baseline":
            predictions[f"{spec.model_id}_forecast"] = data["gm_forecast"].to_numpy()
            predictions[f"{spec.model_id}_correction"] = 0.0
            predictions[f"{spec.model_id}_training_end"] = pd.NaT
            predictions[f"{spec.model_id}_training_events"] = pd.Series(
                pd.NA, index=predictions.index, dtype="Int64"
            )
            predictions[f"{spec.model_id}_training_issuers"] = pd.Series(
                pd.NA, index=predictions.index, dtype="Int64"
            )
            continue
        candidate = _expanding_correction_predictions(data, spec, cfg)
        for column in (
            "forecast",
            "correction",
            "training_end",
            "training_events",
            "training_issuers",
        ):
            predictions[f"{spec.model_id}_{column}"] = candidate[column].to_numpy()
    for spec in MODEL_SPECS:
        model_id = spec.model_id
        predictions[f"{model_id}_error"] = (
            predictions["gm_level"] - predictions[f"{model_id}_forecast"]
        )
        sigma, observations, history_end = _strict_prior_sigmas(
            predictions,
            error_column=f"{model_id}_error",
            minimum_observations=cfg.minimum_prior_oos_errors,
        )
        predictions[f"{model_id}_forecast_sigma"] = sigma
        predictions[f"{model_id}_sigma_observations"] = observations
        predictions[f"{model_id}_error_history_end"] = history_end
        predictions[f"{model_id}_innovation_raw"] = predictions[f"{model_id}_error"]
        predictions[f"{model_id}_innovation_std"] = (
            predictions[f"{model_id}_error"] / predictions[f"{model_id}_forecast_sigma"]
        )
    validate_market_augmented_predictions(predictions, cfg)
    forecast_columns = [f"{spec.model_id}_forecast" for spec in MODEL_SPECS]
    common = predictions.dropna(subset=forecast_columns).copy()
    if common.empty:
        raise MarketAugmentedInnovationError("No common E0-E2 financial-forecast sample exists")
    metrics = _forecast_metrics(common)
    checks, selected = _select_financial_forecast(metrics, cfg)
    selected_spec = next(spec for spec in MODEL_SPECS if spec.model_id == selected)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "market_ready_events": len(data),
        "market_ready_issuers": int(data["cik"].nunique()),
        "common_forecast_events": len(common),
        "common_forecast_issuers": int(common["cik"].nunique()),
        "selected_model_id": selected,
        "selected_feature_set": selected_spec.feature_set,
        "financial_forecast_improvement_gate": bool(
            checks.loc[checks["check_id"].eq("relative_mae_improvement"), "passed"].iloc[0]
        ),
        "selection_uses_return_values": False,
        "return_values_loaded": False,
        "return_metrics_computed": False,
        "strategy_pnl_computed": False,
        "validation_status": "exploratory_outcome_independent_forecast_selection",
        "config": asdict(cfg),
        "models": [asdict(spec) for spec in MODEL_SPECS],
    }
    return MarketAugmentedInnovationTables(predictions, common, metrics, checks, report)


def validate_market_augmented_predictions(
    predictions: pd.DataFrame,
    config: MarketAugmentedInnovationConfig | None = None,
) -> None:
    """Validate forecast clocks, training cutoffs, and strictly prior sigma histories."""
    cfg = config or MarketAugmentedInnovationConfig()
    cfg.validate()
    if predictions.empty or predictions["validation_event_id"].duplicated().any():
        raise MarketAugmentedInnovationError("Augmented predictions are empty or duplicated")
    actual_at = _utc(predictions["actual_available_at"], "actual_available_at")
    cutoff = _utc(predictions["model_feature_cutoff"], "model_feature_cutoff")
    if not (cutoff < actual_at).all():
        raise MarketAugmentedInnovationError("Forecast generation is not strictly pre-actual")
    for spec in MODEL_SPECS:
        model_id = spec.model_id
        rows = predictions[f"{model_id}_forecast"].notna()
        if not rows.any():
            raise MarketAugmentedInnovationError(f"{model_id} produced no forecasts")
        if spec.algorithm != "baseline":
            training_end = _utc(
                predictions.loc[rows, f"{model_id}_training_end"], f"{model_id} training end"
            )
            if not (training_end < cutoff.loc[rows]).all():
                raise MarketAugmentedInnovationError(f"{model_id} uses a same-time training actual")
            if (
                not predictions.loc[rows, f"{model_id}_training_events"]
                .ge(cfg.minimum_training_events)
                .all()
            ):
                raise MarketAugmentedInnovationError(f"{model_id} violates its training gate")
        standardized = predictions[f"{model_id}_innovation_std"].notna()
        history_end = _utc(
            predictions.loc[standardized, f"{model_id}_error_history_end"],
            f"{model_id} error history end",
        )
        if not (history_end < cutoff.loc[standardized]).all():
            raise MarketAugmentedInnovationError(f"{model_id} sigma uses a current error")


def calculate_market_augmented_return_diagnostics(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    selected_model_id: str,
    config: MarketAugmentedInnovationConfig | None = None,
) -> MarketAugmentedReturnTables:
    """Evaluate the financially selected Innovation on the outcome-observed sample."""
    cfg = config or MarketAugmentedInnovationConfig()
    cfg.validate()
    validate_market_augmented_predictions(predictions, cfg)
    if selected_model_id not in {spec.model_id for spec in MODEL_SPECS}:
        raise MarketAugmentedInnovationError("Selected model is not a frozen candidate")
    _require_columns(
        returns,
        {
            "validation_event_id",
            "cik",
            "market_excess_return_arithmetic",
            "market_excess_return_ready",
        },
        "formal validation returns",
    )
    if returns["validation_event_id"].duplicated().any():
        raise MarketAugmentedInnovationError("Formal validation returns are duplicated")
    return_rows = returns.loc[returns["market_excess_return_ready"].astype(bool)].copy()
    base_columns = [
        "validation_event_id",
        "cik",
        "actual_available_at",
        "gm_delta_yoy",
        "gm_innovation_raw",
        "gm_innovation_std",
    ]
    selected_source_raw = (
        "gm_innovation_raw"
        if selected_model_id == "E0_financial_baseline"
        else f"{selected_model_id}_innovation_raw"
    )
    selected_source_std = (
        "gm_innovation_std"
        if selected_model_id == "E0_financial_baseline"
        else f"{selected_model_id}_innovation_std"
    )
    selected_predictions = predictions[base_columns].copy()
    selected_predictions["selected_innovation_raw"] = predictions[selected_source_raw]
    selected_predictions["selected_innovation_std"] = predictions[selected_source_std]
    sample = selected_predictions.merge(
        return_rows[["validation_event_id", "market_excess_return_arithmetic"]],
        on="validation_event_id",
        how="inner",
        validate="one_to_one",
    )
    sample = sample.dropna(
        subset=[
            "gm_delta_yoy",
            "gm_innovation_std",
            "selected_innovation_std",
            "market_excess_return_arithmetic",
        ]
    ).reset_index(drop=True)
    if len(sample) < 20 or sample["cik"].nunique() < 3:
        raise MarketAugmentedInnovationError("Return diagnostic lacks a usable issuer sample")
    feature_map = {
        "gm_delta_yoy": "Delta",
        "gm_innovation_raw": "Original Innovation raw",
        "gm_innovation_std": "Original Innovation std",
        "selected_innovation_raw": "Selected Innovation raw",
        "selected_innovation_std": "Selected Innovation std",
    }
    ic_rows: list[dict[str, Any]] = []
    for feature, label in feature_map.items():
        subset = sample.dropna(subset=[feature])
        point = _spearman(subset[feature], subset["market_excess_return_arithmetic"])
        draws = _cluster_bootstrap_stat(
            subset,
            lambda frame, name=feature: _spearman(
                frame[name], frame["market_excess_return_arithmetic"]
            ),
            cfg,
        )
        ic_rows.append(
            {
                "feature": feature,
                "label": label,
                "events": len(subset),
                "issuers": int(subset["cik"].nunique()),
                "spearman_ic": point,
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    ic = pd.DataFrame(ic_rows)
    contrast_specs = (
        (
            "selected_minus_original_std_ic",
            "selected_innovation_std",
            "gm_innovation_std",
        ),
        ("selected_minus_delta_ic", "selected_innovation_std", "gm_delta_yoy"),
    )
    contrast_rows: list[dict[str, Any]] = []
    for test_id, left, right in contrast_specs:

        def statistic(
            frame: pd.DataFrame, left_column: str = left, right_column: str = right
        ) -> float:
            return _spearman(
                frame[left_column], frame["market_excess_return_arithmetic"]
            ) - _spearman(frame[right_column], frame["market_excess_return_arithmetic"])

        point = statistic(sample)
        draws = _cluster_bootstrap_stat(sample, statistic, cfg)
        contrast_rows.append(
            {
                "test_id": test_id,
                "point_estimate": point,
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_p_value": float(
                    min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
                ),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    selected_ic = ic.loc[ic["feature"].eq("selected_innovation_std")].iloc[0]
    original_contrast = contrasts.loc[
        contrasts["test_id"].eq("selected_minus_original_std_ic")
    ].iloc[0]
    verdict = (
        "exploratory_supportive_evidence"
        if selected_ic["ci_low"] > 0.0 and original_contrast["ci_low"] > 0.0
        else "effectiveness_not_established"
    )
    report: dict[str, Any] = {
        "contract_version": MARKET_AUGMENTED_RETURN_VERSION,
        "selected_model_id": selected_model_id,
        "same_complete_case_events": len(sample),
        "same_complete_case_issuers": int(sample["cik"].nunique()),
        "selected_innovation_ic": float(selected_ic["spearman_ic"]),
        "selected_innovation_ic_ci_low": float(selected_ic["ci_low"]),
        "selected_innovation_ic_ci_high": float(selected_ic["ci_high"]),
        "selected_minus_original_ic": float(original_contrast["point_estimate"]),
        "selected_minus_original_ci_low": float(original_contrast["ci_low"]),
        "selected_minus_original_ci_high": float(original_contrast["ci_high"]),
        "verdict": verdict,
        "evidence_role": "exploratory_reuse_of_outcome_observed_sample",
        "clean_unseen_holdout": False,
        "strategy_pnl_computed": False,
        "paper_trading_authorized": False,
        "live_trading_authorized": False,
        "config": asdict(cfg),
    }
    return MarketAugmentedReturnTables(sample, ic, contrasts, report)


def build_market_augmented_innovation(
    market_state_root: Path,
    *,
    output_root: Path = Path("data/processed/market_augmented_innovation"),
    config: MarketAugmentedInnovationConfig | None = None,
) -> MarketAugmentedResult:
    """Persist outcome-independent forecast selection before reading returns."""
    cfg = config or MarketAugmentedInnovationConfig()
    cfg.validate()
    market_manifest = _sha256_file(market_state_root / "output_manifest.json")
    tables = calculate_market_augmented_innovation(
        load_catalog_table(market_state_root, "validation_market_state_features.parquet"), cfg
    )
    payload = json.dumps(
        {"market_state_manifest_sha256": market_manifest, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MarketAugmentedResult(
            target.resolve(), _read_json(target / "market_augmented_innovation_report.json")
        )
    if target.exists():
        raise MarketAugmentedInnovationError(f"Incomplete forecast directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "market_augmented_predictions.parquet": tables.predictions,
        "market_augmented_common_sample.parquet": tables.common_sample,
        "market_augmented_forecast_metrics.parquet": tables.forecast_metrics,
        "market_augmented_selection_checks.parquet": tables.selection_checks,
    }
    artifacts = _write_tables(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_state_manifest_sha256": market_manifest,
        "artifacts": artifacts,
    }
    report_path = staging / "market_augmented_innovation_report.json"
    _finalize(staging, target, report_path, report, artifacts, {"market_state": market_manifest})
    return MarketAugmentedResult(target.resolve(), report)


def build_market_augmented_return_diagnostics(
    model_root: Path,
    return_ledger_root: Path,
    *,
    output_root: Path = Path("data/processed/market_augmented_return_diagnostics"),
    config: MarketAugmentedInnovationConfig | None = None,
) -> MarketAugmentedResult:
    """Persist exploratory return diagnostics after immutable forecast selection."""
    cfg = config or MarketAugmentedInnovationConfig()
    cfg.validate()
    model_manifest = _sha256_file(model_root / "output_manifest.json")
    return_manifest = _sha256_file(return_ledger_root / "output_manifest.json")
    model_report = _read_json(model_root / "market_augmented_innovation_report.json")
    if model_report.get("return_values_loaded") is not False:
        raise MarketAugmentedInnovationError("Forecast selection was not outcome-independent")
    selected = str(model_report["selected_model_id"])
    tables = calculate_market_augmented_return_diagnostics(
        load_catalog_table(model_root, "market_augmented_predictions.parquet"),
        load_catalog_table(return_ledger_root, "validation_event_return_ledger.parquet"),
        selected,
        cfg,
    )
    payload = json.dumps(
        {
            "model_manifest_sha256": model_manifest,
            "return_manifest_sha256": return_manifest,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        MARKET_AUGMENTED_RETURN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MarketAugmentedResult(
            target.resolve(), _read_json(target / "market_augmented_return_report.json")
        )
    if target.exists():
        raise MarketAugmentedInnovationError(f"Incomplete return directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "market_augmented_return_sample.parquet": tables.analysis_sample,
        "market_augmented_representation_ic.parquet": tables.representation_ic,
        "market_augmented_incremental_tests.parquet": tables.incremental_tests,
    }
    artifacts = _write_tables(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256": model_manifest,
        "return_manifest_sha256": return_manifest,
        "artifacts": artifacts,
    }
    report_path = staging / "market_augmented_return_report.json"
    _finalize(
        staging,
        target,
        report_path,
        report,
        artifacts,
        {"model": model_manifest, "returns": return_manifest},
    )
    return MarketAugmentedResult(target.resolve(), report)


def _expanding_correction_predictions(
    data: pd.DataFrame,
    spec: ExpectationModelSpec,
    config: MarketAugmentedInnovationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in data.itertuples(index=False):
        cutoff = pd.Timestamp(cast(Any, event.model_feature_cutoff))
        train = data.loc[data["actual_available_at"].lt(cutoff)].dropna(
            subset=["baseline_error", *spec.features]
        )
        if (
            len(train) < config.minimum_training_events
            or train["cik"].nunique() < config.minimum_training_issuers
        ):
            rows.append(_empty_prediction())
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
        model.fit(
            train.loc[:, list(spec.features)].to_numpy(dtype=float),
            train["baseline_error"].to_numpy(dtype=float),
        )
        event_frame = pd.DataFrame([cast(Any, event)._asdict()])
        correction = float(
            model.predict(event_frame.loc[:, list(spec.features)].to_numpy(dtype=float))[0]
        )
        rows.append(
            {
                "forecast": float(cast(Any, event.gm_forecast)) + correction,
                "correction": correction,
                "training_end": train["actual_available_at"].max(),
                "training_events": len(train),
                "training_issuers": int(train["cik"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _strict_prior_sigmas(
    predictions: pd.DataFrame,
    *,
    error_column: str,
    minimum_observations: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sigmas: list[float] = []
    counts: list[Any] = []
    history_ends: list[Any] = []
    for event in predictions.itertuples(index=False):
        cutoff = pd.Timestamp(cast(Any, event.model_feature_cutoff))
        prior = predictions.loc[
            predictions["actual_available_at"].lt(cutoff) & predictions[error_column].notna()
        ]
        if len(prior) < minimum_observations:
            sigmas.append(np.nan)
            counts.append(pd.NA)
            history_ends.append(pd.NaT)
            continue
        sigma = float(prior[error_column].std(ddof=1))
        sigmas.append(sigma if math.isfinite(sigma) and sigma > 0.0 else np.nan)
        counts.append(len(prior))
        history_ends.append(prior["actual_available_at"].max())
    return (
        pd.Series(sigmas, index=predictions.index, dtype=float),
        pd.Series(counts, index=predictions.index, dtype="Int64"),
        pd.Series(history_ends, index=predictions.index, dtype="datetime64[ns, UTC]"),
    )


def _forecast_metrics(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual = common["gm_level"].to_numpy(dtype=float)
    for spec in MODEL_SPECS:
        forecast = common[f"{spec.model_id}_forecast"].to_numpy(dtype=float)
        error = actual - forecast
        rows.append(
            {
                "model_id": spec.model_id,
                "feature_set": spec.feature_set,
                "events": len(common),
                "issuers": int(common["cik"].nunique()),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(np.mean(error)),
                "forecast_actual_spearman": _spearman(pd.Series(forecast), pd.Series(actual)),
                "error_delta_spearman": _spearman(
                    pd.Series(error), common["gm_delta_yoy"].reset_index(drop=True)
                ),
                "complexity_rank": spec.complexity_rank,
            }
        )
    return pd.DataFrame(rows)


def _select_financial_forecast(
    metrics: pd.DataFrame, config: MarketAugmentedInnovationConfig
) -> tuple[pd.DataFrame, str]:
    baseline = metrics.loc[metrics["model_id"].eq("E0_financial_baseline")].iloc[0]
    alternatives = metrics.loc[~metrics["model_id"].eq("E0_financial_baseline")].sort_values(
        ["mae", "complexity_rank", "model_id"], kind="mergesort"
    )
    best = alternatives.iloc[0]
    relative = float((baseline["mae"] - best["mae"]) / baseline["mae"])
    passed = bool(relative >= config.minimum_relative_mae_improvement)
    selected = str(best["model_id"]) if passed else "E0_financial_baseline"
    checks = pd.DataFrame(
        [
            {
                "check_id": "best_candidate_mae_below_baseline",
                "observed": float(baseline["mae"] - best["mae"]),
                "required": 0.0,
                "passed": bool(best["mae"] < baseline["mae"]),
            },
            {
                "check_id": "relative_mae_improvement",
                "observed": relative,
                "required": config.minimum_relative_mae_improvement,
                "passed": passed,
            },
        ]
    )
    return checks, selected


def _cluster_bootstrap_stat(
    sample: pd.DataFrame,
    statistic: Any,
    config: MarketAugmentedInnovationConfig,
) -> np.ndarray:
    issuers = sample["cik"].drop_duplicates().astype(str).to_numpy()
    groups = {issuer: sample.loc[sample["cik"].astype(str).eq(issuer)] for issuer in issuers}
    rng = np.random.default_rng(config.bootstrap_seed)
    draws: list[float] = []
    while len(draws) < config.bootstrap_repetitions:
        selected = rng.choice(issuers, size=len(issuers), replace=True)
        frame = pd.concat([groups[str(issuer)] for issuer in selected], ignore_index=True)
        value = float(statistic(frame))
        if math.isfinite(value):
            draws.append(value)
    return np.asarray(draws, dtype=float)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(spearmanr(pair["left"], pair["right"]).statistic)


def _empty_prediction() -> dict[str, Any]:
    return {
        "forecast": np.nan,
        "correction": np.nan,
        "training_end": pd.NaT,
        "training_events": pd.NA,
        "training_issuers": pd.NA,
    }


def _write_tables(staging: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    return artifacts


def _finalize(
    staging: Path,
    target: Path,
    report_path: Path,
    report: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> None:
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
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


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MarketAugmentedInnovationError(f"{label} lacks required columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise MarketAugmentedInnovationError(f"{label} contains invalid timestamps")
    return result


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
