"""Strict walk-forward baseline tournament for the first local equity alpha model."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from tradeable_dataset import (
    COMBINED_FEATURE_COLUMNS,
    FILING_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    validate_tradeable_dataset,
)

BASELINE_TOURNAMENT_VERSION: Final[str] = "baseline-alpha-tournament-v1"
PRIMARY_LABEL: Final[str] = "target_market_excess_return_20d"


@dataclass(frozen=True)
class BaselineTournamentConfig:
    """Frozen candidates, temporal gates, and model hyperparameters."""

    minimum_training_events: int = 2_000
    minimum_training_issuers: int = 100
    minimum_validation_events: int = 500
    minimum_validation_periods: int = 8
    minimum_events_per_period_ic: int = 10
    minimum_selected_mean_period_ic: float = 0.02
    minimum_combined_minus_market_ic: float = 0.01
    ridge_alpha: float = 10.0
    hist_learning_rate: float = 0.05
    hist_max_iter: int = 100
    hist_max_leaf_nodes: int = 7
    hist_min_samples_leaf: int = 50
    hist_l2_regularization: float = 1.0
    random_seed: int = 42
    holdout_metrics_authorized: bool = False
    version: str = BASELINE_TOURNAMENT_VERSION

    def validate(self) -> None:
        """Reject weak gates or an accidental holdout reveal."""
        if self.minimum_training_events < 20:
            raise ValueError("At least 20 training events are required even in a test fixture")
        if self.minimum_training_issuers < 3:
            raise ValueError("At least three training issuers are required")
        if self.minimum_validation_events < 20:
            raise ValueError("At least 20 validation events are required")
        if self.minimum_validation_periods < 2:
            raise ValueError("At least two validation periods are required")
        if self.minimum_events_per_period_ic < 5:
            raise ValueError("A period IC requires at least five events")
        if self.ridge_alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive")
        if not 0.0 < self.hist_learning_rate <= 0.2:
            raise ValueError("Histogram learning rate must be in (0, 0.2]")
        if self.hist_max_iter < 20 or self.hist_max_leaf_nodes < 3:
            raise ValueError("Histogram boosting is underspecified")
        if self.hist_min_samples_leaf < 2:
            raise ValueError("Histogram leaves require at least two observations")
        if self.holdout_metrics_authorized:
            raise ValueError("The v1 command is not authorized to reveal sealed holdout metrics")


@dataclass(frozen=True)
class ModelSpec:
    """One frozen algorithm and feature-set candidate."""

    model_id: str
    algorithm: Literal["zero", "ridge", "hist_gradient_boosting"]
    feature_set: Literal["none", "filing_only", "market_only", "combined"]
    features: tuple[str, ...]
    complexity_rank: int


@dataclass(frozen=True)
class BaselineTournamentTables:
    """Pure tournament outputs before persistence."""

    predictions: pd.DataFrame
    fold_audit: pd.DataFrame
    model_diagnostics: pd.DataFrame
    selection_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class BaselineTournamentResult:
    """Immutable on-disk tournament bundle."""

    output_root: Path
    report: dict[str, Any]


MODEL_SPECS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec("M0_zero", "zero", "none", (), 0),
    ModelSpec("M1_ridge_filing", "ridge", "filing_only", FILING_FEATURE_COLUMNS, 1),
    ModelSpec("M2_ridge_market", "ridge", "market_only", MARKET_FEATURE_COLUMNS, 2),
    ModelSpec("M3_ridge_combined", "ridge", "combined", COMBINED_FEATURE_COLUMNS, 3),
    ModelSpec(
        "M4_hist_filing",
        "hist_gradient_boosting",
        "filing_only",
        FILING_FEATURE_COLUMNS,
        4,
    ),
    ModelSpec(
        "M5_hist_market",
        "hist_gradient_boosting",
        "market_only",
        MARKET_FEATURE_COLUMNS,
        5,
    ),
    ModelSpec(
        "M6_hist_combined",
        "hist_gradient_boosting",
        "combined",
        COMBINED_FEATURE_COLUMNS,
        6,
    ),
)


def calculate_baseline_tournament(
    event_dataset: pd.DataFrame,
    config: BaselineTournamentConfig | None = None,
) -> BaselineTournamentTables:
    """Generate quarterly, label-aware walk-forward validation predictions."""
    cfg = config or BaselineTournamentConfig()
    cfg.validate()
    validate_tradeable_dataset(event_dataset)
    _require_columns(
        event_dataset,
        {
            PRIMARY_LABEL,
            "cik",
            "feature_cutoff",
            "target_available_at_20d",
            "time_role",
            "model_training_candidate",
            "model_evaluation_candidate",
            *COMBINED_FEATURE_COLUMNS,
        },
        "event feature store",
    )
    data = event_dataset.copy()
    data["feature_cutoff"] = _utc(data["feature_cutoff"], "feature_cutoff")
    data["target_available_at_20d"] = _utc(
        data["target_available_at_20d"], "target_available_at_20d"
    )
    data[PRIMARY_LABEL] = pd.to_numeric(data[PRIMARY_LABEL], errors="coerce")
    validation = data.loc[
        data["model_evaluation_candidate"].astype(bool) & data["time_role"].eq("validation")
    ].copy()
    holdout = data.loc[
        data["model_evaluation_candidate"].astype(bool) & data["time_role"].eq("sealed_holdout")
    ].copy()
    validation["validation_period"] = (
        validation["feature_cutoff"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    )

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for period, test in validation.groupby("validation_period", sort=True):
        period_start = pd.Period(str(period), freq="Q").start_time.tz_localize("UTC")
        train = data.loc[
            data["model_training_candidate"].astype(bool)
            & ~data["time_role"].eq("sealed_holdout")
            & data["target_available_at_20d"].lt(period_start)
            & data["feature_cutoff"].lt(period_start)
        ].copy()
        training_gate = (
            len(train) >= cfg.minimum_training_events
            and train["cik"].nunique() >= cfg.minimum_training_issuers
        )
        fold_rows.append(
            {
                "validation_period": period,
                "period_start": period_start,
                "test_events": len(test),
                "test_issuers": int(test["cik"].nunique()),
                "training_events": len(train),
                "training_issuers": int(train["cik"].nunique()),
                "training_label_end": train["target_available_at_20d"].max(),
                "training_gate": training_gate,
                "fold_status": "ready" if training_gate else "insufficient_training_history",
            }
        )
        if not training_gate:
            continue
        for spec in MODEL_SPECS:
            model_predictions = _fit_predict(train, test, spec, cfg)
            for event, prediction in zip(
                test.itertuples(index=False), model_predictions, strict=True
            ):
                prediction_rows.append(
                    {
                        "longitudinal_event_id": str(event.longitudinal_event_id),
                        "cik": str(event.cik),
                        "feature_cutoff": event.feature_cutoff,
                        "target_available_at_20d": event.target_available_at_20d,
                        "validation_period": period,
                        "model_id": spec.model_id,
                        "algorithm": spec.algorithm,
                        "feature_set": spec.feature_set,
                        "prediction": float(prediction),
                        "actual": float(getattr(event, PRIMARY_LABEL)),
                        "training_events": len(train),
                        "training_issuers": int(train["cik"].nunique()),
                        "training_label_end": train["target_available_at_20d"].max(),
                    }
                )
    predictions_frame = pd.DataFrame(prediction_rows, columns=_prediction_columns())
    fold_audit = pd.DataFrame(fold_rows, columns=_fold_columns())
    diagnostics = _model_diagnostics(predictions_frame, cfg)
    checks, verdict, selected = _selection_checks(diagnostics, cfg)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "candidate_models": len(MODEL_SPECS),
        "validation_candidate_events": len(validation),
        "validation_candidate_issuers": int(validation["cik"].nunique()),
        "validation_periods": int(validation["validation_period"].nunique()),
        "walk_forward_prediction_rows": len(predictions_frame),
        "sealed_holdout_candidate_events": len(holdout),
        "holdout_status": "sealed_not_evaluated",
        "holdout_metrics_computed": False,
        "selected_model_id": selected,
        "selection_verdict": verdict,
        "paper_trading_authorized": False,
        "live_trading_authorized": False,
        "config": asdict(cfg),
        "models": [asdict(spec) for spec in MODEL_SPECS],
    }
    return BaselineTournamentTables(predictions_frame, fold_audit, diagnostics, checks, report)


def build_baseline_tournament(
    dataset_root: Path,
    *,
    output_root: Path = Path("data/processed/baseline_alpha_tournament"),
    config: BaselineTournamentConfig | None = None,
) -> BaselineTournamentResult:
    """Persist immutable validation diagnostics while leaving the holdout sealed."""
    cfg = config or BaselineTournamentConfig()
    cfg.validate()
    dataset_manifest = _sha256_file(dataset_root / "output_manifest.json")
    event_dataset = load_catalog_table(dataset_root, "event_feature_store.parquet")
    tables = calculate_baseline_tournament(event_dataset, cfg)
    payload = json.dumps(
        {"dataset_manifest_sha256": dataset_manifest, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return BaselineTournamentResult(
            target.resolve(), _read_json(target / "baseline_tournament_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete baseline tournament exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "walk_forward_predictions.parquet": tables.predictions,
        "fold_audit.parquet": tables.fold_audit,
        "model_diagnostics.parquet": tables.model_diagnostics,
        "selection_checks.parquet": tables.selection_checks,
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
        "dataset_manifest_sha256": dataset_manifest,
        "artifacts": artifacts,
    }
    report_path = staging / "baseline_tournament_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "dataset_manifest_sha256": dataset_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return BaselineTournamentResult(target.resolve(), report)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ModelSpec,
    config: BaselineTournamentConfig,
) -> np.ndarray:
    if spec.algorithm == "zero":
        return np.zeros(len(test), dtype=float)
    training_x = train.loc[:, list(spec.features)].to_numpy(dtype=float)
    test_x = test.loc[:, list(spec.features)].to_numpy(dtype=float)
    training_y = train[PRIMARY_LABEL].to_numpy(dtype=float)
    if not (
        np.isfinite(training_x).all()
        and np.isfinite(test_x).all()
        and np.isfinite(training_y).all()
    ):
        raise FilingCatalogError(f"{spec.model_id} received a non-finite training matrix")
    if spec.algorithm == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=config.hist_learning_rate,
            max_iter=config.hist_max_iter,
            max_leaf_nodes=config.hist_max_leaf_nodes,
            min_samples_leaf=config.hist_min_samples_leaf,
            l2_regularization=config.hist_l2_regularization,
            early_stopping=False,
            random_state=config.random_seed,
        )
    model.fit(training_x, training_y)
    result = np.asarray(model.predict(test_x), dtype=float)
    if not np.isfinite(result).all():
        raise FilingCatalogError(f"{spec.model_id} produced a non-finite prediction")
    return result


def _model_diagnostics(predictions: pd.DataFrame, config: BaselineTournamentConfig) -> pd.DataFrame:
    columns = [
        "model_id",
        "algorithm",
        "feature_set",
        "events",
        "issuers",
        "periods",
        "valid_ic_periods",
        "pooled_spearman_ic",
        "mean_period_ic",
        "median_period_ic",
        "mae",
        "rmse",
        "direction_accuracy",
        "top_bottom_spread",
        "validation_gate",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        period_ics = [
            _spearman(period["prediction"], period["actual"])
            for _, period in group.groupby("validation_period", sort=True)
            if len(period) >= config.minimum_events_per_period_ic
        ]
        finite_period_ics = [value for value in period_ics if math.isfinite(value)]
        residual = group["actual"].astype(float) - group["prediction"].astype(float)
        rank = group["prediction"].rank(method="first", pct=True)
        top = group.loc[rank.gt(0.8), "actual"]
        bottom = group.loc[rank.le(0.2), "actual"]
        rows.append(
            {
                "model_id": str(model_id),
                "algorithm": str(group["algorithm"].iloc[0]),
                "feature_set": str(group["feature_set"].iloc[0]),
                "events": len(group),
                "issuers": int(group["cik"].nunique()),
                "periods": int(group["validation_period"].nunique()),
                "valid_ic_periods": len(finite_period_ics),
                "pooled_spearman_ic": _spearman(group["prediction"], group["actual"]),
                "mean_period_ic": (
                    float(np.mean(finite_period_ics)) if finite_period_ics else float("nan")
                ),
                "median_period_ic": (
                    float(np.median(finite_period_ics)) if finite_period_ics else float("nan")
                ),
                "mae": float(residual.abs().mean()),
                "rmse": float(math.sqrt(float((residual**2).mean()))),
                "direction_accuracy": float(
                    (np.sign(group["prediction"]) == np.sign(group["actual"])).mean()
                ),
                "top_bottom_spread": (
                    float(top.mean() - bottom.mean())
                    if not top.empty and not bottom.empty
                    else float("nan")
                ),
                "validation_gate": bool(
                    len(group) >= config.minimum_validation_events
                    and len(finite_period_ics) >= config.minimum_validation_periods
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("model_id").reset_index(drop=True)


def _selection_checks(
    diagnostics: pd.DataFrame, config: BaselineTournamentConfig
) -> tuple[pd.DataFrame, str, str | None]:
    check_rows: list[dict[str, Any]] = []
    eligible = diagnostics.loc[
        diagnostics.get("validation_gate", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    if eligible.empty:
        check_rows.append(
            {
                "check": "minimum_validation_evidence",
                "passed": False,
                "value": 0.0,
                "threshold": float(config.minimum_validation_events),
            }
        )
        return (
            pd.DataFrame(check_rows),
            "insufficient_unobserved_validation_evidence",
            None,
        )
    combined = eligible.loc[eligible["feature_set"].eq("combined")].copy()
    market = eligible.loc[eligible["feature_set"].eq("market_only")].copy()
    if combined.empty or market.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "check": "same_sample_combined_and_market_models",
                        "passed": False,
                        "value": 0.0,
                        "threshold": 1.0,
                    }
                ]
            ),
            "incomplete_same_sample_model_comparison",
            None,
        )
    combined = combined.sort_values(
        ["mean_period_ic", "model_id"], ascending=[False, True], kind="mergesort"
    )
    market = market.sort_values(
        ["mean_period_ic", "model_id"], ascending=[False, True], kind="mergesort"
    )
    best_combined = combined.iloc[0]
    best_market = market.iloc[0]
    selected_ic = float(best_combined["mean_period_ic"])
    incremental = selected_ic - float(best_market["mean_period_ic"])
    check_rows.extend(
        [
            {
                "check": "selected_combined_mean_period_ic",
                "passed": selected_ic >= config.minimum_selected_mean_period_ic,
                "value": selected_ic,
                "threshold": config.minimum_selected_mean_period_ic,
            },
            {
                "check": "combined_minus_market_mean_period_ic",
                "passed": incremental >= config.minimum_combined_minus_market_ic,
                "value": incremental,
                "threshold": config.minimum_combined_minus_market_ic,
            },
            {
                "check": "sealed_holdout_untouched",
                "passed": True,
                "value": 1.0,
                "threshold": 1.0,
            },
        ]
    )
    checks = pd.DataFrame(check_rows)
    verdict = (
        "baseline_alpha_research_gate_passed"
        if checks["passed"].astype(bool).all()
        else "combined_incremental_value_not_established"
    )
    return checks, verdict, str(best_combined["model_id"])


def _spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    value = frame.iloc[:, 0].rank(method="average").corr(frame.iloc[:, 1].rank(method="average"))
    return float(value) if pd.notna(value) else float("nan")


def _prediction_columns() -> list[str]:
    return [
        "longitudinal_event_id",
        "cik",
        "feature_cutoff",
        "target_available_at_20d",
        "validation_period",
        "model_id",
        "algorithm",
        "feature_set",
        "prediction",
        "actual",
        "training_events",
        "training_issuers",
        "training_label_end",
    ]


def _fold_columns() -> list[str]:
    return [
        "validation_period",
        "period_start",
        "test_events",
        "test_issuers",
        "training_events",
        "training_issuers",
        "training_label_end",
        "training_gate",
        "fold_status",
    ]


def _utc(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise FilingCatalogError(f"{name} contains invalid timestamps")
    return converted


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise FilingCatalogError(f"{name} is missing columns: {sorted(missing)}")


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
