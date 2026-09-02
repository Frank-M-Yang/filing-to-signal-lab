"""Freeze and score the Conditional Drift candidate without outcome leakage.

This module turns the historical v2 development result into an immutable model
candidate.  It deliberately separates three operations:

1. fit the fixed O1 challenger and O3 Alignment-only candidate on the already
   observed development sample;
2. register a temporal, outcome-blind evaluation queue; and
3. score only feature rows that contain no target columns and whose feature
   clocks precede the formal execution timestamp.

The output is a research score, not a trading recommendation.
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
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from conditional_drift import ALL_FEATURE_COLUMNS
from conditional_drift_v2 import ALIGNMENT_COLUMNS
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

CONDITIONAL_DRIFT_FORWARD_VERSION: Final[str] = "conditional-drift-forward-candidate-v1"
PRIMARY_TARGET: Final[str] = "target_net_market_excess_return_20d_liquidity_base"
PRIMARY_TARGET_AVAILABLE_AT: Final[str] = "target_available_at_20d"
FORWARD_EVIDENCE_ROLES: Final[frozenset[str]] = frozenset(
    {
        "unobserved_validation_candidate",
        "unobserved_evaluation_candidate",
        "sealed_holdout",
    }
)
CHALLENGER_FEATURES: Final[tuple[str, ...]] = ALL_FEATURE_COLUMNS
ALIGNMENT_FEATURES: Final[tuple[str, ...]] = (*ALL_FEATURE_COLUMNS, *ALIGNMENT_COLUMNS)


@dataclass(frozen=True)
class ConditionalDriftForwardConfig:
    """Frozen model, target, and outcome-firewall conventions."""

    primary_horizon_sessions: int = 20
    ridge_alpha: float = 10.0
    minimum_training_events: int = 30
    minimum_training_issuers: int = 10
    version: str = CONDITIONAL_DRIFT_FORWARD_VERSION

    def validate(self) -> None:
        """Reject changes that would silently retune the frozen candidate."""
        if self.primary_horizon_sessions != 20:
            raise ValueError("The forward candidate freezes a 20-session target")
        if self.ridge_alpha != 10.0:
            raise ValueError("The forward candidate freezes Ridge alpha at 10")
        if self.minimum_training_events < 30 or self.minimum_training_issuers < 10:
            raise ValueError("The forward freeze training gate is too weak")


@dataclass(frozen=True)
class FrozenLinearModel:
    """Serializable StandardScaler plus Ridge parameters for one model."""

    model_id: str
    feature_set: str
    features: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    training_feature_min: tuple[float, ...]
    training_feature_max: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge_alpha: float
    training_events: int
    training_issuers: int
    training_label_start: str
    training_label_end: str

    def validate(self) -> None:
        """Validate parameter dimensions and finite numerical values."""
        width = len(self.features)
        if not (
            len(self.scaler_mean)
            == len(self.scaler_scale)
            == len(self.training_feature_min)
            == len(self.training_feature_max)
            == len(self.coefficients)
            == width
        ):
            raise ValueError(f"Frozen model {self.model_id} has inconsistent dimensions")
        values = (
            *self.scaler_mean,
            *self.scaler_scale,
            *self.training_feature_min,
            *self.training_feature_max,
            *self.coefficients,
            self.intercept,
        )
        if not all(np.isfinite(values)) or not all(value > 0.0 for value in self.scaler_scale):
            raise ValueError(f"Frozen model {self.model_id} has invalid parameters")
        if not all(
            low <= high
            for low, high in zip(self.training_feature_min, self.training_feature_max, strict=True)
        ):
            raise ValueError(f"Frozen model {self.model_id} has invalid training support")


@dataclass(frozen=True)
class ConditionalDriftForwardTables:
    """Pure forward-freeze outputs before immutable persistence."""

    model_parameters: pd.DataFrame
    scoring_queue: pd.DataFrame
    queue_funnel: pd.DataFrame
    scored_candidates: pd.DataFrame
    gates: pd.DataFrame
    models: tuple[FrozenLinearModel, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftForwardResult:
    """Immutable forward candidate bundle."""

    output_root: Path
    report: dict[str, Any]


def freeze_conditional_drift_models(
    development: pd.DataFrame,
    config: ConditionalDriftForwardConfig | None = None,
) -> tuple[FrozenLinearModel, ...]:
    """Fit the preregistered challenger and candidate on observed development rows."""
    cfg = config or ConditionalDriftForwardConfig()
    cfg.validate()
    required = {
        "cik",
        "evidence_role",
        "v2_development_candidate",
        PRIMARY_TARGET,
        PRIMARY_TARGET_AVAILABLE_AT,
        *ALIGNMENT_FEATURES,
    }
    _require_columns(development, required, "Conditional Drift v2 development dataset")
    sample = development.loc[development["v2_development_candidate"].astype(bool)].copy()
    if not sample["evidence_role"].eq("legacy_observed_training_only").all():
        raise FilingCatalogError("Forward freeze received a non-development evidence role")
    sample[PRIMARY_TARGET_AVAILABLE_AT] = _utc(
        sample[PRIMARY_TARGET_AVAILABLE_AT], PRIMARY_TARGET_AVAILABLE_AT
    )
    complete_columns = [PRIMARY_TARGET, *ALIGNMENT_FEATURES]
    if sample.loc[:, complete_columns].isna().any().any():
        raise FilingCatalogError("Forward freeze development sample contains missing model data")
    if len(sample) < cfg.minimum_training_events:
        raise FilingCatalogError("Forward freeze has too few development events")
    if sample["cik"].nunique() < cfg.minimum_training_issuers:
        raise FilingCatalogError("Forward freeze has too few development issuers")
    start = sample[PRIMARY_TARGET_AVAILABLE_AT].min().isoformat()
    end = sample[PRIMARY_TARGET_AVAILABLE_AT].max().isoformat()
    specifications = (
        ("F1_v1_challenger", "v1_full", CHALLENGER_FEATURES),
        ("F2_alignment_candidate", "v1_full_plus_alignment", ALIGNMENT_FEATURES),
    )
    models: list[FrozenLinearModel] = []
    target = sample[PRIMARY_TARGET].astype(float).to_numpy()
    for model_id, feature_set, features in specifications:
        features_frame = sample.loc[:, list(features)].astype(float)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(features_frame)
        ridge = Ridge(alpha=cfg.ridge_alpha)
        ridge.fit(standardized, target)
        model = FrozenLinearModel(
            model_id=model_id,
            feature_set=feature_set,
            features=features,
            scaler_mean=tuple(float(value) for value in scaler.mean_),
            scaler_scale=tuple(float(value) for value in scaler.scale_),
            training_feature_min=tuple(float(value) for value in features_frame.min()),
            training_feature_max=tuple(float(value) for value in features_frame.max()),
            coefficients=tuple(float(value) for value in ridge.coef_),
            intercept=float(ridge.intercept_),
            ridge_alpha=cfg.ridge_alpha,
            training_events=len(sample),
            training_issuers=int(sample["cik"].nunique()),
            training_label_start=start,
            training_label_end=end,
        )
        model.validate()
        models.append(model)
    return tuple(models)


def score_forward_candidates(
    candidates: pd.DataFrame,
    models: tuple[FrozenLinearModel, ...],
) -> pd.DataFrame:
    """Score outcome-free, temporally forward feature rows with frozen parameters.

    Target columns are rejected rather than ignored so an accidental outcome join
    cannot silently pass through the scoring boundary.
    """
    if not models:
        raise ValueError("At least one frozen model is required")
    forbidden = sorted(
        column
        for column in candidates.columns
        if column.startswith("target_")
        or column.startswith("actual_")
        or column in {"market_excess_return_arithmetic", "resolved_outcome_status"}
    )
    if forbidden:
        raise FilingCatalogError(
            "Forward scoring input contains forbidden outcome columns: " + ", ".join(forbidden)
        )
    feature_union = tuple(dict.fromkeys(feature for model in models for feature in model.features))
    required = {
        "validation_event_id",
        "cik",
        "ticker",
        "accepted_at",
        "reaction_feature_available_at",
        "execution_entry_at",
        "evidence_role",
        "prior_outcome_observed",
        *feature_union,
    }
    _require_columns(candidates, required, "forward scoring input")
    if candidates["validation_event_id"].duplicated().any():
        raise FilingCatalogError("Forward scoring input contains duplicate events")
    if not set(candidates["evidence_role"].astype(str)).issubset(FORWARD_EVIDENCE_ROLES):
        raise FilingCatalogError("Forward scoring input contains a training-only evidence role")
    if candidates["prior_outcome_observed"].astype(bool).any():
        raise FilingCatalogError("Forward scoring input contains an observed outcome event")
    if (
        "outcome_materialized_locally" in candidates
        and candidates["outcome_materialized_locally"].astype(bool).any()
    ):
        raise FilingCatalogError("Forward scoring input contains a locally materialized outcome")
    accepted = _utc(candidates["accepted_at"], "accepted_at")
    reaction_at = _utc(candidates["reaction_feature_available_at"], "reaction_feature_available_at")
    entry_at = _utc(candidates["execution_entry_at"], "execution_entry_at")
    if not ((accepted <= reaction_at) & (reaction_at < entry_at)).all():
        raise FilingCatalogError("Forward scoring input violates the pre-entry reaction clock")
    feature_frame = candidates.loc[:, list(feature_union)].astype(float)
    if feature_frame.isna().any().any() or not np.isfinite(feature_frame.to_numpy()).all():
        raise FilingCatalogError("Forward scoring input contains missing or non-finite features")
    training_ends = {pd.Timestamp(model.training_label_end) for model in models}
    if len(training_ends) != 1:
        raise FilingCatalogError("Frozen models do not share one training cutoff")
    training_end = next(iter(training_ends))
    if not accepted.gt(training_end).all():
        raise FilingCatalogError("Forward scoring event is not strictly after the training cutoff")

    output = candidates.loc[
        :,
        [
            "validation_event_id",
            "cik",
            "ticker",
            "accepted_at",
            "reaction_feature_available_at",
            "execution_entry_at",
            "evidence_role",
        ],
    ].copy()
    for model in models:
        model.validate()
        values = candidates.loc[:, list(model.features)].astype(float).to_numpy()
        means = np.asarray(model.scaler_mean)
        scales = np.asarray(model.scaler_scale)
        lower = np.asarray(model.training_feature_min)
        upper = np.asarray(model.training_feature_max)
        coefficients = np.asarray(model.coefficients)
        standardized = (values - means) / scales
        clipped = np.clip(values, lower, upper)
        output[f"{model.model_id}_research_score"] = standardized @ coefficients + model.intercept
        output[f"{model.model_id}_support_clipped_score"] = (
            clipped - means
        ) / scales @ coefficients + model.intercept
        output[f"{model.model_id}_max_abs_standardized_feature"] = np.abs(standardized).max(axis=1)
        output[f"{model.model_id}_outside_training_range_features"] = (
            (values < lower) | (values > upper)
        ).sum(axis=1)
    output["alignment_minus_challenger_score"] = (
        output["F2_alignment_candidate_research_score"] - output["F1_v1_challenger_research_score"]
    )
    outside = output["F2_alignment_candidate_outside_training_range_features"].astype(int)
    max_abs = output["F2_alignment_candidate_max_abs_standardized_feature"].astype(float)
    output["alignment_candidate_within_training_support"] = outside.eq(0)
    output["alignment_candidate_severe_extrapolation"] = max_abs.gt(4.0) | outside.gt(
        math.ceil(len(ALIGNMENT_FEATURES) * 0.20)
    )
    output["prediction_generated_at"] = reaction_at
    output["model_training_label_end"] = training_end
    output["outcome_columns_loaded"] = False
    output["research_only"] = True
    return output.sort_values(
        ["prediction_generated_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)


def calculate_conditional_drift_forward_candidate(
    development: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    filing_features: pd.DataFrame,
    identity_readiness: pd.DataFrame,
    outcome_materialized_event_ids: set[str],
    *,
    candidate_feature_rows: pd.DataFrame | None = None,
    config: ConditionalDriftForwardConfig | None = None,
) -> ConditionalDriftForwardTables:
    """Freeze models, register the holdout queue, and score ready rows if supplied."""
    cfg = config or ConditionalDriftForwardConfig()
    cfg.validate()
    models = freeze_conditional_drift_models(development, cfg)
    queue = _build_scoring_queue(
        candidate_registry,
        filing_features,
        identity_readiness,
        outcome_materialized_event_ids,
        pd.Timestamp(models[0].training_label_end),
    )
    scored = (
        score_forward_candidates(candidate_feature_rows, models)
        if candidate_feature_rows is not None and not candidate_feature_rows.empty
        else _empty_scored_candidates()
    )
    if not scored.empty:
        unexpected = set(scored["validation_event_id"].astype(str)).difference(
            set(queue.loc[queue["forward_queue_eligible"], "validation_event_id"].astype(str))
        )
        if unexpected:
            raise FilingCatalogError("Scored candidates are absent from the clean forward queue")
    funnel = _queue_funnel(candidate_registry, filing_features, queue, len(scored))
    parameters = _model_parameter_table(models)
    clean = queue["forward_queue_eligible"].astype(bool)
    identity_ready = clean & queue["event_security_identity_ready"].fillna(False).astype(bool)
    within_support = (
        int(scored["alignment_candidate_within_training_support"].sum()) if not scored.empty else 0
    )
    severe_extrapolation = (
        int(scored["alignment_candidate_severe_extrapolation"].sum()) if not scored.empty else 0
    )
    gates = pd.DataFrame(
        [
            _gate("frozen_development_model", models[0].training_events >= 30),
            _gate("frozen_issuer_breadth", models[0].training_issuers >= 10),
            _gate("outcome_firewall", not queue.loc[clean, "outcome_materialized_locally"].any()),
            _gate("temporal_holdout_queue", int(clean.sum()) >= 10),
            _gate("identity_ready_queue", int(identity_ready.sum()) >= 10),
            _gate("outcome_blind_reaction_features", not scored.empty),
            _gate("minimum_covariate_overlap", within_support >= 8),
            _gate("external_validation_metrics", False),
            _gate("live_trading_authorization", False),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "model_status": (
            "frozen_predictions_recorded_before_evaluation"
            if not scored.empty
            else "frozen_before_holdout_scoring"
        ),
        "frozen_models": [model.model_id for model in models],
        "selected_candidate": "F2_alignment_candidate",
        "fixed_challenger": "F1_v1_challenger",
        "selection_basis": "historical_development_only_not_external_validation",
        "training_events": models[0].training_events,
        "training_issuers": models[0].training_issuers,
        "training_label_start": models[0].training_label_start,
        "training_label_end": models[0].training_label_end,
        "registered_unobserved_events": int(
            candidate_registry["evidence_role"].eq("unobserved_validation_candidate").sum()
        ),
        "temporal_complete_case_queue_events": len(queue),
        "locally_materialized_outcome_quarantine_events": int(
            queue["outcome_materialized_locally"].sum()
        ),
        "clean_temporal_queue_events": int(clean.sum()),
        "identity_ready_clean_queue_events": int(identity_ready.sum()),
        "scored_candidate_events": len(scored),
        "within_training_support_scored_events": within_support,
        "severe_extrapolation_scored_events": severe_extrapolation,
        "outcome_values_loaded_for_queue_construction": False,
        "external_validation_metrics_computed": False,
        "sealed_holdout_opened": False,
        "portfolio_backtest_computed": False,
        "live_trading_authorized": False,
        "next_required_asset": (
            "separately_authorized_holdout_outcomes_and_preregistered_evaluation"
            if not scored.empty
            else "outcome_blind_pre_entry_market_and_reaction_features"
        ),
        "config": asdict(cfg),
    }
    return ConditionalDriftForwardTables(parameters, queue, funnel, scored, gates, models, report)


def build_conditional_drift_forward_candidate(
    optimization_root: Path,
    validation_root: Path,
    filing_feature_root: Path,
    identity_root: Path,
    *,
    outcome_exposure_root: Path | None = None,
    candidate_feature_path: Path | None = None,
    output_root: Path = Path("data/processed/conditional_drift_forward_candidate"),
    config: ConditionalDriftForwardConfig | None = None,
) -> ConditionalDriftForwardResult:
    """Persist the immutable forward model bundle and outcome-blind queue."""
    cfg = config or ConditionalDriftForwardConfig()
    cfg.validate()
    manifests: dict[str, str | None] = {
        "implementation_sha256": _sha256_file(Path(__file__)),
        "optimization_manifest_sha256": _sha256_file(optimization_root / "output_manifest.json"),
        "validation_manifest_sha256": _sha256_file(validation_root / "output_manifest.json"),
        "filing_feature_manifest_sha256": _sha256_file(
            filing_feature_root / "output_manifest.json"
        ),
        "identity_manifest_sha256": _sha256_file(identity_root / "output_manifest.json"),
        "outcome_exposure_manifest_sha256": (
            _sha256_file(outcome_exposure_root / "output_manifest.json")
            if outcome_exposure_root is not None
            else None
        ),
        "candidate_feature_sha256": (
            _sha256_file(candidate_feature_path) if candidate_feature_path is not None else None
        ),
    }
    development = load_catalog_table(
        optimization_root, "conditional_drift_v2_event_dataset.parquet"
    )
    candidates = load_catalog_table(validation_root, "validation_candidate_events.parquet")
    filing = load_catalog_table(filing_feature_root, "target_filing_features.parquet")
    identity = load_catalog_table(
        identity_root, "validation_event_identity_readiness_final.parquet"
    )
    materialized_ids: set[str] = set()
    if outcome_exposure_root is not None:
        exposure_path = outcome_exposure_root / "validation_event_return_ledger.parquet"
        exposure = pd.read_parquet(
            exposure_path,
            columns=["validation_event_id", "returns_computed"],
        )
        materialized_ids = set(
            exposure.loc[exposure["returns_computed"].astype(bool), "validation_event_id"].astype(
                str
            )
        )
    feature_rows = pd.read_parquet(candidate_feature_path) if candidate_feature_path else None
    tables = calculate_conditional_drift_forward_candidate(
        development,
        candidates,
        filing,
        identity,
        materialized_ids,
        candidate_feature_rows=feature_rows,
        config=cfg,
    )
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConditionalDriftForwardResult(
            target.resolve(), _read_json(target / "conditional_drift_forward_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete forward candidate directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "frozen_model_parameters.parquet": tables.model_parameters,
        "forward_scoring_queue.parquet": tables.scoring_queue,
        "forward_queue_funnel.parquet": tables.queue_funnel,
        "forward_candidate_scores.parquet": tables.scored_candidates,
        "conditional_drift_forward_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    bundle_path = staging / "frozen_model_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": cfg.version,
                "research_only": True,
                "target": PRIMARY_TARGET,
                "target_availability_clock": PRIMARY_TARGET_AVAILABLE_AT,
                "outcome_columns_forbidden_at_scoring": True,
                "models": [asdict(model) for model in tables.models],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts[bundle_path.name] = _artifact_record(bundle_path, len(tables.models))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "conditional_drift_forward_report.json"
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
    return ConditionalDriftForwardResult(target.resolve(), report)


def _build_scoring_queue(
    candidate_registry: pd.DataFrame,
    filing_features: pd.DataFrame,
    identity_readiness: pd.DataFrame,
    outcome_materialized_event_ids: set[str],
    training_label_end: pd.Timestamp,
) -> pd.DataFrame:
    _require_columns(
        candidate_registry,
        {
            "validation_event_id",
            "cik",
            "accession_number",
            "accepted_at",
            "form",
            "company_name_snapshot",
            "collection_wave",
            "evidence_role",
            "exact_outcome_previously_observed",
            "return_outcome_read_authorized",
        },
        "validation candidate registry",
    )
    _require_columns(
        filing_features,
        {"validation_event_id", "feature_status", "gross_margin_applicability_complete_case"},
        "validation filing features",
    )
    _require_columns(
        identity_readiness,
        {
            "validation_event_id",
            "event_security_identity_ready",
            "assigned_ticker",
            "assigned_exchange_mic",
        },
        "validation identity readiness",
    )
    if candidate_registry["validation_event_id"].duplicated().any():
        raise FilingCatalogError("Validation candidate registry contains duplicate event ids")
    filing = filing_features.loc[
        :,
        [
            "validation_event_id",
            "feature_status",
            "gross_margin_applicability_complete_case",
        ],
    ].drop_duplicates("validation_event_id")
    identity = identity_readiness.loc[
        :,
        [
            "validation_event_id",
            "event_security_identity_ready",
            "assigned_ticker",
            "assigned_exchange_mic",
        ],
    ].drop_duplicates("validation_event_id")
    columns = [
        "validation_event_id",
        "cik",
        "accession_number",
        "accepted_at",
        "form",
        "company_name_snapshot",
        "collection_wave",
        "evidence_role",
        "exact_outcome_previously_observed",
        "return_outcome_read_authorized",
    ]
    queue = (
        candidate_registry.loc[:, columns]
        .merge(filing, on="validation_event_id", how="left", validate="one_to_one")
        .merge(identity, on="validation_event_id", how="left", validate="one_to_one")
    )
    queue["accepted_at"] = _utc(queue["accepted_at"], "accepted_at")
    queue["filing_complete_case_ready"] = queue["feature_status"].eq("complete_case_ready") & queue[
        "gross_margin_applicability_complete_case"
    ].fillna(False).astype(bool)
    queue["strictly_after_training_label_end"] = queue["accepted_at"].gt(training_label_end)
    queue["outcome_materialized_locally"] = (
        queue["validation_event_id"].astype(str).isin(outcome_materialized_event_ids)
    )
    unobserved = queue["evidence_role"].eq("unobserved_validation_candidate")
    not_previously_observed = ~queue["exact_outcome_previously_observed"].astype(bool)
    not_authorized = ~queue["return_outcome_read_authorized"].astype(bool)
    temporal_complete = (
        unobserved
        & not_previously_observed
        & not_authorized
        & queue["filing_complete_case_ready"]
        & queue["strictly_after_training_label_end"]
    )
    queue = queue.loc[temporal_complete].copy()
    queue["forward_queue_eligible"] = ~queue["outcome_materialized_locally"]
    identity_ready = queue["event_security_identity_ready"].fillna(False).astype(bool)
    queue["queue_status"] = np.select(
        [
            queue["outcome_materialized_locally"],
            ~identity_ready,
        ],
        [
            "quarantined_local_outcome_exists",
            "awaiting_security_identity",
        ],
        default="awaiting_outcome_blind_market_reaction_features",
    )
    queue["model_training_label_end"] = training_label_end
    queue["outcome_values_loaded_for_queue"] = False
    return queue.sort_values(
        ["accepted_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _queue_funnel(
    candidate_registry: pd.DataFrame,
    filing_features: pd.DataFrame,
    queue: pd.DataFrame,
    scored_events: int,
) -> pd.DataFrame:
    unobserved = candidate_registry["evidence_role"].eq("unobserved_validation_candidate")
    filing_unobserved = filing_features["evidence_role"].eq("unobserved_validation_candidate")
    complete = filing_unobserved & filing_features["feature_status"].eq("complete_case_ready")
    clean = queue["forward_queue_eligible"].astype(bool)
    identity = clean & queue["event_security_identity_ready"].fillna(False).astype(bool)
    return pd.DataFrame(
        [
            {"stage_order": 1, "stage": "registered_unobserved", "events": int(unobserved.sum())},
            {
                "stage_order": 2,
                "stage": "filing_complete_case",
                "events": int(complete.sum()),
            },
            {"stage_order": 3, "stage": "post_training_temporal_queue", "events": len(queue)},
            {"stage_order": 4, "stage": "outcome_firewall_clean", "events": int(clean.sum())},
            {"stage_order": 5, "stage": "security_identity_ready", "events": int(identity.sum())},
            {"stage_order": 6, "stage": "reaction_feature_ready", "events": scored_events},
            {"stage_order": 7, "stage": "scored_before_outcome", "events": scored_events},
        ]
    )


def _model_parameter_table(models: tuple[FrozenLinearModel, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        for order, feature in enumerate(model.features, start=1):
            rows.append(
                {
                    "model_id": model.model_id,
                    "feature_set": model.feature_set,
                    "feature_order": order,
                    "feature": feature,
                    "scaler_mean": model.scaler_mean[order - 1],
                    "scaler_scale": model.scaler_scale[order - 1],
                    "training_feature_min": model.training_feature_min[order - 1],
                    "training_feature_max": model.training_feature_max[order - 1],
                    "ridge_coefficient": model.coefficients[order - 1],
                    "ridge_intercept": model.intercept,
                    "ridge_alpha": model.ridge_alpha,
                    "training_events": model.training_events,
                    "training_issuers": model.training_issuers,
                    "training_label_start": model.training_label_start,
                    "training_label_end": model.training_label_end,
                }
            )
    return pd.DataFrame(rows)


def _empty_scored_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "validation_event_id",
            "cik",
            "ticker",
            "accepted_at",
            "reaction_feature_available_at",
            "execution_entry_at",
            "evidence_role",
            "F1_v1_challenger_research_score",
            "F1_v1_challenger_support_clipped_score",
            "F1_v1_challenger_max_abs_standardized_feature",
            "F1_v1_challenger_outside_training_range_features",
            "F2_alignment_candidate_research_score",
            "F2_alignment_candidate_support_clipped_score",
            "F2_alignment_candidate_max_abs_standardized_feature",
            "F2_alignment_candidate_outside_training_range_features",
            "alignment_minus_challenger_score",
            "alignment_candidate_within_training_support",
            "alignment_candidate_severe_extrapolation",
            "prediction_generated_at",
            "model_training_label_end",
            "outcome_columns_loaded",
            "research_only",
        ]
    )


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} is missing columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise FilingCatalogError(f"{label} contains invalid timestamps")
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
