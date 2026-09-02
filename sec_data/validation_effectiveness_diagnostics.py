"""Independent validation diagnostics for the formal 20-session event ledger."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.preliminary_effectiveness_diagnostics import (
    FEATURES,
    EffectivenessDiagnosticConfig,
    calculate_primary_effectiveness_tables,
)

VALIDATION_EFFECTIVENESS_VERSION: Final[str] = "validation-effectiveness-diagnostic-v1"


class ValidationEffectivenessError(FilingCatalogError):
    """Raised when formal validation inference violates its frozen contract."""


@dataclass(frozen=True)
class ValidationEffectivenessTables:
    """Pure same-sample validation outputs."""

    analysis_sample: pd.DataFrame
    representation_ic: pd.DataFrame
    incremental_test: pd.DataFrame
    leave_one_issuer_out: pd.DataFrame
    issuer_influence: pd.DataFrame
    quantile_returns: pd.DataFrame
    robustness: pd.DataFrame
    outcome_summary: pd.DataFrame
    verdict_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationEffectivenessResult:
    """Immutable validation diagnostic bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_effectiveness_diagnostics(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    config: EffectivenessDiagnosticConfig | None = None,
) -> ValidationEffectivenessTables:
    """Join the frozen feature and return event sets and run primary inference."""
    cfg = config or EffectivenessDiagnosticConfig()
    cfg.validate()
    _validate_tables(returns, features)
    feature_columns = [
        "validation_event_id",
        "cik",
        "target_accession_number",
        "accepted_at",
        "actual_available_at",
        "gross_margin_history_eligible",
        "gross_margin_applicability_complete_case",
        "feature_status",
        *FEATURES,
    ]
    joined = returns.merge(
        features[feature_columns].rename(
            columns={
                "cik": "feature_cik",
                "accepted_at": "feature_accepted_at",
                "target_accession_number": "feature_accession_number",
            }
        ),
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    joined["cik"] = joined["cik"].astype(str).str.zfill(10)
    joined["feature_cik"] = joined["feature_cik"].astype(str).str.zfill(10)
    accepted = pd.to_datetime(joined["accepted_at"], utc=True, errors="coerce")
    feature_accepted = pd.to_datetime(joined["feature_accepted_at"], utc=True, errors="coerce")
    available = pd.to_datetime(joined["actual_available_at"], utc=True, errors="coerce")
    joined["issuer_match"] = joined["cik"].eq(joined["feature_cik"])
    joined["accession_match"] = joined["accession_number"].eq(joined["feature_accession_number"])
    joined["availability_clock_match"] = accepted.eq(feature_accepted) & accepted.eq(available)
    joined["same_complete_case_ready"] = (
        joined["issuer_match"]
        & joined["accession_match"]
        & joined["availability_clock_match"]
        & joined["gross_margin_history_eligible"].fillna(False).astype(bool)
        & joined["gross_margin_applicability_complete_case"].fillna(False).astype(bool)
        & joined["feature_status"].eq("complete_case_ready")
        & joined.loc[:, list(FEATURES)].notna().all(axis="columns")
        & joined["market_excess_return_ready"].astype(bool)
    )
    if not joined["same_complete_case_ready"].all():
        counts = joined.loc[~joined["same_complete_case_ready"], "feature_status"].value_counts()
        raise ValidationEffectivenessError(
            f"The authorized sample is not one complete case: {counts.to_dict()}"
        )
    if len(joined) < cfg.minimum_complete_events:
        raise ValidationEffectivenessError("Validation sample is below the frozen minimum")
    analysis = joined.sort_values(
        ["accepted_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    primary = calculate_primary_effectiveness_tables(analysis, cfg)
    robustness = _robustness(analysis)
    outcome_summary = _outcome_summary(analysis)
    verdict, checks = _verdict(
        analysis,
        primary.representation_ic,
        primary.incremental_test,
        primary.leave_one_issuer_out,
        primary.issuer_influence,
        primary.quantile_returns,
        cfg,
    )
    innovation = primary.representation_ic.loc[
        primary.representation_ic["feature"].eq("gm_innovation_std")
    ].iloc[0]
    delta = primary.representation_ic.loc[
        primary.representation_ic["feature"].eq("gm_delta_yoy")
    ].iloc[0]
    contrast = primary.incremental_test.loc[
        primary.incremental_test["statistic"].eq("innovation_minus_delta_spearman_ic")
    ].iloc[0]
    report: dict[str, Any] = {
        "contract_version": VALIDATION_EFFECTIVENESS_VERSION,
        "events": len(analysis),
        "issuers": int(analysis["cik"].nunique()),
        "complete_case_coverage": float(analysis["same_complete_case_ready"].mean()),
        "innovation_ic": float(innovation["spearman_ic"]),
        "innovation_ic_ci_low": float(innovation["ci_low"]),
        "innovation_ic_ci_high": float(innovation["ci_high"]),
        "delta_ic": float(delta["spearman_ic"]),
        "innovation_minus_delta_ic": float(contrast["point_estimate"]),
        "innovation_minus_delta_ci_low": float(contrast["ci_low"]),
        "innovation_minus_delta_ci_high": float(contrast["ci_high"]),
        "innovation_minus_delta_bootstrap_p": float(contrast["bootstrap_p_value"]),
        "leave_one_issuer_out_positive_share": float(
            (primary.leave_one_issuer_out["innovation_ic"] > 0.0).mean()
        ),
        "maximum_single_issuer_effect_share": float(
            primary.issuer_influence["absolute_effect_share"].max()
        ),
        "verdict": verdict,
        "verdict_checks_passed": int(checks["passed"].sum()),
        "verdict_checks_total": len(checks),
        "hypothesis_and_inference_settings_preregistered": True,
        "validation_event_set_preregistered": False,
        "event_set_frozen_before_outcome_read": True,
        "same_complete_case_sample_required": True,
        "sector_benchmark_in_scope": False,
        "annualized_performance_computed": False,
        "strategy_pnl_computed": False,
        "investment_recommendation_allowed": False,
        "config": asdict(cfg),
    }
    return ValidationEffectivenessTables(
        analysis,
        primary.representation_ic,
        primary.incremental_test,
        primary.leave_one_issuer_out,
        primary.issuer_influence,
        primary.quantile_returns,
        robustness,
        outcome_summary,
        checks,
        report,
    )


def build_validation_effectiveness_diagnostics(
    return_ledger_root: Path,
    feature_root: Path,
    prior_preregistration_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_effectiveness_diagnostics"),
    config: EffectivenessDiagnosticConfig | None = None,
) -> ValidationEffectivenessResult:
    """Persist validation inference tied to returns, features, and prior hypotheses."""
    cfg = config or EffectivenessDiagnosticConfig()
    cfg.validate()
    manifests = {
        "return_ledger_manifest_sha256": _sha256_file(return_ledger_root / "output_manifest.json"),
        "feature_manifest_sha256": _sha256_file(feature_root / "output_manifest.json"),
        "prior_preregistration_manifest_sha256": _sha256_file(
            prior_preregistration_root / "output_manifest.json"
        ),
    }
    return_report = _read_json(return_ledger_root / "validation_event_return_report.json")
    feature_report = _read_json(feature_root / "validation_feature_report.json")
    prereg = _read_json(prior_preregistration_root / "preregistration_report.json")
    if not bool(return_report.get("formal_return_gate")) or int(
        return_report.get("market_excess_returns_ready", -1)
    ) != int(return_report.get("authorized_event_components", -2)):
        raise ValidationEffectivenessError("The formal return ledger is incomplete")
    if bool(feature_report.get("market_returns_loaded")) or bool(
        feature_report.get("outcome_values_loaded")
    ):
        raise ValidationEffectivenessError("Validation features were not frozen outcome-blind")
    _validate_prior_hypothesis(prereg, cfg)
    tables = calculate_validation_effectiveness_diagnostics(
        load_catalog_table(return_ledger_root, "validation_event_return_ledger.parquet"),
        load_catalog_table(feature_root, "target_filing_features.parquet"),
        cfg,
    )
    payload = json.dumps(
        {"version": VALIDATION_EFFECTIVENESS_VERSION, **manifests, "config": asdict(cfg)},
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_EFFECTIVENESS_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationEffectivenessResult(
            target.resolve(), _read_json(target / "validation_effectiveness_report.json")
        )
    if target.exists():
        raise ValidationEffectivenessError(f"Incomplete validation diagnostic exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_analysis_sample.parquet": tables.analysis_sample,
        "validation_representation_ic.parquet": tables.representation_ic,
        "validation_incremental_test.parquet": tables.incremental_test,
        "validation_leave_one_issuer_out.parquet": tables.leave_one_issuer_out,
        "validation_issuer_influence.parquet": tables.issuer_influence,
        "validation_quantile_returns.parquet": tables.quantile_returns,
        "validation_robustness.parquet": tables.robustness,
        "validation_outcome_summary.parquet": tables.outcome_summary,
        "validation_verdict_checks.parquet": tables.verdict_checks,
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
        **manifests,
        "prior_preregistration_version": prereg["dataset_version"],
        "artifacts": artifacts,
    }
    report_path = staging / "validation_effectiveness_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "dataset_version": dataset_version,
        **manifests,
        "artifacts": artifacts,
        "report_sha256": _sha256_file(report_path),
    }
    (staging / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    staging.replace(target)
    return ValidationEffectivenessResult(target.resolve(), report)


def _robustness(frame: pd.DataFrame) -> pd.DataFrame:
    specifications: list[tuple[str, pd.DataFrame]] = [
        ("full_primary", frame),
        ("10-Q_only", frame[frame["form"].eq("10-Q")]),
        ("10-K_only", frame[frame["form"].isin(["10-K", "20-F"])]),
        (
            "survivors_only_biased",
            frame[~frame["later_delisted_after_formal_exit"].astype(bool)],
        ),
        (
            "later_delisted_retained",
            frame[frame["later_delisted_after_formal_exit"].astype(bool)],
        ),
    ]
    quarter = (
        pd.to_datetime(frame["accepted_at"], utc=True)
        .dt.tz_convert(None)
        .dt.to_period("Q")
        .astype(str)
    )
    for value in sorted(quarter.unique()):
        sample = frame[quarter.eq(value)]
        if len(sample) >= 8:
            specifications.append((f"calendar_quarter_{value}", sample))
    midpoint = len(frame) // 2
    specifications.extend(
        [
            ("chronological_first_half", frame.iloc[:midpoint]),
            ("chronological_second_half", frame.iloc[midpoint:]),
        ]
    )
    rows: list[dict[str, Any]] = []
    for diagnostic_id, sample in specifications:
        innovation = _spearman(
            sample["gm_innovation_std"], sample["market_excess_return_arithmetic"]
        )
        delta = _spearman(sample["gm_delta_yoy"], sample["market_excess_return_arithmetic"])
        rows.append(
            {
                "diagnostic_id": diagnostic_id,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()) if not sample.empty else 0,
                "innovation_ic": innovation,
                "delta_ic": delta,
                "innovation_minus_delta_ic": innovation - delta,
            }
        )
    return pd.DataFrame(rows)


def _outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("all_events", frame),
        ("survivors_only_biased", frame[~frame["later_delisted_after_formal_exit"]]),
        ("later_delisted_retained", frame[frame["later_delisted_after_formal_exit"]]),
    ]
    for sample_id, sample in groups:
        outcome = sample["market_excess_return_arithmetic"]
        rows.append(
            {
                "sample_id": sample_id,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()) if not sample.empty else 0,
                "mean_market_excess_return": float(outcome.mean()) if len(sample) else np.nan,
                "median_market_excess_return": float(outcome.median()) if len(sample) else np.nan,
                "positive_market_excess_rate": float((outcome > 0.0).mean())
                if len(sample)
                else np.nan,
                "minimum_market_excess_return": float(outcome.min()) if len(sample) else np.nan,
                "maximum_market_excess_return": float(outcome.max()) if len(sample) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _verdict(
    analysis: pd.DataFrame,
    ic: pd.DataFrame,
    incremental: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    influence: pd.DataFrame,
    quantiles: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> tuple[str, pd.DataFrame]:
    innovation = ic[ic["feature"].eq("gm_innovation_std")].iloc[0]
    contrast = incremental[incremental["statistic"].eq("innovation_minus_delta_spearman_ic")].iloc[
        0
    ]
    beta = incremental[
        incremental["statistic"].eq("incremental_innovation_rank_beta_given_delta")
    ].iloc[0]
    innovation_quantiles = quantiles[quantiles["feature"].eq("gm_innovation_std")].set_index(
        "quantile"
    )
    means = innovation_quantiles["mean_market_excess_return"]
    monotonic = bool(means["Low"] < means["Middle"] < means["High"])
    loo_share = float((leave_one_out["innovation_ic"] > 0.0).mean())
    max_share = float(influence["absolute_effect_share"].max())
    checks = pd.DataFrame(
        [
            _check(
                "data_quality_and_coverage",
                len(analysis) >= config.minimum_complete_events,
                len(analysis),
                config.minimum_complete_events,
            ),
            _check(
                "innovation_ic_positive_and_ci_excludes_zero",
                bool(innovation["spearman_ic"] > 0.0 and innovation["ci_low"] > 0.0),
                float(innovation["ci_low"]),
                0.0,
            ),
            _check(
                "innovation_minus_delta_ci_excludes_zero",
                bool(contrast["ci_low"] > 0.0),
                float(contrast["ci_low"]),
                0.0,
            ),
            _check(
                "incremental_innovation_beta_positive",
                bool(beta["point_estimate"] > 0.0),
                float(beta["point_estimate"]),
                0.0,
            ),
            _check(
                "innovation_quantile_means_monotonic",
                monotonic,
                float(means["High"] - means["Low"]),
                0.0,
            ),
            _check(
                "leave_one_issuer_out_positive_sign_share",
                loo_share >= config.minimum_leave_one_issuer_out_sign_share,
                loo_share,
                config.minimum_leave_one_issuer_out_sign_share,
            ),
            _check(
                "maximum_single_issuer_absolute_effect_share",
                max_share <= config.maximum_single_issuer_effect_share,
                max_share,
                config.maximum_single_issuer_effect_share,
            ),
        ]
    )
    if not bool(
        checks.loc[
            checks["check_id"].eq("innovation_ic_positive_and_ci_excludes_zero"), "passed"
        ].iloc[0]
    ):
        verdict = "effectiveness_not_established"
    elif not bool(
        checks.loc[
            checks["check_id"].isin(
                ["innovation_minus_delta_ci_excludes_zero", "incremental_innovation_beta_positive"]
            ),
            "passed",
        ].all()
    ):
        verdict = "innovation_not_distinct_from_delta"
    elif not bool(checks.iloc[4:]["passed"].all()):
        verdict = "issuer_concentrated_or_economic_ordering_fragile"
    else:
        verdict = "preliminary_supportive_validation_evidence"
    return verdict, checks


def _validate_tables(returns: pd.DataFrame, features: pd.DataFrame) -> None:
    return_required = {
        "security_event_component_id",
        "validation_event_id",
        "cik",
        "accepted_at",
        "form",
        "accession_number",
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "market_excess_return_ready",
        "later_delisted_after_formal_exit",
        "return_outcome_read_authorized",
        "selection_uses_return_values",
        "returns_computed",
    }
    feature_required = {
        "validation_event_id",
        "cik",
        "target_accession_number",
        "accepted_at",
        "actual_available_at",
        "gross_margin_history_eligible",
        "gross_margin_applicability_complete_case",
        "feature_status",
        *FEATURES,
    }
    _require_columns(returns, return_required, "formal return ledger")
    _require_columns(features, feature_required, "validation features")
    if returns.empty or returns["validation_event_id"].duplicated().any():
        raise ValidationEffectivenessError("Formal validation returns are empty or duplicated")
    if (
        not returns["return_outcome_read_authorized"].astype(bool).all()
        or not returns["returns_computed"].astype(bool).all()
    ):
        raise ValidationEffectivenessError("Formal return rows are not authorized and computed")
    if returns["selection_uses_return_values"].astype(bool).any():
        raise ValidationEffectivenessError("Validation sample selection used return values")
    selected_features = features[
        features["validation_event_id"].isin(returns["validation_event_id"])
    ]
    if (
        len(selected_features) != len(returns)
        or selected_features["validation_event_id"].duplicated().any()
    ):
        raise ValidationEffectivenessError(
            "Validation feature rows do not match returns one-to-one"
        )


def _validate_prior_hypothesis(
    report: dict[str, Any], config: EffectivenessDiagnosticConfig
) -> None:
    prior = report.get("config", {})
    expected = {
        "primary_horizon_sessions": config.primary_horizon_sessions,
        "bootstrap_replications": config.bootstrap_replications,
        "bootstrap_seed": config.bootstrap_seed,
        "confidence_level": config.confidence_level,
        "minimum_leave_one_issuer_out_sign_share": config.minimum_leave_one_issuer_out_sign_share,
        "maximum_single_issuer_effect_share": config.maximum_single_issuer_effect_share,
    }
    for key, value in expected.items():
        if prior.get(key) != value:
            raise ValidationEffectivenessError(f"Prior hypothesis setting changed: {key}")
    if report.get("primary_contrast") != "IC(gm_innovation_std,R20)-IC(gm_delta_yoy,R20)":
        raise ValidationEffectivenessError("Prior primary contrast does not match validation")


def _check(
    check_id: str, passed: bool, observed: int | float, threshold: int | float
) -> dict[str, Any]:
    return {"check_id": check_id, "passed": passed, "observed": observed, "threshold": threshold}


def _spearman(left: pd.Series[Any], right: pd.Series[Any]) -> float:
    if len(left) < 3:
        return math.nan
    value = left.rank(method="average").corr(right.rank(method="average"))
    return float(value) if pd.notna(value) else math.nan


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationEffectivenessError(f"{label} lacks required columns: {missing}")


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
