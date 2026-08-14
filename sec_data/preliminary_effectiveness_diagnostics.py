"""Pre-registered same-sample diagnostics for longitudinal filing signals."""

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

EFFECTIVENESS_DIAGNOSTIC_VERSION: Final[str] = "preliminary-effectiveness-diagnostic-v1"
FEATURES: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
)


@dataclass(frozen=True)
class EffectivenessDiagnosticConfig:
    """Frozen inference and robustness settings copied from the pre-registration."""

    primary_horizon_sessions: int = 20
    secondary_horizons_sessions: tuple[int, ...] = (1, 5)
    minimum_complete_events: int = 47
    bootstrap_replications: int = 10_000
    bootstrap_seed: int = 20260814
    confidence_level: float = 0.95
    minimum_leave_one_issuer_out_sign_share: float = 0.80
    maximum_single_issuer_effect_share: float = 0.25
    version: str = EFFECTIVENESS_DIAGNOSTIC_VERSION

    def validate(self) -> None:
        """Reject thresholds that diverge from the immutable analysis contract."""
        if self.primary_horizon_sessions != 20:
            raise ValueError("The primary horizon is frozen at 20 sessions")
        if tuple(self.secondary_horizons_sessions) != (1, 5):
            raise ValueError("Secondary horizons are frozen at one and five sessions")
        if self.minimum_complete_events < 20:
            raise ValueError("The combined feature-return sample cannot be below 20")
        if self.bootstrap_replications < 1_000:
            raise ValueError("At least 1,000 issuer-cluster bootstrap draws are required")
        if not 0.90 <= self.confidence_level < 1.0:
            raise ValueError("Confidence level must be in [0.90, 1.0)")


@dataclass(frozen=True)
class EffectivenessDiagnosticTables:
    """Pure diagnostic outputs before immutable persistence."""

    analysis_sample: pd.DataFrame
    representation_ic: pd.DataFrame
    incremental_test: pd.DataFrame
    robustness: pd.DataFrame
    leave_one_issuer_out: pd.DataFrame
    issuer_influence: pd.DataFrame
    quantile_returns: pd.DataFrame
    survivorship_diagnostic: pd.DataFrame
    secondary_multiplicity: pd.DataFrame
    verdict_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class EffectivenessDiagnosticResult:
    """Immutable diagnostic bundle for report generation."""

    output_root: Path
    report: dict[str, Any]


def calculate_effectiveness_diagnostics(
    ledger: pd.DataFrame,
    config: EffectivenessDiagnosticConfig | None = None,
) -> EffectivenessDiagnosticTables:
    """Run the frozen same-sample test, or emit a transparent blocked verdict."""
    cfg = config or EffectivenessDiagnosticConfig()
    cfg.validate()
    required = {
        "longitudinal_event_id",
        "cik",
        "accepted_at",
        "form",
        "horizon_sessions",
        "analysis_sample",
        "market_excess_return_ready",
        "market_excess_return_arithmetic",
        "later_delisted_after_formal_exit",
        *FEATURES,
    }
    _require_columns(ledger, required, "longitudinal return ledger")
    if ledger.duplicated(["longitudinal_event_id", "horizon_sessions"]).any():
        raise FilingCatalogError("Longitudinal return rows are duplicated")
    primary_candidates = ledger.loc[
        ledger["horizon_sessions"].eq(cfg.primary_horizon_sessions)
        & ledger["analysis_sample"].astype(bool)
    ].copy()
    analysis = primary_candidates.loc[
        primary_candidates["market_excess_return_ready"].astype(bool)
    ].copy()
    complete_columns = [*FEATURES, "market_excess_return_arithmetic"]
    analysis = analysis.dropna(subset=complete_columns)
    analysis = analysis.sort_values(
        ["accepted_at", "cik", "longitudinal_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    gate = len(analysis) >= cfg.minimum_complete_events
    empty_ic = pd.DataFrame(columns=["feature", "spearman_ic", "ci_low", "ci_high"])
    empty_incremental = pd.DataFrame(columns=["statistic", "point_estimate", "ci_low", "ci_high"])
    empty_robustness = pd.DataFrame(
        columns=[
            "diagnostic_id",
            "horizon_sessions",
            "events",
            "issuers",
            "innovation_ic",
            "delta_ic",
            "innovation_minus_delta_ic",
        ]
    )
    empty_loo = pd.DataFrame(
        columns=["excluded_cik", "events", "innovation_ic", "innovation_minus_delta_ic"]
    )
    empty_influence = pd.DataFrame(
        columns=["cik", "events", "rank_covariance_contribution", "absolute_effect_share"]
    )
    empty_quantiles = pd.DataFrame(
        columns=["feature", "quantile", "events", "mean_market_excess_return"]
    )
    survivorship = _survivorship_diagnostic(analysis)
    empty_secondary = pd.DataFrame(
        columns=[
            "horizon_sessions",
            "innovation_minus_delta_ic",
            "bootstrap_p_value",
            "bh_adjusted_p_value",
        ]
    )
    if not gate:
        verdict = "insufficient_data_quality_or_coverage"
        checks = _blocked_verdict_checks(len(analysis), cfg.minimum_complete_events)
        report = _report(
            cfg,
            primary_candidates,
            analysis,
            verdict,
            checks,
            empty_ic,
            empty_incremental,
            empty_loo,
            empty_influence,
            analysis_gate=False,
        )
        return EffectivenessDiagnosticTables(
            analysis,
            empty_ic,
            empty_incremental,
            empty_robustness,
            empty_loo,
            empty_influence,
            empty_quantiles,
            survivorship,
            empty_secondary,
            checks,
            report,
        )

    bootstrap = _cluster_bootstrap(analysis, cfg)
    representation_ic = _representation_ic(analysis, bootstrap, cfg)
    incremental = _incremental_test(analysis, bootstrap, cfg)
    loo = _leave_one_issuer_out(analysis)
    influence = _issuer_influence(analysis)
    quantiles = _quantile_returns(analysis)
    robustness = _robustness_table(ledger, analysis, cfg)
    secondary = _secondary_multiplicity(ledger, analysis, cfg)
    verdict, checks = _verdict(
        analysis,
        representation_ic,
        incremental,
        loo,
        influence,
        robustness,
        cfg,
    )
    report = _report(
        cfg,
        primary_candidates,
        analysis,
        verdict,
        checks,
        representation_ic,
        incremental,
        loo,
        influence,
        analysis_gate=True,
    )
    return EffectivenessDiagnosticTables(
        analysis,
        representation_ic,
        incremental,
        robustness,
        loo,
        influence,
        quantiles,
        survivorship,
        secondary,
        checks,
        report,
    )


def build_effectiveness_diagnostics(
    ledger_root: Path,
    preregistration_root: Path,
    *,
    output_root: Path = Path("data/processed/preliminary_effectiveness_diagnostics"),
    config: EffectivenessDiagnosticConfig | None = None,
) -> EffectivenessDiagnosticResult:
    """Build immutable diagnostics tied to the ledger and pre-registration hashes."""
    cfg = config or EffectivenessDiagnosticConfig()
    cfg.validate()
    manifests = {
        "ledger_manifest_sha256": _sha256_file(ledger_root / "output_manifest.json"),
        "preregistration_manifest_sha256": _sha256_file(
            preregistration_root / "output_manifest.json"
        ),
    }
    prereg = _read_json(preregistration_root / "preregistration_report.json")
    _validate_preregistration(prereg, cfg)
    ledger = load_catalog_table(ledger_root, "longitudinal_event_return_ledger.parquet")
    tables = calculate_effectiveness_diagnostics(ledger, cfg)
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return EffectivenessDiagnosticResult(
            target.resolve(), _read_json(target / "effectiveness_diagnostic_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete effectiveness diagnostic exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "analysis_sample.parquet": tables.analysis_sample,
        "representation_ic.parquet": tables.representation_ic,
        "incremental_test.parquet": tables.incremental_test,
        "robustness_diagnostics.parquet": tables.robustness,
        "leave_one_issuer_out.parquet": tables.leave_one_issuer_out,
        "issuer_influence.parquet": tables.issuer_influence,
        "quantile_returns.parquet": tables.quantile_returns,
        "survivorship_diagnostic.parquet": tables.survivorship_diagnostic,
        "secondary_multiplicity.parquet": tables.secondary_multiplicity,
        "verdict_checks.parquet": tables.verdict_checks,
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
        "preregistration_version": prereg["dataset_version"],
        "artifacts": artifacts,
    }
    report_path = staging / "effectiveness_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
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
    return EffectivenessDiagnosticResult(target.resolve(), report)


def _cluster_bootstrap(frame: pd.DataFrame, config: EffectivenessDiagnosticConfig) -> pd.DataFrame:
    issuers = sorted(frame["cik"].astype(str).unique())
    if len(issuers) < 3:
        raise FilingCatalogError("Issuer-cluster bootstrap requires at least three issuers")
    clusters = {issuer: frame.loc[frame["cik"].eq(issuer)].copy() for issuer in issuers}
    rng = np.random.default_rng(config.bootstrap_seed)
    rows: list[dict[str, float]] = []
    for _ in range(config.bootstrap_replications):
        sampled = rng.choice(issuers, size=len(issuers), replace=True)
        boot = pd.concat([clusters[str(issuer)] for issuer in sampled], ignore_index=True)
        metrics = _core_metrics(boot)
        if all(math.isfinite(value) for value in metrics.values()):
            rows.append(metrics)
    if len(rows) < int(config.bootstrap_replications * 0.90):
        raise FilingCatalogError("Too many invalid issuer-cluster bootstrap draws")
    return pd.DataFrame(rows)


def _core_metrics(frame: pd.DataFrame) -> dict[str, float]:
    outcome = frame["market_excess_return_arithmetic"]
    values = {f"ic_{feature}": _spearman(frame[feature], outcome) for feature in FEATURES}
    values["innovation_minus_delta_ic"] = values["ic_gm_innovation_std"] - values["ic_gm_delta_yoy"]
    values["incremental_innovation_beta"] = _rank_regression_beta(frame)
    return values


def _representation_ic(
    frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> pd.DataFrame:
    alpha = (1.0 - config.confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        column = f"ic_{feature}"
        rows.append(
            {
                "feature": feature,
                "events": len(frame),
                "issuers": int(frame["cik"].nunique()),
                "spearman_ic": _spearman(frame[feature], frame["market_excess_return_arithmetic"]),
                "ci_low": float(bootstrap[column].quantile(alpha)),
                "ci_high": float(bootstrap[column].quantile(1.0 - alpha)),
                "bootstrap_replications_valid": len(bootstrap),
            }
        )
    return pd.DataFrame(rows)


def _incremental_test(
    frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> pd.DataFrame:
    alpha = (1.0 - config.confidence_level) / 2.0
    delta = bootstrap["innovation_minus_delta_ic"]
    beta = bootstrap["incremental_innovation_beta"]
    return pd.DataFrame(
        [
            {
                "statistic": "innovation_minus_delta_spearman_ic",
                "point_estimate": _core_metrics(frame)["innovation_minus_delta_ic"],
                "ci_low": float(delta.quantile(alpha)),
                "ci_high": float(delta.quantile(1.0 - alpha)),
                "bootstrap_p_value": _bootstrap_two_sided_p(delta),
            },
            {
                "statistic": "incremental_innovation_rank_beta_given_delta",
                "point_estimate": _rank_regression_beta(frame),
                "ci_low": float(beta.quantile(alpha)),
                "ci_high": float(beta.quantile(1.0 - alpha)),
                "bootstrap_p_value": _bootstrap_two_sided_p(beta),
            },
        ]
    )


def _robustness_table(
    ledger: pd.DataFrame,
    primary_analysis: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> pd.DataFrame:
    event_ids = set(primary_analysis["longitudinal_event_id"])
    specifications: list[tuple[str, int, pd.DataFrame]] = [
        ("full_primary", 20, primary_analysis),
        (
            "exclude_ASNS_tail_case",
            20,
            primary_analysis.loc[~primary_analysis["cik"].eq("0001141284")],
        ),
        ("10-Q_only", 20, primary_analysis.loc[primary_analysis["form"].eq("10-Q")]),
    ]
    for horizon in config.secondary_horizons_sessions:
        sample = ledger.loc[
            ledger["horizon_sessions"].eq(horizon)
            & ledger["longitudinal_event_id"].isin(event_ids)
            & ledger["market_excess_return_ready"].astype(bool)
        ].dropna(subset=[*FEATURES, "market_excess_return_arithmetic"])
        specifications.append((f"horizon_{horizon}", horizon, sample))
    primary_with_quarter = primary_analysis.copy()
    primary_with_quarter["calendar_quarter"] = (
        pd.to_datetime(primary_with_quarter["accepted_at"], utc=True)
        .dt.tz_convert(None)
        .dt.to_period("Q")
        .astype(str)
    )
    for quarter, sample in primary_with_quarter.groupby("calendar_quarter", sort=True):
        if len(sample) >= 4:
            specifications.append((f"calendar_quarter_{quarter}", 20, sample))
    rows: list[dict[str, Any]] = []
    for diagnostic_id, horizon, sample in specifications:
        metrics = _core_metrics(sample) if len(sample) >= 3 else _nan_metrics()
        rows.append(
            {
                "diagnostic_id": diagnostic_id,
                "horizon_sessions": horizon,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()) if not sample.empty else 0,
                "innovation_ic": metrics["ic_gm_innovation_std"],
                "delta_ic": metrics["ic_gm_delta_yoy"],
                "innovation_minus_delta_ic": metrics["innovation_minus_delta_ic"],
            }
        )
    return pd.DataFrame(rows)


def _leave_one_issuer_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for issuer in sorted(frame["cik"].astype(str).unique()):
        sample = frame.loc[~frame["cik"].eq(issuer)]
        metrics = _core_metrics(sample)
        rows.append(
            {
                "excluded_cik": issuer,
                "events": len(sample),
                "innovation_ic": metrics["ic_gm_innovation_std"],
                "innovation_minus_delta_ic": metrics["innovation_minus_delta_ic"],
            }
        )
    return pd.DataFrame(rows)


def _issuer_influence(frame: pd.DataFrame) -> pd.DataFrame:
    ranked_x = frame["gm_innovation_std"].rank(method="average")
    ranked_y = frame["market_excess_return_arithmetic"].rank(method="average")
    contribution = (ranked_x - ranked_x.mean()) * (ranked_y - ranked_y.mean())
    working = frame[["cik"]].copy()
    working["rank_covariance_contribution"] = contribution
    grouped = (
        working.groupby("cik", sort=True)
        .agg(
            events=("cik", "size"),
            rank_covariance_contribution=("rank_covariance_contribution", "sum"),
        )
        .reset_index()
    )
    denominator = float(grouped["rank_covariance_contribution"].abs().sum())
    grouped["absolute_effect_share"] = (
        grouped["rank_covariance_contribution"].abs() / denominator if denominator > 0.0 else np.nan
    )
    return grouped.sort_values("absolute_effect_share", ascending=False, kind="mergesort")


def _quantile_returns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        working = frame[[feature, "market_excess_return_arithmetic"]].copy()
        working["quantile"] = pd.qcut(
            working[feature].rank(method="first"),
            q=3,
            labels=["Low", "Middle", "High"],
        )
        grouped = working.groupby("quantile", observed=True, sort=False)
        for quantile, sample in grouped:
            rows.append(
                {
                    "feature": feature,
                    "quantile": str(quantile),
                    "events": len(sample),
                    "mean_market_excess_return": float(
                        sample["market_excess_return_arithmetic"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _survivorship_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, sample in (
        ("all_events", frame),
        (
            "survivors_only_biased",
            frame.loc[~frame["later_delisted_after_formal_exit"].astype(bool)],
        ),
        (
            "later_delisted_retained",
            frame.loc[frame["later_delisted_after_formal_exit"].astype(bool)],
        ),
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "events": len(sample),
                "issuers": int(sample["cik"].nunique()) if not sample.empty else 0,
                "mean_market_excess_return": (
                    float(sample["market_excess_return_arithmetic"].mean())
                    if not sample.empty
                    else np.nan
                ),
                "median_market_excess_return": (
                    float(sample["market_excess_return_arithmetic"].median())
                    if not sample.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _secondary_multiplicity(
    ledger: pd.DataFrame,
    primary_analysis: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> pd.DataFrame:
    event_ids = set(primary_analysis["longitudinal_event_id"])
    rows: list[dict[str, Any]] = []
    for offset, horizon in enumerate(config.secondary_horizons_sessions, start=1):
        sample = ledger.loc[
            ledger["horizon_sessions"].eq(horizon)
            & ledger["longitudinal_event_id"].isin(event_ids)
            & ledger["market_excess_return_ready"].astype(bool)
        ].dropna(subset=[*FEATURES, "market_excess_return_arithmetic"])
        secondary_config = EffectivenessDiagnosticConfig(
            primary_horizon_sessions=config.primary_horizon_sessions,
            secondary_horizons_sessions=config.secondary_horizons_sessions,
            minimum_complete_events=config.minimum_complete_events,
            bootstrap_replications=config.bootstrap_replications,
            bootstrap_seed=config.bootstrap_seed + offset,
            confidence_level=config.confidence_level,
            minimum_leave_one_issuer_out_sign_share=(
                config.minimum_leave_one_issuer_out_sign_share
            ),
            maximum_single_issuer_effect_share=config.maximum_single_issuer_effect_share,
        )
        bootstrap = _cluster_bootstrap(sample, secondary_config)
        distribution = bootstrap["innovation_minus_delta_ic"]
        rows.append(
            {
                "horizon_sessions": horizon,
                "innovation_minus_delta_ic": _core_metrics(sample)["innovation_minus_delta_ic"],
                "bootstrap_p_value": _bootstrap_two_sided_p(distribution),
            }
        )
    result = pd.DataFrame(rows)
    result["bh_adjusted_p_value"] = _benjamini_hochberg(result["bootstrap_p_value"])
    return result


def _verdict(
    analysis: pd.DataFrame,
    ic: pd.DataFrame,
    incremental: pd.DataFrame,
    loo: pd.DataFrame,
    influence: pd.DataFrame,
    robustness: pd.DataFrame,
    config: EffectivenessDiagnosticConfig,
) -> tuple[str, pd.DataFrame]:
    innovation = ic.loc[ic["feature"].eq("gm_innovation_std")].iloc[0]
    contrast = incremental.loc[
        incremental["statistic"].eq("innovation_minus_delta_spearman_ic")
    ].iloc[0]
    beta = incremental.loc[
        incremental["statistic"].eq("incremental_innovation_rank_beta_given_delta")
    ].iloc[0]
    loo_share = float((loo["innovation_ic"] > 0.0).mean())
    asns = robustness.loc[robustness["diagnostic_id"].eq("exclude_ASNS_tail_case")]
    asns_positive = bool(not asns.empty and float(asns.iloc[0]["innovation_ic"]) > 0.0)
    max_share = float(influence["absolute_effect_share"].max())
    checks = pd.DataFrame(
        [
            {
                "check_id": "data_quality_and_coverage",
                "passed": len(analysis) >= config.minimum_complete_events,
                "observed": len(analysis),
                "threshold": config.minimum_complete_events,
            },
            {
                "check_id": "innovation_ic_positive_and_ci_excludes_zero",
                "passed": bool(innovation["spearman_ic"] > 0.0 and innovation["ci_low"] > 0.0),
                "observed": float(innovation["ci_low"]),
                "threshold": 0.0,
            },
            {
                "check_id": "innovation_minus_delta_ci_excludes_zero",
                "passed": bool(contrast["ci_low"] > 0.0),
                "observed": float(contrast["ci_low"]),
                "threshold": 0.0,
            },
            {
                "check_id": "incremental_innovation_beta_positive",
                "passed": bool(beta["point_estimate"] > 0.0),
                "observed": float(beta["point_estimate"]),
                "threshold": 0.0,
            },
            {
                "check_id": "leave_one_issuer_out_positive_sign_share",
                "passed": loo_share >= config.minimum_leave_one_issuer_out_sign_share,
                "observed": loo_share,
                "threshold": config.minimum_leave_one_issuer_out_sign_share,
            },
            {
                "check_id": "exclude_ASNS_innovation_sign_positive",
                "passed": asns_positive,
                "observed": float(asns.iloc[0]["innovation_ic"]) if not asns.empty else np.nan,
                "threshold": 0.0,
            },
            {
                "check_id": "maximum_single_issuer_absolute_effect_share",
                "passed": max_share <= config.maximum_single_issuer_effect_share,
                "observed": max_share,
                "threshold": config.maximum_single_issuer_effect_share,
            },
        ]
    )
    if not bool(checks.iloc[1]["passed"]):
        verdict = "effectiveness_not_established"
    elif not bool(checks.iloc[2:4]["passed"].all()):
        verdict = "innovation_not_distinct_from_delta"
    elif not bool(checks.iloc[4:]["passed"].all()):
        verdict = "issuer_concentrated_or_tail_fragile"
    else:
        verdict = "preliminary_supportive_evidence"
    return verdict, checks


def _blocked_verdict_checks(observed: int, required: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_id": "data_quality_and_coverage",
                "passed": False,
                "observed": observed,
                "threshold": required,
            }
        ]
    )


def _report(
    config: EffectivenessDiagnosticConfig,
    candidates: pd.DataFrame,
    analysis: pd.DataFrame,
    verdict: str,
    checks: pd.DataFrame,
    ic: pd.DataFrame,
    incremental: pd.DataFrame,
    loo: pd.DataFrame,
    influence: pd.DataFrame,
    *,
    analysis_gate: bool,
) -> dict[str, Any]:
    innovation = ic.loc[ic["feature"].eq("gm_innovation_std")]
    contrast = incremental.loc[incremental["statistic"].eq("innovation_minus_delta_spearman_ic")]
    return {
        "contract_version": config.version,
        "primary_candidate_events": len(candidates),
        "primary_analysis_events": len(analysis),
        "primary_analysis_issuers": int(analysis["cik"].nunique()) if not analysis.empty else 0,
        "primary_analysis_gate": analysis_gate,
        "verdict": verdict,
        "verdict_checks_passed": int(checks["passed"].sum()),
        "verdict_checks_total": len(checks),
        "innovation_ic": float(innovation.iloc[0]["spearman_ic"]) if not innovation.empty else None,
        "innovation_ic_ci_low": float(innovation.iloc[0]["ci_low"])
        if not innovation.empty
        else None,
        "innovation_ic_ci_high": float(innovation.iloc[0]["ci_high"])
        if not innovation.empty
        else None,
        "innovation_minus_delta_ic": float(contrast.iloc[0]["point_estimate"])
        if not contrast.empty
        else None,
        "innovation_minus_delta_ci_low": float(contrast.iloc[0]["ci_low"])
        if not contrast.empty
        else None,
        "innovation_minus_delta_ci_high": float(contrast.iloc[0]["ci_high"])
        if not contrast.empty
        else None,
        "leave_one_issuer_out_positive_share": float((loo["innovation_ic"] > 0.0).mean())
        if not loo.empty
        else None,
        "maximum_single_issuer_effect_share": float(influence["absolute_effect_share"].max())
        if not influence.empty
        else None,
        "annualized_performance_computed": False,
        "strategy_pnl_computed": False,
        "investment_recommendation_allowed": False,
        "config": asdict(config),
    }


def _rank_regression_beta(frame: pd.DataFrame) -> float:
    ranked = frame[["gm_delta_yoy", "gm_innovation_std", "market_excess_return_arithmetic"]].rank(
        method="average"
    )
    standardized = (ranked - ranked.mean()) / ranked.std(ddof=0)
    if standardized.isna().any().any():
        return float("nan")
    design = np.column_stack(
        [
            np.ones(len(standardized)),
            standardized["gm_delta_yoy"].to_numpy(),
            standardized["gm_innovation_std"].to_numpy(),
        ]
    )
    outcome = standardized["market_excess_return_arithmetic"].to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < design.shape[1]:
        return float("nan")
    return float(coefficients[2])


def _spearman(left: pd.Series[Any], right: pd.Series[Any]) -> float:
    value = left.rank(method="average").corr(right.rank(method="average"), method="pearson")
    return float(value) if value is not None else float("nan")


def _bootstrap_two_sided_p(distribution: pd.Series[Any]) -> float:
    values = pd.to_numeric(distribution, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    lower = (np.count_nonzero(values <= 0.0) + 1.0) / (len(values) + 1.0)
    upper = (np.count_nonzero(values >= 0.0) + 1.0) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _benjamini_hochberg(values: pd.Series[Any]) -> pd.Series[Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = (valid * count / np.arange(1, count + 1)).iloc[::-1].cummin().iloc[::-1]
    result.loc[valid.index] = adjusted.clip(upper=1.0)
    return result


def _nan_metrics() -> dict[str, float]:
    return {
        **{f"ic_{feature}": float("nan") for feature in FEATURES},
        "innovation_minus_delta_ic": float("nan"),
        "incremental_innovation_beta": float("nan"),
    }


def _validate_preregistration(
    report: dict[str, Any], config: EffectivenessDiagnosticConfig
) -> None:
    prereg_config = report["config"]
    comparisons = {
        "primary_horizon_sessions": config.primary_horizon_sessions,
        "secondary_horizons_sessions": list(config.secondary_horizons_sessions),
        "bootstrap_replications": config.bootstrap_replications,
        "bootstrap_seed": config.bootstrap_seed,
        "confidence_level": config.confidence_level,
        "minimum_leave_one_issuer_out_sign_share": (config.minimum_leave_one_issuer_out_sign_share),
        "maximum_single_issuer_effect_share": config.maximum_single_issuer_effect_share,
    }
    for key, expected in comparisons.items():
        if prereg_config.get(key) != expected:
            raise FilingCatalogError(f"Diagnostic setting {key} differs from pre-registration")
    if int(report["minimum_combined_feature_return_events"]) != config.minimum_complete_events:
        raise FilingCatalogError("Diagnostic minimum sample differs from pre-registration")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


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
