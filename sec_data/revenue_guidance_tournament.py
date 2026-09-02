"""Point-in-time revenue-guidance forecast tournament against a seasonal baseline."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import GuidanceEvidenceError

REVENUE_GUIDANCE_TOURNAMENT_VERSION: Final[str] = "revenue-guidance-tournament-v1"


@dataclass(frozen=True)
class RevenueGuidanceTournamentConfig:
    """Frozen accuracy, diversity, and clustered uncertainty gates."""

    minimum_events: int = 30
    minimum_issuers: int = 10
    minimum_relative_mape_improvement: float = 0.10
    minimum_guidance_win_rate: float = 0.60
    bootstrap_repetitions: int = 5_000
    bootstrap_seed: int = 41
    version: str = REVENUE_GUIDANCE_TOURNAMENT_VERSION

    def validate(self) -> None:
        """Reject weak or invalid tournament settings."""
        if self.minimum_events < 10 or self.minimum_issuers < 3:
            raise ValueError("Revenue guidance sample gates are too small")
        if not 0.0 < self.minimum_relative_mape_improvement < 1.0:
            raise ValueError("Relative MAPE gate must be in (0, 1)")
        if not 0.5 <= self.minimum_guidance_win_rate <= 1.0:
            raise ValueError("Guidance win-rate gate must be in [0.5, 1]")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("Cluster bootstrap requires at least 1,000 repetitions")


@dataclass(frozen=True)
class RevenueGuidanceTournamentTables:
    """Complete-case forecasts, metrics, robustness, and preregistered gates."""

    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    issuer_leave_one_out: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueGuidanceTournamentResult:
    """Immutable persisted revenue-guidance forecast tournament."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_tournament(
    event_metric_features: pd.DataFrame,
    event_links: pd.DataFrame,
    filing_targets: pd.DataFrame,
    quarterly_history: pd.DataFrame,
    config: RevenueGuidanceTournamentConfig | None = None,
) -> RevenueGuidanceTournamentTables:
    """Compare prior management guidance with a strictly available seasonal forecast."""
    cfg = config or RevenueGuidanceTournamentConfig()
    cfg.validate()
    _require_columns(
        event_metric_features,
        {
            "validation_event_id",
            "cik",
            "metric",
            "selected_guidance_low",
            "selected_guidance_high",
            "selected_guidance_midpoint",
            "target_actual",
            "model_feature_ready",
        },
        "guidance event metrics",
    )
    _require_columns(
        event_links,
        {
            "validation_event_id",
            "prior_accepted_at",
            "target_accepted_at",
            "guidance_age_days",
        },
        "guidance event links",
    )
    _require_columns(
        filing_targets,
        {
            "validation_event_id",
            "company_id",
            "fiscal_year",
            "fiscal_quarter",
            "actual_available_at",
        },
        "filing targets",
    )
    _require_columns(
        quarterly_history,
        {
            "event_id",
            "company_id",
            "fiscal_year",
            "fiscal_quarter",
            "revenue",
            "unit",
            "actual_available_at",
        },
        "quarterly history",
    )
    guidance = event_metric_features.loc[
        event_metric_features["metric"].eq("revenue")
        & event_metric_features["model_feature_ready"].astype(bool)
    ].copy()
    context = filing_targets[
        [
            "validation_event_id",
            "company_id",
            "fiscal_year",
            "fiscal_quarter",
            "actual_available_at",
        ]
    ].copy()
    links = event_links[
        [
            "validation_event_id",
            "prior_accepted_at",
            "target_accepted_at",
            "guidance_age_days",
        ]
    ].copy()
    events = guidance.merge(
        context,
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    ).merge(links, on="validation_event_id", how="left", validate="one_to_one")
    history = quarterly_history.loc[
        quarterly_history["revenue"].notna() & quarterly_history["unit"].astype(str).eq("USD")
    ].copy()
    history["actual_available_at"] = _utc(
        history["actual_available_at"], "history actual_available_at"
    )
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        row = _forecast_row(event, history)
        if row is not None:
            rows.append(row)
    forecasts = pd.DataFrame(rows)
    metrics = _metric_table(forecasts)
    leave_one_out = _leave_one_issuer_out(forecasts)
    bootstrap = _cluster_bootstrap(forecasts, cfg)
    event_count = len(forecasts)
    issuer_count = int(forecasts["cik"].nunique()) if event_count else 0
    metric_index = metrics.set_index("model_id")
    guidance_mape = float(cast(Any, metric_index.loc["management_guidance", "mean_ape"]))
    baseline_mape = float(cast(Any, metric_index.loc["seasonal_same_quarter", "mean_ape"]))
    relative_improvement = 1.0 - guidance_mape / baseline_mape if baseline_mape > 0.0 else np.nan
    guidance_win_rate = float(
        (forecasts["management_guidance_ape"] < forecasts["seasonal_baseline_ape"]).mean()
    )
    gate_rows = [
        _gate("minimum_events", event_count, cfg.minimum_events, event_count >= cfg.minimum_events),
        _gate(
            "minimum_issuers",
            issuer_count,
            cfg.minimum_issuers,
            issuer_count >= cfg.minimum_issuers,
        ),
        _gate(
            "relative_mape_improvement",
            relative_improvement,
            cfg.minimum_relative_mape_improvement,
            bool(relative_improvement >= cfg.minimum_relative_mape_improvement),
        ),
        _gate(
            "guidance_event_win_rate",
            guidance_win_rate,
            cfg.minimum_guidance_win_rate,
            guidance_win_rate >= cfg.minimum_guidance_win_rate,
        ),
        _gate(
            "cluster_bootstrap_upper_ci_below_zero",
            bootstrap["mean_ape_difference_ci_high"],
            0.0,
            bootstrap["mean_ape_difference_ci_high"] < 0.0,
        ),
    ]
    gates = pd.DataFrame(gate_rows)
    forecast_gate = bool(gates["passed"].all())
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "complete_case_events": event_count,
        "complete_case_issuers": issuer_count,
        "seasonal_baseline_mape": baseline_mape,
        "management_guidance_mape": guidance_mape,
        "relative_mape_improvement": relative_improvement,
        "seasonal_baseline_median_ape": float(
            cast(Any, metric_index.loc["seasonal_same_quarter", "median_ape"])
        ),
        "management_guidance_median_ape": float(
            cast(Any, metric_index.loc["management_guidance", "median_ape"])
        ),
        "guidance_event_win_rate": guidance_win_rate,
        **bootstrap,
        "minimum_leave_one_issuer_out_improvement": (
            float(leave_one_out["relative_mape_improvement"].min())
            if not leave_one_out.empty
            else np.nan
        ),
        "forecast_gate_passed": forecast_gate,
        "selected_revenue_expectation": (
            "management_guidance" if forecast_gate else "seasonal_same_quarter"
        ),
        "model_fitting_performed": False,
        "selection_uses_return_values": False,
        "return_values_loaded": False,
        "config": asdict(cfg),
    }
    return RevenueGuidanceTournamentTables(
        forecasts,
        metrics,
        leave_one_out,
        gates,
        report,
    )


def build_revenue_guidance_tournament(
    coverage_root: Path,
    guidance_plan_root: Path,
    filing_feature_root: Path,
    *,
    output_root: Path = Path("data/processed/revenue_guidance_tournament"),
    config: RevenueGuidanceTournamentConfig | None = None,
) -> RevenueGuidanceTournamentResult:
    """Persist the guidance-versus-seasonal complete-case tournament."""
    cfg = config or RevenueGuidanceTournamentConfig()
    cfg.validate()
    input_hashes = {
        "guidance_coverage": _sha256_file(coverage_root / "output_manifest.json"),
        "guidance_plan": _sha256_file(guidance_plan_root / "output_manifest.json"),
        "filing_features": _sha256_file(filing_feature_root / "output_manifest.json"),
    }
    tables = calculate_revenue_guidance_tournament(
        load_catalog_table(coverage_root, "guidance_event_metric_features.parquet"),
        load_catalog_table(guidance_plan_root, "guidance_event_links.parquet"),
        load_catalog_table(filing_feature_root, "target_filing_features.parquet"),
        load_catalog_table(filing_feature_root, "quarterly_feature_history.parquet"),
        cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": asdict(cfg)},
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceTournamentResult(
            target.resolve(), _read_json(target / "revenue_guidance_tournament_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete revenue-guidance tournament: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "revenue_guidance_forecasts.parquet": tables.forecasts,
        "revenue_guidance_forecast_metrics.parquet": tables.metrics,
        "revenue_guidance_leave_one_issuer_out.parquet": tables.issuer_leave_one_out,
        "revenue_guidance_tournament_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "revenue_guidance_tournament_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
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
    return RevenueGuidanceTournamentResult(target.resolve(), report)


def _forecast_row(event: Any, history: pd.DataFrame) -> dict[str, Any] | None:
    prior_accepted_at = pd.Timestamp(event.prior_accepted_at)
    target_available_at = pd.Timestamp(event.actual_available_at)
    if not prior_accepted_at < target_available_at:
        raise GuidanceEvidenceError("Guidance must precede the target actual")
    target_year = int(event.fiscal_year)
    target_quarter = str(event.fiscal_quarter)
    candidates = history.loc[
        history["company_id"].eq(str(event.company_id))
        & history["fiscal_year"].eq(float(target_year - 1))
        & history["fiscal_quarter"].astype(str).eq(target_quarter)
        & history["actual_available_at"].lt(prior_accepted_at)
    ].sort_values("actual_available_at", kind="mergesort")
    if candidates.empty:
        return None
    baseline = candidates.iloc[-1]
    actual = float(event.target_actual)
    guidance = float(event.selected_guidance_midpoint)
    seasonal = float(baseline["revenue"])
    if actual <= 0.0 or guidance <= 0.0 or seasonal <= 0.0:
        return None
    guidance_ape = abs(actual - guidance) / actual
    seasonal_ape = abs(actual - seasonal) / actual
    return {
        "validation_event_id": str(event.validation_event_id),
        "cik": str(event.cik),
        "company_id": str(event.company_id),
        "target_fiscal_year": target_year,
        "target_fiscal_quarter": target_quarter,
        "guidance_accepted_at": prior_accepted_at,
        "guidance_age_days": float(event.guidance_age_days),
        "seasonal_source_event_id": str(baseline["event_id"]),
        "seasonal_source_available_at": pd.Timestamp(baseline["actual_available_at"]),
        "target_actual_available_at": target_available_at,
        "target_revenue": actual,
        "management_guidance_low": float(event.selected_guidance_low),
        "management_guidance_high": float(event.selected_guidance_high),
        "management_guidance_forecast": guidance,
        "seasonal_baseline_forecast": seasonal,
        "management_guidance_ape": guidance_ape,
        "seasonal_baseline_ape": seasonal_ape,
        "management_guidance_signed_error": (actual - guidance) / actual,
        "seasonal_baseline_signed_error": (actual - seasonal) / actual,
        "management_guidance_range_hit": bool(
            float(event.selected_guidance_low) <= actual <= float(event.selected_guidance_high)
        ),
        "guidance_beats_seasonal": guidance_ape < seasonal_ape,
        "seasonal_information_precedes_guidance": bool(
            pd.Timestamp(baseline["actual_available_at"]) < prior_accepted_at
        ),
        "selection_uses_return_values": False,
    }


def _metric_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, prefix in (
        ("seasonal_same_quarter", "seasonal_baseline"),
        ("management_guidance", "management_guidance"),
    ):
        ape = forecasts[f"{prefix}_ape"].astype(float)
        signed = forecasts[f"{prefix}_signed_error"].astype(float)
        rows.append(
            {
                "model_id": model_id,
                "events": len(forecasts),
                "issuers": int(forecasts["cik"].nunique()),
                "mean_ape": float(ape.mean()),
                "median_ape": float(ape.median()),
                "mean_signed_percentage_error": float(signed.mean()),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_issuer_out(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cik in sorted(forecasts["cik"].astype(str).unique()):
        sample = forecasts.loc[~forecasts["cik"].astype(str).eq(cik)]
        guidance = float(sample["management_guidance_ape"].mean())
        baseline = float(sample["seasonal_baseline_ape"].mean())
        rows.append(
            {
                "excluded_cik": cik,
                "remaining_events": len(sample),
                "remaining_issuers": int(sample["cik"].nunique()),
                "management_guidance_mape": guidance,
                "seasonal_baseline_mape": baseline,
                "relative_mape_improvement": 1.0 - guidance / baseline,
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    forecasts: pd.DataFrame,
    config: RevenueGuidanceTournamentConfig,
) -> dict[str, float]:
    differences = forecasts["management_guidance_ape"] - forecasts["seasonal_baseline_ape"]
    grouped = (
        pd.DataFrame({"cik": forecasts["cik"].astype(str), "difference": differences})
        .groupby("cik")["difference"]
        .agg(["sum", "count"])
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = rng.integers(
        0,
        len(grouped),
        size=(config.bootstrap_repetitions, len(grouped)),
    )
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    observed = float(differences.mean())
    return {
        "mean_ape_difference_guidance_minus_seasonal": observed,
        "mean_ape_difference_ci_low": float(np.quantile(draws, 0.025)),
        "mean_ape_difference_ci_high": float(np.quantile(draws, 0.975)),
        "one_sided_cluster_bootstrap_p_value": float(
            (np.count_nonzero(draws >= 0.0) + 1) / (len(draws) + 1)
        ),
    }


def _gate(gate_id: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise GuidanceEvidenceError(f"{label} contains invalid timestamps")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GuidanceEvidenceError(f"{label} lacks required columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
