"""Return-readiness audit for the selected point-in-time revenue expectation."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import GuidanceEvidenceError

REVENUE_GUIDANCE_RETURN_AUDIT_VERSION: Final[str] = (
    "revenue-guidance-return-audit-v2-gated-inference"
)


@dataclass(frozen=True)
class RevenueGuidanceReturnAuditConfig:
    """Frozen breadth gates before any return relationship is estimated."""

    minimum_events: int = 20
    minimum_issuers: int = 10
    minimum_positive_innovations: int = 5
    minimum_negative_innovations: int = 5
    bootstrap_repetitions: int = 5_000
    bootstrap_seed: int = 53
    version: str = REVENUE_GUIDANCE_RETURN_AUDIT_VERSION

    def validate(self) -> None:
        """Reject an under-specified diagnostic contract."""
        if self.minimum_events < 10 or self.minimum_issuers < 3:
            raise ValueError("Return breadth gates are too small")
        if self.minimum_positive_innovations < 3 or self.minimum_negative_innovations < 3:
            raise ValueError("Both innovation tails require at least three events")
        if self.bootstrap_repetitions < 1_000:
            raise ValueError("Cluster bootstrap requires at least 1,000 repetitions")


@dataclass(frozen=True)
class RevenueGuidanceReturnAuditTables:
    """Coverage audit, authorized complete cases, gates, and optional statistics."""

    event_audit: pd.DataFrame
    complete_cases: pd.DataFrame
    gates: pd.DataFrame
    descriptives: pd.DataFrame
    statistics: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueGuidanceReturnAuditResult:
    """Immutable persisted return-readiness audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_return_audit(
    forecasts: pd.DataFrame,
    authorized_cohort: pd.DataFrame,
    return_ledger: pd.DataFrame,
    *,
    forecast_gate_passed: bool,
    config: RevenueGuidanceReturnAuditConfig | None = None,
) -> RevenueGuidanceReturnAuditTables:
    """Audit return coverage and estimate IC only after frozen breadth gates pass."""
    cfg = config or RevenueGuidanceReturnAuditConfig()
    cfg.validate()
    _require_columns(
        forecasts,
        {
            "validation_event_id",
            "cik",
            "guidance_accepted_at",
            "target_actual_available_at",
            "target_revenue",
            "management_guidance_forecast",
            "selection_uses_return_values",
        },
        "revenue-guidance forecasts",
    )
    _require_columns(
        authorized_cohort,
        {
            "validation_event_id",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "authorization_uses_return_values",
        },
        "authorized return cohort",
    )
    _require_columns(
        return_ledger,
        {
            "validation_event_id",
            "accepted_at",
            "entry_at",
            "entry_price_type",
            "exit_at",
            "exit_price_type",
            "holding_sessions",
            "market_excess_return_arithmetic",
            "market_excess_return_ready",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "returns_computed",
        },
        "authorized return ledger",
    )
    for frame, label in (
        (forecasts, "forecast"),
        (authorized_cohort, "authorization"),
        (return_ledger, "return ledger"),
    ):
        if (
            frame["validation_event_id"].isna().any()
            or frame["validation_event_id"].duplicated().any()
        ):
            raise GuidanceEvidenceError(f"{label} event IDs must be non-null and unique")
    if forecasts["selection_uses_return_values"].astype(bool).any():
        raise GuidanceEvidenceError("Forecast selection must not use return outcomes")

    auth = authorized_cohort[
        [
            "validation_event_id",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "authorization_uses_return_values",
        ]
    ].rename(
        columns={
            "return_outcome_read_authorized": "cohort_return_authorized",
            "selection_uses_return_values": "cohort_selection_uses_returns",
        }
    )
    ledger_columns = [
        "validation_event_id",
        "accepted_at",
        "entry_at",
        "entry_price_type",
        "exit_at",
        "exit_price_type",
        "holding_sessions",
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "market_excess_return_ready",
        "return_outcome_read_authorized",
        "selection_uses_return_values",
        "returns_computed",
    ]
    ledger_columns = [column for column in ledger_columns if column in return_ledger.columns]
    ledger = return_ledger[ledger_columns].rename(
        columns={
            "return_outcome_read_authorized": "ledger_return_authorized",
            "selection_uses_return_values": "ledger_selection_uses_returns",
        }
    )
    audit = forecasts.merge(auth, on="validation_event_id", how="left", validate="one_to_one")
    audit = audit.merge(ledger, on="validation_event_id", how="left", validate="one_to_one")
    bool_columns = [
        "cohort_return_authorized",
        "cohort_selection_uses_returns",
        "authorization_uses_return_values",
        "market_excess_return_ready",
        "ledger_return_authorized",
        "ledger_selection_uses_returns",
        "returns_computed",
    ]
    for column in bool_columns:
        audit[column] = audit[column].fillna(False).astype(bool)
    audit["execution_contract_ready"] = (
        audit["entry_price_type"].eq("unadjusted_open")
        & audit["exit_price_type"].eq("unadjusted_close")
        & audit["holding_sessions"].eq(20)
    )
    audit["clock_contract_ready"] = _utc(audit["accepted_at"]).eq(
        _utc(audit["target_actual_available_at"])
    ) & _utc(audit["entry_at"]).gt(_utc(audit["accepted_at"]))
    audit["return_complete_case"] = (
        audit["cohort_return_authorized"]
        & ~audit["cohort_selection_uses_returns"]
        & ~audit["authorization_uses_return_values"]
        & audit["ledger_return_authorized"]
        & ~audit["ledger_selection_uses_returns"]
        & audit["returns_computed"]
        & audit["market_excess_return_ready"]
        & audit["execution_contract_ready"]
        & audit["clock_contract_ready"]
        & audit["market_excess_return_arithmetic"].notna()
    )
    audit["return_audit_status"] = np.where(
        audit["return_complete_case"], "authorized_complete_case", "market_evidence_missing"
    )
    complete = audit.loc[audit["return_complete_case"]].copy()
    complete["revenue_guidance_innovation_raw"] = (
        complete["target_revenue"] - complete["management_guidance_forecast"]
    )
    complete["revenue_guidance_innovation_pct"] = (
        complete["revenue_guidance_innovation_raw"] / complete["management_guidance_forecast"].abs()
    )
    complete["innovation_sign"] = np.select(
        [
            complete["revenue_guidance_innovation_pct"].gt(0.0),
            complete["revenue_guidance_innovation_pct"].lt(0.0),
        ],
        ["positive", "negative"],
        default="zero",
    )
    event_count = len(complete)
    issuer_count = int(complete["cik"].nunique()) if event_count else 0
    positive_count = int(complete["revenue_guidance_innovation_pct"].gt(0.0).sum())
    negative_count = int(complete["revenue_guidance_innovation_pct"].lt(0.0).sum())
    gates = pd.DataFrame(
        [
            _gate("forecast_gate", forecast_gate_passed, True, forecast_gate_passed),
            _gate(
                "minimum_events", event_count, cfg.minimum_events, event_count >= cfg.minimum_events
            ),
            _gate(
                "minimum_issuers",
                issuer_count,
                cfg.minimum_issuers,
                issuer_count >= cfg.minimum_issuers,
            ),
            _gate(
                "minimum_positive_innovations",
                positive_count,
                cfg.minimum_positive_innovations,
                positive_count >= cfg.minimum_positive_innovations,
            ),
            _gate(
                "minimum_negative_innovations",
                negative_count,
                cfg.minimum_negative_innovations,
                negative_count >= cfg.minimum_negative_innovations,
            ),
        ]
    )
    diagnostic_gate = bool(gates["passed"].all())
    descriptives = _return_descriptives(complete, inference_allowed=diagnostic_gate)
    statistics = _return_statistics(complete, cfg) if diagnostic_gate else _empty_statistics()
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "forecast_events": len(forecasts),
        "forecast_issuers": int(forecasts["cik"].nunique()),
        "authorized_complete_case_events": event_count,
        "authorized_complete_case_issuers": issuer_count,
        "positive_innovation_events": positive_count,
        "negative_innovation_events": negative_count,
        "market_evidence_missing_events": int((~audit["return_complete_case"]).sum()),
        "market_evidence_coverage": event_count / len(forecasts) if len(forecasts) else 0.0,
        "return_diagnostic_gate_passed": diagnostic_gate,
        "return_diagnostic_performed": diagnostic_gate,
        "return_descriptives_performed": bool(event_count),
        "return_inference_status": (
            "descriptive_and_cluster_bootstrap_complete"
            if diagnostic_gate
            else "withheld_until_breadth_gates_pass"
        ),
        "standardized_innovation_computed": False,
        "standardized_innovation_reason": "insufficient strictly prior OOS guidance errors",
        "selection_uses_return_values": False,
        "config": asdict(cfg),
    }
    if event_count:
        report.update(
            {f"descriptive_{key}": value for key, value in descriptives.iloc[0].to_dict().items()}
        )
    if diagnostic_gate:
        report.update({str(key): value for key, value in statistics.iloc[0].to_dict().items()})
    return RevenueGuidanceReturnAuditTables(
        audit, complete, gates, descriptives, statistics, report
    )


def build_revenue_guidance_return_audit(
    tournament_root: Path,
    authorization_root: Path,
    return_ledger_root: Path,
    *,
    output_root: Path = Path("data/processed/revenue_guidance_return_audit"),
    config: RevenueGuidanceReturnAuditConfig | None = None,
) -> RevenueGuidanceReturnAuditResult:
    """Persist a manifest-tied revenue-guidance return-readiness audit."""
    cfg = config or RevenueGuidanceReturnAuditConfig()
    cfg.validate()
    tournament_report = _read_json(tournament_root / "revenue_guidance_tournament_report.json")
    input_hashes = {
        "revenue_guidance_tournament": _sha256_file(tournament_root / "output_manifest.json"),
        "return_authorization": _sha256_file(authorization_root / "output_manifest.json"),
        "return_ledger": _sha256_file(return_ledger_root / "output_manifest.json"),
    }
    tables = calculate_revenue_guidance_return_audit(
        load_catalog_table(tournament_root, "revenue_guidance_forecasts.parquet"),
        load_catalog_table(authorization_root, "validation_authorized_return_cohort.parquet"),
        load_catalog_table(return_ledger_root, "validation_event_return_ledger.parquet"),
        forecast_gate_passed=bool(tournament_report["forecast_gate_passed"]),
        config=cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceReturnAuditResult(
            target.resolve(), _read_json(target / "revenue_guidance_return_audit_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete revenue-guidance return audit: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "revenue_guidance_return_event_audit.parquet": tables.event_audit,
        "revenue_guidance_return_complete_cases.parquet": tables.complete_cases,
        "revenue_guidance_return_gates.parquet": tables.gates,
        "revenue_guidance_return_descriptives.parquet": tables.descriptives,
        "revenue_guidance_return_statistics.parquet": tables.statistics,
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
    report_path = staging / "revenue_guidance_return_audit_report.json"
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
    return RevenueGuidanceReturnAuditResult(target.resolve(), report)


def _return_statistics(
    complete_cases: pd.DataFrame, config: RevenueGuidanceReturnAuditConfig
) -> pd.DataFrame:
    innovation = complete_cases["revenue_guidance_innovation_pct"].astype(float)
    returns = complete_cases["market_excess_return_arithmetic"].astype(float)
    observed_ic = float(innovation.corr(returns, method="spearman"))
    grouped = pd.DataFrame(
        {"cik": complete_cases["cik"].astype(str), "innovation": innovation, "returns": returns}
    )
    issuers = grouped["cik"].unique()
    rng = np.random.default_rng(config.bootstrap_seed)
    draws: list[float] = []
    for _ in range(config.bootstrap_repetitions):
        sampled = rng.choice(issuers, size=len(issuers), replace=True)
        sample = pd.concat(
            [grouped.loc[grouped["cik"].eq(cik)] for cik in sampled], ignore_index=True
        )
        value = sample["innovation"].corr(sample["returns"], method="spearman")
        if pd.notna(value):
            draws.append(float(value))
    positive = returns.loc[innovation.gt(0.0)]
    negative = returns.loc[innovation.lt(0.0)]
    return pd.DataFrame(
        [
            {
                "spearman_ic": observed_ic,
                "issuer_cluster_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                "issuer_cluster_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                "positive_innovation_mean_market_excess_return": float(positive.mean()),
                "negative_innovation_mean_market_excess_return": float(negative.mean()),
                "positive_minus_negative_mean_market_excess_return": float(
                    positive.mean() - negative.mean()
                ),
            }
        ]
    )


def _return_descriptives(complete_cases: pd.DataFrame, *, inference_allowed: bool) -> pd.DataFrame:
    """Summarize observed complete cases without implying statistical inference."""
    innovation = complete_cases["revenue_guidance_innovation_pct"].astype(float)
    returns = complete_cases["market_excess_return_arithmetic"].astype(float)
    positive = returns.loc[innovation.gt(0.0)]
    negative = returns.loc[innovation.lt(0.0)]
    return pd.DataFrame(
        [
            {
                "events": len(complete_cases),
                "issuers": int(complete_cases["cik"].nunique()),
                "positive_innovation_events": int(innovation.gt(0.0).sum()),
                "negative_innovation_events": int(innovation.lt(0.0).sum()),
                "zero_innovation_events": int(innovation.eq(0.0).sum()),
                "spearman_ic": float(innovation.corr(returns, method="spearman")),
                "pearson_correlation": float(innovation.corr(returns, method="pearson")),
                "mean_market_excess_return": float(returns.mean()),
                "median_market_excess_return": float(returns.median()),
                "positive_excess_return_fraction": float(returns.gt(0.0).mean()),
                "positive_innovation_mean_market_excess_return": float(positive.mean()),
                "positive_innovation_median_market_excess_return": float(positive.median()),
                "negative_innovation_mean_market_excess_return": float(negative.mean()),
                "negative_innovation_median_market_excess_return": float(negative.median()),
                "inference_allowed": bool(inference_allowed),
                "interpretation_status": (
                    "inferential_gate_passed"
                    if inference_allowed
                    else "descriptive_only_breadth_gate_failed"
                ),
            }
        ]
    )


def _empty_statistics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "spearman_ic",
            "issuer_cluster_bootstrap_ci_low",
            "issuer_cluster_bootstrap_ci_high",
            "positive_innovation_mean_market_excess_return",
            "negative_innovation_mean_market_excess_return",
            "positive_minus_negative_mean_market_excess_return",
        ]
    )


def _gate(gate_id: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "observed": json.dumps(observed),
        "required": json.dumps(required),
        "passed": bool(passed),
    }


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


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
