"""Same-sample diagnostics for an auditable real-return event ledger.

The module deliberately separates return accounting from signal research.  It
can describe the observed event-return ledger and quantify a deliberately biased
survivors-only counterfactual, but it will not calculate a Level/Delta/Innovation
IC until the exact filing accessions have complete point-in-time features.
"""

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
from numpy.typing import NDArray

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

EVENT_RETURN_DIAGNOSTIC_VERSION: Final[str] = "event-return-diagnostics-v1.3-asof-applicability"
REPRESENTATION_FEATURES: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
)


@dataclass(frozen=True)
class EventReturnDiagnosticConfig:
    """Precommitted thresholds for the first exploratory return diagnostic."""

    minimum_feature_events_for_ic: int = 20
    minimum_feature_coverage_ratio: float = 0.80
    uncertainty_resamples: int = 5000
    uncertainty_seed: int = 20250827
    version: str = EVENT_RETURN_DIAGNOSTIC_VERSION

    def validate(self) -> None:
        """Reject thresholds that would make a tiny-sample IC look authoritative."""
        if self.minimum_feature_events_for_ic < 10:
            raise ValueError("Exploratory IC requires at least ten complete events")
        if not 0.0 < self.minimum_feature_coverage_ratio <= 1.0:
            raise ValueError("Feature coverage ratio must be in (0, 1]")
        if self.uncertainty_resamples < 1000:
            raise ValueError("IC uncertainty requires at least 1,000 resamples")


@dataclass(frozen=True)
class EventReturnDiagnosticTables:
    """Pure diagnostic tables before persistence."""

    events: pd.DataFrame
    survivorship_counterfactual: pd.DataFrame
    form_diagnostics: pd.DataFrame
    feature_join_audit: pd.DataFrame
    representation_return_sample: pd.DataFrame
    representation_ic: pd.DataFrame
    representation_ic_uncertainty: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class EventReturnDiagnosticResult:
    """Immutable on-disk diagnostic bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_event_return_diagnostics(
    ledger: pd.DataFrame,
    readiness: pd.DataFrame,
    representation_panel: pd.DataFrame,
    config: EventReturnDiagnosticConfig | None = None,
) -> EventReturnDiagnosticTables:
    """Diagnose returns and audit the exact-accession feature join.

    The survivors-only row is a counterfactual bias demonstration.  Securities
    that delist only after the formal exit remain valid observations in the
    unbiased all-event row.
    """
    cfg = config or EventReturnDiagnosticConfig()
    cfg.validate()
    events = _prepare_events(ledger, readiness)
    feature_audit, feature_sample = _audit_representation_join(events, representation_panel)
    counterfactual = _group_diagnostics(events)
    form_diagnostics = _form_diagnostics(events)
    total_events = len(events)
    applicable_events = int(feature_audit["gross_margin_history_eligible"].sum())
    all_complete_events = int(feature_audit["representation_features_complete"].sum())
    complete_events = len(feature_sample)
    complete_ratio = complete_events / applicable_events if applicable_events else 0.0
    minimum_count_pass = complete_events >= cfg.minimum_feature_events_for_ic
    coverage_pass = complete_ratio >= cfg.minimum_feature_coverage_ratio
    exploratory_ic_allowed = minimum_count_pass and coverage_pass
    representation_ic = _representation_ic(
        feature_sample,
        cfg,
        coverage_pass=coverage_pass,
    )
    representation_ic_uncertainty = _representation_ic_uncertainty(
        feature_sample,
        cfg,
        allowed=exploratory_ic_allowed,
    )
    gates = _diagnostic_gates(
        events,
        complete_events=complete_events,
        applicable_events=applicable_events,
        minimum_feature_events=cfg.minimum_feature_events_for_ic,
        minimum_count_pass=minimum_count_pass,
        coverage_pass=coverage_pass,
    )
    all_row = counterfactual.loc[counterfactual["sample_id"].eq("all_events")].iloc[0]
    active_row = counterfactual.loc[
        counterfactual["sample_id"].eq("active_at_outcome_cutoff_only")
    ].iloc[0]
    delisted_row = counterfactual.loc[
        counterfactual["sample_id"].eq("later_delisted_after_exit")
    ].iloc[0]
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "events": total_events,
        "issuer_returns_ready": int(events["issuer_return_ready"].sum()),
        "market_excess_returns_ready": int(events["market_excess_return_ready"].sum()),
        "outcomes_ready": int(events["delisting_outcome_ready"].sum()),
        "later_delisted_after_exit": int(events["later_delisted_after_formal_exit"].sum()),
        "mean_issuer_return": float(all_row["mean_issuer_return"]),
        "mean_market_return": float(all_row["mean_market_return"]),
        "mean_market_excess_return": float(all_row["mean_market_excess_return"]),
        "median_market_excess_return": float(all_row["median_market_excess_return"]),
        "positive_market_excess_events": int(
            (events["market_excess_return_arithmetic"] > 0.0).sum()
        ),
        "survivors_only_mean_market_excess_return": float(active_row["mean_market_excess_return"]),
        "survivors_only_mean_inflation_pp": 100.0
        * (
            float(active_row["mean_market_excess_return"])
            - float(all_row["mean_market_excess_return"])
        ),
        "survivors_only_median_inflation_pp": 100.0
        * (
            float(active_row["median_market_excess_return"])
            - float(all_row["median_market_excess_return"])
        ),
        "later_delisted_mean_market_excess_return": float(
            delisted_row["mean_market_excess_return"]
        ),
        "representation_issuer_overlap": int(
            feature_audit["issuer_in_representation_universe"].sum()
        ),
        "representation_accession_overlap": int(feature_audit["accession_match"].sum()),
        "representation_level_events": int(feature_audit["representation_gm_level_ready"].sum()),
        "representation_delta_events": int(
            feature_audit["representation_gm_delta_yoy_ready"].sum()
        ),
        "representation_raw_innovation_events": int(
            feature_audit["representation_gm_innovation_raw_ready"].sum()
        ),
        "representation_standardized_innovation_events": int(
            feature_audit["representation_gm_innovation_std_ready"].sum()
        ),
        "representation_complete_events": complete_events,
        "representation_complete_coverage": complete_ratio,
        "representation_all_complete_events": all_complete_events,
        "representation_applicable_events": applicable_events,
        "representation_applicability_rule_used": bool(
            representation_panel.get("gross_margin_history_eligible", pd.Series(dtype=bool))
            .notna()
            .any()
        ),
        "exploratory_ic_allowed": exploratory_ic_allowed,
        "signal_effectiveness_conclusion_allowed": False,
        "annualized_performance_computed": False,
        "full_return_study_gate": False,
        "historical_sector_benchmark_gate": bool(events["sector_excess_return_ready"].all()),
        "interpretation": (
            "Pilot return accounting and survivorship-bias diagnostic only; "
            "not evidence that a filing signal predicts returns."
        ),
        "config": asdict(cfg),
    }
    return EventReturnDiagnosticTables(
        events=events,
        survivorship_counterfactual=counterfactual,
        form_diagnostics=form_diagnostics,
        feature_join_audit=feature_audit,
        representation_return_sample=feature_sample,
        representation_ic=representation_ic,
        representation_ic_uncertainty=representation_ic_uncertainty,
        gates=gates,
        report=report,
    )


def build_event_return_diagnostics(
    ledger_root: Path,
    readiness_root: Path,
    representation_root: Path,
    *,
    output_root: Path = Path("data/processed/event_return_diagnostics"),
    representation_artifact_name: str = "representation_panel.csv",
    config: EventReturnDiagnosticConfig | None = None,
) -> EventReturnDiagnosticResult:
    """Build an immutable diagnostic bundle tied to all upstream manifests."""
    cfg = config or EventReturnDiagnosticConfig()
    cfg.validate()
    ledger_report = _read_json(ledger_root / "event_return_report.json")
    ledger_manifest_sha256 = _sha256_file(ledger_root / "output_manifest.json")
    readiness_manifest_sha256 = _sha256_file(readiness_root / "output_manifest.json")
    if ledger_report.get("readiness_manifest_sha256") != readiness_manifest_sha256:
        raise FilingCatalogError("Diagnostic readiness does not match the return ledger")
    representation_manifest_sha256 = _sha256_file(representation_root / "output_manifest.json")
    representation_artifact = Path(representation_artifact_name)
    if representation_artifact.name != representation_artifact_name:
        raise FilingCatalogError("Representation artifact must be a filename within its root")
    representation_path = representation_root / representation_artifact
    representation_artifact_sha256 = _sha256_file(representation_path)
    ledger = load_catalog_table(ledger_root, "event_return_ledger.parquet")
    readiness = load_catalog_table(readiness_root, "sample_price_readiness.parquet")
    representation = _read_representation_panel(representation_path)
    tables = calculate_event_return_diagnostics(ledger, readiness, representation, cfg)
    payload = json.dumps(
        {
            "ledger_manifest_sha256": ledger_manifest_sha256,
            "readiness_manifest_sha256": readiness_manifest_sha256,
            "representation_manifest_sha256": representation_manifest_sha256,
            "representation_artifact_name": representation_artifact_name,
            "representation_artifact_sha256": representation_artifact_sha256,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return EventReturnDiagnosticResult(
            target.resolve(), _read_json(target / "event_return_diagnostic_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete event-return diagnostic directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "event_return_diagnostics.parquet": tables.events,
        "survivorship_counterfactual.parquet": tables.survivorship_counterfactual,
        "form_diagnostics.parquet": tables.form_diagnostics,
        "feature_join_audit.parquet": tables.feature_join_audit,
        "representation_return_sample.parquet": tables.representation_return_sample,
        "representation_ic.parquet": tables.representation_ic,
        "representation_ic_uncertainty.parquet": tables.representation_ic_uncertainty,
        "diagnostic_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "event_return_diagnostic_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_version": ledger_report["dataset_version"],
        "ledger_manifest_sha256": ledger_manifest_sha256,
        "readiness_manifest_sha256": readiness_manifest_sha256,
        "representation_manifest_sha256": representation_manifest_sha256,
        "representation_artifact_name": representation_artifact_name,
        "representation_artifact_sha256": representation_artifact_sha256,
        "artifacts": artifacts,
    }
    report_path = staging / "event_return_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "ledger_manifest_sha256": ledger_manifest_sha256,
                "readiness_manifest_sha256": readiness_manifest_sha256,
                "representation_manifest_sha256": representation_manifest_sha256,
                "representation_artifact_name": representation_artifact_name,
                "representation_artifact_sha256": representation_artifact_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return EventReturnDiagnosticResult(target.resolve(), report)


def _prepare_events(ledger: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    required_ledger = {
        "target_id",
        "cik",
        "ticker",
        "form",
        "accepted_at",
        "entry_at",
        "exit_at",
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "issuer_return_ready",
        "market_excess_return_ready",
        "sector_excess_return_ready",
        "later_delisted_after_formal_exit",
    }
    required_readiness = {
        "target_id",
        "cik",
        "selected_ticker",
        "accession_number",
        "report_date",
        "accepted_at",
        "return_study_eligible",
        "delisting_outcome_ready",
    }
    _require_columns(ledger, required_ledger, "return ledger")
    _require_columns(readiness, required_readiness, "price readiness")
    if ledger.empty or ledger["target_id"].isna().any() or ledger["target_id"].duplicated().any():
        raise FilingCatalogError("Diagnostic ledger events must be non-null and unique")
    eligible = readiness.loc[readiness["return_study_eligible"].astype(bool)].copy()
    if eligible["target_id"].duplicated().any() or set(eligible["target_id"]) != set(
        ledger["target_id"]
    ):
        raise FilingCatalogError("Readiness and ledger target IDs do not match")
    metadata = eligible[
        [
            "target_id",
            "cik",
            "selected_ticker",
            "accession_number",
            "report_date",
            "accepted_at",
            "delisting_outcome_ready",
        ]
    ].rename(
        columns={
            "cik": "readiness_cik",
            "accepted_at": "readiness_accepted_at",
        }
    )
    frame = ledger.merge(metadata, on="target_id", how="inner", validate="one_to_one")
    if not frame["cik"].astype(str).eq(frame["readiness_cik"].astype(str)).all():
        raise FilingCatalogError("CIK changed between readiness and return ledger")
    if not frame["ticker"].astype(str).eq(frame["selected_ticker"].astype(str)).all():
        raise FilingCatalogError("Ticker changed between readiness and return ledger")
    ledger_times = pd.to_datetime(frame["accepted_at"], utc=True, errors="coerce")
    readiness_times = pd.to_datetime(frame["readiness_accepted_at"], utc=True, errors="coerce")
    if (
        ledger_times.isna().any()
        or readiness_times.isna().any()
        or not ledger_times.eq(readiness_times).all()
    ):
        raise FilingCatalogError("Filing acceptance timestamps do not tie exactly")
    numeric_columns = (
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise FilingCatalogError(f"Diagnostic {column} contains a non-finite value")
    ready_columns = (
        "issuer_return_ready",
        "market_excess_return_ready",
        "delisting_outcome_ready",
    )
    for column in ready_columns:
        frame[column] = frame[column].astype(bool)
        if not frame[column].all():
            raise FilingCatalogError(f"Diagnostic requires complete {column}")
    frame["later_delisted_after_formal_exit"] = frame["later_delisted_after_formal_exit"].astype(
        bool
    )
    frame["outcome_group"] = np.where(
        frame["later_delisted_after_formal_exit"],
        "later_delisted_after_exit",
        "active_at_outcome_cutoff",
    )
    frame["issuer_return_pp"] = 100.0 * frame["issuer_total_return"]
    frame["market_return_pp"] = 100.0 * frame["market_total_return"]
    frame["market_excess_return_pp"] = 100.0 * frame["market_excess_return_arithmetic"]
    frame["excess_return_positive"] = frame["market_excess_return_arithmetic"] > 0.0
    frame["excess_rank_worst_to_best"] = (
        frame["market_excess_return_arithmetic"].rank(method="first", ascending=True).astype(int)
    )
    return frame.sort_values(
        ["market_excess_return_arithmetic", "ticker", "target_id"], kind="mergesort"
    ).reset_index(drop=True)


def _audit_representation_join(
    events: pd.DataFrame, representation_panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "company_id",
        "ticker",
        "filing_accession",
        "actual_available_at",
        *REPRESENTATION_FEATURES,
    }
    _require_columns(representation_panel, required, "representation panel")
    representation = representation_panel.copy()
    applicability_supplied = "gross_margin_history_eligible" in representation.columns
    if not applicability_supplied:
        representation["gross_margin_history_eligible"] = True
    representation["gross_margin_history_eligible"] = (
        representation["gross_margin_history_eligible"].fillna(False).astype(bool)
    )
    if (
        representation["filing_accession"].isna().any()
        or representation["filing_accession"].duplicated().any()
    ):
        raise FilingCatalogError("Representation accessions must be non-null and unique")
    representation["company_id"] = representation["company_id"].astype(str).str.zfill(10)
    issuer_universe = set(representation["company_id"])
    keep = [
        "company_id",
        "ticker",
        "filing_accession",
        "actual_available_at",
        "gross_margin_history_eligible",
        *REPRESENTATION_FEATURES,
    ]
    renamed = representation[keep].rename(
        columns={
            "company_id": "representation_company_id",
            "ticker": "representation_ticker",
            "actual_available_at": "representation_available_at",
        }
    )
    joined = events.merge(
        renamed,
        left_on="accession_number",
        right_on="filing_accession",
        how="left",
        validate="one_to_one",
    )
    joined["issuer_in_representation_universe"] = (
        joined["cik"].astype(str).str.zfill(10).isin(issuer_universe)
    )
    joined["accession_match"] = joined["filing_accession"].notna()
    expected_time = pd.to_datetime(joined["readiness_accepted_at"], utc=True, errors="coerce")
    observed_time = pd.to_datetime(joined["representation_available_at"], utc=True, errors="coerce")
    joined["availability_clock_match"] = joined["accession_match"] & expected_time.eq(observed_time)
    joined["representation_features_complete"] = (
        joined.loc[:, list(REPRESENTATION_FEATURES)].notna().all(axis="columns")
    )
    joined["gross_margin_history_eligible"] = (
        joined["gross_margin_history_eligible"].fillna(False).astype(bool)
    )
    for feature in REPRESENTATION_FEATURES:
        joined[f"representation_{feature}_ready"] = (
            joined["availability_clock_match"] & joined[feature].notna()
        )
    joined["feature_join_status"] = "issuer_not_in_representation_universe"
    joined.loc[
        joined["issuer_in_representation_universe"] & ~joined["accession_match"],
        "feature_join_status",
    ] = "event_not_in_representation_panel"
    joined.loc[
        joined["accession_match"] & ~joined["availability_clock_match"],
        "feature_join_status",
    ] = "availability_clock_mismatch"
    joined.loc[
        joined["availability_clock_match"] & ~joined["representation_features_complete"],
        "feature_join_status",
    ] = "partial_representation"
    joined.loc[
        joined["availability_clock_match"]
        & joined["gross_margin_history_eligible"]
        & joined["representation_features_complete"],
        "feature_join_status",
    ] = "complete_case_ready"
    joined.loc[
        joined["availability_clock_match"] & ~joined["gross_margin_history_eligible"],
        "feature_join_status",
    ] = "gross_margin_history_ineligible"
    audit_columns = [
        "target_id",
        "cik",
        "ticker",
        "accession_number",
        "accepted_at",
        "issuer_in_representation_universe",
        "accession_match",
        "availability_clock_match",
        "representation_gm_level_ready",
        "representation_gm_delta_yoy_ready",
        "representation_gm_innovation_raw_ready",
        "representation_gm_innovation_std_ready",
        "representation_features_complete",
        "gross_margin_history_eligible",
        "feature_join_status",
    ]
    audit = joined[audit_columns].copy()
    sample = joined.loc[joined["feature_join_status"].eq("complete_case_ready")].copy()
    sample_columns = [
        "target_id",
        "cik",
        "ticker",
        "accession_number",
        "accepted_at",
        *REPRESENTATION_FEATURES,
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "later_delisted_after_formal_exit",
    ]
    return audit, sample[sample_columns].reset_index(drop=True)


def _group_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    groups = (
        (
            "all_events",
            f"全部 {len(events)} 个事件（无偏主样本）",
            pd.Series(True, index=events.index),
        ),
        (
            "active_at_outcome_cutoff_only",
            "只保留结果截止日仍活跃（故意幸存者偏差）",
            ~events["later_delisted_after_formal_exit"],
        ),
        (
            "later_delisted_after_exit",
            "正式退出后后来退市",
            events["later_delisted_after_formal_exit"],
        ),
    )
    rows = []
    for sample_id, label, mask in groups:
        frame = events.loc[mask]
        excess = frame["market_excess_return_arithmetic"]
        rows.append(
            {
                "sample_id": sample_id,
                "sample_label": label,
                "events": len(frame),
                "mean_issuer_return": float(frame["issuer_total_return"].mean()),
                "mean_market_return": float(frame["market_total_return"].mean()),
                "mean_market_excess_return": float(excess.mean()),
                "median_market_excess_return": float(excess.median()),
                "positive_market_excess_rate": float((excess > 0.0).mean()),
                "market_excess_standard_deviation": float(excess.std(ddof=1)),
                "minimum_market_excess_return": float(excess.min()),
                "maximum_market_excess_return": float(excess.max()),
            }
        )
    return pd.DataFrame(rows)


def _form_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for form, frame in events.groupby("form", sort=True):
        excess = frame["market_excess_return_arithmetic"]
        rows.append(
            {
                "form": str(form),
                "events": len(frame),
                "mean_market_excess_return": float(excess.mean()),
                "median_market_excess_return": float(excess.median()),
                "positive_market_excess_rate": float((excess > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _representation_ic(
    sample: pd.DataFrame,
    config: EventReturnDiagnosticConfig,
    *,
    coverage_pass: bool,
) -> pd.DataFrame:
    event_count = len(sample)
    coverage_count_pass = event_count >= config.minimum_feature_events_for_ic
    rows = []
    for feature in REPRESENTATION_FEATURES:
        unique_feature_values = sample[feature].nunique(dropna=True)
        unique_return_values = sample["market_excess_return_arithmetic"].nunique(dropna=True)
        variation_pass = unique_feature_values >= 2 and unique_return_values >= 2
        allowed = coverage_count_pass and coverage_pass and variation_pass
        correlation = (
            float(
                sample[feature]
                .rank(method="average")
                .corr(sample["market_excess_return_arithmetic"].rank(method="average"))
            )
            if allowed
            else math.nan
        )
        rows.append(
            {
                "representation": feature,
                "events": event_count,
                "spearman_ic": correlation,
                "minimum_event_count_pass": coverage_count_pass,
                "variation_pass": variation_pass,
                "coverage_ratio_pass": coverage_pass,
                "status": "exploratory_only" if allowed else "not_computed_insufficient_gate",
            }
        )
    return pd.DataFrame(rows)


def _representation_ic_uncertainty(
    sample: pd.DataFrame,
    config: EventReturnDiagnosticConfig,
    *,
    allowed: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    returns = sample["market_excess_return_arithmetic"].to_numpy(dtype=float)
    for feature_index, feature in enumerate(REPRESENTATION_FEATURES):
        values = sample[feature].to_numpy(dtype=float)
        observed = _spearman(values, returns) if allowed else math.nan
        bootstrap_low = bootstrap_high = permutation_p = loo_low = loo_high = math.nan
        valid_bootstrap_draws = 0
        if allowed:
            rng = np.random.default_rng(config.uncertainty_seed + feature_index)
            bootstrap_values: list[float] = []
            for _ in range(config.uncertainty_resamples):
                draw = rng.integers(0, len(sample), size=len(sample))
                correlation = _spearman(values[draw], returns[draw])
                if np.isfinite(correlation):
                    bootstrap_values.append(correlation)
            valid_bootstrap_draws = len(bootstrap_values)
            if bootstrap_values:
                bootstrap_low, bootstrap_high = np.quantile(
                    np.asarray(bootstrap_values), [0.025, 0.975]
                ).tolist()
            permuted = np.asarray(
                [
                    _spearman(values, rng.permutation(returns))
                    for _ in range(config.uncertainty_resamples)
                ]
            )
            finite_permuted = permuted[np.isfinite(permuted)]
            if finite_permuted.size:
                permutation_p = float(
                    (1 + (np.abs(finite_permuted) >= abs(observed)).sum())
                    / (1 + finite_permuted.size)
                )
            leave_one_out = np.asarray(
                [
                    _spearman(np.delete(values, index), np.delete(returns, index))
                    for index in range(len(sample))
                ]
            )
            finite_loo = leave_one_out[np.isfinite(leave_one_out)]
            if finite_loo.size:
                loo_low = float(finite_loo.min())
                loo_high = float(finite_loo.max())
        rows.append(
            {
                "representation": feature,
                "events": len(sample),
                "spearman_ic": observed,
                "bootstrap_95_low": bootstrap_low,
                "bootstrap_95_high": bootstrap_high,
                "permutation_p_two_sided": permutation_p,
                "leave_one_out_min": loo_low,
                "leave_one_out_max": loo_high,
                "resamples_requested": config.uncertainty_resamples,
                "valid_bootstrap_draws": valid_bootstrap_draws,
                "status": "exploratory_only" if allowed else "not_computed_insufficient_gate",
            }
        )
    return pd.DataFrame(rows)


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    correlation = left_rank.corr(right_rank)
    return float(correlation) if pd.notna(correlation) else math.nan


def _diagnostic_gates(
    events: pd.DataFrame,
    *,
    complete_events: int,
    applicable_events: int,
    minimum_feature_events: int,
    minimum_count_pass: bool,
    coverage_pass: bool,
) -> pd.DataFrame:
    total = len(events)
    rows = [
        (
            "issuer_return_coverage",
            int(events["issuer_return_ready"].sum()),
            total,
            bool(events["issuer_return_ready"].all()),
            "公司收益账本完整",
        ),
        (
            "market_excess_coverage",
            int(events["market_excess_return_ready"].sum()),
            total,
            bool(events["market_excess_return_ready"].all()),
            "SPY 同窗口超额收益完整",
        ),
        (
            "delisting_outcome_coverage",
            int(events["delisting_outcome_ready"].sum()),
            total,
            bool(events["delisting_outcome_ready"].all()),
            "窗口终值与后来退市状态完整",
        ),
        (
            "sector_benchmark_coverage",
            int(events["sector_excess_return_ready"].sum()),
            total,
            bool(events["sector_excess_return_ready"].all()),
            "历史点时行业基准尚未接入",
        ),
        (
            "ldi_complete_case_coverage",
            complete_events,
            applicable_events,
            coverage_pass,
            "公告前历史预合格事件的 Level/Delta/Innovation 完整覆盖",
        ),
        (
            "ldi_minimum_event_count",
            complete_events,
            minimum_feature_events,
            minimum_count_pass,
            "第一版探索性 IC 的最低事件数",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "numerator": numerator,
                "denominator": denominator,
                "coverage": numerator / denominator if denominator else 0.0,
                "passed": passed,
                "meaning": meaning,
            }
            for gate, numerator, denominator, passed, meaning in rows
        ]
    )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


def _read_representation_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"company_id": "string", "sic": "string"})
    raise FilingCatalogError("Representation artifact must be CSV or Parquet")


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
