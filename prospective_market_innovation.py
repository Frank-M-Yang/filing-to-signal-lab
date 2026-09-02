"""Preregister the forward Market-Innovation v3 research hypothesis.

The protocol is deliberately frozen before any registered filing actual or
post-filing return exists.  It replaces the failed Conditional Drift candidate
with a different economic hypothesis: the surprise relative to the last
point-in-time analyst revenue consensus, not another transformation of the
same historical gross-margin model.
"""

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

from sec_data.filing_catalog import FilingCatalogError

PROSPECTIVE_MARKET_INNOVATION_VERSION: Final[str] = (
    "prospective-market-innovation-v3-preregistration-v1"
)


class ProspectiveMarketInnovationError(FilingCatalogError):
    """Raised when the prospective registry or its outcome firewall is invalid."""


@dataclass(frozen=True)
class ProspectiveMarketInnovationConfig:
    """Frozen sample, signal, execution, and inference rules for v3."""

    minimum_registered_events: int = 30
    minimum_registered_issuers: int = 15
    minimum_complete_events: int = 30
    minimum_complete_issuers: int = 15
    minimum_positive_innovations: int = 8
    minimum_negative_innovations: int = 8
    minimum_prior_errors_for_sigma: int = 10
    primary_horizon_sessions: int = 20
    secondary_horizon_sessions: int = 5
    bootstrap_replications: int = 5_000
    bootstrap_seed: int = 20_260_829
    leave_one_issuer_out_positive_share: float = 0.80
    fixed_slippage_bps: float = 5.0
    minimum_base_cost_bps: float = 10.0
    maximum_base_cost_bps: float = 100.0

    def validate(self) -> None:
        """Reject changes that would silently weaken the prospective test."""
        frozen = (
            self.minimum_registered_events,
            self.minimum_registered_issuers,
            self.minimum_complete_events,
            self.minimum_complete_issuers,
            self.minimum_positive_innovations,
            self.minimum_negative_innovations,
            self.minimum_prior_errors_for_sigma,
            self.primary_horizon_sessions,
            self.secondary_horizon_sessions,
            self.bootstrap_replications,
            self.bootstrap_seed,
            self.leave_one_issuer_out_positive_share,
            self.fixed_slippage_bps,
            self.minimum_base_cost_bps,
            self.maximum_base_cost_bps,
        )
        expected = (
            30,
            15,
            30,
            15,
            8,
            8,
            10,
            20,
            5,
            5_000,
            20_260_829,
            0.80,
            5.0,
            10.0,
            100.0,
        )
        if frozen != expected:
            raise ValueError("The prospective Market-Innovation v3 protocol is frozen")


@dataclass(frozen=True)
class ProspectiveMarketInnovationTables:
    """Outcome-blind event registry and preregistered research rules."""

    event_registry: pd.DataFrame
    hypotheses: pd.DataFrame
    retired_models: pd.DataFrame
    observed_event_firewall: pd.DataFrame
    registration_gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveMarketInnovationResult:
    """Immutable prospective v3 preregistration artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_market_innovation_preregistration(
    forward_watchlist: pd.DataFrame,
    consensus_coverage: pd.DataFrame,
    observed_holdout_events: pd.DataFrame,
    config: ProspectiveMarketInnovationConfig | None = None,
) -> ProspectiveMarketInnovationTables:
    """Freeze a future event cohort without reading actuals or returns."""
    cfg = config or ProspectiveMarketInnovationConfig()
    cfg.validate()
    _require_columns(
        forward_watchlist,
        {
            "ticker",
            "target_period_end",
            "captured_at",
            "capture_date",
            "consensus_revenue_level",
            "consensus_range_pct",
            "revenue_estimate_analyst_count",
            "source_sha256",
            "forward_only_admission",
            "historical_backtest_admission",
            "innovation_status",
            "actual_revenue",
            "revenue_innovation",
            "revenue_innovation_pct",
        },
        "forward consensus watchlist",
    )
    _require_columns(
        consensus_coverage,
        {
            "live_consensus_target_id",
            "cik",
            "ticker",
            "strict_forward_revenue_consensus_observed",
        },
        "consensus coverage",
    )
    _require_columns(
        observed_holdout_events,
        {"validation_event_id", "cik", "ticker", "accepted_at"},
        "observed holdout firewall",
    )
    if forward_watchlist.empty or consensus_coverage.empty:
        raise ProspectiveMarketInnovationError("The prospective registry inputs cannot be empty")
    forbidden_return_columns = {
        "return",
        "market_excess_return",
        "net_market_excess_return",
        "target_return",
        "forward_return",
    }
    present_forbidden = sorted(forbidden_return_columns.intersection(forward_watchlist.columns))
    if present_forbidden:
        raise ProspectiveMarketInnovationError(
            f"The watchlist contains forbidden return columns: {present_forbidden}"
        )
    for column in ("actual_revenue", "revenue_innovation", "revenue_innovation_pct"):
        if forward_watchlist[column].notna().any():
            raise ProspectiveMarketInnovationError(
                f"The preregistration cannot read a future value from {column}"
            )
    watchlist = forward_watchlist.copy()
    watchlist["ticker"] = watchlist["ticker"].astype(str).str.strip().str.upper()
    watchlist["target_period_end"] = _utc(
        watchlist["target_period_end"], "target_period_end"
    ).dt.normalize()
    watchlist["captured_at"] = _utc(watchlist["captured_at"], "captured_at")
    watchlist["capture_date"] = _utc(watchlist["capture_date"], "capture_date").dt.normalize()
    if not watchlist["forward_only_admission"].astype(bool).all():
        raise ProspectiveMarketInnovationError("A registered consensus row is not forward-only")
    if watchlist["historical_backtest_admission"].astype(bool).any():
        raise ProspectiveMarketInnovationError("Historical consensus backfill is forbidden")
    if not watchlist["innovation_status"].astype(str).eq("pending_future_sec_actual").all():
        raise ProspectiveMarketInnovationError("A registered event is not outcome-blind")
    if not watchlist["capture_date"].le(watchlist["target_period_end"]).all():
        raise ProspectiveMarketInnovationError("A consensus capture occurs after its target period")
    if watchlist.duplicated(["ticker", "target_period_end"]).any():
        raise ProspectiveMarketInnovationError("Duplicate ticker-period registry rows exist")
    coverage = consensus_coverage.copy()
    coverage["ticker"] = coverage["ticker"].astype(str).str.strip().str.upper()
    if coverage.duplicated("ticker").any():
        raise ProspectiveMarketInnovationError("Consensus coverage contains duplicate tickers")
    if not coverage["strict_forward_revenue_consensus_observed"].astype(bool).all():
        raise ProspectiveMarketInnovationError("A frozen issuer lacks a strict forward snapshot")
    registry = watchlist.merge(
        coverage[["live_consensus_target_id", "cik", "ticker"]],
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    if registry["cik"].isna().any():
        raise ProspectiveMarketInnovationError("A registered event lacks a frozen CIK")
    registry["cik"] = registry["cik"].astype(str).str.zfill(10)
    registry["prospective_event_id"] = registry.apply(_event_id, axis=1)
    registry = registry.sort_values(
        ["target_period_end", "ticker", "prospective_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    registry["registration_order"] = np.arange(1, len(registry) + 1)
    registry["evidence_role"] = "unobserved_prospective_market_innovation"
    registry["event_status"] = "registered_awaiting_future_sec_actual"
    registry["consensus_selection_rule"] = (
        "last_snapshot_strictly_before_sec_acceptance_for_same_cik_and_period"
    )
    registry["replacement_allowed"] = False
    registry["actual_value_loaded_at_registration"] = False
    registry["return_value_loaded_at_registration"] = False
    registry["selection_uses_return_values"] = False
    registry = registry[
        [
            "prospective_event_id",
            "registration_order",
            "live_consensus_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "captured_at",
            "source_sha256",
            "consensus_revenue_level",
            "consensus_range_pct",
            "revenue_estimate_analyst_count",
            "evidence_role",
            "event_status",
            "consensus_selection_rule",
            "replacement_allowed",
            "actual_value_loaded_at_registration",
            "return_value_loaded_at_registration",
            "selection_uses_return_values",
        ]
    ]
    observed = observed_holdout_events[
        ["validation_event_id", "cik", "ticker", "accepted_at"]
    ].copy()
    observed["cik"] = observed["cik"].astype(str).str.zfill(10)
    observed["firewall_status"] = "permanently_observed_not_eligible_for_v3_selection"
    observed["eligible_for_v3_model_selection"] = False
    observed["eligible_for_future_validation"] = False
    retired = _retired_models()
    hypotheses = _hypothesis_table(cfg)
    event_count = len(registry)
    issuer_count = int(registry["cik"].nunique())
    unique_atomic_unit = not registry.duplicated(["cik", "target_period_end"]).any()
    gates = pd.DataFrame(
        [
            _gate(
                "minimum_registered_events",
                event_count,
                cfg.minimum_registered_events,
                event_count >= cfg.minimum_registered_events,
            ),
            _gate(
                "minimum_registered_issuers",
                issuer_count,
                cfg.minimum_registered_issuers,
                issuer_count >= cfg.minimum_registered_issuers,
            ),
            _gate("unique_company_period_unit", event_count, event_count, unique_atomic_unit),
            _gate(
                "strict_forward_consensus_clock",
                event_count,
                event_count,
                bool(watchlist["capture_date"].le(watchlist["target_period_end"]).all()),
            ),
            _gate(
                "observed_holdout_firewall",
                int((~observed["eligible_for_future_validation"]).sum()),
                len(observed),
                bool((~observed["eligible_for_future_validation"]).all()),
            ),
            _gate("actual_values_loaded", 0, 0, True),
            _gate("return_values_loaded", 0, 0, True),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": PROSPECTIVE_MARKET_INNOVATION_VERSION,
        "registered_events": event_count,
        "registered_issuers": issuer_count,
        "registered_target_periods": int(registry["target_period_end"].nunique()),
        "first_target_period_end": registry["target_period_end"].min().isoformat(),
        "last_target_period_end": registry["target_period_end"].max().isoformat(),
        "initial_capture_dates": int(watchlist["capture_date"].nunique()),
        "observed_holdout_events_firewalled": len(observed),
        "retired_models": len(retired),
        "registration_gates_passed": int(gates["passed"].sum()),
        "registration_gates_total": len(gates),
        "preregistration_gate_passed": bool(gates["passed"].all()),
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_fitting_performed": False,
        "primary_signal_frozen": True,
        "external_validation_metrics_computed": False,
        "portfolio_backtest_computed": False,
        "live_trading_authorized": False,
        "config": asdict(cfg),
    }
    return ProspectiveMarketInnovationTables(registry, hypotheses, retired, observed, gates, report)


def build_prospective_market_innovation_preregistration(
    consensus_panel_root: Path,
    consensus_audit_root: Path,
    observed_holdout_preregistration_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_market_innovation_preregistration"),
    config: ProspectiveMarketInnovationConfig | None = None,
) -> ProspectiveMarketInnovationResult:
    """Persist the immutable v3 protocol before any registered outcome exists."""
    cfg = config or ProspectiveMarketInnovationConfig()
    cfg.validate()
    manifests = {
        "consensus_panel_manifest_sha256": _sha256_file(
            consensus_panel_root / "output_manifest.json"
        ),
        "consensus_audit_manifest_sha256": _sha256_file(
            consensus_audit_root / "output_manifest.json"
        ),
        "observed_holdout_preregistration_manifest_sha256": _sha256_file(
            observed_holdout_preregistration_root / "output_manifest.json"
        ),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    tables = calculate_prospective_market_innovation_preregistration(
        pd.read_parquet(consensus_panel_root / "forward_consensus_watchlist.parquet"),
        pd.read_parquet(consensus_audit_root / "alpha_vantage_live_coverage.parquet"),
        pd.read_parquet(
            observed_holdout_preregistration_root / "preregistered_holdout_events.parquet"
        ),
        cfg,
    )
    payload = json.dumps(
        {
            "version": PROSPECTIVE_MARKET_INNOVATION_VERSION,
            **manifests,
            "event_ids": tables.event_registry["prospective_event_id"].tolist(),
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        PROSPECTIVE_MARKET_INNOVATION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ProspectiveMarketInnovationResult(
            target.resolve(), _read_json(target / "prospective_market_innovation_report.json")
        )
    if target.exists():
        raise ProspectiveMarketInnovationError(f"Incomplete v3 preregistration exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "prospective_market_innovation_events.parquet": tables.event_registry,
        "prospective_market_innovation_hypotheses.parquet": tables.hypotheses,
        "prospective_market_innovation_retired_models.parquet": tables.retired_models,
        "prospective_market_innovation_observed_firewall.parquet": (tables.observed_event_firewall),
        "prospective_market_innovation_registration_gates.parquet": (tables.registration_gates),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    protocol_path = staging / "prospective_market_innovation_protocol.json"
    protocol_path.write_text(json.dumps(_protocol(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[protocol_path.name] = _artifact(protocol_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "prospective_market_innovation_report.json"
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
    return ProspectiveMarketInnovationResult(target.resolve(), report)


def _hypothesis_table(config: ProspectiveMarketInnovationConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "representation": "State",
                "definition": "consensus_expected_revenue_yoy_growth",
                "role": "same_sample_benchmark",
                "uses_return_trained_parameters": False,
            },
            {
                "representation": "Delta",
                "definition": "last_minus_earliest_eligible_consensus_divided_by_earliest",
                "role": "same_sample_benchmark",
                "uses_return_trained_parameters": False,
            },
            {
                "representation": "Raw Innovation",
                "definition": (
                    "actual_revenue_minus_last_preacceptance_consensus_divided_by_consensus"
                ),
                "role": "preregistered_primary_signal",
                "uses_return_trained_parameters": False,
            },
            {
                "representation": "Standardized Innovation",
                "definition": (
                    "raw_innovation_divided_by_sigma_of_strictly_prior_oos_consensus_errors"
                ),
                "role": "secondary_after_prior_error_gate",
                "uses_return_trained_parameters": False,
            },
            {
                "representation": "Reaction",
                "definition": "market_excess_reaction_observable_before_second_session_open",
                "role": "diagnostic_not_part_of_primary_v3_score",
                "uses_return_trained_parameters": False,
            },
        ]
    ).assign(minimum_prior_errors=config.minimum_prior_errors_for_sigma)


def _retired_models() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "F1_v1_challenger",
                "status": "retired_after_external_holdout",
                "eligible_for_v3": False,
                "reason": "negative_external_rank_ic",
            },
            {
                "model_id": "F2_alignment_candidate",
                "status": "retired_after_external_holdout",
                "eligible_for_v3": False,
                "reason": "failed_preregistered_external_holdout_checks",
            },
        ]
    )


def _protocol(config: ProspectiveMarketInnovationConfig) -> dict[str, Any]:
    return {
        "protocol_version": PROSPECTIVE_MARKET_INNOVATION_VERSION,
        "economic_hypothesis": (
            "post_filing_drift_is_related_to_reported_revenue_minus_the_last_analyst_"
            "consensus_observable_before_sec_acceptance"
        ),
        "primary_score": "raw_revenue_market_innovation_pct",
        "score_formula": "(actual_revenue / last_preacceptance_consensus_revenue) - 1",
        "score_available_at": "sec_acceptance_timestamp",
        "representations_compared": [
            "State",
            "Delta",
            "Raw Innovation",
            "Standardized Innovation",
        ],
        "complete_case_rule": "all_representations_and_outcome_on_same_registered_event",
        "forecast_prerequisite": (
            "consensus_forecast_error_must_improve_on_prior_year_same_quarter_baseline"
        ),
        "primary_return_estimand": (
            "spearman_raw_innovation_vs_20_session_net_market_excess_return"
        ),
        "primary_window": (
            "second_trading_day_open_after_sec_acceptance_to_twentieth_session_close"
        ),
        "secondary_window": "same_entry_to_fifth_session_close",
        "benchmark": "SPY_cash_retained_total_return_over_identical_sessions",
        "cost": "preacceptance_corwin_schultz_plus_5bps_clipped_10_to_100bps",
        "inference": "issuer_cluster_bootstrap_5000_replications",
        "primary_success_checks": [
            "complete_event_and_issuer_minimums",
            "minimum_positive_and_negative_innovation_tails",
            "consensus_forecast_prerequisite",
            "raw_innovation_spearman_above_zero",
            "issuer_cluster_interval_lower_bound_above_zero",
            "leave_one_issuer_out_positive_share_at_least_0_80",
            "top_minus_bottom_tertile_net_return_above_zero",
        ],
        "state_delta_innovation_same_sample_required": True,
        "analyst_range_is_dispersion_state_not_forecast_sigma": True,
        "reaction_in_primary_score": False,
        "return_trained_model_in_primary_test": False,
        "registered_event_replacement_allowed": False,
        "annualized_performance_in_scope": False,
        "investment_recommendation_allowed": False,
        "config": asdict(config),
    }


def _event_id(row: pd.Series) -> str:
    payload = "|".join(
        [
            PROSPECTIVE_MARKET_INNOVATION_VERSION,
            str(row["cik"]).zfill(10),
            str(row["ticker"]),
            pd.Timestamp(row["target_period_end"]).date().isoformat(),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ProspectiveMarketInnovationError(f"Invalid {label}")
    return parsed


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ProspectiveMarketInnovationError(f"{label} lacks required columns: {missing}")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
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
