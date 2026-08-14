"""Construct separated representative and stress sampling frames from SEC metadata."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

STRESS_DOMAIN: Final[dict[str, str]] = {
    "periodic_amendment": "revision_reporting",
    "late_filing_notice": "revision_reporting",
    "financial_statement_non_reliance": "revision_reporting",
    "auditor_change": "revision_reporting",
    "bankruptcy_or_receivership": "exit_distress",
    "listing_compliance_or_delisting_notice": "exit_distress",
    "formal_delisting": "exit_distress",
    "reporting_deregistration": "exit_distress",
    "acquisition_or_disposition_completed": "corporate_action",
    "change_in_control": "corporate_action",
}

SMALL_ASSET_REVIEW_USD: Final[float] = 10_000.0
LARGE_ASSET_REVIEW_USD: Final[float] = 5_000_000_000_000.0


@dataclass(frozen=True)
class UniverseDesignResult:
    """Immutable universe-design tables and gate report."""

    output_root: Path
    report: dict[str, Any]


def build_universe_design(
    catalog_root: Path,
    *,
    size_root: Path | None = None,
    cover_root: Path | None = None,
    probability_sample_root: Path | None = None,
    scope_audit_root: Path | None = None,
    price_readiness_root: Path | None = None,
    output_root: Path = Path("data/processed/sec_universe_design"),
) -> UniverseDesignResult:
    """Separate population candidates from enriched stress and exit samples.

    When ``size_root`` is supplied, accession-matched total-assets anchors are
    joined by subject CIK and filing-acceptance calendar year. Coarse absolute
    screens only flag facts for semantic review; they never exclude issuers.
    """
    catalog_report = json.loads((catalog_root / "catalog_report.json").read_text(encoding="utf-8"))
    if not bool(catalog_report.get("archive_coverage_complete")):
        raise FilingCatalogError("Representative frame requires complete archive coverage")
    catalog_manifest_path = catalog_root / "output_manifest.json"
    catalog_manifest_sha256 = _sha256_file(catalog_manifest_path)
    inventory = load_catalog_table(catalog_root, "issuer_year_inventory.parquet")
    signals = load_catalog_table(catalog_root, "filing_signals.parquet")
    archive_audit = load_catalog_table(catalog_root, "archive_coverage_audit.parquet")

    representative = inventory.loc[inventory["representative_candidate"].astype(bool)].copy()
    representative_ciks = set(representative["cik"].astype(str))
    unresolved_archive_ciks = set(
        archive_audit.loc[
            ~archive_audit["coverage_status"].eq("resolved_from_bulk_snapshot"), "cik"
        ].astype(str)
    )
    representative["archive_coverage_complete"] = ~representative["cik"].astype(str).isin(
        unresolved_archive_ciks
    )
    cover_report: dict[str, Any] | None = None
    cover_manifest_sha256: str | None = None
    if cover_root is not None:
        cover_report = json.loads(
            (cover_root / "filing_cover_report.json").read_text(encoding="utf-8")
        )
        if cover_report.get("catalog_manifest_sha256") != catalog_manifest_sha256:
            raise FilingCatalogError("Filing-cover history does not tie to the catalog manifest")
        cover_manifest_sha256 = _sha256_file(cover_root / "output_manifest.json")
        cover_anchors = load_catalog_table(
            cover_root, "issuer_year_identity_industry_anchors.parquet"
        )
        selected_cover_columns = cover_anchors[
            [
                "cik",
                "calendar_year",
                "target_id",
                "index_artifact_preserved",
                "cover_artifact_preserved",
                "filing_identity_ready",
                "security_identity_observed",
                "tradable_security_mapping_ready",
                "historical_identity_ready",
                "historical_industry_ready",
                "registrant_name",
                "identity_status",
                "sic",
                "sic_description",
                "industry_evidence_status",
            ]
        ].rename(
            columns={
                "target_id": "cover_target_id",
                "registrant_name": "filing_time_registrant_name",
                "sic": "filing_time_sic",
                "sic_description": "filing_time_sic_description",
            }
        )
        representative = representative.merge(
            selected_cover_columns,
            on=["cik", "calendar_year"],
            how="left",
            validate="one_to_one",
        )
        for column in (
            "index_artifact_preserved",
            "cover_artifact_preserved",
            "filing_identity_ready",
            "security_identity_observed",
            "tradable_security_mapping_ready",
            "historical_identity_ready",
            "historical_industry_ready",
        ):
            representative[column] = representative[column].fillna(False).astype(bool)
        representative["historical_equity_scope_status"] = np.select(
            [
                ~representative["index_artifact_preserved"],
                representative["filing_time_sic"].eq("6189"),
                representative["filing_time_sic"].eq("6770"),
                ~representative["historical_industry_ready"],
            ],
            [
                "awaiting_filing_index",
                "asset_backed_security_review_not_common_equity",
                "blank_check_special_purpose_review",
                "filing_index_has_no_sic",
            ],
            default="issuer_industry_observed_security_scope_pending",
        )
    else:
        representative["index_artifact_preserved"] = False
        representative["cover_artifact_preserved"] = False
        representative["filing_identity_ready"] = False
        representative["security_identity_observed"] = False
        representative["tradable_security_mapping_ready"] = False
        representative["historical_identity_ready"] = False
        representative["historical_industry_ready"] = False
        representative["historical_equity_scope_status"] = "cover_asset_not_supplied"
    size_report: dict[str, Any] | None = None
    size_manifest_sha256: str | None = None
    if size_root is not None:
        size_report = json.loads(
            (size_root / "size_feature_report.json").read_text(encoding="utf-8")
        )
        if size_report.get("catalog_manifest_sha256") != catalog_manifest_sha256:
            raise FilingCatalogError("Size features do not tie to the selected catalog manifest")
        size_manifest_sha256 = _sha256_file(size_root / "output_manifest.json")
        size_anchors = load_catalog_table(size_root, "issuer_year_size_anchors.parquet")
        selected_size_columns = size_anchors[
            [
                "cik",
                "calendar_year",
                "accession_number",
                "form",
                "accepted_at",
                "total_assets_usd",
                "log_total_assets",
                "asset_size_percentile",
                "asset_size_stratum",
                "size_anchor_rule",
                "size_measure_not_market_cap",
            ]
        ].rename(
            columns={
                "accession_number": "size_anchor_accession_number",
                "form": "size_anchor_form",
                "accepted_at": "size_anchor_accepted_at",
                "total_assets_usd": "size_anchor_total_assets_usd",
                "log_total_assets": "size_anchor_log_total_assets",
                "asset_size_percentile": "size_anchor_percentile",
                "asset_size_stratum": "size_anchor_stratum",
            }
        )
        representative = representative.merge(
            selected_size_columns,
            on=["cik", "calendar_year"],
            how="left",
            validate="one_to_one",
        )
        representative["event_time_size_ready"] = representative[
            "size_anchor_total_assets_usd"
        ].notna()
        small_review = representative["size_anchor_total_assets_usd"].lt(SMALL_ASSET_REVIEW_USD)
        large_review = representative["size_anchor_total_assets_usd"].gt(LARGE_ASSET_REVIEW_USD)
        representative["size_plausibility_review_required"] = small_review | large_review
        representative["size_plausibility_status"] = np.select(
            [
                ~representative["event_time_size_ready"],
                small_review,
                large_review,
            ],
            [
                "missing_event_time_size",
                "review_below_10k_usd_not_excluded",
                "review_above_5t_usd_not_excluded",
            ],
            default="not_flagged_by_coarse_absolute_screen",
        )
    else:
        representative["event_time_size_ready"] = False
        representative["size_plausibility_review_required"] = False
        representative["size_plausibility_status"] = "size_asset_not_supplied"
    representative["sample_freeze_eligible"] = (
        representative["archive_coverage_complete"]
        & representative["historical_identity_ready"]
        & representative["event_time_size_ready"]
        & representative["historical_industry_ready"]
        & ~representative["size_plausibility_review_required"]
    )
    representative["sampling_role"] = "representative_candidate"
    representative["sampling_stratum"] = pd.NA
    representative["sampling_weight"] = np.nan
    representative["blocked_reason"] = "awaiting_filing_index_identity_and_industry"
    representative.loc[
        representative["index_artifact_preserved"]
        & representative["filing_identity_ready"]
        & representative["historical_industry_ready"]
        & ~representative["security_identity_observed"],
        "blocked_reason",
    ] = "issuer_and_sic_ready;awaiting_filing_cover_security_identity"
    representative.loc[
        representative["historical_identity_ready"] & representative["historical_industry_ready"],
        "blocked_reason",
    ] = "identity_and_industry_ready;awaiting_remaining_gates"
    missing_size = ~representative["event_time_size_ready"]
    representative.loc[missing_size, "blocked_reason"] = (
        "missing_event_time_size;" + representative.loc[missing_size, "blocked_reason"].astype(str)
    )
    size_review = representative["size_plausibility_review_required"]
    representative.loc[size_review, "blocked_reason"] = (
        "size_value_requires_semantic_review;"
        + representative.loc[size_review, "blocked_reason"].astype(str)
    )
    representative["future_exit_information_allowed_for_sampling"] = False
    representative = representative.sort_values(["calendar_year", "cik"]).reset_index(drop=True)

    scoped_signals = signals.loc[signals["cik"].astype(str).isin(representative_ciks)].copy()
    scoped_signals["stress_domain"] = scoped_signals["signal_type"].map(STRESS_DOMAIN)
    scoped_signals = scoped_signals.loc[scoped_signals["stress_domain"].notna()].copy()
    stress_membership = (
        scoped_signals.groupby(["cik", "stress_domain"], as_index=False)
        .agg(
            trigger_count=("signal_id", "size"),
            first_trigger_at=("accepted_at", "min"),
            last_trigger_at=("accepted_at", "max"),
            trigger_types=(
                "signal_type",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
            source_accessions=(
                "accession_number",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
        )
        .sort_values(["stress_domain", "cik"])
        .reset_index(drop=True)
    )
    stress_membership["sampling_role"] = "stress_only"
    stress_membership["population_weight"] = np.nan
    stress_membership["may_estimate_population_rate"] = False
    stress_membership["stress_membership_version"] = "revision-stress-membership-v1"

    exit_events = scoped_signals.loc[scoped_signals["event_family"].eq("exit")].copy()
    exit_events["exit_taxonomy_status"] = "candidate_requires_filing_resolution"
    exit_events["final_exit_reason"] = "unknown"
    exit_events["status_interval_ready"] = False
    exit_events["label_use"] = "outcome_or_stress_only_not_sampling_feature"
    exit_events = exit_events.sort_values(["accepted_at", "cik", "signal_id"]).reset_index(
        drop=True
    )

    probability_report: dict[str, Any] | None = None
    probability_manifest_sha256: str | None = None
    if probability_sample_root is not None:
        probability_report = json.loads(
            (probability_sample_root / "sample_plan_report.json").read_text(encoding="utf-8")
        )
        if probability_report.get("catalog_manifest_sha256") != catalog_manifest_sha256:
            raise FilingCatalogError("Probability sample does not tie to the catalog manifest")
        probability_manifest_sha256 = _sha256_file(probability_sample_root / "output_manifest.json")
    scope_report: dict[str, Any] | None = None
    scope_manifest_sha256: str | None = None
    if scope_audit_root is not None:
        scope_report = json.loads(
            (scope_audit_root / "scope_audit_report.json").read_text(encoding="utf-8")
        )
        if scope_report.get("catalog_manifest_sha256") != catalog_manifest_sha256:
            raise FilingCatalogError("Equity-scope audit does not tie to the catalog manifest")
        if probability_report is None or scope_report.get(
            "sample_plan_version"
        ) != probability_report.get("dataset_version"):
            raise FilingCatalogError("Equity-scope audit does not tie to the probability sample")
        scope_manifest_sha256 = _sha256_file(scope_audit_root / "output_manifest.json")
    price_report: dict[str, Any] | None = None
    price_manifest_sha256: str | None = None
    if price_readiness_root is not None:
        price_report = json.loads(
            (price_readiness_root / "price_readiness_report.json").read_text(encoding="utf-8")
        )
        if scope_report is None or price_report.get("scope_version") != scope_report.get(
            "dataset_version"
        ):
            raise FilingCatalogError("Price-readiness audit does not tie to the equity-scope audit")
        if price_report.get("scope_manifest_sha256") != scope_manifest_sha256:
            raise FilingCatalogError("Price-readiness scope manifest hash does not match")
        price_manifest_sha256 = _sha256_file(price_readiness_root / "output_manifest.json")

    dataset_payload = json.dumps(
        {
            "catalog_version": catalog_report["dataset_version"],
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "size_version": size_report["dataset_version"] if size_report else None,
            "size_manifest_sha256": size_manifest_sha256,
            "cover_version": cover_report["dataset_version"] if cover_report else None,
            "cover_manifest_sha256": cover_manifest_sha256,
            "probability_sample_version": (
                probability_report["dataset_version"] if probability_report else None
            ),
            "probability_sample_manifest_sha256": probability_manifest_sha256,
            "scope_audit_version": scope_report["dataset_version"] if scope_report else None,
            "scope_audit_manifest_sha256": scope_manifest_sha256,
            "price_readiness_version": price_report["dataset_version"] if price_report else None,
            "price_readiness_manifest_sha256": price_manifest_sha256,
            "stress_domain": STRESS_DOMAIN,
            "small_asset_review_usd": SMALL_ASSET_REVIEW_USD,
            "large_asset_review_usd": LARGE_ASSET_REVIEW_USD,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "sec-universe-design-v6-" + hashlib.sha256(dataset_payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads(
            (target / "universe_gate_report.json").read_text(encoding="utf-8")
        )
        return UniverseDesignResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete universe-design directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    artifacts = {
        "representative_candidates.parquet": representative,
        "stress_membership.parquet": stress_membership,
        "exit_signal_events.parquet": exit_events,
    }
    for filename, frame in artifacts.items():
        frame.to_parquet(staging / filename, index=False)
    artifact_manifest = {
        filename: {
            "rows": len(frame),
            "bytes": (staging / filename).stat().st_size,
            "sha256": _sha256_file(staging / filename),
        }
        for filename, frame in artifacts.items()
    }
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": catalog_report["dataset_version"],
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "size_asset_integrated": size_report is not None,
        "size_feature_version": size_report["dataset_version"] if size_report else None,
        "size_manifest_sha256": size_manifest_sha256,
        "filing_cover_asset_integrated": cover_report is not None,
        "filing_cover_version": cover_report["dataset_version"] if cover_report else None,
        "filing_cover_manifest_sha256": cover_manifest_sha256,
        "stage1_probability_sample_frozen": probability_report is not None,
        "probability_sample_version": (
            probability_report["dataset_version"] if probability_report else None
        ),
        "probability_sample_manifest_sha256": probability_manifest_sha256,
        "probability_sample_issuer_years": (
            probability_report["selected_issuer_years"] if probability_report else 0
        ),
        "probability_sampling_strata": (
            probability_report["sampling_strata"] if probability_report else 0
        ),
        "probability_weights_ready": bool(
            probability_report and probability_report.get("probability_weights_ready")
        ),
        "equity_scope_audit_version": scope_report["dataset_version"] if scope_report else None,
        "equity_scope_audit_manifest_sha256": scope_manifest_sha256,
        "equity_scope_adjudicated_issuer_years": (
            scope_report["scope_adjudicated"] if scope_report else 0
        ),
        "provisional_equity_candidate_issuer_years": (
            scope_report["provisional_equity_candidates"] if scope_report else 0
        ),
        "equity_scope_gate": bool(scope_report and scope_report.get("scope_gate")),
        "price_readiness_version": price_report["dataset_version"] if price_report else None,
        "price_readiness_manifest_sha256": price_manifest_sha256,
        "price_mapping_candidate_issuer_years": (
            price_report["price_mapping_candidates"] if price_report else 0
        ),
        "provider_queries_complete": (
            price_report["provider_queries_complete"] if price_report else 0
        ),
        "event_date_reference_candidate_issuer_years": (
            price_report["event_date_reference_candidates"] if price_report else 0
        ),
        "provider_price_queries_complete": (
            price_report.get("provider_price_queries_complete", 0) if price_report else 0
        ),
        "observed_event_window_candidate_issuer_years": (
            price_report.get("observed_event_window_candidates", 0) if price_report else 0
        ),
        "formal_event_window_ready_issuer_years": (
            price_report.get("formal_event_window_ready", 0) if price_report else 0
        ),
        "exchange_session_calendar_ready_issuer_years": (
            price_report.get("exchange_session_calendars_ready", 0) if price_report else 0
        ),
        "pilot_exchange_calendar_gate": bool(
            price_report and price_report.get("pilot_exchange_calendar_gate")
        ),
        "price_identity_gate": bool(price_report and price_report.get("price_identity_gate")),
        "daily_price_gate": bool(price_report and price_report.get("daily_price_gate")),
        "delisting_outcome_gate": bool(price_report and price_report.get("delisting_outcome_gate")),
        "exchange_calendar_gate": bool(price_report and price_report.get("exchange_calendar_gate")),
        "return_study_gate": bool(price_report and price_report.get("return_study_gate")),
        "archive_coverage_gate": True,
        "representative_candidate_issuer_years": len(representative),
        "representative_candidate_ciks": representative["cik"].nunique(),
        "stress_membership_rows": len(stress_membership),
        "stress_candidate_ciks": stress_membership["cik"].nunique(),
        "exit_signal_events": len(exit_events),
        "exit_candidate_ciks": exit_events["cik"].nunique(),
        "filing_index_ready_issuer_years": int(representative["index_artifact_preserved"].sum()),
        "filing_identity_ready_issuer_years": int(representative["filing_identity_ready"].sum()),
        "security_identity_ready_issuer_years": int(
            representative["security_identity_observed"].sum()
        ),
        "tradable_security_mapping_ready_issuer_years": int(
            representative["tradable_security_mapping_ready"].sum()
        ),
        "historical_identity_gate": bool(representative["historical_identity_ready"].all()),
        "event_time_size_ready_issuer_years": int(representative["event_time_size_ready"].sum()),
        "event_time_size_coverage": float(representative["event_time_size_ready"].mean()),
        "size_plausibility_review_issuer_years": int(
            representative["size_plausibility_review_required"].sum()
        ),
        "size_review_thresholds_usd": {
            "small": SMALL_ASSET_REVIEW_USD,
            "large": LARGE_ASSET_REVIEW_USD,
        },
        "size_review_flags_exclude_issuers": False,
        "event_time_size_gate": bool(
            (
                representative["event_time_size_ready"]
                & ~representative["size_plausibility_review_required"]
            ).all()
        ),
        "historical_industry_ready_issuer_years": int(
            representative["historical_industry_ready"].sum()
        ),
        "asset_backed_security_review_issuer_years": int(
            representative["historical_equity_scope_status"]
            .eq("asset_backed_security_review_not_common_equity")
            .sum()
        ),
        "blank_check_review_issuer_years": int(
            representative["historical_equity_scope_status"]
            .eq("blank_check_special_purpose_review")
            .sum()
        ),
        "historical_industry_gate": bool(representative["historical_industry_ready"].all()),
        "representative_sample_frozen": bool(representative["sample_freeze_eligible"].all()),
        "equity_analysis_sample_frozen": bool(
            scope_report
            and scope_report.get("scope_gate")
            and price_report
            and price_report.get("price_identity_gate")
        ),
        "stress_sample_probability_warning": (
            "The stress sample is deliberately enriched and must never estimate population rates."
        ),
        "next_required_asset": (
            "validate_historical_security_intervals_corporate_actions_and_delisting_outcomes"
        ),
        "artifacts": artifact_manifest,
    }
    (staging / "universe_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "artifacts": artifact_manifest,
                "report_sha256": _sha256_file(staging / "universe_gate_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return UniverseDesignResult(target.resolve(), report)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
