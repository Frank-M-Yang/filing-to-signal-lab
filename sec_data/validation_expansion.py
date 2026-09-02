"""Outcome-blind expansion plan and filing-feature build for model validation."""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.cleaning import parse_companyfacts
from sec_data.event_cleaning import derive_point_in_time_yoy_lags
from sec_data.filing_catalog import FilingCatalogError
from sec_data.pilot_filing_features import (
    PilotFilingFeatureConfig,
    attach_company_history_features,
    attach_oos_innovations,
    audit_target_feature_coverage,
    build_company_quarterly_history,
)

VALIDATION_EXPANSION_VERSION: Final[str] = "validation-expansion-v1"
VALIDATION_FEATURE_VERSION: Final[str] = "validation-expansion-features-v1.1-asof-applicability"


@dataclass(frozen=True)
class ValidationExpansionConfig:
    """Frozen cohort, time-window, and collection-wave conventions."""

    event_start: str = "2022-01-01T00:00:00Z"
    sealed_holdout_start: str = "2026-08-28T00:00:00Z"
    documented_massive_basic_history_start: str = "2024-08-14T00:00:00Z"
    collection_wave_issuers: int = 25
    minimum_planned_validation_events: int = 500
    minimum_planned_validation_issuers: int = 100
    version: str = VALIDATION_EXPANSION_VERSION

    def validate(self) -> None:
        """Reject an expansion that cannot preserve the already frozen holdout."""
        event_start = pd.Timestamp(self.event_start)
        holdout = pd.Timestamp(self.sealed_holdout_start)
        provider_start = pd.Timestamp(self.documented_massive_basic_history_start)
        if any(value.tzinfo is None for value in (event_start, holdout, provider_start)):
            raise ValueError("Expansion timestamps must be timezone-aware")
        if not event_start < provider_start < holdout:
            raise ValueError("Event, provider-history, and holdout clocks are not ordered")
        if self.collection_wave_issuers < 1:
            raise ValueError("Collection waves must contain at least one issuer")
        if self.minimum_planned_validation_events < 1:
            raise ValueError("The planned validation event threshold must be positive")
        if self.minimum_planned_validation_issuers < 1:
            raise ValueError("The planned validation issuer threshold must be positive")


@dataclass(frozen=True)
class ValidationExpansionTables:
    """Pure expansion tables before immutable persistence."""

    cohort_registry: pd.DataFrame
    candidate_events: pd.DataFrame
    collection_waves: pd.DataFrame
    sample_funnel: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationExpansionResult:
    """Immutable expansion-plan or filing-feature result."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_expansion(
    scope: pd.DataFrame,
    catalog: pd.DataFrame,
    legacy_outcomes: pd.DataFrame,
    reference_audit: pd.DataFrame,
    config: ValidationExpansionConfig | None = None,
) -> ValidationExpansionTables:
    """Freeze issuer and event eligibility without reading any return value."""
    cfg = config or ValidationExpansionConfig()
    cfg.validate()
    _require_columns(
        scope,
        {
            "cik",
            "calendar_year",
            "company_name_snapshot",
            "sampling_hash",
            "sample_size_bucket",
            "stage1_inclusion_probability",
            "stage1_sampling_weight",
            "provisional_equity_eligible",
        },
        "probability-sample scope audit",
    )
    _require_columns(
        catalog,
        {
            "cik",
            "accession_number",
            "form",
            "base_form",
            "is_amendment",
            "report_date",
            "accepted_at",
        },
        "filing catalog",
    )
    _require_columns(legacy_outcomes, {"cik", "source_study"}, "legacy outcome registry")
    _require_columns(
        reference_audit,
        {
            "target_id",
            "cik",
            "provider_query_complete",
            "event_date_reference_candidate",
        },
        "Massive reference audit",
    )

    eligible = scope.loc[scope["provisional_equity_eligible"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        raise FilingCatalogError("The probability sample has no provisional equity candidates")
    eligible["cik"] = eligible["cik"].astype(str).str.zfill(10)
    eligible["calendar_year"] = pd.to_numeric(eligible["calendar_year"], errors="raise").astype(int)
    eligible = eligible.sort_values(["cik", "calendar_year", "sampling_hash"], kind="mergesort")
    grouped = eligible.groupby("cik", sort=True)
    latest = grouped.tail(1).set_index("cik")
    cohort = grouped.agg(
        sampled_issuer_year_count=("calendar_year", "size"),
        first_sampled_year=("calendar_year", "min"),
        last_sampled_year=("calendar_year", "max"),
        cohort_sampling_hash=("sampling_hash", "min"),
    )
    cohort["sampled_issuer_years"] = grouped["calendar_year"].agg(
        lambda values: "|".join(str(value) for value in sorted(set(values)))
    )
    cohort["company_name_snapshot"] = latest["company_name_snapshot"]
    cohort["sample_size_bucket_reference"] = latest["sample_size_bucket"]
    cohort["stage1_inclusion_probability_reference"] = latest["stage1_inclusion_probability"]
    cohort["stage1_sampling_weight_reference"] = latest["stage1_sampling_weight"]
    cohort = cohort.reset_index()

    legacy = legacy_outcomes.copy()
    legacy["cik"] = legacy["cik"].astype(str).str.zfill(10)
    legacy_summary = legacy.groupby("cik", as_index=False).agg(
        legacy_outcome_rows=("source_study", "size"),
        legacy_outcome_studies=(
            "source_study",
            lambda values: "|".join(sorted(set(str(value) for value in values))),
        ),
    )
    cohort = cohort.merge(legacy_summary, on="cik", how="left", validate="one_to_one")
    cohort["legacy_outcome_rows"] = cohort["legacy_outcome_rows"].fillna(0).astype(int)
    cohort["legacy_outcome_studies"] = cohort["legacy_outcome_studies"].fillna("")
    cohort["legacy_outcome_issuer"] = cohort["legacy_outcome_rows"].gt(0)
    cohort["evaluation_eligible_issuer"] = ~cohort["legacy_outcome_issuer"]

    reference = reference_audit.copy()
    reference["cik"] = reference["cik"].astype(str).str.zfill(10)
    reference["provider_query_complete"] = reference["provider_query_complete"].fillna(False)
    reference["event_date_reference_candidate"] = reference[
        "event_date_reference_candidate"
    ].fillna(False)
    reference_summary = reference.groupby("cik", as_index=False).agg(
        reference_target_rows=("target_id", "size"),
        reference_queries_complete=("provider_query_complete", "sum"),
        reference_candidates=("event_date_reference_candidate", "sum"),
    )
    cohort = cohort.merge(reference_summary, on="cik", how="left", validate="one_to_one")
    for column in (
        "reference_target_rows",
        "reference_queries_complete",
        "reference_candidates",
    ):
        cohort[column] = cohort[column].fillna(0).astype(int)
    cohort["reference_collection_complete"] = cohort["reference_queries_complete"].eq(
        cohort["reference_target_rows"]
    ) & cohort["reference_target_rows"].gt(0)
    cohort["reference_candidate_ready"] = cohort["reference_candidates"].gt(0)
    cohort = cohort.sort_values(["cohort_sampling_hash", "cik"], kind="mergesort").reset_index(
        drop=True
    )
    cohort["expansion_rank"] = range(1, len(cohort) + 1)
    cohort["collection_wave"] = (cohort["expansion_rank"] - 1) // cfg.collection_wave_issuers + 1
    cohort["selection_uses_return_values"] = False
    cohort["replacement_after_collection_failure_allowed"] = False
    cohort["longitudinal_probability_weight_valid"] = False

    filings = catalog.copy()
    filings["cik"] = filings["cik"].astype(str).str.zfill(10)
    filings["accepted_at"] = pd.to_datetime(filings["accepted_at"], utc=True, errors="coerce")
    event_start = pd.Timestamp(cfg.event_start)
    holdout_start = pd.Timestamp(cfg.sealed_holdout_start)
    events = filings.loc[
        filings["cik"].isin(cohort["cik"])
        & filings["form"].isin(["10-Q", "10-K"])
        & ~filings["is_amendment"].fillna(False).astype(bool)
        & filings["accepted_at"].ge(event_start)
        & filings["accepted_at"].lt(holdout_start)
    ].copy()
    if events.empty:
        raise FilingCatalogError("No original periodic filings fall inside the expansion window")
    if events["accession_number"].duplicated().any():
        raise FilingCatalogError("Validation expansion accessions must be unique")
    event_columns = [
        "cik",
        "accession_number",
        "form",
        "base_form",
        "report_date",
        "accepted_at",
    ]
    cohort_columns = [
        "cik",
        "company_name_snapshot",
        "cohort_sampling_hash",
        "expansion_rank",
        "collection_wave",
        "legacy_outcome_issuer",
        "evaluation_eligible_issuer",
        "reference_collection_complete",
        "reference_candidate_ready",
    ]
    events = events[event_columns].merge(
        cohort[cohort_columns], on="cik", how="inner", validate="many_to_one"
    )
    observed_accessions = set(
        legacy.get("accession_number", pd.Series(dtype=str)).dropna().astype(str)
    )
    events["exact_outcome_previously_observed"] = events["accession_number"].isin(
        observed_accessions
    )
    events["evidence_role"] = events["legacy_outcome_issuer"].map(
        {True: "legacy_issuer_training_only", False: "unobserved_validation_candidate"}
    )
    events["validation_event_id"] = events["accession_number"].map(
        lambda accession: _stable_hash(
            {
                "accession_number": str(accession),
                "contract_version": cfg.version,
            }
        )
    )
    provider_start = pd.Timestamp(cfg.documented_massive_basic_history_start)
    events["within_documented_massive_basic_window"] = events["accepted_at"].ge(provider_start)
    events["return_outcome_read_authorized"] = False
    events["selection_uses_provider_coverage"] = False
    events = events.sort_values(
        ["accepted_at", "cik", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)

    event_counts = events.groupby("cik").size().rename("candidate_events")
    unobserved_counts = (
        events.loc[events["evidence_role"].eq("unobserved_validation_candidate")]
        .groupby("cik")
        .size()
        .rename("unobserved_candidate_events")
    )
    cohort = cohort.merge(event_counts, on="cik", how="left")
    cohort = cohort.merge(unobserved_counts, on="cik", how="left")
    cohort[["candidate_events", "unobserved_candidate_events"]] = (
        cohort[["candidate_events", "unobserved_candidate_events"]].fillna(0).astype(int)
    )
    cohort["filing_feature_build_candidate"] = cohort["candidate_events"].gt(0)

    waves = (
        cohort.groupby("collection_wave", as_index=False)
        .agg(
            issuers=("cik", "size"),
            evaluation_eligible_issuers=("evaluation_eligible_issuer", "sum"),
            candidate_events=("candidate_events", "sum"),
            unobserved_candidate_events=("unobserved_candidate_events", "sum"),
            reference_target_rows=("reference_target_rows", "sum"),
            reference_queries_complete=("reference_queries_complete", "sum"),
            reference_candidates=("reference_candidates", "sum"),
        )
        .sort_values("collection_wave")
        .reset_index(drop=True)
    )
    waves["reference_queries_pending"] = (
        waves["reference_target_rows"] - waves["reference_queries_complete"]
    )
    waves["collection_status"] = waves["reference_queries_pending"].map(
        lambda value: "reference_complete" if value == 0 else "reference_pending"
    )

    unobserved = events["evidence_role"].eq("unobserved_validation_candidate")
    provider_window = events["within_documented_massive_basic_window"].astype(bool)
    sample_funnel = pd.DataFrame(
        [
            _funnel_row(
                1, "probability_sample_equity_units", len(eligible), eligible["cik"].nunique()
            ),
            _funnel_row(2, "unique_equity_issuers", len(cohort), len(cohort)),
            _funnel_row(3, "candidate_periodic_events", len(events), events["cik"].nunique()),
            _funnel_row(
                4,
                "unobserved_validation_candidates",
                int(unobserved.sum()),
                int(events.loc[unobserved, "cik"].nunique()),
            ),
            _funnel_row(
                5,
                "unobserved_inside_basic_price_window",
                int((unobserved & provider_window).sum()),
                int(events.loc[unobserved & provider_window, "cik"].nunique()),
            ),
            _funnel_row(
                6,
                "unobserved_with_current_reference_candidate",
                int((unobserved & events["reference_candidate_ready"]).sum()),
                int(events.loc[unobserved & events["reference_candidate_ready"], "cik"].nunique()),
            ),
        ]
    )
    planned_events = int(unobserved.sum())
    planned_issuers = int(events.loc[unobserved, "cik"].nunique())
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "probability_sample_equity_units": len(eligible),
        "cohort_issuers": len(cohort),
        "legacy_outcome_issuers": int(cohort["legacy_outcome_issuer"].sum()),
        "evaluation_eligible_issuers": int(cohort["evaluation_eligible_issuer"].sum()),
        "candidate_events": len(events),
        "candidate_event_issuers": int(events["cik"].nunique()),
        "unobserved_validation_candidate_events": planned_events,
        "unobserved_validation_candidate_issuers": planned_issuers,
        "unobserved_inside_documented_basic_window": int((unobserved & provider_window).sum()),
        "reference_target_rows": int(cohort["reference_target_rows"].sum()),
        "reference_queries_complete": int(cohort["reference_queries_complete"].sum()),
        "reference_queries_pending": int(
            cohort["reference_target_rows"].sum() - cohort["reference_queries_complete"].sum()
        ),
        "current_reference_candidate_issuers": int(cohort["reference_candidate_ready"].sum()),
        "collection_waves": len(waves),
        "planned_validation_size_gate": bool(
            planned_events >= cfg.minimum_planned_validation_events
            and planned_issuers >= cfg.minimum_planned_validation_issuers
        ),
        "outcome_values_loaded": False,
        "selection_uses_return_values": False,
        "selection_uses_provider_coverage": False,
        "replacement_after_collection_failure_allowed": False,
        "probability_weights_valid_for_longitudinal_estimand": False,
        "model_evaluation_authorized": False,
        "config": asdict(cfg),
    }
    return ValidationExpansionTables(cohort, events, waves, sample_funnel, report)


def build_validation_expansion_plan(
    scope_root: Path,
    catalog_root: Path,
    reference_root: Path,
    legacy_return_roots: Iterable[Path],
    *,
    output_root: Path = Path("data/processed/validation_expansion"),
    config: ValidationExpansionConfig | None = None,
) -> ValidationExpansionResult:
    """Persist the outcome-blind expansion cohort and candidate event registry."""
    cfg = config or ValidationExpansionConfig()
    cfg.validate()
    legacy_roots = tuple(legacy_return_roots)
    manifests = {
        "scope_manifest_sha256": _sha256_file(scope_root / "output_manifest.json"),
        "catalog_manifest_sha256": _sha256_file(catalog_root / "output_manifest.json"),
        "reference_manifest_sha256": _sha256_file(reference_root / "output_manifest.json"),
        "legacy_manifest_sha256s": [
            _sha256_file(root / "output_manifest.json") for root in legacy_roots
        ],
    }
    scope = _read_verified_table(scope_root, "sample_scope_audit.parquet")
    _verify_artifact(catalog_root, "filing_catalog.parquet")
    candidate_ciks = (
        scope.loc[scope["provisional_equity_eligible"].fillna(False).astype(bool), "cik"]
        .astype(str)
        .str.zfill(10)
        .unique()
        .tolist()
    )
    catalog = pd.read_parquet(
        catalog_root / "filing_catalog.parquet",
        filters=[("cik", "in", candidate_ciks)],
    )
    reference = _read_verified_table(reference_root, "sample_massive_reference_audit.parquet")
    legacy = _legacy_outcome_registry(legacy_roots)
    tables = calculate_validation_expansion(scope, catalog, legacy, reference, cfg)
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationExpansionResult(
            target.resolve(), _read_json(target / "validation_expansion_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete validation expansion exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_cohort_registry.parquet": tables.cohort_registry,
        "validation_candidate_events.parquet": tables.candidate_events,
        "validation_collection_waves.parquet": tables.collection_waves,
        "validation_sample_funnel.parquet": tables.sample_funnel,
        "legacy_outcome_registry.parquet": legacy,
    }
    artifacts = _write_frames(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    _finalize(staging, target, report, "validation_expansion_report.json", artifacts, manifests)
    return ValidationExpansionResult(target.resolve(), report)


def build_validation_expansion_features(
    plan_root: Path,
    catalog_root: Path,
    companyfacts_snapshot_manifest: Path,
    *,
    output_root: Path = Path("data/processed/validation_expansion_features"),
    config: PilotFilingFeatureConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ValidationExpansionResult:
    """Build PIT filing features for the frozen events without loading market outcomes."""
    cfg = config or PilotFilingFeatureConfig()
    cfg.validate()
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    companyfacts_manifest = _read_json(companyfacts_snapshot_manifest)
    companyfacts_sha256 = str(companyfacts_manifest["sha256"])
    companyfacts_path = Path(str(companyfacts_manifest["path"]))
    if not companyfacts_path.is_file() or _sha256_file(companyfacts_path) != companyfacts_sha256:
        raise FilingCatalogError("Companyfacts ZIP does not match its immutable manifest")
    targets = _read_verified_table(plan_root, "validation_candidate_events.parquet")
    target_ciks = targets["cik"].astype(str).str.zfill(10).sort_values().unique().tolist()
    _verify_artifact(catalog_root, "filing_catalog.parquet")
    _verify_artifact(catalog_root, "issuer_registry_snapshot.parquet")
    filings = pd.read_parquet(
        catalog_root / "filing_catalog.parquet",
        filters=[("cik", "in", target_ciks)],
    )
    filings = filings.loc[filings["form"].isin(["10-Q", "10-Q/A", "10-K", "10-K/A"])].copy()
    issuer_registry = pd.read_parquet(
        catalog_root / "issuer_registry_snapshot.parquet",
        filters=[("cik", "in", target_ciks)],
    )
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "companyfacts_sha256": companyfacts_sha256,
            "feature_config": asdict(cfg),
            "version": VALIDATION_FEATURE_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = VALIDATION_FEATURE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return ValidationExpansionResult(
            target_root.resolve(), _read_json(target_root / "validation_feature_report.json")
        )
    if target_root.exists():
        raise FilingCatalogError(f"Incomplete validation feature directory exists: {target_root}")

    event_frames: list[pd.DataFrame] = []
    selection_audits: list[pd.DataFrame] = []
    normalization_rejections: list[pd.DataFrame] = []
    company_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(companyfacts_path) as archive:
        members = set(archive.namelist())
        for number, cik in enumerate(target_ciks, start=1):
            member = f"CIK{cik}.json"
            company_filings = filings.loc[filings["cik"].astype(str).eq(cik)].copy()
            issuer = issuer_registry.loc[issuer_registry["cik"].astype(str).eq(cik)]
            company_name = (
                str(issuer.iloc[0]["company_name_snapshot"])
                if not issuer.empty
                else str(
                    targets.loc[targets["cik"].astype(str).eq(cik), "company_name_snapshot"].iloc[0]
                )
            )
            if member not in members:
                company_rows.append(
                    {
                        "cik": cik,
                        "company_name": company_name,
                        "companyfacts_status": "member_missing",
                        "periodic_filings": len(company_filings),
                        "normalized_facts": 0,
                        "normalization_rejections": 0,
                        "quarterly_events_before_lags": 0,
                    }
                )
                if progress is not None:
                    progress(number, len(target_ciks), cik)
                continue
            document = json.loads(archive.read(member))
            parsed = parse_companyfacts(
                document,
                company_filings,
                source_url=f"{companyfacts_manifest['source_url']}#{member}",
                snapshot_hash=companyfacts_sha256,
            )
            events, audit = build_company_quarterly_history(
                cik=cik,
                ticker="UNRESOLVED",
                company_name=company_name,
                filings=company_filings,
                facts=parsed.data,
                dataset_version=dataset_version,
                config=cfg,
            )
            if not events.empty:
                event_frames.append(events)
            if not audit.empty:
                selection_audits.append(audit)
            if not parsed.rejections.empty:
                rejected = parsed.rejections.copy()
                rejected.insert(0, "cik", cik)
                normalization_rejections.append(rejected)
            company_rows.append(
                {
                    "cik": cik,
                    "company_name": company_name,
                    "companyfacts_status": "parsed",
                    "periodic_filings": len(company_filings),
                    "normalized_facts": len(parsed.data),
                    "normalization_rejections": len(parsed.rejections),
                    "quarterly_events_before_lags": len(events),
                }
            )
            if progress is not None:
                progress(number, len(target_ciks), cik)
    if not event_frames:
        raise FilingCatalogError("No quarterly PIT events could be constructed for the expansion")
    history = pd.concat(event_frames, ignore_index=True)
    history = derive_point_in_time_yoy_lags(history)
    history = attach_company_history_features(history, cfg)
    history = attach_oos_innovations(history, cfg)
    history["pilot_feature_version"] = dataset_version
    history["validation_expansion_feature_version"] = dataset_version
    history = history.sort_values(
        ["actual_available_at", "company_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    feature_targets = targets.copy()
    feature_targets["target_id"] = feature_targets["validation_event_id"]
    feature_targets["selected_ticker"] = "UNRESOLVED"
    feature_targets["formal_entry_open_at"] = pd.NaT
    feature_targets["formal_exit_close_at"] = pd.NaT
    target_features, target_audit = audit_target_feature_coverage(feature_targets, history)
    role_columns = [
        "validation_event_id",
        "evidence_role",
        "legacy_outcome_issuer",
        "within_documented_massive_basic_window",
        "collection_wave",
    ]
    target_features = target_features.merge(
        targets[role_columns],
        left_on="target_id",
        right_on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    target_audit = target_audit.merge(
        targets[role_columns],
        left_on="target_id",
        right_on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    selection_audit = (
        pd.concat(selection_audits, ignore_index=True) if selection_audits else pd.DataFrame()
    )
    fact_rejections = (
        pd.concat(normalization_rejections, ignore_index=True)
        if normalization_rejections
        else pd.DataFrame()
    )
    company_coverage = pd.DataFrame(company_rows)
    unobserved = target_features["evidence_role"].eq("unobserved_validation_candidate")
    complete = target_features["analysis_ready_innovation_std"].fillna(False).astype(bool)
    history_eligible = target_features["gross_margin_history_eligible"].fillna(False).astype(bool)
    applicable_complete = (
        target_features["gross_margin_applicability_complete_case"].fillna(False).astype(bool)
    )
    basic_window = target_features["within_documented_massive_basic_window"].fillna(False)
    unobserved_applicable = unobserved & history_eligible
    unobserved_applicable_complete = unobserved & applicable_complete
    unobserved_basic_applicable = unobserved_applicable & basic_window
    unobserved_basic_applicable_complete = unobserved_applicable_complete & basic_window
    applicable_count = int(unobserved_applicable.sum())
    applicable_complete_count = int(unobserved_applicable_complete.sum())
    applicable_coverage = applicable_complete_count / applicable_count if applicable_count else 0.0
    basic_applicable_count = int(unobserved_basic_applicable.sum())
    basic_applicable_complete_count = int(unobserved_basic_applicable_complete.sum())
    basic_applicable_coverage = (
        basic_applicable_complete_count / basic_applicable_count if basic_applicable_count else 0.0
    )
    status_counts = target_audit["feature_status"].value_counts().sort_index().to_dict()
    report: dict[str, Any] = {
        "contract_version": VALIDATION_FEATURE_VERSION,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_manifest_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "companyfacts_sha256": companyfacts_sha256,
        "target_events": len(target_features),
        "target_companies": int(target_features["company_id"].nunique()),
        "quarterly_history_events": len(history),
        "history_level_ready": int(history["analysis_ready_level"].sum()),
        "history_delta_ready": int(history["analysis_ready_delta"].sum()),
        "history_oos_forecasts": int(history["gm_forecast"].notna().sum()),
        "history_standardized_innovations": int(history["gm_innovation_std"].notna().sum()),
        "target_level_ready": int(target_features["analysis_ready_level"].fillna(False).sum()),
        "target_delta_ready": int(target_features["analysis_ready_delta"].fillna(False).sum()),
        "target_standardized_innovation_ready": int(complete.sum()),
        "unobserved_target_events": int(unobserved.sum()),
        "unobserved_target_companies": int(target_features.loc[unobserved, "company_id"].nunique()),
        "unobserved_complete_feature_events": int((unobserved & complete).sum()),
        "unobserved_complete_feature_companies": int(
            target_features.loc[unobserved & complete, "company_id"].nunique()
        ),
        "unobserved_complete_inside_basic_price_window": int(
            (unobserved & complete & basic_window).sum()
        ),
        "unobserved_history_eligible_events": applicable_count,
        "unobserved_applicability_complete_events": applicable_complete_count,
        "unobserved_applicability_complete_coverage": applicable_coverage,
        "unobserved_basic_window_history_eligible_events": basic_applicable_count,
        "unobserved_basic_window_applicability_complete_events": (basic_applicable_complete_count),
        "unobserved_basic_window_applicability_complete_coverage": (basic_applicable_coverage),
        "unobserved_applicability_minimum_100_events_pass": (applicable_complete_count >= 100),
        "unobserved_applicability_80_percent_coverage_pass": (applicable_coverage >= 0.80),
        "market_identity_collection_candidate_events": basic_applicable_count,
        "market_price_collection_candidate_events": basic_applicable_complete_count,
        "market_collection_selection_uses_return_values": False,
        "target_feature_status_counts": status_counts,
        "outcome_values_loaded": False,
        "market_returns_loaded": False,
        "model_evaluation_authorized": False,
        "config": asdict(cfg),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "quarterly_feature_history.parquet": history,
        "target_filing_features.parquet": target_features,
        "target_feature_audit.parquet": target_audit,
        "quarterly_selection_audit.parquet": selection_audit,
        "fact_normalization_rejections.parquet": fact_rejections,
        "company_feature_coverage.parquet": company_coverage,
    }
    artifacts = _write_frames(staging, outputs)
    report["artifacts"] = artifacts
    manifests: dict[str, Any] = {
        "plan_manifest_sha256": plan_manifest_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "companyfacts_sha256": companyfacts_sha256,
    }
    _finalize(staging, target_root, report, "validation_feature_report.json", artifacts, manifests)
    return ValidationExpansionResult(target_root.resolve(), report)


def _legacy_outcome_registry(roots: Iterable[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for root in roots:
        if (root / "event_return_ledger.parquet").is_file():
            filename = "event_return_ledger.parquet"
            available = ["cik", "accepted_at", "form", "target_id"]
        elif (root / "longitudinal_event_return_ledger.parquet").is_file():
            filename = "longitudinal_event_return_ledger.parquet"
            available = ["cik", "accession_number", "accepted_at", "form"]
        else:
            raise FilingCatalogError(f"No supported event-return ledger exists under {root}")
        _verify_artifact(root, filename)
        frame = pd.read_parquet(root / filename, columns=available).drop_duplicates()
        frame["source_study"] = root.name
        if "accession_number" not in frame:
            frame["accession_number"] = pd.NA
        rows.append(frame[["cik", "accession_number", "accepted_at", "form", "source_study"]])
    if not rows:
        return pd.DataFrame(
            columns=["cik", "accession_number", "accepted_at", "form", "source_study"]
        )
    result = pd.concat(rows, ignore_index=True).drop_duplicates()
    result["cik"] = result["cik"].astype(str).str.zfill(10)
    return result.sort_values(["cik", "accepted_at", "source_study"], kind="mergesort").reset_index(
        drop=True
    )


def _read_verified_table(root: Path, filename: str) -> pd.DataFrame:
    _verify_artifact(root, filename)
    return pd.read_parquet(root / filename)


def _verify_artifact(root: Path, filename: str) -> None:
    manifest = _read_json(root / "output_manifest.json")
    artifact = manifest.get("artifacts", {}).get(filename)
    path = root / filename
    if not isinstance(artifact, dict) or not path.is_file():
        raise FilingCatalogError(f"Artifact is not declared: {filename}")
    if _sha256_file(path) != artifact.get("sha256"):
        raise FilingCatalogError(f"Artifact hash mismatch: {filename}")


def _funnel_row(order: int, stage: str, events: int, issuers: int) -> dict[str, Any]:
    return {"stage_order": order, "stage": stage, "events": events, "issuers": issuers}


def _write_frames(root: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    return artifacts


def _finalize(
    staging: Path,
    target: Path,
    report: dict[str, Any],
    report_name: str,
    artifacts: dict[str, dict[str, Any]],
    manifests: dict[str, Any],
) -> None:
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
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


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise FilingCatalogError(f"{label} is missing columns: {sorted(missing)}")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FilingCatalogError(f"Expected a JSON object in {path}")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
