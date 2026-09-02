"""Point-in-time quarterly Level/Delta/Innovation features for the 28-event pilot.

The builder uses the exact target accessions from price readiness, reconstructs
each issuer's 2016-2025 quarterly history from the immutable SEC companyfacts
snapshot, and never substitutes the separate Global 500 representation cohort.
Q4 flows are either reported directly in the 10-K or derived as FY minus the
strictly earlier original nine-month 10-Q facts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from event_panel import ForecastConfig, generate_expanding_gm_forecasts
from sec_data.cleaning import parse_companyfacts
from sec_data.event_cleaning import (
    DIRECT_GROSS_PROFIT_TAGS,
    REVENUE_TAGS,
    TOTAL_COST_TAGS,
    EventCleaningConfig,
    clean_company_quarterly_events,
    derive_point_in_time_yoy_lags,
)
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

PILOT_FEATURE_VERSION: Final[str] = "pilot-filing-features-v1.2-asof-applicability"
FLOW_TAGS: Final[tuple[str, ...]] = (
    *REVENUE_TAGS,
    *DIRECT_GROSS_PROFIT_TAGS,
    *TOTAL_COST_TAGS,
)
FORECAST_FEATURES: Final[tuple[str, ...]] = (
    "gm_yoy_lag",
    "company_gm_lag1",
    "company_gm_delta_yoy_lag1",
    "company_trailing_gm_mean",
    "company_trailing_gm_volatility",
    "revenue_growth_lag1",
)


@dataclass(frozen=True)
class PilotFilingFeatureConfig:
    """Frozen history, Q4 derivation, and OOS forecast rules."""

    start_year: int = 2016
    end_year: int = 2025
    quarter_min_duration_days: int = 70
    quarter_max_duration_days: int = 110
    annual_min_duration_days: int = 330
    annual_max_duration_days: int = 400
    ytd_min_duration_days: int = 240
    ytd_max_duration_days: int = 300
    q3_to_year_end_min_days: int = 60
    q3_to_year_end_max_days: int = 130
    min_gross_margin: float = -0.50
    max_gross_margin: float = 1.00
    trailing_history_events: int = 8
    minimum_trailing_events: int = 4
    minimum_forecast_training_events: int = 100
    minimum_prior_oos_errors: int = 50
    minimum_prior_level_events_for_applicability: int = 8
    minimum_prior_delta_events_for_applicability: int = 4
    minimum_prior_standardized_innovations_for_applicability: int = 4
    version: str = PILOT_FEATURE_VERSION

    def validate(self) -> None:
        """Reject ambiguous temporal or forecast thresholds."""
        if self.start_year > self.end_year:
            raise ValueError("Pilot feature start year must not exceed end year")
        if self.quarter_min_duration_days >= self.quarter_max_duration_days:
            raise ValueError("Quarter duration bounds are invalid")
        if self.annual_min_duration_days >= self.annual_max_duration_days:
            raise ValueError("Annual duration bounds are invalid")
        if self.ytd_min_duration_days >= self.ytd_max_duration_days:
            raise ValueError("YTD duration bounds are invalid")
        if self.q3_to_year_end_min_days >= self.q3_to_year_end_max_days:
            raise ValueError("Q3-to-year-end gap bounds are invalid")
        if self.minimum_trailing_events < 2:
            raise ValueError("Trailing features require at least two events")
        if self.minimum_forecast_training_events <= len(FORECAST_FEATURES):
            raise ValueError("Forecast training count must exceed feature count")
        if self.minimum_prior_oos_errors < 2:
            raise ValueError("Forecast sigma requires at least two prior OOS errors")
        if self.minimum_prior_level_events_for_applicability < 4:
            raise ValueError("Gross-margin applicability requires prior Level history")
        if self.minimum_prior_delta_events_for_applicability < 2:
            raise ValueError("Gross-margin applicability requires prior Delta history")
        if self.minimum_prior_standardized_innovations_for_applicability < 2:
            raise ValueError("Gross-margin applicability requires prior OOS innovations")


@dataclass(frozen=True)
class FlowFact:
    """One uniquely resolved duration fact used by a direct or derived quarter."""

    fact_id: str
    taxonomy: str
    tag: str
    unit: str
    value: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    fiscal_year: int | None
    accepted_at: pd.Timestamp
    source_snapshot_sha256: str
    priority: int


@dataclass(frozen=True)
class FlowPair:
    """Revenue with either direct gross profit or total cost."""

    revenue: FlowFact
    component: FlowFact
    method: str


@dataclass(frozen=True)
class PilotFilingFeatureResult:
    """Immutable pilot feature bundle."""

    output_root: Path
    report: dict[str, Any]


def build_pilot_filing_features(
    readiness_root: Path,
    catalog_root: Path,
    companyfacts_snapshot_manifest: Path,
    *,
    output_root: Path = Path("data/processed/pilot_filing_features"),
    config: PilotFilingFeatureConfig | None = None,
) -> PilotFilingFeatureResult:
    """Build issuer histories, target features, and strict OOS innovations."""
    cfg = config or PilotFilingFeatureConfig()
    cfg.validate()
    readiness_report = _read_json(readiness_root / "price_readiness_report.json")
    readiness_manifest_sha256 = _sha256_file(readiness_root / "output_manifest.json")
    catalog_report = _read_json(catalog_root / "catalog_report.json")
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    companyfacts_manifest = _read_json(companyfacts_snapshot_manifest)
    companyfacts_sha256 = str(companyfacts_manifest["sha256"])
    companyfacts_path = Path(str(companyfacts_manifest["path"]))
    if not companyfacts_path.is_file() or _sha256_file(companyfacts_path) != companyfacts_sha256:
        raise FilingCatalogError("Companyfacts ZIP does not match its immutable manifest")
    readiness = load_catalog_table(readiness_root, "sample_price_readiness.parquet")
    targets = readiness.loc[readiness["return_study_eligible"].astype(bool)].copy()
    if len(targets) != int(readiness_report["return_study_eligible"]):
        raise FilingCatalogError("Pilot feature targets do not match price readiness")
    target_ciks = targets["cik"].astype(str).str.zfill(10).sort_values().unique().tolist()
    filings = pd.read_parquet(
        catalog_root / "filing_catalog.parquet",
        filters=[("cik", "in", target_ciks)],
    )
    filings = filings.loc[filings["form"].isin(["10-Q", "10-Q/A", "10-K", "10-K/A"])].copy()
    if set(targets["accession_number"]).difference(filings["accession_number"]):
        raise FilingCatalogError("A target accession is absent from the SEC filing catalog")
    issuer_registry = pd.read_parquet(
        catalog_root / "issuer_registry_snapshot.parquet",
        filters=[("cik", "in", target_ciks)],
    )
    payload = json.dumps(
        {
            "readiness_manifest_sha256": readiness_manifest_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "companyfacts_sha256": companyfacts_sha256,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return PilotFilingFeatureResult(
            target_root.resolve(), _read_json(target_root / "pilot_filing_feature_report.json")
        )
    if target_root.exists():
        raise FilingCatalogError(f"Incomplete pilot feature directory exists: {target_root}")

    event_frames: list[pd.DataFrame] = []
    selection_audits: list[pd.DataFrame] = []
    normalization_rejections: list[pd.DataFrame] = []
    company_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(companyfacts_path) as archive:
        members = set(archive.namelist())
        for cik in target_ciks:
            member = f"CIK{cik}.json"
            if member not in members:
                raise FilingCatalogError(f"Companyfacts snapshot lacks {member}")
            document = json.loads(archive.read(member))
            company_filings = filings.loc[filings["cik"].astype(str).eq(cik)].copy()
            parsed = parse_companyfacts(
                document,
                company_filings,
                source_url=f"{companyfacts_manifest['source_url']}#{member}",
                snapshot_hash=companyfacts_sha256,
            )
            company_target = targets.loc[targets["cik"].astype(str).str.zfill(10).eq(cik)].iloc[0]
            issuer = issuer_registry.loc[issuer_registry["cik"].astype(str).eq(cik)]
            company_name = (
                str(issuer.iloc[0]["company_name_snapshot"])
                if not issuer.empty
                else str(company_target["registrant_name"])
            )
            events, audit = build_company_quarterly_history(
                cik=cik,
                ticker=str(company_target["selected_ticker"]),
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
                    "ticker": str(company_target["selected_ticker"]),
                    "periodic_filings": len(company_filings),
                    "normalized_facts": len(parsed.data),
                    "normalization_rejections": len(parsed.rejections),
                    "quarterly_events_before_lags": len(events),
                }
            )
    history = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    if history.empty:
        raise FilingCatalogError("No quarterly PIT events could be constructed")
    history = derive_point_in_time_yoy_lags(history)
    history = attach_company_history_features(history, cfg)
    history = attach_oos_innovations(history, cfg)
    history["pilot_feature_version"] = dataset_version
    history = history.sort_values(
        ["actual_available_at", "company_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    target_features, target_audit = audit_target_feature_coverage(targets, history, cfg)
    selection_audit = (
        pd.concat(selection_audits, ignore_index=True) if selection_audits else pd.DataFrame()
    )
    fact_rejections = (
        pd.concat(normalization_rejections, ignore_index=True)
        if normalization_rejections
        else pd.DataFrame()
    )
    company_coverage = pd.DataFrame(company_rows)
    return _persist_feature_bundle(
        target_root=target_root,
        output_root=output_root,
        history=history,
        target_features=target_features,
        target_audit=target_audit,
        selection_audit=selection_audit,
        fact_rejections=fact_rejections,
        company_coverage=company_coverage,
        dataset_version=dataset_version,
        readiness_report=readiness_report,
        catalog_report=catalog_report,
        readiness_manifest_sha256=readiness_manifest_sha256,
        catalog_manifest_sha256=catalog_manifest_sha256,
        companyfacts_sha256=companyfacts_sha256,
        config=cfg,
    )


def build_company_quarterly_history(
    *,
    cik: str,
    ticker: str,
    company_name: str,
    filings: pd.DataFrame,
    facts: pd.DataFrame,
    dataset_version: str,
    config: PilotFilingFeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct Q1-Q4 flow events for one issuer from original disclosures."""
    cfg = config or PilotFilingFeatureConfig()
    cfg.validate()
    normalized_filings = filings.copy()
    normalized_filings["report_date"] = pd.to_datetime(
        normalized_filings["report_date"], errors="coerce"
    )
    normalized_filings["filing_date"] = pd.to_datetime(
        normalized_filings["filing_date"], errors="coerce"
    )
    normalized_filings["accepted_at"] = pd.to_datetime(
        normalized_filings["accepted_at"], utc=True, errors="coerce"
    )
    normalized_filings = normalized_filings.loc[
        normalized_filings["report_date"].dt.year.between(cfg.start_year, cfg.end_year)
        & normalized_filings["accepted_at"].notna()
    ].copy()
    normalized_facts = _prepare_facts(facts)
    cleaner = clean_company_quarterly_events(
        {
            "sec_cik": cik,
            "sec_ticker": ticker,
            "company_name": company_name,
            "country": "US",
            "fortune_rank": None,
        },
        normalized_filings,
        normalized_facts,
        core_asset_version=dataset_version,
        company_profile={},
        config=EventCleaningConfig(
            start_year=cfg.start_year,
            end_year=cfg.end_year,
            min_duration_days=cfg.quarter_min_duration_days,
            max_duration_days=cfg.quarter_max_duration_days,
            min_gross_margin=cfg.min_gross_margin,
            max_gross_margin=cfg.max_gross_margin,
        ),
    )
    q1_q3 = _standardize_base_events(cleaner.events)
    q4, q4_audit = _build_q4_events(
        cik=cik,
        ticker=ticker,
        company_name=company_name,
        filings=normalized_filings,
        facts=normalized_facts,
        dataset_version=dataset_version,
        config=cfg,
    )
    events = pd.concat([q1_q3, q4], ignore_index=True)
    if events.empty:
        return events, q4_audit
    events = events.sort_values(
        ["actual_available_at", "filing_accession"], kind="mergesort"
    ).drop_duplicates("filing_accession", keep="first")
    q_audit = cleaner.selection_audit.copy()
    if not q_audit.empty:
        q_audit["event_family"] = "reported_q1_q3"
    audit = pd.concat([q_audit, q4_audit], ignore_index=True, sort=False)
    return events.reset_index(drop=True), audit


def _standardize_base_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_history()
    output = events[
        [
            "event_id",
            "core_asset_version",
            "company_id",
            "company_name",
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "period_start",
            "period_end",
            "filing_accession",
            "filing_date",
            "filing_accepted_at",
            "actual_available_at",
            "revenue",
            "gross_profit",
            "gross_margin",
            "gm_level",
            "gm_yoy_lag",
            "gm_yoy_lag_event_id",
            "gm_yoy_lag_available_at",
            "gm_delta_yoy",
            "revenue_fact_id",
            "gross_profit_fact_id",
            "cost_fact_id",
            "unit",
            "derivation_method",
            "plausibility_status",
            "analysis_ready_level",
            "analysis_ready_delta",
        ]
    ].copy()
    output["filing_form"] = "10-Q"
    output["q4_source_method"] = "not_applicable"
    output["ytd_filing_accession"] = pd.NA
    output["ytd_available_at"] = pd.NaT
    output["ytd_revenue_fact_id"] = pd.NA
    output["ytd_component_fact_id"] = pd.NA
    output["annual_revenue_fact_id"] = pd.NA
    output["annual_component_fact_id"] = pd.NA
    output["tag_bridge_evidence"] = pd.NA
    return output.reindex(columns=_history_columns())


def _build_q4_events(
    *,
    cik: str,
    ticker: str,
    company_name: str,
    filings: pd.DataFrame,
    facts: pd.DataFrame,
    dataset_version: str,
    config: PilotFilingFeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_10k = filings.loc[filings["form"].eq("10-K")].sort_values(
        ["accepted_at", "accession_number"], kind="mergesort"
    )
    original_10q = filings.loc[filings["form"].eq("10-Q")].copy()
    original_facts = facts.loc[~facts["is_amendment"].astype(bool)].copy()
    event_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for filing in original_10k.itertuples(index=False):
        accession = str(filing.accession_number)
        report_date = pd.Timestamp(str(filing.report_date))
        accepted_at = pd.Timestamp(str(filing.accepted_at))
        direct_candidates = original_facts.loc[
            original_facts["accession_number"].eq(accession)
            & original_facts["period_end"].eq(report_date)
            & original_facts["duration_days"].between(
                config.quarter_min_duration_days,
                config.quarter_max_duration_days,
                inclusive="both",
            )
        ]
        direct_pair = _select_flow_pair(direct_candidates)
        revenue_bridges = _validated_asof_tag_bridges(original_facts, accepted_at, REVENUE_TAGS)
        profit_bridges = _validated_asof_tag_bridges(
            original_facts, accepted_at, DIRECT_GROSS_PROFIT_TAGS
        )
        cost_bridges = _validated_asof_tag_bridges(original_facts, accepted_at, TOTAL_COST_TAGS)
        q3_filing = _select_prior_q3_filing(
            original_10q,
            report_date=report_date,
            accepted_at=accepted_at,
            config=config,
        )
        derived: dict[str, Any] | None = None
        if direct_pair is not None:
            derived = _direct_q4_values(direct_pair)
            source_method = "direct_10k_q4_fact"
        elif q3_filing is not None:
            annual_candidates = original_facts.loc[
                original_facts["accession_number"].eq(accession)
                & original_facts["period_end"].eq(report_date)
                & original_facts["duration_days"].between(
                    config.annual_min_duration_days,
                    config.annual_max_duration_days,
                    inclusive="both",
                )
            ]
            ytd_candidates = original_facts.loc[
                original_facts["accession_number"].eq(str(q3_filing["accession_number"]))
                & original_facts["period_end"].eq(pd.Timestamp(q3_filing["report_date"]))
                & original_facts["duration_days"].between(
                    config.ytd_min_duration_days,
                    config.ytd_max_duration_days,
                    inclusive="both",
                )
            ]
            derived = _annual_minus_ytd_values(
                annual_candidates,
                ytd_candidates,
                revenue_tag_bridges=revenue_bridges,
                profit_tag_bridges=profit_bridges,
                cost_tag_bridges=cost_bridges,
            )
            source_method = "annual_minus_original_nine_months"
        else:
            source_method = "missing_prior_q3_filing"
        status = "accepted" if derived is not None else "rejected"
        reason = source_method if derived is not None else "no_strict_q4_fact_pair"
        audit_rows.append(
            {
                "core_asset_version": dataset_version,
                "company_id": cik,
                "ticker": ticker,
                "filing_accession": accession,
                "report_date": report_date,
                "filing_accepted_at": accepted_at,
                "event_family": "q4_from_10k",
                "selection_status": status,
                "selection_reason": reason,
                "q4_source_method": source_method,
                "prior_q3_accession": (
                    str(q3_filing["accession_number"]) if q3_filing is not None else None
                ),
                "tag_bridge_evidence": (
                    derived.get("tag_bridge_evidence") if derived is not None else None
                ),
            }
        )
        if derived is None:
            continue
        revenue = float(derived["revenue"])
        gross_profit = float(derived["gross_profit"])
        gross_margin = gross_profit / revenue if revenue > 0.0 else math.nan
        fiscal_year = int(derived["fiscal_year"])
        plausibility = (
            "within_bounds"
            if config.min_gross_margin <= gross_margin <= config.max_gross_margin
            else "outlier_review_required"
        )
        analysis_ready = bool(
            revenue > 0.0 and np.isfinite(gross_profit) and plausibility == "within_bounds"
        )
        event_rows.append(
            {
                "event_id": _stable_id(
                    {
                        "company_id": cik,
                        "filing_accession": accession,
                        "dataset_version": dataset_version,
                    }
                ),
                "core_asset_version": dataset_version,
                "company_id": cik,
                "company_name": company_name,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": "Q4",
                "period_start": pd.Timestamp(derived["quarter_start"]),
                "period_end": report_date,
                "filing_accession": accession,
                "filing_date": pd.Timestamp(str(filing.filing_date)),
                "filing_accepted_at": accepted_at,
                "actual_available_at": accepted_at,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "gross_margin": gross_margin,
                "gm_level": gross_margin,
                "gm_yoy_lag": np.nan,
                "gm_yoy_lag_event_id": None,
                "gm_yoy_lag_available_at": pd.NaT,
                "gm_delta_yoy": np.nan,
                "revenue_fact_id": str(derived["revenue_fact_id"]),
                "gross_profit_fact_id": derived["gross_profit_fact_id"],
                "cost_fact_id": derived["cost_fact_id"],
                "unit": str(derived["unit"]),
                "derivation_method": str(derived["derivation_method"]),
                "plausibility_status": plausibility,
                "analysis_ready_level": analysis_ready,
                "analysis_ready_delta": False,
                "filing_form": "10-K",
                "q4_source_method": source_method,
                "ytd_filing_accession": (
                    str(q3_filing["accession_number"])
                    if q3_filing is not None and source_method.startswith("annual_minus")
                    else None
                ),
                "ytd_available_at": (
                    pd.Timestamp(q3_filing["accepted_at"])
                    if q3_filing is not None and source_method.startswith("annual_minus")
                    else pd.NaT
                ),
                "ytd_revenue_fact_id": derived.get("ytd_revenue_fact_id"),
                "ytd_component_fact_id": derived.get("ytd_component_fact_id"),
                "annual_revenue_fact_id": derived.get("annual_revenue_fact_id"),
                "annual_component_fact_id": derived.get("annual_component_fact_id"),
                "tag_bridge_evidence": derived.get("tag_bridge_evidence"),
            }
        )
    return (
        pd.DataFrame(event_rows).reindex(columns=_history_columns()),
        pd.DataFrame(audit_rows),
    )


def _prepare_facts(facts: pd.DataFrame) -> pd.DataFrame:
    result = facts.copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    result["accepted_at"] = pd.to_datetime(result["accepted_at"], utc=True, errors="coerce")
    result["duration_days"] = pd.to_numeric(result["duration_days"], errors="coerce")
    result["value_numeric"] = pd.to_numeric(result["value_numeric"], errors="coerce")
    result["is_amendment"] = result["is_amendment"].astype(str).str.lower().eq("true")
    return result.loc[
        result["tag"].isin(FLOW_TAGS)
        & result["value_numeric"].notna()
        & result["period_start"].notna()
        & result["period_end"].notna()
        & result["accepted_at"].notna()
    ].copy()


def _select_prior_q3_filing(
    original_10q: pd.DataFrame,
    *,
    report_date: pd.Timestamp,
    accepted_at: pd.Timestamp,
    config: PilotFilingFeatureConfig,
) -> pd.Series | None:
    candidates = original_10q.loc[
        (original_10q["report_date"] < report_date) & (original_10q["accepted_at"] < accepted_at)
    ].copy()
    candidates["gap_days"] = (report_date - candidates["report_date"]).dt.days
    candidates = candidates.loc[
        candidates["gap_days"].between(
            config.q3_to_year_end_min_days,
            config.q3_to_year_end_max_days,
            inclusive="both",
        )
    ].sort_values(["report_date", "accepted_at", "accession_number"])
    if candidates.empty:
        return None
    latest_report = candidates["report_date"].max()
    latest = candidates.loc[candidates["report_date"].eq(latest_report)]
    if len(latest) != 1:
        return None
    return latest.iloc[0]


def _flow_choices(candidates: pd.DataFrame, tags: tuple[str, ...]) -> list[FlowFact]:
    choices: list[FlowFact] = []
    for priority, tag in enumerate(tags):
        tagged = candidates.loc[candidates["tag"].eq(tag)]
        for (taxonomy, unit), rows in tagged.groupby(["taxonomy", "unit"], sort=True):
            duration = pd.to_numeric(rows["duration_days"], errors="coerce")
            if duration.isna().all():
                continue
            shortest = float(duration.min())
            resolved = rows.loc[duration.eq(shortest)].sort_values("fact_id")
            values = resolved["value_numeric"].dropna().astype(float).unique()
            periods = resolved[["period_start", "period_end"]].drop_duplicates()
            if len(values) != 1 or len(periods) != 1:
                continue
            row = resolved.iloc[0]
            choices.append(
                FlowFact(
                    fact_id=str(row["fact_id"]),
                    taxonomy=str(taxonomy),
                    tag=tag,
                    unit=str(unit),
                    value=float(values[0]),
                    period_start=pd.Timestamp(row["period_start"]),
                    period_end=pd.Timestamp(row["period_end"]),
                    fiscal_year=_optional_int(row["fiscal_year"]),
                    accepted_at=pd.Timestamp(row["accepted_at"]),
                    source_snapshot_sha256=str(row["source_snapshot_sha256"]),
                    priority=priority,
                )
            )
    return choices


def _select_flow_pair(candidates: pd.DataFrame) -> FlowPair | None:
    revenues = _flow_choices(candidates, REVENUE_TAGS)
    profits = _flow_choices(candidates, DIRECT_GROSS_PROFIT_TAGS)
    costs = _flow_choices(candidates, TOTAL_COST_TAGS)
    direct = [
        FlowPair(revenue, profit, "direct_gross_profit")
        for revenue in revenues
        for profit in profits
        if _same_flow_period(revenue, profit)
    ]
    if direct:
        return min(direct, key=_flow_pair_score)
    derived = [
        FlowPair(revenue, cost, "revenue_minus_total_cost")
        for revenue in revenues
        for cost in costs
        if _same_flow_period(revenue, cost)
    ]
    return min(derived, key=_flow_pair_score) if derived else None


def _same_flow_period(left: FlowFact, right: FlowFact) -> bool:
    return bool(
        left.unit == right.unit
        and left.period_start == right.period_start
        and left.period_end == right.period_end
    )


def _flow_pair_score(pair: FlowPair) -> tuple[int, int, str, str]:
    return (
        pair.revenue.priority,
        pair.component.priority,
        pair.revenue.fact_id,
        pair.component.fact_id,
    )


def _direct_q4_values(pair: FlowPair) -> dict[str, Any] | None:
    if pair.method == "direct_gross_profit":
        gross_profit = pair.component.value
        profit_id: str | None = pair.component.fact_id
        cost_id: str | None = None
    else:
        gross_profit = pair.revenue.value - pair.component.value
        profit_id = None
        cost_id = pair.component.fact_id
    fiscal_year = pair.revenue.fiscal_year
    if fiscal_year is None:
        return None
    return {
        "revenue": pair.revenue.value,
        "gross_profit": gross_profit,
        "fiscal_year": fiscal_year,
        "quarter_start": pair.revenue.period_start,
        "unit": pair.revenue.unit,
        "derivation_method": pair.method,
        "revenue_fact_id": pair.revenue.fact_id,
        "gross_profit_fact_id": profit_id,
        "cost_fact_id": cost_id,
    }


def _annual_minus_ytd_values(
    annual_candidates: pd.DataFrame,
    ytd_candidates: pd.DataFrame,
    *,
    revenue_tag_bridges: dict[tuple[str, str], tuple[str, str]] | None = None,
    profit_tag_bridges: dict[tuple[str, str], tuple[str, str]] | None = None,
    cost_tag_bridges: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    revenue_bridges = revenue_tag_bridges or {}
    annual_revenues = _flow_choices(annual_candidates, REVENUE_TAGS)
    ytd_revenues = _flow_choices(ytd_candidates, REVENUE_TAGS)
    revenue_matches = [
        (annual, ytd)
        for annual in annual_revenues
        for ytd in ytd_revenues
        if _same_semantic_flow(annual, ytd, revenue_bridges)
    ]
    for component_tags, method, component_bridges in (
        (
            DIRECT_GROSS_PROFIT_TAGS,
            "direct_gross_profit",
            profit_tag_bridges or {},
        ),
        (TOTAL_COST_TAGS, "revenue_minus_total_cost", cost_tag_bridges or {}),
    ):
        annual_components = _flow_choices(annual_candidates, component_tags)
        ytd_components = _flow_choices(ytd_candidates, component_tags)
        component_matches = [
            (annual, ytd)
            for annual in annual_components
            for ytd in ytd_components
            if _same_semantic_flow(annual, ytd, component_bridges)
        ]
        candidates: list[tuple[tuple[int, int, str, str], dict[str, Any]]] = []
        for annual_revenue, ytd_revenue in revenue_matches:
            for annual_component, ytd_component in component_matches:
                if (
                    annual_revenue.unit != annual_component.unit
                    or annual_revenue.period_start != annual_component.period_start
                    or ytd_revenue.period_start != ytd_component.period_start
                ):
                    continue
                q4_revenue = annual_revenue.value - ytd_revenue.value
                q4_component = annual_component.value - ytd_component.value
                q4_profit = (
                    q4_component if method == "direct_gross_profit" else q4_revenue - q4_component
                )
                if q4_revenue <= 0.0 or not np.isfinite(q4_profit):
                    continue
                fiscal_year = annual_revenue.fiscal_year
                if fiscal_year is None:
                    continue
                payload = {
                    "revenue": q4_revenue,
                    "gross_profit": q4_profit,
                    "fiscal_year": fiscal_year,
                    "quarter_start": ytd_revenue.period_end + pd.Timedelta(days=1),
                    "unit": annual_revenue.unit,
                    "derivation_method": method,
                    "revenue_fact_id": _derived_fact_id(
                        annual_revenue.fact_id, ytd_revenue.fact_id
                    ),
                    "gross_profit_fact_id": (
                        _derived_fact_id(annual_component.fact_id, ytd_component.fact_id)
                        if method == "direct_gross_profit"
                        else None
                    ),
                    "cost_fact_id": (
                        _derived_fact_id(annual_component.fact_id, ytd_component.fact_id)
                        if method == "revenue_minus_total_cost"
                        else None
                    ),
                    "annual_revenue_fact_id": annual_revenue.fact_id,
                    "annual_component_fact_id": annual_component.fact_id,
                    "ytd_revenue_fact_id": ytd_revenue.fact_id,
                    "ytd_component_fact_id": ytd_component.fact_id,
                    "tag_bridge_evidence": _tag_bridge_evidence_json(
                        annual_revenue,
                        ytd_revenue,
                        revenue_bridges,
                        annual_component,
                        ytd_component,
                        component_bridges,
                    ),
                }
                score = (
                    annual_revenue.priority,
                    annual_component.priority,
                    annual_revenue.fact_id,
                    annual_component.fact_id,
                )
                candidates.append((score, payload))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    return None


def _same_semantic_flow(
    annual: FlowFact,
    ytd: FlowFact,
    tag_bridges: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> bool:
    bridges = tag_bridges or {}
    tags_match = annual.tag == ytd.tag or _tag_pair(annual.tag, ytd.tag) in bridges
    return bool(
        annual.taxonomy == ytd.taxonomy
        and tags_match
        and annual.unit == ytd.unit
        and annual.period_start == ytd.period_start
        and ytd.accepted_at < annual.accepted_at
    )


def _validated_asof_tag_bridges(
    facts: pd.DataFrame,
    cutoff: pd.Timestamp,
    tag_family: tuple[str, ...],
) -> dict[tuple[str, str], tuple[str, str]]:
    history = facts.loc[facts["tag"].isin(tag_family) & (facts["accepted_at"] < cutoff)].copy()
    bridges: dict[tuple[str, str], tuple[str, str]] = {}
    grouping = [
        "taxonomy",
        "unit",
        "period_start",
        "period_end",
        "value_numeric",
    ]
    for _, equivalent in history.groupby(grouping, dropna=False, sort=True):
        tags = sorted(equivalent["tag"].astype(str).unique())
        for left_tag, right_tag in itertools.combinations(tags, 2):
            left = equivalent.loc[equivalent["tag"].eq(left_tag)].sort_values("fact_id")
            right = equivalent.loc[equivalent["tag"].eq(right_tag)].sort_values("fact_id")
            evidence = (str(left.iloc[0]["fact_id"]), str(right.iloc[0]["fact_id"]))
            pair = _tag_pair(left_tag, right_tag)
            if pair not in bridges or evidence < bridges[pair]:
                bridges[pair] = evidence
    return bridges


def _tag_bridge_evidence_json(
    annual_revenue: FlowFact,
    ytd_revenue: FlowFact,
    revenue_bridges: dict[tuple[str, str], tuple[str, str]],
    annual_component: FlowFact,
    ytd_component: FlowFact,
    component_bridges: dict[tuple[str, str], tuple[str, str]],
) -> str | None:
    evidence: list[dict[str, Any]] = []
    for family, annual, ytd, bridges in (
        ("revenue", annual_revenue, ytd_revenue, revenue_bridges),
        ("component", annual_component, ytd_component, component_bridges),
    ):
        if annual.tag == ytd.tag:
            continue
        pair = _tag_pair(annual.tag, ytd.tag)
        fact_ids = bridges.get(pair)
        if fact_ids is None:
            raise FilingCatalogError("Cross-tag flow match lacks prior as-of evidence")
        evidence.append(
            {
                "family": family,
                "annual_tag": annual.tag,
                "ytd_tag": ytd.tag,
                "prior_evidence_fact_ids": list(fact_ids),
            }
        )
    return json.dumps(evidence, sort_keys=True) if evidence else None


def _tag_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _derived_fact_id(annual_fact_id: str, ytd_fact_id: str) -> str:
    return "derived-q4-" + _stable_id(
        {"annual_fact_id": annual_fact_id, "ytd_fact_id": ytd_fact_id}
    )


def attach_company_history_features(
    events: pd.DataFrame,
    config: PilotFilingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Attach only strictly prior company trajectory features."""
    cfg = config or PilotFilingFeatureConfig()
    cfg.validate()
    result = events.copy()
    result["period_end"] = pd.to_datetime(result["period_end"], errors="raise")
    result["actual_available_at"] = pd.to_datetime(
        result["actual_available_at"], utc=True, errors="raise"
    )
    result["company_gm_lag1"] = np.nan
    result["company_gm_lag1_available_at"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["company_gm_delta_yoy_lag1"] = np.nan
    result["company_trailing_gm_mean"] = np.nan
    result["company_trailing_gm_volatility"] = np.nan
    result["company_trailing_gm_history_end"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["revenue_growth_lag1"] = np.nan
    result["forecast_feature_cutoff"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    revenue_by_event = result.set_index("event_id")["revenue"]
    for index, event in result.iterrows():
        history = result.loc[
            result["company_id"].eq(event["company_id"])
            & (result["period_end"] < event["period_end"])
            & (result["actual_available_at"] < event["actual_available_at"])
            & result["analysis_ready_level"].astype(bool)
        ].copy()
        if history.empty:
            continue
        history["period_gap_days"] = event["period_end"] - history["period_end"]
        history["period_gap_days"] = history["period_gap_days"].dt.days
        prior = history.loc[
            history["period_gap_days"].between(45, 150, inclusive="both")
        ].sort_values(["period_end", "actual_available_at", "event_id"], kind="mergesort")
        cutoff_values: list[pd.Timestamp] = []
        if not prior.empty:
            previous = prior.iloc[-1]
            previous_available = pd.Timestamp(previous["actual_available_at"])
            result.at[index, "company_gm_lag1"] = float(previous["gross_margin"])
            result.at[index, "company_gm_lag1_available_at"] = previous_available
            result.at[index, "company_gm_delta_yoy_lag1"] = previous["gm_delta_yoy"]
            lag_event_id = previous["gm_yoy_lag_event_id"]
            if pd.notna(lag_event_id) and str(lag_event_id) in revenue_by_event.index:
                lag_revenue = float(revenue_by_event.loc[str(lag_event_id)])
                prior_revenue = float(previous["revenue"])
                if lag_revenue > 0.0:
                    result.at[index, "revenue_growth_lag1"] = prior_revenue / lag_revenue - 1.0
            cutoff_values.append(previous_available)
        trailing = history.sort_values(["actual_available_at", "event_id"], kind="mergesort").tail(
            cfg.trailing_history_events
        )
        if len(trailing) >= cfg.minimum_trailing_events:
            margins = trailing["gross_margin"].astype(float)
            result.at[index, "company_trailing_gm_mean"] = float(margins.mean())
            result.at[index, "company_trailing_gm_volatility"] = float(margins.std(ddof=1))
            trailing_end = pd.Timestamp(trailing["actual_available_at"].max())
            result.at[index, "company_trailing_gm_history_end"] = trailing_end
            cutoff_values.append(trailing_end)
        yoy_available = pd.to_datetime(
            pd.Series([event["gm_yoy_lag_available_at"]]), utc=True, errors="coerce"
        ).iloc[0]
        if pd.notna(yoy_available):
            cutoff_values.append(pd.Timestamp(yoy_available))
        if cutoff_values:
            cutoff = max(cutoff_values)
            if cutoff >= pd.Timestamp(event["actual_available_at"]):
                raise FilingCatalogError("A pilot forecast feature is not strictly point in time")
            result.at[index, "forecast_feature_cutoff"] = cutoff
    datetime_columns = (
        "company_gm_lag1_available_at",
        "company_trailing_gm_history_end",
        "forecast_feature_cutoff",
    )
    for column in datetime_columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return result


def attach_oos_innovations(
    events: pd.DataFrame,
    config: PilotFilingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Generate pooled OOS forecasts and strictly prior-error standardization."""
    cfg = config or PilotFilingFeatureConfig()
    forecast_input = events.loc[
        events["analysis_ready_delta"].astype(bool)
        & events.loc[:, list(FORECAST_FEATURES)].notna().all(axis="columns")
        & events["forecast_feature_cutoff"].notna()
    ].copy()
    output_columns = (
        "gm_forecast",
        "forecast_model_version",
        "forecast_training_end",
        "forecast_training_observations",
        "forecast_generated_at",
        "forecast_sigma",
        "forecast_error_history_end",
        "forecast_sigma_observations",
        "gm_innovation_raw",
        "gm_innovation_std",
    )
    result = events.copy()
    for column in output_columns:
        result[column] = np.nan
    result["analysis_ready_innovation_raw"] = False
    result["analysis_ready_innovation_std"] = False
    if forecast_input.empty:
        return result
    forecasted = generate_expanding_gm_forecasts(
        forecast_input,
        ForecastConfig(
            feature_columns=FORECAST_FEATURES,
            model_version="gm-pilot-company-history-ols-v1",
            min_train_observations=cfg.minimum_forecast_training_events,
            min_oos_errors=cfg.minimum_prior_oos_errors,
        ),
    )
    lookup = forecasted.set_index("event_id")
    for column in output_columns:
        if column in lookup:
            result[column] = result["event_id"].map(lookup[column])
    result["analysis_ready_innovation_raw"] = result["gm_innovation_raw"].notna()
    result["analysis_ready_innovation_std"] = result["gm_innovation_std"].notna()
    return result


def audit_target_feature_coverage(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    config: PilotFilingFeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join exact target accessions and label every feature-stage failure."""
    cfg = config or PilotFilingFeatureConfig()
    cfg.validate()
    if history["filing_accession"].duplicated().any():
        raise FilingCatalogError("Quarterly history contains duplicate filing accessions")
    target_columns = [
        "target_id",
        "cik",
        "selected_ticker",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "formal_entry_open_at",
        "formal_exit_close_at",
    ]
    history_columns = [
        "company_id",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "period_end",
        "filing_accession",
        "actual_available_at",
        "gm_level",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gm_forecast",
        "forecast_sigma",
        "gm_innovation_raw",
        "gm_innovation_std",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_training_observations",
        "forecast_error_history_end",
        "forecast_sigma_observations",
        "revenue_fact_id",
        "gross_profit_fact_id",
        "cost_fact_id",
        "q4_source_method",
        "ytd_filing_accession",
        "tag_bridge_evidence",
        "analysis_ready_level",
        "analysis_ready_delta",
        "analysis_ready_innovation_raw",
        "analysis_ready_innovation_std",
        "pilot_feature_version",
    ]
    joined = targets[target_columns].merge(
        history[history_columns],
        left_on="accession_number",
        right_on="filing_accession",
        how="left",
        validate="one_to_one",
    )
    expected_time = pd.to_datetime(joined["accepted_at"], utc=True, errors="coerce")
    observed_time = pd.to_datetime(joined["actual_available_at"], utc=True, errors="coerce")
    joined["quarterly_feature_event_match"] = joined["filing_accession"].notna()
    joined["availability_clock_match"] = joined["quarterly_feature_event_match"] & expected_time.eq(
        observed_time
    )
    level_ready = joined["analysis_ready_level"].fillna(False).astype(bool)
    delta_ready = joined["analysis_ready_delta"].fillna(False).astype(bool)
    joined["feature_status"] = "no_quarterly_level_event"
    joined.loc[
        joined["quarterly_feature_event_match"] & ~joined["availability_clock_match"],
        "feature_status",
    ] = "availability_clock_mismatch"
    joined.loc[
        joined["availability_clock_match"] & ~level_ready,
        "feature_status",
    ] = "level_not_analysis_ready"
    joined.loc[
        joined["availability_clock_match"] & level_ready & ~delta_ready,
        "feature_status",
    ] = "delta_unavailable"
    joined.loc[
        delta_ready & joined["gm_forecast"].isna(),
        "feature_status",
    ] = "forecast_unavailable"
    joined.loc[
        joined["gm_forecast"].notna() & joined["forecast_sigma"].isna(), "feature_status"
    ] = "forecast_sigma_unavailable"
    joined.loc[joined["gm_innovation_std"].notna(), "feature_status"] = "complete_case_ready"
    joined["company_id"] = joined["company_id"].fillna(joined["cik"].astype(str).str.zfill(10))
    joined["ticker"] = joined["ticker"].fillna(joined["selected_ticker"])
    joined["filing_accession"] = joined["filing_accession"].fillna(joined["accession_number"])
    joined["actual_available_at"] = joined["actual_available_at"].fillna(joined["accepted_at"])
    joined["accession_match"] = joined["filing_accession"].eq(joined["accession_number"])
    joined["prior_level_events"] = 0
    joined["prior_delta_events"] = 0
    joined["prior_standardized_innovations"] = 0
    history_available_at = pd.to_datetime(history["actual_available_at"], utc=True, errors="coerce")
    for index, target in joined.iterrows():
        accepted_at = pd.Timestamp(target["accepted_at"])
        prior = history.loc[
            history["company_id"].eq(str(target["company_id"]))
            & (history_available_at < accepted_at)
        ]
        joined.at[index, "prior_level_events"] = int(
            prior["analysis_ready_level"].astype(bool).sum()
        )
        joined.at[index, "prior_delta_events"] = int(
            prior["analysis_ready_delta"].astype(bool).sum()
        )
        joined.at[index, "prior_standardized_innovations"] = int(
            prior["analysis_ready_innovation_std"].astype(bool).sum()
        )
    joined["gross_margin_history_eligible"] = (
        joined["prior_level_events"].ge(cfg.minimum_prior_level_events_for_applicability)
        & joined["prior_delta_events"].ge(cfg.minimum_prior_delta_events_for_applicability)
        & joined["prior_standardized_innovations"].ge(
            cfg.minimum_prior_standardized_innovations_for_applicability
        )
    )
    joined["gross_margin_applicability_complete_case"] = joined[
        "gross_margin_history_eligible"
    ] & joined["feature_status"].eq("complete_case_ready")
    target_features = joined.rename(
        columns={
            "accession_number": "target_accession_number",
            "filing_accession": "filing_accession",
        }
    )
    audit_columns = [
        "target_id",
        "company_id",
        "ticker",
        "target_accession_number",
        "form",
        "accepted_at",
        "accession_match",
        "quarterly_feature_event_match",
        "availability_clock_match",
        "analysis_ready_level",
        "analysis_ready_delta",
        "analysis_ready_innovation_raw",
        "analysis_ready_innovation_std",
        "prior_level_events",
        "prior_delta_events",
        "prior_standardized_innovations",
        "gross_margin_history_eligible",
        "gross_margin_applicability_complete_case",
        "feature_status",
    ]
    return target_features, target_features[audit_columns].copy()


def _persist_feature_bundle(
    *,
    target_root: Path,
    output_root: Path,
    history: pd.DataFrame,
    target_features: pd.DataFrame,
    target_audit: pd.DataFrame,
    selection_audit: pd.DataFrame,
    fact_rejections: pd.DataFrame,
    company_coverage: pd.DataFrame,
    dataset_version: str,
    readiness_report: dict[str, Any],
    catalog_report: dict[str, Any],
    readiness_manifest_sha256: str,
    catalog_manifest_sha256: str,
    companyfacts_sha256: str,
    config: PilotFilingFeatureConfig,
) -> PilotFilingFeatureResult:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "quarterly_feature_history.parquet": history,
        "target_filing_features.parquet": target_features,
        "target_feature_audit.parquet": target_audit,
        "quarterly_selection_audit.parquet": selection_audit,
        "fact_normalization_rejections.parquet": fact_rejections,
        "company_feature_coverage.parquet": company_coverage,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "pilot_filing_feature_contract.json"
    contract_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    status_counts = target_audit["feature_status"].value_counts().sort_index().to_dict()
    target_count = len(target_features)
    level_ready = int(target_features["analysis_ready_level"].fillna(False).sum())
    delta_ready = int(target_features["analysis_ready_delta"].fillna(False).sum())
    raw_ready = int(target_features["analysis_ready_innovation_raw"].fillna(False).sum())
    std_ready = int(target_features["analysis_ready_innovation_std"].fillna(False).sum())
    applicable = int(target_features["gross_margin_history_eligible"].astype(bool).sum())
    applicable_complete = int(
        target_features["gross_margin_applicability_complete_case"].astype(bool).sum()
    )
    applicable_coverage = applicable_complete / applicable if applicable else 0.0
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_version": config.version,
        "readiness_version": readiness_report["dataset_version"],
        "catalog_version": catalog_report["dataset_version"],
        "readiness_manifest_sha256": readiness_manifest_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "companyfacts_sha256": companyfacts_sha256,
        "target_events": target_count,
        "target_companies": int(target_features["company_id"].nunique()),
        "quarterly_history_events": len(history),
        "history_level_ready": int(history["analysis_ready_level"].sum()),
        "history_delta_ready": int(history["analysis_ready_delta"].sum()),
        "history_oos_forecasts": int(history["gm_forecast"].notna().sum()),
        "history_standardized_innovations": int(history["gm_innovation_std"].notna().sum()),
        "target_accession_matches": int(target_features["accession_match"].sum()),
        "target_quarterly_feature_event_matches": int(
            target_features["quarterly_feature_event_match"].sum()
        ),
        "target_level_ready": level_ready,
        "target_delta_ready": delta_ready,
        "target_raw_innovation_ready": raw_ready,
        "target_standardized_innovation_ready": std_ready,
        "target_complete_case_ready": std_ready,
        "gross_margin_history_eligible_events": applicable,
        "gross_margin_applicability_complete_events": applicable_complete,
        "gross_margin_applicability_complete_coverage": applicable_coverage,
        "gross_margin_applicability_minimum_20_events_pass": applicable_complete >= 20,
        "gross_margin_applicability_80_percent_coverage_pass": (applicable_coverage >= 0.80),
        "gross_margin_applicability_ic_gate_pass": (
            applicable_complete >= 20 and applicable_coverage >= 0.80
        ),
        "target_feature_status_counts": status_counts,
        "return_ic_minimum_20_events_pass": std_ready >= 20,
        "return_ic_minimum_80_percent_coverage_pass": (
            std_ready / target_count >= 0.80 if target_count else False
        ),
        "return_signal_conclusion_allowed": False,
        "historical_industry_features_used": False,
        "q4_derivation_contract": (
            "direct 10-K quarter flow or FY minus strictly earlier original nine-month 10-Q; "
            "cross-tag subtraction requires strictly earlier same-period same-value evidence"
        ),
        "forecast_model_version": "gm-pilot-company-history-ols-v1",
        "artifacts": artifacts,
    }
    report_path = staging / "pilot_filing_feature_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "readiness_manifest_sha256": readiness_manifest_sha256,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "companyfacts_sha256": companyfacts_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target_root)
    return PilotFilingFeatureResult(target_root.resolve(), report)


def _history_columns() -> list[str]:
    return [
        "event_id",
        "core_asset_version",
        "company_id",
        "company_name",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "period_start",
        "period_end",
        "filing_accession",
        "filing_date",
        "filing_accepted_at",
        "actual_available_at",
        "revenue",
        "gross_profit",
        "gross_margin",
        "gm_level",
        "gm_yoy_lag",
        "gm_yoy_lag_event_id",
        "gm_yoy_lag_available_at",
        "gm_delta_yoy",
        "revenue_fact_id",
        "gross_profit_fact_id",
        "cost_fact_id",
        "unit",
        "derivation_method",
        "plausibility_status",
        "analysis_ready_level",
        "analysis_ready_delta",
        "filing_form",
        "q4_source_method",
        "ytd_filing_accession",
        "ytd_available_at",
        "ytd_revenue_fact_id",
        "ytd_component_fact_id",
        "annual_revenue_fact_id",
        "annual_component_fact_id",
        "tag_bridge_evidence",
    ]


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=_history_columns())


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _stable_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
