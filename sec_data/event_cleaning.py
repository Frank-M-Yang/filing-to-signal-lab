"""Point-in-time cleaning for quarterly gross-margin filing events."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

REVENUE_TAGS: Final[tuple[str, ...]] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
)
DIRECT_GROSS_PROFIT_TAGS: Final[tuple[str, ...]] = ("GrossProfit",)
TOTAL_COST_TAGS: Final[tuple[str, ...]] = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
)
REVIEW_COST_TAGS: Final[tuple[str, ...]] = (
    "CostOfGoodsSold",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
)

EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "core_asset_version",
    "fortune_rank",
    "company_id",
    "company_name",
    "ticker",
    "country",
    "sic",
    "sic_description",
    "fiscal_year_end",
    "fiscal_year",
    "fiscal_quarter",
    "period_start",
    "period_end",
    "filing_accession",
    "filing_date",
    "filing_accepted_at",
    "actual_available_at",
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "gross_margin",
    "gm_level",
    "gm_yoy_lag",
    "gm_yoy_lag_event_id",
    "gm_yoy_lag_available_at",
    "gm_delta_yoy",
    "revenue_fact_id",
    "revenue_taxonomy",
    "revenue_tag",
    "gross_profit_fact_id",
    "gross_profit_taxonomy",
    "gross_profit_tag",
    "cost_fact_id",
    "cost_taxonomy",
    "cost_tag",
    "unit",
    "derivation_method",
    "accounting_scope_status",
    "economic_scope_status",
    "plausibility_status",
    "revenue_source_snapshot_sha256",
    "profit_source_snapshot_sha256",
    "revenue_latest",
    "cost_of_revenue_latest",
    "gross_profit_latest",
    "gross_margin_latest",
    "latest_available_at",
    "latest_filing_accession",
    "latest_base_form",
    "latest_is_amendment",
    "revision_source_type",
    "latest_revenue_fact_id",
    "latest_profit_fact_id",
    "latest_cost_fact_id",
    "was_revised",
    "revision_size",
    "analysis_ready_level",
    "analysis_ready_delta",
)

REJECTION_COLUMNS: Final[tuple[str, ...]] = (
    "rejection_id",
    "core_asset_version",
    "company_id",
    "ticker",
    "filing_accession",
    "report_date",
    "filing_accepted_at",
    "reason",
    "detail",
    "rejected_at_stage",
)

AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "core_asset_version",
    "company_id",
    "ticker",
    "filing_accession",
    "report_date",
    "filing_accepted_at",
    "candidate_fact_rows",
    "revenue_candidate_count",
    "direct_profit_candidate_count",
    "total_cost_candidate_count",
    "review_cost_candidate_count",
    "selection_status",
    "selection_reason",
    "selected_revenue_tag",
    "selected_profit_tag",
    "selected_cost_tag",
    "selected_unit",
    "derivation_method",
)


@dataclass(frozen=True)
class EventCleaningConfig:
    """Fixed temporal and plausibility rules for the first cleaning pass."""

    start_year: int = 2016
    end_year: int = 2025
    min_duration_days: int = 70
    max_duration_days: int = 110
    min_gross_margin: float = -0.50
    max_gross_margin: float = 1.00

    def __post_init__(self) -> None:
        if self.start_year > self.end_year:
            raise ValueError("start_year must not exceed end_year")
        if self.min_duration_days <= 0 or self.min_duration_days >= self.max_duration_days:
            raise ValueError("Quarter-duration bounds are invalid")
        if self.min_gross_margin >= self.max_gross_margin:
            raise ValueError("Gross-margin plausibility bounds are invalid")


@dataclass(frozen=True)
class CompanyEventCleaningResult:
    """Accepted events plus event-level rejection and selection audit tables."""

    events: pd.DataFrame
    rejections: pd.DataFrame
    selection_audit: pd.DataFrame


@dataclass(frozen=True)
class FactChoice:
    """One uniquely resolved entity-wide fact for a tag, unit, and quarter."""

    fact_id: str
    taxonomy: str
    tag: str
    unit: str
    value: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    accession_number: str
    base_form: str
    is_amendment: bool
    accepted_at: pd.Timestamp
    fiscal_year: int | None
    fiscal_period: str | None
    snapshot_sha256: str
    priority: int


@dataclass(frozen=True)
class FactPair:
    """Resolved revenue and profit/cost facts used for one event."""

    revenue: FactChoice
    profit: FactChoice | None
    cost: FactChoice | None
    method: str
    accounting_scope_status: str


def freeze_core_asset(core: pd.DataFrame, *, frozen_at: str) -> tuple[pd.DataFrame, str]:
    """Return a deterministic 82-company registry with a content version."""
    required = (
        "fortune_rank",
        "company_name",
        "country",
        "sec_cik",
        "sec_ticker",
        "dataset_version",
        "normalized_directory",
    )
    _require_columns(core, required)
    if core.empty or core["sec_cik"].astype(str).duplicated().any():
        raise ValueError("Core asset must contain unique SEC CIKs")
    ordered = core.sort_values(["fortune_rank", "sec_cik"]).reset_index(drop=True).copy()
    version_records = ordered.loc[:, required[:-1]].fillna("").astype(str).to_dict("records")
    encoded = json.dumps(
        version_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    version = f"g500-2026-gm82-{hashlib.sha256(encoded).hexdigest()[:16]}"
    ordered.insert(0, "core_asset_id", range(1, len(ordered) + 1))
    ordered.insert(1, "core_asset_version", version)
    ordered.insert(2, "core_asset_frozen_at", frozen_at)
    ordered.insert(3, "core_asset_status", "active")
    return ordered, version


def clean_company_quarterly_events(
    core_record: Mapping[str, Any],
    filings: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    core_asset_version: str,
    company_profile: Mapping[str, Any] | None = None,
    config: EventCleaningConfig | None = None,
) -> CompanyEventCleaningResult:
    """Select current-quarter original facts without silently imputing values."""
    cfg = config or EventCleaningConfig()
    _require_columns(
        filings,
        ("accession_number", "form", "filing_date", "report_date", "accepted_at"),
    )
    _require_columns(
        facts,
        (
            "fact_id",
            "taxonomy",
            "tag",
            "unit",
            "value_numeric",
            "period_start",
            "period_end",
            "duration_days",
            "fiscal_year",
            "fiscal_period",
            "accession_number",
            "base_form",
            "is_amendment",
            "accepted_at",
            "source_snapshot_sha256",
        ),
    )
    profile = company_profile or {}
    company_id = _normalized_cik(core_record.get("sec_cik"))
    ticker = str(core_record.get("sec_ticker") or "")
    prepared_filings = _prepare_filings(filings, cfg)
    prepared_facts = _prepare_facts(facts)
    original_facts = prepared_facts.loc[
        (prepared_facts["base_form"] == "10-Q") & ~prepared_facts["is_amendment"].astype(bool)
    ].copy()

    event_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    filing_records: list[dict[str, Any]] = [
        {str(key): value for key, value in record.items()}
        for record in prepared_filings.to_dict("records")
    ]
    for filing in filing_records:
        accession = str(filing["accession_number"])
        report_date = pd.Timestamp(str(filing["report_date"]))
        accepted_at = pd.Timestamp(str(filing["accepted_at"]))
        candidates = original_facts.loc[
            (original_facts["accession_number"] == accession)
            & (original_facts["period_end"] == report_date)
            & original_facts["duration_days"].between(
                cfg.min_duration_days,
                cfg.max_duration_days,
                inclusive="both",
            )
        ].copy()
        revenue_choices = _fact_choices(candidates, REVENUE_TAGS)
        direct_choices = _fact_choices(candidates, DIRECT_GROSS_PROFIT_TAGS)
        total_cost_choices = _fact_choices(candidates, TOTAL_COST_TAGS)
        review_cost_choices = _fact_choices(candidates, REVIEW_COST_TAGS)
        pair = _select_fact_pair(
            revenue_choices,
            direct_choices,
            total_cost_choices,
            review_cost_choices,
        )
        audit_base: dict[str, Any] = {
            "core_asset_version": core_asset_version,
            "company_id": company_id,
            "ticker": ticker,
            "filing_accession": accession,
            "report_date": report_date,
            "filing_accepted_at": accepted_at,
            "candidate_fact_rows": len(candidates),
            "revenue_candidate_count": len(revenue_choices),
            "direct_profit_candidate_count": len(direct_choices),
            "total_cost_candidate_count": len(total_cost_choices),
            "review_cost_candidate_count": len(review_cost_choices),
        }
        if pair is None:
            reason = _missing_pair_reason(
                candidates,
                revenue_choices,
                direct_choices,
                total_cost_choices,
                review_cost_choices,
            )
            detail = "No unique same-unit revenue and gross-profit/cost pair"
            rejection_rows.append(
                _rejection_row(
                    core_asset_version=core_asset_version,
                    company_id=company_id,
                    ticker=ticker,
                    accession=accession,
                    report_date=report_date,
                    accepted_at=accepted_at,
                    reason=reason,
                    detail=detail,
                )
            )
            audit_rows.append(
                {
                    **audit_base,
                    "selection_status": "rejected",
                    "selection_reason": reason,
                    "selected_revenue_tag": None,
                    "selected_profit_tag": None,
                    "selected_cost_tag": None,
                    "selected_unit": None,
                    "derivation_method": None,
                }
            )
            continue

        revenue = pair.revenue.value
        cost = pair.cost.value if pair.cost is not None else math.nan
        gross_profit = pair.profit.value if pair.profit is not None else revenue - cost
        gross_margin = gross_profit / revenue if revenue != 0.0 else math.nan
        fiscal_year = pair.revenue.fiscal_year
        fiscal_quarter = pair.revenue.fiscal_period
        validation_reason = _validate_selected_values(
            revenue=revenue,
            cost=cost,
            gross_profit=gross_profit,
            gross_margin=gross_margin,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        if validation_reason is not None:
            rejection_rows.append(
                _rejection_row(
                    core_asset_version=core_asset_version,
                    company_id=company_id,
                    ticker=ticker,
                    accession=accession,
                    report_date=report_date,
                    accepted_at=accepted_at,
                    reason=validation_reason,
                    detail="Selected facts failed accounting or fiscal-period validation",
                )
            )
            audit_rows.append(
                {
                    **audit_base,
                    "selection_status": "rejected",
                    "selection_reason": validation_reason,
                    "selected_revenue_tag": pair.revenue.tag,
                    "selected_profit_tag": pair.profit.tag if pair.profit else None,
                    "selected_cost_tag": pair.cost.tag if pair.cost else None,
                    "selected_unit": pair.revenue.unit,
                    "derivation_method": pair.method,
                }
            )
            continue

        assert fiscal_year is not None
        assert fiscal_quarter is not None
        latest = _latest_pair_values(pair, prepared_facts)
        event_id = _stable_id(
            {
                "company_id": company_id,
                "filing_accession": accession,
                "core_asset_version": core_asset_version,
            }
        )
        plausibility = (
            "within_bounds"
            if cfg.min_gross_margin <= gross_margin <= cfg.max_gross_margin
            else "outlier_review_required"
        )
        economic_scope = _economic_scope_status(profile)
        analysis_ready_level = (
            plausibility == "within_bounds"
            and pair.accounting_scope_status == "high_confidence"
            and economic_scope == "standard_review"
        )
        event_rows.append(
            {
                "event_id": event_id,
                "core_asset_version": core_asset_version,
                "fortune_rank": _optional_int(core_record.get("fortune_rank")),
                "company_id": company_id,
                "company_name": str(core_record.get("company_name") or ""),
                "ticker": ticker,
                "country": str(core_record.get("country") or ""),
                "sic": _optional_int(profile.get("sic")),
                "sic_description": _optional_text(profile.get("sicDescription")),
                "fiscal_year_end": _optional_text(profile.get("fiscalYearEnd")),
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "period_start": pair.revenue.period_start,
                "period_end": pair.revenue.period_end,
                "filing_accession": accession,
                "filing_date": pd.Timestamp(str(filing["filing_date"])),
                "filing_accepted_at": accepted_at,
                "actual_available_at": accepted_at,
                "revenue": revenue,
                "cost_of_revenue": cost if pair.cost is not None else np.nan,
                "gross_profit": gross_profit,
                "gross_margin": gross_margin,
                "gm_level": gross_margin,
                "gm_yoy_lag": np.nan,
                "gm_yoy_lag_event_id": None,
                "gm_yoy_lag_available_at": pd.NaT,
                "gm_delta_yoy": np.nan,
                "revenue_fact_id": pair.revenue.fact_id,
                "revenue_taxonomy": pair.revenue.taxonomy,
                "revenue_tag": pair.revenue.tag,
                "gross_profit_fact_id": pair.profit.fact_id if pair.profit else None,
                "gross_profit_taxonomy": pair.profit.taxonomy if pair.profit else None,
                "gross_profit_tag": pair.profit.tag if pair.profit else None,
                "cost_fact_id": pair.cost.fact_id if pair.cost else None,
                "cost_taxonomy": pair.cost.taxonomy if pair.cost else None,
                "cost_tag": pair.cost.tag if pair.cost else None,
                "unit": pair.revenue.unit,
                "derivation_method": pair.method,
                "accounting_scope_status": pair.accounting_scope_status,
                "economic_scope_status": economic_scope,
                "plausibility_status": plausibility,
                "revenue_source_snapshot_sha256": pair.revenue.snapshot_sha256,
                "profit_source_snapshot_sha256": (
                    pair.profit.snapshot_sha256
                    if pair.profit is not None
                    else pair.cost.snapshot_sha256
                    if pair.cost is not None
                    else None
                ),
                **latest,
                "analysis_ready_level": analysis_ready_level,
                "analysis_ready_delta": False,
            }
        )
        audit_rows.append(
            {
                **audit_base,
                "selection_status": "accepted",
                "selection_reason": pair.accounting_scope_status,
                "selected_revenue_tag": pair.revenue.tag,
                "selected_profit_tag": pair.profit.tag if pair.profit else None,
                "selected_cost_tag": pair.cost.tag if pair.cost else None,
                "selected_unit": pair.revenue.unit,
                "derivation_method": pair.method,
            }
        )

    events = pd.DataFrame(event_rows).reindex(columns=EVENT_COLUMNS)
    rejections = pd.DataFrame(rejection_rows).reindex(columns=REJECTION_COLUMNS)
    audit = pd.DataFrame(audit_rows).reindex(columns=AUDIT_COLUMNS)
    return CompanyEventCleaningResult(events, rejections, audit)


def derive_point_in_time_yoy_lags(events: pd.DataFrame) -> pd.DataFrame:
    """Attach same-fiscal-quarter prior-year margins only when already available."""
    _require_columns(
        events,
        (
            "event_id",
            "company_id",
            "fiscal_year",
            "fiscal_quarter",
            "period_end",
            "actual_available_at",
            "gross_margin",
            "analysis_ready_level",
        ),
    )
    result = events.copy()
    result["actual_available_at"] = pd.to_datetime(
        result["actual_available_at"], utc=True, errors="raise"
    )
    result["gm_yoy_lag_available_at"] = pd.to_datetime(
        result["gm_yoy_lag_available_at"], utc=True, errors="coerce"
    )
    result["period_end"] = pd.to_datetime(result["period_end"], errors="raise")
    result["_row_order"] = np.arange(len(result))
    lag_source = result.loc[
        :,
        [
            "company_id",
            "fiscal_quarter",
            "event_id",
            "period_end",
            "actual_available_at",
            "gross_margin",
            "analysis_ready_level",
        ],
    ].copy()
    lag_source = lag_source.rename(
        columns={
            "event_id": "_lag_event_id",
            "period_end": "_lag_period_end",
            "actual_available_at": "_lag_available_at",
            "gross_margin": "_lag_gross_margin",
            "analysis_ready_level": "_lag_analysis_ready_level",
        }
    )
    candidates = result.loc[
        :,
        ["_row_order", "company_id", "fiscal_quarter", "period_end", "actual_available_at"],
    ].merge(
        lag_source,
        on=["company_id", "fiscal_quarter"],
        how="left",
        validate="many_to_many",
    )
    candidates["_period_gap_days"] = (
        candidates["period_end"] - candidates["_lag_period_end"]
    ).dt.days
    candidates = candidates.loc[
        candidates["_period_gap_days"].between(330, 400, inclusive="both")
        & (candidates["_lag_available_at"] < candidates["actual_available_at"])
    ].copy()
    candidates["_distance_from_year"] = (candidates["_period_gap_days"] - 365).abs()
    candidates["_minimum_distance"] = candidates.groupby("_row_order")[
        "_distance_from_year"
    ].transform("min")
    nearest = candidates.loc[
        candidates["_distance_from_year"] == candidates["_minimum_distance"]
    ].copy()
    unique_nearest = nearest.loc[
        nearest.groupby("_row_order")["_lag_event_id"].transform("size") == 1
    ].drop_duplicates("_row_order")
    selected_lags = unique_nearest.set_index("_row_order")

    result = result.set_index("_row_order")
    lag_index = result.index[result.index.isin(selected_lags.index)]
    result.loc[lag_index, "gm_yoy_lag"] = selected_lags.loc[
        lag_index, "_lag_gross_margin"
    ].to_numpy()
    result.loc[lag_index, "gm_yoy_lag_event_id"] = selected_lags.loc[
        lag_index, "_lag_event_id"
    ].to_numpy()
    result.loc[lag_index, "gm_yoy_lag_available_at"] = selected_lags.loc[
        lag_index, "_lag_available_at"
    ].to_numpy()
    result.loc[lag_index, "gm_delta_yoy"] = (
        result.loc[lag_index, "gross_margin"].to_numpy()
        - selected_lags.loc[lag_index, "_lag_gross_margin"].to_numpy()
    )
    result.loc[lag_index, "analysis_ready_delta"] = (
        result.loc[lag_index, "analysis_ready_level"].astype(bool).to_numpy()
        & selected_lags.loc[lag_index, "_lag_analysis_ready_level"].astype(bool).to_numpy()
    )
    return pd.DataFrame(result.sort_index().reset_index(drop=True))


def validate_cleaned_quarterly_events(events: pd.DataFrame) -> None:
    """Validate accounting identities, provenance, and PIT lag ordering."""
    required = (
        "event_id",
        "company_id",
        "filing_accession",
        "filing_accepted_at",
        "actual_available_at",
        "revenue",
        "gross_profit",
        "gross_margin",
        "gm_yoy_lag",
        "gm_yoy_lag_event_id",
        "gm_yoy_lag_available_at",
        "gm_delta_yoy",
        "latest_available_at",
        "latest_filing_accession",
        "was_revised",
        "analysis_ready_level",
        "analysis_ready_delta",
        "accounting_scope_status",
        "economic_scope_status",
        "plausibility_status",
    )
    _require_columns(events, required)
    if events.empty:
        raise ValueError("Cleaned quarterly events cannot be empty")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("event_id must be non-null and unique")
    if events.duplicated(["company_id", "filing_accession"]).any():
        raise ValueError("A filing accession appears more than once for one company")

    revenue = pd.to_numeric(events["revenue"], errors="coerce")
    gross_profit = pd.to_numeric(events["gross_profit"], errors="coerce")
    gross_margin = pd.to_numeric(events["gross_margin"], errors="coerce")
    if revenue.isna().any() or gross_profit.isna().any() or gross_margin.isna().any():
        raise ValueError("Cleaned accounting values must be finite")
    if (revenue <= 0.0).any():
        raise ValueError("Cleaned revenue must be positive")
    if not np.allclose(
        gross_margin.to_numpy(dtype=float),
        (gross_profit / revenue).to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("gross_margin does not tie to gross_profit/revenue")

    accepted_at = pd.to_datetime(events["filing_accepted_at"], utc=True, errors="coerce")
    actual_at = pd.to_datetime(events["actual_available_at"], utc=True, errors="coerce")
    latest_at = pd.to_datetime(events["latest_available_at"], utc=True, errors="coerce")
    if accepted_at.isna().any() or actual_at.isna().any() or latest_at.isna().any():
        raise ValueError("Availability timestamps must be valid UTC times")
    if not accepted_at.equals(actual_at):
        raise ValueError("actual_available_at must equal the original filing acceptance time")
    if (latest_at < actual_at).any():
        raise ValueError("latest_available_at precedes the PIT fact")

    revised = events["was_revised"].astype(str).str.lower().eq("true")
    if (
        revised
        & (events["latest_filing_accession"].astype(str) == events["filing_accession"].astype(str))
    ).any():
        raise ValueError("A revised event must come from a later coherent filing accession")

    lag_rows = events["gm_yoy_lag"].notna()
    lag_available = pd.to_datetime(
        events.loc[lag_rows, "gm_yoy_lag_available_at"], utc=True, errors="coerce"
    )
    if lag_available.isna().any() or (lag_available >= actual_at.loc[lag_rows]).any():
        raise ValueError("A YoY lag was not available strictly before the current event")
    expected_delta = gross_margin.loc[lag_rows] - pd.to_numeric(
        events.loc[lag_rows, "gm_yoy_lag"], errors="coerce"
    )
    if not np.allclose(
        pd.to_numeric(events.loc[lag_rows, "gm_delta_yoy"], errors="coerce").to_numpy(dtype=float),
        expected_delta.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("gm_delta_yoy does not tie to current and prior PIT margins")

    ready_level = events["analysis_ready_level"].astype(str).str.lower().eq("true")
    invalid_ready_level = ready_level & (
        events["accounting_scope_status"].ne("high_confidence")
        | events["economic_scope_status"].ne("standard_review")
        | events["plausibility_status"].ne("within_bounds")
    )
    if invalid_ready_level.any():
        raise ValueError("An analysis-ready level event still requires accounting review")
    ready_delta = events["analysis_ready_delta"].astype(str).str.lower().eq("true")
    if (ready_delta & ~ready_level).any():
        raise ValueError("An analysis-ready delta has an invalid current level")
    lag_ready_by_id = events.set_index("event_id")["analysis_ready_level"].astype(bool)
    ready_delta_lag_ids = events.loc[ready_delta, "gm_yoy_lag_event_id"]
    if ready_delta_lag_ids.isna().any():
        raise ValueError("An analysis-ready delta has no prior-year event")
    lag_ready = ready_delta_lag_ids.astype(str).map(lag_ready_by_id)
    if lag_ready.isna().any() or not lag_ready.all():
        raise ValueError("An analysis-ready delta has an invalid prior-year level")


def _prepare_filings(filings: pd.DataFrame, config: EventCleaningConfig) -> pd.DataFrame:
    result = filings.copy()
    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
    result["filing_date"] = pd.to_datetime(result["filing_date"], errors="coerce")
    result["accepted_at"] = pd.to_datetime(result["accepted_at"], utc=True, errors="coerce")
    result = result.loc[
        (result["form"] == "10-Q")
        & result["report_date"].dt.year.between(config.start_year, config.end_year)
        & result["accepted_at"].notna()
    ].copy()
    result = result.sort_values(["accepted_at", "accession_number"]).drop_duplicates(
        "accession_number", keep="first"
    )
    return pd.DataFrame(result)


def _prepare_facts(facts: pd.DataFrame) -> pd.DataFrame:
    result = facts.copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    result["accepted_at"] = pd.to_datetime(result["accepted_at"], utc=True, errors="coerce")
    result["duration_days"] = pd.to_numeric(result["duration_days"], errors="coerce")
    result["value_numeric"] = pd.to_numeric(result["value_numeric"], errors="coerce")
    result["is_amendment"] = result["is_amendment"].astype(str).str.lower().eq("true")
    return result.loc[
        result["tag"].isin(
            (
                *REVENUE_TAGS,
                *DIRECT_GROSS_PROFIT_TAGS,
                *TOTAL_COST_TAGS,
                *REVIEW_COST_TAGS,
            )
        )
        & result["value_numeric"].notna()
        & result["accepted_at"].notna()
    ].copy()


def _fact_choices(facts: pd.DataFrame, tags: tuple[str, ...]) -> list[FactChoice]:
    choices: list[FactChoice] = []
    for priority, tag in enumerate(tags):
        tagged = facts.loc[facts["tag"] == tag]
        for unit, unit_rows in tagged.groupby("unit", sort=True):
            duration = pd.to_numeric(unit_rows["duration_days"], errors="coerce")
            if duration.isna().all():
                continue
            shortest = float(duration.min())
            resolved = unit_rows.loc[duration == shortest].sort_values("fact_id")
            unique_values = resolved["value_numeric"].dropna().astype(float).unique()
            if len(unique_values) != 1:
                continue
            row = resolved.iloc[0]
            period_start = pd.Timestamp(row["period_start"])
            period_end = pd.Timestamp(row["period_end"])
            if pd.isna(period_start) or pd.isna(period_end):
                continue
            choices.append(
                FactChoice(
                    fact_id=str(row["fact_id"]),
                    taxonomy=str(row["taxonomy"]),
                    tag=tag,
                    unit=str(unit),
                    value=float(unique_values[0]),
                    period_start=period_start,
                    period_end=period_end,
                    accession_number=str(row["accession_number"]),
                    base_form=str(row["base_form"]),
                    is_amendment=bool(row["is_amendment"]),
                    accepted_at=pd.Timestamp(row["accepted_at"]),
                    fiscal_year=_optional_int(row["fiscal_year"]),
                    fiscal_period=_optional_text(row["fiscal_period"]),
                    snapshot_sha256=str(row["source_snapshot_sha256"]),
                    priority=priority,
                )
            )
    return choices


def _select_fact_pair(
    revenues: list[FactChoice],
    profits: list[FactChoice],
    total_costs: list[FactChoice],
    review_costs: list[FactChoice],
) -> FactPair | None:
    direct_pairs = [
        FactPair(revenue, profit, None, "direct_gross_profit", "high_confidence")
        for revenue in revenues
        for profit in profits
        if revenue.unit == profit.unit and revenue.period_start == profit.period_start
    ]
    if direct_pairs:
        return min(direct_pairs, key=_pair_score)
    total_pairs = [
        FactPair(revenue, None, cost, "revenue_minus_total_cost", "high_confidence")
        for revenue in revenues
        for cost in total_costs
        if revenue.unit == cost.unit and revenue.period_start == cost.period_start
    ]
    if total_pairs:
        return min(total_pairs, key=_pair_score)
    review_pairs = [
        FactPair(
            revenue,
            None,
            cost,
            "revenue_minus_scope_sensitive_cost",
            "review_required",
        )
        for revenue in revenues
        for cost in review_costs
        if revenue.unit == cost.unit and revenue.period_start == cost.period_start
    ]
    return min(review_pairs, key=_pair_score) if review_pairs else None


def _pair_score(pair: FactPair) -> tuple[int, int, str]:
    component = pair.profit if pair.profit is not None else pair.cost
    assert component is not None
    return (pair.revenue.priority, component.priority, pair.revenue.fact_id)


def _latest_pair_values(pair: FactPair, facts: pd.DataFrame) -> dict[str, Any]:
    revenue_versions = _fact_versions(pair.revenue, facts)
    profit_versions = _fact_versions(pair.profit, facts) if pair.profit is not None else {}
    cost_versions = _fact_versions(pair.cost, facts) if pair.cost is not None else {}
    component_versions = profit_versions if pair.profit is not None else cost_versions
    common_accessions = set(revenue_versions).intersection(component_versions)
    if common_accessions:
        latest_accession = max(
            common_accessions,
            key=lambda accession: (
                max(
                    revenue_versions[accession].accepted_at,
                    component_versions[accession].accepted_at,
                ),
                accession,
            ),
        )
        latest_revenue = revenue_versions[latest_accession]
        latest_profit = profit_versions.get(latest_accession)
        latest_cost = cost_versions.get(latest_accession)
    else:
        latest_revenue = pair.revenue
        latest_profit = pair.profit
        latest_cost = pair.cost
    revenue_value = latest_revenue.value if latest_revenue is not None else math.nan
    cost_value = latest_cost.value if latest_cost is not None else math.nan
    if latest_profit is not None:
        profit_value = latest_profit.value
    elif latest_cost is not None and latest_revenue is not None:
        profit_value = revenue_value - cost_value
    else:
        profit_value = math.nan
    latest_margin = (
        profit_value / revenue_value
        if np.isfinite(profit_value) and np.isfinite(revenue_value) and revenue_value != 0.0
        else math.nan
    )
    latest_choices = [
        choice for choice in (latest_revenue, latest_profit, latest_cost) if choice is not None
    ]
    latest_available = max((choice.accepted_at for choice in latest_choices), default=pd.NaT)
    original_margin = (
        pair.profit.value / pair.revenue.value
        if pair.profit is not None
        else (pair.revenue.value - pair.cost.value) / pair.revenue.value
        if pair.cost is not None
        else math.nan
    )
    revised = bool(
        np.isfinite(latest_margin)
        and np.isfinite(original_margin)
        and not math.isclose(latest_margin, original_margin, rel_tol=1e-12, abs_tol=1e-12)
    )
    return {
        "revenue_latest": revenue_value,
        "cost_of_revenue_latest": cost_value if latest_cost is not None else np.nan,
        "gross_profit_latest": profit_value,
        "gross_margin_latest": latest_margin,
        "latest_available_at": latest_available,
        "latest_filing_accession": latest_revenue.accession_number,
        "latest_base_form": latest_revenue.base_form,
        "latest_is_amendment": latest_revenue.is_amendment,
        "revision_source_type": (
            "original_same_accession"
            if latest_revenue.accession_number == pair.revenue.accession_number
            else "amendment"
            if latest_revenue.is_amendment
            else "later_comparative_filing"
        ),
        "latest_revenue_fact_id": latest_revenue.fact_id if latest_revenue else None,
        "latest_profit_fact_id": latest_profit.fact_id if latest_profit else None,
        "latest_cost_fact_id": latest_cost.fact_id if latest_cost else None,
        "was_revised": revised,
        "revision_size": latest_margin - original_margin if np.isfinite(latest_margin) else np.nan,
    }


def _fact_versions(choice: FactChoice | None, facts: pd.DataFrame) -> dict[str, FactChoice]:
    if choice is None:
        return {}
    candidates = facts.loc[
        (facts["taxonomy"] == choice.taxonomy)
        & (facts["tag"] == choice.tag)
        & (facts["unit"] == choice.unit)
        & (facts["period_start"] == choice.period_start)
        & (facts["period_end"] == choice.period_end)
        & (facts["accepted_at"] >= choice.accepted_at)
    ].copy()
    versions: dict[str, FactChoice] = {}
    for accession, accession_rows in candidates.groupby("accession_number", sort=True):
        latest_at = pd.Timestamp(accession_rows["accepted_at"].max())
        latest_rows = accession_rows.loc[accession_rows["accepted_at"] == latest_at].sort_values(
            "fact_id"
        )
        values = latest_rows["value_numeric"].dropna().astype(float).unique()
        if len(values) != 1:
            continue
        row = latest_rows.iloc[0]
        versions[str(accession)] = FactChoice(
            fact_id=str(row["fact_id"]),
            taxonomy=str(row["taxonomy"]),
            tag=str(row["tag"]),
            unit=str(row["unit"]),
            value=float(values[0]),
            period_start=pd.Timestamp(row["period_start"]),
            period_end=pd.Timestamp(row["period_end"]),
            accession_number=str(accession),
            base_form=str(row["base_form"]),
            is_amendment=bool(row["is_amendment"]),
            accepted_at=latest_at,
            fiscal_year=_optional_int(row["fiscal_year"]),
            fiscal_period=_optional_text(row["fiscal_period"]),
            snapshot_sha256=str(row["source_snapshot_sha256"]),
            priority=choice.priority,
        )
    if choice.accession_number not in versions:
        versions[choice.accession_number] = choice
    return versions


def _validate_selected_values(
    *,
    revenue: float,
    cost: float,
    gross_profit: float,
    gross_margin: float,
    fiscal_year: int | None,
    fiscal_quarter: str | None,
) -> str | None:
    if not np.isfinite(revenue) or revenue <= 0.0:
        return "nonpositive_or_invalid_revenue"
    if not np.isfinite(gross_profit) or not np.isfinite(gross_margin):
        return "invalid_gross_profit"
    if np.isfinite(cost) and cost < 0.0:
        return "negative_cost_of_revenue"
    if fiscal_year is None or fiscal_quarter not in {"Q1", "Q2", "Q3"}:
        return "invalid_fiscal_period"
    return None


def _missing_pair_reason(
    candidates: pd.DataFrame,
    revenues: list[FactChoice],
    profits: list[FactChoice],
    total_costs: list[FactChoice],
    review_costs: list[FactChoice],
) -> str:
    if candidates.empty:
        return "no_current_quarter_facts"
    if not revenues:
        return "no_unique_revenue_fact"
    if not profits and not total_costs and not review_costs:
        return "no_unique_profit_or_cost_fact"
    return "no_same_unit_same_period_fact_pair"


def _rejection_row(
    *,
    core_asset_version: str,
    company_id: str,
    ticker: str,
    accession: str,
    report_date: pd.Timestamp,
    accepted_at: pd.Timestamp,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    rejection_id = _stable_id(
        {
            "core_asset_version": core_asset_version,
            "company_id": company_id,
            "filing_accession": accession,
            "reason": reason,
        }
    )
    return {
        "rejection_id": rejection_id,
        "core_asset_version": core_asset_version,
        "company_id": company_id,
        "ticker": ticker,
        "filing_accession": accession,
        "report_date": report_date,
        "filing_accepted_at": accepted_at,
        "reason": reason,
        "detail": detail,
        "rejected_at_stage": "quarterly_fact_selection",
    }


def _economic_scope_status(profile: Mapping[str, Any]) -> str:
    sic = _optional_int(profile.get("sic"))
    description = str(profile.get("sicDescription") or "").upper()
    if sic is not None and 6000 <= sic <= 6799:
        return "industry_review_required"
    if any(word in description for word in ("INSURANCE", "SECURITY BROKERS", "BANK")):
        return "industry_review_required"
    return "standard_review"


def _normalized_cik(value: Any) -> str:
    if value is None:
        raise ValueError("Core record is missing sec_cik")
    raw = str(value).split(".")[0].strip()
    if not raw.isdigit() or len(raw) > 10:
        raise ValueError(f"Invalid core CIK: {value}")
    return raw.zfill(10)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _stable_id(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing cleaning columns: {sorted(missing)}")
