"""Separate release-value proof, proxy chronology and exact session coverage; no admission."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp

VERSION = "announcement-admission-v1"
FORECAST_CLOCKS = (
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_generated_at",
    "forecast_error_history_end",
    "gm_yoy_lag_available_at",
    "company_gm_lag1_available_at",
    "company_trailing_gm_history_end",
)


def join_candidate_clocks(events: pd.DataFrame, clocks: pd.DataFrame) -> pd.DataFrame:
    """Keep the full cohort and join by issuer plus accession, never by accession alone."""
    if (
        events.validation_event_id.duplicated().any()
        or clocks.duplicated(["cik", "accession_number"]).any()
    ):
        raise ValueError("Ambiguous event or clock identity")
    selected = clocks.rename(columns={"accession_number": "target_accession_number"})
    result = events.merge(
        selected, on=["cik", "target_accession_number"], how="left", validate="many_to_one"
    )
    result["filing_clock_matches_catalog"] = result.apply(
        lambda r: (
            pd.notna(r.report_accepted_at)
            and utc_timestamp(r.accepted_at) == utc_timestamp(r.report_accepted_at)
        ),
        axis=1,
    )
    if len(result) != len(events):
        raise ValueError("Frozen denominator changed")
    return result


def review_candidate_pair(event: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind the GM pair to the exact selected release/period, not a later matching release."""
    result: dict[str, Any] = {
        "release_value_pair_verified": False,
        "release_gross_margin": None,
        "release_pair_source_sha256": None,
        "release_pair_exhibit_sha256": None,
        "candidate_value_status": "selected_release_unreviewed",
    }
    if event.get("clock_source") != "item_2_02_8k":
        result["candidate_value_status"] = "no_candidate_not_proof_of_first_public"
        return result
    matched = [
        r
        for r in cells
        if r["validation_event_id"] == event["validation_event_id"]
        and r["cik"] == event["cik"]
        and r["release_accession"] == event["release_accession_number"]
    ]
    if not matched:
        return result
    pair = [r for r in matched if r["metric"] in ("revenue", "gross_profit")]
    if len(pair) != 2 or {r["metric"] for r in pair} != {"revenue", "gross_profit"}:
        result["candidate_value_status"] = "reviewed_but_GM_pair_incomplete_or_ambiguous"
        return result
    if any(
        r["period_end"] != event["period_end"] or r["period_end"] != event["report_date"]
        for r in pair
    ):
        result["candidate_value_status"] = "release_period_mismatch"
        return result
    if any(
        r["clock_status"] != "consistent_acceptance_proxy"
        or utc_timestamp(r["sec_acceptance_proxy_at"])
        != utc_timestamp(event["earliest_disclosure_at"])
        for r in pair
    ):
        result["candidate_value_status"] = "release_clock_conflict_requires_reconciliation"
        return result
    if any(
        r["comparison_status"] != "exact_original_match"
        or r["original_status"] != "original_quarter"
        for r in pair
    ):
        result["candidate_value_status"] = "not_direct_original_quarter_match"
        return result
    for key in ("period_start", "period_end", "unit", "source_sha256", "exhibit_sha256"):
        if len({r[key] for r in pair}) != 1 or pair[0][key] is None:
            result["candidate_value_status"] = "incompatible_release_pair"
            return result
    values = {r["metric"]: r["value"] for r in pair}
    if (
        pair[0]["unit"] != "USD"
        or not np.isfinite(list(values.values())).all()
        or values["revenue"] <= 0
    ):
        result["candidate_value_status"] = "invalid_release_pair_values"
        return result
    result.update(
        release_value_pair_verified=True,
        release_gross_margin=values["gross_profit"] / values["revenue"],
        release_pair_source_sha256=pair[0]["source_sha256"],
        release_pair_exhibit_sha256=pair[0]["exhibit_sha256"],
        candidate_value_status="reviewed_direct_quarter_GM_pair_at_selected_proxy",
    )
    return result


def audit_forecast_metadata(metadata: dict[str, Any], candidate_at: str) -> dict[str, Any]:
    """Precheck existing dependency times only; passing does not substitute for history replay."""
    cutoff = utc_timestamp(candidate_at)
    missing = [k for k in FORECAST_CLOCKS if pd.isna(metadata.get(k))]
    failed = [
        k for k in FORECAST_CLOCKS if k not in missing and utc_timestamp(metadata[k]) >= cutoff
    ]
    return {
        "existing_forecast_clocks_before_proxy": not missing and not failed,
        "missing_forecast_clocks": missing,
        "non_prior_forecast_clocks": failed,
        "history_replay_completed": False,
    }


def exact_session_coverage(
    calendar: pd.DataFrame, observed_dates: list[str], candidate_at: str, *, required: int = 61
) -> dict[str, Any]:
    """Use exact official sessions; older bars cannot replace missing interior days."""
    if required <= 0:
        raise ValueError("Positive history length required")
    day = utc_timestamp(candidate_at).tz_convert("America/New_York").date().isoformat()
    if calendar.session_date.duplicated().any() or len(observed_dates) != len(set(observed_dates)):
        raise ValueError("Duplicate session or observed price date")
    prior = calendar.loc[calendar.session_date.lt(day)].sort_values("session_date").tail(required)
    dates = prior.session_date.tolist()
    if any(
        utc_timestamp(t).tz_convert("America/New_York").date().isoformat() != d
        for d, t in zip(prior.session_date, prior.regular_close_at, strict=True)
    ):
        raise ValueError("Calendar date/close disagreement")
    if any(utc_timestamp(t) >= utc_timestamp(candidate_at) for t in prior.regular_close_at):
        raise ValueError("Non-prior official market close")
    missing = sorted(set(dates) - set(observed_dates))
    return {
        "calendar_ready": len(dates) == required,
        "required_dates": dates,
        "missing_dates": missing,
        "history_dates_complete": len(dates) == required and not missing,
        "required_count": required,
    }


def price_query_covers_history(
    event: dict[str, Any], audit: dict[str, Any], dates: list[str]
) -> bool:
    """Require issuer identity, complete parsing and a query covering the exact history."""
    return bool(
        audit
        and dates
        and all(audit.get(k) == event[k] for k in ("price_target_id", "cik", "ticker"))
        and audit.get("provider_query_complete") is True
        and audit.get("parser_rejection_rows") == 0
        and audit["price_window_start"] <= dates[0] <= dates[-1] <= audit["price_window_end"]
    )


def gap_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose every blocker, including benchmark dates and events without a valid clock."""
    date_keys = {
        "issuer_missing_exact_61_session_history": "missing_issuer_dates",
        "benchmark_missing_exact_61_session_history": "missing_benchmark_dates",
        "execution_issuer_dates_missing": "missing_execution_issuer_dates",
        "execution_benchmark_dates_missing": "missing_execution_benchmark_dates",
    }
    return [
        {
            "validation_event_id": row["validation_event_id"],
            "ticker": row["assigned_ticker"],
            "cik": row["cik"],
            "release_accession": row.get("release_accession_number"),
            "gap": code,
            "missing_dates": row.get(date_keys.get(code, ""), []),
            "automatic_download_authorized": False,
        }
        for code in row["blocking_reasons"]
    ]
