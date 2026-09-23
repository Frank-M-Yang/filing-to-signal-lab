"""Disarmed publisher-label sensitivity; never a first-public certification.

Quarter-level evidence does not establish when YTD components became available.
This module deliberately leaves SEC acceptance and Q4/YTD source clocks intact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.clock_layer import NUMERIC_COLUMNS, TIME_COLUMNS
from sec_data.disclosure_audit import utc_timestamp
from sec_data.pilot_filing_features import FORECAST_FEATURES

VERSION = "publisher-replay-v1"
MARKET_COLUMNS = ["validation_event_id", "cik", "market_feature_available_at", "market_state_ready"]
EVENT_COLUMNS = ["validation_event_id", "cik", "target_accession_number"]


def market_review_requests(
    readiness: pd.DataFrame, target_event_id: str, policy: str = "late_cutoffs"
) -> pd.DataFrame:
    """Select metadata for re-verification, without implying the old cutoff was late.

    A release-time change within one date can preserve the market cutoff and
    trading schedule. It still needs an explicitly clock-bound evidence audit.
    The opt-in selected-target policy does not select by prices or returns.
    """
    if policy not in {"late_cutoffs", "selected_target"}:
        raise ValueError("Unsupported market-review policy")
    if readiness.validation_event_id.duplicated().any():
        raise ValueError("Duplicate market-review event")
    target = readiness.validation_event_id.eq(target_event_id)
    if target.sum() != 1:
        raise ValueError("Market-review target missing or ambiguous")
    if readiness.return_admission.ne(False).any():
        raise ValueError("Market-review requests must remain disarmed")
    selected = (
        target
        if policy == "selected_target"
        else readiness.market_cutoff_status.eq("at_or_after_event")
    )
    return readiness.loc[selected].copy()


def apply_publisher_clock(
    history: pd.DataFrame,
    request: dict[str, Any],
    spec: dict[str, Any],
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Change only one verified quarter's availability in an isolated copy."""
    if (
        any(
            request[k] is not False
            for k in (
                "change_filing_acceptance_timestamp",
                "first_public_certified",
                "price_collection_authorized",
                "return_admission",
            )
        )
        or request["purpose"] != "disarmed_publisher_label_sensitivity_only"
    ):
        raise ValueError("Publisher request would weaken admission gates")
    if any(request[k] != spec[k] for k in ("cik", "source_id", "snapshot_sha256")):
        raise ValueError("Publisher request/source identity mismatch")
    if spec["content_scope"] != "full_financial_table" or spec["clock_kind"] not in {
        "visible_EST_minute",
        "visible_EDT_minute",
    }:
        raise ValueError("Replay requires exact quarter values and an explicit minute label")
    lower = utc_timestamp(request["source_reported_at_utc"])
    timestamp = utc_timestamp(timestamp)
    if not lower <= timestamp < lower + pd.Timedelta(minutes=1):
        raise ValueError("Scenario is outside the publisher minute")
    if (
        history.event_id.duplicated().any()
        or history.duplicated(["company_id", "filing_accession"]).any()
    ):
        raise ValueError("Ambiguous history identity")
    match = history.company_id.eq(request["cik"]) & history.filing_accession.eq(
        request["target_filing_accession"]
    )
    if match.sum() != 1:
        raise ValueError("Publisher target missing or ambiguous")
    row = history.loc[match].iloc[0]
    for field in ("period_start", "period_end"):
        if pd.Timestamp(row[field]).date().isoformat() != spec[field]:
            raise ValueError("Publisher quarter differs from history")
    cells = {c["metric"]: c["value_usd"] for c in spec["cells"]}
    if len(cells) != len(spec["cells"]) or set(cells) != {"revenue", "gross_profit"}:
        raise ValueError("Exactly two quarter component cells are required")
    if any(row[k] != v for k, v in cells.items()) or not np.isclose(
        row.gross_margin, cells["gross_profit"] / cells["revenue"], rtol=0, atol=1e-12
    ):
        raise ValueError("Publisher values differ from frozen original values")
    old = utc_timestamp(request["same_contract_filing_proxy_window"]["announcement_at"])
    if utc_timestamp(row.actual_available_at) != old or not (
        pd.Timestamp(spec["period_end"], tz="UTC") < timestamp < old
    ):
        raise ValueError("Invalid availability ordering or changed baseline")
    result = history.copy(deep=True)
    result.loc[match, "actual_available_at"] = timestamp
    return result


def verify_baseline_projection(history: pd.DataFrame, saved: pd.DataFrame) -> None:
    """Fail closed if the existing clock-layer forecasts are not reproduced."""
    if history.event_id.duplicated().any() or saved.event_id.duplicated().any():
        raise ValueError("Duplicate baseline event")
    left, right = (
        history.set_index("event_id").sort_index(),
        saved.set_index("event_id").sort_index(),
    )
    if not left.index.equals(right.index):
        raise ValueError("Baseline sample changed")
    for field in right.columns:
        if field not in left:
            continue  # Saved disarm/version metadata, not research values.
        a, b = left[field], right[field]
        if field in TIME_COLUMNS:
            a, b = pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True)
        if field in NUMERIC_COLUMNS:
            equal = np.isclose(
                a.to_numpy(dtype=float, na_value=np.nan),
                b.to_numpy(dtype=float, na_value=np.nan),
                rtol=0,
                atol=1e-10,
                equal_nan=True,
            ).all()
        else:
            equal = (a.eq(b).fillna(False) | (a.isna() & b.isna())).all()
        if not equal:
            raise ValueError(f"Clock-layer baseline not reproduced: {field}")


def target_dependencies(history: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """Enumerate the exact training rows and earlier OOS errors for one forecast."""
    target = history.loc[history.event_id.eq(event_id)]
    if len(target) != 1 or target.gm_forecast.isna().any():
        raise ValueError("Dependency audit requires one forecast-ready target")
    cutoff = utc_timestamp(target.iloc[0].actual_available_at)
    prior = pd.to_datetime(history.actual_available_at, utc=True).lt(cutoff)
    training = (
        prior
        & history.analysis_ready_delta.astype(bool)
        & history[list(FORECAST_FEATURES)].notna().all(axis=1)
        & history.forecast_feature_cutoff.notna()
        & history.gross_margin.notna()
    )
    errors = prior & history.gm_forecast.notna() & history.gross_margin.notna()
    result = history.loc[
        training | errors,
        ["event_id", "company_id", "filing_accession", "actual_available_at", "gm_innovation_raw"],
    ].copy()
    result["in_training"] = training.loc[result.index]
    result["in_prior_oos_errors"] = errors.loc[result.index]
    row = target.iloc[0]
    if (
        training.sum() != row.forecast_training_observations
        or errors.sum() != row.forecast_sigma_observations
    ):
        raise ValueError("Training/error membership does not reproduce forecast counts")
    sigma = history.loc[errors, "gm_innovation_raw"].std(ddof=1)
    if pd.notna(row.forecast_sigma) and not np.isclose(
        sigma, row.forecast_sigma, atol=1e-12, rtol=0
    ):
        raise ValueError("Sigma is not computed from strictly prior OOS errors")
    return result.reset_index(drop=True)


def three_arm_readiness(
    events: pd.DataFrame, history: pd.DataFrame, market: pd.DataFrame, publisher_event_id: str
) -> pd.DataFrame:
    """Keep the full frozen cohort; metadata coverage is NOT return-study admission."""
    if set(events) != set(EVENT_COLUMNS) or set(market) != set(MARKET_COLUMNS):
        raise ValueError("Readiness requires metadata-only projections")
    if (
        events.validation_event_id.duplicated().any()
        or market.validation_event_id.duplicated().any()
    ):
        raise ValueError("Duplicate market/execution event")
    if not set(market.validation_event_id).issubset(set(events.validation_event_id)):
        raise ValueError("Market row outside frozen cohort")
    identities = events[["validation_event_id", "cik"]].merge(
        market[["validation_event_id", "cik"]],
        on="validation_event_id",
        suffixes=("_event", "_market"),
    )
    if not identities.cik_event.eq(identities.cik_market).all():
        raise ValueError("Market/execution issuer identity differs")
    financial = history[
        [
            "company_id",
            "filing_accession",
            "actual_available_at",
            "analysis_ready_level",
            "analysis_ready_delta",
            "analysis_ready_innovation_std",
        ]
    ].rename(columns={"company_id": "cik", "filing_accession": "target_accession_number"})
    result = events.merge(
        financial, on=["cik", "target_accession_number"], how="left", validate="one_to_one"
    )
    result = result.merge(
        market,
        on=["validation_event_id", "cik"],
        how="left",
        validate="one_to_one",
        indicator="market_join",
    )
    cutoff = pd.to_datetime(result.market_feature_available_at, utc=True)
    actual = pd.to_datetime(result.actual_available_at, utc=True)
    result["market_cutoff_status"] = np.select(
        [result.market_join.eq("left_only"), cutoff.isna(), actual.isna(), cutoff.ge(actual)],
        [
            "missing_market_row",
            "missing_market_cutoff",
            "missing_financial_clock",
            "at_or_after_event",
        ],
        default="before_proxy_only",
    )
    result["financial_representation_coverage"] = (
        result[["analysis_ready_level", "analysis_ready_delta", "analysis_ready_innovation_std"]]
        .eq(True)
        .all(axis=1)
    )
    result["market_metadata_coverage"] = result.market_state_ready.eq(True) & cutoff.lt(actual)
    result["joint_metadata_coverage"] = (
        result.financial_representation_coverage & result.market_metadata_coverage
    )
    result["clock_basis"] = np.where(
        result.validation_event_id.eq(publisher_event_id),
        "publisher_minute_lower_bound_sensitivity",
        "mixed_acceptance_proxies_unverified",
    )
    for field in (
        "first_public_certified",
        "new_return_window_verified",
        "model_admission",
        "return_admission",
    ):
        result[field] = False
    return result.drop(columns="market_join")
