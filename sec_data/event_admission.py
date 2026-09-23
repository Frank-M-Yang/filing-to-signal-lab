"""Outcome-blind event checklist; publisher scenarios never authorize returns."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.publisher_market_rebuild import BAR_VALUE_COLUMNS
from sec_data.validation_liquidity import ValidationLiquidityConfig

VERSION = "event-admission-audit-v1"
LIQUIDITY_COLUMNS = ["price_target_id", *BAR_VALUE_COLUMNS]
REPRESENTATIONS = [
    "analysis_ready_level",
    "analysis_ready_delta",
    "analysis_ready_innovation_raw",
    "analysis_ready_innovation_std",
]
FORECAST_CLOCKS = [
    "forecast_feature_cutoff",
    "forecast_generated_at",
    "forecast_training_end",
    "forecast_error_history_end",
]
HISTORY_CHECKS = {
    "identity_interval",
    "issuer_actions_interval",
    "issuer_price_query_interval",
    "benchmark_query_interval",
}
WINDOW_CHECKS = {
    "issuer_parsed_bars_all_sessions",
    "benchmark_parsed_bars_all_sessions",
    "historical_security_interval_covers_new_window",
    "issuer_actions_query_covers_new_window",
    "benchmark_queries_cover_new_window",
    "listing_outcome_evidence_covers_new_window",
}


def prior_liquidity_sessions(event: dict[str, Any], calendar: pd.DataFrame) -> pd.DataFrame:
    """Select 20 official sessions strictly before the publisher's New York date.

    A date cut is deliberately conservative: even an after-close announcement
    cannot use its same-day bar. These are reconstructed close-time proxies,
    not certified historical vendor delivery timestamps.
    """
    local_date = utc_timestamp(event["announcement_at"]).tz_convert("America/New_York").date()
    rows = calendar.loc[
        calendar.exchange_mic.eq(event["exchange_mic"])
        & calendar.session_date.lt(local_date.isoformat())
    ].sort_values("session_date")
    if rows.session_date.duplicated().any():
        raise ValueError("Duplicate calendar sessions")
    rows = rows.tail(ValidationLiquidityConfig().lookback_sessions).copy()
    if not rows.source_snapshot_sha256.str.fullmatch(r"[0-9a-f]{64}").eq(True).all():
        raise ValueError("Missing calendar provenance")
    for row in rows.to_dict("records"):
        start, end = utc_timestamp(row["regular_open_at"]), utc_timestamp(row["regular_close_at"])
        if (
            not start < end < utc_timestamp(event["announcement_at"])
            or end.tz_convert("America/New_York").date().isoformat() != row["session_date"]
        ):
            raise ValueError("Invalid prior-session clock")
    return rows.reset_index(drop=True)


def audit_label_liquidity(
    event: dict[str, Any],
    sessions: pd.DataFrame,
    bars: pd.DataFrame,
    evidence_checks: dict[str, bool],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply the unchanged legacy 20-session / 80% / $1m median-volume rule.

    Missing rows remain unknown, not zero-volume sessions. The $10k capacity
    reference is a research assumption, not an order or investment recommendation.
    Only the exact pre-event price projection may enter this function.
    """
    cfg = ValidationLiquidityConfig()
    if set(bars) != set(LIQUIDITY_COLUMNS):
        raise ValueError("Liquidity requires the restricted pre-event price projection")
    expected = prior_liquidity_sessions(event, sessions)
    if list(expected.session_date) != list(sessions.session_date):
        raise ValueError("Liquidity sessions must be the exact ordered prior window")
    if (
        not bars.price_target_id.eq(event["price_target_id"]).all()
        or not bars.ticker.eq(event["ticker"]).all()
        or bars.session_date.duplicated().any()
        or not bars.session_date.isin(sessions.session_date).all()
    ):
        raise ValueError("Wrong price identity, duplicate date or non-prior price")
    if not bars.source_sha256.str.fullmatch(r"[0-9a-f]{64}").eq(True).all():
        raise ValueError("Missing price provenance")
    audit = sessions.merge(bars, on="session_date", how="left", validate="one_to_one")
    close = pd.to_numeric(audit.close_unadjusted, errors="coerce")
    volume = pd.to_numeric(audit.volume, errors="coerce")
    audit["price_row_observed"] = audit.source_sha256.notna()
    audit["numeric_evidence_valid"] = (
        np.isfinite(close) & np.isfinite(volume) & close.gt(0) & volume.ge(0)
    )
    audit["dollar_volume_proxy_usd"] = (close * volume).where(audit.numeric_evidence_valid)
    audit["numeric_evidence_valid"] &= np.isfinite(audit.dollar_volume_proxy_usd)
    audit["print_observed"] = audit.numeric_evidence_valid & volume.gt(0)
    printed = int(audit.print_observed.sum())
    coverage = printed / cfg.lookback_sessions
    median = audit.loc[audit.print_observed, "dollar_volume_proxy_usd"].median()
    calendar_complete = len(sessions) == cfg.lookback_sessions
    evidence_complete = bool(
        calendar_complete
        and audit.price_row_observed.all()
        and audit.numeric_evidence_valid.all()
        and set(evidence_checks) == HISTORY_CHECKS
        and all(v is True for v in evidence_checks.values())
    )
    adequate_prints = coverage >= cfg.minimum_print_coverage
    adequate_capacity = bool(pd.notna(median) and median >= cfg.minimum_median_dollar_volume_usd)
    status = "eligible_under_publisher_label_only"
    if not calendar_complete:
        status = "insufficient_calendar_history"
    elif not evidence_complete:
        status = "liquidity_evidence_incomplete"
    elif not adequate_prints and not adequate_capacity:
        status = "insufficient_print_coverage_and_capacity"
    elif not adequate_prints:
        status = "insufficient_print_coverage"
    elif not adequate_capacity:
        status = "insufficient_median_dollar_volume"
    summary = {
        "validation_event_id": event["validation_event_id"],
        "cik": event["cik"],
        "ticker": event["ticker"],
        "announcement_at": event["announcement_at"],
        "required_sessions": cfg.lookback_sessions,
        "calendar_sessions": len(sessions),
        "observed_price_rows": int(audit.price_row_observed.sum()),
        "printed_sessions": printed,
        "print_coverage": coverage,
        "median_dollar_volume_usd": float(median) if pd.notna(median) else None,
        "minimum_median_dollar_volume_usd": cfg.minimum_median_dollar_volume_usd,
        "minimum_print_coverage": cfg.minimum_print_coverage,
        "first_session": sessions.session_date.iloc[0] if len(sessions) else None,
        "last_session": sessions.session_date.iloc[-1] if len(sessions) else None,
        "evidence_complete": evidence_complete,
        "evidence_checks": evidence_checks,
        "status": status,
        "label_liquidity_eligible": evidence_complete and adequate_prints and adequate_capacity,
        "first_public_certified": False,
        "market_clock_basis": "official_close_proxy_not_vendor_delivery_certification",
        "forward_price_values_loaded": False,
        "returns_computed": False,
        "model_admission": False,
        "trading_admission": False,
    }
    return summary, audit


def _unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate {label} identity")


def _same_events(base: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    _unique(other, ["validation_event_id"], label)
    if set(base.validation_event_id) != set(other.validation_event_id):
        raise ValueError(f"Frozen cohort differs from {label}")
    comparison = base[["validation_event_id", "cik"]].merge(
        other[["validation_event_id", "cik"]], on="validation_event_id", validate="one_to_one"
    )
    if not comparison.cik_x.eq(comparison.cik_y).all():
        raise ValueError(f"Issuer identity differs from {label}")


def build_admission_ledger(
    readiness: pd.DataFrame,
    identities: pd.DataFrame,
    legacy_gates: pd.DataFrame,
    financial_metadata: pd.DataFrame,
    chronology: list[dict[str, Any]],
    publisher: dict[str, Any],
    rebuilt_event: dict[str, Any],
    liquidity: dict[str, Any],
    market_state: dict[str, Any],
    window: dict[str, Any],
) -> pd.DataFrame:
    """Keep all frozen events, distinguishing coverage, label repair and admission.

    The four financial representations retain separate availability flags. Prior
    acceptance-clock permissions are descriptive only. This version intentionally
    has no code path that grants a model, return or trading authorization.
    """
    _unique(readiness, ["validation_event_id"], "readiness")
    _same_events(readiness, identities, "execution identities")
    _same_events(readiness, legacy_gates, "legacy gates")
    financial = financial_metadata.rename(
        columns={"company_id": "cik", "filing_accession": "target_accession_number"}
    )
    _unique(financial, ["cik", "target_accession_number"], "financial history")
    # Do not trust ticker alone: one company has several independent filing events.
    result = readiness.merge(
        identities[["validation_event_id", "cik", "ticker"]],
        on=["validation_event_id", "cik"],
        how="left",
        validate="one_to_one",
    )
    flags = financial[["cik", "target_accession_number", *REPRESENTATIONS]]
    result = result.drop(columns=[k for k in REPRESENTATIONS if k in result]).merge(
        flags, on=["cik", "target_accession_number"], how="left", validate="one_to_one"
    )
    result["financial_four_representation_coverage"] = result[REPRESENTATIONS].eq(True).all(axis=1)
    result = result.merge(
        legacy_gates[
            ["validation_event_id", "cik", "liquidity_eligible", "liquidity_status"]
        ].rename(
            columns={
                "liquidity_eligible": "legacy_liquidity_eligible",
                "liquidity_status": "legacy_liquidity_status",
            }
        ),
        on=["validation_event_id", "cik"],
        how="left",
        validate="one_to_one",
    )
    reviews = {r["validation_event_id"]: r for r in chronology}
    if len(reviews) != len(chronology) or not set(reviews).issubset(
        set(result.validation_event_id)
    ):
        raise ValueError("Duplicate or out-of-cohort clock review")
    for row in result.to_dict("records"):
        review = reviews.get(row["validation_event_id"])
        if review and (
            review["cik"] != row["cik"]
            or review["filing_accession"] != row["target_accession_number"]
        ):
            raise ValueError("Clock review issuer/accession mismatch")
    result["clock_review_status"] = result.validation_event_id.map(
        {k: v["status"] for k, v in reviews.items()}
    ).fillna("not_in_30_event_clock_review")
    result["reviewed_publisher_source_count"] = (
        result.validation_event_id.map({k: len(v["sources"]) for k, v in reviews.items()})
        .fillna(0)
        .astype(int)
    )
    for field in (
        "label_financial_replay_ready",
        "label_market_state_ready",
        "label_liquidity_eligible",
        "label_window_evidence_ready",
        "label_scenario_inputs_complete",
    ):
        result[field] = False
    result["current_liquidity_status"] = "not_rebuilt_for_release_clock"
    event_id = rebuilt_event["validation_event_id"]
    match = result.validation_event_id.eq(event_id)
    if match.sum() != 1 or event_id not in reviews:
        raise ValueError("Rebuilt event missing from frozen reviewed cohort")
    target_row = result.loc[match].iloc[0]
    review = reviews[event_id]
    for artifact in (liquidity, market_state, window):
        if artifact["validation_event_id"] != event_id:
            raise ValueError("Rebuilt artifact belongs to another event")
    for artifact in (liquidity, market_state, rebuilt_event):
        if artifact["cik"] != target_row.cik or artifact["ticker"] != target_row.ticker:
            raise ValueError("Rebuilt artifact issuer mismatch")
    lower = utc_timestamp(rebuilt_event["announcement_at"])
    for artifact in (liquidity, market_state, window["window"]):
        if utc_timestamp(artifact["announcement_at"]) != lower:
            raise ValueError("Rebuilt artifact clock mismatch")
    financial_ready = True
    for name in ("minute_start", "minute_end"):
        target = publisher["scenarios"][name]["target"]
        if (
            target["company_id"] != target_row.cik
            or target["filing_accession"] != target_row.target_accession_number
        ):
            raise ValueError("Financial replay target mismatch")
        stamp = utc_timestamp(target["actual_available_at"])
        if not lower <= stamp < lower + pd.Timedelta(minutes=1):
            raise ValueError("Financial replay is outside publisher minute")
        financial_ready &= all(target[k] is True for k in REPRESENTATIONS)
        financial_ready &= all(utc_timestamp(target[k]) < lower for k in FORECAST_CLOCKS)
    precise = [
        s
        for s in review["chronology_scenarios"]
        if s["applies_to_exact_gross_margin"]
        and s["source_id"] in review["precise_label_replay_candidates"]
    ]
    if not any(
        utc_timestamp(s["lower_inclusive"]) == lower
        and utc_timestamp(s["upper_exclusive"]) == lower + pd.Timedelta(minutes=1)
        for s in precise
    ):
        raise ValueError("Rebuild has no matching precise, exact-value source review")
    financial_ready &= publisher["baseline_reproduced"] is True
    financial_ready &= publisher["minute_endpoint_signals_equal"] is True
    financial_ready &= publisher["hyperparameters_changed"] is False
    market_ready = (
        market_state["market_state_ready"] is True
        and set(market_state["history_evidence_checks"]) == HISTORY_CHECKS
        and all(v is True for v in market_state["history_evidence_checks"].values())
        and utc_timestamp(market_state["market_feature_available_at"]) < lower
    )
    window_ready = (
        window["window_evidence_ready"] is True
        and set(window["checks"]) == WINDOW_CHECKS
        and all(v is True for v in window["checks"].values())
    )
    result.loc[match, "label_financial_replay_ready"] = financial_ready
    result.loc[match, "label_market_state_ready"] = market_ready
    result.loc[match, "label_liquidity_eligible"] = (
        liquidity["label_liquidity_eligible"] is True and liquidity["evidence_complete"] is True
    )
    result.loc[match, "current_liquidity_status"] = liquidity["status"]
    result.loc[match, "label_window_evidence_ready"] = window_ready
    result["label_scenario_inputs_complete"] = (
        result[
            [
                "financial_four_representation_coverage",
                "label_financial_replay_ready",
                "label_market_state_ready",
                "label_liquidity_eligible",
                "label_window_evidence_ready",
            ]
        ]
        .eq(True)
        .all(axis=1)
    )
    result["blockers"] = [admission_blockers(r) for r in result.to_dict("records")]
    result["admission_status"] = np.where(
        result.label_scenario_inputs_complete,
        "label_scenario_prepared_not_admitted",
        "blocked_dependencies_not_rebuilt",
    )
    for field in (
        "first_public_certified",
        "model_admission",
        "return_admission",
        "trading_admission",
    ):
        result[field] = False
    return result


def admission_blockers(row: Mapping[Hashable, Any]) -> list[str]:
    """Give overlapping reasons; legacy liquidity exclusions remain separate facts."""
    reasons = ["first_public_clock_not_verified", "return_experiment_not_frozen"]
    for field, reason in (
        ("financial_four_representation_coverage", "incomplete_four_representation_metadata"),
        ("label_financial_replay_ready", "financial_chain_not_rebuilt_for_release_clock"),
        ("label_market_state_ready", "market_state_not_rebuilt_for_release_clock"),
        ("label_liquidity_eligible", "release_clock_liquidity_not_passed"),
        ("label_window_evidence_ready", "release_clock_window_evidence_not_verified"),
    ):
        if not row[field]:
            reasons.append(reason)
    return reasons
