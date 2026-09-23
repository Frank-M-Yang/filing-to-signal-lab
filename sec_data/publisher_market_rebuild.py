"""Isolated pre-release market reconstruction and revised-window evidence checks.

This diagnostic does not certify first-public clocks or authorize return reads.
Reuse the frozen 13-feature math, but require consecutive official sessions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.execution_contract import execution_window
from sec_data.validation_market_state import (
    MARKET_STATE_FEATURES,
    RAW_PRICE_BASIS,
    SPLIT_POLICY,
    ValidationMarketStateConfig,
    _feature_row,
    _prepare_bars,
)

VERSION = "publisher-market-rebuild-v1"
BAR_VALUE_COLUMNS = ["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"]
BAR_METADATA_COLUMNS = ["ticker", "session_date", "source_sha256"]


def compare_market_features(old: dict[str, Any], rebuilt: dict[str, Any]) -> pd.DataFrame:
    """Preserve missing values in differences; absent features are never zero returns."""
    result = pd.DataFrame(
        {
            "feature": MARKET_STATE_FEATURES,
            "old_filing_proxy_value": [old[k] for k in MARKET_STATE_FEATURES],
            "rebuilt_publisher_value": [rebuilt[k] for k in MARKET_STATE_FEATURES],
        }
    )
    for column in ("old_filing_proxy_value", "rebuilt_publisher_value"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    result["change"] = result.rebuilt_publisher_value - result.old_filing_proxy_value
    return result


def _valid_hashes(values: pd.Series) -> bool:
    return bool(values.notna().all() and values.astype(str).str.fullmatch(r"[a-f0-9]{64}").all())


def rebuild_pre_event_state(
    event: dict[str, Any],
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    splits: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    price_basis: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use only dates before the New York release date, never the release-day close.

    Callers must project/filter before reading numeric price columns. A timestamp
    below is the official close proxy, not a certified vendor delivery timestamp.
    """
    stamp = utc_timestamp(event["announcement_at"])
    if price_basis not in (RAW_PRICE_BASIS, "diagnostic_quote_no_splits"):
        raise ValueError("Explicit raw or diagnostic-only quote basis required")
    if price_basis == "diagnostic_quote_no_splits" and not splits.empty:
        raise ValueError("Unknown quote basis cannot enter split adjustment")
    date = stamp.tz_convert("America/New_York").date().isoformat()
    calendar = sessions.loc[sessions.exchange_mic.eq(event["exchange_mic"])].copy()
    if calendar.empty or calendar.session_date.duplicated().any():
        raise ValueError("Missing or ambiguous official calendar")
    calendar = calendar.sort_values("session_date")
    calendar["regular_open_at"] = calendar.regular_open_at.map(utc_timestamp)
    calendar["regular_close_at"] = calendar.regular_close_at.map(utc_timestamp)
    if calendar.regular_open_at.ge(calendar.regular_close_at).any():
        raise ValueError("Invalid session open/close ordering")
    if not _valid_hashes(calendar.source_snapshot_sha256):
        raise ValueError("Missing calendar provenance")
    if (
        not calendar.regular_close_at.dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
        .eq(calendar.session_date)
        .all()
    ):
        raise ValueError("Calendar date/close disagreement")
    prior = calendar.loc[calendar.session_date.lt(date)].tail(61)
    if len(prior) != 61 or prior.regular_close_at.ge(stamp).any():
        raise ValueError("Need 61 strictly earlier official sessions")
    result: dict[str, Any] = {
        "validation_event_id": event["validation_event_id"],
        "cik": event["cik"],
        "ticker": event["ticker"],
        "exchange_mic": event["exchange_mic"],
        "announcement_at": stamp.isoformat(),
        "market_clock_policy": "previous_New_York_date_sessions_only",
        "market_clock_basis": "official_close_proxy_not_vendor_delivery_certification",
        "input_price_basis": price_basis,
        "split_adjustment_policy": SPLIT_POLICY,
        "market_first_required_session": prior.session_date.iloc[0],
        "market_last_required_session": prior.session_date.iloc[-1],
        "first_public_certified": False,
        "model_admission": False,
        "return_admission": False,
        "market_state_ready": False,
        "market_state_status": "missing_required_official_session",
        "market_feature_available_at": None,
        **{field: np.nan for field in MARKET_STATE_FEATURES},
    }
    prepared = []
    audit = prior[
        ["session_date", "regular_open_at", "regular_close_at", "source_snapshot_sha256"]
    ].copy()
    for label, frame, ticker, target_id in (
        ("issuer", issuer_bars, event["ticker"], event["price_target_id"]),
        ("benchmark", benchmark_bars, "SPY", event["benchmark_target_id"]),
    ):
        if set(frame) != set(BAR_VALUE_COLUMNS):
            raise ValueError("Unexpected numeric bar projection")
        if not frame.ticker.eq(ticker).all() or frame.session_date.duplicated().any():
            raise ValueError("Wrong ticker or duplicate price date")
        if not frame.session_date.lt(date).all():
            raise ValueError("Numeric prices include release-day or later values")
        if not _valid_hashes(frame.source_sha256):
            raise ValueError("Missing bar provenance")
        if not np.isfinite(frame[["close_unadjusted", "volume"]].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite prices or volume")
        audit = audit.merge(
            frame[["session_date", "source_sha256"]].rename(
                columns={"source_sha256": f"{label}_source_sha256"}
            ),
            on="session_date",
            how="left",
            validate="one_to_one",
        )
        target_col = "price_target_id" if label == "issuer" else "benchmark_target_id"
        exact_history = frame.loc[frame.session_date.isin(prior.session_date)]
        prepared.append(
            # Diagnostic quote checks only compare algebra with zero actions;
            # they do not certify the provider's basis or export admission.
            _prepare_bars(
                exact_history.assign(**{target_col: target_id}),
                target_col,
                label,
                price_basis=RAW_PRICE_BASIS,
            )
        )
    audit["required_session_present"] = (
        audit[["issuer_source_sha256", "benchmark_source_sha256"]].notna().all(axis=1)
    )
    if not audit.required_session_present.all():
        return result, audit
    splits = splits.copy()
    if not splits.empty:
        splits["execution_date"] = pd.to_datetime(splits.execution_date)
        if not splits.execution_date.lt(pd.Timestamp(date)).all():
            raise ValueError("Split input must be strictly before the release date")
    closes = {
        (event["exchange_mic"], pd.Timestamp(str(r.session_date)).date()): r.regular_close_at
        for r in calendar.itertuples(index=False)
    }
    # The legacy kernel names its cutoff accepted_at. This temporary adapter is
    # NOT a change to filing_accepted_at in the canonical SEC or feature tables.
    adapted = SimpleNamespace(
        validation_event_id=event["validation_event_id"],
        exchange_mic=event["exchange_mic"],
        accepted_at=stamp,
    )
    row = _feature_row(
        adapted, prepared[0], prepared[1], splits, closes, ValidationMarketStateConfig()
    )
    if (
        row["market_state_ready"]
        and pd.Timestamp(row["market_last_session_date"]).date().isoformat()
        != prior.session_date.iloc[-1]
    ):
        raise ValueError("Computed feature history skips the last required session")
    result.update(row)
    return result, audit


def interval_evidence_checks(
    event: dict[str, Any],
    first_at: str,
    last_at: str,
    mapping: dict[str, Any],
    action_audit: dict[str, Any],
    price_audit: dict[str, Any],
    benchmark_queries: pd.DataFrame,
    benchmark_report: dict[str, Any],
) -> dict[str, bool]:
    """Bind identity, complete queries and date coverage to an explicit interval."""
    first, last = utc_timestamp(first_at), utc_timestamp(last_at)
    if first > last:
        raise ValueError("Reversed evidence interval")
    start = first.tz_convert("America/New_York").date().isoformat()
    end = last.tz_convert("America/New_York").date().isoformat()
    identity = all(
        mapping[k] == event[k]
        for k in (
            "cik",
            "ticker",
            "exchange_mic",
            "price_target_id",
            "security_id",
            "security_id_kind",
        )
    )
    hashes = pd.Series(
        [
            mapping[k]
            for k in (
                "source_snapshot_sha256",
                "end_source_snapshot_sha256",
                "ticker_events_source_snapshot_sha256",
            )
        ]
    )
    identity = (
        identity
        and _valid_hashes(hashes)
        and (
            utc_timestamp(mapping["valid_from_inclusive"]) <= first
            and utc_timestamp(mapping["valid_to_exclusive"]) > last
        )
    )
    action_identity = all(
        action_audit[k] == event[v]
        for k, v in (
            ("price_target_id", "price_target_id"),
            ("cik", "cik"),
            ("selected_ticker", "ticker"),
            ("selected_exchange_mic", "exchange_mic"),
            ("stable_security_id", "security_id"),
        )
    )
    actions = (
        action_identity
        and all(
            action_audit[k] is True
            for k in (
                "splits_query_complete",
                "dividends_query_complete",
                "corporate_actions_ready",
                "historical_symbol_interval_ready",
            )
        )
        and action_audit["price_window_start"] <= start <= end <= action_audit["price_window_end"]
    )
    prices = all(price_audit[k] == event[k] for k in ("price_target_id", "cik", "ticker")) and (
        price_audit["provider_query_complete"] is True
        and price_audit["parser_rejection_rows"] == 0
        and price_audit["price_window_start"] <= start <= end <= price_audit["price_window_end"]
    )
    benchmark = (
        set(benchmark_queries.query_kind) == {"daily_bars", "splits", "dividends"}
        and benchmark_queries.status_code.eq(200).all()
        and benchmark_queries.ticker.eq("SPY").all()
        and benchmark_queries.benchmark_target_id.eq(event["benchmark_target_id"]).all()
        and _valid_hashes(benchmark_queries.sha256)
        and benchmark_queries.window_start.le(start).all()
        and benchmark_queries.window_end.ge(end).all()
        and benchmark_report["queries_complete"] == benchmark_report["queries_required"] == 3
        and benchmark_report["rejections"] == benchmark_report["action_normalization_failures"] == 0
        and benchmark_report["query_adjusted"] is False
        and benchmark_report["splits_explicit"] is True
        and benchmark_report["dividends_explicit"] is True
    )
    return {
        "identity_interval": bool(identity),
        "issuer_actions_interval": bool(actions),
        "issuer_price_query_interval": bool(prices),
        "benchmark_query_interval": bool(benchmark),
    }


def audit_revised_window(
    event: dict[str, Any],
    window: dict[str, Any],
    sessions: pd.DataFrame,
    issuer_metadata: pd.DataFrame,
    benchmark_metadata: pd.DataFrame,
    mapping: dict[str, Any],
    action_audit: dict[str, Any],
    price_audit: dict[str, Any],
    listing: dict[str, Any],
    benchmark_queries: pd.DataFrame,
    benchmark_report: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit new date coverage without opening forward price/return values.

    No terminal payout is inferred. In-window delisting always blocks this narrow
    adapter, even when a vendor's historical price rows exist.
    """
    expected = execution_window(event["announcement_at"], event["exchange_mic"], sessions)
    if window != expected:
        raise ValueError("Window differs from frozen execution contract")
    start, end = window["entry_session_date"], window["exit_session_date"]
    dates = sessions.loc[
        sessions.exchange_mic.eq(event["exchange_mic"]) & sessions.session_date.between(start, end)
    ].sort_values("session_date")
    if len(dates) != 20 or dates.session_date.duplicated().any():
        raise ValueError("Revised window must contain exactly 20 official sessions")
    audit = dates[
        ["session_date", "regular_open_at", "regular_close_at", "source_snapshot_sha256"]
    ].copy()
    for label, frame, ticker in (
        ("issuer", issuer_metadata, event["ticker"]),
        ("benchmark", benchmark_metadata, "SPY"),
    ):
        if set(frame) != set(BAR_METADATA_COLUMNS):
            raise ValueError("Forward-window check requires metadata only")
        if not frame.ticker.eq(ticker).all() or frame.session_date.duplicated().any():
            raise ValueError("Wrong ticker or duplicate window date")
        if not _valid_hashes(frame.source_sha256):
            raise ValueError("Missing window bar provenance")
        audit = audit.merge(
            frame[["session_date", "source_sha256"]].rename(
                columns={"source_sha256": f"{label}_source_sha256"}
            ),
            on="session_date",
            how="left",
            validate="one_to_one",
        )
    intervals = interval_evidence_checks(
        event,
        window["entry_at"],
        window["exit_at"],
        mapping,
        action_audit,
        price_audit,
        benchmark_queries,
        benchmark_report,
    )
    listing_identity = all(
        listing[k] == event[k] for k in ("cik", "exchange_mic", "security_id", "security_id_kind")
    )
    delisted = listing.get("delisting_date")
    active = listing["listing_outcome_status"] == "active_at_outcome_cutoff" and pd.isna(delisted)
    later = (
        pd.notna(delisted)
        and str(delisted)[:10] > end
        and listing["listing_outcome_status"] == "inactive_with_delisting_date"
    )
    listing_ready = bool(
        listing_identity
        and listing["listing_outcome_ready"] is True
        and listing["outcome_cutoff_date"] >= end
        and (active or later)
        and _valid_hashes(pd.Series([listing["source_snapshot_sha256"]]))
    )
    checks = {
        "issuer_parsed_bars_all_sessions": bool(
            audit.issuer_source_sha256.notna().all() and intervals["issuer_price_query_interval"]
        ),
        "benchmark_parsed_bars_all_sessions": bool(audit.benchmark_source_sha256.notna().all()),
        "historical_security_interval_covers_new_window": intervals["identity_interval"],
        "issuer_actions_query_covers_new_window": intervals["issuer_actions_interval"],
        "benchmark_queries_cover_new_window": intervals["benchmark_query_interval"],
        "listing_outcome_evidence_covers_new_window": listing_ready,
    }
    return {
        "validation_event_id": event["validation_event_id"],
        "window": window,
        "required_sessions": len(dates),
        "checks": checks,
        "window_evidence_ready": all(checks.values()),
        "evidence_basis": "pinned_provider_parsers_intervals_actions_and_listing_status",
        "historical_identity_method": mapping["mapping_method"],
        "identity_endpoint_experimental": mapping["ticker_events_endpoint_experimental"],
        "missing_checks": [k for k, v in checks.items() if not v],
        "listing_status_is_outcome_evidence_not_selection": True,
        "terminal_value_imputed": False,
        "first_public_certified": False,
        "forward_price_values_loaded": False,
        "returns_computed": False,
        "return_admission": False,
        "model_admission": False,
    }, audit
