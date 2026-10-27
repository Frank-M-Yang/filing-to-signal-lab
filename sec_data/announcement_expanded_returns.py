"""Frozen announcement expansion, metadata-only identity reuse and fail-closed gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from sec_data.announcement_returns import select_reviewed_cases
from sec_data.disclosure_audit import utc_timestamp
from sec_data.margin_scope_replay import ARMS
from sec_data.validation_action_continuity import calculate_validation_action_continuity

VERSION = "announcement-expanded-returns-v1"


def validate_contract(config: dict[str, Any]) -> None:
    """Keep the reviewed fourteen, fixed model and retrospective-only permissions frozen."""
    expected = {
        "version": VERSION,
        "frozen_cohort_events": 316,
        "frozen_new_issuer_selections": 20,
        "frozen_case_count": 14,
        "frozen_history_events": 3254,
        "arms": list(ARMS),
        "primary_arm": ARMS[2],
        "issuer_cik": "0001552275",
        "minimum_industry_events": 20,
        "entry_session_offset": 2,
        "holding_sessions_inclusive": 20,
        "benchmark": "SPY",
        "policy_basis": "retrospective_accounting_scope_sensitivity_not_a_historical_live_filter",
        "cached_identity_reauthorization_allowed": True,
        "fixed_specification_refit_allowed": True,
        "all_frozen_cases_required_for_returns": True,
        **dict.fromkeys(
            (
                "network_allowed",
                "sample_replacement_allowed",
                "hyperparameter_search_allowed",
                "market_features_in_forecast",
                "general_scope_gate_cleared",
                "first_public_certified",
                "trading_admission",
                "inferential_tests_allowed",
                "strategy_pnl_allowed",
                "overwrite_originals",
            ),
            False,
        ),
    }
    if any(type(config.get(k)) is not type(v) or config[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported or weakened expanded-return contract")
    ids = config.get("case_event_ids", [])
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError("Fourteen unique frozen event identities required")


def freeze_reviewed_expansion(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    dispositions: list[dict[str, Any]],
    cohort_ids: list[str],
    expected_ids: list[str],
) -> list[dict[str, Any]]:
    """Verify six controls plus eight source-qualified cases; never substitute rejected cases."""

    def ids(rows: list[dict[str, Any]]) -> set[str]:
        result = {str(r["validation_event_id"]) for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate frozen expansion identity")
        return result

    selected_ids, control_ids = ids(selected), ids(controls)
    if (
        len(selected) != 20
        or len(controls) != 6
        or len(cohort_ids) != 316
        or len(set(cohort_ids)) != 316
        or len({r["cik"] for r in selected}) != 20
        or {r["cik"] for r in selected} & {r["cik"] for r in controls}
        or ids(dispositions) != selected_ids
        or not (selected_ids | control_ids) <= set(cohort_ids)
    ):
        raise ValueError("Frozen selection, controls or cohort denominator changed")
    verified = {
        r["validation_event_id"]
        for r in dispositions
        if r.get("release_value_pair_verified") is True
    }
    if len(verified) != 8 or ids(candidates) != (control_ids | verified):
        raise ValueError("Keep all eight reviewed additions and all six controls")
    cases = select_reviewed_cases(candidates, expected_ids)
    if len(cases) != 14 or any(
        r.get(k) is not False
        for r in cases
        for k in ("first_public_certified", "model_admission", "return_admission")
    ):
        raise ValueError("Conditional cases cannot promote first-public or trading admission")
    originals = {r["validation_event_id"]: r for r in [*selected, *controls]}
    for case in cases:
        if any(
            case[k] != originals[case["validation_event_id"]][k]
            for k in (
                "cik",
                "target_accession_number",
                "earliest_disclosure_at",
                "price_target_id",
                "assigned_ticker",
                "assigned_exchange_mic",
            )
        ):
            raise ValueError("Reviewed expansion changed the frozen event identity or clock")
    return cases


def reauthorize_cached_continuity(
    events: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    actions: pd.DataFrame,
    identities: pd.DataFrame,
    ticker_events: pd.DataFrame,
    metadata: pd.DataFrame,
    sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Recompute provider continuity for new proxy windows within exact cached query bounds.

    This changes analysis authorization, not source data. A later old authorization
    does not itself invalidate an earlier cached identity observation. Conversely,
    no earlier authorization may be inferred when the original queries start late.
    The existing 95% identity fallback stays unchanged; the return audit separately
    requires every one of the twenty price sessions. No numeric prices are inputs.
    """
    if set(metadata) != {"price_target_id", "ticker", "session_date", "source_sha256"}:
        raise ValueError("Identity reauthorization requires price metadata only")
    by_id = {r["validation_event_id"]: r for r in windows}
    if len(by_id) != len(windows) or set(by_id) != {r["validation_event_id"] for r in events}:
        raise ValueError("Missing or duplicate frozen window")
    components, authorized, admitted, ledger = [], [], [], []
    for event in events:
        w = by_id[event["validation_event_id"]]["window"]
        rows = actions.loc[actions.action_target_id.eq(event["action_target_id"])]
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous cached action target")
        target = rows.iloc[0]
        # A CIK can have multiple traded share classes. Bind the selected class,
        # not every security returned by the same issuer-level reference query.
        endpoints = identities.loc[
            identities.action_target_id.eq(event["action_target_id"])
            & identities.cik.eq(event["cik"])
            & identities.ticker.eq(event["ticker"])
            & identities.exchange_mic.eq(event["exchange_mic"])
            & identities.security_type.eq("CS")
            & identities.share_class_figi.eq(event["security_id"])
        ]
        endpoint_dates_valid = all(
            len(endpoints.loc[endpoints.query_kind.eq(kind)]) == 1
            and str(endpoints.loc[endpoints.query_kind.eq(kind), "as_of_date"].iloc[0]) == bound
            for kind, bound in (
                ("reference_start", target.price_window_start),
                ("reference_end", target.price_window_end),
            )
        )
        within_queries = bool(
            target.price_window_start
            <= w["entry_session_date"]
            <= w["exit_session_date"]
            <= target.price_window_end
        )
        ledger.append(
            {
                "validation_event_id": event["validation_event_id"],
                "ticker": event["ticker"],
                "cached_query_start": target.price_window_start,
                "cached_query_end": target.price_window_end,
                "new_entry_session": w["entry_session_date"],
                "new_exit_session": w["exit_session_date"],
                "cached_endpoint_dates_verified": endpoint_dates_valid,
                "new_window_inside_cached_query_interval": within_queries,
                "reauthorization_attempted": endpoint_dates_valid and within_queries,
                "first_public_certified": False,
                "trading_admission": False,
            }
        )
        if not (endpoint_dates_valid and within_queries):
            continue
        payload = json.dumps([event["validation_event_id"], w], sort_keys=True).encode()
        component_id = hashlib.sha256(payload).hexdigest()
        components.append(
            {
                "security_event_component_id": component_id,
                "cik": event["cik"],
                "ticker": event["ticker"],
                "exchange_mic": event["exchange_mic"],
                "identity_key": event["security_id"],
                "formal_entry_session_date": w["entry_session_date"],
                "formal_exit_session_date": w["exit_session_date"],
            }
        )
        authorized.append(
            {
                "security_event_component_id": component_id,
                "return_outcome_read_authorized": True,
                "authorization_uses_return_values": False,
                "returns_computed": False,
            }
        )
        admitted.append(event["action_target_id"])
    if not admitted:
        return pd.DataFrame(), ledger
    result = calculate_validation_action_continuity(
        pd.DataFrame(authorized),
        pd.DataFrame(components),
        actions.loc[actions.action_target_id.isin(admitted)],
        identities,
        ticker_events,
        metadata[["price_target_id", "session_date"]],
        sessions,
    )
    return result, ledger


def all_case_return_gate(windows: list[dict[str, Any]], case_ids: list[str]) -> dict[str, Any]:
    """Keep the whole frozen sample blocked if any return window is incomplete."""
    if (
        len(case_ids) != 14
        or len(set(case_ids)) != 14
        or len(windows) != 14
        or ({r["validation_event_id"] for r in windows} != set(case_ids))
    ):
        raise ValueError("Fourteen identical frozen cases required; no readiness subset")
    if any(
        type(row.get("conditional_return_diagnostic_ready")) is not bool
        or bool(row.get("missing_checks")) == row["conditional_return_diagnostic_ready"]
        for row in windows
    ):
        raise ValueError("Window readiness must agree with explicit missing checks")
    ready = sum(r["conditional_return_diagnostic_ready"] is True for r in windows)
    return {
        "frozen_events": 14,
        "metadata_ready_events": ready,
        "all_case_return_gate": ready == 14,
        "blocked_tickers": [
            r["ticker"] for r in windows if r["conditional_return_diagnostic_ready"] is not True
        ],
        "partial_sample_returns_allowed": False,
    }


def supplementation_plan(
    events: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    session_audits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Specify exact window evidence gaps, not extra features or outcome-based replacements."""
    gate = all_case_return_gate(windows, [e["validation_event_id"] for e in events])
    jobs = []
    lookup = {e["validation_event_id"]: e for e in events}
    for row in windows:
        if row["conditional_return_diagnostic_ready"]:
            continue
        event, w = lookup[row["validation_event_id"]], row["window"]
        jobs.append(
            {
                **event,
                "window": w,
                "missing_checks": row["missing_checks"],
                "missing_issuer_sessions": session_audits[event["ticker"]]
                .loc[session_audits[event["ticker"]].issuer_source_sha256.isna(), "session_date"]
                .tolist(),
                "price_query": {
                    "ticker": event["ticker"],
                    "start": w["entry_session_date"],
                    "end": w["exit_session_date"],
                    "adjusted": False,
                    "timespan": "day",
                    "multiplier": 1,
                },
                "identity_query_dates": [w["entry_session_date"], w["exit_session_date"]],
                "actions_query_dates": [w["entry_session_date"], w["exit_session_date"]],
                "required_query_kinds": [
                    "daily_bars",
                    "reference_start",
                    "reference_end",
                    "ticker_events",
                    "splits",
                    "dividends",
                ],
                "first_public_certified": False,
                "trading_admission": False,
            }
        )
    return {
        **gate,
        "jobs": jobs,
        "network_requests_made": 0,
        "api_key_required": bool(jobs),
        "sample_replacements_allowed": False,
        "cached_bytes_reusable_only_after_window_specific_reaudit": True,
        "ticker_events_unavailable": "freeze_official_session_identity_fallback_before_collection",
        "new_bundle_required": True,
        "existing_evidence_must_not_be_overwritten": True,
        "numeric_forward_prices_loaded": False,
    }


def prior_industry_summary(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check prior SIC coverage without pretending fourteen events satisfy the twenty gate."""
    selected = {c["validation_event_id"]: c for c in cases}
    matched = [r for r in rows if r["validation_event_id"] in selected]
    if len(matched) != 14 or len({r["validation_event_id"] for r in matched}) != 14:
        raise ValueError("Industry metadata must retain the same fourteen cases")
    for row in matched:
        case = selected[row["validation_event_id"]]
        if row["cik"] != case["cik"] or (
            row["industry_ready"]
            and not utc_timestamp(row["industry_available_at"])
            < utc_timestamp(case["earliest_disclosure_at"])
        ):
            raise ValueError("Industry metadata is not a strictly prior issuer information set")
    frame = pd.DataFrame(matched)
    composition = (
        frame.loc[frame.industry_ready]
        .groupby("sic_2digit", as_index=False)
        .agg(
            events=("validation_event_id", "size"),
            issuers=("cik", "nunique"),
        )
    )
    return {
        "events": 14,
        "minimum_events": 20,
        "prior_industry_matches": int(frame.industry_ready.sum()),
        "composition": composition.to_dict("records"),
        "industry_adjusted_ic_computed": False,
        "gate_reason": "fewer_than_twenty_events_even_before_two_issuers_per_group_filter",
        "current_quarter_industry_forecast_feature_added": False,
    }
