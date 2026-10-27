"""Window-specific evidence closure for frozen publication-label sensitivities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sec_data.announcement_expanded_returns import reauthorize_cached_continuity
from sec_data.announcement_returns import audit_return_window
from sec_data.next_publication import (
    BENCHMARK_COLUMNS,
    DISARM_FLAGS,
    PRICE_COLUMNS,
    label_window_scenarios,
)

VERSION = "publication-market-bridge-v1"
TICKERS = {"NOBH", "AOS", "ROCK", "RAMP", "ORA", "CTSO"}


def validate_contract(c: dict[str, Any]) -> None:
    """Keep all six cases, unchanged execution and all model/return gates closed."""
    expected = {
        "version": VERSION,
        "frozen_queue_issuers": 20,
        "frozen_pair_cases": 6,
        "frozen_financial_scenarios": 17,
        "entry_session_offset": 2,
        "holding_sessions_inclusive": 20,
        "cached_window_reauthorization_allowed": True,
        "network_allowed": False,
        "price_values_allowed": False,
        "returns_allowed": False,
        "forecast_refit_allowed": False,
        "sample_replacement_allowed": False,
        "missing_price_imputation_allowed": False,
        "partial_sample_returns_allowed": False,
        "overwrite_originals": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported or weakened publication-market bridge contract")


def _one(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if len(frame) != 1:
        raise ValueError(f"Missing or ambiguous {label}")
    return {str(k): v for k, v in frame.to_dict("records")[0].items()}


def audit_case_windows(
    case: dict[str, Any],
    source: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    benchmark_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Reaudit every possible local date; missing action targets stay explicitly missing.

    Noon is only a date-based scheduling device. It is not a financial cutoff,
    first-public timestamp or claim that a reconstructed forecast existed live.
    Absence of a parsed bar is not evidence of no trades, a halt or zero return.
    """
    if (
        any(source[k] != case[k] for k in ("cik", "ticker", "validation_event_id"))
        or any(source.get(k) is not False for k in DISARM_FLAGS)
        or tables["metadata"].columns.tolist() != PRICE_COLUMNS
        or tables["benchmark_metadata"].columns.tolist() != BENCHMARK_COLUMNS
    ):
        raise ValueError("Source identity or metadata-only projection changed")
    target = _one(
        tables["targets"].loc[
            tables["targets"].cik.eq(case["cik"])
            & tables["targets"].selected_ticker.eq(case["ticker"])
            & tables["targets"].selected_exchange_mic.eq(case["assigned_exchange_mic"])
        ],
        "frozen price target",
    )
    if (
        ("price_target_id" in case and target["price_target_id"] != case["price_target_id"])
        or any(
            target[k] != case[v]
            for k, v in (
                ("cik", "cik"),
                ("selected_ticker", "ticker"),
                ("selected_exchange_mic", "assigned_exchange_mic"),
            )
        )
        or target["adjusted"] is not False
        or target["selection_uses_return_values"] is not False
        or target["security_identity_key_kind"] != "share_class_figi"
    ):
        raise ValueError("Wrong or outcome-selected parent security identity")
    prices = tables["metadata"].loc[
        tables["metadata"].price_target_id.eq(target["price_target_id"])
    ]
    price = _one(
        tables["price_audit"].loc[
            tables["price_audit"].price_target_id.eq(target["price_target_id"])
        ],
        "frozen price audit",
    )
    actions = tables["actions"].loc[tables["actions"].price_target_id.eq(target["price_target_id"])]
    if len(actions) > 1:
        raise ValueError("Ambiguous action target")
    listed = tables["listing"].loc[
        tables["listing"].cik.eq(case["cik"])
        & tables["listing"].security_id.eq(target["security_identity_key"])
    ]
    listing = _one(listed, "frozen listing outcome")
    if tables["queries"].benchmark_target_id.nunique() != 1:
        raise ValueError("Ambiguous benchmark target")
    audits, date_frames, gaps = [], [], []
    for scenario in label_window_scenarios(
        source, case["assigned_exchange_mic"], tables["sessions"]
    ):
        if scenario["schedule"] is None:
            raise ValueError("Frozen calendar cannot schedule one of the retained source scenarios")
        date = scenario["scenario_local_date"]
        event = {
            "validation_event_id": case["validation_event_id"],
            "cik": case["cik"],
            "ticker": case["ticker"],
            "exchange_mic": case["assigned_exchange_mic"],
            "price_target_id": target["price_target_id"],
            "security_id": target["security_identity_key"],
            "security_id_kind": target["security_identity_key_kind"],
            "benchmark_target_id": str(tables["queries"].benchmark_target_id.iloc[0]),
            "announcement_at": (pd.Timestamp(date) + pd.Timedelta(hours=12))
            .tz_localize("America/New_York")
            .isoformat(),
        }
        missing_action = actions.empty
        reauthorization: list[dict[str, Any]]
        if missing_action:
            # Failed checks, not fabricated zero corporate actions or a ready identity.
            event["action_target_id"] = None
            action = {
                "action_target_id": None,
                "price_target_id": event["price_target_id"],
                "cik": event["cik"],
                "selected_ticker": event["ticker"],
                "selected_exchange_mic": event["exchange_mic"],
                "security_identity_key": event["security_id"],
                "security_identity_key_kind": event["security_id_kind"],
                "splits_query_complete": False,
                "dividends_query_complete": False,
                "price_window_start": "",
                "price_window_end": "",
                "rejected_action_rows": None,
                "selection_uses_return_values": False,
            }
            continuity = pd.DataFrame()
            reauthorization = []
        else:
            action = _one(actions, "action audit")
            event["action_target_id"] = action["action_target_id"]
            continuity, reauthorization = reauthorize_cached_continuity(
                [event],
                [
                    {
                        "validation_event_id": event["validation_event_id"],
                        "window": scenario["schedule"],
                    }
                ],
                actions,
                tables["identities"],
                tables["ticker_events"],
                prices,
                tables["sessions"],
            )
        if continuity.empty:
            identity = {
                **{k: event[k] for k in ("cik", "ticker", "exchange_mic", "price_target_id")},
                "action_target_id": event["action_target_id"],
                "resolved_provider_security_id": None,
                "resolved_identity_ready": False,
                "historical_symbol_interval_ready": False,
                "corporate_actions_ready": False,
                "formal_action_evidence_ready": False,
                "identity_continuity_route": "missing_cached_action_target"
                if missing_action
                else "cached_interval_or_endpoint_not_reauthorized",
            }
        else:
            identity = _one(continuity, "new window continuity")
        result, dates = audit_return_window(
            event,
            tables["sessions"],
            prices.drop(columns="price_target_id"),
            tables["benchmark_metadata"],
            identity,
            action,
            price,
            listing,
            tables["queries"],
            benchmark_report,
        )
        if any(
            result["window"][k] != scenario["schedule"][k]
            for k in ("entry_at", "exit_at", "entry_price_type", "exit_price_type")
        ):
            raise ValueError("The scheduling representative changed the execution contract")
        result["window"]["announcement_at"] = None
        result.update(
            **{
                k: event[k]
                for k in (
                    "cik",
                    "exchange_mic",
                    "price_target_id",
                    "security_id",
                    "security_id_kind",
                )
            },
            scenario_local_date=date,
            representative_time_is_not_publication_time=True,
            missing_cached_action_target=missing_action,
            window_specific_reauthorization=reauthorization,
            **dict.fromkeys(DISARM_FLAGS, False),
        )
        audits.append(result)
        dates["validation_event_id"] = event["validation_event_id"]
        dates["ticker"] = event["ticker"]
        dates["scenario_local_date"] = date
        date_frames.append(dates)
        for day in dates.loc[dates.issuer_source_sha256.isna(), "session_date"]:
            within = price["price_window_start"] <= day <= price["price_window_end"]
            classification = (
                "outside_cached_query_interval"
                if not within
                else "incomplete_price_query"
                if price["provider_query_complete"] is not True
                else "parser_rejection_present"
                if price["parser_rejection_rows"] != 0
                else "no_parsed_bar_despite_complete_query"
            )
            gaps.append(
                {
                    "validation_event_id": event["validation_event_id"],
                    "ticker": event["ticker"],
                    "scenario_local_date": date,
                    "session_date": day,
                    "classification": classification,
                    "is_entry_session": day == result["window"]["entry_session_date"],
                    "is_exit_session": day == result["window"]["exit_session_date"],
                    "no_trading_halt_or_zero_return_inferred": True,
                }
            )
    return audits, pd.concat(date_frames, ignore_index=True), gaps


def close_common_gate(
    audits: list[dict[str, Any]], financial: list[dict[str, Any]]
) -> dict[str, Any]:
    """Retain every financial arm and date scenario; never evaluate a convenient subset."""
    by_ticker = {t: [r for r in audits if r["ticker"] == t] for t in TICKERS}
    if (
        {r["ticker"] for r in audits} != TICKERS
        or any(not rows for rows in by_ticker.values())
        or len({r["validation_event_id"] for r in audits}) != 6
        or len({(r["validation_event_id"], r["scenario_local_date"]) for r in audits})
        != len(audits)
        or len(financial) != 102
        or len({r["scenario_id"] for r in financial}) != 17
        or any(r.get(k) is not False for r in audits for k in DISARM_FLAGS)
        or any(r.get(k) is not False for r in financial for k in DISARM_FLAGS)
        or any(
            r["conditional_return_diagnostic_ready"] is not all(r["checks"].values())
            or any(type(v) is not bool for v in r["checks"].values())
            or set(r["checks"])
            != {
                "issuer_all_20_parsed_sessions",
                "benchmark_all_20_parsed_sessions",
                "provider_identity_interval",
                "issuer_action_query_interval",
                "issuer_price_query_interval",
                "benchmark_query_interval",
                "listing_outcome_interval",
            }
            or set(r["missing_checks"]) != {k for k, v in r["checks"].items() if not v}
            or len(r["missing_checks"]) != len(set(r["missing_checks"]))
            or r["window"]["announcement_at"] is not None
            or r["representative_time_is_not_publication_time"] is not True
            for r in audits
        )
    ):
        raise ValueError("Frozen six-case/seventeen-arm denominator changed")
    matched = set()
    for row in financial:
        identity = (row["scenario_id"], row["validation_event_id"])
        if identity in matched or row["ticker"] not in TICKERS:
            raise ValueError("Duplicate financial scenario identity")
        matched.add(identity)
        cutoff = pd.Timestamp(row["actual_available_at"])
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("Financial availability needs an explicit timezone")
        date = cutoff.tz_convert("America/New_York").date().isoformat()
        if (
            len(
                [
                    r
                    for r in by_ticker[row["ticker"]]
                    if r["validation_event_id"] == row["validation_event_id"]
                    and r["cik"] == row["cik"]
                    and r["scenario_local_date"] == date
                ]
            )
            != 1
        ):
            raise ValueError("Financial ordering arm lacks its exact calendar-date window")
    ready = [
        t
        for t, rows in by_ticker.items()
        if all(r["conditional_return_diagnostic_ready"] for r in rows)
    ]
    return {
        "metadata_window_ready_tickers": sorted(ready),
        "metadata_window_blocked_tickers": sorted(TICKERS - set(ready)),
        "all_six_metadata_windows_ready": len(ready) == 6,
        "common_financial_gate": all(
            r["same_six_common_financial_gate"] is True
            and r["complete_financial_representations"] is True
            for r in financial
        ),
        "all_102_financial_arms_matched_to_calendar_date": True,
        "historical_publication_evidence_still_required": True,
        "partial_sample_returns_allowed": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
