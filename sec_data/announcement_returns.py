"""Same-case, proxy-clock return diagnostics; no strategy or first-public certification."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_replay import SIGNALS, case_row, verify_target_reconstruction
from sec_data.disclosure_audit import utc_timestamp
from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.execution_contract import execution_window

VERSION = "announcement-return-diagnostic-v1"
ACTION_COLUMNS = [
    "action_type",
    "effective_at",
    "cash_amount",
    "currency",
    "split_from",
    "split_to",
    "provider_event_id",
    "source_snapshot_sha256",
]


def select_reviewed_cases(
    cohort: list[dict[str, Any]], event_ids: list[str]
) -> list[dict[str, Any]]:
    """Keep every reviewed original GM pair, without selecting on market history or outcomes."""
    if len({r["validation_event_id"] for r in cohort}) != len(cohort):
        raise ValueError("Duplicate cohort event")
    cases = [r for r in cohort if r.get("release_value_pair_verified") is True]
    if (
        not cases
        or len(event_ids) != len(set(event_ids))
        or {r["validation_event_id"] for r in cases} != set(event_ids)
    ):
        raise ValueError("Select the entire frozen reviewed-value category")
    return sorted(cases, key=lambda r: r["earliest_disclosure_at"])


def _hashes(values: list[Any]) -> bool:
    return bool(values) and all(
        isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in values
    )


def audit_return_window(
    event: dict[str, Any],
    sessions: pd.DataFrame,
    issuer_metadata: pd.DataFrame,
    benchmark_metadata: pd.DataFrame,
    continuity: dict[str, Any],
    action: dict[str, Any],
    price: dict[str, Any],
    listing: dict[str, Any],
    queries: pd.DataFrame,
    benchmark_report: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Bind date-only provider continuity evidence to the newly scheduled 20-session window.

    Only dates, identities, query completeness and source hashes are allowed here.
    The alternative AMWD identity route stays labelled; later delisting does not
    remove an otherwise observed earlier window, but in-window delisting blocks it.
    """
    window = execution_window(event["announcement_at"], event["exchange_mic"], sessions)
    start, end = window["entry_session_date"], window["exit_session_date"]
    audit = (
        sessions.loc[
            sessions.exchange_mic.eq(event["exchange_mic"])
            & sessions.session_date.between(start, end),
            ["session_date", "regular_open_at", "regular_close_at", "source_snapshot_sha256"],
        ]
        .sort_values("session_date")
        .copy()
    )
    if len(audit) != 20:
        raise ValueError("Exactly 20 official holding sessions required")
    for role, rows, ticker in (
        ("issuer", issuer_metadata, event["ticker"]),
        ("benchmark", benchmark_metadata, "SPY"),
    ):
        if set(rows) != {"ticker", "session_date", "source_sha256"}:
            raise ValueError("Window eligibility may read metadata only")
        if not rows.ticker.eq(ticker).all() or rows.session_date.duplicated().any():
            raise ValueError("Ambiguous window price identity")
        if not _hashes(rows.source_sha256.tolist()):
            raise ValueError("Missing parsed-bar provenance")
        audit = audit.merge(
            rows[["session_date", "source_sha256"]].rename(
                columns={"source_sha256": role + "_source_sha256"}
            ),
            on="session_date",
            how="left",
            validate="one_to_one",
        )
    identity = all(
        continuity[k] == event[k] for k in ("cik", "ticker", "exchange_mic", "price_target_id")
    ) and (
        continuity["action_target_id"] == action["action_target_id"]
        and continuity["resolved_provider_security_id"] == event["security_id"]
        and event["security_id_kind"] == "share_class_figi"
        and all(
            continuity[k] is True
            for k in (
                "resolved_identity_ready",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
                "formal_action_evidence_ready",
            )
        )
        and continuity["identity_continuity_route"]
        in {
            "massive_ticker_events_or_session_grid",
            "same_ticker_event_stable_endpoints_plus_daily_bars",
            "stable_reference_endpoints_plus_official_session_daily_bars",
        }
        and _hashes(continuity["endpoint_identity_source_hashes"].split("|"))
        and continuity["authorized_entry_start"]
        <= start
        <= end
        <= continuity["authorized_exit_end"]
        and continuity["selection_uses_return_values"] is False
        and continuity["returns_computed"] is False
    )
    action_ready = all(
        action[k] == event[v]
        for k, v in (
            ("price_target_id", "price_target_id"),
            ("cik", "cik"),
            ("selected_ticker", "ticker"),
            ("selected_exchange_mic", "exchange_mic"),
            ("security_identity_key", "security_id"),
            ("security_identity_key_kind", "security_id_kind"),
        )
    ) and (
        all(action[k] is True for k in ("splits_query_complete", "dividends_query_complete"))
        and action["price_window_start"] <= start <= end <= action["price_window_end"]
        and action["rejected_action_rows"] == 0
        and action["selection_uses_return_values"] is False
    )
    price_ready = all(price[k] == event[k] for k in ("price_target_id", "cik", "ticker")) and (
        price["provider_query_complete"] is True
        and price["parser_rejection_rows"] == 0
        and price["price_window_start"] <= start <= end <= price["price_window_end"]
    )
    benchmark_ready = bool(
        len(queries) == 3
        and set(queries.query_kind) == {"daily_bars", "splits", "dividends"}
        and queries.status_code.eq(200).all()
        and queries.ticker.eq("SPY").all()
        and queries.benchmark_target_id.eq(event["benchmark_target_id"]).all()
        and queries.window_start.le(start).all()
        and queries.window_end.ge(end).all()
        and _hashes(queries.sha256.tolist())
        and benchmark_report["queries_complete"] == benchmark_report["queries_required"] == 3
        and benchmark_report["rejections"] == benchmark_report["action_normalization_failures"] == 0
        and benchmark_report["query_adjusted"] is False
        and benchmark_report["splits_explicit"] is True
        and benchmark_report["dividends_explicit"] is True
    )
    delisted = listing["delisting_date"]
    later = pd.notna(delisted) and str(delisted)[:10] > end
    listing_ready = all(
        listing[k] == event[k] for k in ("cik", "exchange_mic", "security_id", "security_id_kind")
    ) and (
        listing["listing_outcome_ready"] is True
        and listing["outcome_cutoff_date"] >= end
        and _hashes([listing["source_snapshot_sha256"]])
        and (
            (listing["listing_outcome_status"] == "active_at_outcome_cutoff" and pd.isna(delisted))
            or (listing["listing_outcome_status"] == "inactive_with_delisting_date" and later)
        )
    )
    checks = {
        "issuer_all_20_parsed_sessions": bool(audit.issuer_source_sha256.notna().all()),
        "benchmark_all_20_parsed_sessions": bool(audit.benchmark_source_sha256.notna().all()),
        "provider_identity_interval": bool(identity),
        "issuer_action_query_interval": bool(action_ready),
        "issuer_price_query_interval": bool(price_ready),
        "benchmark_query_interval": benchmark_ready,
        "listing_outcome_interval": bool(listing_ready),
    }
    return {
        "validation_event_id": event["validation_event_id"],
        "ticker": event["ticker"],
        "window": window,
        "checks": checks,
        "conditional_return_diagnostic_ready": all(checks.values()),
        "missing_checks": [k for k, passed in checks.items() if not passed],
        "identity_continuity_route": continuity["identity_continuity_route"],
        "later_delisted_after_window": bool(later),
        "first_public_certified": False,
        "trading_admission": False,
        "forward_numeric_price_values_loaded": False,
    }, audit


def case_signal_panel(
    baseline: pd.DataFrame, joint: pd.DataFrame, cases: list[dict[str, Any]]
) -> pd.DataFrame:
    """Extract identical cases after independently checking prior OOS dependencies."""
    rows = []
    for case in cases:
        target = case_row(joint, case)
        old = case_row(baseline, case)
        if (
            not all(
                bool(target[k])
                for k in (
                    "analysis_ready_level",
                    "analysis_ready_delta",
                    "analysis_ready_innovation_raw",
                    "analysis_ready_innovation_std",
                )
            )
            or not np.isfinite(target[list(SIGNALS)].to_numpy(dtype=float)).all()
        ):
            raise ValueError("Incomplete replay: keep case blocked, never replace it")
        verify_target_reconstruction(joint, case)
        if utc_timestamp(target.actual_available_at) != utc_timestamp(
            case["earliest_disclosure_at"]
        ):
            raise ValueError("Financial and execution proxy clocks differ")
        rows.append(
            {
                "validation_event_id": case["validation_event_id"],
                "cik": case["cik"],
                "ticker": case["assigned_ticker"],
                "target_accession_number": case["target_accession_number"],
                "actual_available_at": target.actual_available_at.isoformat(),
                **{k: float(target[k]) for k in SIGNALS},
                **{"filing_clock_" + k: float(old[k]) for k in SIGNALS},
            }
        )
    return pd.DataFrame(rows)


def normalize_holding_actions(
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    sessions: pd.DataFrame,
    window: dict[str, Any],
    ticker: str,
) -> pd.DataFrame:
    """Use local split ratios and ex-date cash entitlements, strictly after entry open."""
    calendar = sessions.loc[sessions.exchange_mic.eq(window["exchange_mic"])].set_index(
        "session_date"
    )
    entry, exit = utc_timestamp(window["entry_at"]), utc_timestamp(window["exit_at"])
    rows = []
    for kind, frame, date_key in (
        ("split", splits, "execution_date"),
        ("cash_dividend", dividends, "ex_dividend_date"),
    ):
        seen = set()
        for r in frame.to_dict("records"):
            day = str(r[date_key])[:10]
            if not window["entry_session_date"] <= day <= window["exit_session_date"]:
                continue
            if r["ticker"] != ticker or day not in calendar.index:
                raise ValueError("Action identity/date not covered by the official calendar")
            effective = utc_timestamp(calendar.loc[day, "regular_open_at"])
            if not entry < effective <= exit:
                continue
            if not _hashes([r["source_sha256"]]):
                raise ValueError("Corporate action lacks provenance")
            key = r["provider_event_id"]
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("Missing or duplicate corporate action identity")
            seen.add(key)
            row = {
                "action_type": kind,
                "effective_at": effective.isoformat(),
                "cash_amount": r["cash_amount"] if kind == "cash_dividend" else None,
                "currency": str(r["currency"]).upper() if kind == "cash_dividend" else "USD",
                "split_from": r["split_from"] if kind == "split" else None,
                "split_to": r["split_to"] if kind == "split" else None,
                "provider_event_id": key,
                "source_snapshot_sha256": r["source_sha256"],
            }
            if kind == "split" and any(
                not np.isfinite(float(row[k])) or float(row[k]) <= 0
                for k in ("split_from", "split_to")
            ):
                raise ValueError("Invalid local split ratio")
            rows.append(row)
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def calculate_case_return(
    event: dict[str, Any],
    evidence: dict[str, Any],
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute issuer and SPY cash-retained returns only after a frozen metadata pass."""
    if (
        evidence["validation_event_id"] != event["validation_event_id"]
        or evidence["ticker"] != event["ticker"]
        or evidence["conditional_return_diagnostic_ready"] is not True
        or not evidence["checks"]
        or not all(evidence["checks"].values())
        or evidence["first_public_certified"] is not False
        or evidence["trading_admission"] is not False
    ):
        raise ValueError("Return window is not conditionally authorized")
    window = evidence["window"]
    if utc_timestamp(window["announcement_at"]) != utc_timestamp(event["announcement_at"]):
        raise ValueError("Return clock does not match the admitted event")
    entry, exit = utc_timestamp(window["entry_at"]), utc_timestamp(window["exit_at"])
    detail: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for role, bars, actions, ticker in (
        ("issuer", issuer_bars, issuer_actions, event["ticker"]),
        ("benchmark", benchmark_bars, benchmark_actions, "SPY"),
    ):
        if not bars.ticker.eq(ticker).all() or bars.session_date.duplicated().any():
            raise ValueError("Ambiguous outcome bar identity")
        key = "price_target_id" if role == "issuer" else "benchmark_target_id"
        if not bars[key].eq(event[key]).all():
            raise ValueError("Outcome price target changed")
        endpoints = []
        for day, column in (
            (window["entry_session_date"], "open_unadjusted"),
            (window["exit_session_date"], "close_unadjusted"),
        ):
            matched = bars.loc[bars.session_date.eq(day)]
            if len(matched) != 1 or not _hashes(matched.source_sha256.tolist()):
                raise ValueError("Missing or ambiguous raw endpoint")
            endpoints.append(float(matched.iloc[0][column]))
            result[role + "_" + column + "_source_sha256"] = matched.iloc[0].source_sha256
        if not actions.empty:
            times = pd.to_datetime(actions.effective_at, utc=True)
            if not (times.gt(entry) & times.le(exit)).all():
                raise ValueError("Holding action outside the admitted window")
        holding, flows = calculate_cash_retained_holding_return(
            endpoints[0], endpoints[1], actions, currency="USD"
        )
        result.update(
            {
                role + "_entry_open": endpoints[0],
                role + "_exit_close": endpoints[1],
                role + "_total_return": holding.total_return,
                role + "_cash_dividend": holding.cash_dividend_value,
                role + "_split_factor": holding.cumulative_split_factor,
            }
        )
        detail.extend(
            {"validation_event_id": event["validation_event_id"], "instrument": role, **f}
            for f in flows
        )
    result.update(
        {
            **event,
            **window,
            "market_excess_return": result["issuer_total_return"]
            - result["benchmark_total_return"],
            "market_relative_wealth_return": (1 + result["issuer_total_return"])
            / (1 + result["benchmark_total_return"])
            - 1,
            "later_delisted_after_window": evidence["later_delisted_after_window"],
            "clock_basis": "reviewed_8k_acceptance_proxy_not_certified_first_public",
            "transaction_costs_included": False,
            "trading_admission": False,
        }
    )
    return result, detail


def descriptive_same_case_diagnostics(
    signals: pd.DataFrame, returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank diagnostics and leave-one-out sensitivity, never small-sample inference."""
    if (
        signals.validation_event_id.duplicated().any()
        or returns.validation_event_id.duplicated().any()
        or set(signals.validation_event_id) != set(returns.validation_event_id)
    ):
        raise ValueError("All representations require exactly the same event set")
    sample = signals.merge(
        returns, on="validation_event_id", suffixes=("", "_return"), validate="one_to_one"
    )
    if (
        not sample.cik.eq(sample.cik_return).all()
        or not sample.ticker.eq(sample.ticker_return).all()
    ):
        raise ValueError("Outcome and signal issuer identities differ")
    for actual, announced in zip(sample.actual_available_at, sample.announcement_at, strict=True):
        if utc_timestamp(actual) != utc_timestamp(announced):
            raise ValueError("Outcome and signal information sets differ")
    columns = [*SIGNALS, *("filing_clock_" + s for s in SIGNALS), "market_excess_return"]
    if not np.isfinite(sample[columns].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite complete case: never silently drop an event")

    def ic(frame: pd.DataFrame, column: str) -> float | None:
        x, y = frame[column], frame.market_excess_return
        if len(frame) < 3 or x.nunique() < 2 or y.nunique() < 2:
            return None
        return float(x.rank(method="average").corr(y.rank(method="average")))

    stats, influence = [], []
    for arm, prefix in (
        ("proxy_joint", ""),
        ("later_filing_signal_on_same_proxy_window", "filing_clock_"),
    ):
        for signal in SIGNALS:
            column = prefix + signal
            stats.append(
                {
                    "arm": arm,
                    "representation": signal,
                    "events": len(sample),
                    "spearman_ic": ic(sample, column),
                    "inferential_test": False,
                }
            )
            for issuer in sorted(sample.cik.unique()):
                reduced = sample.loc[sample.cik.ne(issuer)]
                influence.append(
                    {
                        "arm": arm,
                        "representation": signal,
                        "removed_cik": issuer,
                        "events": len(reduced),
                        "spearman_ic": ic(reduced, column),
                    }
                )
    return sample, pd.DataFrame(stats), pd.DataFrame(influence)
