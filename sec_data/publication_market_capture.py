"""Source-bound parsing and conservative market-evidence overlays, not returns."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from sec_data.massive_prices import parse_massive_daily_bars
from sec_data.massive_reference import MassiveRequestError, parse_massive_ticker_response
from sec_data.massive_security_actions import (
    parse_massive_dividends,
    parse_massive_splits,
    parse_massive_ticker_events,
)
from sec_data.next_publication import DISARM_FLAGS, PRICE_COLUMNS


def parse_query(
    request: dict[str, Any], content: bytes, sha256: str, action_target_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use existing validators; a 200/OK response alone never proves identity or coverage."""
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Captured response hash changed before parsing")
    kind = request["query_kind"]
    status: dict[str, Any] = {
        "query_id": request["query_id"],
        "ticker": request["ticker"],
        "query_kind": kind,
        "source_sha256": sha256,
        "query_complete": False,
        "parser_error": None,
        "parsed_rows": 0,
        "rejected_rows": 0,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    try:
        document = json.loads(content)
        if not isinstance(document, dict) or document.get("status") != "OK":
            raise MassiveRequestError("Invalid response status", reason="bad_status")
        if document.get("next_url"):
            raise MassiveRequestError("Uncollected pagination", reason="pagination_incomplete")
        base = {"source_sha256": sha256, "action_target_id": action_target_id}
        if kind.startswith("reference_") or kind.startswith("listing_"):
            day = (
                request["listing_outcome_cutoff"]
                if kind.startswith("listing_")
                else request["window_start" if kind == "reference_start" else "window_end"]
            )
            rows, rejected, _ = parse_massive_ticker_response(
                content,
                expected_cik=request["cik"],
                target_id=action_target_id,
                query_active=kind != "listing_inactive",
                as_of_date=day,
                source_sha256=sha256,
            )
            for row in rows:
                row["action_target_id"] = row.pop("target_id")
                row["query_kind"] = kind
        elif kind == "ticker_events":
            result = document.get("results")
            if (
                not isinstance(result, dict)
                or result.get("composite_figi") != request["selected_composite_figi"]
            ):
                raise MassiveRequestError(
                    "Timeline identity differs", reason="timeline_identity_mismatch"
                )
            rows, rejected, _ = parse_massive_ticker_events(content, **base)
        elif kind in {"splits", "dividends"}:
            args = {
                **base,
                "expected_ticker": request["ticker"],
                "window_start": request["window_start"],
                "window_end": request["window_end"],
            }
            rows, rejected, _ = (
                parse_massive_splits(content, **args)
                if kind == "splits"
                else parse_massive_dividends(content, expected_currency="USD", **args)
            )
        elif kind == "daily_bars":
            validated, rejected, _ = parse_massive_daily_bars(
                content,
                expected_ticker=request["ticker"],
                price_target_id=request["price_target_id"],
                window_start=request["window_start"],
                window_end=request["window_end"],
                source_sha256=sha256,
            )
            # OHLC is validated only for source normalization, then discarded.
            # No price values enter a signal, sample choice or return calculation.
            rows = [{k: r[k] for k in PRICE_COLUMNS} for r in validated]
        else:
            raise ValueError("Unknown frozen query kind")
    except (json.JSONDecodeError, UnicodeDecodeError, MassiveRequestError) as exc:
        status["parser_error"] = (
            exc.reason if isinstance(exc, MassiveRequestError) else "malformed_json"
        )
        return status, [], []
    status.update(query_complete=True, parsed_rows=len(rows), rejected_rows=len(rejected))
    return status, rows, rejected


def action_audit(
    targets: pd.DataFrame,
    identities: pd.DataFrame,
    timeline: pd.DataFrame,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    statuses: list[dict[str, Any]],
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Authorize exact endpoint IDs plus a bound timeline, never invent an OTC FIGI.

    The established continuity checker remains responsible for any endpoint/bar
    fallback. A queried empty action response can establish zero observed events;
    a missing query cannot. Actions outside official sessions remain rejected.
    """
    output = []
    for t in targets.to_dict("records"):
        ticker, aid = t["selected_ticker"], t["action_target_id"]
        q = {r["query_kind"]: r for r in statuses if r["ticker"] == ticker}
        r = {**t, "historical_symbol_interval_ready": False, "corporate_actions_ready": False}
        for kind in ("reference_start", "reference_end", "ticker_events", "splits", "dividends"):
            r[kind + "_query_complete"] = q.get(kind, {}).get("query_complete") is True
        exact = identities.loc[identities.action_target_id.eq(aid)]
        for field, wanted in (
            ("cik", t["cik"]),
            ("ticker", ticker),
            ("exchange_mic", t["selected_exchange_mic"]),
            ("security_type", "CS"),
            ("provider_security_id", t["selected_provider_security_id"]),
            ("share_class_figi", t["selected_share_class_figi"]),
            ("composite_figi", t["selected_composite_figi"]),
        ):
            exact = exact.iloc[:0] if pd.isna(wanted) else exact.loc[exact[field].eq(wanted)]
        endpoint = all(
            len(exact.loc[exact.query_kind.eq(kind) & exact.as_of_date.eq(day)]) == 1
            for kind, day in (
                ("reference_start", t["price_window_start"]),
                ("reference_end", t["price_window_end"]),
            )
        )
        changes = timeline.loc[
            timeline.action_target_id.eq(aid)
            & timeline.event_date.between(t["price_window_start"], t["price_window_end"])
        ]
        r["rejected_identity_rows"] = sum(
            q.get(k, {}).get("rejected_rows", 0)
            for k in ("reference_start", "reference_end", "ticker_events")
        )
        r["rejected_action_rows"] = sum(
            q.get(k, {}).get("rejected_rows", 0) for k in ("splits", "dividends")
        )
        r["historical_symbol_interval_ready"] = (
            endpoint
            and changes.empty
            and r["rejected_identity_rows"] == 0
            and all(
                r[k + "_query_complete"]
                for k in ("reference_start", "reference_end", "ticker_events")
            )
        )
        official = set(
            sessions.loc[sessions.exchange_mic.eq(t["selected_exchange_mic"]), "session_date"]
        )
        observed_dates = [
            *splits.loc[splits.action_target_id.eq(aid), "execution_date"].tolist(),
            *dividends.loc[dividends.action_target_id.eq(aid), "ex_dividend_date"].tolist(),
        ]
        r["corporate_actions_ready"] = (
            r["splits_query_complete"]
            and r["dividends_query_complete"]
            and r["rejected_action_rows"] == 0
            and set(observed_dates).issubset(official)
        )
        r["observed_split_count"] = len(splits.loc[splits.action_target_id.eq(aid)])
        r["observed_dividend_count"] = len(dividends.loc[dividends.action_target_id.eq(aid)])
        r["empty_actions_are_evidence_only_if_both_queries_complete"] = True
        output.append(r)
    return pd.DataFrame(output)


def overlay_price_dates(
    cached: pd.DataFrame, new: pd.DataFrame, request: dict[str, Any], status: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quarantine new coverage losses; never replace or forward-fill cached observations.

    Query ranges may differ. This detects an overlap-coverage discrepancy, not a
    same-query revision, price revision, actual halt or absence of trading.
    """
    if cached.columns.tolist() != PRICE_COLUMNS or new.columns.tolist() != PRICE_COLUMNS:
        raise ValueError("Date overlay requires metadata-only projections")
    tid, ticker = request["price_target_id"], request["ticker"]
    selected = cached.loc[cached.price_target_id.eq(tid) & cached.ticker.eq(ticker)]
    if (
        not new.price_target_id.eq(tid).all()
        or not new.ticker.eq(ticker).all()
        or new.session_date.duplicated().any()
    ):
        raise ValueError("New price dates have ambiguous identities")
    prior = selected.loc[
        selected.session_date.between(request["window_start"], request["window_end"])
    ]
    disappeared = sorted(set(prior.session_date) - set(new.session_date))
    usable = status["query_complete"] is True and status["rejected_rows"] == 0 and not disappeared
    additions = new.loc[~new.session_date.isin(selected.session_date)] if usable else new.iloc[:0]
    merged = (
        cached.copy(deep=True)
        if additions.empty
        else pd.concat([cached, additions], ignore_index=True)
    )
    return merged, {
        "ticker": ticker,
        "price_target_id": tid,
        "query_id": request["query_id"],
        "new_source_sha256": status["source_sha256"],
        "cached_overlap_dates": sorted(set(prior.session_date)),
        "cached_overlap_source_hashes": sorted(set(prior.source_sha256)),
        "new_parsed_dates": sorted(set(new.session_date)),
        "cached_dates_absent_from_new_response": disappeared,
        "overlap_coverage_discrepancy": bool(disappeared) and status["query_complete"] is True,
        "new_query_reauthorized": usable,
        "new_dates_added": len(additions),
        "cached_observations_replaced": False,
        "zero_return_or_no_trade_inferred": False,
        "queries_not_assumed_identical": True,
        "price_value_changes_not_tested": True,
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def history_access_sensitivity(
    dates: list[str], old_fetched_at: str, new_fetched_at: str, *, history_days: int = 730
) -> dict[str, Any]:
    """Test a declared rolling-day hypothesis, not the user's actual plan entitlement.

    Two calendar years need not be exactly 730 days. Date-only UTC cutoffs here
    are explicit sensitivities, not verified vendor permission boundaries.
    They never override the overlap quarantine or imply no trading.
    """
    old, new = pd.Timestamp(old_fetched_at), pd.Timestamp(new_fetched_at)
    if old.tzinfo is None or new.tzinfo is None or pd.isna(old) or pd.isna(new):
        raise ValueError("Capture times require explicit timezones")
    if type(history_days) is not int or history_days < 1 or old > new:
        raise ValueError("Invalid declared history-window sensitivity")
    old, new = old.tz_convert("UTC").normalize(), new.tz_convert("UTC").normalize()
    rows = []
    for d in dates:
        day = pd.Timestamp(d, tz="UTC")
        old_age, new_age = (old - day).days, (new - day).days
        rows.append(
            {
                "session_date": d,
                "age_days_at_old_capture": old_age,
                "age_days_at_new_capture": new_age,
                "within_assumed_old_history_window": 0 <= old_age <= history_days,
                "outside_assumed_new_history_window": new_age > history_days,
            }
        )
    return {
        "declared_history_days": history_days,
        "old_capture_utc_date": old.date().isoformat(),
        "new_capture_utc_date": new.date().isoformat(),
        "new_assumed_lower_date": (new - pd.Timedelta(days=history_days)).date().isoformat(),
        "date_precision_only_not_verified_permission_boundary": True,
        "actual_key_plan_verified": False,
        "history_horizon_rule_verified": False,
        "all_discrepant_dates_fit_expiry_hypothesis": bool(rows)
        and all(
            r["within_assumed_old_history_window"] and r["outside_assumed_new_history_window"]
            for r in rows
        ),
        "expiry_cause_confirmed": False,
        "rows": rows,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
