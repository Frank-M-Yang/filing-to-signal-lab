"""Strict parsing of a bounded MBUU historical-data probe, never sample admission."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sec_data.massive_prices import parse_massive_daily_bars
from sec_data.massive_reference import MassiveRequestError, parse_massive_ticker_response
from sec_data.massive_security_actions import (
    parse_massive_dividends,
    parse_massive_splits,
    parse_massive_ticker_events,
)

START, END = "2024-08-06", "2024-10-30"
CIK, SHARE_FIGI, COMPOSITE_FIGI = "0001590976", "BBG005PXTR61", "BBG005PXTR70"


def audit_primary_response(
    content: bytes, query: dict[str, str], required_sessions: list[str]
) -> dict[str, Any]:
    """Distinguish complete evidence from empty/truncated/rejected provider responses.

    This audits successful HTTP responses only. No price following the frozen
    event date is accepted; a partial 200 never establishes price coverage.
    Endpoint identity observations do not by themselves certify an interval.
    """
    digest = hashlib.sha256(content).hexdigest()
    kind, ticker = query["kind"], query["ticker"]
    kwargs = {
        "expected_ticker": ticker,
        "window_start": START,
        "window_end": END,
        "source_sha256": digest,
    }
    try:
        document = json.loads(content)
        if not isinstance(document, dict) or document.get("status") != "OK":
            raise ValueError("Provider status is not OK")
        if "results" not in document:
            raise ValueError("Missing result collection is unknown, not zero observations")
        if kind == "bars":
            rows, rejected, next_url = parse_massive_daily_bars(
                content, price_target_id=query["id"], **kwargs
            )
        elif kind == "reference":
            rows, rejected, next_url = parse_massive_ticker_response(
                content,
                expected_cik=CIK,
                target_id=query["id"],
                query_active=True,
                as_of_date=query["date"],
                source_sha256=digest,
            )
        elif kind == "splits":
            rows, rejected, next_url = parse_massive_splits(
                content, action_target_id=query["id"], **kwargs
            )
        elif kind == "dividends":
            rows, rejected, next_url = parse_massive_dividends(
                content, action_target_id=query["id"], expected_currency="USD", **kwargs
            )
        else:
            raise ValueError("Unsupported query kind")
        complete = not rejected and not next_url and not document.get("next_url")
        report: dict[str, Any] = {
            "status": "parsed",
            "rows": rows,
            "rejections": rejected,
            "observed_rows": len(rows),
            "pagination_remaining": bool(next_url or document.get("next_url")),
            "evidence_complete": bool(complete),
        }
        if kind == "bars":
            dates = [r["session_date"] for r in rows]
            exact = len(dates) == len(set(dates)) and sorted(dates) == required_sessions
            report.update(
                missing_sessions=sorted(set(required_sessions) - set(dates)),
                unexpected_sessions=sorted(set(dates) - set(required_sessions)),
                evidence_complete=bool(complete and exact),
                unadjusted=True,
            )
        elif kind == "reference":
            exact = len(rows) == 1 and all(
                rows[0].get(k) == v
                for k, v in {
                    "ticker": "MBUU",
                    "share_class_figi": SHARE_FIGI,
                    "composite_figi": COMPOSITE_FIGI,
                    "exchange_mic": "XNAS",
                    "currency_name": "usd",
                    "security_type": "CS",
                    "market": "stocks",
                    "locale": "us",
                    "provider_active": True,
                }.items()
            )
            report["evidence_complete"] = bool(complete and exact)
        return report
    except (ValueError, KeyError, TypeError, MassiveRequestError):
        # No provider message is copied to logs: it could contain reflected secrets.
        return {"status": "invalid_response", "evidence_complete": False, "observed_rows": None}


def audit_cached_timeline(content: bytes) -> dict[str, Any]:
    """Check pinned experimental timeline identity and MBUU symbol continuity.

    This is retrospective corroboration, not exchange-certified exhaustive
    coverage. Events outside the pre-event interval are not market features.
    """
    digest = hashlib.sha256(content).hexdigest()
    try:
        doc = json.loads(content)
        result = doc["results"]
        if result.get("cik") != CIK or result.get("composite_figi") != COMPOSITE_FIGI:
            raise ValueError("Timeline identity mismatch")
        if "events" not in result:
            raise ValueError("Missing ticker history")
        rows, rejected, _ = parse_massive_ticker_events(
            content, action_target_id="mbuu-pre-event", source_sha256=digest
        )
        past = sorted((r for r in rows if r["event_date"] <= START), key=lambda r: r["event_date"])
        inside = [r for r in rows if START < r["event_date"] <= END]
        unique = len({r["event_date"] for r in rows}) == len(rows)
        consistent = bool(
            past and past[-1]["new_ticker"] == "MBUU" and not inside and not rejected and unique
        )
        return {"consistent": consistent, "experimental_endpoint": True, "sha256": digest}
    except (ValueError, KeyError, TypeError, MassiveRequestError):
        return {"consistent": False, "experimental_endpoint": True, "sha256": digest}
