from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from sec_data.massive_reference import MassiveReferenceClient
from shadow_market_collection import (
    ShadowMarketReadinessConfig,
    build_shadow_market_evidence_audit,
    calculate_shadow_market_order_readiness,
    collect_shadow_market_evidence_queries,
)


class _Sender:
    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del headers, timeout
        payload = {
            "status": "OK",
            "ticker": "AAA",
            "adjusted": False,
            "results": [
                {
                    "t": 1788134400000,
                    "o": 100.0,
                    "h": 102.0,
                    "l": 99.0,
                    "c": 101.0,
                    "v": 1000.0,
                }
            ],
        }
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def test_price_query_preserves_bytes_and_builds_audit(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    queries = pd.DataFrame([_price_query()])
    queries.to_parquet(plan / "shadow_market_evidence_queries.parquet", index=False)
    pd.DataFrame([_event()]).to_parquet(plan / "shadow_market_evidence_events.parquet", index=False)
    (plan / "output_manifest.json").write_text("{}", encoding="utf-8")
    client = MassiveReferenceClient(
        "abcdefgh",
        request_interval_seconds=0.0,
        max_retries=0,
        sender=_Sender(),
        now=lambda: datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
    )

    result = collect_shadow_market_evidence_queries(
        queries,
        client,
        raw_root=tmp_path / "raw",
        capture_date=date(2026, 9, 1),
    )
    audit = build_shadow_market_evidence_audit(
        plan,
        raw_root=tmp_path / "raw",
        raw_calendar_root=tmp_path / "calendar",
        output_root=tmp_path / "audit",
    )

    assert result.completed_queries == 1
    assert result.requests_made == 1
    assert audit.report["queries_with_captures"] == 1
    assert audit.report["price_observations"] == 1
    snapshots = pd.read_parquet(audit.output_root / "shadow_market_response_snapshots.parquet")
    raw_path = Path(snapshots.iloc[0]["path"])
    assert raw_path.is_file()
    assert len(snapshots.iloc[0]["sha256"]) == 64


def test_empty_plan_makes_no_request_and_builds_empty_audit(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    pd.DataFrame(columns=list(_price_query())).to_parquet(
        plan / "shadow_market_evidence_queries.parquet", index=False
    )
    pd.DataFrame([_event()]).to_parquet(plan / "shadow_market_evidence_events.parquet", index=False)
    (plan / "output_manifest.json").write_text(json.dumps({}), encoding="utf-8")

    result = collect_shadow_market_evidence_queries(
        pd.read_parquet(plan / "shadow_market_evidence_queries.parquet"),
        None,
        raw_root=tmp_path / "raw",
    )
    audit = build_shadow_market_evidence_audit(
        plan,
        raw_root=tmp_path / "raw",
        raw_calendar_root=tmp_path / "calendar",
        output_root=tmp_path / "audit",
    )

    assert result.requests_made == 0
    assert audit.report["planned_queries"] == 0
    assert audit.report["source_snapshots"] == 0


def test_readiness_uses_only_evidence_observed_before_second_session_open() -> None:
    dates = pd.bdate_range("2026-07-20", "2026-10-15")
    sessions = pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": dates.strftime("%Y-%m-%d"),
            "regular_open_at": dates.strftime("%Y-%m-%dT13:30:00Z"),
            "regular_close_at": dates.strftime("%Y-%m-%dT20:00:00Z"),
        }
    )
    accepted = pd.Timestamp("2026-08-28T20:00:00Z")
    session_close = pd.to_datetime(sessions["regular_close_at"], utc=True)
    prior = sessions.loc[session_close.lt(accepted)].tail(20)
    cost_rows = []
    for number, session_date in enumerate(prior["session_date"], start=1):
        close = 100.0 + number / 10.0
        cost_rows.append(
            {
                "prospective_event_id": "event-1",
                "evidence_class": "preacceptance_cost_bars",
                "session_date": session_date,
                "high_unadjusted": close + 1.0,
                "low_unadjusted": close - 1.0,
                "close_unadjusted": close,
                "observed_at": "2026-08-28T20:05:00Z",
            }
        )
    prices = pd.DataFrame(
        [
            *cost_rows,
            {
                "prospective_event_id": "event-1",
                "evidence_class": "execution_window_bars",
                "session_date": "2026-08-31",
                "high_unadjusted": 103.0,
                "low_unadjusted": 100.0,
                "close_unadjusted": 102.0,
                "observed_at": "2026-08-31T20:05:00Z",
            },
        ]
    )
    identities = pd.DataFrame(
        [
            {
                "prospective_event_id": "event-1",
                "ticker": "AAA",
                "exchange_mic": "XNAS",
                "observed_at": "2026-08-28T20:03:00Z",
            }
        ]
    )
    query_audit = pd.DataFrame(
        [
            {
                "market_evidence_query_id": f"query-{number}",
                "prospective_event_id": "event-1",
                "query_type": query_type,
                "query_capture_observed": True,
                "parse_rejections": 0,
                "first_fetched_at": "2026-08-31T20:06:00Z",
            }
            for number, query_type in enumerate(
                [
                    "point_in_time_identity",
                    "preacceptance_cost_bars",
                    "execution_window_bars",
                    "split_actions",
                    "cash_dividend_actions",
                ],
                start=1,
            )
        ]
    )

    result = calculate_shadow_market_order_readiness(
        pd.DataFrame([_event()]),
        query_audit,
        identities,
        prices,
        sessions,
        ShadowMarketReadinessConfig(),
    ).iloc[0]

    assert result["order_evidence_ready"]
    assert result["planned_entry_at"] == pd.Timestamp("2026-09-01T13:30:00Z")
    assert result["planned_exit_at"] == pd.Timestamp("2026-09-28T20:00:00Z")
    assert result["sizing_reference_price"] == 102.0
    assert 10.0 <= result["round_trip_cost_bps"] <= 100.0
    assert not result["return_values_loaded"]


def _price_query() -> dict[str, object]:
    return {
        "market_evidence_query_id": "query-1",
        "prospective_event_id": "event-1",
        "cik": "0000000001",
        "ticker": "AAA",
        "exchange_mic": "XNAS",
        "sec_accepted_at": "2026-08-31T20:00:00Z",
        "query_not_before": "2026-08-31T20:00:00Z",
        "query_type": "preacceptance_cost_bars",
        "provider_endpoint": "massive_unadjusted_daily_bars",
        "window_start": "2026-08-01",
        "window_end": "2026-08-31",
        "adjusted_prices": False,
        "query_status": "planned_not_collected",
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _event() -> dict[str, object]:
    return {
        "prospective_event_id": "event-1",
        "decision_as_of": "2026-08-31T20:10:00Z",
        "sec_accepted_at": "2026-08-28T20:00:00Z",
        "filing_first_observed_at": "2026-08-28T20:01:00Z",
        "registered_ticker": "AAA",
        "registered_exchange_mic": "XNAS",
        "security_candidate_ready": True,
        "point_in_time_identity_verified": False,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def test_query_audit_with_queries_but_no_snapshots_reports_zero_captures() -> None:
    # Regression: the first live run produced market queries before any market
    # capture existed, and the empty-snapshot summary lacked the count columns.
    from shadow_market_collection import _query_audit

    queries = pd.DataFrame({"market_evidence_query_id": ["q-1", "q-2"], "ticker": ["AAA", "BBB"]})
    audit = _query_audit(queries, pd.DataFrame(), pd.DataFrame())
    assert list(audit["capture_dates"]) == [0, 0]
    assert list(audit["snapshot_pages"]) == [0, 0]
    assert list(audit["parse_rejections"]) == [0, 0]
    assert not audit["query_capture_observed"].any()
    assert audit["first_fetched_at"].isna().all()
