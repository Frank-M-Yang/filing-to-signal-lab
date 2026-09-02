"""Tests for official historical exchange-session alignment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

import sec_data.exchange_calendar as exchange_calendar
from sec_data.exchange_calendar import (
    build_exchange_calendar_audit,
    collect_official_calendar_sources,
    official_calendar_sources,
)


def test_official_calendar_inventory_covers_guidance_study_years() -> None:
    """The longitudinal guidance cohort must never fall back to inferred sessions."""
    source_ids = {source.source_id for source in official_calendar_sources()}

    for year in range(2022, 2027):
        assert f"nyse-annual-{year}" in source_ids
        assert f"nasdaq-annual-{year}" in source_ids

    assert (
        exchange_calendar._closed_dates(2022)[datetime(2022, 6, 20).date()]
        == "Juneteenth National Independence Day observed"
    )
    assert (
        exchange_calendar._early_close_dates(2023)[datetime(2023, 7, 3).date()]
        == "Independence Day early close"
    )


class _Sender:
    def __init__(self, sources: tuple[exchange_calendar.OfficialCalendarSource, ...]) -> None:
        self.sources = sources
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del headers, timeout
        self.calls.append(url)
        source = self.sources[len(self.calls) - 1]
        if source.media_type == "application/pdf":
            content = f"%PDF-1.4\nfixture {source.source_id}\n%%EOF".encode()
        else:
            content = f"<html><body>fixture {source.source_id}</body></html>".encode()
        return httpx.Response(
            200,
            content=content,
            headers={"Content-Type": source.media_type},
            request=httpx.Request("GET", url),
        )


def test_official_calendar_handles_premarket_and_carter_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    fixture_sources = _fixture_sources()
    monkeypatch.setattr(exchange_calendar, "official_calendar_sources", lambda: fixture_sources)
    sender = _Sender(fixture_sources)
    collection = collect_official_calendar_sources(
        raw_root=raw_root,
        sender=sender,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    price_root = _write_price_fixture(tmp_path / "price")

    result = build_exchange_calendar_audit(
        price_root,
        raw_root=raw_root,
        output_root=tmp_path / "processed",
    )
    sessions = pd.read_parquet(result.output_root / "historical_exchange_sessions.parquet")
    alignment = pd.read_parquet(result.output_root / "event_calendar_alignment.parquet")
    exceptions = pd.read_parquet(result.output_root / "exchange_calendar_exceptions.parquet")

    assert collection.sources_completed == len(fixture_sources)
    assert collection.requests_made == len(fixture_sources)
    assert not sessions["session_date"].eq("2025-01-09").any()
    early = sessions[
        sessions["exchange_mic"].eq("XNAS") & sessions["session_date"].eq("2024-12-24")
    ].iloc[0]
    assert early["is_early_close"]
    assert early["regular_close_at"] == "2024-12-24T18:00:00+00:00"
    row = alignment.iloc[0]
    assert row["acceptance_session_relation"] == "accepted_before_regular_open"
    assert row["formal_entry_session_date"] == "2025-01-10"
    assert row["formal_entry_open_at"] == "2025-01-10T14:30:00+00:00"
    assert row["entry_date_changed_from_observed_candidate"]
    assert row["formal_event_window_ready"]
    carter = exceptions[
        exceptions["exchange_mic"].eq("XNAS") & exceptions["exception_date"].eq("2025-01-09")
    ].iloc[0]
    assert carter["source_id"] == "nasdaq-carter-closure-2025"
    american_early = sessions[
        sessions["exchange_mic"].eq("XASE") & sessions["session_date"].eq("2024-12-24")
    ].iloc[0]
    assert american_early["is_early_close"]
    assert american_early["source_id"] == "nyse-annual-2024"
    american_carter = exceptions[
        exceptions["exchange_mic"].eq("XASE") & exceptions["exception_date"].eq("2025-01-09")
    ].iloc[0]
    assert american_carter["source_id"] == "nyse-carter-closure-2025"
    assert set(alignment["exchange_mic"]) == {"XNAS", "XASE"}
    assert result.report["pilot_exchange_calendar_gate"] is True
    assert result.report["return_study_gate"] is False


def _fixture_sources() -> tuple[exchange_calendar.OfficialCalendarSource, ...]:
    rows: list[exchange_calendar.OfficialCalendarSource] = []
    for source in official_calendar_sources():
        if source.media_type == "application/pdf":
            content = f"%PDF-1.4\nfixture {source.source_id}\n%%EOF".encode()
        else:
            content = f"<html><body>fixture {source.source_id}</body></html>".encode()
        rows.append(
            exchange_calendar.OfficialCalendarSource(
                source_id=source.source_id,
                exchange_mic=source.exchange_mic,
                calendar_year=source.calendar_year,
                source_role=source.source_role,
                url=source.url,
                media_type=source.media_type,
                expected_sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return tuple(rows)


def _write_price_fixture(root: Path) -> Path:
    root.mkdir()
    sessions = pd.bdate_range("2024-12-20", "2025-02-28")
    closed = pd.to_datetime(["2024-12-25", "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17"])
    sessions = sessions[~sessions.isin(closed)]
    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "price_target_id": price_target_id,
                    "session_date": sessions.strftime("%Y-%m-%d"),
                    "open_unadjusted": 10.0,
                    "close_unadjusted": 10.5,
                }
            )
            for price_target_id in ("price-event-1", "price-event-xase")
        ],
        ignore_index=True,
    )
    targets = pd.DataFrame(
        [
            {
                "price_target_id": "price-event-1",
                "target_id": "event-1",
                "cik": "0000000001",
                "accepted_at": "2025-01-08T13:00:00+00:00",
                "selected_ticker": "DEMO",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "BBG001SABCDE",
                "entry_session_offset": 2,
                "holding_sessions": 20,
                "price_window_start": "2024-12-20",
                "price_window_end": "2025-02-28",
            },
            {
                "price_target_id": "price-event-xase",
                "target_id": "event-xase",
                "cik": "0000000002",
                "accepted_at": "2025-01-08T13:00:00+00:00",
                "selected_ticker": "AMER",
                "selected_exchange_mic": "XASE",
                "selected_provider_security_id": "BBG001SXYZ12",
                "entry_session_offset": 2,
                "holding_sessions": 20,
                "price_window_start": "2024-12-20",
                "price_window_end": "2025-02-28",
            },
        ]
    )
    price_audit = pd.DataFrame(
        [
            {
                "price_target_id": "price-event-1",
                "candidate_entry_session_date": "2025-01-13",
                "candidate_exit_session_date": "2025-02-10",
            },
            {
                "price_target_id": "price-event-xase",
                "candidate_entry_session_date": "2025-01-13",
                "candidate_exit_session_date": "2025-02-10",
            },
        ]
    )
    outputs = {
        "massive_daily_price_targets.parquet": targets,
        "massive_daily_price_observations.parquet": bars,
        "sample_massive_price_audit.parquet": price_audit,
    }
    artifacts: dict[str, dict[str, str]] = {}
    for filename, frame in outputs.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "massive_price_report.json").write_text(
        json.dumps({"dataset_version": "price-fixture-v1"}), encoding="utf-8"
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"dataset_version": "price-fixture-v1", "artifacts": artifacts}),
        encoding="utf-8",
    )
    return root
