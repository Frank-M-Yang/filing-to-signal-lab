"""Tests for auditable Massive unadjusted daily-price ingestion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from sec_data.massive_prices import (
    build_massive_price_audit,
    build_massive_price_target_plan,
    collect_massive_price_artifacts,
    load_massive_price_targets,
    massive_daily_bars_url,
    parse_massive_daily_bars,
)
from sec_data.massive_reference import (
    MassiveReferenceClient,
    MassiveRequestError,
)


class _Sender:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del timeout
        self.calls.append((url, headers))
        return self.responses.pop(0)


def test_daily_bar_url_is_unadjusted_and_secret_free() -> None:
    url = massive_daily_bars_url("BRK/B", "2025-01-01", "2025-02-01")

    assert "/BRK%2FB/range/1/day/" in url
    assert "adjusted=false" in url
    assert "apiKey" not in url


def test_daily_bar_parser_rejects_adjusted_response() -> None:
    payload = json.dumps(
        {"status": "OK", "ticker": "DEMO", "adjusted": True, "results": []}
    ).encode()

    with pytest.raises(MassiveRequestError, match="unadjusted"):
        parse_massive_daily_bars(
            payload,
            expected_ticker="DEMO",
            price_target_id="price-event-1",
            window_start="2025-01-01",
            window_end="2025-02-01",
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_daily_bar_parser_rejects_inconsistent_ohlc() -> None:
    payload = _daily_payload("DEMO", count=1, inconsistent=True)

    rows, rejections, next_url = parse_massive_daily_bars(
        payload,
        expected_ticker="DEMO",
        price_target_id="price-event-1",
        window_start="2025-01-01",
        window_end="2025-02-28",
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert not rows
    assert rejections[0]["reason"] == "high_price_inconsistent"
    assert next_url is None


def test_price_plan_collection_and_audit_create_candidate_only(tmp_path: Path) -> None:
    reference = _write_reference_fixture(tmp_path / "reference")
    plan = build_massive_price_target_plan(reference, output_root=tmp_path / "targets")
    targets = load_massive_price_targets(plan.output_root)
    target = targets.iloc[0]
    query_url = massive_daily_bars_url(
        str(target["selected_ticker"]),
        str(target["price_window_start"]),
        str(target["price_window_end"]),
    )
    payload = _daily_payload("DEMO", count=25)
    sender = _Sender(
        [
            httpx.Response(
                200,
                content=payload,
                headers={"Content-Type": "application/json"},
                request=httpx.Request("GET", query_url),
            )
        ]
    )
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=sender,
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    collection = collect_massive_price_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_massive_price_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_massive_price_audit.parquet")
    windows = pd.read_parquet(audit.output_root / "event_window_candidates.parquet")

    assert collection.requests_made == 1
    assert collection.targets_completed == 1
    assert sender.calls[0][1]["Authorization"] == "Bearer secret-api-key"
    assert sample.loc[0, "daily_bar_count"] == 25
    assert sample.loc[0, "observed_event_window_candidate"]
    assert len(windows) == 1
    assert windows.loc[0, "holding_sessions"] == 20
    assert audit.report["candidate_returns_computed"] is False
    assert audit.report["historical_exchange_calendar_gate"] is False
    assert audit.report["return_study_gate"] is False


def _daily_payload(ticker: str, *, count: int, inconsistent: bool = False) -> bytes:
    current = datetime(2025, 1, 3, 5, tzinfo=timezone.utc)
    bars: list[dict[str, float | int]] = []
    while len(bars) < count:
        if current.weekday() < 5:
            number = len(bars)
            opening = 10.0 + number
            high = opening - 1.0 if inconsistent else opening + 1.0
            bars.append(
                {
                    "t": int(current.timestamp() * 1000),
                    "o": opening,
                    "h": high,
                    "l": opening - 1.0,
                    "c": opening + 0.5,
                    "v": 1000 + number,
                    "vw": opening + 0.25,
                    "n": 100 + number,
                }
            )
        current += timedelta(days=1)
    return json.dumps(
        {
            "status": "OK",
            "ticker": ticker,
            "adjusted": False,
            "resultsCount": len(bars),
            "results": bars,
        }
    ).encode()


def _write_reference_fixture(root: Path) -> Path:
    root.mkdir()
    source_sha256 = hashlib.sha256(b"reference-source").hexdigest()
    sample = pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "cik": "0000000001",
                "accepted_at": "2025-01-02T21:00:00+00:00",
                "query_date": "2025-01-02",
                "form": "10-Q",
                "accession_number": "0000000001-25-000001",
                "event_date_reference_candidate": True,
                "selected_ticker": "DEMO",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "BBG001SABCDE",
                "selected_provider_security_id_kind": "share_class_figi",
            }
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "ticker": "DEMO",
                "provider_security_id": "BBG001SABCDE",
                "currency_name": "usd",
                "source_sha256": source_sha256,
            }
        ]
    )
    sample_path = root / "sample_massive_reference_audit.parquet"
    observations_path = root / "massive_reference_observations.parquet"
    sample.to_parquet(sample_path, index=False)
    observations.to_parquet(observations_path, index=False)
    (root / "massive_reference_report.json").write_text(
        json.dumps({"dataset_version": "reference-fixture-v1"}), encoding="utf-8"
    )
    (root / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "reference-fixture-v1",
                "artifacts": {
                    sample_path.name: {
                        "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()
                    },
                    observations_path.name: {
                        "sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest()
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return root
