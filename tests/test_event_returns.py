"""Tests for explicit event cash flows and the SPY benchmark audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from sec_data.event_returns import calculate_cash_retained_holding_return
from sec_data.massive_benchmark import (
    build_massive_benchmark_audit,
    build_massive_benchmark_target_plan,
    collect_massive_benchmark_artifacts,
)
from sec_data.massive_reference import MassiveReferenceClient


class _Sender:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del headers, timeout
        return httpx.Response(
            200,
            content=self.payloads.pop(0),
            headers={"Content-Type": "application/json"},
            request=httpx.Request("GET", url),
        )


def test_cash_retained_return_applies_split_before_same_time_dividend() -> None:
    actions = pd.DataFrame(
        [
            {
                "action_type": "cash_dividend",
                "effective_at": "2025-01-15T14:30:00+00:00",
                "cash_amount": 0.5,
                "currency": "USD",
                "split_from": None,
                "split_to": None,
                "provider_event_id": "div-1",
                "source_snapshot_sha256": "a" * 64,
            },
            {
                "action_type": "forward_split",
                "effective_at": "2025-01-15T14:30:00+00:00",
                "cash_amount": None,
                "currency": None,
                "split_from": 1.0,
                "split_to": 2.0,
                "provider_event_id": "split-1",
                "source_snapshot_sha256": "b" * 64,
            },
        ]
    )

    result, details = calculate_cash_retained_holding_return(10.0, 6.0, actions, currency="USD")

    assert [item["action_type"] for item in details] == ["forward_split", "cash_dividend"]
    assert result.shares_at_exit == 2.0
    assert result.cash_dividend_value == 1.0
    assert result.split_adjusted_price_return == pytest.approx(0.2)
    assert result.cash_dividend_return == pytest.approx(0.1)
    assert result.total_return == pytest.approx(0.3)


def test_benchmark_collection_requires_prices_and_explicit_actions(tmp_path: Path) -> None:
    calendar_root = _write_calendar(tmp_path)
    plan = build_massive_benchmark_target_plan(calendar_root, output_root=tmp_path / "targets")
    sender = _Sender(
        [
            _bars_payload(),
            _ok([]),
            _ok(
                [
                    {
                        "id": "spy-div",
                        "ticker": "SPY",
                        "ex_dividend_date": "2025-01-15",
                        "cash_amount": 1.0,
                        "currency": "USD",
                    }
                ]
            ),
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

    collection = collect_massive_benchmark_artifacts(
        plan.output_root, client, raw_root=tmp_path / "raw"
    )
    audit = build_massive_benchmark_audit(
        plan.output_root,
        calendar_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    report = audit.report
    actions = pd.read_parquet(audit.output_root / "normalized_benchmark_actions.parquet")

    assert collection.requests_made == 3
    assert report["benchmark_data_ready"] is True
    assert report["required_event_dates_ready"] == 2
    assert report["dividends_observed"] == 1
    assert actions.iloc[0]["effective_at"] == "2025-01-15T14:30:00+00:00"


def _write_calendar(tmp_path: Path) -> Path:
    root = tmp_path / "calendar"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "price_target_id": "price-1",
                "target_id": "event-1",
                "formal_entry_session_date": "2025-01-10",
                "formal_exit_session_date": "2025-02-07",
                "formal_event_window_ready": True,
            }
        ]
    ).to_parquet(root / "event_calendar_alignment.parquet", index=False)
    pd.DataFrame(
        [
            {
                "exchange_mic": "XNYS",
                "session_date": date,
                "regular_open_at": f"{date}T14:30:00+00:00",
            }
            for date in ["2025-01-10", "2025-01-15", "2025-02-07"]
        ]
    ).to_parquet(root / "historical_exchange_sessions.parquet", index=False)
    (root / "exchange_calendar_report.json").write_text(
        json.dumps({"dataset_version": "calendar-fixture-v1"}), encoding="utf-8"
    )
    _write_manifest(root)
    return root


def _bars_payload() -> bytes:
    def millis(value: str) -> int:
        return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)

    return json.dumps(
        {
            "status": "OK",
            "adjusted": False,
            "ticker": "SPY",
            "results": [
                {"t": millis("2025-01-10"), "o": 100, "h": 102, "l": 99, "c": 101, "v": 1},
                {"t": millis("2025-02-07"), "o": 104, "h": 106, "l": 103, "c": 105, "v": 1},
            ],
        }
    ).encode()


def _ok(results: list[dict[str, object]]) -> bytes:
    return json.dumps({"status": "OK", "results": results}).encode()


def _write_manifest(root: Path) -> None:
    artifacts = {path.name: {"sha256": _sha256(path)} for path in sorted(root.glob("*.parquet"))}
    (root / "output_manifest.json").write_text(
        json.dumps({"dataset_version": root.name, "artifacts": artifacts}), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
