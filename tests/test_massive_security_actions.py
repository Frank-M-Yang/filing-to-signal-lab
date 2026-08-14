"""Tests for Massive security continuity and corporate-action evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    build_massive_security_actions_target_plan,
    collect_massive_security_actions_artifacts,
    collect_massive_security_session_fallbacks,
    load_massive_security_actions_targets,
    massive_dividends_url,
    massive_splits_url,
    massive_ticker_events_url,
    parse_massive_dividends,
    parse_massive_splits,
)
from sec_data.price_research import validate_security_mapping_frame


class _Sender:
    def __init__(self, payloads: list[bytes | tuple[int, bytes]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del timeout
        self.calls.append((url, headers))
        response = self.payloads.pop(0)
        status_code, content = response if isinstance(response, tuple) else (200, response)
        return httpx.Response(
            status_code,
            content=content,
            headers={"Content-Type": "application/json"},
            request=httpx.Request("GET", url),
        )


def test_action_urls_are_window_filtered_and_secret_free() -> None:
    split_url = massive_splits_url("BRK/B", "2025-01-01", "2025-02-01")
    dividend_url = massive_dividends_url("BRK/B", "2025-01-01", "2025-02-01")
    events_url = massive_ticker_events_url("BBG000DEMO12")

    assert "execution_date.gte=2025-01-01" in split_url
    assert "ex_dividend_date.lte=2025-02-01" in dividend_url
    assert "ticker=BRK%2FB" in split_url
    assert events_url.endswith("BBG000DEMO12/events?types=ticker_change")
    assert "apiKey" not in split_url + dividend_url + events_url


def test_action_parsers_reject_wrong_currency_and_out_of_window() -> None:
    dividend_payload = _ok(
        [
            {
                "id": "div-1",
                "ticker": "DEMO",
                "ex_dividend_date": "2025-01-20",
                "cash_amount": 0.25,
                "currency": "CAD",
            }
        ]
    )
    split_payload = _ok(
        [
            {
                "id": "split-1",
                "ticker": "DEMO",
                "execution_date": "2025-03-01",
                "adjustment_type": "forward_split",
                "split_from": 1,
                "split_to": 2,
            }
        ]
    )

    dividends, dividend_rejections, _ = parse_massive_dividends(
        dividend_payload,
        action_target_id="action-1",
        expected_ticker="DEMO",
        expected_currency="usd",
        window_start="2025-01-01",
        window_end="2025-02-01",
        source_sha256=hashlib.sha256(dividend_payload).hexdigest(),
    )
    splits, split_rejections, _ = parse_massive_splits(
        split_payload,
        action_target_id="action-1",
        expected_ticker="DEMO",
        window_start="2025-01-01",
        window_end="2025-02-01",
        source_sha256=hashlib.sha256(split_payload).hexdigest(),
    )

    assert not dividends and dividend_rejections[0]["reason"] == "dividend_currency_mismatch"
    assert not splits and split_rejections[0]["reason"] == "split_outside_requested_window"


def test_five_query_collection_validates_interval_and_actions(tmp_path: Path) -> None:
    reference_root, price_root, calendar_root = _write_upstream_fixtures(tmp_path)
    plan = build_massive_security_actions_target_plan(
        reference_root,
        price_root,
        calendar_root,
        output_root=tmp_path / "targets",
    )
    targets = load_massive_security_actions_targets(plan.output_root)
    sender = _Sender(
        [
            _reference_payload(),
            _reference_payload(),
            json.dumps(
                {
                    "status": "OK",
                    "results": {"name": "Demo Corp", "events": []},
                }
            ).encode(),
            _ok(
                [
                    {
                        "id": "split-1",
                        "ticker": "DEMO",
                        "execution_date": "2025-01-15",
                        "adjustment_type": "forward_split",
                        "split_from": 1,
                        "split_to": 2,
                        "historical_adjustment_factor": 0.5,
                    }
                ]
            ),
            _ok(
                [
                    {
                        "id": "div-1",
                        "ticker": "DEMO",
                        "ex_dividend_date": "2025-01-20",
                        "cash_amount": 0.25,
                        "currency": "USD",
                        "declaration_date": "2024-12-20",
                        "record_date": "2025-01-20",
                        "pay_date": "2025-02-01",
                        "distribution_type": "recurring",
                        "frequency": 4,
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

    collection = collect_massive_security_actions_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_massive_security_actions_audit(
        plan.output_root,
        calendar_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_security_actions_audit.parquet")
    mappings = pd.read_parquet(audit.output_root / "validated_security_mappings.parquet")
    actions = pd.read_parquet(audit.output_root / "normalized_corporate_actions.parquet")

    assert collection.requests_made == 5
    assert collection.targets_completed == 1
    assert len(sender.calls) == 5
    assert all(call[1]["Authorization"] == "Bearer secret-api-key" for call in sender.calls)
    assert sample.loc[0, "historical_symbol_interval_ready"]
    assert sample.loc[0, "corporate_actions_ready"]
    assert sample.loc[0, "corporate_action_status"] == "complete_including_zero_events"
    assert len(actions) == 2
    assert set(actions["action_type"]) == {"forward_split", "cash_dividend"}
    validate_security_mapping_frame(mappings)
    assert audit.report["pilot_identity_gate"] is True
    assert audit.report["pilot_corporate_actions_gate"] is True
    assert audit.report["return_study_gate"] is False


def test_session_grid_fallback_replaces_experimental_event_404(tmp_path: Path) -> None:
    reference_root, price_root, calendar_root = _write_upstream_fixtures(tmp_path)
    plan = build_massive_security_actions_target_plan(
        reference_root,
        price_root,
        calendar_root,
        output_root=tmp_path / "targets",
    )
    targets = load_massive_security_actions_targets(plan.output_root)
    base_sender = _Sender(
        [
            _reference_payload(),
            _reference_payload(),
            (404, b'{"status":"NOT_FOUND"}'),
            _ok([]),
            _ok([]),
        ]
    )
    base_client = _client(base_sender)

    base = collect_massive_security_actions_artifacts(
        targets,
        base_client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    sessions = pd.read_parquet(calendar_root / "historical_exchange_sessions.parquet")
    expected_sessions = len(sessions)
    fallback_sender = _Sender([_reference_payload()] * expected_sessions)
    fallback_client = _client(fallback_sender)
    fallback = collect_massive_security_session_fallbacks(
        targets,
        sessions,
        fallback_client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_massive_security_actions_audit(
        plan.output_root,
        calendar_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_security_actions_audit.parquet")
    mapping = pd.read_parquet(audit.output_root / "validated_security_mappings.parquet")

    assert base.targets_failed == 1
    assert base.requests_made == 5
    assert fallback.targets_completed == 1
    assert fallback.requests_made == expected_sessions
    assert sample.loc[0, "reference_session_grid_complete"]
    assert sample.loc[0, "ticker_events_unavailable_fallback_used"]
    assert sample.loc[0, "historical_symbol_interval_status"] == (
        "validated_window_continuity_via_session_grid_fallback"
    )
    assert sample.loc[0, "historical_symbol_interval_ready"]
    assert sample.loc[0, "corporate_actions_ready"]
    assert mapping.loc[0, "mapping_method"] == "massive_pit_every_exchange_session"
    assert audit.report["targets_fully_collected"] == 0
    assert audit.report["targets_with_complete_required_evidence"] == 1
    assert audit.report["return_study_gate"] is False


def _write_upstream_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference_root = tmp_path / "reference"
    price_root = tmp_path / "price"
    calendar_root = tmp_path / "calendar"
    reference_root.mkdir()
    price_root.mkdir()
    calendar_root.mkdir()
    source_hash = hashlib.sha256(b"reference-source").hexdigest()
    pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "provider_security_id": "BBG001SABCDE",
                "share_class_figi": "BBG001SABCDE",
                "composite_figi": "BBG000DEMO12",
                "source_sha256": source_hash,
            }
        ]
    ).to_parquet(reference_root / "massive_reference_observations.parquet", index=False)
    (reference_root / "massive_reference_report.json").write_text(
        json.dumps({"dataset_version": "reference-fixture-v1"}), encoding="utf-8"
    )
    _write_manifest(reference_root)
    reference_manifest_hash = _sha256(reference_root / "output_manifest.json")

    pd.DataFrame(
        [
            {
                "price_target_id": "price-event-1",
                "price_acquisition_order": 1,
                "target_id": "event-1",
                "cik": "0000000001",
                "accepted_at": "2025-01-08T21:00:00+00:00",
                "query_date": "2025-01-08",
                "form": "10-Q",
                "accession_number": "0000000001-25-000001",
                "selected_ticker": "DEMO",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "BBG001SABCDE",
                "selected_provider_security_id_kind": "share_class_figi",
                "currency_name": "usd",
                "price_window_start": "2025-01-01",
                "price_window_end": "2025-02-28",
            }
        ]
    ).to_parquet(price_root / "massive_daily_price_targets.parquet", index=False)
    (price_root / "massive_price_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "price-fixture-v1",
                "reference_manifest_sha256": reference_manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(price_root)
    price_manifest_hash = _sha256(price_root / "output_manifest.json")

    sessions = pd.bdate_range("2025-01-01", "2025-02-28")
    pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": sessions.strftime("%Y-%m-%d"),
            "regular_open_at": [f"{value:%Y-%m-%d}T14:30:00+00:00" for value in sessions],
        }
    ).to_parquet(calendar_root / "historical_exchange_sessions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "price_target_id": "price-event-1",
                "formal_entry_session_date": "2025-01-10",
                "formal_exit_session_date": "2025-02-07",
                "formal_event_window_ready": True,
            }
        ]
    ).to_parquet(calendar_root / "event_calendar_alignment.parquet", index=False)
    (calendar_root / "exchange_calendar_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "calendar-fixture-v1",
                "price_manifest_sha256": price_manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(calendar_root)
    return reference_root, price_root, calendar_root


def _reference_payload() -> bytes:
    return json.dumps(
        {
            "status": "OK",
            "results": [
                {
                    "active": True,
                    "cik": "0000000001",
                    "ticker": "DEMO",
                    "name": "Demo Corp",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                    "currency_name": "usd",
                    "share_class_figi": "BBG001SABCDE",
                    "composite_figi": "BBG000DEMO12",
                }
            ],
        }
    ).encode()


def _client(sender: _Sender) -> MassiveReferenceClient:
    return MassiveReferenceClient(
        "secret-api-key",
        sender=sender,
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )


def _ok(results: list[dict[str, object]]) -> bytes:
    return json.dumps({"status": "OK", "results": results}).encode()


def _write_manifest(root: Path) -> None:
    artifacts = {path.name: {"sha256": _sha256(path)} for path in sorted(root.glob("*.parquet"))}
    (root / "output_manifest.json").write_text(
        json.dumps({"dataset_version": root.name, "artifacts": artifacts}), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
