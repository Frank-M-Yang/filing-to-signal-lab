"""Tests for point-in-time Massive security-reference ingestion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from sec_data.massive_reference import (
    MassiveReferenceClient,
    MassiveRequestError,
    build_massive_reference_audit,
    build_massive_reference_target_plan,
    collect_massive_reference_artifacts,
    collect_massive_symbol_reference_artifacts,
    load_massive_reference_targets,
    parse_massive_ticker_response,
)


class _Sender:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del timeout
        self.calls.append((url, headers))
        return self.responses.pop(0)


def test_client_uses_header_auth_and_secret_free_url() -> None:
    url = "https://api.massive.com/v3/reference/tickers?cik=0000000001&date=2025-01-02&active=true"
    sender = _Sender([_response(url, _ticker_payload())])
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=sender,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    result = client.get(url)

    called_url, headers = sender.calls[0]
    assert "secret-api-key" not in called_url
    assert headers["Authorization"] == "Bearer secret-api-key"
    assert "secret-api-key" not in result.requested_url
    assert result.content == _ticker_payload()


def test_client_counts_rejected_authentication_request() -> None:
    url = "https://api.massive.com/v3/reference/tickers?cik=0000000001&date=2025-01-02&active=true"
    sender = _Sender(
        [
            httpx.Response(
                401,
                content=b'{"status":"NOT_AUTHORIZED"}',
                request=httpx.Request("GET", url),
            )
        ]
    )
    client = MassiveReferenceClient(
        "rejected-api-key",
        sender=sender,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(MassiveRequestError) as caught:
        client.get(url)

    assert caught.value.reason == "authentication_failed"
    assert caught.value.requests_made == 1


def test_symbol_fallback_query_enforces_expected_cik_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    client_for_url = MassiveReferenceClient("secret-api-key", sender=_Sender([]))
    query_url = client_for_url.ticker_symbol_reference_url("DEMO", "2025-01-02", active=True)
    sender = _Sender([_response(query_url, _ticker_payload())])
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=sender,
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )
    targets = pd.DataFrame(
        [
            {
                "target_id": "symbol-event-1",
                "cik": "0000000001",
                "query_ticker": "DEMO",
                "query_date": "2025-01-02",
                "massive_acquisition_order": 1,
            }
        ]
    )

    result = collect_massive_symbol_reference_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )

    assert "ticker=DEMO" in query_url
    assert "cik=" not in query_url
    assert result.targets_completed == 1
    assert result.requests_made == 1


def test_parser_rejects_wrong_cik_and_preserves_stable_ids() -> None:
    payload = json.dumps(
        {
            "status": "OK",
            "results": [
                {
                    "active": True,
                    "cik": "0000000001",
                    "ticker": "DEMO",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                    "share_class_figi": "BBG001SABCDE",
                    "composite_figi": "BBG000BWQYZ5",
                },
                {"active": True, "cik": "2", "ticker": "WRONG"},
            ],
        }
    ).encode()

    rows, rejections, next_url = parse_massive_ticker_response(
        payload,
        expected_cik="1",
        target_id="event-1",
        query_active=True,
        as_of_date="2025-01-02",
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert len(rows) == 1
    assert rows[0]["provider_security_id"] == "BBG001SABCDE"
    assert rows[0]["provider_security_id_kind"] == "share_class_figi"
    assert rejections[0]["reason"] == "response_cik_mismatch"
    assert next_url is None


def test_target_plan_collection_and_audit_keep_return_gate_closed(tmp_path: Path) -> None:
    scope = _write_scope_fixture(tmp_path / "scope")
    plan = build_massive_reference_target_plan(scope, output_root=tmp_path / "targets")
    targets = load_massive_reference_targets(plan.output_root)
    query_url = MassiveReferenceClient("secret-api-key", sender=_Sender([])).ticker_reference_url(
        "1", "2025-01-02", active=True
    )
    sender = _Sender([_response(query_url, _ticker_payload())])
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=sender,
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    collection = collect_massive_reference_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_massive_reference_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_massive_reference_audit.parquet")
    snapshots = pd.read_parquet(audit.output_root / "massive_reference_snapshots.parquet")

    assert collection.requests_made == 1
    assert collection.targets_completed == 1
    assert sample.loc[0, "event_date_reference_candidate"]
    assert sample.loc[0, "selected_provider_security_id"] == "BBG001SABCDE"
    assert not sample.loc[0, "historical_symbol_interval_ready"]
    assert audit.report["event_date_reference_candidates"] == 1
    assert audit.report["final_price_identity_ready"] == 0
    assert audit.report["return_study_gate"] is False
    assert not snapshots.loc[0, "requested_url"].endswith("secret-api-key")


def test_audit_rejects_unique_common_stock_without_stable_id(tmp_path: Path) -> None:
    scope = _write_scope_fixture(tmp_path / "scope")
    plan = build_massive_reference_target_plan(scope, output_root=tmp_path / "targets")
    targets = load_massive_reference_targets(plan.output_root)
    query_url = MassiveReferenceClient("secret-api-key", sender=_Sender([])).ticker_reference_url(
        "1", "2025-01-02", active=True
    )
    payload = json.dumps(
        {
            "status": "OK",
            "count": 1,
            "results": [
                {
                    "active": True,
                    "cik": "0000000001",
                    "ticker": "DEMO",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                }
            ],
        }
    ).encode()
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=_Sender([_response(query_url, payload)]),
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    collect_massive_reference_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_massive_reference_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_massive_reference_audit.parquet")

    assert sample.loc[0, "event_date_reference_status"] == "missing_provider_security_id"
    assert not sample.loc[0, "event_date_reference_candidate"]
    assert audit.report["event_date_reference_candidates"] == 0


def test_completed_empty_query_is_not_reported_as_not_collected(tmp_path: Path) -> None:
    scope = _write_scope_fixture(tmp_path / "scope")
    plan = build_massive_reference_target_plan(scope, output_root=tmp_path / "targets")
    targets = load_massive_reference_targets(plan.output_root)
    active_url = MassiveReferenceClient("secret-api-key", sender=_Sender([])).ticker_reference_url(
        "1", "2025-01-02", active=True
    )
    inactive_url = MassiveReferenceClient(
        "secret-api-key", sender=_Sender([])
    ).ticker_reference_url("1", "2025-01-02", active=False)
    empty = json.dumps({"status": "OK", "count": 0, "results": []}).encode()
    client = MassiveReferenceClient(
        "secret-api-key",
        sender=_Sender([_response(active_url, empty), _response(inactive_url, empty)]),
        request_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    collect_massive_reference_artifacts(targets, client, raw_root=tmp_path / "raw", limit=1)
    audit = build_massive_reference_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    sample = pd.read_parquet(audit.output_root / "sample_massive_reference_audit.parquet")

    assert sample.loc[0, "provider_query_complete"]
    assert sample.loc[0, "event_date_reference_status"] == "provider_no_cik_match"


def _ticker_payload() -> bytes:
    return json.dumps(
        {
            "status": "OK",
            "count": 1,
            "results": [
                {
                    "active": True,
                    "cik": "0000000001",
                    "ticker": "DEMO",
                    "name": "Demo Corporation",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                    "currency_name": "usd",
                    "share_class_figi": "BBG001SABCDE",
                    "composite_figi": "BBG000BWQYZ5",
                    "last_updated_utc": "2025-01-02T00:00:00Z",
                }
            ],
        }
    ).encode()


def _response(url: str, content: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"Content-Type": "application/json"},
        request=httpx.Request("GET", url),
    )


def _write_scope_fixture(root: Path) -> Path:
    root.mkdir()
    sample = pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "cik": "0000000001",
                "calendar_year": 2025,
                "accepted_at": "2025-01-02T20:00:00Z",
                "form": "10-Q",
                "accession_number": "0000000001-25-000001",
                "sample_size_bucket": "q4",
                "sampling_stratum": "2025|q4",
                "sampling_hash": "a" * 64,
                "stage1_inclusion_probability": 0.5,
                "stage1_sampling_weight": 2.0,
                "equity_scope_status": "provisional_common_equity_candidate",
                "equity_security_titles": "Common Stock",
                "equity_trading_symbols": "DEMO",
                "provisional_equity_eligible": True,
            },
            {
                "target_id": "event-2",
                "cik": "0000000002",
                "calendar_year": 2025,
                "accepted_at": "2025-02-02T20:00:00Z",
                "form": "10-K",
                "accession_number": "0000000002-25-000001",
                "sample_size_bucket": "q4",
                "sampling_stratum": "2025|q4",
                "sampling_hash": "b" * 64,
                "stage1_inclusion_probability": 0.5,
                "stage1_sampling_weight": 2.0,
                "equity_scope_status": "security_identity_unresolved",
                "equity_security_titles": "",
                "equity_trading_symbols": "",
                "provisional_equity_eligible": False,
            },
        ]
    )
    sample_path = root / "sample_scope_audit.parquet"
    sample.to_parquet(sample_path, index=False)
    (root / "scope_audit_report.json").write_text(
        json.dumps({"dataset_version": "fixture-scope"}), encoding="utf-8"
    )
    (root / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-scope",
                "artifacts": {
                    "sample_scope_audit.parquet": {
                        "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root
