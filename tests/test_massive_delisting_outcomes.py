"""Tests for listing-status and event-window delisting outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from sec_data.massive_delisting_outcomes import (
    build_massive_delisting_audit,
    build_massive_delisting_target_plan,
    collect_massive_delisting_artifacts,
    load_massive_delisting_targets,
)
from sec_data.massive_reference import MassiveReferenceClient


class _Sender:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del headers, timeout
        self.calls.append(url)
        return httpx.Response(
            200,
            content=self.payloads.pop(0),
            headers={"Content-Type": "application/json"},
            request=httpx.Request("GET", url),
        )


def test_active_and_later_delisted_events_are_window_ready(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, inactive_last_trade="2025-06-30T00:00:00Z")
    sample = result["sample"]
    outcomes = result["outcomes"]

    assert result["requests"] == 3
    assert sample["delisting_outcome_ready"].all()
    assert sample["active_at_outcome_cutoff"].sum() == 1
    assert sample["later_delisted_after_formal_exit"].sum() == 1
    assert not sample["delisted_on_or_before_formal_exit"].any()
    assert outcomes["delisting_return"].isna().all()
    assert outcomes["terminal_value"].isna().all()
    later = outcomes[outcomes["later_delisted_after_formal_exit"]].iloc[0]
    assert later["delisting_date"] == "2025-06-30"
    assert later["missing_return_reason"] == "not_applicable_delisting_after_formal_exit"
    assert result["report"]["pilot_delisting_outcome_gate"] is True
    assert result["report"]["full_return_study_gate"] is False
    assert result["report"]["zero_delisting_return_imputed"] is False


def test_delisting_before_formal_exit_keeps_terminal_gate_closed(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, inactive_last_trade="2025-02-01T00:00:00Z")
    sample = result["sample"]
    outcomes = result["outcomes"]
    inactive = sample[sample["selected_ticker"].eq("OLD")].iloc[0]
    inactive_outcome = outcomes[outcomes["ticker_at_event"].eq("OLD")].iloc[0]

    assert not inactive["delisting_outcome_ready"]
    assert inactive["delisted_on_or_before_formal_exit"]
    assert inactive["listing_outcome_status"] == (
        "delisted_on_or_before_formal_exit_terminal_value_required"
    )
    assert inactive_outcome["missing_return_reason"] == (
        "provider_does_not_supply_delisting_return_or_terminal_value"
    )
    assert result["report"]["pilot_delisting_outcome_gate"] is False


def test_ticker_aliases_resolve_at_the_share_class_level(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        inactive_last_trade="2025-06-30T00:00:00Z",
        include_ticker_aliases=True,
    )
    sample = result["sample"]

    active = sample[sample["selected_ticker"].eq("LIVE")].iloc[0]
    inactive = sample[sample["selected_ticker"].eq("OLD")].iloc[0]
    assert active["active_exact_matches"] == 1
    assert active["inactive_exact_matches"] == 0
    assert active["listing_outcome_status"] == "active_at_outcome_cutoff"
    assert active["delisting_outcome_ready"]
    assert inactive["inactive_exact_matches"] == 2
    assert inactive["outcome_ticker"] == "OLD"
    assert inactive["listing_outcome_status"] == "later_delisted_after_formal_exit"
    assert inactive["delisting_outcome_ready"]


def _run_fixture(
    tmp_path: Path,
    *,
    inactive_last_trade: str,
    include_ticker_aliases: bool = False,
) -> dict[str, object]:
    actions_root, calendar_root = _write_upstream_fixtures(tmp_path)
    plan = build_massive_delisting_target_plan(
        actions_root,
        calendar_root,
        output_root=tmp_path / "targets",
    )
    targets = load_massive_delisting_targets(plan.output_root)
    empty_audit = build_massive_delisting_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    empty_outcomes = pd.read_parquet(
        empty_audit.output_root / "event_window_delisting_outcomes.parquet"
    )
    assert len(empty_outcomes) == len(targets)
    assert empty_outcomes["missing_return_reason"].eq("outcome_query_not_collected").all()
    sender = _Sender(
        [
            _status_payload("LIVE", "BBG001LIVE12", active=True),
            _empty_payload(),
            (
                _status_alias_payload(
                    [
                        ("OLD", "BBG001OLD123", inactive_last_trade),
                        ("OLDW", "BBG001OLD123", None),
                    ]
                )
                if include_ticker_aliases
                else _status_payload(
                    "OLD",
                    "BBG001OLD123",
                    active=False,
                    delisted_utc=inactive_last_trade,
                )
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
    collection = collect_massive_delisting_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=2,
    )
    audit = build_massive_delisting_audit(
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    return {
        "requests": collection.requests_made,
        "sample": pd.read_parquet(audit.output_root / "sample_delisting_outcome_audit.parquet"),
        "outcomes": pd.read_parquet(audit.output_root / "event_window_delisting_outcomes.parquet"),
        "report": audit.report,
    }


def _write_upstream_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    actions_root = tmp_path / "actions"
    calendar_root = tmp_path / "calendar"
    actions_root.mkdir()
    calendar_root.mkdir()
    rows: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    alignments: list[dict[str, object]] = []
    for order, (ticker, security_id) in enumerate(
        [("LIVE", "BBG001LIVE12"), ("OLD", "BBG001OLD123")], start=1
    ):
        action_id = f"action-{order}"
        price_id = f"price-{order}"
        target_id = f"event-{order}"
        rows.append(
            {
                "action_target_id": action_id,
                "action_acquisition_order": order,
                "price_target_id": price_id,
                "target_id": target_id,
                "cik": f"{order:010d}",
                "accepted_at": "2025-01-08T21:00:00+00:00",
                "query_date": "2025-01-08",
                "selected_ticker": ticker,
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": security_id,
                "selected_provider_security_id_kind": "share_class_figi",
                "selected_share_class_figi": security_id,
                "selected_composite_figi": f"BBG000COMP{order:02d}",
                "price_window_end": "2025-02-28",
                "formal_exit_session_date": "2025-02-07",
            }
        )
        mappings.append(
            {
                "action_target_id": action_id,
                "security_id": security_id,
                "security_id_kind": "share_class_figi",
                "source_snapshot_sha256": hashlib.sha256(action_id.encode()).hexdigest(),
            }
        )
        alignments.append(
            {
                "price_target_id": price_id,
                "formal_exit_session_date": "2025-02-07",
                "formal_exit_close_at": "2025-02-07T21:00:00+00:00",
                "formal_exit_close_unadjusted": 10.0,
                "formal_event_window_ready": True,
            }
        )
    targets = pd.DataFrame(rows)
    targets.to_parquet(actions_root / "massive_security_actions_targets.parquet", index=False)
    pd.DataFrame(
        {
            "action_target_id": ["action-1", "action-2"],
            "historical_symbol_interval_ready": [True, True],
            "corporate_actions_ready": [True, True],
        }
    ).to_parquet(actions_root / "sample_security_actions_audit.parquet", index=False)
    pd.DataFrame(mappings).to_parquet(
        actions_root / "validated_security_mappings.parquet", index=False
    )
    pd.DataFrame(alignments).to_parquet(
        calendar_root / "event_calendar_alignment.parquet", index=False
    )
    _write_manifest(calendar_root)
    calendar_hash = _sha256(calendar_root / "output_manifest.json")
    (actions_root / "massive_security_actions_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "actions-fixture-v1",
                "calendar_version": "calendar-fixture-v1",
                "calendar_manifest_sha256": calendar_hash,
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(actions_root)
    return actions_root, calendar_root


def _status_payload(
    ticker: str,
    security_id: str,
    *,
    active: bool,
    delisted_utc: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "status": "OK",
            "results": [
                {
                    "active": active,
                    "cik": "0000000001" if ticker == "LIVE" else "0000000002",
                    "ticker": ticker,
                    "name": f"{ticker} Corp",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                    "currency_name": "usd",
                    "share_class_figi": security_id,
                    "composite_figi": "BBG000COMP02",
                    "delisted_utc": delisted_utc,
                }
            ],
        }
    ).encode()


def _empty_payload() -> bytes:
    return json.dumps({"status": "OK", "results": []}).encode()


def _status_alias_payload(rows: list[tuple[str, str, str | None]]) -> bytes:
    return json.dumps(
        {
            "status": "OK",
            "results": [
                {
                    "active": False,
                    "cik": "0000000002",
                    "ticker": ticker,
                    "name": f"{ticker} Corp",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "primary_exchange": "XNAS",
                    "currency_name": "usd",
                    "share_class_figi": security_id,
                    "composite_figi": "BBG000COMP02",
                    "delisted_utc": delisted_utc,
                }
                for ticker, security_id, delisted_utc in rows
            ],
        }
    ).encode()


def _write_manifest(root: Path) -> None:
    artifacts = {path.name: {"sha256": _sha256(path)} for path in sorted(root.glob("*.parquet"))}
    (root / "output_manifest.json").write_text(
        json.dumps({"dataset_version": root.name, "artifacts": artifacts}), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
