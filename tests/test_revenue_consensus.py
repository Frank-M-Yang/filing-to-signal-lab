"""Tests for the frozen point-in-time analyst-consensus pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sec_data.revenue_consensus import (
    assemble_revenue_consensus_identities,
    calculate_revenue_consensus_audit,
    calculate_revenue_consensus_plan,
    parse_massive_revenue_consensus_response,
)


def test_consensus_plan_freezes_all_events_without_return_values() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("event-1", "0000000001", 2024, "Q1", "2024-05-01 20:00:00Z"),
            _forecast("event-2", "0000000002", 2024, "Q2", "2024-08-01 20:00:00Z"),
        ]
    )
    market_targets = pd.DataFrame(
        [
            _market_target("event-1", "0000000001", "2024-03-31", "2024-05-01 20:00:00Z"),
            _market_target("event-2", "0000000002", "2024-06-30", "2024-08-01 20:00:00Z"),
        ]
    )
    primary = pd.DataFrame(
        [
            {
                "validation_event_id": "event-1",
                "cik": "1",
                "ticker": "AAA",
                "exchange_mic": "XNAS",
            }
        ]
    )
    supplemental = pd.DataFrame(
        [
            {
                "validation_event_id": "event-2",
                "cik": "2",
                "assigned_ticker": "BBB",
                "assigned_exchange_mic": "XNYS",
                "event_security_identity_ready": True,
            }
        ]
    )
    identities = assemble_revenue_consensus_identities(primary, supplemental)
    result = calculate_revenue_consensus_plan(forecasts, market_targets, identities)

    assert len(result.targets) == 2
    assert result.report["return_values_loaded"] is False
    assert result.targets["selection_uses_return_values"].eq(False).all()
    assert result.targets["record_version_must_precede_cutoff"].all()
    assert result.targets.loc[0, "query_url"].startswith(
        "https://api.massive.com/benzinga/v1/earnings?"
    )
    assert set(result.targets["identity_source"]) == {
        "prior_authorized_market_ledger",
        "outcome_blind_identity_resolution",
    }


def test_parser_keeps_post_event_consensus_retrospective_only() -> None:
    target = {
        "consensus_target_id": "target-1",
        "validation_event_id": "event-1",
        "cik": "0000000001",
        "ticker": "AAA",
        "target_fiscal_year": 2024,
        "target_fiscal_quarter": "Q1",
        "target_actual_available_at": "2024-05-01T20:00:00Z",
        "consensus_cutoff_at": "2024-05-01T20:00:00Z",
    }
    before = _provider_record("2024-05-01T19:00:00Z")
    after = _provider_record("2024-05-01T21:00:00Z")
    content = json.dumps({"status": "OK", "results": [before, after]}).encode()
    observations, rejections = parse_massive_revenue_consensus_response(
        content,
        target=target,
        source_sha256=hashlib.sha256(content).hexdigest(),
        maximum_event_date_distance_days=3,
    )

    assert not rejections
    assert observations[0]["point_in_time_consensus_ready"] is True
    assert observations[1]["retrospective_consensus_candidate"] is True
    assert observations[1]["point_in_time_consensus_ready"] is False
    assert observations[1]["record_version_precedes_cutoff"] is False


def test_consensus_audit_persists_entitlement_failure(tmp_path: Path) -> None:
    target_id = "target-1"
    failure_dir = tmp_path / "failures"
    failure_dir.mkdir()
    (failure_dir / f"{target_id}.json").write_text(
        json.dumps(
            {
                "consensus_target_id": target_id,
                "reason": "entitlement_denied",
                "detail": "HTTP 403",
                "failed_at": "2026-08-29T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    targets = pd.DataFrame(
        [
            {
                "consensus_target_id": target_id,
                "validation_event_id": "event-1",
                "cik": "0000000001",
                "ticker": "AAA",
                "target_fiscal_year": 2024,
                "target_fiscal_quarter": "Q1",
                "target_actual_available_at": "2024-05-01T20:00:00Z",
                "consensus_cutoff_at": "2024-05-01T20:00:00Z",
            }
        ]
    )
    result = calculate_revenue_consensus_audit(targets, tmp_path)

    assert result.report["provider_query_failures"] == 1
    assert result.report["provider_entitlement_denied"] == 1
    assert result.event_audit.iloc[0]["consensus_audit_status"] == ("provider_entitlement_denied")


def _forecast(
    event_id: str, cik: str, fiscal_year: int, fiscal_quarter: str, accepted_at: str
) -> dict[str, object]:
    return {
        "validation_event_id": event_id,
        "cik": cik,
        "target_fiscal_year": fiscal_year,
        "target_fiscal_quarter": fiscal_quarter,
        "target_actual_available_at": accepted_at,
        "target_revenue": 110.0,
        "management_guidance_forecast": 100.0,
        "selection_uses_return_values": False,
    }


def _market_target(
    event_id: str, cik: str, report_date: str, accepted_at: str
) -> dict[str, object]:
    return {
        "validation_event_id": event_id,
        "cik": cik,
        "report_date": report_date,
        "accepted_at": accepted_at,
    }


def _provider_record(last_updated: str) -> dict[str, object]:
    return {
        "ticker": "AAA",
        "fiscal_year": 2024,
        "fiscal_period": "Q1",
        "date": "2024-05-01",
        "time": "16:00:00",
        "estimated_revenue": 105.0,
        "currency": "USD",
        "revenue_method": "gaap",
        "last_updated": last_updated,
        "benzinga_id": f"record-{last_updated}",
    }
