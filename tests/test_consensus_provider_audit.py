"""Tests for the evidence-backed consensus-provider capability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sec_data.consensus_provider_audit import calculate_consensus_provider_audit


def test_provider_audit_rejects_current_eps_only_probe(tmp_path: Path) -> None:
    content = json.dumps(
        [
            {
                "id": "1",
                "ticker": "AAA",
                "field": "eps_estimate",
                "period": "2024-03-31T00:00:00Z",
                "value": "1.0",
                "as_of_date": "2026-08-27T22:00:00Z",
                "source": "example",
            }
        ]
    ).encode()
    sha256 = hashlib.sha256(content).hexdigest()
    target = tmp_path / "AAA" / sha256
    target.mkdir(parents=True)
    raw_path = target / "response.bin"
    raw_path.write_bytes(content)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "provider": "eulerpool",
                "endpoint": "https://example.invalid/AAA",
                "ticker": "AAA",
                "limit": 100,
                "status_code": 200,
                "fetched_at": "2026-08-28T00:00:00Z",
                "sha256": sha256,
                "bytes": len(content),
                "path": str(raw_path.resolve()),
                "response_headers": {},
            }
        ),
        encoding="utf-8",
    )
    targets = pd.DataFrame(
        [
            {
                "validation_event_id": "event-1",
                "cik": "0000000001",
                "ticker": "AAA",
                "consensus_cutoff_at": "2024-05-01T20:00:00Z",
                "selection_uses_return_values": False,
            }
        ]
    )
    massive = pd.DataFrame(
        [
            {
                "validation_event_id": "event-1",
                "provider_query_complete": False,
                "provider_failure_reason": "entitlement_denied",
                "point_in_time_consensus_ready": False,
                "selection_uses_return_values": False,
            }
        ]
    )
    result = calculate_consensus_provider_audit(targets, massive, tmp_path)

    assert result.report["provider_gate"] is False
    assert result.report["eulerpool_revenue_field_present"] is False
    assert result.report["eulerpool_historical_asof_precedes_any_target"] is False
    assert set(result.provider_decisions["decision"]) == {
        "rejected_current_entitlement",
        "rejected_for_revenue_consensus_v1",
    }
