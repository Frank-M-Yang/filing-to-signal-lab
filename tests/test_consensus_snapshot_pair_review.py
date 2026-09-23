"""Forensic comparisons preserve semantic keys, integrity, and research gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.review_consensus_snapshot_pair import review_consensus_snapshot_pair


def _row(**changes: Any) -> dict[str, Any]:
    return {
        "horizon": "fiscal quarter",
        "date": "2026-10-31",
        "revenue_estimate_average": 100,
        "revenue_estimate_low": 90,
        "revenue_estimate_high": 110,
        "revenue_estimate_analyst_count": 4,
        **changes,
    }


def _snapshot(root: Path, day: int, rows: list[dict[str, Any]], **changes: Any) -> Path:
    folder = root / str(day)
    folder.mkdir()
    content = json.dumps({"symbol": "AAA", "estimates": rows}).encode()
    (folder / "response.bin").write_bytes(content)
    manifest = {
        "ticker": "AAA",
        "cik": "0000000001",
        "coverage_probe_target_id": "fixed-target",
        "fetched_at": f"2026-09-{day:02}T12:00:00Z",
        "status": "ok",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        **changes,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_inserted_quarter_and_annual_same_date_do_not_mismatch(tmp_path: Path) -> None:
    previous = _snapshot(tmp_path, 14, [_row()])
    current = _snapshot(
        tmp_path,
        19,
        [_row(date="2027-01-31"), _row(horizon="fiscal year"), _row()],
    )
    result = review_consensus_snapshot_pair(previous, current, period_end="2026-10-31")
    assert result["previous_occurrence"] == 0
    assert result["current_occurrence"] == 2
    assert result["aggregate_mean_revision_fraction"] == 0
    assert result["elapsed_hours"] == 120
    assert len(result["new_keys"]) == 2
    assert result["provider_updated_at"] is None
    for field in (
        "formal_model_admission",
        "order_authorized",
        "new_market_information_verified",
        "bounds_assumptions_verified",
        "provider_currency_unit_scope_verified",
        "provider_fiscal_calendar_mapping_verified",
        "target_actual_values_loaded",
        "return_values_loaded",
    ):
        assert result[field] is False


def test_simple_one_member_removal_is_rejected_without_causal_attribution(tmp_path: Path) -> None:
    previous = _snapshot(tmp_path, 14, [_row()])
    current = _snapshot(
        tmp_path,
        19,
        [
            _row(
                revenue_estimate_average=120,
                revenue_estimate_low=119,
                revenue_estimate_high=121,
                revenue_estimate_analyst_count=3,
            )
        ],
    )
    result = review_consensus_snapshot_pair(previous, current, period_end="2026-10-31")
    bounds = result["conditional_membership_bounds"]
    assert bounds["required_added_or_removed_mean"] == 40
    assert not bounds["composition_not_ruled_out_by_necessary_bounds"]
    assert not result["new_market_information_verified"]


@pytest.mark.parametrize(
    "mutation",
    [
        "corrupt",
        "wrong_issuer",
        "duplicate",
        "annual_only",
        "naive_time",
        "reverse_time",
        "fractional_count",
        "nonfinite",
        "out_of_range",
        "error",
        "redacted",
        "boolean",
    ],
)
def test_invalid_evidence_fails_closed(tmp_path: Path, mutation: str) -> None:
    previous = _snapshot(tmp_path, 14, [_row()])
    rows = [_row()]
    overrides: dict[str, Any] = {}
    if mutation == "duplicate":
        rows.append(_row())
    if mutation == "annual_only":
        rows = [_row(horizon="fiscal year")]
    if mutation == "wrong_issuer":
        overrides["cik"] = "0000000002"
    if mutation == "naive_time":
        overrides["fetched_at"] = "2026-09-19T12:00:00"
    if mutation == "reverse_time":
        overrides["fetched_at"] = "2026-09-13T12:00:00Z"
    if mutation == "fractional_count":
        rows = [_row(revenue_estimate_analyst_count=2.5)]
    if mutation == "nonfinite":
        rows = [_row(revenue_estimate_average="NaN")]
    if mutation == "out_of_range":
        rows = [_row(revenue_estimate_average=200)]
    if mutation == "error":
        overrides["status"] = "rate_limited"
    if mutation == "redacted":
        overrides["credentials_redacted"] = True
    if mutation == "boolean":
        rows = [_row(revenue_estimate_analyst_count=True)]
    current = _snapshot(tmp_path, 19, rows, **overrides)
    if mutation == "corrupt":
        (current.parent / "response.bin").write_bytes(b"changed")
    with pytest.raises(ValueError):
        review_consensus_snapshot_pair(previous, current, period_end="2026-10-31")


def test_target_already_ended_is_not_future_evidence(tmp_path: Path) -> None:
    previous = _snapshot(tmp_path, 14, [_row(date="2026-09-19")])
    current = _snapshot(tmp_path, 19, [_row(date="2026-09-19")])
    with pytest.raises(ValueError, match="future"):
        review_consensus_snapshot_pair(previous, current, period_end="2026-09-19")
