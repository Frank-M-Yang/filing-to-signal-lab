"""Capture dates, source versions and analyst aggregates are different evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from sec_data.alpha_vantage_consensus import AlphaVantageConsensusError
from sec_data.consensus_probe_continuity import compare_coverage_probe_snapshots


def _attempt(time: str, digest: str = "same-bytes", status: str = "ok") -> dict[str, Any]:
    return {
        "coverage_probe_target_id": "fixed-issuer",
        "ticker": "AAA",
        "captured_at": time,
        "capture_date": pd.Timestamp(time).tz_convert("UTC").date().isoformat(),
        "source_sha256": digest if status != "transport_failure" else None,
        "status": status,
    }


def _quarter(attempt: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {
        **attempt,
        "target_period_end": "2026-12-31",
        "revenue_estimate_average": 100.0,
        "revenue_estimate_low": 90.0,
        "revenue_estimate_high": 110.0,
        "revenue_estimate_analyst_count": 4.0,
        "source_occurrence": 0,
        "aggregate_fields_valid": True,
        "candidate_rejection_reason": "none",
        **changes,
    }


def _compare(
    attempts: list[dict[str, Any]],
    quarters: list[dict[str, Any]],
    rejections: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    return compare_coverage_probe_snapshots(
        pd.DataFrame(attempts), pd.DataFrame(quarters), pd.DataFrame(rejections or [])
    )


def test_crossing_midnight_is_not_a_day_or_independent_information() -> None:
    attempts = [_attempt("2026-09-14T23:50:00Z"), _attempt("2026-09-15T00:10:00Z")]
    row = _compare(attempts, [_quarter(a) for a in attempts]).iloc[0]
    assert row["distinct_utc_dates"]
    assert row["elapsed_hours"] == pytest.approx(1 / 3)
    assert not row["spans_24_hours"]
    assert not row["response_hash_changed"]
    assert row["comparison_status"] == "aggregate_fields_unchanged"
    assert row["aggregate_mean_revision_fraction"] == 0
    assert not row["independent_information_verified"]
    assert not row["formal_model_admission"]


def test_changed_bytes_or_row_position_are_not_revenue_revisions() -> None:
    attempts = [_attempt("2026-09-14T12:00:00Z"), _attempt("2026-09-15T12:00:00Z", "EPS-change")]
    rows = [_quarter(attempts[0]), _quarter(attempts[1], source_occurrence=7)]
    row = _compare(attempts, rows).iloc[0]
    assert row["spans_24_hours"] and row["response_hash_changed"]
    assert row["source_occurrence_changed"]
    assert row["comparison_status"] == "aggregate_fields_unchanged"
    assert not row["analyst_membership_identity_verified"]


def test_mean_range_and_count_changes_are_separately_visible() -> None:
    attempts = [_attempt("2026-09-14T12:00:00Z"), _attempt("2026-09-16T12:00:00Z", "change")]
    rows = [
        _quarter(attempts[0]),
        _quarter(
            attempts[1],
            revenue_estimate_average=105,
            revenue_estimate_high=120,
            revenue_estimate_analyst_count=5,
        ),
    ]
    row = _compare(attempts, rows).iloc[0]
    assert row["comparison_status"] == "aggregate_fields_changed"
    assert row["mean_changed"] and row["range_changed"] and row["analyst_count_changed"]
    assert row["aggregate_mean_revision_fraction"] == pytest.approx(0.05)
    assert not row["independent_information_verified"]


@pytest.mark.parametrize("middle_status", ["transport_failure", "rate_limited", "ok"])
def test_missing_or_failed_middle_attempt_is_never_bridged(middle_status: str) -> None:
    attempts = [
        _attempt("2026-09-14T12:00:00Z"),
        _attempt("2026-09-15T12:00:00Z", status=middle_status),
        _attempt("2026-09-16T12:00:00Z"),
    ]
    result = _compare(attempts, [_quarter(attempts[0]), _quarter(attempts[2])])
    assert len(result) == 2
    assert result["comparison_status"].eq("not_comparable").all()
    assert result["aggregate_mean_revision_fraction"].isna().all()
    assert result["elapsed_hours"].eq(24).all()


def test_zero_provider_coverage_remains_rejected_and_visible() -> None:
    attempts = [_attempt("2026-09-14T12:00:00Z"), _attempt("2026-09-15T12:00:00Z")]
    rejections = [
        {
            **_quarter(a, revenue_estimate_average=0, revenue_estimate_analyst_count=0),
            "field_rejection_reason": "zero_revenue_and_zero_analyst_count_no_valid_coverage",
        }
        for a in attempts
    ]
    row = _compare(attempts, [], rejections).iloc[0]
    assert row["comparison_status"] == "not_comparable"
    assert row["current_revenue_estimate_average"] == 0
    assert row["current_revenue_estimate_analyst_count"] == 0
    assert pd.isna(row["aggregate_mean_revision_fraction"])


def test_rejected_parser_occurrence_is_reconciled_but_duplicate_quarters_are_blocked() -> None:
    attempts = [_attempt("2026-09-14T12:00:00Z"), _attempt("2026-09-15T12:00:00Z")]
    invalid = _quarter(attempts[0], aggregate_fields_valid=False, revenue_estimate_low=None)
    rejection = {**invalid, "field_rejection_reason": "valid_revenue_range_missing"}
    result = _compare(attempts, [invalid, _quarter(attempts[1])], [rejection])
    assert result.iloc[0]["previous_quarter_status"] == "valid_revenue_range_missing"
    duplicated = [_quarter(attempts[0]), _quarter(attempts[0], source_occurrence=1)]
    result = _compare(attempts, [*duplicated, _quarter(attempts[1])])
    assert result.iloc[0]["previous_quarter_status"] == "duplicate_quarter_rows"
    assert not result.iloc[0]["aggregate_fields_comparable"]


def test_expired_quarter_not_mislabelled_as_newly_missing_coverage() -> None:
    attempts = [_attempt("2026-12-31T12:00:00Z"), _attempt("2027-01-01T12:00:00Z")]
    row = _compare(attempts, [_quarter(attempts[0])]).iloc[0]
    assert row["current_quarter_status"] == "period_ended_before_capture"
    assert not row["aggregate_fields_comparable"]


def test_ambiguous_time_or_wrong_hash_fail_closed() -> None:
    attempt = _attempt("2026-09-14T12:00:00Z")
    with pytest.raises(AlphaVantageConsensusError, match="simultaneous"):
        _compare([attempt, attempt], [])
    with pytest.raises(AlphaVantageConsensusError, match="match its capture"):
        _compare([attempt], [_quarter(attempt, source_sha256="wrong")])
    with pytest.raises(AlphaVantageConsensusError, match="aware capture"):
        _compare([{**attempt, "captured_at": "2026-09-14T12:00:00"}], [])


def test_empty_and_single_capture_do_not_invent_pairs() -> None:
    assert _compare([], []).empty
    attempt = _attempt("2026-09-14T12:00:00Z")
    assert _compare([attempt], [_quarter(attempt)]).empty
    with pytest.raises(AlphaVantageConsensusError, match="lack capture"):
        _compare([], [_quarter(attempt)])
