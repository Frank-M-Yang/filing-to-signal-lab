"""No in-window changes cannot erase a conflicting pre-window ticker state."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.review_otc_timeline_prefix import assess_prefix
from sec_data.next_publication import DISARM_FLAGS


def audits(events: list[tuple[str, str]]) -> dict[str, Any]:
    """Create normalized synthetic endpoints and a deliberately reversed timeline."""
    return {
        **{
            k: {
                "query_complete": True,
                "identity_candidate_match": True,
                "identities": [{"ticker": "TEST"}],
            }
            for k in ("reference_start", "reference_end")
        },
        "ticker_events": {
            "query_complete": True,
            "events": [
                {"event_date": day, "new_ticker": symbol, "event_type": "ticker_change"}
                for day, symbol in reversed(events)
            ],
        },
    }


def review(events: list[tuple[str, str]]) -> dict[str, Any]:
    """A synthetic date-only window, unrelated to live market requests."""
    return assess_prefix(audits(events), "TEST", "2024-09-11", "2024-10-09")


def test_pre_window_conflict_not_erased_by_zero_in_window_events() -> None:
    r = review([("2003-09-10", "TEST"), ("2023-11-18", "OTHER")])
    assert r["assessment"] == "conditional_timeline_reference_conflict"
    assert r["prefix_conflict_detected"] is True and not r["in_window_nonmatching_symbols"]
    assert all(s["conditional_ticker"] == "OTHER" for s in r["conditional_boundary_states"])
    assert all(r[k] is False for k in DISARM_FLAGS)
    assert not r["historical_security_interval_ready"]


def test_later_rename_cannot_be_carried_backwards() -> None:
    r = review([("2023-11-18", "OTHER"), ("2024-09-30", "TEST")])
    assert r["conditional_boundary_states"][0]["conditional_ticker"] == "OTHER"
    assert r["conditional_boundary_states"][1]["conditional_ticker"] == "TEST"
    assert r["prefix_conflict_detected"] is True


@pytest.mark.parametrize("day", ["2024-09-11", "2024-10-09"])
def test_same_boundary_date_is_not_intraday_state(day: str) -> None:
    r = review([("2003-09-10", "TEST"), (day, "TEST")])
    assert r["assessment"] == "insufficient_evidence"
    assert any(s["conditional_ticker"] is None for s in r["conditional_boundary_states"])


def test_future_event_not_used_and_input_order_does_not_matter() -> None:
    events = [("2003-09-10", "TEST"), ("2026-10-01", "OTHER")]
    r = review(events)
    assert r["assessment"] == "candidate_consistent_not_certified"
    assert r["events_after_window_ignored"] == 1 and r["prefix_conflict_detected"] is False
    assert r == review(list(reversed(events)))
    assert not r["historical_security_interval_ready"]


def test_same_date_conflicting_prior_symbols_are_ambiguous() -> None:
    r = review([("2023-11-18", "TEST"), ("2023-11-18", "OTHER")])
    assert r["assessment"] == "insufficient_evidence"
    assert all(s["conditional_ticker"] is None for s in r["conditional_boundary_states"])


def test_in_window_transition_conflict_retained_even_after_reversal() -> None:
    r = review([("2003-09-10", "TEST"), ("2024-09-12", "OTHER"), ("2024-09-13", "TEST")])
    assert r["prefix_conflict_detected"] is True and r["in_window_nonmatching_symbols"] == ["OTHER"]


def test_no_prior_state_is_unknown_not_zero_changes_certificate() -> None:
    r = review([])
    assert r["assessment"] == "insufficient_evidence"
    assert r["prefix_conflict_detected"] is None
    assert all(s["conditional_ticker"] is None for s in r["conditional_boundary_states"])


def test_incomplete_timeline_or_conflicting_endpoint_is_not_evaluable() -> None:
    d = audits([("2003-09-10", "OTHER")])
    d["ticker_events"]["query_complete"] = False
    r = assess_prefix(d, "TEST", "2024-09-11", "2024-10-09")
    assert r["prefix_conflict_detected"] is None and r["conditional_boundary_states"] == []
    d = audits([("2003-09-10", "OTHER")])
    d["reference_start"]["identity_candidate_match"] = False
    r = assess_prefix(d, "TEST", "2024-09-11", "2024-10-09")
    assert r["assessment"] == "insufficient_evidence" and r["prefix_conflict_detected"] is None


def test_dates_and_events_fail_closed_and_sources_not_mutated() -> None:
    d = audits([("2003-09-10", "TEST")])
    before = copy.deepcopy(d)
    assess_prefix(d, "TEST", "2024-09-11", "2024-10-09")
    assert d == before
    with pytest.raises(ValueError, match="Reversed"):
        assess_prefix(d, "TEST", "2024-10-09", "2024-09-11")
    d["ticker_events"]["events"][0]["event_type"] = "unsupported"
    with pytest.raises(ValueError, match="Malformed"):
        assess_prefix(d, "TEST", "2024-09-11", "2024-10-09")
