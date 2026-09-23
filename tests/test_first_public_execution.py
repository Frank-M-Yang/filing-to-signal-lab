"""Execution-window comparison tests (synthetic calendar, no prices)."""

import pandas as pd
import pytest

from sec_data.first_public_execution import (
    FirstPublicExecutionError,
    compare_execution_clocks,
    summarize_execution_comparison,
)

SHA = "a" * 64


def _sessions() -> pd.DataFrame:
    # Synthetic XNAS weekdays inside daylight-saving time (open 13:30Z, close 20:00Z).
    days = pd.bdate_range("2025-04-01", "2025-07-31")
    return pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": [day.strftime("%Y-%m-%d") for day in days],
            "regular_open_at": [f"{day:%Y-%m-%d}T13:30:00+00:00" for day in days],
            "regular_close_at": [f"{day:%Y-%m-%d}T20:00:00+00:00" for day in days],
            "source_snapshot_sha256": SHA,
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_event_id": ["e1", "e2", "e3", "e4"],
            "cik": ["A", "B", "C", "D"],
            "target_accession_number": ["A-10Q", "B-10Q", "C-10Q", "D-10Q"],
            "accepted_at": [
                "2025-05-01T20:30:00+00:00",
                "2025-05-01T20:30:00+00:00",
                "2025-05-01T20:30:00+00:00",
                "2025-05-01T20:30:00+00:00",
            ],
            "assigned_exchange_mic": "XNAS",
        }
    )


def _clocks() -> pd.DataFrame:
    report = pd.Timestamp("2025-05-01T20:30:00Z")
    return pd.DataFrame(
        {
            "cik": ["A", "B", "D"],
            "accession_number": ["A-10Q", "B-10Q", "D-10Q"],
            "report_accepted_at": [report, report, report],
            "earliest_disclosure_at": [
                pd.Timestamp("2025-04-24T20:05:00Z"),  # release a week earlier
                report,  # no release
                pd.Timestamp("2025-05-01T12:00:00Z"),  # release on the same New York date
            ],
            "clock_source": ["item_2_02_8k", "periodic_report", "item_2_02_8k"],
            "disclosure_lead_days": [7.017, 0.0, 0.354],
        }
    )


def _compare() -> pd.DataFrame:
    return compare_execution_clocks(_events(), _clocks(), _sessions()).set_index(
        "validation_event_id"
    )


def test_earlier_release_moves_entry_by_counted_sessions() -> None:
    result = _compare()
    # Filing on May 1 enters May 5; the April 24 release enters April 28.
    assert result.loc["e1", "filing_entry_session_date"] == "2025-05-05"
    assert result.loc["e1", "disclosure_entry_session_date"] == "2025-04-28"
    assert bool(result.loc["e1", "entry_changed"]) is True
    assert result.loc["e1", "entry_sessions_earlier"] == 5


def test_unchanged_and_missing_clocks_are_kept() -> None:
    result = _compare()
    assert bool(result.loc["e2", "entry_changed"]) is False
    assert bool(result.loc["e4", "entry_changed"]) is False  # same New York date
    assert result.loc["e3", "window_status"] == "clock_not_found"
    assert result.loc["e3", "filing_entry_session_date"] == "2025-05-05"
    assert not result["returns_computed"].any()


def test_summary_counts() -> None:
    summary = summarize_execution_comparison(
        compare_execution_clocks(_events(), _clocks(), _sessions())
    )
    assert summary["events"] == 4
    assert summary["compared"] == 3
    assert summary["entry_changed"] == 1
    assert summary["entry_sessions_earlier_counts"] == {"5": 1}
    assert summary["lead_at_least_3_days"] == 1
    assert summary["filing_clock_mismatches"] == 0


def test_rejects_duplicate_identities() -> None:
    events = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    with pytest.raises(FirstPublicExecutionError):
        compare_execution_clocks(events, _clocks(), _sessions())
    clocks = pd.concat([_clocks(), _clocks().iloc[[0]]], ignore_index=True)
    with pytest.raises(FirstPublicExecutionError):
        compare_execution_clocks(_events(), clocks, _sessions())
