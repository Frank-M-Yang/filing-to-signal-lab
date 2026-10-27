"""v3.1 disclosure-clock consensus quality tests (synthetic, outcome-free)."""

import pandas as pd
import pytest

from prospective_disclosure_quality import (
    AWAITING_DISCLOSURE,
    CLEAN_PRE_DISCLOSURE,
    STALE_AT_DISCLOSURE,
    ProspectiveDisclosureQualityConfig,
    ProspectiveDisclosureQualityError,
    calculate_prospective_disclosure_quality,
)

DISCLOSURE_AT = pd.Timestamp("2026-10-22T20:10:00Z")


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": ["a", "b"],
            "registration_order": [1, 2],
            "cik": ["42", "43"],
            "ticker": ["aaa", "bbb"],
            "target_period_end": pd.to_datetime(["2026-09-30", "2026-12-31"], utc=True),
            "accepted_at": [pd.NaT, pd.NaT],
            "actual_value_loaded": [False, False],
            "return_value_loaded": [False, False],
            "selection_uses_return_values": [False, False],
        }
    )


def _snapshots(spec: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for event_id, stamps in spec.items():
        for stamp in stamps:
            captured = pd.Timestamp(stamp)
            rows.append(
                {
                    "prospective_event_id": event_id,
                    "captured_at": captured,
                    "capture_date": captured.normalize(),
                    "revenue_estimate_average": 100.0,
                    "revenue_estimate_analyst_count": 10,
                    "source_sha256": "a" * 64,
                    "historical_backtest_admission": False,
                }
            )
    return pd.DataFrame(rows)


def _clocks(event_date: str = "2026-10-22") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": ["a", "b"],
            "clock_source": ["item_2_02_8k", "none_observed"],
            "earliest_disclosure_at": [DISCLOSURE_AT, pd.NaT],
            "release_event_date": [pd.Timestamp(event_date), pd.NaT],
        }
    )


def _v3(status: str = "clean", reasons: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": ["a", "b"],
            "quality_status": [status, "clean"],
            "quality_reasons": [reasons, ""],
            "last_snapshot_at": [pd.Timestamp("2026-10-27T01:00:00Z")] * 2,
            "actual_values_loaded": [False, False],
            "return_values_loaded": [False, False],
        }
    )


FULL = {
    "a": ["2026-10-10T01:00Z", "2026-10-20T01:00Z", "2026-10-23T01:00Z", "2026-10-27T01:00Z"],
    "b": ["2026-10-10T01:00Z", "2026-10-20T01:00Z"],
}


def test_post_disclosure_captures_are_excluded_and_v3_would_have_used_one() -> None:
    tables = calculate_prospective_disclosure_quality(
        _registry(), _snapshots(FULL), _clocks(), _v3()
    )
    quality = tables.event_quality.set_index("prospective_event_id")
    a = quality.loc["a"]
    assert a["v31_status"] == CLEAN_PRE_DISCLOSURE
    assert a["pre_disclosure_capture_dates"] == 2
    assert a["post_disclosure_capture_dates"] == 2
    assert a["last_pre_disclosure_snapshot_at"] == pd.Timestamp("2026-10-20T01:00Z")
    assert a["staleness_calendar_days_at_disclosure"] == 3.0  # 01:00Z is the prior New York day
    assert bool(a["stale_at_disclosure"]) is False
    # v3 (no acceptance yet) would select the latest capture, which is post-disclosure.
    assert a["v3_candidate_snapshot_at"] == pd.Timestamp("2026-10-27T01:00Z")
    assert bool(a["v3_candidate_post_disclosure"]) is True
    assert bool(a["downstream_admission"]) is True
    b = quality.loc["b"]
    assert b["v31_status"] == AWAITING_DISCLOSURE
    assert b["post_disclosure_capture_dates"] == 0
    assert bool(b["downstream_admission"]) is False
    report = tables.report
    assert report["events_with_excluded_post_disclosure_captures"] == 1
    assert report["v3_candidate_post_disclosure_events"] == 1
    assert report["admissible_pre_disclosure_events"] == 1
    assert report["gates_passed"] == 0 and report["gates_total"] == 4
    assert report["thresholds"]["minimum_complete_events"] == 30


def test_stale_pre_disclosure_consensus_is_quarantined() -> None:
    sparse = {"a": ["2026-10-10T01:00Z", "2026-10-27T01:00Z"], "b": FULL["b"]}
    quality = calculate_prospective_disclosure_quality(
        _registry(), _snapshots(sparse), _clocks(), _v3()
    ).event_quality.set_index("prospective_event_id")
    assert quality.loc["a", "v31_status"] == STALE_AT_DISCLOSURE
    assert quality.loc["a", "staleness_calendar_days_at_disclosure"] == 13.0
    assert bool(quality.loc["a", "downstream_admission"]) is False


def test_event_date_cut_is_stricter_than_acceptance_when_8k_is_furnished_late() -> None:
    # The 8-K is accepted on Oct 22 but reports an Oct 20 event: a capture on the
    # New York date of Oct 21 precedes acceptance yet follows the announcement.
    late = {"a": ["2026-10-21T01:00Z", "2026-10-22T01:00Z"], "b": FULL["b"]}
    quality = calculate_prospective_disclosure_quality(
        _registry(), _snapshots(late), _clocks(event_date="2026-10-20"), _v3()
    ).event_quality.set_index("prospective_event_id")
    assert quality.loc["a", "pre_disclosure_capture_dates"] == 1  # Oct 20 New York date
    assert quality.loc["a", "post_disclosure_capture_dates"] == 1  # Oct 21 New York date
    assert quality.loc["a", "v31_status"] == CLEAN_PRE_DISCLOSURE


def test_v3_structural_quarantine_blocks_admission_but_staleness_alone_does_not() -> None:
    structural = calculate_prospective_disclosure_quality(
        _registry(),
        _snapshots(FULL),
        _clocks(),
        _v3("quarantined", "source_occurrence_changed|stale_preacceptance_consensus"),
    ).event_quality.set_index("prospective_event_id")
    assert structural.loc["a", "v31_status"] == CLEAN_PRE_DISCLOSURE
    assert bool(structural.loc["a", "v3_structural_clean"]) is False
    assert bool(structural.loc["a", "downstream_admission"]) is False
    stale_only = calculate_prospective_disclosure_quality(
        _registry(),
        _snapshots(FULL),
        _clocks(),
        _v3("quarantined", "stale_preacceptance_consensus"),
    ).event_quality.set_index("prospective_event_id")
    assert bool(stale_only.loc["a", "downstream_admission"]) is True


def test_outcome_firewall_and_frozen_thresholds() -> None:
    with pytest.raises(ProspectiveDisclosureQualityError):
        ProspectiveDisclosureQualityConfig(minimum_complete_events=20).validate()
    loaded = _registry()
    loaded.loc[0, "return_value_loaded"] = True
    with pytest.raises(ProspectiveDisclosureQualityError):
        calculate_prospective_disclosure_quality(loaded, _snapshots(FULL), _clocks(), _v3())
    backfilled = _snapshots(FULL)
    backfilled.loc[0, "historical_backtest_admission"] = True
    with pytest.raises(ProspectiveDisclosureQualityError):
        calculate_prospective_disclosure_quality(_registry(), backfilled, _clocks(), _v3())
    with pytest.raises(ProspectiveDisclosureQualityError):
        calculate_prospective_disclosure_quality(
            _registry(), _snapshots(FULL), _clocks().iloc[[0]], _v3()
        )
