"""No backdated financial actuals, no missing-session substitution, no promoted forecasts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.audit_announcement_admission import build_audit
from sec_data.announcement_admission import (
    FORECAST_CLOCKS,
    audit_forecast_metadata,
    exact_session_coverage,
    gap_records,
    join_candidate_clocks,
    price_query_covers_history,
    review_candidate_pair,
)


def _event() -> dict[str, Any]:
    return {
        "validation_event_id": "event",
        "cik": "1",
        "clock_source": "item_2_02_8k",
        "release_accession_number": "early",
        "period_end": "2025-06-30",
        "report_date": "2025-06-30",
        "earliest_disclosure_at": "2025-08-01T12:00:00Z",
    }


def _cells() -> list[dict[str, Any]]:
    return [
        {
            "validation_event_id": "event",
            "cik": "1",
            "release_accession": "early",
            "metric": metric,
            "value": value,
            "period_start": "2025-04-01",
            "period_end": "2025-06-30",
            "unit": "USD",
            "source_sha256": "a" * 64,
            "exhibit_sha256": "b" * 64,
            "clock_status": "consistent_acceptance_proxy",
            "sec_acceptance_proxy_at": "2025-08-01T12:00:00Z",
            "comparison_status": "exact_original_match",
            "original_status": "original_quarter",
        }
        for metric, value in (("revenue", 100.0), ("gross_profit", 40.0))
    ]


def test_exact_original_release_pair_is_only_value_evidence() -> None:
    result = review_candidate_pair(_event(), _cells())
    assert result["release_value_pair_verified"]
    assert result["release_gross_margin"] == 0.4
    assert "model_admission" not in result and "first_public_certified" not in result


@pytest.mark.parametrize(
    "key,value",
    [
        ("cik", "other"),
        ("validation_event_id", "other"),
        ("release_accession", "later"),
        ("period_end", "2025-03-31"),
        ("period_start", "2025-01-01"),
        ("unit", "EUR"),
        ("source_sha256", "c" * 64),
        ("exhibit_sha256", "d" * 64),
        ("original_status", "derived_ytd_difference"),
        ("comparison_status", "rounded_match"),
        ("clock_status", "clock_conflict_quarantined"),
        ("sec_acceptance_proxy_at", "2025-08-02T12:00:00Z"),
    ],
)
def test_wrong_release_identity_period_unit_or_vintage_never_backdated(
    key: str, value: Any
) -> None:
    cells = _cells()
    cells[0][key] = value
    assert not review_candidate_pair(_event(), cells)["release_value_pair_verified"]


def test_rule_candidate_earlier_than_reviewed_final_release_cannot_borrow_its_values() -> None:
    event = _event()
    event["release_accession_number"] = "preliminary"
    result = review_candidate_pair(event, _cells())
    assert result["candidate_value_status"] == "selected_release_unreviewed"
    assert not result["release_value_pair_verified"]


def test_absent_8k_not_proof_of_first_public_and_missing_or_duplicate_cells_fail() -> None:
    event = _event()
    event["clock_source"] = "periodic_report"
    assert (
        review_candidate_pair(event, _cells())["candidate_value_status"]
        == "no_candidate_not_proof_of_first_public"
    )
    for cells in ([], _cells()[:1], _cells() + _cells()[:1]):
        assert not review_candidate_pair(_event(), cells)["release_value_pair_verified"]


def test_forecast_clock_precheck_requires_strictly_earlier_sigma_errors_and_all_dependencies() -> (
    None
):
    metadata = dict.fromkeys(FORECAST_CLOCKS, "2025-07-31T12:00:00Z")
    good = audit_forecast_metadata(metadata, _event()["earliest_disclosure_at"])
    assert good["existing_forecast_clocks_before_proxy"]
    assert not good["history_replay_completed"]
    for key in FORECAST_CLOCKS:
        same_time = {**metadata, key: _event()["earliest_disclosure_at"]}
        result = audit_forecast_metadata(same_time, _event()["earliest_disclosure_at"])
        assert result["non_prior_forecast_clocks"] == [key]
        assert not result["existing_forecast_clocks_before_proxy"]
        missing = {**metadata, key: None}
        result = audit_forecast_metadata(missing, _event()["earliest_disclosure_at"])
        assert result["missing_forecast_clocks"] == [key]
        assert not result["existing_forecast_clocks_before_proxy"]


def _calendar() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=70).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {"session_date": dates, "regular_close_at": [d + "T20:00:00Z" for d in dates]}
    )


def test_sixty_one_consecutive_sessions_not_any_sixty_one_observed_rows() -> None:
    cal = _calendar()
    cutoff = (pd.Timestamp(cal.session_date.iloc[-1]) + pd.Timedelta(days=1)).isoformat() + "Z"
    complete = exact_session_coverage(cal, cal.session_date.tolist(), cutoff)
    assert complete["history_dates_complete"]
    dates = cal.session_date.tolist()
    absent = dates.pop(-10)
    # There are still 69 observed rows, but one is missing inside the exact 61-day window.
    result = exact_session_coverage(cal, dates, cutoff)
    assert not result["history_dates_complete"] and result["missing_dates"] == [absent]


def test_new_york_date_not_utc_date_and_future_dates_do_not_fill_history() -> None:
    cal = _calendar()
    last = cal.session_date.iloc[-1]
    cutoff = (pd.Timestamp(last) + pd.Timedelta(days=1, hours=1)).isoformat() + "Z"
    a = exact_session_coverage(cal, cal.session_date.tolist(), cutoff)
    assert a["required_dates"][-1] == cal.session_date.iloc[-2]
    b = exact_session_coverage(cal, [*cal.session_date.tolist(), "2030-01-01"], cutoff)
    assert a == b
    short = exact_session_coverage(cal.tail(60), cal.session_date.tolist(), cutoff)
    assert not short["calendar_ready"] and not short["history_dates_complete"]


def test_duplicate_market_dates_rejected() -> None:
    cal = _calendar()
    with pytest.raises(ValueError, match="Duplicate"):
        exact_session_coverage(
            cal, [*cal.session_date.tolist(), cal.session_date.iloc[0]], "2025-06-01T12:00Z"
        )


def test_calendar_close_must_match_session_and_history_length_positive() -> None:
    cal = _calendar()
    cal.loc[cal.index[-1], "regular_close_at"] = "2025-06-01T20:00Z"
    with pytest.raises(ValueError, match="date/close"):
        exact_session_coverage(cal, cal.session_date.tolist(), "2025-06-02T12:00Z")
    with pytest.raises(ValueError, match="Positive"):
        exact_session_coverage(_calendar(), [], "2025-06-02T12:00Z", required=0)


def test_price_dates_alone_do_not_certify_query_completeness_or_identity() -> None:
    event = {"price_target_id": "p", "cik": "1", "ticker": "TEST"}
    audit = {
        **event,
        "provider_query_complete": True,
        "parser_rejection_rows": 0,
        "price_window_start": "2025-01-01",
        "price_window_end": "2025-06-30",
    }
    dates = ["2025-03-01", "2025-05-01"]
    assert price_query_covers_history(event, audit, dates)
    for key, value in (
        ("cik", "2"),
        ("ticker", "OTHER"),
        ("price_target_id", "other"),
        ("provider_query_complete", False),
        ("parser_rejection_rows", 1),
        ("price_window_start", "2025-03-02"),
        ("price_window_end", "2025-04-30"),
    ):
        assert not price_query_covers_history(event, {**audit, key: value}, dates)
    assert not price_query_covers_history(event, {}, dates)
    assert not price_query_covers_history(event, audit, [])


def test_gap_ledger_preserves_missing_clock_and_benchmark_execution_dates() -> None:
    row = {
        **_event(),
        "assigned_ticker": "TEST",
        "blocking_reasons": [
            "missing_or_conflicting_catalog_clock",
            "benchmark_missing_exact_61_session_history",
            "execution_issuer_dates_missing",
        ],
        "missing_benchmark_dates": ["2025-02-03"],
        "missing_execution_issuer_dates": ["2025-08-04"],
    }
    gaps = gap_records(row)
    assert [g["gap"] for g in gaps] == row["blocking_reasons"]
    assert [g["missing_dates"] for g in gaps] == [[], ["2025-02-03"], ["2025-08-04"]]
    assert all(not g["automatic_download_authorized"] for g in gaps)


def test_clock_join_requires_issuer_plus_accession_and_retains_unmatched_events() -> None:
    events = pd.DataFrame(
        {
            "validation_event_id": ["a", "b"],
            "cik": ["1", "2"],
            "target_accession_number": ["shared", "shared"],
            "accepted_at": ["2025-08-03T12:00Z"] * 2,
        }
    )
    clocks = pd.DataFrame(
        {
            "cik": ["1"],
            "accession_number": ["shared"],
            "report_accepted_at": ["2025-08-03T12:00Z"],
            "earliest_disclosure_at": ["2025-08-01T12:00Z"],
        }
    )
    result = join_candidate_clocks(events, clocks)
    assert len(result) == 2
    assert result.filing_clock_matches_catalog.tolist() == [True, False]
    assert pd.isna(result.earliest_disclosure_at.iloc[1])
    with pytest.raises(ValueError, match="identity"):
        join_candidate_clocks(events, pd.concat([clocks, clocks]))


@pytest.mark.parametrize(
    "key,value",
    [
        ("network_allowed", True),
        ("returns_allowed", True),
        ("model_fitting_allowed", True),
        ("post_event_numeric_prices_allowed", True),
        ("sample_replacement_allowed", True),
        ("frozen_event_count", 30),
        ("required_sessions", 20),
    ],
)
def test_weakened_contract_stops_before_data_or_outputs(
    tmp_path: Path, key: str, value: Any
) -> None:
    config = json.loads(Path("docs/announcement_admission_v1.json").read_bytes())
    changed = copy.deepcopy(config)
    changed[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="contract"):
        build_audit(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
