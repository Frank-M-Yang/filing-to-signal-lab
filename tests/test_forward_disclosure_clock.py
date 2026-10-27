"""Forward earliest-disclosure clock tests (synthetic submissions, no network)."""

import json

import pandas as pd
import pytest

from sec_data.forward_disclosure_clock import (
    ForwardDisclosureClockConfig,
    ForwardDisclosureClockError,
    assign_forward_disclosure_clocks,
    parse_disclosure_filings,
    summarize_forward_disclosure_clocks,
)

CIK = "0000000042"


def _submissions(filings: list[tuple[str, str, str, str, str]]) -> bytes:
    columns = {
        "accessionNumber": [f[0] for f in filings],
        "form": [f[1] for f in filings],
        "items": [f[2] for f in filings],
        "reportDate": [f[3] for f in filings],
        "acceptanceDateTime": [f[4] for f in filings],
        "filingDate": [f[4][:10] for f in filings],
        "primaryDocument": ["doc.htm"] * len(filings),
    }
    return json.dumps({"cik": "42", "filings": {"recent": columns}}).encode()


SEPTEMBER = [
    # Prior-period 10-Q: a release for Q3 must come after this.
    ("ACC-10Q-Q2", "10-Q", "", "2026-06-30", "2026-08-05T20:30:00.000Z"),
    # Unrelated 8-K without Item 2.02.
    ("ACC-8K-HR", "8-K", "5.02,9.01", "2026-09-15", "2026-09-15T20:05:00.000Z"),
    # A 2.02 8-K on the period-end date itself (before the NY day after).
    ("ACC-8K-EARLY", "8-K", "2.02", "2026-09-30", "2026-09-30T21:00:00.000Z"),
]
OCTOBER = [
    *SEPTEMBER,
    # The Q3 earnings release.
    ("ACC-8K-Q3", "8-K", "2.02,9.01", "2026-10-22", "2026-10-22T20:10:00.000Z"),
    # An amended release never counts.
    ("ACC-8K-Q3A", "8-K/A", "2.02", "2026-10-22", "2026-10-22T22:00:00.000Z"),
]
NOVEMBER = [
    *OCTOBER,
    ("ACC-10Q-Q3", "10-Q", "", "2026-09-30", "2026-11-04T21:15:00.000Z"),
]


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": ["q3", "q4"],
            "cik": ["42", "42"],
            "ticker": ["tst", "tst"],
            "target_period_end": pd.to_datetime(["2026-09-30", "2026-12-31"], utc=True),
            "accepted_at": [pd.NaT, pd.NaT],
        }
    )


def _filings(*snapshots: tuple[list[tuple[str, str, str, str, str]], str]) -> pd.DataFrame:
    rows = []
    for filings, fetched_at in snapshots:
        rows.extend(
            parse_disclosure_filings(
                _submissions(filings),
                cik=CIK,
                ticker="TST",
                observed_at=fetched_at,
                source_sha256="a" * 64,
            )
        )
    return pd.DataFrame(rows)


def test_parser_keeps_only_release_and_periodic_originals() -> None:
    rows = _filings((OCTOBER, "2026-10-23T01:00:00+00:00")).set_index("accession_number")
    # Amendments are kept with their flag so the assignment step can exclude them explicitly.
    assert sorted(rows.index) == ["ACC-10Q-Q2", "ACC-8K-EARLY", "ACC-8K-Q3", "ACC-8K-Q3A"]
    assert rows.loc["ACC-8K-Q3", "items"] == "2.02,9.01"
    assert bool(rows.loc["ACC-8K-Q3A", "is_amendment"]) is True
    assert bool(rows.loc["ACC-8K-Q3", "is_amendment"]) is False
    with pytest.raises(ForwardDisclosureClockError):
        parse_disclosure_filings(
            _submissions(OCTOBER),
            cik="99",
            ticker="X",
            observed_at="2026-10-23T01:00:00Z",
            source_sha256="a" * 64,
        )


def test_release_clock_appears_before_the_periodic_report() -> None:
    clocks = assign_forward_disclosure_clocks(
        _registry(), _filings((OCTOBER, "2026-10-23T01:00:00+00:00"))
    ).set_index("prospective_event_id")
    q3 = clocks.loc["q3"]
    assert q3["clock_source"] == "item_2_02_8k"
    assert q3["release_accession_number"] == "ACC-8K-Q3"
    assert q3["earliest_disclosure_at"] == pd.Timestamp("2026-10-22T20:10:00Z")
    assert q3["disclosure_first_observed_at"] == pd.Timestamp("2026-10-23T01:00:00Z")
    assert q3["release_candidate_count"] == 1  # ACC-8K-EARLY is before the NY day after period end
    assert q3["previous_periodic_report_accepted_at"] == pd.Timestamp("2026-08-05T20:30:00Z")
    assert bool(q3["release_event_date_reliable"]) is True
    assert bool(q3["release_filed_after_event_date"]) is False
    assert pd.isna(q3["periodic_accepted_at"])
    assert clocks.loc["q4", "clock_source"] == "none_observed"
    assert pd.isna(clocks.loc["q4", "earliest_disclosure_at"])
    assert not clocks["first_public_certified"].any()


def test_periodic_report_does_not_move_an_earlier_release_clock() -> None:
    clocks = assign_forward_disclosure_clocks(
        _registry(),
        _filings((OCTOBER, "2026-10-23T01:00:00+00:00"), (NOVEMBER, "2026-11-05T01:00:00+00:00")),
    ).set_index("prospective_event_id")
    q3 = clocks.loc["q3"]
    assert q3["periodic_accession_number"] == "ACC-10Q-Q3"
    assert q3["earliest_disclosure_at"] == pd.Timestamp("2026-10-22T20:10:00Z")
    assert q3["disclosure_first_observed_at"] == pd.Timestamp("2026-10-23T01:00:00Z")
    assert q3["release_lead_days_over_periodic"] == pytest.approx(13.045, abs=1e-3)
    assert bool(q3["periodic_matches_registry"]) is False  # registry acceptance still NaT


def test_periodic_report_is_the_clock_when_no_release_exists() -> None:
    without_release = [f for f in NOVEMBER if f[0] != "ACC-8K-Q3"]
    clocks = assign_forward_disclosure_clocks(
        _registry(), _filings((without_release, "2026-11-05T01:00:00+00:00"))
    ).set_index("prospective_event_id")
    assert clocks.loc["q3", "clock_source"] == "periodic_report"
    assert clocks.loc["q3", "earliest_disclosure_at"] == pd.Timestamp("2026-11-04T21:15:00Z")
    assert clocks.loc["q3", "release_candidate_count"] == 0


def test_summary_and_empty_filings() -> None:
    clocks = assign_forward_disclosure_clocks(_registry(), pd.DataFrame())
    assert clocks["clock_source"].eq("none_observed").all()
    summary = summarize_forward_disclosure_clocks(
        assign_forward_disclosure_clocks(
            _registry(), _filings((NOVEMBER, "2026-11-05T01:00:00+00:00"))
        )
    )
    assert summary["events"] == 2
    assert summary["clock_source_counts"] == {"item_2_02_8k": 1, "none_observed": 1}
    assert summary["release_lead_at_least_3_days"] == 1
    assert summary["by_target_period_end"]["2026-09-30"] == {"item_2_02_8k": 1}


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ForwardDisclosureClockError):
        ForwardDisclosureClockConfig(max_release_days_after_period_end=0).validate()
    duplicate = pd.concat([_registry(), _registry().iloc[[0]]], ignore_index=True)
    with pytest.raises(ForwardDisclosureClockError):
        assign_forward_disclosure_clocks(duplicate, pd.DataFrame())
    early = _filings((OCTOBER, "2026-10-01T00:00:00+00:00"))  # observed before acceptance
    with pytest.raises(ForwardDisclosureClockError):
        assign_forward_disclosure_clocks(_registry(), early)
