"""Frozen expansion selection and original, strictly prior industry classification."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from sec_data.announcement_expansion import (
    attach_prior_industry,
    extract_header_industry,
    industry_rank_diagnostic,
    select_expansion_batch,
)
from sec_data.announcement_replay import SIGNALS


def candidate(issuer: int, event: int, **changes: Any) -> dict[str, Any]:
    """Construct a metadata-only candidate; financial values are not selection inputs."""
    return {
        "cik": f"{issuer:010d}",
        "validation_event_id": f"event-{event:03d}",
        "assigned_ticker": f"T{issuer}",
        "release_accession_number": f"{issuer:010d}-24-{event:06d}",
        "clock_source": "item_2_02_8k",
        "earliest_disclosure_at": (
            pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=event)
        ).isoformat(),
        **changes,
    }


def test_selection_is_issuer_unique_chronological_and_outcome_blind() -> None:
    rows = [candidate(i, i) for i in range(1, 24)]
    rows += [candidate(1, 30), candidate(3, 31)]
    rows[0]["release_value_pair_verified"] = True
    keys = {(r["cik"], r["release_accession_number"]) for r in rows}
    selected = select_expansion_batch(list(reversed(rows)), keys)
    assert [r["cik"] for r in selected] == [f"{i:010d}" for i in range(2, 22)]
    altered = [
        {**r, "market_excess_return": -i, "financial_pair_ready": False} for i, r in enumerate(rows)
    ]
    assert [r["validation_event_id"] for r in select_expansion_batch(altered, keys)] == [
        r["validation_event_id"] for r in selected
    ]


def test_missing_cache_does_not_create_replacements_or_duplicate_events() -> None:
    rows = [candidate(i, i) for i in range(1, 21)]
    keys = {(r["cik"], r["release_accession_number"]) for r in rows[1:]}
    with pytest.raises(ValueError, match="Insufficient"):
        select_expansion_batch(rows, keys)
    with pytest.raises(ValueError, match="duplicate"):
        select_expansion_batch([rows[0], rows[0]], keys)


def header(clock: str = "20240701120000") -> bytes:
    """Provide distinct filer and reporting-owner classifications for provenance tests."""
    return (
        "<SEC-HEADER>\nACCESSION NUMBER: 0000000001-24-000001\n"
        f"<ACCEPTANCE-DATETIME>{clock}\nFILER:\n"
        " CENTRAL INDEX KEY: 1\nSTANDARD INDUSTRIAL CLASSIFICATION: SOFTWARE [7372]\n"
        "REPORTING-OWNER:\n CENTRAL INDEX KEY: 2\n"
        "STANDARD INDUSTRIAL CLASSIFICATION: WRONG OWNER [9999]\n</SEC-HEADER>"
    ).encode()


def test_original_filer_sic_and_eastern_clock_not_reporting_owner() -> None:
    row = extract_header_industry(header(), "0000000001", "0000000001-24-000001")
    assert row["sic"] == "7372"
    assert row["industry_available_at"] == "2024-07-01T16:00:00+00:00"
    assert row["industry_basis"] == "original_SEC_FILER_header"


@pytest.mark.parametrize(
    ("content", "cik", "accession"),
    [
        (header(), "0000000002", "0000000001-24-000001"),
        (header(), "0000000001", "0000000001-24-000002"),
        (header("20241103013000"), "0000000001", "0000000001-24-000001"),
        (header("20240310023000"), "0000000001", "0000000001-24-000001"),
    ],
)
def test_header_identity_and_ambiguous_wall_clocks_fail(
    content: bytes, cik: str, accession: str
) -> None:
    with pytest.raises(ValueError):
        extract_header_industry(content, cik, accession)


def industry(case: dict[str, Any], clock: str, sic: str = "7372") -> dict[str, Any]:
    """One source-hashed SIC observation from the original SEC filing header."""
    return {
        "cik": case["cik"],
        "industry_available_at": clock,
        "sic": sic,
        "source_sha256": "a" * 64,
        "accession_number": case["release_accession_number"],
    }


def test_prior_industry_excludes_same_time_future_and_other_issuer() -> None:
    case = candidate(1, 1)
    observations = [
        industry(case, "2024-07-01T12:00:00Z"),
        industry(case, case["earliest_disclosure_at"], "9999"),
        industry(case, "2025-01-01T00:00:00Z", "9999"),
        industry(candidate(2, 1), "2024-08-01T00:00:00Z", "8888"),
    ]
    matched = attach_prior_industry([case], observations).iloc[0]
    assert matched.industry_ready
    assert matched.sic == "7372"
    assert matched.sic_2digit == "73"
    unmatched = attach_prior_industry([case], observations[1:]).iloc[0]
    assert not unmatched.industry_ready


def test_prior_industry_conflicting_ties_missing_and_invalid_hash_rejected() -> None:
    case = candidate(1, 1)
    source = industry(case, "2024-07-01T12:00:00Z")
    conflicting = attach_prior_industry([case], [source, {**source, "sic": "9999"}])
    assert conflicting.iloc[0].industry_status == "same_time_conflicting_SIC"
    assert not conflicting.iloc[0].industry_ready
    with pytest.raises(ValueError, match="provenance"):
        attach_prior_industry([case], [{**source, "source_sha256": "unhashed"}])


def industry_sample(count: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic two-industry sample, solely for testing diagnostic gates."""
    rows = [candidate(i, i) for i in range(1, count + 1)]
    analysis = pd.DataFrame(
        [
            {
                "validation_event_id": r["validation_event_id"],
                "cik": r["cik"],
                "announcement_at": r["earliest_disclosure_at"],
                "market_excess_return": i / 100,
                **{name: i / (j + 1) for j, name in enumerate(SIGNALS)},
            }
            for i, r in enumerate(rows)
        ]
    )
    sources = [
        industry(r, "2024-07-01T12:00:00Z", "7372" if i % 2 else "2834") for i, r in enumerate(rows)
    ]
    return analysis, attach_prior_industry(rows, sources)


def test_industry_diagnostic_same_sample_gate_and_no_inference() -> None:
    analysis, prior = industry_sample()
    _, stats, report = industry_rank_diagnostic(analysis, prior)
    assert report["industry_composition_diagnostic_allowed"]
    assert stats.computed.all()
    assert stats.events.eq(20).all()
    assert not report["tradable_sector_benchmark"]
    assert not report["inferential_tests_computed"]
    _, short_stats, short_report = industry_rank_diagnostic(*industry_sample(19))
    assert not short_report["industry_composition_diagnostic_allowed"]
    assert short_stats.prior_SIC2_rank_residual_ic.isna().all()
    with pytest.raises(ValueError, match="weaken"):
        industry_rank_diagnostic(analysis, prior, minimum_events=6)


def test_industry_requires_two_issuers_not_just_repeated_quarters() -> None:
    analysis, prior = industry_sample()
    for frame in (analysis, prior):
        frame.loc[frame.index % 2 == 0, "cik"] = "0000000001"
        frame.loc[frame.index % 2 != 0, "cik"] = "0000000002"
    _, stats, report = industry_rank_diagnostic(analysis, prior)
    assert report["eligible_industries"] == 0
    assert not stats.computed.any()


@pytest.mark.parametrize("violation", ["duplicate", "issuer", "future", "missing"])
def test_industry_identity_and_future_classification_fail(violation: str) -> None:
    analysis, prior = industry_sample()
    prior = copy.deepcopy(prior)
    if violation == "duplicate":
        prior = pd.concat([prior, prior.iloc[[0]]], ignore_index=True)
    elif violation == "issuer":
        prior.loc[0, "cik"] = "0000009999"
    elif violation == "future":
        prior.loc[0, "industry_available_at"] = "2025-01-01T00:00:00Z"
    else:
        prior = prior.iloc[1:]
    with pytest.raises(ValueError):
        industry_rank_diagnostic(analysis, prior)
