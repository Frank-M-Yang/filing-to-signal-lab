"""Offline SEC parsing, point-in-time, and amendment-comparison tests."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sec_data.cleaning import (
    compare_amendment_facts,
    parse_companyfacts,
    parse_submissions_documents,
)

FIXTURES = Path(__file__).parent / "fixtures"
PARSED_AT = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _filings():
    document = _load("sec_submissions.json")
    return parse_submissions_documents(
        [(document, "submissions.recent", "a" * 64)],
        cik="1",
        ticker="RDD",
        company_name="Real Data Demo Corp",
        parsed_at=PARSED_AT,
    )


def test_submissions_preserve_utc_acceptance_and_amendment() -> None:
    result = _filings()

    assert len(result.data) == 2
    assert result.data["accepted_at"].dt.tz is not None
    assert result.data["form"].tolist() == ["10-Q", "10-Q/A"]
    assert result.data["is_amendment"].tolist() == [False, True]
    assert result.data.iloc[0]["filing_date"].strftime("%Y-%m-%d") == "2024-05-03"
    assert result.data.iloc[0]["accepted_at"].isoformat() == "2024-05-02T22:05:00+00:00"


def test_companyfacts_reject_unmatched_time_and_invalid_unit() -> None:
    filings = _filings().data
    result = parse_companyfacts(
        _load("sec_companyfacts.json"),
        filings,
        source_url="fixture://companyfacts.json",
        snapshot_hash="b" * 64,
        parsed_at=PARSED_AT,
    )

    assert len(result.data) == 7
    assert set(result.rejections["reason"]) == {
        "invalid_unit_bucket",
        "missing_acceptance_timestamp",
    }
    original = result.data.loc[result.data["accession_number"] == "0000000001-24-000001"]
    assert original["accepted_at"].eq(filings.iloc[0]["accepted_at"]).all()


def test_amendments_compare_strictly_without_zero_imputation() -> None:
    facts = parse_companyfacts(
        _load("sec_companyfacts.json"),
        _filings().data,
        source_url="fixture://companyfacts.json",
        snapshot_hash="b" * 64,
        parsed_at=PARSED_AT,
    ).data

    comparisons = compare_amendment_facts(facts)

    assert set(comparisons["comparison_status"]) == {"changed", "unchanged", "no_prior_match"}
    revenue = comparisons.loc[
        comparisons["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    ].iloc[0]
    assert revenue["original_value"] == 100_000_000
    assert revenue["amendment_value"] == 98_000_000
    assert revenue["absolute_difference"] == -2_000_000
    assets = comparisons.loc[comparisons["tag"] == "Assets"].iloc[0]
    assert assets["comparison_status"] == "no_prior_match"
    assert assets["original_value"] != 0.0
