"""Point-in-time quarterly fact-selection tests."""

import pandas as pd
import pytest

from sec_data.event_cleaning import (
    clean_company_quarterly_events,
    derive_point_in_time_yoy_lags,
    freeze_core_asset,
    validate_cleaned_quarterly_events,
)


def test_freeze_core_asset_is_order_invariant() -> None:
    core = pd.DataFrame([_core_row(2, "2", "B"), _core_row(1, "1", "A")])

    first, first_version = freeze_core_asset(core, frozen_at="2026-01-01T00:00:00Z")
    _second, second_version = freeze_core_asset(core.iloc[::-1], frozen_at="2026-01-02T00:00:00Z")

    assert first_version == second_version
    assert first["sec_cik"].astype(str).tolist() == ["1", "2"]


def test_cleaning_prefers_direct_profit_and_keeps_latest_separate() -> None:
    filings = _filings([("a1", "2020-03-31", "2020-04-20T20:00:00Z")])
    facts = pd.DataFrame(
        [
            _fact("r1", "Revenues", 100.0, "a1", "2020-01-01", "2020-03-31"),
            _fact("g1", "GrossProfit", 40.0, "a1", "2020-01-01", "2020-03-31"),
            _fact(
                "r2",
                "Revenues",
                110.0,
                "a2",
                "2020-01-01",
                "2020-03-31",
                accepted="2021-02-01T20:00:00Z",
            ),
            _fact(
                "g2",
                "GrossProfit",
                55.0,
                "a2",
                "2020-01-01",
                "2020-03-31",
                accepted="2021-02-01T20:00:00Z",
            ),
        ]
    )

    cleaned = clean_company_quarterly_events(
        _core_row(1, "1", "ABC"),
        filings,
        facts,
        core_asset_version="v1",
    )

    event = cleaned.events.iloc[0]
    assert event["revenue"] == 100.0
    assert event["gross_margin"] == 0.4
    assert event["gross_margin_latest"] == 0.5
    assert bool(event["was_revised"]) is True
    assert event["derivation_method"] == "direct_gross_profit"


def test_cleaning_rejects_unit_mismatch() -> None:
    filings = _filings([("a1", "2020-03-31", "2020-04-20T20:00:00Z")])
    revenue = _fact("r1", "Revenues", 100.0, "a1", "2020-01-01", "2020-03-31")
    profit = _fact("g1", "GrossProfit", 40.0, "a1", "2020-01-01", "2020-03-31")
    profit["unit"] = "EUR"

    cleaned = clean_company_quarterly_events(
        _core_row(1, "1", "ABC"),
        filings,
        pd.DataFrame([revenue, profit]),
        core_asset_version="v1",
    )

    assert cleaned.events.empty
    assert cleaned.rejections.loc[0, "reason"] == "no_same_unit_same_period_fact_pair"


def test_latest_values_never_mix_different_accessions() -> None:
    filings = _filings([("a1", "2020-03-31", "2020-04-20T20:00:00Z")])
    facts = pd.DataFrame(
        [
            _fact("r1", "Revenues", 100.0, "a1", "2020-01-01", "2020-03-31"),
            _fact("g1", "GrossProfit", 40.0, "a1", "2020-01-01", "2020-03-31"),
            _fact(
                "r2",
                "Revenues",
                200.0,
                "a2",
                "2020-01-01",
                "2020-03-31",
                accepted="2021-01-01T20:00:00Z",
            ),
            _fact(
                "g2",
                "GrossProfit",
                90.0,
                "a3",
                "2020-01-01",
                "2020-03-31",
                accepted="2021-02-01T20:00:00Z",
            ),
        ]
    )

    cleaned = clean_company_quarterly_events(
        _core_row(1, "1", "ABC"), filings, facts, core_asset_version="v1"
    )

    event = cleaned.events.iloc[0]
    assert event["gross_margin_latest"] == 0.4
    assert bool(event["was_revised"]) is False


def test_yoy_lag_requires_prior_availability() -> None:
    events = pd.DataFrame(
        [
            _event("e1", 2020, "2020-04-20T20:00:00Z", 0.40),
            _event("e2", 2021, "2021-04-20T20:00:00Z", 0.45),
        ]
    )

    result = derive_point_in_time_yoy_lags(events)

    assert pd.isna(result.loc[0, "gm_delta_yoy"])
    assert result.loc[1, "gm_yoy_lag"] == 0.40
    assert result.loc[1, "gm_delta_yoy"] == pytest.approx(0.05)
    assert result.loc[1, "gm_yoy_lag_event_id"] == "e1"


def test_validation_accepts_tied_point_in_time_delta() -> None:
    events = pd.DataFrame(
        [
            _validated_event("e1", "a1", 2020, "2020-04-20T20:00:00Z", 0.40),
            _validated_event("e2", "a2", 2021, "2021-04-20T20:00:00Z", 0.45),
        ]
    )
    result = derive_point_in_time_yoy_lags(events)

    validate_cleaned_quarterly_events(result)

    assert bool(result.loc[1, "analysis_ready_delta"]) is True


def _core_row(rank: int, cik: str, ticker: str) -> dict[str, object]:
    return {
        "fortune_rank": rank,
        "company_name": ticker,
        "country": "U.S.",
        "sec_cik": cik,
        "sec_ticker": ticker,
        "dataset_version": "dataset",
        "normalized_directory": "fixture",
    }


def _filings(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accession_number": accession,
                "form": "10-Q",
                "filing_date": accepted[:10],
                "report_date": report_date,
                "accepted_at": accepted,
            }
            for accession, report_date, accepted in rows
        ]
    )


def _fact(
    fact_id: str,
    tag: str,
    value: float,
    accession: str,
    start: str,
    end: str,
    *,
    accepted: str = "2020-04-20T20:00:00Z",
) -> dict[str, object]:
    return {
        "fact_id": fact_id,
        "taxonomy": "us-gaap",
        "tag": tag,
        "unit": "USD",
        "value_numeric": value,
        "period_start": start,
        "period_end": end,
        "duration_days": 90,
        "fiscal_year": 2020,
        "fiscal_period": "Q1",
        "accession_number": accession,
        "base_form": "10-Q",
        "is_amendment": False,
        "accepted_at": accepted,
        "source_snapshot_sha256": "a" * 64,
    }


def _event(event_id: str, fiscal_year: int, accepted: str, margin: float) -> dict[str, object]:
    return {
        "event_id": event_id,
        "company_id": "0000000001",
        "fiscal_year": fiscal_year,
        "fiscal_quarter": "Q1",
        "period_end": f"{fiscal_year}-03-31",
        "actual_available_at": accepted,
        "gross_margin": margin,
        "gm_yoy_lag": float("nan"),
        "gm_yoy_lag_event_id": None,
        "gm_yoy_lag_available_at": pd.NaT,
        "gm_delta_yoy": float("nan"),
        "analysis_ready_level": True,
        "analysis_ready_delta": False,
    }


def _validated_event(
    event_id: str,
    accession: str,
    fiscal_year: int,
    accepted: str,
    margin: float,
) -> dict[str, object]:
    event = _event(event_id, fiscal_year, accepted, margin)
    event.update(
        {
            "filing_accession": accession,
            "filing_accepted_at": accepted,
            "revenue": 100.0,
            "gross_profit": 100.0 * margin,
            "latest_available_at": accepted,
            "latest_filing_accession": accession,
            "was_revised": False,
            "accounting_scope_status": "high_confidence",
            "economic_scope_status": "standard_review",
            "plausibility_status": "within_bounds",
        }
    )
    return event
