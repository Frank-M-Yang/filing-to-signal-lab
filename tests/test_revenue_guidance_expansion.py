"""Tests for the outcome-blind revenue-guidance expansion sample."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_expansion import (
    calculate_revenue_guidance_expansion,
)


def test_issuer_year_sample_is_deterministic_and_revenue_value_invariant() -> None:
    filings = _filing_targets()
    quarterly = _quarterly_history()
    first = calculate_revenue_guidance_expansion(filings, quarterly, _catalog())

    changed = quarterly.copy()
    changed["revenue"] = [9_000_000_000.0, 3.0, 8_000_000_000.0]
    second = calculate_revenue_guidance_expansion(filings, changed, _catalog())

    assert len(first.event_universe) == 2
    assert first.report["selection_uses_revenue_values"] is False
    assert first.report["selection_uses_return_values"] is False
    assert set(first.event_universe["validation_event_id"]) == set(
        second.event_universe["validation_event_id"]
    )
    assert first.guidance_plan.report["selected_guidance_events"] == 2


def _filing_targets() -> pd.DataFrame:
    accepted = pd.to_datetime(
        ["2025-05-01 20:00:00Z", "2025-08-01 20:00:00Z", "2025-06-01 20:00:00Z"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "validation_event_id": ["event-a", "event-b", "event-c"],
            "cik": ["0000000001", "0000000001", "0000000002"],
            "accepted_at": accepted,
            "actual_available_at": accepted,
            "filing_accession": ["filing-a", "filing-b", "filing-c"],
            "fiscal_year": [2025, 2025, 2025],
            "fiscal_quarter": ["Q1", "Q2", "Q1"],
        }
    )


def _quarterly_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filing_accession": ["filing-a", "filing-b", "filing-c"],
            "revenue": [100.0, 110.0, 200.0],
            "unit": ["USD", "USD", "USD"],
        }
    )


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": ["0000000001", "0000000002"],
            "accession_number": ["0000000001-25-000001", "0000000002-25-000001"],
            "form": ["8-K", "8-K"],
            "accepted_at": pd.to_datetime(
                ["2025-02-01 20:00:00Z", "2025-03-01 20:00:00Z"], utc=True
            ),
            "items": ["2.02", "2.02"],
            "primary_document": ["filing.htm", "filing.htm"],
            "source_url": ["source-a", "source-b"],
        }
    )
