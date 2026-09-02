"""Tests for the outcome-blind guidance return evidence union."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_return_bridge import (
    calculate_revenue_guidance_return_bridge,
)


def test_return_bridge_unions_disjoint_authorized_batches() -> None:
    forecasts = pd.DataFrame(
        {"validation_event_id": ["one", "two", "missing"], "cik": ["1", "2", "3"]}
    )
    authorizations = [_authorization("one"), _authorization("two")]
    ledgers = [_ledger("one", "1"), _ledger("two", "2")]

    authorized, ledger, coverage = calculate_revenue_guidance_return_bridge(
        forecasts, authorizations, ledgers
    )

    assert set(authorized["validation_event_id"]) == {"one", "two"}
    assert set(ledger["validation_event_id"]) == {"one", "two"}
    assert coverage["authorized_return_ready"].sum() == 2
    assert not coverage["selection_uses_return_values"].any()


def _authorization(event_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_event_id": [event_id],
            "return_outcome_read_authorized": [True],
            "selection_uses_return_values": [False],
            "authorization_uses_return_values": [False],
            "returns_computed": [False],
        }
    )


def _ledger(event_id: str, cik: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_event_id": [event_id],
            "cik": [cik],
            "market_excess_return_ready": [True],
            "return_outcome_read_authorized": [True],
            "selection_uses_return_values": [False],
            "returns_computed": [True],
        }
    )
