"""Tests for frozen validation security identity endpoints."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.validation_security_identity import (
    calculate_validation_event_identity_readiness,
    calculate_validation_identity_intervals,
)


def test_same_figi_ticker_and_exchange_authorizes_only_interval_propagation() -> None:
    audit = _reference_audit()

    intervals, gates, report = calculate_validation_identity_intervals(audit)

    assert intervals.loc[0, "stable_security_endpoint_evidence"]
    assert intervals.loc[0, "automatic_interval_propagation_ready"]
    assert not intervals.loc[0, "price_collection_authorized"]
    assert report["issuers_ready_for_automatic_interval_propagation"] == 1
    assert report["return_outcome_read_authorized"] is False
    assert not gates.loc[gates["gate"].eq("return_outcome_read"), "passed"].item()


def test_same_figi_but_changed_ticker_requires_transition_repair() -> None:
    audit = _reference_audit()
    audit.loc[1, "selected_ticker"] = "NEW"

    intervals, _, report = calculate_validation_identity_intervals(audit)

    assert intervals.loc[0, "stable_security_endpoint_evidence"]
    assert not intervals.loc[0, "automatic_interval_propagation_ready"]
    assert (
        intervals.loc[0, "identity_interval_status"]
        == "stable_security_with_symbol_or_exchange_transition"
    )
    assert report["issuers_requiring_symbol_or_identity_repair"] == 1


def test_changed_figi_is_not_treated_as_one_security_interval() -> None:
    audit = _reference_audit()
    audit.loc[1, "selected_provider_security_id"] = "BBG000OTHER1"

    intervals, _, _ = calculate_validation_identity_intervals(audit)

    assert not intervals.loc[0, "stable_security_endpoint_evidence"]
    assert not intervals.loc[0, "automatic_interval_propagation_ready"]
    assert (
        intervals.loc[0, "identity_interval_status"] == "provider_security_id_changed_or_ambiguous"
    )


def test_outcome_column_is_rejected() -> None:
    audit = _reference_audit()
    audit["future_return"] = 0.01

    with pytest.raises(FilingCatalogError, match="contains outcome"):
        calculate_validation_identity_intervals(audit)


def test_event_readiness_keeps_unresolved_issuers_in_repair_queue() -> None:
    audit = _reference_audit()
    intervals, _, _ = calculate_validation_identity_intervals(audit)
    unresolved = intervals.copy()
    unresolved.loc[0, "automatic_interval_propagation_ready"] = False
    unresolved.loc[0, "exact_event_identity_ready"] = False
    unresolved.loc[0, "identity_interval_status"] = "endpoint_reference_unresolved"
    events = pd.DataFrame(
        [
            {
                "market_event_id": "market-1",
                "validation_event_id": "event-1",
                "cik": "0000000001",
                "price_collection_candidate": True,
                "selection_uses_return_values": False,
            }
        ]
    )

    result = calculate_validation_event_identity_readiness(events, unresolved)

    assert not result.loc[0, "event_security_identity_ready"]
    assert result.loc[0, "identity_repair_required"]
    assert result.loc[0, "identity_assignment_method"] == "unresolved_identity_repair_required"
    assert not result.loc[0, "price_collection_authorized"]


def _reference_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": "first",
                "cik": "0000000001",
                "anchor_role": "first_candidate",
                "query_date": "2024-08-01",
                "provider_query_complete": True,
                "event_date_reference_status": "unique_reference_candidate",
                "event_date_reference_candidate": True,
                "selected_ticker": "DEMO",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "BBG001SABCDE",
                "selected_provider_security_id_kind": "share_class_figi",
                "selection_uses_return_values": False,
            },
            {
                "target_id": "last",
                "cik": "0000000001",
                "anchor_role": "last_candidate",
                "query_date": "2025-08-01",
                "provider_query_complete": True,
                "event_date_reference_status": "unique_reference_candidate",
                "event_date_reference_candidate": True,
                "selected_ticker": "DEMO",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "BBG001SABCDE",
                "selected_provider_security_id_kind": "share_class_figi",
                "selection_uses_return_values": False,
            },
        ]
    )
