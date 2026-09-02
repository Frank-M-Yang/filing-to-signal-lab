"""Tests for outcome-blind validation action identity continuity."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_action_continuity import calculate_validation_action_continuity


def test_repairs_unavailable_ticker_events_with_endpoints_and_daily_bars() -> None:
    authorized = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "return_outcome_read_authorized": [True],
            "authorization_uses_return_values": [False],
            "returns_computed": [False],
        }
    )
    components = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "cik": ["1"],
            "ticker": ["ABC"],
            "exchange_mic": ["XNAS"],
            "identity_key": ["share-1"],
            "formal_entry_session_date": ["2025-01-02"],
            "formal_exit_session_date": ["2025-01-06"],
        }
    )
    audit = pd.DataFrame(
        {
            "action_target_id": ["a1"],
            "action_acquisition_order": [1],
            "price_target_id": ["p1"],
            "cik": ["1"],
            "selected_ticker": ["ABC"],
            "selected_exchange_mic": ["XNAS"],
            "selected_provider_security_id": [None],
            "selected_share_class_figi": [None],
            "selected_composite_figi": [None],
            "security_identity_key": ["share-1"],
            "reference_start_query_complete": [True],
            "reference_end_query_complete": [True],
            "ticker_events_query_complete": [False],
            "historical_symbol_interval_ready": [False],
            "corporate_actions_ready": [True],
            "selection_uses_return_values": [False],
            "returns_computed": [False],
        }
    )
    identities = pd.DataFrame(
        {
            "action_target_id": ["a1", "a1"],
            "query_kind": ["reference_start", "reference_end"],
            "ticker": ["ABC", "ABC"],
            "exchange_mic": ["XNAS", "XNAS"],
            "security_type": ["CS", "CS"],
            "provider_security_id": ["share-1", "share-1"],
            "share_class_figi": ["share-1", "share-1"],
            "composite_figi": ["composite-1", "composite-1"],
            "name": ["ABC Corp", "ABC Corp"],
            "source_sha256": ["a" * 64, "b" * 64],
        }
    )
    ticker_events = pd.DataFrame(columns=["action_target_id", "event_date", "new_ticker"])
    price_dates = pd.DataFrame({"price_target_id": ["p1", "p1", "p1"], "session_date": _dates()})
    sessions = pd.DataFrame({"exchange_mic": ["XNAS"] * 3, "session_date": _dates()})
    result = calculate_validation_action_continuity(
        authorized, components, audit, identities, ticker_events, price_dates, sessions
    )
    row = result.iloc[0]
    assert row["formal_action_evidence_ready"]
    assert row["resolved_provider_security_id"] == "share-1"
    assert row["daily_bar_session_coverage"] == 1.0
    assert row["identity_continuity_route"] == (
        "stable_reference_endpoints_plus_official_session_daily_bars"
    )


def _dates() -> list[str]:
    return ["2025-01-02", "2025-01-03", "2025-01-06"]
