"""Tests for the formal validation security-action adapter."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_security_actions import (
    calculate_validation_security_action_targets,
    filter_authorized_security_action_targets,
)


def test_action_adapter_preserves_target_and_aggregates_event_span() -> None:
    frozen = pd.DataFrame([_frozen()])
    components = pd.DataFrame(
        [
            _component("event-2", "2025-04-01T20:00:00+00:00", "2025-04-03", "2025-05-01"),
            _component("event-1", "2025-01-01T20:00:00+00:00", "2025-01-03", "2025-01-31"),
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "price_target_id": "price-1",
                "selected_provider_security_id": "ID-1",
                "selected_provider_security_id_kind": "share_class_figi",
            }
        ]
    )

    targets = calculate_validation_security_action_targets(frozen, components, prices)

    assert targets.loc[0, "action_target_id"] == "action-1"
    assert targets.loc[0, "accepted_at"] == "2025-01-01T20:00:00+00:00"
    assert targets.loc[0, "formal_entry_session_date"] == "2025-01-03"
    assert targets.loc[0, "formal_exit_session_date"] == "2025-05-01"
    assert targets.loc[0, "event_component_rows"] == 2
    assert not targets["selection_uses_return_values"].any()


def test_action_adapter_filters_only_authorized_security_spans() -> None:
    components = pd.DataFrame(
        [
            _component("event-1", "2025-01-01T20:00:00+00:00", "2025-01-03", "2025-01-31"),
            {
                **_component("event-2", "2025-02-01T20:00:00+00:00", "2025-02-03", "2025-03-03"),
                "cik": "0000000002",
                "ticker": "OTHER",
                "identity_key": "ID-2",
            },
        ]
    )
    target = calculate_validation_security_action_targets(
        pd.DataFrame([_frozen()]),
        components.loc[components["security_event_component_id"].eq("event-1")],
        pd.DataFrame(
            [
                {
                    "price_target_id": "price-1",
                    "selected_provider_security_id": "ID-1",
                    "selected_provider_security_id_kind": "share_class_figi",
                }
            ]
        ),
    )
    authorized = pd.DataFrame(
        [
            {
                "security_event_component_id": "event-1",
                "return_outcome_read_authorized": True,
                "authorization_uses_return_values": False,
                "returns_computed": False,
            }
        ]
    )

    selected = filter_authorized_security_action_targets(target, authorized, components)

    assert selected["action_target_id"].tolist() == ["action-1"]
    assert selected.loc[0, "authorized_event_components"] == 1


def _frozen() -> dict[str, object]:
    return {
        "price_target_id": "price-1",
        "cik": "0000000001",
        "selected_ticker": "TEST",
        "selected_exchange_mic": "XNAS",
        "security_identity_key": "ID-1",
        "security_identity_key_kind": "share_class_figi",
        "price_window_start": "2025-01-01",
        "price_window_end": "2025-05-02",
        "action_target_id": "action-1",
        "action_acquisition_order": 1,
        "selected_composite_figi": "COMP-1",
        "selected_share_class_figi": "ID-1",
        "currency_name": "usd",
        "action_identity_evidence_source_hashes": "hash-1",
        "ticker_change_query_mode": "composite_figi_ticker_events",
        "session_grid_fallback_frozen_before_collection": False,
        "action_collection_authorized": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
    }


def _component(component: str, accepted_at: str, entry: str, exit_date: str) -> dict[str, object]:
    return {
        "security_event_component_id": component,
        "cik": "0000000001",
        "ticker": "TEST",
        "exchange_mic": "XNAS",
        "identity_key": "ID-1",
        "accepted_at": accepted_at,
        "form": "10-Q",
        "accession_number": f"accession-{component}",
        "formal_entry_session_date": entry,
        "formal_exit_session_date": exit_date,
    }
