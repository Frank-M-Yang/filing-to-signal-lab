"""Tests for exact event-date revenue-guidance identity transition repair."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_identity_transition import (
    calculate_revenue_guidance_identity_resolution,
    calculate_revenue_guidance_transition_targets,
)


def test_transition_plan_queries_only_missing_middle_event_and_resolves_all() -> None:
    market, base_identity, base_reference, middle_reference = _fixtures()
    targets, audit, report = calculate_revenue_guidance_transition_targets(
        market, base_identity, base_reference
    )

    assert report["transition_events"] == 3
    assert report["existing_exact_endpoint_references"] == 2
    assert report["new_exact_event_queries"] == 1
    assert targets.iloc[0]["validation_event_id"] == "event-1"
    assert audit["selection_uses_return_values"].eq(False).all()

    resolved, components, resolution = calculate_revenue_guidance_identity_resolution(
        base_identity, base_reference, middle_reference
    )
    assert resolution["price_candidate_events_identity_ready"] == 3
    assert resolution["transition_events_resolved"] == 3
    assert resolved["event_security_identity_ready"].all()
    assert (
        resolved.set_index("validation_event_id").loc["event-1", "assigned_exchange_mic"] == "XNYS"
    )
    assert components.empty


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_rows = []
    base_rows = []
    for index, date in enumerate(("2022-11-14", "2023-08-14", "2024-03-28")):
        accepted = pd.Timestamp(f"{date} 21:00:00", tz="UTC")
        market_rows.append(
            {
                "market_event_id": f"market-{index}",
                "validation_event_id": f"event-{index}",
                "cik": "0001784851",
                "accepted_at": accepted,
                "form": "10-Q",
                "target_accession_number": f"0001784851-2{index + 2}-000001",
                "price_collection_candidate": True,
                "selection_uses_return_values": False,
            }
        )
        base_rows.append(
            {
                "market_event_id": f"market-{index}",
                "validation_event_id": f"event-{index}",
                "cik": "0001784851",
                "issuer_price_candidate_events": 3,
                "identity_anchor_count": 2,
                "identity_interval_status": "stable_security_with_symbol_or_exchange_transition",
                "event_security_identity_ready": False,
                "identity_assignment_method": "unresolved_identity_repair_required",
                "assigned_ticker": None,
                "assigned_exchange_mic": None,
                "assigned_provider_security_id": None,
                "assigned_provider_security_id_kind": None,
                "identity_repair_required": True,
                "price_collection_authorized": False,
                "return_outcome_read_authorized": False,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    base_reference = pd.DataFrame(
        [
            _reference("event-0", "XNYS"),
            _reference("event-2", "XNAS"),
        ]
    )
    middle_reference = pd.DataFrame([_reference("event-1", "XNYS")])
    return (
        pd.DataFrame(market_rows),
        pd.DataFrame(base_rows),
        base_reference,
        middle_reference,
    )


def _reference(event_id: str, exchange: str) -> dict[str, object]:
    return {
        "validation_event_id": event_id,
        "event_date_reference_candidate": True,
        "selected_ticker": "SHPW",
        "selected_exchange_mic": exchange,
        "selected_provider_security_id": "BBG00QLC97F1",
        "selected_provider_security_id_kind": "share_class_figi",
        "selection_uses_return_values": False,
    }
