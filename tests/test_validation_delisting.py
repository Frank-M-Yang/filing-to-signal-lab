"""Tests for formal event-level listing outcomes."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_delisting import (
    calculate_validation_delisting_targets,
    calculate_validation_listing_outcomes,
    calculate_validation_outcome_repair_queue,
)


def test_validation_delisting_targets_preserve_outcome_blind_identity() -> None:
    source = pd.DataFrame([_delisting_target("ONE", 1)])

    targets = calculate_validation_delisting_targets(source)

    assert targets.loc[0, "outcome_target_id"] == "delist-ONE"
    assert targets.loc[0, "security_id"] == "FIGI-ONE"
    assert not targets.loc[0, "selection_uses_return_values"]
    assert not targets.loc[0, "returns_computed"]


def test_listing_outcome_is_compared_to_each_event_exit() -> None:
    targets = calculate_validation_delisting_targets(pd.DataFrame([_delisting_target("ONE", 1)]))
    observations = pd.DataFrame(
        [
            {
                "outcome_target_id": "delist-ONE",
                "query_active": False,
                "provider_active": False,
                "ticker": "ONE",
                "security_type": "CS",
                "exchange_mic": "XNAS",
                "provider_security_id": "FIGI-ONE",
                "delisted_utc": "2024-02-15T00:00:00Z",
                "last_updated_utc": "2024-02-15T00:00:00Z",
                "source_sha256": "a" * 64,
            }
        ]
    )
    components = pd.DataFrame(
        [
            _component("early", "2024-02-01"),
            _component("late", "2024-03-01"),
        ]
    )
    components.loc[1, "ticker"] = "ONE.NEW"
    components.loc[1, "exchange_mic"] = "XNYS"
    reconciled = pd.DataFrame(
        [
            {"security_event_component_id": "early", "reconciled_formal_price_ready": True},
            {"security_event_component_id": "late", "reconciled_formal_price_ready": False},
        ]
    )

    security, events = calculate_validation_listing_outcomes(
        targets,
        observations,
        pd.DataFrame(
            columns=[
                "outcome_target_id",
                "query_active",
                "source_occurrence",
                "reason",
                "source_sha256",
            ]
        ),
        pd.DataFrame(
            [
                {
                    "outcome_target_id": "delist-ONE",
                    "active_query_complete": True,
                    "inactive_query_complete": True,
                    "completed_at": "2026-01-01T00:00:00Z",
                }
            ]
        ),
        components,
        reconciled,
    )

    events = events.set_index("security_event_component_id")
    assert security.loc[0, "listing_outcome_status"] == "inactive_with_delisting_date"
    assert events.loc["early", "later_delisted_after_formal_exit"]
    assert events.loc["early", "delisting_window_ready"]
    assert events.loc["late", "terminal_value_required"]
    assert events.loc["late", "outcome_target_id"] == "delist-ONE"
    assert events.loc["late", "outcome_exception_status"] == "terminal_value_required"
    assert not events["returns_computed"].any()

    queue = calculate_validation_outcome_repair_queue(events.reset_index())
    assert len(queue) == 1
    assert queue.loc[0, "affected_event_count"] == 1
    assert queue.loc[0, "repair_priority"] == 1
    assert (
        queue.loc[0, "repair_action"]
        == "collect_primary_terminal_consideration_and_settlement_evidence"
    )
    assert not queue.loc[0, "repair_selection_uses_return_values"]


def test_observed_formal_exit_price_supersedes_unresolved_later_status() -> None:
    targets = calculate_validation_delisting_targets(pd.DataFrame([_delisting_target("ONE", 1)]))
    components = pd.DataFrame([_component("observed", "2024-02-01")])
    reconciled = pd.DataFrame(
        [
            {
                "security_event_component_id": "observed",
                "reconciled_formal_price_ready": True,
            }
        ]
    )

    _, events = calculate_validation_listing_outcomes(
        targets,
        pd.DataFrame(
            columns=[
                "outcome_target_id",
                "query_active",
                "provider_active",
                "ticker",
                "security_type",
                "exchange_mic",
                "provider_security_id",
                "delisted_utc",
                "last_updated_utc",
                "source_sha256",
            ]
        ),
        pd.DataFrame(
            columns=[
                "outcome_target_id",
                "query_active",
                "source_occurrence",
                "reason",
                "source_sha256",
            ]
        ),
        pd.DataFrame(
            [
                {
                    "outcome_target_id": "delist-ONE",
                    "active_query_complete": True,
                    "inactive_query_complete": True,
                    "completed_at": "2026-01-01T00:00:00Z",
                }
            ]
        ),
        components,
        reconciled,
    )

    assert not events.loc[0, "listing_outcome_ready"]
    assert events.loc[0, "delisting_window_ready"]
    assert events.loc[0, "market_outcome_ready"]
    assert events.loc[0, "outcome_exception_status"] == "listing_status_advisory_nonblocking"
    assert calculate_validation_outcome_repair_queue(events).empty


def _delisting_target(ticker: str, order: int) -> dict[str, object]:
    return {
        "delisting_target_id": f"delist-{ticker}",
        "delisting_acquisition_order": order,
        "cik": "0000000001",
        "selected_ticker": ticker,
        "selected_exchange_mic": "XNAS",
        "security_identity_key": f"FIGI-{ticker}",
        "security_identity_key_kind": "share_class_figi",
        "outcome_cutoff_date": "2026-08-14",
        "delisting_collection_authorized": True,
        "selection_uses_later_listing_status": False,
        "outcome_values_loaded": False,
    }


def _component(component_id: str, exit_date: str) -> dict[str, object]:
    return {
        "security_event_component_id": component_id,
        "cik": "0000000001",
        "ticker": "ONE",
        "exchange_mic": "XNAS",
        "identity_key": "FIGI-ONE",
        "formal_exit_session_date": exit_date,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }
