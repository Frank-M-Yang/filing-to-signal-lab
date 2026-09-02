"""Tests for outcome-blind formal validation execution planning."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.validation_execution_plan import (
    ValidationExecutionPlanConfig,
    ValidationExecutionPlanError,
    _build_delisting_targets,
    calculate_validation_execution_plan,
)


def test_execution_plan_resolves_timestamps_and_multiclass_components() -> None:
    market, identities, repair, sessions, benchmark = _inputs()

    tables = calculate_validation_execution_plan(market, identities, repair, sessions, benchmark)

    assert tables.report["formal_execution_windows_ready"] == 2
    assert tables.report["security_event_components"] == 3
    assert tables.report["distinct_security_spans"] == 3
    assert tables.report["price_collection_authorized"] is True
    assert tables.report["return_outcome_read_authorized"] is False
    first = tables.events.set_index("market_event_id").loc["event-1"]
    assert first["acceptance_session_relation"] == "accepted_before_regular_open"
    assert first["formal_entry_session_date"] == "2025-01-03"
    assert first["formal_entry_open_at"] == "2025-01-03T14:30:00+00:00"
    assert first["formal_entry_price_type"] == "unadjusted_open"
    second = tables.events.set_index("market_event_id").loc["event-2"]
    assert second["acceptance_session_relation"] == "accepted_after_regular_close"
    assert second["formal_entry_session_date"] == "2025-01-07"
    weights = tables.security_components.groupby("market_event_id")["component_weight"].sum()
    assert weights.to_dict() == {"event-1": 1.0, "event-2": 1.0}
    assert set(
        tables.security_components.loc[
            tables.security_components["market_event_id"].eq("event-2"), "ticker"
        ]
    ) == {"DUALA", "DUALB"}
    assert not tables.events["outcome_values_loaded"].any()


def test_execution_plan_rejects_loaded_outcomes() -> None:
    market, identities, repair, sessions, benchmark = _inputs()
    market["future_return"] = 0.1

    with pytest.raises(ValidationExecutionPlanError, match="contains outcomes"):
        calculate_validation_execution_plan(market, identities, repair, sessions, benchmark)


def test_delisting_plan_deduplicates_one_security_across_ticker_transition() -> None:
    prices = pd.DataFrame(
        [
            {
                "price_target_id": "old-span",
                "cik": "0000000001",
                "selected_ticker": "OLD",
                "selected_exchange_mic": "XNYS",
                "security_identity_key": "FIGI-ONE",
                "security_identity_key_kind": "share_class_figi",
                "price_window_end": "2023-01-31",
                "price_acquisition_order": 1,
            },
            {
                "price_target_id": "new-span",
                "cik": "0000000001",
                "selected_ticker": "NEW",
                "selected_exchange_mic": "XNAS",
                "security_identity_key": "FIGI-ONE",
                "security_identity_key_kind": "share_class_figi",
                "price_window_end": "2024-01-31",
                "price_acquisition_order": 2,
            },
        ]
    )

    targets = _build_delisting_targets(prices, ValidationExecutionPlanConfig())

    assert len(targets) == 1
    assert targets.loc[0, "selected_ticker"] == "NEW"
    assert targets.loc[0, "selected_exchange_mic"] == "XNAS"
    assert targets.loc[0, "event_ticker_aliases"] == "NEW|OLD"
    assert targets.loc[0, "event_exchange_mics"] == "XNAS|XNYS"
    assert targets.loc[0, "security_span_count"] == 2


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.DataFrame(
        [
            {
                "market_event_id": "event-1",
                "validation_event_id": "validation-1",
                "cik": "1",
                "accepted_at": "2025-01-02T13:00:00+00:00",
                "form": "10-Q",
                "target_accession_number": "accession-1",
                "price_collection_candidate": True,
                "selection_uses_return_values": False,
            },
            {
                "market_event_id": "event-2",
                "validation_event_id": "validation-2",
                "cik": "2",
                "accepted_at": "2025-01-03T22:00:00+00:00",
                "form": "10-Q",
                "target_accession_number": "accession-2",
                "price_collection_candidate": True,
                "selection_uses_return_values": False,
            },
        ]
    )
    identities = pd.DataFrame(
        [
            {
                "market_event_id": "event-1",
                "validation_event_id": "validation-1",
                "cik": "1",
                "event_security_identity_ready": True,
                "identity_assignment_method": "endpoint_interval_propagation",
                "assigned_ticker": "ONE",
                "assigned_exchange_mic": "XNAS",
                "assigned_provider_security_id": "FIGI-ONE",
                "assigned_provider_security_id_kind": "share_class_figi",
                "security_component_count": 1,
                "issuer_return_construction": "single_security",
                "identity_repair_status": "not_required",
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            },
            {
                "market_event_id": "event-2",
                "validation_event_id": "validation-2",
                "cik": "2",
                "event_security_identity_ready": True,
                "identity_assignment_method": ("identity_repair_equal_weighted_share_class_basket"),
                "assigned_ticker": "DUALA|DUALB",
                "assigned_exchange_mic": "XNAS",
                "assigned_provider_security_id": "FIGI-A|FIGI-B",
                "assigned_provider_security_id_kind": None,
                "security_component_count": 2,
                "issuer_return_construction": "equal_weighted_share_class_basket",
                "identity_repair_status": "stable_components_across_both_endpoints",
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            },
        ]
    )
    repair = pd.DataFrame(
        [
            {
                "cik": "2",
                "anchor_role": anchor,
                "ticker": ticker,
                "exchange_mic": "XNAS",
                "provider_security_id": identity,
                "provider_security_id_kind": "share_class_figi",
                "identity_key": identity,
                "identity_key_kind": "share_class_figi",
                "component_identity_ready": True,
                "evidence_source": "fixture",
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
            for anchor in ("first_candidate", "last_candidate")
            for ticker, identity in (("DUALA", "FIGI-A"), ("DUALB", "FIGI-B"))
        ]
    )
    dates = pd.bdate_range("2025-01-02", "2025-03-14")
    sessions = pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": dates.strftime("%Y-%m-%d"),
            "regular_open_at": [f"{value:%Y-%m-%d}T14:30:00+00:00" for value in dates],
            "regular_close_at": [f"{value:%Y-%m-%d}T21:00:00+00:00" for value in dates],
            "source_id": "nasdaq-fixture",
            "source_snapshot_sha256": "a" * 64,
            "schedule_basis_exchange_mic": "XNAS",
            "rule_version": "fixture-v1",
        }
    )
    benchmark = pd.DataFrame(
        [
            {
                "benchmark_role": "broad_market_total_return_proxy",
                "benchmark_ticker": "SPY",
                "selection_uses_return_values": False,
            }
        ]
    )
    return market, identities, repair, sessions, benchmark
