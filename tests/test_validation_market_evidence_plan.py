"""Tests for formal validation market-evidence route freezing."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_market_evidence_plan import (
    calculate_validation_market_evidence_plan,
)


def test_market_evidence_plan_freezes_direct_and_session_fallback_routes() -> None:
    actions = pd.DataFrame([_action("one", "ONE", 1), _action("two", "TWO", 2)])
    delistings = pd.DataFrame([_delisting("one", 1), _delisting("two", 2)])
    benchmark = pd.DataFrame(
        [
            {
                "benchmark_target_id": "spy",
                "benchmark_ticker": "SPY",
                "adjusted": False,
                "benchmark_collection_authorized": True,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "ticker": "ONE",
                "exchange_mic": "XNAS",
                "currency_name": "usd",
                "share_class_figi": "FIGI-ONE",
                "composite_figi": "COMP-ONE",
                "source_sha256": "a" * 64,
            }
        ]
    )

    result = calculate_validation_market_evidence_plan(actions, delistings, benchmark, observations)

    routed = result.action_targets.set_index("selected_ticker")
    assert routed.loc["ONE", "ticker_change_query_mode"] == "composite_figi_ticker_events"
    assert (
        routed.loc["TWO", "ticker_change_query_mode"] == "official_session_grid_reference_fallback"
    )
    assert result.metrics["ticker_events_direct_targets"] == 1
    assert result.metrics["session_grid_fallback_targets"] == 1
    assert result.metrics["minimum_action_queries"] == 9
    assert not result.action_targets["selection_uses_action_coverage"].any()
    assert not result.action_targets["returns_computed"].any()


def _action(target_id: str, ticker: str, order: int) -> dict[str, object]:
    return {
        "price_target_id": f"price-{target_id}",
        "cik": f"{order:010d}",
        "selected_ticker": ticker,
        "selected_exchange_mic": "XNAS",
        "security_identity_key": f"FIGI-{ticker}",
        "security_identity_key_kind": "share_class_figi",
        "price_window_start": "2024-01-01",
        "price_window_end": "2025-01-01",
        "action_target_id": f"action-{target_id}",
        "required_query_kinds": ("reference_start|reference_end|ticker_events|splits|dividends"),
        "action_collection_authorized": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "action_acquisition_order": order,
    }


def _delisting(target_id: str, order: int) -> dict[str, object]:
    return {
        "delisting_target_id": f"delist-{target_id}",
        "cik": f"{order:010d}",
        "security_identity_key": f"FIGI-{target_id}",
        "delisting_collection_authorized": True,
        "selection_uses_later_listing_status": False,
        "outcome_values_loaded": False,
        "delisting_acquisition_order": order,
    }
