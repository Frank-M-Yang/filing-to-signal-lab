"""Tests for source-tiered validation price readiness."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_price_reconciliation import (
    calculate_validation_price_reconciliation,
)


def test_price_reconciliation_resolves_declared_history_gaps_only() -> None:
    events = pd.DataFrame(
        [
            _event("primary", True, True, "price_ready"),
            _event(
                "fallback",
                False,
                False,
                "unresolved_external_history_or_listing_start",
            ),
            _event("pending", False, True, "non_executable_entry_under_frozen_rule"),
        ]
    )
    fallback_targets = pd.DataFrame(
        [
            {
                "fallback_target_id": "fb-one",
                "consumer_kind": "issuer_formal_entry",
                "consumer_ids": "fallback",
                "ticker": "ONE",
            },
            {
                "fallback_target_id": "fb-one-exit",
                "consumer_kind": "issuer_formal_exit",
                "consumer_ids": "fallback",
                "ticker": "ONE",
            },
            {
                "fallback_target_id": "fb-spy",
                "consumer_kind": "benchmark_required_session",
                "consumer_ids": "primary|fallback",
                "ticker": "SPY",
            },
        ]
    )
    fallback_prices = pd.DataFrame(
        [
            _fallback_price("fb-one", "2024-08-16"),
            _fallback_price("fb-one-exit", "2024-09-16"),
            _fallback_price("fb-spy", "2024-08-16"),
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "target_id": "primary",
                "required_session_date": "2024-08-16",
                "benchmark_price_ready": False,
                "open_unadjusted": float("nan"),
                "close_unadjusted": float("nan"),
            },
            {
                "target_id": "fallback",
                "required_session_date": "2024-08-16",
                "benchmark_price_ready": False,
                "open_unadjusted": float("nan"),
                "close_unadjusted": float("nan"),
            },
        ]
    )

    reconciled, benchmark_result = calculate_validation_price_reconciliation(
        events, fallback_targets, fallback_prices, benchmark
    )
    reconciled = reconciled.set_index("security_event_component_id")

    assert reconciled.loc["primary", "reconciled_formal_price_ready"]
    assert reconciled.loc["fallback", "reconciled_formal_price_ready"]
    assert not reconciled.loc["pending", "reconciled_formal_price_ready"]
    assert reconciled.loc["fallback", "reconciled_price_source_tier"] == "secondary_yahoo"
    assert benchmark_result["reconciled_benchmark_price_ready"].all()
    assert benchmark_result["open_unadjusted"].eq(10.0).all()
    assert benchmark_result["close_unadjusted"].eq(11.0).all()
    assert not reconciled["prices_imputed"].any()


def _event(
    component_id: str, entry_ready: bool, exit_ready: bool, status: str
) -> dict[str, object]:
    return {
        "security_event_component_id": component_id,
        "entry_price_date_observed": entry_ready,
        "exit_price_date_observed": exit_ready,
        "required_endpoint_dates_ready": entry_ready and exit_ready,
        "formal_price_status": status,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
    }


def _fallback_price(target_id: str, session_date: str) -> dict[str, object]:
    return {
        "fallback_target_id": target_id,
        "session_date": session_date,
        "open_unadjusted": 10.0,
        "close_unadjusted": 11.0,
        "source_provider": "Yahoo Chart API",
        "source_sha256": "a" * 64,
        "adjusted_close_used": False,
    }
