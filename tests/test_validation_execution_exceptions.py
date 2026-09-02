"""Tests for non-imputing validation execution exceptions."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_execution_exceptions import (
    calculate_validation_execution_exceptions,
)


def test_execution_exceptions_distinguish_history_no_print_and_ended_trading() -> None:
    endpoints = pd.DataFrame(
        [
            _endpoint("history", "A", "2024-01-02", "2024-01-10", False, True),
            _endpoint("no-print", "B", "2024-01-03", "2024-01-10", False, True),
            _endpoint("ended", "C", "2024-01-02", "2024-01-10", True, False),
            _endpoint("ready", "D", "2024-01-02", "2024-01-10", True, True),
        ]
    )
    bars = pd.DataFrame(
        [
            _bar("history", "2024-01-05"),
            _bar("history", "2024-01-10"),
            _bar("no-print", "2024-01-02"),
            _bar("no-print", "2024-01-04"),
            _bar("no-print", "2024-01-10"),
            _bar("ended", "2024-01-02"),
            _bar("ended", "2024-01-05"),
            _bar("ready", "2024-01-02"),
            _bar("ready", "2024-01-10"),
        ]
    )

    result = calculate_validation_execution_exceptions(endpoints, bars).set_index("price_target_id")

    assert (
        result.loc["history", "formal_price_status"]
        == "unresolved_external_history_or_listing_start"
    )
    assert result.loc["no-print", "formal_price_status"] == "non_executable_entry_under_frozen_rule"
    assert result.loc["ended", "formal_price_status"] == "unresolved_terminal_or_sparse_exit"
    assert result.loc["ready", "formal_price_status"] == "price_ready"
    assert not result["event_removed"].any()
    assert not result["price_imputed"].any()
    assert not result["returns_computed"].any()


def _endpoint(
    target_id: str,
    ticker: str,
    entry: str,
    exit_date: str,
    entry_ready: bool,
    exit_ready: bool,
) -> dict[str, object]:
    return {
        "security_event_component_id": f"component-{target_id}",
        "market_event_id": f"market-{target_id}",
        "validation_event_id": f"event-{target_id}",
        "cik": "0000000001",
        "ticker": ticker,
        "exchange_mic": "XNAS",
        "price_target_id": target_id,
        "formal_entry_session_date": entry,
        "formal_exit_session_date": exit_date,
        "entry_price_date_observed": entry_ready,
        "exit_price_date_observed": exit_ready,
        "required_endpoint_dates_ready": entry_ready and exit_ready,
    }


def _bar(target_id: str, session_date: str) -> dict[str, str]:
    return {"price_target_id": target_id, "session_date": session_date}
