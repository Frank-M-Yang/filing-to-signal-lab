"""Tests for formal SPY benchmark target translation."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_benchmark import calculate_validation_benchmark_inputs


def test_validation_benchmark_freezes_spy_and_all_component_dates() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "benchmark_target_id": "spy-target",
                "benchmark_role": "broad_market_total_return_proxy",
                "benchmark_ticker": "SPY",
                "price_window_start": "2024-01-01",
                "price_window_end": "2025-01-31",
                "adjusted": False,
                "benchmark_collection_authorized": True,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        ]
    )
    components = pd.DataFrame(
        [
            {
                "security_event_component_id": "one",
                "formal_entry_session_date": "2024-01-03",
                "formal_exit_session_date": "2024-02-01",
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            },
            {
                "security_event_component_id": "two",
                "formal_entry_session_date": "2024-03-04",
                "formal_exit_session_date": "2024-04-01",
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            },
        ]
    )

    target, dates = calculate_validation_benchmark_inputs(benchmark, components)

    assert target.loc[0, "ticker"] == "SPY"
    assert target.loc[0, "cik"] == "0000884394"
    assert not target.loc[0, "adjusted"]
    assert len(dates) == 4
    assert set(dates["target_id"]) == {"one", "two"}
