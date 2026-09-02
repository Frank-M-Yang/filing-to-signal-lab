"""Tests for secondary pre-event liquidity gap evidence."""

from __future__ import annotations

import json

import pandas as pd

from sec_data.validation_liquidity_fallback import (
    calculate_validation_liquidity_fallback_targets,
    parse_yahoo_liquidity_bars,
)


def test_fallback_targets_only_unique_primary_gaps() -> None:
    sessions = pd.DataFrame(
        [
            _session("component-1", "2024-08-26"),
            _session("component-1", "2024-08-27"),
            _session("component-2", "2024-08-27"),
        ]
    )
    bars = pd.DataFrame([{"price_target_id": "primary", "session_date": "2024-08-27"}])

    targets, gaps = calculate_validation_liquidity_fallback_targets(sessions, bars)

    assert len(targets) == 1
    assert targets.loc[0, "required_gap_sessions"] == 1
    assert len(gaps) == 1
    assert gaps.loc[0, "session_date"] == "2024-08-26"
    assert not targets["selection_uses_return_values"].any()


def test_yahoo_liquidity_parser_uses_raw_quote_and_allows_no_print_dates() -> None:
    document = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "TEST",
                        "exchangeTimezoneName": "America/New_York",
                    },
                    "timestamp": [1724682600, 1724769000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [10.0, None],
                                "high": [11.0, None],
                                "low": [9.0, None],
                                "close": [10.5, None],
                                "volume": [200_000, None],
                            }
                        ],
                        "adjclose": [{"adjclose": [999.0, 999.0]}],
                    },
                }
            ],
        }
    }

    rows = parse_yahoo_liquidity_bars(
        json.dumps(document).encode(),
        expected_ticker="TEST",
        fallback_target_id="fallback",
        primary_price_target_id="primary",
        window_start="2024-08-26",
        window_end="2024-08-27",
        source_sha256="abc",
    )

    assert len(rows) == 1
    assert rows[0]["session_date"] == "2024-08-26"
    assert rows[0]["close_unadjusted"] == 10.5
    assert not rows[0]["adjusted_close_used"]


def _session(component: str, session_date: str) -> dict[str, object]:
    return {
        "security_event_component_id": component,
        "price_target_id": "primary",
        "ticker": "TEST",
        "session_date": session_date,
        "accepted_at": "2024-08-28T20:00:00+00:00",
    }
