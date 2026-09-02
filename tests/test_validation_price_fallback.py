"""Tests for frozen secondary-source price gaps."""

from __future__ import annotations

import json

import pandas as pd

from sec_data.validation_price_fallback import (
    calculate_validation_price_fallback_targets,
    parse_yahoo_chart_price,
)


def test_price_fallback_targets_freeze_both_missing_endpoints_and_benchmark() -> None:
    exceptions = pd.DataFrame(
        [
            _exception("one", "ONE", "alternate_history_or_listing_start_evidence_required"),
            _exception("two", "TWO", "non_executable_entry_under_frozen_rule"),
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "target_id": "event-one",
                "required_session_date": "2024-08-16",
                "benchmark_price_ready": False,
            },
            {
                "target_id": "event-two",
                "required_session_date": "2024-08-16",
                "benchmark_price_ready": False,
            },
        ]
    )

    targets = calculate_validation_price_fallback_targets(exceptions, benchmark)

    assert set(targets["ticker"]) == {"ONE", "TWO", "SPY"}
    assert len(targets) == 5
    issuer_kinds = set(targets.loc[targets["ticker"].ne("SPY"), "consumer_kind"])
    assert issuer_kinds == {"issuer_formal_entry", "issuer_formal_exit"}
    assert not targets["price_imputation_allowed"].any()
    assert not targets["selection_uses_return_values"].any()


def test_parse_yahoo_chart_uses_raw_quote_not_adjusted_close() -> None:
    content = json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": "ONE",
                            "exchangeTimezoneName": "America/New_York",
                        },
                        "timestamp": [1723815000],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0],
                                    "high": [12.0],
                                    "low": [9.0],
                                    "close": [11.0],
                                    "volume": [100.0],
                                }
                            ],
                            "adjclose": [{"adjclose": [1.0]}],
                        },
                    }
                ],
            }
        }
    ).encode()

    row = parse_yahoo_chart_price(
        content,
        expected_ticker="ONE",
        required_session_date="2024-08-16",
        fallback_target_id="target",
        source_sha256="a" * 64,
    )

    assert row["open_unadjusted"] == 10.0
    assert row["close_unadjusted"] == 11.0
    assert not row["adjusted_close_used"]


def _exception(target_id: str, ticker: str, route: str) -> dict[str, object]:
    return {
        "security_event_component_id": f"component-{target_id}",
        "price_target_id": f"price-{target_id}",
        "ticker": ticker,
        "formal_entry_session_date": "2024-08-16",
        "formal_exit_session_date": "2024-09-16",
        "entry_price_date_observed": False,
        "exit_price_date_observed": False,
        "execution_exception_route": route,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
    }
