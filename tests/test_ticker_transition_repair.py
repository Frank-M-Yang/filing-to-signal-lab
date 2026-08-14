"""Tests for same-security ticker-transition reconciliation."""

from __future__ import annotations

import pandas as pd

from sec_data.ticker_transition_repair import (
    TickerTransitionRepairConfig,
    calculate_ticker_transition_reconciliation,
)


def test_transition_repair_maps_new_ticker_to_original_security_target() -> None:
    config = TickerTransitionRepairConfig()
    mapping = pd.DataFrame(
        [
            {
                "original_price_target_id": "old-price",
                "repair_price_target_id": "new-price",
                "original_action_target_id": "old-action",
                "repair_action_target_id": "new-action",
                "security_id": "BBG001V0BLG6",
                "share_class_figi": "BBG001V0BLG6",
                "composite_figi": "BBG001HPDS98",
                "exchange_mic": "XNAS",
                "ticker_event_source_sha256": "event-source",
            }
        ]
    )
    original_bars = _bars("old-price", "OB", ["2025-06-06", "2025-06-09"], [2.8, 2.74])
    repair_bars = _bars("new-price", "TEAD", ["2025-06-10", "2025-06-11"], [2.80, 2.90])
    identity = pd.DataFrame(
        [
            _identity("reference_start"),
            _identity("reference_end"),
        ]
    )
    events = pd.DataFrame(
        {
            "action_target_id": ["new-action"],
            "event_date": ["2025-06-10"],
            "new_ticker": ["TEAD"],
        }
    )
    original_audit = pd.DataFrame(
        {
            "action_target_id": ["old-action"],
            "target_id": ["old-target"],
            "corporate_actions_ready": [True],
        }
    )
    repair_audit = pd.DataFrame(
        {
            "action_target_id": ["new-action"],
            "corporate_actions_ready": [True],
        }
    )
    actions = pd.DataFrame(
        columns=[
            "action_target_id",
            "price_target_id",
            "target_id",
            "security_id",
            "ticker",
            "exchange_mic",
            "provider_event_id",
            "effective_at",
            "action_type",
            "cash_amount",
            "currency",
            "split_from",
            "split_to",
            "source_snapshot_sha256",
        ]
    )

    result = calculate_ticker_transition_reconciliation(
        mapping,
        original_bars,
        repair_bars,
        identity,
        events,
        original_audit,
        repair_audit,
        actions,
        _sessions(),
        config,
    )

    assert result.report["transition_gate"] is True
    assert result.price_additions["price_target_id"].eq("old-price").all()
    assert result.price_additions["ticker_at_session"].eq("TEAD").all()
    assert bool(result.identity_overrides.iloc[0]["historical_symbol_interval_ready"])
    assert result.evidence["passed"].all()


def _bars(target_id: str, ticker: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_target_id": [target_id] * len(dates),
            "ticker": [ticker] * len(dates),
            "session_date": dates,
            "open_unadjusted": closes,
            "close_unadjusted": closes,
            "source_sha256": [f"source-{ticker}"] * len(dates),
        }
    )


def _identity(query_kind: str) -> dict[str, object]:
    return {
        "action_target_id": "new-action",
        "query_kind": query_kind,
        "ticker": "TEAD",
        "provider_security_id": "BBG001V0BLG6",
        "share_class_figi": "BBG001V0BLG6",
        "composite_figi": "BBG001HPDS98",
        "exchange_mic": "XNAS",
        "security_type": "CS",
    }


def _sessions() -> pd.DataFrame:
    dates = pd.bdate_range("2025-06-01", "2025-06-20")
    return pd.DataFrame(
        {
            "exchange_mic": ["XNAS"] * len(dates),
            "session_date": dates.strftime("%Y-%m-%d"),
        }
    )
