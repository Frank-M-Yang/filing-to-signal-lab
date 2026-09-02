"""Tests for strictly pre-event liquidity eligibility."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.validation_liquidity import (
    ValidationLiquidityConfig,
    ValidationLiquidityError,
    _calculate_evidence_resolution,
    calculate_pre_event_liquidity_audit,
    calculate_pre_event_liquidity_targets,
)


def test_liquidity_target_sessions_are_strictly_before_acceptance() -> None:
    components = pd.DataFrame([_component()])
    security_targets = pd.DataFrame([_security_target()])
    sessions = _sessions(25)
    config = ValidationLiquidityConfig(lookback_sessions=20)

    targets, windows, event_sessions = calculate_pre_event_liquidity_targets(
        components,
        security_targets,
        sessions,
        config=config,
    )

    assert len(targets) == 1
    assert len(event_sessions) == 20
    assert windows.loc[0, "liquidity_window_end"] == "2025-01-24"
    assert (
        pd.to_datetime(event_sessions["regular_close_at"], utc=True)
        .lt(pd.Timestamp("2025-01-27T20:15:00Z"))
        .all()
    )
    assert not windows["selection_uses_return_values"].any()


def test_liquidity_audit_classifies_eligible_and_illiquid_without_returns() -> None:
    components = pd.DataFrame([_component()])
    targets, windows, event_sessions = calculate_pre_event_liquidity_targets(
        components,
        pd.DataFrame([_security_target()]),
        _sessions(25),
    )
    target_id = str(targets.loc[0, "price_target_id"])
    bars = event_sessions[["session_date"]].copy()
    bars["price_target_id"] = target_id
    bars["close_unadjusted"] = 10.0
    bars["volume"] = 200_000.0
    bars["transactions"] = 100
    completions = pd.DataFrame([{"price_target_id": target_id}])
    empty_rejections = pd.DataFrame(columns=["price_target_id"])

    eligible, _, _ = calculate_pre_event_liquidity_audit(
        windows,
        event_sessions,
        targets,
        bars,
        empty_rejections,
        completions,
    )
    assert bool(eligible.loc[0, "liquidity_eligible"])
    assert eligible.loc[0, "liquidity_status"] == "eligible"
    assert not eligible["returns_computed"].any()

    illiquid_bars = bars.iloc[:10].copy()
    illiquid_bars["volume"] = 100.0
    illiquid, _, _ = calculate_pre_event_liquidity_audit(
        windows,
        event_sessions,
        targets,
        illiquid_bars,
        empty_rejections,
        completions,
    )
    assert not bool(illiquid.loc[0, "liquidity_eligible"])
    assert illiquid.loc[0, "liquidity_status"] == "insufficient_print_coverage_and_capacity"


def test_secondary_liquidity_evidence_can_complete_a_missing_primary_query() -> None:
    targets, windows, event_sessions = calculate_pre_event_liquidity_targets(
        pd.DataFrame([_component()]),
        pd.DataFrame([_security_target()]),
        _sessions(25),
    )
    target_id = str(targets.loc[0, "price_target_id"])
    bars = event_sessions[["session_date"]].copy()
    bars["price_target_id"] = target_id
    bars["close_unadjusted"] = 10.0
    bars["volume"] = 200_000.0
    bars["transactions"] = None
    resolution = pd.DataFrame(
        [
            {
                "security_event_component_id": "component-1",
                "pre_event_liquidity_evidence_complete": True,
                "unresolved_history_sessions": 0,
            }
        ]
    )

    audit, _, _ = calculate_pre_event_liquidity_audit(
        windows,
        event_sessions,
        targets,
        bars,
        pd.DataFrame(columns=["price_target_id"]),
        pd.DataFrame(columns=["price_target_id"]),
        evidence_resolution=resolution,
    )

    assert not bool(audit.loc[0, "primary_price_query_complete"])
    assert bool(audit.loc[0, "price_query_complete"])
    assert bool(audit.loc[0, "liquidity_eligible"])


def test_no_primary_history_routes_every_session_to_secondary_evidence() -> None:
    event_sessions = pd.DataFrame(
        {
            "security_event_component_id": ["component-1", "component-1"],
            "price_target_id": ["price-1", "price-1"],
            "session_date": ["2025-01-02", "2025-01-03"],
        }
    )

    resolution = _calculate_evidence_resolution(
        event_sessions,
        pd.DataFrame(columns=["price_target_id", "session_date"]),
        pd.DataFrame(),
    )

    assert not bool(resolution.loc[0, "pre_event_liquidity_evidence_complete"])
    assert resolution.loc[0, "unresolved_history_sessions"] == 2
    assert resolution.loc[0, "primary_history_left_censored_sessions"] == 2


def test_liquidity_targets_reject_outcome_informed_components() -> None:
    components = pd.DataFrame([_component()])
    components.loc[0, "outcome_values_loaded"] = True

    with pytest.raises(ValidationLiquidityError, match="opened outcomes"):
        calculate_pre_event_liquidity_targets(
            components,
            pd.DataFrame([_security_target()]),
            _sessions(25),
        )


def _component() -> dict[str, object]:
    return {
        "security_event_component_id": "component-1",
        "market_event_id": "market-1",
        "validation_event_id": "event-1",
        "cik": "0000000001",
        "accepted_at": "2025-01-27T20:15:00+00:00",
        "ticker": "TEST",
        "exchange_mic": "XNAS",
        "identity_key": "ID-TEST",
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }


def _security_target() -> dict[str, object]:
    return {
        "cik": "0000000001",
        "selected_ticker": "TEST",
        "selected_exchange_mic": "XNAS",
        "security_identity_key": "ID-TEST",
        "security_identity_key_kind": "share_class_figi",
        "selected_provider_security_id": "ID-TEST",
        "selected_provider_security_id_kind": "share_class_figi",
    }


def _sessions(count: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-12-23", periods=count)
    return pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": dates.strftime("%Y-%m-%d"),
            "regular_open_at": dates.strftime("%Y-%m-%dT14:30:00+00:00"),
            "regular_close_at": dates.strftime("%Y-%m-%dT21:00:00+00:00"),
        }
    )
