"""Tests for outcome-blind validation return authorization."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_return_gate import calculate_validation_return_authorization


def test_return_gate_authorizes_only_liquidity_complete_cases() -> None:
    liquidity = pd.DataFrame(
        [
            _liquidity("eligible", eligible=True, evidence=True),
            _liquidity("illiquid", eligible=False, evidence=True),
            _liquidity("unresolved", eligible=False, evidence=False),
        ]
    )
    outcomes = pd.DataFrame([_outcome(value) for value in liquidity.security_event_component_id])
    benchmark = pd.DataFrame(
        [
            _benchmark(component, date)
            for component in liquidity.security_event_component_id
            for date in ("2025-01-02", "2025-01-31")
        ]
    )

    universe, authorized, excluded = calculate_validation_return_authorization(
        liquidity, outcomes, benchmark
    )

    assert authorized["security_event_component_id"].tolist() == ["eligible"]
    assert len(excluded) == 2
    assert (
        universe.set_index("security_event_component_id").loc["illiquid", "return_gate_status"]
        == "excluded_pre_event_liquidity"
    )
    assert (
        universe.set_index("security_event_component_id").loc["unresolved", "return_gate_status"]
        == "excluded_liquidity_evidence_incomplete"
    )
    assert not authorized["returns_computed"].any()


def _liquidity(component: str, *, eligible: bool, evidence: bool) -> dict[str, object]:
    return {
        "security_event_component_id": component,
        "market_event_id": f"market-{component}",
        "validation_event_id": f"event-{component}",
        "cik": "0000000001",
        "accepted_at": "2025-01-01T21:00:00+00:00",
        "ticker": "TEST",
        "liquidity_eligible": eligible,
        "liquidity_status": "eligible" if eligible else "excluded",
        "pre_event_liquidity_evidence_complete": evidence,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
    }


def _outcome(component: str) -> dict[str, object]:
    return {
        "security_event_component_id": component,
        "resolved_market_outcome_ready": True,
        "resolved_outcome_status": "formal_price_outcome_ready",
        "returns_computed": False,
    }


def _benchmark(component: str, session_date: str) -> dict[str, object]:
    return {
        "target_id": component,
        "required_session_date": session_date,
        "reconciled_benchmark_price_ready": True,
        "returns_computed": False,
    }
