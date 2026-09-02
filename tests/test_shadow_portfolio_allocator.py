from __future__ import annotations

import pandas as pd
import pytest

from shadow_portfolio_allocator import (
    ShadowPortfolioAllocatorError,
    calculate_shadow_portfolio_allocations,
)


def test_empty_intent_set_keeps_full_paper_cash() -> None:
    tables = calculate_shadow_portfolio_allocations(
        pd.DataFrame([_decision("one", 1, "no_trade", False)]),
        pd.DataFrame([_evidence("one", False)]),
        _policy_report(),
        "2026-08-31T21:00:00Z",
    )

    assert tables.order_candidates.empty
    assert tables.account_snapshot.iloc[0]["available_cash_usd"] == 100_000.0
    assert tables.report["allocation_status_counts"] == {"no_shadow_intent": 1}
    assert tables.gates["passed"].all()


def test_ready_buy_reserves_cash_and_creates_candidate_not_order() -> None:
    tables = calculate_shadow_portfolio_allocations(
        pd.DataFrame([_decision("one", 1, "buy", True)]),
        pd.DataFrame([_evidence("one", True)]),
        _policy_report(),
        "2026-08-31T21:00:00Z",
    )

    candidate = tables.order_candidates.iloc[0]
    assert candidate["requested_quantity"] == 20
    assert candidate["target_notional_usd"] == pytest.approx(2_000.0)
    assert candidate["estimated_entry_cash_usd"] == pytest.approx(2_001.5)
    assert candidate["candidate_status"] == "ready_not_submitted"
    assert not candidate["broker_submission_attempted"]
    assert tables.report["orders_submitted"] == 0
    assert tables.report["positions_opened"] == 0


def test_unready_market_evidence_blocks_buy() -> None:
    tables = calculate_shadow_portfolio_allocations(
        pd.DataFrame([_decision("one", 1, "buy", True)]),
        pd.DataFrame([_evidence("one", False)]),
        _policy_report(),
        "2026-08-31T21:00:00Z",
    )

    assert tables.order_candidates.empty
    assert tables.allocations.iloc[0]["allocation_status"] == "market_evidence_blocked"


def test_pre_entry_clock_violation_is_rejected() -> None:
    evidence = _evidence("one", True)
    evidence["planned_entry_at"] = "2026-08-31T20:00:00Z"

    with pytest.raises(ShadowPortfolioAllocatorError):
        calculate_shadow_portfolio_allocations(
            pd.DataFrame([_decision("one", 1, "buy", True)]),
            pd.DataFrame([evidence]),
            _policy_report(),
            "2026-08-31T21:00:00Z",
        )


def _decision(event_id: str, order: int, action: str, eligible: bool) -> dict[str, object]:
    return {
        "shadow_decision_id": f"decision-{event_id}",
        "prospective_event_id": event_id,
        "registration_order": order,
        "cik": str(order).zfill(10),
        "ticker": f"T{order}",
        "decision_recorded_at": "2026-08-31T21:00:00Z",
        "decision_status": "shadow_buy_intent_ready" if eligible else "awaiting_innovation",
        "signal_action": action,
        "eligible_for_shadow_intent": eligible,
        "target_weight": 0.02 if action == "buy" else 0.0,
        "broker_submission_attempted": False,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _evidence(event_id: str, ready: bool) -> dict[str, object]:
    return {
        "prospective_event_id": event_id,
        "market_evidence_status": "ready" if ready else "awaiting_sec_acceptance",
        "market_evidence_blockers": "none" if ready else "sec_acceptance_pending",
        "order_evidence_ready": ready,
        "planned_entry_at": "2026-09-01T13:30:00Z" if ready else None,
        "planned_exit_at": "2026-09-29T20:00:00Z" if ready else None,
        "sizing_reference_price": 100.0 if ready else None,
        "sizing_reference_observed_at": "2026-08-31T20:30:00Z" if ready else None,
        "round_trip_cost_bps": 15.0 if ready else None,
        "cost_evidence_observed_at": "2026-08-31T20:30:00Z" if ready else None,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _policy_report() -> dict[str, object]:
    return {
        "policy_gate_passed": True,
        "config": {
            "signal_name": "raw_revenue_market_innovation_pct",
            "buy_threshold": 0.01,
            "exit_or_avoid_threshold": -0.01,
            "target_event_weight": 0.02,
            "maximum_issuer_weight": 0.02,
            "maximum_gross_weight": 0.20,
            "maximum_concurrent_positions": 10,
            "cash_only": True,
            "short_selling_allowed": False,
            "leverage_allowed": False,
            "entry_session_offset": 2,
            "holding_sessions": 20,
            "entry_price_type": "unadjusted_open",
            "exit_price_type": "unadjusted_close",
            "order_simulation_type": "market_on_open",
            "cost_model": "preacceptance_corwin_schultz_plus_fixed_slippage",
            "fixed_slippage_bps": 5.0,
            "minimum_round_trip_cost_bps": 10.0,
            "maximum_round_trip_cost_bps": 100.0,
        },
    }
