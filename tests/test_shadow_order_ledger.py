from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shadow_order_ledger import (
    AWAITING_INNOVATION,
    QUALITY_QUARANTINED,
    QUALITY_REVIEW_REQUIRED,
    RESEARCH_SIGNAL_READY_POLICY_CLOSED,
    SHADOW_BUY_INTENT_READY,
    SHADOW_EXIT_OR_AVOID_INTENT_READY,
    SIGNAL_DEAD_ZONE,
    TERMINAL_REJECTED,
    ShadowOrderLedgerError,
    build_shadow_order_ledger,
    calculate_shadow_order_ledger,
)


def test_every_event_gets_a_blocked_decision_without_an_order() -> None:
    states = pd.DataFrame(
        [
            _state("clean", "clean"),
            _state("review", "review_required"),
            _state("quarantine", "quarantined"),
            _state("rejected", "clean", rejected=True),
        ]
    )

    tables = calculate_shadow_order_ledger(states, "2026-08-31T12:00:00Z")

    statuses = tables.decisions.set_index("prospective_event_id")["decision_status"]
    assert statuses.to_dict() == {
        "clean": AWAITING_INNOVATION,
        "review": QUALITY_REVIEW_REQUIRED,
        "quarantine": QUALITY_QUARANTINED,
        "rejected": TERMINAL_REJECTED,
    }
    assert len(tables.blocks) == len(states)
    assert tables.orders.empty
    assert tables.positions.empty
    assert not tables.decisions["eligible_for_shadow_order"].any()
    assert not tables.decisions["broker_submission_attempted"].any()
    assert tables.report["operational_gates_passed"] is True
    assert tables.report["paper_order_gate_passed"] is False


def test_clean_innovation_is_preserved_but_policy_remains_closed() -> None:
    states = pd.DataFrame([_state("ready", "clean", innovation_ready=True)])

    tables = calculate_shadow_order_ledger(states, "2026-08-31T12:00:00Z")

    decision = tables.decisions.iloc[0]
    assert decision["decision_status"] == RESEARCH_SIGNAL_READY_POLICY_CLOSED
    assert decision["revenue_innovation_pct"] == pytest.approx(0.08)
    assert decision["block_type"] == "policy_not_preregistered"
    assert decision["target_side"] is None
    assert decision["requested_quantity"] == 0
    assert tables.orders.empty


@pytest.mark.parametrize(
    ("innovation", "expected_status", "expected_action", "expected_weight"),
    [
        (0.02, SHADOW_BUY_INTENT_READY, "buy", 0.02),
        (-0.02, SHADOW_EXIT_OR_AVOID_INTENT_READY, "exit_or_avoid", 0.0),
        (0.005, SIGNAL_DEAD_ZONE, "no_trade", 0.0),
    ],
)
def test_preregistered_policy_maps_signal_without_creating_orders(
    innovation: float,
    expected_status: str,
    expected_action: str,
    expected_weight: float,
) -> None:
    state = _state("ready", "clean", innovation_ready=True)
    state["revenue_innovation_pct"] = innovation
    states = pd.DataFrame([state])
    policy_events = pd.DataFrame(
        [
            {
                "prospective_event_id": "ready",
                "shadow_policy_admitted": True,
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        ]
    )

    tables = calculate_shadow_order_ledger(
        states,
        "2026-08-31T12:00:00Z",
        _policy_report(),
        policy_events,
    )

    decision = tables.decisions.iloc[0]
    assert decision["decision_status"] == expected_status
    assert decision["signal_action"] == expected_action
    assert decision["target_weight"] == pytest.approx(expected_weight)
    assert tables.report["trading_policy_preregistered"] is True
    assert tables.orders.empty


@pytest.mark.parametrize("column", ["return_values_loaded", "live_trading_authorized"])
def test_outcome_and_live_authorization_firewalls(column: str) -> None:
    states = pd.DataFrame([_state("bad", "clean")])
    states.loc[0, column] = True

    with pytest.raises(ShadowOrderLedgerError):
        calculate_shadow_order_ledger(states, "2026-08-31T12:00:00Z")


def test_build_is_immutable_for_the_same_state_snapshot(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    pd.DataFrame([_state("event", "clean")]).to_parquet(
        state_root / "prospective_event_states.parquet", index=False
    )
    (state_root / "output_manifest.json").write_text("{}", encoding="utf-8")
    (state_root / "prospective_event_state_machine_report.json").write_text(
        '{"state_as_of":"2026-08-31T12:00:00+00:00"}', encoding="utf-8"
    )

    first = build_shadow_order_ledger(state_root, output_root=tmp_path / "ledger")
    second = build_shadow_order_ledger(state_root, output_root=tmp_path / "ledger")

    assert first.output_root == second.output_root
    decisions = pd.read_parquet(first.output_root / "shadow_order_decisions.parquet")
    assert len(decisions) == 1
    assert first.report["shadow_orders"] == 0


def _state(
    event_id: str,
    quality: str,
    *,
    rejected: bool = False,
    innovation_ready: bool = False,
) -> dict[str, object]:
    review = quality in {"review_required", "quarantined"}
    return {
        "prospective_event_id": event_id,
        "registration_order": {"clean": 1, "review": 2, "quarantine": 3}.get(event_id, 4),
        "cik": "320193",
        "ticker": "aapl",
        "target_period_end": "2026-09-30",
        "current_state": (
            "rejected"
            if rejected
            else "innovation_ready"
            if innovation_ready
            else "consensus_revision_ready"
        ),
        "state_entered_at": "2026-08-31T10:00:00Z",
        "state_effective_at": "2026-08-31T09:00:00Z",
        "rejected": rejected,
        "rejection_reason": "invalid_event" if rejected else None,
        "blocking_reason": "future_filing_not_observed",
        "innovation_ready": innovation_ready,
        "raw_innovation_computed": innovation_ready,
        "revenue_innovation_pct": 0.08 if innovation_ready else None,
        "consensus_quality_status": quality,
        "consensus_quality_reasons": "provider_structure" if review else "none",
        "downstream_admission": quality == "clean",
        "consensus_quality_review_required": review,
        "consensus_quality_quarantined": quality == "quarantined",
        "price_window_ready": False,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _policy_report() -> dict[str, object]:
    return {
        "policy_gate_passed": True,
        "shadow_intent_generation_authorized": True,
        "shadow_order_execution_authorized": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_fitting_performed": False,
        "broker_submission_authorized": False,
        "live_trading_authorized": False,
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
