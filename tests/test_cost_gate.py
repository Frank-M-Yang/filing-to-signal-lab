"""Execution-cost and liquidity gate tests."""

import pandas as pd
import pytest

from cost_gate import (
    CostGateConfig,
    CostGateError,
    apply_net_edge_hurdle,
    evaluate_cost_gate,
    summarize_cost_gate,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # A: liquid, tight spread, priced above the floor -> tradable.
            # B: priced below the floor.
            # C: spread wider than the cap.
            # D: too illiquid; a reference position breaches the participation cap.
            "candidate_id": ["A", "B", "C", "D"],
            "price": [20.0, 3.0, 20.0, 20.0],
            "relative_spread": [0.005, 0.005, 0.05, 0.005],
            "median_dollar_volume_usd": [5_000_000.0, 5_000_000.0, 5_000_000.0, 500_000.0],
        }
    )


def test_config_floor_and_validation() -> None:
    config = CostGateConfig()
    assert config.minimum_median_dollar_volume_usd == pytest.approx(1_000_000.0)
    config.validate()
    with pytest.raises(CostGateError):
        CostGateConfig(max_participation=0.0).validate()
    with pytest.raises(CostGateError):
        CostGateConfig(max_relative_spread=1.0).validate()
    with pytest.raises(CostGateError):
        CostGateConfig(commission_bps=-1.0).validate()


def test_evaluate_costs_and_tradability() -> None:
    scored = evaluate_cost_gate(_candidates()).set_index("candidate_id")

    # A: half-spread 25bps + commission 1bps + slippage 2bps = 28bps one way.
    assert scored.loc["A", "one_way_cost_bps"] == pytest.approx(28.0)
    assert scored.loc["A", "round_trip_cost_bps"] == pytest.approx(56.0)
    assert scored.loc["A", "net_edge_hurdle_bps"] == pytest.approx(56.0)
    assert scored.loc["A", "participation"] == pytest.approx(0.002)
    assert bool(scored.loc["A", "controllable"]) is True
    assert bool(scored.loc["A", "tradable"]) is True
    assert scored.loc["A", "exclusion_reason"] == "tradable"

    assert scored.loc["B", "exclusion_reason"] == "excluded_min_price"
    assert scored.loc["C", "exclusion_reason"] == "excluded_wide_spread"
    assert scored.loc["D", "exclusion_reason"] == "excluded_illiquid_uncontrollable"
    assert bool(scored.loc["D", "controllable"]) is False
    assert not bool(scored.loc[["B", "C", "D"], "tradable"].any())


def test_net_edge_hurdle_requires_tradable_and_positive() -> None:
    scored = evaluate_cost_gate(_candidates())
    # A hurdle is 56bps: 80bps clears, 40bps does not; D is untradable regardless.
    scored["gross_edge_bps"] = scored["candidate_id"].map(
        {"A": 80.0, "B": 80.0, "C": 80.0, "D": 1_000.0}
    )
    graded = apply_net_edge_hurdle(scored).set_index("candidate_id")
    assert graded.loc["A", "net_edge_bps"] == pytest.approx(24.0)
    assert bool(graded.loc["A", "clears_hurdle"]) is True
    assert bool(graded.loc["D", "clears_hurdle"]) is False  # huge edge, but untradable

    scored.loc[scored["candidate_id"] == "A", "gross_edge_bps"] = 40.0
    graded_low = apply_net_edge_hurdle(scored).set_index("candidate_id")
    assert graded_low.loc["A", "net_edge_bps"] == pytest.approx(-16.0)
    assert bool(graded_low.loc["A", "clears_hurdle"]) is False


def test_summary_counts_only_tradable_costs() -> None:
    summary = summarize_cost_gate(evaluate_cost_gate(_candidates()))
    assert summary["candidates"] == 4
    assert summary["tradable"] == 1
    assert summary["excluded"] == 3
    assert summary["median_round_trip_cost_bps"] == pytest.approx(56.0)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(CostGateError):
        evaluate_cost_gate(pd.DataFrame({"candidate_id": ["A"], "price": [10.0]}))
    duplicate = pd.DataFrame(
        {
            "candidate_id": ["A", "A"],
            "price": [10.0, 10.0],
            "relative_spread": [0.01, 0.01],
            "median_dollar_volume_usd": [2_000_000.0, 2_000_000.0],
        }
    )
    with pytest.raises(CostGateError):
        evaluate_cost_gate(duplicate)
    non_finite = _candidates()
    non_finite.loc[0, "price"] = float("nan")
    with pytest.raises(CostGateError):
        evaluate_cost_gate(non_finite)
