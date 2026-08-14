"""Bias semantics, portfolio construction, and calibrated acceptance tests."""

import pandas as pd

from backtest import (
    BacktestConfig,
    BiasConfig,
    compare_standard_scenarios,
    run_backtest,
)
from data_generator import SimulatedDataSource


def test_execution_rule_uses_second_trading_day_open() -> None:
    data = SimulatedDataSource().load()
    corrected = run_backtest(data, "sue", biases=BiasConfig())
    biased = run_backtest(data, "sue", biases=BiasConfig(announcement_timing=True))
    calendar = pd.DatetimeIndex(data.benchmark["date"])
    corrected_sample = corrected.events.iloc[0]
    biased_sample = biased.events.iloc[0]

    announce_position = calendar.get_loc(corrected_sample["announce_date"])
    assert corrected_sample["entry_date"] == calendar[announce_position + 2]
    assert biased_sample["entry_date"] == biased_sample["announce_date"]


def test_survivorship_bias_removes_eventual_delistings() -> None:
    data = SimulatedDataSource().load()
    honest = run_backtest(data, "sue", biases=BiasConfig())
    survivor_only = run_backtest(data, "sue", biases=BiasConfig(survivorship=True))

    assert (~honest.events["is_listed"]).any()
    assert survivor_only.events["is_listed"].all()
    assert len(survivor_only.events) < len(honest.events)


def test_lookahead_bias_uses_restated_values_only_when_enabled() -> None:
    data = SimulatedDataSource().load()
    honest = run_backtest(data, "sue", biases=BiasConfig())
    leaked = run_backtest(data, "sue", biases=BiasConfig(lookahead=True))

    assert honest.events["signal_value_source"].eq("eps").all()
    assert leaked.events["signal_value_source"].eq("eps_restated").all()


def test_default_calibration_exposes_material_bias_gap() -> None:
    scenarios = compare_standard_scenarios(SimulatedDataSource().load(), "sue")
    biased = scenarios["原始回测（全部偏差）"].metrics.annual_return
    corrected = scenarios["修正全部偏差后"].metrics.annual_return

    assert 0.10 <= biased <= 0.20
    assert corrected < 0.03
    assert biased - corrected >= 0.08


def test_portfolio_has_both_legs_and_five_quantiles() -> None:
    result = run_backtest(
        SimulatedDataSource().load(),
        "gross_margin_change",
        BacktestConfig(holding_days=20, quantile_fraction=0.20),
        BiasConfig(),
    )

    assert (result.cohort_returns[["long_count", "short_count"]] > 0).all().all()
    assert result.quantile_returns["quantile"].tolist() == [1, 2, 3, 4, 5]
    assert result.equity_curve.iloc[0]["equity"] == 1.0
