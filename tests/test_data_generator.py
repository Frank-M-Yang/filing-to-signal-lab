"""Simulation contract and reproducibility tests."""

import pandas as pd

from data_generator import SimulatedDataSource, SimulationConfig, generate_simulated_data


def test_simulation_is_reproducible_for_fixed_seed() -> None:
    config = SimulationConfig(seed=7, n_stocks=20, n_quarters=20)

    first = generate_simulated_data(config)
    second = generate_simulated_data(config)

    pd.testing.assert_frame_equal(first.fundamentals, second.fundamentals)
    pd.testing.assert_frame_equal(first.prices, second.prices)
    pd.testing.assert_frame_equal(first.benchmark, second.benchmark)


def test_default_data_contains_restatements_and_delistings() -> None:
    data = SimulatedDataSource().load()
    fundamentals = data.fundamentals

    assert fundamentals["ticker"].nunique() == 50
    assert fundamentals["has_restatement"].any()
    assert (~fundamentals["is_listed"]).any()
    assert (
        fundamentals.loc[fundamentals["has_restatement"], "restatement_date"]
        > fundamentals.loc[fundamentals["has_restatement"], "announce_date"]
    ).all()
    assert (fundamentals["announce_date"] > fundamentals["report_date"]).all()
    assert not data.prices.duplicated(["date", "ticker"]).any()


def test_restatement_columns_equal_original_when_no_revision_occurred() -> None:
    fundamentals = SimulatedDataSource().load().fundamentals
    unchanged = fundamentals.loc[~fundamentals["has_restatement"]]

    assert unchanged["eps"].equals(unchanged["eps_restated"])
    assert unchanged["revenue"].equals(unchanged["revenue_restated"])
    assert unchanged["gross_margin"].equals(unchanged["gross_margin_restated"])
