"""Financial signal timing and formula tests."""

import numpy as np
import pandas as pd

from signals import calculate_signal


def _fundamentals() -> pd.DataFrame:
    report_dates = pd.date_range("2020-03-31", periods=12, freq=pd.offsets.QuarterEnd())
    eps = np.array([1.0, 1.1, 0.9, 1.2, 1.05, 1.2, 1.0, 1.3, 1.4, 1.5, 1.2, 1.6])
    margin = np.array([0.30, 0.31, 0.29, 0.32, 0.31, 0.32, 0.30, 0.33, 0.34, 0.35, 0.32, 0.36])
    return pd.DataFrame(
        {
            "ticker": "TEST",
            "report_date": report_dates,
            "announce_date": report_dates + pd.offsets.BusinessDay(30),
            "eps": eps,
            "eps_restated": eps + 0.25,
            "gross_margin": margin,
            "gross_margin_restated": margin + np.linspace(0.0, 0.02, len(margin)),
        }
    )


def test_sue_uses_lagged_year_and_only_prior_eight_quarters() -> None:
    inputs = _fundamentals()
    output = calculate_signal(inputs, "sue")
    row = output.iloc[8]
    expected_std = inputs["eps"].iloc[:8].std(ddof=1)

    assert row["expected_eps"] == inputs["eps"].iloc[4]
    assert row["std_eps"] == expected_std
    assert row["signal"] == (inputs["eps"].iloc[8] - inputs["eps"].iloc[4]) / expected_std


def test_future_eps_change_does_not_change_past_sue() -> None:
    inputs = _fundamentals()
    before = calculate_signal(inputs, "sue").loc[8, "signal"]
    inputs.loc[11, "eps"] = 1000.0
    after = calculate_signal(inputs, "sue").loc[8, "signal"]

    assert before == after


def test_restatement_switch_selects_explicit_source_column() -> None:
    inputs = _fundamentals()
    original = calculate_signal(inputs, "gross_margin_change", use_restated=False)
    restated = calculate_signal(inputs, "gross_margin_change", use_restated=True)

    assert original["signal_value_source"].eq("gross_margin").all()
    assert restated["signal_value_source"].eq("gross_margin_restated").all()
    assert not original["signal"].equals(restated["signal"])
