"""Point-in-time financial-signal calculations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

SignalName = Literal["sue", "gross_margin_change"]

SIGNAL_LABELS: dict[SignalName, str] = {
    "sue": "盈余惊喜（SUE）",
    "gross_margin_change": "毛利率同比变化",
}


def calculate_signal(
    fundamentals: pd.DataFrame,
    signal_name: SignalName,
    *,
    use_restated: bool = False,
) -> pd.DataFrame:
    """Calculate one signal without using observations after the current row.

    Setting ``use_restated`` is intentionally unsafe for historical research: it
    selects values that became known after the announcement and exists solely to
    demonstrate look-ahead bias.
    """
    required = {"ticker", "report_date", "announce_date"}
    missing = required.difference(fundamentals.columns)
    if missing:
        raise ValueError(f"Missing signal input columns: {sorted(missing)}")
    frame = fundamentals.copy().sort_values(["ticker", "report_date"])
    if signal_name == "sue":
        return _calculate_sue(frame, use_restated=use_restated)
    if signal_name == "gross_margin_change":
        return _calculate_margin_change(frame, use_restated=use_restated)
    raise ValueError(f"Unsupported signal: {signal_name}")


def _calculate_sue(frame: pd.DataFrame, *, use_restated: bool) -> pd.DataFrame:
    column = "eps_restated" if use_restated else "eps"
    if column not in frame:
        raise ValueError(f"Missing SUE input column: {column}")
    grouped = frame.groupby("ticker", sort=False)[column]
    expected_eps = grouped.shift(4)
    historical_std = grouped.transform(
        lambda values: values.shift(1).rolling(8, min_periods=8).std(ddof=1)
    )
    safe_std = historical_std.where(historical_std.abs() > 1e-10)
    frame["expected_eps"] = expected_eps
    frame["std_eps"] = safe_std
    frame["signal"] = (frame[column] - expected_eps) / safe_std
    frame["signal"] = frame["signal"].replace([np.inf, -np.inf], np.nan)
    frame["signal_value_source"] = column
    frame["signal_available_date"] = frame["announce_date"]
    return frame


def _calculate_margin_change(frame: pd.DataFrame, *, use_restated: bool) -> pd.DataFrame:
    column = "gross_margin_restated" if use_restated else "gross_margin"
    if column not in frame:
        raise ValueError(f"Missing gross-margin input column: {column}")
    prior_year = frame.groupby("ticker", sort=False)[column].shift(4)
    frame["prior_year_gross_margin"] = prior_year
    frame["signal"] = frame[column] - prior_year
    frame["signal_value_source"] = column
    frame["signal_available_date"] = frame["announce_date"]
    return frame
