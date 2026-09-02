from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from conditional_drift_v2 import (
    ConditionalDriftV2Config,
    _corwin_schultz_spread,
    calculate_conditional_drift_v2,
)


def test_v2_decomposition_and_liquidity_cost_obey_pre_entry_clocks() -> None:
    dates = _business_dates(date(2024, 1, 2), 90)
    accepted_index = 20
    entry_index = 22
    closes = np.asarray([100.0 + index * 0.1 for index in range(len(dates))])
    issuer_bars = _bars(dates, closes, price_target_id="price-1")
    benchmark_bars = _bars(dates, np.full(len(dates), 100.0), price_target_id=None)
    sessions = _calendar(dates)
    entry_price = float(issuer_bars.loc[entry_index, "open_unadjusted"])
    label_5d = float(closes[entry_index + 4] / entry_price - 1.0)
    label_20d = float(closes[entry_index + 19] / entry_price - 1.0)
    accepted_at = pd.Timestamp(f"{dates[accepted_index].isoformat()} 22:00:00", tz="UTC")
    entry_at = pd.Timestamp(f"{dates[entry_index].isoformat()} 14:30:00", tz="UTC")

    result = calculate_conditional_drift_v2(
        _feature_store(accepted_at, entry_at, label_5d, label_20d),
        issuer_bars,
        benchmark_bars,
        _empty_actions(include_price_target=True),
        _empty_actions(include_price_target=False),
        sessions,
    )

    event = result.event_dataset.iloc[0]
    assert bool(event["v2_features_complete"])
    assert accepted_at < event["reaction_gap_available_at"] < entry_at
    assert event["liquidity_cost_available_at"] <= accepted_at
    assert event["liquidity_cost_low_bps"] <= event["liquidity_cost_base_bps"]
    assert event["liquidity_cost_base_bps"] <= event["liquidity_cost_stress_bps"]
    assert event["target_net_market_excess_return_20d_liquidity_base"] == pytest.approx(
        label_20d - event["liquidity_cost_base_bps"] / 10_000.0
    )
    assert result.report["external_validation_metrics_computed"] is False


def test_corwin_schultz_uses_official_overnight_adjustment() -> None:
    history = pd.DataFrame(
        {
            "high_unadjusted": [101.0, 111.0],
            "low_unadjusted": [99.0, 109.0],
            "close_unadjusted": [100.0, 110.0],
        }
    )
    observed = _corwin_schultz_spread(history)
    adjusted_high = [101.0, 102.0]
    adjusted_low = [99.0, 100.0]
    beta = (
        math.log(adjusted_high[0] / adjusted_low[0]) ** 2
        + math.log(adjusted_high[1] / adjusted_low[1]) ** 2
    )
    gamma = math.log(max(adjusted_high) / min(adjusted_low)) ** 2
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = max(
        (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denominator - math.sqrt(gamma / denominator),
        0.0,
    )
    expected = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    assert observed == pytest.approx(expected)


def test_v2_cost_calibration_is_frozen() -> None:
    config = ConditionalDriftV2Config(fixed_slippage_bps=0.0)
    with pytest.raises(ValueError, match="calibration is frozen"):
        config.validate()


def _feature_store(
    accepted_at: pd.Timestamp,
    entry_at: pd.Timestamp,
    label_5d: float,
    label_20d: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitudinal_event_id": ["event-1"],
            "cik": ["0000000001"],
            "accession_number": ["0000000001-24-000001"],
            "accepted_at": [accepted_at],
            "form": ["10-Q"],
            "ticker": ["AAA"],
            "security_id": ["security-1"],
            "price_target_id": ["price-1"],
            "execution_entry_at": [entry_at],
            "analysis_sample": [True],
            "evidence_role": ["legacy_observed_training_only"],
            "prior_outcome_observed": [True],
            "target_market_excess_return_5d": [label_5d],
            "target_ready_5d": [True],
            "target_available_at_5d": [entry_at + pd.Timedelta(days=7)],
            "target_market_excess_return_20d": [label_20d],
            "target_ready_20d": [True],
            "target_available_at_20d": [entry_at + pd.Timedelta(days=28)],
            "gm_level": [0.4],
            "form_is_10k": [0.0],
            "issuer_momentum_5d": [0.01],
            "issuer_momentum_20d": [0.02],
            "issuer_volatility_20d": [0.01],
            "issuer_volume_ratio_5_20": [1.0],
            "market_momentum_5d": [0.0],
            "market_momentum_20d": [0.0],
            "relative_momentum_20d": [0.02],
            "market_beta_20d": [1.0],
            "gm_delta_yoy": [0.01],
            "gm_innovation_raw": [0.02],
            "gm_innovation_std": [0.5],
        }
    )


def _bars(dates: list[date], closes: np.ndarray, price_target_id: str | None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session_date": [value.isoformat() for value in dates],
            "open_unadjusted": closes,
            "high_unadjusted": closes + 0.5,
            "low_unadjusted": closes - 0.5,
            "close_unadjusted": closes,
            "volume": np.full(len(dates), 1_000_000.0),
        }
    )
    if price_target_id is not None:
        frame.insert(0, "price_target_id", price_target_id)
    return frame


def _calendar(dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange_mic": ["XNYS"] * len(dates),
            "session_date": [value.isoformat() for value in dates],
            "regular_open_at": [f"{value.isoformat()}T14:30:00Z" for value in dates],
            "regular_close_at": [f"{value.isoformat()}T21:00:00Z" for value in dates],
            "is_early_close": [False] * len(dates),
        }
    )


def _empty_actions(*, include_price_target: bool) -> pd.DataFrame:
    columns = [
        "effective_at",
        "action_type",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
        "provider_event_id",
        "source_snapshot_sha256",
    ]
    if include_price_target:
        columns.insert(0, "price_target_id")
    return pd.DataFrame(columns=columns)


def _business_dates(start: date, count: int) -> list[date]:
    values: list[date] = []
    cursor = start
    while len(values) < count:
        if cursor.weekday() < 5:
            values.append(cursor)
        cursor += timedelta(days=1)
    return values
