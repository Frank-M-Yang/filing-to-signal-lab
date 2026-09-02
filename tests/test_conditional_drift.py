from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from conditional_drift import (
    ConditionalDriftConfig,
    calculate_conditional_drift_study,
    validate_conditional_drift_dataset,
)
from sec_data.filing_catalog import FilingCatalogError


def test_same_event_reaction_and_cost_labels_obey_clocks() -> None:
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
    features = _feature_store(accepted_at, entry_at, label_5d, label_20d)
    actions = _empty_actions(include_price_target=True)
    benchmark_actions = _empty_actions(include_price_target=False)

    result = calculate_conditional_drift_study(
        features,
        issuer_bars,
        benchmark_bars,
        actions,
        benchmark_actions,
        sessions,
    )

    event = result.event_dataset.iloc[0]
    assert bool(event["reaction_clock_valid"])
    assert event["reaction_reference_close_at"] <= accepted_at
    assert accepted_at <= event["reaction_feature_available_at"] < entry_at
    assert event["reaction_session_count"] == 1.0
    assert event["reaction_volume_ratio_20d"] == pytest.approx(1.0)
    assert bool(event["same_event_features_complete"])
    assert event["target_market_excess_return_5d"] == pytest.approx(label_5d)
    assert event["target_market_excess_return_20d"] == pytest.approx(label_20d)
    assert event["target_net_market_excess_return_20d_base"] == pytest.approx(label_20d - 0.0025)
    assert bool(event["target_ready_60d"])
    assert result.report["external_validation_metrics_computed"] is False
    assert result.report["portfolio_backtest_computed"] is False


def test_reaction_feature_at_entry_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "longitudinal_event_id": ["event-1"],
            "accepted_at": ["2024-01-02T21:30:00Z"],
            "execution_entry_at": ["2024-01-04T14:30:00Z"],
            "reaction_feature_available_at": ["2024-01-04T14:30:00Z"],
            "reaction_clock_valid": [True],
            "same_event_features_complete": [True],
            **{column: [1.0] for column in _all_feature_columns()},
            **{
                f"target_available_at_{horizon}d": ["2024-04-01T20:00:00Z"]
                for horizon in (5, 20, 60)
            },
            **{f"target_ready_{horizon}d": [True] for horizon in (5, 20, 60)},
        }
    )
    with pytest.raises(FilingCatalogError, match="between acceptance and entry"):
        validate_conditional_drift_dataset(frame)


def test_protocol_rejects_hidden_cost_change() -> None:
    config = ConditionalDriftConfig(
        cost_scenarios_bps=(("low", 0.0), ("base", 0.0), ("stress", 0.0))
    )
    with pytest.raises(ValueError, match="freezes 10/25/50"):
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


def _all_feature_columns() -> tuple[str, ...]:
    from conditional_drift import ALL_FEATURE_COLUMNS

    return ALL_FEATURE_COLUMNS
