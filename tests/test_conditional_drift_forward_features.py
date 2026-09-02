from __future__ import annotations

import numpy as np
import pandas as pd

from conditional_drift_forward_features import calculate_conditional_drift_forward_features


def test_forward_features_stop_before_entry_and_write_no_targets() -> None:
    session_dates = pd.bdate_range("2025-01-01", periods=30)
    accepted_at = pd.Timestamp(session_dates[24], tz="UTC") + pd.Timedelta(hours=22)
    entry_date = session_dates[27]
    entry_at = pd.Timestamp(entry_date, tz="UTC") + pd.Timedelta(hours=14, minutes=30)
    queue = pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "cik": ["0000000001"],
            "assigned_ticker": ["TEST"],
            "accepted_at": [accepted_at],
            "evidence_role": ["unobserved_validation_candidate"],
            "forward_queue_eligible": [True],
            "outcome_materialized_locally": [False],
            "model_training_label_end": [pd.Timestamp("2024-12-31", tz="UTC")],
        }
    )
    filing = pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "form": ["10-Q"],
            "gm_level": [0.45],
            "gm_delta_yoy": [0.02],
            "gm_innovation_raw": [0.01],
            "gm_innovation_std": [0.5],
        }
    )
    execution = pd.DataFrame(
        {"validation_event_id": ["event-1"], "formal_entry_open_at": [entry_at]}
    )
    endpoints = pd.DataFrame({"validation_event_id": ["event-1"], "price_target_id": ["price-1"]})
    issuer_close = 20.0 + np.arange(30) * 0.1
    market_close = 100.0 + np.arange(30) * 0.2
    issuer_bars = _bars(session_dates, issuer_close, "price-1")
    benchmark_bars = _bars(session_dates, market_close, None)
    action_columns = [
        "price_target_id",
        "action_type",
        "effective_at",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
        "provider_event_id",
        "source_snapshot_sha256",
    ]
    empty_actions = pd.DataFrame(columns=action_columns)
    benchmark_actions = pd.DataFrame(columns=action_columns)
    sessions = pd.DataFrame(
        {
            "exchange_mic": ["XNYS"] * len(session_dates),
            "session_date": session_dates,
            "regular_open_at": [
                pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=14, minutes=30)
                for date in session_dates
            ],
            "regular_close_at": [
                pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=21) for date in session_dates
            ],
            "is_early_close": [False] * len(session_dates),
        }
    )

    tables = calculate_conditional_drift_forward_features(
        queue,
        filing,
        execution,
        endpoints,
        issuer_bars,
        empty_actions,
        benchmark_bars,
        benchmark_actions,
        sessions,
    )

    assert len(tables.features) == 1
    assert tables.report["outcome_values_loaded"] is False
    assert tables.report["target_columns_written"] is False
    assert not any(column.startswith("target_") for column in tables.features.columns)
    row = tables.features.iloc[0]
    assert row["accepted_at"] <= row["reaction_feature_available_at"] < row["execution_entry_at"]
    assert tables.clock_audit["market_clock_validated"].all()
    assert tables.clock_audit["reaction_clock_validated"].all()


def _bars(
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    price_target_id: str | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "open_unadjusted": closes - 0.02,
            "high_unadjusted": closes + 0.10,
            "low_unadjusted": closes - 0.10,
            "close_unadjusted": closes,
            "volume": 100_000.0 + np.arange(len(dates)) * 100.0,
        }
    )
    if price_target_id is not None:
        frame["price_target_id"] = price_target_id
    return frame
