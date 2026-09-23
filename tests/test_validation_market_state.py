"""Tests for strict pre-filing market-state features."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sec_data.validation_market_state import (
    ValidationMarketStateConfig,
    ValidationMarketStateError,
    _feature_row,
    _prepare_bars,
    _prepare_splits,
    _split_adjusted_history,
    calculate_validation_market_state,
)


def test_market_state_is_split_adjusted_and_future_invariant() -> None:
    dates = pd.bdate_range("2025-01-02", periods=70)
    underlying = np.linspace(100.0, 120.0, len(dates))
    raw_close = underlying.copy()
    raw_close[:40] *= 2.0
    raw_volume = np.full(len(dates), 1_000.0)
    raw_volume[:40] /= 2.0
    accepted_at = (dates[-1] + pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=20)
    issuer_bars = pd.DataFrame(
        {
            "price_target_id": "price-1",
            "session_date": dates,
            "close_unadjusted": raw_close,
            "volume": raw_volume,
        }
    )
    benchmark_bars = pd.DataFrame(
        {
            "benchmark_target_id": "spy",
            "session_date": dates,
            "close_unadjusted": np.linspace(400.0, 430.0, len(dates)),
            "volume": 10_000.0,
        }
    )
    tables = calculate_validation_market_state(
        _authorized(accepted_at),
        _endpoints(),
        issuer_bars,
        benchmark_bars,
        pd.DataFrame(
            {
                "action_target_id": ["action-1"],
                "execution_date": [dates[40]],
                "historical_adjustment_factor": [0.5],
                "split_from": [1],
                "split_to": [2],
            }
        ),
        pd.DataFrame({"action_target_id": ["action-1"], "price_target_id": ["price-1"]}),
        _filing_features(accepted_at),
        _sessions(dates),
        issuer_price_basis="raw_as_traded",
        benchmark_price_basis="raw_as_traded",
    )
    row = tables.features.iloc[0]
    assert bool(row["market_state_ready"])
    assert row["split_adjustments_applied"] == 1
    assert row["pre_price_momentum_60d"] == pytest.approx(underlying[-1] / underlying[-61] - 1.0)
    assert pd.Timestamp(row["market_feature_available_at"]) < accepted_at

    future = issuer_bars.copy()
    future.loc[len(future)] = [
        "price-1",
        accepted_at.tz_convert(None).normalize() + pd.Timedelta(days=1),
        9_999.0,
        1.0,
    ]
    rerun = calculate_validation_market_state(
        _authorized(accepted_at),
        _endpoints(),
        future,
        benchmark_bars,
        pd.DataFrame(
            {
                "action_target_id": ["action-1"],
                "execution_date": [dates[40]],
                "historical_adjustment_factor": [0.5],
                "split_from": [1],
                "split_to": [2],
            }
        ),
        pd.DataFrame({"action_target_id": ["action-1"], "price_target_id": ["price-1"]}),
        _filing_features(accepted_at),
        _sessions(dates),
        issuer_price_basis="raw_as_traded",
        benchmark_price_basis="raw_as_traded",
    )
    assert rerun.features.loc[0, "pre_price_momentum_60d"] == pytest.approx(
        row["pre_price_momentum_60d"]
    )


def _authorized(accepted_at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_event_component_id": ["component-1"],
            "accepted_at": [accepted_at],
            "form": ["10-Q"],
        }
    )


def _endpoints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_event_component_id": ["component-1"],
            "validation_event_id": ["event-1"],
            "cik": ["0000000001"],
            "ticker": ["TEST"],
            "exchange_mic": ["XNYS"],
            "price_target_id": ["price-1"],
        }
    )


def _filing_features(accepted_at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_event_id": ["event-1"],
            "form": ["10-Q"],
            "actual_available_at": [accepted_at],
            "gm_level": [0.40],
            "gm_yoy_lag": [0.39],
            "gm_delta_yoy": [0.01],
            "gm_forecast": [0.395],
            "gm_innovation_raw": [0.005],
            "gm_innovation_std": [0.20],
            "forecast_feature_cutoff": [accepted_at - pd.Timedelta(days=30)],
            "forecast_training_end": [accepted_at - pd.Timedelta(days=40)],
            "forecast_training_observations": [100],
        }
    )


def _sessions(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange_mic": "XNYS",
            "session_date": dates,
            "regular_close_at": dates.tz_localize("UTC") + pd.Timedelta(hours=21),
        }
    )


def _raw_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_target_id": "p",
            "session_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "close_unadjusted": [600.0, 300.0, 100.0],
            "volume": [100.0, 200.0, 600.0],
        }
    )


def _local_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "execution_date": pd.to_datetime(["2025-01-03", "2025-01-06"]),
            "split_from": [1.0, 1.0],
            "split_to": [2.0, 3.0],
            "historical_adjustment_factor": [1 / 6, 1 / 3],
        }
    )


def test_multiple_splits_use_each_local_ratio_once_and_preserve_raw() -> None:
    raw, actions = _raw_history(), _local_actions()
    out, count = _split_adjusted_history(
        raw, actions, pd.Timestamp("2025-01-07"), price_basis="raw_as_traded"
    )
    assert count == 2
    np.testing.assert_allclose(out.close_adjusted_to_cutoff, [100, 100, 100])
    np.testing.assert_allclose(out.volume_adjusted_to_cutoff, [600, 600, 600])
    np.testing.assert_allclose(out.share_unit_factor, [1 / 6, 1 / 3, 1])
    pd.testing.assert_series_equal(out.close_unadjusted, raw.close_unadjusted)
    np.testing.assert_allclose(
        out.close_adjusted_to_cutoff * out.volume_adjusted_to_cutoff,
        raw.close_unadjusted * raw.volume,
    )
    actions["historical_adjustment_factor"] = [1e-12, np.nan]
    changed, _ = _split_adjusted_history(
        raw, actions, pd.Timestamp("2025-01-07"), price_basis="raw_as_traded"
    )
    pd.testing.assert_frame_equal(out, changed)


def test_future_and_cutoff_day_actions_cannot_rewrite_earlier_units() -> None:
    raw = _raw_history().iloc[:2].copy()
    actions = _local_actions()
    out, n = _split_adjusted_history(
        raw, actions, pd.Timestamp("2025-01-06"), price_basis="raw_as_traded"
    )
    np.testing.assert_allclose(out.close_adjusted_to_cutoff, [300, 300])
    assert n == 1
    future = pd.concat(
        [
            actions,
            pd.DataFrame(
                {
                    "execution_date": [pd.Timestamp("2030-01-01")],
                    "split_from": [10],
                    "split_to": [1],
                    "historical_adjustment_factor": [10],
                }
            ),
        ],
        ignore_index=True,
    )
    future["historical_adjustment_factor"] = -12345
    after, _ = _split_adjusted_history(
        raw, future, pd.Timestamp("2025-01-06"), price_basis="raw_as_traded"
    )
    pd.testing.assert_frame_equal(out, after)


def test_reverse_split_is_not_inverted() -> None:
    raw = _raw_history().iloc[:2].copy()
    raw["close_unadjusted"], raw["volume"] = [1.0, 10.0], [1000.0, 100.0]
    action = _local_actions().iloc[:1].copy()
    action["split_from"], action["split_to"] = 10.0, 1.0
    out, _ = _split_adjusted_history(
        raw, action, pd.Timestamp("2025-01-06"), price_basis="raw_as_traded"
    )
    np.testing.assert_allclose(out.close_adjusted_to_cutoff, [10, 10])
    np.testing.assert_allclose(out.volume_adjusted_to_cutoff, [100, 100])


@pytest.mark.parametrize(
    "bad", ["zero", "negative", "nan", "infinity", "duplicate", "missing_ratio", "intraday"]
)
def test_malformed_single_event_actions_fail_closed(bad: str) -> None:
    a = _local_actions()
    if bad in ("zero", "negative", "nan", "infinity"):
        a.loc[0, "split_to"] = {"zero": 0, "negative": -1, "nan": np.nan, "infinity": np.inf}[bad]
    elif bad == "duplicate":
        a = pd.concat([a, a.iloc[:1]])
    elif bad == "missing_ratio":
        a = a.drop(columns="split_from")
    else:
        a.loc[0, "execution_date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        _split_adjusted_history(
            _raw_history(), a, pd.Timestamp("2025-01-07"), price_basis="raw_as_traded"
        )


def test_mapping_requires_event_ratios_not_cumulative_fallback() -> None:
    a = _local_actions().assign(action_target_id="a").drop(columns="split_from")
    with pytest.raises(ValueError, match="split_from"):
        _prepare_splits(a, pd.DataFrame({"action_target_id": ["a"], "price_target_id": ["p"]}))


@pytest.mark.parametrize("basis", ["unknown", "split_adjusted", "total_return_adjusted", ""])
def test_unknown_or_vendor_adjusted_basis_is_not_treated_as_raw(basis: str) -> None:
    with pytest.raises(ValidationMarketStateError, match="basis"):
        _prepare_bars(_raw_history(), "price_target_id", "test", price_basis=basis)


def test_already_adjusted_output_is_rejected_even_if_caller_claims_raw() -> None:
    out, _ = _split_adjusted_history(
        _raw_history(), _local_actions(), pd.Timestamp("2025-01-07"), price_basis="raw_as_traded"
    )
    with pytest.raises(ValueError):
        _split_adjusted_history(
            out, _local_actions(), pd.Timestamp("2025-01-07"), price_basis="raw_as_traded"
        )
    out.attrs.clear()  # Column markers survive serialization without DataFrame attrs.
    with pytest.raises(ValueError, match="twice"):
        _prepare_bars(out, "price_target_id", "test", price_basis="raw_as_traded")


def test_adjusted_query_marker_and_old_version_are_rejected() -> None:
    with pytest.raises(ValueError, match="vendor"):
        _prepare_bars(
            _raw_history().assign(adjusted=True),
            "price_target_id",
            "test",
            price_basis="raw_as_traded",
        )
    with pytest.raises(ValueError, match="v2"):
        ValidationMarketStateConfig(version="validation-market-state-v1").validate()


def test_canonical_cutoff_uses_new_york_date_across_utc_midnight() -> None:
    dates = pd.bdate_range("2025-01-02", periods=70)
    bars = pd.DataFrame(
        {"session_date": dates, "close_unadjusted": np.linspace(100, 120, 70), "volume": 1000.0}
    )
    accepted = dates[-1].tz_localize("UTC") + pd.Timedelta(days=1, hours=1)
    event = SimpleNamespace(validation_event_id="e", accepted_at=accepted, exchange_mic="XNYS")
    lookup = {("XNYS", d.date()): d.tz_localize("UTC") + pd.Timedelta(hours=21) for d in dates}
    result = _feature_row(event, bars, bars, pd.DataFrame(), lookup, ValidationMarketStateConfig())
    assert result["market_last_session_date"] == dates[-2]
    before = dict(result)
    bars.loc[bars.session_date.eq(dates[-1]), "close_unadjusted"] = 1e12
    assert (
        _feature_row(event, bars, bars, pd.DataFrame(), lookup, ValidationMarketStateConfig())
        == before
    )
