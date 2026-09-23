"""Source-independent math parity, exact-window missingness and contamination guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_mbuu_source_sensitivity import build_sensitivity
from sec_data.market_source_sensitivity import (
    calculate_quote_feature_proxy,
    compare_feature_proxies,
)
from sec_data.publisher_market_rebuild import rebuild_pre_event_state
from sec_data.validation_market_state import MARKET_STATE_FEATURES

ANNOUNCEMENT = "2024-10-31T11:00:00Z"


def _inputs(seed: int = 1) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-08-06", "2024-10-30").difference(pd.DatetimeIndex(["2024-09-02"]))
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame(
            {
                "ticker": t,
                "session_date": dates.strftime("%Y-%m-%d"),
                "quote_close": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
                "volume": rng.integers(1000, 10000, len(dates)),
                "source_sha256": "b" * 64,
            }
        )
        for t in ("MBUU", "SPY")
    ]
    return dates.strftime("%Y-%m-%d").tolist(), frames[0], frames[1]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_all_thirteen_equations_match_frozen_kernel(seed: int) -> None:
    dates, stock, market = _inputs(seed)
    calendar = pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": dates,
            "regular_open_at": [d + "T13:30:00Z" for d in dates],
            "regular_close_at": [d + "T20:00:00Z" for d in dates],
            "source_snapshot_sha256": "a" * 64,
        }
    )
    event = {
        "validation_event_id": "test",
        "cik": "0001590976",
        "ticker": "MBUU",
        "exchange_mic": "XNAS",
        "price_target_id": "p",
        "benchmark_target_id": "b",
        "announcement_at": ANNOUNCEMENT,
    }
    canonical, _ = rebuild_pre_event_state(
        event,
        stock.rename(columns={"quote_close": "close_unadjusted"}),
        market.rename(columns={"quote_close": "close_unadjusted"}),
        pd.DataFrame(),
        calendar,
        price_basis="raw_as_traded",
    )
    assert canonical["market_state_ready"]
    observed = calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)
    np.testing.assert_allclose(
        observed.value.to_numpy(), [canonical[f] for f in MARKET_STATE_FEATURES], rtol=0, atol=1e-12
    )
    assert not observed.model_admission.any()
    assert "market_state_ready" not in observed


@pytest.mark.parametrize("remaining", [45, 28, 21, 20, 6, 0])
def test_missing_long_history_does_not_shorten_feature_windows(remaining: int) -> None:
    dates, stock, market = _inputs()
    s, m = (
        (stock.tail(remaining), market.tail(remaining))
        if remaining
        else (stock.iloc[:0], market.iloc[:0])
    )
    actual = calculate_quote_feature_proxy(s, m, dates, ANNOUNCEMENT)
    assert actual.value.notna().tolist() == actual.required_closes.le(remaining).tolist()
    assert actual.loc[actual.required_closes.gt(remaining), "value"].isna().all()
    if remaining in {45, 28, 21}:
        assert actual.value.notna().sum() == 7
    expected = calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)
    comparison = compare_feature_proxies(actual, expected)
    assert comparison.candidate_minus_primary.dropna().eq(0).all()
    assert comparison.loc[~comparison.comparable, "candidate_minus_primary"].isna().all()


def test_missing_interior_date_cannot_be_bridged_or_counted_as_adjacent() -> None:
    dates, stock, market = _inputs()
    result = calculate_quote_feature_proxy(stock.drop(index=50), market, dates, ANNOUNCEMENT)
    assert result.loc[result.feature.eq("pre_price_momentum_20d"), "value"].isna().all()
    assert result.loc[result.feature.eq("pre_price_momentum_5d"), "value"].notna().all()
    assert result.loc[result.feature.eq("pre_market_momentum_20d"), "value"].notna().all()
    last_missing = calculate_quote_feature_proxy(
        stock.iloc[:-1], market.iloc[:-1], dates, ANNOUNCEMENT
    )
    assert last_missing.value.isna().all()


def test_volume_change_does_not_change_price_features() -> None:
    dates, stock, market = _inputs()
    base = calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)
    changed = stock.copy()
    changed.loc[changed.index[-5:], "volume"] *= 2
    candidate = calculate_quote_feature_proxy(changed, market, dates, ANNOUNCEMENT)
    compared = compare_feature_proxies(base, candidate)
    assert compared.loc[compared.unit.eq("decimal_fraction"), "difference_basis_points"].eq(0).all()
    assert set(compared.loc[compared.candidate_minus_primary.ne(0), "feature"]) == {
        "pre_volume_ratio_5_20",
        "pre_log_dollar_volume_20d",
    }
    assert compared.implied_dollar_volume_difference_pct.dropna().gt(0).all()


def test_undefined_ratios_and_zero_variance_stay_unknown() -> None:
    dates, stock, market = _inputs()
    stock["volume"] = 0
    market["quote_close"] = 50.0
    result = calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)
    assert set(result.loc[result.value.isna(), "feature"]) == {
        "pre_volume_ratio_5_20",
        "pre_log_dollar_volume_20d",
        "pre_market_beta_60d",
    }
    assert result.loc[result.value.isna(), "status"].eq("undefined_statistic").all()


@pytest.mark.parametrize(
    "bad",
    [
        "ticker",
        "duplicate",
        "future",
        "gap_outside",
        "unsorted",
        "negative_price",
        "negative_volume",
        "nan",
        "hash",
        "extra_outcome",
    ],
)
def test_contamination_and_malformed_inputs_rejected(bad: str) -> None:
    dates, stock, market = _inputs()
    if bad == "ticker":
        stock["ticker"] = "OTHER"
    elif bad == "duplicate":
        stock = pd.concat([stock, stock.tail(1)], ignore_index=True)
    elif bad in {"future", "gap_outside"}:
        stock.loc[0, "session_date"] = "2024-10-31" if bad == "future" else "2024-08-03"
    elif bad == "unsorted":
        stock = stock.iloc[::-1]
    elif bad == "negative_price":
        stock.loc[0, "quote_close"] = -1
    elif bad == "negative_volume":
        stock.loc[0, "volume"] = -1
    elif bad == "nan":
        stock.loc[0, "quote_close"] = np.nan
    elif bad == "hash":
        stock.loc[0, "source_sha256"] = "unknown"
    else:
        stock["post_event_return"] = 0.5
    with pytest.raises(ValueError):
        calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)


def test_clock_and_comparison_contracts() -> None:
    dates, stock, market = _inputs()
    for wrong_dates in (dates[:-1], dates[::-1], [*dates[:-1], dates[0]]):
        with pytest.raises(ValueError):
            calculate_quote_feature_proxy(stock, market, wrong_dates, ANNOUNCEMENT)
    with pytest.raises(ValueError):
        calculate_quote_feature_proxy(stock, market, dates, "2024-10-30T23:00:00Z")
    result = calculate_quote_feature_proxy(stock, market, dates, ANNOUNCEMENT)
    wrong = result.copy()
    wrong.loc[0, "window_start"] = dates[0]
    with pytest.raises(ValueError, match="window"):
        compare_feature_proxies(result, wrong)


@pytest.mark.parametrize(
    "key,value",
    [
        ("network_allowed", True),
        ("returns_allowed", True),
        ("model_admission", True),
        ("provider_splicing", True),
        ("shorten_feature_lookbacks", True),
        ("required_sessions", 45),
        ("economic_acceptance_threshold", 0.01),
    ],
)
def test_weakened_experiment_contract_fails_before_reading_data(
    tmp_path: Path, key: str, value: Any
) -> None:
    c = json.loads(Path("docs/mbuu_source_sensitivity_v1.json").read_bytes())
    c[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        build_sensitivity(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
