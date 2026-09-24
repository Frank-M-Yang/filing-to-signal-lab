"""Disclosure-cut market-state tests (synthetic prices, no outcomes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sec_data.disclosure_market_state import (
    DisclosureMarketStateError,
    calculate_disclosure_market_state,
)
from sec_data.validation_market_state import ValidationMarketStateInputs

DATES = pd.bdate_range("2025-01-02", periods=90)
RELEASE_INDEX = 80
FILING_AT = (DATES[-1] + pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=20)
# Released after the fixture's 21:00 UTC close; the reaction starts next session.
RELEASE_AT = DATES[RELEASE_INDEX].tz_localize("UTC") + pd.Timedelta(hours=21, minutes=30)


def _prices(post_release_multiplier: float = 1.2) -> np.ndarray:
    close = np.linspace(100.0, 110.0, len(DATES))
    close[RELEASE_INDEX + 1 :] *= post_release_multiplier
    return close


def _inputs(close: np.ndarray) -> ValidationMarketStateInputs:
    return ValidationMarketStateInputs(
        authorized_cohort=pd.DataFrame(
            {"security_event_component_id": ["component-1"], "accepted_at": [FILING_AT]}
        ),
        event_endpoints=pd.DataFrame(
            {
                "security_event_component_id": ["component-1"],
                "validation_event_id": ["event-1"],
                "cik": ["0000000001"],
                "ticker": ["TEST"],
                "exchange_mic": ["XNYS"],
                "price_target_id": ["price-1"],
            }
        ),
        issuer_bars=pd.DataFrame(
            {
                "price_target_id": "price-1",
                "session_date": DATES,
                "close_unadjusted": close,
                "volume": 1_000.0,
            }
        ),
        benchmark_bars=pd.DataFrame(
            {
                "benchmark_target_id": "spy",
                "session_date": DATES,
                "close_unadjusted": 400.0 + np.sin(np.arange(len(DATES))) + np.arange(len(DATES)),
                "volume": 10_000.0,
            }
        ),
        split_observations=pd.DataFrame(),
        action_targets=pd.DataFrame(columns=["action_target_id", "price_target_id"]),
        filing_features=pd.DataFrame(
            {
                "validation_event_id": ["event-1"],
                "target_accession_number": ["ACC-1"],
                "form": ["10-Q"],
                "actual_available_at": [FILING_AT],
                "gm_level": [0.40],
                "gm_yoy_lag": [0.39],
                "gm_delta_yoy": [0.01],
                "gm_forecast": [0.395],
                "gm_innovation_raw": [0.005],
                "gm_innovation_std": [0.20],
                "forecast_feature_cutoff": [FILING_AT - pd.Timedelta(days=30)],
                "forecast_training_end": [FILING_AT - pd.Timedelta(days=40)],
                "forecast_training_observations": [100],
            }
        ),
        exchange_sessions=pd.DataFrame(
            {
                "exchange_mic": "XNYS",
                "session_date": DATES,
                "regular_close_at": DATES.tz_localize("UTC") + pd.Timedelta(hours=21),
            }
        ),
    )


def _clocks(earliest: pd.Timestamp, *, accession: str = "ACC-1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": ["0000000001"],
            "accession_number": [accession],
            "report_accepted_at": [FILING_AT],
            "earliest_disclosure_at": [earliest],
            "clock_source": ["item_2_02_8k" if earliest < FILING_AT else "periodic_report"],
            "disclosure_lead_days": [(FILING_AT - earliest).total_seconds() / 86_400],
        }
    )


def test_release_cut_excludes_post_release_prices() -> None:
    close = _prices()
    tables = calculate_disclosure_market_state(_inputs(close), _clocks(RELEASE_AT))
    row = tables.features.iloc[0]
    assert bool(row["market_state_ready"])
    # The New York date cut drops the release day itself as well as later sessions.
    assert pd.Timestamp(row["market_last_session_date"]) == DATES[RELEASE_INDEX - 1]
    assert pd.Timestamp(row["market_feature_available_at"]) < RELEASE_AT
    expected = close[RELEASE_INDEX - 1] / close[RELEASE_INDEX - 6] - 1.0
    assert row["pre_price_momentum_5d"] == pytest.approx(expected)

    comparison = tables.comparison.iloc[0]
    assert bool(comparison["filing_view_uses_post_disclosure_close"]) is True
    assert bool(comparison["features_changed"]) is True
    assert comparison["readiness_transition"] == "ready_both"
    assert comparison["sessions_removed"] == len(DATES) - RELEASE_INDEX
    assert tables.report["filing_view_post_disclosure_close_events"] == 1
    assert tables.report["returns_computed"] is False


def test_release_cut_is_invariant_to_post_release_prices() -> None:
    calm = calculate_disclosure_market_state(_inputs(_prices(1.2)), _clocks(RELEASE_AT))
    shock = calculate_disclosure_market_state(_inputs(_prices(3.0)), _clocks(RELEASE_AT))
    for column in ("pre_price_momentum_5d", "pre_price_volatility_20d", "pre_market_beta_60d"):
        assert shock.features.loc[0, column] == pytest.approx(calm.features.loc[0, column])
    # The filing-cut 20-session window spans the release, so it does see the shock.
    assert shock.comparison.loc[0, "filing_pre_price_momentum_20d"] != pytest.approx(
        calm.comparison.loc[0, "filing_pre_price_momentum_20d"]
    )


def test_periodic_clock_reproduces_the_filing_view() -> None:
    tables = calculate_disclosure_market_state(_inputs(_prices()), _clocks(FILING_AT))
    comparison = tables.comparison.iloc[0]
    assert bool(comparison["features_changed"]) is False
    assert bool(comparison["filing_view_uses_post_disclosure_close"]) is False
    assert bool(comparison["cutoff_date_moved"]) is False
    assert comparison["max_abs_feature_change"] == pytest.approx(0.0)


def test_missing_clock_is_kept_and_labelled() -> None:
    tables = calculate_disclosure_market_state(
        _inputs(_prices()), _clocks(RELEASE_AT, accession="OTHER")
    )
    assert tables.features.loc[0, "market_state_status"] == "disclosure_clock_not_found"
    assert tables.comparison.loc[0, "readiness_transition"] == "clock_not_found"
    assert pd.isna(tables.comparison.loc[0, "sessions_removed"])
    assert tables.report["clock_found"] == 0


def test_rejects_inconsistent_clock_tables() -> None:
    mismatched = _clocks(RELEASE_AT)
    mismatched["report_accepted_at"] = FILING_AT - pd.Timedelta(hours=1)
    with pytest.raises(DisclosureMarketStateError):
        calculate_disclosure_market_state(_inputs(_prices()), mismatched)
    later = _clocks(RELEASE_AT)
    later["earliest_disclosure_at"] = FILING_AT + pd.Timedelta(hours=1)
    with pytest.raises(DisclosureMarketStateError):
        calculate_disclosure_market_state(_inputs(_prices()), later)
    with pytest.raises(DisclosureMarketStateError):
        calculate_disclosure_market_state(
            _inputs(_prices()), _clocks(RELEASE_AT).drop(columns=["clock_source"])
        )
