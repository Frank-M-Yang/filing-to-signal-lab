"""Conditional quote-field sensitivity with fixed windows, no imputation or admission."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.pre_event_source_audit import QUOTE_COLUMNS
from sec_data.validation_market_state import MARKET_STATE_FEATURES

# Number of closes, not number of returns. Drawdown follows the existing kernel's
# 61-close peak definition. No shortened 60-session substitute is calculated.
WINDOW_CLOSES = dict(
    zip(MARKET_STATE_FEATURES, (6, 21, 61, 21, 61, 20, 20, 21, 61, 21, 61, 61, 61), strict=True)
)


def _validate(frame: pd.DataFrame, ticker: str, dates: list[str]) -> pd.DataFrame:
    if (
        set(frame) != set(QUOTE_COLUMNS)
        or frame.isna().any().any()
        or not frame.ticker.eq(ticker).all()
        or frame.session_date.duplicated().any()
        or not frame.session_date.is_monotonic_increasing
        or not frame.session_date.isin(dates).all()
        or not frame.source_sha256.str.fullmatch(r"[a-f0-9]{64}").all()
        or not np.isfinite(frame[["quote_close", "volume"]].to_numpy(dtype=float)).all()
        or not frame.quote_close.gt(0).all()
        or not frame.volume.ge(0).all()
    ):
        raise ValueError("Invalid quote projection, provenance, dates, price or volume")
    return frame.set_index("session_date")


def _calculate(feature: str, close: np.ndarray, volume: np.ndarray, market: np.ndarray) -> float:
    if feature.startswith("pre_price_momentum"):
        return float(close[-1] / close[0] - 1)
    if feature.startswith("pre_market_momentum"):
        return float(market[-1] / market[0] - 1)
    if feature.startswith("pre_relative_momentum"):
        return float((close[-1] / close[0] - 1) - (market[-1] / market[0] - 1))
    if feature.startswith("pre_price_volatility"):
        return float(np.std(np.diff(np.log(close)), ddof=1) * np.sqrt(252))
    if feature == "pre_volume_ratio_5_20":
        return float(np.mean(volume[-5:]) / np.mean(volume)) if np.mean(volume) > 0 else np.nan
    if feature == "pre_log_dollar_volume_20d":
        mean = float(np.mean(close * volume))
        return float(np.log(mean)) if mean > 0 else np.nan
    if feature == "pre_market_beta_60d":
        stock_returns, market_returns = np.diff(np.log(close)), np.diff(np.log(market))
        variance = float(np.var(market_returns, ddof=1))
        return (
            float(np.cov(stock_returns, market_returns, ddof=1)[0, 1] / variance)
            if variance > 0
            else np.nan
        )
    if feature == "pre_price_drawdown_60d":
        return float(close[-1] / np.max(close) - 1)
    raise ValueError("Unsupported feature")


def calculate_quote_feature_proxy(
    issuer: pd.DataFrame,
    benchmark: pd.DataFrame,
    required_sessions: list[str],
    announcement_at: str,
    *,
    issuer_ticker: str = "MBUU",
) -> pd.DataFrame:
    """Apply the frozen no-split 13-feature equations without certifying raw quotes.

    Each feature retains its exact session window. This diagnostic can describe
    short-window quantities with incomplete long history, but it never yields a
    canonical market_state_ready flag. Missing inputs and undefined ratios stay
    null. Returns here are PRE-event price changes, never post-event outcomes.
    """
    cutoff = utc_timestamp(announcement_at).tz_convert("America/New_York").date().isoformat()
    if (
        len(required_sessions) != 61
        or sorted(set(required_sessions)) != required_sessions
        or any(pd.Timestamp(d).date().isoformat() != d for d in required_sessions)
        or required_sessions[-1] >= cutoff
    ):
        raise ValueError("Need 61 distinct ordered strictly pre-event official sessions")
    stock, market = (
        _validate(issuer, issuer_ticker, required_sessions),
        _validate(benchmark, "SPY", required_sessions),
    )
    rows: list[dict[str, Any]] = []
    for feature in MARKET_STATE_FEATURES:
        dates = required_sessions[-WINDOW_CLOSES[feature] :]
        needs_stock = not feature.startswith("pre_market_momentum")
        needs_market = feature.startswith(
            ("pre_market_momentum", "pre_relative_momentum", "pre_market_beta")
        )
        missing_stock = sorted(set(dates) - set(stock.index)) if needs_stock else []
        missing_market = sorted(set(dates) - set(market.index)) if needs_market else []
        value = np.nan
        status = "missing_required_sessions"
        if not missing_stock and not missing_market:
            s = stock.reindex(dates)
            b = market.reindex(dates)
            value = _calculate(
                feature,
                s.quote_close.to_numpy(float),
                s.volume.to_numpy(float),
                b.quote_close.to_numpy(float),
            )
            status = "conditional_quote_proxy" if np.isfinite(value) else "undefined_statistic"
        unit = (
            "log_USD"
            if "log_dollar" in feature
            else "ratio"
            if "ratio" in feature or "beta" in feature
            else "decimal_fraction"
        )
        rows.append(
            {
                "feature": feature,
                "value": value,
                "status": status,
                "unit": unit,
                "required_closes": len(dates),
                "window_start": dates[0],
                "window_end": dates[-1],
                "missing_issuer_sessions": missing_stock,
                "missing_benchmark_sessions": missing_market,
                "model_admission": False,
            }
        )
    return pd.DataFrame(rows)


def compare_feature_proxies(primary: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Compare matched equations/windows only, without inventing acceptance thresholds."""
    contract = ["feature", "unit", "required_closes", "window_start", "window_end"]
    if any(
        list(f.feature) != list(MARKET_STATE_FEATURES) for f in (primary, candidate)
    ) or not primary[contract].equals(candidate[contract]):
        raise ValueError("Feature or window mismatch")
    result = primary[contract].copy()
    result["primary_value"] = primary.value.to_numpy()
    result["candidate_value"] = candidate.value.to_numpy()
    result["comparable"] = np.isfinite(result.primary_value) & np.isfinite(result.candidate_value)
    result["candidate_minus_primary"] = result.candidate_value - result.primary_value
    result["difference_basis_points"] = (
        result.candidate_minus_primary.where(result.unit.eq("decimal_fraction")) * 10000
    )
    # Exp(log dollar-volume difference) has an interpretable multiplicative scale.
    result["implied_dollar_volume_difference_pct"] = (
        np.expm1(result.candidate_minus_primary.where(result.unit.eq("log_USD"))) * 100
    )
    result["model_admission"] = False
    return result
