"""Quote-field validation and descriptive cross-provider checks, never admission."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp

VERSION = "pre-event-source-audit-v1"
QUOTE_COLUMNS = ["ticker", "session_date", "quote_close", "volume", "source_sha256"]


def parse_pre_event_quotes(
    content: bytes,
    ticker: str,
    start: str,
    end: str,
    announcement_at: str,
    *,
    expected_exchange_mic: str = "XNAS",
) -> pd.DataFrame:
    """Use only Yahoo quote.close/volume, not adjclose or current-price metadata.

    The name quote_close intentionally does not certify historical unadjusted
    semantics. Current Yahoo identity metadata is not a historical security map.
    Unexpected bar dates are rejected, rather than silently filtered away.
    """
    boundary = utc_timestamp(announcement_at)
    if not start <= end < boundary.tz_convert("America/New_York").date().isoformat():
        raise ValueError("Quote query must end strictly before the announcement date")
    chart = json.loads(content)["chart"]
    if chart.get("error") is not None or len(chart.get("result", [])) != 1:
        raise ValueError("Yahoo error or ambiguous chart result")
    item = chart["result"][0]
    meta = item["meta"]
    exchanges = {"XNAS": {"NMS", "NGM", "NCM"}, "XNYS": {"NYQ"}, "ARCX": {"PCX"}}
    allowed = {"PCX"} if ticker == "SPY" else exchanges.get(expected_exchange_mic, set())
    if (
        meta.get("symbol") != ticker
        or meta.get("currency") != "USD"
        or meta.get("exchangeTimezoneName") != "America/New_York"
        or meta.get("instrumentType") != ("ETF" if ticker == "SPY" else "EQUITY")
        or meta.get("exchangeName") not in allowed
    ):
        raise ValueError("Quote symbol/currency/timezone/instrument/exchange mismatch")
    stamps = pd.to_datetime(item["timestamp"], unit="s", utc=True)
    dates = stamps.tz_convert("America/New_York").strftime("%Y-%m-%d")
    if len(stamps) == 0 or any(stamps >= boundary) or any((dates < start) | (dates > end)):
        raise ValueError("Quote timestamps empty or outside the frozen pre-event window")
    quotes = item["indicators"]["quote"]
    if len(quotes) != 1 or any(len(quotes[0][k]) != len(stamps) for k in ("close", "volume")):
        raise ValueError("Quote arrays misaligned")
    result = pd.DataFrame(
        {
            "ticker": ticker,
            "session_date": dates,
            "quote_close": pd.to_numeric(pd.Series(quotes[0]["close"]), errors="raise"),
            "volume": pd.to_numeric(pd.Series(quotes[0]["volume"]), errors="raise"),
            "source_sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    if (
        result.session_date.duplicated().any()
        or not result.session_date.is_monotonic_increasing
        or not np.isfinite(result[["quote_close", "volume"]].to_numpy(dtype=float)).all()
        or not result.quote_close.gt(0).all()
        or not result.volume.ge(0).all()
        or not result.volume.mod(1).eq(0).all()
    ):
        raise ValueError("Duplicate/unordered dates or invalid price/volume")
    return result[QUOTE_COLUMNS]


def compare_quote_overlap(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    price_tolerance: float = 0.01,
    minimum_sessions: int = 45,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare all shared dates with fixed cent/zero-volume tolerances; never splice."""
    if price_tolerance not in {0.0, 0.01} or minimum_sessions < 1:
        raise ValueError("Unsupported overlap tolerances")
    for frame in (primary, secondary):
        if (
            set(frame) != set(QUOTE_COLUMNS)
            or frame.empty
            or frame.ticker.nunique() != 1
            or frame.session_date.duplicated().any()
            or frame[QUOTE_COLUMNS].isna().any().any()
            or not frame.source_sha256.str.fullmatch(r"[a-f0-9]{64}").all()
            or not np.isfinite(frame[["quote_close", "volume"]].to_numpy(dtype=float)).all()
            or not frame.quote_close.gt(0).all()
            or not frame.volume.ge(0).all()
        ):
            raise ValueError("Invalid overlap input")
    if primary.ticker.iloc[0] != secondary.ticker.iloc[0]:
        raise ValueError("Cannot join different securities by date")
    rows = primary.merge(
        secondary,
        on=["ticker", "session_date"],
        suffixes=("_primary", "_secondary"),
        validate="one_to_one",
    )
    rows["close_difference"] = rows.quote_close_secondary - rows.quote_close_primary
    rows["volume_difference"] = rows.volume_secondary - rows.volume_primary
    rows["absolute_volume_relative_difference"] = rows.volume_difference.abs().div(
        rows.volume_primary.where(rows.volume_primary.gt(0))
    )
    rows["close_agrees"] = rows.close_difference.abs().le(price_tolerance)
    rows["volume_agrees"] = rows.volume_difference.eq(0)
    enough = len(rows) >= minimum_sessions
    return {
        "ticker": str(primary.ticker.iloc[0]),
        "overlap_sessions": len(rows),
        "minimum_overlap_sessions": minimum_sessions,
        "price_tolerance_usd": price_tolerance,
        "volume_tolerance": 0,
        "close_disagreements": int((~rows.close_agrees).sum()),
        "volume_disagreements": int((~rows.volume_agrees).sum()),
        "maximum_absolute_close_difference": None
        if rows.empty
        else float(rows.close_difference.abs().max()),
        "maximum_absolute_volume_difference": None
        if rows.empty
        else float(rows.volume_difference.abs().max()),
        "maximum_absolute_volume_relative_difference": None
        if rows.absolute_volume_relative_difference.dropna().empty
        else float(rows.absolute_volume_relative_difference.max()),
        "median_absolute_volume_relative_difference": None
        if rows.absolute_volume_relative_difference.dropna().empty
        else float(rows.absolute_volume_relative_difference.median()),
        "zero_primary_volume_sessions": int(rows.volume_primary.eq(0).sum()),
        "overlap_consistent": bool(enough and rows.close_agrees.all() and rows.volume_agrees.all()),
        "historical_identity_certified": False,
        "unadjusted_semantics_certified": False,
        "automatic_splicing": False,
        "model_admission": False,
    }, rows
