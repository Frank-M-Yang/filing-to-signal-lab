"""Corporate-action share-unit diagnostics and missing-price exception classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.pre_event_source_audit import QUOTE_COLUMNS


def split_to_cutoff_units(
    quotes: pd.DataFrame, actions: pd.DataFrame, cutoff_at: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Convert pre-split prices and volumes using event ratios only.

    This is an explicit hypothesis for Yahoo, not a certification that its quotes
    are raw. Actions on/after the cutoff date never affect prior input units.
    Source hashes continue to refer to the unmodified input, not transformed bytes.
    """
    day = utc_timestamp(cutoff_at).tz_convert("America/New_York").date().isoformat()
    if (
        set(quotes) != set(QUOTE_COLUMNS)
        or quotes.empty
        or quotes.isna().any().any()
        or quotes.ticker.nunique() != 1
        or quotes.session_date.duplicated().any()
        or not quotes.session_date.is_monotonic_increasing
        or not quotes.session_date.lt(day).all()
        or not np.isfinite(quotes[["quote_close", "volume"]].to_numpy(float)).all()
        or not quotes.quote_close.gt(0).all()
        or not quotes.volume.ge(0).all()
        or not quotes.source_sha256.str.fullmatch(r"[a-f0-9]{64}").all()
    ):
        raise ValueError("Invalid or non-prior quote inputs")
    required = {"ticker", "execution_date", "split_from", "split_to", "source_sha256"}
    if not required.issubset(actions):
        raise ValueError("Split event ratios/provenance missing")
    if any(pd.Timestamp(d).date().isoformat() != d for d in actions.execution_date):
        raise ValueError("Split dates must be unambiguous ISO dates")
    eligible = actions.loc[actions.execution_date.lt(day)].sort_values("execution_date")
    if (
        eligible.execution_date.duplicated().any()
        or not eligible.ticker.eq(quotes.ticker.iloc[0]).all()
    ):
        raise ValueError("Ambiguous split identity/date")
    if not eligible.source_sha256.str.fullmatch(r"[a-f0-9]{64}").all():
        raise ValueError("Missing split provenance")
    ratios = eligible[["split_from", "split_to"]].to_numpy(float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("Invalid split ratio")
    converted = quotes.copy()
    converted["quote_close"] = converted.quote_close.astype(float)
    converted["volume"] = converted.volume.astype(float)
    ledger = []
    for a in eligible.to_dict("records"):
        factor = float(a["split_from"] / a["split_to"])
        mask = converted.session_date.lt(a["execution_date"])
        converted.loc[mask, "quote_close"] *= factor
        converted.loc[mask, "volume"] /= factor
        ledger.append(
            {
                "execution_date": a["execution_date"],
                "event_ratio_factor": factor,
                "rows_adjusted": int(mask.sum()),
                "source_sha256": a["source_sha256"],
                "vendor_cumulative_factor_used": False,
            }
        )
    if not np.allclose(
        (quotes.quote_close * quotes.volume).to_numpy(),
        (converted.quote_close * converted.volume).to_numpy(),
        rtol=1e-12,
        atol=0,
    ):
        raise ValueError("Share-unit conversion changed price-times-volume")
    return converted, ledger


def split_boundary_changes(
    series: dict[str, pd.DataFrame], actions: pd.DataFrame, official_dates: list[str]
) -> list[dict[str, Any]]:
    """Report mechanical close changes across exact split sessions, never trade returns."""
    rows = []
    for action in actions.to_dict("records"):
        day = action["execution_date"]
        if day not in official_dates or official_dates.index(day) == 0:
            continue
        before = official_dates[official_dates.index(day) - 1]
        for label, frame in series.items():
            indexed = frame.set_index("session_date")
            present = before in indexed.index and day in indexed.index
            rows.append(
                {
                    "series": label,
                    "previous_session": before,
                    "split_session": day,
                    "exact_endpoints_present": present,
                    "close_change_pct": float(
                        (indexed.loc[day, "quote_close"] / indexed.loc[before, "quote_close"] - 1)
                        * 100
                    )
                    if present
                    else None,
                    "not_a_strategy_return": True,
                }
            )
    return rows


def classify_execution_exceptions(exceptions: pd.DataFrame, listing: pd.DataFrame) -> pd.DataFrame:
    """Retain all missing-price cases; distinguish dated delisting from unknown halt/payout."""
    if (
        exceptions.validation_event_id.duplicated().any()
        or listing.duplicated(["cik", "ticker_at_event"]).any()
    ):
        raise ValueError("Ambiguous exception/listing identity")
    rows = []
    for e in exceptions.to_dict("records"):
        match = listing.loc[listing.cik.eq(e["cik"]) & listing.ticker_at_event.eq(e["ticker"])]
        record = match.iloc[0].to_dict() if len(match) == 1 else {}
        day = record.get("delisting_date")
        known = pd.notna(day) and isinstance(day, str) and len(day) >= 10
        inside = bool(
            known and e["formal_entry_session_date"] <= day <= e["formal_exit_session_date"]
        )
        later = bool(known and day > e["formal_exit_session_date"])
        rows.append(
            {
                **e,
                "delisting_date": day if known else None,
                "dated_delisting_inside_window": inside,
                "later_delisting_not_explanation_for_prior_gap": later,
                "halt_status": "unknown_no_official_halt_evidence",
                "terminal_value_observed": record.get("terminal_value_observed") is True,
                "terminal_value_required_but_unobserved": inside
                and record.get("terminal_value_observed") is not True,
                "listing_status": record.get("listing_outcome_status", "unresolved"),
                "event_retained": True,
                "price_imputed": False,
                "zero_return_imputed": False,
                "terminal_return_computed": False,
                "model_admission": False,
            }
        )
    return pd.DataFrame(rows)
