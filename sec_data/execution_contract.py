"""New, versioned date-based execution contract; legacy ledgers stay immutable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import utc_timestamp


@dataclass(frozen=True)
class ExecutionContract:
    """Second exchange session after the New York announcement date, not time."""

    version: str = "execution-contract-v1-after-local-date"
    timezone: str = "America/New_York"
    entry_session_offset: int = 2
    holding_sessions_inclusive: int = 20
    entry_price_type: str = "unadjusted_open"
    exit_price_type: str = "unadjusted_close"

    def validate(self) -> None:
        """Require a new contract version for any change in economic conventions."""
        if asdict(self) != asdict(ExecutionContract()):
            raise ValueError("Execution convention differs from frozen version")


def execution_window(
    announcement_at: str,
    exchange_mic: str,
    sessions: pd.DataFrame,
    contract: ExecutionContract | None = None,
) -> dict[str, Any]:
    """Build a schedule only; a supplied clock never becomes certified by scheduling.

    Entry excludes the entire local announcement date for pre-open, intraday and
    after-close news alike. Holding day one is entry day. No missing date or price
    is imputed. The caller must pin a complete, official exchange calendar.
    """
    cfg = contract or ExecutionContract()
    cfg.validate()
    event = utc_timestamp(announcement_at)
    event_date = event.tz_convert(cfg.timezone).date().isoformat()
    frame = sessions.loc[sessions.exchange_mic.eq(exchange_mic)].copy()
    if frame.empty or frame.session_date.duplicated().any():
        raise ValueError("Missing or duplicated exchange calendar")
    frame = frame.sort_values("session_date")
    for column in ("regular_open_at", "regular_close_at"):
        frame[column] = frame[column].map(utc_timestamp)
        local_dates = frame[column].dt.tz_convert(cfg.timezone).dt.strftime("%Y-%m-%d")
        if not local_dates.eq(frame.session_date).all():
            raise ValueError("Session date disagrees with local open/close date")
    if (
        (frame.regular_open_at >= frame.regular_close_at).any()
        or not frame.regular_open_at.is_monotonic_increasing
        or not frame.regular_close_at.is_monotonic_increasing
        or frame.source_snapshot_sha256.isna().any()
        or not frame.source_snapshot_sha256.str.fullmatch(r"[a-f0-9]{64}").all()
    ):
        raise ValueError("Invalid calendar boundaries or provenance")
    if not frame.session_date.min() <= event_date <= frame.session_date.max():
        raise ValueError("Announcement outside frozen calendar coverage")
    future = frame.loc[frame.session_date.gt(event_date)].reset_index(drop=True)
    entry_index = cfg.entry_session_offset - 1
    exit_index = entry_index + cfg.holding_sessions_inclusive - 1
    if len(future) <= exit_index:
        raise ValueError("Insufficient calendar for the complete holding window")
    entry, exit = future.iloc[entry_index], future.iloc[exit_index]
    return {
        **asdict(cfg),
        "announcement_at": event.isoformat(),
        "announcement_local_date": event_date,
        "exchange_mic": exchange_mic,
        "entry_session_date": entry.session_date,
        "entry_at": entry.regular_open_at.isoformat(),
        "exit_session_date": exit.session_date,
        "exit_at": exit.regular_close_at.isoformat(),
        "entry_calendar_sha256": entry.source_snapshot_sha256,
        "exit_calendar_sha256": exit.source_snapshot_sha256,
        "schedule_resolved": True,
        "clock_certified": False,
        "price_readiness": False,
        "return_admission": False,
    }
