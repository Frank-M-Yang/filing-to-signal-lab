"""Execution windows under the filing clock versus the earliest disclosure clock.

Schedule-only: no price is loaded and no return is computed.  Both windows use
the same frozen, date-based ``ExecutionContract``, so any difference in entry
or exit dates is attributable to the information clock alone.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.execution_contract import ExecutionContract, execution_window

EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "validation_event_id",
    "cik",
    "target_accession_number",
    "accepted_at",
    "assigned_exchange_mic",
)
CLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "cik",
    "accession_number",
    "report_accepted_at",
    "earliest_disclosure_at",
    "clock_source",
    "disclosure_lead_days",
)


class FirstPublicExecutionError(ValueError):
    """Raised when the frozen cohort, clock table, or calendar cannot be joined safely."""


def compare_execution_clocks(
    events: pd.DataFrame,
    clocks: pd.DataFrame,
    sessions: pd.DataFrame,
    contract: ExecutionContract | None = None,
) -> pd.DataFrame:
    """Schedule every frozen event twice under one contract and compare entry dates.

    Events whose report is absent from the clock table, or whose window cannot
    be resolved from the pinned calendar, are kept with an explicit status.
    ``entry_sessions_earlier`` counts the official sessions by which the
    earliest-disclosure entry precedes the filing-clock entry.
    """
    cfg = contract or ExecutionContract()
    cfg.validate()
    _require(events, EVENT_COLUMNS, "events")
    _require(clocks, CLOCK_COLUMNS, "clocks")
    if events["validation_event_id"].duplicated().any():
        raise FirstPublicExecutionError("Frozen event identities must be unique")
    if clocks.duplicated(["cik", "accession_number"]).any():
        raise FirstPublicExecutionError("Clock rows must be unique per (cik, accession)")

    joined = events.loc[:, list(EVENT_COLUMNS)].merge(
        clocks.loc[:, list(CLOCK_COLUMNS)],
        left_on=["cik", "target_accession_number"],
        right_on=["cik", "accession_number"],
        how="left",
        validate="many_to_one",
    )
    calendars = {
        mic: frame.sort_values("session_date")["session_date"].to_numpy(dtype=object)
        for mic, frame in sessions.groupby("exchange_mic")
    }

    rows: list[dict[str, Any]] = []
    for event in joined.to_dict("records"):
        filing_at = utc_timestamp(event["accepted_at"])
        clock_found = bool(pd.notna(event["earliest_disclosure_at"]))
        row: dict[str, Any] = {
            "validation_event_id": event["validation_event_id"],
            "cik": event["cik"],
            "target_accession_number": event["target_accession_number"],
            "exchange_mic": event["assigned_exchange_mic"],
            "clock_found": clock_found,
            "clock_source": event["clock_source"] if clock_found else None,
            "disclosure_lead_days": event["disclosure_lead_days"] if clock_found else None,
            "filing_clock_matches_catalog": (
                bool(filing_at == utc_timestamp(event["report_accepted_at"]))
                if clock_found
                else None
            ),
            "filing_entry_session_date": None,
            "filing_exit_session_date": None,
            "disclosure_entry_session_date": None,
            "disclosure_exit_session_date": None,
            "entry_changed": None,
            "entry_sessions_earlier": None,
            "returns_computed": False,
        }
        filing, filing_error = _window(filing_at, event["assigned_exchange_mic"], sessions, cfg)
        if filing is not None:
            row["filing_entry_session_date"] = filing["entry_session_date"]
            row["filing_exit_session_date"] = filing["exit_session_date"]
        if not clock_found:
            row["window_status"] = "clock_not_found"
            rows.append(row)
            continue
        disclosure_at = utc_timestamp(event["earliest_disclosure_at"])
        disclosure, disclosure_error = _window(
            disclosure_at, event["assigned_exchange_mic"], sessions, cfg
        )
        if disclosure is not None:
            row["disclosure_entry_session_date"] = disclosure["entry_session_date"]
            row["disclosure_exit_session_date"] = disclosure["exit_session_date"]
        if filing is None:
            row["window_status"] = f"filing_window_unresolved: {filing_error}"
        elif disclosure is None:
            row["window_status"] = f"disclosure_window_unresolved: {disclosure_error}"
        else:
            row["window_status"] = "compared"
            row["entry_changed"] = filing["entry_session_date"] != disclosure["entry_session_date"]
            dates = calendars[event["assigned_exchange_mic"]]
            row["entry_sessions_earlier"] = int(
                np.searchsorted(dates, filing["entry_session_date"])
                - np.searchsorted(dates, disclosure["entry_session_date"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_execution_comparison(comparison: pd.DataFrame) -> dict[str, Any]:
    """Headline counts for the frozen cohort; no return or admission is implied."""
    compared = comparison.loc[comparison["window_status"].eq("compared")]
    lead = pd.to_numeric(compared["disclosure_lead_days"], errors="coerce")
    changed = compared["entry_changed"].astype(bool)
    return {
        "events": len(comparison),
        "issuers": int(comparison["cik"].nunique()),
        "clock_found": int(comparison["clock_found"].sum()),
        "compared": len(compared),
        "status_counts": {
            str(k): int(v) for k, v in comparison["window_status"].value_counts().items()
        },
        "clock_source_counts": {
            str(k): int(v) for k, v in compared["clock_source"].value_counts().items()
        },
        "lead_at_least_1_day": int(lead.ge(1.0).sum()),
        "lead_at_least_3_days": int(lead.ge(3.0).sum()),
        "entry_changed": int(changed.sum()),
        "share_entry_changed": float(changed.mean()) if len(compared) else None,
        "entry_sessions_earlier_counts": {
            str(k): int(v)
            for k, v in compared.loc[changed, "entry_sessions_earlier"]
            .astype("int64")
            .value_counts()
            .sort_index()
            .items()
        },
        "filing_clock_mismatches": int(comparison["filing_clock_matches_catalog"].eq(False).sum()),
        "returns_computed": False,
        "return_admission": False,
    }


def _window(
    announcement_at: pd.Timestamp,
    exchange_mic: str,
    sessions: pd.DataFrame,
    contract: ExecutionContract,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return execution_window(announcement_at.isoformat(), exchange_mic, sessions, contract), None
    except ValueError as error:
        return None, str(error)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise FirstPublicExecutionError(f"{label} lack columns: {missing}")
