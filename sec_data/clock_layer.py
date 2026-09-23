"""Evidence-bound acceptance proxies and a disarmed, clock-only research replay.

This is retrospective source reconciliation, not a claim that corrected clocks
or replayed predictions were captured live. SEC acceptance is not first-public time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.event_cleaning import derive_point_in_time_yoy_lags
from sec_data.pilot_filing_features import (
    FORECAST_FEATURES,
    PilotFilingFeatureConfig,
    attach_company_history_features,
    attach_oos_innovations,
)

VERSION = "sec-clock-layer-v1"
CORRECTED = "source_label_conflict_resolved_by_header_and_index"
TIME_COLUMNS = (
    "filing_accepted_at",
    "actual_available_at",
    "ytd_available_at",
    "gm_yoy_lag_available_at",
    "company_gm_lag1_available_at",
    "company_trailing_gm_history_end",
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_generated_at",
    "forecast_error_history_end",
)
NUMERIC_COLUMNS = (
    "gm_level",
    "gm_delta_yoy",
    *FORECAST_FEATURES,
    "gm_forecast",
    "forecast_sigma",
    "gm_innovation_raw",
    "gm_innovation_std",
    "forecast_training_observations",
    "forecast_sigma_observations",
)
FLAG_COLUMNS = (
    "analysis_ready_level",
    "analysis_ready_delta",
    "analysis_ready_innovation_raw",
    "analysis_ready_innovation_std",
)
HISTORY_COLUMNS = list(
    dict.fromkeys(
        [
            "event_id",
            "company_id",
            "ticker",
            "filing_accession",
            "fiscal_year",
            "fiscal_quarter",
            "period_start",
            "period_end",
            "filing_form",
            "ytd_filing_accession",
            "q4_source_method",
            "revenue",
            "gross_profit",
            "gross_margin",
            "gm_yoy_lag_event_id",
            "forecast_model_version",
            *TIME_COLUMNS,
            *NUMERIC_COLUMNS,
            *FLAG_COLUMNS,
        ]
    )
)


def build_clock_layer(
    catalog: pd.DataFrame,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind reviewed clocks to exact original catalog rows, preserving uncertainty."""
    if catalog.duplicated(["cik", "accession_number"]).any():
        raise ValueError("Duplicate catalog identity")
    lookup = catalog.set_index(["cik", "accession_number"])
    seen = set()
    result = []
    for row in evidence:
        key = (row["cik"], row["accession_number"])
        if key in seen or key not in lookup.index:
            raise ValueError("Duplicate or absent evidence identity")
        seen.add(key)
        original = lookup.loc[key]
        old = utc_timestamp(row["catalog_accepted_at"])
        if (
            original["form"] != row["form"]
            or utc_timestamp(original["accepted_at"]) != old
            or original["source_zip_sha256"] != row["source_zip_sha256"]
            or original["source_member"] != row["source_member"]
            or int(original["source_row_index"]) != row["source_row_index"]
        ):
            raise ValueError("Evidence no longer matches original catalog")
        status = row["resolution_status"]
        proxy = None
        if status in {CORRECTED, "three_representations_consistent", "consistent_header_control"}:
            if (
                row["catalog_reproduces_literal"] is not True
                or utc_timestamp(row["raw_acceptance_literal"]) != old
                or utc_timestamp(row["literal_interpreted_as_labelled_utc"]) != old
            ):
                raise ValueError("Explicit source literal is not reproduced")
            header = utc_timestamp(row["header_accepted_at_utc"])
            if status == "consistent_header_control":
                if header != old:
                    raise ValueError("Control header disagrees with catalog")
                proxy = old.isoformat()
            else:
                if (
                    utc_timestamp(row["index_accepted_at_utc"]) != header
                    or utc_timestamp(row["reviewed_sec_acceptance_proxy_at"]) != header
                    or (header != old) != (status == CORRECTED)
                ):
                    raise ValueError("Header/index reconciliation is inconsistent")
                proxy = header.isoformat()
        result.append(
            {
                **row,
                "clock_layer_version": VERSION,
                "experimental_acceptance_proxy_at": proxy,
                "clock_changed": proxy is not None and utc_timestamp(proxy) != old,
                "overlay_applied": status == CORRECTED,
                "application_scope": "isolated_research_view_only",
                "first_public_at": None,
                "public_available_by": None,
                "model_admission": False,
                "trading_admission": False,
            }
        )
    return result


def apply_history_clocks(
    history: pd.DataFrame,
    layer: list[dict[str, Any]],
) -> pd.DataFrame:
    """Update accession-bound aliases and Q4 source clocks, never timestamp-match blindly."""
    result = history.copy(deep=True)
    if (
        result.event_id.duplicated().any()
        or result.duplicated(["company_id", "filing_accession"]).any()
    ):
        raise ValueError("History identity is not unique")
    for column in TIME_COLUMNS:
        result[column] = pd.to_datetime(
            result[column].map(lambda x: utc_timestamp(x) if pd.notna(x) else pd.NaT),
            utc=True,
        )
    seen = set()
    for row in layer:
        key = (row["cik"], row["accession_number"])
        if key in seen:
            raise ValueError("Duplicate clock layer identity")
        seen.add(key)
        primary = result.company_id.eq(key[0]) & result.filing_accession.eq(key[1])
        ytd = result.company_id.eq(key[0]) & result.ytd_filing_accession.eq(key[1])
        if not primary.any() and not ytd.any():
            continue
        if row["experimental_acceptance_proxy_at"] is None:
            raise ValueError("Unresolved clock dependency cannot enter replay")
        old = utc_timestamp(row["catalog_accepted_at"])
        new = utc_timestamp(row["experimental_acceptance_proxy_at"])
        for field in ("filing_accepted_at", "actual_available_at"):
            if not result.loc[primary, field].eq(old).all():
                raise ValueError("History is not an exact acceptance-proxy alias")
            result.loc[primary, field] = new
        if not result.loc[ytd, "ytd_available_at"].eq(old).all():
            raise ValueError("Q4 dependency clock differs from catalog")
        result.loc[ytd, "ytd_available_at"] = new
    derived = result.ytd_filing_accession.notna()
    if (
        result.loc[derived, "ytd_available_at"].isna().any()
        or not (
            result.loc[derived, "ytd_available_at"] < result.loc[derived, "actual_available_at"]
        ).all()
    ):
        raise ValueError("Q4 source is no longer strictly earlier; reselect facts before replay")
    return result


def replay_history(history: pd.DataFrame, config: PilotFilingFeatureConfig) -> pd.DataFrame:
    """Rebuild all lag joins, fixed-spec OOS forecasts and prior-error sigma.

    Original accounting values/event selection are frozen. This is not a new
    fact-selection pass; a Q4 dependency ordering failure must stop upstream.
    """
    result = apply_history_clocks(history, []).reset_index(drop=True)
    for col in ("gm_yoy_lag", "gm_delta_yoy"):
        result[col] = np.nan
    result["gm_yoy_lag_event_id"] = None
    result["gm_yoy_lag_available_at"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["analysis_ready_delta"] = False
    result = derive_point_in_time_yoy_lags(result)
    result = attach_company_history_features(result, config)
    result = attach_oos_innovations(result, config)
    audit_replay_chronology(result)
    return result


def audit_replay_chronology(history: pd.DataFrame) -> dict[str, int]:
    """Fail on same-time/future features, training, generation or OOS errors."""
    actual = pd.to_datetime(history.actual_available_at, utc=True)
    for field in (
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_generated_at",
        "forecast_error_history_end",
    ):
        stamps = pd.to_datetime(history[field], utc=True)
        if (stamps.notna() & (stamps >= actual)).any():
            raise ValueError(f"Non-prior replay clock: {field}")
    ready = history.gm_forecast.notna()
    generated = pd.to_datetime(history.forecast_generated_at, utc=True)
    for field in ("forecast_feature_cutoff", "forecast_training_end"):
        stamps = pd.to_datetime(history[field], utc=True)
        if (ready & (stamps.isna() | generated.isna() | (stamps > generated))).any():
            raise ValueError("Forecast generation precedes a dependency")
    standardized = history.gm_innovation_std.notna()
    sigma_end = pd.to_datetime(history.forecast_error_history_end, utc=True)
    if (standardized & (sigma_end.isna() | history.forecast_sigma.le(0))).any():
        raise ValueError("Invalid prior-OOS sigma")
    return {"forecasts": int(ready.sum()), "standardized": int(standardized.sum()), "violations": 0}


def compare_histories(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Compare the identical event sample; report float changes above explicit tolerance."""
    if tolerance < 0 or before.event_id.duplicated().any() or after.event_id.duplicated().any():
        raise ValueError("Invalid comparison inputs")
    old = before.set_index("event_id").sort_index()
    new = after.set_index("event_id").sort_index()
    if not old.index.equals(new.index):
        raise ValueError("Replay changed event sample")
    counts = {}
    changes = []
    for field in (*TIME_COLUMNS, *NUMERIC_COLUMNS, *FLAG_COLUMNS, "gm_yoy_lag_event_id"):
        left, right = old[field], new[field]
        if field in TIME_COLUMNS:
            left, right = pd.to_datetime(left, utc=True), pd.to_datetime(right, utc=True)
        if field in NUMERIC_COLUMNS:
            different = pd.Series(
                ~np.isclose(
                    pd.to_numeric(left).to_numpy(dtype=float, na_value=np.nan),
                    pd.to_numeric(right).to_numpy(dtype=float, na_value=np.nan),
                    atol=tolerance,
                    rtol=0,
                    equal_nan=True,
                ),
                index=old.index,
            )
        else:
            different = ~(left.eq(right).fillna(False) | (left.isna() & right.isna()))
        counts[field] = int(different.sum())
        for event_id in old.index[different]:
            changes.append(
                {
                    "event_id": event_id,
                    "cik": old.at[event_id, "company_id"],
                    "accession_number": old.at[event_id, "filing_accession"],
                    "field": field,
                    "old": left.loc[event_id],
                    "new": right.loc[event_id],
                }
            )
    return counts, pd.DataFrame(
        changes, columns=["event_id", "cik", "accession_number", "field", "old", "new"]
    )


def calendar_windows(
    timestamp: str,
    exchange: str,
    sessions: pd.DataFrame,
    *,
    holding_sessions: int = 20,
) -> dict[str, Any]:
    """Rebuild daily-price proxy brackets and two explicitly distinct entry conventions.

    Conservative contract: second session strictly AFTER the New York event date.
    Legacy implementation: second regular open strictly AFTER the timestamp; for
    pre-open news this counts the same-day open. Report both, never conflate them.
    """
    if holding_sessions < 1:
        raise ValueError("Holding sessions must be positive")
    event = utc_timestamp(timestamp)
    date = event.tz_convert("America/New_York").date().isoformat()
    frame = sessions.loc[sessions.exchange_mic.eq(exchange)].copy()
    if frame.empty or frame.session_date.duplicated().any():
        raise ValueError("Missing or duplicated exchange schedule")
    frame = frame.sort_values("session_date")
    for field in ("regular_open_at", "regular_close_at"):
        frame[field] = frame[field].map(utc_timestamp)
    if (frame.regular_open_at >= frame.regular_close_at).any():
        raise ValueError("Invalid session boundaries")
    if not frame.session_date.min() <= date <= frame.session_date.max():
        raise ValueError("Event outside frozen calendar")
    same = frame.loc[frame.session_date.eq(date)]
    relation = "non_session"
    if not same.empty:
        day = same.iloc[0]
        relation = (
            "pre_open"
            if event < day.regular_open_at
            else "in_session"
            if event < day.regular_close_at
            else "at_or_after_close"
        )
    result: dict[str, Any] = {"local_date": date, "session_relation": relation}
    for label, mask in {
        "legacy_second_future_open": frame.regular_open_at > event,
        "second_session_after_local_date": frame.session_date > date,
    }.items():
        future = frame.loc[mask].reset_index(drop=True)
        if len(future) < holding_sessions + 1:
            raise ValueError("Insufficient calendar for complete holding window")
        result[label] = {
            "entry_at": future.iloc[1].regular_open_at.isoformat(),
            "entry_price_type": "open_unadjusted",
            "exit_at": future.iloc[holding_sessions].regular_close_at.isoformat(),
            "exit_price_type": "close_unadjusted",
            "holding_sessions_inclusive": holding_sessions,
            "entry_calendar_sha256": future.iloc[1].source_snapshot_sha256,
            "exit_calendar_sha256": future.iloc[holding_sessions].source_snapshot_sha256,
        }
    previous = frame.loc[frame.regular_close_at < event]
    following = frame.loc[frame.regular_close_at > event]
    if previous.empty or following.empty:
        raise ValueError("Missing close-to-close bracket")
    result["daily_close_bracket"] = {
        "start_at": previous.iloc[-1].regular_close_at.isoformat(),
        "end_at": following.iloc[0].regular_close_at.isoformat(),
        "price_type": "close_unadjusted",
        "isolated_event_reaction": False,
    }
    result["prices_loaded"] = False
    result["trading_admission"] = False
    return result
