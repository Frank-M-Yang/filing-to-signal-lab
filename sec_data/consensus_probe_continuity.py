"""Compare adjacent capture attempts without manufacturing independent information."""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import Any, cast

import pandas as pd

from sec_data.alpha_vantage_consensus import AlphaVantageConsensusError

REVENUE_FIELDS = (
    "revenue_estimate_average",
    "revenue_estimate_low",
    "revenue_estimate_high",
    "revenue_estimate_analyst_count",
)
PAIR_COLUMNS = [
    "snapshot_pair_id",
    "coverage_probe_target_id",
    "ticker",
    "target_period_end",
    "previous_captured_at",
    "current_captured_at",
    "elapsed_hours",
    "distinct_utc_dates",
    "spans_24_hours",
    "previous_source_sha256",
    "current_source_sha256",
    "response_hash_changed",
    "previous_response_status",
    "current_response_status",
    "previous_quarter_status",
    "current_quarter_status",
    "comparison_status",
    "aggregate_fields_comparable",
    "mean_changed",
    "range_changed",
    "analyst_count_changed",
    "source_occurrence_changed",
    "aggregate_mean_revision_fraction",
    "analyst_membership_identity_verified",
    "independent_information_verified",
    "formal_model_admission",
    *[f"{side}_{field}" for side in ("previous", "current") for field in REVENUE_FIELDS],
    "previous_source_occurrence",
    "current_source_occurrence",
]


def compare_coverage_probe_snapshots(
    attempts: pd.DataFrame,
    observations: pd.DataFrame,
    rejections: pd.DataFrame,
) -> pd.DataFrame:
    """Compare consecutive attempts for each issuer and the union of their quarters.

    Inputs must come from the raw-byte-verified coverage audit. An error, missing
    period or duplicate source quarter blocks that pair; it is never skipped to
    compare two convenient valid endpoints. A 24-hour span is descriptive only,
    not an admission rule or evidence of a new analyst information set.
    """
    if attempts.empty:
        if not observations.empty or not rejections.empty:
            raise AlphaVantageConsensusError("Quarter rows lack capture attempts")
        return pd.DataFrame(columns=PAIR_COLUMNS)
    timeline = attempts.copy()
    timeline["_time"] = timeline["captured_at"].map(_aware_time)
    if timeline.duplicated(["coverage_probe_target_id", "_time"]).any():
        raise AlphaVantageConsensusError("Ambiguous simultaneous probe attempts")
    timeline_rows = cast(list[dict[str, Any]], timeline.to_dict("records"))
    for row in timeline_rows:
        if row["_time"].date().isoformat() != row["capture_date"]:
            raise AlphaVantageConsensusError("Continuity capture date is inconsistent")
    endpoint_index = {(row["coverage_probe_target_id"], row["_time"]): row for row in timeline_rows}
    fields: dict[tuple[str, pd.Timestamp], dict[str, dict[int, dict[str, Any]]]] = {}
    # A rejected parser row may also occur in the observation table. Reconcile
    # only that identical physical occurrence, never duplicate quarter rows.
    for frame, is_rejection in ((observations, False), (rejections, True)):
        seen: set[tuple[str, pd.Timestamp, str, int]] = set()
        for row in cast(list[dict[str, Any]], frame.to_dict("records")):
            time = _aware_time(row["captured_at"])
            key = (str(row["coverage_probe_target_id"]), time)
            attempt = endpoint_index.get(key)
            if (
                attempt is None
                or attempt["status"] != "ok"
                or attempt["source_sha256"] != row["source_sha256"]
                or attempt["ticker"] != row["ticker"]
            ):
                raise AlphaVantageConsensusError("Quarter row does not match its capture")
            period = str(row["target_period_end"])
            period_time = pd.Timestamp(period)
            if period_time.tzinfo is not None or period_time.date().isoformat() != period:
                raise AlphaVantageConsensusError("Noncanonical probe quarter date")
            occurrence = int(row["source_occurrence"])
            physical_key = (*key, period, occurrence)
            if physical_key in seen:
                raise AlphaVantageConsensusError("Duplicated physical source occurrence")
            seen.add(physical_key)
            quarter = fields.setdefault(key, {}).setdefault(period, {})
            if occurrence in quarter:
                if not is_rejection or quarter[occurrence]["aggregate_fields_valid"]:
                    raise AlphaVantageConsensusError("Conflicting probe field evidence")
                quarter[occurrence]["candidate_rejection_reason"] = row["field_rejection_reason"]
            else:
                quarter[occurrence] = {
                    **row,
                    "aggregate_fields_valid": False
                    if is_rejection
                    else bool(row["aggregate_fields_valid"]),
                    "candidate_rejection_reason": row["field_rejection_reason"]
                    if is_rejection
                    else row["candidate_rejection_reason"],
                }

    pairs: list[dict[str, Any]] = []
    for target_id, group in timeline.groupby("coverage_probe_target_id", sort=True):
        ordered = cast(list[dict[str, Any]], group.sort_values("_time").to_dict("records"))
        if len({row["ticker"] for row in ordered}) != 1:
            raise AlphaVantageConsensusError("Ticker changed inside a probe identity")
        for previous, current in pairwise(ordered):
            before = fields.get((str(target_id), previous["_time"]), {})
            after = fields.get((str(target_id), current["_time"]), {})
            periods: list[str | None] = list(sorted(set(before) | set(after))) or [None]
            for pair_period in periods:
                pairs.append(_compare_pair(previous, current, pair_period, before, after))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _compare_pair(
    previous: dict[str, Any],
    current: dict[str, Any],
    period: str | None,
    before: dict[str, dict[int, dict[str, Any]]],
    after: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    elapsed_hours = (current["_time"] - previous["_time"]).total_seconds() / 3600
    pair_id = hashlib.sha256(
        "|".join(
            str(value)
            for value in (
                previous["coverage_probe_target_id"],
                previous["_time"].isoformat(),
                current["_time"].isoformat(),
            )
        ).encode()
    ).hexdigest()
    result: dict[str, Any] = {
        "snapshot_pair_id": pair_id,
        "coverage_probe_target_id": previous["coverage_probe_target_id"],
        "ticker": previous["ticker"],
        "target_period_end": period,
        "previous_captured_at": previous["_time"].isoformat(),
        "current_captured_at": current["_time"].isoformat(),
        "elapsed_hours": elapsed_hours,
        "distinct_utc_dates": previous["capture_date"] != current["capture_date"],
        "spans_24_hours": elapsed_hours >= 24,
        "response_hash_changed": None,
        "comparison_status": "not_comparable",
        "aggregate_fields_comparable": False,
        "analyst_membership_identity_verified": False,
        "independent_information_verified": False,
        "formal_model_admission": False,
    }
    if pd.notna(previous["source_sha256"]) and pd.notna(current["source_sha256"]):
        result["response_hash_changed"] = previous["source_sha256"] != current["source_sha256"]
    rows = []
    for side, attempt, quarters in (("previous", previous, before), ("current", current, after)):
        status, row = _endpoint(attempt, period, quarters)
        result[f"{side}_quarter_status"] = status
        result[f"{side}_response_status"] = attempt["status"]
        result[f"{side}_source_sha256"] = attempt["source_sha256"]
        for field in (*REVENUE_FIELDS, "source_occurrence"):
            result[f"{side}_{field}"] = row.get(field) if row else None
        rows.append(row)
    if result["previous_quarter_status"] == result["current_quarter_status"] == "valid_fields":
        first, second = rows
        assert first is not None and second is not None
        result.update(
            aggregate_fields_comparable=True,
            mean_changed=first[REVENUE_FIELDS[0]] != second[REVENUE_FIELDS[0]],
            range_changed=any(first[field] != second[field] for field in REVENUE_FIELDS[1:3]),
            analyst_count_changed=first[REVENUE_FIELDS[3]] != second[REVENUE_FIELDS[3]],
            source_occurrence_changed=first["source_occurrence"] != second["source_occurrence"],
            aggregate_mean_revision_fraction=(
                second[REVENUE_FIELDS[0]] / first[REVENUE_FIELDS[0]] - 1
            ),
        )
        result["comparison_status"] = (
            "aggregate_fields_changed"
            if any(
                result[field]
                for field in ("mean_changed", "range_changed", "analyst_count_changed")
            )
            else "aggregate_fields_unchanged"
        )
    return result


def _endpoint(
    attempt: dict[str, Any],
    period: str | None,
    quarters: dict[str, dict[int, dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None]:
    if attempt["status"] != "ok":
        return "response_unavailable", None
    if period is None:
        return "no_future_quarter_rows", None
    rows = list(quarters.get(period, {}).values())
    if len(rows) > 1:
        return "duplicate_quarter_rows", None
    if not rows:
        return (
            "period_ended_before_capture"
            if period < attempt["capture_date"]
            else "period_not_observed_as_future",
            None,
        )
    row = rows[0]
    return (
        "valid_fields" if row["aggregate_fields_valid"] else row["candidate_rejection_reason"],
        row,
    )


def _aware_time(value: Any) -> pd.Timestamp:
    time = pd.Timestamp(value)
    if pd.isna(time) or time.tzinfo is None:
        raise AlphaVantageConsensusError("Probe continuity requires an aware capture time")
    return time.tz_convert("UTC")
