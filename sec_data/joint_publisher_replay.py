"""Two event-clock corrections on one causal history, never a return authorization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.clock_layer import FLAG_COLUMNS, NUMERIC_COLUMNS, compare_histories
from sec_data.disclosure_audit import utc_timestamp
from sec_data.event_admission import admission_blockers
from sec_data.publisher_replay import apply_publisher_clock, target_dependencies

VERSION = "joint-publisher-replay-v1"
IMMUTABLE_COLUMNS = [
    "company_id",
    "filing_accession",
    "filing_accepted_at",
    "revenue",
    "gross_profit",
    "gross_margin",
    "period_start",
    "period_end",
    "ytd_filing_accession",
    "ytd_available_at",
]
CASE_FLAGS = [
    "label_financial_replay_ready",
    "label_market_state_ready",
    "label_liquidity_eligible",
    "label_window_evidence_ready",
    "label_scenario_inputs_complete",
    "current_liquidity_status",
]


def validate_joint_intervals(history: pd.DataFrame, cases: list[dict[str, Any]]) -> None:
    """Reject ambiguous targets or an event inside either uncertain publisher minute.

    This bounded protocol has two distinct events. Endpoint invariance can only
    describe the whole minute when no other information arrival lies inside it.
    The intervals concern source-label precision, not first-public certification.
    """
    if len(cases) != 2:
        raise ValueError("The joint protocol requires exactly two publisher events")
    for keys in (("source_id",), ("cik", "target_filing_accession"), ("validation_event_id",)):
        if len({tuple(c["request"][k] for k in keys) for c in cases}) != len(cases):
            raise ValueError("Duplicate joint source or event")
    intervals = sorted(utc_timestamp(c["request"]["source_reported_at_utc"]) for c in cases)
    if intervals[1] < intervals[0] + pd.Timedelta(minutes=1):
        raise ValueError("Overlapping publisher minutes require a different interval protocol")
    for case in cases:
        req = case["request"]
        lower = utc_timestamp(req["source_reported_at_utc"])
        selected = history.company_id.eq(req["cik"]) & history.filing_accession.eq(
            req["target_filing_accession"]
        )
        if selected.sum() != 1:
            raise ValueError("Joint target missing or ambiguous")
        actual = pd.to_datetime(history.actual_available_at, utc=True)
        if ((~selected) & actual.ge(lower) & actual.lt(lower + pd.Timedelta(minutes=1))).any():
            raise ValueError("Other event inside uncertain publisher minute")


def apply_joint_clocks(
    history: pd.DataFrame, cases: list[dict[str, Any]], endpoints: dict[str, str]
) -> pd.DataFrame:
    """Apply both availability corrections before recomputing any feature or error."""
    validate_joint_intervals(history, cases)
    sources = {c["request"]["source_id"] for c in cases}
    if set(endpoints) != sources or any(v not in {"start", "end"} for v in endpoints.values()):
        raise ValueError("Every source needs an explicit minute endpoint")
    result = history.copy(deep=True)
    for case in cases:
        request, spec = case["request"], case["spec"]
        stamp = utc_timestamp(request["source_reported_at_utc"])
        if endpoints[request["source_id"]] == "end":
            stamp += pd.Timedelta(minutes=1) - pd.Timedelta(microseconds=1)
        result = apply_publisher_clock(result, request, spec, stamp)
    assert_originals_preserved(history, result)
    return result


def assert_originals_preserved(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Keep exact accounting values, accession/acceptance and YTD dependencies intact."""
    if before.event_id.duplicated().any() or after.event_id.duplicated().any():
        raise ValueError("Duplicate historical event")
    left = before.set_index("event_id").sort_index()[IMMUTABLE_COLUMNS]
    right = after.set_index("event_id").sort_index()[IMMUTABLE_COLUMNS]
    pd.testing.assert_frame_equal(left, right)


def require_endpoint_invariance(reference: pd.DataFrame, scenario: pd.DataFrame) -> dict[str, int]:
    """Fail closed if minute endpoints alter a numerical representation or readiness flag."""
    counts, _ = compare_histories(reference, scenario)
    if any(counts[k] for k in (*NUMERIC_COLUMNS, *FLAG_COLUMNS, "gm_yoy_lag_event_id")):
        raise ValueError("Joint minute endpoints change signals; interval-aware study required")
    return counts


def compare_error_histories(
    independent: pd.DataFrame, joint: pd.DataFrame, event_id: str
) -> tuple[dict[str, int], pd.DataFrame]:
    """Audit membership and values separately; unchanged counts need not mean same sigma."""
    a = target_dependencies(independent, event_id).set_index("event_id")
    b = target_dependencies(joint, event_id).set_index("event_id")
    joined = a.join(b, how="outer", lsuffix="_independent", rsuffix="_joint")
    counts = {}
    for flag, label in (("in_training", "training"), ("in_prior_oos_errors", "prior_errors")):
        left, right = joined[f"{flag}_independent"].eq(True), joined[f"{flag}_joint"].eq(True)
        counts[f"{label}_added"] = int((~left & right).sum())
        counts[f"{label}_removed"] = int((left & ~right).sum())
    shared = joined.in_prior_oos_errors_independent.eq(True) & joined.in_prior_oos_errors_joint.eq(
        True
    )
    changed = ~np.isclose(
        joined.gm_innovation_raw_independent.to_numpy(dtype=float, na_value=np.nan),
        joined.gm_innovation_raw_joint.to_numpy(dtype=float, na_value=np.nan),
        atol=1e-10,
        rtol=0,
        equal_nan=True,
    )
    joined["shared_prior_error_changed"] = shared & changed
    counts["shared_prior_error_values_changed"] = int(joined.shared_prior_error_changed.sum())
    return counts, joined.reset_index()


def consolidate_joint_admission(
    case_ledgers: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Merge only case-specific checks after every ledger used the SAME joint history.

    Refuse different common metadata, duplicate cases, or any attempted admission.
    Independent histories must never be spliced into a nominally joint sample.
    """
    if len(case_ledgers) != 2 or len({k for k, _ in case_ledgers}) != 2:
        raise ValueError("Expected two distinct joint case ledgers")
    result = case_ledgers[0][1].copy(deep=True)
    dynamic = {*CASE_FLAGS, "blockers", "admission_status"}
    common = [k for k in result if k not in dynamic]
    for event_id, frame in case_ledgers:
        if frame.validation_event_id.duplicated().any() or list(frame.validation_event_id) != list(
            result.validation_event_id
        ):
            raise ValueError("Joint admission cohort changed")
        if set(frame) != set(result):
            raise ValueError("Joint admission columns differ")
        pd.testing.assert_frame_equal(result[common], frame[common])
        if (
            frame[
                [
                    "model_admission",
                    "return_admission",
                    "trading_admission",
                    "first_public_certified",
                ]
            ]
            .ne(False)
            .any()
            .any()
        ):
            raise ValueError("Joint sensitivity cannot authorize model/return/trading admission")
        target = frame.validation_event_id.eq(event_id)
        if target.sum() != 1 or frame.loc[~target, CASE_FLAGS[:-1]].eq(True).any().any():
            raise ValueError("Case checks leaked into another event")
        result.loc[target, CASE_FLAGS] = frame.loc[target, CASE_FLAGS]
    result["blockers"] = [admission_blockers(r) for r in result.to_dict("records")]
    result["admission_status"] = np.where(
        result.label_scenario_inputs_complete,
        "joint_label_scenario_prepared_not_admitted",
        "blocked_dependencies_not_rebuilt",
    )
    result["rebuild_scope"] = "one_shared_history_two_publisher_label_sensitivities"
    return result
