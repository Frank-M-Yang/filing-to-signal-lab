"""Build an auditable, monotone lifecycle for frozen prospective events.

The state machine is deliberately outcome-neutral.  It records which evidence is
available, why an event is waiting, and whether an event must be rejected.  It
does not score returns, size positions, or authorize live trading.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

PROSPECTIVE_EVENT_STATE_MACHINE_VERSION: Final[str] = "prospective-event-state-machine-v1"

REGISTERED: Final[str] = "registered"
CONSENSUS_LEVEL_READY: Final[str] = "consensus_level_ready"
CONSENSUS_REVISION_READY: Final[str] = "consensus_revision_ready"
FILING_OBSERVED: Final[str] = "filing_observed"
CONSENSUS_FROZEN: Final[str] = "consensus_frozen"
ACTUAL_REVIEW_REQUIRED: Final[str] = "actual_review_required"
ACTUAL_READY: Final[str] = "actual_ready"
INNOVATION_READY: Final[str] = "innovation_ready"
PRICE_WINDOW_READY: Final[str] = "price_window_ready"
PAPER_DECISION_READY: Final[str] = "paper_decision_ready"
PAPER_POSITION_OPEN: Final[str] = "paper_position_open"
PAPER_POSITION_CLOSED: Final[str] = "paper_position_closed"
REJECTED: Final[str] = "rejected"

_READY_ACTUAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"direct_quarterly_fact_ready", "derived_q4_fy_minus_prior_9m_ready"}
)
_STATE_RANK: Final[dict[str, int]] = {
    REGISTERED: 0,
    CONSENSUS_LEVEL_READY: 1,
    CONSENSUS_REVISION_READY: 2,
    FILING_OBSERVED: 3,
    CONSENSUS_FROZEN: 4,
    ACTUAL_REVIEW_REQUIRED: 5,
    ACTUAL_READY: 6,
    INNOVATION_READY: 7,
    PRICE_WINDOW_READY: 8,
    PAPER_DECISION_READY: 9,
    PAPER_POSITION_OPEN: 10,
    PAPER_POSITION_CLOSED: 11,
    REJECTED: 99,
}


class ProspectiveEventStateMachineError(FilingCatalogError):
    """Raised when event evidence violates a lifecycle or point-in-time clock."""


@dataclass(frozen=True)
class ProspectiveEventStateMachineTables:
    """Current event states, reached transitions, state catalog, gates, and report."""

    event_states: pd.DataFrame
    transitions: pd.DataFrame
    state_catalog: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveEventStateMachineResult:
    """One immutable state-machine audit artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_event_state_machine(
    event_readiness: pd.DataFrame,
    consensus_snapshots: pd.DataFrame,
    actual_observations: pd.DataFrame,
    actual_snapshots: pd.DataFrame,
    innovation_events: pd.DataFrame,
    consensus_quality: pd.DataFrame,
    previous_states: pd.DataFrame | None = None,
) -> ProspectiveEventStateMachineTables:
    """Resolve the highest justified state for every frozen prospective event.

    The function consumes only timestamped evidence.  Later-stage execution states
    are reserved in the catalog but remain unreachable in v1 because price and
    paper-order evidence are intentionally outside this first implementation.
    """
    _require_columns(
        event_readiness,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "registered_at",
            "consensus_snapshot_rows",
            "consensus_capture_dates",
            "earliest_consensus_captured_at",
            "latest_consensus_captured_at",
            "latest_consensus_revenue",
            "revision_ready",
            "future_accession_number",
            "accepted_at",
            "future_filing_first_observed_at",
            "future_filing_observed",
        },
        "prospective event readiness",
    )
    _require_columns(
        consensus_snapshots,
        {
            "prospective_event_id",
            "captured_at",
            "capture_date",
            "revenue_estimate_average",
        },
        "prospective consensus snapshots",
    )
    _require_columns(
        actual_observations,
        {
            "forward_actual_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "accepted_at",
            "actual_status",
            "actual_revenue",
            "actual_available_at",
            "source_snapshot_sha256",
            "historical_backfill_admission",
            "return_values_loaded",
        },
        "forward actual observations",
    )
    _require_columns(
        actual_snapshots,
        {"forward_actual_target_id", "fetched_at", "sha256"},
        "forward actual snapshots",
    )
    _require_columns(
        innovation_events,
        {
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "actual_status",
            "raw_innovation_ready",
            "revenue_innovation_pct",
            "consensus_precedes_actual",
            "join_status",
            "historical_backfill_admission",
            "return_values_loaded",
        },
        "forward innovation events",
    )
    _require_columns(
        consensus_quality,
        {
            "prospective_event_id",
            "quality_status",
            "quality_reasons",
            "downstream_admission",
            "manual_review_required",
            "quarantined",
        },
        "prospective consensus quality",
    )
    events = _prepare_events(event_readiness)
    consensus = _prepare_consensus(consensus_snapshots)
    actuals = _prepare_actuals(actual_observations, actual_snapshots)
    innovations = _prepare_innovations(innovation_events)
    _enforce_outcome_firewall(actuals, innovations)

    second_capture = _second_independent_capture(consensus)
    events = events.merge(
        second_capture, on="prospective_event_id", how="left", validate="one_to_one"
    )
    events = _attach_actuals(events, actuals)
    events = _attach_innovations(events, innovations)
    events = _attach_quality(events, consensus_quality)
    states, transitions = _resolve_states(events)
    cross_run_passed = _validate_previous_states(states, previous_states)
    transition_passed = _validate_transitions(transitions)
    state_catalog = _state_catalog()
    gates = pd.DataFrame(
        [
            _gate(
                "frozen_event_coverage",
                len(states),
                len(event_readiness),
                len(states) == len(event_readiness),
            ),
            _gate(
                "unique_atomic_event_state",
                int(states["prospective_event_id"].nunique()),
                len(states),
                not states.duplicated("prospective_event_id").any(),
            ),
            _gate("monotone_transition_history", len(states), len(states), transition_passed),
            _gate("cross_run_non_regression", len(states), len(states), cross_run_passed),
            _gate(
                "no_return_values_loaded",
                int(states["return_values_loaded"].sum()),
                0,
                not states["return_values_loaded"].any(),
            ),
            _gate(
                "consensus_quality_evidence_coverage",
                int(states["consensus_quality_status"].notna().sum()),
                len(states),
                bool(states["consensus_quality_status"].notna().all()),
            ),
            _gate(
                "no_paper_positions_open",
                int(states["paper_position_open"].sum()),
                0,
                not states["paper_position_open"].any(),
            ),
            _gate("live_trading_not_authorized", 0, 0, True),
        ]
    )
    state_counts = {
        str(key): int(value)
        for key, value in states["current_state"].value_counts().sort_index().items()
    }
    as_of_candidates = pd.concat(
        [
            states["state_entered_at"],
            consensus["captured_at"]
            if not consensus.empty
            else pd.Series(dtype="datetime64[ns, UTC]"),
        ],
        ignore_index=True,
    ).dropna()
    report: dict[str, Any] = {
        "contract_version": PROSPECTIVE_EVENT_STATE_MACHINE_VERSION,
        "events": len(states),
        "issuers": int(states["cik"].nunique()),
        "state_counts": state_counts,
        "transition_rows": len(transitions),
        "rejected_events": int(states["rejected"].sum()),
        "waiting_events": int(states["waiting"].sum()),
        "innovation_ready_events": int(states["innovation_ready"].sum()),
        "consensus_quality_status_counts": {
            str(key): int(value)
            for key, value in states["consensus_quality_status"].value_counts().sort_index().items()
        },
        "downstream_admitted_events": int(states["downstream_admission"].sum()),
        "paper_decision_ready_events": 0,
        "paper_positions_open": 0,
        "paper_positions_closed": 0,
        "return_values_loaded": False,
        "live_trading_authorized": False,
        "state_as_of": as_of_candidates.max().isoformat() if not as_of_candidates.empty else None,
        "all_state_gates_passed": bool(gates["passed"].all()),
    }
    return ProspectiveEventStateMachineTables(states, transitions, state_catalog, gates, report)


def build_prospective_event_state_machine(
    readiness_root: Path,
    actual_audit_root: Path,
    innovation_ledger_root: Path,
    consensus_quality_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_event_state_machine"),
    previous_state_root: Path | None = None,
) -> ProspectiveEventStateMachineResult:
    """Persist an immutable state snapshot with complete upstream lineage."""
    manifests = {
        "readiness_manifest_sha256": _sha256_file(readiness_root / "output_manifest.json"),
        "actual_audit_manifest_sha256": _sha256_file(actual_audit_root / "output_manifest.json"),
        "innovation_ledger_manifest_sha256": _sha256_file(
            innovation_ledger_root / "output_manifest.json"
        ),
        "consensus_quality_manifest_sha256": _sha256_file(
            consensus_quality_root / "output_manifest.json"
        ),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    payload = json.dumps(
        {"version": PROSPECTIVE_EVENT_STATE_MACHINE_VERSION, **manifests}, sort_keys=True
    ).encode()
    dataset_version = (
        PROSPECTIVE_EVENT_STATE_MACHINE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ProspectiveEventStateMachineResult(
            target.resolve(), _read_json(target / "prospective_event_state_machine_report.json")
        )
    if target.exists():
        raise ProspectiveEventStateMachineError(f"Incomplete state-machine audit exists: {target}")
    prior_root = previous_state_root or _latest_complete_state_root(output_root)
    previous_states = (
        pd.read_parquet(prior_root / "prospective_event_states.parquet")
        if prior_root is not None
        else None
    )
    tables = calculate_prospective_event_state_machine(
        pd.read_parquet(readiness_root / "prospective_market_innovation_event_readiness.parquet"),
        pd.read_parquet(
            readiness_root / "prospective_market_innovation_consensus_snapshots.parquet"
        ),
        pd.read_parquet(actual_audit_root / "forward_actual_revenue_observations.parquet"),
        pd.read_parquet(actual_audit_root / "forward_actual_revenue_snapshots.parquet"),
        pd.read_parquet(innovation_ledger_root / "forward_innovation_events.parquet"),
        pd.read_parquet(consensus_quality_root / "prospective_consensus_event_quality.parquet"),
        previous_states,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "prospective_event_states.parquet": tables.event_states,
        "prospective_event_state_transitions.parquet": tables.transitions,
        "prospective_event_state_catalog.parquet": tables.state_catalog,
        "prospective_event_state_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "prospective_event_state_machine_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ProspectiveEventStateMachineResult(target.resolve(), report)


def _prepare_events(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame.copy()
    if events.empty or events.duplicated("prospective_event_id").any():
        raise ProspectiveEventStateMachineError("The frozen event registry is empty or non-atomic")
    events["cik"] = events["cik"].astype(str).str.zfill(10)
    events["ticker"] = events["ticker"].astype(str).str.strip().str.upper()
    events["target_period_end"] = _utc(
        events["target_period_end"], "target_period_end"
    ).dt.normalize()
    for column in (
        "registered_at",
        "earliest_consensus_captured_at",
        "latest_consensus_captured_at",
        "accepted_at",
        "future_filing_first_observed_at",
    ):
        events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    observed_filing = events["future_filing_observed"].astype(bool)
    if (
        events.loc[observed_filing, ["accepted_at", "future_filing_first_observed_at"]]
        .isna()
        .any()
        .any()
    ):
        raise ProspectiveEventStateMachineError(
            "An observed filing lacks acceptance or observation time"
        )
    if (
        events.loc[observed_filing, "future_filing_first_observed_at"]
        < events.loc[observed_filing, "accepted_at"]
    ).any():
        raise ProspectiveEventStateMachineError("A filing was observed before SEC acceptance")
    return events


def _prepare_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    consensus = frame.copy()
    consensus["captured_at"] = _utc(consensus["captured_at"], "consensus captured_at")
    consensus["capture_date"] = _utc(
        consensus["capture_date"], "consensus capture_date"
    ).dt.normalize()
    if consensus.duplicated(["prospective_event_id", "captured_at"]).any():
        raise ProspectiveEventStateMachineError("Duplicate prospective consensus clock")
    return consensus


def _prepare_actuals(observations: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    actuals = observations.copy()
    if actuals.empty:
        actuals["actual_observed_at"] = pd.Series(dtype="datetime64[ns, UTC]")
        return actuals
    actuals["cik"] = actuals["cik"].astype(str).str.zfill(10)
    actuals["ticker"] = actuals["ticker"].astype(str).str.strip().str.upper()
    actuals["target_period_end"] = _utc(
        actuals["target_period_end"], "actual target_period_end"
    ).dt.normalize()
    actuals["accepted_at"] = _utc(actuals["accepted_at"], "actual accepted_at")
    actuals["actual_available_at"] = pd.to_datetime(
        actuals["actual_available_at"], utc=True, errors="coerce"
    )
    snapshot_times = snapshots.copy()
    if snapshot_times.empty:
        actuals["actual_observed_at"] = pd.NaT
    else:
        snapshot_times["fetched_at"] = _utc(snapshot_times["fetched_at"], "actual fetched_at")
        snapshot_times = snapshot_times.sort_values("fetched_at", kind="mergesort")
        by_hash = snapshot_times.drop_duplicates("sha256", keep="last")[["sha256", "fetched_at"]]
        actuals = actuals.merge(
            by_hash.rename(
                columns={"sha256": "source_snapshot_sha256", "fetched_at": "actual_observed_at"}
            ),
            on="source_snapshot_sha256",
            how="left",
            validate="many_to_one",
        )
    actuals = actuals.sort_values(
        ["cik", "target_period_end", "accession_number", "actual_observed_at"],
        kind="mergesort",
    ).drop_duplicates(["cik", "target_period_end", "accession_number"], keep="last")
    return actuals


def _prepare_innovations(frame: pd.DataFrame) -> pd.DataFrame:
    innovations = frame.copy()
    if innovations.empty:
        return innovations
    innovations["cik"] = innovations["cik"].astype(str).str.zfill(10)
    innovations["ticker"] = innovations["ticker"].astype(str).str.strip().str.upper()
    innovations["target_period_end"] = _utc(
        innovations["target_period_end"], "innovation target_period_end"
    ).dt.normalize()
    if innovations.duplicated(["cik", "target_period_end", "accession_number"]).any():
        raise ProspectiveEventStateMachineError("Innovation events are not atomic")
    return innovations


def _enforce_outcome_firewall(actuals: pd.DataFrame, innovations: pd.DataFrame) -> None:
    if (
        (not actuals.empty and actuals["historical_backfill_admission"].astype(bool).any())
        or (not actuals.empty and actuals["return_values_loaded"].astype(bool).any())
        or (
            not innovations.empty
            and innovations["historical_backfill_admission"].astype(bool).any()
        )
        or (not innovations.empty and innovations["return_values_loaded"].astype(bool).any())
    ):
        raise ProspectiveEventStateMachineError("The forward state machine outcome firewall failed")


def _second_independent_capture(consensus: pd.DataFrame) -> pd.DataFrame:
    if consensus.empty:
        return pd.DataFrame(
            columns=[
                "prospective_event_id",
                "second_consensus_capture_date",
                "second_consensus_captured_at",
            ]
        )
    dates = consensus.sort_values(
        ["prospective_event_id", "capture_date", "captured_at"], kind="mergesort"
    ).drop_duplicates(["prospective_event_id", "capture_date"], keep="first")
    dates["capture_order"] = dates.groupby("prospective_event_id").cumcount() + 1
    second = dates.loc[
        dates["capture_order"].eq(2), ["prospective_event_id", "capture_date", "captured_at"]
    ]
    return second.rename(
        columns={
            "capture_date": "second_consensus_capture_date",
            "captured_at": "second_consensus_captured_at",
        }
    ).reset_index(drop=True)


def _attach_actuals(events: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    if actuals.empty:
        for column, dtype in {
            "actual_status": "object",
            "actual_revenue": "float64",
            "actual_available_at": "datetime64[ns, UTC]",
            "actual_observed_at": "datetime64[ns, UTC]",
        }.items():
            output[column] = pd.Series(index=output.index, dtype=dtype)
        return output
    selected = actuals[
        [
            "cik",
            "target_period_end",
            "accession_number",
            "actual_status",
            "actual_revenue",
            "actual_available_at",
            "actual_observed_at",
        ]
    ]
    return output.merge(
        selected,
        left_on=["cik", "target_period_end", "future_accession_number"],
        right_on=["cik", "target_period_end", "accession_number"],
        how="left",
        validate="one_to_one",
    ).drop(columns="accession_number")


def _attach_innovations(events: pd.DataFrame, innovations: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    if innovations.empty:
        output["raw_innovation_ready"] = False
        output["revenue_innovation_pct"] = np.nan
        output["consensus_precedes_actual"] = False
        output["innovation_join_status"] = pd.Series(index=output.index, dtype="object")
        return output
    selected = innovations[
        [
            "cik",
            "target_period_end",
            "accession_number",
            "raw_innovation_ready",
            "revenue_innovation_pct",
            "consensus_precedes_actual",
            "join_status",
        ]
    ].rename(columns={"join_status": "innovation_join_status"})
    output = output.merge(
        selected,
        left_on=["cik", "target_period_end", "future_accession_number"],
        right_on=["cik", "target_period_end", "accession_number"],
        how="left",
        validate="one_to_one",
    ).drop(columns="accession_number")
    output["raw_innovation_ready"] = output["raw_innovation_ready"].fillna(False)
    output["consensus_precedes_actual"] = output["consensus_precedes_actual"].fillna(False)
    return output


def _attach_quality(events: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    selected = quality[
        [
            "prospective_event_id",
            "quality_status",
            "quality_reasons",
            "downstream_admission",
            "manual_review_required",
            "quarantined",
        ]
    ].copy()
    if selected.duplicated("prospective_event_id").any():
        raise ProspectiveEventStateMachineError("Consensus quality evidence is not atomic")
    if set(selected["prospective_event_id"].astype(str)) != set(
        events["prospective_event_id"].astype(str)
    ):
        raise ProspectiveEventStateMachineError(
            "Consensus quality does not cover the frozen registry"
        )
    return events.merge(selected, on="prospective_event_id", how="left", validate="one_to_one")


def _resolve_states(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for raw_event in events.sort_values("registration_order", kind="mergesort").to_dict("records"):
        event = {str(key): value for key, value in raw_event.items()}
        state, event_transitions = _resolve_event(event)
        state_rows.append(state)
        transition_rows.extend(event_transitions)
    states = pd.DataFrame(state_rows).sort_values("registration_order").reset_index(drop=True)
    transitions = pd.DataFrame(transition_rows, columns=_transition_columns())
    return states, transitions


def _resolve_event(event: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_id = str(event["prospective_event_id"])
    registered_at = _required_timestamp(event["registered_at"], "registered_at")
    transitions: list[dict[str, Any]] = []
    _append_transition(
        transitions, event_id, None, REGISTERED, registered_at, registered_at, "frozen_registry"
    )
    current = REGISTERED
    state_entered_at = registered_at
    state_effective_at = registered_at
    blocking_reason = "consensus_snapshot_missing"
    next_state: str | None = CONSENSUS_LEVEL_READY
    rejection_reason: str | None = None

    snapshot_ready = int(event["consensus_snapshot_rows"]) >= 1
    revision_ready = bool(event["revision_ready"])
    filing_observed = bool(event["future_filing_observed"])
    earliest = _optional_timestamp(event["earliest_consensus_captured_at"])
    latest = _optional_timestamp(event["latest_consensus_captured_at"])
    accepted_at = _optional_timestamp(event["accepted_at"])
    filing_seen_at = _optional_timestamp(event["future_filing_first_observed_at"])
    frozen_consensus_ready = bool(
        filing_observed and accepted_at is not None and latest is not None and latest < accepted_at
    )
    if snapshot_ready:
        if earliest is None:
            raise ProspectiveEventStateMachineError("A level-ready event lacks a capture clock")
        _append_transition(
            transitions,
            event_id,
            current,
            CONSENSUS_LEVEL_READY,
            earliest,
            earliest,
            "first_live_consensus_snapshot",
        )
        current = CONSENSUS_LEVEL_READY
        state_entered_at = earliest
        state_effective_at = earliest
        blocking_reason = "second_independent_capture_date_missing"
        next_state = CONSENSUS_REVISION_READY
    if revision_ready:
        second = _required_timestamp(
            event["second_consensus_captured_at"], "second_consensus_captured_at"
        )
        _append_transition(
            transitions,
            event_id,
            current,
            CONSENSUS_REVISION_READY,
            second,
            second,
            "second_independent_consensus_capture_date",
        )
        current = CONSENSUS_REVISION_READY
        state_entered_at = second
        state_effective_at = second
        blocking_reason = "future_sec_filing_not_observed"
        next_state = FILING_OBSERVED
    if filing_observed:
        if accepted_at is None or filing_seen_at is None:
            raise ProspectiveEventStateMachineError("An observed filing lacks its clocks")
        _append_transition(
            transitions,
            event_id,
            current,
            FILING_OBSERVED,
            accepted_at,
            filing_seen_at,
            "original_10q_or_10k_accession_observed",
        )
        current = FILING_OBSERVED
        state_entered_at = filing_seen_at
        state_effective_at = accepted_at
        blocking_reason = "preacceptance_consensus_not_frozen"
        next_state = CONSENSUS_FROZEN
        if not frozen_consensus_ready:
            _append_transition(
                transitions,
                event_id,
                current,
                REJECTED,
                accepted_at,
                filing_seen_at,
                "missing_strictly_preacceptance_consensus",
            )
            current = REJECTED
            state_entered_at = filing_seen_at
            state_effective_at = accepted_at
            blocking_reason = "terminal_rejection"
            next_state = None
            rejection_reason = "missing_strictly_preacceptance_consensus"
        else:
            _append_transition(
                transitions,
                event_id,
                current,
                CONSENSUS_FROZEN,
                accepted_at,
                filing_seen_at,
                "last_consensus_snapshot_strictly_before_sec_acceptance",
            )
            current = CONSENSUS_FROZEN
            state_entered_at = filing_seen_at
            state_effective_at = accepted_at
            blocking_reason = "accession_matched_actual_revenue_not_ready"
            next_state = ACTUAL_READY

    actual_status = None if pd.isna(event.get("actual_status")) else str(event["actual_status"])
    actual_observed_at = _optional_timestamp(event.get("actual_observed_at"))
    actual_available_at = _optional_timestamp(event.get("actual_available_at"))
    actual_ready = bool(
        current != REJECTED
        and actual_status in _READY_ACTUAL_STATUSES
        and pd.notna(event.get("actual_revenue"))
        and actual_observed_at is not None
        and actual_available_at is not None
    )
    if current == CONSENSUS_FROZEN and actual_status is not None and not actual_ready:
        review_at = actual_observed_at or filing_seen_at or accepted_at
        if review_at is None:
            raise ProspectiveEventStateMachineError("Actual review evidence lacks a clock")
        _append_transition(
            transitions,
            event_id,
            current,
            ACTUAL_REVIEW_REQUIRED,
            accepted_at or review_at,
            review_at,
            actual_status,
        )
        current = ACTUAL_REVIEW_REQUIRED
        state_entered_at = review_at
        state_effective_at = accepted_at or review_at
        blocking_reason = f"actual_status:{actual_status}"
        next_state = ACTUAL_READY
    if actual_ready and current != REJECTED:
        assert actual_observed_at is not None
        assert actual_available_at is not None
        if actual_observed_at < actual_available_at:
            raise ProspectiveEventStateMachineError("An actual was observed before availability")
        _append_transition(
            transitions,
            event_id,
            current,
            ACTUAL_READY,
            actual_available_at,
            actual_observed_at,
            actual_status or "actual_ready",
        )
        current = ACTUAL_READY
        state_entered_at = actual_observed_at
        state_effective_at = actual_available_at
        blocking_reason = "innovation_join_not_ready"
        next_state = INNOVATION_READY

    raw_innovation_computed = bool(
        current != REJECTED
        and event.get("raw_innovation_ready", False)
        and event.get("consensus_precedes_actual", False)
        and pd.notna(event.get("revenue_innovation_pct"))
    )
    downstream_admission = bool(event["downstream_admission"])
    innovation_ready = raw_innovation_computed and downstream_admission
    if innovation_ready:
        if not actual_ready or actual_observed_at is None or actual_available_at is None:
            raise ProspectiveEventStateMachineError("Innovation became ready before its actual")
        _append_transition(
            transitions,
            event_id,
            current,
            INNOVATION_READY,
            actual_available_at,
            actual_observed_at,
            "raw_market_innovation_join_ready",
        )
        current = INNOVATION_READY
        state_entered_at = actual_observed_at
        state_effective_at = actual_available_at
        blocking_reason = "point_in_time_price_window_not_loaded"
        next_state = PRICE_WINDOW_READY
    elif raw_innovation_computed and not downstream_admission and current == ACTUAL_READY:
        blocking_reason = f"consensus_quality_not_admitted:{event['quality_status']}"

    state = {
        "prospective_event_id": event_id,
        "registration_order": int(event["registration_order"]),
        "cik": str(event["cik"]),
        "ticker": str(event["ticker"]),
        "target_period_end": event["target_period_end"],
        "current_state": current,
        "state_rank": _STATE_RANK[current],
        "state_entered_at": state_entered_at,
        "state_effective_at": state_effective_at,
        "next_expected_state": next_state,
        "waiting": current not in {REJECTED, PAPER_POSITION_CLOSED},
        "terminal": current in {REJECTED, PAPER_POSITION_CLOSED},
        "rejected": current == REJECTED,
        "rejection_reason": rejection_reason,
        "blocking_reason": blocking_reason,
        "consensus_level_ready": snapshot_ready,
        "consensus_revision_ready": revision_ready,
        "filing_observed": filing_observed,
        "consensus_frozen": frozen_consensus_ready,
        "actual_ready": actual_ready,
        "innovation_ready": innovation_ready,
        "raw_innovation_computed": raw_innovation_computed,
        "consensus_quality_status": str(event["quality_status"]),
        "consensus_quality_reasons": str(event["quality_reasons"]),
        "downstream_admission": downstream_admission,
        "consensus_quality_review_required": bool(event["manual_review_required"]),
        "consensus_quality_quarantined": bool(event["quarantined"]),
        "price_window_ready": False,
        "paper_decision_ready": False,
        "paper_position_open": False,
        "paper_position_closed": False,
        "future_accession_number": event.get("future_accession_number"),
        "sec_accepted_at": accepted_at,
        "filing_first_observed_at": filing_seen_at,
        "frozen_consensus_captured_at": latest if frozen_consensus_ready else None,
        "frozen_consensus_revenue": (
            float(event["latest_consensus_revenue"])
            if frozen_consensus_ready and pd.notna(event["latest_consensus_revenue"])
            else None
        ),
        "actual_status": actual_status,
        "actual_available_at": actual_available_at,
        "actual_observed_at": actual_observed_at,
        "actual_revenue": float(event["actual_revenue"]) if actual_ready else None,
        "innovation_join_status": event.get("innovation_join_status"),
        "revenue_innovation_pct": (
            float(event["revenue_innovation_pct"]) if innovation_ready else None
        ),
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }
    return state, transitions


def _append_transition(
    rows: list[dict[str, Any]],
    event_id: str,
    from_state: str | None,
    to_state: str,
    effective_at: pd.Timestamp,
    observed_at: pd.Timestamp,
    evidence: str,
) -> None:
    rows.append(
        {
            "prospective_event_id": event_id,
            "transition_order": len(rows) + 1,
            "from_state": from_state,
            "to_state": to_state,
            "to_state_rank": _STATE_RANK[to_state],
            "effective_at": effective_at,
            "observed_at": observed_at,
            "evidence": evidence,
            "live_trading_authorized": False,
        }
    )


def _validate_previous_states(current: pd.DataFrame, previous: pd.DataFrame | None) -> bool:
    if previous is None:
        return True
    _require_columns(
        previous, {"prospective_event_id", "current_state", "state_rank"}, "previous states"
    )
    if set(previous["prospective_event_id"]) != set(current["prospective_event_id"]):
        raise ProspectiveEventStateMachineError(
            "The frozen event set changed across state snapshots"
        )
    comparison = previous[["prospective_event_id", "current_state", "state_rank"]].merge(
        current[["prospective_event_id", "current_state", "state_rank"]],
        on="prospective_event_id",
        validate="one_to_one",
        suffixes=("_previous", "_current"),
    )
    resurrected = comparison["current_state_previous"].eq(REJECTED) & ~comparison[
        "current_state_current"
    ].eq(REJECTED)
    regressed = (
        ~comparison["current_state_previous"].eq(REJECTED)
        & ~comparison["current_state_current"].eq(REJECTED)
        & comparison["state_rank_current"].lt(comparison["state_rank_previous"])
    )
    if resurrected.any() or regressed.any():
        raise ProspectiveEventStateMachineError("An event state regressed across daily snapshots")
    return True


def _validate_transitions(transitions: pd.DataFrame) -> bool:
    for _, group in transitions.groupby("prospective_event_id", sort=False):
        ordered = group.sort_values("transition_order")
        if not ordered["transition_order"].tolist() == list(range(1, len(ordered) + 1)):
            raise ProspectiveEventStateMachineError("A transition order is not contiguous")
        non_rejected = ordered.loc[~ordered["to_state"].eq(REJECTED), "to_state_rank"]
        if not non_rejected.is_monotonic_increasing or non_rejected.duplicated().any():
            raise ProspectiveEventStateMachineError("An event transition history is not monotone")
        if ordered["to_state"].eq(REJECTED).any() and not ordered.iloc[-1]["to_state"] == REJECTED:
            raise ProspectiveEventStateMachineError("Rejected must be the final event state")
        if (ordered["observed_at"] < ordered["effective_at"]).any():
            raise ProspectiveEventStateMachineError(
                "Evidence was observed before it became effective"
            )
    return True


def _state_catalog() -> pd.DataFrame:
    rows = [
        (
            REGISTERED,
            0,
            False,
            True,
            CONSENSUS_LEVEL_READY,
            "Frozen event exists; no live consensus yet.",
        ),
        (
            CONSENSUS_LEVEL_READY,
            1,
            False,
            True,
            CONSENSUS_REVISION_READY,
            "At least one forward-only consensus snapshot exists.",
        ),
        (
            CONSENSUS_REVISION_READY,
            2,
            False,
            True,
            FILING_OBSERVED,
            "At least two independent capture dates exist.",
        ),
        (
            FILING_OBSERVED,
            3,
            False,
            True,
            CONSENSUS_FROZEN,
            "Original 10-Q or 10-K accession has been observed.",
        ),
        (
            CONSENSUS_FROZEN,
            4,
            False,
            True,
            ACTUAL_READY,
            "Last snapshot strictly before SEC acceptance is frozen.",
        ),
        (
            ACTUAL_REVIEW_REQUIRED,
            5,
            False,
            True,
            ACTUAL_READY,
            "Accession fact needs repair or manual evidence review.",
        ),
        (
            ACTUAL_READY,
            6,
            False,
            True,
            INNOVATION_READY,
            "Accession-matched quarterly revenue is available.",
        ),
        (
            INNOVATION_READY,
            7,
            False,
            True,
            PRICE_WINDOW_READY,
            "Raw market innovation is available without returns.",
        ),
        (
            PRICE_WINDOW_READY,
            8,
            False,
            False,
            PAPER_DECISION_READY,
            "Reserved for point-in-time price evidence.",
        ),
        (
            PAPER_DECISION_READY,
            9,
            False,
            False,
            PAPER_POSITION_OPEN,
            "Reserved for the shadow decision ledger.",
        ),
        (
            PAPER_POSITION_OPEN,
            10,
            False,
            False,
            PAPER_POSITION_CLOSED,
            "Reserved for a simulated open position.",
        ),
        (PAPER_POSITION_CLOSED, 11, True, False, None, "Reserved terminal paper-trade state."),
        (
            REJECTED,
            99,
            True,
            True,
            None,
            "Terminal evidence failure; event cannot re-enter automatically.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "state_code",
            "state_rank",
            "terminal",
            "implemented_in_v1",
            "default_next_state",
            "description",
        ],
    )


def _transition_columns() -> list[str]:
    return [
        "prospective_event_id",
        "transition_order",
        "from_state",
        "to_state",
        "to_state_rank",
        "effective_at",
        "observed_at",
        "evidence",
        "live_trading_authorized",
    ]


def _latest_complete_state_root(output_root: Path) -> Path | None:
    candidates = list(output_root.glob("*/prospective_event_state_machine_report.json"))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: str(_read_json(path).get("generated_at", "")),
    ).parent


def _required_timestamp(value: Any, label: str) -> pd.Timestamp:
    result = _optional_timestamp(value)
    if result is None:
        raise ProspectiveEventStateMachineError(f"Missing {label}")
    return result


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _utc(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ProspectiveEventStateMachineError(f"Invalid {label}")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ProspectiveEventStateMachineError(f"{label} lacks required columns: {missing}")


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
