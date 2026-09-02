"""Auditable post-window listing-status and terminal-outcome evidence from Massive."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_reference import (
    MassiveConfigurationError,
    MassiveReferenceClient,
    MassiveRequestError,
    MassiveResponse,
    parse_massive_ticker_response,
)

MASSIVE_DELISTING_PARSER_VERSION: Final[str] = (
    "massive-delisting-outcome-parser-v3-security-aliases"
)


@dataclass(frozen=True)
class MassiveDelistingConfig:
    """Explicit common cutoff for observing later listing outcomes."""

    outcome_cutoff_date: str = "2026-08-14"
    version: str = "massive-delisting-outcome-v1"

    def validate(self) -> None:
        """Reject invalid or future cutoff assumptions."""
        cutoff = _parse_date(self.outcome_cutoff_date)
        if cutoff > date(2026, 8, 14):
            raise MassiveConfigurationError(
                "Delisting outcome cutoff exceeds the frozen as-of date"
            )


@dataclass(frozen=True)
class MassiveDelistingPlanResult:
    """Immutable set of post-window listing-status targets."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveDelistingCollectionResult:
    """Counts and stopping reason from one resumable status batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class MassiveDelistingAuditResult:
    """Immutable event-window terminal-outcome audit."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveDelistingSnapshot:
    """Content-addressed raw listing-status response."""

    outcome_target_id: str
    query_id: str
    query_active: bool
    outcome_cutoff_date: str
    page_number: int
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


def build_massive_delisting_target_plan(
    security_actions_root: Path,
    calendar_root: Path,
    *,
    output_root: Path = Path("data/market/massive_delisting/targets"),
    config: MassiveDelistingConfig | None = None,
) -> MassiveDelistingPlanResult:
    """Freeze one common post-window listing-status cutoff for all ready events."""
    cfg = config or MassiveDelistingConfig()
    cfg.validate()
    actions_report = _read_json(security_actions_root / "massive_security_actions_report.json")
    actions_manifest_sha256 = _sha256_file(security_actions_root / "output_manifest.json")
    calendar_manifest_sha256 = _sha256_file(calendar_root / "output_manifest.json")
    if actions_report.get("calendar_manifest_sha256") != calendar_manifest_sha256:
        raise FilingCatalogError("Security-actions audit does not tie to the calendar audit")
    targets = load_catalog_table(security_actions_root, "massive_security_actions_targets.parquet")
    readiness = load_catalog_table(security_actions_root, "sample_security_actions_audit.parquet")
    mappings = load_catalog_table(security_actions_root, "validated_security_mappings.parquet")
    alignment = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
    ready = readiness[
        readiness["historical_symbol_interval_ready"].astype(bool)
        & readiness["corporate_actions_ready"].astype(bool)
    ][["action_target_id"]]
    targets = targets.merge(ready, on="action_target_id", how="inner", validate="one_to_one")
    targets = targets.merge(
        mappings[
            [
                "action_target_id",
                "security_id",
                "security_id_kind",
                "source_snapshot_sha256",
            ]
        ].rename(columns={"source_snapshot_sha256": "mapping_source_sha256"}),
        on="action_target_id",
        how="left",
        validate="one_to_one",
    )
    targets = targets.merge(
        alignment[
            [
                "price_target_id",
                "formal_exit_session_date",
                "formal_exit_close_at",
                "formal_exit_close_unadjusted",
                "formal_event_window_ready",
            ]
        ],
        on="price_target_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_calendar"),
    )
    if targets.empty:
        raise FilingCatalogError("Delisting outcome plan requires ready security-action events")
    required = [
        "security_id",
        "security_id_kind",
        "mapping_source_sha256",
        "formal_exit_session_date_calendar",
        "formal_exit_close_at",
        "formal_exit_close_unadjusted",
    ]
    if targets[required].isna().any().any():
        raise FilingCatalogError("A delisting target lacks identity or formal-exit evidence")
    cutoff = _parse_date(cfg.outcome_cutoff_date)
    if targets["price_window_end"].map(lambda value: _parse_date(str(value)) >= cutoff).any():
        raise FilingCatalogError("Outcome cutoff must be after every price window")
    targets["outcome_cutoff_date"] = cutoff.isoformat()
    targets["outcome_target_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "action_target_id": str(row["action_target_id"]),
                "security_id": str(row["security_id"]),
                "outcome_cutoff_date": cutoff.isoformat(),
                "config_version": cfg.version,
            }
        ),
        axis=1,
    )
    targets = targets.sort_values("action_acquisition_order").reset_index(drop=True)
    targets["outcome_acquisition_order"] = np.arange(1, len(targets) + 1)
    columns = [
        "outcome_target_id",
        "outcome_acquisition_order",
        "action_target_id",
        "price_target_id",
        "target_id",
        "cik",
        "accepted_at",
        "query_date",
        "selected_ticker",
        "selected_exchange_mic",
        "security_id",
        "security_id_kind",
        "selected_share_class_figi",
        "selected_composite_figi",
        "mapping_source_sha256",
        "price_window_end",
        "formal_exit_session_date_calendar",
        "formal_exit_close_at",
        "formal_exit_close_unadjusted",
        "outcome_cutoff_date",
    ]
    targets = targets[columns].rename(
        columns={"formal_exit_session_date_calendar": "formal_exit_session_date"}
    )
    payload = json.dumps(
        {
            "security_actions_manifest_sha256": actions_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
            "config": asdict(cfg),
            "targets": targets.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-delisting-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MassiveDelistingPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete delisting target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "massive_delisting_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "security_actions_version": actions_report["dataset_version"],
        "security_actions_manifest_sha256": actions_manifest_sha256,
        "calendar_version": actions_report["calendar_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "config": asdict(cfg),
        "targets": len(targets),
        "outcome_cutoff_date": cutoff.isoformat(),
        "query_method": "cik_cutoff_active_true_then_inactive_if_expected_security_absent",
        "selection_uses_later_listing_status": False,
        "current_ticker_join_used": False,
        "artifacts": {target_path.name: artifact},
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "security_actions_manifest_sha256": actions_manifest_sha256,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "artifacts": {target_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveDelistingPlanResult(target.resolve(), report)


def load_massive_delisting_targets(plan_root: Path) -> pd.DataFrame:
    """Load and minimally validate an immutable delisting target plan."""
    targets = load_catalog_table(plan_root, "massive_delisting_targets.parquet")
    if targets.empty or targets["outcome_target_id"].duplicated().any():
        raise FilingCatalogError("Delisting targets are empty or duplicated")
    if targets["security_id"].astype(str).str.strip().eq("").any():
        raise FilingCatalogError("A delisting target lacks a stable security ID")
    return targets


def collect_massive_delisting_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_delisting"),
    limit: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveDelistingCollectionResult:
    """Collect active status first and query inactive records only when necessary."""
    if limit < 1:
        raise MassiveConfigurationError("Delisting collection limit must be positive")
    ordered = targets.sort_values("outcome_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_target_completion(raw_root, str(row.outcome_target_id))
    ][:limit]
    completed = failed = requests_made = reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        try:
            active, requests, reused_pages = _collect_status_query(
                client, raw_root, row, active=True
            )
            requests_made += requests
            reused += reused_pages
            active_matches = _completion_exact_matches(active, row)
            inactive: dict[str, Any] | None = None
            if active_matches == 0:
                inactive, requests, reused_pages = _collect_status_query(
                    client, raw_root, row, active=False
                )
                requests_made += requests
                reused += reused_pages
            _write_target_completion(raw_root, row, active, inactive)
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, str(row.outcome_target_id), exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), str(row.outcome_target_id))
    return MassiveDelistingCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def build_massive_delisting_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/massive_delisting"),
    output_root: Path = Path("data/processed/massive_delisting"),
) -> MassiveDelistingAuditResult:
    """Classify later delistings and require terminal evidence only inside study windows."""
    plan_report = _read_json(plan_root / "target_report.json")
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    targets = load_massive_delisting_targets(plan_root)
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    completions: dict[str, dict[str, Any]] = {}
    for row in targets.itertuples(index=False):
        outcome_target_id = str(row.outcome_target_id)
        completion = _load_target_completion(raw_root, outcome_target_id)
        if completion is None:
            continue
        completions[outcome_target_id] = completion
        query_documents = [completion["active_query"]]
        if completion.get("inactive_query") is not None:
            query_documents.append(completion["inactive_query"])
        for query in query_documents:
            for snapshot_path in query["snapshots"]:
                snapshot = _load_snapshot(Path(snapshot_path))
                snapshots.append(_snapshot_row(snapshot))
                parsed, rejected, _ = parse_massive_ticker_response(
                    Path(snapshot.path).read_bytes(),
                    expected_cik=str(row.cik),
                    target_id=outcome_target_id,
                    query_active=snapshot.query_active,
                    as_of_date=str(row.outcome_cutoff_date),
                    source_sha256=snapshot.sha256,
                )
                for item in parsed:
                    item["outcome_target_id"] = item.pop("target_id")
                observations.extend(parsed)
                rejections.extend(
                    {
                        "outcome_target_id": item["target_id"],
                        "query_active": snapshot.query_active,
                        "source_occurrence": item["source_occurrence"],
                        "reason": item["reason"],
                        "source_sha256": item["source_sha256"],
                    }
                    for item in rejected
                )
    observation_frame = pd.DataFrame(observations, columns=_observation_columns())
    rejection_frame = pd.DataFrame(rejections, columns=_rejection_columns())
    snapshot_frame = pd.DataFrame(snapshots, columns=_snapshot_columns())
    audit, outcomes = _build_outcome_tables(
        targets, observation_frame, rejection_frame, completions
    )
    source_hashes = sorted(snapshot_frame.get("sha256", pd.Series(dtype=str)).astype(str).tolist())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "source_hashes": source_hashes,
            "parser_version": MASSIVE_DELISTING_PARSER_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-delisting-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MassiveDelistingAuditResult(
            target.resolve(), _read_json(target / "massive_delisting_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete delisting audit directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_delisting_targets.parquet": targets,
        "listing_status_observations.parquet": observation_frame,
        "listing_status_rejections.parquet": rejection_frame,
        "listing_status_snapshots.parquet": snapshot_frame,
        "sample_delisting_outcome_audit.parquet": audit,
        "event_window_delisting_outcomes.parquet": outcomes,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    collected = int(audit["provider_outcome_query_complete"].astype(bool).sum())
    outcome_ready = int(audit["delisting_outcome_ready"].astype(bool).sum())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "security_actions_version": plan_report["security_actions_version"],
        "security_actions_manifest_sha256": plan_report["security_actions_manifest_sha256"],
        "calendar_manifest_sha256": plan_report["calendar_manifest_sha256"],
        "parser_version": MASSIVE_DELISTING_PARSER_VERSION,
        "targets": len(targets),
        "targets_collected": collected,
        "raw_snapshots": len(snapshot_frame),
        "rejected_rows": len(rejection_frame),
        "active_at_outcome_cutoff": int(audit["active_at_outcome_cutoff"].sum()),
        "later_delisted_after_formal_exit": int(audit["later_delisted_after_formal_exit"].sum()),
        "delisted_on_or_before_formal_exit": int(audit["delisted_on_or_before_formal_exit"].sum()),
        "delisting_outcomes_ready": outcome_ready,
        "pilot_delisting_outcome_gate": outcome_ready == len(targets),
        "provider_supplies_delisting_return": False,
        "zero_delisting_return_imputed": False,
        "pilot_event_return_gate": outcome_ready == len(targets),
        "full_return_study_gate": False,
        "selection_uses_later_listing_status": False,
        "artifacts": artifacts,
    }
    report_path = staging / "massive_delisting_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "security_actions_manifest_sha256": plan_report["security_actions_manifest_sha256"],
                "calendar_manifest_sha256": plan_report["calendar_manifest_sha256"],
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveDelistingAuditResult(target.resolve(), report)


def load_collected_massive_listing_status(
    targets: pd.DataFrame,
    *,
    raw_root: Path = Path("data/market/massive_delisting"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse preserved listing-status snapshots for compatible formal targets."""
    required = {"outcome_target_id", "cik", "outcome_cutoff_date"}
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise FilingCatalogError(f"Compatible delisting targets lack columns: {missing}")
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        outcome_target_id = str(row.outcome_target_id)
        completion = _load_target_completion(raw_root, outcome_target_id)
        if completion is None:
            continue
        completion_rows.append(
            {
                "outcome_target_id": outcome_target_id,
                "active_query_complete": True,
                "inactive_query_complete": completion.get("inactive_query") is not None,
                "completed_at": str(completion.get("completed_at", "")),
            }
        )
        query_documents = [completion["active_query"]]
        if completion.get("inactive_query") is not None:
            query_documents.append(completion["inactive_query"])
        for query in query_documents:
            for snapshot_path in query["snapshots"]:
                snapshot = _load_snapshot(Path(snapshot_path))
                parsed, rejected, _ = parse_massive_ticker_response(
                    Path(snapshot.path).read_bytes(),
                    expected_cik=str(row.cik),
                    target_id=outcome_target_id,
                    query_active=snapshot.query_active,
                    as_of_date=str(row.outcome_cutoff_date),
                    source_sha256=snapshot.sha256,
                )
                for item in parsed:
                    item["outcome_target_id"] = item.pop("target_id")
                observations.extend(parsed)
                rejections.extend(
                    {
                        "outcome_target_id": item["target_id"],
                        "query_active": snapshot.query_active,
                        "source_occurrence": item["source_occurrence"],
                        "reason": item["reason"],
                        "source_sha256": item["source_sha256"],
                    }
                    for item in rejected
                )
    return (
        pd.DataFrame(observations, columns=_observation_columns()),
        pd.DataFrame(rejections, columns=_rejection_columns()),
        pd.DataFrame(
            completion_rows,
            columns=[
                "outcome_target_id",
                "active_query_complete",
                "inactive_query_complete",
                "completed_at",
            ],
        ),
    )


def _collect_status_query(
    client: MassiveReferenceClient,
    raw_root: Path,
    row: Any,
    *,
    active: bool,
) -> tuple[dict[str, Any], int, int]:
    outcome_target_id = str(row.outcome_target_id)
    existing = _load_query_completion(raw_root, outcome_target_id, active)
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = client.ticker_reference_url(
        str(row.cik), str(row.outcome_cutoff_date), active=active
    )
    snapshot_paths: list[str] = []
    requests_made = reused = page_number = result_count = rejection_count = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError("Delisting pagination exceeded 20 pages", reason="pagination")
        query_id = _stable_hash(
            {
                "outcome_target_id": outcome_target_id,
                "query_active": active,
                "page_url": url,
            }
        )
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(raw_root, row, query_id, active, page_number, response)
            requests_made += response.requests_made
        else:
            reused += 1
        parsed, rejected, next_url = parse_massive_ticker_response(
            Path(snapshot.path).read_bytes(),
            expected_cik=str(row.cik),
            target_id=outcome_target_id,
            query_active=active,
            as_of_date=str(row.outcome_cutoff_date),
            source_sha256=snapshot.sha256,
        )
        result_count += len(parsed)
        rejection_count += len(rejected)
        snapshot_paths.append(str(_snapshot_manifest_path(raw_root, snapshot).resolve()))
        url = next_url
    completion = {
        "outcome_target_id": outcome_target_id,
        "query_active": active,
        "outcome_cutoff_date": str(row.outcome_cutoff_date),
        "result_count": result_count,
        "rejection_count": rejection_count,
        "page_count": len(snapshot_paths),
        "pagination_complete": True,
        "snapshots": snapshot_paths,
    }
    _write_query_completion(raw_root, outcome_target_id, active, completion)
    return completion, requests_made, reused


def _completion_exact_matches(completion: dict[str, Any], row: Any) -> int:
    count = 0
    for snapshot_path in completion["snapshots"]:
        snapshot = _load_snapshot(Path(snapshot_path))
        parsed, _, _ = parse_massive_ticker_response(
            Path(snapshot.path).read_bytes(),
            expected_cik=str(row.cik),
            target_id=str(row.outcome_target_id),
            query_active=snapshot.query_active,
            as_of_date=str(row.outcome_cutoff_date),
            source_sha256=snapshot.sha256,
        )
        count += sum(
            item["provider_security_id"] == str(row.security_id) and item["security_type"] == "CS"
            for item in parsed
        )
    return count


def _build_outcome_tables(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    rejections: pd.DataFrame,
    completions: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = targets.copy()
    audit["provider_outcome_query_complete"] = audit["outcome_target_id"].isin(completions)
    audit["active_query_complete"] = False
    audit["inactive_query_complete"] = False
    audit["active_exact_matches"] = 0
    audit["inactive_exact_matches"] = 0
    audit["listing_outcome_status"] = "not_collected"
    audit["outcome_ticker"] = pd.NA
    audit["delisted_utc"] = pd.NA
    audit["active_at_outcome_cutoff"] = False
    audit["later_delisted_after_formal_exit"] = False
    audit["delisted_on_or_before_formal_exit"] = False
    audit["delisting_outcome_ready"] = False
    outcome_rows: list[dict[str, Any]] = []
    for index, target in audit.iterrows():
        outcome_target_id = str(target["outcome_target_id"])
        completion = completions.get(outcome_target_id)
        if completion is None:
            outcome_rows.append(
                {
                    "outcome_target_id": outcome_target_id,
                    "price_target_id": target["price_target_id"],
                    "target_id": target["target_id"],
                    "security_id": target["security_id"],
                    "ticker_at_event": target["selected_ticker"],
                    "ticker_at_outcome": None,
                    "formal_exit_at": target["formal_exit_close_at"],
                    "outcome_cutoff_date": target["outcome_cutoff_date"],
                    "delisting_date": None,
                    "delisting_return": np.nan,
                    "terminal_value": np.nan,
                    "missing_return_reason": "outcome_query_not_collected",
                    "window_terminal_outcome_ready": False,
                    "later_delisted_after_formal_exit": False,
                    "source_snapshot_sha256": None,
                }
            )
            continue
        audit.at[index, "active_query_complete"] = True
        audit.at[index, "inactive_query_complete"] = completion.get("inactive_query") is not None
        target_observations = observations[observations["outcome_target_id"].eq(outcome_target_id)]
        exact = target_observations[
            target_observations["provider_security_id"].eq(str(target["security_id"]))
            & target_observations["security_type"].eq("CS")
        ]
        active = exact[exact["query_active"].astype(bool)]
        inactive = exact[~exact["query_active"].astype(bool)]
        audit.at[index, "active_exact_matches"] = len(active)
        audit.at[index, "inactive_exact_matches"] = len(inactive)
        target_rejections = rejections[rejections["outcome_target_id"].eq(outcome_target_id)]
        status = "unresolved_security_at_cutoff"
        ready = False
        active_at_cutoff = later_delisted = delisted_before_exit = False
        observed_ticker: str | None = None
        delisted_utc: str | None = None
        source_sha256: str | None = None
        missing_reason = "provider_security_not_resolved_at_outcome_cutoff"
        if not target_rejections.empty:
            status = "listing_status_parse_rejection"
            missing_reason = "listing_status_parse_rejection"
        elif not active.empty:
            active_semantic_conflict = active.apply(
                lambda row: (
                    not _provider_state_is(row["provider_active"], True)
                    or pd.notna(row["delisted_utc"])
                ),
                axis=1,
            ).any()
            if active_semantic_conflict:
                status = "active_record_semantic_conflict"
                missing_reason = "active_record_semantic_conflict"
            else:
                row = _preferred_ticker_alias(active, str(target["selected_ticker"]))
                status = "active_at_outcome_cutoff"
                ready = active_at_cutoff = True
                observed_ticker = str(row["ticker"])
                source_sha256 = str(row["source_sha256"])
                missing_reason = "not_applicable_active_through_outcome_cutoff"
        elif not inactive.empty:
            inactive_semantic_conflict = inactive.apply(
                lambda row: not _provider_state_is(row["provider_active"], False),
                axis=1,
            ).any()
            dated_inactive = inactive.copy()
            dated_inactive["_normalized_delisted_utc"] = dated_inactive["delisted_utc"].map(
                _normalized_delisted_utc
            )
            dated_inactive = dated_inactive[
                dated_inactive["_normalized_delisted_utc"].notna()
            ].copy()
            if inactive_semantic_conflict:
                status = "inactive_record_semantic_conflict"
                missing_reason = "inactive_record_semantic_conflict"
            elif dated_inactive.empty:
                status = "inactive_record_missing_last_trade_date"
                missing_reason = "provider_inactive_without_delisted_utc"
            else:
                dated_inactive["_delisted_at"] = dated_inactive["_normalized_delisted_utc"].map(
                    _parse_timestamp
                )
                latest_delisted_at = dated_inactive["_delisted_at"].max()
                latest_aliases = dated_inactive[
                    dated_inactive["_delisted_at"].eq(latest_delisted_at)
                ]
                row = _preferred_ticker_alias(latest_aliases, str(target["selected_ticker"]))
                observed_ticker = str(row["ticker"])
                source_sha256 = str(row["source_sha256"])
                delisted_utc = str(row["_normalized_delisted_utc"])
                exit_date = _parse_date(str(target["formal_exit_session_date"]))
                if latest_delisted_at.date() > exit_date:
                    status = "later_delisted_after_formal_exit"
                    ready = later_delisted = True
                    missing_reason = "not_applicable_delisting_after_formal_exit"
                else:
                    status = "delisted_on_or_before_formal_exit_terminal_value_required"
                    delisted_before_exit = True
                    missing_reason = "provider_does_not_supply_delisting_return_or_terminal_value"
        audit.at[index, "listing_outcome_status"] = status
        audit.at[index, "outcome_ticker"] = observed_ticker
        audit.at[index, "delisted_utc"] = delisted_utc
        audit.at[index, "active_at_outcome_cutoff"] = active_at_cutoff
        audit.at[index, "later_delisted_after_formal_exit"] = later_delisted
        audit.at[index, "delisted_on_or_before_formal_exit"] = delisted_before_exit
        audit.at[index, "delisting_outcome_ready"] = ready
        outcome_rows.append(
            {
                "outcome_target_id": outcome_target_id,
                "price_target_id": target["price_target_id"],
                "target_id": target["target_id"],
                "security_id": target["security_id"],
                "ticker_at_event": target["selected_ticker"],
                "ticker_at_outcome": observed_ticker,
                "formal_exit_at": target["formal_exit_close_at"],
                "outcome_cutoff_date": target["outcome_cutoff_date"],
                "delisting_date": (
                    _parse_timestamp(delisted_utc).date().isoformat()
                    if delisted_utc is not None
                    else None
                ),
                "delisting_return": np.nan,
                "terminal_value": np.nan,
                "missing_return_reason": missing_reason,
                "window_terminal_outcome_ready": ready,
                "later_delisted_after_formal_exit": later_delisted,
                "source_snapshot_sha256": source_sha256,
            }
        )
    outcomes = pd.DataFrame(outcome_rows, columns=_outcome_columns())
    return audit, outcomes


def _preserve_response(
    raw_root: Path,
    row: Any,
    query_id: str,
    active: bool,
    page_number: int,
    response: MassiveResponse,
) -> MassiveDelistingSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Delisting raw snapshot hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Delisting response changed before preservation")
        temporary.replace(path)
    snapshot = MassiveDelistingSnapshot(
        outcome_target_id=str(row.outcome_target_id),
        query_id=query_id,
        query_active=active,
        outcome_cutoff_date=str(row.outcome_cutoff_date),
        page_number=page_number,
        requested_url=response.requested_url,
        response_url=response.response_url,
        sha256=digest,
        fetched_at=response.fetched_at.isoformat(),
        path=str(path.resolve()),
        bytes=len(response.content),
        status_code=response.status_code,
        headers=response.headers,
    )
    manifest_path = _snapshot_manifest_path(raw_root, snapshot)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8"
        )
    return snapshot


def _existing_snapshot(
    raw_root: Path, query_id: str, requested_url: str
) -> MassiveDelistingSnapshot | None:
    for path in sorted((raw_root / "snapshots" / query_id).glob("*.json")):
        snapshot = _load_snapshot(path)
        if snapshot.requested_url == requested_url:
            return snapshot
    return None


def _load_snapshot(path: Path) -> MassiveDelistingSnapshot:
    snapshot = MassiveDelistingSnapshot(**_read_json(path))
    raw_path = Path(snapshot.path)
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Delisting snapshot is missing or invalid: {raw_path}")
    return snapshot


def _snapshot_manifest_path(raw_root: Path, snapshot: MassiveDelistingSnapshot) -> Path:
    return raw_root / "snapshots" / snapshot.query_id / f"{snapshot.sha256}.json"


def _write_query_completion(
    raw_root: Path,
    outcome_target_id: str,
    active: bool,
    completion: dict[str, Any],
) -> None:
    name = "active.json" if active else "inactive.json"
    path = raw_root / "queries" / outcome_target_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")


def _load_query_completion(
    raw_root: Path, outcome_target_id: str, active: bool
) -> dict[str, Any] | None:
    name = "active.json" if active else "inactive.json"
    path = raw_root / "queries" / outcome_target_id / name
    if not path.is_file():
        return None
    document = _read_json(path)
    if document.get("query_active") is not active or not document.get("pagination_complete"):
        return None
    for snapshot_path in document.get("snapshots", []):
        snapshot = _load_snapshot(Path(snapshot_path))
        if snapshot.query_active is not active:
            return None
    return document


def _write_target_completion(
    raw_root: Path,
    row: Any,
    active: dict[str, Any],
    inactive: dict[str, Any] | None,
) -> None:
    path = raw_root / "completed" / f"{row.outcome_target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "outcome_target_id": str(row.outcome_target_id),
                    "target_id": str(row.target_id),
                    "security_id": str(row.security_id),
                    "outcome_cutoff_date": str(row.outcome_cutoff_date),
                    "active_query": active,
                    "inactive_query": inactive,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _load_target_completion(raw_root: Path, outcome_target_id: str) -> dict[str, Any] | None:
    path = raw_root / "completed" / f"{outcome_target_id}.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    active = _load_query_completion(raw_root, outcome_target_id, True)
    if active is None or document.get("active_query") != active:
        return None
    inactive_document = document.get("inactive_query")
    if inactive_document is not None:
        inactive = _load_query_completion(raw_root, outcome_target_id, False)
        if inactive is None or inactive_document != inactive:
            return None
    return document


def _valid_target_completion(raw_root: Path, outcome_target_id: str) -> bool:
    try:
        return _load_target_completion(raw_root, outcome_target_id) is not None
    except (FilingCatalogError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _write_failure(raw_root: Path, outcome_target_id: str, reason: str, detail: str) -> None:
    path = (
        raw_root
        / "failures"
        / outcome_target_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "outcome_target_id": outcome_target_id,
                "reason": reason,
                "detail": detail,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _preferred_ticker_alias(records: pd.DataFrame, preferred_ticker: str) -> pd.Series[Any]:
    candidates = records
    preferred = records[records["ticker"].eq(preferred_ticker)]
    if not preferred.empty:
        candidates = preferred
    return candidates.sort_values(
        ["last_updated_utc", "ticker", "source_occurrence"],
        na_position="first",
        kind="stable",
    ).iloc[-1]


def _normalized_delisted_utc(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        timestamp = pd.Timestamp(str(value))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _provider_state_is(value: Any, expected: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        return False
    return bool(value) is expected


def _parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveConfigurationError(f"Invalid ISO date: {value}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _snapshot_row(snapshot: MassiveDelistingSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def _observation_columns() -> list[str]:
    return [
        "outcome_target_id",
        "cik",
        "as_of_date",
        "query_active",
        "provider_active",
        "ticker",
        "name",
        "market",
        "locale",
        "security_type",
        "exchange_mic",
        "currency_name",
        "share_class_figi",
        "composite_figi",
        "provider_security_id",
        "provider_security_id_kind",
        "delisted_utc",
        "last_updated_utc",
        "source_occurrence",
        "source_sha256",
    ]


def _rejection_columns() -> list[str]:
    return [
        "outcome_target_id",
        "query_active",
        "source_occurrence",
        "reason",
        "source_sha256",
    ]


def _snapshot_columns() -> list[str]:
    return [field.name for field in MassiveDelistingSnapshot.__dataclass_fields__.values()]


def _outcome_columns() -> list[str]:
    return [
        "outcome_target_id",
        "price_target_id",
        "target_id",
        "security_id",
        "ticker_at_event",
        "ticker_at_outcome",
        "formal_exit_at",
        "outcome_cutoff_date",
        "delisting_date",
        "delisting_return",
        "terminal_value",
        "missing_return_reason",
        "window_terminal_outcome_ready",
        "later_delisted_after_formal_exit",
        "source_snapshot_sha256",
    ]


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
