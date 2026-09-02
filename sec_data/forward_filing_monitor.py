"""Forward SEC filing monitor for future analyst-consensus Innovation events."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from sec_data.client import SUBMISSIONS_URL, SecClient, SecRequestError, normalize_cik
from sec_data.filing_catalog import FilingCatalogError
from sec_data.models import SecResponse

FORWARD_FILING_MONITOR_PLAN_VERSION: Final[str] = "forward-filing-monitor-plan-v1"
FORWARD_FILING_MONITOR_PARSER_VERSION: Final[str] = "forward-filing-monitor-parser-v2"
ELIGIBLE_FORMS: Final[frozenset[str]] = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})


class ForwardFilingMonitorError(FilingCatalogError):
    """Raised when a future-filing observation violates its point-in-time contract."""


@dataclass(frozen=True)
class ForwardFilingCollectionResult:
    """One resumable daily SEC submissions collection summary."""

    selected_targets: int
    completed_targets: int
    failed_targets: int
    requests_made: int


@dataclass(frozen=True)
class ForwardFilingMonitorResult:
    """Immutable monitor-plan or audit output."""

    output_root: Path
    report: dict[str, Any]


def calculate_forward_filing_monitor_plan(next_events: pd.DataFrame) -> pd.DataFrame:
    """Freeze one SEC submissions query per future consensus event without returns."""
    _require_columns(
        next_events,
        {
            "live_consensus_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "captured_at",
            "consensus_revenue_level",
            "source_sha256",
            "innovation_status",
            "future_actual_join_rule",
        },
        "forward consensus next events",
    )
    targets = next_events.copy()
    targets["cik"] = targets["cik"].astype(str).map(normalize_cik)
    targets["ticker"] = targets["ticker"].astype(str).str.strip().str.upper()
    targets["consensus_captured_at"] = _utc(targets["captured_at"], "captured_at")
    targets["target_period_end"] = _utc(
        targets["target_period_end"], "target_period_end"
    ).dt.normalize()
    if targets["target_period_end"].lt(targets["consensus_captured_at"].dt.normalize()).any():
        raise ForwardFilingMonitorError("A monitored target period predates its consensus capture")
    if targets.duplicated("ticker").any() or targets.duplicated("cik").any():
        raise ForwardFilingMonitorError(
            "The first monitor version requires one security per issuer"
        )
    targets["submissions_url"] = targets["cik"].map(lambda cik: SUBMISSIONS_URL.format(cik=cik))
    targets["eligible_forms"] = "10-Q|10-Q/A|10-K|10-K/A"
    targets["maximum_period_end_distance_days"] = 7
    targets["monitor_status"] = "awaiting_future_sec_filing"
    targets["historical_backfill_allowed"] = False
    targets["actual_values_loaded"] = False
    targets["return_values_loaded"] = False
    targets["selection_uses_return_values"] = False
    targets["sec_monitor_target_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "live_consensus_target_id": row["live_consensus_target_id"],
                "cik": row["cik"],
                "ticker": row["ticker"],
                "target_period_end": row["target_period_end"].date().isoformat(),
                "consensus_captured_at": row["consensus_captured_at"].isoformat(),
            }
        ),
        axis="columns",
    )
    targets = targets.sort_values(
        ["target_period_end", "ticker", "sec_monitor_target_id"], kind="mergesort"
    ).reset_index(drop=True)
    targets["collection_order"] = range(1, len(targets) + 1)
    return targets[
        [
            "sec_monitor_target_id",
            "collection_order",
            "live_consensus_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "consensus_captured_at",
            "consensus_revenue_level",
            "source_sha256",
            "submissions_url",
            "eligible_forms",
            "maximum_period_end_distance_days",
            "monitor_status",
            "future_actual_join_rule",
            "historical_backfill_allowed",
            "actual_values_loaded",
            "return_values_loaded",
            "selection_uses_return_values",
        ]
    ]


def build_forward_filing_monitor_plan(
    forward_consensus_root: Path,
    *,
    output_root: Path = Path("data/sec/forward_filing_monitor/targets"),
) -> ForwardFilingMonitorResult:
    """Persist the frozen future-filing watchlist before any SEC query."""
    input_hash = _sha256_file(forward_consensus_root / "output_manifest.json")
    targets = calculate_forward_filing_monitor_plan(
        pd.read_parquet(forward_consensus_root / "forward_consensus_next_events.parquet")
    )
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hash,
            "contract_version": FORWARD_FILING_MONITOR_PLAN_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        FORWARD_FILING_MONITOR_PLAN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ForwardFilingMonitorResult(
            target.resolve(), _read_json(target / "forward_filing_monitor_plan_report.json")
        )
    if target.exists():
        raise ForwardFilingMonitorError(f"Incomplete monitor plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    targets_path = staging / "forward_filing_monitor_targets.parquet"
    targets.to_parquet(targets_path, index=False)
    artifacts = {targets_path.name: _artifact(targets_path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "contract_version": FORWARD_FILING_MONITOR_PLAN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hash,
        "monitor_targets": len(targets),
        "monitor_issuers": int(targets["cik"].nunique()),
        "eligible_forms": sorted(ELIGIBLE_FORMS),
        "historical_backfill_allowed": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_filing_monitor_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(
        staging, dataset_version, {"forward_consensus": input_hash}, artifacts, report_path
    )
    staging.replace(target)
    return ForwardFilingMonitorResult(target.resolve(), report)


def collect_forward_filing_snapshots(
    targets: pd.DataFrame,
    client: SecClient,
    *,
    raw_root: Path,
    capture_date: date | None = None,
    limit: int = 20,
    progress: Callable[[int, int, str], None] | None = None,
) -> ForwardFilingCollectionResult:
    """Capture one immutable SEC submissions snapshot per pending issuer and UTC date."""
    _require_columns(
        targets,
        {"sec_monitor_target_id", "collection_order", "cik", "ticker"},
        "forward filing targets",
    )
    if limit < 1:
        raise ValueError("Monitor collection limit must be positive")
    day = capture_date or datetime.now(timezone.utc).date()
    pending = [
        row
        for row in targets.sort_values("collection_order").itertuples(index=False)
        if not _completion_path(raw_root, day, str(row.sec_monitor_target_id)).is_file()
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    for number, row in enumerate(pending, start=1):
        target_id = str(row.sec_monitor_target_id)
        try:
            response = client.get_submissions(str(row.cik))
            requests_made += 1
            manifest_path = _preserve_response(
                raw_root,
                target_id=target_id,
                cik=str(row.cik),
                ticker=str(row.ticker),
                capture_date=day,
                response=response,
            )
            if _submission_response_status(response.content, str(row.cik)) != "ok":
                _write_failure(raw_root, day, target_id, "bad_submissions_schema", manifest_path)
                failed += 1
            else:
                _write_json_immutable(
                    _completion_path(raw_root, day, target_id),
                    {
                        "sec_monitor_target_id": target_id,
                        "capture_date": day.isoformat(),
                        "cik": normalize_cik(str(row.cik)),
                        "ticker": str(row.ticker),
                        "snapshot_manifest": str(manifest_path.resolve()),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                completed += 1
        except SecRequestError:
            failed += 1
            _write_failure(raw_root, day, target_id, "sec_request_failed", None)
        if progress is not None:
            progress(number, len(pending), target_id)
    return ForwardFilingCollectionResult(
        selected_targets=len(pending),
        completed_targets=completed,
        failed_targets=failed,
        requests_made=requests_made,
    )


def parse_forward_filing_snapshot(
    content: bytes,
    *,
    target: dict[str, Any],
    observed_at: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Parse eligible forms and classify only filings after the consensus clock."""
    if _submission_response_status(content, str(target["cik"])) != "ok":
        raise ForwardFilingMonitorError("SEC submissions snapshot is not parseable")
    document: dict[str, Any] = json.loads(content)
    recent: dict[str, Any] = document["filings"]["recent"]
    accessions = recent["accessionNumber"]
    if not isinstance(accessions, list):
        raise ForwardFilingMonitorError("SEC accession list is invalid")
    consensus_captured_at = _timestamp(target["consensus_captured_at"], "consensus clock")
    observed_timestamp = _timestamp(observed_at, "observed_at")
    target_period_end = _timestamp(target["target_period_end"], "target period").normalize()
    rows: list[dict[str, Any]] = []
    for index, accession_value in enumerate(accessions):
        form = str(_list_value(recent, "form", index)).strip().upper()
        if form not in ELIGIBLE_FORMS:
            continue
        accession = str(accession_value).strip()
        accepted_at = _timestamp_or_none(_list_value(recent, "acceptanceDateTime", index))
        report_date = _timestamp_or_none(_list_value(recent, "reportDate", index))
        distance_days = (
            abs((report_date.normalize() - target_period_end).days)
            if report_date is not None
            else None
        )
        period_match_status = (
            "exact"
            if distance_days == 0
            else "within_7_days_provisional"
            if distance_days is not None and distance_days <= 7
            else "period_mismatch"
        )
        accepted_after_consensus = bool(
            accepted_at is not None and accepted_at > consensus_captured_at
        )
        observed_after_acceptance = bool(
            accepted_at is not None and observed_timestamp >= accepted_at
        )
        future_candidate = bool(
            accepted_after_consensus
            and observed_after_acceptance
            and period_match_status in {"exact", "within_7_days_provisional"}
        )
        rows.append(
            {
                "sec_monitor_target_id": target["sec_monitor_target_id"],
                "live_consensus_target_id": target["live_consensus_target_id"],
                "cik": normalize_cik(str(target["cik"])),
                "ticker": str(target["ticker"]),
                "target_period_end": target_period_end,
                "consensus_captured_at": consensus_captured_at,
                "consensus_revenue_level": float(target["consensus_revenue_level"]),
                "accession_number": accession,
                "form": form,
                "filing_date": _list_value(recent, "filingDate", index),
                "report_date": report_date,
                "accepted_at": accepted_at,
                "primary_document": _list_value(recent, "primaryDocument", index),
                "period_end_distance_days": distance_days,
                "period_match_status": period_match_status,
                "accepted_after_consensus": accepted_after_consensus,
                "observed_after_acceptance": observed_after_acceptance,
                "future_filing_candidate": future_candidate,
                "amendment_form": form.endswith("/A"),
                "observed_at": observed_timestamp,
                "availability_clock": "sec_acceptance_datetime",
                "historical_backtest_admission": False,
                "source_sha256": source_sha256,
            }
        )
    return rows


def build_forward_filing_monitor_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/forward_filing_monitor"),
    output_root: Path = Path("data/processed/forward_filing_monitor"),
) -> ForwardFilingMonitorResult:
    """Audit preserved submissions snapshots and emit first-seen future filing candidates."""
    plan_hash = _sha256_file(plan_root / "output_manifest.json")
    targets = pd.read_parquet(plan_root / "forward_filing_monitor_targets.parquet")
    target_index = targets.set_index("sec_monitor_target_id", drop=False)
    snapshots: list[dict[str, Any]] = []
    filing_rows: list[dict[str, Any]] = []
    for completion_path in raw_root.glob("completions/*/*.json"):
        completion = _read_json(completion_path)
        target_id = str(completion["sec_monitor_target_id"])
        if target_id not in target_index.index:
            continue
        manifest = _load_snapshot_manifest(Path(str(completion["snapshot_manifest"])))
        snapshots.append(manifest)
        target_record = cast(dict[str, Any], target_index.loc[target_id].to_dict())
        filing_rows.extend(
            parse_forward_filing_snapshot(
                Path(str(manifest["path"])).read_bytes(),
                target=target_record,
                observed_at=str(manifest["fetched_at"]),
                source_sha256=str(manifest["sha256"]),
            )
        )
    snapshot_frame = pd.DataFrame(snapshots, columns=_snapshot_columns())
    filings = pd.DataFrame(filing_rows, columns=_filing_columns())
    candidates = filings.loc[filings["future_filing_candidate"].astype(bool)].copy()
    if not candidates.empty:
        candidates = (
            candidates.sort_values(
                ["sec_monitor_target_id", "accession_number", "observed_at"],
                kind="mergesort",
            )
            .drop_duplicates(["sec_monitor_target_id", "accession_number"], keep="first")
            .reset_index(drop=True)
        )
        candidates["candidate_status"] = candidates["period_match_status"].map(
            {
                "exact": "ready_for_accession_xbrl_actual_collection",
                "within_7_days_provisional": "requires_xbrl_period_confirmation",
            }
        )
    else:
        candidates = pd.DataFrame(columns=[*_filing_columns(), "candidate_status"])
    prior_filings = filings.loc[
        filings["accepted_at"].notna() & filings["accepted_at"].lt(filings["consensus_captured_at"])
    ].copy()
    if not prior_filings.empty:
        latest_prior = (
            prior_filings.sort_values(
                ["sec_monitor_target_id", "accepted_at", "accession_number"],
                kind="mergesort",
            )
            .groupby("sec_monitor_target_id", as_index=False, sort=True)
            .tail(1)
            .sort_values("ticker", kind="mergesort")
            .reset_index(drop=True)
        )
        latest_prior["consensus_to_prior_filing_days"] = (
            latest_prior["consensus_captured_at"] - latest_prior["accepted_at"]
        ).dt.total_seconds() / 86_400.0
        latest_prior["baseline_role"] = "latest_eligible_filing_before_consensus_capture"
    else:
        latest_prior = pd.DataFrame(
            columns=[
                *_filing_columns(),
                "consensus_to_prior_filing_days",
                "baseline_role",
            ]
        )
    captured_target_ids = set(snapshot_frame["sec_monitor_target_id"].astype(str))
    coverage = targets[["sec_monitor_target_id", "cik", "ticker", "target_period_end"]].copy()
    coverage["submissions_snapshot_observed"] = coverage["sec_monitor_target_id"].isin(
        captured_target_ids
    )
    coverage["future_filing_candidate_observed"] = coverage["sec_monitor_target_id"].isin(
        set(candidates["sec_monitor_target_id"].astype(str))
    )
    snapshot_hashes = sorted(snapshot_frame["sha256"].astype(str))
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_hash,
            "parser_version": FORWARD_FILING_MONITOR_PARSER_VERSION,
            "snapshot_sha256": snapshot_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "forward-filing-monitor-audit-v2-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ForwardFilingMonitorResult(
            target.resolve(), _read_json(target / "forward_filing_monitor_audit_report.json")
        )
    if target.exists():
        raise ForwardFilingMonitorError(f"Incomplete monitor audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "forward_filing_submission_snapshots.parquet": snapshot_frame,
        "forward_filing_observations.parquet": filings,
        "forward_filing_candidates.parquet": candidates,
        "forward_filing_latest_prior_baseline.parquet": latest_prior,
        "forward_filing_monitor_coverage.parquet": coverage,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    baseline_gate = bool(coverage["submissions_snapshot_observed"].all())
    exact_candidates = int(candidates["period_match_status"].astype(str).eq("exact").sum())
    provisional_candidates = int(
        candidates["period_match_status"].astype(str).eq("within_7_days_provisional").sum()
    )
    latest_prior_at = (
        pd.to_datetime(latest_prior["accepted_at"], utc=True, errors="coerce").max()
        if not latest_prior.empty
        else pd.NaT
    )
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "parser_version": FORWARD_FILING_MONITOR_PARSER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_hash,
        "monitor_targets": len(targets),
        "captured_targets": int(coverage["submissions_snapshot_observed"].sum()),
        "submission_snapshots": len(snapshot_frame),
        "eligible_filing_observations": len(filings),
        "future_filing_candidates": len(candidates),
        "exact_period_candidates": exact_candidates,
        "provisional_period_candidates": provisional_candidates,
        "latest_prior_baseline_filings": len(latest_prior),
        "latest_prior_baseline_issuers": int(latest_prior["cik"].nunique()),
        "latest_prior_filing_accepted_at": (
            latest_prior_at.isoformat() if pd.notna(latest_prior_at) else None
        ),
        "monitor_baseline_gate_passed": baseline_gate,
        "future_filing_candidate_gate_passed": exact_candidates > 0,
        "actual_revenue_join_gate_passed": False,
        "actual_revenue_blocking_reason": (
            "awaiting_future_filing" if candidates.empty else "accession_xbrl_actual_not_collected"
        ),
        "return_research_gate_passed": False,
        "historical_backfill_rows": 0,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "snapshot_sha256": snapshot_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_filing_monitor_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(staging, dataset_version, {"plan": plan_hash}, artifacts, report_path)
    staging.replace(target)
    return ForwardFilingMonitorResult(target.resolve(), report)


def _preserve_response(
    raw_root: Path,
    *,
    target_id: str,
    cik: str,
    ticker: str,
    capture_date: date,
    response: SecResponse,
) -> Path:
    sha256 = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "snapshots" / capture_date.isoformat() / ticker / sha256
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "submissions.json"
    if raw_path.exists() and _sha256_file(raw_path) != sha256:
        raise ForwardFilingMonitorError(f"SEC snapshot hash mismatch: {raw_path}")
    if not raw_path.exists():
        raw_path.write_bytes(response.content)
    manifest = {
        "sec_monitor_target_id": target_id,
        "cik": normalize_cik(cik),
        "ticker": ticker,
        "capture_date": capture_date.isoformat(),
        "source_url": response.url,
        "status_code": response.status_code,
        "fetched_at": response.fetched_at.isoformat(),
        "sha256": sha256,
        "bytes": len(response.content),
        "path": str(raw_path.resolve()),
        "response_headers": {
            key: value
            for key, value in response.headers.items()
            if key in {"content-type", "date", "etag", "last-modified"}
        },
    }
    manifest_path = directory / "manifest.json"
    _write_json_immutable(manifest_path, manifest)
    return manifest_path


def _load_snapshot_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    raw_path = Path(str(manifest.get("path", "")))
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != int(manifest.get("bytes", -1))
        or _sha256_file(raw_path) != str(manifest.get("sha256", ""))
    ):
        raise ForwardFilingMonitorError(f"Invalid SEC submissions snapshot: {raw_path}")
    return manifest


def _submission_response_status(content: bytes, expected_cik: str) -> str:
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "malformed_json"
    if not isinstance(document, dict):
        return "bad_schema"
    cik_value = str(document.get("cik", "")).strip()
    try:
        cik_matches = normalize_cik(cik_value) == normalize_cik(expected_cik)
    except ValueError:
        cik_matches = False
    filings = document.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    accessions = recent.get("accessionNumber") if isinstance(recent, dict) else None
    return "ok" if cik_matches and isinstance(accessions, list) else "bad_schema"


def _list_value(document: dict[str, Any], key: str, index: int) -> Any:
    values = document.get(key)
    return values[index] if isinstance(values, list) and index < len(values) else None


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = _timestamp_or_none(value)
    if result is None:
        raise ForwardFilingMonitorError(f"Invalid {label}")
    return result


def _timestamp_or_none(value: Any) -> pd.Timestamp | None:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ForwardFilingMonitorError(f"Invalid {label}")
    return parsed


def _completion_path(raw_root: Path, day: date, target_id: str) -> Path:
    return raw_root / "completions" / day.isoformat() / f"{target_id}.json"


def _write_failure(
    raw_root: Path,
    day: date,
    target_id: str,
    reason: str,
    manifest_path: Path | None,
) -> None:
    path = raw_root / "failures" / day.isoformat() / f"{target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sec_monitor_target_id": target_id,
                "capture_date": day.isoformat(),
                "reason": reason,
                "snapshot_manifest": (
                    str(manifest_path.resolve()) if manifest_path is not None else None
                ),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _snapshot_columns() -> list[str]:
    return [
        "sec_monitor_target_id",
        "cik",
        "ticker",
        "capture_date",
        "source_url",
        "status_code",
        "fetched_at",
        "sha256",
        "bytes",
        "path",
        "response_headers",
    ]


def _filing_columns() -> list[str]:
    return [
        "sec_monitor_target_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "consensus_captured_at",
        "consensus_revenue_level",
        "accession_number",
        "form",
        "filing_date",
        "report_date",
        "accepted_at",
        "primary_document",
        "period_end_distance_days",
        "period_match_status",
        "accepted_after_consensus",
        "observed_after_acceptance",
        "future_filing_candidate",
        "amendment_form",
        "observed_at",
        "availability_clock",
        "historical_backtest_admission",
        "source_sha256",
    ]


def _stable_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForwardFilingMonitorError(f"{label} lacks required columns: {missing}")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json_immutable(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ForwardFilingMonitorError(f"Immutable monitor metadata differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _write_manifest(
    staging: Path,
    dataset_version: str,
    input_hashes: dict[str, str],
    artifacts: dict[str, dict[str, Any]],
    report_path: Path,
) -> None:
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
