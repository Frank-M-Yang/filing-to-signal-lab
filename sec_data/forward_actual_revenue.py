"""Accession-level SEC revenue actuals for the forward consensus experiment.

This module deliberately stops before prices or returns.  It admits an actual only
when an original 10-Q/10-K accession was accepted after the consensus capture and
the preserved companyfacts response contains a unique, same-accession quarterly
revenue fact for the SEC report period.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from sec_data.cleaning import parse_companyfacts
from sec_data.client import COMPANYFACTS_URL, SecClient, SecRequestError, normalize_cik
from sec_data.event_cleaning import REVENUE_TAGS
from sec_data.filing_catalog import FilingCatalogError
from sec_data.models import SecResponse

FORWARD_ACTUAL_PLAN_VERSION: Final[str] = "forward-actual-revenue-plan-v2"
FORWARD_ACTUAL_PARSER_VERSION: Final[str] = "forward-actual-revenue-parser-v2"
DIRECT_QUARTER_MIN_DAYS: Final[int] = 70
DIRECT_QUARTER_MAX_DAYS: Final[int] = 110
ANNUAL_MIN_DAYS: Final[int] = 330
ANNUAL_MAX_DAYS: Final[int] = 400
YTD_MIN_DAYS: Final[int] = 240
YTD_MAX_DAYS: Final[int] = 300
Q3_TO_YEAR_END_MIN_DAYS: Final[int] = 60
Q3_TO_YEAR_END_MAX_DAYS: Final[int] = 130


class ForwardActualRevenueError(FilingCatalogError):
    """Raised when a future SEC actual violates the point-in-time contract."""


@dataclass(frozen=True)
class ForwardActualCollectionResult:
    """Summary of a resumable companyfacts collection run."""

    selected_targets: int
    completed_targets: int
    failed_targets: int
    requests_made: int


@dataclass(frozen=True)
class ForwardActualRevenueResult:
    """Immutable plan or actual-audit output."""

    output_root: Path
    report: dict[str, Any]


def calculate_forward_actual_targets(
    candidates: pd.DataFrame,
    filing_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select original future filings that require accession-level XBRL actuals."""
    required = {
        "sec_monitor_target_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "consensus_captured_at",
        "consensus_revenue_level",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "period_match_status",
        "future_filing_candidate",
        "amendment_form",
        "observed_at",
        "source_sha256",
        "candidate_status",
    }
    _require_columns(candidates, required, "forward filing candidates")
    if candidates.empty:
        return pd.DataFrame(columns=_target_columns())
    frame = candidates.copy()
    frame["cik"] = frame["cik"].astype(str).map(normalize_cik)
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    for column in [
        "target_period_end",
        "consensus_captured_at",
        "report_date",
        "accepted_at",
        "observed_at",
    ]:
        frame[column] = _utc(frame[column], column)
    if not frame["future_filing_candidate"].astype(bool).all():
        raise ForwardActualRevenueError("An actual target is not a future filing candidate")
    if not frame["accepted_at"].gt(frame["consensus_captured_at"]).all():
        raise ForwardActualRevenueError("An actual target was not accepted after consensus")
    if not frame["observed_at"].ge(frame["accepted_at"]).all():
        raise ForwardActualRevenueError("A filing was observed before SEC acceptance")
    originals = frame.loc[
        frame["form"].astype(str).isin(["10-Q", "10-K"]) & ~frame["amendment_form"].astype(bool)
    ].copy()
    if originals.empty:
        return pd.DataFrame(columns=_target_columns())
    if originals.duplicated(["cik", "accession_number"]).any():
        raise ForwardActualRevenueError("Duplicate future accession targets exist")
    originals["companyfacts_url"] = originals["cik"].map(
        lambda cik: COMPANYFACTS_URL.format(cik=cik)
    )
    originals["actual_availability_clock"] = "sec_acceptance_datetime"
    originals["historical_backfill_admission"] = False
    originals["return_values_loaded"] = False
    originals = _attach_q4_prior_filing_metadata(originals, filing_observations)
    originals["forward_actual_target_id"] = originals.apply(
        lambda row: _stable_hash(
            {
                "sec_monitor_target_id": row["sec_monitor_target_id"],
                "accession_number": row["accession_number"],
                "accepted_at": row["accepted_at"].isoformat(),
            }
        ),
        axis="columns",
    )
    originals = originals.sort_values(
        ["accepted_at", "ticker", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)
    originals["collection_order"] = range(1, len(originals) + 1)
    return originals[_target_columns()]


def _attach_q4_prior_filing_metadata(
    targets: pd.DataFrame,
    filing_observations: pd.DataFrame | None,
) -> pd.DataFrame:
    output = targets.copy()
    output["q4_prior_10q_accession"] = None
    output["q4_prior_10q_accepted_at"] = pd.NaT
    output["q4_prior_10q_report_date"] = pd.NaT
    output["q4_prior_10q_form"] = None
    output["q4_derivation_status"] = "not_applicable"
    ten_k_indexes = output.index[output["form"].astype(str).eq("10-K")]
    if len(ten_k_indexes) == 0:
        return output
    output.loc[ten_k_indexes, "q4_derivation_status"] = "missing_prior_9m_filing_metadata"
    if filing_observations is None or filing_observations.empty:
        return output
    _require_columns(
        filing_observations,
        {"cik", "accession_number", "form", "report_date", "accepted_at", "amendment_form"},
        "forward filing observations",
    )
    observations = filing_observations.copy()
    observations["cik"] = observations["cik"].astype(str).map(normalize_cik)
    observations["report_date"] = pd.to_datetime(
        observations["report_date"], utc=True, errors="coerce"
    )
    observations["accepted_at"] = pd.to_datetime(
        observations["accepted_at"], utc=True, errors="coerce"
    )
    observations = observations.loc[
        observations["form"].astype(str).eq("10-Q")
        & ~observations["amendment_form"].astype(bool)
        & observations["report_date"].notna()
        & observations["accepted_at"].notna()
    ].copy()
    for index in ten_k_indexes:
        target = output.loc[index]
        target_report = _timestamp(target["report_date"], "10-K report_date").normalize()
        target_accepted = _timestamp(target["accepted_at"], "10-K accepted_at")
        prior = observations.loc[
            observations["cik"].eq(str(target["cik"]))
            & observations["accepted_at"].lt(target_accepted)
            & observations["report_date"].lt(target_report)
        ].copy()
        if prior.empty:
            continue
        prior["q3_to_year_end_days"] = (target_report - prior["report_date"].dt.normalize()).dt.days
        prior = prior.loc[
            prior["q3_to_year_end_days"].between(Q3_TO_YEAR_END_MIN_DAYS, Q3_TO_YEAR_END_MAX_DAYS)
        ].copy()
        if prior.empty:
            continue
        selected = prior.sort_values(
            ["report_date", "accepted_at", "accession_number"], kind="mergesort"
        ).iloc[-1]
        output.at[index, "q4_prior_10q_accession"] = str(selected["accession_number"])
        output.at[index, "q4_prior_10q_accepted_at"] = selected["accepted_at"]
        output.at[index, "q4_prior_10q_report_date"] = selected["report_date"]
        output.at[index, "q4_prior_10q_form"] = "10-Q"
        output.at[index, "q4_derivation_status"] = "prior_9m_filing_candidate_identified"
    return output


def build_forward_actual_plan(
    filing_monitor_root: Path,
    *,
    output_root: Path = Path("data/sec/forward_actual_revenue/targets"),
) -> ForwardActualRevenueResult:
    """Persist immutable companyfacts targets from a forward filing audit."""
    input_hash = _sha256_file(filing_monitor_root / "output_manifest.json")
    candidates = pd.read_parquet(filing_monitor_root / "forward_filing_candidates.parquet")
    observations = pd.read_parquet(filing_monitor_root / "forward_filing_observations.parquet")
    targets = calculate_forward_actual_targets(candidates, observations)
    payload = json.dumps(
        {"input_manifest_sha256": input_hash, "contract_version": FORWARD_ACTUAL_PLAN_VERSION},
        sort_keys=True,
    ).encode()
    dataset_version = FORWARD_ACTUAL_PLAN_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ForwardActualRevenueResult(
            target.resolve(), _read_json(target / "forward_actual_revenue_plan_report.json")
        )
    if target.exists():
        raise ForwardActualRevenueError(f"Incomplete actual plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    targets_path = staging / "forward_actual_revenue_targets.parquet"
    targets.to_parquet(targets_path, index=False)
    artifacts = {targets_path.name: _artifact(targets_path, len(targets))}
    amendments_excluded = (
        int(candidates["amendment_form"].astype(bool).sum()) if not candidates.empty else 0
    )
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "contract_version": FORWARD_ACTUAL_PLAN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hash,
        "filing_candidates": len(candidates),
        "original_actual_targets": len(targets),
        "amendment_candidates_excluded": amendments_excluded,
        "historical_backfill_admission": False,
        "return_values_loaded": False,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_actual_revenue_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(
        staging, dataset_version, {"filing_monitor": input_hash}, artifacts, report_path
    )
    staging.replace(target)
    return ForwardActualRevenueResult(target.resolve(), report)


def collect_forward_actual_snapshots(
    targets: pd.DataFrame,
    client: SecClient,
    *,
    raw_root: Path,
    capture_date: date | None = None,
    limit: int = 20,
    progress: Callable[[int, int, str], None] | None = None,
) -> ForwardActualCollectionResult:
    """Capture immutable companyfacts bytes for pending future accessions."""
    _require_columns(
        targets,
        {"forward_actual_target_id", "collection_order", "cik", "ticker", "accession_number"},
        "forward actual targets",
    )
    if limit < 1:
        raise ValueError("Actual collection limit must be positive")
    day = capture_date or datetime.now(timezone.utc).date()
    pending = [
        row
        for row in targets.sort_values("collection_order").itertuples(index=False)
        if not _completion_path(raw_root, str(row.forward_actual_target_id)).is_file()
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    for number, row in enumerate(pending, start=1):
        target_id = str(row.forward_actual_target_id)
        try:
            response = client.get_companyfacts(str(row.cik))
            requests_made += 1
            manifest_path = _preserve_response(
                raw_root,
                target_id=target_id,
                cik=str(row.cik),
                ticker=str(row.ticker),
                accession_number=str(row.accession_number),
                capture_date=day,
                response=response,
            )
            if _companyfacts_response_status(response.content, str(row.cik)) != "ok":
                _write_failure(raw_root, day, target_id, "bad_companyfacts_schema", manifest_path)
                failed += 1
            else:
                _write_json_immutable(
                    _completion_path(raw_root, target_id),
                    {
                        "forward_actual_target_id": target_id,
                        "capture_date": day.isoformat(),
                        "cik": normalize_cik(str(row.cik)),
                        "ticker": str(row.ticker),
                        "accession_number": str(row.accession_number),
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
    return ForwardActualCollectionResult(len(pending), completed, failed, requests_made)


def select_accession_revenue_actual(target: dict[str, Any], facts: pd.DataFrame) -> dict[str, Any]:
    """Resolve one unique direct quarterly revenue fact or an explicit rejection."""
    _require_columns(
        facts,
        {
            "fact_id",
            "taxonomy",
            "tag",
            "unit",
            "value_numeric",
            "period_end",
            "duration_days",
            "accession_number",
            "form",
            "accepted_at",
            "source_snapshot_sha256",
        },
        "parsed companyfacts",
    )
    accession = str(target["accession_number"])
    form = str(target["form"])
    accepted_at = _timestamp(target["accepted_at"], "accepted_at")
    report_date = _timestamp(target["report_date"], "report_date").normalize()
    base = facts.loc[
        facts["accession_number"].astype(str).eq(accession)
        & facts["form"].astype(str).eq(form)
        & facts["taxonomy"].astype(str).eq("us-gaap")
        & facts["tag"].astype(str).isin(REVENUE_TAGS)
        & facts["unit"].astype(str).eq("USD")
    ].copy()
    if not base.empty:
        base["period_end"] = _utc(base["period_end"], "fact period_end").dt.normalize()
        base["accepted_at"] = _utc(base["accepted_at"], "fact accepted_at")
        if not base["accepted_at"].eq(accepted_at).all():
            raise ForwardActualRevenueError("Fact availability differs from accession acceptance")
        base["duration_days"] = pd.to_numeric(base["duration_days"], errors="coerce")
        base["value_numeric"] = pd.to_numeric(base["value_numeric"], errors="coerce")
        base = base.loc[
            base["period_end"].eq(report_date)
            & base["value_numeric"].map(lambda value: math.isfinite(float(value)))
            & base["value_numeric"].gt(0.0)
        ].copy()
    direct = base.loc[
        base["duration_days"].between(DIRECT_QUARTER_MIN_DAYS, DIRECT_QUARTER_MAX_DAYS)
    ].copy()
    selected, status = _unique_priority_fact(direct)
    if selected is not None:
        return _actual_record(
            target,
            selected,
            "direct_quarterly_fact_ready",
            derivation_method="direct_quarterly_fact",
        )
    if status == "ambiguous":
        return _status_record(target, "ambiguous_exact_accession_revenue_fact")
    if form == "10-K":
        annual = base.loc[base["duration_days"].between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)].copy()
        annual_fact, annual_status = _unique_priority_fact(annual)
        if annual_status == "ambiguous":
            return _status_record(target, "ambiguous_exact_accession_annual_revenue_fact")
        if annual_fact is not None:
            derived = _derive_q4_from_prior_nine_months(target, facts, annual_fact)
            if derived is not None:
                return derived
            record = _status_record(target, "annual_only_q4_derivation_required")
            record["annual_revenue_fact_id"] = str(annual_fact["fact_id"])
            record["annual_revenue_value"] = float(annual_fact["value_numeric"])
            return record
    return _status_record(target, "missing_exact_accession_revenue_fact")


def build_forward_actual_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/forward_actual_revenue"),
    output_root: Path = Path("data/processed/forward_actual_revenue"),
) -> ForwardActualRevenueResult:
    """Parse preserved companyfacts and audit accession-level future actuals."""
    plan_hash = _sha256_file(plan_root / "output_manifest.json")
    targets = pd.read_parquet(plan_root / "forward_actual_revenue_targets.parquet")
    target_index = targets.set_index("forward_actual_target_id", drop=False)
    snapshots: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for completion_path in raw_root.glob("completions/*.json"):
        completion = _read_json(completion_path)
        target_id = str(completion["forward_actual_target_id"])
        if target_id not in target_index.index:
            continue
        manifest = _load_snapshot_manifest(Path(str(completion["snapshot_manifest"])))
        snapshots.append(manifest)
        target = cast(dict[str, Any], target_index.loc[target_id].to_dict())
        document: dict[str, Any] = json.loads(Path(str(manifest["path"])).read_bytes())
        accession_metadata = [
            {
                "cik": normalize_cik(str(target["cik"])),
                "accession_number": str(target["accession_number"]),
                "accepted_at": _timestamp(target["accepted_at"], "accepted_at"),
            }
        ]
        prior_accession = target.get("q4_prior_10q_accession")
        prior_accepted = _timestamp_or_none(
            target.get("q4_prior_10q_accepted_at"), "prior 10-Q accepted_at"
        )
        if prior_accession and prior_accepted is not None:
            accession_metadata.append(
                {
                    "cik": normalize_cik(str(target["cik"])),
                    "accession_number": str(prior_accession),
                    "accepted_at": prior_accepted,
                }
            )
        filtered = _filter_companyfacts_for_accessions(
            document,
            {str(row["accession_number"]) for row in accession_metadata},
        )
        filing_metadata = pd.DataFrame(accession_metadata)
        parsed = parse_companyfacts(
            filtered,
            filing_metadata,
            source_url=str(manifest["source_url"]),
            snapshot_hash=str(manifest["sha256"]),
        )
        actual_rows.append(select_accession_revenue_actual(target, parsed.data))
        if not parsed.rejections.empty:
            for rejection in parsed.rejections.to_dict("records"):
                typed_rejection = cast(dict[str, Any], rejection)
                rejection_rows.append({"forward_actual_target_id": target_id, **typed_rejection})
    snapshot_frame = pd.DataFrame(snapshots, columns=_snapshot_columns())
    actuals = pd.DataFrame(actual_rows, columns=_actual_columns())
    rejections = pd.DataFrame(rejection_rows)
    if rejections.empty:
        rejections = pd.DataFrame(
            columns=[
                "forward_actual_target_id",
                "rejection_id",
                "stage",
                "reason",
                "detail",
                "source_record_path",
                "source_snapshot_sha256",
                "source_data",
                "rejected_at",
            ]
        )
    captured_ids = set(snapshot_frame["forward_actual_target_id"].astype(str))
    coverage = targets[
        ["forward_actual_target_id", "cik", "ticker", "accession_number", "accepted_at"]
    ].copy()
    coverage["companyfacts_snapshot_observed"] = coverage["forward_actual_target_id"].isin(
        captured_ids
    )
    ready_ids = set(
        actuals.loc[
            actuals["actual_status"]
            .astype(str)
            .isin(["direct_quarterly_fact_ready", "derived_q4_fy_minus_prior_9m_ready"]),
            "forward_actual_target_id",
        ].astype(str)
    )
    coverage["direct_quarterly_actual_ready"] = coverage["forward_actual_target_id"].isin(ready_ids)
    snapshot_hashes = sorted(snapshot_frame["sha256"].astype(str))
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_hash,
            "parser_version": FORWARD_ACTUAL_PARSER_VERSION,
            "snapshot_sha256": snapshot_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "forward-actual-revenue-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target_root = output_root / dataset_version
    if (target_root / "output_manifest.json").is_file():
        return ForwardActualRevenueResult(
            target_root.resolve(),
            _read_json(target_root / "forward_actual_revenue_audit_report.json"),
        )
    if target_root.exists():
        raise ForwardActualRevenueError(f"Incomplete actual audit exists: {target_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "forward_actual_revenue_snapshots.parquet": snapshot_frame,
        "forward_actual_revenue_observations.parquet": actuals,
        "forward_actual_revenue_rejections.parquet": rejections,
        "forward_actual_revenue_coverage.parquet": coverage,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    direct_ready = int(actuals["actual_status"].astype(str).eq("direct_quarterly_fact_ready").sum())
    derived_ready = int(
        actuals["actual_status"].astype(str).eq("derived_q4_fy_minus_prior_9m_ready").sum()
    )
    ready = direct_ready + derived_ready
    annual_only = int(
        actuals["actual_status"].astype(str).eq("annual_only_q4_derivation_required").sum()
    )
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "parser_version": FORWARD_ACTUAL_PARSER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_hash,
        "actual_targets": len(targets),
        "captured_targets": int(coverage["companyfacts_snapshot_observed"].sum()),
        "actual_observations": len(actuals),
        "direct_quarterly_actuals_ready": direct_ready,
        "derived_q4_actuals_ready": derived_ready,
        "total_quarterly_actuals_ready": ready,
        "annual_only_q4_derivations_pending": annual_only,
        "actual_revenue_join_gate_passed": ready > 0,
        "actual_revenue_blocking_reason": (
            None
            if ready > 0
            else "awaiting_future_filing_candidate"
            if targets.empty
            else "awaiting_accession_companyfacts_or_strict_quarterly_fact"
        ),
        "historical_backfill_rows": 0,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "snapshot_sha256": snapshot_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "forward_actual_revenue_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(staging, dataset_version, {"actual_plan": plan_hash}, artifacts, report_path)
    staging.replace(target_root)
    return ForwardActualRevenueResult(target_root.resolve(), report)


def _unique_priority_fact(frame: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if frame.empty:
        return None, "missing"
    for tag in REVENUE_TAGS:
        tagged = frame.loc[frame["tag"].astype(str).eq(tag)].copy()
        if tagged.empty:
            continue
        values = tagged["value_numeric"].astype(float).drop_duplicates()
        if len(values) != 1:
            return None, "ambiguous"
        tagged = tagged.sort_values(["fact_id"], kind="mergesort")
        return tagged.iloc[0], "ready"
    return None, "missing"


def _derive_q4_from_prior_nine_months(
    target: dict[str, Any], facts: pd.DataFrame, annual_fact: pd.Series
) -> dict[str, Any] | None:
    prior_accession = target.get("q4_prior_10q_accession")
    prior_accepted = _timestamp_or_none(
        target.get("q4_prior_10q_accepted_at"), "prior 10-Q accepted_at"
    )
    prior_report = _timestamp_or_none(
        target.get("q4_prior_10q_report_date"), "prior 10-Q report_date"
    )
    if not prior_accession or prior_accepted is None or prior_report is None:
        return None
    candidates = facts.loc[
        facts["accession_number"].astype(str).eq(str(prior_accession))
        & facts["form"].astype(str).eq("10-Q")
        & facts["taxonomy"].astype(str).eq(str(annual_fact["taxonomy"]))
        & facts["tag"].astype(str).eq(str(annual_fact["tag"]))
        & facts["unit"].astype(str).eq(str(annual_fact["unit"]))
    ].copy()
    if candidates.empty:
        return None
    candidates["period_end"] = _utc(candidates["period_end"], "prior 9M period_end").dt.normalize()
    candidates["accepted_at"] = _utc(candidates["accepted_at"], "prior 9M accepted_at")
    candidates["duration_days"] = pd.to_numeric(candidates["duration_days"], errors="coerce")
    candidates["value_numeric"] = pd.to_numeric(candidates["value_numeric"], errors="coerce")
    candidates = candidates.loc[
        candidates["period_end"].eq(prior_report.normalize())
        & candidates["accepted_at"].eq(prior_accepted)
        & candidates["duration_days"].between(YTD_MIN_DAYS, YTD_MAX_DAYS)
        & candidates["value_numeric"].notna()
        & candidates["value_numeric"].gt(0.0)
    ].copy()
    ytd_fact, status = _unique_priority_fact(candidates)
    if ytd_fact is None or status != "ready":
        return None
    q4_revenue = float(annual_fact["value_numeric"]) - float(ytd_fact["value_numeric"])
    if not math.isfinite(q4_revenue) or q4_revenue <= 0.0:
        return None
    record = _actual_record(
        target,
        annual_fact,
        "derived_q4_fy_minus_prior_9m_ready",
        derivation_method="annual_10k_minus_strictly_prior_original_10q_9m",
    )
    record["actual_revenue"] = q4_revenue
    record["revenue_fact_id"] = _stable_hash(
        {"annual_fact_id": annual_fact["fact_id"], "ytd_fact_id": ytd_fact["fact_id"]}
    )
    record["revenue_duration_days"] = int(
        (
            _timestamp(target["report_date"], "report_date").normalize() - prior_report.normalize()
        ).days
    )
    record["annual_revenue_fact_id"] = str(annual_fact["fact_id"])
    record["annual_revenue_value"] = float(annual_fact["value_numeric"])
    record["ytd_revenue_fact_id"] = str(ytd_fact["fact_id"])
    record["ytd_revenue_value"] = float(ytd_fact["value_numeric"])
    record["ytd_revenue_accession"] = str(ytd_fact["accession_number"])
    return record


def _actual_record(
    target: dict[str, Any],
    fact: pd.Series,
    status: str,
    *,
    derivation_method: str,
) -> dict[str, Any]:
    record = _status_record(target, status)
    record.update(
        {
            "actual_revenue": float(fact["value_numeric"]),
            "revenue_fact_id": str(fact["fact_id"]),
            "revenue_taxonomy": str(fact["taxonomy"]),
            "revenue_tag": str(fact["tag"]),
            "revenue_unit": str(fact["unit"]),
            "revenue_period_end": fact["period_end"],
            "revenue_duration_days": int(fact["duration_days"]),
            "actual_available_at": fact["accepted_at"],
            "source_snapshot_sha256": str(fact["source_snapshot_sha256"]),
            "period_confirmed_by_xbrl": True,
            "derivation_method": derivation_method,
        }
    )
    return record


def _status_record(target: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "forward_actual_target_id": str(target["forward_actual_target_id"]),
        "sec_monitor_target_id": str(target["sec_monitor_target_id"]),
        "live_consensus_target_id": str(target["live_consensus_target_id"]),
        "cik": normalize_cik(str(target["cik"])),
        "ticker": str(target["ticker"]),
        "target_period_end": target["target_period_end"],
        "consensus_captured_at": target["consensus_captured_at"],
        "consensus_revenue_level": float(target["consensus_revenue_level"]),
        "accession_number": str(target["accession_number"]),
        "form": str(target["form"]),
        "report_date": target["report_date"],
        "accepted_at": target["accepted_at"],
        "actual_status": status,
        "actual_revenue": None,
        "revenue_fact_id": None,
        "revenue_taxonomy": None,
        "revenue_tag": None,
        "revenue_unit": None,
        "revenue_period_end": None,
        "revenue_duration_days": None,
        "actual_available_at": None,
        "source_snapshot_sha256": None,
        "period_confirmed_by_xbrl": False,
        "annual_revenue_fact_id": None,
        "annual_revenue_value": None,
        "ytd_revenue_fact_id": None,
        "ytd_revenue_value": None,
        "ytd_revenue_accession": None,
        "derivation_method": None,
        "historical_backfill_admission": False,
        "return_values_loaded": False,
    }


def _filter_companyfacts_for_accessions(
    document: dict[str, Any], accession_numbers: set[str]
) -> dict[str, Any]:
    facts_root = document.get("facts")
    filtered_facts: dict[str, Any] = {}
    if isinstance(facts_root, dict):
        for taxonomy, concepts in facts_root.items():
            if not isinstance(concepts, dict):
                continue
            filtered_concepts: dict[str, Any] = {}
            for tag, concept in concepts.items():
                if tag not in REVENUE_TAGS or not isinstance(concept, dict):
                    continue
                units = concept.get("units")
                if not isinstance(units, dict):
                    continue
                filtered_units: dict[str, Any] = {}
                for unit, rows in units.items():
                    if not isinstance(rows, list):
                        continue
                    selected = [
                        row
                        for row in rows
                        if isinstance(row, dict) and str(row.get("accn", "")) in accession_numbers
                    ]
                    if selected:
                        filtered_units[str(unit)] = selected
                if filtered_units:
                    filtered_concepts[str(tag)] = {**concept, "units": filtered_units}
            if filtered_concepts:
                filtered_facts[str(taxonomy)] = filtered_concepts
    return {
        "cik": document.get("cik"),
        "entityName": document.get("entityName"),
        "facts": filtered_facts,
    }


def _preserve_response(
    raw_root: Path,
    *,
    target_id: str,
    cik: str,
    ticker: str,
    accession_number: str,
    capture_date: date,
    response: SecResponse,
) -> Path:
    sha256 = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "snapshots" / capture_date.isoformat() / ticker / sha256
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "companyfacts.json"
    if raw_path.exists() and _sha256_file(raw_path) != sha256:
        raise ForwardActualRevenueError(f"SEC companyfacts hash mismatch: {raw_path}")
    if not raw_path.exists():
        raw_path.write_bytes(response.content)
    manifest = {
        "forward_actual_target_id": target_id,
        "cik": normalize_cik(cik),
        "ticker": ticker,
        "accession_number": accession_number,
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


def _companyfacts_response_status(content: bytes, expected_cik: str) -> str:
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "malformed_json"
    if not isinstance(document, dict) or not isinstance(document.get("facts"), dict):
        return "bad_schema"
    try:
        return (
            "ok"
            if normalize_cik(str(document.get("cik", ""))) == normalize_cik(expected_cik)
            else "bad_schema"
        )
    except ValueError:
        return "bad_schema"


def _load_snapshot_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    raw_path = Path(str(manifest.get("path", "")))
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != int(manifest.get("bytes", -1))
        or _sha256_file(raw_path) != str(manifest.get("sha256", ""))
    ):
        raise ForwardActualRevenueError(f"Invalid SEC companyfacts snapshot: {raw_path}")
    return manifest


def _target_columns() -> list[str]:
    return [
        "forward_actual_target_id",
        "collection_order",
        "sec_monitor_target_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "consensus_captured_at",
        "consensus_revenue_level",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "observed_at",
        "period_match_status",
        "candidate_status",
        "source_sha256",
        "q4_prior_10q_accession",
        "q4_prior_10q_accepted_at",
        "q4_prior_10q_report_date",
        "q4_prior_10q_form",
        "q4_derivation_status",
        "companyfacts_url",
        "actual_availability_clock",
        "historical_backfill_admission",
        "return_values_loaded",
    ]


def _actual_columns() -> list[str]:
    return [
        "forward_actual_target_id",
        "sec_monitor_target_id",
        "live_consensus_target_id",
        "cik",
        "ticker",
        "target_period_end",
        "consensus_captured_at",
        "consensus_revenue_level",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "actual_status",
        "actual_revenue",
        "revenue_fact_id",
        "revenue_taxonomy",
        "revenue_tag",
        "revenue_unit",
        "revenue_period_end",
        "revenue_duration_days",
        "actual_available_at",
        "source_snapshot_sha256",
        "period_confirmed_by_xbrl",
        "annual_revenue_fact_id",
        "annual_revenue_value",
        "ytd_revenue_fact_id",
        "ytd_revenue_value",
        "ytd_revenue_accession",
        "derivation_method",
        "historical_backfill_admission",
        "return_values_loaded",
    ]


def _snapshot_columns() -> list[str]:
    return [
        "forward_actual_target_id",
        "cik",
        "ticker",
        "accession_number",
        "capture_date",
        "source_url",
        "status_code",
        "fetched_at",
        "sha256",
        "bytes",
        "path",
        "response_headers",
    ]


def _completion_path(raw_root: Path, target_id: str) -> Path:
    return raw_root / "completions" / f"{target_id}.json"


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
                "forward_actual_target_id": target_id,
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


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ForwardActualRevenueError(f"Invalid {label}")
    return pd.Timestamp(result)


def _timestamp_or_none(value: Any, label: str) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return _timestamp(value, label)


def _utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ForwardActualRevenueError(f"Invalid {label}")
    return parsed


def _stable_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForwardActualRevenueError(f"{label} lacks required columns: {missing}")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json_immutable(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ForwardActualRevenueError(f"Immutable actual metadata differs: {path}")
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
