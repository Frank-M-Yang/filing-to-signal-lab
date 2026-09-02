"""Auditable Massive daily-price ingestion for filing-event pilot windows."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_reference import (
    MassiveConfigurationError,
    MassiveReferenceClient,
    MassiveRequestError,
    MassiveResponse,
)

MASSIVE_AGGREGATES_PREFIX: Final[str] = "https://api.massive.com/v2/aggs/ticker"
MASSIVE_PRICE_PARSER_VERSION: Final[str] = "massive-unadjusted-daily-parser-v1"


@dataclass(frozen=True)
class MassivePriceConfig:
    """Explicit filing-event price-window assumptions for the pilot."""

    pre_event_calendar_days: int = 7
    post_event_calendar_days: int = 45
    entry_session_offset: int = 2
    holding_sessions: int = 20
    adjusted: bool = False
    version: str = "massive-daily-price-pilot-v1"

    def validate(self) -> None:
        """Reject windows that cannot contain the declared execution rule."""
        if self.pre_event_calendar_days < 1 or self.post_event_calendar_days < 30:
            raise MassiveConfigurationError("Massive daily-price window is too short")
        if self.entry_session_offset < 1 or self.holding_sessions < 1:
            raise MassiveConfigurationError("Massive event-session assumptions are invalid")
        if self.adjusted:
            raise MassiveConfigurationError("The research contract requires adjusted=false")


@dataclass(frozen=True)
class MassivePriceTargetPlanResult:
    """Immutable set of event-ticker daily-bar queries."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassivePriceCollectionResult:
    """Counts and stopping reason from one resumable price batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class MassivePriceAuditResult:
    """Immutable audit of observed daily bars and candidate event windows."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassivePriceSnapshot:
    """Content-addressed raw Massive aggregate response."""

    price_target_id: str
    query_id: str
    ticker: str
    window_start: str
    window_end: str
    page_number: int
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


def build_massive_price_target_plan(
    reference_root: Path,
    *,
    output_root: Path = Path("data/market/massive_prices/targets"),
    config: MassivePriceConfig | None = None,
) -> MassivePriceTargetPlanResult:
    """Freeze unadjusted daily-bar requests for unique reference candidates."""
    cfg = config or MassivePriceConfig()
    cfg.validate()
    reference_report = json.loads(
        (reference_root / "massive_reference_report.json").read_text(encoding="utf-8")
    )
    reference_manifest_sha256 = _sha256_file(reference_root / "output_manifest.json")
    sample = load_catalog_table(reference_root, "sample_massive_reference_audit.parquet")
    observations = load_catalog_table(reference_root, "massive_reference_observations.parquet")
    candidates = sample[sample["event_date_reference_candidate"].astype(bool)].copy()
    if candidates.empty:
        raise FilingCatalogError("Massive price plan requires reference candidates")
    selected_ids = candidates["selected_provider_security_id"]
    if selected_ids.isna().any() or selected_ids.astype(str).str.strip().eq("").any():
        raise FilingCatalogError("A reference candidate is missing its stable provider ID")
    selected_sources = _selected_reference_sources(candidates, observations)
    candidates = candidates.merge(
        selected_sources,
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    if candidates["reference_source_sha256"].isna().any():
        raise FilingCatalogError("A selected reference candidate has no traceable raw source")
    event_dates = pd.to_datetime(candidates["query_date"], errors="coerce")
    accepted = pd.to_datetime(candidates["accepted_at"], utc=True, errors="coerce")
    if event_dates.isna().any() or accepted.isna().any():
        raise FilingCatalogError("Massive price targets contain invalid event timestamps")
    candidates["price_window_start"] = (
        event_dates - pd.to_timedelta(cfg.pre_event_calendar_days, unit="D")
    ).dt.strftime("%Y-%m-%d")
    candidates["price_window_end"] = (
        event_dates + pd.to_timedelta(cfg.post_event_calendar_days, unit="D")
    ).dt.strftime("%Y-%m-%d")
    candidates["accepted_at"] = accepted.map(lambda value: value.isoformat())
    candidates["adjusted"] = cfg.adjusted
    candidates["entry_session_offset"] = cfg.entry_session_offset
    candidates["holding_sessions"] = cfg.holding_sessions
    candidates["price_target_id"] = candidates.apply(
        lambda row: _stable_hash(
            {
                "target_id": str(row["target_id"]),
                "provider_security_id": str(row["selected_provider_security_id"]),
                "ticker": str(row["selected_ticker"]),
                "window_start": str(row["price_window_start"]),
                "window_end": str(row["price_window_end"]),
                "adjusted": False,
                "config_version": cfg.version,
            }
        ),
        axis=1,
    )
    candidates = candidates.sort_values(
        ["query_date", "target_id"], ascending=[False, True]
    ).reset_index(drop=True)
    candidates["price_acquisition_order"] = np.arange(1, len(candidates) + 1)
    columns = [
        "price_target_id",
        "price_acquisition_order",
        "target_id",
        "cik",
        "accepted_at",
        "query_date",
        "form",
        "accession_number",
        "selected_ticker",
        "selected_exchange_mic",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
        "reference_source_sha256",
        "currency_name",
        "price_window_start",
        "price_window_end",
        "adjusted",
        "entry_session_offset",
        "holding_sessions",
    ]
    targets = candidates[columns].copy()
    payload = json.dumps(
        {
            "reference_manifest_sha256": reference_manifest_sha256,
            "config": asdict(cfg),
            "targets": targets.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-price-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        existing_report = json.loads((target / "target_report.json").read_text(encoding="utf-8"))
        return MassivePriceTargetPlanResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete Massive price target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "massive_daily_price_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": reference_report["dataset_version"],
        "reference_manifest_sha256": reference_manifest_sha256,
        "config": asdict(cfg),
        "price_targets": len(targets),
        "query_adjusted": False,
        "selection_uses_price_coverage": False,
        "current_ticker_join_used": False,
        "artifacts": {target_path.name: artifact},
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "reference_manifest_sha256": reference_manifest_sha256,
                "artifacts": {target_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassivePriceTargetPlanResult(target.resolve(), report)


def load_massive_price_targets(plan_root: Path) -> pd.DataFrame:
    """Load and minimally validate an immutable Massive price target plan."""
    targets = load_catalog_table(plan_root, "massive_daily_price_targets.parquet")
    if targets.empty or targets["price_target_id"].duplicated().any():
        raise FilingCatalogError("Massive price targets are empty or duplicated")
    if targets["adjusted"].astype(bool).any():
        raise FilingCatalogError("Massive price targets must request adjusted=false")
    return targets


def massive_daily_bars_url(ticker: str, window_start: str, window_end: str) -> str:
    """Build a deterministic secret-free unadjusted daily aggregate URL."""
    normalized_ticker = _required_text(ticker, "ticker")
    start = _parse_date(window_start)
    end = _parse_date(window_end)
    if start > end:
        raise MassiveConfigurationError("Massive price range starts after it ends")
    path_ticker = quote(normalized_ticker, safe="")
    query = urlencode({"adjusted": "false", "sort": "asc", "limit": "50000"})
    return (
        f"{MASSIVE_AGGREGATES_PREFIX}/{path_ticker}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}?{query}"
    )


def collect_massive_price_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_prices"),
    limit: int = 5,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassivePriceCollectionResult:
    """Collect a resumable batch of exact unadjusted aggregate responses."""
    if limit < 1:
        raise MassiveConfigurationError("Massive price collection limit must be positive")
    ordered = targets.sort_values("price_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_target_completion(raw_root, str(row.price_target_id))
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        try:
            query, requests, reused_pages = _collect_price_query(client, raw_root, row)
            requests_made += requests
            reused += reused_pages
            _write_target_completion(raw_root, row, query)
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, str(row.price_target_id), exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), str(row.price_target_id))
    return MassivePriceCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def parse_massive_daily_bars(
    content: bytes,
    *,
    expected_ticker: str,
    price_target_id: str,
    window_start: str,
    window_end: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse one unadjusted aggregate page and reject malformed bars."""
    try:
        document = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MassiveRequestError(
            f"Malformed Massive price JSON: {exc}", reason="malformed_json"
        ) from exc
    if not isinstance(document, dict) or document.get("status") != "OK":
        raise MassiveRequestError("Massive price response status is not OK", reason="bad_status")
    if document.get("adjusted") is not False:
        raise MassiveRequestError(
            "Massive price response is not explicitly unadjusted", reason="adjustment_mismatch"
        )
    response_ticker = _required_text(document.get("ticker"), "response ticker")
    if response_ticker != expected_ticker:
        raise MassiveRequestError(
            "Massive price response ticker mismatch", reason="ticker_mismatch"
        )
    results = document.get("results", [])
    if not isinstance(results, list):
        raise MassiveRequestError("Massive price results must be a list", reason="bad_schema")
    start = _parse_date(window_start)
    end = _parse_date(window_end)
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_timestamps: set[int] = set()
    for occurrence, item in enumerate(results):
        reason = _bar_rejection_reason(item, start, end, seen_timestamps)
        if reason is not None:
            rejections.append(
                {
                    "price_target_id": price_target_id,
                    "ticker": expected_ticker,
                    "source_occurrence": occurrence,
                    "reason": reason,
                    "source_sha256": source_sha256,
                }
            )
            continue
        assert isinstance(item, dict)
        timestamp_ms = int(item["t"])
        seen_timestamps.add(timestamp_ms)
        timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        rows.append(
            {
                "price_target_id": price_target_id,
                "ticker": expected_ticker,
                "session_date": timestamp_utc.date().isoformat(),
                "aggregate_start_at_utc": timestamp_utc.isoformat(),
                "open_unadjusted": float(item["o"]),
                "high_unadjusted": float(item["h"]),
                "low_unadjusted": float(item["l"]),
                "close_unadjusted": float(item["c"]),
                "volume": float(item["v"]),
                "vwap": _optional_finite_float(item.get("vw")),
                "transactions": _optional_nonnegative_int(item.get("n")),
                "otc": bool(item.get("otc", False)),
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    next_url = document.get("next_url")
    return rows, rejections, _safe_optional_next_url(next_url)


def build_massive_price_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/massive_prices"),
    output_root: Path = Path("data/processed/massive_prices"),
) -> MassivePriceAuditResult:
    """Build candidate price windows while keeping all return gates closed."""
    plan_report = json.loads((plan_root / "target_report.json").read_text(encoding="utf-8"))
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    targets = load_massive_price_targets(plan_root)
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    completion_by_target: dict[str, dict[str, Any]] = {}
    for row in targets.itertuples(index=False):
        price_target_id = str(row.price_target_id)
        completion = _load_target_completion(raw_root, price_target_id)
        if completion is None:
            continue
        completion_by_target[price_target_id] = completion
        for snapshot_path in completion["query"]["snapshots"]:
            snapshot = _load_snapshot(Path(snapshot_path))
            snapshots.append(_snapshot_row(snapshot))
            parsed, page_rejections, _ = parse_massive_daily_bars(
                Path(snapshot.path).read_bytes(),
                expected_ticker=str(row.selected_ticker),
                price_target_id=price_target_id,
                window_start=str(row.price_window_start),
                window_end=str(row.price_window_end),
                source_sha256=snapshot.sha256,
            )
            observations.extend(parsed)
            rejections.extend(page_rejections)
    bars = pd.DataFrame(observations, columns=_bar_columns())
    rejected_frame = pd.DataFrame(rejections, columns=_rejection_columns())
    snapshot_frame = pd.DataFrame(snapshots, columns=_snapshot_columns())
    audit, windows = _build_target_audit(targets, bars, rejected_frame, completion_by_target)
    source_hashes = sorted(snapshot_frame.get("sha256", pd.Series(dtype=str)).astype(str).tolist())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "source_hashes": source_hashes,
            "parser_version": MASSIVE_PRICE_PARSER_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-price-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        existing_report = json.loads(
            (target / "massive_price_report.json").read_text(encoding="utf-8")
        )
        return MassivePriceAuditResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete Massive price audit directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs: dict[str, pd.DataFrame] = {
        "massive_daily_price_targets.parquet": targets,
        "massive_daily_price_observations.parquet": bars,
        "massive_daily_price_rejections.parquet": rejected_frame,
        "massive_daily_price_snapshots.parquet": snapshot_frame,
        "sample_massive_price_audit.parquet": audit,
        "event_window_candidates.parquet": windows,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    candidate_count = int(audit["observed_event_window_candidate"].astype(bool).sum())
    collected_count = int(audit["provider_price_query_complete"].astype(bool).sum())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "reference_version": plan_report["reference_version"],
        "reference_manifest_sha256": plan_report["reference_manifest_sha256"],
        "parser_version": MASSIVE_PRICE_PARSER_VERSION,
        "price_targets": len(targets),
        "targets_collected": collected_count,
        "raw_snapshots": len(snapshot_frame),
        "parsed_daily_bars": len(bars),
        "rejected_daily_bars": len(rejected_frame),
        "observed_event_window_candidates": candidate_count,
        "pilot_collection_gate": collected_count == len(targets),
        "pilot_observed_event_window_gate": candidate_count == len(targets),
        "query_adjusted": False,
        "stable_security_interval_gate": False,
        "historical_exchange_calendar_gate": False,
        "corporate_action_gate": False,
        "delisting_outcome_gate": False,
        "daily_price_gate": False,
        "return_study_gate": False,
        "candidate_returns_computed": False,
        "current_ticker_join_used": False,
        "selection_uses_price_coverage": False,
        "artifacts": artifacts,
    }
    report_path = staging / "massive_price_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "reference_manifest_sha256": plan_report["reference_manifest_sha256"],
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassivePriceAuditResult(target.resolve(), report)


def load_collected_massive_daily_bars(
    targets: pd.DataFrame,
    *,
    raw_root: Path = Path("data/market/massive_prices"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse preserved snapshots for arbitrary compatible price-target rows.

    The helper exposes only validated unadjusted bars, parser rejections, and
    target completion metadata.  It does not calculate candidate or realized
    returns.
    """
    required = {
        "price_target_id",
        "selected_ticker",
        "price_window_start",
        "price_window_end",
    }
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise FilingCatalogError(f"Massive compatible price targets lack columns: {missing}")
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        price_target_id = str(row.price_target_id)
        completion = _load_target_completion(raw_root, price_target_id)
        if completion is None:
            continue
        completion_rows.append(
            {
                "price_target_id": price_target_id,
                "ticker": str(row.selected_ticker),
                "status": str(completion.get("status", "")),
                "bar_count": int(completion["query"].get("bar_count", 0)),
                "rejection_count": int(completion["query"].get("rejection_count", 0)),
                "page_count": int(completion["query"].get("page_count", 0)),
                "completed_at": str(completion.get("completed_at", "")),
            }
        )
        for snapshot_path in completion["query"]["snapshots"]:
            snapshot = _load_snapshot(Path(snapshot_path))
            parsed, page_rejections, _ = parse_massive_daily_bars(
                Path(snapshot.path).read_bytes(),
                expected_ticker=str(row.selected_ticker),
                price_target_id=price_target_id,
                window_start=str(row.price_window_start),
                window_end=str(row.price_window_end),
                source_sha256=snapshot.sha256,
            )
            observations.extend(parsed)
            rejections.extend(page_rejections)
    return (
        pd.DataFrame(observations, columns=_bar_columns()),
        pd.DataFrame(rejections, columns=_rejection_columns()),
        pd.DataFrame(
            completion_rows,
            columns=[
                "price_target_id",
                "ticker",
                "status",
                "bar_count",
                "rejection_count",
                "page_count",
                "completed_at",
            ],
        ),
    )


def _selected_reference_sources(
    candidates: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    selected = candidates[["target_id", "selected_ticker", "selected_provider_security_id"]].merge(
        observations[
            [
                "target_id",
                "ticker",
                "provider_security_id",
                "currency_name",
                "source_sha256",
            ]
        ],
        left_on=["target_id", "selected_ticker", "selected_provider_security_id"],
        right_on=["target_id", "ticker", "provider_security_id"],
        how="left",
        validate="one_to_many",
    )
    selected = selected.drop_duplicates(
        ["target_id", "ticker", "provider_security_id", "source_sha256"]
    )
    counts = selected.groupby("target_id")["source_sha256"].nunique(dropna=True)
    if (counts > 1).any():
        raise FilingCatalogError("A reference candidate traces to multiple raw responses")
    return selected.rename(columns={"source_sha256": "reference_source_sha256"})[
        ["target_id", "currency_name", "reference_source_sha256"]
    ]


def _collect_price_query(
    client: MassiveReferenceClient,
    raw_root: Path,
    row: Any,
) -> tuple[dict[str, Any], int, int]:
    existing = _load_query_completion(raw_root, str(row.price_target_id))
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = massive_daily_bars_url(
        str(row.selected_ticker), str(row.price_window_start), str(row.price_window_end)
    )
    snapshot_paths: list[str] = []
    bar_count = 0
    rejection_count = 0
    requests_made = 0
    reused = 0
    page_number = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError(
                "Massive price pagination exceeded 20 pages", reason="pagination"
            )
        query_id = _stable_hash(
            {
                "price_target_id": str(row.price_target_id),
                "page_url": url,
            }
        )
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(raw_root, row, query_id, page_number, response)
            requests_made += response.requests_made
        else:
            reused += 1
        parsed, rejected, next_url = parse_massive_daily_bars(
            Path(snapshot.path).read_bytes(),
            expected_ticker=str(row.selected_ticker),
            price_target_id=str(row.price_target_id),
            window_start=str(row.price_window_start),
            window_end=str(row.price_window_end),
            source_sha256=snapshot.sha256,
        )
        bar_count += len(parsed)
        rejection_count += len(rejected)
        snapshot_paths.append(str(_snapshot_manifest_path(raw_root, snapshot).resolve()))
        url = next_url
    completion = {
        "price_target_id": str(row.price_target_id),
        "ticker": str(row.selected_ticker),
        "window_start": str(row.price_window_start),
        "window_end": str(row.price_window_end),
        "adjusted": False,
        "bar_count": bar_count,
        "rejection_count": rejection_count,
        "page_count": len(snapshot_paths),
        "pagination_complete": True,
        "snapshots": snapshot_paths,
    }
    _write_query_completion(raw_root, str(row.price_target_id), completion)
    return completion, requests_made, reused


def _preserve_response(
    raw_root: Path,
    row: Any,
    query_id: str,
    page_number: int,
    response: MassiveResponse,
) -> MassivePriceSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Massive price raw snapshot hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Massive price response changed before preservation")
        temporary.replace(path)
    snapshot = MassivePriceSnapshot(
        price_target_id=str(row.price_target_id),
        query_id=query_id,
        ticker=str(row.selected_ticker),
        window_start=str(row.price_window_start),
        window_end=str(row.price_window_end),
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
) -> MassivePriceSnapshot | None:
    for path in sorted((raw_root / "snapshots" / query_id).glob("*.json")):
        snapshot = _load_snapshot(path)
        if snapshot.requested_url == requested_url:
            return snapshot
    return None


def _load_snapshot(path: Path) -> MassivePriceSnapshot:
    snapshot = MassivePriceSnapshot(**json.loads(path.read_text(encoding="utf-8")))
    raw_path = Path(snapshot.path)
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Massive price snapshot is missing or invalid: {raw_path}")
    return snapshot


def _snapshot_manifest_path(raw_root: Path, snapshot: MassivePriceSnapshot) -> Path:
    return raw_root / "snapshots" / snapshot.query_id / f"{snapshot.sha256}.json"


def _write_query_completion(
    raw_root: Path, price_target_id: str, completion: dict[str, Any]
) -> None:
    path = raw_root / "queries" / price_target_id / "daily-bars.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")


def _load_query_completion(raw_root: Path, price_target_id: str) -> dict[str, Any] | None:
    path = raw_root / "queries" / price_target_id / "daily-bars.json"
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("pagination_complete") or document.get("adjusted") is not False:
        return None
    for snapshot_path in document.get("snapshots", []):
        _load_snapshot(Path(snapshot_path))
    return document


def _write_target_completion(raw_root: Path, row: Any, query: dict[str, Any]) -> None:
    path = raw_root / "completed" / f"{row.price_target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "price_target_id": str(row.price_target_id),
                "target_id": str(row.target_id),
                "ticker": str(row.selected_ticker),
                "status": "bars_observed" if query["bar_count"] else "no_bars_observed",
                "query": query,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_target_completion(raw_root: Path, price_target_id: str) -> dict[str, Any] | None:
    path = raw_root / "completed" / f"{price_target_id}.json"
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    query = document.get("query", {})
    if not query.get("pagination_complete") or query.get("adjusted") is not False:
        return None
    for snapshot_path in query.get("snapshots", []):
        _load_snapshot(Path(snapshot_path))
    return document


def _valid_target_completion(raw_root: Path, price_target_id: str) -> bool:
    try:
        return _load_target_completion(raw_root, price_target_id) is not None
    except (FilingCatalogError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _write_failure(raw_root: Path, price_target_id: str, reason: str, detail: str) -> None:
    path = (
        raw_root
        / "failures"
        / price_target_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "price_target_id": price_target_id,
                "reason": reason,
                "detail": detail,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _build_target_audit(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    rejections: pd.DataFrame,
    completions: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = targets.copy()
    audit["provider_price_query_complete"] = audit["price_target_id"].isin(completions)
    audit["daily_bar_count"] = 0
    audit["rejected_bar_count"] = 0
    audit["daily_price_status"] = "not_collected"
    audit["observed_event_window_candidate"] = False
    audit["candidate_entry_session_date"] = pd.NA
    audit["candidate_entry_open_unadjusted"] = np.nan
    audit["candidate_exit_session_date"] = pd.NA
    audit["candidate_exit_close_unadjusted"] = np.nan
    audit["stable_security_interval_ready"] = False
    audit["historical_exchange_calendar_ready"] = False
    audit["corporate_actions_ready"] = False
    audit["delisting_outcome_ready"] = False
    audit["return_study_eligible"] = False
    window_rows: list[dict[str, Any]] = []
    for index, target in audit.iterrows():
        price_target_id = str(target["price_target_id"])
        target_bars = bars[bars["price_target_id"].eq(price_target_id)].copy()
        target_rejections = rejections[rejections["price_target_id"].eq(price_target_id)]
        audit.at[index, "daily_bar_count"] = len(target_bars)
        audit.at[index, "rejected_bar_count"] = len(target_rejections)
        if price_target_id not in completions:
            continue
        if not target_rejections.empty:
            audit.at[index, "daily_price_status"] = "bar_validation_rejection"
            continue
        if target_bars.empty:
            audit.at[index, "daily_price_status"] = "no_daily_bars_observed"
            continue
        if target_bars["session_date"].duplicated().any():
            audit.at[index, "daily_price_status"] = "duplicate_session_date"
            continue
        target_bars["session_date_value"] = pd.to_datetime(target_bars["session_date"])
        target_bars = target_bars.sort_values("session_date_value").reset_index(drop=True)
        event_date = pd.Timestamp(str(target["query_date"]))
        post_event = target_bars[target_bars["session_date_value"] > event_date].reset_index(
            drop=True
        )
        entry_index = int(target["entry_session_offset"]) - 1
        exit_index = entry_index + int(target["holding_sessions"]) - 1
        if len(post_event) <= exit_index:
            audit.at[index, "daily_price_status"] = "insufficient_post_event_observed_bars"
            continue
        entry = post_event.iloc[entry_index]
        exit_bar = post_event.iloc[exit_index]
        audit.at[index, "daily_price_status"] = "observed_bar_window_candidate_calendar_unvalidated"
        audit.at[index, "observed_event_window_candidate"] = True
        audit.at[index, "candidate_entry_session_date"] = entry["session_date"]
        audit.at[index, "candidate_entry_open_unadjusted"] = entry["open_unadjusted"]
        audit.at[index, "candidate_exit_session_date"] = exit_bar["session_date"]
        audit.at[index, "candidate_exit_close_unadjusted"] = exit_bar["close_unadjusted"]
        window_rows.append(
            {
                "price_target_id": price_target_id,
                "target_id": target["target_id"],
                "cik": target["cik"],
                "accepted_at": target["accepted_at"],
                "event_date": target["query_date"],
                "security_id_candidate": target["selected_provider_security_id"],
                "ticker": target["selected_ticker"],
                "exchange_mic": target["selected_exchange_mic"],
                "entry_session_offset": target["entry_session_offset"],
                "holding_sessions": target["holding_sessions"],
                "candidate_entry_session_date": entry["session_date"],
                "candidate_entry_open_unadjusted": entry["open_unadjusted"],
                "candidate_entry_source_sha256": entry["source_sha256"],
                "candidate_exit_session_date": exit_bar["session_date"],
                "candidate_exit_close_unadjusted": exit_bar["close_unadjusted"],
                "candidate_exit_source_sha256": exit_bar["source_sha256"],
                "candidate_only_reason": (
                    "historical security interval and exchange calendar are not validated"
                ),
            }
        )
    windows = pd.DataFrame(window_rows, columns=_window_columns())
    return audit, windows


def _bar_rejection_reason(
    item: Any, start: date, end: date, seen_timestamps: set[int]
) -> str | None:
    if not isinstance(item, dict):
        return "bar_not_object"
    timestamp = item.get("t")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return "invalid_timestamp"
    if not math.isfinite(float(timestamp)) or int(timestamp) != timestamp or int(timestamp) <= 0:
        return "invalid_timestamp"
    timestamp_ms = int(timestamp)
    if timestamp_ms in seen_timestamps:
        return "duplicate_timestamp_in_page"
    session_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()
    if not start <= session_date <= end:
        return "timestamp_outside_requested_window"
    values: dict[str, float] = {}
    for key in ("o", "h", "l", "c", "v"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"invalid_{key}"
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or (key != "v" and numeric == 0):
            return f"invalid_{key}"
        values[key] = numeric
    if values["h"] < max(values["o"], values["l"], values["c"]):
        return "high_price_inconsistent"
    if values["l"] > min(values["o"], values["h"], values["c"]):
        return "low_price_inconsistent"
    return None


def _safe_optional_next_url(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _required_text(value, "next_url")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
        raise MassiveRequestError("Unsafe Massive price next_url", reason="unsafe_pagination")
    safe_query = urlencode(
        [(key, item) for key, item in parse_qsl(parsed.query) if key.casefold() != "apikey"]
    )
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", safe_query, ""))


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or int(value) != value or int(value) < 0:
        return None
    return int(value)


def _required_text(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise MassiveConfigurationError(f"Massive price {label} is missing")
    return normalized


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveConfigurationError(f"Invalid ISO date: {value}") from exc


def _snapshot_row(snapshot: MassivePriceSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def _bar_columns() -> list[str]:
    return [
        "price_target_id",
        "ticker",
        "session_date",
        "aggregate_start_at_utc",
        "open_unadjusted",
        "high_unadjusted",
        "low_unadjusted",
        "close_unadjusted",
        "volume",
        "vwap",
        "transactions",
        "otc",
        "source_occurrence",
        "source_sha256",
    ]


def _rejection_columns() -> list[str]:
    return ["price_target_id", "ticker", "source_occurrence", "reason", "source_sha256"]


def _snapshot_columns() -> list[str]:
    return [field.name for field in MassivePriceSnapshot.__dataclass_fields__.values()]


def _window_columns() -> list[str]:
    return [
        "price_target_id",
        "target_id",
        "cik",
        "accepted_at",
        "event_date",
        "security_id_candidate",
        "ticker",
        "exchange_mic",
        "entry_session_offset",
        "holding_sessions",
        "candidate_entry_session_date",
        "candidate_entry_open_unadjusted",
        "candidate_entry_source_sha256",
        "candidate_exit_session_date",
        "candidate_exit_close_unadjusted",
        "candidate_exit_source_sha256",
        "candidate_only_reason",
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
