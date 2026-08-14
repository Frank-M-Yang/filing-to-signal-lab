"""Auditable Massive SPY benchmark prices and corporate actions."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_prices import massive_daily_bars_url, parse_massive_daily_bars
from sec_data.massive_reference import (
    MassiveConfigurationError,
    MassiveReferenceClient,
    MassiveRequestError,
    MassiveResponse,
)
from sec_data.massive_security_actions import (
    massive_dividends_url,
    massive_splits_url,
    parse_massive_dividends,
    parse_massive_splits,
)

MASSIVE_BENCHMARK_PARSER_VERSION: Final[str] = "massive-spy-benchmark-parser-v2"
_QUERY_KINDS: Final[tuple[str, ...]] = ("daily_bars", "splits", "dividends")


@dataclass(frozen=True)
class MassiveBenchmarkConfig:
    """Explicit market-proxy specification for pilot excess returns."""

    ticker: str = "SPY"
    cik: str = "0000884394"
    currency: str = "USD"
    benchmark_role: str = "sp500_total_return_proxy"
    adjusted: bool = False
    version: str = "massive-spy-benchmark-v1"

    def validate(self) -> None:
        """Reject implicit or adjusted benchmark substitutions."""
        if self.ticker != "SPY" or self.cik != "0000884394":
            raise MassiveConfigurationError("Benchmark v1 is explicitly frozen to SPY")
        if self.currency != "USD" or self.adjusted:
            raise MassiveConfigurationError("Benchmark v1 requires raw USD prices")


@dataclass(frozen=True)
class MassiveBenchmarkPlanResult:
    """Immutable SPY acquisition target."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveBenchmarkCollectionResult:
    """Counts from one resumable benchmark collection."""

    queries_selected: int
    queries_completed: int
    queries_failed: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class MassiveBenchmarkAuditResult:
    """Immutable benchmark market-data audit."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveBenchmarkSnapshot:
    """Content-addressed raw benchmark response."""

    benchmark_target_id: str
    query_kind: str
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


def build_massive_benchmark_target_plan(
    calendar_root: Path,
    *,
    output_root: Path = Path("data/market/massive_benchmark/targets"),
    config: MassiveBenchmarkConfig | None = None,
) -> MassiveBenchmarkPlanResult:
    """Freeze one SPY range covering every formal pilot entry and exit session."""
    cfg = config or MassiveBenchmarkConfig()
    cfg.validate()
    calendar_report = _read_json(calendar_root / "exchange_calendar_report.json")
    calendar_manifest = _sha256_file(calendar_root / "output_manifest.json")
    alignments = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
    ready = alignments[alignments["formal_event_window_ready"].astype(bool)].copy()
    if ready.empty:
        raise FilingCatalogError("Benchmark plan requires formal event windows")
    entry_dates = pd.to_datetime(ready["formal_entry_session_date"], errors="coerce")
    exit_dates = pd.to_datetime(ready["formal_exit_session_date"], errors="coerce")
    if entry_dates.isna().any() or exit_dates.isna().any():
        raise FilingCatalogError("Benchmark plan contains invalid event dates")
    window_start = entry_dates.min().date().isoformat()
    window_end = exit_dates.max().date().isoformat()
    benchmark_target_id = _stable_hash(
        {
            "ticker": cfg.ticker,
            "cik": cfg.cik,
            "window_start": window_start,
            "window_end": window_end,
            "adjusted": False,
            "config_version": cfg.version,
            "calendar_manifest_sha256": calendar_manifest,
        }
    )
    target_frame = pd.DataFrame(
        [
            {
                "benchmark_target_id": benchmark_target_id,
                "ticker": cfg.ticker,
                "cik": cfg.cik,
                "currency": cfg.currency,
                "benchmark_role": cfg.benchmark_role,
                "window_start": window_start,
                "window_end": window_end,
                "adjusted": False,
            }
        ]
    )
    required_dates = pd.concat(
        [
            ready[["target_id", "formal_entry_session_date"]].rename(
                columns={"formal_entry_session_date": "required_session_date"}
            ),
            ready[["target_id", "formal_exit_session_date"]].rename(
                columns={"formal_exit_session_date": "required_session_date"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    payload = json.dumps(
        {
            "calendar_manifest_sha256": calendar_manifest,
            "config": asdict(cfg),
            "target": target_frame.to_dict(orient="records"),
            "required_dates": required_dates.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-benchmark-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MassiveBenchmarkPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete benchmark target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_benchmark_target.parquet": target_frame,
        "benchmark_required_session_dates.parquet": required_dates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest,
        "config": asdict(cfg),
        "benchmark_target_id": benchmark_target_id,
        "benchmark_ticker": cfg.ticker,
        "window_start": window_start,
        "window_end": window_end,
        "required_event_dates": len(required_dates),
        "queries": list(_QUERY_KINDS),
        "query_adjusted": False,
        "benchmark_identity_method": "explicit_versioned_instrument_specification",
        "current_ticker_join_used": False,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "calendar_manifest_sha256": calendar_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveBenchmarkPlanResult(target.resolve(), report)


def collect_massive_benchmark_artifacts(
    plan_root: Path,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_benchmark"),
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveBenchmarkCollectionResult:
    """Collect unadjusted SPY bars, splits, and dividends with resume support."""
    target = _load_target(plan_root).iloc[0]
    selected = [
        kind
        for kind in _QUERY_KINDS
        if _load_query_completion(raw_root, str(target["benchmark_target_id"]), kind) is None
    ]
    completed = failed = requests_made = reused = 0
    blocking_reason: str | None = None
    for number, kind in enumerate(selected, start=1):
        try:
            _, requests, reused_pages = _collect_query(client, raw_root, target, kind)
            requests_made += requests
            reused += reused_pages
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, str(target["benchmark_target_id"]), kind, exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(selected), kind)
    return MassiveBenchmarkCollectionResult(
        queries_selected=len(selected),
        queries_completed=completed,
        queries_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def build_massive_benchmark_audit(
    plan_root: Path,
    calendar_root: Path,
    *,
    raw_root: Path = Path("data/market/massive_benchmark"),
    output_root: Path = Path("data/processed/massive_benchmark"),
) -> MassiveBenchmarkAuditResult:
    """Audit benchmark bars and actions against every required event date."""
    plan_report = _read_json(plan_root / "target_report.json")
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    calendar_manifest = _sha256_file(calendar_root / "output_manifest.json")
    if plan_report.get("calendar_manifest_sha256") != calendar_manifest:
        raise FilingCatalogError("Benchmark plan does not tie to the calendar")
    target = _load_target(plan_root).iloc[0]
    benchmark_target_id = str(target["benchmark_target_id"])
    completions = {
        kind: completion
        for kind in _QUERY_KINDS
        if (completion := _load_query_completion(raw_root, benchmark_target_id, kind)) is not None
    }
    bars_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    dividend_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    for kind, completion in completions.items():
        for snapshot_path in completion["snapshots"]:
            snapshot = _load_snapshot(Path(snapshot_path))
            snapshot_rows.append(asdict(snapshot))
            parsed, rejected, _ = _parse_snapshot(snapshot, target)
            if kind == "daily_bars":
                bars_rows.extend(parsed)
            elif kind == "splits":
                split_rows.extend(parsed)
            else:
                dividend_rows.extend(parsed)
            rejection_rows.extend(
                {
                    "benchmark_target_id": benchmark_target_id,
                    "query_kind": kind,
                    "source_occurrence": item.get("source_occurrence"),
                    "reason": item.get("reason"),
                    "source_sha256": item.get("source_sha256"),
                }
                for item in rejected
            )
    bars = pd.DataFrame(bars_rows, columns=_bar_columns()).rename(
        columns={"price_target_id": "benchmark_target_id"}
    )
    splits = pd.DataFrame(split_rows, columns=_split_columns())
    dividends = pd.DataFrame(dividend_rows, columns=_dividend_columns())
    rejections = pd.DataFrame(rejection_rows, columns=_rejection_columns())
    snapshots = pd.DataFrame(snapshot_rows, columns=_snapshot_columns())
    actions, action_failures = _normalize_benchmark_actions(
        benchmark_target_id, splits, dividends, calendar_root
    )
    split_duplicates = int(
        splits.duplicated(
            ["execution_date", "split_from", "split_to", "provider_event_id"], keep=False
        ).sum()
    )
    dividend_duplicates = int(
        dividends.duplicated(
            ["ex_dividend_date", "cash_amount", "currency", "provider_event_id"], keep=False
        ).sum()
    )
    action_duplicates = split_duplicates + dividend_duplicates
    required = load_catalog_table(plan_root, "benchmark_required_session_dates.parquet")
    alignment = _build_benchmark_alignment(required, bars)
    queries_complete = len(completions) == len(_QUERY_KINDS)
    price_dates_ready = bool(not alignment.empty and alignment["benchmark_price_ready"].all())
    benchmark_ready = bool(
        queries_complete
        and rejections.empty
        and action_failures == 0
        and action_duplicates == 0
        and price_dates_ready
        and not bars["session_date"].duplicated().any()
    )
    source_hashes = sorted(snapshots.get("sha256", pd.Series(dtype=str)).astype(str).tolist())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "source_hashes": source_hashes,
            "parser_version": MASSIVE_BENCHMARK_PARSER_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-benchmark-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    output = output_root / dataset_version
    if (output / "output_manifest.json").is_file():
        return MassiveBenchmarkAuditResult(
            output.resolve(), _read_json(output / "massive_benchmark_report.json")
        )
    if output.exists():
        raise FilingCatalogError(f"Incomplete benchmark audit directory exists: {output}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "benchmark_daily_price_observations.parquet": bars,
        "benchmark_split_observations.parquet": splits,
        "benchmark_dividend_observations.parquet": dividends,
        "normalized_benchmark_actions.parquet": actions,
        "benchmark_event_window_alignment.parquet": alignment,
        "benchmark_rejections.parquet": rejections,
        "massive_benchmark_snapshots.parquet": snapshots,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "calendar_version": plan_report["calendar_version"],
        "calendar_manifest_sha256": calendar_manifest,
        "parser_version": MASSIVE_BENCHMARK_PARSER_VERSION,
        "benchmark_ticker": str(target["ticker"]),
        "benchmark_role": str(target["benchmark_role"]),
        "queries_complete": len(completions),
        "queries_required": len(_QUERY_KINDS),
        "parsed_daily_bars": len(bars),
        "splits_observed": len(splits),
        "dividends_observed": len(dividends),
        "rejections": len(rejections),
        "action_normalization_failures": action_failures,
        "duplicate_action_rows": action_duplicates,
        "required_event_dates": len(alignment),
        "required_event_dates_ready": int(alignment["benchmark_price_ready"].sum()),
        "benchmark_data_ready": benchmark_ready,
        "query_adjusted": False,
        "dividends_explicit": True,
        "splits_explicit": True,
        "benchmark_identity_method": "explicit_versioned_instrument_specification",
        "full_return_study_gate": False,
        "artifacts": artifacts,
    }
    report_path = staging / "massive_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest,
                "calendar_manifest_sha256": calendar_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(output)
    return MassiveBenchmarkAuditResult(output.resolve(), report)


def _collect_query(
    client: MassiveReferenceClient,
    raw_root: Path,
    target: pd.Series[Any],
    kind: str,
) -> tuple[dict[str, Any], int, int]:
    target_id = str(target["benchmark_target_id"])
    existing = _load_query_completion(raw_root, target_id, kind)
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = _query_url(target, kind)
    snapshot_paths: list[str] = []
    result_count = rejection_count = requests_made = reused = page = 0
    while url is not None:
        page += 1
        if page > 20:
            raise MassiveRequestError("Benchmark pagination exceeded 20 pages", reason="pagination")
        query_id = _stable_hash({"benchmark_target_id": target_id, "kind": kind, "url": url})
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(raw_root, target, kind, query_id, page, response)
            requests_made += response.requests_made
        else:
            reused += 1
        parsed, rejected, next_url = _parse_snapshot(snapshot, target)
        result_count += len(parsed)
        rejection_count += len(rejected)
        snapshot_paths.append(str(_snapshot_manifest_path(raw_root, snapshot).resolve()))
        url = next_url
    completion = {
        "benchmark_target_id": target_id,
        "query_kind": kind,
        "pagination_complete": True,
        "result_count": result_count,
        "rejection_count": rejection_count,
        "snapshots": snapshot_paths,
    }
    _write_query_completion(raw_root, target_id, kind, completion)
    return completion, requests_made, reused


def _query_url(target: pd.Series[Any], kind: str) -> str:
    ticker = str(target["ticker"])
    start, end = str(target["window_start"]), str(target["window_end"])
    if kind == "daily_bars":
        return massive_daily_bars_url(ticker, start, end)
    if kind == "splits":
        return massive_splits_url(ticker, start, end)
    if kind == "dividends":
        return massive_dividends_url(ticker, start, end)
    raise MassiveConfigurationError(f"Unknown benchmark query kind: {kind}")


def _parse_snapshot(
    snapshot: MassiveBenchmarkSnapshot, target: pd.Series[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    content = Path(snapshot.path).read_bytes()
    target_id = str(target["benchmark_target_id"])
    ticker = str(target["ticker"])
    start, end = str(target["window_start"]), str(target["window_end"])
    if snapshot.query_kind == "daily_bars":
        return parse_massive_daily_bars(
            content,
            expected_ticker=ticker,
            price_target_id=target_id,
            window_start=start,
            window_end=end,
            source_sha256=snapshot.sha256,
        )
    if snapshot.query_kind == "splits":
        return parse_massive_splits(
            content,
            action_target_id=target_id,
            expected_ticker=ticker,
            window_start=start,
            window_end=end,
            source_sha256=snapshot.sha256,
        )
    return parse_massive_dividends(
        content,
        action_target_id=target_id,
        expected_ticker=ticker,
        expected_currency=str(target["currency"]),
        window_start=start,
        window_end=end,
        source_sha256=snapshot.sha256,
    )


def _normalize_benchmark_actions(
    benchmark_target_id: str,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    calendar_root: Path,
) -> tuple[pd.DataFrame, int]:
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    opens = sessions.groupby("session_date")["regular_open_at"].agg(
        lambda values: sorted(set(values))
    )
    lookup = {date: values[0] for date, values in opens.items() if len(values) == 1}
    rows: list[dict[str, Any]] = []
    failures = 0
    for item in splits.itertuples(index=False):
        effective_at = lookup.get(str(item.execution_date))
        if effective_at is None:
            failures += 1
            continue
        rows.append(
            {
                "benchmark_target_id": benchmark_target_id,
                "provider_event_id": item.provider_event_id,
                "effective_at": effective_at,
                "action_type": str(item.adjustment_type),
                "cash_amount": np.nan,
                "currency": None,
                "split_from": float(str(item.split_from)),
                "split_to": float(str(item.split_to)),
                "source_snapshot_sha256": str(item.source_sha256),
            }
        )
    for item in dividends.itertuples(index=False):
        effective_at = lookup.get(str(item.ex_dividend_date))
        if effective_at is None:
            failures += 1
            continue
        rows.append(
            {
                "benchmark_target_id": benchmark_target_id,
                "provider_event_id": item.provider_event_id,
                "effective_at": effective_at,
                "action_type": "cash_dividend",
                "cash_amount": float(str(item.cash_amount)),
                "currency": str(item.currency),
                "split_from": np.nan,
                "split_to": np.nan,
                "source_snapshot_sha256": str(item.source_sha256),
            }
        )
    return pd.DataFrame(rows, columns=_normalized_action_columns()), failures


def _build_benchmark_alignment(required: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    frame = required.copy()
    frame = frame.merge(
        bars[
            [
                "session_date",
                "open_unadjusted",
                "close_unadjusted",
                "source_sha256",
            ]
        ],
        left_on="required_session_date",
        right_on="session_date",
        how="left",
        validate="many_to_one",
    )
    frame["benchmark_price_ready"] = (
        frame["open_unadjusted"].notna()
        & frame["close_unadjusted"].notna()
        & frame["source_sha256"].notna()
    )
    return frame.drop(columns="session_date")


def _preserve_response(
    raw_root: Path,
    target: pd.Series[Any],
    kind: str,
    query_id: str,
    page: int,
    response: MassiveResponse,
) -> MassiveBenchmarkSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Benchmark raw snapshot hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Benchmark response changed before preservation")
        temporary.replace(path)
    snapshot = MassiveBenchmarkSnapshot(
        benchmark_target_id=str(target["benchmark_target_id"]),
        query_kind=kind,
        query_id=query_id,
        ticker=str(target["ticker"]),
        window_start=str(target["window_start"]),
        window_end=str(target["window_end"]),
        page_number=page,
        requested_url=response.requested_url,
        response_url=response.response_url,
        sha256=digest,
        fetched_at=response.fetched_at.isoformat(),
        path=str(path.resolve()),
        bytes=len(response.content),
        status_code=response.status_code,
        headers=response.headers,
    )
    manifest = _snapshot_manifest_path(raw_root, snapshot)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(
            json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8"
        )
    return snapshot


def _existing_snapshot(
    raw_root: Path, query_id: str, requested_url: str
) -> MassiveBenchmarkSnapshot | None:
    for path in sorted((raw_root / "snapshots" / query_id).glob("*.json")):
        snapshot = _load_snapshot(path)
        if snapshot.requested_url == requested_url:
            return snapshot
    return None


def _load_snapshot(path: Path) -> MassiveBenchmarkSnapshot:
    snapshot = MassiveBenchmarkSnapshot(**_read_json(path))
    raw_path = Path(snapshot.path)
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Benchmark snapshot is missing or invalid: {raw_path}")
    return snapshot


def _snapshot_manifest_path(raw_root: Path, snapshot: MassiveBenchmarkSnapshot) -> Path:
    return raw_root / "snapshots" / snapshot.query_id / f"{snapshot.sha256}.json"


def _write_query_completion(
    raw_root: Path, target_id: str, kind: str, completion: dict[str, Any]
) -> None:
    path = raw_root / "queries" / target_id / f"{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")


def _load_query_completion(raw_root: Path, target_id: str, kind: str) -> dict[str, Any] | None:
    path = raw_root / "queries" / target_id / f"{kind}.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    if not document.get("pagination_complete") or document.get("query_kind") != kind:
        return None
    for snapshot_path in document.get("snapshots", []):
        _load_snapshot(Path(snapshot_path))
    return document


def _write_failure(raw_root: Path, target_id: str, kind: str, reason: str, detail: str) -> None:
    path = (
        raw_root
        / "failures"
        / target_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{kind}.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "benchmark_target_id": target_id,
                "query_kind": kind,
                "reason": reason,
                "detail": detail,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_target(plan_root: Path) -> pd.DataFrame:
    target = load_catalog_table(plan_root, "massive_benchmark_target.parquet")
    if len(target) != 1 or target["benchmark_target_id"].duplicated().any():
        raise FilingCatalogError("Benchmark plan must contain exactly one target")
    if target["adjusted"].astype(bool).any():
        raise FilingCatalogError("Benchmark target must use adjusted=false")
    return target


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


def _normalized_action_columns() -> list[str]:
    return [
        "benchmark_target_id",
        "provider_event_id",
        "effective_at",
        "action_type",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
        "source_snapshot_sha256",
    ]


def _split_columns() -> list[str]:
    return [
        "action_target_id",
        "provider_event_id",
        "ticker",
        "execution_date",
        "adjustment_type",
        "split_from",
        "split_to",
        "historical_adjustment_factor",
        "source_occurrence",
        "source_sha256",
    ]


def _dividend_columns() -> list[str]:
    return [
        "action_target_id",
        "provider_event_id",
        "ticker",
        "ex_dividend_date",
        "cash_amount",
        "currency",
        "declaration_date",
        "record_date",
        "pay_date",
        "distribution_type",
        "frequency",
        "split_adjusted_cash_amount",
        "historical_adjustment_factor",
        "source_occurrence",
        "source_sha256",
    ]


def _rejection_columns() -> list[str]:
    return [
        "benchmark_target_id",
        "query_kind",
        "source_occurrence",
        "reason",
        "source_sha256",
    ]


def _snapshot_columns() -> list[str]:
    return [
        "benchmark_target_id",
        "query_kind",
        "query_id",
        "ticker",
        "window_start",
        "window_end",
        "page_number",
        "requested_url",
        "response_url",
        "sha256",
        "fetched_at",
        "path",
        "bytes",
        "status_code",
        "headers",
    ]


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
