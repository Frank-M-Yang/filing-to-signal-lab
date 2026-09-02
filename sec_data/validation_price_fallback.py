"""Auditable secondary-source prices for frozen provider-history gaps."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_PRICE_FALLBACK_VERSION: Final[str] = (
    "validation-price-fallback-v2-yahoo-dual-endpoint-raw-ohlc"
)
YAHOO_CHART_PREFIX: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart"


class ValidationPriceFallbackError(FilingCatalogError):
    """Raised when secondary-source price evidence is invalid or outcome-informed."""


@dataclass(frozen=True)
class ValidationPriceFallbackResult:
    """Immutable fallback target plan or audit."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationPriceFallbackCollectionResult:
    """Counts from one resumable secondary-source collection batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int


@dataclass(frozen=True)
class ValidationPriceFallbackSnapshot:
    """Content-addressed Yahoo Chart response."""

    fallback_target_id: str
    ticker: str
    required_session_date: str
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


def calculate_validation_price_fallback_targets(
    execution_exceptions: pd.DataFrame,
    benchmark_alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze only pre-classified external-history gaps and benchmark analogues."""
    _require_columns(
        execution_exceptions,
        {
            "security_event_component_id",
            "price_target_id",
            "ticker",
            "formal_entry_session_date",
            "formal_exit_session_date",
            "entry_price_date_observed",
            "exit_price_date_observed",
            "execution_exception_route",
            "selection_uses_return_values",
            "outcome_values_loaded",
            "returns_computed",
        },
        "execution exceptions",
    )
    _require_columns(
        benchmark_alignment,
        {"target_id", "required_session_date", "benchmark_price_ready"},
        "benchmark alignment",
    )
    if execution_exceptions["selection_uses_return_values"].astype(bool).any():
        raise ValidationPriceFallbackError("Fallback selection used return values")
    if execution_exceptions["outcome_values_loaded"].astype(bool).any():
        raise ValidationPriceFallbackError("Fallback selection opened outcomes")
    if execution_exceptions["returns_computed"].astype(bool).any():
        raise ValidationPriceFallbackError("Fallback selection followed return computation")
    rows: list[dict[str, Any]] = []
    for row in execution_exceptions.itertuples(index=False):
        if not bool(row.entry_price_date_observed):
            rows.append(
                _fallback_target(
                    ticker=str(row.ticker),
                    required_session_date=str(row.formal_entry_session_date),
                    asset_class="stocks",
                    consumer_kind="issuer_formal_entry",
                    consumer_ids=str(row.security_event_component_id),
                    primary_price_target_ids=str(row.price_target_id),
                )
            )
        if not bool(row.exit_price_date_observed):
            rows.append(
                _fallback_target(
                    ticker=str(row.ticker),
                    required_session_date=str(row.formal_exit_session_date),
                    asset_class="stocks",
                    consumer_kind="issuer_formal_exit",
                    consumer_ids=str(row.security_event_component_id),
                    primary_price_target_ids=str(row.price_target_id),
                )
            )
    missing_benchmark = benchmark_alignment.loc[
        ~benchmark_alignment["benchmark_price_ready"].astype(bool)
    ].copy()
    for required_date, group in missing_benchmark.groupby("required_session_date"):
        rows.append(
            _fallback_target(
                ticker="SPY",
                required_session_date=str(required_date),
                asset_class="etf",
                consumer_kind="benchmark_required_session",
                consumer_ids="|".join(sorted(group["target_id"].astype(str).unique())),
                primary_price_target_ids="SPY",
            )
        )
    targets = pd.DataFrame(rows)
    if targets.empty or targets["fallback_target_id"].duplicated().any():
        raise ValidationPriceFallbackError("Fallback targets are empty or duplicated")
    targets = targets.sort_values(
        ["required_session_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    targets["fallback_acquisition_order"] = targets.index + 1
    targets["selection_uses_price_coverage"] = False
    targets["selection_uses_return_values"] = False
    targets["outcome_values_loaded"] = False
    targets["returns_computed"] = False
    targets["price_imputation_allowed"] = False
    return targets


def build_validation_price_fallback_plan(
    exception_root: Path,
    benchmark_audit_root: Path,
    *,
    output_root: Path = Path("data/market/validation_price_fallback/targets"),
) -> ValidationPriceFallbackResult:
    """Persist exact secondary-source queries before fetching fallback prices."""
    exception_manifest = _sha256_file(exception_root / "output_manifest.json")
    benchmark_manifest = _sha256_file(benchmark_audit_root / "output_manifest.json")
    exception_report = _read_json(exception_root / "validation_execution_exception_report.json")
    benchmark_report = _read_json(benchmark_audit_root / "massive_benchmark_report.json")
    targets = calculate_validation_price_fallback_targets(
        load_catalog_table(exception_root, "validation_execution_exceptions.parquet"),
        load_catalog_table(benchmark_audit_root, "benchmark_event_window_alignment.parquet"),
    )
    payload = json.dumps(
        {
            "version": VALIDATION_PRICE_FALLBACK_VERSION,
            "exception_manifest_sha256": exception_manifest,
            "benchmark_manifest_sha256": benchmark_manifest,
            "targets": targets.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_PRICE_FALLBACK_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPriceFallbackResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationPriceFallbackError(f"Incomplete fallback plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    path = staging / "validation_price_fallback_targets.parquet"
    targets.to_parquet(path, index=False)
    artifacts = {path.name: _artifact_record(path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception_version": exception_report["dataset_version"],
        "exception_manifest_sha256": exception_manifest,
        "benchmark_version": benchmark_report["dataset_version"],
        "benchmark_manifest_sha256": benchmark_manifest,
        "fallback_targets": len(targets),
        "source_provider": "Yahoo Chart API",
        "source_role": "secondary_history_gap_evidence",
        "price_fields": "raw_open_high_low_close_volume",
        "adjusted_close_used": False,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "exception_manifest_sha256": exception_manifest,
                "benchmark_manifest_sha256": benchmark_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPriceFallbackResult(target.resolve(), report)


def yahoo_chart_url(ticker: str, required_session_date: str) -> str:
    """Build a secret-free Yahoo Chart range surrounding one required date."""
    required = date.fromisoformat(required_session_date)
    start = datetime.combine(required - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(required + timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc)
    query = urlencode(
        {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": "1d",
            "events": "history",
        }
    )
    return f"{YAHOO_CHART_PREFIX}/{ticker}?{query}"


def collect_validation_price_fallback_artifacts(
    targets: pd.DataFrame,
    *,
    raw_root: Path = Path("data/market/validation_price_fallback"),
    limit: int = 10,
    request_interval: float = 1.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> ValidationPriceFallbackCollectionResult:
    """Collect exact Yahoo responses with resumable content-addressed storage."""
    if limit < 1 or request_interval < 0:
        raise ValidationPriceFallbackError("Invalid fallback collection limits")
    pending = [
        row
        for row in targets.sort_values("fallback_acquisition_order").itertuples(index=False)
        if not (raw_root / "completed" / f"{row.fallback_target_id}.json").is_file()
    ][:limit]
    completed = failed = requests = 0
    headers = {
        "User-Agent": "Filing-to-Signal-Lab/0.4 educational-research",
        "Accept": "application/json",
    }
    with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client:
        for number, row in enumerate(pending, start=1):
            try:
                url = yahoo_chart_url(str(row.ticker), str(row.required_session_date))
                response = client.get(url)
                requests += 1
                response.raise_for_status()
                snapshot = _preserve_response(raw_root, row, url, response)
                parse_yahoo_chart_price(
                    response.content,
                    expected_ticker=str(row.ticker),
                    required_session_date=str(row.required_session_date),
                    fallback_target_id=str(row.fallback_target_id),
                    source_sha256=snapshot.sha256,
                )
                _write_completion(raw_root, row, snapshot)
                completed += 1
            except (httpx.HTTPError, ValidationPriceFallbackError) as exc:
                failed += 1
                _write_failure(raw_root, str(row.fallback_target_id), str(exc))
            if progress is not None:
                progress(number, len(pending), str(row.fallback_target_id))
            if number < len(pending) and request_interval:
                time.sleep(request_interval)
    return ValidationPriceFallbackCollectionResult(len(pending), completed, failed, requests)


def parse_yahoo_chart_price(
    content: bytes,
    *,
    expected_ticker: str,
    required_session_date: str,
    fallback_target_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Extract one exact raw OHLC row and ignore Yahoo adjusted-close fields."""
    try:
        document = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationPriceFallbackError(f"Malformed Yahoo JSON: {exc}") from exc
    chart = document.get("chart") if isinstance(document, dict) else None
    if not isinstance(chart, dict) or chart.get("error") is not None:
        raise ValidationPriceFallbackError("Yahoo Chart returned an error")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValidationPriceFallbackError("Yahoo Chart result is missing or ambiguous")
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict) or str(meta.get("symbol")) != expected_ticker:
        raise ValidationPriceFallbackError("Yahoo Chart ticker mismatch")
    timezone_name = str(meta.get("exchangeTimezoneName", "America/New_York"))
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ValidationPriceFallbackError("Yahoo Chart has no timestamps or indicators")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ValidationPriceFallbackError("Yahoo Chart raw quote block is missing")
    quote = quotes[0]
    required = date.fromisoformat(required_session_date)
    matches = [
        index
        for index, timestamp in enumerate(timestamps)
        if datetime.fromtimestamp(int(timestamp), tz=ZoneInfo(timezone_name)).date() == required
    ]
    if len(matches) != 1:
        raise ValidationPriceFallbackError("Yahoo Chart required date is missing or duplicated")
    index = matches[0]
    values = {
        field: _finite_number(quote.get(field), index, field)
        for field in ("open", "high", "low", "close", "volume")
    }
    if min(values[field] for field in ("open", "high", "low", "close")) <= 0:
        raise ValidationPriceFallbackError("Yahoo Chart contains a non-positive price")
    if values["volume"] < 0:
        raise ValidationPriceFallbackError("Yahoo Chart contains negative volume")
    return {
        "fallback_target_id": fallback_target_id,
        "ticker": expected_ticker,
        "session_date": required.isoformat(),
        "open_unadjusted": values["open"],
        "high_unadjusted": values["high"],
        "low_unadjusted": values["low"],
        "close_unadjusted": values["close"],
        "volume": values["volume"],
        "source_provider": "Yahoo Chart API",
        "source_sha256": source_sha256,
        "adjusted_close_used": False,
    }


def build_validation_price_fallback_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_price_fallback"),
    output_root: Path = Path("data/processed/validation_price_fallback"),
) -> ValidationPriceFallbackResult:
    """Audit exact fallback dates while keeping all return calculations closed."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    targets = load_catalog_table(plan_root, "validation_price_fallback_targets.parquet")
    observations: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        completion_path = raw_root / "completed" / f"{row.fallback_target_id}.json"
        if not completion_path.is_file():
            continue
        completion = _read_json(completion_path)
        snapshot = ValidationPriceFallbackSnapshot(**completion["snapshot"])
        if _sha256_file(Path(snapshot.path)) != snapshot.sha256:
            raise ValidationPriceFallbackError("Fallback raw response hash mismatch")
        observations.append(
            parse_yahoo_chart_price(
                Path(snapshot.path).read_bytes(),
                expected_ticker=str(row.ticker),
                required_session_date=str(row.required_session_date),
                fallback_target_id=str(row.fallback_target_id),
                source_sha256=snapshot.sha256,
            )
        )
        snapshot_rows.append(asdict(snapshot))
        completion_rows.append(
            {
                "fallback_target_id": str(row.fallback_target_id),
                "completed_at": completion["completed_at"],
                "source_sha256": snapshot.sha256,
            }
        )
    prices = pd.DataFrame(observations)
    snapshots = pd.DataFrame(snapshot_rows)
    completions = pd.DataFrame(completion_rows)
    ready = len(prices)
    source_hashes = sorted(prices.get("source_sha256", pd.Series(dtype=str)).tolist())
    payload = json.dumps(
        {"plan_manifest_sha256": plan_manifest, "source_hashes": source_hashes},
        sort_keys=True,
    ).encode()
    dataset_version = (
        "validation-price-fallback-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationPriceFallbackResult(
            target.resolve(), _read_json(target / "validation_price_fallback_report.json")
        )
    if target.exists():
        raise ValidationPriceFallbackError(f"Incomplete fallback audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "fallback_required_dates",
                ready,
                len(targets),
                ready == len(targets),
                "Every frozen external-history gap has an exact raw OHLC observation",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(targets),
                False,
                "Fallback collection does not compute returns",
            ),
        ]
    )
    outputs = {
        "validation_price_fallback_targets.parquet": targets,
        "validation_price_fallback_observations.parquet": prices,
        "validation_price_fallback_snapshots.parquet": snapshots,
        "validation_price_fallback_completions.parquet": completions,
        "validation_price_fallback_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "fallback_targets": len(targets),
        "fallback_prices_ready": ready,
        "fallback_price_gate": ready == len(targets),
        "source_provider": "Yahoo Chart API",
        "source_role": "secondary_history_gap_evidence",
        "adjusted_close_used": False,
        "prices_imputed": 0,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_price_fallback_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest,
                "source_hashes": source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationPriceFallbackResult(target.resolve(), report)


def _fallback_target(
    *,
    ticker: str,
    required_session_date: str,
    asset_class: str,
    consumer_kind: str,
    consumer_ids: str,
    primary_price_target_ids: str,
) -> dict[str, Any]:
    fallback_target_id = _stable_hash(
        {
            "provider": "yahoo_chart",
            "ticker": ticker,
            "required_session_date": required_session_date,
            "consumer_kind": consumer_kind,
            "raw_fields": ["open", "high", "low", "close", "volume"],
        }
    )
    return {
        "fallback_target_id": fallback_target_id,
        "ticker": ticker,
        "required_session_date": required_session_date,
        "asset_class": asset_class,
        "consumer_kind": consumer_kind,
        "consumer_ids": consumer_ids,
        "primary_price_target_ids": primary_price_target_ids,
        "source_provider": "Yahoo Chart API",
        "source_role": "secondary_history_gap_evidence",
        "raw_price_fields": "open|high|low|close|volume",
        "adjusted_close_used": False,
    }


def _preserve_response(
    raw_root: Path,
    row: Any,
    requested_url: str,
    response: httpx.Response,
) -> ValidationPriceFallbackSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise ValidationPriceFallbackError("Fallback response changed before preservation")
        temporary.replace(path)
    snapshot = ValidationPriceFallbackSnapshot(
        fallback_target_id=str(row.fallback_target_id),
        ticker=str(row.ticker),
        required_session_date=str(row.required_session_date),
        requested_url=requested_url,
        response_url=str(response.url),
        sha256=digest,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        path=str(path.resolve()),
        bytes=len(response.content),
        status_code=response.status_code,
        headers={
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "date", "etag", "last-modified"}
        },
    )
    manifest = raw_root / "snapshots" / f"{row.fallback_target_id}-{digest}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(
            json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8"
        )
    return snapshot


def _write_completion(raw_root: Path, row: Any, snapshot: ValidationPriceFallbackSnapshot) -> None:
    path = raw_root / "completed" / f"{row.fallback_target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "fallback_target_id": str(row.fallback_target_id),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "snapshot": asdict(snapshot),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _write_failure(raw_root: Path, target_id: str, detail: str) -> None:
    path = (
        raw_root
        / "failures"
        / target_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"fallback_target_id": target_id, "detail": detail},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _finite_number(values: Any, index: int, field: str) -> float:
    if not isinstance(values, list) or index >= len(values):
        raise ValidationPriceFallbackError(f"Yahoo Chart {field} array is invalid")
    value = values[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationPriceFallbackError(f"Yahoo Chart {field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationPriceFallbackError(f"Yahoo Chart {field} is not finite")
    return number


def _gate(
    gate: str, numerator: int, denominator: int, passed: bool, meaning: str
) -> dict[str, Any]:
    return {
        "gate": gate,
        "numerator": numerator,
        "denominator": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
        "passed": passed,
        "meaning": meaning,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationPriceFallbackError(f"{label} lacks required columns: {missing}")


def _stable_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
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
