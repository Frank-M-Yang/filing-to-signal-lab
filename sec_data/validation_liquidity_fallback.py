"""Secondary raw-OHLCV evidence for pre-event liquidity history gaps."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_LIQUIDITY_FALLBACK_VERSION: Final[str] = (
    "validation-liquidity-fallback-v1-yahoo-raw-ohlcv"
)
VALIDATION_LIQUIDITY_FALLBACK_AUDIT_VERSION: Final[str] = (
    "validation-liquidity-fallback-audit-v2-positive-volume"
)
YAHOO_CHART_PREFIX: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart"


class ValidationLiquidityFallbackError(FilingCatalogError):
    """Raised when secondary liquidity evidence is invalid or outcome-informed."""


@dataclass(frozen=True)
class ValidationLiquidityFallbackResult:
    """Immutable fallback target plan or audit."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationLiquidityFallbackCollectionResult:
    """Counts from one resumable secondary-source collection."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int


@dataclass(frozen=True)
class ValidationLiquidityFallbackSnapshot:
    """Content-addressed Yahoo Chart response for one security span."""

    fallback_target_id: str
    ticker: str
    window_start: str
    window_end: str
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


def calculate_validation_liquidity_fallback_targets(
    event_sessions: pd.DataFrame,
    primary_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze every unique primary-source session gap without opening returns."""
    _require_columns(
        event_sessions,
        {
            "security_event_component_id",
            "price_target_id",
            "ticker",
            "session_date",
            "accepted_at",
        },
        "liquidity event sessions",
    )
    _require_columns(primary_bars, {"price_target_id", "session_date"}, "primary bars")
    expected = event_sessions[
        [
            "security_event_component_id",
            "price_target_id",
            "ticker",
            "session_date",
            "accepted_at",
        ]
    ].drop_duplicates()
    observed = primary_bars[["price_target_id", "session_date"]].drop_duplicates()
    gaps = expected.merge(
        observed.assign(primary_print_observed=True),
        on=["price_target_id", "session_date"],
        how="left",
        validate="many_to_one",
    )
    gaps = gaps.loc[gaps["primary_print_observed"].isna()].drop(columns="primary_print_observed")
    if gaps.empty:
        raise ValidationLiquidityFallbackError("Primary liquidity evidence has no gaps")
    gap_map = (
        gaps.groupby(["price_target_id", "ticker", "session_date"], as_index=False)
        .agg(
            consumer_ids=("security_event_component_id", _joined_unique),
            latest_acceptance=("accepted_at", "max"),
        )
        .sort_values(["session_date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    spans = (
        gap_map.groupby(["price_target_id", "ticker"], as_index=False)
        .agg(
            window_start=("session_date", "min"),
            window_end=("session_date", "max"),
            required_gap_sessions=("session_date", "size"),
        )
        .reset_index(drop=True)
    )
    spans["fallback_target_id"] = spans.apply(
        lambda row: _stable_hash(
            {
                "provider": "yahoo_chart",
                "ticker": str(row["ticker"]),
                "window_start": str(row["window_start"]),
                "window_end": str(row["window_end"]),
                "raw_fields": ["open", "high", "low", "close", "volume"],
                "version": VALIDATION_LIQUIDITY_FALLBACK_VERSION,
            }
        ),
        axis=1,
    )
    spans = spans.sort_values(["window_start", "ticker"], kind="mergesort").reset_index(drop=True)
    spans["fallback_acquisition_order"] = spans.index + 1
    spans["source_provider"] = "Yahoo Chart API"
    spans["source_role"] = "secondary_pre_event_liquidity_gap_evidence"
    spans["raw_price_fields"] = "open|high|low|close|volume"
    spans["adjusted_close_used"] = False
    spans["selection_uses_price_coverage"] = True
    spans["selection_uses_return_values"] = False
    spans["outcome_values_loaded"] = False
    spans["returns_computed"] = False
    fallback_lookup = spans.set_index(["price_target_id", "ticker"])["fallback_target_id"].to_dict()
    gap_map["fallback_target_id"] = gap_map.apply(
        lambda row: fallback_lookup[(str(row["price_target_id"]), str(row["ticker"]))],
        axis=1,
    )
    gap_map["selection_uses_return_values"] = False
    gap_map["outcome_values_loaded"] = False
    gap_map["returns_computed"] = False
    return spans, gap_map


def build_validation_liquidity_fallback_plan(
    liquidity_plan_root: Path,
    liquidity_audit_root: Path,
    *,
    output_root: Path = Path("data/market/validation_liquidity_fallback/targets"),
) -> ValidationLiquidityFallbackResult:
    """Persist secondary queries selected solely from primary pre-event gaps."""
    liquidity_plan_manifest = _sha256_file(liquidity_plan_root / "output_manifest.json")
    liquidity_audit_manifest = _sha256_file(liquidity_audit_root / "output_manifest.json")
    liquidity_report = _read_json(liquidity_audit_root / "validation_liquidity_report.json")
    if bool(liquidity_report.get("outcome_values_loaded")) or bool(
        liquidity_report.get("returns_computed")
    ):
        raise ValidationLiquidityFallbackError("Liquidity fallback selection opened outcomes")
    event_sessions = load_catalog_table(
        liquidity_plan_root, "validation_liquidity_event_sessions.parquet"
    )
    primary_bars = load_catalog_table(
        liquidity_audit_root, "validation_liquidity_daily_bars.parquet"
    )
    targets, gaps = calculate_validation_liquidity_fallback_targets(event_sessions, primary_bars)
    payload = json.dumps(
        {
            "version": VALIDATION_LIQUIDITY_FALLBACK_VERSION,
            "liquidity_plan_manifest_sha256": liquidity_plan_manifest,
            "liquidity_audit_manifest_sha256": liquidity_audit_manifest,
            "targets": targets.to_dict(orient="records"),
            "gaps": gaps.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_LIQUIDITY_FALLBACK_VERSION
        + "-targets-"
        + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationLiquidityFallbackResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationLiquidityFallbackError(
            f"Incomplete liquidity fallback plan exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    tables = {
        "validation_liquidity_fallback_targets.parquet": targets,
        "validation_liquidity_primary_gaps.parquet": gaps,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "liquidity_plan_manifest_sha256": liquidity_plan_manifest,
        "liquidity_audit_manifest_sha256": liquidity_audit_manifest,
        "fallback_targets": len(targets),
        "primary_gap_sessions": len(gaps),
        "source_provider": "Yahoo Chart API",
        "source_role": "secondary_pre_event_liquidity_gap_evidence",
        "adjusted_close_used": False,
        "selection_uses_price_coverage": True,
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
                "liquidity_plan_manifest_sha256": liquidity_plan_manifest,
                "liquidity_audit_manifest_sha256": liquidity_audit_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationLiquidityFallbackResult(target.resolve(), report)


def yahoo_liquidity_chart_url(ticker: str, window_start: str, window_end: str) -> str:
    """Build a secret-free Yahoo raw daily-bar query around a frozen span."""
    start = datetime.fromisoformat(window_start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(window_end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    query = urlencode(
        {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": "1d",
            "events": "history",
        }
    )
    return f"{YAHOO_CHART_PREFIX}/{ticker}?{query}"


def parse_yahoo_liquidity_bars(
    content: bytes,
    *,
    expected_ticker: str,
    fallback_target_id: str,
    primary_price_target_id: str,
    window_start: str,
    window_end: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Extract raw daily OHLCV rows and ignore every adjusted-close field."""
    try:
        document = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationLiquidityFallbackError(f"Malformed Yahoo JSON: {exc}") from exc
    chart = document.get("chart") if isinstance(document, dict) else None
    if not isinstance(chart, dict) or chart.get("error") is not None:
        raise ValidationLiquidityFallbackError("Yahoo Chart returned an error")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValidationLiquidityFallbackError("Yahoo Chart result is missing or ambiguous")
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict) or str(meta.get("symbol")) != expected_ticker:
        raise ValidationLiquidityFallbackError("Yahoo Chart ticker mismatch")
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ValidationLiquidityFallbackError("Yahoo Chart has no timestamps or indicators")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ValidationLiquidityFallbackError("Yahoo Chart raw quote block is missing")
    quote = quotes[0]
    timezone_name = str(meta.get("exchangeTimezoneName", "America/New_York"))
    zone = ZoneInfo(timezone_name)
    start = datetime.fromisoformat(window_start).date()
    end = datetime.fromisoformat(window_end).date()
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        session_date = datetime.fromtimestamp(int(timestamp), tz=zone).date()
        if not start <= session_date <= end:
            continue
        values = {
            field: _optional_finite_number(quote.get(field), index)
            for field in ("open", "high", "low", "close", "volume")
        }
        if any(value is None for value in values.values()):
            continue
        numeric_values: dict[str, float] = {}
        for field, value in values.items():
            assert value is not None
            numeric_values[field] = value
        if min(numeric_values[field] for field in ("open", "high", "low", "close")) <= 0:
            continue
        if numeric_values["volume"] < 0:
            continue
        rows.append(
            {
                "fallback_target_id": fallback_target_id,
                "price_target_id": primary_price_target_id,
                "ticker": expected_ticker,
                "session_date": session_date.isoformat(),
                "open_unadjusted": numeric_values["open"],
                "high_unadjusted": numeric_values["high"],
                "low_unadjusted": numeric_values["low"],
                "close_unadjusted": numeric_values["close"],
                "volume": numeric_values["volume"],
                "transactions": None,
                "source_provider": "Yahoo Chart API",
                "source_sha256": source_sha256,
                "adjusted_close_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and frame["session_date"].duplicated().any():
        raise ValidationLiquidityFallbackError("Yahoo Chart returned duplicate session dates")
    return rows


def collect_validation_liquidity_fallback_artifacts(
    targets: pd.DataFrame,
    *,
    raw_root: Path = Path("data/market/validation_liquidity_fallback/raw_store"),
    limit: int = 25,
    request_interval: float = 1.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> ValidationLiquidityFallbackCollectionResult:
    """Collect resumable Yahoo range responses with byte-exact preservation."""
    if limit < 1 or request_interval < 0:
        raise ValidationLiquidityFallbackError("Invalid fallback collection limits")
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
                url = yahoo_liquidity_chart_url(
                    str(row.ticker), str(row.window_start), str(row.window_end)
                )
                response = client.get(url)
                requests += 1
                response.raise_for_status()
                snapshot = _preserve_response(raw_root, row, url, response)
                parse_yahoo_liquidity_bars(
                    response.content,
                    expected_ticker=str(row.ticker),
                    fallback_target_id=str(row.fallback_target_id),
                    primary_price_target_id=str(row.price_target_id),
                    window_start=str(row.window_start),
                    window_end=str(row.window_end),
                    source_sha256=snapshot.sha256,
                )
                _write_completion(raw_root, row, snapshot)
                completed += 1
            except (httpx.HTTPError, ValidationLiquidityFallbackError) as exc:
                failed += 1
                _write_failure(raw_root, str(row.fallback_target_id), str(exc))
            if progress is not None:
                progress(number, len(pending), str(row.fallback_target_id))
            if number < len(pending) and request_interval:
                time.sleep(request_interval)
    return ValidationLiquidityFallbackCollectionResult(len(pending), completed, failed, requests)


def build_validation_liquidity_fallback_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_liquidity_fallback/raw_store"),
    output_root: Path = Path("data/processed/validation_liquidity_fallback"),
) -> ValidationLiquidityFallbackResult:
    """Audit secondary bars and explicit no-print dates without reading returns."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    targets = load_catalog_table(plan_root, "validation_liquidity_fallback_targets.parquet")
    gaps = load_catalog_table(plan_root, "validation_liquidity_primary_gaps.parquet")
    bar_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        path = raw_root / "completed" / f"{row.fallback_target_id}.json"
        if not path.is_file():
            continue
        completion = _read_json(path)
        snapshot = ValidationLiquidityFallbackSnapshot(**completion["snapshot"])
        raw_path = Path(snapshot.path)
        if _sha256_file(raw_path) != snapshot.sha256:
            raise ValidationLiquidityFallbackError("Fallback raw response hash mismatch")
        bar_rows.extend(
            parse_yahoo_liquidity_bars(
                raw_path.read_bytes(),
                expected_ticker=str(row.ticker),
                fallback_target_id=str(row.fallback_target_id),
                primary_price_target_id=str(row.price_target_id),
                window_start=str(row.window_start),
                window_end=str(row.window_end),
                source_sha256=snapshot.sha256,
            )
        )
        completion_rows.append(
            {
                "fallback_target_id": str(row.fallback_target_id),
                "completed_at": completion["completed_at"],
                "source_sha256": snapshot.sha256,
            }
        )
    bars = pd.DataFrame(bar_rows, columns=_bar_columns())
    completions = pd.DataFrame(
        completion_rows, columns=["fallback_target_id", "completed_at", "source_sha256"]
    )
    observed = (
        bars.loc[bars["volume"].astype(float).gt(0), ["price_target_id", "session_date"]]
        .drop_duplicates()
        .assign(fallback_print_observed=True)
    )
    gap_audit = gaps.merge(
        observed,
        on=["price_target_id", "session_date"],
        how="left",
        validate="many_to_one",
    )
    gap_audit["fallback_print_observed"] = (
        gap_audit["fallback_print_observed"].fillna(False).astype(bool)
    )
    completed_ids = set(completions["fallback_target_id"].astype(str))
    gap_audit["fallback_query_complete"] = gap_audit["fallback_target_id"].isin(completed_ids)
    gap_audit["secondary_no_print_observed"] = (
        gap_audit["fallback_query_complete"] & ~gap_audit["fallback_print_observed"]
    )
    target_audit = targets.copy()
    target_audit["provider_query_complete"] = target_audit["fallback_target_id"].isin(completed_ids)
    target_audit["validated_bar_rows"] = (
        target_audit["fallback_target_id"]
        .map(bars.groupby("fallback_target_id").size())
        .fillna(0)
        .astype(int)
    )
    complete = len(completions) == len(targets)
    source_hashes = sorted(completions["source_sha256"].astype(str).unique())
    payload = json.dumps(
        {
            "audit_version": VALIDATION_LIQUIDITY_FALLBACK_AUDIT_VERSION,
            "plan_manifest_sha256": plan_manifest,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_LIQUIDITY_FALLBACK_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationLiquidityFallbackResult(
            target.resolve(), _read_json(target / "validation_liquidity_fallback_report.json")
        )
    if target.exists():
        raise ValidationLiquidityFallbackError(
            f"Incomplete liquidity fallback audit exists: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    tables = {
        "validation_liquidity_fallback_bars.parquet": bars,
        "validation_liquidity_fallback_completions.parquet": completions,
        "validation_liquidity_fallback_gap_audit.parquet": gap_audit,
        "validation_liquidity_fallback_target_audit.parquet": target_audit,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "fallback_targets": len(targets),
        "targets_collected": len(completions),
        "targets_remaining": len(targets) - len(completions),
        "primary_gap_sessions": len(gaps),
        "fallback_prints_observed": int(gap_audit["fallback_print_observed"].sum()),
        "secondary_no_print_sessions": int(gap_audit["secondary_no_print_observed"].sum()),
        "fallback_evidence_gate": complete,
        "source_provider": "Yahoo Chart API",
        "source_role": "secondary_pre_event_liquidity_gap_evidence",
        "adjusted_close_used": False,
        "selection_uses_price_coverage": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_liquidity_fallback_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationLiquidityFallbackResult(target.resolve(), report)


def _preserve_response(
    raw_root: Path,
    row: Any,
    requested_url: str,
    response: httpx.Response,
) -> ValidationLiquidityFallbackSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise ValidationLiquidityFallbackError("Fallback response changed before preservation")
        temporary.replace(path)
    snapshot = ValidationLiquidityFallbackSnapshot(
        fallback_target_id=str(row.fallback_target_id),
        ticker=str(row.ticker),
        window_start=str(row.window_start),
        window_end=str(row.window_end),
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


def _write_completion(
    raw_root: Path, row: Any, snapshot: ValidationLiquidityFallbackSnapshot
) -> None:
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
        json.dumps({"fallback_target_id": target_id, "detail": detail}, indent=2),
        encoding="utf-8",
    )


def _optional_finite_number(values: Any, index: int) -> float | None:
    if not isinstance(values, list) or index >= len(values):
        raise ValidationLiquidityFallbackError("Yahoo Chart raw quote array is invalid")
    value = values[index]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationLiquidityFallbackError("Yahoo Chart raw quote is not numeric")
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _joined_unique(values: pd.Series) -> str:
    return "|".join(sorted(values.astype(str).unique()))


def _bar_columns() -> list[str]:
    return [
        "fallback_target_id",
        "price_target_id",
        "ticker",
        "session_date",
        "open_unadjusted",
        "high_unadjusted",
        "low_unadjusted",
        "close_unadjusted",
        "volume",
        "transactions",
        "source_provider",
        "source_sha256",
        "adjusted_close_used",
    ]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationLiquidityFallbackError(f"{label} lacks columns: {missing}")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
