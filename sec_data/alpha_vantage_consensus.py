"""Forward-only point-in-time revenue-consensus snapshots from Alpha Vantage."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import pandas as pd

from sec_data.client import normalize_cik
from sec_data.filing_catalog import FilingCatalogError

ALPHA_VANTAGE_URL: Final[str] = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_LIVE_PLAN_VERSION: Final[str] = "alpha-vantage-live-consensus-plan-v1-top20"
ALPHA_VANTAGE_PARSER_VERSION: Final[str] = "alpha-vantage-live-consensus-parser-v2"


class AlphaVantageConsensusError(FilingCatalogError):
    """Raised when a live consensus snapshot violates its frozen contract."""


class AlphaVantageHttpSender(Protocol):
    """Injectable HTTP sender for live consensus collection."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        """Return one HTTP response."""
        ...


@dataclass(frozen=True)
class AlphaVantageLiveConfig:
    """Frozen daily breadth and point-in-time capture settings."""

    universe_size: int = 20
    daily_request_cap: int = 20
    version: str = ALPHA_VANTAGE_LIVE_PLAN_VERSION

    def validate(self) -> None:
        """Keep the plan inside the documented free daily allowance."""
        if not 1 <= self.universe_size <= 25:
            raise ValueError("Live consensus universe must contain 1-25 issuers")
        if not self.universe_size <= self.daily_request_cap <= 25:
            raise ValueError("Live consensus daily cap must cover the universe and not exceed 25")


@dataclass(frozen=True)
class AlphaVantageResponse:
    """Exact response bytes and secret-free transport metadata."""

    requested_url: str
    response_url: str
    content: bytes
    fetched_at: datetime
    status_code: int
    headers: dict[str, str]


@dataclass(frozen=True)
class AlphaVantageCollectionResult:
    """One resumable daily collection summary."""

    selected_targets: int
    completed_targets: int
    failed_targets: int
    requests_made: int
    blocking_reason: str | None


@dataclass(frozen=True)
class AlphaVantageConsensusResult:
    """Immutable live plan or audit result."""

    output_root: Path
    report: dict[str, Any]


class AlphaVantageConsensusClient:
    """Rate-limited client that never persists its query-string API key."""

    def __init__(
        self,
        api_key: str,
        *,
        request_interval_seconds: float = 12.1,
        timeout_seconds: float = 30.0,
        sender: AlphaVantageHttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        normalized = api_key.strip()
        if len(normalized) < 8 or re.search(r"\s", normalized):
            raise ValueError("ALPHA_VANTAGE_API_KEY is missing or malformed")
        if request_interval_seconds < 12.0 or timeout_seconds <= 0.0:
            raise ValueError("Alpha Vantage timing settings are unsafe")
        self._api_key = normalized
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.sender = sender or httpx.Client(follow_redirects=True)
        self.sleep = sleep
        self.monotonic = monotonic
        self.now = now
        self._last_request_started: float | None = None

    def get_earnings_estimates(self, ticker: str) -> AlphaVantageResponse:
        """Fetch one current estimates snapshot and redact the API key from metadata."""
        symbol = _ticker(ticker)
        self._throttle()
        response = self.sender.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "EARNINGS_ESTIMATES",
                "symbol": symbol,
                "apikey": self._api_key,
            },
            headers={"Accept": "application/json", "User-Agent": "Filing-to-Signal-Lab/1.0"},
            timeout=self.timeout_seconds,
        )
        public_query = urlencode({"function": "EARNINGS_ESTIMATES", "symbol": symbol})
        return AlphaVantageResponse(
            requested_url=_secret_free_url(f"{ALPHA_VANTAGE_URL}?{public_query}"),
            response_url=_secret_free_url(str(response.url)),
            content=response.content,
            fetched_at=self.now(),
            status_code=response.status_code,
            headers={key.lower(): value for key, value in response.headers.items()},
        )

    def _throttle(self) -> None:
        current = self.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (current - self._last_request_started)
            if remaining > 0.0:
                self.sleep(remaining)
                current = self.monotonic()
        self._last_request_started = current


def calculate_alpha_vantage_live_plan(
    core_assets: pd.DataFrame,
    config: AlphaVantageLiveConfig | None = None,
) -> pd.DataFrame:
    """Select a deterministic 20-name forward cohort without using market outcomes."""
    cfg = config or AlphaVantageLiveConfig()
    cfg.validate()
    required = {
        "core_asset_id",
        "core_asset_version",
        "core_asset_status",
        "company_name",
        "sec_cik",
        "sec_ticker",
        "mapping_status",
        "selection_order",
    }
    _require_columns(core_assets, required, "core assets")
    eligible = core_assets.loc[
        core_assets["core_asset_status"].eq("active")
        & core_assets["mapping_status"].eq("accepted")
        & core_assets["sec_cik"].notna()
        & core_assets["sec_ticker"].notna()
        & core_assets["selection_order"].notna()
    ].copy()
    eligible["selection_order"] = pd.to_numeric(eligible["selection_order"], errors="raise").astype(
        int
    )
    targets = eligible.sort_values(["selection_order", "core_asset_id"], kind="mergesort").head(
        cfg.universe_size
    )
    if len(targets) != cfg.universe_size:
        raise AlphaVantageConsensusError("Core assets cannot fill the live consensus universe")
    targets["cik"] = targets["sec_cik"].map(lambda value: normalize_cik(str(value)))
    targets["ticker"] = targets["sec_ticker"].map(_ticker)
    targets["provider"] = "alpha_vantage_earnings_estimates"
    targets["endpoint"] = ALPHA_VANTAGE_URL
    targets["capture_frequency"] = "daily"
    targets["forward_only"] = True
    targets["historical_backfill_allowed"] = False
    targets["selection_uses_return_values"] = False
    targets["return_values_loaded"] = False
    targets["live_consensus_target_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "core_asset_version": row["core_asset_version"],
                "core_asset_id": int(row["core_asset_id"]),
                "cik": row["cik"],
                "ticker": row["ticker"],
                "provider": row["provider"],
            }
        ),
        axis="columns",
    )
    targets["collection_order"] = range(1, len(targets) + 1)
    return targets[
        [
            "live_consensus_target_id",
            "collection_order",
            "core_asset_id",
            "core_asset_version",
            "company_name",
            "cik",
            "ticker",
            "provider",
            "endpoint",
            "capture_frequency",
            "forward_only",
            "historical_backfill_allowed",
            "selection_uses_return_values",
            "return_values_loaded",
        ]
    ].reset_index(drop=True)


def build_alpha_vantage_live_plan(
    core_asset_root: Path,
    *,
    output_root: Path = Path("data/market/alpha_vantage_consensus/targets"),
    config: AlphaVantageLiveConfig | None = None,
) -> AlphaVantageConsensusResult:
    """Persist the forward-only daily consensus cohort."""
    cfg = config or AlphaVantageLiveConfig()
    cfg.validate()
    source_hash = _sha256_file(core_asset_root / "output_manifest.json")
    targets = calculate_alpha_vantage_live_plan(
        pd.read_csv(core_asset_root / "core_asset_82.csv", dtype={"sec_cik": "string"}), cfg
    )
    payload = json.dumps(
        {"core_asset_manifest_sha256": source_hash, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return AlphaVantageConsensusResult(
            target.resolve(), _read_json(target / "alpha_vantage_live_plan_report.json")
        )
    if target.exists():
        raise AlphaVantageConsensusError(f"Incomplete live consensus plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "alpha_vantage_live_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifacts = {"alpha_vantage_live_targets.parquet": _artifact(target_path, len(targets))}
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "contract_version": cfg.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "core_asset_manifest_sha256": source_hash,
        "live_targets": len(targets),
        "live_target_issuers": int(targets["cik"].nunique()),
        "daily_request_cap": cfg.daily_request_cap,
        "forward_only": True,
        "historical_backfill_allowed": False,
        "selection_uses_return_values": False,
        "return_values_loaded": False,
        "config": asdict(cfg),
        "artifacts": artifacts,
    }
    report_path = staging / "alpha_vantage_live_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(staging, dataset_version, {"core_asset": source_hash}, artifacts, report_path)
    staging.replace(target)
    return AlphaVantageConsensusResult(target.resolve(), report)


def collect_alpha_vantage_live_snapshots(
    targets: pd.DataFrame,
    client: AlphaVantageConsensusClient,
    *,
    raw_root: Path,
    capture_date: date | None = None,
    limit: int = 20,
    progress: Callable[[int, int, str], None] | None = None,
) -> AlphaVantageCollectionResult:
    """Capture one immutable current snapshot per pending target for a UTC date."""
    required = {"live_consensus_target_id", "collection_order", "cik", "ticker"}
    _require_columns(targets, required, "live consensus targets")
    if not 1 <= limit <= 25:
        raise ValueError("Live collection limit must be between 1 and 25")
    day = capture_date or datetime.now(timezone.utc).date()
    pending = [
        row
        for row in targets.sort_values("collection_order").itertuples(index=False)
        if not _completion_path(raw_root, day, str(row.live_consensus_target_id)).is_file()
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        response = client.get_earnings_estimates(str(row.ticker))
        requests_made += 1
        manifest_path = _preserve_response(
            raw_root,
            target_id=str(row.live_consensus_target_id),
            cik=str(row.cik),
            ticker=str(row.ticker),
            capture_date=day,
            response=response,
        )
        status = alpha_vantage_response_status(response.content, response.status_code)
        if status == "ok":
            completion = {
                "live_consensus_target_id": str(row.live_consensus_target_id),
                "capture_date": day.isoformat(),
                "ticker": str(row.ticker),
                "snapshot_manifest": str(manifest_path.resolve()),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_immutable(
                _completion_path(raw_root, day, str(row.live_consensus_target_id)), completion
            )
            completed += 1
        else:
            failed += 1
            _write_failure(
                raw_root,
                day,
                str(row.live_consensus_target_id),
                status,
                manifest_path,
            )
            if status in {"rate_limited", "premium_endpoint_or_invalid_key"}:
                blocking_reason = status
                break
        if progress is not None:
            progress(number, len(pending), str(row.live_consensus_target_id))
    return AlphaVantageCollectionResult(
        selected_targets=len(pending),
        completed_targets=completed,
        failed_targets=failed,
        requests_made=requests_made,
        blocking_reason=blocking_reason,
    )


def alpha_vantage_response_status(content: bytes, status_code: int) -> str:
    """Classify a preserved response without exposing its values."""
    if not 200 <= status_code < 300:
        return f"http_{status_code}"
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "malformed_json"
    if not isinstance(document, dict):
        return "bad_schema"
    if "Note" in document:
        return "rate_limited"
    if "Information" in document:
        return "premium_endpoint_or_invalid_key"
    if "Error Message" in document:
        return "provider_error"
    return "ok" if isinstance(document.get("estimates"), list) else "bad_schema"


def parse_alpha_vantage_live_response(
    content: bytes,
    *,
    expected_ticker: str,
    captured_at: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Parse quarterly revenue consensus from one exact live snapshot."""
    if alpha_vantage_response_status(content, 200) != "ok":
        raise AlphaVantageConsensusError("Alpha Vantage snapshot is not parseable")
    document: dict[str, Any] = json.loads(content)
    symbol = str(document.get("symbol", "")).strip().upper()
    if symbol != _ticker(expected_ticker):
        raise AlphaVantageConsensusError("Alpha Vantage response symbol mismatch")
    capture_timestamp = _aware_datetime(captured_at)
    observations: list[dict[str, Any]] = []
    for occurrence, item in enumerate(document["estimates"]):
        if not isinstance(item, dict) or str(item.get("horizon", "")).lower() != "fiscal quarter":
            continue
        average = _float_or_none(item.get("revenue_estimate_average"))
        target_period_end = _date_or_none(item.get("date"))
        if average is None or average <= 0.0 or target_period_end is None:
            continue
        target_period_after_capture = target_period_end >= capture_timestamp.date()
        observations.append(
            {
                "ticker": symbol,
                "target_period_end": target_period_end.isoformat(),
                "horizon": "fiscal quarter",
                "revenue_estimate_average": average,
                "revenue_estimate_high": _float_or_none(item.get("revenue_estimate_high")),
                "revenue_estimate_low": _float_or_none(item.get("revenue_estimate_low")),
                "revenue_estimate_analyst_count": _float_or_none(
                    item.get("revenue_estimate_analyst_count")
                ),
                "captured_at": captured_at,
                "availability_clock": "collector_fetched_at",
                "forward_only_admission": True,
                "historical_backtest_admission": False,
                "target_period_after_capture": target_period_after_capture,
                "forward_event_candidate": target_period_after_capture,
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    return observations


def build_alpha_vantage_live_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/alpha_vantage_consensus"),
    output_root: Path = Path("data/processed/alpha_vantage_consensus"),
) -> AlphaVantageConsensusResult:
    """Persist all verified forward snapshots without backfilling a historical event."""
    plan_hash = _sha256_file(plan_root / "output_manifest.json")
    targets = pd.read_parquet(plan_root / "alpha_vantage_live_targets.parquet")
    snapshots: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for completion_path in raw_root.glob("completions/*/*.json"):
        completion = _read_json(completion_path)
        manifest = _load_snapshot_manifest(Path(str(completion["snapshot_manifest"])))
        snapshots.append(manifest)
        observations.extend(
            parse_alpha_vantage_live_response(
                Path(str(manifest["path"])).read_bytes(),
                expected_ticker=str(manifest["ticker"]),
                captured_at=str(manifest["fetched_at"]),
                source_sha256=str(manifest["sha256"]),
            )
        )
    snapshot_frame = pd.DataFrame(snapshots, columns=_snapshot_columns())
    observation_frame = pd.DataFrame(observations, columns=_observation_columns())
    coverage = targets[["live_consensus_target_id", "cik", "ticker"]].copy()
    available_tickers = set(observation_frame["ticker"].astype(str))
    forward_observations = observation_frame.loc[
        observation_frame["forward_event_candidate"].eq(True)
    ].copy()
    forward_tickers = set(forward_observations["ticker"].astype(str))
    coverage["current_revenue_consensus_observed"] = coverage["ticker"].isin(available_tickers)
    coverage["strict_forward_revenue_consensus_observed"] = coverage["ticker"].isin(forward_tickers)
    latest_capture = (
        pd.to_datetime(observation_frame["captured_at"], utc=True, errors="coerce").max()
        if not observation_frame.empty
        else pd.NaT
    )
    snapshot_hashes = sorted(snapshot_frame["sha256"].astype(str))
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_hash,
            "parser_version": ALPHA_VANTAGE_PARSER_VERSION,
            "snapshot_sha256": snapshot_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        "alpha-vantage-live-consensus-audit-v2-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return AlphaVantageConsensusResult(
            target.resolve(), _read_json(target / "alpha_vantage_live_audit_report.json")
        )
    if target.exists():
        raise AlphaVantageConsensusError(f"Incomplete live consensus audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "alpha_vantage_live_snapshots.parquet": snapshot_frame,
        "alpha_vantage_live_revenue_consensus.parquet": observation_frame,
        "alpha_vantage_live_coverage.parquet": coverage,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    raw_snapshot_gate = bool(coverage["current_revenue_consensus_observed"].all())
    strict_forward_gate = bool(coverage["strict_forward_revenue_consensus_observed"].all())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "parser_version": ALPHA_VANTAGE_PARSER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_hash,
        "live_targets": len(targets),
        "captured_tickers": int(coverage["current_revenue_consensus_observed"].sum()),
        "quarterly_revenue_consensus_observations": len(observation_frame),
        "strict_forward_tickers": int(coverage["strict_forward_revenue_consensus_observed"].sum()),
        "strict_forward_observations": len(forward_observations),
        "retrospective_snapshot_observations": len(observation_frame) - len(forward_observations),
        "latest_capture_at": latest_capture.isoformat() if pd.notna(latest_capture) else None,
        "raw_snapshot_coverage_gate": raw_snapshot_gate,
        "strict_forward_snapshot_gate": strict_forward_gate,
        "live_snapshot_gate": strict_forward_gate,
        "forward_event_join_authorized": strict_forward_gate,
        "historical_backtest_admission": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "snapshot_sha256": snapshot_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "alpha_vantage_live_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(staging, dataset_version, {"plan": plan_hash}, artifacts, report_path)
    staging.replace(target)
    return AlphaVantageConsensusResult(target.resolve(), report)


def _preserve_response(
    raw_root: Path,
    *,
    target_id: str,
    cik: str,
    ticker: str,
    capture_date: date,
    response: AlphaVantageResponse,
) -> Path:
    sha256 = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "snapshots" / capture_date.isoformat() / _ticker(ticker) / sha256
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "response.bin"
    if raw_path.exists() and _sha256_file(raw_path) != sha256:
        raise AlphaVantageConsensusError(f"Live snapshot hash mismatch: {raw_path}")
    if not raw_path.exists():
        raw_path.write_bytes(response.content)
    manifest = {
        "live_consensus_target_id": target_id,
        "cik": normalize_cik(cik),
        "ticker": _ticker(ticker),
        "capture_date": capture_date.isoformat(),
        "requested_url": response.requested_url,
        "response_url": response.response_url,
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
        raise AlphaVantageConsensusError(f"Invalid live consensus snapshot: {raw_path}")
    for url_field in ("requested_url", "response_url"):
        if "apikey=" in str(manifest.get(url_field, "")).lower():
            raise AlphaVantageConsensusError("Live consensus manifest contains an API key")
    return manifest


def _completion_path(raw_root: Path, day: date, target_id: str) -> Path:
    return raw_root / "completions" / day.isoformat() / f"{target_id}.json"


def _write_failure(
    raw_root: Path, day: date, target_id: str, reason: str, manifest_path: Path
) -> None:
    path = raw_root / "failures" / day.isoformat() / f"{target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "live_consensus_target_id": target_id,
                "capture_date": day.isoformat(),
                "reason": reason,
                "snapshot_manifest": str(manifest_path.resolve()),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _secret_free_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {
        "www.alphavantage.co",
        "alphavantage.co",
    }:
        raise AlphaVantageConsensusError("Unexpected Alpha Vantage URL")
    query = urlencode(
        [(key, item) for key, item in parse_qsl(parsed.query) if key.lower() != "apikey"]
    )
    return urlunparse(("https", parsed.netloc.lower(), parsed.path, "", query, ""))


def _ticker(value: Any) -> str:
    ticker = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-/]{0,31}", ticker):
        raise AlphaVantageConsensusError(f"Invalid ticker: {value}")
    return ticker


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _date_or_none(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _aware_datetime(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlphaVantageConsensusError("Invalid consensus capture timestamp") from exc
    if result.tzinfo is None:
        raise AlphaVantageConsensusError("Consensus capture timestamp lacks a timezone")
    return result


def _stable_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AlphaVantageConsensusError(f"{label} lacks required columns: {missing}")


def _observation_columns() -> list[str]:
    return [
        "ticker",
        "target_period_end",
        "horizon",
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
        "captured_at",
        "availability_clock",
        "forward_only_admission",
        "historical_backtest_admission",
        "target_period_after_capture",
        "forward_event_candidate",
        "source_occurrence",
        "source_sha256",
    ]


def _snapshot_columns() -> list[str]:
    return [
        "live_consensus_target_id",
        "cik",
        "ticker",
        "capture_date",
        "requested_url",
        "response_url",
        "status_code",
        "fetched_at",
        "sha256",
        "bytes",
        "path",
        "response_headers",
    ]


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json_immutable(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise AlphaVantageConsensusError(f"Immutable live metadata differs: {path}")
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
