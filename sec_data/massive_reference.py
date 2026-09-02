"""Auditable point-in-time security-reference ingestion from Massive."""

from __future__ import annotations

import email.utils
import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import numpy as np
import pandas as pd

from sec_data.client import normalize_cik
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

MASSIVE_TICKERS_URL: Final[str] = "https://api.massive.com/v3/reference/tickers"
MASSIVE_PARSER_VERSION: Final[str] = "massive-pit-reference-parser-v3-empty-status"
MASSIVE_HISTORY_START: Final[date] = date(2003, 9, 10)
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
_FIGI: Final[re.Pattern[str]] = re.compile(r"BBG[A-Z0-9]{9}")
_MIC: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{4}")
_TICKER: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9][A-Z0-9.\-/]{0,31}")


class MassiveConfigurationError(ValueError):
    """Raised when Massive access or query configuration is unsafe."""


class MassiveRequestError(RuntimeError):
    """Raised when a Massive request cannot be completed or decoded."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "request_failed",
        requests_made: int = 0,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.requests_made = requests_made


class MassiveHttpSender(Protocol):
    """Injectable HTTP sender used by the reference client."""

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        """Return one HTTP response."""
        ...


@dataclass(frozen=True)
class MassiveResponse:
    """Exact response bytes plus non-secret request metadata."""

    requested_url: str
    response_url: str
    content: bytes
    fetched_at: datetime
    status_code: int
    headers: dict[str, str]
    requests_made: int


@dataclass(frozen=True)
class MassiveSnapshot:
    """Content-addressed raw Massive reference response."""

    target_id: str
    query_id: str
    query_active: bool
    as_of_date: str
    page_number: int
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


@dataclass(frozen=True)
class MassiveReferenceConfig:
    """Explicit target-priority assumptions for the reference pilot."""

    pilot_as_of_date: str = "2026-08-14"
    basic_history_days: int = 730
    version: str = "massive-reference-pilot-v1"

    def validate(self) -> None:
        """Reject invalid or historically impossible pilot settings."""
        pilot_date = _parse_date(self.pilot_as_of_date)
        if pilot_date < MASSIVE_HISTORY_START or self.basic_history_days < 1:
            raise MassiveConfigurationError("Massive reference pilot configuration is invalid")


@dataclass(frozen=True)
class MassiveTargetPlanResult:
    """Immutable Massive CIK-by-event-date target plan."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveCollectionResult:
    """Counts and stopping reason from one resumable reference batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class MassiveReferenceAuditResult:
    """Immutable point-in-time reference coverage audit."""

    output_root: Path
    report: dict[str, Any]


class MassiveReferenceClient:
    """Header-authenticated, rate-limited client for Massive reference data."""

    def __init__(
        self,
        api_key: str,
        *,
        user_agent: str = "Filing-to-Signal-Lab/1.0",
        request_interval_seconds: float = 12.1,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sender: MassiveHttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        normalized_key = api_key.strip()
        if len(normalized_key) < 8 or re.search(r"\s", normalized_key):
            raise MassiveConfigurationError("MASSIVE_API_KEY is missing or malformed")
        if request_interval_seconds < 0.0 or timeout_seconds <= 0.0:
            raise MassiveConfigurationError("Massive timing configuration is invalid")
        if not 0 <= max_retries <= 8:
            raise MassiveConfigurationError("Massive retry count is invalid")
        self._api_key = normalized_key
        self.user_agent = user_agent.strip()
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sender = sender or httpx.Client(follow_redirects=True)
        self.sleep = sleep
        self.monotonic = monotonic
        self.now = now
        self._last_request_started: float | None = None

    def ticker_reference_url(self, cik: str, as_of_date: str, *, active: bool) -> str:
        """Build a deterministic secret-free CIK/date reference URL."""
        normalized_cik = normalize_cik(cik)
        query_date = _parse_date(as_of_date)
        if query_date < MASSIVE_HISTORY_START:
            raise MassiveConfigurationError("Massive ticker history begins on 2003-09-10")
        query = urlencode(
            {
                "cik": normalized_cik,
                "date": query_date.isoformat(),
                "active": str(active).lower(),
                "order": "asc",
                "limit": "1000",
                "sort": "ticker",
            }
        )
        return f"{MASSIVE_TICKERS_URL}?{query}"

    def ticker_symbol_reference_url(self, ticker: str, as_of_date: str, *, active: bool) -> str:
        """Build a deterministic secret-free ticker/date reference URL."""
        normalized_ticker = _normalized_ticker(ticker)
        if normalized_ticker is None:
            raise MassiveConfigurationError("Massive ticker query symbol is invalid")
        query_date = _parse_date(as_of_date)
        if query_date < MASSIVE_HISTORY_START:
            raise MassiveConfigurationError("Massive ticker history begins on 2003-09-10")
        query = urlencode(
            {
                "ticker": normalized_ticker,
                "date": query_date.isoformat(),
                "active": str(active).lower(),
                "order": "asc",
                "limit": "1000",
                "sort": "ticker",
            }
        )
        return f"{MASSIVE_TICKERS_URL}?{query}"

    def get(self, url: str) -> MassiveResponse:
        """Fetch one safe Massive page without exposing the API key in its URL."""
        safe_url = _safe_massive_url(url)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": self.user_agent,
        }
        attempts = self.max_retries + 1
        last_detail = "request did not start"
        for attempt in range(attempts):
            self._throttle()
            try:
                response = self.sender.get(
                    safe_url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except httpx.RequestError as exc:
                last_detail = str(exc)
                if attempt + 1 == attempts:
                    break
                self.sleep(_backoff_seconds(attempt))
                continue
            if 200 <= response.status_code < 300:
                return MassiveResponse(
                    requested_url=safe_url,
                    response_url=_safe_massive_url(str(response.url)),
                    content=response.content,
                    fetched_at=self.now(),
                    status_code=response.status_code,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    requests_made=attempt + 1,
                )
            if response.status_code == 401:
                raise MassiveRequestError(
                    f"Massive authorization failed for {safe_url}",
                    reason="authentication_failed",
                    requests_made=attempt + 1,
                )
            if response.status_code == 403:
                raise MassiveRequestError(
                    f"Massive plan does not permit {safe_url}",
                    reason="entitlement_denied",
                    requests_made=attempt + 1,
                )
            if response.status_code == 404:
                raise MassiveRequestError(
                    f"Massive resource was not found for {safe_url}",
                    reason="not_found",
                    requests_made=attempt + 1,
                )
            last_detail = f"HTTP {response.status_code}"
            if response.status_code not in _RETRYABLE_STATUSES or attempt + 1 == attempts:
                break
            self.sleep(_retry_delay(response, attempt, self.now()))
        raise MassiveRequestError(
            f"Massive request failed for {safe_url}: {last_detail}",
            requests_made=attempts,
        )

    def _throttle(self) -> None:
        current = self.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (current - self._last_request_started)
            if remaining > 0.0:
                self.sleep(remaining)
                current = self.monotonic()
        self._last_request_started = current


def build_massive_reference_target_plan(
    scope_root: Path,
    *,
    output_root: Path = Path("data/market/massive_reference/targets"),
    config: MassiveReferenceConfig | None = None,
) -> MassiveTargetPlanResult:
    """Freeze all provisional-equity CIK/event-date reference queries."""
    cfg = config or MassiveReferenceConfig()
    cfg.validate()
    scope_report = json.loads((scope_root / "scope_audit_report.json").read_text("utf-8"))
    scope_manifest_sha256 = _sha256_file(scope_root / "output_manifest.json")
    sample = load_catalog_table(scope_root, "sample_scope_audit.parquet")
    targets = sample.loc[sample["provisional_equity_eligible"].astype(bool)].copy()
    if targets.empty:
        raise FilingCatalogError("Massive reference plan requires provisional equity candidates")
    targets["accepted_at"] = pd.to_datetime(targets["accepted_at"], utc=True)
    targets["query_date"] = targets["accepted_at"].dt.date.astype(str)
    cutoff = _parse_date(cfg.pilot_as_of_date) - timedelta(days=cfg.basic_history_days)
    targets["within_documented_basic_history_window"] = targets["query_date"].map(
        lambda value: _parse_date(str(value)) >= cutoff
    )
    targets["provider_query_method"] = "cik_and_point_in_time_date"
    targets["query_active_first"] = True
    targets["query_inactive_only_after_no_active_match"] = True
    targets["ticker_events_automatic_join_allowed"] = False
    targets["selection_uses_provider_coverage"] = False
    targets["provider_history_cutoff_assumption"] = cutoff.isoformat()
    targets = targets.sort_values(
        [
            "within_documented_basic_history_window",
            "calendar_year",
            "sampling_hash",
            "target_id",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    targets["massive_acquisition_order"] = np.arange(1, len(targets) + 1)
    columns = [
        "target_id",
        "massive_acquisition_order",
        "cik",
        "calendar_year",
        "accepted_at",
        "query_date",
        "form",
        "accession_number",
        "sample_size_bucket",
        "sampling_stratum",
        "sampling_hash",
        "stage1_inclusion_probability",
        "stage1_sampling_weight",
        "equity_scope_status",
        "equity_security_titles",
        "equity_trading_symbols",
        "within_documented_basic_history_window",
        "provider_history_cutoff_assumption",
        "provider_query_method",
        "query_active_first",
        "query_inactive_only_after_no_active_match",
        "ticker_events_automatic_join_allowed",
        "selection_uses_provider_coverage",
    ]
    targets = targets[columns]
    payload = json.dumps(
        {
            "scope_manifest_sha256": scope_manifest_sha256,
            "config": asdict(cfg),
            "query_method": "active_true_then_active_false_only_if_empty",
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-reference-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads((target / "target_report.json").read_text("utf-8"))
        return MassiveTargetPlanResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete Massive target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    targets.to_parquet(staging / "massive_reference_targets.parquet", index=False)
    artifact = _artifact_record(staging / "massive_reference_targets.parquet", len(targets))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_version": scope_report["dataset_version"],
        "scope_manifest_sha256": scope_manifest_sha256,
        "config": asdict(cfg),
        "price_mapping_targets": len(targets),
        "distinct_ciks": targets["cik"].nunique(),
        "within_documented_basic_history_window": int(
            targets["within_documented_basic_history_window"].sum()
        ),
        "selection_uses_provider_coverage": False,
        "unresolved_units_replaced": False,
        "query_method": "active_true_then_active_false_only_if_empty",
        "ticker_events_automatic_join_allowed": False,
        "artifacts": {"massive_reference_targets.parquet": artifact},
    }
    (staging / "target_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "scope_manifest_sha256": scope_manifest_sha256,
                "artifacts": {"massive_reference_targets.parquet": artifact},
                "report_sha256": _sha256_file(staging / "target_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveTargetPlanResult(target.resolve(), report)


def load_massive_reference_targets(plan_root: Path) -> pd.DataFrame:
    """Load and minimally validate an immutable Massive target plan."""
    targets = load_catalog_table(plan_root, "massive_reference_targets.parquet")
    if targets["target_id"].duplicated().any() or targets.empty:
        raise FilingCatalogError("Massive reference targets are empty or duplicated")
    return targets


def collect_massive_reference_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_reference"),
    limit: int = 5,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveCollectionResult:
    """Collect a resumable target batch, preserving bytes before parsing."""
    if limit < 1:
        raise MassiveConfigurationError("Massive collection limit must be positive")
    ordered = targets.sort_values("massive_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_target_completion(raw_root, str(row.target_id))
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        target_id = str(row.target_id)
        try:
            active = _collect_reference_query(
                client,
                raw_root=raw_root,
                target_id=target_id,
                cik=str(row.cik),
                as_of_date=str(row.query_date),
                active=True,
            )
            requests_made += active[1]
            reused += active[2]
            queries = {"active": active[0]}
            status = "active_match" if active[0]["record_count"] > 0 else "active_no_match"
            if active[0]["record_count"] == 0:
                inactive = _collect_reference_query(
                    client,
                    raw_root=raw_root,
                    target_id=target_id,
                    cik=str(row.cik),
                    as_of_date=str(row.query_date),
                    active=False,
                )
                requests_made += inactive[1]
                reused += inactive[2]
                queries["inactive"] = inactive[0]
                status = (
                    "inactive_match" if inactive[0]["record_count"] > 0 else "no_provider_match"
                )
            _write_target_completion(
                raw_root,
                target_id=target_id,
                cik=str(row.cik),
                as_of_date=str(row.query_date),
                status=status,
                queries=queries,
            )
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, target_id, exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), target_id)
    return MassiveCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def collect_massive_symbol_reference_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path,
    limit: int = 5,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveCollectionResult:
    """Collect outcome-blind ticker/date fallbacks while enforcing the expected CIK."""
    required = {
        "target_id",
        "cik",
        "query_ticker",
        "query_date",
        "massive_acquisition_order",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise FilingCatalogError(f"Massive symbol targets lack columns: {sorted(missing)}")
    if limit < 1:
        raise MassiveConfigurationError("Massive collection limit must be positive")
    ordered = targets.sort_values("massive_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_target_completion(raw_root, str(row.target_id))
    ][:limit]
    completed = 0
    failed = 0
    requests_made = 0
    reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        target_id = str(row.target_id)
        try:
            active = _collect_symbol_reference_query(
                client,
                raw_root=raw_root,
                target_id=target_id,
                cik=str(row.cik),
                ticker=str(row.query_ticker),
                as_of_date=str(row.query_date),
                active=True,
            )
            requests_made += active[1]
            reused += active[2]
            queries = {"active": active[0]}
            status = "active_match" if active[0]["record_count"] > 0 else "active_no_match"
            if active[0]["record_count"] == 0:
                inactive = _collect_symbol_reference_query(
                    client,
                    raw_root=raw_root,
                    target_id=target_id,
                    cik=str(row.cik),
                    ticker=str(row.query_ticker),
                    as_of_date=str(row.query_date),
                    active=False,
                )
                requests_made += inactive[1]
                reused += inactive[2]
                queries["inactive"] = inactive[0]
                status = (
                    "inactive_match" if inactive[0]["record_count"] > 0 else "no_provider_match"
                )
            _write_target_completion(
                raw_root,
                target_id=target_id,
                cik=str(row.cik),
                as_of_date=str(row.query_date),
                status=status,
                queries=queries,
            )
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, target_id, exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), target_id)
    return MassiveCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def parse_massive_ticker_response(
    content: bytes,
    *,
    expected_cik: str,
    target_id: str,
    query_active: bool,
    as_of_date: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse one preserved ticker-reference page with row-level rejections."""
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRequestError(
            f"Malformed Massive JSON: {exc}", reason="malformed_json"
        ) from exc
    if not isinstance(document, dict) or document.get("status") != "OK":
        raise MassiveRequestError("Massive ticker response status is not OK", reason="bad_status")
    results = document.get("results", [])
    if not isinstance(results, list):
        raise MassiveRequestError("Massive ticker results must be a list", reason="bad_schema")
    normalized_expected = normalize_cik(expected_cik)
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for occurrence, item in enumerate(results):
        if not isinstance(item, dict):
            rejections.append(
                _reference_rejection(
                    target_id,
                    occurrence,
                    "result_not_object",
                    source_sha256,
                )
            )
            continue
        raw_cik = str(item.get("cik", ""))
        try:
            observed_cik = normalize_cik(raw_cik)
        except ValueError:
            observed_cik = ""
        if observed_cik != normalized_expected:
            rejections.append(
                _reference_rejection(
                    target_id,
                    occurrence,
                    "response_cik_mismatch",
                    source_sha256,
                )
            )
            continue
        ticker = _normalized_ticker(item.get("ticker"))
        share_class_figi = _normalized_figi(item.get("share_class_figi"))
        composite_figi = _normalized_figi(item.get("composite_figi"))
        exchange_mic = _normalized_mic(item.get("primary_exchange"))
        provider_active = item.get("active") if isinstance(item.get("active"), bool) else None
        if provider_active is not None and provider_active != query_active:
            rejections.append(
                _reference_rejection(
                    target_id,
                    occurrence,
                    "active_state_mismatch",
                    source_sha256,
                )
            )
            continue
        observations.append(
            {
                "target_id": target_id,
                "cik": observed_cik,
                "as_of_date": as_of_date,
                "query_active": query_active,
                "provider_active": provider_active,
                "ticker": ticker,
                "name": _normalized_optional_text(item.get("name")),
                "market": _normalized_optional_text(item.get("market")),
                "locale": _normalized_optional_text(item.get("locale")),
                "security_type": _normalized_optional_text(item.get("type")),
                "exchange_mic": exchange_mic,
                "currency_name": _normalized_optional_text(item.get("currency_name")),
                "share_class_figi": share_class_figi,
                "composite_figi": composite_figi,
                "provider_security_id": share_class_figi or composite_figi,
                "provider_security_id_kind": (
                    "share_class_figi"
                    if share_class_figi
                    else "composite_figi"
                    if composite_figi
                    else None
                ),
                "delisted_utc": _normalized_optional_text(item.get("delisted_utc")),
                "last_updated_utc": _normalized_optional_text(item.get("last_updated_utc")),
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    next_url = document.get("next_url")
    return (
        observations,
        rejections,
        _safe_massive_url(next_url) if isinstance(next_url, str) else None,
    )


def build_massive_reference_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/massive_reference"),
    output_root: Path = Path("data/processed/massive_reference"),
) -> MassiveReferenceAuditResult:
    """Build point-in-time reference coverage without opening price gates."""
    plan_report = json.loads((plan_root / "target_report.json").read_text("utf-8"))
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    targets = load_massive_reference_targets(plan_root)
    snapshot_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    completion_by_target: dict[str, dict[str, Any]] = {}
    for row in targets.itertuples(index=False):
        target_id = str(row.target_id)
        completion = _load_target_completion(raw_root, target_id)
        if completion is None:
            continue
        completion_by_target[target_id] = completion
        for query in completion["queries"].values():
            query_active = bool(query["query_active"])
            for snapshot_manifest in query["snapshots"]:
                snapshot = _load_snapshot(Path(snapshot_manifest))
                snapshot_rows.append(_snapshot_row(snapshot))
                observations, rejections, _ = parse_massive_ticker_response(
                    Path(snapshot.path).read_bytes(),
                    expected_cik=str(row.cik),
                    target_id=target_id,
                    query_active=query_active,
                    as_of_date=str(row.query_date),
                    source_sha256=snapshot.sha256,
                )
                observation_rows.extend(observations)
                rejection_rows.extend(rejections)
    observations_frame = pd.DataFrame(observation_rows, columns=_observation_columns())
    rejections_frame = pd.DataFrame(
        rejection_rows,
        columns=["target_id", "source_occurrence", "reason", "source_sha256"],
    )
    audit = _reference_coverage_audit(
        targets,
        observations_frame,
        completion_by_target=completion_by_target,
    )
    snapshots_frame = pd.DataFrame(snapshot_rows, columns=_snapshot_columns())
    snapshot_payload = [
        {"target_id": row["target_id"], "query_id": row["query_id"], "sha256": row["sha256"]}
        for row in sorted(snapshot_rows, key=lambda value: (value["target_id"], value["query_id"]))
    ]
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "parser_version": MASSIVE_PARSER_VERSION,
            "snapshots": snapshot_payload,
            "automatic_ticker_events_join": False,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "massive-reference-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads((target / "massive_reference_report.json").read_text("utf-8"))
        return MassiveReferenceAuditResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete Massive reference audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_reference_targets.parquet": targets,
        "massive_reference_snapshots.parquet": snapshots_frame,
        "massive_reference_observations.parquet": observations_frame,
        "massive_reference_rejections.parquet": rejections_frame,
        "sample_massive_reference_audit.parquet": audit,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "scope_version": plan_report["scope_version"],
        "scope_manifest_sha256": plan_report["scope_manifest_sha256"],
        "parser_version": MASSIVE_PARSER_VERSION,
        "price_mapping_targets": len(targets),
        "targets_collected": len(completion_by_target),
        "raw_snapshots": len(snapshots_frame),
        "parsed_reference_observations": len(observations_frame),
        "active_reference_matches": int(audit["active_reference_observed"].sum()),
        "inactive_reference_matches": int(audit["inactive_reference_observed"].sum()),
        "event_date_reference_candidates": int(audit["event_date_reference_candidate"].sum()),
        "historical_symbol_interval_ready": 0,
        "final_price_identity_ready": 0,
        "provider_collection_gate": bool(audit["provider_query_complete"].all()),
        "event_date_reference_gate": bool(audit["event_date_reference_candidate"].all()),
        "historical_symbol_interval_gate": False,
        "daily_price_gate": False,
        "corporate_action_gate": False,
        "delisting_outcome_gate": False,
        "exchange_calendar_gate": False,
        "return_study_gate": False,
        "ticker_events_automatic_join_used": False,
        "current_ticker_join_used": False,
        "unresolved_units_replaced": False,
        "artifacts": artifacts,
    }
    (staging / "massive_reference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(staging / "massive_reference_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveReferenceAuditResult(target.resolve(), report)


def _collect_reference_query(
    client: MassiveReferenceClient,
    *,
    raw_root: Path,
    target_id: str,
    cik: str,
    as_of_date: str,
    active: bool,
) -> tuple[dict[str, Any], int, int]:
    existing = _load_query_completion(raw_root, target_id, active)
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = client.ticker_reference_url(cik, as_of_date, active=active)
    snapshots: list[str] = []
    record_count = 0
    requests_made = 0
    reused = 0
    page_number = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError("Massive pagination exceeded 20 pages", reason="pagination")
        query_id = _stable_hash(
            {
                "target_id": target_id,
                "active": active,
                "as_of_date": as_of_date,
                "requested_url": url,
            }
        )
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(
                raw_root,
                target_id=target_id,
                query_id=query_id,
                query_active=active,
                as_of_date=as_of_date,
                page_number=page_number,
                response=response,
            )
            requests_made += response.requests_made
        else:
            reused += 1
        observations, _, next_url = parse_massive_ticker_response(
            Path(snapshot.path).read_bytes(),
            expected_cik=cik,
            target_id=target_id,
            query_active=active,
            as_of_date=as_of_date,
            source_sha256=snapshot.sha256,
        )
        record_count += len(observations)
        snapshots.append(str(_snapshot_manifest_path(raw_root, snapshot)))
        url = next_url
    completion = {
        "query_active": active,
        "as_of_date": as_of_date,
        "record_count": record_count,
        "page_count": len(snapshots),
        "pagination_complete": True,
        "snapshots": snapshots,
    }
    _write_query_completion(raw_root, target_id, active, completion)
    return completion, requests_made, reused


def _collect_symbol_reference_query(
    client: MassiveReferenceClient,
    *,
    raw_root: Path,
    target_id: str,
    cik: str,
    ticker: str,
    as_of_date: str,
    active: bool,
) -> tuple[dict[str, Any], int, int]:
    existing = _load_query_completion(raw_root, target_id, active)
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = client.ticker_symbol_reference_url(ticker, as_of_date, active=active)
    snapshots: list[str] = []
    record_count = 0
    requests_made = 0
    reused = 0
    page_number = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError("Massive pagination exceeded 20 pages", reason="pagination")
        query_id = _stable_hash(
            {
                "target_id": target_id,
                "ticker": ticker,
                "active": active,
                "as_of_date": as_of_date,
                "requested_url": url,
            }
        )
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(
                raw_root,
                target_id=target_id,
                query_id=query_id,
                query_active=active,
                as_of_date=as_of_date,
                page_number=page_number,
                response=response,
            )
            requests_made += response.requests_made
        else:
            reused += 1
        observations, _, next_url = parse_massive_ticker_response(
            Path(snapshot.path).read_bytes(),
            expected_cik=cik,
            target_id=target_id,
            query_active=active,
            as_of_date=as_of_date,
            source_sha256=snapshot.sha256,
        )
        record_count += len(observations)
        snapshots.append(str(_snapshot_manifest_path(raw_root, snapshot)))
        url = next_url
    completion = {
        "query_active": active,
        "query_ticker": ticker,
        "as_of_date": as_of_date,
        "record_count": record_count,
        "page_count": len(snapshots),
        "pagination_complete": True,
        "snapshots": snapshots,
    }
    _write_query_completion(raw_root, target_id, active, completion)
    return completion, requests_made, reused


def _preserve_response(
    raw_root: Path,
    *,
    target_id: str,
    query_id: str,
    query_active: bool,
    as_of_date: str,
    page_number: int,
    response: MassiveResponse,
) -> MassiveSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Massive raw snapshot hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Massive response changed before preservation")
        temporary.replace(path)
    snapshot = MassiveSnapshot(
        target_id=target_id,
        query_id=query_id,
        query_active=query_active,
        as_of_date=as_of_date,
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
        manifest_path.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True), "utf-8")
    return snapshot


def _existing_snapshot(raw_root: Path, query_id: str, requested_url: str) -> MassiveSnapshot | None:
    directory = raw_root / "snapshots" / query_id
    for path in sorted(directory.glob("*.json")):
        snapshot = _load_snapshot(path)
        if snapshot.requested_url == requested_url:
            return snapshot
    return None


def _load_snapshot(path: Path) -> MassiveSnapshot:
    document = json.loads(path.read_text("utf-8"))
    snapshot = MassiveSnapshot(**document)
    raw_path = Path(snapshot.path)
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Massive snapshot is missing or invalid: {raw_path}")
    return snapshot


def _snapshot_manifest_path(raw_root: Path, snapshot: MassiveSnapshot) -> Path:
    return raw_root / "snapshots" / snapshot.query_id / f"{snapshot.sha256}.json"


def _write_query_completion(
    raw_root: Path, target_id: str, active: bool, completion: dict[str, Any]
) -> None:
    path = raw_root / "queries" / target_id / f"active-{str(active).lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(completion, indent=2, sort_keys=True), "utf-8")


def _load_query_completion(raw_root: Path, target_id: str, active: bool) -> dict[str, Any] | None:
    path = raw_root / "queries" / target_id / f"active-{str(active).lower()}.json"
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if not document.get("pagination_complete"):
        return None
    for snapshot_path in document.get("snapshots", []):
        _load_snapshot(Path(snapshot_path))
    return document


def _write_target_completion(
    raw_root: Path,
    *,
    target_id: str,
    cik: str,
    as_of_date: str,
    status: str,
    queries: dict[str, Any],
) -> None:
    path = raw_root / "completed" / f"{target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "target_id": target_id,
                "cik": normalize_cik(cik),
                "as_of_date": as_of_date,
                "status": status,
                "queries": queries,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        "utf-8",
    )


def _load_target_completion(raw_root: Path, target_id: str) -> dict[str, Any] | None:
    path = raw_root / "completed" / f"{target_id}.json"
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    for query in document.get("queries", {}).values():
        if not query.get("pagination_complete"):
            return None
        for snapshot_path in query.get("snapshots", []):
            _load_snapshot(Path(snapshot_path))
    return document


def _valid_target_completion(raw_root: Path, target_id: str) -> bool:
    try:
        return _load_target_completion(raw_root, target_id) is not None
    except (OSError, TypeError, json.JSONDecodeError, FilingCatalogError):
        return False


def _write_failure(raw_root: Path, target_id: str, reason: str, detail: str) -> None:
    directory = raw_root / "failures" / target_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "target_id": target_id,
                "reason": reason,
                "detail": detail,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        "utf-8",
    )


def _reference_coverage_audit(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    completion_by_target: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    audit = targets.copy()
    audit["provider_query_complete"] = audit["target_id"].isin(completion_by_target)
    audit["provider_query_status"] = audit["target_id"].map(
        lambda value: completion_by_target.get(str(value), {}).get("status", "not_collected")
    )
    audit["active_reference_observed"] = False
    audit["inactive_reference_observed"] = False
    audit["reference_candidate_count"] = 0
    audit["event_date_reference_status"] = "not_collected"
    audit["selected_ticker"] = pd.NA
    audit["selected_exchange_mic"] = pd.NA
    audit["selected_provider_security_id"] = pd.NA
    audit["selected_provider_security_id_kind"] = pd.NA
    audit["event_date_reference_candidate"] = False
    audit["historical_symbol_interval_ready"] = False
    audit["final_price_identity_ready"] = False
    if observations.empty:
        audit.loc[
            audit["provider_query_complete"].astype(bool),
            "event_date_reference_status",
        ] = "provider_no_cik_match"
        return audit
    for index, target in audit.iterrows():
        group = observations.loc[observations["target_id"].eq(target["target_id"])].copy()
        active = group.loc[group["query_active"].astype(bool)]
        inactive = group.loc[~group["query_active"].astype(bool)]
        audit.at[index, "active_reference_observed"] = not active.empty
        audit.at[index, "inactive_reference_observed"] = not inactive.empty
        if active.empty:
            audit.at[index, "event_date_reference_status"] = (
                "only_inactive_security_at_event_date"
                if not inactive.empty
                else "provider_no_cik_match"
            )
            continue
        common = active.loc[active["security_type"].eq("CS")].copy()
        if common.empty:
            audit.at[index, "event_date_reference_status"] = "no_common_stock_type"
            continue
        sec_tickers = {
            ticker.strip().upper()
            for ticker in str(target.get("equity_trading_symbols") or "").split("|")
            if ticker.strip()
        }
        if sec_tickers:
            exact = common.loc[common["ticker"].isin(sec_tickers)]
            if exact.empty:
                audit.at[index, "event_date_reference_status"] = "sec_ticker_conflict"
                audit.at[index, "reference_candidate_count"] = len(common)
                continue
            common = exact
        audit.at[index, "reference_candidate_count"] = len(common)
        if len(common) != 1:
            audit.at[index, "event_date_reference_status"] = "ambiguous_common_stock"
            continue
        selected = common.iloc[0]
        provider_security_id = selected["provider_security_id"]
        if pd.isna(provider_security_id) or not str(provider_security_id).strip():
            audit.at[index, "event_date_reference_status"] = "missing_provider_security_id"
            continue
        exchange_mic = selected["exchange_mic"]
        if pd.isna(exchange_mic) or not str(exchange_mic).strip():
            audit.at[index, "event_date_reference_status"] = "missing_primary_exchange"
            continue
        audit.at[index, "selected_ticker"] = selected["ticker"]
        audit.at[index, "selected_exchange_mic"] = selected["exchange_mic"]
        audit.at[index, "selected_provider_security_id"] = selected["provider_security_id"]
        audit.at[index, "selected_provider_security_id_kind"] = selected[
            "provider_security_id_kind"
        ]
        audit.at[index, "event_date_reference_status"] = "unique_reference_candidate"
        audit.at[index, "event_date_reference_candidate"] = True
    return audit


def _reference_rejection(
    target_id: str, source_occurrence: int, reason: str, source_sha256: str
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "source_occurrence": source_occurrence,
        "reason": reason,
        "source_sha256": source_sha256,
    }


def _snapshot_row(snapshot: MassiveSnapshot) -> dict[str, Any]:
    row = asdict(snapshot)
    row["headers_json"] = json.dumps(row.pop("headers"), sort_keys=True)
    return row


def _snapshot_columns() -> list[str]:
    return [
        "target_id",
        "query_id",
        "query_active",
        "as_of_date",
        "page_number",
        "requested_url",
        "response_url",
        "sha256",
        "fetched_at",
        "path",
        "bytes",
        "status_code",
        "headers_json",
    ]


def _observation_columns() -> list[str]:
    return [
        "target_id",
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


def _normalized_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized or None


def _normalized_ticker(value: Any) -> str | None:
    normalized = (_normalized_optional_text(value) or "").upper()
    return normalized if _TICKER.fullmatch(normalized) else None


def _normalized_figi(value: Any) -> str | None:
    normalized = (_normalized_optional_text(value) or "").upper()
    return normalized if _FIGI.fullmatch(normalized) else None


def _normalized_mic(value: Any) -> str | None:
    normalized = (_normalized_optional_text(value) or "").upper()
    return normalized if _MIC.fullmatch(normalized) else None


def _safe_massive_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
        raise MassiveConfigurationError(f"Unsafe Massive URL: {value}")
    safe_query = urlencode(
        [(key, item) for key, item in parse_qsl(parsed.query) if key.casefold() != "apikey"]
    )
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", safe_query, ""))


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveConfigurationError(f"Invalid ISO date: {value}") from exc


def _backoff_seconds(attempt: int) -> float:
    return float(min(1.0 * (2**attempt), 30.0))


def _retry_delay(response: httpx.Response, attempt: int, now: datetime) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return _backoff_seconds(attempt)
    try:
        return max(0.0, min(float(retry_after), 120.0))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return _backoff_seconds(attempt)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return float(max(0.0, min((retry_at - now).total_seconds(), 120.0)))


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
