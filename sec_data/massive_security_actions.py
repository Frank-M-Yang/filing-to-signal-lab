"""Auditable security-continuity and corporate-action ingestion from Massive."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo

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

MASSIVE_TICKER_EVENTS_PREFIX: Final[str] = "https://api.massive.com/vX/reference/tickers"
MASSIVE_SPLITS_URL: Final[str] = "https://api.massive.com/stocks/v1/splits"
MASSIVE_DIVIDENDS_URL: Final[str] = "https://api.massive.com/stocks/v1/dividends"
MASSIVE_SECURITY_ACTIONS_PARSER_VERSION: Final[str] = (
    "massive-security-continuity-actions-parser-v2"
)
_QUERY_KINDS: Final[tuple[str, ...]] = (
    "reference_start",
    "reference_end",
    "ticker_events",
    "splits",
    "dividends",
)
_NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MassiveSecurityActionsConfig:
    """Explicit evidence requirements for one filing-event price window."""

    require_start_reference: bool = True
    require_end_reference: bool = True
    require_ticker_events: bool = True
    require_splits: bool = True
    require_dividends: bool = True
    version: str = "massive-security-continuity-actions-v1"

    def validate(self) -> None:
        """Reject a plan that silently omits one of the required evidence classes."""
        requirements = (
            self.require_start_reference,
            self.require_end_reference,
            self.require_ticker_events,
            self.require_splits,
            self.require_dividends,
        )
        if not all(requirements):
            raise MassiveConfigurationError("All security-continuity evidence is required")


@dataclass(frozen=True)
class MassiveSecurityActionsPlanResult:
    """Immutable set of security-continuity and corporate-action queries."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveSecurityActionsCollectionResult:
    """Counts and stopping reason from one resumable collection batch."""

    selected_targets: int
    targets_completed: int
    targets_failed: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class MassiveSecurityActionsAuditResult:
    """Immutable security-continuity and corporate-action audit."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MassiveSecurityActionsSnapshot:
    """Content-addressed raw response for one evidence query page."""

    action_target_id: str
    query_id: str
    query_kind: str
    page_number: int
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    headers: dict[str, str]


def build_massive_security_actions_target_plan(
    reference_root: Path,
    price_root: Path,
    calendar_root: Path,
    *,
    cohort_root: Path | None = None,
    output_root: Path = Path("data/market/massive_security_actions/targets"),
    config: MassiveSecurityActionsConfig | None = None,
) -> MassiveSecurityActionsPlanResult:
    """Freeze five evidence queries for every calendar-aligned price target."""
    cfg = config or MassiveSecurityActionsConfig()
    cfg.validate()
    reference_manifest_sha256 = _sha256_file(reference_root / "output_manifest.json")
    price_manifest_sha256 = _sha256_file(price_root / "output_manifest.json")
    calendar_manifest_sha256 = _sha256_file(calendar_root / "output_manifest.json")
    price_report = _read_json(price_root / "massive_price_report.json")
    calendar_report = _read_json(calendar_root / "exchange_calendar_report.json")
    if price_report.get("reference_manifest_sha256") != reference_manifest_sha256:
        raise FilingCatalogError("Price audit does not tie to the selected reference audit")
    if calendar_report.get("price_manifest_sha256") != price_manifest_sha256:
        raise FilingCatalogError("Calendar audit does not tie to the selected price audit")

    targets = load_catalog_table(price_root, "massive_daily_price_targets.parquet")
    cohort_manifest_sha256: str | None = None
    cohort_report: dict[str, Any] | None = None
    if cohort_root is not None:
        cohort_manifest_sha256 = _sha256_file(cohort_root / "output_manifest.json")
        cohort_report = _read_json(cohort_root / "exploratory_cohort_report.json")
        if cohort_report.get("reference_manifest_sha256") != reference_manifest_sha256:
            raise FilingCatalogError("Exploratory cohort does not tie to the reference audit")
        if cohort_report.get("price_manifest_sha256") != price_manifest_sha256:
            raise FilingCatalogError("Exploratory cohort does not tie to the price audit")
        if cohort_report.get("calendar_manifest_sha256") != calendar_manifest_sha256:
            raise FilingCatalogError("Exploratory cohort does not tie to the calendar audit")
        if not cohort_report.get("exploratory_only"):
            raise FilingCatalogError("Cohort is not labelled exploratory-only")
        if not cohort_report.get("downstream_evidence_collection_gate"):
            raise FilingCatalogError("Exploratory cohort is not ready for evidence collection")
        cohort = load_catalog_table(cohort_root, "selected_exploratory_events.parquet")
        if cohort["price_target_id"].duplicated().any():
            raise FilingCatalogError("Exploratory cohort contains duplicate price targets")
        targets = targets.merge(
            cohort[["price_target_id"]],
            on="price_target_id",
            how="inner",
            validate="one_to_one",
        )
        if len(targets) != int(cohort_report.get("selected_events", -1)):
            raise FilingCatalogError("Exploratory cohort targets do not match the price audit")
    alignment = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
    reference = load_catalog_table(reference_root, "massive_reference_observations.parquet")
    selected = _selected_reference_evidence(targets, reference)
    targets = targets.merge(selected, on="target_id", how="left", validate="one_to_one")
    targets = targets.merge(
        alignment[
            [
                "price_target_id",
                "formal_entry_session_date",
                "formal_exit_session_date",
                "formal_event_window_ready",
            ]
        ],
        on="price_target_id",
        how="left",
        validate="one_to_one",
    )
    required = [
        "selected_composite_figi",
        "selected_share_class_figi",
        "selected_reference_source_sha256",
    ]
    if targets[required].isna().any().any():
        raise FilingCatalogError("A security-actions target lacks FIGI reference evidence")
    if not targets["formal_event_window_ready"].fillna(False).astype(bool).all():
        raise FilingCatalogError("Security-actions targets require formal calendar windows")
    targets["action_target_id"] = targets.apply(
        lambda row: _stable_hash(
            {
                "price_target_id": str(row["price_target_id"]),
                "share_class_figi": str(row["selected_share_class_figi"]),
                "composite_figi": str(row["selected_composite_figi"]),
                "window_start": str(row["price_window_start"]),
                "window_end": str(row["price_window_end"]),
                "config_version": cfg.version,
            }
        ),
        axis=1,
    )
    targets = targets.sort_values("price_acquisition_order").reset_index(drop=True)
    targets["action_acquisition_order"] = np.arange(1, len(targets) + 1)
    columns = [
        "action_target_id",
        "action_acquisition_order",
        "price_target_id",
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
        "selected_share_class_figi",
        "selected_composite_figi",
        "selected_reference_source_sha256",
        "currency_name",
        "price_window_start",
        "price_window_end",
        "formal_entry_session_date",
        "formal_exit_session_date",
    ]
    targets = targets[columns].copy()
    payload = json.dumps(
        {
            "reference_manifest_sha256": reference_manifest_sha256,
            "price_manifest_sha256": price_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "config": asdict(cfg),
            "targets": targets.to_dict(orient="records"),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        "massive-security-actions-targets-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MassiveSecurityActionsPlanResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete security-actions target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / "massive_security_actions_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": _read_json(reference_root / "massive_reference_report.json")[
            "dataset_version"
        ],
        "reference_manifest_sha256": reference_manifest_sha256,
        "price_version": price_report["dataset_version"],
        "price_manifest_sha256": price_manifest_sha256,
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "exploratory_cohort_version": (
            cohort_report["dataset_version"] if cohort_report is not None else None
        ),
        "exploratory_cohort_manifest_sha256": cohort_manifest_sha256,
        "exploratory_only": cohort_report is not None,
        "config": asdict(cfg),
        "targets": len(targets),
        "queries_per_target": len(_QUERY_KINDS),
        "planned_queries": len(targets) * len(_QUERY_KINDS),
        "ticker_events_endpoint_experimental": True,
        "current_ticker_join_used": False,
        "selection_uses_action_coverage": False,
        "artifacts": {target_path.name: artifact},
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "reference_manifest_sha256": reference_manifest_sha256,
                "price_manifest_sha256": price_manifest_sha256,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "artifacts": {target_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveSecurityActionsPlanResult(target.resolve(), report)


def load_massive_security_actions_targets(plan_root: Path) -> pd.DataFrame:
    """Load and minimally validate an immutable security-actions target plan."""
    targets = load_catalog_table(plan_root, "massive_security_actions_targets.parquet")
    if targets.empty or targets["action_target_id"].duplicated().any():
        raise FilingCatalogError("Security-actions targets are empty or duplicated")
    for column in ("selected_share_class_figi", "selected_composite_figi"):
        if targets[column].astype(str).str.strip().eq("").any():
            raise FilingCatalogError(f"Security-actions target has an empty {column}")
    return targets


def massive_ticker_events_url(composite_figi: str) -> str:
    """Build a secret-free ticker-change timeline URL using Composite FIGI."""
    identifier = _required_text(composite_figi, "composite FIGI")
    return f"{MASSIVE_TICKER_EVENTS_PREFIX}/{quote(identifier, safe='')}/events?types=ticker_change"


def massive_splits_url(ticker: str, window_start: str, window_end: str) -> str:
    """Build a filtered, secret-free split-history URL."""
    return _action_url(MASSIVE_SPLITS_URL, ticker, "execution_date", window_start, window_end)


def massive_dividends_url(ticker: str, window_start: str, window_end: str) -> str:
    """Build a filtered, secret-free dividend-history URL."""
    return _action_url(MASSIVE_DIVIDENDS_URL, ticker, "ex_dividend_date", window_start, window_end)


def collect_massive_security_actions_artifacts(
    targets: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_security_actions"),
    limit: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveSecurityActionsCollectionResult:
    """Collect a resumable batch, preserving partial query success by evidence class."""
    if limit < 1:
        raise MassiveConfigurationError("Security-actions collection limit must be positive")
    ordered = targets.sort_values("action_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_target_completion(raw_root, str(row.action_target_id))
    ][:limit]
    completed = failed = requests_made = reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        queries: dict[str, dict[str, Any]] = {}
        target_failed = False
        for query_kind in _QUERY_KINDS:
            try:
                query, requests, reused_pages = _collect_query(client, raw_root, row, query_kind)
                queries[query_kind] = query
                requests_made += requests
                reused += reused_pages
            except MassiveRequestError as exc:
                target_failed = True
                requests_made += exc.requests_made
                _write_failure(
                    raw_root,
                    str(row.action_target_id),
                    exc.reason,
                    str(exc),
                    query_kind=query_kind,
                )
                if exc.reason in {"authentication_failed", "entitlement_denied"}:
                    blocking_reason = exc.reason
                    break
        if blocking_reason is not None:
            failed += 1
            break
        if target_failed:
            failed += 1
        else:
            _write_target_completion(raw_root, row, queries)
            completed += 1
        if progress is not None:
            progress(number, len(pending), str(row.action_target_id))
    return MassiveSecurityActionsCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def collect_massive_security_session_fallbacks(
    targets: pd.DataFrame,
    sessions: pd.DataFrame,
    client: MassiveReferenceClient,
    *,
    raw_root: Path = Path("data/market/massive_security_actions"),
    limit: int = 2,
    progress: Callable[[int, int, str], None] | None = None,
) -> MassiveSecurityActionsCollectionResult:
    """Validate every exchange session when the experimental events endpoint is absent."""
    if limit < 1:
        raise MassiveConfigurationError("Session-fallback collection limit must be positive")
    ordered = targets.sort_values("action_acquisition_order")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if _ticker_events_not_found(raw_root, str(row.action_target_id))
        and _load_session_fallback_completion(raw_root, row, sessions) is None
    ][:limit]
    completed = failed = requests_made = reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        session_dates = _target_session_dates(row, sessions)
        query_kinds: list[str] = []
        target_failed = False
        for session_date in session_dates:
            query_kind = f"reference_session_{session_date}"
            try:
                _, requests, reused_pages = _collect_query(client, raw_root, row, query_kind)
                query_kinds.append(query_kind)
                requests_made += requests
                reused += reused_pages
            except MassiveRequestError as exc:
                target_failed = True
                requests_made += exc.requests_made
                _write_failure(
                    raw_root,
                    str(row.action_target_id),
                    exc.reason,
                    str(exc),
                    query_kind=query_kind,
                )
                if exc.reason in {"authentication_failed", "entitlement_denied"}:
                    blocking_reason = exc.reason
                    break
        if blocking_reason is not None:
            failed += 1
            break
        if target_failed:
            failed += 1
        else:
            _write_session_fallback_completion(raw_root, row, session_dates, query_kinds)
            completed += 1
        if progress is not None:
            progress(number, len(pending), str(row.action_target_id))
    return MassiveSecurityActionsCollectionResult(
        selected_targets=len(pending),
        targets_completed=completed,
        targets_failed=failed,
        requests_made=requests_made,
        snapshots_reused=reused,
        blocking_reason=blocking_reason,
    )


def parse_massive_ticker_events(
    content: bytes,
    *,
    action_target_id: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse an experimental ticker-change timeline without assuming completeness."""
    document = _response_document(content, "ticker events")
    results = document.get("results", {})
    if not isinstance(results, dict):
        raise MassiveRequestError("Ticker-events results must be an object", reason="bad_schema")
    events = results.get("events", [])
    if not isinstance(events, list):
        raise MassiveRequestError("Ticker-events events must be a list", reason="bad_schema")
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for occurrence, item in enumerate(events):
        reason: str | None = None
        event_date: str | None = None
        new_ticker: str | None = None
        if not isinstance(item, dict):
            reason = "ticker_event_not_object"
        elif item.get("type") != "ticker_change":
            reason = "unsupported_ticker_event_type"
        else:
            event_date = _valid_date_text(item.get("date"))
            detail = item.get("ticker_change")
            new_ticker = _optional_text(detail.get("ticker")) if isinstance(detail, dict) else None
            if event_date is None:
                reason = "invalid_ticker_event_date"
            elif new_ticker is None:
                reason = "missing_ticker_change_symbol"
        if reason is not None:
            rejections.append(
                _rejection(action_target_id, "ticker_events", occurrence, reason, source_sha256)
            )
            continue
        rows.append(
            {
                "action_target_id": action_target_id,
                "event_date": event_date,
                "event_type": "ticker_change",
                "new_ticker": new_ticker,
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    return rows, rejections, None


def parse_massive_splits(
    content: bytes,
    *,
    action_target_id: str,
    expected_ticker: str,
    window_start: str,
    window_end: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse one filtered split page and reject out-of-window or malformed rows."""
    document = _response_document(content, "splits")
    results = _result_list(document, "splits")
    start, end = _parse_date(window_start), _parse_date(window_end)
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for occurrence, item in enumerate(results):
        reason = _split_rejection_reason(item, expected_ticker, start, end)
        if reason is not None:
            rejections.append(
                _rejection(action_target_id, "splits", occurrence, reason, source_sha256)
            )
            continue
        assert isinstance(item, dict)
        rows.append(
            {
                "action_target_id": action_target_id,
                "provider_event_id": _optional_text(item.get("id")),
                "ticker": expected_ticker,
                "execution_date": str(item["execution_date"]),
                "adjustment_type": str(item["adjustment_type"]),
                "split_from": float(item["split_from"]),
                "split_to": float(item["split_to"]),
                "historical_adjustment_factor": _optional_positive_float(
                    item.get("historical_adjustment_factor")
                ),
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    return rows, rejections, _safe_optional_next_url(document.get("next_url"))


def parse_massive_dividends(
    content: bytes,
    *,
    action_target_id: str,
    expected_ticker: str,
    expected_currency: str,
    window_start: str,
    window_end: str,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse one filtered cash-dividend page without imputing missing amounts."""
    document = _response_document(content, "dividends")
    results = _result_list(document, "dividends")
    start, end = _parse_date(window_start), _parse_date(window_end)
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for occurrence, item in enumerate(results):
        reason = _dividend_rejection_reason(item, expected_ticker, expected_currency, start, end)
        if reason is not None:
            rejections.append(
                _rejection(action_target_id, "dividends", occurrence, reason, source_sha256)
            )
            continue
        assert isinstance(item, dict)
        rows.append(
            {
                "action_target_id": action_target_id,
                "provider_event_id": _optional_text(item.get("id")),
                "ticker": expected_ticker,
                "ex_dividend_date": str(item["ex_dividend_date"]),
                "cash_amount": float(item["cash_amount"]),
                "currency": str(item["currency"]).upper(),
                "declaration_date": _valid_date_text(item.get("declaration_date")),
                "record_date": _valid_date_text(item.get("record_date")),
                "pay_date": _valid_date_text(item.get("pay_date")),
                "distribution_type": _optional_text(item.get("distribution_type")),
                "frequency": _optional_nonnegative_int(item.get("frequency")),
                "split_adjusted_cash_amount": _optional_positive_float(
                    item.get("split_adjusted_cash_amount")
                ),
                "historical_adjustment_factor": _optional_positive_float(
                    item.get("historical_adjustment_factor")
                ),
                "source_occurrence": occurrence,
                "source_sha256": source_sha256,
            }
        )
    return rows, rejections, _safe_optional_next_url(document.get("next_url"))


def build_massive_security_actions_audit(
    plan_root: Path,
    calendar_root: Path,
    *,
    raw_root: Path = Path("data/market/massive_security_actions"),
    output_root: Path = Path("data/processed/massive_security_actions"),
) -> MassiveSecurityActionsAuditResult:
    """Build interval and action readiness assets while keeping return gates closed."""
    plan_report = _read_json(plan_root / "target_report.json")
    plan_manifest_sha256 = _sha256_file(plan_root / "output_manifest.json")
    calendar_manifest_sha256 = _sha256_file(calendar_root / "output_manifest.json")
    if plan_report.get("calendar_manifest_sha256") != calendar_manifest_sha256:
        raise FilingCatalogError("Security-actions plan does not tie to the calendar audit")
    targets = load_massive_security_actions_targets(plan_root)
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    identity_rows: list[dict[str, Any]] = []
    ticker_event_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    dividend_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    completions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in targets.itertuples(index=False):
        action_target_id = str(row.action_target_id)
        query_kinds = list(_QUERY_KINDS)
        fallback = _load_session_fallback_completion(raw_root, row, sessions)
        if fallback is not None:
            query_kinds.extend(str(value) for value in fallback["query_kinds"])
        for query_kind in query_kinds:
            completion = _load_query_completion(raw_root, action_target_id, query_kind)
            if completion is None:
                continue
            completions[(action_target_id, query_kind)] = completion
            for snapshot_path in completion["snapshots"]:
                snapshot = _load_snapshot(Path(snapshot_path))
                snapshot_rows.append(_snapshot_row(snapshot))
                parsed, rejected, _ = _parse_snapshot(snapshot, row)
                if query_kind in {"reference_start", "reference_end"} or query_kind.startswith(
                    "reference_session_"
                ):
                    for item in parsed:
                        item["query_kind"] = query_kind
                    identity_rows.extend(parsed)
                elif query_kind == "ticker_events":
                    ticker_event_rows.extend(parsed)
                elif query_kind == "splits":
                    split_rows.extend(parsed)
                else:
                    dividend_rows.extend(parsed)
                rejection_rows.extend(rejected)
    identity = pd.DataFrame(identity_rows, columns=_identity_columns())
    ticker_events = pd.DataFrame(ticker_event_rows, columns=_ticker_event_columns())
    splits = pd.DataFrame(split_rows, columns=_split_columns())
    dividends = pd.DataFrame(dividend_rows, columns=_dividend_columns())
    rejections = pd.DataFrame(rejection_rows, columns=_rejection_columns())
    snapshots = pd.DataFrame(snapshot_rows, columns=_snapshot_columns())
    audit, mappings, actions = _build_readiness_tables(
        targets, sessions, identity, ticker_events, splits, dividends, rejections, completions
    )
    source_hashes = sorted(snapshots.get("sha256", pd.Series(dtype=str)).astype(str).tolist())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
            "source_hashes": source_hashes,
            "parser_version": MASSIVE_SECURITY_ACTIONS_PARSER_VERSION,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        "massive-security-actions-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return MassiveSecurityActionsAuditResult(
            target.resolve(), _read_json(target / "massive_security_actions_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete security-actions audit directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "massive_security_actions_targets.parquet": targets,
        "massive_security_actions_snapshots.parquet": snapshots,
        "security_reference_interval_observations.parquet": identity,
        "ticker_change_observations.parquet": ticker_events,
        "split_observations.parquet": splits,
        "dividend_observations.parquet": dividends,
        "security_actions_rejections.parquet": rejections,
        "sample_security_actions_audit.parquet": audit,
        "validated_security_mappings.parquet": mappings,
        "normalized_corporate_actions.parquet": actions,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    identity_ready = int(audit["historical_symbol_interval_ready"].astype(bool).sum())
    actions_ready = int(audit["corporate_actions_ready"].astype(bool).sum())
    collected = int(audit["all_security_action_queries_complete"].astype(bool).sum())
    fallback_ready = int(audit["reference_session_grid_complete"].astype(bool).sum())
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest_sha256,
        "reference_version": plan_report["reference_version"],
        "reference_manifest_sha256": plan_report["reference_manifest_sha256"],
        "price_version": plan_report["price_version"],
        "price_manifest_sha256": plan_report["price_manifest_sha256"],
        "calendar_version": plan_report["calendar_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "parser_version": MASSIVE_SECURITY_ACTIONS_PARSER_VERSION,
        "targets": len(targets),
        "targets_fully_collected": collected,
        "session_grid_fallbacks_ready": fallback_ready,
        "targets_with_complete_required_evidence": int(
            (
                audit["historical_symbol_interval_ready"].astype(bool)
                & audit["corporate_actions_ready"].astype(bool)
            ).sum()
        ),
        "raw_snapshots": len(snapshots),
        "historical_symbol_intervals_ready": identity_ready,
        "corporate_action_windows_ready": actions_ready,
        "ticker_changes_inside_windows": int(audit["ticker_changes_inside_window"].sum()),
        "splits_observed": len(splits),
        "dividends_observed": len(dividends),
        "rejected_rows": len(rejections),
        "pilot_identity_gate": identity_ready == len(targets),
        "pilot_corporate_actions_gate": actions_ready == len(targets),
        "ticker_events_endpoint_experimental": True,
        "delisting_outcome_gate": False,
        "return_study_gate": False,
        "returns_computed": False,
        "current_ticker_join_used": False,
        "selection_uses_action_coverage": False,
        "artifacts": artifacts,
    }
    report_path = staging / "massive_security_actions_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "target_plan_manifest_sha256": plan_manifest_sha256,
                "reference_manifest_sha256": plan_report["reference_manifest_sha256"],
                "price_manifest_sha256": plan_report["price_manifest_sha256"],
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return MassiveSecurityActionsAuditResult(target.resolve(), report)


def _selected_reference_evidence(targets: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    selected = targets[["target_id", "selected_provider_security_id"]].merge(
        observations[
            [
                "target_id",
                "provider_security_id",
                "share_class_figi",
                "composite_figi",
                "source_sha256",
            ]
        ],
        left_on=["target_id", "selected_provider_security_id"],
        right_on=["target_id", "provider_security_id"],
        how="left",
        validate="one_to_many",
    )
    selected = selected.drop_duplicates(["target_id", "provider_security_id", "source_sha256"])
    if selected["target_id"].duplicated().any():
        raise FilingCatalogError("Selected reference evidence is not unique per target")
    return selected.rename(
        columns={
            "share_class_figi": "selected_share_class_figi",
            "composite_figi": "selected_composite_figi",
            "source_sha256": "selected_reference_source_sha256",
        }
    )[
        [
            "target_id",
            "selected_share_class_figi",
            "selected_composite_figi",
            "selected_reference_source_sha256",
        ]
    ]


def _collect_query(
    client: MassiveReferenceClient,
    raw_root: Path,
    row: Any,
    query_kind: str,
) -> tuple[dict[str, Any], int, int]:
    action_target_id = str(row.action_target_id)
    existing = _load_query_completion(raw_root, action_target_id, query_kind)
    if existing is not None:
        return existing, 0, len(existing["snapshots"])
    url: str | None = _initial_query_url(client, row, query_kind)
    snapshots: list[str] = []
    result_count = rejection_count = requests_made = reused = page_number = 0
    while url is not None:
        page_number += 1
        if page_number > 20:
            raise MassiveRequestError(
                "Security-actions pagination exceeded 20 pages", reason="pagination"
            )
        query_id = _stable_hash(
            {"action_target_id": action_target_id, "query_kind": query_kind, "page_url": url}
        )
        snapshot = _existing_snapshot(raw_root, query_id, url)
        if snapshot is None:
            response = client.get(url)
            snapshot = _preserve_response(
                raw_root, row, query_id, query_kind, page_number, response
            )
            requests_made += response.requests_made
        else:
            reused += 1
        parsed, rejected, next_url = _parse_snapshot(snapshot, row)
        result_count += len(parsed)
        rejection_count += len(rejected)
        snapshots.append(str(_snapshot_manifest_path(raw_root, snapshot).resolve()))
        url = next_url
    completion = {
        "action_target_id": action_target_id,
        "query_kind": query_kind,
        "result_count": result_count,
        "rejection_count": rejection_count,
        "page_count": len(snapshots),
        "pagination_complete": True,
        "snapshots": snapshots,
    }
    _write_query_completion(raw_root, action_target_id, query_kind, completion)
    return completion, requests_made, reused


def _initial_query_url(client: MassiveReferenceClient, row: Any, query_kind: str) -> str:
    if query_kind == "reference_start":
        return client.ticker_reference_url(str(row.cik), str(row.price_window_start), active=True)
    if query_kind == "reference_end":
        return client.ticker_reference_url(str(row.cik), str(row.price_window_end), active=True)
    if query_kind == "ticker_events":
        return massive_ticker_events_url(str(row.selected_composite_figi))
    if query_kind == "splits":
        return massive_splits_url(
            str(row.selected_ticker), str(row.price_window_start), str(row.price_window_end)
        )
    if query_kind == "dividends":
        return massive_dividends_url(
            str(row.selected_ticker), str(row.price_window_start), str(row.price_window_end)
        )
    if query_kind.startswith("reference_session_"):
        session_date = query_kind.removeprefix("reference_session_")
        _parse_date(session_date)
        return client.ticker_reference_url(str(row.cik), session_date, active=True)
    raise MassiveConfigurationError(f"Unknown security-actions query kind: {query_kind}")


def _parse_snapshot(
    snapshot: MassiveSecurityActionsSnapshot, row: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    content = Path(snapshot.path).read_bytes()
    if snapshot.query_kind in {
        "reference_start",
        "reference_end",
    } or snapshot.query_kind.startswith("reference_session_"):
        if snapshot.query_kind == "reference_start":
            as_of = str(row.price_window_start)
        elif snapshot.query_kind == "reference_end":
            as_of = str(row.price_window_end)
        else:
            as_of = snapshot.query_kind.removeprefix("reference_session_")
        parsed, rejected, next_url = parse_massive_ticker_response(
            content,
            expected_cik=str(row.cik),
            target_id=str(row.action_target_id),
            query_active=True,
            as_of_date=as_of,
            source_sha256=snapshot.sha256,
        )
        for item in parsed:
            item["action_target_id"] = item.pop("target_id")
        return (
            parsed,
            [
                {
                    "action_target_id": item["target_id"],
                    "query_kind": snapshot.query_kind,
                    "source_occurrence": item["source_occurrence"],
                    "reason": item["reason"],
                    "source_sha256": item["source_sha256"],
                }
                for item in rejected
            ],
            next_url,
        )
    if snapshot.query_kind == "ticker_events":
        return parse_massive_ticker_events(
            content,
            action_target_id=str(row.action_target_id),
            source_sha256=snapshot.sha256,
        )
    if snapshot.query_kind == "splits":
        return parse_massive_splits(
            content,
            action_target_id=str(row.action_target_id),
            expected_ticker=str(row.selected_ticker),
            window_start=str(row.price_window_start),
            window_end=str(row.price_window_end),
            source_sha256=snapshot.sha256,
        )
    return parse_massive_dividends(
        content,
        action_target_id=str(row.action_target_id),
        expected_ticker=str(row.selected_ticker),
        expected_currency=str(row.currency_name),
        window_start=str(row.price_window_start),
        window_end=str(row.price_window_end),
        source_sha256=snapshot.sha256,
    )


def _build_readiness_tables(
    targets: pd.DataFrame,
    sessions: pd.DataFrame,
    identity: pd.DataFrame,
    ticker_events: pd.DataFrame,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    rejections: pd.DataFrame,
    completions: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = targets.copy()
    for query_kind in _QUERY_KINDS:
        audit[f"{query_kind}_query_complete"] = audit["action_target_id"].map(
            lambda value, kind=query_kind: (str(value), kind) in completions
        )
    query_columns = [f"{kind}_query_complete" for kind in _QUERY_KINDS]
    audit["all_security_action_queries_complete"] = audit[query_columns].all(axis=1)
    audit["reference_session_grid_complete"] = False
    audit["reference_session_expected"] = 0
    audit["reference_session_exact_matches"] = 0
    audit["ticker_events_unavailable_fallback_used"] = False
    audit["reference_start_exact_matches"] = 0
    audit["reference_end_exact_matches"] = 0
    audit["ticker_changes_inside_window"] = 0
    audit["split_count"] = 0
    audit["dividend_count"] = 0
    audit["rejected_identity_rows"] = 0
    audit["rejected_action_rows"] = 0
    audit["historical_symbol_interval_status"] = "not_collected"
    audit["historical_symbol_interval_ready"] = False
    audit["corporate_action_status"] = "not_collected"
    audit["corporate_actions_ready"] = False
    audit["stable_security_id"] = pd.NA
    mapping_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    session_lookup = {
        (str(row.exchange_mic), str(row.session_date)): str(row.regular_open_at)
        for row in sessions.itertuples(index=False)
    }
    snapshot_hashes = _completion_source_hashes(completions)
    for index, target in audit.iterrows():
        action_target_id = str(target["action_target_id"])
        target_identity = identity[identity["action_target_id"].eq(action_target_id)]
        exact = target_identity[
            target_identity["ticker"].eq(str(target["selected_ticker"]))
            & target_identity["exchange_mic"].eq(str(target["selected_exchange_mic"]))
            & target_identity["provider_security_id"].eq(
                str(target["selected_provider_security_id"])
            )
            & target_identity["share_class_figi"].eq(str(target["selected_share_class_figi"]))
            & target_identity["composite_figi"].eq(str(target["selected_composite_figi"]))
            & target_identity["security_type"].eq("CS")
        ]
        start_matches = exact[exact["query_kind"].eq("reference_start")]
        end_matches = exact[exact["query_kind"].eq("reference_end")]
        start_count, end_count = len(start_matches), len(end_matches)
        audit.at[index, "reference_start_exact_matches"] = start_count
        audit.at[index, "reference_end_exact_matches"] = end_count
        start = _parse_date(str(target["price_window_start"]))
        end = _parse_date(str(target["price_window_end"]))
        target_sessions = sessions[
            sessions["exchange_mic"].eq(str(target["selected_exchange_mic"]))
            & sessions["session_date"].between(start.isoformat(), end.isoformat())
        ]
        expected_session_dates = target_sessions["session_date"].astype(str).tolist()
        session_query_kinds = [
            f"reference_session_{session_date}" for session_date in expected_session_dates
        ]
        session_grid_complete = bool(session_query_kinds) and all(
            (action_target_id, query_kind) in completions for query_kind in session_query_kinds
        )
        session_matches = exact[exact["query_kind"].isin(session_query_kinds)]
        session_exact_counts = session_matches.groupby("query_kind").size()
        session_exact_match_count = int((session_exact_counts == 1).sum())
        audit.at[index, "reference_session_grid_complete"] = session_grid_complete
        audit.at[index, "reference_session_expected"] = len(session_query_kinds)
        audit.at[index, "reference_session_exact_matches"] = session_exact_match_count
        changes = ticker_events[ticker_events["action_target_id"].eq(action_target_id)]
        if not changes.empty:
            change_dates = pd.to_datetime(changes["event_date"], errors="coerce").dt.date
            changes = changes[(change_dates >= start) & (change_dates <= end)]
        audit.at[index, "ticker_changes_inside_window"] = len(changes)
        target_rejections = rejections[rejections["action_target_id"].eq(action_target_id)]
        identity_rejection_mask = target_rejections["query_kind"].isin(
            ["reference_start", "reference_end", "ticker_events"]
        ) | target_rejections["query_kind"].astype(str).str.startswith("reference_session_")
        identity_rejections = target_rejections[identity_rejection_mask]
        action_rejections = target_rejections[
            target_rejections["query_kind"].isin(["splits", "dividends"])
        ]
        audit.at[index, "rejected_identity_rows"] = len(identity_rejections)
        audit.at[index, "rejected_action_rows"] = len(action_rejections)
        ticker_event_route_ready = all(
            bool(target[f"{kind}_query_complete"])
            for kind in ("reference_start", "reference_end", "ticker_events")
        )
        session_grid_route_ready = (
            bool(target["reference_start_query_complete"])
            and bool(target["reference_end_query_complete"])
            and session_grid_complete
            and session_exact_match_count == len(session_query_kinds)
            and len(session_matches) == len(session_query_kinds)
        )
        identity_queries_ready = ticker_event_route_ready or session_grid_route_ready
        identity_ready = (
            identity_queries_ready
            and start_count == 1
            and end_count == 1
            and (changes.empty or session_grid_route_ready)
            and identity_rejections.empty
        )
        audit.at[index, "ticker_events_unavailable_fallback_used"] = (
            session_grid_route_ready and not ticker_event_route_ready
        )
        if not identity_queries_ready:
            identity_status = "not_collected"
        elif not identity_rejections.empty:
            identity_status = "identity_parse_rejection"
        elif start_count != 1 or end_count != 1:
            identity_status = "endpoint_identity_mismatch"
        elif not changes.empty and not session_grid_route_ready:
            identity_status = "ticker_change_inside_window"
        elif session_grid_route_ready and not ticker_event_route_ready:
            identity_status = "validated_window_continuity_via_session_grid_fallback"
        else:
            identity_status = "validated_window_continuity"
        audit.at[index, "historical_symbol_interval_status"] = identity_status
        audit.at[index, "historical_symbol_interval_ready"] = identity_ready
        if identity_ready:
            audit.at[index, "stable_security_id"] = str(target["selected_provider_security_id"])
            mapping_rows.append(
                {
                    "action_target_id": action_target_id,
                    "price_target_id": target["price_target_id"],
                    "target_id": target["target_id"],
                    "cik": target["cik"],
                    "security_id": target["selected_provider_security_id"],
                    "security_id_kind": target["selected_provider_security_id_kind"],
                    "ticker": target["selected_ticker"],
                    "exchange_mic": target["selected_exchange_mic"],
                    "valid_from_inclusive": _local_midnight(start).isoformat(),
                    "valid_to_exclusive": _local_midnight(end + timedelta(days=1)).isoformat(),
                    "mapping_method": (
                        "massive_pit_every_exchange_session"
                        if session_grid_route_ready and not ticker_event_route_ready
                        else "massive_pit_endpoints_plus_ticker_event_timeline"
                    ),
                    "source_snapshot_sha256": _first_hash(
                        snapshot_hashes, action_target_id, "reference_start"
                    ),
                    "end_source_snapshot_sha256": _first_hash(
                        snapshot_hashes, action_target_id, "reference_end"
                    ),
                    "ticker_events_source_snapshot_sha256": (
                        _first_hash(snapshot_hashes, action_target_id, "ticker_events")
                        if ticker_event_route_ready
                        else None
                    ),
                    "session_grid_snapshot_count": (
                        len(session_query_kinds)
                        if session_grid_route_ready and not ticker_event_route_ready
                        else 0
                    ),
                    "ticker_events_endpoint_experimental": True,
                }
            )
        target_splits = splits[splits["action_target_id"].eq(action_target_id)]
        target_dividends = dividends[dividends["action_target_id"].eq(action_target_id)]
        audit.at[index, "split_count"] = len(target_splits)
        audit.at[index, "dividend_count"] = len(target_dividends)
        normalized_actions, session_failures = _normalize_actions(
            target, target_splits, target_dividends, session_lookup
        )
        action_rows.extend(normalized_actions)
        action_queries_ready = bool(target["splits_query_complete"]) and bool(
            target["dividends_query_complete"]
        )
        action_ready = action_queries_ready and action_rejections.empty and session_failures == 0
        if not action_queries_ready:
            action_status = "not_collected"
        elif not action_rejections.empty:
            action_status = "action_parse_rejection"
        elif session_failures:
            action_status = "action_date_not_exchange_session"
        else:
            action_status = "complete_including_zero_events"
        audit.at[index, "corporate_action_status"] = action_status
        audit.at[index, "corporate_actions_ready"] = action_ready
    mappings = pd.DataFrame(mapping_rows, columns=_mapping_columns())
    actions = pd.DataFrame(action_rows, columns=_normalized_action_columns())
    return audit, mappings, actions


def _normalize_actions(
    target: pd.Series[Any],
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    session_lookup: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0
    exchange = str(target["selected_exchange_mic"])
    base = {
        "action_target_id": str(target["action_target_id"]),
        "price_target_id": str(target["price_target_id"]),
        "target_id": str(target["target_id"]),
        "security_id": str(target["selected_provider_security_id"]),
        "ticker": str(target["selected_ticker"]),
        "exchange_mic": exchange,
    }
    for item in splits.itertuples(index=False):
        effective_at = session_lookup.get((exchange, str(item.execution_date)))
        if effective_at is None:
            failures += 1
            continue
        rows.append(
            {
                **base,
                "provider_event_id": item.provider_event_id,
                "effective_at": effective_at,
                "action_type": str(item.adjustment_type),
                "cash_amount": np.nan,
                "currency": None,
                "split_from": _numeric_scalar(item.split_from),
                "split_to": _numeric_scalar(item.split_to),
                "source_snapshot_sha256": str(item.source_sha256),
            }
        )
    for item in dividends.itertuples(index=False):
        effective_at = session_lookup.get((exchange, str(item.ex_dividend_date)))
        if effective_at is None:
            failures += 1
            continue
        rows.append(
            {
                **base,
                "provider_event_id": item.provider_event_id,
                "effective_at": effective_at,
                "action_type": "cash_dividend",
                "cash_amount": _numeric_scalar(item.cash_amount),
                "currency": str(item.currency),
                "split_from": np.nan,
                "split_to": np.nan,
                "source_snapshot_sha256": str(item.source_sha256),
            }
        )
    return rows, failures


def _completion_source_hashes(
    completions: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for key, completion in completions.items():
        result[key] = [_load_snapshot(Path(path)).sha256 for path in completion["snapshots"]]
    return result


def _first_hash(
    hashes: dict[tuple[str, str], list[str]], action_target_id: str, query_kind: str
) -> str:
    values = hashes.get((action_target_id, query_kind), [])
    if not values:
        raise FilingCatalogError("Validated identity lacks a traceable raw snapshot")
    return values[0]


def _preserve_response(
    raw_root: Path,
    row: Any,
    query_id: str,
    query_kind: str,
    page_number: int,
    response: MassiveResponse,
) -> MassiveSecurityActionsSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Security-actions raw snapshot hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Security-actions response changed before preservation")
        temporary.replace(path)
    snapshot = MassiveSecurityActionsSnapshot(
        action_target_id=str(row.action_target_id),
        query_id=query_id,
        query_kind=query_kind,
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
) -> MassiveSecurityActionsSnapshot | None:
    for path in sorted((raw_root / "snapshots" / query_id).glob("*.json")):
        snapshot = _load_snapshot(path)
        if snapshot.requested_url == requested_url:
            return snapshot
    return None


def _load_snapshot(path: Path) -> MassiveSecurityActionsSnapshot:
    snapshot = MassiveSecurityActionsSnapshot(**_read_json(path))
    raw_path = Path(snapshot.path)
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Security-actions snapshot is missing or invalid: {raw_path}")
    return snapshot


def _snapshot_manifest_path(raw_root: Path, snapshot: MassiveSecurityActionsSnapshot) -> Path:
    return raw_root / "snapshots" / snapshot.query_id / f"{snapshot.sha256}.json"


def _write_query_completion(
    raw_root: Path,
    action_target_id: str,
    query_kind: str,
    completion: dict[str, Any],
) -> None:
    path = raw_root / "queries" / action_target_id / f"{query_kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")


def _load_query_completion(
    raw_root: Path, action_target_id: str, query_kind: str
) -> dict[str, Any] | None:
    path = raw_root / "queries" / action_target_id / f"{query_kind}.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    if document.get("query_kind") != query_kind or not document.get("pagination_complete"):
        return None
    for snapshot_path in document.get("snapshots", []):
        snapshot = _load_snapshot(Path(snapshot_path))
        if snapshot.query_kind != query_kind:
            return None
    return document


def _write_target_completion(raw_root: Path, row: Any, queries: dict[str, dict[str, Any]]) -> None:
    path = raw_root / "completed" / f"{row.action_target_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "action_target_id": str(row.action_target_id),
                    "price_target_id": str(row.price_target_id),
                    "target_id": str(row.target_id),
                    "ticker": str(row.selected_ticker),
                    "status": "all_evidence_queries_complete",
                    "queries": queries,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _load_target_completion(raw_root: Path, action_target_id: str) -> dict[str, Any] | None:
    path = raw_root / "completed" / f"{action_target_id}.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    queries = document.get("queries", {})
    if not isinstance(queries, dict) or set(queries) != set(_QUERY_KINDS):
        return None
    for query_kind in _QUERY_KINDS:
        if _load_query_completion(raw_root, action_target_id, query_kind) is None:
            return None
    return document


def _valid_target_completion(raw_root: Path, action_target_id: str) -> bool:
    try:
        return _load_target_completion(raw_root, action_target_id) is not None
    except (FilingCatalogError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _target_session_dates(row: Any, sessions: pd.DataFrame) -> list[str]:
    selected = sessions[
        sessions["exchange_mic"].eq(str(row.selected_exchange_mic))
        & sessions["session_date"].between(str(row.price_window_start), str(row.price_window_end))
    ]
    dates = selected["session_date"].astype(str).sort_values().tolist()
    if not dates or len(dates) != len(set(dates)):
        raise FilingCatalogError("Session fallback requires a unique nonempty calendar grid")
    return dates


def _write_session_fallback_completion(
    raw_root: Path,
    row: Any,
    session_dates: list[str],
    query_kinds: list[str],
) -> None:
    path = raw_root / "fallbacks" / str(row.action_target_id) / "reference-session-grid.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "action_target_id": str(row.action_target_id),
                    "fallback_method": "massive_pit_reference_every_exchange_session",
                    "trigger": "experimental_ticker_events_composite_figi_not_found",
                    "session_dates": session_dates,
                    "query_kinds": query_kinds,
                    "pagination_complete": True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _load_session_fallback_completion(
    raw_root: Path, row: Any, sessions: pd.DataFrame
) -> dict[str, Any] | None:
    path = raw_root / "fallbacks" / str(row.action_target_id) / "reference-session-grid.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    expected_dates = _target_session_dates(row, sessions)
    expected_kinds = [f"reference_session_{value}" for value in expected_dates]
    if (
        document.get("session_dates") != expected_dates
        or document.get("query_kinds") != expected_kinds
        or not document.get("pagination_complete")
    ):
        return None
    for query_kind in expected_kinds:
        if _load_query_completion(raw_root, str(row.action_target_id), query_kind) is None:
            return None
    return document


def _ticker_events_not_found(raw_root: Path, action_target_id: str) -> bool:
    failure_root = raw_root / "failures" / action_target_id
    for path in sorted(failure_root.glob("*.json"), reverse=True):
        document = _read_json(path)
        detail = str(document.get("detail", ""))
        query_kind = document.get("query_kind")
        if query_kind not in {None, "ticker_events"}:
            continue
        if document.get("reason") == "not_found" or (
            "HTTP 404" in detail and "/events?types=ticker_change" in detail
        ):
            return True
    return False


def _write_failure(
    raw_root: Path,
    action_target_id: str,
    reason: str,
    detail: str,
    *,
    query_kind: str | None = None,
) -> None:
    path = (
        raw_root
        / "failures"
        / action_target_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "action_target_id": action_target_id,
                "query_kind": query_kind,
                "reason": reason,
                "detail": detail,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _action_url(
    base_url: str, ticker: str, date_field: str, window_start: str, window_end: str
) -> str:
    normalized_ticker = _required_text(ticker, "ticker")
    start, end = _parse_date(window_start), _parse_date(window_end)
    if start > end:
        raise MassiveConfigurationError("Corporate-action range starts after it ends")
    query = urlencode(
        {
            "ticker": normalized_ticker,
            f"{date_field}.gte": start.isoformat(),
            f"{date_field}.lte": end.isoformat(),
            "limit": "5000",
            "sort": f"{date_field}.asc",
        }
    )
    return f"{base_url}?{query}"


def _response_document(content: bytes, label: str) -> dict[str, Any]:
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRequestError(
            f"Malformed Massive {label} JSON: {exc}", reason="malformed_json"
        ) from exc
    if not isinstance(document, dict) or document.get("status") != "OK":
        raise MassiveRequestError(f"Massive {label} response status is not OK", reason="bad_status")
    return document


def _result_list(document: dict[str, Any], label: str) -> list[Any]:
    results = document.get("results", [])
    if not isinstance(results, list):
        raise MassiveRequestError(f"Massive {label} results must be a list", reason="bad_schema")
    return results


def _split_rejection_reason(item: Any, expected_ticker: str, start: date, end: date) -> str | None:
    if not isinstance(item, dict):
        return "split_not_object"
    if _optional_text(item.get("ticker")) != expected_ticker:
        return "split_ticker_mismatch"
    execution_date = _valid_date_text(item.get("execution_date"))
    if execution_date is None:
        return "invalid_split_execution_date"
    if not start <= _parse_date(execution_date) <= end:
        return "split_outside_requested_window"
    if item.get("adjustment_type") not in {
        "forward_split",
        "reverse_split",
        "stock_dividend",
    }:
        return "invalid_split_adjustment_type"
    if _required_positive_float(item.get("split_from")) is None:
        return "invalid_split_from"
    if _required_positive_float(item.get("split_to")) is None:
        return "invalid_split_to"
    factor = item.get("historical_adjustment_factor")
    if factor is not None and _optional_positive_float(factor) is None:
        return "invalid_split_historical_factor"
    return None


def _dividend_rejection_reason(
    item: Any,
    expected_ticker: str,
    expected_currency: str,
    start: date,
    end: date,
) -> str | None:
    if not isinstance(item, dict):
        return "dividend_not_object"
    if _optional_text(item.get("ticker")) != expected_ticker:
        return "dividend_ticker_mismatch"
    event_date = _valid_date_text(item.get("ex_dividend_date"))
    if event_date is None:
        return "invalid_ex_dividend_date"
    if not start <= _parse_date(event_date) <= end:
        return "dividend_outside_requested_window"
    if _required_positive_float(item.get("cash_amount")) is None:
        return "invalid_dividend_cash_amount"
    currency = _optional_text(item.get("currency"))
    if currency is None or currency.casefold() != expected_currency.casefold():
        return "dividend_currency_mismatch"
    for field in ("declaration_date", "record_date", "pay_date"):
        if item.get(field) is not None and _valid_date_text(item.get(field)) is None:
            return f"invalid_{field}"
    for field in ("split_adjusted_cash_amount", "historical_adjustment_factor"):
        if item.get(field) is not None and _optional_positive_float(item.get(field)) is None:
            return f"invalid_dividend_{field}"
    return None


def _rejection(
    action_target_id: str,
    query_kind: str,
    occurrence: int,
    reason: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "action_target_id": action_target_id,
        "query_kind": query_kind,
        "source_occurrence": occurrence,
        "reason": reason,
        "source_sha256": source_sha256,
    }


def _safe_optional_next_url(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _required_text(value, "next_url")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
        raise MassiveRequestError("Unsafe Massive pagination URL", reason="unsafe_pagination")
    safe_query = urlencode(
        [(key, item) for key, item in parse_qsl(parsed.query) if key.casefold() != "apikey"]
    )
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", safe_query, ""))


def _required_text(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise MassiveConfigurationError(f"Massive {label} is missing")
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _valid_date_text(value: Any) -> str | None:
    normalized = _optional_text(value)
    if normalized is None:
        return None
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError:
        return None


def _required_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    return _required_positive_float(value)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or int(numeric) != numeric or numeric < 0:
        return None
    return int(numeric)


def _numeric_scalar(value: Any) -> float:
    """Convert a previously validated pandas scalar to float."""
    return float(value)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveConfigurationError(f"Invalid ISO date: {value}") from exc


def _local_midnight(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=_NEW_YORK)


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _snapshot_row(snapshot: MassiveSecurityActionsSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def _identity_columns() -> list[str]:
    return [
        "action_target_id",
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
        "query_kind",
    ]


def _ticker_event_columns() -> list[str]:
    return [
        "action_target_id",
        "event_date",
        "event_type",
        "new_ticker",
        "source_occurrence",
        "source_sha256",
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
    return ["action_target_id", "query_kind", "source_occurrence", "reason", "source_sha256"]


def _snapshot_columns() -> list[str]:
    return [field.name for field in MassiveSecurityActionsSnapshot.__dataclass_fields__.values()]


def _mapping_columns() -> list[str]:
    return [
        "action_target_id",
        "price_target_id",
        "target_id",
        "cik",
        "security_id",
        "security_id_kind",
        "ticker",
        "exchange_mic",
        "valid_from_inclusive",
        "valid_to_exclusive",
        "mapping_method",
        "source_snapshot_sha256",
        "end_source_snapshot_sha256",
        "ticker_events_source_snapshot_sha256",
        "session_grid_snapshot_count",
        "ticker_events_endpoint_experimental",
    ]


def _normalized_action_columns() -> list[str]:
    return [
        "action_target_id",
        "price_target_id",
        "target_id",
        "security_id",
        "ticker",
        "exchange_mic",
        "provider_event_id",
        "effective_at",
        "action_type",
        "cash_amount",
        "currency",
        "split_from",
        "split_to",
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
