"""Collect and audit exact Massive responses for shadow market evidence."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.exchange_calendar import (
    build_historical_exchange_sessions,
    load_preserved_official_calendar_sources,
)
from sec_data.filing_catalog import FilingCatalogError
from sec_data.massive_prices import massive_daily_bars_url, parse_massive_daily_bars
from sec_data.massive_reference import (
    MassiveReferenceClient,
    MassiveRequestError,
    MassiveResponse,
    parse_massive_ticker_response,
)
from sec_data.massive_security_actions import (
    massive_dividends_url,
    massive_splits_url,
    parse_massive_dividends,
    parse_massive_splits,
)

SHADOW_MARKET_COLLECTION_VERSION: Final[str] = "shadow-market-collection-v1"


class ShadowMarketCollectionError(FilingCatalogError):
    """Raised when shadow market collection or source lineage is invalid."""


@dataclass(frozen=True)
class ShadowMarketSnapshot:
    """One content-addressed, secret-free Massive response snapshot."""

    market_evidence_query_id: str
    prospective_event_id: str
    query_type: str
    query_variant: str
    capture_date: str
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
class ShadowMarketCollectionResult:
    """Counts from one resumable daily market-evidence capture."""

    selected_queries: int
    completed_queries: int
    failed_queries: int
    requests_made: int
    snapshots_reused: int
    blocking_reason: str | None


@dataclass(frozen=True)
class ShadowMarketAuditResult:
    """One immutable parsed audit over all preserved query snapshots."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ShadowMarketReadinessConfig:
    """Frozen pre-entry evidence requirements for order-candidate promotion."""

    liquidity_history_sessions: int = 20
    entry_session_offset: int = 2
    holding_sessions: int = 20
    fixed_slippage_bps: float = 5.0
    minimum_round_trip_cost_bps: float = 10.0
    maximum_round_trip_cost_bps: float = 100.0

    def validate(self) -> None:
        """Reject departures from the preregistered shadow trading policy."""
        if tuple(asdict(self).values()) != (20, 2, 20, 5.0, 10.0, 100.0):
            raise ValueError("Shadow market readiness assumptions are frozen")


def collect_shadow_market_evidence_queries(
    queries: pd.DataFrame,
    client: MassiveReferenceClient | None,
    *,
    raw_root: Path = Path("data/market/shadow_market_evidence"),
    limit: int = 20,
    capture_date: date | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ShadowMarketCollectionResult:
    """Capture each eligible query at most once per UTC date.

    Identity and pre-acceptance cost queries become permanently reusable after
    their first successful capture. Execution bars, splits, and dividends may
    receive a new immutable snapshot on later dates while an event is monitored.
    """
    _require_query_columns(queries)
    if limit < 1:
        raise ShadowMarketCollectionError("Market-evidence collection limit must be positive")
    day = capture_date or datetime.now(timezone.utc).date()
    if queries.empty:
        return ShadowMarketCollectionResult(0, 0, 0, 0, 0, None)
    eligible = queries.loc[
        pd.to_datetime(queries["query_not_before"], utc=True).dt.date.le(day)
    ].copy()
    eligible = eligible.sort_values(
        ["sec_accepted_at", "prospective_event_id", "query_type"], kind="mergesort"
    )
    pending = [
        row
        for row in eligible.itertuples(index=False)
        if _query_capture_needed(
            raw_root, str(row.market_evidence_query_id), str(row.query_type), day
        )
    ][:limit]
    if pending and client is None:
        raise ShadowMarketCollectionError("MASSIVE_API_KEY is required for eligible market queries")
    completed = failed = requests_made = reused = 0
    blocking_reason: str | None = None
    for number, row in enumerate(pending, start=1):
        assert client is not None
        query_id = str(row.market_evidence_query_id)
        try:
            snapshots, requests, reused_pages = _collect_query(row, client, raw_root, day)
            _write_capture_completion(raw_root, row, day, snapshots)
            requests_made += requests
            reused += reused_pages
            completed += 1
        except MassiveRequestError as exc:
            failed += 1
            requests_made += exc.requests_made
            _write_failure(raw_root, query_id, exc.reason, str(exc))
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocking_reason = exc.reason
                break
        if progress is not None:
            progress(number, len(pending), query_id)
    return ShadowMarketCollectionResult(
        len(pending), completed, failed, requests_made, reused, blocking_reason
    )


def build_shadow_market_evidence_audit(
    market_plan_root: Path,
    *,
    raw_root: Path = Path("data/market/shadow_market_evidence"),
    raw_calendar_root: Path = Path("data/market/exchange_calendar"),
    output_root: Path = Path("data/processed/shadow_market_evidence_audit"),
    config: ShadowMarketReadinessConfig | None = None,
) -> ShadowMarketAuditResult:
    """Parse preserved responses and audit query coverage without calculating returns."""
    cfg = config or ShadowMarketReadinessConfig()
    cfg.validate()
    plan_manifest = _sha256_file(market_plan_root / "output_manifest.json")
    plan_events = pd.read_parquet(market_plan_root / "shadow_market_evidence_events.parquet")
    queries = pd.read_parquet(market_plan_root / "shadow_market_evidence_queries.parquet")
    _require_query_columns(queries)
    snapshots = _load_all_snapshots(raw_root, set(queries["market_evidence_query_id"].astype(str)))
    parsed = _parse_snapshots(queries, snapshots)
    query_audit = _query_audit(queries, snapshots, parsed["rejections"])
    sessions, calendar_errors, calendar_source_hashes = _load_required_sessions(
        plan_events, raw_calendar_root
    )
    event_audit = calculate_shadow_market_order_readiness(
        plan_events,
        query_audit,
        parsed["identity"],
        parsed["prices"],
        sessions,
        cfg,
        calendar_errors=calendar_errors,
    )
    source_hashes = sorted(snapshots.get("sha256", pd.Series(dtype=str)).astype(str).unique())
    lineage = {
        "market_plan_manifest_sha256": plan_manifest,
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    payload = json.dumps(
        {
            "version": SHADOW_MARKET_COLLECTION_VERSION,
            **lineage,
            "source_hashes": source_hashes,
            "calendar_source_hashes": calendar_source_hashes,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "shadow-market-evidence-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ShadowMarketAuditResult(
            target.resolve(), _read_json(target / "shadow_market_evidence_audit_report.json")
        )
    if target.exists():
        raise ShadowMarketCollectionError(f"Incomplete market audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "shadow_market_query_audit.parquet": query_audit,
        "shadow_market_event_audit.parquet": event_audit,
        "shadow_market_response_snapshots.parquet": snapshots,
        "shadow_market_identity_observations.parquet": parsed["identity"],
        "shadow_market_price_observations.parquet": parsed["prices"],
        "shadow_market_split_observations.parquet": parsed["splits"],
        "shadow_market_dividend_observations.parquet": parsed["dividends"],
        "shadow_market_parse_rejections.parquet": parsed["rejections"],
        "shadow_market_official_sessions.parquet": sessions,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    query_complete = (
        int(query_audit["query_capture_observed"].sum()) if not query_audit.empty else 0
    )
    report: dict[str, Any] = {
        "contract_version": SHADOW_MARKET_COLLECTION_VERSION,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **lineage,
        "planned_queries": len(queries),
        "queries_with_captures": query_complete,
        "source_snapshots": len(snapshots),
        "identity_observations": len(parsed["identity"]),
        "price_observations": len(parsed["prices"]),
        "split_observations": len(parsed["splits"]),
        "dividend_observations": len(parsed["dividends"]),
        "parse_rejections": len(parsed["rejections"]),
        "point_in_time_identity_verified_events": int(
            event_audit["point_in_time_identity_verified"].sum()
        ),
        "official_calendar_ready_events": int(event_audit["official_calendar_window_ready"].sum()),
        "preacceptance_cost_ready_events": int(event_audit["preacceptance_cost_ready"].sum()),
        "sizing_reference_ready_events": int(event_audit["sizing_reference_price_ready"].sum()),
        "preentry_action_evidence_ready_events": int(
            event_audit["corporate_action_evidence_ready"].sum()
        ),
        "order_evidence_ready_events": int(event_audit["order_evidence_ready"].sum()),
        "calendar_errors": calendar_errors,
        "calendar_source_hashes": calendar_source_hashes,
        "config": asdict(cfg),
        "return_values_loaded": False,
        "broker_submission_authorized": False,
        "live_trading_authorized": False,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "shadow_market_evidence_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **lineage,
                "source_hashes": source_hashes,
                "calendar_source_hashes": calendar_source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ShadowMarketAuditResult(target.resolve(), report)


def _collect_query(
    row: Any,
    client: MassiveReferenceClient,
    raw_root: Path,
    day: date,
) -> tuple[list[ShadowMarketSnapshot], int, int]:
    query_type = str(row.query_type)
    variants: list[tuple[str, str]]
    if query_type == "point_in_time_identity":
        variants = [
            (
                "active_true",
                client.ticker_reference_url(
                    str(row.cik), pd.Timestamp(row.sec_accepted_at).date().isoformat(), active=True
                ),
            ),
            (
                "active_false",
                client.ticker_reference_url(
                    str(row.cik), pd.Timestamp(row.sec_accepted_at).date().isoformat(), active=False
                ),
            ),
        ]
    elif query_type in {"preacceptance_cost_bars", "execution_window_bars"}:
        variants = [
            (
                "primary",
                massive_daily_bars_url(str(row.ticker), str(row.window_start), str(row.window_end)),
            )
        ]
    elif query_type == "split_actions":
        variants = [
            (
                "primary",
                massive_splits_url(str(row.ticker), str(row.window_start), str(row.window_end)),
            )
        ]
    elif query_type == "cash_dividend_actions":
        variants = [
            (
                "primary",
                massive_dividends_url(str(row.ticker), str(row.window_start), str(row.window_end)),
            )
        ]
    else:
        raise ShadowMarketCollectionError(f"Unsupported market query type: {query_type}")
    all_snapshots: list[ShadowMarketSnapshot] = []
    requests = reused = 0
    for variant, first_url in variants:
        url: str | None = first_url
        page = 0
        while url is not None:
            page += 1
            if page > 20:
                raise MassiveRequestError(
                    "Shadow market pagination exceeded 20 pages", reason="pagination"
                )
            existing = _existing_snapshot(
                raw_root, str(row.market_evidence_query_id), day, variant, page, url
            )
            if existing is None:
                response = client.get(url)
                snapshot = _preserve_response(raw_root, row, day, variant, page, response)
                requests += response.requests_made
            else:
                snapshot = existing
                reused += 1
            all_snapshots.append(snapshot)
            url = _next_url(row, snapshot, variant)
    return all_snapshots, requests, reused


def _next_url(row: Any, snapshot: ShadowMarketSnapshot, variant: str) -> str | None:
    content = Path(snapshot.path).read_bytes()
    query_type = str(row.query_type)
    if query_type == "point_in_time_identity":
        _, _, next_url = parse_massive_ticker_response(
            content,
            expected_cik=str(row.cik),
            target_id=str(row.market_evidence_query_id),
            query_active=variant == "active_true",
            as_of_date=pd.Timestamp(row.sec_accepted_at).date().isoformat(),
            source_sha256=snapshot.sha256,
        )
        return next_url
    if query_type in {"preacceptance_cost_bars", "execution_window_bars"}:
        _, _, next_url = parse_massive_daily_bars(
            content,
            expected_ticker=str(row.ticker),
            price_target_id=str(row.market_evidence_query_id),
            window_start=str(row.window_start),
            window_end=str(row.window_end),
            source_sha256=snapshot.sha256,
        )
        return next_url
    if query_type == "split_actions":
        _, _, next_url = parse_massive_splits(
            content,
            action_target_id=str(row.market_evidence_query_id),
            expected_ticker=str(row.ticker),
            window_start=str(row.window_start),
            window_end=str(row.window_end),
            source_sha256=snapshot.sha256,
        )
        return next_url
    _, _, next_url = parse_massive_dividends(
        content,
        action_target_id=str(row.market_evidence_query_id),
        expected_ticker=str(row.ticker),
        expected_currency="USD",
        window_start=str(row.window_start),
        window_end=str(row.window_end),
        source_sha256=snapshot.sha256,
    )
    return next_url


def _preserve_response(
    raw_root: Path,
    row: Any,
    day: date,
    variant: str,
    page: int,
    response: MassiveResponse,
) -> ShadowMarketSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    raw_path = raw_root / "raw" / digest[:2] / f"{digest}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        temporary = raw_path.parent / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise ShadowMarketCollectionError("Market response changed before preservation")
        temporary.replace(raw_path)
    elif _sha256_file(raw_path) != digest:
        raise ShadowMarketCollectionError("Preserved market response hash mismatch")
    snapshot = ShadowMarketSnapshot(
        market_evidence_query_id=str(row.market_evidence_query_id),
        prospective_event_id=str(row.prospective_event_id),
        query_type=str(row.query_type),
        query_variant=variant,
        capture_date=day.isoformat(),
        page_number=page,
        requested_url=response.requested_url,
        response_url=response.response_url,
        sha256=digest,
        fetched_at=response.fetched_at.isoformat(),
        path=str(raw_path.resolve()),
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


def _parse_snapshots(queries: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    identity: list[dict[str, Any]] = []
    prices: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    dividends: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    if snapshots.empty:
        return _empty_parsed()
    query_lookup = queries.set_index("market_evidence_query_id")
    for raw in snapshots.to_dict("records"):
        snapshot = {str(key): value for key, value in raw.items()}
        query_id = str(snapshot["market_evidence_query_id"])
        if query_id not in query_lookup.index:
            continue
        row = query_lookup.loc[[query_id]].iloc[0]
        content = Path(str(snapshot["path"])).read_bytes()
        source_hash = str(snapshot["sha256"])
        query_type = str(snapshot["query_type"])
        parsed: list[dict[str, Any]]
        rejected: list[dict[str, Any]]
        if query_type == "point_in_time_identity":
            parsed, rejected, _ = parse_massive_ticker_response(
                content,
                expected_cik=str(row["cik"]),
                target_id=query_id,
                query_active=str(snapshot["query_variant"]) == "active_true",
                as_of_date=pd.Timestamp(row["sec_accepted_at"]).date().isoformat(),
                source_sha256=source_hash,
            )
            identity.extend(_with_snapshot_clock(parsed, snapshot))
        elif query_type in {"preacceptance_cost_bars", "execution_window_bars"}:
            parsed, rejected, _ = parse_massive_daily_bars(
                content,
                expected_ticker=str(row["ticker"]),
                price_target_id=query_id,
                window_start=str(row["window_start"]),
                window_end=str(row["window_end"]),
                source_sha256=source_hash,
            )
            prices.extend(_with_snapshot_clock(parsed, snapshot, evidence_class=query_type))
        elif query_type == "split_actions":
            parsed, rejected, _ = parse_massive_splits(
                content,
                action_target_id=query_id,
                expected_ticker=str(row["ticker"]),
                window_start=str(row["window_start"]),
                window_end=str(row["window_end"]),
                source_sha256=source_hash,
            )
            splits.extend(_with_snapshot_clock(parsed, snapshot))
        else:
            parsed, rejected, _ = parse_massive_dividends(
                content,
                action_target_id=query_id,
                expected_ticker=str(row["ticker"]),
                expected_currency="USD",
                window_start=str(row["window_start"]),
                window_end=str(row["window_end"]),
                source_sha256=source_hash,
            )
            dividends.extend(_with_snapshot_clock(parsed, snapshot))
        rejections.extend(_with_snapshot_clock(rejected, snapshot))
    return {
        "identity": pd.DataFrame(identity),
        "prices": pd.DataFrame(prices),
        "splits": pd.DataFrame(splits),
        "dividends": pd.DataFrame(dividends),
        "rejections": pd.DataFrame(rejections),
    }


def _query_audit(
    queries: pd.DataFrame, snapshots: pd.DataFrame, rejections: pd.DataFrame
) -> pd.DataFrame:
    audit = queries.copy()
    if audit.empty:
        return audit.assign(
            capture_dates=pd.Series(dtype="int64"),
            snapshot_pages=pd.Series(dtype="int64"),
            parse_rejections=pd.Series(dtype="int64"),
            first_fetched_at=pd.Series(dtype="datetime64[ns, UTC]"),
            last_fetched_at=pd.Series(dtype="datetime64[ns, UTC]"),
            query_capture_observed=pd.Series(dtype="bool"),
        )
    summary = (
        snapshots.groupby("market_evidence_query_id", as_index=False).agg(
            capture_dates=("capture_date", "nunique"),
            snapshot_pages=("sha256", "size"),
            first_fetched_at=("fetched_at", "min"),
            last_fetched_at=("fetched_at", "max"),
        )
        if not snapshots.empty
        else pd.DataFrame(columns=["market_evidence_query_id"])
    )
    rejection_counts = (
        rejections.groupby("market_evidence_query_id", as_index=False)
        .size()
        .rename(columns={"size": "parse_rejections"})
        if not rejections.empty
        else pd.DataFrame(columns=["market_evidence_query_id", "parse_rejections"])
    )
    audit = audit.merge(summary, on="market_evidence_query_id", how="left")
    audit = audit.merge(rejection_counts, on="market_evidence_query_id", how="left")
    for column in ("capture_dates", "snapshot_pages", "parse_rejections"):
        audit[column] = audit[column].fillna(0).astype(int)
    audit["first_fetched_at"] = pd.to_datetime(audit["first_fetched_at"], utc=True, errors="coerce")
    audit["last_fetched_at"] = pd.to_datetime(audit["last_fetched_at"], utc=True, errors="coerce")
    audit["query_capture_observed"] = audit["capture_dates"].gt(0)
    return audit


def calculate_shadow_market_order_readiness(
    plan_events: pd.DataFrame,
    query_audit: pd.DataFrame,
    identities: pd.DataFrame,
    prices: pd.DataFrame,
    sessions: pd.DataFrame,
    config: ShadowMarketReadinessConfig | None = None,
    *,
    calendar_errors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Promote only pre-entry, timestamped evidence into shadow order readiness.

    This function calculates no return and cannot submit an order.  A sizing
    price must be observed after the first post-acceptance session closes and
    before the second session opens.  Cost evidence uses only the 20 completed
    sessions strictly before SEC acceptance.
    """
    cfg = config or ShadowMarketReadinessConfig()
    cfg.validate()
    required = {
        "prospective_event_id",
        "decision_as_of",
        "sec_accepted_at",
        "filing_first_observed_at",
        "registered_ticker",
        "registered_exchange_mic",
        "security_candidate_ready",
        "return_values_loaded",
        "live_trading_authorized",
    }
    missing = sorted(required.difference(plan_events.columns))
    if missing:
        raise ShadowMarketCollectionError(f"Market plan events lack columns: {missing}")
    if plan_events["prospective_event_id"].duplicated().any():
        raise ShadowMarketCollectionError("Market plan events are non-atomic")
    if plan_events["return_values_loaded"].fillna(False).astype(bool).any():
        raise ShadowMarketCollectionError("Return values entered market readiness")
    if plan_events["live_trading_authorized"].fillna(False).astype(bool).any():
        raise ShadowMarketCollectionError("Live authorization entered market readiness")

    counts = (
        query_audit.groupby("prospective_event_id", as_index=False).agg(
            planned_queries=("market_evidence_query_id", "size"),
            captured_queries=("query_capture_observed", "sum"),
        )
        if not query_audit.empty
        else pd.DataFrame(columns=["prospective_event_id", "planned_queries", "captured_queries"])
    )
    base = plan_events.drop(columns=["planned_queries", "captured_queries"], errors="ignore").merge(
        counts, on="prospective_event_id", how="left", validate="one_to_one"
    )
    base["planned_queries"] = base["planned_queries"].fillna(0).astype(int)
    base["captured_queries"] = base["captured_queries"].fillna(0).astype(int)
    identity_frame = _timestamped_frame(identities)
    price_frame = _timestamped_frame(prices)
    query_frame = query_audit.copy()
    if not query_frame.empty:
        query_frame["first_fetched_at"] = pd.to_datetime(
            query_frame["first_fetched_at"], utc=True, errors="coerce"
        )
    session_frame = sessions.copy()
    if not session_frame.empty:
        session_frame["regular_open_at"] = pd.to_datetime(
            session_frame["regular_open_at"], utc=True, errors="coerce"
        )
        session_frame["regular_close_at"] = pd.to_datetime(
            session_frame["regular_close_at"], utc=True, errors="coerce"
        )
        session_frame["session_date"] = session_frame["session_date"].astype(str)

    errors = calendar_errors or {}
    output_rows: list[dict[str, Any]] = []
    for raw in base.to_dict("records"):
        row = {str(key): value for key, value in raw.items()}
        event_id = str(row["prospective_event_id"])
        accepted = _optional_utc(row.get("sec_accepted_at"))
        observed = _optional_utc(row.get("filing_first_observed_at"))
        decision_at = _optional_utc(row.get("decision_as_of"))
        ticker = str(row.get("registered_ticker") or "")
        exchange = str(row.get("registered_exchange_mic") or "")
        candidate_ready = bool(row.get("security_candidate_ready", False))
        blockers: list[str] = []

        event_sessions = session_frame.loc[
            session_frame.get("exchange_mic", pd.Series(dtype=str)).astype(str).eq(exchange)
        ].sort_values("regular_open_at", kind="mergesort")
        future = (
            event_sessions.loc[event_sessions["regular_open_at"].gt(accepted)]
            if accepted is not None and not event_sessions.empty
            else event_sessions.iloc[0:0]
        )
        entry_index = cfg.entry_session_offset - 1
        exit_index = entry_index + cfg.holding_sessions - 1
        calendar_ready = bool(len(future) > exit_index and event_id not in errors)
        entry_at = (
            pd.Timestamp(future.iloc[entry_index]["regular_open_at"]) if calendar_ready else None
        )
        exit_at = (
            pd.Timestamp(future.iloc[exit_index]["regular_close_at"]) if calendar_ready else None
        )
        entry_date = str(future.iloc[entry_index]["session_date"]) if calendar_ready else None
        exit_date = str(future.iloc[exit_index]["session_date"]) if calendar_ready else None

        identity_ready, identity_at = _identity_ready(
            identity_frame, event_id, ticker, exchange, accepted, entry_at
        )
        cost_ready, cost_bps, cost_at = _cost_ready(
            price_frame, event_sessions, event_id, accepted, entry_at, cfg
        )
        sizing_ready, sizing_price, sizing_at = _sizing_ready(
            price_frame, future, event_id, accepted, entry_at
        )
        actions_ready, action_at = _actions_ready(query_frame, event_id, accepted, entry_at)
        filing_clock_ready = bool(
            accepted is not None
            and observed is not None
            and accepted <= observed
            and entry_at is not None
            and observed < entry_at
        )

        if not candidate_ready:
            blockers.append("registered_security_candidate_missing")
        if accepted is None:
            blockers.append("sec_acceptance_pending")
        elif observed is None:
            blockers.append("filing_observation_clock_missing")
        elif not filing_clock_ready:
            blockers.append("filing_not_observed_before_entry")
        if accepted is not None and not identity_ready:
            blockers.append("point_in_time_identity_not_verified_before_entry")
        if accepted is not None and not calendar_ready:
            blockers.append(
                "official_calendar_source_missing"
                if event_id in errors
                else "official_execution_calendar_not_resolved"
            )
        if accepted is not None and not cost_ready:
            blockers.append("preacceptance_cost_evidence_not_ready")
        if accepted is not None and not sizing_ready:
            blockers.append("preentry_sizing_price_not_ready")
        if accepted is not None and not actions_ready:
            blockers.append("preentry_corporate_action_evidence_not_ready")

        ready = bool(
            candidate_ready
            and filing_clock_ready
            and identity_ready
            and calendar_ready
            and cost_ready
            and sizing_ready
            and actions_ready
        )
        if ready:
            status = "order_evidence_ready"
        elif accepted is None:
            status = "awaiting_sec_acceptance"
        elif entry_at is not None and decision_at is not None and decision_at >= entry_at:
            status = "preentry_evidence_window_missed"
        else:
            status = "collecting_preentry_evidence"
        row.update(
            {
                "point_in_time_identity_verified": identity_ready,
                "identity_evidence_observed_at": identity_at,
                "official_calendar_window_ready": calendar_ready,
                "planned_entry_session_date": entry_date,
                "planned_exit_session_date": exit_date,
                "planned_entry_at": entry_at,
                "planned_exit_at": exit_at,
                "preacceptance_cost_ready": cost_ready,
                "round_trip_cost_bps": cost_bps,
                "cost_evidence_observed_at": cost_at,
                "sizing_reference_price_ready": sizing_ready,
                "sizing_reference_price": sizing_price,
                "sizing_reference_observed_at": sizing_at,
                "corporate_action_evidence_ready": actions_ready,
                "corporate_action_evidence_observed_at": action_at,
                "market_evidence_status": status,
                "market_evidence_blockers": "|".join(blockers),
                "order_evidence_ready": ready,
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        )
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _load_required_sessions(
    plan_events: pd.DataFrame, raw_root: Path
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    accepted_values = pd.to_datetime(plan_events["sec_accepted_at"], utc=True, errors="coerce")
    accepted = plan_events.loc[accepted_values.notna()].copy()
    if accepted.empty:
        return _empty_sessions(), {}, []
    try:
        sources = load_preserved_official_calendar_sources(raw_root)
    except FilingCatalogError as exc:
        source_errors = {
            str(event_id): str(exc) for event_id in accepted["prospective_event_id"].astype(str)
        }
        return _empty_sessions(), source_errors, []
    session_frames: list[pd.DataFrame] = []
    source_hashes: set[str] = set()
    errors: dict[str, str] = {}
    for row in accepted.itertuples(index=False):
        event_id = str(row.prospective_event_id)
        accepted_at = _optional_utc(row.sec_accepted_at)
        if accepted_at is None:
            continue
        exchange = str(row.registered_exchange_mic)
        try:
            event_sessions, exceptions = build_historical_exchange_sessions(
                accepted_at.date() - timedelta(days=45),
                accepted_at.date() + timedelta(days=45),
                sources,
                {exchange},
            )
        except FilingCatalogError as exc:
            errors[event_id] = str(exc)
            continue
        session_frames.append(event_sessions)
        for frame in (event_sessions, exceptions):
            for column in ("source_snapshot_sha256", "venue_policy_source_sha256"):
                if column in frame:
                    source_hashes.update(frame[column].dropna().astype(str))
    sessions = (
        pd.concat(session_frames, ignore_index=True)
        .drop_duplicates(["exchange_mic", "session_date"])
        .sort_values(["exchange_mic", "session_date"], kind="mergesort")
        .reset_index(drop=True)
        if session_frames
        else _empty_sessions()
    )
    return sessions, errors, sorted(source_hashes)


def _identity_ready(
    identities: pd.DataFrame,
    event_id: str,
    ticker: str,
    exchange: str,
    accepted: pd.Timestamp | None,
    entry_at: pd.Timestamp | None,
) -> tuple[bool, pd.Timestamp | None]:
    if accepted is None or entry_at is None or identities.empty:
        return False, None
    matches = identities.loc[
        identities["prospective_event_id"].astype(str).eq(event_id)
        & identities["ticker"].astype(str).eq(ticker)
        & identities["exchange_mic"].astype(str).eq(exchange)
        & identities["observed_at"].ge(accepted)
        & identities["observed_at"].lt(entry_at)
    ]
    if matches.empty:
        return False, None
    return True, pd.Timestamp(matches["observed_at"].min())


def _cost_ready(
    prices: pd.DataFrame,
    sessions: pd.DataFrame,
    event_id: str,
    accepted: pd.Timestamp | None,
    entry_at: pd.Timestamp | None,
    config: ShadowMarketReadinessConfig,
) -> tuple[bool, float | None, pd.Timestamp | None]:
    if accepted is None or entry_at is None or prices.empty or sessions.empty:
        return False, None, None
    prior_sessions = sessions.loc[sessions["regular_close_at"].lt(accepted)].tail(
        config.liquidity_history_sessions
    )
    if len(prior_sessions) != config.liquidity_history_sessions:
        return False, None, None
    bars = prices.loc[
        prices["prospective_event_id"].astype(str).eq(event_id)
        & prices["evidence_class"].astype(str).eq("preacceptance_cost_bars")
        & prices["observed_at"].ge(accepted)
        & prices["observed_at"].lt(entry_at)
    ].copy()
    if bars.empty:
        return False, None, None
    bars["session_date"] = bars["session_date"].astype(str)
    bars = bars.sort_values("observed_at", kind="mergesort").drop_duplicates(
        "session_date", keep="first"
    )
    history = prior_sessions[["session_date"]].merge(
        bars, on="session_date", how="left", validate="one_to_one"
    )
    if history["close_unadjusted"].isna().any() or not _valid_ohlc(history):
        return False, None, None
    spread = _corwin_schultz_spread(history)
    if not math.isfinite(spread):
        return False, None, None
    cost = float(
        np.clip(
            spread * 10_000.0 + config.fixed_slippage_bps,
            config.minimum_round_trip_cost_bps,
            config.maximum_round_trip_cost_bps,
        )
    )
    return True, cost, pd.Timestamp(history["observed_at"].max())


def _sizing_ready(
    prices: pd.DataFrame,
    future_sessions: pd.DataFrame,
    event_id: str,
    accepted: pd.Timestamp | None,
    entry_at: pd.Timestamp | None,
) -> tuple[bool, float | None, pd.Timestamp | None]:
    if accepted is None or entry_at is None or prices.empty or future_sessions.empty:
        return False, None, None
    reference = future_sessions.iloc[0]
    reference_close = pd.Timestamp(reference["regular_close_at"])
    bars = prices.loc[
        prices["prospective_event_id"].astype(str).eq(event_id)
        & prices["evidence_class"].astype(str).eq("execution_window_bars")
        & prices["session_date"].astype(str).eq(str(reference["session_date"]))
        & prices["observed_at"].ge(reference_close)
        & prices["observed_at"].lt(entry_at)
    ].sort_values("observed_at", kind="mergesort")
    if bars.empty:
        return False, None, None
    price = pd.to_numeric(bars.iloc[0]["close_unadjusted"], errors="coerce")
    if pd.isna(price) or not math.isfinite(float(price)) or float(price) <= 0.0:
        return False, None, None
    return True, float(price), pd.Timestamp(bars.iloc[0]["observed_at"])


def _actions_ready(
    query_audit: pd.DataFrame,
    event_id: str,
    accepted: pd.Timestamp | None,
    entry_at: pd.Timestamp | None,
) -> tuple[bool, pd.Timestamp | None]:
    if accepted is None or entry_at is None or query_audit.empty:
        return False, None
    actions = query_audit.loc[
        query_audit["prospective_event_id"].astype(str).eq(event_id)
        & query_audit["query_type"].isin(["split_actions", "cash_dividend_actions"])
    ].copy()
    if set(actions["query_type"].astype(str)) != {"split_actions", "cash_dividend_actions"}:
        return False, None
    eligible = (
        actions["query_capture_observed"].astype(bool)
        & actions["parse_rejections"].eq(0)
        & actions["first_fetched_at"].ge(accepted)
        & actions["first_fetched_at"].lt(entry_at)
    )
    if len(actions) != 2 or not bool(eligible.all()):
        return False, None
    return True, pd.Timestamp(actions["first_fetched_at"].max())


def _timestamped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "observed_at" not in output:
        output["observed_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    output["observed_at"] = pd.to_datetime(output["observed_at"], utc=True, errors="coerce")
    return output


def _optional_utc(value: Any) -> pd.Timestamp | None:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _valid_ohlc(history: pd.DataFrame) -> bool:
    high = pd.to_numeric(history["high_unadjusted"], errors="coerce")
    low = pd.to_numeric(history["low_unadjusted"], errors="coerce")
    close = pd.to_numeric(history["close_unadjusted"], errors="coerce")
    return bool(
        high.notna().all()
        and low.notna().all()
        and close.notna().all()
        and high.ge(low).all()
        and close.ge(low).all()
        and close.le(high).all()
        and low.gt(0.0).all()
    )


def _corwin_schultz_spread(history: pd.DataFrame) -> float:
    high = pd.to_numeric(history["high_unadjusted"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(history["low_unadjusted"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(history["close_unadjusted"], errors="coerce").to_numpy(float)
    adjusted_high = high.copy()
    adjusted_low = low.copy()
    for index in range(1, len(history)):
        previous_close = close[index - 1]
        if not math.isfinite(previous_close) or previous_close <= 0.0:
            continue
        if previous_close < low[index]:
            gap = low[index] - previous_close
            adjusted_high[index] = high[index] - gap
            adjusted_low[index] = previous_close
        elif previous_close > high[index]:
            gap = previous_close - high[index]
            adjusted_high[index] = previous_close
            adjusted_low[index] = low[index] + gap
    estimates: list[float] = []
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    for index in range(1, len(history)):
        if (
            min(
                adjusted_high[index - 1],
                adjusted_high[index],
                adjusted_low[index - 1],
                adjusted_low[index],
            )
            <= 0.0
        ):
            continue
        beta = math.log(adjusted_high[index - 1] / adjusted_low[index - 1]) ** 2
        beta += math.log(adjusted_high[index] / adjusted_low[index]) ** 2
        gamma = (
            math.log(
                max(adjusted_high[index - 1], adjusted_high[index])
                / min(adjusted_low[index - 1], adjusted_low[index])
            )
            ** 2
        )
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denominator
        alpha -= math.sqrt(gamma / denominator)
        alpha = max(alpha, 0.0)
        spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        if math.isfinite(spread):
            estimates.append(spread)
    return float(np.median(estimates)) if estimates else float("nan")


def _empty_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange_mic",
            "session_date",
            "regular_open_at",
            "regular_close_at",
            "source_snapshot_sha256",
            "venue_policy_source_sha256",
        ]
    )


def _with_snapshot_clock(
    rows: list[dict[str, Any]],
    snapshot: dict[str, Any],
    *,
    evidence_class: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "market_evidence_query_id": snapshot["market_evidence_query_id"],
            "prospective_event_id": snapshot["prospective_event_id"],
            "evidence_class": evidence_class or snapshot["query_type"],
            "capture_date": snapshot["capture_date"],
            "observed_at": snapshot["fetched_at"],
        }
        for row in rows
    ]


def _empty_parsed() -> dict[str, pd.DataFrame]:
    common = [
        "market_evidence_query_id",
        "prospective_event_id",
        "evidence_class",
        "capture_date",
        "observed_at",
    ]
    return {
        "identity": pd.DataFrame(columns=["target_id", "ticker", "exchange_mic", *common]),
        "prices": pd.DataFrame(columns=["price_target_id", "ticker", "session_date", *common]),
        "splits": pd.DataFrame(columns=["action_target_id", "ticker", "execution_date", *common]),
        "dividends": pd.DataFrame(
            columns=["action_target_id", "ticker", "ex_dividend_date", *common]
        ),
        "rejections": pd.DataFrame(columns=["reason", *common]),
    }


def _query_capture_needed(raw_root: Path, query_id: str, query_type: str, day: date) -> bool:
    completed_root = raw_root / "completed" / query_id
    if query_type in {"point_in_time_identity", "preacceptance_cost_bars"}:
        return not any(completed_root.glob("*.json"))
    return not (completed_root / f"{day.isoformat()}.json").is_file()


def _write_capture_completion(
    raw_root: Path, row: Any, day: date, snapshots: list[ShadowMarketSnapshot]
) -> None:
    path = raw_root / "completed" / str(row.market_evidence_query_id) / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "market_evidence_query_id": str(row.market_evidence_query_id),
        "prospective_event_id": str(row.prospective_event_id),
        "query_type": str(row.query_type),
        "capture_date": day.isoformat(),
        "snapshots": [str(_snapshot_manifest_path(raw_root, item).resolve()) for item in snapshots],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot_manifest_path(raw_root: Path, snapshot: ShadowMarketSnapshot) -> Path:
    return (
        raw_root
        / "snapshots"
        / snapshot.market_evidence_query_id
        / snapshot.capture_date
        / snapshot.query_variant
        / f"{snapshot.page_number:03d}-{snapshot.sha256}.json"
    )


def _existing_snapshot(
    raw_root: Path, query_id: str, day: date, variant: str, page: int, requested_url: str
) -> ShadowMarketSnapshot | None:
    root = raw_root / "snapshots" / query_id / day.isoformat() / variant
    for path in root.glob(f"{page:03d}-*.json"):
        snapshot = ShadowMarketSnapshot(**_read_json(path))
        if snapshot.requested_url == requested_url:
            _validate_snapshot(snapshot)
            return snapshot
    return None


def _load_all_snapshots(raw_root: Path, query_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        for path in sorted((raw_root / "snapshots" / query_id).glob("*/*/*.json")):
            snapshot = ShadowMarketSnapshot(**_read_json(path))
            _validate_snapshot(snapshot)
            rows.append(asdict(snapshot))
    return pd.DataFrame(
        rows, columns=[field.name for field in ShadowMarketSnapshot.__dataclass_fields__.values()]
    )


def _validate_snapshot(snapshot: ShadowMarketSnapshot) -> None:
    path = Path(snapshot.path)
    if (
        not path.is_file()
        or path.stat().st_size != snapshot.bytes
        or _sha256_file(path) != snapshot.sha256
    ):
        raise ShadowMarketCollectionError(f"Missing or invalid raw market snapshot: {path}")


def _write_failure(raw_root: Path, query_id: str, reason: str, detail: str) -> None:
    path = (
        raw_root
        / "failures"
        / query_id
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reason": reason, "detail": detail}, indent=2), encoding="utf-8")


def _require_query_columns(queries: pd.DataFrame) -> None:
    required = {
        "market_evidence_query_id",
        "prospective_event_id",
        "cik",
        "ticker",
        "sec_accepted_at",
        "query_not_before",
        "query_type",
        "window_start",
        "window_end",
        "return_values_loaded",
        "live_trading_authorized",
    }
    missing = sorted(required.difference(queries.columns))
    if missing:
        raise ShadowMarketCollectionError(f"Market queries lack columns: {missing}")
    if not queries.empty and queries["market_evidence_query_id"].duplicated().any():
        raise ShadowMarketCollectionError("Market query IDs are duplicated")
    if queries["return_values_loaded"].fillna(False).astype(bool).any():
        raise ShadowMarketCollectionError("Return values entered market collection")
    if queries["live_trading_authorized"].fillna(False).astype(bool).any():
        raise ShadowMarketCollectionError("Live authorization entered market collection")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
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
