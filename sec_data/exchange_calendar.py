"""Auditable historical NYSE/Nasdaq sessions for filing-event windows."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
CALENDAR_RULE_VERSION: Final[str] = "us-equity-official-calendar-rules-v4-2022-2026-otcm"
SUPPORTED_EXCHANGES: Final[frozenset[str]] = frozenset({"XNYS", "XNAS", "XASE", "OTCM"})
OTC_SESSION_POLICY_SOURCE_ID: Final[str] = "otcmarkets-session-policy"


@dataclass(frozen=True)
class OfficialCalendarSource:
    """One official exchange calendar or exceptional-closure document."""

    source_id: str
    exchange_mic: str
    calendar_year: int
    source_role: str
    url: str
    media_type: str
    expected_sha256: str


@dataclass(frozen=True)
class CalendarSourceSnapshot:
    """Exact official source bytes plus retrieval metadata."""

    source_id: str
    exchange_mic: str
    calendar_year: int
    source_role: str
    requested_url: str
    response_url: str
    sha256: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    content_type: str


@dataclass(frozen=True)
class CalendarCollectionResult:
    """Counts from one resumable official-source collection."""

    selected_sources: int
    sources_completed: int
    requests_made: int
    snapshots_reused: int


@dataclass(frozen=True)
class ExchangeCalendarAuditResult:
    """Immutable historical-session and event-alignment audit."""

    output_root: Path
    report: dict[str, Any]


class CalendarHttpSender(Protocol):
    """Injectable HTTP sender for official calendar documents."""

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        """Return one HTTP response."""
        ...


def official_calendar_sources() -> tuple[OfficialCalendarSource, ...]:
    """Return the frozen official-document inventory for the pilot period."""
    return (
        OfficialCalendarSource(
            source_id="nyse-annual-2022",
            exchange_mic="XNYS",
            calendar_year=2022,
            source_role="annual_calendar",
            url="https://www.nyse.com/publicdocs/ICE_NYSE_2022_Yearly_Trading_Calendar.pdf",
            media_type="application/pdf",
            expected_sha256="2c2e63b780c4a373d8d546904ae46c8ba7e8070ad9357d23d2c92749985bb6c9",
        ),
        OfficialCalendarSource(
            source_id="nyse-annual-2023",
            exchange_mic="XNYS",
            calendar_year=2023,
            source_role="annual_calendar",
            url="https://www.nyse.com/publicdocs/ICE_NYSE_2023_Yearly_Trading_Calendar.pdf",
            media_type="application/pdf",
            expected_sha256="3d28b46a7197424c822087fb7117cdf3f303bd9a73f29660525b605f542bcfe2",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-annual-2022",
            exchange_mic="XNAS",
            calendar_year=2022,
            source_role="annual_calendar",
            url=("https://www.nasdaqtrader.com/content/technicalsupport/2022tradingcalendar.pdf"),
            media_type="application/pdf",
            expected_sha256="c886ec149e38d2cc84f33355c600d59d946a270fce6054323df147b4dd50b6ce",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-annual-2023",
            exchange_mic="XNAS",
            calendar_year=2023,
            source_role="annual_calendar",
            url=("https://www.nasdaqtrader.com/content/technicalsupport/2023tradingcalendar.pdf"),
            media_type="application/pdf",
            expected_sha256="3781aa7ec98e429b39c9b1cbdb3677043b8ba622084b8c9d5e10fd7836d9a13b",
        ),
        OfficialCalendarSource(
            source_id="nyse-annual-2024",
            exchange_mic="XNYS",
            calendar_year=2024,
            source_role="annual_calendar",
            url="https://www.nyse.com/publicdocs/ICE_NYSE_2024_Yearly_Trading_Calendar.pdf",
            media_type="application/pdf",
            expected_sha256="02282e733512d1f112230372efccfd18f81599c682af94bb6f21fd9bccbfd145",
        ),
        OfficialCalendarSource(
            source_id="nyse-annual-2025",
            exchange_mic="XNYS",
            calendar_year=2025,
            source_role="annual_calendar",
            url="https://www.nyse.com/publicdocs/ICE_NYSE_2025_Yearly_Trading_Calendar.pdf",
            media_type="application/pdf",
            expected_sha256="3e8b540ad74474bc12a0aba473dbca28f37e0a5c1795cf66701b9a506a34226e",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-annual-2024",
            exchange_mic="XNAS",
            calendar_year=2024,
            source_role="annual_calendar",
            url=("https://www.nasdaqtrader.com/content/technicalsupport/2024tradingcalendar.pdf"),
            media_type="application/pdf",
            expected_sha256="5a417265c706abe1e2eaa24e0c1fde1787e79fdd472eca105466a019f8ff1101",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-annual-2025",
            exchange_mic="XNAS",
            calendar_year=2025,
            source_role="annual_calendar",
            url=("https://www.nasdaqtrader.com/content/technicalsupport/2025tradingcalendar.pdf"),
            media_type="application/pdf",
            expected_sha256="fb761082d449ef76461d2ca8448ccebb79862a0acd124fad2e914469c9086037",
        ),
        OfficialCalendarSource(
            source_id="nyse-annual-2026",
            exchange_mic="XNYS",
            calendar_year=2026,
            source_role="annual_calendar",
            url=("https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf"),
            media_type="application/pdf",
            expected_sha256="70f5577eb43e60a9dbbecaae3cec23d0f02028c05c7f175013bb3e97816d394f",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-annual-2026",
            exchange_mic="XNAS",
            calendar_year=2026,
            source_role="annual_calendar",
            url="https://www.nasdaq.com/trading-calendar",
            media_type="application/pdf",
            expected_sha256="2c0caeb2513eed4990c01551fad9b0b0cd0049818ba5d561dbb61bdb017abeb8",
        ),
        OfficialCalendarSource(
            source_id=OTC_SESSION_POLICY_SOURCE_ID,
            exchange_mic="OTCM",
            calendar_year=0,
            source_role="venue_hours_and_holiday_policy",
            url=("https://www.otcmarkets.com/files/OTC%20Link%20ECN%20FIX%20Specification.pdf"),
            media_type="application/pdf",
            expected_sha256="04be3a90ca9ba6f39759902aa83dcd7db6ea2b9e9112cd557859ac4719e278dc",
        ),
        OfficialCalendarSource(
            source_id="nyse-carter-closure-2025",
            exchange_mic="XNYS",
            calendar_year=2025,
            source_role="exceptional_closure",
            url=(
                "https://www.nyse.com/publicdocs/nyse/markets/american-options/"
                "rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf"
            ),
            media_type="application/pdf",
            expected_sha256="a4640ce5466cfa262cfe387a20f97b3bbc7ee213a9ce819c4ac7d3717db493c7",
        ),
        OfficialCalendarSource(
            source_id="nasdaq-carter-closure-2025",
            exchange_mic="XNAS",
            calendar_year=2025,
            source_role="exceptional_closure",
            url="https://nasdaqtrader.com/TraderNews.aspx?id=ECA2024-632",
            media_type="text/html",
            expected_sha256="50ae14d3328b5ee54a509de043c37343d9c81006e2441fde63729c5fdc5d9215",
        ),
    )


def collect_official_calendar_sources(
    *,
    raw_root: Path = Path("data/market/exchange_calendar"),
    user_agent: str = "Filing-to-Signal-Lab/1.0",
    sender: CalendarHttpSender | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CalendarCollectionResult:
    """Download and preserve every official calendar source exactly once."""
    client = sender or httpx.Client(follow_redirects=True)
    headers = {
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.1",
        "User-Agent": user_agent.strip() or "Filing-to-Signal-Lab/1.0",
    }
    completed = 0
    requests_made = 0
    reused = 0
    for source in official_calendar_sources():
        existing = _load_source_completion(raw_root, source)
        if existing is not None:
            completed += 1
            reused += 1
            continue
        response = client.get(source.url, headers=headers, timeout=60.0)
        requests_made += 1
        if not 200 <= response.status_code < 300:
            raise FilingCatalogError(
                f"Official calendar source failed: {source.source_id} HTTP {response.status_code}"
            )
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not _media_type_matches(source.media_type, content_type, response.content):
            raise FilingCatalogError(
                f"Official calendar source has unexpected media type: {source.source_id}"
            )
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != source.expected_sha256:
            raise FilingCatalogError(
                f"Official calendar source changed and requires manual review: {source.source_id}"
            )
        snapshot = _preserve_source_response(
            raw_root,
            source,
            response,
            fetched_at=now(),
        )
        _write_source_completion(raw_root, snapshot)
        completed += 1
    return CalendarCollectionResult(
        selected_sources=len(official_calendar_sources()),
        sources_completed=completed,
        requests_made=requests_made,
        snapshots_reused=reused,
    )


def build_exchange_calendar_audit(
    price_root: Path,
    *,
    raw_root: Path = Path("data/market/exchange_calendar"),
    output_root: Path = Path("data/processed/exchange_calendar"),
) -> ExchangeCalendarAuditResult:
    """Build official sessions and align every observed price-event window."""
    price_report = json.loads(
        (price_root / "massive_price_report.json").read_text(encoding="utf-8")
    )
    price_manifest_sha256 = _sha256_file(price_root / "output_manifest.json")
    targets = load_catalog_table(price_root, "massive_daily_price_targets.parquet")
    bars = load_catalog_table(price_root, "massive_daily_price_observations.parquet")
    price_audit = load_catalog_table(price_root, "sample_massive_price_audit.parquet")
    sources = _load_all_official_sources(raw_root)
    required_exchanges = set(targets["selected_exchange_mic"].astype(str))
    unsupported = sorted(required_exchanges.difference(SUPPORTED_EXCHANGES))
    if unsupported:
        raise FilingCatalogError(f"Unsupported exchange MICs in price targets: {unsupported}")
    start = date.fromisoformat(str(targets["price_window_start"].min()))
    end = date.fromisoformat(str(targets["price_window_end"].max()))
    sessions, exceptions = _build_official_sessions(start, end, sources, required_exchanges)
    alignment = _align_events_to_sessions(targets, bars, price_audit, sessions)
    source_hashes = sorted(sources["sha256"].astype(str).tolist())
    payload = json.dumps(
        {
            "price_manifest_sha256": price_manifest_sha256,
            "source_hashes": source_hashes,
            "rule_version": CALENDAR_RULE_VERSION,
            "calendar_start": start.isoformat(),
            "calendar_end": end.isoformat(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = "exchange-calendar-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        existing_report = json.loads(
            (target / "exchange_calendar_report.json").read_text(encoding="utf-8")
        )
        return ExchangeCalendarAuditResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete exchange-calendar directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs: dict[str, pd.DataFrame] = {
        "official_calendar_sources.parquet": sources,
        "historical_exchange_sessions.parquet": sessions,
        "exchange_calendar_exceptions.parquet": exceptions,
        "event_calendar_alignment.parquet": alignment,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    ready = alignment["formal_event_window_ready"].astype(bool)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "price_version": price_report["dataset_version"],
        "price_manifest_sha256": price_manifest_sha256,
        "rule_version": CALENDAR_RULE_VERSION,
        "construction_method": "explicit_rules_transcribed_from_preserved_official_documents",
        "timezone": "America/New_York",
        "calendar_start": start.isoformat(),
        "calendar_end": end.isoformat(),
        "official_sources_required": len(official_calendar_sources()),
        "official_sources_preserved": len(sources),
        "exchange_mics": sorted(required_exchanges),
        "exchange_session_rows": len(sessions),
        "early_close_session_rows": int(sessions["is_early_close"].astype(bool).sum()),
        "closed_date_exception_rows": int(exceptions["exception_type"].eq("closed").sum()),
        "price_targets": len(targets),
        "calendar_aligned_events": int(alignment["calendar_window_resolved"].sum()),
        "formal_event_window_ready": int(ready.sum()),
        "missing_entry_prices": int((~alignment["entry_price_observed"]).sum()),
        "missing_exit_prices": int((~alignment["exit_price_observed"]).sum()),
        "entry_date_changes_vs_observed_bars": int(
            alignment["entry_date_changed_from_observed_candidate"].sum()
        ),
        "exit_date_changes_vs_observed_bars": int(
            alignment["exit_date_changed_from_observed_candidate"].sum()
        ),
        "acceptance_timing_counts": {
            str(key): int(value)
            for key, value in alignment["acceptance_session_relation"].value_counts().items()
        },
        "official_source_gate": len(sources) == len(official_calendar_sources()),
        "pilot_exchange_calendar_gate": bool(ready.all()),
        "stable_security_interval_gate": False,
        "corporate_action_gate": False,
        "delisting_outcome_gate": False,
        "return_study_gate": False,
        "returns_computed": False,
        "current_ticker_join_used": False,
        "artifacts": artifacts,
    }
    report_path = staging / "exchange_calendar_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "price_manifest_sha256": price_manifest_sha256,
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
    return ExchangeCalendarAuditResult(target.resolve(), report)


def load_preserved_official_calendar_sources(
    raw_root: Path = Path("data/market/exchange_calendar"),
) -> pd.DataFrame:
    """Load every hash-verified official calendar source snapshot."""
    return _load_all_official_sources(raw_root)


def build_historical_exchange_sessions(
    start: date,
    end: date,
    sources: pd.DataFrame,
    exchanges: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct timestamped sessions from preserved official source evidence."""
    unsupported = sorted(exchanges.difference(SUPPORTED_EXCHANGES))
    if unsupported:
        raise FilingCatalogError(f"Unsupported exchange MICs: {unsupported}")
    if start > end:
        raise FilingCatalogError("Exchange calendar start follows its end")
    return _build_official_sessions(start, end, sources, exchanges)


def _build_official_sessions(
    start: date,
    end: date,
    sources: pd.DataFrame,
    exchanges: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_lookup = sources.set_index("source_id")["sha256"].astype(str).to_dict()
    otc_policy_hash = source_lookup.get(OTC_SESSION_POLICY_SOURCE_ID)
    if "OTCM" in exchanges and otc_policy_hash is None:
        raise FilingCatalogError("Missing OTC Markets session-policy source")
    session_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    for exchange_mic in sorted(exchanges):
        for calendar_year in range(start.year, end.year + 1):
            annual_id = _annual_source_id(exchange_mic, calendar_year)
            annual_hash = source_lookup.get(annual_id)
            if annual_hash is None:
                raise FilingCatalogError(f"Missing annual calendar source: {annual_id}")
            closed = _closed_dates(calendar_year)
            early = _early_close_dates(calendar_year)
            for session_date, description in sorted(closed.items()):
                if not start <= session_date <= end:
                    continue
                source_id = (
                    _carter_source_id(exchange_mic)
                    if session_date == date(2025, 1, 9)
                    else annual_id
                )
                source_hash = source_lookup.get(source_id)
                if source_hash is None:
                    raise FilingCatalogError(f"Missing exception source: {source_id}")
                exception_rows.append(
                    {
                        "exchange_mic": exchange_mic,
                        "exception_date": session_date.isoformat(),
                        "exception_type": "closed",
                        "description": description,
                        "source_id": source_id,
                        "source_snapshot_sha256": source_hash,
                        "schedule_basis_exchange_mic": _schedule_basis_exchange(exchange_mic),
                        "venue_policy_source_id": (
                            OTC_SESSION_POLICY_SOURCE_ID if exchange_mic == "OTCM" else None
                        ),
                        "venue_policy_source_sha256": (
                            otc_policy_hash if exchange_mic == "OTCM" else None
                        ),
                    }
                )
            for session_date, description in sorted(early.items()):
                if not start <= session_date <= end:
                    continue
                exception_rows.append(
                    {
                        "exchange_mic": exchange_mic,
                        "exception_date": session_date.isoformat(),
                        "exception_type": "early_close_13_et",
                        "description": description,
                        "source_id": annual_id,
                        "source_snapshot_sha256": annual_hash,
                        "schedule_basis_exchange_mic": _schedule_basis_exchange(exchange_mic),
                        "venue_policy_source_id": (
                            OTC_SESSION_POLICY_SOURCE_ID if exchange_mic == "OTCM" else None
                        ),
                        "venue_policy_source_sha256": (
                            otc_policy_hash if exchange_mic == "OTCM" else None
                        ),
                    }
                )
            year_start = max(start, date(calendar_year, 1, 1))
            year_end = min(end, date(calendar_year, 12, 31))
            for timestamp in pd.date_range(year_start, year_end, freq="D"):
                session_date = timestamp.date()
                if session_date.weekday() >= 5 or session_date in closed:
                    continue
                close_time = time(13, 0) if session_date in early else time(16, 0)
                open_at = datetime.combine(session_date, time(9, 30), tzinfo=NEW_YORK)
                close_at = datetime.combine(session_date, close_time, tzinfo=NEW_YORK)
                session_rows.append(
                    {
                        "exchange_mic": exchange_mic,
                        "session_date": session_date.isoformat(),
                        "regular_open_at": open_at.astimezone(timezone.utc).isoformat(),
                        "regular_close_at": close_at.astimezone(timezone.utc).isoformat(),
                        "is_early_close": session_date in early,
                        "calendar_timezone": "America/New_York",
                        "source_id": annual_id,
                        "source_snapshot_sha256": annual_hash,
                        "schedule_basis_exchange_mic": _schedule_basis_exchange(exchange_mic),
                        "venue_policy_source_id": (
                            OTC_SESSION_POLICY_SOURCE_ID if exchange_mic == "OTCM" else None
                        ),
                        "venue_policy_source_sha256": (
                            otc_policy_hash if exchange_mic == "OTCM" else None
                        ),
                        "rule_version": CALENDAR_RULE_VERSION,
                    }
                )
    sessions = pd.DataFrame(session_rows, columns=_session_columns())
    exceptions = pd.DataFrame(exception_rows, columns=_exception_columns())
    if sessions.duplicated(["exchange_mic", "session_date"]).any():
        raise FilingCatalogError("Exchange calendar contains duplicate sessions")
    return sessions, exceptions


def _align_events_to_sessions(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    price_audit: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    observed = price_audit[
        [
            "price_target_id",
            "candidate_entry_session_date",
            "candidate_exit_session_date",
        ]
    ]
    targets = targets.merge(observed, on="price_target_id", how="left", validate="one_to_one")
    session_frame = sessions.copy()
    session_frame["regular_open_value"] = pd.to_datetime(session_frame["regular_open_at"], utc=True)
    session_frame["regular_close_value"] = pd.to_datetime(
        session_frame["regular_close_at"], utc=True
    )
    bar_lookup = bars.set_index(["price_target_id", "session_date"])
    if not bar_lookup.index.is_unique:
        raise FilingCatalogError("Price observations contain duplicate target-session rows")
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        accepted_at = pd.Timestamp(str(target.accepted_at))
        exchange_sessions = session_frame[
            session_frame["exchange_mic"].eq(str(target.selected_exchange_mic))
        ].sort_values("regular_open_value")
        event_date = accepted_at.date().isoformat()
        event_session = exchange_sessions["session_date"].eq(event_date)
        if not event_session.any():
            relation = "accepted_on_non_session"
        else:
            same_day = exchange_sessions.loc[event_session].iloc[0]
            if accepted_at < same_day["regular_open_value"]:
                relation = "accepted_before_regular_open"
            elif accepted_at <= same_day["regular_close_value"]:
                relation = "accepted_during_regular_session"
            else:
                relation = "accepted_after_regular_close"
        future = exchange_sessions[
            exchange_sessions["regular_open_value"] > accepted_at
        ].reset_index(drop=True)
        entry_index = int(str(target.entry_session_offset)) - 1
        exit_index = entry_index + int(str(target.holding_sessions)) - 1
        calendar_resolved = len(future) > exit_index
        entry_date: str | None = None
        exit_date: str | None = None
        entry_open_at: str | None = None
        exit_close_at: str | None = None
        entry_source_hash: str | None = None
        exit_source_hash: str | None = None
        if calendar_resolved:
            entry_session = future.iloc[entry_index]
            exit_session = future.iloc[exit_index]
            entry_date = str(entry_session["session_date"])
            exit_date = str(exit_session["session_date"])
            entry_open_at = str(entry_session["regular_open_at"])
            exit_close_at = str(exit_session["regular_close_at"])
            entry_source_hash = str(entry_session["source_snapshot_sha256"])
            exit_source_hash = str(exit_session["source_snapshot_sha256"])
        entry_key = (str(target.price_target_id), str(entry_date))
        exit_key = (str(target.price_target_id), str(exit_date))
        entry_observed = calendar_resolved and entry_key in bar_lookup.index
        exit_observed = calendar_resolved and exit_key in bar_lookup.index
        entry_price = (
            _numeric_scalar(bar_lookup.at[entry_key, "open_unadjusted"])
            if entry_observed
            else np.nan
        )
        exit_price = (
            _numeric_scalar(bar_lookup.at[exit_key, "close_unadjusted"])
            if exit_observed
            else np.nan
        )
        observed_entry = _optional_string(target.candidate_entry_session_date)
        observed_exit = _optional_string(target.candidate_exit_session_date)
        ready = bool(calendar_resolved and entry_observed and exit_observed)
        rows.append(
            {
                "price_target_id": target.price_target_id,
                "target_id": target.target_id,
                "cik": target.cik,
                "accepted_at": target.accepted_at,
                "ticker": target.selected_ticker,
                "exchange_mic": target.selected_exchange_mic,
                "security_id_candidate": target.selected_provider_security_id,
                "acceptance_session_relation": relation,
                "entry_session_offset": target.entry_session_offset,
                "holding_sessions": target.holding_sessions,
                "calendar_window_resolved": calendar_resolved,
                "formal_entry_session_date": entry_date,
                "formal_entry_open_at": entry_open_at,
                "formal_entry_open_unadjusted": entry_price,
                "entry_price_observed": entry_observed,
                "entry_calendar_source_sha256": entry_source_hash,
                "formal_exit_session_date": exit_date,
                "formal_exit_close_at": exit_close_at,
                "formal_exit_close_unadjusted": exit_price,
                "exit_price_observed": exit_observed,
                "exit_calendar_source_sha256": exit_source_hash,
                "observed_candidate_entry_session_date": observed_entry,
                "observed_candidate_exit_session_date": observed_exit,
                "entry_date_changed_from_observed_candidate": entry_date != observed_entry,
                "exit_date_changed_from_observed_candidate": exit_date != observed_exit,
                "formal_event_window_ready": ready,
                "alignment_status": _alignment_status(
                    calendar_resolved, entry_observed, exit_observed
                ),
            }
        )
    return pd.DataFrame(rows, columns=_alignment_columns())


def _closed_dates(calendar_year: int) -> dict[date, str]:
    values: dict[int, tuple[tuple[str, str], ...]] = {
        2022: (
            ("2022-01-17", "Martin Luther King Jr. Day"),
            ("2022-02-21", "Washington's Birthday"),
            ("2022-04-15", "Good Friday"),
            ("2022-05-30", "Memorial Day"),
            ("2022-06-20", "Juneteenth National Independence Day observed"),
            ("2022-07-04", "Independence Day"),
            ("2022-09-05", "Labor Day"),
            ("2022-11-24", "Thanksgiving Day"),
            ("2022-12-26", "Christmas Day observed"),
        ),
        2023: (
            ("2023-01-02", "New Year's Day observed"),
            ("2023-01-16", "Martin Luther King Jr. Day"),
            ("2023-02-20", "Washington's Birthday"),
            ("2023-04-07", "Good Friday"),
            ("2023-05-29", "Memorial Day"),
            ("2023-06-19", "Juneteenth National Independence Day"),
            ("2023-07-04", "Independence Day"),
            ("2023-09-04", "Labor Day"),
            ("2023-11-23", "Thanksgiving Day"),
            ("2023-12-25", "Christmas Day"),
        ),
        2024: (
            ("2024-01-01", "New Year's Day"),
            ("2024-01-15", "Martin Luther King Jr. Day"),
            ("2024-02-19", "Washington's Birthday"),
            ("2024-03-29", "Good Friday"),
            ("2024-05-27", "Memorial Day"),
            ("2024-06-19", "Juneteenth National Independence Day"),
            ("2024-07-04", "Independence Day"),
            ("2024-09-02", "Labor Day"),
            ("2024-11-28", "Thanksgiving Day"),
            ("2024-12-25", "Christmas Day"),
        ),
        2025: (
            ("2025-01-01", "New Year's Day"),
            ("2025-01-09", "National Day of Mourning for President Jimmy Carter"),
            ("2025-01-20", "Martin Luther King Jr. Day"),
            ("2025-02-17", "Washington's Birthday"),
            ("2025-04-18", "Good Friday"),
            ("2025-05-26", "Memorial Day"),
            ("2025-06-19", "Juneteenth National Independence Day"),
            ("2025-07-04", "Independence Day"),
            ("2025-09-01", "Labor Day"),
            ("2025-11-27", "Thanksgiving Day"),
            ("2025-12-25", "Christmas Day"),
        ),
        2026: (
            ("2026-01-01", "New Year's Day"),
            ("2026-01-19", "Martin Luther King Jr. Day"),
            ("2026-02-16", "Washington's Birthday"),
            ("2026-04-03", "Good Friday"),
            ("2026-05-25", "Memorial Day"),
            ("2026-06-19", "Juneteenth National Independence Day"),
            ("2026-07-03", "Independence Day observed"),
            ("2026-09-07", "Labor Day"),
            ("2026-11-26", "Thanksgiving Day"),
            ("2026-12-25", "Christmas Day"),
        ),
    }
    try:
        return {date.fromisoformat(value): label for value, label in values[calendar_year]}
    except KeyError as exc:
        raise FilingCatalogError(f"No official calendar transcription for {calendar_year}") from exc


def _early_close_dates(calendar_year: int) -> dict[date, str]:
    values: dict[int, tuple[tuple[str, str], ...]] = {
        2022: (("2022-11-25", "Day after Thanksgiving early close"),),
        2023: (
            ("2023-07-03", "Independence Day early close"),
            ("2023-11-24", "Day after Thanksgiving early close"),
        ),
        2024: (
            ("2024-07-03", "Independence Day early close"),
            ("2024-11-29", "Day after Thanksgiving early close"),
            ("2024-12-24", "Christmas Eve early close"),
        ),
        2025: (
            ("2025-07-03", "Independence Day early close"),
            ("2025-11-28", "Day after Thanksgiving early close"),
            ("2025-12-24", "Christmas Eve early close"),
        ),
        2026: (
            ("2026-11-27", "Day after Thanksgiving early close"),
            ("2026-12-24", "Christmas Eve early close"),
        ),
    }
    try:
        return {date.fromisoformat(value): label for value, label in values[calendar_year]}
    except KeyError as exc:
        raise FilingCatalogError(
            f"No official early-close transcription for {calendar_year}"
        ) from exc


def _annual_source_id(exchange_mic: str, calendar_year: int) -> str:
    prefix = "nyse" if exchange_mic in {"XNYS", "XASE"} else "nasdaq"
    return f"{prefix}-annual-{calendar_year}"


def _schedule_basis_exchange(exchange_mic: str) -> str:
    if exchange_mic in {"XNYS", "XASE"}:
        return "XNYS"
    return "XNAS"


def _carter_source_id(exchange_mic: str) -> str:
    prefix = "nyse" if exchange_mic in {"XNYS", "XASE"} else "nasdaq"
    return f"{prefix}-carter-closure-2025"


def _preserve_source_response(
    raw_root: Path,
    source: OfficialCalendarSource,
    response: httpx.Response,
    *,
    fetched_at: datetime,
) -> CalendarSourceSnapshot:
    digest = hashlib.sha256(response.content).hexdigest()
    suffix = ".pdf" if source.media_type == "application/pdf" else ".html"
    directory = raw_root / "raw" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}{suffix}"
    if path.exists() and _sha256_file(path) != digest:
        raise FilingCatalogError(f"Official calendar raw hash mismatch: {path}")
    if not path.exists():
        temporary = directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise FilingCatalogError("Official calendar bytes changed before preservation")
        temporary.replace(path)
    return CalendarSourceSnapshot(
        source_id=source.source_id,
        exchange_mic=source.exchange_mic,
        calendar_year=source.calendar_year,
        source_role=source.source_role,
        requested_url=source.url,
        response_url=str(response.url),
        sha256=digest,
        fetched_at=fetched_at.isoformat(),
        path=str(path.resolve()),
        bytes=len(response.content),
        status_code=response.status_code,
        content_type=response.headers.get("content-type", "").split(";")[0].strip().lower(),
    )


def _write_source_completion(raw_root: Path, snapshot: CalendarSourceSnapshot) -> None:
    path = raw_root / "sources" / f"{snapshot.source_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8")


def _load_source_completion(
    raw_root: Path, source: OfficialCalendarSource
) -> CalendarSourceSnapshot | None:
    path = raw_root / "sources" / f"{source.source_id}.json"
    if not path.is_file():
        return None
    snapshot = CalendarSourceSnapshot(**json.loads(path.read_text(encoding="utf-8")))
    raw_path = Path(snapshot.path)
    if snapshot.requested_url != source.url:
        raise FilingCatalogError(f"Official source URL changed: {source.source_id}")
    if snapshot.sha256 != source.expected_sha256:
        raise FilingCatalogError(f"Official source hash is not reviewed: {source.source_id}")
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != snapshot.bytes
        or _sha256_file(raw_path) != snapshot.sha256
    ):
        raise FilingCatalogError(f"Official calendar snapshot is invalid: {source.source_id}")
    return snapshot


def _load_all_official_sources(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in official_calendar_sources():
        snapshot = _load_source_completion(raw_root, source)
        if snapshot is None:
            raise FilingCatalogError(
                f"Official calendar source is not collected: {source.source_id}"
            )
        rows.append(asdict(snapshot))
    return pd.DataFrame(rows)


def _media_type_matches(expected: str, actual: str, content: bytes) -> bool:
    if expected == "application/pdf":
        return content.startswith(b"%PDF-") and actual in {
            "application/pdf",
            "application/octet-stream",
            "binary/octet-stream",
        }
    return expected == "text/html" and (actual == "text/html" or b"<html" in content[:1000].lower())


def _optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _numeric_scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise FilingCatalogError("Price lookup did not return one numeric scalar")
    return float(value)


def _alignment_status(calendar_resolved: bool, entry: bool, exit_price: bool) -> str:
    if not calendar_resolved:
        return "calendar_window_not_resolved"
    if not entry and not exit_price:
        return "entry_and_exit_prices_missing"
    if not entry:
        return "entry_price_missing"
    if not exit_price:
        return "exit_price_missing"
    return "formal_window_prices_observed"


def _session_columns() -> list[str]:
    return [
        "exchange_mic",
        "session_date",
        "regular_open_at",
        "regular_close_at",
        "is_early_close",
        "calendar_timezone",
        "source_id",
        "source_snapshot_sha256",
        "schedule_basis_exchange_mic",
        "venue_policy_source_id",
        "venue_policy_source_sha256",
        "rule_version",
    ]


def _exception_columns() -> list[str]:
    return [
        "exchange_mic",
        "exception_date",
        "exception_type",
        "description",
        "source_id",
        "source_snapshot_sha256",
        "schedule_basis_exchange_mic",
        "venue_policy_source_id",
        "venue_policy_source_sha256",
    ]


def _alignment_columns() -> list[str]:
    return [
        "price_target_id",
        "target_id",
        "cik",
        "accepted_at",
        "ticker",
        "exchange_mic",
        "security_id_candidate",
        "acceptance_session_relation",
        "entry_session_offset",
        "holding_sessions",
        "calendar_window_resolved",
        "formal_entry_session_date",
        "formal_entry_open_at",
        "formal_entry_open_unadjusted",
        "entry_price_observed",
        "entry_calendar_source_sha256",
        "formal_exit_session_date",
        "formal_exit_close_at",
        "formal_exit_close_unadjusted",
        "exit_price_observed",
        "exit_calendar_source_sha256",
        "observed_candidate_entry_session_date",
        "observed_candidate_exit_session_date",
        "entry_date_changed_from_observed_candidate",
        "exit_date_changed_from_observed_candidate",
        "formal_event_window_ready",
        "alignment_status",
    ]


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
