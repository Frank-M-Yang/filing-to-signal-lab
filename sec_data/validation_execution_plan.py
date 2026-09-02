"""Outcome-blind execution windows and market-evidence targets for validation.

The module converts filing acceptance timestamps and already-audited historical
security identities into explicit, tradable windows.  It does not load prices,
corporate actions, listing outcomes, or returns.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.exchange_calendar import (
    CALENDAR_RULE_VERSION,
    build_historical_exchange_sessions,
    load_preserved_official_calendar_sources,
)
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_EXECUTION_PLAN_VERSION: Final[str] = (
    "validation-execution-plan-v2-transition-aware-delisting"
)
FORBIDDEN_OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "sector_excess_return",
        "future_return",
        "delisting_return",
    }
)


class ValidationExecutionPlanError(FilingCatalogError):
    """Raised when a formal execution plan violates its evidence contract."""


@dataclass(frozen=True)
class ValidationExecutionPlanConfig:
    """Frozen timing, query-context, and outcome-cutoff assumptions."""

    entry_session_offset: int = 2
    holding_sessions: int = 20
    price_context_sessions: int = 2
    delisting_outcome_cutoff_date: str = "2026-08-14"
    entry_price_type: str = "unadjusted_open"
    exit_price_type: str = "unadjusted_close"
    version: str = VALIDATION_EXECUTION_PLAN_VERSION

    def validate(self) -> None:
        """Reject timing rules that depart from the frozen research design."""
        if self.entry_session_offset != 2:
            raise ValueError("Validation entry is frozen to the second future session open")
        if self.holding_sessions != 20:
            raise ValueError("Validation holding period is frozen to twenty sessions")
        if self.price_context_sessions < 1:
            raise ValueError("Price targets require at least one context session")
        cutoff = date.fromisoformat(self.delisting_outcome_cutoff_date)
        if cutoff > date(2026, 8, 14):
            raise ValueError("Delisting cutoff exceeds the frozen outcome as-of date")
        if self.entry_price_type != "unadjusted_open":
            raise ValueError("Entry must use an unadjusted open before action processing")
        if self.exit_price_type != "unadjusted_close":
            raise ValueError("Exit must use an unadjusted close before action processing")


@dataclass(frozen=True)
class ValidationExecutionPlanTables:
    """Pure formal windows and evidence targets before persistence."""

    events: pd.DataFrame
    security_components: pd.DataFrame
    price_targets: pd.DataFrame
    action_targets: pd.DataFrame
    delisting_targets: pd.DataFrame
    benchmark_targets: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationExecutionPlanResult:
    """Immutable outcome-blind validation execution plan."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_execution_plan(
    market_events: pd.DataFrame,
    identity_events: pd.DataFrame,
    identity_repair_components: pd.DataFrame,
    sessions: pd.DataFrame,
    benchmark_plan: pd.DataFrame,
    config: ValidationExecutionPlanConfig | None = None,
) -> ValidationExecutionPlanTables:
    """Calculate formal windows and deduplicated evidence queries without outcomes."""
    cfg = config or ValidationExecutionPlanConfig()
    cfg.validate()
    for frame, label in (
        (market_events, "market events"),
        (identity_events, "identity events"),
        (identity_repair_components, "identity repair components"),
        (sessions, "exchange sessions"),
        (benchmark_plan, "benchmark plan"),
    ):
        forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(frame.columns))
        if forbidden:
            raise ValidationExecutionPlanError(f"{label} contains outcomes: {forbidden}")
    _require_columns(
        market_events,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "accepted_at",
            "form",
            "target_accession_number",
            "price_collection_candidate",
            "selection_uses_return_values",
        },
        "market events",
    )
    _require_columns(
        identity_events,
        {
            "market_event_id",
            "validation_event_id",
            "cik",
            "event_security_identity_ready",
            "identity_assignment_method",
            "assigned_ticker",
            "assigned_exchange_mic",
            "assigned_provider_security_id",
            "assigned_provider_security_id_kind",
            "security_component_count",
            "issuer_return_construction",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "identity events",
    )
    _require_columns(
        sessions,
        {
            "exchange_mic",
            "session_date",
            "regular_open_at",
            "regular_close_at",
            "source_id",
            "source_snapshot_sha256",
            "schedule_basis_exchange_mic",
            "rule_version",
        },
        "exchange sessions",
    )
    candidates = market_events.loc[
        market_events["price_collection_candidate"].fillna(False).astype(bool)
    ].copy()
    identities = identity_events.copy()
    if candidates["selection_uses_return_values"].astype(bool).any():
        raise ValidationExecutionPlanError("Market-event selection used return values")
    if identities["selection_uses_return_values"].astype(bool).any():
        raise ValidationExecutionPlanError("Identity assignment used return values")
    if identities["outcome_values_loaded"].astype(bool).any():
        raise ValidationExecutionPlanError("Identity assignment opened outcome values")
    if not identities["event_security_identity_ready"].astype(bool).all():
        raise ValidationExecutionPlanError("Every validation event requires audited identity")
    events = candidates.merge(
        identities[
            [
                "market_event_id",
                "identity_assignment_method",
                "assigned_ticker",
                "assigned_exchange_mic",
                "assigned_provider_security_id",
                "assigned_provider_security_id_kind",
                "security_component_count",
                "issuer_return_construction",
                "identity_repair_status",
            ]
        ],
        on="market_event_id",
        how="inner",
        validate="one_to_one",
    )
    if len(events) != len(identities):
        raise ValidationExecutionPlanError("Market and identity event sets do not match")
    events["cik"] = events["cik"].astype(str).str.zfill(10)
    events["accepted_value"] = pd.to_datetime(events["accepted_at"], utc=True, errors="coerce")
    if events["accepted_value"].isna().any():
        raise ValidationExecutionPlanError("A validation event has an invalid acceptance clock")
    windows = _calculate_event_windows(events, sessions, cfg)
    events = events.merge(windows, on="market_event_id", how="left", validate="one_to_one")
    events["price_collection_authorized"] = events["formal_execution_window_ready"]
    events["return_outcome_read_authorized"] = False
    events["selection_uses_return_values"] = False
    events["outcome_values_loaded"] = False
    components = _expand_security_components(events, identity_repair_components)
    components = _attach_query_context(components, sessions, cfg.price_context_sessions)
    price_targets = _build_security_spans(components, cfg)
    action_targets = _build_action_targets(price_targets)
    delisting_targets = _build_delisting_targets(price_targets, cfg)
    benchmark_targets = _build_benchmark_targets(events, benchmark_plan, sessions, cfg)
    event_ready = events["formal_execution_window_ready"].astype(bool)
    component_ready = components["security_component_execution_ready"].astype(bool)
    price_ready = price_targets["price_collection_authorized"].astype(bool)
    gates = pd.DataFrame(
        [
            _gate(
                "event_security_identity",
                int(identities["event_security_identity_ready"].sum()),
                len(identities),
                bool(identities["event_security_identity_ready"].astype(bool).all()),
                "Every price-candidate event has an outcome-blind historical identity",
            ),
            _gate(
                "formal_execution_calendar",
                int(event_ready.sum()),
                len(events),
                bool(event_ready.all()),
                "Entry and exit have explicit official-session timestamps and price types",
            ),
            _gate(
                "security_component_queries",
                int(component_ready.sum()),
                len(components),
                bool(component_ready.all()),
                "Every issuer event resolves to one or more fixed-weight security components",
            ),
            _gate(
                "price_collection",
                int(price_ready.sum()),
                len(price_targets),
                bool(price_ready.all()),
                "Deduplicated unadjusted price spans are authorized for collection",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(events),
                False,
                "Locked until prices, actions, listing outcomes, and benchmarks pass audits",
            ),
        ]
    )
    report: dict[str, Any] = {
        "validation_events": len(events),
        "validation_issuers": int(events["cik"].nunique()),
        "formal_execution_windows_ready": int(event_ready.sum()),
        "formal_execution_window_coverage": float(event_ready.mean()),
        "security_event_components": len(components),
        "distinct_security_spans": len(price_targets),
        "multiclass_issuer_events": int(events["security_component_count"].gt(1).sum()),
        "price_targets": len(price_targets),
        "action_targets": len(action_targets),
        "delisting_targets": len(delisting_targets),
        "broad_market_benchmark_targets": len(benchmark_targets),
        "entry_rule": "second_official_trading_session_open_strictly_after_acceptance",
        "exit_rule": "twentieth_official_trading_session_close_including_entry_session",
        "entry_price_type": cfg.entry_price_type,
        "exit_price_type": cfg.exit_price_type,
        "calendar_rule_version": CALENDAR_RULE_VERSION,
        "acceptance_timing_counts": {
            str(key): int(value)
            for key, value in events["acceptance_session_relation"].value_counts().items()
        },
        "exchange_event_counts": {
            str(key): int(value)
            for key, value in events["assigned_exchange_mic"].value_counts().items()
        },
        "otcm_calendar_method": (
            "OTC Link regular DAY session timestamps with Nasdaq US-equity holiday dates; "
            "both source classes are hash preserved"
        ),
        "selection_uses_provider_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_collection_authorized": bool(event_ready.all() and component_ready.all()),
        "corporate_action_collection_authorized": bool(event_ready.all() and component_ready.all()),
        "delisting_evidence_collection_authorized": bool(
            event_ready.all() and component_ready.all()
        ),
        "return_outcome_read_authorized": False,
    }
    return ValidationExecutionPlanTables(
        events=events.drop(columns=["accepted_value"]),
        security_components=components,
        price_targets=price_targets,
        action_targets=action_targets,
        delisting_targets=delisting_targets,
        benchmark_targets=benchmark_targets,
        gates=gates,
        report=report,
    )


def build_validation_execution_plan(
    market_plan_root: Path,
    identity_resolution_root: Path,
    *,
    raw_calendar_root: Path = Path("data/market/exchange_calendar"),
    output_root: Path = Path("data/processed/validation_execution_plan"),
    config: ValidationExecutionPlanConfig | None = None,
) -> ValidationExecutionPlanResult:
    """Build an immutable formal-window and market-evidence acquisition plan."""
    cfg = config or ValidationExecutionPlanConfig()
    cfg.validate()
    market_manifest = _sha256_file(market_plan_root / "output_manifest.json")
    identity_manifest = _sha256_file(identity_resolution_root / "output_manifest.json")
    identity_report = _read_json(identity_resolution_root / "identity_repair_report.json")
    expected_market_manifest = identity_report.get("input_manifest_sha256", {}).get("market_plan")
    if expected_market_manifest != market_manifest:
        raise ValidationExecutionPlanError("Identity resolution does not tie to market plan")
    if bool(identity_report.get("outcome_values_loaded")):
        raise ValidationExecutionPlanError("Identity resolution opened outcome values")
    market_events = load_catalog_table(market_plan_root, "validation_market_event_targets.parquet")
    benchmark_plan = load_catalog_table(market_plan_root, "validation_benchmark_plan.parquet")
    identity_events = load_catalog_table(
        identity_resolution_root, "validation_event_identity_readiness_final.parquet"
    )
    repair_components = load_catalog_table(
        identity_resolution_root, "validation_identity_repair_components.parquet"
    )
    candidates = market_events.loc[
        market_events["price_collection_candidate"].fillna(False).astype(bool)
    ]
    accepted = pd.to_datetime(candidates["accepted_at"], utc=True, errors="coerce")
    if accepted.isna().any():
        raise ValidationExecutionPlanError("Market plan contains invalid acceptance clocks")
    calendar_start = accepted.min().date() - timedelta(days=10)
    calendar_end = accepted.max().date() + timedelta(days=70)
    sources = load_preserved_official_calendar_sources(raw_calendar_root)
    exchanges = set(identity_events["assigned_exchange_mic"].dropna().astype(str))
    sessions, exceptions = build_historical_exchange_sessions(
        calendar_start, calendar_end, sources, exchanges
    )
    tables = calculate_validation_execution_plan(
        market_events,
        identity_events,
        repair_components,
        sessions,
        benchmark_plan,
        cfg,
    )
    source_hashes = sorted(sources["sha256"].astype(str).tolist())
    payload = json.dumps(
        {
            "market_manifest_sha256": market_manifest,
            "identity_manifest_sha256": identity_manifest,
            "calendar_source_hashes": source_hashes,
            "calendar_rule_version": CALENDAR_RULE_VERSION,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationExecutionPlanResult(
            target.resolve(), _read_json(target / "validation_execution_plan_report.json")
        )
    if target.exists():
        raise ValidationExecutionPlanError(f"Incomplete execution plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_execution_events.parquet": tables.events,
        "validation_security_event_components.parquet": tables.security_components,
        "validation_security_price_targets.parquet": tables.price_targets,
        "validation_security_action_targets.parquet": tables.action_targets,
        "validation_security_delisting_targets.parquet": tables.delisting_targets,
        "validation_benchmark_price_targets.parquet": tables.benchmark_targets,
        "validation_execution_gates.parquet": tables.gates,
        "validation_official_calendar_sources.parquet": sources,
        "validation_exchange_sessions.parquet": sessions,
        "validation_exchange_calendar_exceptions.parquet": exceptions,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_plan_manifest_sha256": market_manifest,
        "identity_resolution_manifest_sha256": identity_manifest,
        "calendar_source_hashes": source_hashes,
        "calendar_start": calendar_start.isoformat(),
        "calendar_end": calendar_end.isoformat(),
        "config": asdict(cfg),
        "artifacts": artifacts,
    }
    report_path = staging / "validation_execution_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "market_plan_manifest_sha256": market_manifest,
                "identity_resolution_manifest_sha256": identity_manifest,
                "calendar_source_hashes": source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationExecutionPlanResult(target.resolve(), report)


def _calculate_event_windows(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    cfg: ValidationExecutionPlanConfig,
) -> pd.DataFrame:
    session_frame = sessions.copy()
    session_frame["open_value"] = pd.to_datetime(
        session_frame["regular_open_at"], utc=True, errors="coerce"
    )
    session_frame["close_value"] = pd.to_datetime(
        session_frame["regular_close_at"], utc=True, errors="coerce"
    )
    if session_frame[["open_value", "close_value"]].isna().any().any():
        raise ValidationExecutionPlanError("Exchange sessions contain invalid timestamps")
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        accepted_at = pd.Timestamp(str(event.accepted_value))
        exchange = str(event.assigned_exchange_mic)
        exchange_sessions = session_frame.loc[
            session_frame["exchange_mic"].eq(exchange)
        ].sort_values("open_value", kind="mergesort")
        event_date = accepted_at.date().isoformat()
        same_date = exchange_sessions.loc[exchange_sessions["session_date"].eq(event_date)]
        if same_date.empty:
            relation = "accepted_on_non_session"
        else:
            same_session = same_date.iloc[0]
            if accepted_at < same_session["open_value"]:
                relation = "accepted_before_regular_open"
            elif accepted_at <= same_session["close_value"]:
                relation = "accepted_during_regular_session"
            else:
                relation = "accepted_after_regular_close"
        future = exchange_sessions.loc[exchange_sessions["open_value"] > accepted_at].reset_index(
            drop=True
        )
        entry_index = cfg.entry_session_offset - 1
        exit_index = entry_index + cfg.holding_sessions - 1
        ready = len(future) > exit_index
        entry = future.iloc[entry_index] if ready else None
        exit_session = future.iloc[exit_index] if ready else None
        rows.append(
            {
                "market_event_id": str(event.market_event_id),
                "acceptance_session_relation": relation,
                "entry_session_offset": cfg.entry_session_offset,
                "holding_sessions": cfg.holding_sessions,
                "formal_entry_session_date": (
                    str(entry["session_date"]) if entry is not None else None
                ),
                "formal_entry_open_at": (
                    str(entry["regular_open_at"]) if entry is not None else None
                ),
                "formal_entry_price_type": cfg.entry_price_type,
                "entry_calendar_source_id": (
                    str(entry["source_id"]) if entry is not None else None
                ),
                "entry_calendar_source_sha256": (
                    str(entry["source_snapshot_sha256"]) if entry is not None else None
                ),
                "formal_exit_session_date": (
                    str(exit_session["session_date"]) if exit_session is not None else None
                ),
                "formal_exit_close_at": (
                    str(exit_session["regular_close_at"]) if exit_session is not None else None
                ),
                "formal_exit_price_type": cfg.exit_price_type,
                "exit_calendar_source_id": (
                    str(exit_session["source_id"]) if exit_session is not None else None
                ),
                "exit_calendar_source_sha256": (
                    str(exit_session["source_snapshot_sha256"])
                    if exit_session is not None
                    else None
                ),
                "schedule_basis_exchange_mic": (
                    str(entry["schedule_basis_exchange_mic"]) if entry is not None else None
                ),
                "formal_execution_window_ready": ready,
                "calendar_alignment_status": (
                    "formal_window_resolved" if ready else "insufficient_future_sessions"
                ),
            }
        )
    return pd.DataFrame(rows)


def _expand_security_components(
    events: pd.DataFrame, repair_components: pd.DataFrame
) -> pd.DataFrame:
    repair = repair_components.copy()
    if not repair.empty:
        _require_columns(
            repair,
            {
                "cik",
                "anchor_role",
                "ticker",
                "exchange_mic",
                "provider_security_id",
                "provider_security_id_kind",
                "identity_key",
                "identity_key_kind",
                "component_identity_ready",
                "selection_uses_return_values",
                "outcome_values_loaded",
            },
            "identity repair components",
        )
        if repair["selection_uses_return_values"].astype(bool).any():
            raise ValidationExecutionPlanError("Repair-component selection used returns")
        if repair["outcome_values_loaded"].astype(bool).any():
            raise ValidationExecutionPlanError("Repair components opened outcomes")
        repair["cik"] = repair["cik"].astype(str).str.zfill(10)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        method = str(event.identity_assignment_method)
        event_components: list[dict[str, Any]]
        if method.startswith("identity_repair_"):
            issuer = repair.loc[
                repair["cik"].eq(str(event.cik)) & repair["component_identity_ready"].astype(bool)
            ].copy()
            first = issuer.loc[issuer["anchor_role"].eq("first_candidate")]
            last = issuer.loc[issuer["anchor_role"].eq("last_candidate")]
            keys = ["ticker", "exchange_mic", "identity_key", "identity_key_kind"]
            if set(map(tuple, first[keys].astype(str).to_numpy())) != set(
                map(tuple, last[keys].astype(str).to_numpy())
            ):
                raise ValidationExecutionPlanError("Repair components are not endpoint stable")
            event_components = cast(
                list[dict[str, Any]],
                first.sort_values(keys, kind="mergesort").to_dict(orient="records"),
            )
        else:
            provider_id = _optional_text(event.assigned_provider_security_id)
            provider_kind = _optional_text(event.assigned_provider_security_id_kind)
            if provider_id is None:
                raise ValidationExecutionPlanError("A base identity lacks its stable key")
            event_components = [
                {
                    "ticker": str(event.assigned_ticker),
                    "exchange_mic": str(event.assigned_exchange_mic),
                    "provider_security_id": provider_id,
                    "provider_security_id_kind": provider_kind,
                    "identity_key": provider_id,
                    "identity_key_kind": provider_kind,
                    "evidence_source": "audited_endpoint_identity_interval",
                }
            ]
        expected = int(str(event.security_component_count))
        if len(event_components) != expected or not event_components:
            raise ValidationExecutionPlanError(
                "Security component count does not match identity audit"
            )
        weight = 1.0 / len(event_components)
        for index, component in enumerate(event_components, start=1):
            identity_key = str(component["identity_key"])
            ticker = str(component["ticker"])
            exchange = str(component["exchange_mic"])
            rows.append(
                {
                    "security_event_component_id": _stable_hash(
                        {
                            "market_event_id": str(event.market_event_id),
                            "identity_key": identity_key,
                            "ticker": ticker,
                            "exchange_mic": exchange,
                        }
                    ),
                    "market_event_id": str(event.market_event_id),
                    "validation_event_id": str(event.validation_event_id),
                    "cik": str(event.cik),
                    "accepted_at": str(event.accepted_at),
                    "form": str(event.form),
                    "accession_number": str(event.target_accession_number),
                    "identity_assignment_method": method,
                    "ticker": ticker,
                    "exchange_mic": exchange,
                    "provider_security_id": _optional_text(component.get("provider_security_id")),
                    "provider_security_id_kind": _optional_text(
                        component.get("provider_security_id_kind")
                    ),
                    "identity_key": identity_key,
                    "identity_key_kind": str(component["identity_key_kind"]),
                    "component_index": index,
                    "component_count": len(event_components),
                    "component_weight": weight,
                    "issuer_return_construction": str(event.issuer_return_construction),
                    "identity_evidence_source": str(
                        component.get("evidence_source", "identity_repair_resolution")
                    ),
                    "formal_entry_session_date": str(event.formal_entry_session_date),
                    "formal_entry_open_at": str(event.formal_entry_open_at),
                    "formal_entry_price_type": str(event.formal_entry_price_type),
                    "formal_exit_session_date": str(event.formal_exit_session_date),
                    "formal_exit_close_at": str(event.formal_exit_close_at),
                    "formal_exit_price_type": str(event.formal_exit_price_type),
                    "formal_execution_window_ready": bool(event.formal_execution_window_ready),
                    "security_component_execution_ready": bool(event.formal_execution_window_ready),
                    "selection_uses_return_values": False,
                    "outcome_values_loaded": False,
                }
            )
    result = pd.DataFrame(rows)
    if result["security_event_component_id"].duplicated().any():
        raise ValidationExecutionPlanError("Security event component IDs are duplicated")
    weights = result.groupby("market_event_id")["component_weight"].sum()
    if not np.allclose(weights.to_numpy(dtype=float), 1.0):
        raise ValidationExecutionPlanError("Issuer-event component weights do not sum to one")
    return result


def _attach_query_context(
    components: pd.DataFrame, sessions: pd.DataFrame, context_sessions: int
) -> pd.DataFrame:
    result = components.copy()
    starts: list[str] = []
    ends: list[str] = []
    for component in result.itertuples(index=False):
        venue = sessions.loc[sessions["exchange_mic"].eq(str(component.exchange_mic))].sort_values(
            "session_date", kind="mergesort"
        )
        dates = venue["session_date"].astype(str).tolist()
        entry_index = dates.index(str(component.formal_entry_session_date))
        exit_index = dates.index(str(component.formal_exit_session_date))
        start_index = max(0, entry_index - context_sessions)
        end_index = min(len(dates) - 1, exit_index + context_sessions)
        starts.append(dates[start_index])
        ends.append(dates[end_index])
    result["price_query_window_start"] = starts
    result["price_query_window_end"] = ends
    return result


def _build_security_spans(
    components: pd.DataFrame, cfg: ValidationExecutionPlanConfig
) -> pd.DataFrame:
    keys = ["cik", "ticker", "exchange_mic", "identity_key", "identity_key_kind"]
    rows: list[dict[str, Any]] = []
    for key, group in components.groupby(keys, sort=True, dropna=False):
        cik, ticker, exchange, identity_key, identity_key_kind = key
        provider_ids = group["provider_security_id"].dropna().astype(str).unique().tolist()
        provider_kinds = group["provider_security_id_kind"].dropna().astype(str).unique().tolist()
        if len(provider_ids) > 1 or len(provider_kinds) > 1:
            raise ValidationExecutionPlanError("One security span has inconsistent provider IDs")
        window_start = str(group["price_query_window_start"].min())
        window_end = str(group["price_query_window_end"].max())
        price_target_id = _stable_hash(
            {
                "cik": str(cik),
                "ticker": str(ticker),
                "exchange_mic": str(exchange),
                "identity_key": str(identity_key),
                "window_start": window_start,
                "window_end": window_end,
                "adjusted": False,
                "config_version": cfg.version,
            }
        )
        rows.append(
            {
                "price_target_id": price_target_id,
                "cik": str(cik),
                "selected_ticker": str(ticker),
                "selected_exchange_mic": str(exchange),
                "security_identity_key": str(identity_key),
                "security_identity_key_kind": str(identity_key_kind),
                "selected_provider_security_id": provider_ids[0] if provider_ids else None,
                "selected_provider_security_id_kind": (
                    provider_kinds[0] if provider_kinds else None
                ),
                "price_window_start": window_start,
                "price_window_end": window_end,
                "adjusted": False,
                "event_component_rows": len(group),
                "distinct_events": int(group["market_event_id"].nunique()),
                "entry_price_type": cfg.entry_price_type,
                "exit_price_type": cfg.exit_price_type,
                "price_collection_authorized": bool(
                    group["security_component_execution_ready"].astype(bool).all()
                ),
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    result = (
        pd.DataFrame(rows)
        .sort_values(["price_window_start", "cik", "selected_ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    result["price_acquisition_order"] = result.index + 1
    return result


def _build_action_targets(price_targets: pd.DataFrame) -> pd.DataFrame:
    result = price_targets[
        [
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "security_identity_key_kind",
            "price_window_start",
            "price_window_end",
            "price_acquisition_order",
        ]
    ].copy()
    result["action_target_id"] = result.apply(
        lambda row: _stable_hash(
            {
                "price_target_id": str(row["price_target_id"]),
                "ticker": str(row["selected_ticker"]),
                "window_start": str(row["price_window_start"]),
                "window_end": str(row["price_window_end"]),
                "queries": [
                    "reference_start",
                    "reference_end",
                    "ticker_events",
                    "splits",
                    "dividends",
                ],
            }
        ),
        axis=1,
    )
    result["required_query_kinds"] = "reference_start|reference_end|ticker_events|splits|dividends"
    result["action_collection_authorized"] = True
    result["selection_uses_return_values"] = False
    result["outcome_values_loaded"] = False
    result = result.sort_values("price_acquisition_order").reset_index(drop=True)
    result["action_acquisition_order"] = result.index + 1
    return result.drop(columns=["price_acquisition_order"])


def _build_delisting_targets(
    price_targets: pd.DataFrame, cfg: ValidationExecutionPlanConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_columns = [
        "cik",
        "security_identity_key",
        "security_identity_key_kind",
    ]
    for identity, group in price_targets.groupby(identity_columns, sort=True, dropna=False):
        cik, security_identity_key, security_identity_key_kind = identity
        ordered = group.sort_values(
            ["price_window_end", "price_acquisition_order"], kind="mergesort"
        )
        latest = ordered.iloc[-1]
        rows.append(
            {
                "cik": str(cik),
                "selected_ticker": str(latest["selected_ticker"]),
                "selected_exchange_mic": str(latest["selected_exchange_mic"]),
                "security_identity_key": str(security_identity_key),
                "security_identity_key_kind": str(security_identity_key_kind),
                "price_window_end": str(ordered["price_window_end"].max()),
                "price_acquisition_order": int(ordered["price_acquisition_order"].min()),
                "event_ticker_aliases": "|".join(
                    sorted(ordered["selected_ticker"].astype(str).unique())
                ),
                "event_exchange_mics": "|".join(
                    sorted(ordered["selected_exchange_mic"].astype(str).unique())
                ),
                "security_span_count": len(ordered),
            }
        )
    result = pd.DataFrame(rows)
    result["outcome_cutoff_date"] = cfg.delisting_outcome_cutoff_date
    if (
        result["price_window_end"]
        .map(date.fromisoformat)
        .ge(date.fromisoformat(cfg.delisting_outcome_cutoff_date))
        .any()
    ):
        raise ValidationExecutionPlanError("Delisting cutoff does not follow every price span")
    result["delisting_target_id"] = result.apply(
        lambda row: _stable_hash(
            {
                "cik": str(row["cik"]),
                "identity_key": str(row["security_identity_key"]),
                "outcome_cutoff_date": str(row["outcome_cutoff_date"]),
            }
        ),
        axis=1,
    )
    result["query_method"] = "cik_cutoff_active_true_then_inactive_if_security_absent"
    result["delisting_collection_authorized"] = True
    result["selection_uses_later_listing_status"] = False
    result["outcome_values_loaded"] = False
    result = result.sort_values("price_acquisition_order").reset_index(drop=True)
    result["delisting_acquisition_order"] = result.index + 1
    return result.drop(columns=["price_acquisition_order"])


def _build_benchmark_targets(
    events: pd.DataFrame,
    benchmark_plan: pd.DataFrame,
    sessions: pd.DataFrame,
    cfg: ValidationExecutionPlanConfig,
) -> pd.DataFrame:
    _require_columns(
        benchmark_plan,
        {"benchmark_role", "benchmark_ticker", "selection_uses_return_values"},
        "benchmark plan",
    )
    broad = benchmark_plan.loc[
        benchmark_plan["benchmark_role"].eq("broad_market_total_return_proxy")
        & benchmark_plan["benchmark_ticker"].notna()
    ].copy()
    if len(broad) != 1:
        raise ValidationExecutionPlanError("Exactly one broad-market benchmark is required")
    if broad["selection_uses_return_values"].astype(bool).any():
        raise ValidationExecutionPlanError("Benchmark selection used return outcomes")
    xnas = sessions.loc[sessions["exchange_mic"].eq("XNAS")].sort_values("session_date")
    dates = xnas["session_date"].astype(str).tolist()
    first_entry = str(events["formal_entry_session_date"].min())
    last_exit = str(events["formal_exit_session_date"].max())
    start_index = max(0, dates.index(first_entry) - cfg.price_context_sessions)
    end_index = min(len(dates) - 1, dates.index(last_exit) + cfg.price_context_sessions)
    ticker = str(broad.iloc[0]["benchmark_ticker"])
    window_start = dates[start_index]
    window_end = dates[end_index]
    return pd.DataFrame(
        [
            {
                "benchmark_target_id": _stable_hash(
                    {
                        "role": "broad_market_total_return_proxy",
                        "ticker": ticker,
                        "window_start": window_start,
                        "window_end": window_end,
                        "adjusted": False,
                    }
                ),
                "benchmark_role": "broad_market_total_return_proxy",
                "benchmark_ticker": ticker,
                "price_window_start": window_start,
                "price_window_end": window_end,
                "adjusted": False,
                "entry_price_type": cfg.entry_price_type,
                "exit_price_type": cfg.exit_price_type,
                "benchmark_collection_authorized": True,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        ]
    )


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
        raise ValidationExecutionPlanError(f"{label} lacks required columns: {missing}")


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
