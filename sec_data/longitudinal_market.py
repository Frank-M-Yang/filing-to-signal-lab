"""Freeze the final longitudinal market-data batch for effectiveness v1.

The plan collects one unadjusted daily-bar range per issuer, explicit splits and
dividends, point-in-time security identity, and one SPY benchmark range.  Issuers
whose ticker-events endpoint previously required a session-grid fallback are
split into event-window identity targets to avoid unnecessary full-year calls.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

LONGITUDINAL_MARKET_PLAN_VERSION: Final[str] = "longitudinal-market-plan-v1"
HORIZONS: Final[tuple[int, ...]] = (1, 5, 20)


@dataclass(frozen=True)
class LongitudinalMarketConfig:
    """Frozen market-data scope and execution conventions."""

    benchmark_ticker: str = "SPY"
    benchmark_cik: str = "0000884394"
    benchmark_currency: str = "USD"
    benchmark_role: str = "sp500_total_return_proxy"
    adjusted: bool = False
    entry_session_offset: int = 2
    primary_holding_sessions: int = 20
    secondary_holding_sessions: tuple[int, ...] = (1, 5)
    version: str = LONGITUDINAL_MARKET_PLAN_VERSION

    def validate(self) -> None:
        """Reject a plan that changes the pre-registered market convention."""
        if self.benchmark_ticker != "SPY" or self.benchmark_cik != "0000884394":
            raise ValueError("The market benchmark is frozen to SPY")
        if self.adjusted:
            raise ValueError("Issuer and benchmark bars must be unadjusted")
        if self.entry_session_offset != 2 or self.primary_holding_sessions != 20:
            raise ValueError("The primary window is frozen to +2 open and 20 sessions")
        if tuple(self.secondary_holding_sessions) != (1, 5):
            raise ValueError("Secondary holding windows are frozen to one and five sessions")


@dataclass(frozen=True)
class LongitudinalMarketTables:
    """Pure target and mapping tables before immutable persistence."""

    price_targets: pd.DataFrame
    action_targets: pd.DataFrame
    benchmark_target: pd.DataFrame
    benchmark_required_dates: pd.DataFrame
    event_horizons: pd.DataFrame
    event_market_map: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class LongitudinalMarketPlanResult:
    """Immutable collection plan with compatible child plan roots."""

    output_root: Path
    price_plan_root: Path
    action_plan_root: Path
    benchmark_plan_root: Path
    report: dict[str, Any]


def calculate_longitudinal_market_plan(
    events: pd.DataFrame,
    issuer_targets: pd.DataFrame,
    prior_action_targets: pd.DataFrame,
    prior_action_audit: pd.DataFrame,
    sessions: pd.DataFrame,
    config: LongitudinalMarketConfig | None = None,
) -> LongitudinalMarketTables:
    """Create issuer ranges, action evidence targets, and event horizon mappings."""
    cfg = config or LongitudinalMarketConfig()
    cfg.validate()
    selected_issuers = issuer_targets.loc[issuer_targets["planned_next_batch"].astype(bool)].copy()
    selected_events = events.loc[events["feature_dense_issuer_eligible"].astype(bool)].copy()
    if selected_issuers.empty or selected_events.empty:
        raise FilingCatalogError("Longitudinal market planning requires a nonempty dense cohort")
    if selected_events["longitudinal_event_id"].duplicated().any():
        raise FilingCatalogError("Longitudinal events must be unique")
    planned_ciks = set(selected_issuers["cik"].astype(str).str.zfill(10))
    event_ciks = set(selected_events["cik"].astype(str).str.zfill(10))
    if planned_ciks != event_ciks:
        raise FilingCatalogError("Issuer targets and feature-dense events have different CIKs")

    selected_issuers["cik"] = selected_issuers["cik"].astype(str).str.zfill(10)
    selected_events["cik"] = selected_events["cik"].astype(str).str.zfill(10)
    prior_action_targets = prior_action_targets.copy()
    prior_action_targets["cik"] = prior_action_targets["cik"].astype(str).str.zfill(10)
    prior_action_audit = prior_action_audit.copy()
    prior_action_audit["cik"] = prior_action_audit["cik"].astype(str).str.zfill(10)
    action_identity = _prior_identity(prior_action_targets, prior_action_audit, planned_ciks)
    event_horizons = _event_horizon_registry(selected_events, sessions, cfg)
    price_targets = _issuer_price_targets(selected_issuers, selected_events, cfg)
    action_targets, action_map = _action_targets(
        selected_issuers,
        selected_events,
        action_identity,
        event_horizons,
        sessions,
        cfg,
    )
    price_map = price_targets[["cik", "price_target_id"]]
    event_market_map = (
        selected_events[
            [
                "longitudinal_event_id",
                "cik",
                "accession_number",
                "representation_features_complete",
            ]
        ]
        .merge(price_map, on="cik", how="left", validate="many_to_one")
        .merge(action_map, on=["longitudinal_event_id", "cik"], how="left", validate="one_to_one")
    )
    if event_market_map[["price_target_id", "action_target_id"]].isna().any().any():
        raise FilingCatalogError("An event lacks a price or action target mapping")
    benchmark_target, required_dates = _benchmark_plan(event_horizons, cfg)
    fallback_ciks = set(
        action_identity.loc[
            action_identity["ticker_events_unavailable_fallback_used"].astype(bool), "cik"
        ]
    )
    fallback_sessions = int(
        action_targets.loc[
            action_targets["fallback_event_window_target"], "fallback_sessions"
        ].sum()
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "issuers": len(selected_issuers),
        "events": len(selected_events),
        "complete_feature_events": int(
            selected_events["representation_features_complete"].astype(bool).sum()
        ),
        "event_horizon_rows": len(event_horizons),
        "issuer_price_targets": len(price_targets),
        "security_action_targets": len(action_targets),
        "prior_fallback_issuers": len(fallback_ciks),
        "fallback_event_window_targets": int(
            action_targets["fallback_event_window_target"].astype(bool).sum()
        ),
        "fallback_reference_session_queries_if_needed": fallback_sessions,
        "benchmark_queries": 3,
        "planned_base_queries": len(price_targets) + 5 * len(action_targets) + 3,
        "planned_maximum_queries_including_fallback": (
            len(price_targets) + 5 * len(action_targets) + 3 + fallback_sessions
        ),
        "query_adjusted": False,
        "selection_uses_market_returns": False,
        "returns_computed": False,
        "config": asdict(cfg),
    }
    return LongitudinalMarketTables(
        price_targets=price_targets,
        action_targets=action_targets,
        benchmark_target=benchmark_target,
        benchmark_required_dates=required_dates,
        event_horizons=event_horizons,
        event_market_map=event_market_map,
        report=report,
    )


def build_longitudinal_market_plan(
    longitudinal_root: Path,
    prior_actions_root: Path,
    calendar_root: Path,
    *,
    output_root: Path = Path("data/market/longitudinal_study/targets"),
    config: LongitudinalMarketConfig | None = None,
) -> LongitudinalMarketPlanResult:
    """Persist the final collection plan and three compatible child plan roots."""
    cfg = config or LongitudinalMarketConfig()
    cfg.validate()
    input_hashes = {
        "longitudinal_manifest_sha256": _sha256_file(longitudinal_root / "output_manifest.json"),
        "prior_actions_manifest_sha256": _sha256_file(prior_actions_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
    }
    longitudinal_report = _read_json(longitudinal_root / "longitudinal_expansion_report.json")
    prior_actions_report = _read_json(prior_actions_root / "massive_security_actions_report.json")
    calendar_report = _read_json(calendar_root / "exchange_calendar_report.json")
    events = load_catalog_table(longitudinal_root, "feature_dense_event_panel.parquet")
    issuer_targets = load_catalog_table(
        longitudinal_root, "issuer_market_collection_targets.parquet"
    )
    prior_action_targets = load_catalog_table(
        prior_actions_root, "massive_security_actions_targets.parquet"
    )
    prior_action_audit = load_catalog_table(
        prior_actions_root, "sample_security_actions_audit.parquet"
    )
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    tables = calculate_longitudinal_market_plan(
        events,
        issuer_targets,
        prior_action_targets,
        prior_action_audit,
        sessions,
        cfg,
    )
    payload = json.dumps(
        {**input_hashes, "config": asdict(cfg), "report": tables.report},
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        report = _read_json(target / "longitudinal_market_plan_report.json")
        return LongitudinalMarketPlanResult(
            target.resolve(),
            (target / "prices").resolve(),
            (target / "actions").resolve(),
            (target / "benchmark").resolve(),
            report,
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete longitudinal market plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    price_root = staging / "prices"
    action_root = staging / "actions"
    benchmark_root = staging / "benchmark"
    price_root.mkdir()
    action_root.mkdir()
    benchmark_root.mkdir()

    price_version, price_manifest = _write_price_plan(
        price_root,
        tables.price_targets,
        dataset_version,
        prior_actions_report,
    )
    _write_action_plan(
        action_root,
        tables.action_targets,
        dataset_version,
        prior_actions_report,
        price_version,
        price_manifest,
        calendar_report,
        input_hashes["calendar_manifest_sha256"],
    )
    _write_benchmark_plan(
        benchmark_root,
        tables.benchmark_target,
        tables.benchmark_required_dates,
        dataset_version,
        calendar_report,
        input_hashes["calendar_manifest_sha256"],
        cfg,
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "event_horizon_registry.parquet": tables.event_horizons,
        "event_market_map.parquet": tables.event_market_map,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "longitudinal_version": longitudinal_report["dataset_version"],
        "prior_actions_version": prior_actions_report["dataset_version"],
        "calendar_version": calendar_report["dataset_version"],
        **input_hashes,
        "price_plan_version": price_version,
        "artifacts": artifacts,
    }
    report_path = staging / "longitudinal_market_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **input_hashes,
                "artifacts": artifacts,
                "price_plan_manifest_sha256": price_manifest,
                "action_plan_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
                "benchmark_plan_manifest_sha256": _sha256_file(
                    benchmark_root / "output_manifest.json"
                ),
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return LongitudinalMarketPlanResult(
        target.resolve(),
        (target / "prices").resolve(),
        (target / "actions").resolve(),
        (target / "benchmark").resolve(),
        report,
    )


def _prior_identity(
    targets: pd.DataFrame, audit: pd.DataFrame, planned_ciks: set[str]
) -> pd.DataFrame:
    columns = [
        "cik",
        "selected_share_class_figi",
        "selected_composite_figi",
        "selected_reference_source_sha256",
    ]
    identity = targets[columns].drop_duplicates("cik")
    if identity["cik"].duplicated().any():
        raise FilingCatalogError("Prior action identity must be unique by CIK")
    fallback = audit[
        ["cik", "ticker_events_unavailable_fallback_used", "historical_symbol_interval_ready"]
    ].drop_duplicates("cik")
    identity = identity.merge(fallback, on="cik", how="left", validate="one_to_one")
    identity = identity.loc[identity["cik"].isin(planned_ciks)].copy()
    if set(identity["cik"]) != planned_ciks:
        raise FilingCatalogError("A planned issuer lacks prior FIGI identity evidence")
    if not identity["historical_symbol_interval_ready"].fillna(False).astype(bool).all():
        raise FilingCatalogError("Prior issuer identity must already have passed its pilot gate")
    return identity


def _event_horizon_registry(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    config: LongitudinalMarketConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    session_frame = sessions.copy()
    session_frame["session_date"] = session_frame["session_date"].astype(str)
    for event in events.itertuples(index=False):
        exchange = session_frame.loc[
            session_frame["exchange_mic"].eq(str(event.selected_exchange_mic))
        ].sort_values("session_date", kind="mergesort")
        matches = np.flatnonzero(
            exchange["session_date"].eq(str(event.formal_entry_session_date)).to_numpy()
        )
        if len(matches) != 1:
            raise FilingCatalogError("Formal entry session is absent or duplicated")
        entry_index = int(matches[0])
        for horizon in HORIZONS:
            exit_index = entry_index + horizon - 1
            if exit_index >= len(exchange):
                raise FilingCatalogError("Calendar ends before a frozen event horizon")
            exit_session = exchange.iloc[exit_index]
            exit_date = str(exit_session["session_date"])
            if horizon == config.primary_holding_sessions and exit_date != str(
                event.formal_exit_session_date
            ):
                raise FilingCatalogError("Recomputed 20-session exit disagrees with expansion")
            rows.append(
                {
                    "longitudinal_event_id": str(event.longitudinal_event_id),
                    "cik": str(event.cik),
                    "accession_number": str(event.accession_number),
                    "accepted_at": str(event.accepted_at),
                    "form": str(event.form),
                    "horizon_sessions": horizon,
                    "is_primary_horizon": horizon == config.primary_holding_sessions,
                    "entry_session_date": str(event.formal_entry_session_date),
                    "entry_open_at": str(event.formal_entry_open_at),
                    "exit_session_date": exit_date,
                    "exit_close_at": str(exit_session["regular_close_at"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["longitudinal_event_id", "horizon_sessions"]).any():
        raise FilingCatalogError("Event horizon registry is duplicated")
    return result.sort_values(
        ["accepted_at", "cik", "horizon_sessions"], kind="mergesort"
    ).reset_index(drop=True)


def _issuer_price_targets(
    issuers: pd.DataFrame,
    events: pd.DataFrame,
    config: LongitudinalMarketConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = issuers.sort_values("collection_order", kind="mergesort")
    for order, issuer in enumerate(ordered.itertuples(index=False), start=1):
        issuer_events = events.loc[events["cik"].eq(str(issuer.cik))].sort_values("accepted_at")
        first_event = issuer_events.iloc[0]
        price_target_id = _stable_hash(
            {
                "market_batch_id": str(issuer.market_batch_id),
                "ticker": str(issuer.selected_ticker),
                "window_start": str(issuer.market_window_start),
                "window_end": str(issuer.market_window_end),
                "adjusted": False,
                "config_version": config.version,
            }
        )
        rows.append(
            {
                "price_target_id": price_target_id,
                "price_acquisition_order": order,
                "target_id": str(issuer.market_batch_id),
                "cik": str(issuer.cik),
                "accepted_at": str(first_event["accepted_at"]),
                "query_date": pd.Timestamp(first_event["accepted_at"]).date().isoformat(),
                "form": "ISSUER_BATCH",
                "accession_number": str(issuer.market_batch_id),
                "selected_ticker": str(issuer.selected_ticker),
                "selected_exchange_mic": str(issuer.selected_exchange_mic),
                "selected_provider_security_id": str(issuer.selected_provider_security_id),
                "selected_provider_security_id_kind": str(
                    issuer.selected_provider_security_id_kind
                ),
                "reference_source_sha256": str(issuer.selected_reference_source_sha256),
                "currency_name": str(issuer.currency_name),
                "price_window_start": str(issuer.market_window_start),
                "price_window_end": str(issuer.market_window_end),
                "adjusted": False,
                "entry_session_offset": config.entry_session_offset,
                "holding_sessions": config.primary_holding_sessions,
            }
        )
    return pd.DataFrame(rows)


def _action_targets(
    issuers: pd.DataFrame,
    events: pd.DataFrame,
    identity: pd.DataFrame,
    event_horizons: pd.DataFrame,
    sessions: pd.DataFrame,
    config: LongitudinalMarketConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = identity.set_index("cik")
    rows: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    order = 0
    for issuer in issuers.sort_values("collection_order").itertuples(index=False):
        cik = str(issuer.cik)
        issuer_events = events.loc[events["cik"].eq(cik)].sort_values("accepted_at")
        fallback = bool(identity.at[cik, "ticker_events_unavailable_fallback_used"])
        groups: list[tuple[pd.DataFrame, str, str, bool]] = []
        if fallback:
            for event in issuer_events.itertuples(index=False):
                horizon = event_horizons.loc[
                    event_horizons["longitudinal_event_id"].eq(str(event.longitudinal_event_id))
                    & event_horizons["horizon_sessions"].eq(config.primary_holding_sessions)
                ].iloc[0]
                groups.append(
                    (
                        issuer_events.loc[
                            issuer_events["longitudinal_event_id"].eq(
                                str(event.longitudinal_event_id)
                            )
                        ],
                        str(horizon["entry_session_date"]),
                        str(horizon["exit_session_date"]),
                        True,
                    )
                )
        else:
            groups.append(
                (
                    issuer_events,
                    str(issuer.market_window_start),
                    str(issuer.market_window_end),
                    False,
                )
            )
        for group, window_start, window_end, event_window in groups:
            order += 1
            first = group.iloc[0]
            action_target_id = _stable_hash(
                {
                    "cik": cik,
                    "share_class_figi": str(identity.at[cik, "selected_share_class_figi"]),
                    "window_start": window_start,
                    "window_end": window_end,
                    "config_version": config.version,
                }
            )
            session_count = len(
                sessions.loc[
                    sessions["exchange_mic"].eq(str(issuer.selected_exchange_mic))
                    & sessions["session_date"].astype(str).between(window_start, window_end)
                ]
            )
            rows.append(
                {
                    "action_target_id": action_target_id,
                    "action_acquisition_order": order,
                    "price_target_id": _stable_hash(
                        {
                            "market_batch_id": str(issuer.market_batch_id),
                            "ticker": str(issuer.selected_ticker),
                            "window_start": str(issuer.market_window_start),
                            "window_end": str(issuer.market_window_end),
                            "adjusted": False,
                            "config_version": config.version,
                        }
                    ),
                    "target_id": str(issuer.market_batch_id),
                    "cik": cik,
                    "accepted_at": str(first["accepted_at"]),
                    "query_date": pd.Timestamp(first["accepted_at"]).date().isoformat(),
                    "form": "EVENT_WINDOW" if event_window else "ISSUER_BATCH",
                    "accession_number": str(first["accession_number"]),
                    "selected_ticker": str(issuer.selected_ticker),
                    "selected_exchange_mic": str(issuer.selected_exchange_mic),
                    "selected_provider_security_id": str(issuer.selected_provider_security_id),
                    "selected_provider_security_id_kind": str(
                        issuer.selected_provider_security_id_kind
                    ),
                    "selected_share_class_figi": str(identity.at[cik, "selected_share_class_figi"]),
                    "selected_composite_figi": str(identity.at[cik, "selected_composite_figi"]),
                    "selected_reference_source_sha256": str(
                        identity.at[cik, "selected_reference_source_sha256"]
                    ),
                    "currency_name": str(issuer.currency_name),
                    "price_window_start": window_start,
                    "price_window_end": window_end,
                    "formal_entry_session_date": window_start,
                    "formal_exit_session_date": window_end,
                    "fallback_event_window_target": event_window,
                    "fallback_sessions": session_count if event_window else 0,
                }
            )
            for event_id in group["longitudinal_event_id"]:
                maps.append(
                    {
                        "longitudinal_event_id": str(event_id),
                        "cik": cik,
                        "action_target_id": action_target_id,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(maps)


def _benchmark_plan(
    event_horizons: pd.DataFrame, config: LongitudinalMarketConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = pd.concat(
        [
            event_horizons[["longitudinal_event_id", "entry_session_date"]].rename(
                columns={"entry_session_date": "required_session_date"}
            ),
            event_horizons[["longitudinal_event_id", "exit_session_date"]].rename(
                columns={"exit_session_date": "required_session_date"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    window_start = str(required["required_session_date"].min())
    window_end = str(required["required_session_date"].max())
    target_id = _stable_hash(
        {
            "ticker": config.benchmark_ticker,
            "cik": config.benchmark_cik,
            "window_start": window_start,
            "window_end": window_end,
            "adjusted": False,
            "config_version": config.version,
        }
    )
    target = pd.DataFrame(
        [
            {
                "benchmark_target_id": target_id,
                "ticker": config.benchmark_ticker,
                "cik": config.benchmark_cik,
                "currency": config.benchmark_currency,
                "benchmark_role": config.benchmark_role,
                "window_start": window_start,
                "window_end": window_end,
                "adjusted": False,
            }
        ]
    )
    return target, required.reset_index(drop=True)


def _write_price_plan(
    root: Path,
    targets: pd.DataFrame,
    parent_version: str,
    prior_actions_report: dict[str, Any],
) -> tuple[str, str]:
    version = parent_version + "-prices"
    target_path = root / "massive_daily_price_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report = {
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": prior_actions_report["reference_version"],
        "reference_manifest_sha256": prior_actions_report["reference_manifest_sha256"],
        "price_targets": len(targets),
        "query_adjusted": False,
        "selection_uses_price_coverage": False,
        "current_ticker_join_used": False,
        "artifacts": {target_path.name: artifact},
    }
    report_path = root / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = root / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": version,
                "reference_manifest_sha256": report["reference_manifest_sha256"],
                "artifacts": {target_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return version, _sha256_file(manifest_path)


def _write_action_plan(
    root: Path,
    targets: pd.DataFrame,
    parent_version: str,
    prior_actions_report: dict[str, Any],
    price_version: str,
    price_manifest_sha256: str,
    calendar_report: dict[str, Any],
    calendar_manifest_sha256: str,
) -> None:
    version = parent_version + "-actions"
    target_path = root / "massive_security_actions_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report = {
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": prior_actions_report["reference_version"],
        "reference_manifest_sha256": prior_actions_report["reference_manifest_sha256"],
        "price_version": price_version,
        "price_manifest_sha256": price_manifest_sha256,
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "targets": len(targets),
        "queries_per_target": 5,
        "planned_queries": len(targets) * 5,
        "ticker_events_endpoint_experimental": True,
        "current_ticker_join_used": False,
        "selection_uses_action_coverage": False,
        "artifacts": {target_path.name: artifact},
    }
    report_path = root / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": version,
                "reference_manifest_sha256": report["reference_manifest_sha256"],
                "price_manifest_sha256": price_manifest_sha256,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "artifacts": {target_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_benchmark_plan(
    root: Path,
    target: pd.DataFrame,
    required: pd.DataFrame,
    parent_version: str,
    calendar_report: dict[str, Any],
    calendar_manifest_sha256: str,
    config: LongitudinalMarketConfig,
) -> None:
    version = parent_version + "-benchmark"
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "massive_benchmark_target.parquet": target,
        "benchmark_required_session_dates.parquet": required,
    }.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    row = target.iloc[0]
    report = {
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "config": asdict(config),
        "benchmark_target_id": str(row["benchmark_target_id"]),
        "benchmark_ticker": str(row["ticker"]),
        "window_start": str(row["window_start"]),
        "window_end": str(row["window_end"]),
        "required_event_dates": len(required),
        "queries": ["daily_bars", "splits", "dividends"],
        "query_adjusted": False,
        "benchmark_identity_method": "explicit_versioned_instrument_specification",
        "current_ticker_join_used": False,
        "artifacts": artifacts,
    }
    report_path = root / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": version,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
