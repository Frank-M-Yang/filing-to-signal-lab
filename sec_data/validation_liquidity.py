"""Outcome-blind pre-filing liquidity and execution-eligibility audit."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.exchange_calendar import (
    build_historical_exchange_sessions,
    load_preserved_official_calendar_sources,
)
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.massive_prices import load_collected_massive_daily_bars

VALIDATION_LIQUIDITY_VERSION: Final[str] = "validation-pre-event-liquidity-v1"
VALIDATION_LIQUIDITY_AUDIT_VERSION: Final[str] = (
    "validation-pre-event-liquidity-audit-v3-source-tiered"
)


class ValidationLiquidityError(FilingCatalogError):
    """Raised when pre-event liquidity evidence violates the frozen design."""


@dataclass(frozen=True)
class ValidationLiquidityConfig:
    """Explicit, outcome-blind execution-capacity assumptions."""

    lookback_sessions: int = 20
    minimum_print_coverage: float = 0.80
    reference_position_notional_usd: float = 10_000.0
    maximum_median_dollar_volume_participation: float = 0.01
    calendar_buffer_days: int = 60
    version: str = VALIDATION_LIQUIDITY_VERSION

    @property
    def minimum_median_dollar_volume_usd(self) -> float:
        """Return the median dollar-volume floor implied by capacity assumptions."""
        return (
            self.reference_position_notional_usd / self.maximum_median_dollar_volume_participation
        )

    def validate(self) -> None:
        """Reject ambiguous or economically invalid eligibility parameters."""
        if self.lookback_sessions < 5:
            raise ValidationLiquidityError("Liquidity lookback must span at least five sessions")
        if not 0 < self.minimum_print_coverage <= 1:
            raise ValidationLiquidityError("Print coverage must lie in (0, 1]")
        if self.reference_position_notional_usd <= 0:
            raise ValidationLiquidityError("Reference position notional must be positive")
        if not 0 < self.maximum_median_dollar_volume_participation <= 0.10:
            raise ValidationLiquidityError(
                "Median dollar-volume participation must lie in (0, 0.10]"
            )
        if self.calendar_buffer_days < 30:
            raise ValidationLiquidityError("Calendar buffer is too short for the lookback")


@dataclass(frozen=True)
class ValidationLiquidityResult:
    """Immutable target plan or pre-event liquidity audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_pre_event_liquidity_targets(
    components: pd.DataFrame,
    security_targets: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    config: ValidationLiquidityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze exact prior-session evidence and one price query per security."""
    cfg = config or ValidationLiquidityConfig()
    cfg.validate()
    _require_columns(
        components,
        {
            "security_event_component_id",
            "market_event_id",
            "validation_event_id",
            "cik",
            "accepted_at",
            "ticker",
            "exchange_mic",
            "identity_key",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "execution components",
    )
    _require_columns(
        security_targets,
        {
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "security_identity_key",
            "security_identity_key_kind",
            "selected_provider_security_id",
            "selected_provider_security_id_kind",
        },
        "security targets",
    )
    _require_columns(
        sessions,
        {"exchange_mic", "session_date", "regular_open_at", "regular_close_at"},
        "exchange sessions",
    )
    if components.empty or security_targets.empty or sessions.empty:
        raise ValidationLiquidityError("Liquidity target inputs cannot be empty")
    if components["security_event_component_id"].duplicated().any():
        raise ValidationLiquidityError("Execution component IDs are duplicated")
    if components["selection_uses_return_values"].astype(bool).any():
        raise ValidationLiquidityError("Liquidity target construction used return values")
    if components["outcome_values_loaded"].astype(bool).any():
        raise ValidationLiquidityError("Liquidity target construction opened outcomes")

    session_frame = sessions.copy()
    session_frame["regular_close_value"] = pd.to_datetime(
        session_frame["regular_close_at"], utc=True, errors="coerce"
    )
    if session_frame["regular_close_value"].isna().any():
        raise ValidationLiquidityError("Exchange sessions contain invalid close timestamps")
    event_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    ordered_components = components.sort_values(
        ["accepted_at", "security_event_component_id"], kind="mergesort"
    )
    for component in ordered_components.itertuples(index=False):
        accepted_at = pd.Timestamp(str(component.accepted_at))
        if accepted_at.tzinfo is None:
            raise ValidationLiquidityError("Acceptance timestamps must include a timezone")
        exchange_sessions = session_frame.loc[
            session_frame["exchange_mic"].eq(str(component.exchange_mic))
            & session_frame["regular_close_value"].lt(accepted_at)
        ].sort_values("regular_close_value", kind="mergesort")
        prior = exchange_sessions.tail(cfg.lookback_sessions).reset_index(drop=True)
        if len(prior) != cfg.lookback_sessions:
            raise ValidationLiquidityError(
                "Official calendar does not contain the full pre-event lookback for "
                f"{component.security_event_component_id}"
            )
        component_id = str(component.security_event_component_id)
        for lookback_index, session in enumerate(prior.itertuples(index=False), start=1):
            session_rows.append(
                {
                    "security_event_component_id": component_id,
                    "market_event_id": str(component.market_event_id),
                    "validation_event_id": str(component.validation_event_id),
                    "cik": str(component.cik),
                    "ticker": str(component.ticker),
                    "exchange_mic": str(component.exchange_mic),
                    "identity_key": str(component.identity_key),
                    "accepted_at": accepted_at.isoformat(),
                    "lookback_session_index": lookback_index,
                    "session_date": str(session.session_date),
                    "regular_open_at": str(session.regular_open_at),
                    "regular_close_at": str(session.regular_close_at),
                }
            )
        event_rows.append(
            {
                "security_event_component_id": component_id,
                "market_event_id": str(component.market_event_id),
                "validation_event_id": str(component.validation_event_id),
                "cik": str(component.cik),
                "ticker": str(component.ticker),
                "exchange_mic": str(component.exchange_mic),
                "identity_key": str(component.identity_key),
                "accepted_at": accepted_at.isoformat(),
                "liquidity_window_start": str(prior.iloc[0]["session_date"]),
                "liquidity_window_end": str(prior.iloc[-1]["session_date"]),
                "required_lookback_sessions": cfg.lookback_sessions,
                "latest_evidence_close_at": str(prior.iloc[-1]["regular_close_at"]),
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    events = pd.DataFrame(event_rows)
    event_sessions = pd.DataFrame(session_rows)
    if (
        not pd.to_datetime(event_sessions["regular_close_at"], utc=True)
        .lt(pd.to_datetime(event_sessions["accepted_at"], utc=True))
        .all()
    ):
        raise ValidationLiquidityError("A liquidity observation is not strictly pre-acceptance")

    identity_columns = ["cik", "ticker", "exchange_mic", "identity_key"]
    spans = (
        events.groupby(identity_columns, as_index=False)
        .agg(
            price_window_start=("liquidity_window_start", "min"),
            price_window_end=("liquidity_window_end", "max"),
            event_component_rows=("security_event_component_id", "size"),
        )
        .reset_index(drop=True)
    )
    target_identity = security_targets.rename(
        columns={
            "selected_ticker": "ticker",
            "selected_exchange_mic": "exchange_mic",
            "security_identity_key": "identity_key",
        }
    )
    spans = spans.merge(
        target_identity[
            [
                "cik",
                "ticker",
                "exchange_mic",
                "identity_key",
                "security_identity_key_kind",
                "selected_provider_security_id",
                "selected_provider_security_id_kind",
            ]
        ],
        on=identity_columns,
        how="left",
        validate="one_to_one",
    )
    if spans["security_identity_key_kind"].isna().any():
        raise ValidationLiquidityError("A liquidity span has no frozen security identity")
    spans["adjusted"] = False
    spans["price_collection_authorized"] = True
    spans["selection_uses_return_values"] = False
    spans["outcome_values_loaded"] = False
    spans["price_target_id"] = spans.apply(
        lambda row: _stable_hash(
            {
                "cik": str(row["cik"]),
                "identity_key": str(row["identity_key"]),
                "ticker": str(row["ticker"]),
                "window_start": str(row["price_window_start"]),
                "window_end": str(row["price_window_end"]),
                "adjusted": False,
                "config_version": cfg.version,
            }
        ),
        axis=1,
    )
    targets = spans.sort_values(
        ["price_window_start", "cik", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    targets["price_acquisition_order"] = np.arange(1, len(targets) + 1)
    targets["target_id"] = targets["price_target_id"]
    targets = targets.rename(
        columns={
            "ticker": "selected_ticker",
            "exchange_mic": "selected_exchange_mic",
            "identity_key": "security_identity_key",
        }
    )
    target_key = targets.set_index(
        ["cik", "selected_ticker", "selected_exchange_mic", "security_identity_key"]
    )["price_target_id"]
    lookup = target_key.to_dict()
    events["price_target_id"] = events.apply(
        lambda row: lookup[
            (
                str(row["cik"]),
                str(row["ticker"]),
                str(row["exchange_mic"]),
                str(row["identity_key"]),
            )
        ],
        axis=1,
    )
    event_sessions["price_target_id"] = event_sessions.apply(
        lambda row: lookup[
            (
                str(row["cik"]),
                str(row["ticker"]),
                str(row["exchange_mic"]),
                str(row["identity_key"]),
            )
        ],
        axis=1,
    )
    return targets, events, event_sessions


def calculate_pre_event_liquidity_audit(
    event_windows: pd.DataFrame,
    event_sessions: pd.DataFrame,
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    rejections: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    config: ValidationLiquidityConfig | None = None,
    evidence_resolution: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure only prior-session trading activity and classify eligibility."""
    cfg = config or ValidationLiquidityConfig()
    cfg.validate()
    _require_columns(
        event_sessions,
        {
            "security_event_component_id",
            "accepted_at",
            "price_target_id",
            "session_date",
            "regular_close_at",
        },
        "liquidity event sessions",
    )
    _require_columns(
        bars,
        {"price_target_id", "session_date", "close_unadjusted", "volume", "transactions"},
        "liquidity bars",
    )
    if event_windows["outcome_values_loaded"].astype(bool).any():
        raise ValidationLiquidityError("Liquidity audit opened outcome values")
    if bars.duplicated(["price_target_id", "session_date"]).any():
        raise ValidationLiquidityError("Liquidity bars duplicate a target-session row")
    if (
        not pd.to_datetime(event_sessions["regular_close_at"], utc=True)
        .lt(pd.to_datetime(event_sessions["accepted_at"], utc=True))
        .all()
    ):
        raise ValidationLiquidityError("Liquidity audit contains post-acceptance evidence")

    observed = bars[
        ["price_target_id", "session_date", "close_unadjusted", "volume", "transactions"]
    ].copy()
    observed["print_observed"] = observed["volume"].astype(float).gt(0)
    observed["dollar_volume_usd"] = observed["close_unadjusted"].astype(float) * observed[
        "volume"
    ].astype(float)
    evidence = event_sessions.merge(
        observed,
        on=["price_target_id", "session_date"],
        how="left",
        validate="many_to_one",
    )
    evidence["print_observed"] = evidence["print_observed"].fillna(False).astype(bool)
    completion_ids = set(completions.get("price_target_id", pd.Series(dtype=str)).astype(str))
    rejection_counts = rejections.groupby("price_target_id").size().to_dict()
    evidence_ready_lookup: dict[str, bool] = {}
    unresolved_lookup: dict[str, int] = {}
    secondary_resolution_supplied = evidence_resolution is not None
    if evidence_resolution is not None:
        _require_columns(
            evidence_resolution,
            {
                "security_event_component_id",
                "pre_event_liquidity_evidence_complete",
                "unresolved_history_sessions",
            },
            "liquidity evidence resolution",
        )
        raw_evidence_ready = (
            evidence_resolution.set_index("security_event_component_id")[
                "pre_event_liquidity_evidence_complete"
            ]
            .astype(bool)
            .to_dict()
        )
        evidence_ready_lookup = {str(key): bool(value) for key, value in raw_evidence_ready.items()}
        raw_unresolved = (
            evidence_resolution.set_index("security_event_component_id")[
                "unresolved_history_sessions"
            ]
            .astype(int)
            .to_dict()
        )
        unresolved_lookup = {str(key): int(value) for key, value in raw_unresolved.items()}
    rows: list[dict[str, Any]] = []
    for component_id, group in evidence.groupby("security_event_component_id", sort=False):
        window = event_windows.loc[event_windows["security_event_component_id"].eq(component_id)]
        if len(window) != 1:
            raise ValidationLiquidityError("Liquidity event window is missing or duplicated")
        event = window.iloc[0]
        target_id = str(event["price_target_id"])
        expected = len(group)
        observed_count = int(group["print_observed"].sum())
        coverage = observed_count / expected if expected else 0.0
        printed = group.loc[group["print_observed"]]
        median_dollar_volume = (
            float(printed["dollar_volume_usd"].median()) if not printed.empty else np.nan
        )
        median_transactions = (
            float(printed["transactions"].dropna().median())
            if printed["transactions"].notna().any()
            else np.nan
        )
        primary_query_complete = target_id in completion_ids
        rejection_count = int(rejection_counts.get(target_id, 0))
        evidence_ready = evidence_ready_lookup.get(str(component_id), True)
        query_complete = primary_query_complete or (
            secondary_resolution_supplied and evidence_ready
        )
        unresolved_sessions = unresolved_lookup.get(str(component_id), 0)
        calendar_ready = expected == cfg.lookback_sessions
        coverage_ready = coverage >= cfg.minimum_print_coverage
        capacity_ready = bool(
            np.isfinite(median_dollar_volume)
            and median_dollar_volume >= cfg.minimum_median_dollar_volume_usd
        )
        eligible = bool(
            query_complete
            and rejection_count == 0
            and evidence_ready
            and calendar_ready
            and coverage_ready
            and capacity_ready
        )
        event_values = {str(key): value for key, value in event.to_dict().items()}
        rows.append(
            {
                **event_values,
                "price_query_complete": query_complete,
                "primary_price_query_complete": primary_query_complete,
                "parser_rejection_rows": rejection_count,
                "pre_event_liquidity_evidence_complete": evidence_ready,
                "unresolved_history_sessions": unresolved_sessions,
                "expected_sessions": expected,
                "observed_print_sessions": observed_count,
                "print_coverage": coverage,
                "median_print_dollar_volume_usd": median_dollar_volume,
                "median_print_transactions": median_transactions,
                "minimum_print_coverage": cfg.minimum_print_coverage,
                "minimum_median_dollar_volume_usd": (cfg.minimum_median_dollar_volume_usd),
                "reference_position_notional_usd": cfg.reference_position_notional_usd,
                "maximum_median_dollar_volume_participation": (
                    cfg.maximum_median_dollar_volume_participation
                ),
                "calendar_lookback_ready": calendar_ready,
                "print_coverage_ready": coverage_ready,
                "capacity_ready": capacity_ready,
                "liquidity_eligible": eligible,
                "liquidity_status": _liquidity_status(
                    query_complete=query_complete,
                    rejection_count=rejection_count,
                    evidence_ready=evidence_ready,
                    calendar_ready=calendar_ready,
                    coverage_ready=coverage_ready,
                    capacity_ready=capacity_ready,
                ),
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
                "returns_computed": False,
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["accepted_at", "security_event_component_id"], kind="mergesort"
    )
    ineligible = audit.loc[
        audit["pre_event_liquidity_evidence_complete"].astype(bool)
        & ~audit["liquidity_eligible"].astype(bool)
    ].copy()
    target_audit = targets.copy()
    target_audit["provider_query_complete"] = target_audit["price_target_id"].isin(completion_ids)
    target_audit["validated_bar_rows"] = (
        target_audit["price_target_id"].map(bars.groupby("price_target_id").size()).fillna(0)
    ).astype(int)
    target_audit["parser_rejection_rows"] = (
        target_audit["price_target_id"].map(rejections.groupby("price_target_id").size()).fillna(0)
    ).astype(int)
    target_audit["returns_computed"] = False
    return audit.reset_index(drop=True), ineligible.reset_index(drop=True), target_audit


def build_validation_liquidity_plan(
    execution_plan_root: Path,
    *,
    calendar_raw_root: Path = Path("data/market/exchange_calendar"),
    output_root: Path = Path("data/market/validation_liquidity/targets"),
    config: ValidationLiquidityConfig | None = None,
) -> ValidationLiquidityResult:
    """Persist complete pre-event session evidence and price targets before collection."""
    cfg = config or ValidationLiquidityConfig()
    cfg.validate()
    execution_manifest = _sha256_file(execution_plan_root / "output_manifest.json")
    execution_report = _read_json(execution_plan_root / "validation_execution_plan_report.json")
    components = load_catalog_table(
        execution_plan_root, "validation_security_event_components.parquet"
    )
    security_targets = load_catalog_table(
        execution_plan_root, "validation_security_price_targets.parquet"
    )
    accepted = pd.to_datetime(components["accepted_at"], utc=True, errors="coerce")
    if accepted.isna().any():
        raise ValidationLiquidityError("Execution components contain invalid acceptance times")
    start = accepted.min().date() - timedelta(days=cfg.calendar_buffer_days)
    end = accepted.max().date()
    sources = load_preserved_official_calendar_sources(calendar_raw_root)
    exchanges = set(components["exchange_mic"].astype(str))
    sessions, exceptions = build_historical_exchange_sessions(start, end, sources, exchanges)
    targets, events, event_sessions = calculate_pre_event_liquidity_targets(
        components,
        security_targets,
        sessions,
        config=cfg,
    )
    payload = json.dumps(
        {
            "execution_manifest_sha256": execution_manifest,
            "config": asdict(cfg),
            "target_ids": targets["price_target_id"].astype(str).tolist(),
            "event_session_hash": _frame_hash(event_sessions),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_LIQUIDITY_VERSION + "-targets-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationLiquidityResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationLiquidityError(f"Incomplete liquidity target directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    tables = {
        "massive_daily_price_targets.parquet": targets,
        "validation_liquidity_event_windows.parquet": events,
        "validation_liquidity_event_sessions.parquet": event_sessions,
        "validation_liquidity_exchange_sessions.parquet": sessions,
        "validation_liquidity_calendar_exceptions.parquet": exceptions,
        "validation_liquidity_official_sources.parquet": sources,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact_record(path, len(frame))
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_plan_version": execution_report["dataset_version"],
        "execution_plan_manifest_sha256": execution_manifest,
        "config": asdict(cfg),
        "minimum_median_dollar_volume_usd": cfg.minimum_median_dollar_volume_usd,
        "event_components": len(events),
        "price_targets": len(targets),
        "event_session_rows": len(event_sessions),
        "calendar_start": start.isoformat(),
        "calendar_end": end.isoformat(),
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "threshold_status": "frozen_before_liquidity_price_collection",
        "artifacts": artifacts,
    }
    report_path = staging / "target_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "execution_plan_manifest_sha256": execution_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationLiquidityResult(target.resolve(), report)


def build_validation_liquidity_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/validation_liquidity/raw_store"),
    output_root: Path = Path("data/processed/validation_liquidity"),
    fallback_audit_root: Path | None = None,
) -> ValidationLiquidityResult:
    """Persist the structural liquidity audit while keeping return access closed."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    cfg = ValidationLiquidityConfig(**plan_report["config"])
    targets = load_catalog_table(plan_root, "massive_daily_price_targets.parquet")
    event_windows = load_catalog_table(plan_root, "validation_liquidity_event_windows.parquet")
    event_sessions = load_catalog_table(plan_root, "validation_liquidity_event_sessions.parquet")
    primary_bars, rejections, completions = load_collected_massive_daily_bars(
        targets, raw_root=raw_root
    )
    bars = primary_bars.copy()
    bars["source_provider"] = "Massive"
    bars["adjusted_close_used"] = False
    fallback_manifest: str | None = None
    fallback_report: dict[str, Any] | None = None
    fallback_gap_audit = pd.DataFrame()
    if fallback_audit_root is not None:
        fallback_manifest = _sha256_file(fallback_audit_root / "output_manifest.json")
        fallback_report = _read_json(
            fallback_audit_root / "validation_liquidity_fallback_report.json"
        )
        if bool(fallback_report.get("outcome_values_loaded")) or bool(
            fallback_report.get("returns_computed")
        ):
            raise ValidationLiquidityError("Liquidity fallback audit opened outcomes")
        fallback_bars = load_catalog_table(
            fallback_audit_root, "validation_liquidity_fallback_bars.parquet"
        )
        fallback_gap_audit = load_catalog_table(
            fallback_audit_root, "validation_liquidity_fallback_gap_audit.parquet"
        )
        bars = pd.concat([bars, fallback_bars], ignore_index=True, sort=False)
        bars = bars.sort_values(
            ["price_target_id", "session_date", "source_provider"], kind="mergesort"
        ).drop_duplicates(["price_target_id", "session_date"], keep="first")
    evidence_resolution = _calculate_evidence_resolution(
        event_sessions,
        primary_bars,
        fallback_gap_audit,
    )
    audit, ineligible, target_audit = calculate_pre_event_liquidity_audit(
        event_windows,
        event_sessions,
        targets,
        bars,
        rejections,
        completions,
        config=cfg,
        evidence_resolution=evidence_resolution,
    )
    source_hashes = sorted(bars.get("source_sha256", pd.Series(dtype=str)).astype(str).unique())
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "audit_version": VALIDATION_LIQUIDITY_AUDIT_VERSION,
            "fallback_manifest_sha256": fallback_manifest,
            "source_hashes": source_hashes,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_LIQUIDITY_AUDIT_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationLiquidityResult(
            target.resolve(), _read_json(target / "validation_liquidity_report.json")
        )
    if target.exists():
        raise ValidationLiquidityError(f"Incomplete liquidity audit directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    collected = int(target_audit["provider_query_complete"].sum())
    eligible = int(audit["liquidity_eligible"].sum())
    primary_complete = collected == len(targets)
    resolved = int(audit["pre_event_liquidity_evidence_complete"].sum())
    evidence_complete = resolved == len(audit)
    unresolved = audit.loc[~audit["pre_event_liquidity_evidence_complete"].astype(bool)].copy()
    gates = pd.DataFrame(
        [
            _gate(
                "pre_event_threshold_frozen",
                1,
                1,
                True,
                "Capacity parameters were frozen before this price collection",
            ),
            _gate(
                "pre_event_price_collection",
                collected,
                len(targets),
                primary_complete,
                "Every frozen security span has a preserved provider response",
            ),
            _gate(
                "pre_event_liquidity_classification",
                resolved,
                len(audit),
                evidence_complete,
                "Every event has resolved primary or secondary pre-event evidence",
            ),
            _gate(
                "complete_case_liquidity_eligibility",
                eligible,
                1,
                eligible > 0,
                "At least one outcome-blind, liquidity-eligible complete case exists",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(audit),
                False,
                "No outcome or return is read during liquidity calibration",
            ),
        ]
    )
    tables = {
        "validation_liquidity_event_audit.parquet": audit,
        "validation_liquidity_ineligible_events.parquet": ineligible,
        "validation_liquidity_unresolved_events.parquet": unresolved,
        "validation_liquidity_evidence_resolution.parquet": evidence_resolution,
        "validation_liquidity_target_audit.parquet": target_audit,
        "validation_liquidity_daily_bars.parquet": bars,
        "validation_liquidity_rejections.parquet": rejections,
        "validation_liquidity_completions.parquet": completions,
        "validation_liquidity_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact_record(path, len(frame))
    status_counts = {
        str(key): int(value) for key, value in audit["liquidity_status"].value_counts().items()
    }
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_version": plan_report["dataset_version"],
        "target_plan_manifest_sha256": plan_manifest,
        "config": asdict(cfg),
        "minimum_median_dollar_volume_usd": cfg.minimum_median_dollar_volume_usd,
        "price_targets": len(targets),
        "targets_collected": collected,
        "targets_remaining": len(targets) - collected,
        "event_components": len(audit),
        "liquidity_eligible_events": eligible,
        "liquidity_ineligible_events": len(ineligible),
        "liquidity_unresolved_events": len(unresolved),
        "liquidity_resolved_events": resolved,
        "liquidity_status_counts": status_counts,
        "liquidity_gate": evidence_complete,
        "complete_case_liquidity_gate": eligible > 0,
        "eligibility_final": evidence_complete,
        "fallback_evidence_used": fallback_audit_root is not None,
        "fallback_audit_version": (
            fallback_report.get("dataset_version") if fallback_report is not None else None
        ),
        "fallback_audit_manifest_sha256": fallback_manifest,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "raw_store": str(raw_root),
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_liquidity_report.json"
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
    return ValidationLiquidityResult(target.resolve(), report)


def _liquidity_status(
    *,
    query_complete: bool,
    rejection_count: int,
    evidence_ready: bool,
    calendar_ready: bool,
    coverage_ready: bool,
    capacity_ready: bool,
) -> str:
    if not query_complete:
        return "price_query_incomplete"
    if rejection_count:
        return "parser_rejection"
    if not evidence_ready:
        return "liquidity_evidence_incomplete"
    if not calendar_ready:
        return "insufficient_calendar_history"
    if not coverage_ready and not capacity_ready:
        return "insufficient_print_coverage_and_capacity"
    if not coverage_ready:
        return "insufficient_print_coverage"
    if not capacity_ready:
        return "insufficient_median_dollar_volume"
    return "eligible"


def _calculate_evidence_resolution(
    event_sessions: pd.DataFrame,
    primary_bars: pd.DataFrame,
    fallback_gap_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Separate confirmed no-print sessions from provider-history censoring."""
    observed = pd.DataFrame(columns=["price_target_id", "session_date", "primary_print_observed"])
    if not primary_bars.empty:
        observed = (
            primary_bars[["price_target_id", "session_date"]]
            .drop_duplicates()
            .assign(primary_print_observed=True)
        )
    evidence = event_sessions[
        ["security_event_component_id", "price_target_id", "session_date"]
    ].merge(
        observed,
        on=["price_target_id", "session_date"],
        how="left",
        validate="many_to_one",
    )
    evidence["primary_print_observed"] = (
        evidence["primary_print_observed"].fillna(False).astype(bool)
    )
    if primary_bars.empty:
        evidence["primary_history_left_censored"] = True
    else:
        primary_floor = str(primary_bars["session_date"].astype(str).min())
        evidence["primary_history_left_censored"] = ~evidence["primary_print_observed"] & evidence[
            "session_date"
        ].astype(str).lt(primary_floor)
    evidence["fallback_query_complete"] = False
    if not fallback_gap_audit.empty:
        fallback = fallback_gap_audit[
            ["price_target_id", "session_date", "fallback_query_complete"]
        ].drop_duplicates()
        evidence = evidence.drop(columns="fallback_query_complete").merge(
            fallback,
            on=["price_target_id", "session_date"],
            how="left",
            validate="many_to_one",
        )
        evidence["fallback_query_complete"] = (
            evidence["fallback_query_complete"].fillna(False).astype(bool)
        )
    evidence["session_evidence_resolved"] = (
        ~evidence["primary_history_left_censored"] | evidence["fallback_query_complete"]
    )
    resolution = (
        evidence.groupby("security_event_component_id", as_index=False)
        .agg(
            pre_event_liquidity_evidence_complete=("session_evidence_resolved", "all"),
            unresolved_history_sessions=(
                "session_evidence_resolved",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            primary_history_left_censored_sessions=(
                "primary_history_left_censored",
                "sum",
            ),
            secondary_resolved_sessions=("fallback_query_complete", "sum"),
        )
        .reset_index(drop=True)
    )
    resolution["pre_event_liquidity_evidence_complete"] = resolution[
        "pre_event_liquidity_evidence_complete"
    ].astype(bool)
    return resolution


def _gate(name: str, observed: int, required: int, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "gate": name,
        "observed": observed,
        "required": required,
        "passed": passed,
        "detail": detail,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationLiquidityError(f"{label} is missing columns: {missing}")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values(list(frame.columns), kind="mergesort").to_json(
        orient="records", date_format="iso"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
