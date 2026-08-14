"""Auditable same-security ticker-transition repair for OB to TEAD.

The repair is a data-quality operation on a pre-specified event.  It validates
that the old and new symbols resolve to the same point-in-time security ids,
collects the missing new-symbol prices and actions, and emits additions plus an
identity-readiness override without changing any signal or return definition.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

TICKER_TRANSITION_REPAIR_VERSION: Final[str] = "ticker-transition-repair-v1"


@dataclass(frozen=True)
class TickerTransitionRepairConfig:
    """Explicit security and boundary assumptions for the discovered transition."""

    cik: str = "0001454938"
    from_ticker: str = "OB"
    to_ticker: str = "TEAD"
    transition_date: str = "2025-06-10"
    minimum_price_scale_ratio: float = 0.20
    maximum_price_scale_ratio: float = 5.00
    adjusted: bool = False
    version: str = TICKER_TRANSITION_REPAIR_VERSION

    def validate(self) -> None:
        """Reject a repair that is not the documented same-security transition."""
        if self.cik != "0001454938" or (self.from_ticker, self.to_ticker) != ("OB", "TEAD"):
            raise ValueError("Repair v1 is frozen to the documented OB-to-TEAD transition")
        parsed = pd.Timestamp(self.transition_date)
        if parsed.date().isoformat() != self.transition_date:
            raise ValueError("Transition date must be an ISO calendar date")
        if self.adjusted:
            raise ValueError("Transition repair requires unadjusted prices")
        if not 0.0 < self.minimum_price_scale_ratio < 1.0:
            raise ValueError("Minimum price-scale ratio must be in (0, 1)")
        if self.maximum_price_scale_ratio <= 1.0:
            raise ValueError("Maximum price-scale ratio must exceed one")


@dataclass(frozen=True)
class TickerTransitionRepairPlanResult:
    """Immutable price/action collection plan for one ticker transition."""

    output_root: Path
    price_plan_root: Path
    action_plan_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class TickerTransitionReconciliationTables:
    """Pure reconciled additions and readiness override."""

    price_additions: pd.DataFrame
    action_additions: pd.DataFrame
    identity_overrides: pd.DataFrame
    evidence: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class TickerTransitionReconciliationResult:
    """Immutable transition-repair audit bundle."""

    output_root: Path
    report: dict[str, Any]


def build_ticker_transition_repair_plan(
    market_plan_root: Path,
    original_price_root: Path,
    original_action_root: Path,
    calendar_root: Path,
    *,
    output_root: Path = Path("data/market/longitudinal_study/transition_repairs"),
    config: TickerTransitionRepairConfig | None = None,
) -> TickerTransitionRepairPlanResult:
    """Freeze one TEAD collection target from preserved OB transition evidence."""
    cfg = config or TickerTransitionRepairConfig()
    cfg.validate()
    manifests = {
        "market_plan_manifest_sha256": _sha256_file(market_plan_root / "output_manifest.json"),
        "original_price_manifest_sha256": _sha256_file(
            original_price_root / "output_manifest.json"
        ),
        "original_action_manifest_sha256": _sha256_file(
            original_action_root / "output_manifest.json"
        ),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
    }
    price_targets = load_catalog_table(
        market_plan_root / "prices", "massive_daily_price_targets.parquet"
    )
    action_targets = load_catalog_table(
        market_plan_root / "actions", "massive_security_actions_targets.parquet"
    )
    ticker_events = load_catalog_table(original_action_root, "ticker_change_observations.parquet")
    original_price = _unique_cik(price_targets, cfg.cik, "price target")
    original_action = _unique_cik(action_targets, cfg.cik, "action target")
    observed_transition = ticker_events.loc[
        ticker_events["action_target_id"].eq(str(original_action["action_target_id"]))
        & ticker_events["event_date"].eq(cfg.transition_date)
        & ticker_events["new_ticker"].eq(cfg.to_ticker)
    ]
    if len(observed_transition) != 1:
        raise FilingCatalogError("The preserved action audit lacks one exact OB-to-TEAD event")
    window_end = str(original_price["price_window_end"])
    price_target_id = _stable_hash(
        {
            "cik": cfg.cik,
            "ticker": cfg.to_ticker,
            "window_start": cfg.transition_date,
            "window_end": window_end,
            "adjusted": False,
            "config_version": cfg.version,
        }
    )
    repair_target_id = _stable_hash(
        {
            "original_target_id": str(original_price["target_id"]),
            "repair_price_target_id": price_target_id,
            "config_version": cfg.version,
        }
    )
    price_frame = pd.DataFrame(
        [
            {
                "price_target_id": price_target_id,
                "price_acquisition_order": 1,
                "target_id": repair_target_id,
                "cik": cfg.cik,
                "accepted_at": f"{cfg.transition_date}T13:30:00+00:00",
                "query_date": cfg.transition_date,
                "form": "TICKER_TRANSITION_REPAIR",
                "accession_number": repair_target_id,
                "selected_ticker": cfg.to_ticker,
                "selected_exchange_mic": str(original_price["selected_exchange_mic"]),
                "selected_provider_security_id": str(
                    original_price["selected_provider_security_id"]
                ),
                "selected_provider_security_id_kind": str(
                    original_price["selected_provider_security_id_kind"]
                ),
                "reference_source_sha256": str(original_price["reference_source_sha256"]),
                "currency_name": str(original_price["currency_name"]),
                "price_window_start": cfg.transition_date,
                "price_window_end": window_end,
                "adjusted": False,
                "entry_session_offset": 2,
                "holding_sessions": 20,
            }
        ]
    )
    action_target_id = _stable_hash(
        {
            "repair_price_target_id": price_target_id,
            "share_class_figi": str(original_action["selected_share_class_figi"]),
            "window_start": cfg.transition_date,
            "window_end": window_end,
            "config_version": cfg.version,
        }
    )
    action_frame = pd.DataFrame(
        [
            {
                "action_target_id": action_target_id,
                "action_acquisition_order": 1,
                "price_target_id": price_target_id,
                "target_id": repair_target_id,
                "cik": cfg.cik,
                "accepted_at": f"{cfg.transition_date}T13:30:00+00:00",
                "query_date": cfg.transition_date,
                "form": "TICKER_TRANSITION_REPAIR",
                "accession_number": repair_target_id,
                "selected_ticker": cfg.to_ticker,
                "selected_exchange_mic": str(original_action["selected_exchange_mic"]),
                "selected_provider_security_id": str(
                    original_action["selected_provider_security_id"]
                ),
                "selected_provider_security_id_kind": str(
                    original_action["selected_provider_security_id_kind"]
                ),
                "selected_share_class_figi": str(original_action["selected_share_class_figi"]),
                "selected_composite_figi": str(original_action["selected_composite_figi"]),
                "selected_reference_source_sha256": str(
                    original_action["selected_reference_source_sha256"]
                ),
                "currency_name": str(original_action["currency_name"]),
                "price_window_start": cfg.transition_date,
                "price_window_end": window_end,
                "formal_entry_session_date": cfg.transition_date,
                "formal_exit_session_date": window_end,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "cik": cfg.cik,
                "from_ticker": cfg.from_ticker,
                "to_ticker": cfg.to_ticker,
                "transition_date": cfg.transition_date,
                "original_price_target_id": str(original_price["price_target_id"]),
                "repair_price_target_id": price_target_id,
                "original_action_target_id": str(original_action["action_target_id"]),
                "repair_action_target_id": action_target_id,
                "security_id": str(original_action["selected_provider_security_id"]),
                "share_class_figi": str(original_action["selected_share_class_figi"]),
                "composite_figi": str(original_action["selected_composite_figi"]),
                "exchange_mic": str(original_action["selected_exchange_mic"]),
                "window_end": window_end,
                "ticker_event_source_sha256": str(observed_transition.iloc[0]["source_sha256"]),
            }
        ]
    )
    payload = json.dumps(
        {**manifests, "config": asdict(cfg), "mapping": mapping.to_dict(orient="records")},
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-plan-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return TickerTransitionRepairPlanResult(
            target.resolve(),
            (target / "prices").resolve(),
            (target / "actions").resolve(),
            _read_json(target / "transition_repair_plan_report.json"),
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete ticker-transition plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    price_root = staging / "prices"
    action_root = staging / "actions"
    price_root.mkdir()
    action_root.mkdir()
    original_action_report = _read_json(
        original_action_root / "massive_security_actions_report.json"
    )
    calendar_report = _read_json(calendar_root / "exchange_calendar_report.json")
    price_version, price_manifest = _write_price_plan(
        price_root, price_frame, dataset_version, original_action_report
    )
    _write_action_plan(
        action_root,
        action_frame,
        dataset_version,
        original_action_report,
        price_version,
        price_manifest,
        calendar_report,
        manifests["calendar_manifest_sha256"],
    )
    mapping_path = staging / "transition_mapping.parquet"
    mapping.to_parquet(mapping_path, index=False)
    artifact = _artifact_record(mapping_path, len(mapping))
    report = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "config": asdict(cfg),
        "price_targets": 1,
        "action_targets": 1,
        "planned_queries": 6,
        "selection_uses_event_returns": False,
        "artifacts": {mapping_path.name: artifact},
    }
    report_path = staging / "transition_repair_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **manifests,
                "price_plan_manifest_sha256": price_manifest,
                "action_plan_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
                "artifacts": {mapping_path.name: artifact},
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return TickerTransitionRepairPlanResult(
        target.resolve(),
        (target / "prices").resolve(),
        (target / "actions").resolve(),
        report,
    )


def calculate_ticker_transition_reconciliation(
    mapping: pd.DataFrame,
    original_bars: pd.DataFrame,
    repair_bars: pd.DataFrame,
    repair_identity: pd.DataFrame,
    repair_ticker_events: pd.DataFrame,
    original_action_audit: pd.DataFrame,
    repair_action_audit: pd.DataFrame,
    repair_actions: pd.DataFrame,
    sessions: pd.DataFrame,
    config: TickerTransitionRepairConfig | None = None,
) -> TickerTransitionReconciliationTables:
    """Validate the transition and create source-preserving ledger additions."""
    cfg = config or TickerTransitionRepairConfig()
    cfg.validate()
    if len(mapping) != 1:
        raise FilingCatalogError("Ticker-transition mapping must contain exactly one row")
    transition = mapping.iloc[0]
    original_price_id = str(transition["original_price_target_id"])
    repair_price_id = str(transition["repair_price_target_id"])
    original_action_id = str(transition["original_action_target_id"])
    repair_action_id = str(transition["repair_action_target_id"])
    old = original_bars.loc[original_bars["price_target_id"].eq(original_price_id)].copy()
    new = repair_bars.loc[repair_bars["price_target_id"].eq(repair_price_id)].copy()
    if old.empty or new.empty:
        raise FilingCatalogError("Transition repair requires both old- and new-symbol bars")
    old_before = old.loc[old["session_date"].lt(cfg.transition_date)].sort_values("session_date")
    new_after = new.loc[new["session_date"].ge(cfg.transition_date)].sort_values("session_date")
    if old_before.empty or new_after.empty:
        raise FilingCatalogError("Transition repair lacks bars on one side of the boundary")
    old_last = old_before.iloc[-1]
    new_first = new_after.iloc[0]
    exchange_sessions = sessions.loc[
        sessions["exchange_mic"].eq(str(transition["exchange_mic"]))
    ].sort_values("session_date")
    transition_matches = exchange_sessions.index[
        exchange_sessions["session_date"].eq(cfg.transition_date)
    ].tolist()
    if len(transition_matches) != 1:
        raise FilingCatalogError("Transition date is not one unique exchange session")
    position = exchange_sessions.index.get_loc(transition_matches[0])
    if not isinstance(position, int) or position < 1:
        raise FilingCatalogError("Transition session has no prior exchange session")
    prior_session = str(exchange_sessions.iloc[position - 1]["session_date"])
    boundary_sessions_ready = (
        str(old_last["session_date"]) == prior_session
        and str(new_first["session_date"]) == cfg.transition_date
    )
    ratio = _positive_float(new_first["open_unadjusted"], "new-symbol open") / _positive_float(
        old_last["close_unadjusted"], "old-symbol close"
    )
    price_scale_ready = cfg.minimum_price_scale_ratio <= ratio <= cfg.maximum_price_scale_ratio
    overlap = set(old["session_date"]).intersection(set(new["session_date"]))
    no_bar_overlap = not overlap

    identity = repair_identity.loc[
        repair_identity["action_target_id"].eq(repair_action_id)
        & repair_identity["ticker"].eq(cfg.to_ticker)
        & repair_identity["provider_security_id"].eq(str(transition["security_id"]))
        & repair_identity["share_class_figi"].eq(str(transition["share_class_figi"]))
        & repair_identity["composite_figi"].eq(str(transition["composite_figi"]))
        & repair_identity["exchange_mic"].eq(str(transition["exchange_mic"]))
        & repair_identity["security_type"].eq("CS")
    ]
    start_matches = identity.loc[identity["query_kind"].eq("reference_start")]
    end_matches = identity.loc[identity["query_kind"].eq("reference_end")]
    endpoint_identity_ready = len(start_matches) == 1 and len(end_matches) == 1
    events = repair_ticker_events.loc[
        repair_ticker_events["action_target_id"].eq(repair_action_id)
    ].copy()
    boundary_events = events.loc[
        events["event_date"].eq(cfg.transition_date) & events["new_ticker"].eq(cfg.to_ticker)
    ]
    later_events = events.loc[events["event_date"].gt(cfg.transition_date)]
    ticker_timeline_ready = len(boundary_events) == 1 and later_events.empty
    original_audit = original_action_audit.loc[
        original_action_audit["action_target_id"].eq(original_action_id)
    ]
    repair_audit = repair_action_audit.loc[
        repair_action_audit["action_target_id"].eq(repair_action_id)
    ]
    if len(original_audit) != 1 or len(repair_audit) != 1:
        raise FilingCatalogError("Transition action audits are missing or duplicated")
    corporate_actions_ready = bool(original_audit.iloc[0]["corporate_actions_ready"]) and bool(
        repair_audit.iloc[0]["corporate_actions_ready"]
    )
    transition_gate = all(
        (
            boundary_sessions_ready,
            price_scale_ready,
            no_bar_overlap,
            endpoint_identity_ready,
            ticker_timeline_ready,
            corporate_actions_ready,
        )
    )
    price_additions = new_after.copy()
    price_additions["repair_price_target_id"] = price_additions["price_target_id"]
    price_additions["price_target_id"] = original_price_id
    price_additions["ticker_transition_date"] = cfg.transition_date
    price_additions["ticker_at_session"] = cfg.to_ticker
    action_additions = repair_actions.loc[
        repair_actions["action_target_id"].eq(repair_action_id)
    ].copy()
    if not action_additions.empty:
        action_additions["repair_action_target_id"] = action_additions["action_target_id"]
        action_additions["repair_price_target_id"] = action_additions["price_target_id"]
        action_additions["action_target_id"] = original_action_id
        action_additions["price_target_id"] = original_price_id
        action_additions["target_id"] = str(original_audit.iloc[0]["target_id"])
    overrides = pd.DataFrame(
        [
            {
                "action_target_id": original_action_id,
                "historical_symbol_interval_ready": transition_gate,
                "historical_symbol_interval_status": (
                    "validated_ticker_transition_same_security_id"
                    if transition_gate
                    else "ticker_transition_repair_failed"
                ),
                "corporate_actions_ready": corporate_actions_ready,
                "stable_security_id": str(transition["security_id"]),
                "transition_date": cfg.transition_date,
                "from_ticker": cfg.from_ticker,
                "to_ticker": cfg.to_ticker,
                "repair_action_target_id": repair_action_id,
                "ticker_event_source_sha256": str(transition["ticker_event_source_sha256"]),
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "evidence_check": "consecutive_exchange_sessions",
                "passed": boundary_sessions_ready,
                "observed": f"{old_last['session_date']}->{new_first['session_date']}",
            },
            {
                "evidence_check": "price_scale_ratio",
                "passed": price_scale_ready,
                "observed": f"{ratio:.12g}",
            },
            {
                "evidence_check": "no_overlapping_symbol_bars",
                "passed": no_bar_overlap,
                "observed": str(len(overlap)),
            },
            {
                "evidence_check": "same_security_endpoint_identity",
                "passed": endpoint_identity_ready,
                "observed": f"start={len(start_matches)};end={len(end_matches)}",
            },
            {
                "evidence_check": "boundary_ticker_event_no_later_change",
                "passed": ticker_timeline_ready,
                "observed": f"boundary={len(boundary_events)};later={len(later_events)}",
            },
            {
                "evidence_check": "corporate_actions_complete_both_symbols",
                "passed": corporate_actions_ready,
                "observed": str(corporate_actions_ready),
            },
        ]
    )
    report = {
        "contract_version": cfg.version,
        "transition_gate": transition_gate,
        "cik": cfg.cik,
        "from_ticker": cfg.from_ticker,
        "to_ticker": cfg.to_ticker,
        "transition_date": cfg.transition_date,
        "security_id": str(transition["security_id"]),
        "old_last_session": str(old_last["session_date"]),
        "new_first_session": str(new_first["session_date"]),
        "old_last_close": float(old_last["close_unadjusted"]),
        "new_first_open": float(new_first["open_unadjusted"]),
        "overnight_price_ratio": ratio,
        "overnight_return": ratio - 1.0,
        "price_addition_rows": len(price_additions),
        "action_addition_rows": len(action_additions),
        "selection_uses_event_returns": False,
        "signal_values_changed": False,
        "config": asdict(cfg),
    }
    return TickerTransitionReconciliationTables(
        price_additions,
        action_additions,
        overrides,
        evidence,
        report,
    )


def build_ticker_transition_reconciliation(
    plan_root: Path,
    original_price_root: Path,
    original_action_root: Path,
    repair_price_root: Path,
    repair_action_root: Path,
    calendar_root: Path,
    *,
    output_root: Path = Path("data/processed/ticker_transition_repair"),
    config: TickerTransitionRepairConfig | None = None,
) -> TickerTransitionReconciliationResult:
    """Persist the source-preserving transition additions and identity override."""
    cfg = config or TickerTransitionRepairConfig()
    cfg.validate()
    manifests = {
        "plan_manifest_sha256": _sha256_file(plan_root / "output_manifest.json"),
        "original_price_manifest_sha256": _sha256_file(
            original_price_root / "output_manifest.json"
        ),
        "original_action_manifest_sha256": _sha256_file(
            original_action_root / "output_manifest.json"
        ),
        "repair_price_manifest_sha256": _sha256_file(repair_price_root / "output_manifest.json"),
        "repair_action_manifest_sha256": _sha256_file(repair_action_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
    }
    price_report = _read_json(repair_price_root / "massive_price_report.json")
    action_report = _read_json(repair_action_root / "massive_security_actions_report.json")
    if price_report["target_plan_manifest_sha256"] != _sha256_file(
        plan_root / "prices" / "output_manifest.json"
    ):
        raise FilingCatalogError("Repair price audit does not match its frozen target")
    if action_report["target_plan_manifest_sha256"] != _sha256_file(
        plan_root / "actions" / "output_manifest.json"
    ):
        raise FilingCatalogError("Repair action audit does not match its frozen target")
    mapping = load_catalog_table(plan_root, "transition_mapping.parquet")
    original_bars = load_catalog_table(
        original_price_root, "massive_daily_price_observations.parquet"
    )
    repair_bars = load_catalog_table(repair_price_root, "massive_daily_price_observations.parquet")
    repair_identity = load_catalog_table(
        repair_action_root, "security_reference_interval_observations.parquet"
    )
    repair_events = load_catalog_table(repair_action_root, "ticker_change_observations.parquet")
    original_action_audit = load_catalog_table(
        original_action_root, "sample_security_actions_audit.parquet"
    )
    repair_action_audit = load_catalog_table(
        repair_action_root, "sample_security_actions_audit.parquet"
    )
    repair_actions = load_catalog_table(repair_action_root, "normalized_corporate_actions.parquet")
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    tables = calculate_ticker_transition_reconciliation(
        mapping,
        original_bars,
        repair_bars,
        repair_identity,
        repair_events,
        original_action_audit,
        repair_action_audit,
        repair_actions,
        sessions,
        cfg,
    )
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return TickerTransitionReconciliationResult(
            target.resolve(), _read_json(target / "ticker_transition_repair_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete ticker-transition repair exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "reconciled_daily_price_additions.parquet": tables.price_additions,
        "reconciled_corporate_action_additions.parquet": tables.action_additions,
        "identity_readiness_overrides.parquet": tables.identity_overrides,
        "transition_evidence_checks.parquet": tables.evidence,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifests,
        "artifacts": artifacts,
    }
    report_path = staging / "ticker_transition_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return TickerTransitionReconciliationResult(target.resolve(), report)


def _unique_cik(frame: pd.DataFrame, cik: str, label: str) -> pd.Series[Any]:
    selected = frame.loc[frame["cik"].astype(str).str.zfill(10).eq(cik)]
    if len(selected) != 1:
        raise FilingCatalogError(f"Ticker repair requires one {label} for CIK {cik}")
    return selected.iloc[0]


def _write_price_plan(
    root: Path,
    targets: pd.DataFrame,
    parent_version: str,
    original_action_report: dict[str, Any],
) -> tuple[str, str]:
    version = parent_version + "-prices"
    target_path = root / "massive_daily_price_targets.parquet"
    targets.to_parquet(target_path, index=False)
    artifact = _artifact_record(target_path, len(targets))
    report = {
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": original_action_report["reference_version"],
        "reference_manifest_sha256": original_action_report["reference_manifest_sha256"],
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
    original_action_report: dict[str, Any],
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
        "reference_version": original_action_report["reference_version"],
        "reference_manifest_sha256": original_action_report["reference_manifest_sha256"],
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


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FilingCatalogError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FilingCatalogError(f"{label} must be finite and positive")
    return result


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
