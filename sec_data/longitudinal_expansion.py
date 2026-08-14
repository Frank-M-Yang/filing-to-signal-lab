"""Pre-return longitudinal expansion design for the real filing pilot.

The design keeps the representative 28-issuer parent sample intact while
freezing a separate gross-margin research cohort using information available
strictly before the study window.  Current-event feature availability and
future returns never enter issuer eligibility.
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

LONGITUDINAL_EXPANSION_VERSION: Final[str] = "longitudinal-event-expansion-v1.1"
REPRESENTATION_FEATURES: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
)


@dataclass(frozen=True)
class LongitudinalExpansionConfig:
    """Precommitted eligibility and event-window rules."""

    feature_eligibility_cutoff: str = "2024-10-30T00:00:00+00:00"
    minimum_prior_level_events: int = 8
    minimum_prior_delta_events: int = 4
    minimum_prior_standardized_innovations: int = 4
    entry_session_offset: int = 2
    holding_sessions: int = 20
    market_buffer_calendar_days: int = 7
    version: str = LONGITUDINAL_EXPANSION_VERSION

    def validate(self) -> None:
        """Reject thresholds that could create an undertrained development cohort."""
        cutoff = pd.Timestamp(self.feature_eligibility_cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("Feature eligibility cutoff must be timezone-aware")
        if self.minimum_prior_level_events < 4:
            raise ValueError("Longitudinal eligibility requires at least four Level events")
        if self.minimum_prior_delta_events < 2:
            raise ValueError("Longitudinal eligibility requires at least two Delta events")
        if self.minimum_prior_standardized_innovations < 2:
            raise ValueError("Longitudinal eligibility requires prior OOS innovations")
        if self.entry_session_offset < 1 or self.holding_sessions < 1:
            raise ValueError("Event-window offsets must be positive")
        if self.market_buffer_calendar_days < 0:
            raise ValueError("Market collection buffer cannot be negative")


@dataclass(frozen=True)
class LongitudinalExpansionTables:
    """Pure expansion tables before immutable persistence."""

    issuer_eligibility: pd.DataFrame
    representative_events: pd.DataFrame
    feature_dense_events: pd.DataFrame
    market_collection_targets: pd.DataFrame
    event_to_market_batch: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class LongitudinalExpansionResult:
    """Immutable longitudinal expansion bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_longitudinal_expansion(
    readiness: pd.DataFrame,
    catalog: pd.DataFrame,
    feature_history: pd.DataFrame,
    sessions: pd.DataFrame,
    action_targets: pd.DataFrame,
    ticker_changes: pd.DataFrame,
    config: LongitudinalExpansionConfig | None = None,
) -> LongitudinalExpansionTables:
    """Freeze representative and feature-dense event frames without using returns."""
    cfg = config or LongitudinalExpansionConfig()
    cfg.validate()
    parent = _prepare_parent_issuers(readiness, action_targets, ticker_changes)
    eligibility = _issuer_eligibility(parent, feature_history, cfg)
    events = _candidate_events(catalog, parent, eligibility, sessions, feature_history, cfg)
    representative = events.loc[events["calendar_window_resolved"]].copy()
    representative = representative.sort_values(
        ["accepted_at", "cik", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)
    feature_dense = representative.loc[representative["feature_dense_issuer_eligible"]].reset_index(
        drop=True
    )
    market_targets, event_map = _market_collection_plan(representative, cfg)
    complete_count = int(feature_dense["representation_features_complete"].sum())
    dense_count = len(feature_dense)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "parent_issuers": len(parent),
        "representative_events": len(representative),
        "representative_issuers": int(representative["cik"].nunique()),
        "feature_dense_issuers": int(eligibility["feature_dense_issuer_eligible"].sum()),
        "feature_dense_events": dense_count,
        "feature_dense_level_events": int(feature_dense["gm_level"].notna().sum()),
        "feature_dense_delta_events": int(feature_dense["gm_delta_yoy"].notna().sum()),
        "feature_dense_raw_innovation_events": int(
            feature_dense["gm_innovation_raw"].notna().sum()
        ),
        "feature_dense_standardized_innovation_events": int(
            feature_dense["gm_innovation_std"].notna().sum()
        ),
        "feature_dense_complete_events": complete_count,
        "feature_dense_complete_coverage": complete_count / dense_count if dense_count else 0.0,
        "prospective_minimum_20_event_gate": complete_count >= 20,
        "prospective_80_percent_feature_gate": (
            complete_count / dense_count >= 0.80 if dense_count else False
        ),
        "market_collection_issuers": len(market_targets),
        "planned_next_batch_issuers": int(market_targets["planned_next_batch"].sum()),
        "planned_next_batch_events": int(
            market_targets.loc[market_targets["planned_next_batch"], "event_count"].sum()
        ),
        "planned_direct_price_issuers": int(
            market_targets.loc[
                market_targets["planned_next_batch"], "historical_ticker_query_ready"
            ].sum()
        ),
        "planned_identity_first_issuers": int(
            market_targets.loc[
                market_targets["planned_next_batch"], "identity_discovery_required"
            ].sum()
        ),
        "selection_uses_event_returns": False,
        "selection_uses_current_event_features": False,
        "selection_uses_price_coverage": False,
        "development_estimand": "gross-margin-reporting issuers with sufficient prior PIT history",
        "representative_estimand_preserved": True,
        "new_window_security_identity_gate": False,
        "market_returns_ready": False,
        "signal_effectiveness_conclusion_allowed": False,
        "config": asdict(cfg),
    }
    return LongitudinalExpansionTables(
        issuer_eligibility=eligibility,
        representative_events=representative,
        feature_dense_events=feature_dense,
        market_collection_targets=market_targets,
        event_to_market_batch=event_map,
        report=report,
    )


def build_longitudinal_expansion(
    readiness_root: Path,
    catalog_root: Path,
    feature_root: Path,
    calendar_root: Path,
    security_actions_root: Path,
    reference_root: Path,
    *,
    output_root: Path = Path("data/processed/longitudinal_expansion"),
    config: LongitudinalExpansionConfig | None = None,
) -> LongitudinalExpansionResult:
    """Build an immutable expansion and market-collection target bundle."""
    cfg = config or LongitudinalExpansionConfig()
    cfg.validate()
    input_manifests = {
        "readiness_manifest_sha256": _sha256_file(readiness_root / "output_manifest.json"),
        "catalog_manifest_sha256": _sha256_file(catalog_root / "output_manifest.json"),
        "feature_manifest_sha256": _sha256_file(feature_root / "output_manifest.json"),
        "calendar_manifest_sha256": _sha256_file(calendar_root / "output_manifest.json"),
        "security_actions_manifest_sha256": _sha256_file(
            security_actions_root / "output_manifest.json"
        ),
        "reference_manifest_sha256": _sha256_file(reference_root / "output_manifest.json"),
    }
    readiness = load_catalog_table(readiness_root, "sample_price_readiness.parquet")
    parent_ciks = (
        readiness.loc[readiness["return_study_eligible"].astype(bool), "cik"]
        .astype(str)
        .str.zfill(10)
        .unique()
        .tolist()
    )
    catalog = pd.read_parquet(
        catalog_root / "filing_catalog.parquet",
        filters=[("cik", "in", parent_ciks)],
    )
    history = load_catalog_table(feature_root, "quarterly_feature_history.parquet")
    sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")
    action_targets = load_catalog_table(
        security_actions_root, "massive_security_actions_targets.parquet"
    )
    ticker_changes = load_catalog_table(security_actions_root, "ticker_change_observations.parquet")
    tables = calculate_longitudinal_expansion(
        readiness,
        catalog,
        history,
        sessions,
        action_targets,
        ticker_changes,
        cfg,
    )
    payload = json.dumps({**input_manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return LongitudinalExpansionResult(
            target.resolve(), _read_json(target / "longitudinal_expansion_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete longitudinal expansion directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "issuer_eligibility_audit.parquet": tables.issuer_eligibility,
        "representative_event_frame.parquet": tables.representative_events,
        "feature_dense_event_panel.parquet": tables.feature_dense_events,
        "issuer_market_collection_targets.parquet": tables.market_collection_targets,
        "event_to_market_batch.parquet": tables.event_to_market_batch,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "longitudinal_expansion_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    reference_report = _read_json(reference_root / "massive_reference_report.json")
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **input_manifests,
        "reference_version": reference_report["dataset_version"],
        "artifacts": artifacts,
    }
    report_path = staging / "longitudinal_expansion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                **input_manifests,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return LongitudinalExpansionResult(target.resolve(), report)


def _prepare_parent_issuers(
    readiness: pd.DataFrame,
    action_targets: pd.DataFrame,
    ticker_changes: pd.DataFrame,
) -> pd.DataFrame:
    readiness_required = {
        "target_id",
        "cik",
        "sampling_hash",
        "stage1_inclusion_probability",
        "stage1_sampling_weight",
        "sic",
        "sic_description",
        "return_study_eligible",
    }
    action_required = {
        "action_target_id",
        "target_id",
        "selected_ticker",
        "selected_exchange_mic",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
        "selected_reference_source_sha256",
        "currency_name",
    }
    _require_columns(readiness, readiness_required, "price readiness")
    _require_columns(action_targets, action_required, "security action targets")
    selected_columns = [
        "target_id",
        "cik",
        "sampling_hash",
        "stage1_inclusion_probability",
        "stage1_sampling_weight",
        "sic",
        "sic_description",
        "return_study_eligible",
    ]
    selected = readiness.loc[
        readiness["return_study_eligible"].astype(bool), selected_columns
    ].copy()
    selected["cik"] = selected["cik"].astype(str).str.zfill(10)
    action = action_targets[list(action_required)].copy()
    action["cik"] = action_targets["cik"].astype(str).str.zfill(10)
    parent = selected.merge(action, on=["target_id", "cik"], how="inner", validate="one_to_one")
    if len(parent) != len(selected) or parent["cik"].duplicated().any():
        raise FilingCatalogError("Parent return issuers do not tie one-to-one to security mappings")
    changes = ticker_changes.copy()
    if changes.empty:
        changes = pd.DataFrame(columns=["action_target_id", "event_date", "new_ticker"])
    _require_columns(changes, {"action_target_id", "event_date", "new_ticker"}, "ticker changes")
    changes = changes.merge(
        parent[["action_target_id", "selected_ticker"]],
        on="action_target_id",
        how="inner",
        validate="many_to_one",
    )
    changes = changes.loc[changes["new_ticker"].eq(changes["selected_ticker"])].copy()
    changes["ticker_evidence_from"] = pd.to_datetime(changes["event_date"], errors="coerce")
    inception = changes.groupby("action_target_id", as_index=False).agg(
        ticker_evidence_from=("ticker_evidence_from", "min")
    )
    parent = parent.merge(inception, on="action_target_id", how="left", validate="one_to_one")
    return parent


def _issuer_eligibility(
    parent: pd.DataFrame,
    history: pd.DataFrame,
    config: LongitudinalExpansionConfig,
) -> pd.DataFrame:
    _require_columns(
        history,
        {
            "company_id",
            "actual_available_at",
            "analysis_ready_level",
            "analysis_ready_delta",
            "analysis_ready_innovation_std",
        },
        "feature history",
    )
    cutoff = pd.Timestamp(config.feature_eligibility_cutoff)
    histories = history.copy()
    histories["company_id"] = histories["company_id"].astype(str).str.zfill(10)
    histories["actual_available_at"] = pd.to_datetime(
        histories["actual_available_at"], utc=True, errors="coerce"
    )
    prior = histories.loc[histories["actual_available_at"] < cutoff].copy()
    prior["analysis_ready_level"] = prior["analysis_ready_level"].fillna(False).astype(bool)
    prior["analysis_ready_delta"] = prior["analysis_ready_delta"].fillna(False).astype(bool)
    prior["analysis_ready_innovation_std"] = (
        prior["analysis_ready_innovation_std"].fillna(False).astype(bool)
    )
    summary = prior.groupby("company_id", as_index=False).agg(
        prior_level_events=("analysis_ready_level", "sum"),
        prior_delta_events=("analysis_ready_delta", "sum"),
        prior_standardized_innovation_events=("analysis_ready_innovation_std", "sum"),
        prior_feature_history_end=("actual_available_at", "max"),
    )
    result = parent.merge(summary, left_on="cik", right_on="company_id", how="left")
    count_columns = [
        "prior_level_events",
        "prior_delta_events",
        "prior_standardized_innovation_events",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["feature_dense_issuer_eligible"] = (
        (result["prior_level_events"] >= config.minimum_prior_level_events)
        & (result["prior_delta_events"] >= config.minimum_prior_delta_events)
        & (
            result["prior_standardized_innovation_events"]
            >= config.minimum_prior_standardized_innovations
        )
    )
    result["eligibility_uses_current_event_features"] = False
    result["eligibility_uses_event_returns"] = False
    result["feature_eligibility_cutoff"] = cutoff.isoformat()
    result["eligibility_status"] = np.where(
        result["feature_dense_issuer_eligible"],
        "eligible_from_strictly_prior_history",
        "insufficient_strictly_prior_feature_history",
    )
    return result.drop(columns=["company_id"])


def _candidate_events(
    catalog: pd.DataFrame,
    parent: pd.DataFrame,
    eligibility: pd.DataFrame,
    sessions: pd.DataFrame,
    history: pd.DataFrame,
    config: LongitudinalExpansionConfig,
) -> pd.DataFrame:
    _require_columns(
        catalog,
        {"cik", "accession_number", "form", "report_date", "accepted_at"},
        "filing catalog",
    )
    _require_columns(
        sessions,
        {"exchange_mic", "session_date", "regular_open_at", "regular_close_at"},
        "exchange sessions",
    )
    cutoff = pd.Timestamp(config.feature_eligibility_cutoff)
    filings = catalog.copy()
    filings["cik"] = filings["cik"].astype(str).str.zfill(10)
    filings["accepted_at"] = pd.to_datetime(filings["accepted_at"], utc=True, errors="coerce")
    filings = filings.loc[
        filings["form"].isin(["10-Q", "10-K"])
        & filings["accepted_at"].notna()
        & (filings["accepted_at"] >= cutoff)
        & filings["cik"].isin(parent["cik"])
    ].copy()
    if filings["accession_number"].duplicated().any():
        raise FilingCatalogError("Longitudinal candidate accessions must be unique")
    issuer_columns = [
        "cik",
        "target_id",
        "sampling_hash",
        "stage1_inclusion_probability",
        "stage1_sampling_weight",
        "sic",
        "sic_description",
        "action_target_id",
        "selected_ticker",
        "selected_exchange_mic",
        "selected_provider_security_id",
        "selected_provider_security_id_kind",
        "selected_reference_source_sha256",
        "currency_name",
        "ticker_evidence_from",
    ]
    issuer_eligibility = eligibility[
        [
            "cik",
            "feature_dense_issuer_eligible",
            "prior_level_events",
            "prior_delta_events",
            "prior_standardized_innovation_events",
        ]
    ]
    events = filings.merge(parent[issuer_columns], on="cik", how="inner", validate="many_to_one")
    events = events.merge(issuer_eligibility, on="cik", how="left", validate="many_to_one")
    windows = _resolve_event_windows(events, sessions, config)
    events = pd.concat([events.reset_index(drop=True), windows], axis="columns")
    feature_columns = [
        "filing_accession",
        "actual_available_at",
        *REPRESENTATION_FEATURES,
        "analysis_ready_level",
        "analysis_ready_delta",
        "analysis_ready_innovation_raw",
        "analysis_ready_innovation_std",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_training_observations",
        "forecast_error_history_end",
        "forecast_sigma_observations",
    ]
    _require_columns(history, set(feature_columns), "feature history")
    if history["filing_accession"].duplicated().any():
        raise FilingCatalogError("Feature history accessions must be unique")
    events = events.merge(
        history[feature_columns],
        left_on="accession_number",
        right_on="filing_accession",
        how="left",
        validate="one_to_one",
    )
    events["representation_features_complete"] = (
        events[list(REPRESENTATION_FEATURES)].notna().all(axis="columns")
    )
    events["feature_status"] = "no_quarterly_level_event"
    events.loc[events["gm_level"].notna(), "feature_status"] = "delta_unavailable"
    events.loc[events["gm_delta_yoy"].notna(), "feature_status"] = "forecast_unavailable"
    events.loc[
        events["gm_innovation_raw"].notna() & events["gm_innovation_std"].isna(),
        "feature_status",
    ] = "forecast_sigma_unavailable"
    events.loc[events["representation_features_complete"], "feature_status"] = "complete_case_ready"
    ticker_start = pd.to_datetime(events["ticker_evidence_from"], errors="coerce")
    events["selected_ticker_asof_evidence"] = ticker_start.notna() & (
        events["accepted_at"].dt.tz_localize(None) >= ticker_start
    )
    events["new_window_security_identity_gate"] = False
    events["security_identity_revalidation_required"] = True
    events["selection_uses_current_event_features"] = False
    events["selection_uses_event_returns"] = False
    events["longitudinal_event_id"] = events.apply(
        lambda row: _stable_hash(
            {
                "accession_number": str(row["accession_number"]),
                "accepted_at": pd.Timestamp(row["accepted_at"]).isoformat(),
                "config_version": config.version,
            }
        ),
        axis=1,
    )
    return events


def _resolve_event_windows(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    config: LongitudinalExpansionConfig,
) -> pd.DataFrame:
    session_frame = sessions.copy()
    session_frame["regular_open_value"] = pd.to_datetime(
        session_frame["regular_open_at"], utc=True, errors="coerce"
    )
    session_frame["regular_close_value"] = pd.to_datetime(
        session_frame["regular_close_at"], utc=True, errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    entry_index = config.entry_session_offset - 1
    exit_index = entry_index + config.holding_sessions - 1
    for event in events.itertuples(index=False):
        accepted_at = pd.Timestamp(str(event.accepted_at))
        exchange = session_frame.loc[
            session_frame["exchange_mic"].eq(str(event.selected_exchange_mic))
        ].sort_values("regular_open_value")
        future = exchange.loc[exchange["regular_open_value"] > accepted_at].reset_index(drop=True)
        resolved = len(future) > exit_index
        if resolved:
            entry = future.iloc[entry_index]
            exit_session = future.iloc[exit_index]
            entry_date = str(entry["session_date"])
            entry_at = str(entry["regular_open_at"])
            exit_date = str(exit_session["session_date"])
            exit_at = str(exit_session["regular_close_at"])
        else:
            entry_date = None
            entry_at = None
            exit_date = None
            exit_at = None
        rows.append(
            {
                "entry_session_offset": config.entry_session_offset,
                "holding_sessions": config.holding_sessions,
                "calendar_window_resolved": resolved,
                "formal_entry_session_date": entry_date,
                "formal_entry_open_at": entry_at,
                "formal_exit_session_date": exit_date,
                "formal_exit_close_at": exit_at,
            }
        )
    return pd.DataFrame(rows)


def _market_collection_plan(
    representative: pd.DataFrame,
    config: LongitudinalExpansionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    event_maps: list[dict[str, Any]] = []
    for cik, events in representative.groupby("cik", sort=True):
        accepted_min = pd.to_datetime(events["accepted_at"], utc=True).min()
        exit_max = pd.to_datetime(events["formal_exit_close_at"], utc=True).max()
        buffer = pd.Timedelta(days=config.market_buffer_calendar_days)
        window_start = (accepted_min - buffer).date().isoformat()
        window_end = (exit_max + buffer).date().isoformat()
        parent = events.iloc[0]
        batch_id = _stable_hash(
            {
                "cik": str(cik),
                "security_id": str(parent["selected_provider_security_id"]),
                "window_start": window_start,
                "window_end": window_end,
                "adjusted": False,
                "config_version": config.version,
            }
        )
        dense = bool(parent["feature_dense_issuer_eligible"])
        ticker_ready = bool(events["selected_ticker_asof_evidence"].all())
        rows.append(
            {
                "market_batch_id": batch_id,
                "cik": str(cik),
                "selected_ticker": parent["selected_ticker"],
                "selected_exchange_mic": parent["selected_exchange_mic"],
                "selected_provider_security_id": parent["selected_provider_security_id"],
                "selected_provider_security_id_kind": parent["selected_provider_security_id_kind"],
                "selected_reference_source_sha256": parent["selected_reference_source_sha256"],
                "currency_name": parent["currency_name"],
                "market_window_start": window_start,
                "market_window_end": window_end,
                "adjusted": False,
                "event_count": len(events),
                "complete_feature_events": int(events["representation_features_complete"].sum()),
                "feature_dense_issuer_eligible": dense,
                "planned_next_batch": dense,
                "historical_ticker_query_ready": ticker_ready,
                "identity_discovery_required": not ticker_ready,
                "daily_bar_queries_planned": int(ticker_ready),
                "reference_endpoint_queries_planned": 2,
                "ticker_events_snapshot_reuse_planned": True,
                "split_queries_planned": 1,
                "dividend_queries_planned": 1,
                "delisting_outcome_query_planned": 1,
                "new_window_security_identity_gate": False,
                "return_study_gate": False,
            }
        )
        for event_id in events["longitudinal_event_id"]:
            event_maps.append(
                {
                    "market_batch_id": batch_id,
                    "longitudinal_event_id": event_id,
                    "cik": str(cik),
                    "planned_next_batch": dense,
                }
            )
    targets = pd.DataFrame(rows).sort_values(
        ["planned_next_batch", "cik"], ascending=[False, True], kind="mergesort"
    )
    targets["collection_order"] = np.arange(1, len(targets) + 1)
    return targets.reset_index(drop=True), pd.DataFrame(event_maps)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


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
