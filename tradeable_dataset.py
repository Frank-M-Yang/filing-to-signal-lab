"""Point-in-time feature and return-label contract for the first local alpha model."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

TRADEABLE_DATASET_VERSION: Final[str] = "tradeable-dataset-v1"
PREREGISTRATION_VERSION: Final[str] = "tradeable-dataset-preregistration-v1"

FILING_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
    "form_is_10k",
)
MARKET_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "issuer_momentum_5d",
    "issuer_momentum_20d",
    "issuer_volatility_20d",
    "issuer_volume_ratio_5_20",
    "market_momentum_5d",
    "market_momentum_20d",
    "relative_momentum_20d",
    "market_beta_20d",
)
COMBINED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *FILING_FEATURE_COLUMNS,
    *MARKET_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class TradeableDatasetConfig:
    """Frozen feature, label, and evidence-role conventions."""

    primary_horizon_sessions: int = 20
    secondary_horizon_sessions: tuple[int, ...] = (5,)
    minimum_market_history_sessions: int = 21
    momentum_windows: tuple[int, ...] = (5, 20)
    volatility_window_sessions: int = 20
    volume_short_window_sessions: int = 5
    volume_long_window_sessions: int = 20
    beta_window_sessions: int = 20
    validation_start: str = "2022-01-01T00:00:00Z"
    sealed_holdout_start: str = "2026-08-28T00:00:00Z"
    version: str = TRADEABLE_DATASET_VERSION

    def validate(self) -> None:
        """Reject changes that weaken the frozen point-in-time contract."""
        if self.primary_horizon_sessions != 20:
            raise ValueError("The primary label must remain the 20-session excess return")
        if set(self.secondary_horizon_sessions) != {5}:
            raise ValueError("The v1 secondary label must remain the five-session excess return")
        if self.momentum_windows != (5, 20):
            raise ValueError("The v1 momentum windows are frozen at five and 20 sessions")
        if self.minimum_market_history_sessions < 21:
            raise ValueError("Market features require at least 21 strictly prior sessions")
        if self.volatility_window_sessions != 20 or self.beta_window_sessions != 20:
            raise ValueError("The v1 volatility and beta windows are frozen at 20 sessions")
        validation = pd.Timestamp(self.validation_start)
        holdout = pd.Timestamp(self.sealed_holdout_start)
        if validation.tzinfo is None or holdout.tzinfo is None:
            raise ValueError("Validation and holdout timestamps must include a timezone")
        if validation >= holdout:
            raise ValueError("Validation must begin before the sealed holdout")


@dataclass(frozen=True)
class TradeableDatasetTables:
    """Pure feature-store outputs before immutable persistence."""

    event_dataset: pd.DataFrame
    feature_availability: pd.DataFrame
    sample_funnel: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class TradeableDatasetResult:
    """Immutable on-disk tradeable-dataset bundle."""

    output_root: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class TradeableDatasetPreregistrationResult:
    """Frozen registry of events whose outcomes were already observed."""

    output_root: Path
    report: dict[str, Any]


def calculate_tradeable_dataset(
    ledger: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    legacy_observed_event_ids: Iterable[str],
    config: TradeableDatasetConfig | None = None,
) -> TradeableDatasetTables:
    """Build one auditable row per event without reading future market observations."""
    cfg = config or TradeableDatasetConfig()
    cfg.validate()
    _validate_inputs(ledger, issuer_bars, benchmark_bars, issuer_actions)
    legacy_ids = {str(value) for value in legacy_observed_event_ids}

    primary = ledger.loc[ledger["horizon_sessions"].eq(cfg.primary_horizon_sessions)].copy()
    if primary.empty:
        raise FilingCatalogError("No primary 20-session event rows are available")
    if primary["longitudinal_event_id"].duplicated().any():
        raise FilingCatalogError("Primary event rows are duplicated")
    primary["accepted_at"] = _utc(primary["accepted_at"], "accepted_at")
    primary["entry_at"] = _utc(primary["entry_at"], "entry_at")
    primary["exit_at"] = _utc(primary["exit_at"], "exit_at")
    if not (primary["accepted_at"] < primary["entry_at"]).all():
        raise FilingCatalogError("An event enters before its filing is accepted")
    if not (primary["entry_at"] < primary["exit_at"]).all():
        raise FilingCatalogError("An event label ends before its formal entry")

    stock_histories = _price_histories(issuer_bars, issuer_actions)
    benchmark_history = _benchmark_history(benchmark_bars)
    market_rows = [
        _market_feature_row(event, stock_histories, benchmark_history, cfg)
        for event in primary.itertuples(index=False)
    ]
    market = pd.DataFrame(market_rows)

    columns = [
        "longitudinal_event_id",
        "cik",
        "accession_number",
        "accepted_at",
        "form",
        "ticker",
        "security_id",
        "price_target_id",
        "entry_at",
        "exit_at",
        "market_excess_return_arithmetic",
        "market_excess_return_ready",
        "later_delisted_after_formal_exit",
        "analysis_sample",
        "gm_level",
        "gm_delta_yoy",
        "gm_innovation_raw",
        "gm_innovation_std",
    ]
    dataset = primary.loc[:, columns].rename(
        columns={
            "entry_at": "execution_entry_at",
            "exit_at": "target_available_at_20d",
            "market_excess_return_arithmetic": "target_market_excess_return_20d",
            "market_excess_return_ready": "target_ready_20d",
        }
    )
    dataset = dataset.merge(
        _secondary_labels(ledger, cfg),
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    dataset = dataset.merge(
        market,
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    dataset["feature_cutoff"] = dataset["accepted_at"]
    dataset["form_is_10k"] = dataset["form"].eq("10-K").astype(float)
    dataset["filing_features_complete"] = (
        dataset.loc[:, list(FILING_FEATURE_COLUMNS)].notna().all(axis=1)
    )
    dataset["market_features_complete"] = dataset.loc[:, list(MARKET_FEATURE_COLUMNS)].notna().all(
        axis=1
    ) & dataset["market_feature_clock_valid"].astype(bool)
    dataset["combined_features_complete"] = (
        dataset["filing_features_complete"] & dataset["market_features_complete"]
    )
    dataset["primary_label_ready"] = (
        dataset["target_ready_20d"].astype(bool)
        & dataset["target_market_excess_return_20d"].notna()
    )
    dataset["prior_outcome_observed"] = dataset["longitudinal_event_id"].isin(legacy_ids)
    holdout_start = pd.Timestamp(cfg.sealed_holdout_start)
    validation_start = pd.Timestamp(cfg.validation_start)
    dataset["time_role"] = np.select(
        [
            dataset["feature_cutoff"].ge(holdout_start),
            dataset["feature_cutoff"].ge(validation_start),
        ],
        ["sealed_holdout", "validation"],
        default="development",
    )
    dataset["evidence_role"] = np.select(
        [
            dataset["prior_outcome_observed"],
            dataset["time_role"].eq("sealed_holdout"),
        ],
        ["legacy_observed_training_only", "sealed_holdout"],
        default="unobserved_evaluation_candidate",
    )
    base_candidate = (
        dataset["analysis_sample"].astype(bool)
        & dataset["combined_features_complete"]
        & dataset["primary_label_ready"]
    )
    dataset["model_training_candidate"] = base_candidate
    dataset["model_evaluation_candidate"] = base_candidate & ~dataset["prior_outcome_observed"]
    dataset["sealed_holdout_candidate"] = dataset["model_evaluation_candidate"] & dataset[
        "time_role"
    ].eq("sealed_holdout")
    dataset["data_contract_status"] = dataset.apply(_contract_status, axis=1)
    dataset = dataset.sort_values(
        ["feature_cutoff", "cik", "longitudinal_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_tradeable_dataset(dataset)

    availability_columns = [
        "longitudinal_event_id",
        "accepted_at",
        "market_last_session_date",
        "market_feature_available_at",
        "prior_market_sessions",
        "market_feature_clock_valid",
        "market_feature_status",
    ]
    availability = dataset.loc[:, availability_columns].copy()
    funnel = _sample_funnel(dataset)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "events": len(dataset),
        "issuers": int(dataset["cik"].nunique()),
        "legacy_observed_events": int(dataset["prior_outcome_observed"].sum()),
        "filing_feature_complete_events": int(dataset["filing_features_complete"].sum()),
        "market_feature_complete_events": int(dataset["market_features_complete"].sum()),
        "combined_feature_complete_events": int(dataset["combined_features_complete"].sum()),
        "primary_label_ready_events": int(dataset["primary_label_ready"].sum()),
        "model_training_candidates": int(dataset["model_training_candidate"].sum()),
        "unobserved_evaluation_candidates": int(dataset["model_evaluation_candidate"].sum()),
        "sealed_holdout_candidates": int(dataset["sealed_holdout_candidate"].sum()),
        "current_model_gate": bool(dataset["model_evaluation_candidate"].any()),
        "current_gate_reason": (
            "unobserved_evaluation_events_available"
            if dataset["model_evaluation_candidate"].any()
            else "all_complete_events_are_legacy_observed_or_incomplete"
        ),
        "strategy_pnl_computed": False,
        "holdout_metrics_computed": False,
        "feature_sets": {
            "filing_only": list(FILING_FEATURE_COLUMNS),
            "market_only": list(MARKET_FEATURE_COLUMNS),
            "combined": list(COMBINED_FEATURE_COLUMNS),
        },
        "config": asdict(cfg),
    }
    return TradeableDatasetTables(dataset, availability, funnel, report)


def validate_tradeable_dataset(dataset: pd.DataFrame) -> None:
    """Validate clocks, evidence roles, labels, and same-sample model flags."""
    required = {
        "longitudinal_event_id",
        "feature_cutoff",
        "execution_entry_at",
        "target_available_at_20d",
        "market_feature_available_at",
        "prior_outcome_observed",
        "model_training_candidate",
        "model_evaluation_candidate",
        *COMBINED_FEATURE_COLUMNS,
    }
    _require_columns(dataset, required, "tradeable dataset")
    if dataset["longitudinal_event_id"].duplicated().any():
        raise FilingCatalogError("Tradeable dataset events are duplicated")
    feature_cutoff = _utc(dataset["feature_cutoff"], "feature_cutoff")
    entry = _utc(dataset["execution_entry_at"], "execution_entry_at")
    target_at = _utc(dataset["target_available_at_20d"], "target_available_at_20d")
    if not (feature_cutoff < entry).all() or not (entry < target_at).all():
        raise FilingCatalogError("Feature, execution, and label clocks are not strictly ordered")
    market_rows = dataset["market_feature_available_at"].notna()
    market_at = _utc(
        dataset.loc[market_rows, "market_feature_available_at"],
        "market_feature_available_at",
    )
    if (market_at > feature_cutoff.loc[market_rows]).any():
        raise FilingCatalogError("A market feature uses a same-time or future session")
    evaluation = dataset["model_evaluation_candidate"].astype(bool)
    if dataset.loc[evaluation, "prior_outcome_observed"].astype(bool).any():
        raise FilingCatalogError("A previously observed outcome entered the evaluation sample")
    training = dataset["model_training_candidate"].astype(bool)
    if dataset.loc[training, list(COMBINED_FEATURE_COLUMNS)].isna().any().any():
        raise FilingCatalogError("A model-training row has an incomplete feature set")


def build_tradeable_dataset_preregistration(
    ledger_root: Path,
    *,
    output_root: Path = Path("data/processed/tradeable_dataset_preregistration"),
    config: TradeableDatasetConfig | None = None,
) -> TradeableDatasetPreregistrationResult:
    """Freeze all event outcomes already observed before the new model round begins."""
    cfg = config or TradeableDatasetConfig()
    cfg.validate()
    ledger_manifest = _sha256_file(ledger_root / "output_manifest.json")
    ledger = load_catalog_table(ledger_root, "longitudinal_event_return_ledger.parquet")
    observed = (
        ledger.loc[
            ledger["horizon_sessions"].eq(cfg.primary_horizon_sessions),
            ["longitudinal_event_id"],
        ]
        .drop_duplicates()
        .sort_values("longitudinal_event_id", kind="mergesort")
        .reset_index(drop=True)
    )
    payload = {
        "version": PREREGISTRATION_VERSION,
        "manifest_schema": 2,
        "ledger_manifest_sha256": ledger_manifest,
        "legacy_observed_event_ids": observed["longitudinal_event_id"].astype(str).tolist(),
        "config": asdict(cfg),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    dataset_version = f"{PREREGISTRATION_VERSION}-{digest}"
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return TradeableDatasetPreregistrationResult(
            target.resolve(), _read_json(target / "preregistration_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete tradeable preregistration exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    registry_path = staging / "legacy_observed_event_ids.parquet"
    observed.to_parquet(registry_path, index=False)
    registry_artifact = _artifact_record(registry_path, len(observed))
    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": generated_at,
        "preregistration_cutoff": generated_at,
        "legacy_observed_events": len(observed),
        "legacy_events_permanently_excluded_from_evaluation": True,
        "sealed_holdout_metrics_authorized": False,
        "ledger_manifest_sha256": ledger_manifest,
        "config": asdict(cfg),
    }
    report_path = staging / "preregistration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "ledger_manifest_sha256": ledger_manifest,
                "artifacts": {"legacy_observed_event_ids.parquet": registry_artifact},
                "legacy_registry_sha256": registry_artifact["sha256"],
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return TradeableDatasetPreregistrationResult(target.resolve(), report)


def build_tradeable_dataset(
    ledger_root: Path,
    price_root: Path,
    benchmark_root: Path,
    action_root: Path,
    preregistration_root: Path,
    *,
    output_root: Path = Path("data/processed/tradeable_dataset"),
    config: TradeableDatasetConfig | None = None,
) -> TradeableDatasetResult:
    """Persist an immutable feature store tied to all source manifests."""
    cfg = config or TradeableDatasetConfig()
    cfg.validate()
    manifests = {
        "ledger_manifest_sha256": _sha256_file(ledger_root / "output_manifest.json"),
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
        "action_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
        "preregistration_manifest_sha256": _sha256_file(
            preregistration_root / "output_manifest.json"
        ),
    }
    ledger = load_catalog_table(ledger_root, "longitudinal_event_return_ledger.parquet")
    issuer_bars = load_catalog_table(price_root, "massive_daily_price_observations.parquet")
    benchmark_bars = load_catalog_table(
        benchmark_root, "benchmark_daily_price_observations.parquet"
    )
    issuer_actions = load_catalog_table(action_root, "normalized_corporate_actions.parquet")
    legacy = load_catalog_table(preregistration_root, "legacy_observed_event_ids.parquet")
    tables = calculate_tradeable_dataset(
        ledger,
        issuer_bars,
        benchmark_bars,
        issuer_actions,
        legacy["longitudinal_event_id"].astype(str),
        cfg,
    )
    payload = json.dumps({**manifests, "config": asdict(cfg)}, sort_keys=True).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return TradeableDatasetResult(
            target.resolve(), _read_json(target / "tradeable_dataset_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete tradeable dataset exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "event_feature_store.parquet": tables.event_dataset,
        "feature_availability.parquet": tables.feature_availability,
        "sample_funnel.parquet": tables.sample_funnel,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
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
    report_path = staging / "tradeable_dataset_report.json"
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
    return TradeableDatasetResult(target.resolve(), report)


def _validate_inputs(
    ledger: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
) -> None:
    _require_columns(
        ledger,
        {
            "longitudinal_event_id",
            "cik",
            "accession_number",
            "accepted_at",
            "form",
            "horizon_sessions",
            "price_target_id",
            "ticker",
            "security_id",
            "entry_at",
            "exit_at",
            "market_excess_return_arithmetic",
            "market_excess_return_ready",
            "later_delisted_after_formal_exit",
            "analysis_sample",
            "gm_level",
            "gm_delta_yoy",
            "gm_innovation_raw",
            "gm_innovation_std",
        },
        "longitudinal event ledger",
    )
    if ledger.duplicated(["longitudinal_event_id", "horizon_sessions"]).any():
        raise FilingCatalogError("Longitudinal event-horizon rows are duplicated")
    bar_columns = {
        "price_target_id",
        "session_date",
        "close_unadjusted",
        "volume",
    }
    _require_columns(issuer_bars, bar_columns, "issuer bars")
    _require_columns(
        benchmark_bars,
        {"session_date", "close_unadjusted"},
        "benchmark bars",
    )
    _require_columns(
        issuer_actions,
        {"price_target_id", "effective_at", "action_type", "split_from", "split_to"},
        "issuer actions",
    )


def _price_histories(
    issuer_bars: pd.DataFrame, issuer_actions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    bars = issuer_bars.copy()
    bars["session_date"] = pd.to_datetime(bars["session_date"], errors="coerce")
    bars["close_unadjusted"] = pd.to_numeric(bars["close_unadjusted"], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    if bars[["session_date", "close_unadjusted"]].isna().any().any():
        raise FilingCatalogError("Issuer bars contain invalid dates or closes")
    actions = issuer_actions.loc[issuer_actions["action_type"].eq("split")].copy()
    actions["effective_at"] = pd.to_datetime(actions["effective_at"], utc=True, errors="coerce")
    histories: dict[str, pd.DataFrame] = {}
    for target_id, group in bars.groupby("price_target_id", sort=False):
        history = group.sort_values("session_date", kind="mergesort").drop_duplicates(
            "session_date", keep="last"
        )
        history = history.loc[:, ["session_date", "close_unadjusted", "volume"]].copy()
        split_factor = pd.Series(1.0, index=history.index)
        target_actions = actions.loc[actions["price_target_id"].astype(str).eq(str(target_id))]
        for action in target_actions.itertuples(index=False):
            split_from = float(cast(Any, action.split_from))
            split_to = float(cast(Any, action.split_to))
            if split_from <= 0.0 or split_to <= 0.0 or pd.isna(action.effective_at):
                raise FilingCatalogError("A split action has an invalid ratio or timestamp")
            effective_date = (
                pd.Timestamp(cast(Any, action.effective_at)).tz_convert("UTC").tz_localize(None)
            )
            split_factor.loc[history["session_date"].ge(effective_date.normalize())] *= (
                split_to / split_from
            )
        history["split_adjusted_close"] = history["close_unadjusted"] * split_factor
        histories[str(target_id)] = history.reset_index(drop=True)
    return histories


def _benchmark_history(benchmark_bars: pd.DataFrame) -> pd.DataFrame:
    history = benchmark_bars.copy()
    history["session_date"] = pd.to_datetime(history["session_date"], errors="coerce")
    history["split_adjusted_close"] = pd.to_numeric(history["close_unadjusted"], errors="coerce")
    if history[["session_date", "split_adjusted_close"]].isna().any().any():
        raise FilingCatalogError("Benchmark bars contain invalid dates or closes")
    return (
        history.sort_values("session_date", kind="mergesort")
        .drop_duplicates("session_date", keep="last")
        .loc[:, ["session_date", "split_adjusted_close"]]
        .reset_index(drop=True)
    )


def _market_feature_row(
    event: Any,
    stock_histories: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame,
    config: TradeableDatasetConfig,
) -> dict[str, Any]:
    event_id = str(event.longitudinal_event_id)
    accepted_at = pd.Timestamp(event.accepted_at)
    accepted_date = accepted_at.tz_convert("UTC").tz_localize(None).normalize()
    history = stock_histories.get(str(event.price_target_id))
    empty = {
        "longitudinal_event_id": event_id,
        **{column: np.nan for column in MARKET_FEATURE_COLUMNS},
        "market_last_session_date": pd.NaT,
        "market_feature_available_at": pd.NaT,
        "prior_market_sessions": 0,
        "market_feature_clock_valid": False,
        "market_feature_status": "missing_price_target_history",
    }
    if history is None:
        return empty
    prior = history.loc[history["session_date"].lt(accepted_date)].copy()
    market_prior = benchmark_history.loc[benchmark_history["session_date"].lt(accepted_date)].copy()
    if len(prior) < config.minimum_market_history_sessions:
        empty["prior_market_sessions"] = len(prior)
        empty["market_feature_status"] = "insufficient_strictly_prior_market_history"
        if not prior.empty:
            empty["market_last_session_date"] = prior["session_date"].iloc[-1]
            empty["market_feature_available_at"] = _session_available_at(
                prior["session_date"].iloc[-1]
            )
        return empty
    if len(market_prior) < config.minimum_market_history_sessions:
        empty["prior_market_sessions"] = len(prior)
        empty["market_feature_status"] = "insufficient_strictly_prior_benchmark_history"
        return empty

    close = prior["split_adjusted_close"].to_numpy(dtype=float)
    volume = prior["volume"].to_numpy(dtype=float)
    benchmark_close = market_prior["split_adjusted_close"].to_numpy(dtype=float)
    issuer_momentum_5d = _momentum(close, 5)
    issuer_momentum_20d = _momentum(close, 20)
    market_momentum_5d = _momentum(benchmark_close, 5)
    market_momentum_20d = _momentum(benchmark_close, 20)
    last_session = pd.Timestamp(prior["session_date"].iloc[-1])
    available_at = _session_available_at(last_session)
    row: dict[str, Any] = {
        "longitudinal_event_id": event_id,
        "issuer_momentum_5d": issuer_momentum_5d,
        "issuer_momentum_20d": issuer_momentum_20d,
        "issuer_volatility_20d": _annualized_volatility(close, 20),
        "issuer_volume_ratio_5_20": _volume_ratio(volume, 5, 20),
        "market_momentum_5d": market_momentum_5d,
        "market_momentum_20d": market_momentum_20d,
        "relative_momentum_20d": ((1.0 + issuer_momentum_20d) / (1.0 + market_momentum_20d) - 1.0),
        "market_beta_20d": _market_beta(prior, market_prior, 20),
        "market_last_session_date": last_session,
        "market_feature_available_at": available_at,
        "prior_market_sessions": len(prior),
        "market_feature_clock_valid": bool(available_at <= accepted_at),
        "market_feature_status": "ready",
    }
    if not all(math.isfinite(float(row[column])) for column in MARKET_FEATURE_COLUMNS):
        row["market_feature_status"] = "non_finite_market_feature"
    return row


def _secondary_labels(ledger: pd.DataFrame, config: TradeableDatasetConfig) -> pd.DataFrame:
    event_ids = ledger[["longitudinal_event_id"]].drop_duplicates().copy()
    result = event_ids
    for horizon in config.secondary_horizon_sessions:
        labels = ledger.loc[
            ledger["horizon_sessions"].eq(horizon),
            [
                "longitudinal_event_id",
                "market_excess_return_arithmetic",
                "market_excess_return_ready",
                "exit_at",
            ],
        ].rename(
            columns={
                "market_excess_return_arithmetic": f"target_market_excess_return_{horizon}d",
                "market_excess_return_ready": f"target_ready_{horizon}d",
                "exit_at": f"target_available_at_{horizon}d",
            }
        )
        result = result.merge(labels, on="longitudinal_event_id", how="left", validate="one_to_one")
    return result


def _momentum(close: np.ndarray, window: int) -> float:
    if len(close) < window + 1 or close[-window - 1] <= 0.0:
        return float("nan")
    return float(close[-1] / close[-window - 1] - 1.0)


def _annualized_volatility(close: np.ndarray, window: int) -> float:
    if len(close) < window + 1 or (close[-window - 1 :] <= 0.0).any():
        return float("nan")
    returns = np.diff(np.log(close[-window - 1 :]))
    return float(returns.std(ddof=1) * math.sqrt(252.0))


def _volume_ratio(volume: np.ndarray, short_window: int, long_window: int) -> float:
    if len(volume) < long_window or not np.isfinite(volume[-long_window:]).all():
        return float("nan")
    long_mean = float(volume[-long_window:].mean())
    if long_mean <= 0.0:
        return float("nan")
    return float(volume[-short_window:].mean() / long_mean)


def _market_beta(stock: pd.DataFrame, market: pd.DataFrame, window: int) -> float:
    aligned = stock.loc[:, ["session_date", "split_adjusted_close"]].merge(
        market.loc[:, ["session_date", "split_adjusted_close"]],
        on="session_date",
        how="inner",
        suffixes=("_stock", "_market"),
    )
    if len(aligned) < window + 1:
        return float("nan")
    aligned = aligned.tail(window + 1)
    stock_returns = np.diff(np.log(aligned["split_adjusted_close_stock"].to_numpy(float)))
    market_returns = np.diff(np.log(aligned["split_adjusted_close_market"].to_numpy(float)))
    market_variance = float(market_returns.var(ddof=1))
    if market_variance <= 1e-16:
        return float("nan")
    covariance = float(np.cov(stock_returns, market_returns, ddof=1)[0, 1])
    return covariance / market_variance


def _session_available_at(session_date: Any) -> pd.Timestamp:
    session = pd.Timestamp(session_date).normalize().tz_localize("UTC")
    return session + pd.Timedelta(days=1)


def _contract_status(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["analysis_sample"]):
        reasons.append("outside_frozen_analysis_sample")
    if not bool(row["filing_features_complete"]):
        reasons.append("incomplete_filing_features")
    if not bool(row["market_features_complete"]):
        reasons.append(str(row["market_feature_status"]))
    if not bool(row["primary_label_ready"]):
        reasons.append("primary_label_not_ready")
    if bool(row["prior_outcome_observed"]):
        reasons.append("legacy_observed_training_only")
    return "ready" if not reasons else ";".join(dict.fromkeys(reasons))


def _sample_funnel(dataset: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("frozen_primary_events", pd.Series(True, index=dataset.index)),
        ("analysis_sample", dataset["analysis_sample"].astype(bool)),
        ("filing_features_complete", dataset["filing_features_complete"].astype(bool)),
        ("market_features_complete", dataset["market_features_complete"].astype(bool)),
        ("primary_label_ready", dataset["primary_label_ready"].astype(bool)),
        ("combined_training_candidate", dataset["model_training_candidate"].astype(bool)),
        ("unobserved_evaluation_candidate", dataset["model_evaluation_candidate"].astype(bool)),
        ("sealed_holdout_candidate", dataset["sealed_holdout_candidate"].astype(bool)),
    ]
    return pd.DataFrame(
        [
            {
                "stage_order": order,
                "stage": name,
                "events": int(mask.sum()),
                "issuers": int(dataset.loc[mask, "cik"].nunique()),
            }
            for order, (name, mask) in enumerate(stages, start=1)
        ]
    )


def _utc(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise FilingCatalogError(f"{name} contains invalid timestamps")
    return converted


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise FilingCatalogError(f"{name} is missing columns: {sorted(missing)}")


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FilingCatalogError(f"Expected a JSON object in {path}")
    return document
