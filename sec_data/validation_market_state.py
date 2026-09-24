"""Strict pre-filing market-state features for the formal validation cohort."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

VALIDATION_MARKET_STATE_VERSION: Final[str] = "validation-market-state-v2-local-split-ratios"
RAW_PRICE_BASIS: Final[str] = "raw_as_traded"
SPLIT_POLICY: Final[str] = "prior_local_split_from_over_split_to"
MARKET_STATE_FEATURES: Final[tuple[str, ...]] = (
    "pre_price_momentum_5d",
    "pre_price_momentum_20d",
    "pre_price_momentum_60d",
    "pre_price_volatility_20d",
    "pre_price_volatility_60d",
    "pre_volume_ratio_5_20",
    "pre_log_dollar_volume_20d",
    "pre_market_momentum_20d",
    "pre_market_momentum_60d",
    "pre_relative_momentum_20d",
    "pre_relative_momentum_60d",
    "pre_market_beta_60d",
    "pre_price_drawdown_60d",
)


class ValidationMarketStateError(FilingCatalogError):
    """Raised when a market-state feature violates the point-in-time contract."""


@dataclass(frozen=True)
class ValidationMarketStateConfig:
    """Frozen market-state windows and data-availability requirements."""

    short_window_sessions: int = 5
    medium_window_sessions: int = 20
    long_window_sessions: int = 60
    annualization_sessions: int = 252
    version: str = VALIDATION_MARKET_STATE_VERSION

    def validate(self) -> None:
        """Reject settings that weaken the declared feature history."""
        if self.version != VALIDATION_MARKET_STATE_VERSION:
            raise ValueError("Changed split semantics require the v2 market-state version")
        if self.short_window_sessions != 5 or self.medium_window_sessions != 20:
            raise ValueError("Short and medium market windows are frozen at 5 and 20 sessions")
        if self.long_window_sessions != 60:
            raise ValueError("The long market-state window is frozen at 60 sessions")
        if self.annualization_sessions != 252:
            raise ValueError("Volatility annualization is frozen at 252 sessions")


@dataclass(frozen=True)
class ValidationMarketStateTables:
    """Pure market-state tables before immutable persistence."""

    features: pd.DataFrame
    clock_audit: pd.DataFrame
    sample_funnel: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationMarketStateResult:
    """Immutable on-disk market-state feature bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_market_state(
    authorized_cohort: pd.DataFrame,
    event_endpoints: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    split_observations: pd.DataFrame,
    action_targets: pd.DataFrame,
    filing_features: pd.DataFrame,
    exchange_sessions: pd.DataFrame,
    config: ValidationMarketStateConfig | None = None,
    *,
    issuer_price_basis: str,
    benchmark_price_basis: str,
) -> ValidationMarketStateTables:
    """Build features using only sessions strictly before each filing date."""
    cfg = config or ValidationMarketStateConfig()
    cfg.validate()
    _validate_inputs(
        authorized_cohort,
        event_endpoints,
        issuer_bars,
        benchmark_bars,
        split_observations,
        action_targets,
        filing_features,
        exchange_sessions,
    )
    endpoint_columns = [
        "security_event_component_id",
        "validation_event_id",
        "cik",
        "ticker",
        "exchange_mic",
        "price_target_id",
    ]
    events = authorized_cohort[["security_event_component_id", "accepted_at"]].merge(
        event_endpoints[endpoint_columns],
        on="security_event_component_id",
        how="left",
        validate="one_to_one",
    )
    if events["price_target_id"].isna().any():
        raise ValidationMarketStateError("An authorized event has no formal price target")
    if events["validation_event_id"].duplicated().any():
        raise ValidationMarketStateError(
            "Market-state v1 requires one authorized security component per event"
        )
    feature_columns = [
        "validation_event_id",
        "form",
        "actual_available_at",
        "gm_level",
        "gm_yoy_lag",
        "gm_delta_yoy",
        "gm_forecast",
        "gm_innovation_raw",
        "gm_innovation_std",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_training_observations",
    ]
    financial = filing_features[feature_columns].copy()
    if financial["validation_event_id"].duplicated().any():
        raise ValidationMarketStateError("Filing-feature event IDs are duplicated")
    events = events.merge(financial, on="validation_event_id", how="left", validate="one_to_one")
    events["accepted_at"] = _utc(events["accepted_at"], "accepted_at")
    events["actual_available_at"] = _utc(events["actual_available_at"], "actual_available_at")
    if not events["accepted_at"].eq(events["actual_available_at"]).all():
        raise ValidationMarketStateError("Filing and actual availability clocks do not match")

    price_groups, benchmark, split_map, close_lookup = _prepare_market_inputs(
        issuer_bars,
        benchmark_bars,
        split_observations,
        action_targets,
        exchange_sessions,
        issuer_price_basis=issuer_price_basis,
        benchmark_price_basis=benchmark_price_basis,
    )
    rows = [
        _feature_row(
            event,
            price_groups.get(str(event.price_target_id), pd.DataFrame()),
            benchmark,
            split_map.get(str(event.price_target_id), pd.DataFrame()),
            close_lookup,
            cfg,
        )
        for event in events.itertuples(index=False)
    ]
    market = pd.DataFrame(rows)
    output = events.merge(market, on="validation_event_id", how="left", validate="one_to_one")
    output["form_is_10k"] = output["form"].eq("10-K").astype(float)
    output = output.sort_values(
        ["actual_available_at", "cik", "validation_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_validation_market_state(output, cfg)
    clock_columns = [
        "validation_event_id",
        "accepted_at",
        "market_last_session_date",
        "market_feature_available_at",
        "prior_common_sessions",
        "split_adjustments_applied",
        "market_state_ready",
        "market_state_status",
    ]
    clock_audit = output[clock_columns].copy()
    ready = output["market_state_ready"].astype(bool)
    funnel = pd.DataFrame(
        [
            _funnel("authorized_events", len(output), output["cik"].nunique()),
            _funnel(
                "strictly_prior_market_history",
                int(output["prior_common_sessions"].ge(cfg.long_window_sessions + 1).sum()),
                output.loc[
                    output["prior_common_sessions"].ge(cfg.long_window_sessions + 1), "cik"
                ].nunique(),
            ),
            _funnel("market_state_ready", int(ready.sum()), output.loc[ready, "cik"].nunique()),
        ]
    )
    status_counts = {
        str(key): int(value) for key, value in output["market_state_status"].value_counts().items()
    }
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "authorized_events": len(output),
        "authorized_issuers": int(output["cik"].nunique()),
        "market_state_ready_events": int(ready.sum()),
        "market_state_ready_issuers": int(output.loc[ready, "cik"].nunique()),
        "market_state_ready_coverage": float(ready.mean()),
        "status_counts": status_counts,
        "feature_columns": list(MARKET_STATE_FEATURES),
        "split_adjustment_policy": SPLIT_POLICY,
        "input_price_basis": RAW_PRICE_BASIS,
        "vendor_cumulative_factor_used": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "strategy_pnl_computed": False,
        "config": asdict(cfg),
    }
    return ValidationMarketStateTables(output, clock_audit, funnel, report)


def validate_validation_market_state(
    features: pd.DataFrame,
    config: ValidationMarketStateConfig | None = None,
) -> None:
    """Validate completeness and strict market-feature clocks."""
    cfg = config or ValidationMarketStateConfig()
    cfg.validate()
    required = {
        "validation_event_id",
        "accepted_at",
        "market_feature_available_at",
        "prior_common_sessions",
        "market_state_ready",
        *MARKET_STATE_FEATURES,
    }
    _require_columns(features, required, "validation market-state features")
    if features.empty or features["validation_event_id"].duplicated().any():
        raise ValidationMarketStateError("Market-state event rows are empty or duplicated")
    ready = features["market_state_ready"].astype(bool)
    if features.loc[ready, list(MARKET_STATE_FEATURES)].isna().any().any():
        raise ValidationMarketStateError("A ready event has an incomplete market-state vector")
    if not features.loc[ready, "prior_common_sessions"].ge(cfg.long_window_sessions + 1).all():
        raise ValidationMarketStateError("A ready event lacks 61 strictly prior common sessions")
    available = _utc(
        features.loc[ready, "market_feature_available_at"], "market_feature_available_at"
    )
    accepted = _utc(features.loc[ready, "accepted_at"], "accepted_at")
    if not (available < accepted).all():
        raise ValidationMarketStateError("A market feature is not strictly pre-filing")


@dataclass(frozen=True)
class ValidationMarketStateInputs:
    """Hash-verified input tables shared by every market-state view."""

    authorized_cohort: pd.DataFrame
    event_endpoints: pd.DataFrame
    issuer_bars: pd.DataFrame
    benchmark_bars: pd.DataFrame
    split_observations: pd.DataFrame
    action_targets: pd.DataFrame
    filing_features: pd.DataFrame
    exchange_sessions: pd.DataFrame


def load_validation_market_state_inputs(
    return_gate_root: Path,
    execution_plan_root: Path,
    price_audit_root: Path,
    benchmark_root: Path,
    action_root: Path,
    feature_root: Path,
) -> ValidationMarketStateInputs:
    """Load verified tables and reject any source whose price basis is not raw."""
    # Verify the provider request basis, not merely a column named unadjusted.
    targets = load_catalog_table(price_audit_root, "validation_price_batch_targets.parquet")
    if "adjusted" not in targets or not targets.adjusted.map(lambda v: v is False).all():
        raise ValidationMarketStateError("Issuer query basis is not explicitly unadjusted")
    benchmark_report_path = benchmark_root / "massive_benchmark_report.json"
    if _sha256_file(benchmark_report_path) != _read_json(
        benchmark_root / "output_manifest.json"
    ).get("report_sha256"):
        raise ValidationMarketStateError("Benchmark report hash mismatch")
    benchmark_report = _read_json(benchmark_report_path)
    if benchmark_report.get("query_adjusted") is not False:
        raise ValidationMarketStateError("Benchmark query basis is not explicitly unadjusted")
    if not load_catalog_table(benchmark_root, "benchmark_split_observations.parquet").empty:
        raise ValidationMarketStateError("Benchmark splits need a separately specified adapter")
    return ValidationMarketStateInputs(
        authorized_cohort=load_catalog_table(
            return_gate_root, "validation_authorized_return_cohort.parquet"
        ),
        event_endpoints=load_catalog_table(
            price_audit_root, "validation_price_batch_event_endpoints.parquet"
        ),
        issuer_bars=load_catalog_table(price_audit_root, "validation_price_batch_bars.parquet"),
        benchmark_bars=load_catalog_table(
            benchmark_root, "benchmark_daily_price_observations.parquet"
        ),
        split_observations=load_catalog_table(action_root, "split_observations.parquet"),
        action_targets=load_catalog_table(action_root, "massive_security_actions_targets.parquet"),
        filing_features=load_catalog_table(feature_root, "target_filing_features.parquet"),
        exchange_sessions=load_catalog_table(
            execution_plan_root, "validation_exchange_sessions.parquet"
        ),
    )


def build_validation_market_state(
    return_gate_root: Path,
    execution_plan_root: Path,
    price_audit_root: Path,
    benchmark_root: Path,
    action_root: Path,
    feature_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_market_state"),
    config: ValidationMarketStateConfig | None = None,
) -> ValidationMarketStateResult:
    """Persist the first outcome-independent pre-filing market-state feature store."""
    cfg = config or ValidationMarketStateConfig()
    cfg.validate()
    roots = {
        "return_gate": return_gate_root,
        "execution_plan": execution_plan_root,
        "price_audit": price_audit_root,
        "benchmark": benchmark_root,
        "security_actions": action_root,
        "filing_features": feature_root,
    }
    input_hashes = {
        label: _sha256_file(root / "output_manifest.json") for label, root in roots.items()
    }
    inputs = load_validation_market_state_inputs(
        return_gate_root,
        execution_plan_root,
        price_audit_root,
        benchmark_root,
        action_root,
        feature_root,
    )
    tables = calculate_validation_market_state(
        inputs.authorized_cohort,
        inputs.event_endpoints,
        inputs.issuer_bars,
        inputs.benchmark_bars,
        inputs.split_observations,
        inputs.action_targets,
        inputs.filing_features,
        inputs.exchange_sessions,
        cfg,
        issuer_price_basis=RAW_PRICE_BASIS,
        benchmark_price_basis=RAW_PRICE_BASIS,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationMarketStateResult(
            target.resolve(), _read_json(target / "validation_market_state_report.json")
        )
    if target.exists():
        raise ValidationMarketStateError(f"Incomplete market-state directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_market_state_features.parquet": tables.features,
        "validation_market_state_clock_audit.parquet": tables.clock_audit,
        "validation_market_state_funnel.parquet": tables.sample_funnel,
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
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_market_state_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationMarketStateResult(target.resolve(), report)


def _prepare_market_inputs(
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    split_observations: pd.DataFrame,
    action_targets: pd.DataFrame,
    exchange_sessions: pd.DataFrame,
    *,
    issuer_price_basis: str,
    benchmark_price_basis: str,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[tuple[str, Any], Any],
]:
    """Price groups, benchmark, split map and official close clocks, shared by views."""
    prices = _prepare_bars(
        issuer_bars, "price_target_id", "issuer bars", price_basis=issuer_price_basis
    )
    benchmark = _prepare_bars(
        benchmark_bars, "benchmark_target_id", "benchmark bars", price_basis=benchmark_price_basis
    )
    benchmark = benchmark.sort_values("session_date", kind="mergesort").drop_duplicates(
        "session_date", keep="last"
    )
    split_map = _prepare_splits(split_observations, action_targets)
    calendar = exchange_sessions.copy()
    calendar["session_date"] = pd.to_datetime(calendar["session_date"], errors="raise")
    calendar["regular_close_at"] = _utc(calendar["regular_close_at"], "regular_close_at")
    close_lookup = {
        (
            str(row.exchange_mic),
            pd.Timestamp(cast(Any, row.session_date)).date(),
        ): row.regular_close_at
        for row in calendar.itertuples(index=False)
    }
    price_groups = {
        str(target_id): group.copy()
        for target_id, group in prices.groupby("price_target_id", sort=False)
    }
    return price_groups, benchmark, split_map, close_lookup


def _feature_row(
    event: Any,
    issuer_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    splits: pd.DataFrame,
    close_lookup: dict[tuple[str, Any], Any],
    config: ValidationMarketStateConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_event_id": str(event.validation_event_id),
        "market_last_session_date": pd.NaT,
        "market_feature_available_at": pd.NaT,
        "prior_common_sessions": 0,
        "split_adjustments_applied": 0,
        "split_adjustment_policy": SPLIT_POLICY,
        "market_state_ready": False,
        "market_state_status": "insufficient_strictly_prior_history",
        **{column: np.nan for column in MARKET_STATE_FEATURES},
    }
    if issuer_bars.empty:
        result["market_state_status"] = "missing_issuer_price_history"
        return result
    cutoff = pd.Timestamp(event.accepted_at)
    cutoff_date = cutoff.tz_convert("America/New_York").tz_localize(None).normalize()
    stock = issuer_bars.loc[issuer_bars["session_date"].lt(cutoff_date)].copy()
    market = benchmark_bars.loc[benchmark_bars["session_date"].lt(cutoff_date)].copy()
    if stock.empty or market.empty:
        return result
    stock, applied = _split_adjusted_history(
        stock, splits, cutoff_date, price_basis=RAW_PRICE_BASIS
    )
    common = stock.merge(
        market[["session_date", "close_unadjusted"]].rename(
            columns={"close_unadjusted": "market_close"}
        ),
        on="session_date",
        how="inner",
        validate="one_to_one",
    ).sort_values("session_date", kind="mergesort")
    result["prior_common_sessions"] = len(common)
    result["split_adjustments_applied"] = applied
    if len(common) < config.long_window_sessions + 1:
        return result
    common = common.tail(config.long_window_sessions + 1).copy()
    last_date = pd.Timestamp(common["session_date"].iloc[-1])
    available_at = close_lookup.get((str(event.exchange_mic), last_date.date()))
    if available_at is None or pd.isna(available_at):
        result["market_state_status"] = "missing_official_close_clock"
        return result
    if pd.Timestamp(available_at) >= cutoff:
        result["market_state_status"] = "non_prior_market_clock"
        return result
    close = common["close_adjusted_to_cutoff"].to_numpy(dtype=float)
    volume = common["volume_adjusted_to_cutoff"].to_numpy(dtype=float)
    market_close = common["market_close"].to_numpy(dtype=float)
    stock_returns = np.diff(np.log(close))
    market_returns = np.diff(np.log(market_close))
    market_variance = float(np.var(market_returns[-config.long_window_sessions :], ddof=1))
    if market_variance <= 0.0:
        result["market_state_status"] = "zero_market_variance"
        return result
    covariance = float(
        np.cov(
            stock_returns[-config.long_window_sessions :],
            market_returns[-config.long_window_sessions :],
            ddof=1,
        )[0, 1]
    )
    short = config.short_window_sessions
    medium = config.medium_window_sessions
    long = config.long_window_sessions
    values = {
        "pre_price_momentum_5d": close[-1] / close[-(short + 1)] - 1.0,
        "pre_price_momentum_20d": close[-1] / close[-(medium + 1)] - 1.0,
        "pre_price_momentum_60d": close[-1] / close[-(long + 1)] - 1.0,
        "pre_price_volatility_20d": float(np.std(stock_returns[-medium:], ddof=1))
        * math.sqrt(config.annualization_sessions),
        "pre_price_volatility_60d": float(np.std(stock_returns[-long:], ddof=1))
        * math.sqrt(config.annualization_sessions),
        "pre_volume_ratio_5_20": float(np.mean(volume[-short:]) / np.mean(volume[-medium:])),
        "pre_log_dollar_volume_20d": float(np.log(np.mean(close[-medium:] * volume[-medium:]))),
        "pre_market_momentum_20d": market_close[-1] / market_close[-(medium + 1)] - 1.0,
        "pre_market_momentum_60d": market_close[-1] / market_close[-(long + 1)] - 1.0,
        "pre_relative_momentum_20d": (close[-1] / close[-(medium + 1)] - 1.0)
        - (market_close[-1] / market_close[-(medium + 1)] - 1.0),
        "pre_relative_momentum_60d": (close[-1] / close[-(long + 1)] - 1.0)
        - (market_close[-1] / market_close[-(long + 1)] - 1.0),
        "pre_market_beta_60d": covariance / market_variance,
        "pre_price_drawdown_60d": close[-1] / float(np.max(close)) - 1.0,
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
        result["market_state_status"] = "nonfinite_market_state"
        return result
    result.update(values)
    result["market_last_session_date"] = last_date
    result["market_feature_available_at"] = pd.Timestamp(available_at)
    result["market_state_ready"] = True
    result["market_state_status"] = "ready"
    return result


def _require_raw_basis(frame: pd.DataFrame, price_basis: str) -> None:
    """Reject unknown/adjusted inputs and our own marked output; never infer basis from prices."""
    if (
        price_basis != RAW_PRICE_BASIS
        or frame.attrs.get("price_basis", RAW_PRICE_BASIS) != RAW_PRICE_BASIS
    ):
        raise ValidationMarketStateError("Explicit raw_as_traded price basis required")
    if {"close_adjusted_to_cutoff", "volume_adjusted_to_cutoff", "share_unit_factor"}.intersection(
        frame
    ):
        raise ValidationMarketStateError("Already adjusted history cannot be adjusted twice")
    if "share_unit_basis" in frame and not frame.share_unit_basis.eq(RAW_PRICE_BASIS).all():
        raise ValidationMarketStateError("Already adjusted or unknown share-unit basis")
    if "adjusted" in frame and not frame.adjusted.map(lambda v: v is False).all():
        raise ValidationMarketStateError("Adjusted vendor quotes are not raw inputs")


def _prepare_bars(
    frame: pd.DataFrame, target_column: str, label: str, *, price_basis: str
) -> pd.DataFrame:
    _require_raw_basis(frame, price_basis)
    required = {
        target_column,
        "session_date",
        "close_unadjusted",
        "volume",
    }
    _require_columns(frame, required, label)
    result = frame.copy()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="raise")
    result["close_unadjusted"] = pd.to_numeric(result["close_unadjusted"], errors="raise")
    result["volume"] = pd.to_numeric(result["volume"], errors="raise")
    if result.duplicated([target_column, "session_date"]).any():
        raise ValidationMarketStateError(f"{label} contains duplicate target-session rows")
    if (
        result.session_date.isna().any()
        or not np.isfinite(result[["close_unadjusted", "volume"]].to_numpy(float)).all()
        or (result["close_unadjusted"] <= 0.0).any()
        or (result["volume"] < 0.0).any()
    ):
        raise ValidationMarketStateError(f"{label} contains invalid prices or volume")
    return result


def _prepare_splits(
    split_observations: pd.DataFrame, action_targets: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    if split_observations.empty:
        return {}
    _require_columns(
        split_observations,
        {"action_target_id", "execution_date", "split_from", "split_to"},
        "split observations",
    )
    _require_columns(action_targets, {"action_target_id", "price_target_id"}, "action targets")
    mapping = action_targets[["action_target_id", "price_target_id"]].drop_duplicates()
    if mapping["action_target_id"].duplicated().any():
        raise ValidationMarketStateError("Action targets map to multiple price targets")
    splits = split_observations.merge(
        mapping, on="action_target_id", how="left", validate="many_to_one"
    )
    if splits["price_target_id"].isna().any():
        raise ValidationMarketStateError("A split observation lacks a price target")
    # Validation and ratio calculation happen after each event's date cutoff.
    # Future records and vendor cumulative fields never enter its conversion.
    splits["execution_date"] = _split_dates(splits.execution_date)
    return {
        str(target_id): group.copy()
        for target_id, group in splits.groupby("price_target_id", sort=False)
    }


def _split_adjusted_history(
    bars: pd.DataFrame, splits: pd.DataFrame, cutoff_date: pd.Timestamp, *, price_basis: str
) -> tuple[pd.DataFrame, int]:
    _require_raw_basis(bars, price_basis)
    cutoff_date = pd.Timestamp(cutoff_date)
    if cutoff_date.tzinfo is not None or cutoff_date != cutoff_date.normalize():
        raise ValidationMarketStateError("Split cutoff must be a naive New York calendar date")
    result = bars.sort_values("session_date", kind="mergesort").copy()
    result["session_date"] = _split_dates(result.session_date)
    if (
        not result.session_date.lt(cutoff_date).all()
        or result.session_date.duplicated().any()
        or not np.isfinite(result[["close_unadjusted", "volume"]].to_numpy(float)).all()
        or not result.close_unadjusted.gt(0).all()
        or not result.volume.ge(0).all()
    ):
        raise ValidationMarketStateError("Invalid or non-prior raw history")
    result["close_adjusted_to_cutoff"] = result["close_unadjusted"].astype(float)
    result["volume_adjusted_to_cutoff"] = result["volume"].astype(float)
    result["share_unit_factor"] = 1.0
    result["share_unit_basis"] = "split_adjusted_to_cutoff"
    result.attrs["price_basis"] = "split_adjusted_to_cutoff"
    applied = 0
    if splits.empty:
        return result, applied
    _require_columns(splits, {"execution_date", "split_from", "split_to"}, "split observations")
    splits = splits.copy()
    splits["execution_date"] = _split_dates(splits.execution_date)
    eligible = splits.loc[splits["execution_date"].lt(cutoff_date)].sort_values(
        "execution_date", kind="mergesort"
    )
    ratios = eligible[["split_from", "split_to"]].to_numpy(float)
    if eligible.execution_date.duplicated().any():
        raise ValidationMarketStateError(
            "Ambiguous duplicate split date; no automatic deduplication"
        )
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValidationMarketStateError("Single-event split ratios must be finite and positive")
    for split in eligible.itertuples(index=False):
        factor = float(cast(Any, split.split_from)) / float(cast(Any, split.split_to))
        mask = result["session_date"].lt(pd.Timestamp(cast(Any, split.execution_date)))
        if mask.any():
            result.loc[mask, "close_adjusted_to_cutoff"] *= factor
            result.loc[mask, "volume_adjusted_to_cutoff"] /= factor
            result.loc[mask, "share_unit_factor"] *= factor
            applied += 1
    if (
        not np.isfinite(
            result[
                ["close_adjusted_to_cutoff", "volume_adjusted_to_cutoff", "share_unit_factor"]
            ].to_numpy(float)
        ).all()
        or not result.close_adjusted_to_cutoff.gt(0).all()
        or not np.allclose(
            result.close_unadjusted * result.volume,
            result.close_adjusted_to_cutoff * result.volume_adjusted_to_cutoff,
            rtol=1e-12,
            atol=0,
        )
    ):
        raise ValidationMarketStateError("Invalid split conversion or changed price-times-volume")
    return result, applied


def _split_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValidationMarketStateError("Session and split dates must be naive calendar dates")
    return dates


def _validate_inputs(*frames: pd.DataFrame) -> None:
    if any(frame.empty for frame in frames[:4]):
        raise ValidationMarketStateError("Authorized events and price inputs must be non-empty")
    _require_columns(frames[0], {"security_event_component_id", "accepted_at"}, "authorized cohort")
    _require_columns(
        frames[1],
        {
            "security_event_component_id",
            "validation_event_id",
            "cik",
            "ticker",
            "exchange_mic",
            "price_target_id",
        },
        "event endpoints",
    )
    _require_columns(frames[7], {"exchange_mic", "session_date", "regular_close_at"}, "sessions")


def _funnel(stage: str, events: int, issuers: int) -> dict[str, Any]:
    return {"stage": stage, "events": int(events), "issuers": int(issuers)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationMarketStateError(f"{label} lacks required columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise ValidationMarketStateError(f"{label} contains invalid timestamps")
    return result


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
