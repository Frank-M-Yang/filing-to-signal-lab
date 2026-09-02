"""Build outcome-blind pre-entry features for the frozen drift candidate."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from conditional_drift import (
    ALL_FEATURE_COLUMNS,
    ConditionalDriftConfig,
    _prepare_bars,
    _prepare_calendar,
    _reaction_row,
)
from conditional_drift_forward import ALIGNMENT_FEATURES
from conditional_drift_v2 import ALIGNMENT_COLUMNS
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from tradeable_dataset import (
    MARKET_FEATURE_COLUMNS,
    TradeableDatasetConfig,
    _benchmark_history,
    _market_feature_row,
    _price_histories,
)

CONDITIONAL_DRIFT_FORWARD_FEATURE_VERSION: Final[str] = "conditional-drift-forward-features-v1"


@dataclass(frozen=True)
class ConditionalDriftForwardFeatureTables:
    """Pure outcome-blind candidate features and clock audit."""

    features: pd.DataFrame
    clock_audit: pd.DataFrame
    funnel: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConditionalDriftForwardFeatureResult:
    """Immutable outcome-blind forward feature bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_conditional_drift_forward_features(
    scoring_queue: pd.DataFrame,
    filing_features: pd.DataFrame,
    execution_events: pd.DataFrame,
    event_price_endpoints: pd.DataFrame,
    issuer_bars: pd.DataFrame,
    issuer_actions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_actions: pd.DataFrame,
    calendar_sessions: pd.DataFrame,
) -> ConditionalDriftForwardFeatureTables:
    """Create only features available before formal entry for clean queued events."""
    _validate_inputs(
        scoring_queue,
        filing_features,
        execution_events,
        event_price_endpoints,
        issuer_bars,
        benchmark_bars,
        calendar_sessions,
    )
    queue = scoring_queue.loc[scoring_queue["forward_queue_eligible"].astype(bool)].copy()
    if queue.empty:
        raise FilingCatalogError("The clean forward scoring queue is empty")
    if queue["outcome_materialized_locally"].astype(bool).any():
        raise FilingCatalogError("The clean feature queue contains a materialized outcome")
    filing_columns = [
        "validation_event_id",
        "form",
        "gm_level",
        "gm_delta_yoy",
        "gm_innovation_raw",
        "gm_innovation_std",
    ]
    execution_columns = ["validation_event_id", "formal_entry_open_at"]
    endpoint_columns = ["validation_event_id", "price_target_id"]
    if event_price_endpoints["validation_event_id"].duplicated().any():
        duplicates = event_price_endpoints.loc[
            event_price_endpoints["validation_event_id"].duplicated(False),
            "validation_event_id",
        ]
        raise FilingCatalogError(
            "Forward features require one security component per event: "
            + ", ".join(duplicates.astype(str).head(5))
        )
    base = (
        queue.loc[
            :,
            [
                "validation_event_id",
                "cik",
                "assigned_ticker",
                "accepted_at",
                "evidence_role",
                "outcome_materialized_locally",
                "model_training_label_end",
            ],
        ]
        .merge(
            filing_features.loc[:, filing_columns].drop_duplicates("validation_event_id"),
            on="validation_event_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            execution_events.loc[:, execution_columns].drop_duplicates("validation_event_id"),
            on="validation_event_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            event_price_endpoints.loc[:, endpoint_columns],
            on="validation_event_id",
            how="left",
            validate="one_to_one",
        )
    )
    if base[["form", "formal_entry_open_at", "price_target_id"]].isna().any().any():
        raise FilingCatalogError(
            "A clean queued event is missing filing, execution, or price identity"
        )
    base = base.rename(
        columns={
            "assigned_ticker": "ticker",
            "formal_entry_open_at": "execution_entry_at",
        }
    )
    base["longitudinal_event_id"] = base["validation_event_id"].astype(str)
    base["accepted_at"] = _utc(base["accepted_at"], "accepted_at")
    base["execution_entry_at"] = _utc(base["execution_entry_at"], "execution_entry_at")
    base["form_is_10k"] = base["form"].eq("10-K").astype(float)
    base["prior_outcome_observed"] = False

    stock_histories = _price_histories(issuer_bars, issuer_actions)
    benchmark_history = _benchmark_history(benchmark_bars)
    market_rows = [
        _market_feature_row(
            event,
            stock_histories,
            benchmark_history,
            TradeableDatasetConfig(),
        )
        for event in base.itertuples(index=False)
    ]
    data = base.merge(
        pd.DataFrame(market_rows),
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    stock = _prepare_bars(issuer_bars, "price_target_id")
    market = _prepare_bars(benchmark_bars, None)
    sessions = _prepare_calendar(calendar_sessions)
    reaction_rows: list[dict[str, Any]] = []
    for event in data.itertuples(index=False):
        price_target_id = str(event.price_target_id)
        reaction_rows.append(
            _reaction_row(
                event,
                stock.loc[stock["price_target_id"].astype(str).eq(price_target_id)],
                market,
                issuer_actions.loc[
                    issuer_actions["price_target_id"].astype(str).eq(price_target_id)
                ],
                benchmark_actions,
                sessions,
                ConditionalDriftConfig(),
            )
        )
    data = data.merge(
        pd.DataFrame(reaction_rows),
        on="longitudinal_event_id",
        how="left",
        validate="one_to_one",
    )
    innovation = pd.to_numeric(data["gm_innovation_std"], errors="coerce")
    reaction = pd.to_numeric(data["reaction_market_excess_return"], errors="coerce")
    data["innovation_aligned_reaction"] = np.sign(innovation) * reaction
    data["innovation_reaction_disagreement"] = (
        data["innovation_aligned_reaction"].lt(0.0).astype(float)
    )
    data["innovation_reaction_interaction"] = innovation * reaction
    data["market_features_complete"] = data.loc[:, list(MARKET_FEATURE_COLUMNS)].notna().all(
        axis=1
    ) & data["market_feature_clock_valid"].astype(bool)
    data["reaction_features_complete"] = data.loc[
        :, [column for column in ALL_FEATURE_COLUMNS if column.startswith("reaction_")]
    ].notna().all(axis=1) & data["reaction_clock_valid"].astype(bool)
    data["alignment_features_complete"] = data.loc[:, list(ALIGNMENT_COLUMNS)].notna().all(axis=1)
    data["forward_features_complete"] = data.loc[:, list(ALIGNMENT_FEATURES)].notna().all(
        axis=1
    ) & data["reaction_clock_valid"].astype(bool)
    accepted = _utc(data["accepted_at"], "accepted_at")
    market_at = _utc(data["market_feature_available_at"], "market_feature_available_at")
    reaction_at = _utc(data["reaction_feature_available_at"], "reaction_feature_available_at")
    entry_at = _utc(data["execution_entry_at"], "execution_entry_at")
    data["market_clock_validated"] = market_at.le(accepted)
    data["reaction_clock_validated"] = accepted.le(reaction_at) & reaction_at.lt(entry_at)
    data["forward_score_ready"] = (
        data["forward_features_complete"]
        & data["market_clock_validated"]
        & data["reaction_clock_validated"]
    )
    if not data["market_clock_validated"].all():
        raise FilingCatalogError("A forward market feature uses post-acceptance information")
    if not data["reaction_clock_validated"].all():
        raise FilingCatalogError("A forward reaction feature is not strictly pre-entry")
    ready = data.loc[data["forward_score_ready"]].copy()
    output_columns = [
        "validation_event_id",
        "cik",
        "ticker",
        "accepted_at",
        "reaction_feature_available_at",
        "execution_entry_at",
        "evidence_role",
        "prior_outcome_observed",
        "outcome_materialized_locally",
        *ALIGNMENT_FEATURES,
    ]
    features = ready.loc[:, output_columns].sort_values(
        ["reaction_feature_available_at", "cik", "validation_event_id"], kind="mergesort"
    )
    if any(
        column.startswith("target_") or column.startswith("actual_") for column in features.columns
    ):
        raise FilingCatalogError("Outcome column crossed the forward feature firewall")
    audit_columns = [
        "validation_event_id",
        "cik",
        "ticker",
        "accepted_at",
        "market_feature_available_at",
        "market_feature_status",
        "market_clock_validated",
        "reaction_feature_available_at",
        "reaction_status",
        "reaction_clock_validated",
        "execution_entry_at",
        "market_features_complete",
        "reaction_features_complete",
        "alignment_features_complete",
        "forward_features_complete",
        "forward_score_ready",
    ]
    audit = data.loc[:, audit_columns].sort_values(
        ["accepted_at", "cik", "validation_event_id"], kind="mergesort"
    )
    funnel = pd.DataFrame(
        [
            {"stage_order": 1, "stage": "clean_temporal_queue", "events": len(data)},
            {
                "stage_order": 2,
                "stage": "market_features_complete",
                "events": int(data["market_features_complete"].sum()),
            },
            {
                "stage_order": 3,
                "stage": "reaction_features_complete",
                "events": int(data["reaction_features_complete"].sum()),
            },
            {
                "stage_order": 4,
                "stage": "alignment_features_complete",
                "events": int(data["alignment_features_complete"].sum()),
            },
            {"stage_order": 5, "stage": "forward_score_ready", "events": len(features)},
        ]
    )
    report: dict[str, Any] = {
        "contract_version": CONDITIONAL_DRIFT_FORWARD_FEATURE_VERSION,
        "queue_events": len(data),
        "queue_issuers": int(data["cik"].nunique()),
        "market_feature_complete_events": int(data["market_features_complete"].sum()),
        "reaction_feature_complete_events": int(data["reaction_features_complete"].sum()),
        "alignment_feature_complete_events": int(data["alignment_features_complete"].sum()),
        "forward_score_ready_events": len(features),
        "forward_score_ready_issuers": int(features["cik"].nunique()),
        "outcome_values_loaded": False,
        "return_labels_computed": False,
        "target_columns_written": False,
        "feature_clock": "strictly_prior_market_state_plus_last_close_before_entry",
        "research_only": True,
    }
    return ConditionalDriftForwardFeatureTables(features, audit, funnel, report)


def build_conditional_drift_forward_features(
    forward_candidate_root: Path,
    filing_feature_root: Path,
    execution_root: Path,
    price_root: Path,
    action_root: Path,
    benchmark_root: Path,
    *,
    output_root: Path = Path("data/processed/conditional_drift_forward_features"),
) -> ConditionalDriftForwardFeatureResult:
    """Persist outcome-blind pre-entry features for the frozen candidate queue."""
    manifests = {
        "implementation_sha256": _sha256_file(Path(__file__)),
        "forward_candidate_manifest_sha256": _sha256_file(
            forward_candidate_root / "output_manifest.json"
        ),
        "filing_feature_manifest_sha256": _sha256_file(
            filing_feature_root / "output_manifest.json"
        ),
        "execution_manifest_sha256": _sha256_file(execution_root / "output_manifest.json"),
        "price_manifest_sha256": _sha256_file(price_root / "output_manifest.json"),
        "action_manifest_sha256": _sha256_file(action_root / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(benchmark_root / "output_manifest.json"),
    }
    queue = load_catalog_table(forward_candidate_root, "forward_scoring_queue.parquet")
    filing = load_catalog_table(filing_feature_root, "target_filing_features.parquet")
    execution = load_catalog_table(execution_root, "validation_execution_events.parquet")
    endpoints = pd.read_parquet(
        price_root / "validation_price_batch_event_endpoints.parquet",
        columns=["validation_event_id", "price_target_id"],
    )
    bars = load_catalog_table(price_root, "validation_price_batch_bars.parquet")
    actions = load_catalog_table(action_root, "normalized_corporate_actions.parquet")
    benchmark = load_catalog_table(benchmark_root, "benchmark_daily_price_observations.parquet")
    benchmark_actions = load_catalog_table(benchmark_root, "normalized_benchmark_actions.parquet")
    sessions = load_catalog_table(execution_root, "validation_exchange_sessions.parquet")
    tables = calculate_conditional_drift_forward_features(
        queue,
        filing,
        execution,
        endpoints,
        bars,
        actions,
        benchmark,
        benchmark_actions,
        sessions,
    )
    payload = json.dumps(manifests, sort_keys=True).encode()
    dataset_version = (
        CONDITIONAL_DRIFT_FORWARD_FEATURE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ConditionalDriftForwardFeatureResult(
            target.resolve(), _read_json(target / "conditional_drift_forward_feature_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete forward feature directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "conditional_drift_forward_features.parquet": tables.features,
        "conditional_drift_forward_feature_clock_audit.parquet": tables.clock_audit,
        "conditional_drift_forward_feature_funnel.parquet": tables.funnel,
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
    report_path = staging / "conditional_drift_forward_feature_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
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
    return ConditionalDriftForwardFeatureResult(target.resolve(), report)


def _validate_inputs(
    queue: pd.DataFrame,
    filing: pd.DataFrame,
    execution: pd.DataFrame,
    endpoints: pd.DataFrame,
    bars: pd.DataFrame,
    benchmark: pd.DataFrame,
    sessions: pd.DataFrame,
) -> None:
    required = {
        "queue": {
            "validation_event_id",
            "cik",
            "assigned_ticker",
            "accepted_at",
            "evidence_role",
            "forward_queue_eligible",
            "outcome_materialized_locally",
            "model_training_label_end",
        },
        "filing": {
            "validation_event_id",
            "form",
            "gm_level",
            "gm_delta_yoy",
            "gm_innovation_raw",
            "gm_innovation_std",
        },
        "execution": {"validation_event_id", "formal_entry_open_at"},
        "endpoints": {"validation_event_id", "price_target_id"},
        "bars": {
            "price_target_id",
            "session_date",
            "open_unadjusted",
            "high_unadjusted",
            "low_unadjusted",
            "close_unadjusted",
            "volume",
        },
        "benchmark": {"session_date", "open_unadjusted", "close_unadjusted"},
        "sessions": {"exchange_mic", "session_date", "regular_close_at"},
    }
    frames = {
        "queue": queue,
        "filing": filing,
        "execution": execution,
        "endpoints": endpoints,
        "bars": bars,
        "benchmark": benchmark,
        "sessions": sessions,
    }
    for name, columns in required.items():
        missing = sorted(columns.difference(frames[name].columns))
        if missing:
            raise FilingCatalogError(f"Forward {name} input is missing columns: {missing}")
    if benchmark.empty:
        raise FilingCatalogError("Forward benchmark bars are empty")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise FilingCatalogError(f"{label} contains invalid timestamps")
    return result


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
