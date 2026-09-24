"""Market-state features cut at the earliest SEC disclosure instead of the filing.

The frozen v2 view (``validation_market_state``) keeps market history strictly
before the New York date of the 10-Q/10-K acceptance.  When an Item 2.02
earnings release came earlier, those "pre-filing" features can contain the
price reaction to the release.  This view recomputes the identical thirteen
features with the identical feature function and inputs, changing only the
cutoff to the earliest SEC disclosure from the first-public clock table, and
compares both views event by event.  The v2 view is recomputed, not modified.

The earliest disclosure is used only as an event clock.  It does not make a
financial value available earlier, and it is not a certified first-public time.
No outcome is loaded and no return is computed.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.validation_market_state import (
    MARKET_STATE_FEATURES,
    RAW_PRICE_BASIS,
    ValidationMarketStateConfig,
    ValidationMarketStateInputs,
    _feature_row,
    _prepare_market_inputs,
    calculate_validation_market_state,
    load_validation_market_state_inputs,
)

DISCLOSURE_MARKET_STATE_VERSION: Final[str] = "disclosure-market-state-v1"
CUTOFF_BASIS: Final[str] = "earliest_sec_disclosure_new_york_date"
CLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "cik",
    "accession_number",
    "report_accepted_at",
    "earliest_disclosure_at",
    "clock_source",
    "disclosure_lead_days",
)
FEATURE_TOLERANCE: Final[float] = 1e-12


class DisclosureMarketStateError(FilingCatalogError):
    """Raised when the disclosure-cut view would violate its point-in-time contract."""


@dataclass(frozen=True)
class DisclosureMarketStateTables:
    """Disclosure-cut features, the event-level comparison, and the report."""

    features: pd.DataFrame
    comparison: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class DisclosureMarketStateResult:
    """Immutable on-disk bundle location and its report."""

    output_root: Path
    report: dict[str, Any]


def calculate_disclosure_market_state(
    inputs: ValidationMarketStateInputs,
    clocks: pd.DataFrame,
    config: ValidationMarketStateConfig | None = None,
) -> DisclosureMarketStateTables:
    """Recompute v2 market state at the filing and at the earliest disclosure.

    Every cohort event is kept.  An event whose report is absent from the clock
    table is labelled ``disclosure_clock_not_found`` rather than falling back to
    the filing clock.
    """
    cfg = config or ValidationMarketStateConfig()
    cfg.validate()
    missing = sorted(set(CLOCK_COLUMNS).difference(clocks.columns))
    if missing:
        raise DisclosureMarketStateError(f"Clock table lacks columns: {missing}")
    if clocks.duplicated(["cik", "accession_number"]).any():
        raise DisclosureMarketStateError("Clock rows must be unique per (cik, accession)")

    baseline = calculate_validation_market_state(
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
    ).features

    accessions = inputs.filing_features.loc[:, ["validation_event_id", "target_accession_number"]]
    if accessions["validation_event_id"].duplicated().any():
        raise DisclosureMarketStateError("Filing-feature event IDs are duplicated")
    events = baseline.loc[
        :, ["validation_event_id", "cik", "exchange_mic", "price_target_id", "accepted_at"]
    ].merge(accessions, on="validation_event_id", how="left", validate="one_to_one")
    if events["target_accession_number"].isna().any():
        raise DisclosureMarketStateError("A cohort event has no target accession")
    events = events.merge(
        clocks.loc[:, list(CLOCK_COLUMNS)],
        left_on=["cik", "target_accession_number"],
        right_on=["cik", "accession_number"],
        how="left",
        validate="many_to_one",
    )
    filing_at = pd.to_datetime(events["accepted_at"], utc=True)
    found = events["earliest_disclosure_at"].notna()
    report_at = pd.to_datetime(events["report_accepted_at"], utc=True)
    if (found & report_at.ne(filing_at)).any():
        raise DisclosureMarketStateError("Clock table disagrees with the cohort filing clock")
    cutoff_at = pd.to_datetime(events["earliest_disclosure_at"], utc=True)
    if (found & cutoff_at.gt(filing_at)).any():
        raise DisclosureMarketStateError("An earliest disclosure cannot follow its filing")

    price_groups, benchmark, split_map, close_lookup = _prepare_market_inputs(
        inputs.issuer_bars,
        inputs.benchmark_bars,
        inputs.split_observations,
        inputs.action_targets,
        inputs.exchange_sessions,
        issuer_price_basis=RAW_PRICE_BASIS,
        benchmark_price_basis=RAW_PRICE_BASIS,
    )
    rows: list[dict[str, Any]] = []
    for position, event in enumerate(events.itertuples(index=False)):
        if not found.iloc[position]:
            rows.append(_clock_not_found_row(str(event.validation_event_id)))
            continue
        # The shared feature function reads its cutoff from ``accepted_at``;
        # here that field carries the earliest disclosure, never the filing.
        view = SimpleNamespace(
            validation_event_id=event.validation_event_id,
            exchange_mic=event.exchange_mic,
            accepted_at=cutoff_at.iloc[position],
        )
        target = str(event.price_target_id)
        rows.append(
            _feature_row(
                view,
                price_groups.get(target, pd.DataFrame()),
                benchmark,
                split_map.get(target, pd.DataFrame()),
                close_lookup,
                cfg,
            )
        )
    identity = events.loc[
        :,
        [
            "validation_event_id",
            "cik",
            "target_accession_number",
            "clock_source",
            "disclosure_lead_days",
        ],
    ].assign(filing_accepted_at=filing_at, market_cutoff_at=cutoff_at)
    features = identity.merge(
        pd.DataFrame(rows), on="validation_event_id", how="left", validate="one_to_one"
    )
    features["cutoff_basis"] = CUTOFF_BASIS
    _validate_disclosure_features(features, cfg)
    comparison = _compare_views(baseline, features)
    report = _report(features, comparison, cfg)
    return DisclosureMarketStateTables(features, comparison, report)


def build_disclosure_market_state(
    return_gate_root: Path,
    execution_plan_root: Path,
    price_audit_root: Path,
    benchmark_root: Path,
    action_root: Path,
    feature_root: Path,
    clock_root: Path,
    *,
    output_root: Path = Path("data/processed/disclosure_market_state"),
    config: ValidationMarketStateConfig | None = None,
) -> DisclosureMarketStateResult:
    """Persist a content-addressed disclosure-cut view with its v2 comparison."""
    cfg = config or ValidationMarketStateConfig()
    cfg.validate()
    roots = {
        "return_gate": return_gate_root,
        "execution_plan": execution_plan_root,
        "price_audit": price_audit_root,
        "benchmark": benchmark_root,
        "security_actions": action_root,
        "filing_features": feature_root,
        "first_public_clock": clock_root,
    }
    input_hashes = {
        label: _sha256_file(root / "output_manifest.json") for label, root in roots.items()
    }
    module_root = Path(__file__).parent
    implementation_hashes = {
        name: _sha256_file(module_root / name)
        for name in ("disclosure_market_state.py", "validation_market_state.py")
    }
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "implementation_sha256": implementation_hashes,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        f"{DISCLOSURE_MARKET_STATE_VERSION}-{hashlib.sha256(payload).hexdigest()[:16]}"
    )
    target = output_root / dataset_version
    report_name = "disclosure_market_state_report.json"
    if (target / "output_manifest.json").is_file():
        stored: dict[str, Any] = json.loads((target / report_name).read_text(encoding="utf-8"))
        return DisclosureMarketStateResult(target.resolve(), stored)
    if target.exists():
        raise DisclosureMarketStateError(f"Incomplete disclosure market-state bundle: {target}")

    inputs = load_validation_market_state_inputs(
        return_gate_root,
        execution_plan_root,
        price_audit_root,
        benchmark_root,
        action_root,
        feature_root,
    )
    clocks = load_catalog_table(clock_root, "first_public_clock.parquet")
    tables = calculate_disclosure_market_state(inputs, clocks, cfg)

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "disclosure_market_state_features.parquet": tables.features,
        "disclosure_market_state_comparison.parquet": tables.comparison,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"rows": len(frame), "sha256": _sha256_file(path)}
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "implementation_sha256": implementation_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / report_name
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
    return DisclosureMarketStateResult(target.resolve(), report)


def _clock_not_found_row(validation_event_id: str) -> dict[str, Any]:
    return {
        "validation_event_id": validation_event_id,
        "market_last_session_date": pd.NaT,
        "market_feature_available_at": pd.NaT,
        "prior_common_sessions": 0,
        "split_adjustments_applied": 0,
        "market_state_ready": False,
        "market_state_status": "disclosure_clock_not_found",
        **{column: np.nan for column in MARKET_STATE_FEATURES},
    }


def _validate_disclosure_features(
    features: pd.DataFrame, config: ValidationMarketStateConfig
) -> None:
    if features.empty or features["validation_event_id"].duplicated().any():
        raise DisclosureMarketStateError("Disclosure market-state rows are empty or duplicated")
    ready = features["market_state_ready"].astype(bool)
    if features.loc[ready, list(MARKET_STATE_FEATURES)].isna().any().any():
        raise DisclosureMarketStateError("A ready event has an incomplete market-state vector")
    if not features.loc[ready, "prior_common_sessions"].ge(config.long_window_sessions + 1).all():
        raise DisclosureMarketStateError("A ready event lacks 61 strictly prior common sessions")
    available = pd.to_datetime(features.loc[ready, "market_feature_available_at"], utc=True)
    cutoff = pd.to_datetime(features.loc[ready, "market_cutoff_at"], utc=True)
    if not (available < cutoff).all():
        raise DisclosureMarketStateError("A market feature is not strictly before the disclosure")


def _compare_views(baseline: pd.DataFrame, disclosure: pd.DataFrame) -> pd.DataFrame:
    shared = [
        "validation_event_id",
        "market_state_ready",
        "market_state_status",
        "market_last_session_date",
        "market_feature_available_at",
        "prior_common_sessions",
        *MARKET_STATE_FEATURES,
    ]
    joined = disclosure.loc[
        :,
        [
            "validation_event_id",
            "cik",
            "clock_source",
            "disclosure_lead_days",
            "filing_accepted_at",
            "market_cutoff_at",
        ],
    ].merge(
        baseline.loc[:, shared]
        .add_prefix("filing_")
        .rename(columns={"filing_validation_event_id": "validation_event_id"}),
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        disclosure.loc[:, shared]
        .add_prefix("disclosure_")
        .rename(columns={"disclosure_validation_event_id": "validation_event_id"}),
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    filing_ready = joined["filing_market_state_ready"].astype(bool)
    disclosure_ready = joined["disclosure_market_state_ready"].astype(bool)
    joined["readiness_transition"] = np.select(
        [
            joined["market_cutoff_at"].isna(),
            filing_ready & disclosure_ready,
            filing_ready & ~disclosure_ready,
            ~filing_ready & disclosure_ready,
        ],
        ["clock_not_found", "ready_both", "lost_readiness", "gained_readiness"],
        default="ready_neither",
    )
    new_york = "America/New_York"
    filing_date = joined["filing_accepted_at"].dt.tz_convert(new_york).dt.date
    cutoff_date = pd.to_datetime(joined["market_cutoff_at"], utc=True).dt.tz_convert(new_york)
    joined["cutoff_date_moved"] = joined["market_cutoff_at"].notna() & cutoff_date.dt.date.ne(
        filing_date
    )
    joined["sessions_removed"] = (
        joined["filing_prior_common_sessions"] - joined["disclosure_prior_common_sessions"]
    ).where(joined["market_cutoff_at"].notna())
    filing_available = pd.to_datetime(joined["filing_market_feature_available_at"], utc=True)
    joined["filing_view_uses_post_disclosure_close"] = filing_ready & filing_available.gt(
        pd.to_datetime(joined["market_cutoff_at"], utc=True)
    )
    both = joined["readiness_transition"].eq("ready_both")
    deltas = pd.DataFrame(
        {
            f"delta_{column}": (joined[f"disclosure_{column}"] - joined[f"filing_{column}"]).where(
                both
            )
            for column in MARKET_STATE_FEATURES
        }
    )
    joined = pd.concat([joined, deltas], axis=1)
    joined["max_abs_feature_change"] = deltas.abs().max(axis=1).where(both)
    joined["features_changed"] = both & joined["max_abs_feature_change"].gt(FEATURE_TOLERANCE)
    return joined


def _report(
    features: pd.DataFrame, comparison: pd.DataFrame, config: ValidationMarketStateConfig
) -> dict[str, Any]:
    both = comparison["readiness_transition"].eq("ready_both")
    contaminated = comparison["filing_view_uses_post_disclosure_close"].astype(bool)
    per_feature: dict[str, dict[str, Any]] = {}
    for column in MARKET_STATE_FEATURES:
        delta = comparison.loc[both, f"delta_{column}"].abs()
        changed = delta.gt(FEATURE_TOLERANCE)
        per_feature[column] = {
            "changed_events": int(changed.sum()),
            "median_abs_change_when_changed": (
                float(delta.loc[changed].median()) if changed.any() else None
            ),
        }
    return {
        "contract_version": DISCLOSURE_MARKET_STATE_VERSION,
        "feature_contract_version": config.version,
        "cutoff_basis": CUTOFF_BASIS,
        "events": len(features),
        "issuers": int(features["cik"].nunique()),
        "clock_found": int(features["market_cutoff_at"].notna().sum()),
        "clock_source_counts": {
            str(k): int(v) for k, v in features["clock_source"].value_counts().items()
        },
        "cutoff_date_moved_events": int(comparison["cutoff_date_moved"].sum()),
        "filing_view_ready_events": int(comparison["filing_market_state_ready"].sum()),
        "disclosure_view_ready_events": int(features["market_state_ready"].sum()),
        "readiness_transitions": {
            str(k): int(v) for k, v in comparison["readiness_transition"].value_counts().items()
        },
        "filing_view_post_disclosure_close_events": int(contaminated.sum()),
        "filing_view_post_disclosure_close_issuers": int(
            comparison.loc[contaminated, "cik"].nunique()
        ),
        "features_changed_events": int(comparison["features_changed"].sum()),
        "per_feature": per_feature,
        "financial_value_clock_changed": False,
        "first_public_certified": False,
        "outcome_values_loaded": False,
        "returns_computed": False,
        "model_admission": False,
        "trading_admission": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
