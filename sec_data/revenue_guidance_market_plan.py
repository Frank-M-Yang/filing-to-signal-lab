"""Outcome-blind market-evidence expansion for revenue-guidance events."""

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

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

REVENUE_GUIDANCE_MARKET_PLAN_VERSION: Final[str] = "revenue-guidance-market-plan-v1"
FORBIDDEN_RETURN_VALUE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "issuer_total_return",
        "market_total_return",
        "market_excess_return_arithmetic",
        "market_relative_wealth_return",
        "future_return",
        "delisting_return",
    }
)


class RevenueGuidanceMarketPlanError(FilingCatalogError):
    """Raised when the outcome-blind guidance market plan is violated."""


@dataclass(frozen=True)
class RevenueGuidanceMarketPlanConfig:
    """Frozen evidence windows and acquisition batching rules."""

    provisional_price_window_after_days: int = 45
    issuer_batch_size: int = 10
    benchmark_ticker: str = "SPY"
    version: str = REVENUE_GUIDANCE_MARKET_PLAN_VERSION

    def validate(self) -> None:
        """Reject a plan that cannot cover the formal 20-session horizon."""
        if self.provisional_price_window_after_days < 35:
            raise ValueError("Provisional price windows must cover at least 35 calendar days")
        if self.issuer_batch_size < 1:
            raise ValueError("Issuer batch size must be positive")
        if self.benchmark_ticker != "SPY":
            raise ValueError("The first guidance return audit is frozen to SPY")


@dataclass(frozen=True)
class RevenueGuidanceMarketPlanTables:
    """Frozen event, identity, SEC, span, batch, benchmark, and gate tables."""

    events: pd.DataFrame
    identity_anchors: pd.DataFrame
    filing_evidence_targets: pd.DataFrame
    issuer_price_spans: pd.DataFrame
    collection_batches: pd.DataFrame
    benchmark_plan: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueGuidanceMarketPlanResult:
    """Immutable persisted guidance market-evidence plan."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_market_plan(
    forecasts: pd.DataFrame,
    filing_features: pd.DataFrame,
    authorized_cohort: pd.DataFrame,
    filing_catalog: pd.DataFrame,
    *,
    forecast_gate_passed: bool,
    config: RevenueGuidanceMarketPlanConfig | None = None,
) -> RevenueGuidanceMarketPlanTables:
    """Freeze every missing market-evidence event without reading any return value."""
    cfg = config or RevenueGuidanceMarketPlanConfig()
    cfg.validate()
    for frame, label in (
        (forecasts, "forecasts"),
        (filing_features, "filing features"),
        (authorized_cohort, "authorized cohort"),
        (filing_catalog, "filing catalog"),
    ):
        forbidden = sorted(FORBIDDEN_RETURN_VALUE_COLUMNS.intersection(frame.columns))
        if forbidden:
            raise RevenueGuidanceMarketPlanError(f"{label} contains return values: {forbidden}")
    _require_columns(
        forecasts,
        {
            "validation_event_id",
            "cik",
            "target_actual_available_at",
            "selection_uses_return_values",
        },
        "revenue-guidance forecasts",
    )
    _require_columns(
        filing_features,
        {
            "validation_event_id",
            "target_id",
            "cik",
            "target_accession_number",
            "form",
            "report_date",
            "accepted_at",
        },
        "filing features",
    )
    _require_columns(
        authorized_cohort,
        {
            "validation_event_id",
            "return_outcome_read_authorized",
            "selection_uses_return_values",
            "authorization_uses_return_values",
        },
        "authorized cohort",
    )
    _require_columns(
        filing_catalog,
        {"cik", "accession_number", "accepted_at", "primary_document", "source_url"},
        "filing catalog",
    )
    if not forecast_gate_passed:
        raise RevenueGuidanceMarketPlanError("Forecast gate must pass before market expansion")
    if forecasts["selection_uses_return_values"].astype(bool).any():
        raise RevenueGuidanceMarketPlanError("Forecast selection used return values")
    if (
        authorized_cohort["selection_uses_return_values"].astype(bool).any()
        or authorized_cohort["authorization_uses_return_values"].astype(bool).any()
    ):
        raise RevenueGuidanceMarketPlanError("Return authorization was not outcome blind")
    for frame, label in (
        (forecasts, "forecast"),
        (filing_features, "filing feature"),
        (authorized_cohort, "authorization"),
    ):
        if (
            frame["validation_event_id"].isna().any()
            or frame["validation_event_id"].duplicated().any()
        ):
            raise RevenueGuidanceMarketPlanError(f"{label} event IDs must be unique")

    features = filing_features[
        [
            "validation_event_id",
            "target_id",
            "cik",
            "target_accession_number",
            "form",
            "report_date",
            "accepted_at",
        ]
    ].copy()
    features["cik"] = features["cik"].astype(str).str.zfill(10)
    events = forecasts[["validation_event_id", "cik", "target_actual_available_at"]].merge(
        features,
        on="validation_event_id",
        how="left",
        suffixes=("_forecast", ""),
        validate="one_to_one",
    )
    if events["target_id"].isna().any():
        raise RevenueGuidanceMarketPlanError("A forecast lacks exact filing metadata")
    if not events["cik_forecast"].astype(str).str.zfill(10).eq(events["cik"]).all():
        raise RevenueGuidanceMarketPlanError("Forecast and filing CIKs disagree")
    events["accepted_at"] = _utc(events["accepted_at"], "filing accepted_at")
    events["target_actual_available_at"] = _utc(
        events["target_actual_available_at"], "target actual availability"
    )
    if not events["accepted_at"].eq(events["target_actual_available_at"]).all():
        raise RevenueGuidanceMarketPlanError("Target actual and filing acceptance clocks disagree")

    authorized_ids = set(
        authorized_cohort.loc[
            authorized_cohort["return_outcome_read_authorized"].astype(bool),
            "validation_event_id",
        ].astype(str)
    )
    events["existing_return_evidence_authorized"] = (
        events["validation_event_id"].astype(str).isin(authorized_ids)
    )
    events["market_evidence_collection_required"] = ~events["existing_return_evidence_authorized"]
    events["price_collection_candidate"] = events["market_evidence_collection_required"]
    events["identity_evidence_candidate"] = events["market_evidence_collection_required"]
    events["market_event_id"] = events["validation_event_id"].map(
        lambda event_id: _stable_hash(
            {"validation_event_id": str(event_id), "contract_version": cfg.version}
        )
    )
    events["collection_wave"] = 1
    events["provisional_price_window_start"] = events["accepted_at"].dt.date.astype(str)
    events["provisional_price_window_end"] = (
        events["accepted_at"] + pd.to_timedelta(cfg.provisional_price_window_after_days, unit="D")
    ).dt.date.astype(str)
    events["provider_entitlement_status"] = np.where(
        events["market_evidence_collection_required"], "must_be_tested", "not_required"
    )
    events["historical_security_identity_ready"] = events["existing_return_evidence_authorized"]
    events["official_calendar_window_ready"] = events["existing_return_evidence_authorized"]
    events["price_collection_authorized"] = False
    events["return_outcome_read_authorized"] = events["existing_return_evidence_authorized"]
    events["selection_uses_provider_coverage"] = False
    events["selection_uses_return_values"] = False
    events["outcome_values_loaded"] = False
    events["replacement_after_collection_failure_allowed"] = False
    events["market_plan_status"] = np.where(
        events["market_evidence_collection_required"],
        "awaiting_identity_calendar_actions_prices_and_outcome",
        "existing_authorized_return_evidence",
    )
    events = events.drop(columns=["cik_forecast", "target_actual_available_at"])
    events = events.sort_values(
        ["accepted_at", "cik", "target_accession_number"], kind="mergesort"
    ).reset_index(drop=True)

    collection_events = events.loc[events["market_evidence_collection_required"]].copy()
    if collection_events.empty:
        raise RevenueGuidanceMarketPlanError("No missing market evidence remains")
    anchors = _identity_anchors(collection_events, cfg)
    filing_targets = _filing_evidence_targets(collection_events, filing_catalog)
    spans = _issuer_price_spans(collection_events)
    batches = _collection_batches(spans, cfg)
    benchmark_plan = _benchmark_plan(collection_events, cfg)
    total_events = len(events)
    existing_events = int(events["existing_return_evidence_authorized"].sum())
    missing_events = len(collection_events)
    gates = pd.DataFrame(
        [
            _gate(
                "forecast_selection",
                total_events,
                total_events,
                True,
                "Every target comes from the passing no-return forecast tournament",
            ),
            _gate(
                "existing_market_authorization",
                existing_events,
                total_events,
                existing_events == total_events,
                "Existing outcome-blind return authorization before expansion",
            ),
            _gate(
                "missing_market_evidence_frozen",
                missing_events,
                missing_events,
                True,
                "Every uncovered forecast event enters the no-replacement collection queue",
            ),
            _gate(
                "return_outcome_read",
                0,
                missing_events,
                False,
                "New return values remain locked until all market evidence gates pass",
            ),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "forecast_events": total_events,
        "forecast_issuers": int(events["cik"].nunique()),
        "existing_authorized_return_events": existing_events,
        "market_evidence_collection_events": missing_events,
        "market_evidence_collection_issuers": int(collection_events["cik"].nunique()),
        "identity_anchor_targets": len(anchors),
        "filing_index_evidence_targets": len(filing_targets),
        "issuer_price_spans": len(spans),
        "issuer_collection_batches": int(batches["collection_batch"].nunique()),
        "earliest_missing_event_date": str(collection_events["accepted_at"].dt.date.min()),
        "latest_missing_event_date": str(collection_events["accepted_at"].dt.date.max()),
        "benchmark_ticker": cfg.benchmark_ticker,
        "entry_rule": "second_official_trading_session_open_strictly_after_acceptance",
        "exit_rule": "twentieth_official_trading_session_close_including_entry_session",
        "selection_uses_provider_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "replacement_after_collection_failure_allowed": False,
        "return_outcome_read_authorized": False,
        "config": asdict(cfg),
    }
    return RevenueGuidanceMarketPlanTables(
        events,
        anchors,
        filing_targets,
        spans,
        batches,
        benchmark_plan,
        gates,
        report,
    )


def build_revenue_guidance_market_plan(
    tournament_root: Path,
    filing_feature_root: Path,
    authorization_root: Path,
    filing_catalog_root: Path,
    *,
    output_root: Path = Path("data/processed/revenue_guidance_market_plan"),
    config: RevenueGuidanceMarketPlanConfig | None = None,
) -> RevenueGuidanceMarketPlanResult:
    """Persist the manifest-tied, no-replacement guidance market plan."""
    cfg = config or RevenueGuidanceMarketPlanConfig()
    cfg.validate()
    tournament_report = _read_json(tournament_root / "revenue_guidance_tournament_report.json")
    input_hashes = {
        "revenue_guidance_tournament": _sha256_file(tournament_root / "output_manifest.json"),
        "filing_features": _sha256_file(filing_feature_root / "output_manifest.json"),
        "return_authorization": _sha256_file(authorization_root / "output_manifest.json"),
        "filing_catalog": _sha256_file(filing_catalog_root / "output_manifest.json"),
    }
    tables = calculate_revenue_guidance_market_plan(
        load_catalog_table(tournament_root, "revenue_guidance_forecasts.parquet"),
        load_catalog_table(filing_feature_root, "target_filing_features.parquet"),
        load_catalog_table(authorization_root, "validation_authorized_return_cohort.parquet"),
        load_catalog_table(filing_catalog_root, "filing_catalog.parquet"),
        forecast_gate_passed=bool(tournament_report["forecast_gate_passed"]),
        config=cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": asdict(cfg)}, sort_keys=True
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceMarketPlanResult(
            target.resolve(), _read_json(target / "validation_market_plan_report.json")
        )
    if target.exists():
        raise RevenueGuidanceMarketPlanError(f"Incomplete guidance market plan: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_market_event_targets.parquet": tables.events,
        "validation_identity_anchor_targets.parquet": tables.identity_anchors,
        "validation_filing_evidence_targets.parquet": tables.filing_evidence_targets,
        "validation_issuer_price_spans.parquet": tables.issuer_price_spans,
        "validation_market_collection_batches.parquet": tables.collection_batches,
        "validation_benchmark_plan.parquet": tables.benchmark_plan,
        "validation_market_gates.parquet": tables.gates,
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
    report_path = staging / "validation_market_plan_report.json"
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
    return RevenueGuidanceMarketPlanResult(target.resolve(), report)


def _identity_anchors(
    collection_events: pd.DataFrame, config: RevenueGuidanceMarketPlanConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cik, group in collection_events.groupby("cik", sort=True):
        ordered = group.sort_values(["accepted_at", "validation_event_id"], kind="mergesort")
        selected = [("only_candidate", ordered.iloc[0])]
        if len(ordered) > 1:
            selected = [("first_candidate", ordered.iloc[0]), ("last_candidate", ordered.iloc[-1])]
        for role, event in selected:
            payload = {
                "cik": str(cik),
                "validation_event_id": str(event["validation_event_id"]),
                "anchor_role": role,
                "contract_version": config.version,
            }
            rows.append(
                {
                    "identity_anchor_id": _stable_hash(payload),
                    "market_event_id": str(event["market_event_id"]),
                    "cik": str(cik),
                    "anchor_role": role,
                    "validation_event_id": str(event["validation_event_id"]),
                    "accession_number": str(event["target_accession_number"]),
                    "accepted_at": event["accepted_at"],
                    "query_date": str(pd.Timestamp(event["accepted_at"]).date()),
                    "provider_query_method": "cik_and_point_in_time_date",
                    "query_active_first": True,
                    "query_inactive_only_after_no_active_match": True,
                    "automatic_interval_propagation_allowed": False,
                    "selection_uses_return_values": False,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["query_date", "cik", "anchor_role"], kind="mergesort")
        .reset_index(drop=True)
    )


def _filing_evidence_targets(
    collection_events: pd.DataFrame, filing_catalog: pd.DataFrame
) -> pd.DataFrame:
    catalog = filing_catalog.copy()
    catalog["cik"] = catalog["cik"].astype(str).str.zfill(10)
    catalog["accession_number"] = catalog["accession_number"].astype(str)
    catalog["catalog_accepted_at"] = _utc(catalog["accepted_at"], "catalog accepted_at")
    catalog = catalog.drop(columns=["accepted_at"])
    if catalog.duplicated(["cik", "accession_number"]).any():
        raise RevenueGuidanceMarketPlanError("Filing catalog is not unique by CIK/accession")
    result = collection_events.merge(
        catalog[
            [
                "cik",
                "accession_number",
                "catalog_accepted_at",
                "primary_document",
                "source_url",
            ]
        ],
        left_on=["cik", "target_accession_number"],
        right_on=["cik", "accession_number"],
        how="left",
        validate="many_to_one",
    )
    result["catalog_clock_match"] = result["accepted_at"].eq(result["catalog_accepted_at"])
    if not result["catalog_clock_match"].all():
        raise RevenueGuidanceMarketPlanError("A target filing lacks an exact catalog clock")
    result["index_url"] = result.apply(
        lambda row: _filing_index_url(str(row["cik"]), str(row["accession_number"])),
        axis="columns",
    )
    result["primary_document_url"] = result["source_url"]
    result["sec_index_collection_required"] = True
    result["sec_primary_document_collection_required"] = False
    result["sec_evidence_status"] = "awaiting_filing_index_for_sic"
    return result[
        [
            "market_event_id",
            "validation_event_id",
            "cik",
            "target_accession_number",
            "accepted_at",
            "catalog_accepted_at",
            "catalog_clock_match",
            "index_url",
            "primary_document_url",
            "sec_index_collection_required",
            "sec_primary_document_collection_required",
            "sec_evidence_status",
        ]
    ].copy()


def _issuer_price_spans(collection_events: pd.DataFrame) -> pd.DataFrame:
    grouped = collection_events.groupby("cik", sort=True)
    spans = grouped.agg(
        price_candidate_events=("validation_event_id", "size"),
        first_event_accepted_at=("accepted_at", "min"),
        last_event_accepted_at=("accepted_at", "max"),
        provisional_price_span_start=("provisional_price_window_start", "min"),
        provisional_price_span_end=("provisional_price_window_end", "max"),
    ).reset_index()
    spans["historical_ticker"] = None
    spans["composite_figi"] = None
    spans["share_class_figi"] = None
    spans["identity_interval_ready"] = False
    spans["price_query_authorized"] = False
    spans["selection_uses_return_values"] = False
    return spans


def _collection_batches(
    spans: pd.DataFrame, config: RevenueGuidanceMarketPlanConfig
) -> pd.DataFrame:
    result = spans.copy()
    result["issuer_sampling_hash"] = result["cik"].map(
        lambda cik: _stable_hash({"cik": str(cik), "contract_version": config.version})
    )
    result = result.sort_values(["issuer_sampling_hash", "cik"], kind="mergesort").reset_index(
        drop=True
    )
    result["market_collection_order"] = result.index + 1
    result["collection_batch"] = result.index // config.issuer_batch_size + 1
    result["collection_status"] = "awaiting_identity_anchors"
    return result[
        [
            "collection_batch",
            "market_collection_order",
            "cik",
            "price_candidate_events",
            "provisional_price_span_start",
            "provisional_price_span_end",
            "issuer_sampling_hash",
            "collection_status",
        ]
    ].copy()


def _benchmark_plan(
    collection_events: pd.DataFrame, config: RevenueGuidanceMarketPlanConfig
) -> pd.DataFrame:
    start = str(collection_events["accepted_at"].dt.date.min())
    end = str(
        (
            collection_events["accepted_at"].max()
            + timedelta(days=config.provisional_price_window_after_days + 10)
        ).date()
    )
    return pd.DataFrame(
        [
            {
                "benchmark_role": "broad_market_total_return_proxy",
                "benchmark_ticker": config.benchmark_ticker,
                "price_span_start": start,
                "price_span_end": end,
                "mapping_status": "explicit_instrument",
                "price_collection_authorized": False,
                "authorization_dependency": "official_calendar_event_windows",
                "selection_uses_return_values": False,
            }
        ]
    )


def _filing_index_url(cik: str, accession_number: str) -> str:
    accession_digits = accession_number.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_digits}/{accession_number}-index.html"
    )


def _gate(
    gate: str, numerator: int, denominator: int, passed: bool, meaning: str
) -> dict[str, Any]:
    return {
        "gate": gate,
        "numerator": numerator,
        "denominator": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
        "passed": bool(passed),
        "meaning": meaning,
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise RevenueGuidanceMarketPlanError(f"{label} contains invalid timestamps")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RevenueGuidanceMarketPlanError(f"{label} lacks required columns: {missing}")


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
