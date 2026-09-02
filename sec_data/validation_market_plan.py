"""Outcome-blind market-data plan for the frozen validation cohort.

The plan is built only from pre-event feature readiness and immutable filing
metadata.  It deliberately stops before resolving returns: security-reference
anchors must establish stable historical identity, official calendars must set
the execution window, and filing-time industry evidence must exist before any
price or benchmark outcome can be evaluated.
"""

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

VALIDATION_MARKET_PLAN_VERSION: Final[str] = "validation-market-plan-v1-outcome-blind"
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


@dataclass(frozen=True)
class ValidationMarketPlanConfig:
    """Frozen acquisition order and execution-window preparation rules."""

    identity_anchors_per_issuer: int = 2
    issuer_batch_size: int = 10
    provisional_price_window_before_days: int = 0
    provisional_price_window_after_days: int = 45
    holding_sessions: int = 20
    entry_rule: str = "second_official_trading_session_open_after_accepted_at"
    exit_rule: str = "twentieth_official_trading_session_close_after_entry"
    version: str = VALIDATION_MARKET_PLAN_VERSION

    def validate(self) -> None:
        """Reject plans that cannot bracket identity or a 20-session window."""
        if self.identity_anchors_per_issuer != 2:
            raise ValueError("Validation identity discovery is frozen to endpoint anchors")
        if self.issuer_batch_size < 1:
            raise ValueError("Issuer batch size must be positive")
        if self.provisional_price_window_before_days < 0:
            raise ValueError("The provisional price window cannot begin after the event")
        if self.provisional_price_window_after_days < 35:
            raise ValueError("The provisional price window cannot cover twenty sessions")
        if self.holding_sessions != 20:
            raise ValueError("The first validation return horizon is frozen to twenty sessions")


@dataclass(frozen=True)
class ValidationMarketPlanTables:
    """Pure outcome-blind market-collection tables before persistence."""

    event_targets: pd.DataFrame
    identity_anchors: pd.DataFrame
    filing_evidence_targets: pd.DataFrame
    issuer_price_spans: pd.DataFrame
    collection_batches: pd.DataFrame
    benchmark_plan: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ValidationMarketPlanResult:
    """Immutable validation market-data plan."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_market_plan(
    target_features: pd.DataFrame,
    filing_catalog: pd.DataFrame,
    config: ValidationMarketPlanConfig | None = None,
) -> ValidationMarketPlanTables:
    """Select collection targets without reading provider coverage or outcomes."""
    cfg = config or ValidationMarketPlanConfig()
    cfg.validate()
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(target_features.columns))
    if forbidden:
        raise FilingCatalogError(f"Validation feature input contains outcome columns: {forbidden}")
    _require_columns(
        target_features,
        {
            "target_id",
            "validation_event_id",
            "cik",
            "target_accession_number",
            "form",
            "report_date",
            "accepted_at",
            "evidence_role",
            "within_documented_massive_basic_window",
            "gross_margin_history_eligible",
            "gross_margin_applicability_complete_case",
            "collection_wave",
        },
        "validation feature targets",
    )
    _require_columns(
        filing_catalog,
        {
            "cik",
            "accession_number",
            "accepted_at",
            "primary_document",
            "source_url",
        },
        "filing catalog",
    )
    features = target_features.copy()
    features["cik"] = features["cik"].astype(str).str.zfill(10)
    features["target_accession_number"] = features["target_accession_number"].astype(str)
    features["accepted_at"] = pd.to_datetime(features["accepted_at"], utc=True, errors="coerce")
    if features["target_id"].isna().any() or features["target_id"].duplicated().any():
        raise FilingCatalogError("Validation feature target IDs must be non-null and unique")
    if features["accepted_at"].isna().any():
        raise FilingCatalogError("Validation feature targets contain invalid acceptance clocks")
    unobserved = features["evidence_role"].eq("unobserved_validation_candidate")
    basic_window = features["within_documented_massive_basic_window"].fillna(False).astype(bool)
    history_eligible = features["gross_margin_history_eligible"].fillna(False).astype(bool)
    complete = features["gross_margin_applicability_complete_case"].fillna(False).astype(bool)
    features["identity_evidence_candidate"] = unobserved & basic_window & history_eligible
    features["price_collection_candidate"] = unobserved & basic_window & complete
    selected = features.loc[features["identity_evidence_candidate"]].copy()
    if selected.empty:
        raise FilingCatalogError("No validation events qualify for outcome-blind identity planning")
    selected["market_event_id"] = selected["validation_event_id"].map(
        lambda event_id: _stable_hash(
            {"validation_event_id": str(event_id), "contract_version": cfg.version}
        )
    )
    selected["provisional_price_window_start"] = (
        selected["accepted_at"]
        - pd.to_timedelta(cfg.provisional_price_window_before_days, unit="D")
    ).dt.date.astype(str)
    selected["provisional_price_window_end"] = (
        selected["accepted_at"] + pd.to_timedelta(cfg.provisional_price_window_after_days, unit="D")
    ).dt.date.astype(str)
    selected["historical_security_identity_ready"] = False
    selected["official_calendar_window_ready"] = False
    selected["filing_time_industry_ready"] = False
    selected["price_collection_authorized"] = False
    selected["return_outcome_read_authorized"] = False
    selected["selection_uses_return_values"] = False
    selected["replacement_after_collection_failure_allowed"] = False
    selected["market_plan_status"] = np.where(
        selected["price_collection_candidate"],
        "awaiting_identity_calendar_and_industry",
        "history_eligible_but_current_features_incomplete",
    )
    selected = selected.sort_values(
        ["accepted_at", "cik", "target_accession_number"], kind="mergesort"
    ).reset_index(drop=True)

    price_candidates = selected.loc[selected["price_collection_candidate"]].copy()
    anchors = _identity_anchors(price_candidates, cfg)
    catalog = filing_catalog.copy()
    catalog["cik"] = catalog["cik"].astype(str).str.zfill(10)
    catalog["accession_number"] = catalog["accession_number"].astype(str)
    catalog["catalog_accepted_at"] = pd.to_datetime(
        catalog["accepted_at"], utc=True, errors="coerce"
    )
    catalog = catalog.drop(columns=["accepted_at"])
    catalog["index_url"] = catalog.apply(
        lambda row: _filing_index_url(str(row["cik"]), str(row["accession_number"])),
        axis="columns",
    )
    catalog["primary_document_url"] = catalog["source_url"]
    catalog_keys = selected[["cik", "target_accession_number"]].rename(
        columns={"target_accession_number": "accession_number"}
    )
    catalog = catalog.merge(catalog_keys, on=["cik", "accession_number"], how="inner")
    if catalog.duplicated(["cik", "accession_number"]).any():
        raise FilingCatalogError("Validation filing evidence is not unique by CIK and accession")
    evidence = selected.merge(
        catalog[
            [
                "cik",
                "accession_number",
                "catalog_accepted_at",
                "index_url",
                "primary_document_url",
            ]
        ],
        left_on=["cik", "target_accession_number"],
        right_on=["cik", "accession_number"],
        how="left",
        validate="one_to_one",
    )
    evidence["catalog_clock_match"] = evidence["accepted_at"].eq(evidence["catalog_accepted_at"])
    evidence["sec_index_collection_required"] = (
        evidence["catalog_clock_match"] & evidence["index_url"].notna()
    )
    evidence["sec_primary_document_collection_required"] = False
    evidence["sec_evidence_status"] = np.where(
        evidence["sec_index_collection_required"],
        "awaiting_filing_index_for_sic",
        "catalog_accession_or_clock_mismatch",
    )
    spans = _issuer_price_spans(price_candidates, cfg)
    batches = _collection_batches(spans, cfg)
    benchmark = _benchmark_plan(selected, cfg)
    exact_catalog = int(evidence["catalog_clock_match"].sum())
    gates = pd.DataFrame(
        [
            _gate(
                "feature_applicability",
                int(selected["price_collection_candidate"].sum()),
                len(selected),
                bool(selected["price_collection_candidate"].any()),
                "Complete L/D/I among pre-event history-eligible events inside the price window",
            ),
            _gate(
                "filing_catalog_clock_match",
                exact_catalog,
                len(selected),
                exact_catalog == len(selected),
                "Exact CIK, accession, and accepted_at before SEC evidence collection",
            ),
            _gate(
                "historical_security_identity",
                0,
                len(price_candidates),
                False,
                "Endpoint reference anchors must establish historical security intervals",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(price_candidates),
                False,
                "Remains locked until identity, calendar, actions, industry, "
                "and delisting gates pass",
            ),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "input_feature_events": len(features),
        "unobserved_basic_window_history_eligible_events": len(selected),
        "unobserved_basic_window_complete_price_candidates": len(price_candidates),
        "price_candidate_issuers": int(price_candidates["cik"].nunique()),
        "identity_anchor_queries": len(anchors),
        "filing_index_evidence_targets": int(evidence["sec_index_collection_required"].sum()),
        "issuer_price_spans": len(spans),
        "issuer_collection_batches": int(batches["collection_batch"].nunique()),
        "benchmark_plan_rows": len(benchmark),
        "historical_security_identity_ready": False,
        "official_calendar_window_ready": False,
        "filing_time_industry_ready": False,
        "price_collection_authorized": False,
        "outcome_values_loaded": False,
        "selection_uses_return_values": False,
        "selection_uses_provider_coverage": False,
        "replacement_after_collection_failure_allowed": False,
        "model_evaluation_authorized": False,
        "identity_cost_control": (
            "Query first and last eligible event per issuer; propagate no interval until both "
            "anchors prove the same security identity."
        ),
        "sector_benchmark_status": (
            "awaiting exact filing-time SIC and a separately versioned benchmark mapping"
        ),
        "config": asdict(cfg),
    }
    event_columns = [
        "market_event_id",
        "validation_event_id",
        "target_id",
        "cik",
        "target_accession_number",
        "form",
        "report_date",
        "accepted_at",
        "collection_wave",
        "gross_margin_history_eligible",
        "gross_margin_applicability_complete_case",
        "identity_evidence_candidate",
        "price_collection_candidate",
        "provisional_price_window_start",
        "provisional_price_window_end",
        "historical_security_identity_ready",
        "official_calendar_window_ready",
        "filing_time_industry_ready",
        "price_collection_authorized",
        "return_outcome_read_authorized",
        "selection_uses_return_values",
        "replacement_after_collection_failure_allowed",
        "market_plan_status",
    ]
    evidence_columns = [
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
    return ValidationMarketPlanTables(
        event_targets=selected[event_columns].copy(),
        identity_anchors=anchors,
        filing_evidence_targets=evidence[evidence_columns].copy(),
        issuer_price_spans=spans,
        collection_batches=batches,
        benchmark_plan=benchmark,
        gates=gates,
        report=report,
    )


def build_validation_market_plan(
    validation_plan_root: Path,
    validation_feature_root: Path,
    catalog_root: Path,
    *,
    output_root: Path = Path("data/processed/validation_market_plan"),
    config: ValidationMarketPlanConfig | None = None,
) -> ValidationMarketPlanResult:
    """Persist a manifest-tied validation market plan without any outcome data."""
    cfg = config or ValidationMarketPlanConfig()
    cfg.validate()
    plan_report = _read_json(validation_plan_root / "validation_expansion_report.json")
    feature_report = _read_json(validation_feature_root / "validation_feature_report.json")
    plan_manifest = _sha256_file(validation_plan_root / "output_manifest.json")
    feature_manifest = _sha256_file(validation_feature_root / "output_manifest.json")
    catalog_manifest = _sha256_file(catalog_root / "output_manifest.json")
    if feature_report.get("plan_manifest_sha256") != plan_manifest:
        raise FilingCatalogError("Validation features do not tie to the selected frozen plan")
    if feature_report.get("catalog_manifest_sha256") != catalog_manifest:
        raise FilingCatalogError("Validation features do not tie to the selected filing catalog")
    if any(
        bool(feature_report.get(field))
        for field in (
            "outcome_values_loaded",
            "market_returns_loaded",
            "model_evaluation_authorized",
        )
    ):
        raise FilingCatalogError("Validation feature bundle has already opened outcome data")
    features = load_catalog_table(validation_feature_root, "target_filing_features.parquet")
    catalog = load_catalog_table(catalog_root, "filing_catalog.parquet")
    tables = calculate_validation_market_plan(features, catalog, cfg)
    payload = json.dumps(
        {
            "validation_plan_manifest_sha256": plan_manifest,
            "validation_feature_manifest_sha256": feature_manifest,
            "catalog_manifest_sha256": catalog_manifest,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationMarketPlanResult(
            target.resolve(), _read_json(target / "validation_market_plan_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete validation market plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "validation_market_event_targets.parquet": tables.event_targets,
        "validation_identity_anchor_targets.parquet": tables.identity_anchors,
        "validation_filing_evidence_targets.parquet": tables.filing_evidence_targets,
        "validation_issuer_price_spans.parquet": tables.issuer_price_spans,
        "validation_market_collection_batches.parquet": tables.collection_batches,
        "validation_benchmark_plan.parquet": tables.benchmark_plan,
        "validation_market_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "validation_market_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_plan_version": plan_report["dataset_version"],
        "validation_feature_version": feature_report["dataset_version"],
        "validation_plan_manifest_sha256": plan_manifest,
        "validation_feature_manifest_sha256": feature_manifest,
        "catalog_manifest_sha256": catalog_manifest,
        "artifacts": artifacts,
    }
    report_path = staging / "validation_market_plan_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "validation_plan_manifest_sha256": plan_manifest,
                "validation_feature_manifest_sha256": feature_manifest,
                "catalog_manifest_sha256": catalog_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationMarketPlanResult(target.resolve(), report)


def _identity_anchors(
    price_candidates: pd.DataFrame,
    config: ValidationMarketPlanConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cik, group in price_candidates.groupby("cik", sort=True):
        ordered = group.sort_values(["accepted_at", "target_accession_number"], kind="mergesort")
        positions = [("first_candidate", 0)]
        if len(ordered) > 1:
            positions.append(("last_candidate", len(ordered) - 1))
        for role, position in positions:
            event = ordered.iloc[position]
            query_date = pd.Timestamp(event["accepted_at"]).date().isoformat()
            rows.append(
                {
                    "identity_anchor_id": _stable_hash(
                        {
                            "cik": str(cik),
                            "query_date": query_date,
                            "anchor_role": role,
                            "contract_version": config.version,
                        }
                    ),
                    "cik": str(cik),
                    "anchor_role": role,
                    "validation_event_id": str(event["validation_event_id"]),
                    "accession_number": str(event["target_accession_number"]),
                    "accepted_at": event["accepted_at"],
                    "query_date": query_date,
                    "provider_query_method": "cik_and_point_in_time_date",
                    "query_active_first": True,
                    "query_inactive_only_after_no_active_match": True,
                    "automatic_interval_propagation_allowed": False,
                    "selection_uses_return_values": False,
                }
            )
    return pd.DataFrame(rows)


def _issuer_price_spans(
    price_candidates: pd.DataFrame,
    config: ValidationMarketPlanConfig,
) -> pd.DataFrame:
    rows = []
    for cik, group in price_candidates.groupby("cik", sort=True):
        start = pd.Timestamp(group["accepted_at"].min()) - timedelta(
            days=config.provisional_price_window_before_days
        )
        end = pd.Timestamp(group["accepted_at"].max()) + timedelta(
            days=config.provisional_price_window_after_days
        )
        rows.append(
            {
                "cik": str(cik),
                "price_candidate_events": len(group),
                "first_event_accepted_at": group["accepted_at"].min(),
                "last_event_accepted_at": group["accepted_at"].max(),
                "provisional_price_span_start": start.date().isoformat(),
                "provisional_price_span_end": end.date().isoformat(),
                "historical_ticker": pd.NA,
                "composite_figi": pd.NA,
                "share_class_figi": pd.NA,
                "identity_interval_ready": False,
                "price_query_authorized": False,
                "selection_uses_return_values": False,
            }
        )
    return pd.DataFrame(rows)


def _collection_batches(
    spans: pd.DataFrame,
    config: ValidationMarketPlanConfig,
) -> pd.DataFrame:
    frame = spans.copy()
    frame["issuer_sampling_hash"] = frame["cik"].map(
        lambda cik: _stable_hash({"cik": str(cik), "contract_version": config.version})
    )
    frame = frame.sort_values(
        ["price_candidate_events", "issuer_sampling_hash", "cik"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["market_collection_order"] = np.arange(1, len(frame) + 1)
    frame["collection_batch"] = (
        frame["market_collection_order"] - 1
    ) // config.issuer_batch_size + 1
    frame["collection_status"] = "awaiting_identity_anchors"
    return frame[
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
    ]


def _benchmark_plan(
    events: pd.DataFrame,
    config: ValidationMarketPlanConfig,
) -> pd.DataFrame:
    start = events["provisional_price_window_start"].min()
    end = events["provisional_price_window_end"].max()
    return pd.DataFrame(
        [
            {
                "benchmark_role": "broad_market_total_return_proxy",
                "benchmark_ticker": "SPY",
                "price_span_start": start,
                "price_span_end": end,
                "mapping_status": "explicit_instrument",
                "price_collection_authorized": False,
                "authorization_dependency": "official_calendar_event_windows",
                "selection_uses_return_values": False,
            },
            {
                "benchmark_role": "filing_time_industry_total_return_proxy",
                "benchmark_ticker": pd.NA,
                "price_span_start": start,
                "price_span_end": end,
                "mapping_status": "awaiting_filing_time_sic_and_versioned_mapping",
                "price_collection_authorized": False,
                "authorization_dependency": "filing_time_industry_evidence",
                "selection_uses_return_values": False,
            },
        ]
    )


def _gate(
    gate: str,
    numerator: int,
    denominator: int,
    passed: bool,
    meaning: str,
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
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _filing_index_url(cik: str, accession_number: str) -> str:
    cik_unpadded = str(int(cik))
    accession_compact = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_compact}/"
        f"{accession_number}-index.html"
    )


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
