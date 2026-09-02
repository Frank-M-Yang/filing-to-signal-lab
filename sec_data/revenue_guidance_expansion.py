"""Outcome-blind issuer-year expansion plan for quarterly revenue guidance."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import (
    GuidanceEvidenceError,
    GuidancePlanTables,
    calculate_guidance_plan,
)

REVENUE_GUIDANCE_EXPANSION_VERSION: Final[str] = "revenue-guidance-expansion-plan-v1"


@dataclass(frozen=True)
class RevenueGuidanceExpansionTables:
    """Frozen issuer-year sample and its prior-guidance document plan."""

    event_universe: pd.DataFrame
    guidance_plan: GuidancePlanTables
    report: dict[str, Any]


@dataclass(frozen=True)
class RevenueGuidanceExpansionResult:
    """Immutable persisted revenue-guidance expansion plan."""

    output_root: Path
    report: dict[str, Any]


def calculate_revenue_guidance_expansion(
    filing_targets: pd.DataFrame,
    quarterly_history: pd.DataFrame,
    filing_catalog: pd.DataFrame,
) -> RevenueGuidanceExpansionTables:
    """Select one deterministic revenue-ready event per issuer-year without values."""
    _require_columns(
        filing_targets,
        {
            "validation_event_id",
            "cik",
            "accepted_at",
            "actual_available_at",
            "filing_accession",
            "fiscal_year",
            "fiscal_quarter",
        },
        "target filing features",
    )
    _require_columns(
        quarterly_history,
        {"filing_accession", "revenue", "unit"},
        "quarterly history",
    )
    quarterly = quarterly_history[["filing_accession", "revenue", "unit"]].drop_duplicates(
        "filing_accession"
    )
    joined = filing_targets.merge(
        quarterly,
        on="filing_accession",
        how="left",
        validate="many_to_one",
    )
    accepted = pd.to_datetime(joined["accepted_at"], utc=True, errors="coerce")
    available = pd.to_datetime(joined["actual_available_at"], utc=True, errors="coerce")
    ready = (
        joined["revenue"].notna()
        & joined["unit"].astype(str).eq("USD")
        & accepted.notna()
        & available.notna()
        & available.le(accepted)
    )
    eligible = joined.loc[ready].copy()
    eligible["accepted_at"] = accepted.loc[ready]
    eligible["calendar_year"] = eligible["accepted_at"].dt.year
    eligible["selection_rank"] = (
        eligible["validation_event_id"]
        .astype(str)
        .map(
            lambda event_id: hashlib.sha256(
                f"{REVENUE_GUIDANCE_EXPANSION_VERSION}:{event_id}".encode()
            ).hexdigest()
        )
    )
    selected = (
        eligible.sort_values(
            ["cik", "calendar_year", "selection_rank"],
            kind="mergesort",
        )
        .drop_duplicates(["cik", "calendar_year"])
        .reset_index(drop=True)
    )
    universe = selected[
        [
            "validation_event_id",
            "cik",
            "accepted_at",
            "actual_available_at",
            "filing_accession",
            "fiscal_year",
            "fiscal_quarter",
            "calendar_year",
            "selection_rank",
        ]
    ].copy()
    universe["revenue_actual_ready"] = True
    universe["market_state_ready"] = True
    universe["selection_uses_revenue_values"] = False
    universe["selection_uses_return_values"] = False
    guidance_plan = calculate_guidance_plan(
        universe[["validation_event_id", "cik", "accepted_at", "market_state_ready"]],
        filing_catalog,
    )
    report: dict[str, Any] = {
        "contract_version": REVENUE_GUIDANCE_EXPANSION_VERSION,
        "revenue_ready_events": len(eligible),
        "revenue_ready_issuers": int(eligible["cik"].nunique()),
        "sampled_issuer_year_events": len(universe),
        "sampled_issuers": int(universe["cik"].nunique()),
        "sampled_calendar_years": sorted(
            int(value) for value in universe["calendar_year"].unique()
        ),
        "prior_guidance_candidate_events": guidance_plan.report["selected_guidance_events"],
        "prior_guidance_candidate_issuers": guidance_plan.report["selected_guidance_issuers"],
        "unique_guidance_documents": guidance_plan.report["unique_guidance_documents"],
        "selection_uses_revenue_values": False,
        "selection_uses_return_values": False,
        "current_filing_text_loaded": False,
        "sampling_rule": "minimum_sha256_rank_per_cik_calendar_year",
    }
    return RevenueGuidanceExpansionTables(universe, guidance_plan, report)


def build_revenue_guidance_expansion(
    filing_feature_root: Path,
    filing_catalog_root: Path,
    *,
    output_root: Path = Path("data/sec/revenue_guidance_expansion/targets"),
) -> RevenueGuidanceExpansionResult:
    """Persist the frozen issuer-year expansion and SEC document targets."""
    input_hashes = {
        "filing_features": _sha256_file(filing_feature_root / "output_manifest.json"),
        "filing_catalog": _sha256_file(filing_catalog_root / "output_manifest.json"),
    }
    tables = calculate_revenue_guidance_expansion(
        load_catalog_table(filing_feature_root, "target_filing_features.parquet"),
        load_catalog_table(filing_feature_root, "quarterly_feature_history.parquet"),
        load_catalog_table(filing_catalog_root, "filing_catalog.parquet"),
    )
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "version": REVENUE_GUIDANCE_EXPANSION_VERSION,
            "sampling_rule": tables.report["sampling_rule"],
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        REVENUE_GUIDANCE_EXPANSION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return RevenueGuidanceExpansionResult(
            target.resolve(), _read_json(target / "revenue_guidance_expansion_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete revenue-guidance plan: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "revenue_guidance_event_universe.parquet": tables.event_universe,
        "guidance_event_links.parquet": tables.guidance_plan.event_links,
        "guidance_document_targets.parquet": tables.guidance_plan.document_targets,
        "guidance_sample_funnel.parquet": tables.guidance_plan.sample_funnel,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "revenue_guidance_expansion_report.json"
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
    return RevenueGuidanceExpansionResult(target.resolve(), report)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GuidanceEvidenceError(f"{label} lacks required columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
