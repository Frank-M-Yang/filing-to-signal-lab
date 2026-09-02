"""Outcome-blind calendar-year cohorts for entitlement-limited exploration."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table


@dataclass(frozen=True)
class CalendarYearExploratoryCohortConfig:
    """Pre-outcome rule selecting events from one filing calendar year."""

    event_year: int = 2025
    version: str = "calendar-year-exploratory-cohort-v1"

    def validate(self) -> None:
        """Reject implausible years and blank version labels."""
        if not 2000 <= self.event_year <= 2100:
            raise FilingCatalogError("Exploratory cohort year is outside the supported range")
        if not self.version.strip():
            raise FilingCatalogError("Exploratory cohort version must not be blank")


@dataclass(frozen=True)
class CalendarYearExploratoryCohortResult:
    """Immutable registry and audit for one exploratory event-year cohort."""

    output_root: Path
    report: dict[str, Any]


def build_calendar_year_exploratory_cohort(
    reference_root: Path,
    price_root: Path,
    calendar_root: Path,
    *,
    output_root: Path = Path("data/processed/exploratory_price_cohort"),
    config: CalendarYearExploratoryCohortConfig | None = None,
) -> CalendarYearExploratoryCohortResult:
    """Freeze a date-only exploratory cohort without inspecting returns or actions."""
    cfg = config or CalendarYearExploratoryCohortConfig()
    cfg.validate()
    reference_manifest_sha256 = _sha256_file(reference_root / "output_manifest.json")
    price_manifest_sha256 = _sha256_file(price_root / "output_manifest.json")
    calendar_manifest_sha256 = _sha256_file(calendar_root / "output_manifest.json")
    reference_report = _read_json(reference_root / "massive_reference_report.json")
    price_report = _read_json(price_root / "massive_price_report.json")
    calendar_report = _read_json(calendar_root / "exchange_calendar_report.json")
    if price_report.get("reference_manifest_sha256") != reference_manifest_sha256:
        raise FilingCatalogError("Price audit does not tie to the selected reference audit")
    if calendar_report.get("price_manifest_sha256") != price_manifest_sha256:
        raise FilingCatalogError("Calendar audit does not tie to the selected price audit")

    targets = load_catalog_table(price_root, "massive_daily_price_targets.parquet")
    price_audit = load_catalog_table(price_root, "sample_massive_price_audit.parquet")
    alignment = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
    if targets["price_target_id"].duplicated().any():
        raise FilingCatalogError("Exploratory cohort price targets are not unique")
    registry = targets[
        [
            "price_target_id",
            "target_id",
            "cik",
            "accepted_at",
            "query_date",
            "form",
            "accession_number",
            "selected_ticker",
            "selected_exchange_mic",
        ]
    ].copy()
    registry["event_year"] = pd.to_datetime(registry["query_date"], errors="raise").dt.year
    registry["selection_rule"] = f"query_date_calendar_year_equals_{cfg.event_year}"
    registry["selected_exploratory_cohort"] = registry["event_year"].eq(cfg.event_year)
    registry["exclusion_reason"] = pd.Series(pd.NA, index=registry.index, dtype="string")
    registry.loc[~registry["selected_exploratory_cohort"], "exclusion_reason"] = (
        "outside_registered_event_year"
    )
    diagnostics = price_audit[
        [
            "price_target_id",
            "provider_price_query_complete",
            "observed_event_window_candidate",
        ]
    ].merge(
        alignment[["price_target_id", "formal_event_window_ready"]],
        on="price_target_id",
        how="outer",
        validate="one_to_one",
    )
    registry = registry.merge(
        diagnostics,
        on="price_target_id",
        how="left",
        validate="one_to_one",
    )
    diagnostic_columns = [
        "provider_price_query_complete",
        "observed_event_window_candidate",
        "formal_event_window_ready",
    ]
    registry[diagnostic_columns] = registry[diagnostic_columns].fillna(False).astype(bool)
    registry = registry.sort_values("price_target_id").reset_index(drop=True)
    selected = registry[registry["selected_exploratory_cohort"]].copy()
    if selected.empty:
        raise FilingCatalogError("Calendar-year exploratory cohort selected no events")
    selected = selected.sort_values(["query_date", "price_target_id"]).reset_index(drop=True)
    selected["exploratory_cohort_order"] = np.arange(1, len(selected) + 1)
    selection_columns = [
        "price_target_id",
        "target_id",
        "cik",
        "accepted_at",
        "query_date",
        "form",
        "accession_number",
        "selected_ticker",
        "selected_exchange_mic",
        "event_year",
        "selection_rule",
        "exploratory_cohort_order",
    ]
    selected_events = selected[selection_columns].copy()
    payload = json.dumps(
        {
            "reference_manifest_sha256": reference_manifest_sha256,
            "price_manifest_sha256": price_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
            "config": asdict(cfg),
            "selected_price_target_ids": selected_events["price_target_id"].tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        "calendar-year-exploratory-cohort-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return CalendarYearExploratoryCohortResult(
            target.resolve(), _read_json(target / "exploratory_cohort_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete exploratory-cohort directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "exploratory_cohort_registry.parquet": registry,
        "selected_exploratory_events.parquet": selected_events,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    selected_price_ready = selected["provider_price_query_complete"].astype(bool)
    selected_calendar_ready = selected["formal_event_window_ready"].astype(bool)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "reference_version": reference_report["dataset_version"],
        "reference_manifest_sha256": reference_manifest_sha256,
        "price_version": price_report["dataset_version"],
        "price_manifest_sha256": price_manifest_sha256,
        "calendar_version": calendar_report["dataset_version"],
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "source_price_targets": len(registry),
        "selected_events": len(selected),
        "excluded_events": int((~registry["selected_exploratory_cohort"]).sum()),
        "selected_price_ready": int(selected_price_ready.sum()),
        "selected_calendar_ready": int(selected_calendar_ready.sum()),
        "selection_rule": f"query_date_calendar_year_equals_{cfg.event_year}",
        "selection_uses_provider_price_coverage": False,
        "selection_uses_returns": False,
        "selection_uses_corporate_actions": False,
        "selection_uses_delisting_outcomes": False,
        "exploratory_only": True,
        "primary_effectiveness_eligible": False,
        "downstream_evidence_collection_gate": bool(
            selected_price_ready.all() and selected_calendar_ready.all()
        ),
        "artifacts": artifacts,
    }
    report_path = staging / "exploratory_cohort_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "reference_manifest_sha256": reference_manifest_sha256,
                "price_manifest_sha256": price_manifest_sha256,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return CalendarYearExploratoryCohortResult(target.resolve(), report)


def _read_json(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FilingCatalogError(f"Expected a JSON object: {path}")
    return value


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
