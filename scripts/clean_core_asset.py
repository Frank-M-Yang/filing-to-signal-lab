"""Freeze and clean the 82-company Global 500 gross-margin core asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.event_cleaning import (
    CompanyEventCleaningResult,
    EventCleaningConfig,
    clean_company_quarterly_events,
    derive_point_in_time_yoy_lags,
    freeze_core_asset,
    validate_cleaned_quarterly_events,
)

FACT_USECOLS = [
    "fact_id",
    "taxonomy",
    "tag",
    "unit",
    "value_numeric",
    "period_start",
    "period_end",
    "duration_days",
    "fiscal_year",
    "fiscal_period",
    "accession_number",
    "base_form",
    "is_amendment",
    "accepted_at",
    "source_snapshot_sha256",
]
FILING_USECOLS = [
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "accepted_at",
]


def main() -> int:
    """Run the deterministic fact-selection and PIT feature-cleaning pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-source",
        type=Path,
        default=Path("data/processed/global500_2026/gross_margin_ready_cohort.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/global500_2026/core_asset_v1"),
    )
    parser.add_argument("--sec-raw-root", type=Path, default=Path("data/sec/raw"))
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    frozen_at = datetime.now(timezone.utc).isoformat()
    core_source_sha256 = _sha256_file(args.core_source)
    source = pd.read_csv(args.core_source, dtype={"sec_cik": "string"})
    if len(source) != 82:
        raise ValueError(f"Expected the fixed 82-company core source, found {len(source)} rows")
    core, core_version = freeze_core_asset(source, frozen_at=frozen_at)
    core.to_csv(args.output_root / "core_asset_82.csv", index=False)
    print(f"Frozen core asset: {len(core)} companies, version={core_version}", flush=True)

    event_frames: list[pd.DataFrame] = []
    rejection_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    company_rows: list[dict[str, Any]] = []
    config = EventCleaningConfig(start_year=args.start_year, end_year=args.end_year)
    for position, (_, record) in enumerate(core.iterrows(), start=1):
        normalized = Path(str(record["normalized_directory"]))
        filings = pd.read_csv(normalized / "filings.csv", usecols=FILING_USECOLS)
        facts = pd.read_csv(normalized / "facts.csv", usecols=FACT_USECOLS)
        profile = _load_company_profile(
            args.sec_raw_root,
            cik=str(record["sec_cik"]),
            normalized_directory=normalized,
        )
        cleaned = clean_company_quarterly_events(
            {str(key): value for key, value in record.items()},
            filings,
            facts,
            core_asset_version=core_version,
            company_profile=profile,
            config=config,
        )
        _append_nonempty(event_frames, cleaned.events)
        _append_nonempty(rejection_frames, cleaned.rejections)
        _append_nonempty(audit_frames, cleaned.selection_audit)
        company_rows.append(_company_quality_row(record, cleaned, profile))
        print(
            f"{position:02d}/{len(core)} {record['sec_ticker']}: "
            f"accepted={len(cleaned.events)} rejected={len(cleaned.rejections)}",
            flush=True,
        )

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    events = derive_point_in_time_yoy_lags(events)
    events = events.sort_values(["actual_available_at", "company_id", "event_id"]).reset_index(
        drop=True
    )
    validate_cleaned_quarterly_events(events)
    rejections = (
        pd.concat(rejection_frames, ignore_index=True) if rejection_frames else pd.DataFrame()
    )
    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    quality = pd.DataFrame(company_rows).sort_values("fortune_rank").reset_index(drop=True)
    event_counts = events.groupby("company_id")["event_id"].nunique()
    level_counts = events.groupby("company_id")["analysis_ready_level"].sum()
    delta_counts = events.groupby("company_id")["analysis_ready_delta"].sum()
    quality["clean_event_count"] = quality["company_id"].map(event_counts).fillna(0).astype(int)
    quality["delta_ready_event_count"] = (
        quality["company_id"].map(delta_counts).fillna(0).astype(int)
    )
    quality["level_ready_event_count"] = (
        quality["company_id"].map(level_counts).fillna(0).astype(int)
    )
    research_ready = quality["level_ready_event_count"].ge(16) & quality[
        "delta_ready_event_count"
    ].ge(12)
    quality["cleaning_status"] = research_ready.map(
        {True: "research_ready", False: "manual_review_required"}
    )
    strict_level_events = events.loc[events["analysis_ready_level"].astype(bool)].copy()
    strict_delta_events = events.loc[events["analysis_ready_delta"].astype(bool)].copy()
    research_ready_companies = quality.loc[quality["cleaning_status"].eq("research_ready")].copy()

    events.to_csv(args.output_root / "quarterly_filing_events.csv", index=False)
    strict_level_events.to_csv(args.output_root / "strict_level_events.csv", index=False)
    strict_delta_events.to_csv(args.output_root / "strict_delta_events.csv", index=False)
    rejections.to_csv(args.output_root / "event_rejections.csv", index=False)
    audit.to_csv(args.output_root / "fact_selection_audit.csv", index=False)
    quality.to_csv(args.output_root / "company_quality.csv", index=False)
    research_ready_companies.to_csv(args.output_root / "research_ready_companies.csv", index=False)
    report = _build_report(
        core_version=core_version,
        frozen_at=frozen_at,
        core_source_sha256=core_source_sha256,
        config=config,
        core=core,
        events=events,
        rejections=rejections,
        audit=audit,
        quality=quality,
    )
    report_path = args.output_root / "cleaning_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_output_manifest(
        args.output_root,
        core_source=args.core_source,
        core_source_sha256=core_source_sha256,
        core_asset_version=core_version,
        row_counts={
            "core_asset_82.csv": len(core),
            "quarterly_filing_events.csv": len(events),
            "strict_level_events.csv": len(strict_level_events),
            "strict_delta_events.csv": len(strict_delta_events),
            "event_rejections.csv": len(rejections),
            "fact_selection_audit.csv": len(audit),
            "company_quality.csv": len(quality),
            "research_ready_companies.csv": len(research_ready_companies),
        },
    )
    print(
        f"DONE events={len(events)} delta_ready={int(events['analysis_ready_delta'].sum())} "
        "companies_ready="
        f"{int((quality['cleaning_status'] == 'research_ready').sum())}/{len(core)}",
        flush=True,
    )
    return 0


def _load_company_profile(
    raw_root: Path,
    *,
    cik: str,
    normalized_directory: Path,
) -> dict[str, Any]:
    manifest = json.loads((normalized_directory / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("snapshot_hashes", [])
    for digest in hashes:
        if not isinstance(digest, str):
            continue
        path = raw_root / cik.zfill(10) / digest[:2] / f"submissions-{digest}.json"
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                return document
    return {}


def _company_quality_row(
    record: pd.Series,
    cleaned: CompanyEventCleaningResult,
    profile: dict[str, Any],
) -> dict[str, Any]:
    methods = cleaned.events.get("derivation_method", pd.Series(dtype="string"))
    return {
        "company_id": str(record["sec_cik"]).zfill(10),
        "ticker": record["sec_ticker"],
        "company_name": record["company_name"],
        "fortune_rank": int(record["fortune_rank"]),
        "sic": profile.get("sic"),
        "sic_description": profile.get("sicDescription"),
        "source_10q_count": int(record["original_10q_filings"]),
        "selected_event_count_before_lags": len(cleaned.events),
        "rejected_event_count": len(cleaned.rejections),
        "direct_gross_profit_events": int(methods.eq("direct_gross_profit").sum()),
        "total_cost_derived_events": int(methods.eq("revenue_minus_total_cost").sum()),
        "scope_review_events": int(methods.eq("revenue_minus_scope_sensitive_cost").sum()),
    }


def _append_nonempty(frames: list[pd.DataFrame], frame: pd.DataFrame) -> None:
    if not frame.empty:
        frames.append(frame)


def _build_report(
    *,
    core_version: str,
    frozen_at: str,
    core_source_sha256: str,
    config: EventCleaningConfig,
    core: pd.DataFrame,
    events: pd.DataFrame,
    rejections: pd.DataFrame,
    audit: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "core_asset_version": core_version,
        "core_asset_frozen_at": frozen_at,
        "core_source_sha256": core_source_sha256,
        "core_companies": len(core),
        "research_period": [config.start_year, config.end_year],
        "accepted_events": len(events),
        "rejected_events": len(rejections),
        "selection_audit_rows": len(audit),
        "level_ready_events": int(events["analysis_ready_level"].sum()),
        "delta_ready_events": int(events["analysis_ready_delta"].sum()),
        "revised_events": int(events["was_revised"].sum()),
        "latest_source_types": events["revision_source_type"].value_counts().sort_index().to_dict(),
        "revised_event_source_types": events.loc[
            events["was_revised"].astype(bool), "revision_source_type"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
        "direct_gross_profit_events": int(
            events["derivation_method"].eq("direct_gross_profit").sum()
        ),
        "total_cost_derived_events": int(
            events["derivation_method"].eq("revenue_minus_total_cost").sum()
        ),
        "scope_review_events": int(
            events["derivation_method"].eq("revenue_minus_scope_sensitive_cost").sum()
        ),
        "industry_review_events": int(
            events["economic_scope_status"].eq("industry_review_required").sum()
        ),
        "plausibility_outlier_events": int(
            events["plausibility_status"].eq("outlier_review_required").sum()
        ),
        "companies_research_ready": int(quality["cleaning_status"].eq("research_ready").sum()),
        "companies_requiring_manual_review": int(
            quality["cleaning_status"].eq("manual_review_required").sum()
        ),
        "event_count_per_company": {
            "min": int(quality["clean_event_count"].min()),
            "median": float(quality["clean_event_count"].median()),
            "max": int(quality["clean_event_count"].max()),
        },
        "rejection_reasons": (
            rejections["reason"].value_counts().sort_index().to_dict()
            if not rejections.empty
            else {}
        ),
        "invariants": [
            "Only original 10-Q facts create PIT events",
            "Fact period end must equal filing report date",
            "Quarter duration must be 70-110 days",
            "Revenue and profit/cost must share unit and period start",
            "Latest/restated values are ex-post fields only",
            "Latest revenue and profit/cost come from one coherent filing accession",
            "YoY lag must be accepted strictly before the current event",
            "Strict delta events require both current and lag level events to pass review",
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_output_manifest(
    output_root: Path,
    *,
    core_source: Path,
    core_source_sha256: str,
    core_asset_version: str,
    row_counts: dict[str, int],
) -> None:
    artifacts = {
        name: {
            "bytes": (output_root / name).stat().st_size,
            "rows": rows,
            "sha256": _sha256_file(output_root / name),
        }
        for name, rows in sorted(row_counts.items())
    }
    manifest = {
        "core_asset_version": core_asset_version,
        "source": {
            "path": str(core_source),
            "sha256": core_source_sha256,
        },
        "artifacts": artifacts,
    }
    (output_root / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
