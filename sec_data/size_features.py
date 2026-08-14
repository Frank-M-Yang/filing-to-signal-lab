"""Point-in-time company-size features from accession-matched SEC Companyfacts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.bulk import SecBulkSnapshot
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

COMPANYFACTS_MEMBER: Final[re.Pattern[str]] = re.compile(r"(?:^|/)CIK(\d{10})\.json$")
SIZE_TAG: Final[str] = "Assets"


@dataclass(frozen=True)
class SizeFeatureResult:
    """Immutable event facts, issuer-year anchors, coverage, and report."""

    output_root: Path
    report: dict[str, Any]


def build_event_time_size_features(
    catalog_root: Path,
    companyfacts_snapshot: SecBulkSnapshot,
    *,
    output_root: Path = Path("data/processed/sec_size_features"),
) -> SizeFeatureResult:
    """Extract original-filing total assets with accession acceptance timestamps."""
    if companyfacts_snapshot.kind != "companyfacts":
        raise FilingCatalogError("Size features require a companyfacts bulk snapshot")
    companyfacts_path = Path(companyfacts_snapshot.path)
    if (
        not companyfacts_path.is_file()
        or _sha256_file(companyfacts_path) != companyfacts_snapshot.sha256
    ):
        raise FilingCatalogError("Companyfacts snapshot does not tie to its SHA-256")
    catalog_report = json.loads((catalog_root / "catalog_report.json").read_text(encoding="utf-8"))
    if not bool(catalog_report.get("archive_coverage_complete")):
        raise FilingCatalogError("Size extraction requires complete filing archive coverage")
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    filings = load_catalog_table(catalog_root, "filing_catalog.parquet")
    inventory = load_catalog_table(catalog_root, "issuer_year_inventory.parquet")
    periodic = filings.loc[filings["form"].isin(["10-Q", "10-K"])].copy()
    periodic = periodic[
        [
            "filing_id",
            "cik",
            "accession_number",
            "form",
            "report_date",
            "accepted_at",
            "source_url",
        ]
    ]
    if periodic.duplicated(["cik", "accession_number"]).any():
        raise FilingCatalogError("Original periodic catalog is not subject-accession unique")
    representative_ciks = set(
        inventory.loc[inventory["representative_candidate"].astype(bool), "cik"].astype(str)
    )
    periodic = periodic.loc[periodic["cik"].astype(str).isin(representative_ciks)].copy()
    filing_lookup = {
        (str(row.cik), str(row.accession_number)): row for row in periodic.itertuples(index=False)
    }
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejection_rows: list[dict[str, Any]] = []
    parsed_companyfacts = 0
    missing_companyfacts_ciks = set(representative_ciks)
    with zipfile.ZipFile(companyfacts_path) as archive:
        for member in archive.infolist():
            matched = COMPANYFACTS_MEMBER.search(member.filename.replace("\\", "/"))
            if matched is None or member.is_dir() or matched.group(1) not in representative_ciks:
                continue
            cik = matched.group(1)
            member_bytes = archive.read(member)
            member_sha256 = hashlib.sha256(member_bytes).hexdigest()
            try:
                document = json.loads(member_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                rejection_rows.append(
                    _rejection(cik, "invalid_companyfacts_json", str(exc), member.filename, "")
                )
                continue
            if not isinstance(document, dict):
                rejection_rows.append(
                    _rejection(
                        cik,
                        "invalid_companyfacts_object",
                        "Top-level payload is not an object",
                        member.filename,
                        "",
                    )
                )
                continue
            parsed_companyfacts += 1
            missing_companyfacts_ciks.discard(cik)
            rows = _asset_rows(document)
            for occurrence, fact in enumerate(rows):
                accession = str(fact.get("accn") or "")
                lookup_key = (cik, accession)
                filing = filing_lookup.get(lookup_key)
                if filing is None:
                    continue
                fact_end = str(fact.get("end") or "")
                if fact_end != str(filing.report_date):
                    continue
                value = _positive_float(fact.get("val"))
                if value is None:
                    rejection_rows.append(
                        _rejection(
                            cik,
                            "invalid_total_assets_value",
                            f"Assets value is not positive: {fact.get('val')}",
                            member.filename,
                            accession,
                        )
                    )
                    continue
                candidates.setdefault(lookup_key, []).append(
                    {
                        "filing_id": str(filing.filing_id),
                        "cik": cik,
                        "accession_number": accession,
                        "form": str(filing.form),
                        "report_date": str(filing.report_date),
                        "accepted_at": str(filing.accepted_at),
                        "source_url": str(filing.source_url),
                        "taxonomy": "us-gaap",
                        "tag": SIZE_TAG,
                        "unit": "USD",
                        "total_assets_usd": value,
                        "companyfacts_filed_date_not_used": str(fact.get("filed") or ""),
                        "companyfacts_frame": str(fact.get("frame") or ""),
                        "source_member": member.filename,
                        "source_member_sha256": member_sha256,
                        "source_zip_sha256": companyfacts_snapshot.sha256,
                        "source_occurrence": occurrence,
                    }
                )

    selected_rows: list[dict[str, Any]] = []
    conflicting_fact_events = 0
    duplicate_same_value_events = 0
    for key, rows in sorted(candidates.items()):
        distinct_values = sorted({float(row["total_assets_usd"]) for row in rows})
        if len(distinct_values) != 1:
            conflicting_fact_events += 1
            rejection_rows.append(
                _rejection(
                    key[0],
                    "conflicting_total_assets_values",
                    json.dumps(distinct_values),
                    str(rows[0]["source_member"]),
                    key[1],
                )
            )
            continue
        if len(rows) > 1:
            duplicate_same_value_events += 1
        selected = rows[0].copy()
        selected["source_occurrences"] = len(rows)
        selected["size_fact_status"] = "accession_matched_original_period_end_assets"
        selected_rows.append(selected)
    event_assets = pd.DataFrame(selected_rows)
    if event_assets.empty:
        raise FilingCatalogError("No point-in-time Assets facts were selected")
    event_assets["accepted_at"] = pd.to_datetime(event_assets["accepted_at"], utc=True)
    event_assets["calendar_year"] = event_assets["accepted_at"].dt.year
    event_assets["log_total_assets"] = np.log(event_assets["total_assets_usd"])
    event_assets = event_assets.sort_values(["accepted_at", "cik", "accession_number"])

    anchors = (
        event_assets.sort_values(["accepted_at", "cik", "accession_number"])
        .groupby(["cik", "calendar_year"], as_index=False)
        .first()
    )
    anchors["asset_size_percentile"] = anchors.groupby("calendar_year")["total_assets_usd"].rank(
        method="average", pct=True
    )
    anchors["asset_size_stratum"] = pd.cut(
        anchors["asset_size_percentile"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=[
            "asset_q1_smallest",
            "asset_q2",
            "asset_q3",
            "asset_q4_largest",
        ],
        include_lowest=True,
    ).astype("string")
    anchors["size_anchor_rule"] = "first_original_periodic_filing_with_assets_in_year"
    anchors["size_measure_not_market_cap"] = True

    representative = inventory.loc[inventory["representative_candidate"].astype(bool)].copy()
    coverage = representative[["cik", "calendar_year"]].merge(
        anchors[["cik", "calendar_year", "total_assets_usd", "asset_size_stratum"]],
        on=["cik", "calendar_year"],
        how="left",
        validate="one_to_one",
    )
    coverage_by_year = (
        coverage.groupby("calendar_year", as_index=False)
        .agg(
            candidate_issuer_years=("cik", "size"),
            issuer_years_with_size=("total_assets_usd", "count"),
        )
        .sort_values("calendar_year")
    )
    coverage_by_year["size_coverage"] = (
        coverage_by_year["issuer_years_with_size"] / coverage_by_year["candidate_issuer_years"]
    )

    dataset_payload = json.dumps(
        {
            "catalog": catalog_report["dataset_version"],
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "companyfacts_sha256": companyfacts_snapshot.sha256,
            "tag": SIZE_TAG,
            "selection": "same accession original form USD exact report date",
        },
        sort_keys=True,
    ).encode()
    dataset_version = "event-time-size-v1-" + hashlib.sha256(dataset_payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads(
            (target / "size_feature_report.json").read_text(encoding="utf-8")
        )
        return SizeFeatureResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete size-feature directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    outputs = {
        "event_time_total_assets.parquet": event_assets,
        "issuer_year_size_anchors.parquet": anchors,
        "size_coverage_by_year.parquet": coverage_by_year,
        "size_fact_rejections.parquet": pd.DataFrame(rejection_rows),
    }
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
    artifacts = {
        filename: {
            "rows": len(frame),
            "bytes": (staging / filename).stat().st_size,
            "sha256": _sha256_file(staging / filename),
        }
        for filename, frame in outputs.items()
    }
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": catalog_report["dataset_version"],
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "companyfacts_snapshot": asdict(companyfacts_snapshot),
        "representative_candidate_ciks": len(representative_ciks),
        "companyfacts_ciks_parsed": parsed_companyfacts,
        "companyfacts_ciks_missing": len(missing_companyfacts_ciks),
        "original_periodic_filings": len(periodic),
        "event_filings_with_total_assets": len(event_assets),
        "event_total_assets_coverage": len(event_assets) / max(len(periodic), 1),
        "candidate_issuer_years": len(representative),
        "issuer_years_with_size_anchor": len(anchors),
        "issuer_year_size_coverage": len(anchors) / max(len(representative), 1),
        "conflicting_fact_events_rejected": conflicting_fact_events,
        "duplicate_same_value_events_collapsed": duplicate_same_value_events,
        "filed_date_used_as_availability": False,
        "size_measure": "us-gaap:Assets in USD; asset quartiles are not market-cap tiers",
        "artifacts": artifacts,
    }
    (staging / "size_feature_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "companyfacts_sha256": companyfacts_snapshot.sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(staging / "size_feature_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return SizeFeatureResult(target.resolve(), report)


def _asset_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    facts = document.get("facts")
    us_gaap = facts.get("us-gaap") if isinstance(facts, dict) else None
    concept = us_gaap.get(SIZE_TAG) if isinstance(us_gaap, dict) else None
    units = concept.get("units") if isinstance(concept, dict) else None
    rows = units.get("USD") if isinstance(units, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0.0 else None


def _rejection(
    cik: str,
    reason: str,
    detail: str,
    source_member: str,
    accession_number: str,
) -> dict[str, Any]:
    return {
        "cik": cik,
        "accession_number": accession_number,
        "reason": reason,
        "detail": detail,
        "source_member": source_member,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
