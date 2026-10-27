"""Freeze and inspect twenty new cached SEC issuers, then replay reviewed same cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_announcement_admission import _config
from scripts.build_announcement_return_diagnostic import _rows
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_expansion import (
    VERSION,
    attach_prior_industry,
    extract_header_industry,
    select_expansion_batch,
)
from sec_data.disclosure_audit import (
    METRIC_TAGS,
    audit_financial_fields,
    load_cached_submissions,
    submission_url,
    verify_file,
)
from sec_data.disclosure_review import audit_submission_clock
from sec_data.guidance_evidence import parse_complete_submission


def prepare_batch(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze selection before opening statement values; retain every original cohort row."""
    c = json.loads(config_path.read_bytes())
    if (
        c["version"] != VERSION
        or c["new_issuer_batch_size"] != 20
        or c["frozen_cohort_events"] != 316
        or c["selection_policy"] != "first_cached_candidate_per_new_issuer_then_earliest_proxy"
        or c["minimum_industry_events"] != 20
        or c["minimum_distinct_issuers_per_industry"] != 2
        or c["industry_grouping"] != "strictly_prior_original_header_SIC_2digit"
        or any(
            c[k] is not False
            for k in (
                "network_allowed",
                "sample_replacement_allowed",
                "model_retuning_allowed",
                "first_public_certified",
                "trading_admission",
                "overwrite_originals",
            )
        )
    ):
        raise ValueError("Unsupported or weakened expansion contract")
    base = _config(c["base_return_config"], c["base_return_config_sha256"])
    cohort = _json(
        Path(base["admission_root"]),
        base["admission_manifest_sha256"],
        "event_admission_ledger.json",
    )
    source_plan = _json(
        Path(base["admission_plan_root"]), base["admission_plan_manifest_sha256"], "plan.json"
    )
    source = source_plan["financial_sources"]
    cache = load_cached_submissions([Path(p) for p in source["cached_submission_roots"]])
    selected = select_expansion_batch(cohort, set(cache))
    controls = [r for r in cohort if r["release_value_pair_verified"] is True]
    if len(cohort) != 316 or len(controls) != 6:
        raise ValueError("Original cohort or six-case controls changed")
    history = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        ),
        columns=["company_id", "filing_accession", "period_start", "period_end", "filing_form"],
    )
    events = []
    for r in selected:
        rows = history.loc[
            history.company_id.eq(r["cik"])
            & history.filing_accession.eq(r["target_accession_number"])
        ]
        if len(rows) != 1:
            raise ValueError("Expansion financial source identity is not unique")
        history_row = rows.iloc[0]
        events.append(
            {
                "validation_event_id": r["validation_event_id"],
                "cik": r["cik"],
                "ticker": r["assigned_ticker"],
                "filing_accession": r["target_accession_number"],
                "accepted_at": r["accepted_at"],
                "period_start": history_row.period_start.date().isoformat(),
                "period_end": history_row.period_end.date().isoformat(),
                "form": history_row.filing_form,
            }
        )
    snapshot_index = [
        {"cik": k[0], "accession_number": k[1], "sha256": v["sha256"], "url": v["requested_url"]}
        for k, v in sorted(cache.items())
    ]
    plan = {
        "config": c,
        "base_config": base,
        "source_plan": source_plan,
        "selected_cases": selected,
        "controls": controls,
        "events": events,
        "cached_snapshot_index": snapshot_index,
        "all_cohort_event_ids": [r["validation_event_id"] for r in cohort],
        "financial_values_read": False,
        "return_values_read": False,
        "selection_uses_return_values": False,
        "selection_uses_original_value_match": False,
        "implementation_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "scripts/build_announcement_expansion.py",
                "sec_data/announcement_expansion.py",
                "sec_data/disclosure_review.py",
                "sec_data/disclosure_audit.py",
                "sec_data/guidance_evidence.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    print(
        f"EXPANSION PLAN new_issuers={len(selected)} controls={len(controls)} output={frozen}",
        flush=True,
    )
    catalog = pd.read_parquet(
        _artifact(
            Path(source["catalog_root"]),
            source["catalog_manifest_sha256"],
            "filing_catalog.parquet",
        ),
        columns=["cik", "accession_number", "form", "report_date", "accepted_at", "items"],
    )
    # Only immutable companyfacts and original submissions are opened after the plan.
    verify_file(Path(source["companyfacts_zip"]), source["companyfacts_sha256"])
    fields: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(source["companyfacts_zip"]) as archive:
        for case, event in zip(selected, events, strict=True):
            accession = case["release_accession_number"]
            candidates.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "cik": case["cik"],
                    "accession_number": accession,
                    "form": "8-K",
                    "accepted_at": case["earliest_disclosure_at"],
                    "url": submission_url(case["cik"], accession),
                }
            )
            if event["form"] == "10-Q":
                body = archive.read(f"CIK{case['cik']}.json")
                fields.extend(
                    {
                        **r,
                        "source_member_sha256": hashlib.sha256(body).hexdigest(),
                        "source_zip_sha256": source["companyfacts_sha256"],
                    }
                    for r in audit_financial_fields(event, json.loads(body), catalog)
                )
            else:
                fields.extend(
                    {
                        "validation_event_id": event["validation_event_id"],
                        "cik": event["cik"],
                        "metric": m,
                        "value": None,
                        "status": "unsupported_direct_quarter_base_form",
                        "quarter_or_instant_ready": False,
                        "period_start": event["period_start"],
                        "period_end": event["period_end"],
                    }
                    for m in METRIC_TAGS
                )
            source_row = cache[(case["cik"], accession)]
            verify_file(Path(source_row["path"]), source_row["sha256"])
            if source_row["requested_url"] != candidates[-1]["url"]:
                raise ValueError("Cached submission URL does not bind to candidate")
            content = Path(source_row["path"]).read_bytes()
            for doc in parse_complete_submission(content):
                if not doc["document_type"].upper().startswith("EX-99"):
                    continue
                text = re.sub(r"\s+", " ", doc["visible_text"])
                filename = f"{case['assigned_ticker']}_{doc['source_occurrence']}_exhibit.txt"
                files[filename] = text.encode("utf-8")
                documents.append(
                    {
                        "validation_event_id": case["validation_event_id"],
                        "cik": case["cik"],
                        "ticker": case["assigned_ticker"],
                        "period_end": case["period_end"],
                        "release_accession": accession,
                        "submission_sha256": source_row["sha256"],
                        "exhibit_filename": doc["filename"],
                        "exhibit_sha256": doc["document_sha256"],
                        "visible_text_file": filename,
                        "clock": audit_submission_clock(content, candidates[-1]),
                    }
                )
    # Historical industry observations remain independent of the financial values.
    industries = []
    relevant = {r["cik"] for r in [*selected, *controls]}
    for (cik, accession), row in sorted(cache.items()):
        if cik not in relevant:
            continue
        verify_file(Path(row["path"]), row["sha256"])
        try:
            item = extract_header_industry(Path(row["path"]).read_bytes(), cik, accession)
        except ValueError as exc:
            rejections.append(
                {
                    "cik": cik,
                    "accession_number": accession,
                    "source_sha256": row["sha256"],
                    "reason": str(exc),
                }
            )
            continue
        industries.append(
            {**item, "source_sha256": row["sha256"], "source_url": row["requested_url"]}
        )
    prior = attach_prior_industry([*controls, *selected], industries)
    report = {
        "version": VERSION,
        "selection_plan_root": str(frozen),
        "selection_plan_manifest_sha256": hashlib.sha256(
            (frozen / "output_manifest.json").read_bytes()
        ).hexdigest(),
        "cohort_events_retained": len(cohort),
        "new_issuers_frozen": len(selected),
        "controls": len(controls),
        "exhibits_prepared": len(documents),
        "prior_industry_matches": int(prior.industry_ready.sum()),
        "prior_industry_events": len(prior),
        "industry_observations": len(industries),
        "return_values_read": False,
        "network_requests": 0,
        "model_retuned": False,
        "first_public_certified": False,
        "trading_admission": False,
    }
    files.update(
        {
            "summary.json": _encode(report),
            "documents.json": _encode(documents),
            "field_audit.json": _encode(fields),
            "selected_candidates.json": _encode(candidates),
            "industry_observations.json": _encode(industries),
            "industry_rejections.json": _encode(rejections),
            "prior_industry.json": _encode(_rows(prior)),
        }
    )
    return write_bundle(output_root, VERSION + "-prepared", files), report


def main() -> None:
    """Prepare an offline evidence batch; financial review and returns are separate stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/announcement_expansion_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_expansion")
    )
    args = parser.parse_args()
    root, report = prepare_batch(args.config, args.output_root)
    print(
        f"EXPANSION PREPARED issuers={report['new_issuers_frozen']} "
        f"prior_industry={report['prior_industry_matches']}/{report['prior_industry_events']} "
        f"output={root}"
    )


if __name__ == "__main__":
    main()
