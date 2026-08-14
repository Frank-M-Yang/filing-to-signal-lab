"""Collect a 100-company SEC-available cohort from the Fortune Global 500."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.client import TICKER_MAP_URL, SecClient, decode_json_object
from sec_data.cohort import (
    FortuneClient,
    assess_sec_research_completeness,
    download_fortune_global500,
    map_fortune_members_to_sec,
    preserve_source_bytes,
)
from sec_data.pipeline import SecIngestionService


def main() -> int:
    """Run the resumable ranking, mapping, ingestion, and completeness workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT", ""),
        help="Identified SEC User-Agent; preferably set SEC_USER_AGENT in the process environment.",
    )
    parser.add_argument(
        "--accept-manual-matches",
        action="store_true",
        help="Also ingest high-scoring mappings labelled manual_review.",
    )
    args = parser.parse_args()
    if not args.sec_user_agent:
        parser.error("Set SEC_USER_AGENT or pass --sec-user-agent with a real contact email")

    scope = f"global500_{args.year}"
    raw_root = args.data_root / "raw" / scope
    interim_root = args.data_root / "interim" / scope
    processed_root = args.data_root / "processed" / scope
    assessment_root = interim_root / "assessments"
    for path in (raw_root, interim_root, processed_root, assessment_root):
        path.mkdir(parents=True, exist_ok=True)

    fortune_client = FortuneClient(args.sec_user_agent)
    ranking = download_fortune_global500(
        args.year,
        raw_root=raw_root / "fortune",
        interim_root=interim_root / "fortune",
        client=fortune_client,
        progress=_progress,
    )
    ranking.to_csv(processed_root / "fortune_global500_ranking.csv", index=False)

    sec_client = SecClient(args.sec_user_agent)
    ticker_response = sec_client.get(TICKER_MAP_URL)
    ticker_digest, ticker_path = preserve_source_bytes(
        raw_root / "sec_reference",
        kind="company-tickers",
        content=ticker_response.content,
        suffix="json",
    )
    ticker_document = decode_json_object(ticker_response.content, source_url=ticker_response.url)
    mapping = map_fortune_members_to_sec(ranking, ticker_document)
    mapping.to_csv(processed_root / "fortune_to_sec_mapping.csv", index=False)
    _progress(
        "SEC mapping: "
        f"{int((mapping['mapping_status'] == 'accepted').sum())} accepted, "
        f"{int((mapping['mapping_status'] == 'manual_review').sum())} manual review"
    )

    allowed_statuses = {"accepted"}
    if args.accept_manual_matches:
        allowed_statuses.add("manual_review")
    service = SecIngestionService(
        args.sec_user_agent,
        data_root=args.data_root / "sec",
        client=sec_client,
    )
    audit_rows: list[dict[str, Any]] = []
    seen_ciks: set[str] = set()
    for _, candidate in mapping.sort_values("fortune_rank").iterrows():
        candidate_row = candidate.to_dict()
        status = str(candidate["mapping_status"])
        cik_value = candidate.get("sec_cik")
        cik = str(cik_value) if pd.notna(cik_value) else ""
        if status not in allowed_statuses or not cik:
            candidate_row.update(
                {
                    "cohort_status": "not_sec_mapped",
                    "selection_order": None,
                    "selection_reason": f"mapping_status={status}",
                }
            )
            audit_rows.append(candidate_row)
            continue
        if cik in seen_ciks:
            candidate_row.update(
                {
                    "cohort_status": "duplicate_sec_cik",
                    "selection_order": None,
                    "selection_reason": "A higher-ranked Fortune member maps to the same SEC CIK",
                }
            )
            audit_rows.append(candidate_row)
            continue
        seen_ciks.add(cik)
        assessment_path = assessment_root / f"{cik}.json"
        if assessment_path.exists():
            assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
            _progress(
                f"rank {int(candidate['fortune_rank']):03d} {candidate['company_name']}: cached"
            )
        else:
            try:
                result = service.ingest(cik, include_archives=True)
                assessment = assess_sec_research_completeness(
                    result,
                    start_year=args.start_year,
                    end_year=args.end_year,
                )
                assessment_path.write_text(
                    json.dumps(assessment, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
                _progress(
                    f"rank {int(candidate['fortune_rank']):03d} {candidate['company_name']}: "
                    f"10-Q={assessment['original_10q_filings']}, "
                    f"GM={assessment['gross_margin_ready_events']}"
                )
            except Exception as exc:
                assessment = {
                    "gross_margin_research_ready": False,
                    "download_error": f"{type(exc).__name__}: {exc}",
                }
                assessment_path.write_text(
                    json.dumps(assessment, indent=2, sort_keys=True), encoding="utf-8"
                )
                _progress(
                    f"rank {int(candidate['fortune_rank']):03d} {candidate['company_name']}: "
                    f"ERROR {assessment['download_error']}"
                )
        candidate_row.update(assessment)
        if bool(assessment.get("sec_raw_complete")):
            candidate_row.update(
                {
                    "cohort_status": "quarterly_data_complete",
                    "selection_order": None,
                    "selection_reason": "Passed original 10-Q and quarterly revenue thresholds",
                }
            )
        else:
            candidate_row.update(
                {
                    "cohort_status": "incomplete_quarterly_data",
                    "selection_order": None,
                    "selection_reason": (
                        "Did not pass original 10-Q and quarterly revenue thresholds"
                    ),
                }
            )
        audit_rows.append(candidate_row)

    audit = pd.DataFrame(audit_rows).sort_values("fortune_rank").reset_index(drop=True)
    eligible = audit.loc[audit["cohort_status"] == "quarterly_data_complete"].copy()
    selected_indexes = eligible.head(args.target).index
    audit.loc[selected_indexes, "cohort_status"] = "selected"
    audit.loc[selected_indexes, "selection_order"] = range(1, len(selected_indexes) + 1)
    selected = audit.loc[audit["cohort_status"] == "selected"].copy()
    gross_margin_ready = audit.loc[
        audit["gross_margin_research_ready"].fillna(False).astype(bool)
    ].copy()
    audit.to_csv(processed_root / "candidate_audit.csv", index=False)
    selected.to_csv(processed_root / "cohort_100.csv", index=False)
    gross_margin_ready.to_csv(processed_root / "gross_margin_ready_cohort.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ranking_year": args.year,
        "research_period": [args.start_year, args.end_year],
        "target_companies": args.target,
        "selected_companies": len(selected),
        "target_met": len(selected) == args.target,
        "candidates_audited": len(audit),
        "highest_selected_rank": (
            int(selected["fortune_rank"].max()) if not selected.empty else None
        ),
        "selected_from_original_top_100": int((selected["fortune_rank"] <= 100).sum()),
        "selected_replacements_from_101_500": int((selected["fortune_rank"] > 100).sum()),
        "gross_margin_research_ready_companies": len(gross_margin_ready),
        "gross_margin_target_met": len(gross_margin_ready) >= args.target,
        "sec_ticker_map_source_url": ticker_response.url,
        "sec_ticker_map_snapshot_sha256": ticker_digest,
        "sec_ticker_map_raw_path": str(ticker_path),
        "ranking_source_url": "https://fortune.com/ranking/global500/",
        "selection_definition": {
            "primary_data_cohort": (
                "At least 20 original 10-Q filings and 20 current-quarter revenue events"
            ),
            "gross_margin_research_subset": (
                "Primary data complete plus at least 16 gross-margin events across 6 years"
            ),
            "original_10q_filings_minimum": 20,
            "quarterly_revenue_events_minimum": 20,
            "gross_margin_ready_events_minimum": 16,
            "gross_margin_ready_years_minimum": 6,
        },
    }
    (processed_root / "coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _progress(
        f"DONE quarterly-data={len(selected)}/{args.target}; "
        f"gross-margin-ready={len(gross_margin_ready)}; "
        f"audit={processed_root / 'candidate_audit.csv'}"
    )
    return 0 if len(selected) == args.target else 2


def _progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
