"""Probe at most four unknown issuers; keep every result outside the frozen cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.build_prospective_cohort_feasibility import _latest_report_root
from sec_data.alpha_vantage_budget import AlphaVantageRequestBudget
from sec_data.alpha_vantage_consensus import AlphaVantageConsensusClient
from sec_data.alpha_vantage_coverage_probe import (
    build_alpha_vantage_coverage_probe_audit,
    build_alpha_vantage_coverage_probe_plan,
    collect_alpha_vantage_coverage_probe,
)
from sec_data.filing_catalog import load_catalog_table


def main() -> None:
    """Build an isolated plan/audit, optionally collecting one fixed, protected batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility-root", type=Path)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/market/alpha_vantage_coverage_probe")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/alpha_vantage_coverage_probe")
    )
    parser.add_argument("--original-plan-root", type=Path)
    parser.add_argument(
        "--original-raw-root", type=Path, default=Path("data/market/alpha_vantage_consensus")
    )
    parser.add_argument(
        "--budget-root", type=Path, default=Path("data/market/alpha_vantage_request_budget")
    )
    parser.add_argument("--external-requests-today", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 4 or args.offset < 0 or not 0 <= args.external_requests_today <= 25:
        parser.error("limit must be 1-4, offset nonnegative, external requests 0-25")
    feasibility = args.feasibility_root or _latest_report_root(
        Path("data/processed/prospective_cohort_feasibility"),
        "prospective_cohort_feasibility_report.json",
    )
    original_plan = args.original_plan_root or _latest_report_root(
        args.original_raw_root / "targets", "alpha_vantage_live_plan_report.json"
    )
    original_targets = load_catalog_table(original_plan, "alpha_vantage_live_targets.parquet")
    if len(original_targets) != 20 or original_targets["cik"].nunique() != 20:
        parser.error("the protected original plan must contain exactly 20 issuers")
    plan = build_alpha_vantage_coverage_probe_plan(
        feasibility, output_root=args.raw_root / "targets"
    )
    targets = load_catalog_table(plan.output_root, "coverage_probe_targets.parquet")
    if set(targets["cik"]) & set(original_targets["cik"]):
        parser.error("unknown probe issuers overlap the protected original plan")
    budget = AlphaVantageRequestBudget(
        state_root=args.budget_root,
        legacy_raw_root=args.original_raw_root,
        protected_target_ids=tuple(original_targets["live_consensus_target_id"].astype(str)),
        retry_reserve=1,
        single_attempt_per_ticker_day=True,
        external_requests_today=args.external_requests_today,
    )
    print("PROBE LOCAL BUDGET " + json.dumps(budget.summary(), sort_keys=True), flush=True)
    if args.collect:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
        if not api_key:
            parser.error(
                "ALPHA_VANTAGE_API_KEY is required with --collect; never paste it in output"
            )
        client = AlphaVantageConsensusClient(api_key, request_budget=budget)

        def progress(done: int, total: int, target_id: str) -> None:
            print(f"COVERAGE PROBE progress={done}/{total} target={target_id[:12]}", flush=True)

        result = collect_alpha_vantage_coverage_probe(
            targets,
            client,
            raw_root=args.raw_root,
            offset=args.offset,
            limit=args.limit,
            progress=progress,
        )
        print("COVERAGE PROBE COLLECT " + json.dumps(result, sort_keys=True), flush=True)
    audit = build_alpha_vantage_coverage_probe_audit(
        plan.output_root, raw_root=args.raw_root, output_root=args.output_root
    )
    report = audit.report
    print(
        f"COVERAGE PROBE AUDIT DONE version={report['dataset_version']} "
        f"attempted={report['attempted_issuers']}/{report['probe_targets']} "
        f"candidates={report['candidate_issuers']} repeated={report['repeated_candidate_issuers']} "
        f"pairs={report['adjacent_snapshot_pairs']} "
        f"spans_24h={report['snapshot_pairs_spanning_24_hours']} "
        f"formal_admission=False output={audit.output_root}"
    )


if __name__ == "__main__":
    main()
