"""Build, collect, and audit the revenue-guidance issuer-year expansion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.client import SecClient
from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import collect_guidance_documents
from sec_data.guidance_metric_coverage import build_guidance_metric_coverage
from sec_data.revenue_guidance_expansion import build_revenue_guidance_expansion


def main() -> int:
    """Freeze expansion targets, optionally collect, and rebuild metric coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument("--filing-catalog-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/sec/revenue_guidance_expansion/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/guidance_evidence"),
    )
    parser.add_argument(
        "--coverage-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_expansion_coverage"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.collect and args.request_interval <= 0.20:
        parser.error("--request-interval must be above 0.20 seconds")
    feature_root = args.filing_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    catalog_root = args.filing_catalog_root or _latest_root(
        Path("data/processed/sec_universe"),
        "catalog_report.json",
    )
    plan = build_revenue_guidance_expansion(
        feature_root,
        catalog_root,
        output_root=args.target_output_root,
    )
    print(
        "REVENUE GUIDANCE TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"events={plan.report['sampled_issuer_year_events']} "
        f"issuers={plan.report['sampled_issuers']} "
        f"documents={plan.report['unique_guidance_documents']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        targets = load_catalog_table(plan.output_root, "guidance_document_targets.parquet")
        client = SecClient(
            user_agent,
            request_interval_seconds=args.request_interval,
        )

        def show_progress(done: int, total: int, document_id: str) -> None:
            if done == 1 or done == total or done % 25 == 0:
                print(
                    f"REVENUE GUIDANCE COLLECT progress={done}/{total} target={document_id[:12]}",
                    flush=True,
                )

        collected = collect_guidance_documents(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "REVENUE GUIDANCE COLLECT DONE "
            f"selected={collected.selected_documents} "
            f"completed={collected.completed_documents} "
            f"failed={collected.failed_documents} "
            f"requests={collected.requests_made}",
            flush=True,
        )
    coverage = build_guidance_metric_coverage(
        plan.output_root,
        feature_root,
        raw_root=args.raw_root,
        output_root=args.coverage_output_root,
    )
    revenue = next(row for row in coverage.report["metric_results"] if row["metric"] == "revenue")
    print(
        "REVENUE GUIDANCE COVERAGE DONE "
        f"exact={revenue['exact_quarter_gaap_events']}/"
        f"{revenue['eligible_events']} "
        f"issuers={revenue['exact_quarter_gaap_issuers']} "
        f"gate={revenue['evidence_coverage_gate']} "
        f"output={coverage.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
