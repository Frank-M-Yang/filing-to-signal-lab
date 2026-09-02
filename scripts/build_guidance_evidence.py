"""Plan, collect, and audit prior company-guidance evidence from SEC 8-K submissions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.client import SecClient
from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import (
    build_guidance_evidence_audit,
    build_guidance_plan,
    collect_guidance_documents,
)


def main() -> int:
    """Freeze targets, optionally collect SEC bytes, and rebuild evidence coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-state-root", type=Path, default=None)
    parser.add_argument("--filing-catalog-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/sec/guidance_evidence/targets"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/sec/guidance_evidence"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/guidance_evidence")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze targets without writing an empty pre-collection evidence audit.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.collect and args.plan_only:
        parser.error("--collect and --plan-only cannot be used together")
    if args.collect and args.request_interval <= 0.20:
        parser.error("--request-interval must be above 0.20 seconds (below five requests/second)")
    market_root = args.market_state_root or _latest_root(
        Path("data/processed/validation_market_state"), "validation_market_state_report.json"
    )
    catalog_root = args.filing_catalog_root or _latest_root(
        Path("data/processed/sec_universe"), "catalog_report.json"
    )
    plan = build_guidance_plan(
        market_root,
        catalog_root,
        output_root=args.target_output_root,
    )
    print(
        "GUIDANCE TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"events={plan.report['selected_guidance_events']}/"
        f"{plan.report['market_state_ready_events']} "
        f"documents={plan.report['unique_guidance_documents']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.plan_only:
        return 0
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        client = SecClient(
            user_agent,
            request_interval_seconds=args.request_interval,
        )
        targets = load_catalog_table(plan.output_root, "guidance_document_targets.parquet")

        def show_progress(done: int, total: int, document_id: str) -> None:
            if done == 1 or done == total or done % 10 == 0:
                print(
                    f"GUIDANCE COLLECT progress={done}/{total} target={document_id[:12]}",
                    flush=True,
                )

        collection = collect_guidance_documents(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "GUIDANCE COLLECT DONE "
            f"selected={collection.selected_documents} "
            f"completed={collection.completed_documents} "
            f"failed={collection.failed_documents} "
            f"requests={collection.requests_made}",
            flush=True,
        )
    audit = build_guidance_evidence_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "GUIDANCE EVIDENCE DONE "
        f"version={report['dataset_version']} "
        f"collected={report['documents_collected']}/{report['unique_document_targets']} "
        f"exhibit_text={report['documents_with_exhibit_text']} "
        "gaap_numeric_gm_events="
        f"{report['events_with_gaap_compatible_numeric_gross_margin_guidance']}/"
        f"{report['guidance_candidate_events']} "
        f"model_gate={report['model_run_authorized']} "
        f"output={audit.output_root}",
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
