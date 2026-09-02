"""Collect and audit exact SEC filing-index evidence for frozen validation events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.client import SecClient
from sec_data.filing_cover import collect_filing_cover_artifacts
from sec_data.validation_filing_evidence import (
    build_validation_filing_evidence_audit,
    build_validation_filing_evidence_plan,
    load_validation_filing_evidence_targets,
)


def main() -> int:
    """Create targets, optionally collect raw index pages, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/sec/validation_filing_evidence/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/validation_filing_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_filing_evidence"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.request_interval < 0.20:
        parser.error("--request-interval must remain at or above 0.20 seconds")
    market_plan_root = args.market_plan_root or _latest_root(
        Path("data/processed/validation_market_plan"),
        "validation_market_plan_report.json",
    )
    plan = build_validation_filing_evidence_plan(market_plan_root, output_root=args.target_root)
    targets = load_validation_filing_evidence_targets(plan.output_root)
    print(
        "VALIDATION FILING TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['target_events']:,} "
        f"issuers={plan.report['target_issuers']:,} "
        f"outcomes_loaded={plan.report['outcome_values_loaded']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error(
                "SEC_USER_AGENT must identify the project and include a real contact email"
            )
        client = SecClient(
            user_agent,
            request_interval_seconds=args.request_interval,
            timeout_seconds=60.0,
        )

        def show_progress(done: int, total: int, target_id: str) -> None:
            if done == 1 or done % 5 == 0 or done == total:
                print(
                    f"VALIDATION FILING COLLECT progress={done}/{total} target={target_id[:12]}",
                    flush=True,
                )

        collection = collect_filing_cover_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            artifact_policy="index-only",
            progress=show_progress,
        )
        print(
            "VALIDATION FILING COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,}",
            flush=True,
        )
    audit = build_validation_filing_evidence_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION FILING AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['index_artifacts_preserved']:,}/{report['target_events']:,} "
        f"industry_ready={report['filing_time_industry_ready']:,} "
        f"rejected_or_missing={report['rejected_or_missing']:,} "
        f"return_gate={report['return_outcome_read_authorized']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report = _read_json(report_path)
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


if __name__ == "__main__":
    sys.exit(main())
