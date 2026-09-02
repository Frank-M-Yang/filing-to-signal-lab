"""Build and optionally collect strict forward SEC revenue actuals."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.client import SecClient
from sec_data.forward_actual_revenue import (
    build_forward_actual_audit,
    build_forward_actual_plan,
    collect_forward_actual_snapshots,
)


def main() -> int:
    """Freeze actual targets, optionally collect companyfacts, and audit results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filing-monitor-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/sec/forward_actual_revenue/targets"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/sec/forward_actual_revenue"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/forward_actual_revenue")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.collect and args.request_interval <= 0.20:
        parser.error("--request-interval must remain below five SEC requests per second")
    monitor_root = args.filing_monitor_root or _latest_root(
        Path("data/processed/forward_filing_monitor"),
        "forward_filing_monitor_audit_report.json",
    )
    plan = build_forward_actual_plan(monitor_root, output_root=args.target_output_root)
    print(
        "FORWARD ACTUAL TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['original_actual_targets']} "
        f"amendments_excluded={plan.report['amendment_candidates_excluded']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        client = SecClient(user_agent, request_interval_seconds=args.request_interval)

        def show_progress(done: int, total: int, target_id: str) -> None:
            if done == 1 or done == total or done % 5 == 0:
                print(f"FORWARD ACTUAL progress={done}/{total} target={target_id[:12]}", flush=True)

        collection = collect_forward_actual_snapshots(
            pd.read_parquet(plan.output_root / "forward_actual_revenue_targets.parquet"),
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "FORWARD ACTUAL COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.completed_targets} "
            f"failed={collection.failed_targets} requests={collection.requests_made}",
            flush=True,
        )
    audit = build_forward_actual_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "FORWARD ACTUAL AUDIT DONE "
        f"version={report['dataset_version']} "
        f"captured={report['captured_targets']}/{report['actual_targets']} "
        f"ready={report['total_quarterly_actuals_ready']} "
        f"gate={report['actual_revenue_join_gate_passed']} "
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
