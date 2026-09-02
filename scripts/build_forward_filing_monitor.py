"""Build and optionally collect the forward SEC filing monitor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.client import SecClient
from sec_data.forward_filing_monitor import (
    build_forward_filing_monitor_audit,
    build_forward_filing_monitor_plan,
    collect_forward_filing_snapshots,
)


def main() -> int:
    """Freeze targets, optionally snapshot SEC submissions, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-consensus-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/sec/forward_filing_monitor/targets"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/sec/forward_filing_monitor"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/forward_filing_monitor")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.collect and args.request_interval <= 0.20:
        parser.error("--request-interval must remain below five SEC requests per second")
    consensus_root = args.forward_consensus_root or _latest_root(
        Path("data/processed/forward_consensus_panel"),
        "forward_consensus_panel_report.json",
    )
    plan = build_forward_filing_monitor_plan(
        consensus_root,
        output_root=args.target_output_root,
    )
    print(
        "FORWARD FILING MONITOR TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['monitor_targets']} "
        f"issuers={plan.report['monitor_issuers']} output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        client = SecClient(user_agent, request_interval_seconds=args.request_interval)

        def show_progress(done: int, total: int, target_id: str) -> None:
            if done == 1 or done == total or done % 5 == 0:
                print(
                    f"FORWARD FILING MONITOR progress={done}/{total} target={target_id[:12]}",
                    flush=True,
                )

        collection = collect_forward_filing_snapshots(
            pd.read_parquet(plan.output_root / "forward_filing_monitor_targets.parquet"),
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "FORWARD FILING MONITOR COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.completed_targets} "
            f"failed={collection.failed_targets} requests={collection.requests_made}",
            flush=True,
        )
    audit = build_forward_filing_monitor_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "FORWARD FILING MONITOR AUDIT DONE "
        f"version={report['dataset_version']} "
        f"captured={report['captured_targets']}/{report['monitor_targets']} "
        f"candidates={report['future_filing_candidates']} "
        f"baseline_gate={report['monitor_baseline_gate_passed']} "
        f"actual_gate={report['actual_revenue_join_gate_passed']} "
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
