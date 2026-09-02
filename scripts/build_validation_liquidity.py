"""Freeze, collect, and audit strictly pre-event liquidity evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_prices import collect_massive_price_artifacts, load_massive_price_targets
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.validation_liquidity import (
    build_validation_liquidity_audit,
    build_validation_liquidity_plan,
)


def main() -> int:
    """Run one resumable, outcome-blind pre-event liquidity batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--fallback-audit-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_liquidity/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_liquidity/raw_store"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_liquidity"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.1,
        help="seconds between calls; Massive Basic requires at least 12 seconds",
    )
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for Massive Basic")
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    plan = build_validation_liquidity_plan(
        execution,
        output_root=args.target_output_root,
    )
    targets = load_massive_price_targets(plan.output_root)
    print(
        "VALIDATION LIQUIDITY TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"events={plan.report['event_components']} "
        f"targets={plan.report['price_targets']} "
        f"lookback={plan.report['config']['lookback_sessions']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

        def show_progress(done: int, total: int, price_target_id: str) -> None:
            print(
                f"VALIDATION LIQUIDITY progress={done}/{total} target={price_target_id[:12]}",
                flush=True,
            )

        collection = collect_massive_price_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "VALIDATION LIQUIDITY COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_validation_liquidity_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
        fallback_audit_root=args.fallback_audit_root,
    )
    report = audit.report
    print(
        "VALIDATION LIQUIDITY AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']}/{report['price_targets']} "
        f"eligible={report['liquidity_eligible_events']}/{report['event_components']} "
        f"unresolved={report['liquidity_unresolved_events']} "
        f"gate={report['liquidity_gate']} "
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
