"""Build, optionally collect, and audit the formal SPY benchmark evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_benchmark import (
    build_massive_benchmark_audit,
    collect_massive_benchmark_artifacts,
)
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.validation_benchmark import build_validation_benchmark_plan


def main() -> int:
    """Run the resumable formal SPY benchmark evidence phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_benchmark/targets"),
    )
    parser.add_argument(
        "--calendar-output-root",
        type=Path,
        default=Path("data/processed/validation_calendar_adapter"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_benchmark"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_benchmark"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for Massive Basic")
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    plan = build_validation_benchmark_plan(
        execution,
        output_root=args.target_output_root,
        calendar_output_root=args.calendar_output_root,
    )
    print(
        "VALIDATION BENCHMARK TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"dates={plan.report['required_event_dates']} "
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

        def show_progress(done: int, total: int, query_kind: str) -> None:
            print(
                f"VALIDATION BENCHMARK progress={done}/{total} query={query_kind}",
                flush=True,
            )

        collection = collect_massive_benchmark_artifacts(
            plan.output_root,
            client,
            raw_root=args.raw_root,
            progress=show_progress,
        )
        print(
            "VALIDATION BENCHMARK COLLECT DONE "
            f"selected={collection.queries_selected} "
            f"completed={collection.queries_completed} "
            f"failed={collection.queries_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_massive_benchmark_audit(
        plan.output_root,
        plan.calendar_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION BENCHMARK AUDIT DONE "
        f"version={report['dataset_version']} "
        f"queries={report['queries_complete']}/{report['queries_required']} "
        f"dates={report['required_event_dates_ready']}/"
        f"{report['required_event_dates']} "
        f"ready={report['benchmark_data_ready']} "
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
