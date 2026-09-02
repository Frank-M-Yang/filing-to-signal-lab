"""Build, collect, and audit formal validation listing outcomes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_delisting_outcomes import (
    collect_massive_delisting_artifacts,
    load_massive_delisting_targets,
)
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.validation_delisting import (
    build_validation_delisting_audit,
    build_validation_delisting_plan,
)


def main() -> int:
    """Run one resumable formal listing-status batch and event audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--reconciliation-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_delisting/targets"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/market/validation_delisting"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/validation_delisting")
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for Massive Basic")
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    reconciliation = args.reconciliation_root or _latest_root(
        Path("data/processed/validation_price_reconciliation"),
        "validation_price_reconciliation_report.json",
    )
    plan = build_validation_delisting_plan(execution, output_root=args.target_output_root)
    targets = load_massive_delisting_targets(plan.output_root)
    print(
        "VALIDATION DELISTING TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['targets']} "
        f"cutoff={plan.report['outcome_cutoff_date']} "
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

        def show_progress(done: int, total: int, outcome_target_id: str) -> None:
            print(
                f"VALIDATION DELISTING progress={done}/{total} target={outcome_target_id[:12]}",
                flush=True,
            )

        collection = collect_massive_delisting_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "VALIDATION DELISTING COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_validation_delisting_audit(
        plan.output_root,
        execution,
        reconciliation,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION DELISTING AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']}/{report['targets']} "
        f"security_ready={report['security_listing_outcomes_ready']}/"
        f"{report['targets']} "
        f"events_ready={report['event_delisting_windows_ready']}/"
        f"{report['event_components']} "
        f"market_ready={report['market_outcome_ready_events']}/"
        f"{report['event_components']} "
        f"repair_queue={report['securities_in_repair_queue']} "
        f"terminal_required={report['terminal_value_required_events']} "
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
