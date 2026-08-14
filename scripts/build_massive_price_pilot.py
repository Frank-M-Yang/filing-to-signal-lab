"""Build and optionally collect the Massive unadjusted daily-price pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_prices import (
    build_massive_price_audit,
    build_massive_price_target_plan,
    collect_massive_price_artifacts,
    load_massive_price_targets,
)
from sec_data.massive_reference import MassiveReferenceClient


def main() -> int:
    """Freeze price targets, optionally collect them, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Massive reference audit; defaults to latest audit with candidates",
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/massive_prices/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/massive_prices"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/massive_prices"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.1,
        help="seconds between calls; 12.1 respects the documented Basic 5 calls/minute limit",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="make authenticated API calls; otherwise build only from preserved bytes",
    )
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic pilot")
    reference_root = args.reference_root or _latest_reference_audit(
        Path("data/processed/massive_reference")
    )
    plan = build_massive_price_target_plan(
        reference_root,
        output_root=args.target_output_root,
    )
    targets = load_massive_price_targets(plan.output_root)
    print(
        "PRICE TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['price_targets']:,} "
        f"adjusted={plan.report['query_adjusted']} "
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
                f"PRICE COLLECT progress={done}/{total} target={price_target_id[:12]}", flush=True
            )

        collection = collect_massive_price_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "PRICE COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
        if collection.blocking_reason == "authentication_failed":
            print("AUTH ACTION: Massive rejected MASSIVE_API_KEY (HTTP 401).", flush=True)
        elif collection.blocking_reason == "entitlement_denied":
            print(
                "ENTITLEMENT ACTION: authentication succeeded, but the plan does not permit "
                "this historical aggregate query (HTTP 403).",
                flush=True,
            )
    audit = build_massive_price_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "PRICE AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']:,}/{report['price_targets']:,} "
        f"bars={report['parsed_daily_bars']:,} "
        f"window_candidates={report['observed_event_window_candidates']:,} "
        f"return_gate={report['return_study_gate']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


def _latest_reference_audit(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_reference_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("event_date_reference_candidates", 0)) > 0:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(
            "No Massive reference audit has event-date candidates; run "
            "scripts.build_massive_reference_pilot first"
        )
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
