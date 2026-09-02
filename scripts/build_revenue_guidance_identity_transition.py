"""Repair revenue-guidance exchange transitions with exact event-date identity."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_reference import (
    MassiveReferenceClient,
    build_massive_reference_audit,
    collect_massive_reference_artifacts,
    load_massive_reference_targets,
)
from sec_data.revenue_guidance_identity_transition import (
    build_revenue_guidance_identity_resolution,
    build_revenue_guidance_transition_plan,
)


def main() -> int:
    """Freeze the exact query, optionally collect it, and resolve all events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument("--base-identity-root", type=Path, default=None)
    parser.add_argument("--base-reference-root", type=Path, default=None)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/market/revenue_guidance_identity_transition/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/revenue_guidance_identity_transition"),
    )
    parser.add_argument(
        "--reference-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_identity_transition_reference"),
    )
    parser.add_argument(
        "--resolution-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_identity_resolution"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic plan")
    market = args.market_plan_root or _latest_root(
        Path("data/processed/revenue_guidance_market_plan"),
        "validation_market_plan_report.json",
    )
    identity = args.base_identity_root or _latest_root(
        Path("data/processed/revenue_guidance_security_identity"),
        "validation_security_identity_report.json",
    )
    base_reference = args.base_reference_root or _latest_root(
        Path("data/processed/revenue_guidance_security_reference"),
        "massive_reference_report.json",
    )
    plan = build_revenue_guidance_transition_plan(
        market, identity, base_reference, output_root=args.target_root
    )
    print(
        "GUIDANCE TRANSITION TARGETS DONE "
        f"events={plan.report['transition_events']} "
        f"existing_exact={plan.report['existing_exact_endpoint_references']} "
        f"new_queries={plan.report['new_exact_event_queries']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if not args.collect:
        print("GUIDANCE TRANSITION PENDING set MASSIVE_API_KEY and rerun with --collect")
        return 0
    api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not api_key:
        parser.error("MASSIVE_API_KEY is required with --collect")
    targets = load_massive_reference_targets(plan.output_root)
    client = MassiveReferenceClient(api_key, request_interval_seconds=args.request_interval)
    collection = collect_massive_reference_artifacts(
        targets, client, raw_root=args.raw_root, limit=args.limit
    )
    print(
        "GUIDANCE TRANSITION COLLECT DONE "
        f"selected={collection.selected_targets} "
        f"completed={collection.targets_completed} "
        f"failed={collection.targets_failed} "
        f"requests={collection.requests_made} "
        f"blocked={collection.blocking_reason}",
        flush=True,
    )
    reference = build_massive_reference_audit(
        plan.output_root, raw_root=args.raw_root, output_root=args.reference_output_root
    )
    if not bool(reference.report["event_date_reference_gate"]):
        print(
            "GUIDANCE TRANSITION RESOLUTION BLOCKED "
            f"reference_ready={reference.report['event_date_reference_candidates']}/"
            f"{reference.report['price_mapping_targets']}",
            flush=True,
        )
        return 0
    resolution = build_revenue_guidance_identity_resolution(
        market,
        identity,
        base_reference,
        reference.output_root,
        output_root=args.resolution_output_root,
    )
    print(
        "GUIDANCE IDENTITY RESOLUTION DONE "
        f"ready={resolution.report['price_candidate_events_identity_ready']}/"
        f"{resolution.report['price_candidate_events']} "
        f"transitions={resolution.report['transition_events_resolved']} "
        f"price_gate={resolution.report['price_collection_authorized']} "
        f"output={resolution.output_root}",
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
