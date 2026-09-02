"""Collect and audit planned point-in-time shadow market evidence."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from sec_data.massive_reference import MassiveReferenceClient
from shadow_market_collection import (
    build_shadow_market_evidence_audit,
    collect_shadow_market_evidence_queries,
)


def main() -> int:
    """Optionally collect eligible queries and always rebuild the parsed audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=12.1)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/shadow_market_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/shadow_market_evidence_audit"),
    )
    parser.add_argument(
        "--raw-calendar-root",
        type=Path,
        default=Path("data/market/exchange_calendar"),
    )
    args = parser.parse_args()
    queries = pd.read_parquet(args.market_plan_root / "shadow_market_evidence_queries.parquet")
    if args.collect:
        key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not queries.empty and not key:
            parser.error("MASSIVE_API_KEY is required when eligible market queries exist")
        client = (
            MassiveReferenceClient(key, request_interval_seconds=args.request_interval)
            if key
            else None
        )
        collection = collect_shadow_market_evidence_queries(
            queries,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
        )
        print(
            "SHADOW MARKET COLLECT DONE "
            f"selected={collection.selected_queries} "
            f"completed={collection.completed_queries} "
            f"failed={collection.failed_queries} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}"
        )
    audit = build_shadow_market_evidence_audit(
        args.market_plan_root,
        raw_root=args.raw_root,
        raw_calendar_root=args.raw_calendar_root,
        output_root=args.output_root,
    )
    print(
        "SHADOW MARKET AUDIT DONE "
        f"queries={audit.report['queries_with_captures']}/{audit.report['planned_queries']} "
        f"snapshots={audit.report['source_snapshots']} "
        f"identity={audit.report['point_in_time_identity_verified_events']} "
        f"prices={audit.report['price_observations']} "
        f"order_evidence={audit.report['order_evidence_ready_events']} "
        f"output={audit.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
