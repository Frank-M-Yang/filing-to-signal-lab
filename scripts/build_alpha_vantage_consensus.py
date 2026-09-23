"""Build and optionally capture the forward-only Alpha Vantage consensus ledger."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from sec_data.alpha_vantage_budget import AlphaVantageRequestBudget
from sec_data.alpha_vantage_consensus import (
    AlphaVantageConsensusClient,
    build_alpha_vantage_live_audit,
    build_alpha_vantage_live_plan,
    collect_alpha_vantage_live_snapshots,
)


def main() -> int:
    """Freeze the cohort, optionally capture today, and rebuild the live audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-asset-root",
        type=Path,
        default=Path("data/processed/global500_2026/core_asset_v1"),
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/alpha_vantage_consensus/targets"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/market/alpha_vantage_consensus")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/alpha_vantage_consensus")
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=12.1)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--external-requests-today", type=int, default=0)
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds")
    plan = build_alpha_vantage_live_plan(args.core_asset_root, output_root=args.target_output_root)
    print(
        "ALPHA VANTAGE LIVE TARGETS DONE "
        f"version={plan.report['dataset_version']} targets={plan.report['live_targets']} "
        f"daily_cap={plan.report['daily_request_cap']} output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
        if not api_key:
            parser.error("ALPHA_VANTAGE_API_KEY is required with --collect")
        client = AlphaVantageConsensusClient(
            api_key,
            request_interval_seconds=args.request_interval,
            request_budget=AlphaVantageRequestBudget(
                state_root=args.raw_root.parent / "alpha_vantage_request_budget",
                legacy_raw_root=args.raw_root,
                external_requests_today=args.external_requests_today,
            ),
        )

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(f"LIVE CONSENSUS progress={done}/{total} target={target_id[:12]}", flush=True)

        collection = collect_alpha_vantage_live_snapshots(
            pd.read_parquet(plan.output_root / "alpha_vantage_live_targets.parquet"),
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "ALPHA VANTAGE LIVE COLLECT DONE "
            f"selected={collection.selected_targets} completed={collection.completed_targets} "
            f"failed={collection.failed_targets} requests={collection.requests_made} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_alpha_vantage_live_audit(
        plan.output_root, raw_root=args.raw_root, output_root=args.output_root
    )
    report = audit.report
    print(
        "ALPHA VANTAGE LIVE AUDIT DONE "
        f"version={report['dataset_version']} "
        f"captured={report['captured_tickers']}/{report['live_targets']} "
        f"quarterly_estimates={report['quarterly_revenue_consensus_observations']} "
        f"forward={report['strict_forward_observations']} "
        f"forward_tickers={report['strict_forward_tickers']}/{report['live_targets']} "
        f"gate={report['strict_forward_snapshot_gate']} output={audit.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
