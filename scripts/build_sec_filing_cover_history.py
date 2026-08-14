"""Collect and build filing-time identity, security, and industry evidence."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_data.client import SecClient
from sec_data.filing_cover import (
    build_filing_cover_history,
    build_filing_cover_target_plan,
    collect_filing_cover_artifacts,
    load_filing_cover_targets,
)


def main() -> int:
    """Run a resumable balanced-year acquisition batch and rebuild processed assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/sec/filing_covers/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/filing_covers"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/sec_filing_cover_history"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="new issuer-year targets to collect in this resumable run",
    )
    parser.add_argument(
        "--artifact-policy",
        choices=["index-only", "index-and-cover"],
        default="index-only",
        help="collect small identity/SIC index pages, or also collect large cover documents",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.25,
        help="seconds between SEC requests; must be at least 0.20",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="do not make network requests; rebuild from preserved artifacts",
    )
    args = parser.parse_args()
    if args.request_interval < 0.20:
        parser.error("--request-interval must remain at or above 0.20 seconds")
    plan = build_filing_cover_target_plan(args.catalog_root, output_root=args.target_root)
    targets = load_filing_cover_targets(plan.output_root)
    if not args.build_only:
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
            if done == 1 or done % 25 == 0 or done == total:
                print(f"COLLECT progress={done}/{total} target={target_id[:12]}", flush=True)

        collection = collect_filing_cover_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            artifact_policy=args.artifact_policy,
            progress=show_progress,
        )
        print(
            "COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,} "
            f"policy={args.artifact_policy}",
            flush=True,
        )
    result = build_filing_cover_history(
        args.catalog_root,
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "BUILD DONE "
        f"version={report['dataset_version']} "
        f"acquired={report['acquired_issuer_years']:,}/{report['target_issuer_years']:,} "
        f"identity_ready={report['filing_identity_ready_acquired']:,} "
        f"industry_ready={report['historical_industry_ready_acquired']:,} "
        f"sample_frozen={report['representative_sample_frozen']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
