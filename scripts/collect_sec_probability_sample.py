"""Collect full filing covers for the frozen probability sample and audit scope."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sec_data.client import SecClient
from sec_data.filing_catalog import load_catalog_table
from sec_data.filing_cover import build_filing_cover_history, collect_filing_cover_artifacts
from sec_data.probability_sample import build_equity_scope_audit


def main() -> int:
    """Resume selected cover acquisition, rebuild evidence, and adjudicate scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--filing-target-plan-root",
        type=Path,
        default=Path("data/sec/filing_covers/targets/filing-cover-targets-v1-9d9a9ea7c9e40430"),
    )
    parser.add_argument(
        "--sample-plan-root",
        type=Path,
        default=Path(
            "data/processed/sec_probability_sample/sec-probability-sample-plan-v1-bd8b453b0e9e3a28"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/filing_covers"),
    )
    parser.add_argument(
        "--cover-output-root",
        type=Path,
        default=Path("data/processed/sec_filing_cover_history"),
    )
    parser.add_argument(
        "--scope-output-root",
        type=Path,
        default=Path("data/processed/sec_probability_scope"),
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.request_interval < 0.20:
        parser.error("--request-interval must remain at or above 0.20 seconds")
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        parser.error("SEC_USER_AGENT must identify the project and include a real contact email")
    sample_report = json.loads(
        (args.sample_plan_root / "sample_plan_report.json").read_text(encoding="utf-8")
    )
    targets = load_catalog_table(args.sample_plan_root, "sample_filing_cover_targets.parquet")
    client = SecClient(
        user_agent,
        request_interval_seconds=args.request_interval,
        timeout_seconds=60.0,
    )

    def show_progress(done: int, total: int, target_id: str) -> None:
        if done == 1 or done % 10 == 0 or done == total:
            print(f"COLLECT progress={done}/{total} target={target_id[:12]}", flush=True)

    collection = collect_filing_cover_artifacts(
        targets,
        client,
        raw_root=args.raw_root,
        limit=args.limit,
        artifact_policy="index-and-cover",
        progress=show_progress,
    )
    print(
        "COLLECT DONE "
        f"selected={collection.selected_targets:,} "
        f"completed={collection.targets_completed:,} "
        f"failed={collection.targets_failed:,} "
        f"requests={collection.requests_made:,} "
        f"sample={sample_report['dataset_version']}",
        flush=True,
    )
    history = build_filing_cover_history(
        args.catalog_root,
        args.filing_target_plan_root,
        raw_root=args.raw_root,
        output_root=args.cover_output_root,
    )
    scope = build_equity_scope_audit(
        args.sample_plan_root,
        history.output_root,
        output_root=args.scope_output_root,
    )
    report = scope.report
    print(
        "SCOPE DONE "
        f"cover_version={history.report['dataset_version']} "
        f"cover_ready={report['cover_ready']:,}/{report['selected_issuer_years']:,} "
        f"adjudicated={report['scope_adjudicated']:,} "
        f"provisional_equity={report['provisional_equity_candidates']:,} "
        f"scope_gate={report['scope_gate']} "
        f"output={scope.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
