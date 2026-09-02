"""Build and optionally collect the frozen point-in-time revenue-consensus pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import load_catalog_table
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.revenue_consensus import (
    build_revenue_consensus_audit,
    build_revenue_consensus_plan,
    collect_revenue_consensus_artifacts,
)


def main() -> int:
    """Freeze targets, optionally collect Massive evidence, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-root", type=Path, default=None)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument("--primary-return-root", type=Path, default=None)
    parser.add_argument("--supplemental-identity-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/revenue_consensus/targets"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/market/revenue_consensus"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/revenue_consensus")
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=12.1)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds")
    tournament_root = args.tournament_root or _latest_root(
        Path("data/processed/revenue_guidance_tournament"),
        "revenue_guidance_tournament_report.json",
    )
    market_plan_root = args.market_plan_root or _latest_root(
        Path("data/processed/revenue_guidance_market_plan"),
        "validation_market_plan_report.json",
    )
    primary_return_root = args.primary_return_root or _latest_matching_return_root(tournament_root)
    supplemental_identity_root = args.supplemental_identity_root or _latest_root(
        Path("data/processed/revenue_guidance_identity_resolution"),
        "identity_repair_report.json",
    )
    plan = build_revenue_consensus_plan(
        tournament_root,
        market_plan_root,
        primary_return_root,
        supplemental_identity_root,
        output_root=args.target_output_root,
    )
    print(
        "REVENUE CONSENSUS TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['consensus_targets']} "
        f"issuers={plan.report['consensus_target_issuers']} "
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

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(f"CONSENSUS progress={done}/{total} target={target_id[:12]}", flush=True)

        collection = collect_revenue_consensus_artifacts(
            load_catalog_table(plan.output_root, "revenue_consensus_targets.parquet"),
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "REVENUE CONSENSUS COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_revenue_consensus_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "REVENUE CONSENSUS AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['provider_queries_complete']}/{report['consensus_targets']} "
        f"retrospective={report['retrospective_consensus_candidates']} "
        f"pit_ready={report['point_in_time_consensus_ready_events']} "
        f"gate={report['formal_consensus_gate']} "
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


def _latest_matching_return_root(tournament_root: Path) -> Path:
    forecasts = load_catalog_table(tournament_root, "revenue_guidance_forecasts.parquet")
    event_ids = set(forecasts["validation_event_id"].astype(str))
    candidates: list[tuple[int, str, Path]] = []
    for report_path in Path("data/processed/validation_event_returns").glob(
        "*/validation_event_return_report.json"
    ):
        ledger_path = report_path.parent / "validation_event_return_ledger.parquet"
        if not ledger_path.is_file():
            continue
        ledger = load_catalog_table(report_path.parent, ledger_path.name)
        overlap = len(event_ids.intersection(ledger["validation_event_id"].astype(str)))
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((overlap, str(report.get("generated_at", "")), report_path.parent))
    if not candidates or max(item[0] for item in candidates) == 0:
        raise FileNotFoundError("No primary return ledger overlaps revenue-guidance events")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


if __name__ == "__main__":
    sys.exit(main())
