"""Build, optionally collect, and audit the final longitudinal market batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_data.filing_catalog import load_catalog_table
from sec_data.longitudinal_market import build_longitudinal_market_plan
from sec_data.massive_benchmark import (
    build_massive_benchmark_audit,
    collect_massive_benchmark_artifacts,
)
from sec_data.massive_prices import (
    build_massive_price_audit,
    collect_massive_price_artifacts,
    load_massive_price_targets,
)
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    collect_massive_security_actions_artifacts,
    collect_massive_security_session_fallbacks,
    load_massive_security_actions_targets,
)


def main() -> int:
    """Run the resumable final market-data collection phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--longitudinal-root",
        type=Path,
        default=Path(
            "data/processed/longitudinal_expansion/"
            "longitudinal-event-expansion-v1.1-28158673fcdf01d6"
        ),
    )
    parser.add_argument(
        "--prior-actions-root",
        type=Path,
        default=Path(
            "data/processed/massive_security_actions/"
            "massive-security-actions-audit-v1-36067a60c713841b"
        ),
    )
    parser.add_argument(
        "--calendar-root",
        type=Path,
        default=Path(
            "data/processed/exchange_calendar/exchange-calendar-audit-v1-26f068f384d50b76"
        ),
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/longitudinal_study/targets"),
    )
    parser.add_argument(
        "--price-raw-root",
        type=Path,
        default=Path("data/market/longitudinal_study/prices"),
    )
    parser.add_argument(
        "--action-raw-root",
        type=Path,
        default=Path("data/market/longitudinal_study/actions"),
    )
    parser.add_argument(
        "--benchmark-raw-root",
        type=Path,
        default=Path("data/market/longitudinal_study/benchmark"),
    )
    parser.add_argument(
        "--price-output-root",
        type=Path,
        default=Path("data/processed/longitudinal_prices"),
    )
    parser.add_argument(
        "--action-output-root",
        type=Path,
        default=Path("data/processed/longitudinal_security_actions"),
    )
    parser.add_argument(
        "--benchmark-output-root",
        type=Path,
        default=Path("data/processed/longitudinal_benchmark"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.1,
        help="seconds between Massive calls; Basic requires at least 12 seconds",
    )
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic plan")
    plan = build_longitudinal_market_plan(
        args.longitudinal_root,
        args.prior_actions_root,
        args.calendar_root,
        output_root=args.target_output_root,
    )
    print(
        "LONGITUDINAL MARKET PLAN DONE "
        f"version={plan.report['dataset_version']} "
        f"issuers={plan.report['issuers']} events={plan.report['events']} "
        f"price_targets={plan.report['issuer_price_targets']} "
        f"action_targets={plan.report['security_action_targets']} "
        f"base_queries={plan.report['planned_base_queries']} "
        f"max_queries={plan.report['planned_maximum_queries_including_fallback']} "
        f"output={plan.output_root}",
        flush=True,
    )
    price_targets = load_massive_price_targets(plan.price_plan_root)
    action_targets = load_massive_security_actions_targets(plan.action_plan_root)
    client: MassiveReferenceClient | None = None
    if args.collect:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

        def show_action(done: int, total: int, target_id: str) -> None:
            print(f"ACTION progress={done}/{total} target={target_id[:12]}", flush=True)

        action_collection = collect_massive_security_actions_artifacts(
            action_targets,
            client,
            raw_root=args.action_raw_root,
            limit=args.limit,
            progress=show_action,
        )
        print(
            "ACTION COLLECT DONE "
            f"selected={action_collection.selected_targets} "
            f"completed={action_collection.targets_completed} "
            f"failed={action_collection.targets_failed} "
            f"requests={action_collection.requests_made} "
            f"reused={action_collection.snapshots_reused} "
            f"blocked={action_collection.blocking_reason}",
            flush=True,
        )
        sessions = load_catalog_table(args.calendar_root, "historical_exchange_sessions.parquet")

        def show_fallback(done: int, total: int, target_id: str) -> None:
            print(f"FALLBACK progress={done}/{total} target={target_id[:12]}", flush=True)

        fallback = collect_massive_security_session_fallbacks(
            action_targets,
            sessions,
            client,
            raw_root=args.action_raw_root,
            limit=args.limit,
            progress=show_fallback,
        )
        print(
            "FALLBACK COLLECT DONE "
            f"selected={fallback.selected_targets} completed={fallback.targets_completed} "
            f"failed={fallback.targets_failed} requests={fallback.requests_made} "
            f"reused={fallback.snapshots_reused} blocked={fallback.blocking_reason}",
            flush=True,
        )

        def show_price(done: int, total: int, target_id: str) -> None:
            print(f"PRICE progress={done}/{total} target={target_id[:12]}", flush=True)

        prices = collect_massive_price_artifacts(
            price_targets,
            client,
            raw_root=args.price_raw_root,
            limit=args.limit,
            progress=show_price,
        )
        print(
            "PRICE COLLECT DONE "
            f"selected={prices.selected_targets} completed={prices.targets_completed} "
            f"failed={prices.targets_failed} requests={prices.requests_made} "
            f"reused={prices.snapshots_reused} blocked={prices.blocking_reason}",
            flush=True,
        )

        def show_benchmark(done: int, total: int, query_kind: str) -> None:
            print(f"BENCHMARK progress={done}/{total} query={query_kind}", flush=True)

        benchmark_collection = collect_massive_benchmark_artifacts(
            plan.benchmark_plan_root,
            client,
            raw_root=args.benchmark_raw_root,
            progress=show_benchmark,
        )
        print(
            "BENCHMARK COLLECT DONE "
            f"selected={benchmark_collection.queries_selected} "
            f"completed={benchmark_collection.queries_completed} "
            f"failed={benchmark_collection.queries_failed} "
            f"requests={benchmark_collection.requests_made} "
            f"reused={benchmark_collection.snapshots_reused} "
            f"blocked={benchmark_collection.blocking_reason}",
            flush=True,
        )

    action_audit = build_massive_security_actions_audit(
        plan.action_plan_root,
        args.calendar_root,
        raw_root=args.action_raw_root,
        output_root=args.action_output_root,
    )
    price_audit = build_massive_price_audit(
        plan.price_plan_root,
        raw_root=args.price_raw_root,
        output_root=args.price_output_root,
    )
    benchmark_audit = build_massive_benchmark_audit(
        plan.benchmark_plan_root,
        args.calendar_root,
        raw_root=args.benchmark_raw_root,
        output_root=args.benchmark_output_root,
    )
    print(
        "LONGITUDINAL MARKET AUDIT DONE "
        f"identity={action_audit.report['historical_symbol_intervals_ready']}/"
        f"{action_audit.report['targets']} "
        f"actions={action_audit.report['corporate_action_windows_ready']}/"
        f"{action_audit.report['targets']} "
        f"prices={price_audit.report['targets_collected']}/"
        f"{price_audit.report['price_targets']} "
        f"benchmark={benchmark_audit.report['benchmark_data_ready']} "
        f"action_output={action_audit.output_root} "
        f"price_output={price_audit.output_root} "
        f"benchmark_output={benchmark_audit.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
