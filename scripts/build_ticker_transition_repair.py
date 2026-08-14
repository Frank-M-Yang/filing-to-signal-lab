"""Collect and reconcile the pre-specified OB-to-TEAD ticker transition."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_data.massive_prices import (
    build_massive_price_audit,
    collect_massive_price_artifacts,
    load_massive_price_targets,
)
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    collect_massive_security_actions_artifacts,
    load_massive_security_actions_targets,
)
from sec_data.ticker_transition_repair import (
    build_ticker_transition_reconciliation,
    build_ticker_transition_repair_plan,
)


def main() -> int:
    """Build the six-query repair, optionally collect it, and audit continuity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-plan-root",
        type=Path,
        default=Path(
            "data/market/longitudinal_study/targets/longitudinal-market-plan-v1-54c0331b5073c8ad"
        ),
    )
    parser.add_argument(
        "--original-price-root",
        type=Path,
        default=Path("data/processed/longitudinal_prices/massive-price-audit-v1-465f09abea53b35d"),
    )
    parser.add_argument(
        "--original-action-root",
        type=Path,
        default=Path(
            "data/processed/longitudinal_security_actions/"
            "massive-security-actions-audit-v1-f457be666a6fed59"
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
        default=Path("data/market/longitudinal_study/transition_repairs"),
    )
    parser.add_argument(
        "--price-raw-root",
        type=Path,
        default=Path("data/market/longitudinal_study/transition_repair_prices"),
    )
    parser.add_argument(
        "--action-raw-root",
        type=Path,
        default=Path("data/market/longitudinal_study/transition_repair_actions"),
    )
    parser.add_argument(
        "--price-output-root",
        type=Path,
        default=Path("data/processed/ticker_transition_repair_prices"),
    )
    parser.add_argument(
        "--action-output-root",
        type=Path,
        default=Path("data/processed/ticker_transition_repair_actions"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/ticker_transition_repair"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic plan")
    plan = build_ticker_transition_repair_plan(
        args.market_plan_root,
        args.original_price_root,
        args.original_action_root,
        args.calendar_root,
        output_root=args.target_output_root,
    )
    print(
        "TRANSITION REPAIR PLAN DONE "
        f"version={plan.report['dataset_version']} queries={plan.report['planned_queries']} "
        f"output={plan.output_root}",
        flush=True,
    )
    price_targets = load_massive_price_targets(plan.price_plan_root)
    action_targets = load_massive_security_actions_targets(plan.action_plan_root)
    if args.collect:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

        def action_progress(done: int, total: int, target_id: str) -> None:
            print(f"REPAIR ACTION progress={done}/{total} target={target_id[:12]}", flush=True)

        action_collection = collect_massive_security_actions_artifacts(
            action_targets,
            client,
            raw_root=args.action_raw_root,
            limit=1,
            progress=action_progress,
        )
        print(
            "REPAIR ACTION COLLECT DONE "
            f"completed={action_collection.targets_completed}/"
            f"{action_collection.selected_targets} failed={action_collection.targets_failed} "
            f"requests={action_collection.requests_made} "
            f"blocked={action_collection.blocking_reason}",
            flush=True,
        )

        def price_progress(done: int, total: int, target_id: str) -> None:
            print(f"REPAIR PRICE progress={done}/{total} target={target_id[:12]}", flush=True)

        price_collection = collect_massive_price_artifacts(
            price_targets,
            client,
            raw_root=args.price_raw_root,
            limit=1,
            progress=price_progress,
        )
        print(
            "REPAIR PRICE COLLECT DONE "
            f"completed={price_collection.targets_completed}/{price_collection.selected_targets} "
            f"failed={price_collection.targets_failed} requests={price_collection.requests_made} "
            f"blocked={price_collection.blocking_reason}",
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
    if (
        int(action_audit.report["corporate_action_windows_ready"]) < 1
        or int(price_audit.report["targets_collected"]) < 1
    ):
        print(
            "TRANSITION REPAIR PENDING "
            f"actions_ready={action_audit.report['corporate_action_windows_ready']}/1 "
            f"prices_collected={price_audit.report['targets_collected']}/1; "
            "rerun with --collect in the PowerShell session containing MASSIVE_API_KEY",
            flush=True,
        )
        return 1
    repair = build_ticker_transition_reconciliation(
        plan.output_root,
        args.original_price_root,
        args.original_action_root,
        price_audit.output_root,
        action_audit.output_root,
        args.calendar_root,
        output_root=args.output_root,
    )
    print(
        "TRANSITION REPAIR DONE "
        f"version={repair.report['dataset_version']} "
        f"gate={repair.report['transition_gate']} "
        f"boundary={repair.report['old_last_session']}->{repair.report['new_first_session']} "
        f"overnight_return={repair.report['overnight_return']:.2%} "
        f"price_rows={repair.report['price_addition_rows']} "
        f"action_rows={repair.report['action_addition_rows']} "
        f"output={repair.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
