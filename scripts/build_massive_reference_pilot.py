"""Build and optionally collect the Massive point-in-time reference pilot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_data.massive_reference import (
    MassiveReferenceClient,
    build_massive_reference_audit,
    build_massive_reference_target_plan,
    collect_massive_reference_artifacts,
    load_massive_reference_targets,
)


def main() -> int:
    """Freeze targets, optionally collect a small batch, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope-root",
        type=Path,
        default=Path(
            "data/processed/sec_probability_scope/sec-equity-scope-audit-v1-91be9ca9abf87966"
        ),
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/massive_reference/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/massive_reference"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/massive_reference"),
    )
    parser.add_argument("--limit", type=int, default=5)
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
    plan = build_massive_reference_target_plan(
        args.scope_root,
        output_root=args.target_output_root,
    )
    targets = load_massive_reference_targets(plan.output_root)
    print(
        "TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['price_mapping_targets']:,} "
        f"basic_window={plan.report['within_documented_basic_history_window']:,} "
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
            print(f"COLLECT progress={done}/{total} target={target_id[:12]}", flush=True)

        collection = collect_massive_reference_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
        if collection.blocking_reason == "authentication_failed":
            print(
                "AUTH ACTION: Massive rejected the API key (HTTP 401). Copy a current key "
                "from https://massive.com/dashboard, reset MASSIVE_API_KEY in this "
                "PowerShell session, and rerun; the failed target remains pending.",
                flush=True,
            )
        elif collection.blocking_reason == "entitlement_denied":
            print(
                "ENTITLEMENT ACTION: authentication succeeded, but the account plan does "
                "not permit this historical query (HTTP 403).",
                flush=True,
            )
    audit = build_massive_reference_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']:,}/{report['price_mapping_targets']:,} "
        f"reference_candidates={report['event_date_reference_candidates']:,} "
        f"price_ready={report['final_price_identity_ready']:,} "
        f"return_gate={report['return_study_gate']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
