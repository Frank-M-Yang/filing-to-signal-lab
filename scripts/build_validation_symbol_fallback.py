"""Collect SEC-grounded Massive ticker/date identity fallback queries."""

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
    collect_massive_symbol_reference_artifacts,
    load_massive_reference_targets,
)
from sec_data.validation_identity_repair import build_validation_symbol_fallback_plan


def main() -> int:
    """Build, optionally collect, and audit ticker/date fallback references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cover-audit-root", type=Path, default=None)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/market/validation_identity_symbol_fallback/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_identity_symbol_fallback"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_identity_symbol_reference"),
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic plan")
    cover = args.cover_audit_root or _latest_root(
        Path("data/processed/validation_identity_cover_repair"), "cover_repair_report.json"
    )
    plan = build_validation_symbol_fallback_plan(cover, output_root=args.target_root)
    targets = load_massive_reference_targets(plan.output_root)
    print(
        "SYMBOL FALLBACK TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['symbol_fallback_targets']} "
        f"issuers={plan.report['distinct_ciks']} "
        f"outcomes_loaded={plan.report['outcome_values_loaded']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect")
        client = MassiveReferenceClient(api_key, request_interval_seconds=args.request_interval)

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(
                f"SYMBOL FALLBACK COLLECT progress={done}/{total} target={target_id[:12]}",
                flush=True,
            )

        collection = collect_massive_symbol_reference_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "SYMBOL FALLBACK COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_massive_reference_audit(
        plan.output_root, raw_root=args.raw_root, output_root=args.output_root
    )
    report = audit.report
    print(
        "SYMBOL FALLBACK AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']}/{report['price_mapping_targets']} "
        f"candidates={report['event_date_reference_candidates']} "
        f"return_gate={report['return_study_gate']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report = _read_json(report_path)
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


if __name__ == "__main__":
    sys.exit(main())
