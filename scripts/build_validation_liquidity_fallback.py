"""Collect secondary raw OHLCV for primary pre-event liquidity gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import load_catalog_table
from sec_data.validation_liquidity_fallback import (
    build_validation_liquidity_fallback_audit,
    build_validation_liquidity_fallback_plan,
    collect_validation_liquidity_fallback_artifacts,
)


def main() -> int:
    """Freeze, optionally collect, and audit the secondary liquidity batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquidity-plan-root", type=Path, default=None)
    parser.add_argument("--liquidity-audit-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_liquidity_fallback/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_liquidity_fallback/raw_store"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_liquidity_fallback"),
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    liquidity_plan = args.liquidity_plan_root or _latest_root(
        Path("data/market/validation_liquidity/targets"), "target_report.json"
    )
    liquidity_audit = args.liquidity_audit_root or _latest_root(
        Path("data/processed/validation_liquidity"), "validation_liquidity_report.json"
    )
    plan = build_validation_liquidity_fallback_plan(
        liquidity_plan,
        liquidity_audit,
        output_root=args.target_output_root,
    )
    print(
        "LIQUIDITY FALLBACK TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['fallback_targets']} "
        f"gaps={plan.report['primary_gap_sessions']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        targets = load_catalog_table(
            plan.output_root, "validation_liquidity_fallback_targets.parquet"
        )

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(
                f"LIQUIDITY FALLBACK progress={done}/{total} target={target_id[:12]}",
                flush=True,
            )

        collection = collect_validation_liquidity_fallback_artifacts(
            targets,
            raw_root=args.raw_root,
            limit=args.limit,
            request_interval=args.request_interval,
            progress=show_progress,
        )
        print(
            "LIQUIDITY FALLBACK COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made}",
            flush=True,
        )
    audit = build_validation_liquidity_fallback_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "LIQUIDITY FALLBACK AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']}/{report['fallback_targets']} "
        f"recovered={report['fallback_prints_observed']}/"
        f"{report['primary_gap_sessions']} "
        f"no_print={report['secondary_no_print_sessions']} "
        f"gate={report['fallback_evidence_gate']} "
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
