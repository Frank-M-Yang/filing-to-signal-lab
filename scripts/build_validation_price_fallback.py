"""Build, collect, and audit frozen secondary-source price gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import load_catalog_table
from sec_data.validation_price_fallback import (
    build_validation_price_fallback_audit,
    build_validation_price_fallback_plan,
    collect_validation_price_fallback_artifacts,
)


def main() -> int:
    """Run the small, resumable Yahoo raw-OHLC fallback batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exception-root", type=Path, default=None)
    parser.add_argument("--benchmark-audit-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_price_fallback/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_price_fallback"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_price_fallback"),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    exceptions = args.exception_root or _latest_root(
        Path("data/processed/validation_execution_exceptions"),
        "validation_execution_exception_report.json",
    )
    benchmark = args.benchmark_audit_root or _latest_root(
        Path("data/processed/validation_benchmark"),
        "massive_benchmark_report.json",
    )
    plan = build_validation_price_fallback_plan(
        exceptions,
        benchmark,
        output_root=args.target_output_root,
    )
    print(
        "VALIDATION PRICE FALLBACK TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['fallback_targets']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        targets = load_catalog_table(plan.output_root, "validation_price_fallback_targets.parquet")

        def show_progress(done: int, total: int, fallback_target_id: str) -> None:
            print(
                f"VALIDATION PRICE FALLBACK progress={done}/{total} "
                f"target={fallback_target_id[:12]}",
                flush=True,
            )

        collection = collect_validation_price_fallback_artifacts(
            targets,
            raw_root=args.raw_root,
            limit=args.limit,
            request_interval=args.request_interval,
            progress=show_progress,
        )
        print(
            "VALIDATION PRICE FALLBACK COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made}",
            flush=True,
        )
    audit = build_validation_price_fallback_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION PRICE FALLBACK AUDIT DONE "
        f"version={report['dataset_version']} "
        f"ready={report['fallback_prices_ready']}/{report['fallback_targets']} "
        f"gate={report['fallback_price_gate']} "
        f"return_gate={report['return_outcome_read_authorized']} "
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
