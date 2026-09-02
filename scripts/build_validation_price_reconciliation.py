"""Reconcile primary and secondary validation price readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_price_reconciliation import (
    build_validation_price_reconciliation,
)


def main() -> int:
    """Build the source-tiered event and benchmark readiness ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exception-root", type=Path, default=None)
    parser.add_argument("--benchmark-audit-root", type=Path, default=None)
    parser.add_argument("--fallback-audit-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_price_reconciliation"),
    )
    args = parser.parse_args()
    exceptions = args.exception_root or _latest_root(
        Path("data/processed/validation_execution_exceptions"),
        "validation_execution_exception_report.json",
    )
    benchmark = args.benchmark_audit_root or _latest_root(
        Path("data/processed/validation_benchmark"),
        "massive_benchmark_report.json",
    )
    fallback = args.fallback_audit_root or _latest_root(
        Path("data/processed/validation_price_fallback"),
        "validation_price_fallback_report.json",
    )
    result = build_validation_price_reconciliation(
        exceptions,
        benchmark,
        fallback,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION PRICE RECONCILIATION DONE "
        f"events={report['event_components_price_ready']}/"
        f"{report['event_components']} "
        f"pending={report['event_components_price_pending']} "
        f"benchmark={report['benchmark_required_dates_ready']}/"
        f"{report['benchmark_required_dates']} "
        f"benchmark_gate={report['benchmark_price_gate']} "
        f"return_gate={report['return_outcome_read_authorized']} "
        f"output={result.output_root}",
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
