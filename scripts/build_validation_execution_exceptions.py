"""Build the non-imputing formal execution exception ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_execution_exceptions import (
    build_validation_execution_exception_ledger,
)


def main() -> int:
    """Classify formal price gaps and preserve every validation event."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_execution_exceptions"),
    )
    args = parser.parse_args()
    price = args.price_audit_root or _latest_root(
        Path("data/processed/validation_prices"),
        "validation_price_batch_report.json",
    )
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    result = build_validation_execution_exception_ledger(
        price,
        execution,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION EXECUTION EXCEPTIONS DONE "
        f"version={report['dataset_version']} "
        f"ready={report['price_ready_components']}/{report['event_components']} "
        f"exceptions={report['execution_exceptions']} "
        f"non_executable={report['non_executable_under_frozen_rule']} "
        f"unresolved={report['unresolved_evidence_routes']} "
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
