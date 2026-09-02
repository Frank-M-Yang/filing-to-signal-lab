"""Freeze formal validation action, delisting, and benchmark evidence routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_market_evidence_plan import (
    build_validation_market_evidence_plan,
)


def main() -> int:
    """Build the outcome-blind market-evidence acquisition plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--reference-audit-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/market/validation_market_evidence/targets"),
    )
    args = parser.parse_args()
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    reference = args.reference_audit_root or _latest_complete_reference(
        Path("data/processed/validation_security_reference")
    )
    result = build_validation_market_evidence_plan(
        execution,
        reference,
        output_root=args.output_root,
    )
    report = result.report
    minimum_queries = (
        report["minimum_action_queries"]
        + report["minimum_delisting_queries"]
        + report["benchmark_queries"]
    )
    print(
        "VALIDATION MARKET EVIDENCE PLAN DONE "
        f"version={report['dataset_version']} "
        f"actions={report['action_targets']} "
        f"ticker_events={report['ticker_events_direct_targets']} "
        f"session_fallbacks={report['session_grid_fallback_targets']} "
        f"delisting={report['delisting_targets']} "
        f"benchmark={report['benchmark_targets']} "
        f"min_queries={minimum_queries} "
        f"return_gate={report['return_outcome_read_authorized']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_complete_reference(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_reference_report.json"):
        report = _read_json(report_path)
        if int(report.get("targets_collected", -1)) == int(report.get("price_mapping_targets", -2)):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No complete validation security-reference audit exists")
    return max(candidates, key=lambda item: item[0])[1]


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
