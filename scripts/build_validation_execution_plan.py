"""Build formal validation execution windows and market-evidence query plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_execution_plan import build_validation_execution_plan


def main() -> int:
    """Resolve latest audited inputs and persist the outcome-blind execution plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument("--identity-resolution-root", type=Path, default=None)
    parser.add_argument(
        "--raw-calendar-root",
        type=Path,
        default=Path("data/market/exchange_calendar"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_execution_plan"),
    )
    args = parser.parse_args()
    market = args.market_plan_root or _latest_root(
        Path("data/processed/validation_market_plan"),
        "validation_market_plan_report.json",
    )
    identity = args.identity_resolution_root or _latest_root(
        Path("data/processed/validation_identity_repair_resolution"),
        "identity_repair_report.json",
    )
    result = build_validation_execution_plan(
        market,
        identity,
        raw_calendar_root=args.raw_calendar_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION EXECUTION PLAN DONE "
        f"version={report['dataset_version']} "
        f"windows={report['formal_execution_windows_ready']}/"
        f"{report['validation_events']} "
        f"components={report['security_event_components']} "
        f"price_targets={report['price_targets']} "
        f"actions={report['action_targets']} "
        f"delisting={report['delisting_targets']} "
        f"price_gate={report['price_collection_authorized']} "
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
