"""Reconcile formal validation action identity without reading outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_action_continuity import build_validation_action_continuity


def main() -> int:
    """Resolve latest immutable inputs and run action continuity reconciliation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-gate-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--security-actions-root", type=Path, default=None)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--calendar-adapter-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_action_continuity"),
    )
    args = parser.parse_args()
    result = build_validation_action_continuity(
        args.return_gate_root
        or _latest_root(
            Path("data/processed/validation_return_gate"),
            "validation_return_gate_report.json",
        ),
        args.execution_plan_root
        or _latest_root(
            Path("data/processed/validation_execution_plan"),
            "validation_execution_plan_report.json",
        ),
        args.security_actions_root
        or _latest_root(
            Path("data/processed/validation_security_actions"),
            "massive_security_actions_report.json",
        ),
        args.price_audit_root
        or _latest_root(
            Path("data/processed/validation_prices"), "validation_price_batch_report.json"
        ),
        args.calendar_adapter_root
        or _latest_root(
            Path("data/processed/validation_calendar_adapter"), "exchange_calendar_report.json"
        ),
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION ACTION CONTINUITY DONE "
        f"version={report['dataset_version']} "
        f"ready={report['formal_action_evidence_ready']}/{report['targets']} "
        f"alternative={report['alternative_identity_ready']} "
        f"gate={report['formal_action_evidence_gate']} output={result.output_root}",
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
