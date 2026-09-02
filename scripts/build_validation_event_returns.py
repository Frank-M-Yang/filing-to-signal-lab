"""Build the authorized formal validation event-return ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_event_returns import build_validation_event_return_ledger


def main() -> int:
    """Resolve immutable inputs and build the return ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-gate-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--price-reconciliation-root", type=Path, default=None)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--price-fallback-root", type=Path, default=None)
    parser.add_argument("--terminal-evidence-root", type=Path, default=None)
    parser.add_argument("--security-actions-root", type=Path, default=None)
    parser.add_argument("--action-continuity-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/validation_event_returns")
    )
    args = parser.parse_args()
    result = build_validation_event_return_ledger(
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
        args.price_reconciliation_root
        or _latest_root(
            Path("data/processed/validation_price_reconciliation"),
            "validation_price_reconciliation_report.json",
        ),
        args.price_audit_root
        or _latest_root(
            Path("data/processed/validation_prices"), "validation_price_batch_report.json"
        ),
        args.terminal_evidence_root
        or _latest_root(
            Path("data/processed/validation_terminal_evidence"),
            "validation_terminal_evidence_report.json",
        ),
        args.security_actions_root
        or _latest_root(
            Path("data/processed/validation_security_actions"),
            "massive_security_actions_report.json",
        ),
        args.action_continuity_root
        or _latest_root(
            Path("data/processed/validation_action_continuity"),
            "validation_action_continuity_report.json",
        ),
        args.benchmark_root
        or _latest_root(
            Path("data/processed/validation_benchmark"), "massive_benchmark_report.json"
        ),
        price_fallback_root=args.price_fallback_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION EVENT RETURN LEDGER DONE "
        f"version={report['dataset_version']} "
        f"events={report['market_excess_returns_ready']}/"
        f"{report['authorized_event_components']} "
        f"issuers={report['issuers']} gate={report['formal_return_gate']} "
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
