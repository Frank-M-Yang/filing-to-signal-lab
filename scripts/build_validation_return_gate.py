"""Authorize return reading for the outcome-blind liquidity complete cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_return_gate import build_validation_return_gate


def main() -> int:
    """Build the immutable validation return authorization ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquidity-audit-root", type=Path, default=None)
    parser.add_argument("--terminal-evidence-root", type=Path, default=None)
    parser.add_argument("--price-reconciliation-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_return_gate"),
    )
    args = parser.parse_args()
    liquidity = args.liquidity_audit_root or _latest_root(
        Path("data/processed/validation_liquidity"), "validation_liquidity_report.json"
    )
    terminal = args.terminal_evidence_root or _latest_root(
        Path("data/processed/validation_terminal_evidence"),
        "validation_terminal_evidence_report.json",
    )
    reconciliation = args.price_reconciliation_root or _latest_root(
        Path("data/processed/validation_price_reconciliation"),
        "validation_price_reconciliation_report.json",
    )
    result = build_validation_return_gate(
        liquidity,
        terminal,
        reconciliation,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION RETURN GATE DONE "
        f"version={report['dataset_version']} "
        f"authorized={report['authorized_event_components']}/"
        f"{report['universe_event_components']} "
        f"gate={report['return_outcome_read_authorized']} "
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
