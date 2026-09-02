"""Audit authorized 20-session returns for revenue-guidance innovations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.revenue_guidance_return_audit import build_revenue_guidance_return_audit


def main() -> int:
    """Resolve latest immutable inputs and persist the return-readiness audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-root", type=Path, default=None)
    parser.add_argument("--authorization-root", type=Path, default=None)
    parser.add_argument("--return-ledger-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_return_audit"),
    )
    args = parser.parse_args()
    tournament_root = args.tournament_root or _latest_root(
        Path("data/processed/revenue_guidance_tournament"),
        "revenue_guidance_tournament_report.json",
    )
    authorization_root = args.authorization_root or _latest_root(
        Path("data/processed/validation_return_gate"), "validation_return_gate_report.json"
    )
    ledger_root = args.return_ledger_root or _latest_root(
        Path("data/processed/validation_event_returns"), "validation_event_return_report.json"
    )
    result = build_revenue_guidance_return_audit(
        tournament_root,
        authorization_root,
        ledger_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "REVENUE GUIDANCE RETURN AUDIT DONE "
        f"version={report['dataset_version']} "
        f"coverage={report['authorized_complete_case_events']}/{report['forecast_events']} "
        f"issuers={report['authorized_complete_case_issuers']} "
        f"diagnostic_gate={report['return_diagnostic_gate_passed']} "
        f"inference={report['return_inference_status']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
