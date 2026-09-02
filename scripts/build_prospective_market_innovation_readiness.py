"""Build the registry-aware v3 consensus and SEC-filing readiness ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prospective_market_innovation_readiness import (
    build_prospective_market_innovation_readiness,
)


def main() -> int:
    """Persist the latest outcome-free readiness state for the frozen v3 cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-root", type=Path, default=None)
    parser.add_argument("--consensus-audit-root", type=Path, default=None)
    parser.add_argument("--filing-monitor-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_market_innovation_readiness"),
    )
    args = parser.parse_args()
    preregistration_root = args.preregistration_root or _frozen_preregistration_root(
        Path("data/processed/prospective_market_innovation_preregistration")
    )
    consensus_root = args.consensus_audit_root or _latest_report_root(
        Path("data/processed/alpha_vantage_consensus"),
        "alpha_vantage_live_audit_report.json",
    )
    filing_root = args.filing_monitor_root or _latest_report_root(
        Path("data/processed/forward_filing_monitor"),
        "forward_filing_monitor_audit_report.json",
    )
    result = build_prospective_market_innovation_readiness(
        preregistration_root,
        consensus_root,
        filing_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "PROSPECTIVE MARKET INNOVATION READINESS DONE "
        f"version={report['dataset_version']} "
        f"consensus={report['registered_consensus_events']}/{report['registered_events']} "
        f"capture_dates={report['capture_dates']} "
        f"revisions={report['revision_ready_events']} "
        f"sec_issuers={report['submission_snapshot_issuers']} "
        f"filings={report['future_filing_candidates']} "
        f"foundation_gate={report['collection_foundation_gate_passed']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _frozen_preregistration_root(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/prospective_market_innovation_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if bool(report.get("preregistration_gate_passed")):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No frozen prospective preregistration exists under {search_root}")
    return min(candidates, key=lambda item: item[0])[1]


def _latest_report_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
