"""Assign registered prospective events their earliest SEC disclosure from preserved snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.forward_disclosure_clock import build_forward_disclosure_clock


def main() -> int:
    """Offline: reads hash-verified SEC submissions bytes only; makes no request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, default=None)
    parser.add_argument("--monitor-audit-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/forward_disclosure_clock")
    )
    args = parser.parse_args()
    result = build_forward_disclosure_clock(
        args.readiness_root
        or _latest_root(
            Path("data/processed/prospective_market_innovation_readiness"),
            "prospective_market_innovation_readiness_report.json",
        ),
        args.monitor_audit_root
        or _latest_root(
            Path("data/processed/forward_filing_monitor"),
            "forward_filing_monitor_audit_report.json",
        ),
        output_root=args.output_root,
    )
    summary = result.report["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "FORWARD DISCLOSURE CLOCK DONE "
        f"events={summary['events']} disclosed={summary['disclosed_events']} "
        f"requests={result.report['network_requests_made']} output={result.output_root}"
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
