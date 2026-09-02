"""Build the forward-only analyst-consensus state and revision panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.forward_consensus_panel import build_forward_consensus_panel


def main() -> int:
    """Resolve the latest live-consensus audit and persist panel readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-audit-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/forward_consensus_panel"),
    )
    args = parser.parse_args()
    audit_root = args.consensus_audit_root or _latest_root(
        Path("data/processed/alpha_vantage_consensus"),
        "alpha_vantage_live_audit_report.json",
    )
    result = build_forward_consensus_panel(audit_root, output_root=args.output_root)
    report = result.report
    print(
        "FORWARD CONSENSUS PANEL DONE "
        f"version={report['dataset_version']} "
        f"snapshots={report['forward_snapshot_rows']} "
        f"capture_dates={report['capture_dates']} "
        f"next_events={report['next_events']} "
        f"revisions={report['revision_rows']}/{report['revision_issuers']} "
        f"watchlist_gate={report['event_watchlist_gate_passed']} "
        f"revision_gate={report['revision_research_gate_passed']} "
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
