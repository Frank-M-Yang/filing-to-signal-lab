"""Build the strict forward Level/Revision Delta/Innovation ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.forward_innovation_ledger import build_forward_innovation_ledger


def main() -> int:
    """Resolve latest forward inputs and persist a return-free feature ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-root", type=Path, default=None)
    parser.add_argument("--actual-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/forward_innovation_ledger")
    )
    args = parser.parse_args()
    consensus_root = args.consensus_root or _latest_root(
        Path("data/processed/forward_consensus_panel"),
        "forward_consensus_panel_report.json",
    )
    actual_root = args.actual_root or _latest_root(
        Path("data/processed/forward_actual_revenue"),
        "forward_actual_revenue_audit_report.json",
    )
    result = build_forward_innovation_ledger(
        consensus_root,
        actual_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "FORWARD INNOVATION LEDGER DONE "
        f"version={report['dataset_version']} "
        f"events={report['event_rows']} "
        f"raw={report['raw_innovation_ready_events']} "
        f"revision={report['revision_delta_ready_events']} "
        f"standardized={report['standardized_innovation_ready_events']} "
        f"feature_gate={report['feature_research_gate_passed']} "
        f"return_gate={report['return_research_gate_passed']} "
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
