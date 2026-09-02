"""Build the prospective analyst-consensus structural quality audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prospective_consensus_quality import build_prospective_consensus_quality


def main() -> int:
    """Build and print one immutable event-level quality assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_consensus_quality"),
    )
    args = parser.parse_args()
    result = build_prospective_consensus_quality(
        args.readiness_root,
        output_root=args.output_root,
    )
    counts = ",".join(
        f"{status}:{count}" for status, count in result.report["quality_status_counts"].items()
    )
    print(
        "PROSPECTIVE CONSENSUS QUALITY DONE "
        f"events={result.report['events']} "
        f"status={counts} "
        f"admitted={result.report['downstream_admitted_events']} "
        f"gate={result.report['quality_audit_gate_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
