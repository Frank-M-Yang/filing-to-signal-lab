"""Build the prospective v3 event state machine from immutable daily artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prospective_event_state_machine import build_prospective_event_state_machine


def main() -> int:
    """Build and print one immutable event-state audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument("--actual-audit-root", type=Path, required=True)
    parser.add_argument("--innovation-ledger-root", type=Path, required=True)
    parser.add_argument("--consensus-quality-root", type=Path, required=True)
    parser.add_argument("--previous-state-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_event_state_machine"),
    )
    args = parser.parse_args()
    result = build_prospective_event_state_machine(
        args.readiness_root,
        args.actual_audit_root,
        args.innovation_ledger_root,
        args.consensus_quality_root,
        output_root=args.output_root,
        previous_state_root=args.previous_state_root,
    )
    counts = ",".join(f"{state}:{count}" for state, count in result.report["state_counts"].items())
    print(
        "PROSPECTIVE EVENT STATE MACHINE DONE "
        f"events={result.report['events']} "
        f"states={counts} "
        f"rejected={result.report['rejected_events']} "
        f"gate={result.report['all_state_gates_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
