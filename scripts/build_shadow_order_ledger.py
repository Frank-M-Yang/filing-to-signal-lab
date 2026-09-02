"""Build the policy-closed shadow decision and order ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadow_order_ledger import build_shadow_order_ledger


def main() -> int:
    """Build and summarize one immutable shadow-ledger snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state-root", type=Path, required=True)
    parser.add_argument("--trading-policy-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/shadow_order_ledger"),
    )
    args = parser.parse_args()
    result = build_shadow_order_ledger(
        args.event_state_root,
        trading_policy_root=args.trading_policy_root,
        output_root=args.output_root,
    )
    counts = ",".join(
        f"{status}:{count}" for status, count in result.report["decision_status_counts"].items()
    )
    print(
        "SHADOW ORDER LEDGER DONE "
        f"events={result.report['events']} "
        f"decisions={counts} "
        f"orders={result.report['shadow_orders']} "
        f"positions={result.report['shadow_positions']} "
        f"paper_gate={result.report['paper_order_gate_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
