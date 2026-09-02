"""Build the forward point-in-time shadow market-evidence plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadow_market_evidence import build_shadow_market_evidence_plan


def main() -> int:
    """Build and summarize one immutable market-evidence plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state-root", type=Path, required=True)
    parser.add_argument("--trading-policy-root", type=Path, required=True)
    parser.add_argument("--core-asset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/shadow_market_evidence"),
    )
    args = parser.parse_args()
    result = build_shadow_market_evidence_plan(
        args.event_state_root,
        args.trading_policy_root,
        args.core_asset_root,
        output_root=args.output_root,
    )
    print(
        "SHADOW MARKET EVIDENCE PLAN DONE "
        f"events={result.report['events']} "
        f"security_candidates={result.report['security_candidates_ready']} "
        f"accepted={result.report['accepted_events']} "
        f"queries={result.report['market_queries_planned']} "
        f"order_evidence={result.report['order_evidence_ready_events']} "
        f"gate={result.report['operational_gates_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
