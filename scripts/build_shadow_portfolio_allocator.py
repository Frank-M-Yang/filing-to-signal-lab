"""Build cash-constrained paper allocations from shadow intentions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadow_portfolio_allocator import build_shadow_portfolio_allocations


def main() -> int:
    """Build and summarize one immutable paper-account allocation snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger-root", type=Path, required=True)
    parser.add_argument("--market-evidence-root", type=Path, required=True)
    parser.add_argument("--trading-policy-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/shadow_portfolio_allocator"),
    )
    args = parser.parse_args()
    result = build_shadow_portfolio_allocations(
        args.shadow_ledger_root,
        args.market_evidence_root,
        args.trading_policy_root,
        output_root=args.output_root,
    )
    print(
        "SHADOW PORTFOLIO ALLOCATOR DONE "
        f"events={result.report['events']} "
        f"intents={result.report['shadow_intents']} "
        f"order_candidates={result.report['order_candidates']} "
        f"reserved_cash=${result.report['reserved_cash_usd']:,.2f} "
        f"gate={result.report['allocation_gates_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
