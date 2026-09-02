"""Freeze the outcome-blind cash-only shadow trading policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadow_trading_policy import build_shadow_trading_policy


def main() -> int:
    """Build and summarize the immutable first shadow policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-preregistration-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/shadow_trading_policy"),
    )
    args = parser.parse_args()
    result = build_shadow_trading_policy(
        args.prospective_preregistration_root,
        output_root=args.output_root,
    )
    config = result.report["config"]
    print(
        "SHADOW TRADING POLICY FROZEN "
        f"events={result.report['registered_events']} "
        f"issuers={result.report['registered_issuers']} "
        f"buy_threshold={config['buy_threshold']:.2%} "
        f"exit_or_avoid_threshold={config['exit_or_avoid_threshold']:.2%} "
        f"max_gross={config['maximum_gross_weight']:.0%} "
        f"gate={result.report['policy_gate_passed']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
