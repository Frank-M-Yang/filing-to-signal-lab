"""Combine independent authorized guidance return batches without outcome selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.revenue_guidance_return_bridge import (
    build_revenue_guidance_return_bridge,
)


def main() -> int:
    """Build the frozen guidance-only authorization and return bridge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-root", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, action="append", required=True)
    parser.add_argument("--return-ledger-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_return_bridge"),
    )
    args = parser.parse_args()
    result = build_revenue_guidance_return_bridge(
        args.tournament_root,
        args.authorization_root,
        args.return_ledger_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "REVENUE GUIDANCE RETURN BRIDGE DONE "
        f"coverage={report['authorized_return_events']}/{report['forecast_events']} "
        f"issuers={report['authorized_return_issuers']} "
        f"batches={report['evidence_batches']} output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
