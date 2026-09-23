"""Audit how much earlier Item 2.02 earnings 8-Ks disclose periodic results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.first_public_clock import FirstPublicClockConfig, build_first_public_clock_audit


def main() -> int:
    """Build the rule-based clock table from one hash-verified filing catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/first_public_clock")
    )
    parser.add_argument("--max-release-lead-days", type=int, default=90)
    args = parser.parse_args()
    result = build_first_public_clock_audit(
        args.catalog_root,
        output_root=args.output_root,
        config=FirstPublicClockConfig(max_release_lead_days=args.max_release_lead_days),
    )
    summary = result.report["summary"]
    for label, block in [("all", summary["all"]), *summary["by_form"].items()]:
        released = block["with_item_2_02_release"]
        print(
            f"{label}: reports={block['periodic_reports']} "
            f"with_release={released} ({block['share_with_release']:.1%})"
        )
        if released:
            print(
                f"  same_local_date={block['share_same_local_date']:.1%} "
                f"lead>=1d={block['share_lead_at_least_1_day']:.1%} "
                f"lead>=3d={block['share_lead_at_least_3_days']:.1%} "
                f"median={block['median_lead_days']:.2f}d "
                f"p90={block['p90_lead_days']:.2f}d "
                f"multi_candidate={block['share_multiple_candidates']:.1%} "
                f"8k_after_event_date={block['share_release_filed_after_event_date']:.1%}"
            )
    print(f"FIRST PUBLIC CLOCK AUDIT DONE output={result.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
