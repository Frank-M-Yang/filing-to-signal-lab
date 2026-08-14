"""Build PIT Level/Delta/OOS Innovation for the 28 real-return events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.pilot_filing_features import build_pilot_filing_features


def main() -> int:
    """Build the immutable pilot feature history and target audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness-root",
        type=Path,
        default=Path("data/processed/sec_price_readiness/sec-price-readiness-v6-9e06c6d4b248e770"),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--companyfacts-snapshot-manifest",
        type=Path,
        default=Path(
            "data/sec/bulk/snapshots/companyfacts/"
            "dcebecc7f2d58f71566336ac6cf19d0e1daea1792f3555426b760f6ffcef0590/"
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/pilot_filing_features"),
    )
    args = parser.parse_args()
    result = build_pilot_filing_features(
        args.readiness_root,
        args.catalog_root,
        args.companyfacts_snapshot_manifest,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "PILOT FILING FEATURES DONE "
        f"version={report['dataset_version']} "
        f"history={report['quarterly_history_events']} "
        f"level={report['target_level_ready']}/{report['target_events']} "
        f"delta={report['target_delta_ready']}/{report['target_events']} "
        f"innovation={report['target_standardized_innovation_ready']}/"
        f"{report['target_events']} "
        f"return_gate={report['return_ic_minimum_20_events_pass']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
