"""Build accession-matched point-in-time company asset-size features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.bulk import load_bulk_snapshot_manifest
from sec_data.size_features import build_event_time_size_features


def main() -> int:
    """Create event and issuer-year total-asset size assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--companyfacts-manifest",
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
        default=Path("data/processed/sec_size_features"),
    )
    args = parser.parse_args()
    snapshot = load_bulk_snapshot_manifest(args.companyfacts_manifest)
    result = build_event_time_size_features(
        args.catalog_root,
        snapshot,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "DONE "
        f"version={report['dataset_version']} "
        f"events_with_size={report['event_filings_with_total_assets']:,} "
        f"issuer_year_coverage={report['issuer_year_size_coverage']:.1%} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
