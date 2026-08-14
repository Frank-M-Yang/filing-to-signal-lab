"""Download SEC submissions bulk data and build the historical filing catalog."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_data.bulk import SecBulkDownloader, load_bulk_snapshot_manifest
from sec_data.filing_catalog import FilingCatalogConfig, build_filing_catalog


def main() -> int:
    """Create the first immutable SEC-wide historical metadata asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--bulk-root", type=Path, default=Path("data/sec/bulk"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/sec_universe"))
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        help="Reuse a verified submissions snapshot instead of downloading a new one",
    )
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    if args.snapshot_manifest is not None:
        snapshot = load_bulk_snapshot_manifest(args.snapshot_manifest)
    else:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("Set SEC_USER_AGENT to an identified name and real contact email")
        snapshot = SecBulkDownloader(user_agent, root=args.bulk_root).download("submissions")
    config = FilingCatalogConfig(start_year=args.start_year, end_year=args.end_year)
    result = build_filing_catalog(
        snapshot,
        output_root=args.output_root,
        config=config,
        progress_every=args.progress_every,
    )
    report = result.report
    print(
        "DONE "
        f"version={report['dataset_version']} "
        f"entities={report['parsed_entities']:,} "
        f"filings={report['selected_filings']:,} "
        f"signals={report['filing_signals']:,} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
