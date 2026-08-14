"""Command-line entry point for reproducible SEC company downloads."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from sec_data.client import SecConfigurationError, SecRequestError
from sec_data.pipeline import SecIngestionService


def main(argv: Sequence[str] | None = None) -> int:
    """Download, clean, and summarize one ticker or CIK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("company", help="Ticker or numeric CIK, for example AAPL or 320193")
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="download older submissions shards referenced by the primary response",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/sec"))
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", ""),
        help="declared SEC identity with contact email; defaults to SEC_USER_AGENT",
    )
    arguments = parser.parse_args(argv)
    try:
        result = SecIngestionService(arguments.user_agent, data_root=arguments.data_root).ingest(
            arguments.company, include_archives=arguments.include_archives
        )
    except (SecConfigurationError, SecRequestError, OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = {
        "cik": result.cik,
        "ticker": result.ticker,
        "company_name": result.company_name,
        "filings": len(result.filings),
        "facts": len(result.facts),
        "amendment_comparisons": len(result.amendment_comparisons),
        "changed_amendment_facts": int(
            result.amendment_comparisons["comparison_status"].eq("changed").sum()
        ),
        "rejections": len(result.rejections),
        "dataset_version": result.metadata["dataset_version"],
        "normalized_directory": result.metadata["normalized_directory"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
