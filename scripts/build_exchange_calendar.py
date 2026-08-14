"""Collect official exchange documents and build historical event sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.exchange_calendar import (
    build_exchange_calendar_audit,
    collect_official_calendar_sources,
)


def main() -> int:
    """Optionally collect official files and build the calendar alignment audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-root",
        type=Path,
        default=None,
        help="Massive price audit; defaults to latest complete pilot",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/exchange_calendar"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/exchange_calendar"),
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="download official exchange documents before building",
    )
    args = parser.parse_args()
    if args.collect:
        collection = collect_official_calendar_sources(raw_root=args.raw_root)
        print(
            "CALENDAR SOURCES DONE "
            f"selected={collection.selected_sources:,} "
            f"completed={collection.sources_completed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,}",
            flush=True,
        )
    price_root = args.price_root or _latest_complete_price_audit(
        Path("data/processed/massive_prices")
    )
    result = build_exchange_calendar_audit(
        price_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CALENDAR AUDIT DONE "
        f"version={report['dataset_version']} "
        f"sessions={report['exchange_session_rows']:,} "
        f"aligned={report['calendar_aligned_events']:,}/{report['price_targets']:,} "
        f"price_ready={report['formal_event_window_ready']:,} "
        f"entry_changes={report['entry_date_changes_vs_observed_bars']:,} "
        f"exit_changes={report['exit_date_changes_vs_observed_bars']:,} "
        f"return_gate={report['return_study_gate']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_complete_price_audit(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_price_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("price_targets", 0)) > 0 and int(
            report.get("targets_collected", 0)
        ) == int(report.get("price_targets", -1)):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(
            "No complete Massive price audit exists; run scripts.build_massive_price_pilot first"
        )
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
