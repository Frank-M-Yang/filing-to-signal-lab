"""Build an outcome-blind calendar-year exploratory price cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.exploratory_cohort import (
    CalendarYearExploratoryCohortConfig,
    build_calendar_year_exploratory_cohort,
)


def main() -> int:
    """Build and report one explicitly non-primary exploratory cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--price-root", type=Path, required=True)
    parser.add_argument("--calendar-root", type=Path, required=True)
    parser.add_argument("--event-year", type=int, default=2025)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/exploratory_price_cohort"),
    )
    args = parser.parse_args()
    result = build_calendar_year_exploratory_cohort(
        args.reference_root,
        args.price_root,
        args.calendar_root,
        output_root=args.output_root,
        config=CalendarYearExploratoryCohortConfig(event_year=args.event_year),
    )
    report = result.report
    print(
        "EXPLORATORY COHORT DONE "
        f"version={report['dataset_version']} "
        f"selected={report['selected_events']:,}/{report['source_price_targets']:,} "
        f"price_ready={report['selected_price_ready']:,} "
        f"calendar_ready={report['selected_calendar_ready']:,} "
        f"downstream_gate={report['downstream_evidence_collection_gate']} "
        f"primary_eligible={report['primary_effectiveness_eligible']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
