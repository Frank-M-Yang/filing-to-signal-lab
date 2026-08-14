"""Freeze the first longitudinal effectiveness analysis before collecting returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.preliminary_effectiveness import (
    build_preliminary_effectiveness_preregistration,
)


def main() -> int:
    """Build the immutable hypothesis, event, and verdict registries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--longitudinal-root",
        type=Path,
        default=Path(
            "data/processed/longitudinal_expansion/"
            "longitudinal-event-expansion-v1.1-28158673fcdf01d6"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/preliminary_effectiveness"),
    )
    args = parser.parse_args()
    result = build_preliminary_effectiveness_preregistration(
        args.longitudinal_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "EFFECTIVENESS PREREGISTRATION DONE "
        f"version={report['dataset_version']} "
        f"events={report['complete_feature_events']}/{report['candidate_events']} "
        f"issuers={report['candidate_issuers']} "
        f"minimum={report['minimum_combined_feature_return_events']} "
        f"returns_observed={report['outcomes_observed_at_preregistration']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
