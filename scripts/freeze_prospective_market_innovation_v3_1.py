"""Freeze the v3.1 earliest-disclosure clock amendment before any outcome exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prospective_market_innovation_amendment import (
    freeze_prospective_market_innovation_amendment,
)
from scripts.build_forward_disclosure_clock import _latest_root


def main() -> int:
    """Fail closed if the registry, readiness ledger, or state machine show an outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-root", type=Path, default=None)
    parser.add_argument("--readiness-root", type=Path, default=None)
    parser.add_argument("--state-machine-root", type=Path, default=None)
    parser.add_argument("--clock-root", type=Path, default=None)
    parser.add_argument("--quality-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_market_innovation_amendment"),
    )
    args = parser.parse_args()
    processed = Path("data/processed")
    result = freeze_prospective_market_innovation_amendment(
        args.preregistration_root
        or _latest_root(
            processed / "prospective_market_innovation_preregistration",
            "prospective_market_innovation_report.json",
        ),
        args.readiness_root
        or _latest_root(
            processed / "prospective_market_innovation_readiness",
            "prospective_market_innovation_readiness_report.json",
        ),
        args.state_machine_root
        or _latest_root(
            processed / "prospective_event_state_machine",
            "prospective_event_state_machine_report.json",
        ),
        args.clock_root
        or _latest_root(
            processed / "forward_disclosure_clock", "forward_disclosure_clock_report.json"
        ),
        args.quality_root
        or _latest_root(
            processed / "prospective_disclosure_quality",
            "prospective_disclosure_quality_report.json",
        ),
        output_root=args.output_root,
    )
    report = result.report
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "amendment_version",
                    "declared_at",
                    "registered_events",
                    "registered_issuers",
                    "cohort_changed",
                    "thresholds_changed",
                    "verification_at_declaration",
                    "clock_dataset_version",
                    "quality_dataset_version",
                    "quality_status_counts",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"PROSPECTIVE MARKET INNOVATION V3.1 AMENDMENT FROZEN output={result.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
