"""Freeze the Conditional Drift temporal-holdout event set and evaluation rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditional_drift_holdout import build_conditional_drift_holdout_preregistration


def main() -> int:
    """Persist the outcome-blind holdout preregistration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-candidate-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_holdout_preregistration"),
    )
    args = parser.parse_args()
    forward_root = args.forward_candidate_root or _latest_scored_root(
        Path("data/processed/conditional_drift_forward_candidate")
    )
    result = build_conditional_drift_holdout_preregistration(
        forward_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CONDITIONAL DRIFT HOLDOUT PREREGISTERED "
        f"version={report['dataset_version']} "
        f"events={report['events']} "
        f"issuers={report['issuers']} "
        f"in_support={report['within_training_support_events']} "
        f"outcomes_loaded={report['outcome_values_loaded']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_scored_root(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/conditional_drift_forward_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("scored_candidate_events", 0)) <= 0:
            continue
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No scored forward candidate exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
