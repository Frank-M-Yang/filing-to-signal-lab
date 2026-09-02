"""Freeze the Conditional Drift challenger/candidate and build its holdout queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditional_drift_forward import build_conditional_drift_forward_candidate


def main() -> int:
    """Resolve immutable inputs and persist the outcome-blind forward bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-root", type=Path, default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument("--identity-root", type=Path, default=None)
    parser.add_argument("--outcome-exposure-root", type=Path, default=None)
    parser.add_argument(
        "--without-outcome-exposure-quarantine",
        action="store_true",
        help="Build without the local-outcome quarantine (not recommended).",
    )
    parser.add_argument(
        "--candidate-feature-path",
        type=Path,
        default=None,
        help="Optional outcome-free, pre-entry candidate feature parquet to score.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_forward_candidate"),
    )
    args = parser.parse_args()
    optimization_root = args.optimization_root or _latest_root(
        Path("data/processed/conditional_drift_optimization"),
        "conditional_drift_v2_report.json",
    )
    validation_root = args.validation_root or _latest_root(
        Path("data/processed/validation_expansion"),
        "validation_expansion_report.json",
    )
    filing_feature_root = args.filing_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    identity_root = args.identity_root or _latest_root(
        Path("data/processed/validation_identity_repair_resolution"),
        "identity_repair_report.json",
    )
    outcome_exposure_root = None
    if not args.without_outcome_exposure_quarantine:
        outcome_exposure_root = args.outcome_exposure_root or _latest_root(
            Path("data/processed/validation_event_returns"),
            "validation_event_return_report.json",
        )
    result = build_conditional_drift_forward_candidate(
        optimization_root,
        validation_root,
        filing_feature_root,
        identity_root,
        outcome_exposure_root=outcome_exposure_root,
        candidate_feature_path=args.candidate_feature_path,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CONDITIONAL DRIFT FORWARD FREEZE DONE "
        f"version={report['dataset_version']} "
        f"training={report['training_events']} "
        f"temporal_queue={report['temporal_complete_case_queue_events']} "
        f"quarantined={report['locally_materialized_outcome_quarantine_events']} "
        f"clean={report['clean_temporal_queue_events']} "
        f"identity_ready={report['identity_ready_clean_queue_events']} "
        f"scored={report['scored_candidate_events']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
