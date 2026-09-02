"""Run queue freeze, outcome-blind feature build, and final pre-outcome scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditional_drift_forward import build_conditional_drift_forward_candidate
from conditional_drift_forward_features import build_conditional_drift_forward_features


def main() -> int:
    """Run the three immutable stages without manually copying version paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_forward_candidate"),
    )
    parser.add_argument(
        "--feature-output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_forward_features"),
    )
    args = parser.parse_args()
    optimization_root = _latest_root(
        Path("data/processed/conditional_drift_optimization"),
        "conditional_drift_v2_report.json",
    )
    validation_root = _latest_root(
        Path("data/processed/validation_expansion"),
        "validation_expansion_report.json",
    )
    filing_root = _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    identity_root = _latest_root(
        Path("data/processed/validation_identity_repair_resolution"),
        "identity_repair_report.json",
    )
    outcome_root = _latest_root(
        Path("data/processed/validation_event_returns"),
        "validation_event_return_report.json",
    )
    execution_root = _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    price_root = _latest_root(
        Path("data/processed/validation_prices"),
        "validation_price_batch_report.json",
    )
    action_root = _latest_root(
        Path("data/processed/validation_security_actions"),
        "massive_security_actions_report.json",
    )
    benchmark_root = _latest_nonempty_benchmark_root(Path("data/processed/validation_benchmark"))
    queue = build_conditional_drift_forward_candidate(
        optimization_root,
        validation_root,
        filing_root,
        identity_root,
        outcome_exposure_root=outcome_root,
        output_root=args.candidate_output_root,
    )
    features = build_conditional_drift_forward_features(
        queue.output_root,
        filing_root,
        execution_root,
        price_root,
        action_root,
        benchmark_root,
        output_root=args.feature_output_root,
    )
    candidate_feature_path = features.output_root / "conditional_drift_forward_features.parquet"
    final = build_conditional_drift_forward_candidate(
        optimization_root,
        validation_root,
        filing_root,
        identity_root,
        outcome_exposure_root=outcome_root,
        candidate_feature_path=candidate_feature_path,
        output_root=args.candidate_output_root,
    )
    report = final.report
    print(
        "CONDITIONAL DRIFT FORWARD PIPELINE DONE "
        f"version={report['dataset_version']} "
        f"scored={report['scored_candidate_events']} "
        f"in_support={report['within_training_support_scored_events']} "
        f"severe_extrapolation={report['severe_extrapolation_scored_events']} "
        f"external_metrics={report['external_validation_metrics_computed']} "
        f"output={final.output_root}",
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


def _latest_nonempty_benchmark_root(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_benchmark_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("parsed_daily_bars", 0)) > 0:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No non-empty benchmark bundle exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
