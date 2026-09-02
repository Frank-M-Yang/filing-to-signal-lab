"""Build outcome-blind pre-entry features for the frozen Conditional Drift queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditional_drift_forward_features import build_conditional_drift_forward_features


def main() -> int:
    """Resolve immutable assets and persist the forward feature bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-candidate-root", type=Path, default=None)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument("--execution-root", type=Path, default=None)
    parser.add_argument("--price-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_forward_features"),
    )
    args = parser.parse_args()
    forward_root = args.forward_candidate_root or _latest_root(
        Path("data/processed/conditional_drift_forward_candidate"),
        "conditional_drift_forward_report.json",
    )
    filing_root = args.filing_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    execution_root = args.execution_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    price_root = args.price_root or _latest_root(
        Path("data/processed/validation_prices"),
        "validation_price_batch_report.json",
    )
    action_root = args.action_root or _latest_root(
        Path("data/processed/validation_security_actions"),
        "massive_security_actions_report.json",
    )
    benchmark_root = args.benchmark_root or _latest_nonempty_benchmark_root(
        Path("data/processed/validation_benchmark")
    )
    result = build_conditional_drift_forward_features(
        forward_root,
        filing_root,
        execution_root,
        price_root,
        action_root,
        benchmark_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CONDITIONAL DRIFT FORWARD FEATURES DONE "
        f"version={report['dataset_version']} "
        f"queue={report['queue_events']} "
        f"market={report['market_feature_complete_events']} "
        f"reaction={report['reaction_feature_complete_events']} "
        f"score_ready={report['forward_score_ready_events']} "
        f"outcomes_loaded={report['outcome_values_loaded']} "
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


def _latest_nonempty_benchmark_root(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_benchmark_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("parsed_daily_bars", 0)) <= 0:
            continue
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No non-empty benchmark bundle exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
