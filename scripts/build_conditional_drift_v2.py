"""Build the frozen Conditional Drift v2 historical-development optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditional_drift_v2 import build_conditional_drift_v2


def main() -> int:
    """Resolve immutable local assets and persist the v2 optimization bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--price-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument("--calendar-root", type=Path, default=None)
    parser.add_argument("--transition-repair-root", type=Path, default=None)
    parser.add_argument(
        "--without-transition-repair",
        action="store_true",
        help="Do not apply the audited OB-to-TEAD continuity repair.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/conditional_drift_optimization"),
    )
    args = parser.parse_args()
    feature_store_root = args.feature_store_root or _latest_root(
        Path("data/processed/tradeable_dataset"), "tradeable_dataset_report.json"
    )
    price_root = args.price_root or _latest_root(
        Path("data/processed/longitudinal_prices"), "massive_price_report.json"
    )
    benchmark_root = args.benchmark_root or _latest_root(
        Path("data/processed/longitudinal_benchmark"), "massive_benchmark_report.json"
    )
    action_root = args.action_root or _latest_root(
        Path("data/processed/longitudinal_security_actions"),
        "massive_security_actions_report.json",
    )
    calendar_root = args.calendar_root or _latest_root(
        Path("data/processed/exchange_calendar"), "exchange_calendar_report.json"
    )
    transition_root = None
    if not args.without_transition_repair:
        transition_root = args.transition_repair_root or _latest_root(
            Path("data/processed/ticker_transition_repair"),
            "ticker_transition_repair_report.json",
        )
    result = build_conditional_drift_v2(
        feature_store_root,
        price_root,
        benchmark_root,
        action_root,
        calendar_root,
        transition_repair_root=transition_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CONDITIONAL DRIFT V2 DONE "
        f"version={report['dataset_version']} "
        f"same_event={report['v2_same_event_complete_events']} "
        f"predictions={report['development_prediction_events']} "
        f"improvement={report['primary_horizon_v2_minus_v1_spearman']:+.4f} "
        f"status={report['forward_candidate_status']} "
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
