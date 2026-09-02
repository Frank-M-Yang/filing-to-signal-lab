"""Run the point-in-time revenue-guidance forecast tournament."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.revenue_guidance_tournament import (
    build_revenue_guidance_tournament,
)


def main() -> int:
    """Resolve latest immutable inputs and persist the tournament."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", type=Path, default=None)
    parser.add_argument("--guidance-plan-root", type=Path, default=None)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_tournament"),
    )
    args = parser.parse_args()
    coverage_root = args.coverage_root or _latest_root(
        Path("data/processed/revenue_guidance_expansion_coverage"),
        "guidance_metric_coverage_report.json",
    )
    plan_root = args.guidance_plan_root or _latest_root(
        Path("data/sec/revenue_guidance_expansion/targets"),
        "revenue_guidance_expansion_report.json",
    )
    feature_root = args.filing_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    result = build_revenue_guidance_tournament(
        coverage_root,
        plan_root,
        feature_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "REVENUE GUIDANCE TOURNAMENT DONE "
        f"version={report['dataset_version']} "
        f"sample={report['complete_case_events']}/"
        f"{report['complete_case_issuers']} "
        f"guidance_mape={report['management_guidance_mape']:.4%} "
        f"seasonal_mape={report['seasonal_baseline_mape']:.4%} "
        f"improvement={report['relative_mape_improvement']:.4%} "
        f"gate={report['forecast_gate_passed']} "
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
