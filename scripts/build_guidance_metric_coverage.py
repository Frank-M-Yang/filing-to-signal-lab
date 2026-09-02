"""Build the strict multi-metric prior-guidance coverage audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.guidance_metric_coverage import build_guidance_metric_coverage


def main() -> int:
    """Parse preserved SEC guidance and measure exact-quarter metric coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-plan-root", type=Path, default=None)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/guidance_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/guidance_metric_coverage"),
    )
    args = parser.parse_args()
    plan_root = args.guidance_plan_root or _latest_root(
        Path("data/sec/guidance_evidence/targets"),
        "guidance_plan_report.json",
    )
    feature_root = args.filing_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    result = build_guidance_metric_coverage(
        plan_root,
        feature_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    print(
        "GUIDANCE METRIC COVERAGE DONE "
        f"version={result.report['dataset_version']} "
        f"documents={result.report['documents_audited']} "
        f"observations={result.report['numeric_observations']} "
        f"evidence_authorized={result.report['authorized_evidence_metrics']} "
        f"model_authorized={result.report['authorized_model_metrics']} "
        f"recommended={result.report['recommended_next_metric']} "
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
