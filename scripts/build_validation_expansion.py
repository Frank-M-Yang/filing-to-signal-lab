"""Freeze the unseen validation cohort and build its outcome-blind filing features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_expansion import (
    build_validation_expansion_features,
    build_validation_expansion_plan,
)


def main() -> int:
    """Build the immutable expansion plan and, by default, its SEC feature layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-root", type=Path, default=None)
    parser.add_argument("--catalog-root", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--pilot-return-root", type=Path, default=None)
    parser.add_argument("--longitudinal-return-root", type=Path, default=None)
    parser.add_argument(
        "--companyfacts-snapshot-manifest",
        type=Path,
        default=Path(
            "data/sec/bulk/snapshots/companyfacts/"
            "dcebecc7f2d58f71566336ac6cf19d0e1daea1792f3555426b760f6ffcef0590/"
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--plan-output-root",
        type=Path,
        default=Path("data/processed/validation_expansion"),
    )
    parser.add_argument(
        "--feature-output-root",
        type=Path,
        default=Path("data/processed/validation_expansion_features"),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="freeze the cohort without parsing the local companyfacts ZIP",
    )
    args = parser.parse_args()
    scope_root = args.scope_root or _latest_root(
        Path("data/processed/sec_probability_scope"), "scope_audit_report.json"
    )
    catalog_root = args.catalog_root or _latest_root(
        Path("data/processed/sec_universe"), "catalog_report.json"
    )
    reference_root = args.reference_root or _latest_root(
        Path("data/processed/massive_reference"), "massive_reference_report.json"
    )
    pilot_return_root = args.pilot_return_root or _latest_root(
        Path("data/processed/event_return_ledger"), "event_return_report.json"
    )
    longitudinal_return_root = args.longitudinal_return_root or _latest_root(
        Path("data/processed/longitudinal_event_returns"), "longitudinal_return_report.json"
    )
    plan = build_validation_expansion_plan(
        scope_root,
        catalog_root,
        reference_root,
        [pilot_return_root, longitudinal_return_root],
        output_root=args.plan_output_root,
    )
    print(
        "VALIDATION EXPANSION PLAN DONE "
        f"version={plan.report['dataset_version']} "
        f"issuers={plan.report['cohort_issuers']} "
        f"events={plan.report['candidate_events']} "
        f"unobserved={plan.report['unobserved_validation_candidate_events']} "
        f"reference={plan.report['reference_queries_complete']}/"
        f"{plan.report['reference_target_rows']} "
        f"pending={plan.report['reference_queries_pending']} "
        f"gate={plan.report['planned_validation_size_gate']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.plan_only:
        return 0

    def show_feature_progress(done: int, total: int, cik: str) -> None:
        if done == 1 or done % 10 == 0 or done == total:
            print(f"FEATURE progress={done}/{total} cik={cik}", flush=True)

    features = build_validation_expansion_features(
        plan.output_root,
        catalog_root,
        args.companyfacts_snapshot_manifest,
        output_root=args.feature_output_root,
        progress=show_feature_progress,
    )
    print(
        "VALIDATION EXPANSION FEATURES DONE "
        f"version={features.report['dataset_version']} "
        f"history={features.report['quarterly_history_events']} "
        f"targets={features.report['target_events']} "
        f"unobserved_complete={features.report['unobserved_complete_feature_events']} "
        f"basic_window={features.report['unobserved_complete_inside_basic_price_window']} "
        f"outcomes_loaded={features.report['outcome_values_loaded']} "
        f"output={features.output_root}",
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
