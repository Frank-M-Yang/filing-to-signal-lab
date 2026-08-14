"""Freeze the historical probability sample and optionally adjudicate equity scope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.probability_sample import (
    ProbabilitySampleConfig,
    build_equity_scope_audit,
    build_probability_sample_plan,
)


def main() -> int:
    """Build a deterministic stage-one plan and optional filing-cover scope audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe-root",
        type=Path,
        default=Path("data/processed/sec_universe_design/sec-universe-design-v4-102aa9b3b185f7fb"),
    )
    parser.add_argument(
        "--target-plan-root",
        type=Path,
        default=Path("data/sec/filing_covers/targets/filing-cover-targets-v1-9d9a9ea7c9e40430"),
    )
    parser.add_argument(
        "--sample-output-root",
        type=Path,
        default=Path("data/processed/sec_probability_sample"),
    )
    parser.add_argument(
        "--cover-root",
        type=Path,
        default=None,
        help="processed filing-cover history to use for scope adjudication",
    )
    parser.add_argument(
        "--scope-output-root",
        type=Path,
        default=Path("data/processed/sec_probability_scope"),
    )
    parser.add_argument("--sample-per-stratum", type=int, default=10)
    args = parser.parse_args()
    plan = build_probability_sample_plan(
        args.universe_root,
        args.target_plan_root,
        output_root=args.sample_output_root,
        config=ProbabilitySampleConfig(sample_per_stratum=args.sample_per_stratum),
    )
    report = plan.report
    print(
        "PLAN DONE "
        f"version={report['dataset_version']} "
        f"selected={report['selected_issuer_years']:,} "
        f"strata={report['sampling_strata']:,} "
        f"weighted_total={report['weighted_frame_total']:,.0f} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.cover_root is not None:
        scope = build_equity_scope_audit(
            plan.output_root,
            args.cover_root,
            output_root=args.scope_output_root,
        )
        scope_report = scope.report
        print(
            "SCOPE DONE "
            f"version={scope_report['dataset_version']} "
            f"cover_ready={scope_report['cover_ready']:,} "
            f"adjudicated={scope_report['scope_adjudicated']:,} "
            f"provisional_equity={scope_report['provisional_equity_candidates']:,} "
            f"scope_gate={scope_report['scope_gate']} "
            f"output={scope.output_root}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
