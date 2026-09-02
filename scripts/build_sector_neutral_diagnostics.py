"""Build the historical-SIC fixed-effect return diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.sector_neutral_diagnostics import build_sector_neutral_diagnostics


def main() -> int:
    """Resolve the latest tied assets and persist the immutable diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-diagnostic-root", type=Path, default=None)
    parser.add_argument("--filing-industry-root", type=Path, default=None)
    parser.add_argument("--validation-plan-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/sector_neutral_diagnostics"),
    )
    args = parser.parse_args()
    event_root = args.event_diagnostic_root or _latest_root(
        Path("data/processed/exploratory_event_return_diagnostics"),
        "event_return_diagnostic_report.json",
    )
    industry_root = args.filing_industry_root or _latest_root(
        Path("data/processed/sec_filing_cover_history"),
        "filing_cover_report.json",
    )
    validation_root = args.validation_plan_root or _latest_root(
        Path("data/processed/validation_expansion"),
        "validation_expansion_report.json",
    )
    result = build_sector_neutral_diagnostics(
        event_root,
        industry_root,
        validation_plan_root=validation_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "SECTOR NEUTRAL DIAGNOSTIC DONE "
        f"version={report['dataset_version']} "
        f"industry={report['historical_industry_exact_matches']}/"
        f"{report['complete_representation_events']} "
        f"fixed_effect={report['sector_fixed_effect_events']} "
        f"divisions={report['sic_divisions_fixed_effect_eligible']} "
        f"gate={report['sector_fixed_effect_diagnostic_allowed']} "
        f"tradable_sector_benchmark={report['tradable_sector_benchmark_gate']} "
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
