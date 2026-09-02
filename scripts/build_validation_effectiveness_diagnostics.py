"""Run the frozen Level/Delta/Innovation test on formal validation returns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_effectiveness_diagnostics import (
    build_validation_effectiveness_diagnostics,
)


def main() -> int:
    """Resolve latest immutable inputs and build formal validation diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-ledger-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--prior-preregistration-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_effectiveness_diagnostics"),
    )
    args = parser.parse_args()
    result = build_validation_effectiveness_diagnostics(
        args.return_ledger_root
        or _latest_root(
            Path("data/processed/validation_event_returns"),
            "validation_event_return_report.json",
        ),
        args.feature_root
        or _latest_root(
            Path("data/processed/validation_expansion_features"),
            "validation_feature_report.json",
        ),
        args.prior_preregistration_root
        or _latest_root(
            Path("data/processed/preliminary_effectiveness"),
            "preregistration_report.json",
        ),
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION EFFECTIVENESS DIAGNOSTIC DONE "
        f"version={report['dataset_version']} events={report['events']} "
        f"issuers={report['issuers']} innovation_ic={report['innovation_ic']:.4f} "
        f"delta_ic={report['delta_ic']:.4f} verdict={report['verdict']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report = _read_json(report_path)
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


if __name__ == "__main__":
    sys.exit(main())
