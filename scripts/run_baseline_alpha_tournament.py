"""Run the frozen local baseline tournament without opening the sealed holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from baseline_alpha_tournament import build_baseline_tournament


def main() -> int:
    """Resolve the latest feature store and persist walk-forward diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/baseline_alpha_tournament"),
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root or _latest_root(
        Path("data/processed/tradeable_dataset"), "tradeable_dataset_report.json"
    )
    result = build_baseline_tournament(dataset_root, output_root=args.output_root)
    print(
        "BASELINE ALPHA TOURNAMENT DONE "
        f"version={result.report['dataset_version']} "
        f"validation={result.report['validation_candidate_events']} "
        f"predictions={result.report['walk_forward_prediction_rows']} "
        f"holdout={result.report['holdout_status']} "
        f"verdict={result.report['selection_verdict']} "
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
