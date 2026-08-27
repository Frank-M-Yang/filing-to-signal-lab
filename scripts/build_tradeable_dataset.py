"""Freeze legacy outcomes and build the first point-in-time model feature store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tradeable_dataset import (
    build_tradeable_dataset,
    build_tradeable_dataset_preregistration,
)


def main() -> int:
    """Resolve the latest local inputs and persist the preregistration and dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, default=None)
    parser.add_argument("--price-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument(
        "--preregistration-output-root",
        type=Path,
        default=Path("data/processed/tradeable_dataset_preregistration"),
    )
    parser.add_argument(
        "--dataset-output-root",
        type=Path,
        default=Path("data/processed/tradeable_dataset"),
    )
    args = parser.parse_args()
    ledger_root = args.ledger_root or _latest_root(
        Path("data/processed/longitudinal_event_returns"),
        "output_manifest.json",
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
    preregistration = build_tradeable_dataset_preregistration(
        ledger_root,
        output_root=args.preregistration_output_root,
    )
    print(
        "TRADEABLE PREREGISTRATION DONE "
        f"version={preregistration.report['dataset_version']} "
        f"legacy_observed={preregistration.report['legacy_observed_events']} "
        f"output={preregistration.output_root}",
        flush=True,
    )
    dataset = build_tradeable_dataset(
        ledger_root,
        price_root,
        benchmark_root,
        action_root,
        preregistration.output_root,
        output_root=args.dataset_output_root,
    )
    print(
        "TRADEABLE DATASET DONE "
        f"version={dataset.report['dataset_version']} "
        f"events={dataset.report['events']} "
        f"combined={dataset.report['combined_feature_complete_events']} "
        f"evaluation={dataset.report['unobserved_evaluation_candidates']} "
        f"gate={dataset.report['current_model_gate']} "
        f"output={dataset.output_root}",
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
