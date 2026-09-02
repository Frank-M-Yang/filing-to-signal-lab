"""Run outcome-independent forecast selection, then exploratory return diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.market_augmented_innovation import (
    build_market_augmented_innovation,
    build_market_augmented_return_diagnostics,
)


def main() -> int:
    """Persist financial forecast selection before opening formal returns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-state-root", type=Path, default=None)
    parser.add_argument("--return-ledger-root", type=Path, default=None)
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=Path("data/processed/market_augmented_innovation"),
    )
    parser.add_argument(
        "--return-output-root",
        type=Path,
        default=Path("data/processed/market_augmented_return_diagnostics"),
    )
    parser.add_argument(
        "--skip-return-diagnostic",
        action="store_true",
        help="stop after the outcome-independent financial forecast selection",
    )
    args = parser.parse_args()
    market_root = args.market_state_root or _latest_root(
        Path("data/processed/validation_market_state"), "validation_market_state_report.json"
    )
    model = build_market_augmented_innovation(
        market_root,
        output_root=args.model_output_root,
    )
    report = model.report
    print(
        "MARKET-AUGMENTED FORECAST DONE "
        f"version={report['dataset_version']} "
        f"common={report['common_forecast_events']} "
        f"selected={report['selected_model_id']} "
        f"forecast_gate={report['financial_forecast_improvement_gate']} "
        f"returns_loaded={report['return_values_loaded']} "
        f"output={model.output_root}",
        flush=True,
    )
    if args.skip_return_diagnostic or not bool(report["financial_forecast_improvement_gate"]):
        reason = (
            "explicit_skip"
            if args.skip_return_diagnostic
            else "financial_forecast_improvement_gate_failed"
        )
        print(f"MARKET-AUGMENTED RETURN DIAGNOSTIC SKIPPED reason={reason}", flush=True)
        return 0
    return_root = args.return_ledger_root or _latest_root(
        Path("data/processed/validation_event_returns"), "validation_event_return_report.json"
    )
    diagnostic = build_market_augmented_return_diagnostics(
        model.output_root,
        return_root,
        output_root=args.return_output_root,
    )
    outcome = diagnostic.report
    print(
        "MARKET-AUGMENTED RETURN DIAGNOSTIC DONE "
        f"version={outcome['dataset_version']} "
        f"events={outcome['same_complete_case_events']} "
        f"issuers={outcome['same_complete_case_issuers']} "
        f"selected_ic={outcome['selected_innovation_ic']:.4f} "
        f"verdict={outcome['verdict']} "
        f"output={diagnostic.output_root}",
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
