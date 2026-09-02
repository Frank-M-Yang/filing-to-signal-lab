"""Build strict pre-filing market-state features for authorized validation events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_market_state import build_validation_market_state


def main() -> int:
    """Build the immutable market-state feature store from preserved evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-gate-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/validation_market_state")
    )
    args = parser.parse_args()
    result = build_validation_market_state(
        args.return_gate_root
        or _latest_root(
            Path("data/processed/validation_return_gate"), "validation_return_gate_report.json"
        ),
        args.execution_plan_root
        or _latest_root(
            Path("data/processed/validation_execution_plan"),
            "validation_execution_plan_report.json",
        ),
        args.price_audit_root
        or _latest_root(
            Path("data/processed/validation_prices"), "validation_price_batch_report.json"
        ),
        args.benchmark_root
        or _latest_root(
            Path("data/processed/validation_benchmark"), "massive_benchmark_report.json"
        ),
        args.action_root
        or _latest_root(
            Path("data/processed/validation_security_actions"),
            "massive_security_actions_report.json",
        ),
        args.feature_root
        or _latest_root(
            Path("data/processed/validation_expansion_features"),
            "validation_feature_report.json",
        ),
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION MARKET STATE DONE "
        f"version={report['dataset_version']} "
        f"ready={report['market_state_ready_events']}/{report['authorized_events']} "
        f"issuers={report['market_state_ready_issuers']} "
        f"outcomes_loaded={report['outcome_values_loaded']} "
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
