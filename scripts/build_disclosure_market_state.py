"""Rebuild market-state features cut at the earliest SEC disclosure and compare to v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.build_validation_market_state import _latest_root
from sec_data.disclosure_market_state import build_disclosure_market_state


def main() -> int:
    """Same inputs as v2 market state; only the cutoff clock differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-root", type=Path, required=True)
    parser.add_argument("--return-gate-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/disclosure_market_state")
    )
    args = parser.parse_args()
    processed = Path("data/processed")
    result = build_disclosure_market_state(
        args.return_gate_root
        or _latest_root(processed / "validation_return_gate", "validation_return_gate_report.json"),
        args.execution_plan_root
        or _latest_root(
            processed / "validation_execution_plan", "validation_execution_plan_report.json"
        ),
        args.price_audit_root
        or _latest_root(processed / "validation_prices", "validation_price_batch_report.json"),
        args.benchmark_root
        or _latest_root(processed / "validation_benchmark", "massive_benchmark_report.json"),
        args.action_root
        or _latest_root(
            processed / "validation_security_actions", "massive_security_actions_report.json"
        ),
        args.feature_root
        or _latest_root(
            processed / "validation_expansion_features", "validation_feature_report.json"
        ),
        args.clock_root,
        output_root=args.output_root,
    )
    report = {
        key: value
        for key, value in result.report.items()
        if key not in {"artifacts", "input_manifest_sha256", "implementation_sha256"}
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"DISCLOSURE MARKET STATE DONE output={result.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
