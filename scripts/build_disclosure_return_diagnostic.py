"""Re-time the observed validation returns at the earliest disclosure and compare inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.build_forward_disclosure_clock import _latest_root
from sec_data.disclosure_return_diagnostic import build_disclosure_return_diagnostic


def main() -> int:
    """Diagnostic on previously observed outcomes; never validation or model selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-ledger-root", type=Path, default=None)
    parser.add_argument("--clock-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--price-audit-root", type=Path, default=None)
    parser.add_argument("--security-actions-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/disclosure_return_diagnostic")
    )
    args = parser.parse_args()
    processed = Path("data/processed")
    result = build_disclosure_return_diagnostic(
        args.return_ledger_root
        or _latest_root(
            processed / "validation_event_returns", "validation_event_return_report.json"
        ),
        args.clock_root
        or _latest_root(processed / "first_public_clock", "first_public_clock_report.json"),
        args.feature_root
        or _latest_root(
            processed / "validation_expansion_features", "validation_feature_report.json"
        ),
        args.execution_plan_root
        or _latest_root(
            processed / "validation_execution_plan", "validation_execution_plan_report.json"
        ),
        args.price_audit_root
        or _latest_root(processed / "validation_prices", "validation_price_batch_report.json"),
        args.security_actions_root
        or _latest_root(
            processed / "validation_security_actions", "massive_security_actions_report.json"
        ),
        args.benchmark_root
        or _latest_root(processed / "validation_benchmark", "massive_benchmark_report.json"),
        output_root=args.output_root,
    )
    report = {
        key: value
        for key, value in result.report.items()
        if key not in {"artifacts", "input_manifest_sha256", "implementation_sha256", "config"}
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"DISCLOSURE RETURN DIAGNOSTIC DONE output={result.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
