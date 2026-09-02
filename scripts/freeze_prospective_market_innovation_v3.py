"""Freeze the forward Market-Innovation v3 event registry and test protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prospective_market_innovation import (
    build_prospective_market_innovation_preregistration,
)


def main() -> int:
    """Persist the v3 outcome-blind registry using the latest eligible inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-panel-root", type=Path, default=None)
    parser.add_argument("--consensus-audit-root", type=Path, default=None)
    parser.add_argument("--observed-holdout-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_market_innovation_preregistration"),
    )
    args = parser.parse_args()
    panel_root = args.consensus_panel_root or _latest_report_root(
        Path("data/processed/forward_consensus_panel"),
        "forward_consensus_panel_report.json",
    )
    audit_root = args.consensus_audit_root or _latest_report_root(
        Path("data/processed/alpha_vantage_consensus"),
        "alpha_vantage_live_audit_report.json",
    )
    holdout_root = args.observed_holdout_root or _latest_report_root(
        Path("data/processed/conditional_drift_holdout_preregistration"),
        "holdout_preregistration_report.json",
    )
    result = build_prospective_market_innovation_preregistration(
        panel_root,
        audit_root,
        holdout_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "PROSPECTIVE MARKET INNOVATION V3 PREREGISTERED "
        f"version={report['dataset_version']} "
        f"events={report['registered_events']} "
        f"issuers={report['registered_issuers']} "
        f"firewalled={report['observed_holdout_events_firewalled']} "
        f"gates={report['registration_gates_passed']}/{report['registration_gates_total']} "
        f"actuals_loaded={report['actual_values_loaded']} "
        f"returns_loaded={report['return_values_loaded']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_report_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
