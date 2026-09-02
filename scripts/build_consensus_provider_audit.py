"""Build the evidence-backed analyst-consensus provider capability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.consensus_provider_audit import build_consensus_provider_audit


def main() -> int:
    """Resolve latest immutable inputs and persist provider decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-plan-root", type=Path, default=None)
    parser.add_argument("--massive-audit-root", type=Path, default=None)
    parser.add_argument(
        "--eulerpool-probe-root",
        type=Path,
        default=Path("data/market/eulerpool_consensus/probes"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/consensus_provider_audit"),
    )
    args = parser.parse_args()
    plan_root = args.consensus_plan_root or _latest_root(
        Path("data/market/revenue_consensus/targets"), "revenue_consensus_plan_report.json"
    )
    massive_root = args.massive_audit_root or _latest_root(
        Path("data/processed/revenue_consensus"), "revenue_consensus_audit_report.json"
    )
    result = build_consensus_provider_audit(
        plan_root,
        massive_root,
        eulerpool_probe_root=args.eulerpool_probe_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "CONSENSUS PROVIDER AUDIT DONE "
        f"version={report['dataset_version']} "
        f"massive_entitlement_failures={report['massive_entitlement_failures']} "
        f"eulerpool_tickers={report['eulerpool_probe_tickers']} "
        f"eulerpool_rows={report['eulerpool_probe_rows']} "
        f"revenue_field={report['eulerpool_revenue_field_present']} "
        f"provider_gate={report['provider_gate']} "
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
