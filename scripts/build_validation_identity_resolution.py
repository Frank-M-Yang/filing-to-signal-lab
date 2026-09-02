"""Resolve validation identities from frozen SEC and Massive repair evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_identity_repair import (
    build_validation_identity_repair_resolution,
)


def main() -> int:
    """Resolve all repair inputs and persist final event identity readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument("--base-identity-audit-root", type=Path, default=None)
    parser.add_argument("--original-reference-audit-root", type=Path, default=None)
    parser.add_argument("--repair-plan-root", type=Path, default=None)
    parser.add_argument("--cover-audit-root", type=Path, default=None)
    parser.add_argument("--fallback-plan-root", type=Path, default=None)
    parser.add_argument("--fallback-reference-audit-root", type=Path, default=None)
    parser.add_argument("--filing-evidence-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_identity_repair_resolution"),
    )
    args = parser.parse_args()
    market = args.market_plan_root or _latest_root(
        Path("data/processed/validation_market_plan"), "validation_market_plan_report.json"
    )
    base_identity = args.base_identity_audit_root or _latest_root(
        Path("data/processed/validation_security_identity"),
        "validation_security_identity_report.json",
    )
    original_reference = args.original_reference_audit_root or _latest_root(
        Path("data/processed/validation_security_reference"),
        "massive_reference_report.json",
    )
    repair_plan = args.repair_plan_root or _latest_root(
        Path("data/sec/validation_identity_repair/targets"), "target_report.json"
    )
    cover = args.cover_audit_root or _latest_root(
        Path("data/processed/validation_identity_cover_repair"), "cover_repair_report.json"
    )
    fallback_plan = args.fallback_plan_root or _latest_root(
        Path("data/market/validation_identity_symbol_fallback/targets"),
        "target_report.json",
    )
    fallback_reference = args.fallback_reference_audit_root or _latest_root(
        Path("data/processed/validation_identity_symbol_reference"),
        "massive_reference_report.json",
    )
    filing = args.filing_evidence_root or _latest_root(
        Path("data/processed/validation_filing_evidence"),
        "validation_filing_evidence_report.json",
    )
    result = build_validation_identity_repair_resolution(
        market,
        base_identity,
        original_reference,
        repair_plan,
        cover,
        fallback_plan,
        fallback_reference,
        filing,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION IDENTITY RESOLUTION DONE "
        f"version={report['dataset_version']} "
        f"repair_issuers={report['repair_issuers_ready']}/{report['repair_issuers']} "
        f"events={report['price_candidate_events_identity_ready']}/"
        f"{report['price_candidate_events']} "
        f"coverage={report['price_candidate_event_identity_coverage']:.1%} "
        f"price_gate={report['price_collection_authorized']} "
        f"return_gate={report['return_outcome_read_authorized']} "
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
