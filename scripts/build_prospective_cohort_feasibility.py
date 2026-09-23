"""Audit prospective capacity offline; never collect, register, or fit a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from prospective_cohort_feasibility import (
    CohortFeasibilityError,
    build_prospective_cohort_feasibility,
)


def main() -> None:
    """Resolve hash-linked local parents and write the separate planning bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--provider-audit-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/prospective_cohort_feasibility")
    )
    args = parser.parse_args()
    processed: Path = args.processed_root
    quality = args.quality_root or _latest_report_root(
        processed / "prospective_consensus_quality", "prospective_consensus_quality_report.json"
    )
    quality_report = json.loads(
        (quality / "prospective_consensus_quality_report.json").read_text(encoding="utf-8")
    )
    readiness = _linked_root(
        processed / "prospective_market_innovation_readiness",
        quality_report["readiness_manifest_sha256"],
    )
    ready_manifest = json.loads((readiness / "output_manifest.json").read_text(encoding="utf-8"))
    registry = _linked_root(
        processed / "prospective_market_innovation_preregistration",
        ready_manifest["preregistration_manifest_sha256"],
    )
    consensus = _linked_root(
        processed / "alpha_vantage_consensus", ready_manifest["consensus_audit_manifest_sha256"]
    )
    provider = args.provider_audit_root or _latest_report_root(
        processed / "consensus_provider_audit", "consensus_provider_audit_report.json"
    )
    core = args.core_root or processed / "global500_2026" / "core_asset_v1"
    result = build_prospective_cohort_feasibility(
        quality,
        readiness,
        registry,
        core,
        consensus,
        provider,
        output_root=args.output_root,
    )
    report = result.report
    print(
        f"COHORT FEASIBILITY DONE version={report['dataset_version']} "
        f"quality_events={report['clean_events']}/{report['registered_events']} "
        f"quality_issuers={report['issuers_with_any_clean_event']}/{report['registered_issuers']} "
        f"unprobed={report['eligible_unprobed_queue']} "
        f"registered_new=False collected=False output={result.output_root}"
    )


def _latest_report_root(base: Path, name: str) -> Path:
    reports = list(base.glob(f"*/{name}"))
    if not reports:
        raise CohortFeasibilityError(f"No local report found: {base}/{name}")
    # Source generation time, not filesystem mtime or numerical outcome.
    latest = max(
        reports,
        key=lambda p: (
            datetime.fromisoformat(json.loads(p.read_text(encoding="utf-8"))["generated_at"]),
            p.parent.name,
        ),
    )
    return latest.parent


def _linked_root(base: Path, expected_hash: str) -> Path:
    matches = [
        p.parent
        for p in base.glob("*/output_manifest.json")
        if hashlib.sha256(p.read_bytes()).hexdigest() == expected_hash
    ]
    if len(matches) != 1:
        raise CohortFeasibilityError(f"Expected one exact hash-linked parent under {base}")
    return matches[0]


if __name__ == "__main__":
    main()
