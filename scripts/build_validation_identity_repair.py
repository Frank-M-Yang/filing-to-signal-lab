"""Collect SEC cover evidence for unresolved validation security identities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.client import SecClient
from sec_data.filing_cover import collect_filing_cover_artifacts
from sec_data.validation_identity_repair import (
    build_validation_identity_cover_audit,
    build_validation_identity_repair_plan,
    load_validation_identity_repair_targets,
)


def main() -> int:
    """Freeze repair targets, optionally collect SEC primary documents, and audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-plan-root", type=Path, default=None)
    parser.add_argument("--identity-audit-root", type=Path, default=None)
    parser.add_argument("--reference-audit-root", type=Path, default=None)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/sec/validation_identity_repair/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/sec/validation_filing_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_identity_cover_repair"),
    )
    parser.add_argument("--collect-sec", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.request_interval < 0.20:
        parser.error("--request-interval must remain at or above 0.20 seconds")
    market = args.market_plan_root or _latest_root(
        Path("data/processed/validation_market_plan"), "validation_market_plan_report.json"
    )
    identity = args.identity_audit_root or _latest_root(
        Path("data/processed/validation_security_identity"),
        "validation_security_identity_report.json",
    )
    reference = args.reference_audit_root or _latest_root(
        Path("data/processed/validation_security_reference"),
        "massive_reference_report.json",
    )
    plan = build_validation_identity_repair_plan(
        market, identity, reference, output_root=args.target_root
    )
    targets = load_validation_identity_repair_targets(plan.output_root)
    print(
        "IDENTITY REPAIR TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"issuers={plan.report['repair_issuers']} "
        f"endpoints={plan.report['repair_endpoint_targets']} "
        f"outcomes_loaded={plan.report['outcome_values_loaded']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect_sec:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        client = SecClient(
            user_agent,
            request_interval_seconds=args.request_interval,
            timeout_seconds=60.0,
        )

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(
                f"IDENTITY REPAIR SEC progress={done}/{total} target={target_id[:12]}",
                flush=True,
            )

        collection = collect_filing_cover_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            artifact_policy="index-and-cover",
            progress=show_progress,
        )
        print(
            "IDENTITY REPAIR SEC DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused}",
            flush=True,
        )
    audit = build_validation_identity_cover_audit(
        plan.output_root, raw_root=args.raw_root, output_root=args.output_root
    )
    report = audit.report
    print(
        "IDENTITY COVER AUDIT DONE "
        f"version={report['dataset_version']} "
        f"primary={report['primary_artifacts_preserved']}/"
        f"{report['repair_endpoint_targets']} "
        f"symbol_ready={report['sec_cover_symbol_ready']} "
        f"tradable_ready={report['sec_cover_tradable_mapping_ready']} "
        f"ticker_fallbacks={report['ticker_fallback_queries']} "
        f"return_gate={report['return_outcome_read_authorized']} "
        f"output={audit.output_root}",
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
