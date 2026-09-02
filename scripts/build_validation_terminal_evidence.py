"""Build, collect, and audit SEC terminal consideration evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.client import SecClient
from sec_data.filing_cover import collect_filing_cover_artifacts
from sec_data.validation_terminal_evidence import (
    build_validation_terminal_evidence_audit,
    build_validation_terminal_evidence_plan,
)


def main() -> int:
    """Run the resumable SEC terminal-evidence workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delisting-audit-root", type=Path, default=None)
    parser.add_argument("--catalog-root", type=Path, default=None)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/sec/validation_terminal_evidence/targets"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/sec/validation_terminal_evidence")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/validation_terminal_evidence")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.request_interval < 0.20:
        parser.error("--request-interval must remain at or above 0.20 seconds")
    delisting = args.delisting_audit_root or _latest_root(
        Path("data/processed/validation_delisting"), "validation_delisting_report.json"
    )
    catalog = args.catalog_root or _latest_root(
        Path("data/processed/sec_universe"), "catalog_report.json"
    )
    plan = build_validation_terminal_evidence_plan(delisting, catalog, output_root=args.target_root)
    print(
        "VALIDATION TERMINAL TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"securities={plan.report['terminal_securities']} "
        f"filings={plan.report['filing_targets']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            parser.error("SEC_USER_AGENT must identify the project and include a real email")
        targets = _load_table(plan.output_root / "validation_terminal_filing_targets.parquet")
        client = SecClient(
            user_agent,
            request_interval_seconds=args.request_interval,
            timeout_seconds=60.0,
        )

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(
                f"VALIDATION TERMINAL progress={done}/{total} target={target_id[:12]}",
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
            "VALIDATION TERMINAL COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused}",
            flush=True,
        )
    audit = build_validation_terminal_evidence_audit(
        plan.output_root,
        delisting,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION TERMINAL AUDIT DONE "
        f"version={report['dataset_version']} "
        f"artifacts={report['filing_artifacts_preserved']}/{report['filing_targets']} "
        f"terminal={report['terminal_values_ready']}/{report['terminal_values_required']} "
        f"market_ready={report['resolved_market_outcomes']}/{report['event_components']} "
        f"remaining={report['remaining_event_exceptions']} "
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


def _load_table(path: Path) -> Any:
    import pandas as pd

    return pd.read_parquet(path)


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


if __name__ == "__main__":
    sys.exit(main())
