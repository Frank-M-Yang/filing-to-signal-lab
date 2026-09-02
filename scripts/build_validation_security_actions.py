"""Collect formal company-action evidence for the validation security spans."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import load_catalog_table
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    collect_massive_security_actions_artifacts,
    collect_massive_security_session_fallbacks,
    load_massive_security_actions_targets,
)
from sec_data.validation_security_actions import build_validation_security_actions_plan


def main() -> int:
    """Freeze, optionally collect, and audit the formal validation action batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-evidence-plan-root", type=Path, default=None)
    parser.add_argument("--execution-plan-root", type=Path, default=None)
    parser.add_argument("--calendar-adapter-root", type=Path, default=None)
    parser.add_argument("--return-gate-root", type=Path, default=None)
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/validation_security_actions/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/validation_security_actions/raw_store"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_security_actions"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--collect-session-fallback", action="store_true")
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if (args.collect or args.collect_session_fallback) and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for Massive Basic")
    evidence = args.market_evidence_plan_root or _latest_root(
        Path("data/market/validation_market_evidence/targets"),
        "validation_market_evidence_plan_report.json",
    )
    execution = args.execution_plan_root or _latest_root(
        Path("data/processed/validation_execution_plan"),
        "validation_execution_plan_report.json",
    )
    calendar = args.calendar_adapter_root or _latest_root(
        Path("data/processed/validation_calendar_adapter"), "exchange_calendar_report.json"
    )
    return_gate = args.return_gate_root or _latest_root(
        Path("data/processed/validation_return_gate"), "validation_return_gate_report.json"
    )
    plan = build_validation_security_actions_plan(
        evidence,
        execution,
        calendar,
        output_root=args.target_output_root,
        return_gate_root=return_gate,
    )
    targets = load_massive_security_actions_targets(plan.output_root)
    print(
        "VALIDATION SECURITY ACTION TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['targets']} "
        f"queries={plan.report['planned_queries']} "
        f"output={plan.output_root}",
        flush=True,
    )
    client: MassiveReferenceClient | None = None
    if args.collect or args.collect_session_fallback:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with collection flags")
        client = MassiveReferenceClient(api_key, request_interval_seconds=args.request_interval)
    if args.collect:
        assert client is not None

        def show_progress(done: int, total: int, target_id: str) -> None:
            print(
                f"VALIDATION SECURITY ACTION progress={done}/{total} target={target_id[:12]}",
                flush=True,
            )

        collection = collect_massive_security_actions_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "VALIDATION SECURITY ACTION COLLECT DONE "
            f"selected={collection.selected_targets} "
            f"completed={collection.targets_completed} "
            f"failed={collection.targets_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    if args.collect_session_fallback:
        assert client is not None
        sessions = load_catalog_table(calendar, "historical_exchange_sessions.parquet")

        def show_fallback(done: int, total: int, target_id: str) -> None:
            print(
                f"VALIDATION ACTION FALLBACK progress={done}/{total} target={target_id[:12]}",
                flush=True,
            )

        fallback = collect_massive_security_session_fallbacks(
            targets,
            sessions,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_fallback,
        )
        print(
            "VALIDATION ACTION FALLBACK DONE "
            f"selected={fallback.selected_targets} "
            f"completed={fallback.targets_completed} "
            f"failed={fallback.targets_failed} "
            f"requests={fallback.requests_made} "
            f"reused={fallback.snapshots_reused} "
            f"blocked={fallback.blocking_reason}",
            flush=True,
        )
    audit = build_massive_security_actions_audit(
        plan.output_root,
        calendar,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "VALIDATION SECURITY ACTION AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_fully_collected']}/{report['targets']} "
        f"evidence={report['targets_with_complete_required_evidence']} "
        f"actions={report['corporate_action_windows_ready']} "
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
