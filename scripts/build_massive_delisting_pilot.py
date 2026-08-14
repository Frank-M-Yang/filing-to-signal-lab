"""Build and optionally collect the Massive listing-outcome pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_delisting_outcomes import (
    build_massive_delisting_audit,
    build_massive_delisting_target_plan,
    collect_massive_delisting_artifacts,
    load_massive_delisting_targets,
)
from sec_data.massive_reference import MassiveReferenceClient


def main() -> int:
    """Freeze listing-outcome targets, optionally collect, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--security-actions-root",
        type=Path,
        default=None,
        help="security-actions audit; defaults to latest 28/28 evidence-ready audit",
    )
    parser.add_argument(
        "--calendar-root",
        type=Path,
        default=None,
        help="calendar audit; defaults to the audit tied to security actions",
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/massive_delisting/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/massive_delisting"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/massive_delisting"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.1,
        help="seconds between calls; 12.1 respects the Basic 5-calls/minute limit",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="make authenticated API calls; otherwise audit preserved bytes only",
    )
    args = parser.parse_args()
    if args.collect and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic pilot")
    actions_root = args.security_actions_root or _latest_ready_security_actions(
        Path("data/processed/massive_security_actions")
    )
    calendar_root = args.calendar_root or _matching_calendar(
        Path("data/processed/exchange_calendar"), actions_root
    )
    plan = build_massive_delisting_target_plan(
        actions_root,
        calendar_root,
        output_root=args.target_output_root,
    )
    targets = load_massive_delisting_targets(plan.output_root)
    print(
        "DELISTING TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['targets']:,} "
        f"cutoff={plan.report['outcome_cutoff_date']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

        def show_progress(done: int, total: int, outcome_target_id: str) -> None:
            print(
                f"DELISTING COLLECT progress={done}/{total} target={outcome_target_id[:12]}",
                flush=True,
            )

        collection = collect_massive_delisting_artifacts(
            targets,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_progress,
        )
        print(
            "DELISTING COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    audit = build_massive_delisting_audit(
        plan.output_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "DELISTING AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_collected']:,}/{report['targets']:,} "
        f"active={report['active_at_outcome_cutoff']:,} "
        f"later_delisted={report['later_delisted_after_formal_exit']:,} "
        f"inside_window={report['delisted_on_or_before_formal_exit']:,} "
        f"outcome_ready={report['delisting_outcomes_ready']:,} "
        f"pilot_gate={report['pilot_delisting_outcome_gate']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


def _latest_ready_security_actions(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_security_actions_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("targets_with_complete_required_evidence", 0)) == int(
            report.get("targets", -1)
        ):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No 28/28 evidence-ready security-actions audit is available")
    return max(candidates, key=lambda item: item[0])[1]


def _matching_calendar(search_root: Path, actions_root: Path) -> Path:
    report = json.loads(
        (actions_root / "massive_security_actions_report.json").read_text(encoding="utf-8")
    )
    expected = str(report["calendar_manifest_sha256"])
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/exchange_calendar_report.json"):
        if _sha256_file(report_path.parent / "output_manifest.json") == expected:
            calendar_report = json.loads(report_path.read_text(encoding="utf-8"))
            candidates.append((str(calendar_report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No exchange-calendar audit matches security actions")
    return max(candidates, key=lambda item: item[0])[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
