"""Build and optionally collect the Massive security-continuity/actions pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import load_catalog_table
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    build_massive_security_actions_target_plan,
    collect_massive_security_actions_artifacts,
    collect_massive_security_session_fallbacks,
    load_massive_security_actions_targets,
)


def main() -> int:
    """Freeze evidence targets, optionally collect a batch, and rebuild the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="reference audit; defaults to the one tied to the selected price audit",
    )
    parser.add_argument(
        "--price-root",
        type=Path,
        default=None,
        help="price audit; defaults to the latest fully collected 28-event pilot",
    )
    parser.add_argument(
        "--calendar-root",
        type=Path,
        default=None,
        help="calendar audit; defaults to the latest audit tied to the price audit",
    )
    parser.add_argument(
        "--target-output-root",
        type=Path,
        default=Path("data/market/massive_security_actions/targets"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/market/massive_security_actions"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/massive_security_actions"),
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
    parser.add_argument(
        "--collect-session-fallback",
        action="store_true",
        help=("for ticker-event 404 targets, verify identity on every official exchange session"),
    )
    args = parser.parse_args()
    if (args.collect or args.collect_session_fallback) and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic pilot")
    price_root = args.price_root or _latest_full_price_audit(Path("data/processed/massive_prices"))
    reference_root = args.reference_root or _matching_reference_audit(
        Path("data/processed/massive_reference"), price_root
    )
    calendar_root = args.calendar_root or _matching_calendar_audit(
        Path("data/processed/exchange_calendar"), price_root
    )
    plan = build_massive_security_actions_target_plan(
        reference_root,
        price_root,
        calendar_root,
        output_root=args.target_output_root,
    )
    targets = load_massive_security_actions_targets(plan.output_root)
    print(
        "SECURITY ACTION TARGETS DONE "
        f"version={plan.report['dataset_version']} "
        f"targets={plan.report['targets']:,} "
        f"queries={plan.report['planned_queries']:,} "
        f"output={plan.output_root}",
        flush=True,
    )
    client: MassiveReferenceClient | None = None
    if args.collect or args.collect_session_fallback:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with collection flags")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

    if args.collect:
        assert client is not None

        def show_progress(done: int, total: int, action_target_id: str) -> None:
            print(
                f"SECURITY ACTION COLLECT progress={done}/{total} target={action_target_id[:12]}",
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
            "SECURITY ACTION COLLECT DONE "
            f"selected={collection.selected_targets:,} "
            f"completed={collection.targets_completed:,} "
            f"failed={collection.targets_failed:,} "
            f"requests={collection.requests_made:,} "
            f"reused={collection.snapshots_reused:,} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
        if collection.blocking_reason == "authentication_failed":
            print("AUTH ACTION: Massive rejected MASSIVE_API_KEY (HTTP 401).", flush=True)
        elif collection.blocking_reason == "entitlement_denied":
            print(
                "ENTITLEMENT ACTION: authentication succeeded, but the plan denied one "
                "security-continuity endpoint (HTTP 403).",
                flush=True,
            )
    if args.collect_session_fallback:
        assert client is not None
        sessions = load_catalog_table(calendar_root, "historical_exchange_sessions.parquet")

        def show_fallback_progress(done: int, total: int, action_target_id: str) -> None:
            print(
                f"SESSION FALLBACK progress={done}/{total} target={action_target_id[:12]}",
                flush=True,
            )

        fallback = collect_massive_security_session_fallbacks(
            targets,
            sessions,
            client,
            raw_root=args.raw_root,
            limit=args.limit,
            progress=show_fallback_progress,
        )
        print(
            "SESSION FALLBACK DONE "
            f"selected={fallback.selected_targets:,} "
            f"completed={fallback.targets_completed:,} "
            f"failed={fallback.targets_failed:,} "
            f"requests={fallback.requests_made:,} "
            f"reused={fallback.snapshots_reused:,} "
            f"blocked={fallback.blocking_reason}",
            flush=True,
        )
    audit = build_massive_security_actions_audit(
        plan.output_root,
        calendar_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
    )
    report = audit.report
    print(
        "SECURITY ACTION AUDIT DONE "
        f"version={report['dataset_version']} "
        f"collected={report['targets_fully_collected']:,}/{report['targets']:,} "
        f"evidence_ready={report['targets_with_complete_required_evidence']:,} "
        f"session_fallbacks={report['session_grid_fallbacks_ready']:,} "
        f"identity_ready={report['historical_symbol_intervals_ready']:,} "
        f"actions_ready={report['corporate_action_windows_ready']:,} "
        f"splits={report['splits_observed']:,} "
        f"dividends={report['dividends_observed']:,} "
        f"return_gate={report['return_study_gate']} "
        f"output={audit.output_root}",
        flush=True,
    )
    return 0


def _latest_full_price_audit(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_price_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("targets_collected", 0)) == int(report.get("price_targets", -1)):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No fully collected Massive price audit is available")
    return max(candidates, key=lambda item: item[0])[1]


def _matching_reference_audit(search_root: Path, price_root: Path) -> Path:
    price_report = json.loads(
        (price_root / "massive_price_report.json").read_text(encoding="utf-8")
    )
    expected = str(price_report["reference_manifest_sha256"])
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_reference_report.json"):
        manifest = report_path.parent / "output_manifest.json"
        if _sha256_file(manifest) == expected:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No Massive reference audit matches the selected price audit")
    return max(candidates, key=lambda item: item[0])[1]


def _matching_calendar_audit(search_root: Path, price_root: Path) -> Path:
    expected = _sha256_file(price_root / "output_manifest.json")
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/exchange_calendar_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("price_manifest_sha256") == expected:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No exchange-calendar audit matches the selected price audit")
    return max(candidates, key=lambda item: item[0])[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
