"""Build the SPY benchmark pilot and the auditable 28-event return ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.event_returns import build_event_return_ledger
from sec_data.massive_benchmark import (
    build_massive_benchmark_audit,
    build_massive_benchmark_target_plan,
    collect_massive_benchmark_artifacts,
)
from sec_data.massive_reference import MassiveReferenceClient


def main() -> int:
    """Resolve tied inputs, optionally collect SPY, and build immutable returns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness-root",
        type=Path,
        default=None,
        help="price-readiness audit; defaults to latest passed pilot gate",
    )
    parser.add_argument(
        "--benchmark-target-output-root",
        type=Path,
        default=Path("data/market/massive_benchmark/targets"),
    )
    parser.add_argument(
        "--benchmark-raw-root",
        type=Path,
        default=Path("data/market/massive_benchmark"),
    )
    parser.add_argument(
        "--benchmark-output-root",
        type=Path,
        default=Path("data/processed/massive_benchmark"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/event_return_ledger"),
    )
    parser.add_argument(
        "--collect-benchmark",
        action="store_true",
        help="collect SPY raw prices, dividends, and splits",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.1,
        help="seconds between Massive calls; Basic requires at least 12 seconds",
    )
    args = parser.parse_args()
    if args.collect_benchmark and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic pilot")
    readiness_root = args.readiness_root or _latest_ready_audit(
        Path("data/processed/sec_price_readiness")
    )
    readiness_report = _read_json(readiness_root / "price_readiness_report.json")
    price_root = _root_by_manifest(
        Path("data/processed/massive_prices"),
        "massive_price_report.json",
        str(readiness_report["price_manifest_sha256"]),
    )
    calendar_root = _root_by_manifest(
        Path("data/processed/exchange_calendar"),
        "exchange_calendar_report.json",
        str(readiness_report["calendar_manifest_sha256"]),
    )
    actions_root = _root_by_manifest(
        Path("data/processed/massive_security_actions"),
        "massive_security_actions_report.json",
        str(readiness_report["security_actions_manifest_sha256"]),
    )
    delisting_root = _root_by_manifest(
        Path("data/processed/massive_delisting"),
        "massive_delisting_report.json",
        str(readiness_report["delisting_manifest_sha256"]),
    )
    plan = build_massive_benchmark_target_plan(
        calendar_root,
        output_root=args.benchmark_target_output_root,
    )
    print(
        "BENCHMARK TARGET DONE "
        f"version={plan.report['dataset_version']} "
        f"ticker={plan.report['benchmark_ticker']} "
        f"window={plan.report['window_start']}..{plan.report['window_end']} "
        f"output={plan.output_root}",
        flush=True,
    )
    if args.collect_benchmark:
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            parser.error("MASSIVE_API_KEY is required with --collect-benchmark")
        client = MassiveReferenceClient(
            api_key,
            request_interval_seconds=args.request_interval,
        )

        def show_progress(done: int, total: int, query_kind: str) -> None:
            print(f"BENCHMARK COLLECT progress={done}/{total} query={query_kind}", flush=True)

        collection = collect_massive_benchmark_artifacts(
            plan.output_root,
            client,
            raw_root=args.benchmark_raw_root,
            progress=show_progress,
        )
        print(
            "BENCHMARK COLLECT DONE "
            f"selected={collection.queries_selected} "
            f"completed={collection.queries_completed} "
            f"failed={collection.queries_failed} "
            f"requests={collection.requests_made} "
            f"reused={collection.snapshots_reused} "
            f"blocked={collection.blocking_reason}",
            flush=True,
        )
    benchmark = build_massive_benchmark_audit(
        plan.output_root,
        calendar_root,
        raw_root=args.benchmark_raw_root,
        output_root=args.benchmark_output_root,
    )
    print(
        "BENCHMARK AUDIT DONE "
        f"version={benchmark.report['dataset_version']} "
        f"queries={benchmark.report['queries_complete']}/"
        f"{benchmark.report['queries_required']} "
        f"bars={benchmark.report['parsed_daily_bars']} "
        f"dates={benchmark.report['required_event_dates_ready']}/"
        f"{benchmark.report['required_event_dates']} "
        f"ready={benchmark.report['benchmark_data_ready']} "
        f"output={benchmark.output_root}",
        flush=True,
    )
    ledger = build_event_return_ledger(
        readiness_root,
        price_root,
        calendar_root,
        actions_root,
        delisting_root,
        benchmark_root=benchmark.output_root,
        output_root=args.output_root,
    )
    report = ledger.report
    print(
        "EVENT RETURN LEDGER DONE "
        f"version={report['dataset_version']} "
        f"issuer={report['issuer_returns_ready']}/{report['events']} "
        f"market_excess={report['market_excess_returns_ready']}/{report['events']} "
        f"sector_excess={report['sector_excess_returns_ready']}/{report['events']} "
        f"full_gate={report['full_return_study_gate']} "
        f"output={ledger.output_root}",
        flush=True,
    )
    return 0


def _latest_ready_audit(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/price_readiness_report.json"):
        report = _read_json(report_path)
        if (
            report.get("pilot_event_return_gate")
            and int(report.get("return_study_eligible", 0)) > 0
        ):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No price-readiness audit has a passed pilot event-return gate")
    return max(candidates, key=lambda item: item[0])[1]


def _root_by_manifest(search_root: Path, report_name: str, expected_hash: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        if _sha256_file(report_path.parent / "output_manifest.json") == expected_hash:
            report = _read_json(report_path)
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} matches readiness manifest {expected_hash}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
