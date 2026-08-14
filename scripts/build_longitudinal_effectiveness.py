"""Build the longitudinal return ledger and pre-registered effectiveness diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.longitudinal_event_returns import build_longitudinal_event_returns
from sec_data.preliminary_effectiveness_diagnostics import build_effectiveness_diagnostics


def main() -> int:
    """Resolve tied audits, then build returns and the frozen statistical verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-plan-root",
        type=Path,
        default=Path(
            "data/market/longitudinal_study/targets/longitudinal-market-plan-v1-54c0331b5073c8ad"
        ),
    )
    parser.add_argument(
        "--preregistration-root",
        type=Path,
        default=Path(
            "data/processed/preliminary_effectiveness/"
            "preliminary-effectiveness-preregistration-v1-97df4b07e54937e8"
        ),
    )
    parser.add_argument("--price-root", type=Path, default=None)
    parser.add_argument("--action-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--transition-repair-root",
        type=Path,
        default=None,
        help="passed OB-to-TEAD repair; defaults to the latest passed repair when available",
    )
    parser.add_argument(
        "--delisting-root",
        type=Path,
        default=Path(
            "data/processed/massive_delisting/massive-delisting-audit-v1-6f2e240cfb707464"
        ),
    )
    parser.add_argument(
        "--ledger-output-root",
        type=Path,
        default=Path("data/processed/longitudinal_event_returns"),
    )
    parser.add_argument(
        "--diagnostic-output-root",
        type=Path,
        default=Path("data/processed/preliminary_effectiveness_diagnostics"),
    )
    args = parser.parse_args()
    price_root = args.price_root or _latest_tied_audit(
        Path("data/processed/longitudinal_prices"),
        "massive_price_report.json",
        args.market_plan_root / "prices",
    )
    action_root = args.action_root or _latest_tied_audit(
        Path("data/processed/longitudinal_security_actions"),
        "massive_security_actions_report.json",
        args.market_plan_root / "actions",
    )
    benchmark_root = args.benchmark_root or _latest_tied_audit(
        Path("data/processed/longitudinal_benchmark"),
        "massive_benchmark_report.json",
        args.market_plan_root / "benchmark",
    )
    transition_repair_root = args.transition_repair_root or _latest_passed_transition_repair(
        Path("data/processed/ticker_transition_repair")
    )
    ledger = build_longitudinal_event_returns(
        args.market_plan_root,
        args.preregistration_root,
        price_root,
        action_root,
        benchmark_root,
        args.delisting_root,
        transition_repair_root=transition_repair_root,
        output_root=args.ledger_output_root,
    )
    print(
        "LONGITUDINAL RETURN LEDGER DONE "
        f"version={ledger.report['dataset_version']} "
        f"primary_ready={ledger.report['primary_complete_feature_return_events']}/"
        f"{ledger.report['minimum_complete_feature_return_events']} "
        f"gate={ledger.report['primary_analysis_gate']} "
        f"transition_repair={ledger.report['ticker_transition_repair_applied']} "
        f"output={ledger.output_root}",
        flush=True,
    )
    diagnostics = build_effectiveness_diagnostics(
        ledger.output_root,
        args.preregistration_root,
        output_root=args.diagnostic_output_root,
    )
    print(
        "EFFECTIVENESS DIAGNOSTIC DONE "
        f"version={diagnostics.report['dataset_version']} "
        f"events={diagnostics.report['primary_analysis_events']} "
        f"issuers={diagnostics.report['primary_analysis_issuers']} "
        f"gate={diagnostics.report['primary_analysis_gate']} "
        f"verdict={diagnostics.report['verdict']} "
        f"output={diagnostics.output_root}",
        flush=True,
    )
    return 0


def _latest_tied_audit(search_root: Path, report_name: str, plan_root: Path) -> Path:
    expected = _sha256_file(plan_root / "output_manifest.json")
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("target_plan_manifest_sha256") == expected:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} matches plan {plan_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _latest_passed_transition_repair(search_root: Path) -> Path | None:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/ticker_transition_repair_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("transition_gate"):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
