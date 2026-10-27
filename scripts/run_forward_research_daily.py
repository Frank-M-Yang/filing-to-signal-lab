"""Run the forward-only consensus-to-SEC research pipeline in dependency order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prospective_consensus_quality import (
    ProspectiveConsensusQualityResult,
    build_prospective_consensus_quality,
)
from prospective_consensus_review import (
    ConsensusReviewResult,
    build_prospective_consensus_review,
)
from prospective_disclosure_quality import (
    ProspectiveDisclosureQualityResult,
    build_prospective_disclosure_quality,
)
from prospective_event_state_machine import (
    ProspectiveEventStateMachineResult,
    build_prospective_event_state_machine,
)
from prospective_market_innovation_readiness import (
    ProspectiveMarketInnovationReadinessResult,
    build_prospective_market_innovation_readiness,
)
from sec_data.alpha_vantage_budget import AlphaVantageRequestBudget
from sec_data.alpha_vantage_consensus import (
    AlphaVantageConsensusClient,
    build_alpha_vantage_live_audit,
    build_alpha_vantage_live_plan,
    collect_alpha_vantage_live_snapshots,
)
from sec_data.client import SecClient
from sec_data.forward_actual_revenue import (
    build_forward_actual_audit,
    build_forward_actual_plan,
    collect_forward_actual_snapshots,
)
from sec_data.forward_consensus_panel import build_forward_consensus_panel
from sec_data.forward_disclosure_clock import (
    ForwardDisclosureClockResult,
    build_forward_disclosure_clock,
)
from sec_data.forward_filing_monitor import (
    build_forward_filing_monitor_audit,
    build_forward_filing_monitor_plan,
    collect_forward_filing_snapshots,
)
from sec_data.forward_innovation_ledger import build_forward_innovation_ledger
from sec_data.massive_reference import MassiveReferenceClient
from shadow_market_collection import (
    ShadowMarketAuditResult,
    build_shadow_market_evidence_audit,
    collect_shadow_market_evidence_queries,
)
from shadow_market_evidence import (
    ShadowMarketEvidenceResult,
    build_shadow_market_evidence_plan,
)
from shadow_order_ledger import ShadowOrderLedgerResult, build_shadow_order_ledger
from shadow_portfolio_allocator import (
    ShadowPortfolioAllocatorResult,
    build_shadow_portfolio_allocations,
)
from shadow_trading_policy import (
    ShadowTradingPolicyResult,
    build_shadow_trading_policy,
)


def main() -> int:
    """Capture optional live data and rebuild every downstream immutable layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-consensus", action="store_true")
    parser.add_argument("--collect-sec", action="store_true")
    parser.add_argument("--collect-market", action="store_true")
    parser.add_argument("--consensus-limit", type=int, default=20)
    parser.add_argument("--alpha-external-requests-today", type=int, default=0)
    parser.add_argument("--sec-limit", type=int, default=20)
    parser.add_argument("--consensus-request-interval", type=float, default=12.1)
    parser.add_argument("--sec-request-interval", type=float, default=0.25)
    parser.add_argument("--market-request-interval", type=float, default=12.1)
    parser.add_argument("--market-limit", type=int, default=20)
    parser.add_argument("--v3-preregistration-root", type=Path, default=None)
    parser.add_argument(
        "--core-asset-root",
        type=Path,
        default=Path("data/processed/global500_2026/core_asset_v1"),
    )
    parser.add_argument(
        "--run-report-root", type=Path, default=Path("data/processed/forward_daily_runs")
    )
    args = parser.parse_args()
    if args.collect_consensus and args.consensus_request_interval < 12.0:
        parser.error("--consensus-request-interval must be at least 12 seconds")
    if args.collect_sec and args.sec_request_interval <= 0.20:
        parser.error("--sec-request-interval must remain below five requests per second")
    alpha_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    massive_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    sec_user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if args.collect_consensus and not alpha_key:
        parser.error("ALPHA_VANTAGE_API_KEY is required with --collect-consensus")
    if args.collect_sec and not sec_user_agent:
        parser.error("SEC_USER_AGENT is required with --collect-sec")

    consensus_plan = build_alpha_vantage_live_plan(args.core_asset_root)
    if args.collect_consensus:
        alpha_client = AlphaVantageConsensusClient(
            alpha_key,
            request_interval_seconds=args.consensus_request_interval,
            request_budget=AlphaVantageRequestBudget(
                external_requests_today=args.alpha_external_requests_today,
            ),
        )
        consensus_collection = collect_alpha_vantage_live_snapshots(
            pd.read_parquet(consensus_plan.output_root / "alpha_vantage_live_targets.parquet"),
            alpha_client,
            raw_root=Path("data/market/alpha_vantage_consensus"),
            limit=args.consensus_limit,
            progress=_progress("DAILY CONSENSUS"),
        )
        print(
            "DAILY CONSENSUS COLLECT DONE "
            f"selected={consensus_collection.selected_targets} "
            f"completed={consensus_collection.completed_targets} "
            f"failed={consensus_collection.failed_targets} "
            f"requests={consensus_collection.requests_made} "
            f"blocked={consensus_collection.blocking_reason}",
            flush=True,
        )
    consensus_audit = build_alpha_vantage_live_audit(
        consensus_plan.output_root,
        raw_root=Path("data/market/alpha_vantage_consensus"),
        output_root=Path("data/processed/alpha_vantage_consensus"),
    )
    consensus_panel = build_forward_consensus_panel(consensus_audit.output_root)
    filing_plan = build_forward_filing_monitor_plan(consensus_panel.output_root)

    sec_client = (
        SecClient(sec_user_agent, request_interval_seconds=args.sec_request_interval)
        if args.collect_sec
        else None
    )
    if sec_client is not None:
        filing_collection = collect_forward_filing_snapshots(
            pd.read_parquet(filing_plan.output_root / "forward_filing_monitor_targets.parquet"),
            sec_client,
            raw_root=Path("data/sec/forward_filing_monitor"),
            limit=args.sec_limit,
            progress=_progress("DAILY SEC MONITOR"),
        )
        print(
            "DAILY SEC MONITOR COLLECT DONE "
            f"selected={filing_collection.selected_targets} "
            f"completed={filing_collection.completed_targets} "
            f"failed={filing_collection.failed_targets} "
            f"requests={filing_collection.requests_made}",
            flush=True,
        )
    filing_audit = build_forward_filing_monitor_audit(filing_plan.output_root)
    v3_preregistration_root = args.v3_preregistration_root or _frozen_v3_preregistration_root(
        Path("data/processed/prospective_market_innovation_preregistration")
    )
    v3_readiness: ProspectiveMarketInnovationReadinessResult | None = None
    if v3_preregistration_root is not None:
        v3_readiness = build_prospective_market_innovation_readiness(
            v3_preregistration_root,
            consensus_audit.output_root,
            filing_audit.output_root,
        )
    actual_plan = build_forward_actual_plan(filing_audit.output_root)
    actual_targets = pd.read_parquet(
        actual_plan.output_root / "forward_actual_revenue_targets.parquet"
    )
    if sec_client is not None and not actual_targets.empty:
        actual_collection = collect_forward_actual_snapshots(
            actual_targets,
            sec_client,
            raw_root=Path("data/sec/forward_actual_revenue"),
            limit=args.sec_limit,
            progress=_progress("DAILY SEC ACTUAL"),
        )
        print(
            "DAILY SEC ACTUAL COLLECT DONE "
            f"selected={actual_collection.selected_targets} "
            f"completed={actual_collection.completed_targets} "
            f"failed={actual_collection.failed_targets} "
            f"requests={actual_collection.requests_made}",
            flush=True,
        )
    actual_audit = build_forward_actual_audit(actual_plan.output_root)
    innovation = build_forward_innovation_ledger(
        consensus_panel.output_root,
        actual_audit.output_root,
    )
    consensus_quality: ProspectiveConsensusQualityResult | None = None
    disclosure_clock: ForwardDisclosureClockResult | None = None
    disclosure_quality: ProspectiveDisclosureQualityResult | None = None
    consensus_review: ConsensusReviewResult | None = None
    event_state_machine: ProspectiveEventStateMachineResult | None = None
    shadow_trading_policy: ShadowTradingPolicyResult | None = None
    shadow_order_ledger: ShadowOrderLedgerResult | None = None
    shadow_market_evidence: ShadowMarketEvidenceResult | None = None
    shadow_market_audit: ShadowMarketAuditResult | None = None
    shadow_portfolio_allocator: ShadowPortfolioAllocatorResult | None = None
    if v3_readiness is not None:
        assert v3_preregistration_root is not None
        frozen_policy_root = _frozen_shadow_trading_policy_root(
            Path("data/processed/shadow_trading_policy"), v3_preregistration_root
        )
        shadow_trading_policy = (
            build_shadow_trading_policy(v3_preregistration_root)
            if frozen_policy_root is None
            else ShadowTradingPolicyResult(
                frozen_policy_root.resolve(),
                _read_json(frozen_policy_root / "shadow_trading_policy_report.json"),
            )
        )
        consensus_quality = build_prospective_consensus_quality(v3_readiness.output_root)
        # v3.1 amendment view: re-times the same cohort at the earliest SEC disclosure
        # using preserved snapshot bytes only; the v3 tables above are not modified.
        disclosure_clock = build_forward_disclosure_clock(
            v3_readiness.output_root, filing_audit.output_root
        )
        disclosure_quality = build_prospective_disclosure_quality(
            v3_readiness.output_root,
            disclosure_clock.output_root,
            consensus_quality.output_root,
        )
        consensus_review = build_prospective_consensus_review(
            consensus_quality.output_root, v3_readiness.output_root
        )
        event_state_machine = build_prospective_event_state_machine(
            v3_readiness.output_root,
            actual_audit.output_root,
            innovation.output_root,
            consensus_quality.output_root,
        )
        shadow_order_ledger = build_shadow_order_ledger(
            event_state_machine.output_root,
            trading_policy_root=shadow_trading_policy.output_root,
        )
        shadow_market_evidence = build_shadow_market_evidence_plan(
            event_state_machine.output_root,
            shadow_trading_policy.output_root,
            args.core_asset_root,
        )
        market_queries = pd.read_parquet(
            shadow_market_evidence.output_root / "shadow_market_evidence_queries.parquet"
        )
        if args.collect_market:
            if not market_queries.empty and not massive_key:
                parser.error("MASSIVE_API_KEY is required when eligible market queries exist")
            market_client = (
                MassiveReferenceClient(
                    massive_key,
                    request_interval_seconds=args.market_request_interval,
                )
                if massive_key
                else None
            )
            market_collection = collect_shadow_market_evidence_queries(
                market_queries,
                market_client,
                limit=args.market_limit,
                progress=_progress("DAILY MARKET EVIDENCE"),
            )
            print(
                "DAILY MARKET EVIDENCE COLLECT DONE "
                f"selected={market_collection.selected_queries} "
                f"completed={market_collection.completed_queries} "
                f"failed={market_collection.failed_queries} "
                f"requests={market_collection.requests_made} "
                f"blocked={market_collection.blocking_reason}",
                flush=True,
            )
        shadow_market_audit = build_shadow_market_evidence_audit(shadow_market_evidence.output_root)
        shadow_portfolio_allocator = build_shadow_portfolio_allocations(
            shadow_order_ledger.output_root,
            shadow_market_audit.output_root,
            shadow_trading_policy.output_root,
        )
    report = {
        "run_contract": "forward-research-daily-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": {
            "consensus_requested": bool(args.collect_consensus),
            "sec_requested": bool(args.collect_sec),
            "market_requested": bool(args.collect_market),
            "credentials_persisted": False,
        },
        "consensus_audit": _stage(consensus_audit.output_root, consensus_audit.report),
        "consensus_panel": _stage(consensus_panel.output_root, consensus_panel.report),
        "filing_monitor": _stage(filing_audit.output_root, filing_audit.report),
        "prospective_market_innovation": (
            _stage(v3_readiness.output_root, v3_readiness.report)
            if v3_readiness is not None
            else {"status": "not_preregistered"}
        ),
        "shadow_market_evidence_audit": (
            _stage(shadow_market_audit.output_root, shadow_market_audit.report)
            if shadow_market_audit is not None
            else {"status": "not_preregistered"}
        ),
        "actual_revenue": _stage(actual_audit.output_root, actual_audit.report),
        "innovation_ledger": _stage(innovation.output_root, innovation.report),
        "prospective_consensus_quality": (
            _stage(consensus_quality.output_root, consensus_quality.report)
            if consensus_quality is not None
            else {"status": "not_preregistered"}
        ),
        "prospective_consensus_review": (
            _stage(consensus_review.output_root, consensus_review.report)
            if consensus_review is not None
            else {"status": "not_preregistered"}
        ),
        "forward_disclosure_clock": (
            _stage(disclosure_clock.output_root, disclosure_clock.report)
            if disclosure_clock is not None
            else {"status": "not_preregistered"}
        ),
        "prospective_disclosure_quality": (
            _stage(disclosure_quality.output_root, disclosure_quality.report)
            if disclosure_quality is not None
            else {"status": "not_preregistered"}
        ),
        "prospective_event_state_machine": (
            _stage(event_state_machine.output_root, event_state_machine.report)
            if event_state_machine is not None
            else {"status": "not_preregistered"}
        ),
        "shadow_trading_policy": (
            _stage(shadow_trading_policy.output_root, shadow_trading_policy.report)
            if shadow_trading_policy is not None
            else {"status": "not_preregistered"}
        ),
        "shadow_order_ledger": (
            _stage(shadow_order_ledger.output_root, shadow_order_ledger.report)
            if shadow_order_ledger is not None
            else {"status": "not_preregistered"}
        ),
        "shadow_market_evidence": (
            _stage(shadow_market_evidence.output_root, shadow_market_evidence.report)
            if shadow_market_evidence is not None
            else {"status": "not_preregistered"}
        ),
        "shadow_portfolio_allocator": (
            _stage(
                shadow_portfolio_allocator.output_root,
                shadow_portfolio_allocator.report,
            )
            if shadow_portfolio_allocator is not None
            else {"status": "not_preregistered"}
        ),
        "return_values_loaded": False,
        "model_fitting_performed": False,
    }
    report_path = _write_run_report(args.run_report_root, report)
    v3_status = (
        "not_preregistered"
        if v3_readiness is None
        else (
            f"{v3_readiness.report['registered_consensus_events']}/"
            f"{v3_readiness.report['registered_events']}"
        )
    )
    state_status = (
        "not_preregistered"
        if event_state_machine is None
        else ",".join(
            f"{state}:{count}"
            for state, count in event_state_machine.report["state_counts"].items()
        )
    )
    quality_status = (
        "not_preregistered"
        if consensus_quality is None
        else ",".join(
            f"{status}:{count}"
            for status, count in consensus_quality.report["quality_status_counts"].items()
        )
    )
    shadow_status = (
        "not_preregistered"
        if shadow_order_ledger is None
        else ",".join(
            f"{status}:{count}"
            for status, count in shadow_order_ledger.report["decision_status_counts"].items()
        )
    )
    shadow_order_count = (
        0 if shadow_order_ledger is None else shadow_order_ledger.report["shadow_orders"]
    )
    shadow_policy_status = (
        "not_preregistered"
        if shadow_trading_policy is None
        else f"frozen:{shadow_trading_policy.report['registered_events']}"
    )
    market_evidence_status = (
        "not_preregistered"
        if shadow_market_evidence is None
        else (
            f"accepted:{shadow_market_evidence.report['accepted_events']},"
            f"queries:{shadow_market_evidence.report['market_queries_planned']}"
        )
    )
    order_candidate_count = (
        0
        if shadow_portfolio_allocator is None
        else shadow_portfolio_allocator.report["order_candidates"]
    )
    market_capture_status = (
        "not_preregistered"
        if shadow_market_audit is None
        else (
            f"{shadow_market_audit.report['queries_with_captures']}/"
            f"{shadow_market_audit.report['planned_queries']}"
        )
    )
    review_promotions = (
        0 if consensus_review is None else consensus_review.report["quality_promotions"]
    )
    print(
        "FORWARD DAILY RUN DONE "
        f"capture_dates={consensus_panel.report['capture_dates']} "
        f"revisions={consensus_panel.report['revision_rows']} "
        f"filing_candidates={filing_audit.report['future_filing_candidates']} "
        f"actuals={actual_audit.report['total_quarterly_actuals_ready']} "
        f"innovations={innovation.report['raw_innovation_ready_events']} "
        f"v3_registry={v3_status} "
        f"consensus_quality={quality_status} "
        f"review_promotions={review_promotions} "
        f"event_states={state_status} "
        f"shadow_policy={shadow_policy_status} "
        f"shadow_decisions={shadow_status} "
        f"shadow_orders={shadow_order_count} "
        f"market_evidence={market_evidence_status} "
        f"market_captures={market_capture_status} "
        f"order_candidates={order_candidate_count} "
        f"feature_gate={innovation.report['feature_research_gate_passed']} "
        f"report={report_path}",
        flush=True,
    )
    return 0


def _progress(label: str) -> Any:
    def show(done: int, total: int, target_id: str) -> None:
        if done == 1 or done == total or done % 5 == 0:
            print(f"{label} progress={done}/{total} target={target_id[:12]}", flush=True)

    return show


def _stage(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_version": report["dataset_version"],
        "output_root": str(output_root.resolve()),
    }


def _frozen_v3_preregistration_root(search_root: Path) -> Path | None:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/prospective_market_innovation_report.json"):
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if bool(report.get("preregistration_gate_passed")):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _frozen_shadow_trading_policy_root(
    search_root: Path, prospective_preregistration_root: Path
) -> Path | None:
    expected_manifest = _sha256_file(prospective_preregistration_root / "output_manifest.json")
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/shadow_trading_policy_report.json"):
        report = _read_json(report_path)
        if (
            bool(report.get("policy_gate_passed"))
            and report.get("prospective_preregistration_manifest_sha256") == expected_manifest
        ):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _write_run_report(root: Path, report: dict[str, Any]) -> Path:
    payload = json.dumps(report, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    day = str(report["generated_at"])[:10]
    path = root / day / f"forward-daily-run-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"Immutable daily run report differs: {path}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return path.resolve()


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
