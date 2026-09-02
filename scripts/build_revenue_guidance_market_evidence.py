"""Freeze guidance market gaps and optionally collect historical identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from sec_data.massive_reference import (
    MassiveReferenceClient,
    build_massive_reference_audit,
    collect_massive_reference_artifacts,
    load_massive_reference_targets,
)
from sec_data.revenue_guidance_market_plan import build_revenue_guidance_market_plan
from sec_data.validation_security_identity import (
    build_validation_security_identity_audit,
    build_validation_security_identity_plan,
)


def main() -> int:
    """Build the no-return plan, identity targets, and optional identity audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-root", type=Path, default=None)
    parser.add_argument("--filing-feature-root", type=Path, default=None)
    parser.add_argument("--authorization-root", type=Path, default=None)
    parser.add_argument("--filing-catalog-root", type=Path, default=None)
    parser.add_argument(
        "--market-plan-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_market_plan"),
    )
    parser.add_argument(
        "--identity-target-root",
        type=Path,
        default=Path("data/market/revenue_guidance_security_identity/targets"),
    )
    parser.add_argument(
        "--identity-raw-root",
        type=Path,
        default=Path("data/market/revenue_guidance_security_identity"),
    )
    parser.add_argument(
        "--reference-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_security_reference"),
    )
    parser.add_argument(
        "--identity-output-root",
        type=Path,
        default=Path("data/processed/revenue_guidance_security_identity"),
    )
    parser.add_argument("--collect-identity", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--request-interval", type=float, default=12.1)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.collect_identity and args.request_interval < 12.0:
        parser.error("--request-interval must be at least 12 seconds for the Basic plan")

    tournament_root = args.tournament_root or _latest_root(
        Path("data/processed/revenue_guidance_tournament"),
        "revenue_guidance_tournament_report.json",
    )
    tournament_report = _read_json(tournament_root / "revenue_guidance_tournament_report.json")
    feature_root = args.filing_feature_root or _root_by_manifest(
        Path("data/processed/validation_expansion_features"),
        str(tournament_report["input_manifest_sha256"]["filing_features"]),
    )
    feature_report = _read_json(feature_root / "validation_feature_report.json")
    authorization_root = args.authorization_root or _latest_root(
        Path("data/processed/validation_return_gate"), "validation_return_gate_report.json"
    )
    catalog_root = args.filing_catalog_root or _root_by_manifest(
        Path("data/processed/sec_universe"), str(feature_report["catalog_manifest_sha256"])
    )
    market = build_revenue_guidance_market_plan(
        tournament_root,
        feature_root,
        authorization_root,
        catalog_root,
        output_root=args.market_plan_output_root,
    )
    identity_plan = build_validation_security_identity_plan(
        market.output_root, output_root=args.identity_target_root
    )
    print(
        "REVENUE GUIDANCE MARKET PLAN DONE "
        f"version={market.report['dataset_version']} "
        f"existing={market.report['existing_authorized_return_events']}/"
        f"{market.report['forecast_events']} "
        f"missing={market.report['market_evidence_collection_events']} "
        f"issuers={market.report['market_evidence_collection_issuers']} "
        f"identity_anchors={identity_plan.report['identity_anchor_targets']} "
        f"output={market.output_root}",
        flush=True,
    )
    if not args.collect_identity:
        print(
            "REVENUE GUIDANCE IDENTITY PENDING "
            "set MASSIVE_API_KEY and rerun with --collect-identity",
            flush=True,
        )
        return 0

    api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not api_key:
        parser.error("MASSIVE_API_KEY is required with --collect-identity")
    targets = load_massive_reference_targets(identity_plan.output_root)
    client = MassiveReferenceClient(api_key, request_interval_seconds=args.request_interval)

    def show_progress(done: int, total: int, target_id: str) -> None:
        print(
            f"REVENUE GUIDANCE IDENTITY progress={done}/{total} target={target_id[:12]}",
            flush=True,
        )

    collection = collect_massive_reference_artifacts(
        targets,
        client,
        raw_root=args.identity_raw_root,
        limit=args.limit,
        progress=show_progress,
    )
    print(
        "REVENUE GUIDANCE IDENTITY COLLECT DONE "
        f"selected={collection.selected_targets} "
        f"completed={collection.targets_completed} "
        f"failed={collection.targets_failed} "
        f"requests={collection.requests_made} "
        f"reused={collection.snapshots_reused} "
        f"blocked={collection.blocking_reason}",
        flush=True,
    )
    reference = build_massive_reference_audit(
        identity_plan.output_root,
        raw_root=args.identity_raw_root,
        output_root=args.reference_output_root,
    )
    identity = build_validation_security_identity_audit(
        market.output_root,
        reference.output_root,
        output_root=args.identity_output_root,
    )
    print(
        "REVENUE GUIDANCE IDENTITY AUDIT DONE "
        f"collected={identity.report['provider_queries_complete']}/"
        f"{identity.report['identity_anchor_targets']} "
        f"event_ready={identity.report['price_candidate_events_identity_ready']}/"
        f"{identity.report['price_candidate_events']} "
        f"output={identity.output_root}",
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


def _root_by_manifest(search_root: Path, expected_hash: str) -> Path:
    candidates = [
        path.parent
        for path in search_root.glob("*/output_manifest.json")
        if _sha256_file(path) == expected_hash
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one root for manifest {expected_hash}; found {len(candidates)}"
        )
    return candidates[0]


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
