"""Build the outcome-blind market-data plan for one frozen validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.validation_market_plan import build_validation_market_plan


def main() -> int:
    """Resolve manifest-tied inputs and persist the validation market plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-plan-root", type=Path, default=None)
    parser.add_argument("--validation-feature-root", type=Path, default=None)
    parser.add_argument("--catalog-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_market_plan"),
    )
    args = parser.parse_args()
    feature_root = args.validation_feature_root or _latest_root(
        Path("data/processed/validation_expansion_features"),
        "validation_feature_report.json",
    )
    feature_report = _read_json(feature_root / "validation_feature_report.json")
    plan_root = args.validation_plan_root or _root_by_manifest(
        Path("data/processed/validation_expansion"),
        str(feature_report["plan_manifest_sha256"]),
    )
    catalog_root = args.catalog_root or _root_by_manifest(
        Path("data/processed/sec_universe"),
        str(feature_report["catalog_manifest_sha256"]),
    )
    result = build_validation_market_plan(
        plan_root,
        feature_root,
        catalog_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION MARKET PLAN DONE "
        f"version={report['dataset_version']} "
        f"applicable={report['unobserved_basic_window_history_eligible_events']} "
        f"price_candidates={report['unobserved_basic_window_complete_price_candidates']} "
        f"issuers={report['price_candidate_issuers']} "
        f"identity_anchors={report['identity_anchor_queries']} "
        f"sec_indexes={report['filing_index_evidence_targets']} "
        f"batches={report['issuer_collection_batches']} "
        f"outcomes_loaded={report['outcome_values_loaded']} "
        f"output={result.output_root}",
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
