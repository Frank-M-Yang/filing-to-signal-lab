"""Build separated representative and stress frames from a complete SEC catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sec_data.universe_design import build_universe_design


def main() -> int:
    """Persist deterministic sampling candidates and freeze-gate diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--size-root",
        type=Path,
        default=Path("data/processed/sec_size_features/event-time-size-v1-9920331a12142a9e"),
    )
    parser.add_argument(
        "--cover-root",
        type=Path,
        default=Path(
            "data/processed/sec_filing_cover_history/filing-cover-history-v1-75eb202210a58216"
        ),
    )
    parser.add_argument(
        "--probability-sample-root",
        type=Path,
        default=Path(
            "data/processed/sec_probability_sample/sec-probability-sample-plan-v1-bd8b453b0e9e3a28"
        ),
    )
    parser.add_argument(
        "--scope-audit-root",
        type=Path,
        default=Path(
            "data/processed/sec_probability_scope/sec-equity-scope-audit-v1-91be9ca9abf87966"
        ),
    )
    parser.add_argument(
        "--price-readiness-root",
        type=Path,
        default=None,
        help="price-readiness asset; defaults to latest version matching scope audit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/sec_universe_design"),
    )
    args = parser.parse_args()
    price_readiness_root = args.price_readiness_root or _latest_matching_price_readiness(
        Path("data/processed/sec_price_readiness"), args.scope_audit_root
    )
    result = build_universe_design(
        args.catalog_root,
        size_root=args.size_root,
        cover_root=args.cover_root,
        probability_sample_root=args.probability_sample_root,
        scope_audit_root=args.scope_audit_root,
        price_readiness_root=price_readiness_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "DONE "
        f"version={report['dataset_version']} "
        f"representative_issuer_years={report['representative_candidate_issuer_years']:,} "
        f"size_coverage={report['event_time_size_coverage']:.1%} "
        f"filing_index_ready={report['filing_index_ready_issuer_years']:,} "
        f"stress_ciks={report['stress_candidate_ciks']:,} "
        f"sample_frozen={report['representative_sample_frozen']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_matching_price_readiness(search_root: Path, scope_root: Path) -> Path:
    scope_report = json.loads((scope_root / "scope_audit_report.json").read_text(encoding="utf-8"))
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/price_readiness_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("scope_version") == scope_report.get("dataset_version"):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(
            "No price-readiness asset matches the current scope; run "
            "scripts.build_sec_price_readiness first"
        )
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
