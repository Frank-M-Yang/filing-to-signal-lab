"""Freeze the no-return longitudinal filing expansion and collection plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_data.longitudinal_expansion import build_longitudinal_expansion


def main() -> int:
    """Build the immutable representative and feature-dense expansion bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness-root",
        type=Path,
        default=Path("data/processed/sec_price_readiness/sec-price-readiness-v6-9e06c6d4b248e770"),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d"),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(
            "data/processed/pilot_filing_features/pilot-filing-features-v1.1-b8c407284c843d77"
        ),
    )
    parser.add_argument(
        "--calendar-root",
        type=Path,
        default=Path(
            "data/processed/exchange_calendar/exchange-calendar-audit-v1-26f068f384d50b76"
        ),
    )
    parser.add_argument(
        "--security-actions-root",
        type=Path,
        default=Path(
            "data/processed/massive_security_actions/"
            "massive-security-actions-audit-v1-36067a60c713841b"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "data/processed/massive_reference/massive-reference-audit-v1-c5dd620357f207b1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/longitudinal_expansion"),
    )
    args = parser.parse_args()
    result = build_longitudinal_expansion(
        args.readiness_root,
        args.catalog_root,
        args.feature_root,
        args.calendar_root,
        args.security_actions_root,
        args.reference_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "LONGITUDINAL EXPANSION DONE "
        f"version={report['dataset_version']} "
        f"representative={report['representative_events']} "
        f"dense={report['feature_dense_events']} "
        f"complete={report['feature_dense_complete_events']} "
        f"coverage={report['feature_dense_complete_coverage']:.1%} "
        f"market_batches={report['planned_next_batch_issuers']} "
        f"return_ready={report['market_returns_ready']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
