"""Re-time the frozen prospective cohort's consensus quality at the earliest disclosure (v3.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prospective_disclosure_quality import build_prospective_disclosure_quality
from scripts.build_forward_disclosure_clock import _latest_root


def main() -> int:
    """Offline amendment view; v3 bundles are read, never modified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, default=None)
    parser.add_argument("--clock-root", type=Path, default=None)
    parser.add_argument("--v3-quality-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/prospective_disclosure_quality"),
    )
    args = parser.parse_args()
    processed = Path("data/processed")
    result = build_prospective_disclosure_quality(
        args.readiness_root
        or _latest_root(
            processed / "prospective_market_innovation_readiness",
            "prospective_market_innovation_readiness_report.json",
        ),
        args.clock_root
        or _latest_root(
            processed / "forward_disclosure_clock", "forward_disclosure_clock_report.json"
        ),
        args.v3_quality_root
        or _latest_root(
            processed / "prospective_consensus_quality", "prospective_consensus_quality_report.json"
        ),
        output_root=args.output_root,
    )
    report = {
        key: value
        for key, value in result.report.items()
        if key not in {"artifacts", "input_manifest_sha256", "implementation_sha256"}
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"PROSPECTIVE DISCLOSURE QUALITY DONE output={result.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
