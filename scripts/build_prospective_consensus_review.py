"""Build a non-promoting analyst-membership counterfactual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prospective_consensus_review import build_prospective_consensus_review


def main() -> int:
    """Review preserved inputs without network access or changing admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/prospective_consensus_review")
    )
    args = parser.parse_args()
    result = build_prospective_consensus_review(
        args.quality_root, args.readiness_root, output_root=args.output_root
    )
    print(
        "CONSENSUS REVIEW DONE "
        f"events={result.report['events']} "
        f"review_required={result.report['review_required_events']} "
        f"quarantined={result.report['quarantined_events']} "
        f"released={result.report['quality_promotions']} "
        f"attribution_identified={result.report['membership_attribution_identified_events']} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
