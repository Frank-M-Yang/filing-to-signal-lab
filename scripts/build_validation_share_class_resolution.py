"""Resolve validation multiclass issuers from filing-time SEC share counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.validation_share_class_resolution import (
    build_validation_share_class_resolution,
)


def main() -> int:
    """Build the price-outcome-independent primary-share-class identity asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-resolution-root", type=Path, default=None)
    parser.add_argument("--cover-audit-root", type=Path, default=None)
    parser.add_argument(
        "--raw-filing-root",
        type=Path,
        default=Path("data/sec/validation_filing_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/validation_identity_repair_resolution"),
    )
    args = parser.parse_args()
    identity = args.identity_resolution_root or _latest_multiclass_identity_root(
        Path("data/processed/validation_identity_repair_resolution")
    )
    cover = args.cover_audit_root or _latest_root(
        Path("data/processed/validation_identity_cover_repair"),
        "cover_repair_report.json",
    )
    result = build_validation_share_class_resolution(
        identity,
        cover,
        raw_filing_root=args.raw_filing_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "VALIDATION SHARE CLASS RESOLUTION DONE "
        f"version={report['dataset_version']} "
        f"issuers={report['issuers_with_stable_primary_class']}/"
        f"{report['multiclass_issuers']} "
        f"events_before={report['multiclass_events_before_resolution']} "
        f"events_after={report['multiclass_events_after_resolution']} "
        f"price_input={report['prior_price_pilot_used_as_selection_input']} "
        f"return_gate={report['return_outcome_read_authorized']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_multiclass_identity_root(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/identity_repair_report.json"):
        event_path = report_path.parent / "validation_event_identity_readiness_final.parquet"
        component_path = report_path.parent / "validation_identity_repair_components.parquet"
        if not event_path.is_file() or not component_path.is_file():
            continue
        events = pd.read_parquet(event_path, columns=["security_component_count"])
        if not events["security_component_count"].gt(1).any():
            continue
        report = _read_json(report_path)
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No unresolved multiclass identity resolution is available")
    return max(candidates, key=lambda item: item[0])[1]


def _latest_root(search_root: Path, report_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob(f"*/{report_name}"):
        report = _read_json(report_path)
        candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No {report_name} exists under {search_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


if __name__ == "__main__":
    sys.exit(main())
