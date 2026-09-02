"""Build the first real-return diagnostic and strict feature-join audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sec_data.event_return_diagnostics import build_event_return_diagnostics


def main() -> int:
    """Resolve tied inputs and persist the immutable diagnostic bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="event-return ledger; defaults to the latest complete market-excess pilot",
    )
    parser.add_argument(
        "--readiness-root",
        type=Path,
        default=None,
        help="price-readiness audit; defaults to the ledger-tied manifest",
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=None,
        help="feature bundle; defaults to the latest complete pilot filing-feature asset",
    )
    parser.add_argument(
        "--representation-artifact",
        default="target_filing_features.parquet",
        help="CSV or Parquet feature panel filename within --representation-root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/event_return_diagnostics"),
    )
    args = parser.parse_args()
    ledger_root = args.ledger_root or _latest_complete_ledger(
        Path("data/processed/event_return_ledger")
    )
    ledger_report = _read_json(ledger_root / "event_return_report.json")
    readiness_root = args.readiness_root or _root_by_manifest(
        Path("data/processed/sec_price_readiness"),
        str(ledger_report["readiness_manifest_sha256"]),
    )
    representation_root = args.representation_root or _latest_pilot_feature(
        Path("data/processed/pilot_filing_features")
    )
    result = build_event_return_diagnostics(
        ledger_root,
        readiness_root,
        representation_root,
        output_root=args.output_root,
        representation_artifact_name=args.representation_artifact,
    )
    report = result.report
    applicability_denominator = report.get("representation_applicable_events", report["events"])
    print(
        "EVENT RETURN DIAGNOSTIC DONE "
        f"version={report['dataset_version']} "
        f"events={report['events']} "
        f"later_delisted={report['later_delisted_after_exit']} "
        f"survivor_inflation_pp={report['survivors_only_mean_inflation_pp']:+.4f} "
        f"ldi_complete={report['representation_complete_events']}/{applicability_denominator} "
        f"ic_allowed={report['exploratory_ic_allowed']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_complete_ledger(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/event_return_report.json"):
        report = _read_json(report_path)
        if report.get("pilot_market_excess_return_gate"):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No event-return ledger has complete market-excess returns")
    return max(candidates, key=lambda item: item[0])[1]


def _root_by_manifest(search_root: Path, expected_hash: str) -> Path:
    candidates: list[Path] = []
    for manifest_path in search_root.glob("*/output_manifest.json"):
        if _sha256_file(manifest_path) == expected_hash:
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        count = len(candidates)
        raise FileNotFoundError(
            f"Expected one price-readiness root for manifest {expected_hash}; found {count}"
        )
    return candidates[0]


def _latest_pilot_feature(search_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/pilot_filing_feature_report.json"):
        report = _read_json(report_path)
        if report.get("target_accession_matches") == report.get("target_events"):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError("No complete pilot filing-feature bundle exists")
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
