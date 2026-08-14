"""Run and persist the source-audited v2 revision-robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from revision_robustness import build_revision_robustness
from revision_tail_audit import build_revision_tail_audit


def main() -> int:
    """Build deterministic revision-risk artifacts from the locked M2 sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tournament-root",
        type=Path,
        default=Path("data/processed/global500_2026/forecast_tournament_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/global500_2026/revision_robustness_v2"),
    )
    parser.add_argument("--audit-root", type=Path, default=Path("data/audit/revision_tail_v1"))
    parser.add_argument(
        "--v1-tail-events",
        type=Path,
        default=Path(
            "data/processed/global500_2026/revision_robustness_v1/revision_tail_events.csv"
        ),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_path = args.tournament_root / "tournament_common_sample.csv"
    source_sha256 = _sha256_file(source_path)
    events = pd.read_csv(source_path, dtype={"company_id": "string", "sic": "string"})
    reviews_path = args.audit_root / "manual_reviews.csv"
    filing_manifest_path = args.audit_root / "filing_document_manifest.csv"
    reviews = pd.read_csv(reviews_path)
    v1_tail = pd.read_csv(args.v1_tail_events, dtype={"company_id": "string"})
    filing_manifest = pd.read_csv(filing_manifest_path, dtype={"company_id": "string"})
    audit = build_revision_tail_audit(v1_tail, reviews, filing_manifest)
    result = build_revision_robustness(events, audit_reviews=reviews)
    artifacts = {
        "revision_robustness_panel.csv": result.panel,
        "primary_decisions.csv": result.primary_decisions,
        "confidence_calibration.csv": result.confidence_calibration,
        "diagnostics_by_year.csv": result.diagnostics_by_year,
        "revision_source_audit.csv": result.source_audit,
        "revision_tail_events.csv": result.tail_events,
        "reviewed_tail_events.csv": audit.reviewed_events,
        "tail_classification_summary.csv": audit.classification_summary,
        "event_time_gate_diagnostic.csv": audit.event_time_gate,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(args.output_root / filename, index=False)

    report = {
        **result.report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "tail_audit": audit.report,
    }
    (args.output_root / "revision_robustness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_manifest(
        args.output_root,
        artifacts=artifacts,
        source_path=source_path,
        source_sha256=source_sha256,
        audit_sources={
            str(reviews_path): _sha256_file(reviews_path),
            str(filing_manifest_path): _sha256_file(filing_manifest_path),
            str(args.v1_tail_events): _sha256_file(args.v1_tail_events),
        },
        research_version=str(result.report["revision_robustness_version"]),
    )
    print(
        "DONE "
        f"decisions={result.report['primary_decision_events']} "
        f"resolved_evaluation={result.report['primary_resolved_evaluation_events']} "
        f"median_buffer={result.report['primary_median_revision_at_risk']:.6f} "
        f"coverage={result.report['primary_empirical_coverage']:.3f}",
        flush=True,
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_root: Path,
    *,
    artifacts: dict[str, pd.DataFrame],
    source_path: Path,
    source_sha256: str,
    audit_sources: dict[str, str],
    research_version: str,
) -> None:
    files: dict[str, dict[str, Any]] = {}
    for filename, frame in sorted(artifacts.items()):
        path = output_root / filename
        files[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    manifest = {
        "revision_robustness_version": research_version,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "audit_sources": audit_sources,
        "artifacts": files,
    }
    (output_root / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
