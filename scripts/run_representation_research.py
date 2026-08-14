"""Run the real-data Level, Delta, and Innovation representation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from representation_research import build_representation_research


def main() -> int:
    """Build and persist the deterministic representation research artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-root",
        type=Path,
        default=Path("data/processed/global500_2026/core_asset_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/global500_2026/representation_research_v1"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_path = args.core_root / "quarterly_filing_events.csv"
    source_sha256 = _sha256_file(source_path)
    events = pd.read_csv(source_path, dtype={"company_id": "string", "sic": "string"})
    result = build_representation_research(events)
    artifacts = {
        "representation_panel.csv": result.panel,
        "common_representation_sample.csv": result.common_sample,
        "coverage_by_year.csv": result.coverage_by_year,
        "forecast_diagnostics_by_year.csv": result.forecast_by_year,
        "representation_correlations.csv": result.correlations,
        "representation_distributions.csv": result.distributions,
        "company_coverage.csv": result.company_coverage,
        "revision_diagnostics.csv": result.revision_diagnostics,
        "expectation_adjustment_diagnostics.csv": result.expectation_adjustment_diagnostics,
        "representation_rank_movements.csv": result.rank_movements,
        "forecast_loss_events.csv": result.forecast_loss_events,
        "forecast_diagnostics_by_group.csv": result.forecast_diagnostics_by_group,
        "forecast_bootstrap_intervals.csv": result.forecast_bootstrap_intervals,
        "sigma_diagnostics.csv": result.sigma_diagnostics,
        "universe_timepoint_audit.csv": result.universe_timepoint_audit,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(args.output_root / filename, index=False)

    report = {
        **result.report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
    }
    (args.output_root / "research_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_manifest(
        args.output_root,
        artifacts=artifacts,
        source_path=source_path,
        source_sha256=source_sha256,
        research_version=str(result.report["representation_research_version"]),
    )
    print(
        "DONE "
        f"level={result.report['level_events']} "
        f"delta={result.report['delta_events']} "
        f"raw_innovation={result.report['raw_innovation_events']} "
        f"standardized_innovation={result.report['standardized_innovation_events']} "
        f"common_companies={result.report['common_representation_companies']}",
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
        "representation_research_version": research_version,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "artifacts": files,
    }
    (output_root / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
