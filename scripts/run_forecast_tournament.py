"""Run and persist the preregistered M0-M4 gross-margin forecast tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_tournament import build_forecast_tournament


def main() -> int:
    """Build deterministic v2 tournament artifacts from the v1 PIT panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("data/processed/global500_2026/representation_research_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/global500_2026/forecast_tournament_v2"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_path = args.representation_root / "representation_panel.csv"
    source_sha256 = _sha256_file(source_path)
    panel = pd.read_csv(source_path, dtype={"company_id": "string", "sic": "string"})
    result = build_forecast_tournament(panel)
    artifacts = {
        "tournament_panel.csv": result.panel,
        "tournament_common_sample.csv": result.common_sample,
        "model_metrics.csv": result.metrics,
        "model_metrics_by_year.csv": result.metrics_by_year,
        "clustered_bootstrap_intervals.csv": result.bootstrap_intervals,
        "calibration_diagnostics.csv": result.calibration,
        "eligibility_gate.csv": result.eligibility_gate,
        "representation_diagnostics.csv": result.representation_diagnostics,
        "forecast_loss_events.csv": result.loss_events,
        "subgroup_diagnostics.csv": result.subgroup_diagnostics,
        "feature_clock_audit.csv": result.feature_clock_audit,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(args.output_root / filename, index=False)

    report = {
        **result.report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
    }
    (args.output_root / "tournament_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_manifest(
        args.output_root,
        artifacts=artifacts,
        source_path=source_path,
        source_sha256=source_sha256,
        tournament_version=str(result.report["forecast_tournament_version"]),
    )
    print(
        "DONE "
        f"common_events={result.report['common_sample_events']} "
        f"common_companies={result.report['common_sample_companies']} "
        f"diagnostic_leader={result.report['diagnostic_mae_leader']} "
        f"selected={result.report['selected_model_id'] or 'NONE'}",
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
    tournament_version: str,
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
        "forecast_tournament_version": tournament_version,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "artifacts": files,
    }
    (output_root / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
