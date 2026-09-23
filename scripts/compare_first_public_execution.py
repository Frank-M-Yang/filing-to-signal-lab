"""Compare frozen execution windows under the filing and earliest-disclosure clocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.first_public_execution import (
    EVENT_COLUMNS,
    compare_execution_clocks,
    summarize_execution_comparison,
)


def main() -> int:
    """Schedule-only comparison: loads no prices and computes no returns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--migration-config",
        type=Path,
        default=Path("docs/sec_execution_migration_v1.json"),
        help="Pins the frozen execution-plan root and its manifest SHA-256.",
    )
    parser.add_argument("--clock-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/first_public_execution")
    )
    args = parser.parse_args()

    config = json.loads(args.migration_config.read_bytes())
    root, digest = Path(config["execution_root"]), config["execution_manifest_sha256"]
    events = pd.read_parquet(
        _artifact(root, digest, "validation_execution_events.parquet"),
        columns=list(EVENT_COLUMNS),
    )
    sessions = pd.read_parquet(_artifact(root, digest, "validation_exchange_sessions.parquet"))
    clock_manifest = json.loads((args.clock_root / "output_manifest.json").read_bytes())
    clock_table = _artifact(
        args.clock_root,
        _sha256(args.clock_root / "output_manifest.json"),
        "first_public_clock.parquet",
    )
    clocks = pd.read_parquet(clock_table)

    comparison = compare_execution_clocks(events, clocks, sessions)
    summary = summarize_execution_comparison(comparison)
    report = {
        "summary": summary,
        "execution_root": root.as_posix(),
        "execution_manifest_sha256": digest,
        "clock_dataset_version": clock_manifest["dataset_version"],
        "prices_loaded": False,
        "returns_computed": False,
        "return_admission": False,
    }
    output = write_bundle(
        args.output_root,
        "first-public-execution-v1",
        {"comparison.json": _encode(_records(comparison)), "report.json": _encode(report)},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"FIRST PUBLIC EXECUTION COMPARISON DONE output={output}")
    return 0


def _sha256(path: Path) -> str:
    from sec_data.disclosure_audit import digest_file

    return digest_file(path)


if __name__ == "__main__":
    sys.exit(main())
