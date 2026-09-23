"""Freeze, collect or replay the isolated historical market-risk pilot (no trading)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from typing import Any

from market_stress import StressConfig, forecast_panel, make_panel, score_forecasts
from market_stress_sources import (
    collect_sources,
    freeze_protocol,
    json_bytes,
    load_sources,
    verify_bundle,
)
from scripts.build_sec_disclosure_audit import write_bundle


def run_pilot(protocol: Path, snapshot: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Run one frozen historical specification, with lineage and no trading outputs."""
    files = verify_bundle(protocol)
    plan = json.loads(files["protocol.json"])
    if (
        plan["variance_windows"] != [5, 20, 60]
        or plan["correlation_windows"] != [20, 60]
        or plan["live_availability_certified"]
        or plan["trading_admission"]
        or plan["restore_retired_research"]
        or plan["untouched_holdout"]
        or plan["liquidity_tested"]
    ):
        raise ValueError("Unsupported or weakened research contract")
    config = StressConfig(**{name: plan[name] for name in StressConfig.__dataclass_fields__})
    returns, evidence = load_sources(snapshot, protocol)
    panel = make_panel(returns, config)
    predictions, fits = forecast_panel(panel, config)
    ledger, summary = score_forecasts(panel, predictions, config)
    summary["protocol_sha256"] = hashlib.sha256(files["protocol.json"]).hexdigest()
    summary["source_manifest_sha256"] = hashlib.sha256(
        (snapshot / "output_manifest.json").read_bytes()
    ).hexdigest()
    code_root = Path(__file__).resolve().parent.parent
    evidence["implementation_sha256"] = {
        name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
        for name in (
            "market_stress.py",
            "market_stress_sources.py",
            "scripts/build_market_stress_pilot.py",
        )
    }
    evidence["runtime_versions"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "httpx")
    }
    outputs = {
        "summary.json": json_bytes(summary),
        "protocol.json": files["protocol.json"],
        "freeze_record.json": files["freeze_record.json"],
        "source_evidence.json": json_bytes(evidence),
        "monthly_fits.json": json_bytes(fits),
    }
    for name, frame in (("panel", panel), ("forecasts", predictions), ("scored_ledger", ledger)):
        buffer = io.BytesIO()
        frame.reset_index().to_parquet(buffer, index=False)
        outputs[f"{name}.parquet"] = buffer.getvalue()
    return write_bundle(output_root, "market-stress-pilot-v1", outputs), summary


def main() -> None:
    """Freeze before collection; replay will always require an explicit snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/market_stress_pilot_v1.json"))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--snapshot", type=Path, help="Exact cached source bundle to replay offline"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/market/market_stress"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/market_stress"))
    args = parser.parse_args()
    protocol = freeze_protocol(args.config, args.data_root / "protocols")
    print(f"STRESS PROTOCOL FROZEN output={protocol}", flush=True)
    if args.collect:
        snapshot = collect_sources(args.data_root / "sources", protocol)
        print(f"STRESS SOURCES COLLECTED output={snapshot}", flush=True)
    if args.snapshot:
        root, summary = run_pilot(protocol, args.snapshot, args.output_root)
        print(
            f"STRESS PILOT DONE origins={summary['scored_origins']} "
            f"verdict={summary['verdict']} "
            f"relative_qlike_change={summary['slices']['all']['relative_qlike_change']:.2%} "
            f"trading=False output={root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
