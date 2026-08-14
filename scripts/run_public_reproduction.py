"""Reproduce every research layer that is distributable in the public repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backtest import BacktestConfig, compare_standard_scenarios
from data_generator import SimulatedDataSource
from sec_data.apple_case import CASE_FACTS_PATH, CASE_METADATA_PATH, load_apple_amendment_case

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SNAPSHOT_PATH = REPOSITORY_ROOT / "data" / "examples" / "research_preview_summary.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "output" / "public_reproduction"


def build_public_reproduction(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Recompute public evidence and write a deterministic machine-readable summary.

    Simulation results and the Apple filing case are recomputed from public inputs.
    The empirical longitudinal result is deliberately copied from its frozen aggregate
    snapshot because provider event-level prices are not licensed for redistribution.
    """
    data = SimulatedDataSource().load()
    scenarios = compare_standard_scenarios(data, "sue", BacktestConfig())
    simulation: dict[str, Any] = {
        "evidence_type": "deterministic_simulation_recomputed",
        "seed": data.metadata["seed"],
        "scenario_metrics": {name: asdict(result.metrics) for name, result in scenarios.items()},
    }

    apple_case = load_apple_amendment_case()
    fiscal_2009 = apple_case.gross_margin.loc[apple_case.gross_margin["fiscal_year"].eq(2009)].iloc[
        0
    ]
    apple = {
        "evidence_type": "real_sec_case_recomputed_from_curated_slice",
        "curated_fact_count": len(apple_case.facts),
        "comparable_fact_count": int(apple_case.metadata["comparable_fact_count"]),
        "changed_fact_count": int(apple_case.metadata["changed_fact_count"]),
        "information_delay_days": apple_case.information_delay_days,
        "fiscal_2009": {
            "original_margin": float(fiscal_2009["original_margin"]),
            "amendment_margin": float(fiscal_2009["amendment_margin"]),
            "original_yoy_change": float(fiscal_2009["original_yoy_change"]),
            "amendment_yoy_change": float(fiscal_2009["amendment_yoy_change"]),
            "signal_inflation": float(fiscal_2009["signal_inflation"]),
        },
        "source_sha256": {
            CASE_FACTS_PATH.name: _sha256(CASE_FACTS_PATH),
            CASE_METADATA_PATH.name: _sha256(CASE_METADATA_PATH),
        },
    }

    frozen_snapshot = _read_json_object(PUBLIC_SNAPSHOT_PATH)
    longitudinal = frozen_snapshot["longitudinal_effectiveness"]
    if not isinstance(longitudinal, dict):
        raise ValueError("Public longitudinal effectiveness summary must be an object")
    empirical = {
        "evidence_type": "frozen_aggregate_snapshot_not_recomputed",
        "snapshot_schema_version": frozen_snapshot["schema_version"],
        "snapshot_sha256": _sha256(PUBLIC_SNAPSHOT_PATH),
        "source_version": frozen_snapshot["generated_from"]["longitudinal_effectiveness"],
        "complete_events": longitudinal["complete_events"],
        "issuers": longitudinal["issuers"],
        "innovation_ic": longitudinal["representation_ic"]["Innovation std"],
        "innovation_ci": longitudinal["innovation_ci"],
        "innovation_minus_delta": longitudinal["innovation_minus_delta"],
        "innovation_minus_delta_p": longitudinal["innovation_minus_delta_p"],
        "survivor_only_inflation_pp": longitudinal["survivor_only_inflation_pp"],
        "verdict": longitudinal["verdict"],
    }

    biased_return = simulation["scenario_metrics"]["原始回测（全部偏差）"]["annual_return"]
    corrected_return = simulation["scenario_metrics"]["修正全部偏差后"]["annual_return"]
    document = {
        "schema_version": "public-reproduction-v1",
        "evidence_boundary": (
            "Simulation and the curated Apple case are recomputed offline. The 49-event "
            "market result is verified only as a frozen aggregate snapshot; independently "
            "recomputing it requires newly collected point-in-time provider data."
        ),
        "simulation": simulation,
        "apple_2009": apple,
        "empirical_longitudinal": empirical,
        "checks": {
            "simulation_bias_gap_pass": (
                0.10 <= float(biased_return) <= 0.20
                and float(corrected_return) < 0.03
                and float(biased_return) - float(corrected_return) >= 0.08
            ),
            "apple_signal_inflation_reproduced": abs(
                float(fiscal_2009["signal_inflation"]) - 0.032903320
            )
            < 1e-9,
            "empirical_effectiveness_not_overclaimed": (
                longitudinal["verdict"] == "effectiveness_not_established"
            ),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "public_reproduction_summary.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON document and require a top-level object."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return document


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a public source artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for the deterministic reproduction summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the public offline reproduction command."""
    args = _parse_args()
    output_path = build_public_reproduction(args.output_root)
    document = _read_json_object(output_path)
    checks = document["checks"]
    if not isinstance(checks, dict) or not all(checks.values()):
        raise RuntimeError(f"Public reproduction check failed: {checks}")
    print(f"PUBLIC REPRODUCTION DONE checks={len(checks)} output={output_path}")


if __name__ == "__main__":
    main()
