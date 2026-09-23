"""Offline 2008 cross-period check of the fixed 2020 pilot; no strategy optimization.

Run: uv run python -m scripts.build_market_stress_replication
Freeze only: add --freeze-only. Requires the exact locally cached v1 sources and
result named in the config. Missing inputs fail closed; there is no network path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market_stress import MODELS, StressConfig, forecast_panel, make_panel, score_forecasts
from market_stress_sources import SOURCE_URLS, json_bytes, parse_return_panel, verify_bundle
from scripts.build_sec_disclosure_audit import write_bundle

VERSION = "market-stress-replication-v1"
DATE_FIELDS = {"input_start", "input_end", "train_start", "evaluation_start", "evaluation_end"}


def _pinned(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Pinned input hash mismatch: {path.name}")
    return content


def freeze_replication(config_path: Path, root: Path) -> Path:
    """Freeze date-only changes before reading numeric rows of the new episode."""
    content = config_path.read_bytes()
    config = json.loads(content)
    if config["version"] != VERSION or any(
        config[key]
        for key in (
            "parameter_search_allowed",
            "untouched_holdout",
            "trading_admission",
            "restore_retired_research",
            "network_requests_allowed",
            "pooled_crisis_pvalue_allowed",
        )
    ):
        raise ValueError("Unsupported or weakened replication contract")
    base = _pinned(Path(config["base_config"]), config["base_config_sha256"])
    protocol = json.loads(base)
    changes = config["date_overrides"]
    if set(changes) != DATE_FIELDS:
        raise ValueError("Only the five date fields may change")
    protocol.update(changes)
    start = pd.Timestamp(protocol["evaluation_start"])
    end = pd.Timestamp(protocol["evaluation_end"])
    train = pd.Timestamp(protocol["train_start"])
    if (
        start != pd.Timestamp(start.year, 1, 1)
        or end != pd.Timestamp(start.year + 2, 12, 31)
        or train != pd.Timestamp(start.year - 9, 1, 1)
    ):
        raise ValueError("Preserve nine initial training years and three complete evaluation years")
    periods = config["descriptive_periods"]
    expected = {str(y): [f"{y}-01-01", f"{y}-12-31"] for y in range(start.year, end.year + 1)}
    if periods != expected:
        raise ValueError("Replication reports complete calendar years, no selected crisis days")
    protocol["descriptive_slices"] = ["all", *periods]
    for field, hash_field in (
        ("source_snapshot", "source_manifest_sha256"),
        ("prior_result", "prior_result_manifest_sha256"),
    ):
        _pinned(Path(config[field]) / "output_manifest.json", config[hash_field])
    for previous in sorted(root.glob("market-stress-replication-plan-*")):
        old = verify_bundle(previous)
        if old["replication_config.json"] == content and old["base_protocol.json"] == base:
            return previous
    return write_bundle(
        root,
        "market-stress-replication-plan",
        {
            "replication_config.json": content,
            "base_protocol.json": base,
            "protocol.json": json_bytes(protocol),
            "freeze_record.json": json_bytes(
                {
                    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "new_episode_numeric_rows_opened_in_this_invocation": False,
                    "raw_snapshot_acquired_before_this_plan": True,
                    "prior_episode_results_already_observed": True,
                    "untouched_holdout": False,
                    "history_overlap": config["sample_history_overlap"],
                }
            ),
        },
    )


def _numerical_config(plan: dict[str, Any]) -> StressConfig:
    return StressConfig(**{key: plan[key] for key in StressConfig.__dataclass_fields__})


def _replay_previous(
    source: dict[str, bytes], prior: dict[str, bytes], base: dict[str, Any]
) -> dict[str, Any]:
    """Require exact old forecasts, fits, scored ledger and metrics after refactoring."""
    frame = parse_return_panel(source, base["input_start"], base["input_end"])
    settings = _numerical_config(base)
    panel = make_panel(frame, settings)
    forecasts, fits = forecast_panel(panel, settings)
    ledger, summary = score_forecasts(panel, forecasts, settings)
    for name, computed in (("panel", panel), ("forecasts", forecasts), ("scored_ledger", ledger)):
        expected = pd.read_parquet(io.BytesIO(prior[f"{name}.parquet"]))
        pd.testing.assert_frame_equal(computed.reset_index(), expected, check_exact=True)
    if json.loads(prior["monthly_fits.json"]) != fits:
        raise ValueError("Prior model fits changed")
    previous_summary: dict[str, Any] = json.loads(prior["summary.json"])
    if any(previous_summary[key] != value for key, value in summary.items()):
        raise ValueError("Prior numerical results changed")
    return previous_summary


def compare_episodes(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Keep episodes separate; no pooled daily p-value or claim of independent crises."""
    episodes = {}
    for label, result in (("2007_2009", current), ("2019_2021", previous)):
        overall = result["slices"]["all"]
        episodes[label] = {
            "scored_origins": result["scored_origins"],
            "models": overall["models"],
            "relative_qlike_change": overall["relative_qlike_change"],
            "paired_uncertainty": overall["paired_uncertainty"],
            "years_with_lower_qlike": result["years_with_lower_qlike"],
            "historical_candidate_screen_passed": result["historical_candidate_screen_passed"],
        }
    both = all(item["historical_candidate_screen_passed"] for item in episodes.values())
    return {
        "episodes": episodes,
        "cross_episode_candidate_screen_passed": both,
        "verdict": "cross_period_historical_candidate_only"
        if both
        else "cross_period_incremental_risk_value_not_established",
        "numeric_hyperparameters_changed": False,
        "source_snapshot_changed": False,
        "pooled_independent_crisis_inference": False,
        "untouched_holdout": False,
        "liquidity_tested": False,
        "live_availability_certified": False,
        "strategy_returns_computed": False,
        "trading_admission": False,
    }


def run_replication(frozen: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Verify cached evidence, replay v1 unchanged, then score the new dated episode."""
    files = verify_bundle(frozen)
    config = json.loads(files["replication_config.json"])
    plan = json.loads(files["protocol.json"])
    base = json.loads(files["base_protocol.json"])
    expected = {
        **base,
        **config["date_overrides"],
        "descriptive_slices": ["all", *config["descriptive_periods"]],
    }
    if set(config["date_overrides"]) != DATE_FIELDS or plan != expected:
        raise ValueError("Frozen replication changed more than dates and reporting slices")
    sources, previous = Path(config["source_snapshot"]), Path(config["prior_result"])
    _pinned(sources / "output_manifest.json", config["source_manifest_sha256"])
    _pinned(previous / "output_manifest.json", config["prior_result_manifest_sha256"])
    source, prior = verify_bundle(sources), verify_bundle(previous)
    if (
        source["protocol.json"] != files["base_protocol.json"]
        or prior["protocol.json"] != files["base_protocol.json"]
    ):
        raise ValueError("Cached artifacts do not match the base protocol")
    receipts = json.loads(source["receipts.json"])
    for kind, url in SOURCE_URLS.items():
        if (
            receipts[kind]["url"] != url
            or hashlib.sha256(source[f"{kind}.zip"]).hexdigest()
            != receipts[kind]["response_sha256"]
        ):
            raise ValueError("Cached response identity changed")
    previous_summary = _replay_previous(source, prior, base)
    settings = _numerical_config(plan)
    returns = parse_return_panel(source, plan["input_start"], plan["input_end"])
    panel = make_panel(returns, settings)
    predictions, fits = forecast_panel(panel, settings)
    ledger, summary = score_forecasts(
        panel,
        predictions,
        settings,
        descriptive_periods={k: tuple(v) for k, v in config["descriptive_periods"].items()},
    )
    comparison = compare_episodes(previous_summary, summary)
    comparison["prior_replay_matches_exactly"] = True
    evidence = {
        "version": VERSION,
        "frozen_plan_manifest_sha256": hashlib.sha256(
            (frozen / "output_manifest.json").read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "prior_result_manifest_sha256": config["prior_result_manifest_sha256"],
        "original_source_receipts": receipts,
        "network_requests": 0,
        "source_acquisition_relabelled_as_new": False,
        "sample_history_overlap": config["sample_history_overlap"],
        "input_rows": len(returns),
        "input_first_session": str(returns.index.min().date()),
        "input_last_session": str(returns.index.max().date()),
        "incomplete_source_rows": int(returns.isna().any(axis=1).sum()),
        "training_label_end_before_fit_origin": bool(
            (predictions.training_label_end < predictions.model_fit_origin).all()
        ),
        "targets_start_after_forecast_origin": bool((ledger.target_start > ledger.index).all()),
        "same_scored_origins_all_models": bool(ledger[list(MODELS)].notna().all().all()),
        "calendar_independently_certified": False,
        "implementation_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "market_stress.py",
                "market_stress_sources.py",
                "scripts/build_market_stress_replication.py",
            )
        },
        "runtime_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")
        },
    }
    outputs = {
        **files,
        "summary.json": json_bytes(summary),
        "comparison.json": json_bytes(comparison),
        "evidence.json": json_bytes(evidence),
        "monthly_fits.json": json_bytes(fits),
    }
    for name, frame in (("panel", panel), ("forecasts", predictions), ("scored_ledger", ledger)):
        buffer = io.BytesIO()
        frame.reset_index().to_parquet(buffer, index=False)
        outputs[f"{name}.parquet"] = buffer.getvalue()
    return write_bundle(output_root, VERSION, outputs), comparison


def main() -> None:
    """Freeze the replication and optionally run it using pinned local data only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/market_stress_replication_v1.json")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--plan-root", type=Path, default=Path("data/market/market_stress/replication_plans")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/market_stress_replication")
    )
    args = parser.parse_args()
    frozen = freeze_replication(args.config, args.plan_root)
    print(f"STRESS REPLICATION FROZEN output={frozen}", flush=True)
    if not args.freeze_only:
        root, result = run_replication(frozen, args.output_root)
        print(
            f"STRESS REPLICATION DONE origins={result['episodes']['2007_2009']['scored_origins']} "
            f"relative_qlike_change={result['episodes']['2007_2009']['relative_qlike_change']:.2%} "
            f"prior_unchanged=True verdict={result['verdict']} trading=False output={root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
