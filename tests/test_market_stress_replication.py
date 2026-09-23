"""Cross-period reporting and freeze checks use synthetic outcomes only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_stress import FEATURES, MODELS, StressConfig, score_forecasts
from market_stress_sources import json_bytes, parse_return_panel, verify_bundle
from scripts.build_market_stress_replication import compare_episodes, freeze_replication
from tests.test_market_stress import _zip


def _scored(improved: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.bdate_range("2007-01-01", "2009-12-31")
    panel = pd.DataFrame(index=dates)
    panel["target_variance"] = 0.01
    panel["future_return"] = 0.001
    panel["future_downside_variance"] = 0.002
    panel["target_start"] = dates + pd.offsets.BDay(1)
    panel["target_end"] = dates + pd.offsets.BDay(5)
    panel["target_complete"] = True
    panel["feature_complete"] = True
    for feature in FEATURES:
        panel[feature] = 0.0
    predictions = pd.DataFrame(index=dates)
    predictions["model_fit_origin"] = dates.to_period("M").to_timestamp()
    predictions["alarm_threshold_variance"] = 0.015
    predictions[MODELS[0]] = 0.03
    predictions[MODELS[1]] = 0.02
    predictions[MODELS[2]] = 0.01 if improved else 0.04
    settings = replace(
        StressConfig(),
        train_start="1998-01-01",
        evaluation_start="2007-01-01",
        evaluation_end="2009-12-31",
        bootstrap_repetitions=25,
    )
    periods = {str(y): (f"{y}-01-01", f"{y}-12-31") for y in (2007, 2008, 2009)}
    return score_forecasts(panel, predictions, settings, descriptive_periods=periods)


def test_replication_screen_counts_correct_years() -> None:
    ledger, summary = _scored()
    assert set(summary["slices"]) == {"all", "2007", "2008", "2009"}
    assert summary["years_with_lower_qlike"] == 3
    assert summary["historical_candidate_screen_passed"]
    assert summary["slices"]["all"]["scored_origins"] == len(ledger)
    assert not summary["trading_admission"]
    _, failed = _scored(False)
    assert failed["years_with_lower_qlike"] == 0
    assert not failed["historical_candidate_screen_passed"]


def test_one_good_episode_cannot_erase_the_other_negative_result() -> None:
    _, positive = _scored()
    _, negative = _scored(False)
    for previous, current in ((positive, negative), (negative, positive)):
        result = compare_episodes(previous, current)
        assert not result["cross_episode_candidate_screen_passed"]
        assert not result["pooled_independent_crisis_inference"]
        assert not result["trading_admission"]
        assert len(result["episodes"]) == 2


def test_even_two_good_historical_episodes_never_authorize_trading() -> None:
    _, positive = _scored()
    result = compare_episodes(positive, positive)
    assert result["cross_episode_candidate_screen_passed"]
    assert not result["trading_admission"]
    assert not result["untouched_holdout"]
    assert not result["live_availability_certified"]


def _local_config(tmp_path: Path) -> Path:
    config = json.loads(Path("docs/market_stress_replication_v1.json").read_bytes())
    for field, hash_field in (
        ("source_snapshot", "source_manifest_sha256"),
        ("prior_result", "prior_result_manifest_sha256"),
    ):
        root = tmp_path / field
        root.mkdir()
        content = b"fixture manifest; numeric source access is not required to freeze"
        (root / "output_manifest.json").write_bytes(content)
        config[field] = str(root)
        config[hash_field] = hashlib.sha256(content).hexdigest()
    path = tmp_path / "config.json"
    path.write_bytes(json_bytes(config))
    return path


def test_freeze_only_changes_dates_and_preserves_old_acquisition_semantics(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    frozen = freeze_replication(config, tmp_path / "plans")
    assert freeze_replication(config, tmp_path / "plans") == frozen
    files = verify_bundle(frozen)
    plan = json.loads(files["protocol.json"])
    base = json.loads(files["base_protocol.json"])
    changed = {key for key in plan if plan[key] != base[key]}
    assert changed == {
        "input_start",
        "input_end",
        "train_start",
        "evaluation_start",
        "evaluation_end",
        "descriptive_slices",
    }
    record = json.loads(files["freeze_record.json"])
    assert record["raw_snapshot_acquired_before_this_plan"]
    assert record["prior_episode_results_already_observed"]
    assert not record["untouched_holdout"]


@pytest.mark.parametrize("mutation", ["alpha", "short_train", "crisis_only", "unpin", "trading"])
def test_freeze_rejects_parameter_or_sample_drift(tmp_path: Path, mutation: str) -> None:
    path = _local_config(tmp_path)
    config = json.loads(path.read_bytes())
    if mutation == "alpha":
        config["date_overrides"]["ridge_alpha"] = 20
    elif mutation == "short_train":
        config["date_overrides"]["train_start"] = "2000-01-01"
    elif mutation == "crisis_only":
        config["descriptive_periods"]["2008"][0] = "2008-09-01"
    elif mutation == "unpin":
        config["source_manifest_sha256"] = "0" * 64
    else:
        config["trading_admission"] = True
    path.write_bytes(json_bytes(config))
    with pytest.raises(ValueError):
        freeze_replication(path, tmp_path / "plans")


def test_reused_pair_preserves_union_not_inner_join() -> None:
    from market_stress_sources import INDUSTRIES

    market = _zip(",Mkt-RF,SMB,HML,RF\n20070102,1,0,0,0\n20070103,2,0,0,0\n20070104,3,0,0,0\n")
    industries = _zip(
        "Average Value Weighted Returns -- Daily\n,"
        + ",".join(INDUSTRIES)
        + "\n20070102,"
        + ",".join(["1"] * 10)
        + "\n20070104,"
        + ",".join(["3"] * 10)
        + "\n"
    )
    result = parse_return_panel(
        {"market.zip": market, "industries.zip": industries}, "2007-01-01", "2007-01-04"
    )
    assert len(result) == 3
    assert np.isnan(result.loc["2007-01-03", "industry_NoDur"])
    assert result.loc["2007-01-03", "market_return"] == pytest.approx(0.02)


def test_reused_pair_rejects_insufficient_time_coverage() -> None:
    from market_stress_sources import INDUSTRIES

    files = {
        "market.zip": _zip(",Mkt-RF,SMB,HML,RF\n20080102,1,0,0,0\n"),
        "industries.zip": _zip(
            "Average Value Weighted Returns -- Daily\n,"
            + ",".join(INDUSTRIES)
            + "\n20080102,"
            + ",".join(["1"] * 10)
            + "\n"
        ),
    }
    with pytest.raises(ValueError, match="cover"):
        parse_return_panel(files, "2007-01-01", "2009-12-31")
