"""Leakage and same-sample checks for the baseline local alpha tournament."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_alpha_tournament import (
    MODEL_SPECS,
    BaselineTournamentConfig,
    calculate_baseline_tournament,
)
from tradeable_dataset import COMBINED_FEATURE_COLUMNS


def test_baseline_tournament_is_strict_walk_forward_and_keeps_holdout_sealed() -> None:
    dataset = _model_fixture()
    config = BaselineTournamentConfig(
        minimum_training_events=40,
        minimum_training_issuers=5,
        minimum_validation_events=40,
        minimum_validation_periods=2,
        minimum_events_per_period_ic=5,
        hist_min_samples_leaf=5,
    )
    result = calculate_baseline_tournament(dataset, config)

    assert not result.predictions.empty
    assert result.predictions["model_id"].nunique() == len(MODEL_SPECS)
    fold_start = pd.PeriodIndex(result.predictions["validation_period"], freq="Q").start_time
    fold_start = fold_start.tz_localize("UTC")
    training_end = pd.to_datetime(result.predictions["training_label_end"], utc=True)
    assert (training_end.to_numpy() < fold_start.to_numpy()).all()
    counts = result.predictions.groupby("model_id")["longitudinal_event_id"].size()
    assert counts.nunique() == 1
    assert result.report["holdout_status"] == "sealed_not_evaluated"
    assert result.report["holdout_metrics_computed"] is False
    assert result.report["sealed_holdout_candidate_events"] > 0
    assert result.report["selected_model_id"] in {"M3_ridge_combined", "M6_hist_combined"}


def test_prior_observed_events_train_but_never_score() -> None:
    dataset = _model_fixture()
    validation = dataset["time_role"].eq("validation")
    contaminated_ids = set(dataset.loc[validation].head(12)["longitudinal_event_id"])
    dataset.loc[
        dataset["longitudinal_event_id"].isin(contaminated_ids), "prior_outcome_observed"
    ] = True
    dataset.loc[dataset["longitudinal_event_id"].isin(contaminated_ids), "evidence_role"] = (
        "legacy_observed_training_only"
    )
    dataset.loc[
        dataset["longitudinal_event_id"].isin(contaminated_ids), "model_evaluation_candidate"
    ] = False
    config = BaselineTournamentConfig(
        minimum_training_events=40,
        minimum_training_issuers=5,
        minimum_validation_events=40,
        minimum_validation_periods=2,
        minimum_events_per_period_ic=5,
        hist_min_samples_leaf=5,
    )
    result = calculate_baseline_tournament(dataset, config)

    assert contaminated_ids.isdisjoint(set(result.predictions["longitudinal_event_id"]))


def _model_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(20260827)
    rows: list[dict[str, object]] = []
    event_number = 0
    for quarter_start in pd.date_range("2018-01-01", "2026-10-01", freq="QS"):
        for issuer_index in range(12):
            event_number += 1
            feature_cutoff = pd.Timestamp(quarter_start, tz="UTC") + pd.Timedelta(days=issuer_index)
            filing_signal = float(rng.normal())
            market_signal = float(rng.normal())
            features = {
                "gm_level": float(rng.normal(0.4, 0.08)),
                "gm_delta_yoy": float(rng.normal(0.0, 0.03)),
                "gm_innovation_raw": filing_signal * 0.02,
                "gm_innovation_std": filing_signal,
                "form_is_10k": float(issuer_index % 4 == 0),
                "issuer_momentum_5d": market_signal * 0.02,
                "issuer_momentum_20d": market_signal * 0.04,
                "issuer_volatility_20d": float(abs(rng.normal(0.25, 0.05))),
                "issuer_volume_ratio_5_20": float(rng.normal(1.0, 0.1)),
                "market_momentum_5d": float(rng.normal(0.0, 0.01)),
                "market_momentum_20d": float(rng.normal(0.0, 0.03)),
                "relative_momentum_20d": market_signal * 0.03,
                "market_beta_20d": float(rng.normal(1.0, 0.15)),
            }
            target = 0.035 * filing_signal + 0.004 * market_signal + float(rng.normal(0.0, 0.015))
            if feature_cutoff < pd.Timestamp("2020-01-01", tz="UTC"):
                time_role = "development"
            elif feature_cutoff < pd.Timestamp("2026-01-01", tz="UTC"):
                time_role = "validation"
            else:
                time_role = "sealed_holdout"
            rows.append(
                {
                    "longitudinal_event_id": f"event-{event_number:05d}",
                    "cik": f"{issuer_index + 1:010d}",
                    "feature_cutoff": feature_cutoff,
                    "execution_entry_at": feature_cutoff + pd.Timedelta(days=2),
                    "target_available_at_20d": feature_cutoff + pd.Timedelta(days=32),
                    "market_feature_available_at": feature_cutoff - pd.Timedelta(days=1),
                    "market_feature_clock_valid": True,
                    "prior_outcome_observed": False,
                    "model_training_candidate": True,
                    "model_evaluation_candidate": True,
                    "time_role": time_role,
                    "evidence_role": (
                        "sealed_holdout"
                        if time_role == "sealed_holdout"
                        else "unobserved_evaluation_candidate"
                    ),
                    "target_market_excess_return_20d": target,
                    **features,
                }
            )
    result = pd.DataFrame(rows)
    assert result.loc[:, list(COMBINED_FEATURE_COLUMNS)].notna().all().all()
    return result
