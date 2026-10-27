"""Synthetic fixtures test leakage and four-arm mechanics, not empirical Alpha."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sec_data.information_complexity import (
    ARMS,
    FINANCIAL,
    FINANCIAL_CLOCKS,
    STATE,
    ExperimentConfig,
    audit_prior_history,
    lagged_industry_features,
    run_experiment,
    select_review_batch,
)


def panel_fixture() -> pd.DataFrame:
    """Twenty issuers in twelve synthetic time batches; no real calendar claim."""
    rng = np.random.default_rng(37)
    rows = []
    for i in range(240):
        announce = pd.Timestamp("2020-01-01T20:00:00Z") + pd.Timedelta(days=40 * (i // 20))
        decision = announce + pd.Timedelta(days=2)
        entry = decision + pd.Timedelta(minutes=1)
        exit_at = entry + pd.Timedelta(days=28)
        x = rng.normal(size=8)
        rows.append(
            {
                "validation_event_id": f"event-{i}",
                "cik": f"issuer-{i % 20}",
                "announcement_at": announce,
                "decision_at": decision,
                "entry_at": entry,
                "exit_at": exit_at,
                "label_available_at": exit_at + pd.Timedelta(hours=1),
                "market_feature_available_at": announce - pd.Timedelta(days=1),
                "industry_feature_available_at": announce - pd.Timedelta(days=2),
                **{k: announce - pd.Timedelta(hours=1) for k in FINANCIAL_CLOCKS},
                **dict(zip((*FINANCIAL, *STATE), x, strict=True)),
                "market_excess_return": 0.01 * x[0] + 0.02 * x[1] * x[4],
                "source_admitted": True,
                "information_set": "original_release",
                "entry_session_offset": 2,
                "holding_sessions_inclusive": 20,
                "entry_price_type": "unadjusted_open",
                "exit_price_type": "unadjusted_close",
            }
        )
    return pd.DataFrame(rows)


def test_fourteen_cases_cannot_trigger_fitting() -> None:
    with (
        patch("sec_data.information_complexity.Ridge.fit", side_effect=AssertionError),
        patch(
            "sec_data.information_complexity.DecisionTreeRegressor.fit", side_effect=AssertionError
        ),
    ):
        predictions, metrics, report = run_experiment(panel_fixture().head(14))
    assert predictions.empty and metrics.empty
    assert not report["training_gate"] and report["models_fitted"] == 0
    assert not report["alpha_established"]


def test_common_sample_mature_labels_and_all_representations() -> None:
    frame = panel_fixture()
    frame.loc[239, STATE[0]] = np.nan
    predictions, metrics, report = run_experiment(frame)
    assert report["common_events"] == 239
    assert len(predictions) == 139 and report["evaluation_gate"]
    assert predictions.training_events.min() == 100
    assert (
        predictions.training_label_available_end < predictions.decision_at - pd.Timedelta(days=1)
    ).all()
    assert predictions[list(ARMS)].notna().all().all()
    assert metrics.events.eq(len(predictions)).all()
    assert set(metrics.model_or_representation) == {*ARMS, *FINANCIAL, "training_mean_control"}
    assert len(report["paired_contrasts"]) == 4
    assert not report["financial_expectation_refitted"]
    assert not report["inferential_tests_computed"] and not report["strategy_pnl_computed"]


def test_own_and_same_time_outcomes_cannot_change_current_predictions() -> None:
    frame = panel_fixture()
    original, _, _ = run_experiment(frame)
    frame.loc[100:119, "market_excess_return"] = 1e6
    changed, _, _ = run_experiment(frame)
    for output in (original, changed):
        assert (
            output.loc[output.validation_event_id.eq("event-100"), "training_events"].item() == 100
        )
    pd.testing.assert_frame_equal(original.iloc[:20][list(ARMS)], changed.iloc[:20][list(ARMS)])


def test_future_delivery_labels_and_unadmitted_sources_are_not_training_data() -> None:
    frame = panel_fixture()
    frame["label_available_at"] = pd.Timestamp("2030-01-01T00:00:00Z")
    predictions, metrics, report = run_experiment(frame)
    assert predictions.empty and metrics.empty and report["models_fitted"] == 0
    frame["source_admitted"] = False
    _, _, report = run_experiment(frame)
    assert report["common_events"] == 0 and not report["training_gate"]


@pytest.mark.parametrize(
    "clock", [*FINANCIAL_CLOCKS, "market_feature_available_at", "industry_feature_available_at"]
)
def test_same_time_or_post_announcement_features_fail_closed(clock: str) -> None:
    frame = panel_fixture()
    frame.loc[0, clock] = frame.loc[0, "announcement_at"]
    with pytest.raises(ValueError, match="clock"):
        run_experiment(frame)


@pytest.mark.parametrize(
    "column,value",
    [
        ("information_set", "latest_restated"),
        ("entry_session_offset", 1),
        ("holding_sessions_inclusive", 21),
        ("entry_price_type", "unadjusted_close"),
        ("exit_price_type", "unadjusted_open"),
        ("source_admitted", "False"),
    ],
)
def test_execution_and_information_set_cannot_be_relabelled(column: str, value: object) -> None:
    frame = panel_fixture().astype({column: "object"})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        run_experiment(frame)


def test_duplicate_identity_and_premature_label_are_rejected() -> None:
    frame = panel_fixture()
    with pytest.raises(ValueError, match="identity"):
        run_experiment(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "label_available_at"] = frame.loc[0, "entry_at"]
    with pytest.raises(ValueError, match="exit"):
        run_experiment(frame)
    with pytest.raises(ValueError, match="new experiment"):
        replace(ExperimentConfig(), minimum_common_events=14).validate()


def test_deterministic_replay_and_constant_outcomes() -> None:
    frame = panel_fixture()
    first, _, _ = run_experiment(frame)
    second, _, _ = run_experiment(frame.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(first, second)
    frame["market_excess_return"] = 0.0
    _, metrics, _ = run_experiment(frame)
    assert metrics.rank_ic.isna().all() and metrics.oos_r2_vs_training_mean.isna().all()


def industry_fixture() -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    """A target, two prior-quarter peers and a forbidden current-quarter peer."""
    event = {
        "validation_event_id": "target",
        "cik": "X",
        "period_start": "2024-04-01",
        "announcement_at": "2024-08-01T20:00:00Z",
    }
    rows = [
        {
            "event_id": c,
            "company_id": c,
            "period_end": end,
            "actual_available_at": "2024-05-01T20:00:00Z",
            "gm_level": value,
            "analysis_ready_level": True,
            "scope_quarantined": False,
        }
        for c, end, value in [
            ("X", "2024-03-31", 0.9),
            ("B", "2024-03-31", 0.2),
            ("C", "2024-03-31", 0.4),
            ("D", "2024-06-30", 1.0),
        ]
    ]
    codes = [
        {
            "cik": c,
            "sic": "2400",
            "industry_available_at": "2023-01-01T20:00:00Z",
            "source_sha256": "a" * 64,
            "accession_number": c,
        }
        for c in ["X", "B", "C", "D"]
    ]
    return [event], pd.DataFrame(rows), codes


def test_industry_feature_uses_other_issuers_and_only_lagged_quarters() -> None:
    events, history, codes = industry_fixture()
    result = lagged_industry_features(events, history, codes).iloc[0]
    assert result.industry_state_ready and result.lagged_industry_gm == pytest.approx(0.3)
    assert set(result.industry_peer_issuers) == {"B", "C"}
    history.loc[history.company_id.eq("D"), "gm_level"] = -1e9
    assert lagged_industry_features(
        events, history, codes
    ).lagged_industry_gm.item() == pytest.approx(0.3)


@pytest.mark.parametrize(
    "violation", ["same_time", "quarantined", "future_code", "conflicting_quarter", "stale"]
)
def test_industry_missing_evidence_is_not_filled(violation: str) -> None:
    events, history, codes = industry_fixture()
    if violation == "same_time":
        history.loc[1, "actual_available_at"] = events[0]["announcement_at"]
    elif violation == "quarantined":
        history.loc[1, "scope_quarantined"] = True
    elif violation == "future_code":
        codes[1]["industry_available_at"] = events[0]["announcement_at"]
    elif violation == "stale":
        history.loc[1, "period_end"] = "2020-01-01"
    else:
        extra = history.iloc[[1]].assign(event_id="conflict", gm_level=0.8)
        history = pd.concat([history, extra], ignore_index=True)
    result = lagged_industry_features(events, history, codes).iloc[0]
    assert not result.industry_state_ready and pd.isna(result.lagged_industry_gm)


def test_exact_calendar_dates_cannot_be_replaced_by_future_prices() -> None:
    dates = pd.bdate_range("2024-01-01", periods=61).strftime("%Y-%m-%d")
    event = {
        "validation_event_id": "e",
        "ticker": "X",
        "price_target_id": "p",
        "action_target_id": "a",
        "security_id": "s",
        "exchange_mic": "XNYS",
        "announcement_at": "2024-04-01T20:00:00Z",
    }
    sessions = pd.DataFrame(
        {
            "exchange_mic": "XNYS",
            "session_date": dates,
            "regular_close_at": [d + "T21:00:00Z" for d in dates],
        }
    )
    stock = pd.DataFrame({"price_target_id": "p", "session_date": dates})
    market = pd.DataFrame({"session_date": dates})
    actions = pd.DataFrame(
        {
            "action_target_id": ["a"],
            "price_window_start": [dates[0]],
            "price_window_end": ["2024-06-30"],
            "all_security_action_queries_complete": [True],
        }
    )
    assert audit_prior_history([event], stock, market, sessions, actions)[0][
        "conditional_feature_calculation_allowed"
    ]
    stock.loc[0, "session_date"] = "2024-04-02"
    audit = audit_prior_history([event], stock, market, sessions, actions)[0]
    assert audit["missing_issuer_dates"] == [dates[0]]
    assert not audit["conditional_feature_calculation_allowed"]
    audit = audit_prior_history([event], stock, market, sessions.tail(60), actions)[0]
    assert audit["calendar_extension_required"] and not audit["numeric_history_ready"]


def test_next_review_batch_is_frozen_without_returns_or_survival_selection() -> None:
    rows = [
        {
            "validation_event_id": f"e{i}",
            "cik": f"c{i % 25}",
            "earliest_disclosure_at": "2024-08-01T20:00:00Z",
            "is_listed": False,
            "market_excess_return": -100.0,
        }
        for i in range(50)
    ]
    original, count = select_review_batch(rows, {"c0", "c1"})
    assert count == 23 and len(original) == 20
    assert len({r["cik"] for r in original}) == 20
    assert not ({r["cik"] for r in original} & {"c0", "c1"})
    for row in rows:
        row.update(market_excess_return=100.0, is_listed=True)
    changed, _ = select_review_batch(list(reversed(rows)), {"c0", "c1"})
    assert original == changed
    assert all(r["release_accession_number"] is None for r in original)
