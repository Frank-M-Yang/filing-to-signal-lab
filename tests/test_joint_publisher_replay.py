"""Shared error-history, interval, identity and admission controls for joint replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_joint_publisher_replay import build_joint_replay
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    HISTORY_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
    compare_histories,
    replay_history,
)
from sec_data.joint_publisher_replay import (
    CASE_FLAGS,
    apply_joint_clocks,
    assert_originals_preserved,
    compare_error_histories,
    consolidate_joint_admission,
    require_endpoint_invariance,
    validate_joint_intervals,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig
from sec_data.publisher_replay import apply_publisher_clock


@pytest.fixture()
def joint_case() -> tuple[pd.DataFrame, list[dict[str, Any]], PilotFilingFeatureConfig]:
    rows = []
    rng = np.random.default_rng(129)
    for q, period in enumerate(pd.date_range("2021-03-31", periods=16, freq="QE")):
        for issuer in range(3):
            stamp = (period + pd.Timedelta(days=40, hours=10 + issuer)).tz_localize("UTC")
            revenue = float(100 + q + issuer)
            # A pure sine wave is exactly forecastable by lag features; fixed
            # noise gives this propagation fixture genuinely nonzero OOS errors.
            margin = 0.3 + 0.03 * np.sin(q + issuer) + 0.02 * issuer + rng.normal(0, 0.006)
            rows.append(
                {
                    **dict.fromkeys(NUMERIC_COLUMNS, np.nan),
                    **dict.fromkeys(TIME_COLUMNS, pd.NaT),
                    **dict.fromkeys(FLAG_COLUMNS, False),
                    "event_id": f"{issuer}-{q}",
                    "company_id": str(issuer),
                    "ticker": f"S{issuer}",
                    "filing_accession": f"A{issuer}-{q}",
                    "fiscal_year": period.year,
                    "fiscal_quarter": f"Q{q % 4 + 1}",
                    "period_start": period - pd.DateOffset(months=3) + pd.Timedelta(days=1),
                    "period_end": period,
                    "filing_form": "10-Q",
                    "ytd_filing_accession": None,
                    "q4_source_method": "not_applicable",
                    "gm_yoy_lag_event_id": None,
                    "forecast_model_version": None,
                    "revenue": revenue,
                    "gross_profit": revenue * margin,
                    "gross_margin": margin,
                    "gm_level": margin,
                    "analysis_ready_level": True,
                    "filing_accepted_at": stamp,
                    "actual_available_at": stamp,
                }
            )
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    for column in TIME_COLUMNS:
        history[column] = pd.to_datetime(history[column], utc=True)
    cfg = PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)
    baseline = replay_history(history, cfg)
    cases = []
    for q in (10, 12):
        row = baseline.loc[baseline.event_id.eq(f"2-{q}")].iloc[0]
        lower = row.actual_available_at - pd.Timedelta(hours=3)
        req = {
            "validation_event_id": f"v{q}",
            "cik": row.company_id,
            "source_id": f"s{q}",
            "snapshot_sha256": "a" * 64,
            "target_filing_accession": row.filing_accession,
            "source_reported_at_utc": lower.isoformat(),
            "same_contract_filing_proxy_window": {
                "announcement_at": row.actual_available_at.isoformat()
            },
            "change_filing_acceptance_timestamp": False,
            "first_public_certified": False,
            "price_collection_authorized": False,
            "return_admission": False,
            "purpose": "disarmed_publisher_label_sensitivity_only",
        }
        spec = {
            "cik": row.company_id,
            "source_id": f"s{q}",
            "snapshot_sha256": "a" * 64,
            "content_scope": "full_financial_table",
            "clock_kind": "visible_EST_minute",
            "period_start": row.period_start.date().isoformat(),
            "period_end": row.period_end.date().isoformat(),
            "cells": [{"metric": k, "value_usd": row[k]} for k in ("revenue", "gross_profit")],
        }
        cases.append({"request": req, "spec": spec})
    return baseline, cases, cfg


def test_joint_order_and_originals_invariant(joint_case: Any) -> None:
    baseline, cases, _ = joint_case
    snapshot = baseline.copy(deep=True)
    endpoints = {"s10": "start", "s12": "end"}
    joint = apply_joint_clocks(baseline, cases, endpoints)
    pd.testing.assert_frame_equal(joint, apply_joint_clocks(baseline, cases[::-1], endpoints))
    pd.testing.assert_frame_equal(baseline, snapshot)
    assert_originals_preserved(baseline, joint)
    assert int(joint.actual_available_at.ne(baseline.actual_available_at).sum()) == 2


def test_joint_causal_error_propagation_and_all_corners(joint_case: Any) -> None:
    baseline, cases, cfg = joint_case
    joint = replay_history(
        apply_joint_clocks(baseline, cases, {"s10": "start", "s12": "start"}), cfg
    )
    earlier_req, earlier_spec = cases[0]["request"], cases[0]["spec"]
    first_only = replay_history(
        apply_publisher_clock(
            baseline, earlier_req, earlier_spec, pd.Timestamp(earlier_req["source_reported_at_utc"])
        ),
        cfg,
    )
    before_later = joint.actual_available_at.lt(
        pd.Timestamp(cases[1]["request"]["source_reported_at_utc"])
    )
    assert compare_histories(first_only.loc[before_later], joint.loc[before_later])[1].empty
    later_req, later_spec = cases[1]["request"], cases[1]["spec"]
    later_only = replay_history(
        apply_publisher_clock(
            baseline, later_req, later_spec, pd.Timestamp(later_req["source_reported_at_utc"])
        ),
        cfg,
    )
    changes, evidence = compare_error_histories(later_only, joint, "2-12")
    assert changes["shared_prior_error_values_changed"] > 0
    assert (
        evidence.loc[evidence.shared_prior_error_changed, "actual_available_at_joint"]
        .lt(pd.Timestamp(later_req["source_reported_at_utc"]))
        .all()
    )
    assert changes["training_added"] == changes["training_removed"] == 0
    assert changes["prior_errors_added"] == changes["prior_errors_removed"] == 0
    for endpoints in (
        {"s10": "end", "s12": "start"},
        {"s10": "start", "s12": "end"},
        {"s10": "end", "s12": "end"},
    ):
        other = replay_history(apply_joint_clocks(baseline, cases, endpoints), cfg)
        require_endpoint_invariance(joint, other)
    changed = joint.copy()
    changed.loc[changed.event_id.eq("2-12"), "gm_innovation_std"] += 1
    with pytest.raises(ValueError, match="endpoints"):
        require_endpoint_invariance(joint, changed)


@pytest.mark.parametrize(
    "change", ["duplicate", "overlap", "inside", "missing", "endpoint", "gate"]
)
def test_joint_ambiguous_or_unsafe_inputs_rejected(joint_case: Any, change: str) -> None:
    baseline, cases, _ = joint_case
    cases = copy.deepcopy(cases)
    endpoints = {"s10": "start", "s12": "start"}
    if change == "duplicate":
        cases[1] = copy.deepcopy(cases[0])
    elif change == "overlap":
        cases[1]["request"]["source_reported_at_utc"] = cases[0]["request"][
            "source_reported_at_utc"
        ]
    elif change == "inside":
        baseline.loc[baseline.event_id.eq("0-10"), "actual_available_at"] = pd.Timestamp(
            cases[0]["request"]["source_reported_at_utc"]
        ) + pd.Timedelta(seconds=30)
    elif change == "missing":
        cases[1]["request"]["target_filing_accession"] = "absent"
    elif change == "endpoint":
        endpoints["s12"] = "guess"
    else:
        cases[1]["request"]["return_admission"] = True
    with pytest.raises(ValueError):
        apply_joint_clocks(baseline, cases, endpoints)


def test_only_two_frozen_events_supported(joint_case: Any) -> None:
    baseline, cases, _ = joint_case
    with pytest.raises(ValueError, match="exactly two"):
        validate_joint_intervals(baseline, cases[:1])


def _ledgers() -> list[tuple[str, pd.DataFrame]]:
    common = pd.DataFrame(
        {
            "validation_event_id": ["u", "f", "m"],
            "cik": ["1", "2", "3"],
            "actual_available_at": ["a", "b", "c"],
            "financial_four_representation_coverage": True,
            "model_admission": False,
            "return_admission": False,
            "trading_admission": False,
            "first_public_certified": False,
        }
    )
    result = []
    for event_id in ("u", "f"):
        frame = common.copy()
        for flag in CASE_FLAGS[:-1]:
            frame[flag] = frame.validation_event_id.eq(event_id)
        frame["current_liquidity_status"] = "not_rebuilt"
        frame.loc[frame.validation_event_id.eq(event_id), "current_liquidity_status"] = "eligible"
        frame["blockers"] = [[] for _ in range(3)]
        frame["admission_status"] = "old"
        result.append((event_id, frame))
    return result


def test_joint_admission_keeps_third_event_and_gates_closed() -> None:
    result = consolidate_joint_admission(_ledgers())
    assert list(result.validation_event_id) == ["u", "f", "m"]
    assert list(result.label_scenario_inputs_complete) == [True, True, False]
    assert result.return_admission.eq(False).all()
    assert result.iloc[0].blockers == [
        "first_public_clock_not_verified",
        "return_experiment_not_frozen",
    ]
    assert "financial_chain_not_rebuilt_for_release_clock" in result.iloc[2].blockers


@pytest.mark.parametrize(
    "change", ["different_history", "drop_row", "admit", "spillover", "duplicate"]
)
def test_joint_admission_rejects_splicing_or_promoting(change: str) -> None:
    ledgers = _ledgers()
    frame = ledgers[1][1]
    if change == "different_history":
        frame.loc[0, "actual_available_at"] = "other"
    elif change == "drop_row":
        ledgers[1] = ("f", frame.iloc[:2])
    elif change == "admit":
        frame["model_admission"] = True
    elif change == "spillover":
        frame.loc[2, "label_financial_replay_ready"] = True
    else:
        ledgers[1] = ledgers[0]
    with pytest.raises((ValueError, AssertionError)):
        consolidate_joint_admission(ledgers)


def test_contract_blocks_return_access_before_reading_inputs(tmp_path: Path) -> None:
    config = json.loads(Path("docs/joint_publisher_replay_v1.json").read_bytes())
    config["returns_allowed"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contract"):
        build_joint_replay(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
