"""Offline regression tests: isolated quarter clocks, prior errors, full-cohort coverage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_publisher_replay import build_replay
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    HISTORY_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
    compare_histories,
    replay_history,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig
from sec_data.publisher_replay import (
    EVENT_COLUMNS,
    MARKET_COLUMNS,
    apply_publisher_clock,
    market_review_requests,
    target_dependencies,
    three_arm_readiness,
    verify_baseline_projection,
)


@pytest.fixture()
def case() -> tuple[pd.DataFrame, PilotFilingFeatureConfig, dict[str, Any], dict[str, Any]]:
    rows = []
    for q, period in enumerate(pd.date_range("2021-03-31", periods=16, freq="QE")):
        for issuer in range(3):
            stamp = (period + pd.Timedelta(days=40, hours=10 + issuer)).tz_localize("UTC")
            revenue = float(100 + q + issuer)
            margin = 0.3 + 0.03 * np.sin(q + issuer) + 0.02 * issuer
            row: dict[str, Any] = {k: np.nan for k in NUMERIC_COLUMNS}
            row.update({k: pd.NaT for k in TIME_COLUMNS})
            row.update({k: False for k in FLAG_COLUMNS})
            row.update(
                {
                    "event_id": f"{issuer}-{q}",
                    "company_id": f"{issuer + 1:010d}",
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
            rows.append(row)
    cfg = PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    for field in TIME_COLUMNS:
        history[field] = pd.to_datetime(history[field], utc=True)
    history = replay_history(history, cfg)
    row = history.loc[history.event_id.eq("2-12")].iloc[0]
    # Move the third issuer before the other two reports from the same quarter.
    lower = row.actual_available_at - pd.Timedelta(hours=3)
    request = {
        "cik": row.company_id,
        "source_id": "synthetic",
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
        "source_id": "synthetic",
        "snapshot_sha256": "a" * 64,
        "content_scope": "full_financial_table",
        "clock_kind": "visible_EST_minute",
        "period_start": row.period_start.date().isoformat(),
        "period_end": row.period_end.date().isoformat(),
        "cells": [{"metric": k, "value_usd": row[k]} for k in ("revenue", "gross_profit")],
    }
    return history, cfg, request, spec


def test_quarter_clock_does_not_rewrite_acceptance_ytd_or_source(case: Any) -> None:
    history, _, request, spec = case
    history.loc[history.event_id.eq("2-13"), "ytd_filing_accession"] = request[
        "target_filing_accession"
    ]
    history.loc[history.event_id.eq("2-13"), "ytd_available_at"] = pd.Timestamp(
        request["same_contract_filing_proxy_window"]["announcement_at"]
    )
    original = history.copy(deep=True)
    revised = apply_publisher_clock(
        history, request, spec, pd.Timestamp(request["source_reported_at_utc"])
    )
    pd.testing.assert_frame_equal(history, original)
    pd.testing.assert_frame_equal(
        history.drop(columns="actual_available_at"), revised.drop(columns="actual_available_at")
    )
    assert history.actual_available_at.ne(revised.actual_available_at).sum() == 1


@pytest.mark.parametrize("bad", ["period", "value", "identity", "gate", "outside_minute"])
def test_publisher_evidence_fail_closed(case: Any, bad: str) -> None:
    history, _, request, spec = case
    spec, request = copy.deepcopy(spec), copy.deepcopy(request)
    stamp = pd.Timestamp(request["source_reported_at_utc"])
    if bad == "period":
        spec["period_start"] = "2000-01-01"
    elif bad == "value":
        spec["cells"][0]["value_usd"] += 1
    elif bad == "identity":
        request["cik"] = "0000000999"
    elif bad == "gate":
        request["return_admission"] = True
    else:
        stamp += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        apply_publisher_clock(history, request, spec, stamp)


def test_full_replay_removes_late_training_and_rebuilds_oos_sigma(case: Any) -> None:
    history, cfg, request, spec = case
    stamp = pd.Timestamp(request["source_reported_at_utc"])
    new = replay_history(apply_publisher_clock(history, request, spec, stamp), cfg)
    earlier = history.actual_available_at.lt(stamp)
    assert compare_histories(history.loc[earlier], new.loc[earlier])[1].empty
    prior = target_dependencies(history, "2-12")
    revised = target_dependencies(new, "2-12")
    assert set(prior.event_id) - set(revised.event_id) == {"0-12", "1-12"}
    assert revised.actual_available_at.lt(stamp).all()
    assert "2-12" not in set(revised.event_id)
    target = new.loc[new.event_id.eq("2-12")].iloc[0]
    assert target.forecast_sigma == pytest.approx(revised.gm_innovation_raw.std(ddof=1))
    assert target.forecast_generated_at < stamp
    upper = replay_history(
        apply_publisher_clock(
            history, request, spec, stamp + pd.Timedelta(seconds=59, microseconds=999999)
        ),
        cfg,
    )
    counts, _ = compare_histories(new, upper)
    assert all(counts[k] == 0 for k in ("gm_forecast", "forecast_sigma", "gm_innovation_std"))
    again = replay_history(apply_publisher_clock(history, request, spec, stamp), cfg)
    pd.testing.assert_frame_equal(new, again)


def test_baseline_projection_rejects_value_or_sample_changes(case: Any) -> None:
    history, _, _, _ = case
    saved = history[["event_id", "company_id", "actual_available_at", "gm_forecast"]].copy()
    verify_baseline_projection(history, saved)
    with pytest.raises(ValueError, match="sample"):
        verify_baseline_projection(history, saved.iloc[:-1])
    saved.loc[saved.gm_forecast.notna(), "gm_forecast"] += 0.01
    with pytest.raises(ValueError, match="not reproduced"):
        verify_baseline_projection(history, saved)


def test_edt_replay_allows_only_the_reviewed_minute_kind(case: Any) -> None:
    history, _, request, spec = case
    spec["clock_kind"] = "visible_EDT_minute"
    stamp = pd.Timestamp(request["source_reported_at_utc"])
    result = apply_publisher_clock(history, request, spec, stamp)
    assert result.loc[result.event_id.eq("2-12"), "actual_available_at"].iloc[0] == stamp
    spec["clock_kind"] = "datetime_without_timezone"
    with pytest.raises(ValueError, match="explicit minute"):
        apply_publisher_clock(history, request, spec, stamp)


@pytest.mark.parametrize("hours_after", [0, 1])
def test_market_late_cutoff_blocked_and_missing_events_retained(
    case: Any, hours_after: int
) -> None:
    history, cfg, request, spec = case
    stamp = pd.Timestamp(request["source_reported_at_utc"])
    new = replay_history(apply_publisher_clock(history, request, spec, stamp), cfg)
    events = pd.DataFrame(
        [
            ["v1", request["cik"], request["target_filing_accession"]],
            ["v2", "0000000001", "A0-12"],
            ["v3", "0000000999", "absent"],
        ],
        columns=EVENT_COLUMNS,
    )
    market = pd.DataFrame(
        [
            ["v1", request["cik"], stamp + pd.Timedelta(hours=hours_after), True],
            ["v2", "0000000001", stamp - pd.Timedelta(days=1), True],
        ],
        columns=MARKET_COLUMNS,
    )
    output = three_arm_readiness(events, new, market, "v1").set_index("validation_event_id")
    assert len(output) == 3
    assert output.loc["v1", "market_cutoff_status"] == "at_or_after_event"
    assert not output.loc["v1", "market_metadata_coverage"]
    assert output.loc["v3", "market_cutoff_status"] == "missing_market_row"
    assert not output.loc["v3", "financial_representation_coverage"]
    assert output.joint_metadata_coverage.sum() == 1
    assert not output.return_admission.any()
    with pytest.raises(ValueError, match="metadata-only"):
        three_arm_readiness(events, new, market.assign(forward_return=0.1), "v1")
    market.loc[0, "cik"] = "0000000999"
    with pytest.raises(ValueError, match="issuer identity"):
        three_arm_readiness(events, new, market, "v1")


def test_build_rejects_tampered_manifest_before_replay(tmp_path: Path) -> None:
    config = json.loads(Path("docs/publisher_replay_v1.json").read_text())
    (tmp_path / "output_manifest.json").write_text("{}")
    config["chronology_root"] = str(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        build_replay(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_same_date_market_review_is_not_a_late_cutoff_claim() -> None:
    readiness = pd.DataFrame(
        {
            "validation_event_id": ["chosen", "other"],
            "market_cutoff_status": ["before_proxy_only", "at_or_after_event"],
            "return_admission": [False, False],
        }
    )
    before = readiness.copy(deep=True)
    selected = market_review_requests(readiness, "chosen", "selected_target")
    assert list(selected.validation_event_id) == ["chosen"]
    assert selected.market_cutoff_status.iloc[0] == "before_proxy_only"
    assert not selected.return_admission.any()
    assert list(market_review_requests(readiness, "chosen").validation_event_id) == ["other"]
    pd.testing.assert_frame_equal(readiness, before)


@pytest.mark.parametrize("bad", ["policy", "missing", "duplicate", "admitted"])
def test_market_review_queue_fails_closed(bad: str) -> None:
    rows = pd.DataFrame(
        {
            "validation_event_id": ["v1"],
            "market_cutoff_status": ["before_proxy_only"],
            "return_admission": [bad == "admitted"],
        }
    )
    if bad == "duplicate":
        rows = pd.concat([rows, rows], ignore_index=True)
    with pytest.raises(ValueError):
        market_review_requests(
            rows,
            "absent" if bad == "missing" else "v1",
            "unknown" if bad == "policy" else "selected_target",
        )


def test_bad_market_policy_stops_before_loading_sources(tmp_path: Path) -> None:
    config = json.loads(Path("docs/publisher_replay_v1.json").read_bytes())
    config["market_review_policy"] = "select_by_future_return"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contract"):
        build_replay(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
