"""Two-case clock replay: exact originals, causal training and explainable signal changes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.replay_announcement_cases import build_replay
from sec_data.announcement_replay import (
    apply_release_proxies,
    case_row,
    company_dependencies,
    compare_case_signals,
    select_replay_cases,
    verify_prefix_causality,
    verify_target_reconstruction,
)
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    HISTORY_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
    compare_histories,
    replay_history,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig


@pytest.fixture()
def experiment() -> tuple[
    pd.DataFrame, PilotFilingFeatureConfig, list[dict[str, Any]], list[dict[str, Any]]
]:
    rows = []
    for q, period in enumerate(pd.date_range("2021-03-31", periods=16, freq="QE")):
        for issuer in range(3):
            stamp = (period + pd.Timedelta(days=40, hours=10 + issuer)).tz_localize("UTC")
            revenue = float(100 + q + issuer)
            margin = 0.3 + 0.03 * np.sin(q + issuer) + 0.02 * issuer
            rows.append(
                {
                    **dict.fromkeys(NUMERIC_COLUMNS, np.nan),
                    **dict.fromkeys(TIME_COLUMNS, pd.NaT),
                    **dict.fromkeys(FLAG_COLUMNS, False),
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
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    for col in TIME_COLUMNS:
        history[col] = pd.to_datetime(history[col], utc=True)
    cfg = PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)
    baseline = replay_history(history, cfg)
    cases, cells = [], []
    for event_id in ("2-12", "1-14"):
        row = baseline.loc[baseline.event_id.eq(event_id)].iloc[0]
        new = row.actual_available_at - pd.Timedelta(hours=3)
        case = {
            "validation_event_id": event_id,
            "cik": row.company_id,
            "assigned_ticker": row.ticker,
            "target_accession_number": row.filing_accession,
            "release_accession_number": f"release-{event_id}",
            "clock_source": "item_2_02_8k",
            "period_end": row.period_end.date().isoformat(),
            "report_date": row.period_end.date().isoformat(),
            "accepted_at": row.actual_available_at.isoformat(),
            "earliest_disclosure_at": new.isoformat(),
            "release_accepted_at": new.isoformat(),
            "release_pair_source_sha256": "a" * 64,
            "release_pair_exhibit_sha256": "b" * 64,
            "first_public_certified": False,
            "model_admission": False,
            "return_admission": False,
        }
        cases.append(case)
        for metric in ("revenue", "gross_profit"):
            cells.append(
                {
                    "validation_event_id": event_id,
                    "cik": row.company_id,
                    "release_accession": case["release_accession_number"],
                    "metric": metric,
                    "value": row[metric],
                    "period_start": row.period_start.date().isoformat(),
                    "period_end": case["period_end"],
                    "unit": "USD",
                    "source_sha256": "a" * 64,
                    "exhibit_sha256": "b" * 64,
                    "clock_status": "consistent_acceptance_proxy",
                    "sec_acceptance_proxy_at": new.isoformat(),
                    "comparison_status": "exact_original_match",
                    "original_status": "original_quarter",
                }
            )
    return baseline, cfg, cases, cells


def test_overlay_changes_only_evidence_bound_quarter_clocks(experiment: Any) -> None:
    baseline, _, cases, cells = experiment
    original = baseline.copy(deep=True)
    revised = apply_release_proxies(baseline, cases, cells)
    pd.testing.assert_frame_equal(baseline, original)
    pd.testing.assert_frame_equal(
        baseline.drop(columns="actual_available_at"), revised.drop(columns="actual_available_at")
    )
    assert revised.actual_available_at.ne(baseline.actual_available_at).sum() == 2
    pd.testing.assert_frame_equal(revised, apply_release_proxies(baseline, cases[::-1], cells))


@pytest.mark.parametrize(
    "bad", ["duplicate", "value", "period", "hash", "later_release", "admission", "clock", "ytd"]
)
def test_missing_or_incompatible_evidence_cannot_backdate_values(experiment: Any, bad: str) -> None:
    baseline, _, cases, cells = experiment
    cases, cells = copy.deepcopy(cases), copy.deepcopy(cells)
    if bad == "duplicate":
        cases.append(cases[0])
    elif bad == "value":
        cells[0]["value"] += 1
    elif bad == "period":
        cells[0]["period_start"] = "2000-01-01"
    elif bad == "hash":
        cases[0]["release_pair_source_sha256"] = "c" * 64
    elif bad == "later_release":
        cells[0]["release_accession"] = "later"
    elif bad == "admission":
        cases[0]["return_admission"] = True
    elif bad == "clock":
        cases[0]["accepted_at"] = "2025-01-01T00:00Z"
    else:
        baseline.loc[baseline.event_id.eq("2-12"), "ytd_filing_accession"] = "source"
    with pytest.raises(ValueError):
        apply_release_proxies(baseline, cases, cells)


def test_complete_replay_rebuilds_prior_training_errors_and_company_lags(experiment: Any) -> None:
    baseline, cfg, cases, cells = experiment
    revised = replay_history(apply_release_proxies(baseline, cases, cells), cfg)
    target = case_row(revised, cases[0])
    deps = verify_target_reconstruction(revised, cases[0])
    assert deps.actual_available_at.lt(target.actual_available_at).all()
    assert not {"0-12", "1-12", "2-12"} & set(deps.event_id)
    assert target.forecast_sigma == pytest.approx(
        deps.loc[deps.in_prior_oos_errors, "gm_innovation_raw"].std(ddof=1)
    )
    assert (
        target.forecast_training_observations
        < case_row(baseline, cases[0]).forecast_training_observations
    )
    company = company_dependencies(revised, target, cfg)
    assert company.company_id.eq(target.company_id).all()
    assert company.actual_available_at.lt(target.actual_available_at).all()
    assert "2-8" in set(company.event_id)
    assert company.roles.map(lambda x: "trailing_margin_mean_and_volatility" in x).sum() == 8
    earlier = baseline.actual_available_at.lt(target.actual_available_at)
    assert compare_histories(baseline.loc[earlier], revised.loc[earlier])[1].empty


def test_later_actual_values_cannot_change_earlier_forecasts_or_sigma(experiment: Any) -> None:
    baseline, cfg, cases, cells = experiment
    source = apply_release_proxies(baseline, cases, cells)
    normal = replay_history(source, cfg)
    future = source.actual_available_at.gt(pd.Timestamp(cases[0]["earliest_disclosure_at"]))
    source.loc[future, "gross_margin"] = 0.99
    source.loc[future, "gm_level"] = 0.99
    source.loc[future, "gross_profit"] = source.loc[future, "revenue"] * 0.99
    poisoned = replay_history(source, cfg)
    target = case_row(normal, cases[0])
    changed = case_row(poisoned, cases[0])
    assert target.gm_forecast == changed.gm_forecast
    assert target.forecast_sigma == changed.forecast_sigma
    assert target.gm_innovation_std == changed.gm_innovation_std


def test_comparison_preserves_common_cases_and_attributes_own_vs_shared_shift(
    experiment: Any,
) -> None:
    baseline, cfg, cases, cells = experiment
    arms = {"legacy": baseline, "reconciled": baseline}
    for case in cases:
        arms[f"only_{case['assigned_ticker']}"] = replay_history(
            apply_release_proxies(baseline, [case], cells), cfg
        )
    arms["joint"] = replay_history(apply_release_proxies(baseline, cases, cells), cfg)
    table = compare_case_signals(arms, cases)
    assert len(table) == 12 and table.validation_event_id.nunique() == 2
    assert table.loc[table.representation.eq("gm_level"), "changed"].eq(False).all()
    assert table.clock_reconciliation_change.eq(0).all()
    assert np.allclose(
        table.total_change, table.own_release_shift_change + table.joint_minus_own_only
    )
    changed = arms["joint"].copy()
    changed.loc[changed.event_id.eq("2-12"), "gm_innovation_std"] = np.nan
    with pytest.raises(ValueError, match="Incomplete common-case"):
        compare_case_signals({**arms, "joint": changed}, cases)


@pytest.mark.parametrize(
    "key",
    [
        "hyperparameter_search_allowed",
        "market_features_in_forecast",
        "network_allowed",
        "post_event_prices_allowed",
        "returns_allowed",
        "first_public_certified",
        "trading_admission",
        "sample_replacement_allowed",
        "overwrite_originals",
    ],
)
def test_weakened_contract_stops_before_source_reads(tmp_path: Path, key: str) -> None:
    c = json.loads(Path("docs/announcement_replay_v1.json").read_bytes())
    c[key] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        build_replay(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_stress_selection_retains_all_verified_clock_failures_including_price_gaps() -> None:
    cohort = [
        {
            "validation_event_id": name,
            "earliest_disclosure_at": f"2025-0{i + 1}-01T12:00Z",
            "release_value_pair_verified": True,
            "existing_forecast_clocks_before_proxy": False,
            "market_history_evidence_ready": i != 0,
        }
        for i, name in enumerate(("missing_prices", "complete_a", "complete_b"))
    ]
    cohort.append(
        {
            "validation_event_id": "passing_control",
            "release_value_pair_verified": True,
            "existing_forecast_clocks_before_proxy": True,
            "local_replay_candidate": True,
            "earliest_disclosure_at": "2025-04-01T12:00Z",
        }
    )
    ids = ["missing_prices", "complete_a", "complete_b"]
    selected = select_replay_cases(cohort, ids, "reviewed_values_clock_failures")
    assert [r["validation_event_id"] for r in selected] == ids
    assert not selected[0]["market_history_evidence_ready"]
    with pytest.raises(ValueError, match="entire"):
        select_replay_cases(cohort, ids[1:], "reviewed_values_clock_failures")
    with pytest.raises(ValueError, match="policy"):
        select_replay_cases(cohort, ids, "choose_by_return")
    with pytest.raises(ValueError, match="Duplicate"):
        select_replay_cases([*cohort, cohort[0]], ids, "reviewed_values_clock_failures")
    assert select_replay_cases(cohort, ["passing_control"], "ready_cases")[0][
        "local_replay_candidate"
    ]


def test_three_case_prefixes_preserve_causality_and_common_comparison(experiment: Any) -> None:
    baseline, cfg, cases, cells = experiment
    cases, cells = copy.deepcopy(cases), copy.deepcopy(cells)
    target = baseline.loc[baseline.event_id.eq("0-15")].iloc[0]
    third = {
        **cases[0],
        "validation_event_id": "0-15",
        "cik": target.company_id,
        "assigned_ticker": target.ticker,
        "target_accession_number": target.filing_accession,
        "release_accession_number": "release-0-15",
        "period_end": target.period_end.date().isoformat(),
        "report_date": target.period_end.date().isoformat(),
        "accepted_at": target.actual_available_at.isoformat(),
        "earliest_disclosure_at": (target.actual_available_at - pd.Timedelta(hours=3)).isoformat(),
        "release_accepted_at": (target.actual_available_at - pd.Timedelta(hours=3)).isoformat(),
    }
    cases.append(third)
    for template in cells[:2]:
        cells.append(
            {
                **template,
                "validation_event_id": third["validation_event_id"],
                "cik": third["cik"],
                "release_accession": third["release_accession_number"],
                "value": target[template["metric"]],
                "period_start": target.period_start.date().isoformat(),
                "period_end": third["period_end"],
                "sec_acceptance_proxy_at": third["earliest_disclosure_at"],
            }
        )
    histories = {"legacy": baseline, "reconciled": baseline}
    for case in cases:
        histories[f"only_{case['assigned_ticker']}"] = replay_history(
            apply_release_proxies(baseline, [case], cells), cfg
        )
    prefix = replay_history(apply_release_proxies(baseline, cases[:2], cells), cfg)
    joint = replay_history(apply_release_proxies(baseline, cases, cells), cfg)
    prefixes = {1: histories["only_S2"], 2: prefix}
    checks = verify_prefix_causality(baseline, joint, cases, prefixes)
    assert set(checks) == {"before_first_correction", "before_correction_2", "before_correction_3"}
    table = compare_case_signals({**histories, "prefix_2": prefix, "joint": joint}, cases)
    assert len(table) == 18 and table.validation_event_id.nunique() == 3
    changed = prefix.copy()
    changed.loc[changed.event_id.eq("2-12"), "gm_forecast"] += 1.0
    with pytest.raises(ValueError, match="earlier prefix"):
        verify_prefix_causality(baseline, joint, cases, {**prefixes, 2: changed})
    with pytest.raises(ValueError, match="Missing causal"):
        verify_prefix_causality(baseline, joint, cases, {1: prefixes[1]})
    with pytest.raises(ValueError, match="chronological"):
        verify_prefix_causality(baseline, joint, cases[::-1], prefixes)


def test_stress_contract_cannot_reclassify_or_replace_cases(tmp_path: Path) -> None:
    c = json.loads(Path("docs/announcement_replay_stress_v2.json").read_bytes())
    c["case_selection_policy"] = "ready_cases"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        build_replay(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()
