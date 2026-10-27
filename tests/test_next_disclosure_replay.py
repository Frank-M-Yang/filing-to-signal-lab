"""Synthetic complete/partial timelines test frozen membership and earlier-only replays."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.replay_next_disclosures import validate_contract
from sec_data.announcement_replay import SIGNALS, apply_release_proxies, case_row
from sec_data.clock_layer import FLAG_COLUMNS, NUMERIC_COLUMNS, TIME_COLUMNS, replay_history
from sec_data.next_disclosure_replay import (
    DISARM_FLAGS,
    audit_replayed_cases,
    overlay_complete_pairs,
    select_complete_pairs,
    verify_prefix_outputs,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig


@pytest.fixture()
def experiment() -> tuple[Any, ...]:
    rows = []
    for q, end in enumerate(pd.date_range("2020-03-31", periods=16, freq="QE")):
        for issuer in range(6):
            stamp = (end + pd.Timedelta(days=40, hours=10 + issuer)).tz_localize("UTC")
            revenue = float(100 + issuer + q)
            margin = 0.3 + 0.02 * issuer + 0.03 * np.sin(q + issuer)
            rows.append(
                {
                    **dict.fromkeys(NUMERIC_COLUMNS, np.nan),
                    **dict.fromkeys(TIME_COLUMNS, pd.NaT),
                    **dict.fromkeys(FLAG_COLUMNS, False),
                    "event_id": f"{issuer}-{q}",
                    "company_id": str(issuer + 1).zfill(10),
                    "ticker": f"T{issuer}",
                    "filing_accession": f"report-{issuer}-{q}",
                    "filing_form": "10-Q",
                    "fiscal_year": end.year,
                    "fiscal_quarter": f"Q{q % 4 + 1}",
                    "period_start": end - pd.DateOffset(months=3) + pd.Timedelta(days=1),
                    "period_end": end,
                    "ytd_filing_accession": None,
                    "q4_source_method": "not_applicable",
                    "forecast_model_version": None,
                    "gm_yoy_lag_event_id": None,
                    "revenue": revenue,
                    "gross_profit": revenue * margin,
                    "gross_margin": margin,
                    "gm_level": margin,
                    "analysis_ready_level": True,
                    "filing_accepted_at": stamp,
                    "actual_available_at": stamp,
                    "scope_quarantined": False,
                    "derivation_method": "direct_gross_profit",
                }
            )
    cfg = PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)
    baseline = replay_history(pd.DataFrame(rows), cfg)
    dispositions, cells = [], []
    for i in range(20):
        e: dict[str, Any] = {
            "validation_event_id": f"case-{i}",
            "cik": str(i + 1).zfill(10),
            "conditional_pair_verified": i < 6,
            "existing_scope_quarantine": False,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        if i < 6:
            row = baseline.loc[baseline.event_id.eq(f"{i}-13")].iloc[0]
            cutoff = row.actual_available_at - pd.Timedelta(hours=8)
            early = cutoff - pd.Timedelta(days=10) if i < 3 else cutoff
            pair = {
                "conditional_pair_verified": True,
                "release_accession": f"complete-{i}",
                "submission_sha256": "a" * 64,
                "exhibit_sha256": "b" * 64,
                "gross_margin": row.gross_margin,
                "pair_acceptance_proxy_at": cutoff.isoformat(),
                "exact_original_pair": True,
                "clock_consistent": True,
                "later_than_frozen_earliest_proxy": cutoff > early,
            }
            e.update(
                ticker=row.ticker,
                assigned_ticker=row.ticker,
                target_accession_number=row.filing_accession,
                period_end=row.period_end.date().isoformat(),
                accepted_at=row.actual_available_at.isoformat(),
                earliest_disclosure_at=early.isoformat(),
                release_accession_number=f"partial-{i}" if i < 3 else f"complete-{i}",
                pair_acceptance_proxy_at=cutoff.isoformat(),
                reviewed_pairs=[pair],
            )
            for metric in ("revenue", "gross_profit"):
                cells.append(
                    {
                        "validation_event_id": e["validation_event_id"],
                        "cik": e["cik"],
                        "release_accession": f"complete-{i}",
                        "metric": metric,
                        "value": row[metric],
                        "period_start": row.period_start.date().isoformat(),
                        "period_end": e["period_end"],
                        "unit": "USD",
                        "source_sha256": "a" * 64,
                        "exhibit_sha256": "b" * 64,
                        "clock_status": "consistent_acceptance_proxy",
                        "sec_acceptance_proxy_at": cutoff.isoformat(),
                        "comparison_status": "exact_original_match",
                        "original_status": "original_quarter",
                    }
                )
        dispositions.append(e)
    return baseline, cfg, dispositions, cells


def test_later_complete_pairs_never_inherit_partial_candidate_clocks(experiment: Any) -> None:
    baseline, _, dispositions, cells = experiment
    original = copy.deepcopy(dispositions)
    cases = select_complete_pairs(dispositions, cells)
    assert dispositions == original
    assert len(cases) == 6
    assert (
        sum(c["frozen_earliest_candidate_at"] != c["complete_pair_replay_cutoff_at"] for c in cases)
        == 3
    )
    revised = overlay_complete_pairs(baseline, cases, cells)
    for c in cases:
        assert (
            case_row(revised, c).actual_available_at.isoformat()
            == c["complete_pair_replay_cutoff_at"]
        )
        assert c["earlier_candidate_backfill_allowed"] is False
    pd.testing.assert_frame_equal(revised, overlay_complete_pairs(baseline, cases[::-1], cells))
    pd.testing.assert_frame_equal(
        baseline.drop(columns="actual_available_at"), revised.drop(columns="actual_available_at")
    )


@pytest.mark.parametrize(
    "bad",
    [
        "denominator",
        "duplicate",
        "admission",
        "scope",
        "ambiguous",
        "clock",
        "backfill",
        "hash",
        "cells",
        "period",
        "metric",
        "restated",
    ],
)
def test_unverified_or_ambiguous_pairs_cannot_enter(experiment: Any, bad: str) -> None:
    _, _, dispositions, cells = experiment
    d, c = copy.deepcopy(dispositions), copy.deepcopy(cells)
    if bad == "denominator":
        d.pop()
    elif bad == "duplicate":
        d[-1]["cik"] = d[0]["cik"]
    elif bad == "admission":
        d[0]["model_admission"] = True
    elif bad == "scope":
        d[0]["existing_scope_quarantine"] = True
    elif bad == "ambiguous":
        d[0]["reviewed_pairs"].append(d[0]["reviewed_pairs"][0])
    elif bad == "clock":
        d[0]["reviewed_pairs"][0]["clock_consistent"] = False
    elif bad == "backfill":
        d[0]["earliest_disclosure_at"] = d[0]["accepted_at"]
    elif bad == "hash":
        d[0]["reviewed_pairs"][0]["submission_sha256"] = "c" * 64
    elif bad == "cells":
        c.pop(0)
    elif bad == "period":
        c[0]["period_end"] = "2000-01-01"
    elif bad == "metric":
        c[0]["metric"] = "eps"
    else:
        c[0]["comparison_status"] = "restated"
    with pytest.raises(ValueError):
        select_complete_pairs(d, c)


def test_replay_reconstructs_prior_only_forecasts_errors_and_common_cases(experiment: Any) -> None:
    baseline, cfg, dispositions, cells = experiment
    cases = select_complete_pairs(dispositions, cells)
    revised = replay_history(overlay_complete_pairs(baseline, cases, cells), cfg)
    audit, comparisons, deps, company = audit_replayed_cases(baseline, revised, cases, cfg)
    assert len(audit) == 6 and len(comparisons) == 36
    assert all(r["all_six_common_financial_cases_ready"] for r in audit)
    assert all(r["replayed_forecast_clock_check"]["history_replay_completed"] for r in audit)
    assert all(not r[k] for r in audit for k in DISARM_FLAGS)
    assert any(
        not r["old_forecast_clock_check"]["existing_forecast_clocks_before_proxy"] for r in audit
    )
    for case in cases:
        target = case_row(revised, case)
        d = pd.DataFrame(
            [r for r in deps if r["validation_event_id"] == case["validation_event_id"]]
        )
        assert d.actual_available_at.lt(target.actual_available_at).all()
        assert target.event_id not in set(d.event_id)
        assert target.forecast_sigma == pytest.approx(
            d.loc[d.in_prior_oos_errors, "gm_innovation_raw"].std(ddof=1)
        )
        own = [r for r in company if r["validation_event_id"] == case["validation_event_id"]]
        assert all(r["actual_available_at"] < target.actual_available_at for r in own)
    assert {r["representation"] for r in comparisons} >= set(SIGNALS)
    assert verify_prefix_outputs(baseline, revised, cases[0]["complete_pair_replay_cutoff_at"]) > 0
    for size in range(1, 6):
        prefix = replay_history(apply_release_proxies(baseline, cases[:size], cells), cfg)
        assert (
            verify_prefix_outputs(prefix, revised, cases[size]["complete_pair_replay_cutoff_at"])
            > 0
        )


def test_future_actual_mutation_cannot_change_earlier_complete_pair_forecasts(
    experiment: Any,
) -> None:
    baseline, cfg, dispositions, cells = experiment
    cases = select_complete_pairs(dispositions, cells)
    overlaid = overlay_complete_pairs(baseline, cases, cells)
    reference = replay_history(overlaid, cfg)
    modified = overlaid.copy()
    mask = modified.period_end.gt(pd.Timestamp("2023-06-30"))
    modified.loc[mask, ["gross_margin", "gm_level"]] += 0.2
    modified.loc[mask, "gross_profit"] = (
        modified.loc[mask, "revenue"] * modified.loc[mask, "gross_margin"]
    )
    altered = replay_history(modified, cfg)
    for case in cases:
        a, b = case_row(reference, case), case_row(altered, case)
        np.testing.assert_allclose(
            a[list(SIGNALS)].to_numpy(dtype=float),
            b[list(SIGNALS)].to_numpy(dtype=float),
            rtol=0,
            atol=1e-10,
        )


def test_missing_case_closes_whole_common_sample_without_dropping_it(experiment: Any) -> None:
    baseline, cfg, dispositions, cells = experiment
    cases = select_complete_pairs(dispositions, cells)
    revised = replay_history(overlay_complete_pairs(baseline, cases, cells), cfg)
    row = case_row(revised, cases[-1])
    revised.loc[revised.event_id.eq(row.event_id), "gm_innovation_std"] = np.nan
    revised.loc[revised.event_id.eq(row.event_id), "analysis_ready_innovation_std"] = False
    audit, _, _, _ = audit_replayed_cases(baseline, revised, cases, cfg)
    assert len(audit) == 6
    assert sum(r["complete_financial_representations"] for r in audit) == 5
    assert not any(r["all_six_common_financial_cases_ready"] for r in audit)


@pytest.mark.parametrize(
    "bad",
    [
        "returns_allowed",
        "network_allowed",
        "first_public_certified",
        "model_admission",
        "return_admission",
        "trading_admission",
        "sample_replacement_allowed",
        "overwrite_originals",
        "hyperparameter_search_allowed",
        "market_features_in_forecast",
        "threshold_relaxation_allowed",
        "minimum_prior_oos_errors",
    ],
)
def test_weakened_or_missing_contract_fails_before_any_io(bad: str) -> None:
    c: dict[str, Any] = {
        "version": "next-disclosure-replay-v1",
        "frozen_queue_issuers": 20,
        "frozen_pair_cases": 6,
        "frozen_history_events": 3254,
        "fixed_specification_refit_allowed": True,
        "primary_scope_arm": "SUN_and_all_cost_derived_quarantined",
        "minimum_forecast_training_events": 100,
        "minimum_prior_oos_errors": 50,
        **dict.fromkeys(
            (
                "hyperparameter_search_allowed",
                "threshold_relaxation_allowed",
                "sample_replacement_allowed",
                "network_allowed",
                "market_features_in_forecast",
                "returns_allowed",
                "first_public_certified",
                "model_admission",
                "return_admission",
                "trading_admission",
                "overwrite_originals",
            ),
            False,
        ),
    }
    validate_contract(c)
    c[bad] = True
    with pytest.raises(ValueError):
        validate_contract(c)
    del c[bad]
    with pytest.raises(ValueError):
        validate_contract(c)
