"""Synthetic arrivals, strict ties, frozen samples and shared-history propagation."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_sec_clock_layer import _encode
from scripts.replay_publication_intervals import _parquet, validate_contract
from sec_data.announcement_replay import case_row
from sec_data.clock_layer import replay_history
from sec_data.next_disclosure_replay import (
    overlay_complete_pairs,
    select_complete_pairs,
    verify_prefix_outputs,
)
from sec_data.next_publication import DISARM_FLAGS
from sec_data.publication_interval_replay import (
    COMPARISONS,
    audit_interval_cases,
    build_ordering_grid,
    interval_signal_summary,
    overlay_intervals,
    require_stable_endpoint_equivalence,
    validate_sources,
)
from tests.test_next_disclosure_replay import experiment as financial_fixture


@pytest.fixture(scope="module")
def interval_experiment() -> tuple[Any, ...]:
    history, cfg, dispositions, cells = financial_fixture.__wrapped__()
    cases = select_complete_pairs(dispositions, cells)
    # A seventh synthetic issuer supplies an actual crossing arrival with a
    # complete history, not a fabricated residual or in-sample error.
    foreign = history.loc[history.company_id.eq("0000000001")].copy()
    foreign["company_id"] = "0000000099"
    foreign["event_id"] = foreign.event_id.map(lambda v: "foreign-" + v)
    foreign["filing_accession"] = foreign.filing_accession.map(lambda v: "foreign-" + v)
    foreign["ticker"] = "FOREIGN"
    sources = []
    for case in cases:
        row = case_row(history, case)
        case["period_start"] = row.period_start.date().isoformat()
        lower = pd.Timestamp(case["complete_pair_replay_cutoff_at"]) - pd.Timedelta(hours=2)
        sources.append(
            {
                **{
                    k: case[k]
                    for k in ("ticker", "cik", "period_start", "period_end", "validation_event_id")
                },
                "source_id": "publisher-" + case["ticker"],
                "release_accession": case["release_accession_number"],
                "snapshot_sha256": "c" * 64,
                "exact_original_pair": True,
                "clock_resolution_seconds": 60,
                "clock_precision": "minute_explicit_offset",
                "source_reported_at_utc": lower.isoformat(),
                "lower_inclusive": lower.isoformat(),
                "upper_exclusive": (lower + pd.Timedelta(minutes=1)).isoformat(),
                "matched_cells": [
                    {"metric": k, "value_usd": row[k]} for k in ("revenue", "gross_profit")
                ],
                **dict.fromkeys(DISARM_FLAGS, False),
            }
        )
    arrival = pd.Timestamp(sources[0]["lower_inclusive"]) + pd.Timedelta(seconds=30)
    selected = foreign.event_id.eq("foreign-0-13")
    foreign.loc[selected, ["actual_available_at", "filing_accepted_at"]] = arrival
    foreign.loc[selected, ["gross_margin", "gm_level"]] = 0.8
    foreign.loc[selected, "gross_profit"] = foreign.loc[selected, "revenue"] * 0.8
    combined = pd.concat([history, foreign], ignore_index=True)
    baseline = replay_history(overlay_complete_pairs(combined, cases, cells), cfg)
    return baseline, cfg, cases, sources


def test_grid_enumerates_exact_tie_not_just_endpoints(interval_experiment: Any) -> None:
    history, _, cases, sources = interval_experiment
    grid, crossings = build_ordering_grid(history, cases, sources)
    assert len(crossings) == 1 and crossings[0]["event_id"] == "foreign-0-13"
    assert len(grid) == 5  # four regimes plus stable-minute endpoint check
    arrival = pd.Timestamp(crossings[0]["proxy_arrival_at"])
    cutoffs = [pd.Timestamp(s["cutoffs"][cases[0]["ticker"]]) for s in grid[:-1]]
    assert cutoffs[1] == arrival
    assert cutoffs[2] == arrival + pd.Timedelta(nanoseconds=1)
    assert cutoffs[-1] == pd.Timestamp(sources[0]["upper_exclusive"]) - pd.Timedelta(nanoseconds=1)
    assert all(s["return_admission"] is False for s in grid)


def test_overlay_preserves_values_acceptance_ytd_and_order(interval_experiment: Any) -> None:
    history, _, cases, sources = interval_experiment
    grid, _ = build_ordering_grid(history, cases, sources)
    revised = overlay_intervals(history, cases, sources, grid[0]["cutoffs"])
    pd.testing.assert_frame_equal(
        history.drop(columns="actual_available_at"), revised.drop(columns="actual_available_at")
    )
    pd.testing.assert_frame_equal(
        revised, overlay_intervals(history, cases[::-1], sources, grid[0]["cutoffs"])
    )
    assert history is not revised


def test_each_regime_rebuilds_prior_ols_sigma_and_six_case_denominator(
    interval_experiment: Any,
) -> None:
    baseline, cfg, cases, sources = interval_experiment
    grid, crossings = build_ordering_grid(baseline, cases, sources)
    outputs, rows = {}, []
    for scenario in grid:
        replayed = replay_history(
            overlay_intervals(baseline, cases, sources, scenario["cutoffs"]), cfg
        )
        audit, deps, own = audit_interval_cases(baseline, replayed, cases, scenario, cfg)
        assert len(audit) == 6 and all(r["same_six_common_financial_gate"] for r in audit)
        target = cases[0]
        selected = deps.loc[deps.target_ticker.eq(target["ticker"])]
        cutoff = pd.Timestamp(scenario["cutoffs"][target["ticker"]])
        assert selected.actual_available_at.lt(cutoff).all()
        crossing = selected.loc[selected.event_id.eq(crossings[0]["event_id"])]
        assert bool(not crossing.empty) == (pd.Timestamp(crossings[0]["proxy_arrival_at"]) < cutoff)
        assert own.actual_available_at.max() < max(
            pd.Timestamp(s) for s in scenario["cutoffs"].values()
        )
        row = case_row(replayed, target)
        assert row.forecast_sigma == pytest.approx(
            selected.loc[selected.in_prior_oos_errors, "gm_innovation_raw"].std(ddof=1)
        )
        assert all(r[k] is False for r in audit for k in DISARM_FLAGS)
        outputs[scenario["scenario_id"]] = replayed
        rows.extend(audit)
    end = grid[-1]
    require_stable_endpoint_equivalence(
        outputs[end["equivalent_numeric_scenario"]], outputs[end["scenario_id"]]
    )
    ranges = interval_signal_summary(rows)
    assert len(ranges) == 36 and all(r["all_scenarios_same_six_common_gate"] for r in ranges)
    assert any(
        r["changed_across_clock_scenarios"]
        for r in ranges
        if r["representation"] == "gm_innovation_raw"
    )
    assert (
        verify_prefix_outputs(
            baseline, outputs[grid[0]["scenario_id"]], sources[0]["lower_inclusive"]
        )
        > 0
    )


def test_microsecond_inputs_preserve_immediately_after_tie_precision(
    interval_experiment: Any,
) -> None:
    history, _, cases, sources = interval_experiment
    history = history.copy()
    history["actual_available_at"] = history.actual_available_at.astype("datetime64[us, UTC]")
    grid, crossings = build_ordering_grid(history, cases, sources)
    revised = overlay_intervals(history, cases, sources, grid[2]["cutoffs"])
    expected = pd.Timestamp(crossings[0]["proxy_arrival_at"]) + pd.Timedelta(nanoseconds=1)
    assert case_row(revised, cases[0]).actual_available_at == expected
    assert str(revised.actual_available_at.dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_frame_equal(
        history.drop(columns="actual_available_at"), revised.drop(columns="actual_available_at")
    )
    assert str(history.actual_available_at.dtype) == "datetime64[us, UTC]"


def test_saved_parquet_and_json_do_not_round_an_ordering_probe_to_a_tie() -> None:
    tie = pd.Timestamp("2024-11-06T21:05:46Z")
    after = tie + pd.Timedelta(nanoseconds=1)
    frame = pd.DataFrame({"event_id": ["tie", "after"], "actual_available_at": [tie, after]})
    restored = pd.read_parquet(io.BytesIO(_parquet(frame)))
    pd.testing.assert_frame_equal(frame, restored)
    assert restored.actual_available_at.iloc[0] < restored.actual_available_at.iloc[1]
    clocks = json.loads(_encode({"tie": tie.isoformat(), "after": after.isoformat()}))
    assert pd.Timestamp(clocks["after"]) - pd.Timestamp(clocks["tie"]) == pd.Timedelta(
        nanoseconds=1
    )


@pytest.mark.parametrize(
    "bad",
    [
        "cohort",
        "source",
        "identity",
        "period",
        "accepted",
        "values",
        "scale",
        "clock",
        "timezone",
        "overlap",
        "scope",
        "derived",
        "flag",
        "market",
        "duplicate",
    ],
)
def test_input_drift_or_weakening_is_rejected(interval_experiment: Any, bad: str) -> None:
    baseline, _, cases, sources = interval_experiment
    history, c, s = baseline.copy(), copy.deepcopy(cases), copy.deepcopy(sources)
    selected = history.event_id.eq(case_row(history, c[0]).event_id)
    if bad == "cohort":
        c.pop()
    elif bad == "source":
        s[1]["source_id"] = s[0]["source_id"]
    elif bad == "identity":
        s[0]["cik"] = "other"
    elif bad == "period":
        s[0]["period_end"] = "2000-01-01"
    elif bad == "accepted":
        history.loc[selected, "filing_accepted_at"] += pd.Timedelta(seconds=1)
    elif bad == "values":
        s[0]["matched_cells"][0]["value_usd"] += 1
    elif bad == "scale":
        s[0]["clock_resolution_seconds"] = 1
    elif bad == "clock":
        s[0]["upper_exclusive"] = s[0]["lower_inclusive"]
    elif bad == "timezone":
        s[0]["clock_precision"] = "minute_unknown_timezone"
    elif bad == "overlap":
        s[1]["source_reported_at_utc"] = s[0]["lower_inclusive"]
        s[1]["lower_inclusive"] = s[0]["lower_inclusive"]
        s[1]["upper_exclusive"] = s[0]["upper_exclusive"]
    elif bad == "scope":
        history.loc[selected, "scope_quarantined"] = True
    elif bad == "derived":
        history.loc[selected, "ytd_filing_accession"] = "ytd"
    elif bad == "flag":
        s[0]["first_public_certified"] = True
    elif bad == "market":
        history["future_return"] = 0.5
    else:
        history = pd.concat([history, history.iloc[:1]])
    with pytest.raises(ValueError):
        validate_sources(history, c, s)


def test_scenario_cap_never_drops_crossing_events(interval_experiment: Any) -> None:
    baseline, _, cases, sources = interval_experiment
    extras = []
    for i in range(16):
        row = baseline.loc[baseline.event_id.eq("foreign-0-13")].copy()
        row["event_id"] = "extra-" + str(i)
        row["filing_accession"] = "extra-" + str(i)
        row["actual_available_at"] = pd.Timestamp(sources[0]["lower_inclusive"]) + pd.Timedelta(
            seconds=i + 1
        )
        extras.append(row)
    with pytest.raises(ValueError, match="cap"):
        build_ordering_grid(pd.concat([baseline, *extras], ignore_index=True), cases, sources)
    with pytest.raises(ValueError, match="protocol"):
        build_ordering_grid(baseline, cases, sources, max_scenarios=999)


def test_unknown_timezone_is_a_declared_sensitivity_not_an_imputed_et_clock(
    interval_experiment: Any,
) -> None:
    baseline, _, cases, sources = interval_experiment
    s = copy.deepcopy(sources)
    lower = pd.Timestamp(cases[0]["complete_pair_replay_cutoff_at"]) - pd.Timedelta(days=2)
    s[0].update(
        clock_precision="minute_unknown_timezone",
        source_reported_at_utc=None,
        lower_inclusive=lower.isoformat(),
        upper_exclusive=(lower + pd.Timedelta(hours=28, minutes=1)).isoformat(),
    )
    assert validate_sources(baseline, cases, s)[0]["source_reported_at_utc"] is None
    grid, _ = build_ordering_grid(baseline, cases, s)
    assert grid[0]["cutoffs"][cases[0]["ticker"]] == lower.isoformat()
    assert grid[-1]["first_public_certified"] is False


@pytest.mark.parametrize("bad", ["missing", "out_of_range", "empty_partial", "extra"])
def test_overlays_require_explicit_complete_cutoffs(interval_experiment: Any, bad: str) -> None:
    history, _, cases, sources = interval_experiment
    grid, _ = build_ordering_grid(history, cases, sources)
    cutoffs = copy.deepcopy(grid[0]["cutoffs"])
    kwargs: dict[str, Any] = {}
    if bad == "missing":
        cutoffs.pop(cases[0]["ticker"])
    elif bad == "out_of_range":
        cutoffs[cases[0]["ticker"]] = sources[0]["upper_exclusive"]
    elif bad == "empty_partial":
        kwargs["selected_tickers"] = set()
    else:
        cutoffs["EXTRA"] = sources[0]["lower_inclusive"]
    with pytest.raises(ValueError):
        overlay_intervals(history, cases, sources, cutoffs, **kwargs)


def test_missing_case_closes_all_signal_ranges_without_outcome_based_deletion(
    interval_experiment: Any,
) -> None:
    history, cfg, cases, sources = interval_experiment
    grid, _ = build_ordering_grid(history, cases, sources)
    replayed = replay_history(overlay_intervals(history, cases, sources, grid[0]["cutoffs"]), cfg)
    mask = replayed.event_id.eq(case_row(replayed, cases[-1]).event_id)
    replayed.loc[mask, "gm_innovation_std"] = np.nan
    replayed.loc[mask, "analysis_ready_innovation_std"] = False
    audit, _, _ = audit_interval_cases(history, replayed, cases, grid[0], cfg)
    assert len(audit) == 6 and sum(r["complete_financial_representations"] for r in audit) == 5
    ranges = interval_signal_summary(audit)
    assert len(ranges) == 36 and all(r["minimum"] is None for r in ranges)
    with pytest.raises(ValueError, match="six"):
        interval_signal_summary(audit[:-1])


def test_future_mutation_cannot_change_earlier_reconstructed_signals(
    interval_experiment: Any,
) -> None:
    history, cfg, cases, sources = interval_experiment
    grid, _ = build_ordering_grid(history, cases, sources)
    overlaid = overlay_intervals(history, cases, sources, grid[0]["cutoffs"])
    reference = replay_history(overlaid, cfg)
    mutated = overlaid.copy()
    mask = mutated.period_end.gt(pd.Timestamp(cases[0]["period_end"]))
    mutated.loc[mask, ["gross_margin", "gm_level"]] += 0.1
    mutated.loc[mask, "gross_profit"] = (
        mutated.loc[mask, "revenue"] * mutated.loc[mask, "gross_margin"]
    )
    altered = replay_history(mutated, cfg)
    for case in cases:
        np.testing.assert_allclose(
            case_row(reference, case)[list(COMPARISONS)].to_numpy(dtype=float),
            case_row(altered, case)[list(COMPARISONS)].to_numpy(dtype=float),
            rtol=0,
            atol=1e-10,
        )
    with pytest.raises(ValueError, match="equivalence"):
        require_stable_endpoint_equivalence(history, reference)


def test_contract_fails_before_input_access_for_any_weakening() -> None:
    config = json.loads(Path("docs/publication_interval_replay_v1.json").read_bytes())
    validate_contract(config)
    for field in (
        "network_allowed",
        "price_values_allowed",
        "returns_allowed",
        "market_features_in_forecast",
        "hyperparameter_search_allowed",
        "threshold_relaxation_allowed",
        "sample_replacement_allowed",
        "overwrite_originals",
        *DISARM_FLAGS,
        "minimum_prior_oos_errors",
        "max_joint_scenarios",
    ):
        changed = copy.deepcopy(config)
        changed[field] = True
        with pytest.raises(ValueError):
            validate_contract(changed)
        del changed[field]
        with pytest.raises(ValueError):
            validate_contract(changed)


def test_missing_or_duplicate_sample_identity_does_not_pass_summary() -> None:
    rows = [
        {
            "scenario_id": "s",
            "validation_event_id": str(i),
            "ticker": str(i),
            "same_six_common_financial_gate": True,
            "values": dict.fromkeys(COMPARISONS, 0.1),
            "baseline_values": dict.fromkeys(COMPARISONS, 0.2),
        }
        for i in range(6)
    ]
    assert len(interval_signal_summary(rows)) == 36
    with pytest.raises(ValueError):
        interval_signal_summary([*rows, rows[0]])
    rows[1]["validation_event_id"] = rows[0]["validation_event_id"]
    with pytest.raises(ValueError):
        interval_signal_summary(rows)
