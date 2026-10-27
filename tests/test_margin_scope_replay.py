"""Accounting provenance, quarantine propagation, frozen samples and causal forecasts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_margin_scope_replay import build_scope_replay, validate_contract
from sec_data.announcement_replay import SIGNALS
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
)
from sec_data.margin_scope_replay import (
    ARMS,
    audit_original_recipes,
    replay_scope_view,
    scope_case_signals,
    scope_quarantine_view,
    scope_rank_diagnostics,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig, _derived_fact_id
from sec_data.publisher_replay import target_dependencies


@pytest.fixture()
def history() -> pd.DataFrame:
    rows = []
    for q, period in enumerate(pd.date_range("2021-03-31", periods=20, freq="QE")):
        for issuer in range(3):
            stamp = (period + pd.Timedelta(days=40, hours=10 + issuer)).tz_localize("UTC")
            revenue = float(100 + q + issuer)
            margin = 0.3 + 0.03 * np.sin(q + issuer) + issuer * 0.02
            rows.append(
                {
                    **dict.fromkeys(NUMERIC_COLUMNS, np.nan),
                    **dict.fromkeys(TIME_COLUMNS, pd.NaT),
                    **dict.fromkeys(FLAG_COLUMNS, False),
                    "event_id": f"{issuer}-{q}",
                    "company_id": f"{issuer + 1:010d}",
                    "ticker": f"S{issuer}",
                    "filing_accession": f"{issuer + 1:010d}-24-{q:06d}",
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
                    "derivation_method": "revenue_minus_total_cost"
                    if issuer == 0 or (issuer == 1 and q == 6)
                    else "direct_gross_profit",
                }
            )
    frame = pd.DataFrame(rows)
    for col in TIME_COLUMNS:
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


@pytest.fixture()
def cfg() -> PilotFilingFeatureConfig:
    return PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)


def test_quarantine_preserves_originals_and_masks_analytical_level(history: pd.DataFrame) -> None:
    before = history.copy(deep=True)
    view = scope_quarantine_view(history, ARMS[1], "0000000001")
    assert len(view) == len(history)
    assert view.scope_quarantined.sum() == 20
    assert view.loc[view.scope_quarantined, ["gross_margin", "gm_level"]].isna().all().all()
    pd.testing.assert_series_equal(
        view.source_gross_margin, history.gross_margin, check_names=False
    )
    pd.testing.assert_frame_equal(before, history)
    for col in ("event_id", "revenue", "gross_profit", "actual_available_at"):
        pd.testing.assert_series_equal(view[col], history[col])
    broad = scope_quarantine_view(history, ARMS[2], "0000000001")
    assert broad.scope_quarantined.sum() == 21
    with pytest.raises(ValueError, match="twice"):
        scope_quarantine_view(view, ARMS[1], "0000000001")
    with pytest.raises(ValueError, match="Unsupported"):
        scope_quarantine_view(history, "picked_best_return_arm", "0000000001")


def test_rebuild_excludes_quarantined_labels_lags_and_oos_errors(
    history: pd.DataFrame, cfg: PilotFilingFeatureConfig
) -> None:
    result = replay_scope_view(history, ARMS[2], "0000000001", cfg)
    excluded_ids = set(result.loc[result.scope_quarantined, "event_id"])
    assert not result.loc[result.scope_quarantined, list(FLAG_COLUMNS)].any().any()
    assert result.loc[result.scope_quarantined, "gm_forecast"].isna().all()
    # In particular, the year-after delta of issuer 2 cannot reuse quarantined q6.
    assert pd.isna(result.loc[result.event_id.eq("1-10"), "gm_delta_yoy"].iloc[0])
    assert pd.isna(result.loc[result.event_id.eq("1-11"), "company_gm_delta_yoy_lag1"].iloc[0])
    deps = target_dependencies(result, "1-19")
    assert not set(deps.event_id) & excluded_ids
    assert result.loc[result.event_id.eq("1-19"), "gm_forecast"].notna().all()


def test_future_quarantine_does_not_change_prior_forecasts(
    history: pd.DataFrame, cfg: PilotFilingFeatureConfig
) -> None:
    # Relabel only the last future record as cost-derived. Prior actuals/forecasts
    # must remain identical: excluding a future row cannot repair an earlier fit.
    baseline = replay_scope_view(history, ARMS[2], "0000000001", cfg)
    changed = history.copy()
    changed.loc[changed.event_id.eq("2-19"), "derivation_method"] = "revenue_minus_total_cost"
    replayed = replay_scope_view(changed, ARMS[2], "0000000001", cfg)
    earlier = history.actual_available_at.lt(history.iloc[-1].actual_available_at)
    for field in ("gm_forecast", "forecast_sigma", "gm_innovation_raw", "gm_innovation_std"):
        pd.testing.assert_series_equal(baseline.loc[earlier, field], replayed.loc[earlier, field])


def test_same_case_scope_signals_independent_reconstruction_and_no_replacement(
    history: pd.DataFrame, cfg: PilotFilingFeatureConfig
) -> None:
    histories = {arm: replay_scope_view(history, arm, "0000000001", cfg) for arm in ARMS}
    cases = [
        {
            "validation_event_id": f"v{i}",
            "cik": f"{i + 1:010d}",
            "assigned_ticker": f"S{i}",
            "target_accession_number": f"{i + 1:010d}-24-000019",
            "earliest_disclosure_at": history.loc[
                history.event_id.eq(f"{i}-19"), "actual_available_at"
            ]
            .iloc[0]
            .isoformat(),
        }
        for i in (1, 2)
    ]
    signals = scope_case_signals(histories, cases)
    assert len(signals) == 6
    assert signals.groupby("arm").validation_event_id.nunique().eq(2).all()
    broken = {a: f.copy(deep=True) for a, f in histories.items()}
    broken[ARMS[2]].loc[broken[ARMS[2]].event_id.eq("1-19"), "gm_innovation_std"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        scope_case_signals(broken, cases)


def recipe_fixture(q4: bool = False, direct: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small exact-decimal source facts with known original accession clocks."""
    stamp = pd.Timestamp("2025-02-01T12:00:00Z")
    end = pd.Timestamp("2024-12-31" if q4 else "2024-09-30")
    start = pd.Timestamp("2024-01-01" if q4 else "2024-07-01")
    facts = []
    for fid, tag, value in [
        ("r", "Revenues", "100.0"),
        ("c", "GrossProfit" if direct else "CostOfRevenue", "30.0" if direct else "70.0"),
    ]:
        facts.append(
            {
                "fact_id": fid,
                "cik": "0000000001",
                "taxonomy": "us-gaap",
                "unit": "USD",
                "tag": tag,
                "value": value,
                "period_start": start,
                "period_end": end,
                "accession_number": "original",
                "is_amendment": False,
                "accepted_at": stamp,
                "source_snapshot_sha256": "a" * 64,
            }
        )
    row: dict[str, Any] = {
        "event_id": "e",
        "company_id": "0000000001",
        "filing_accession": "original",
        "filing_accepted_at": stamp,
        "actual_available_at": stamp,
        "period_start": start,
        "period_end": end,
        "unit": "USD",
        "revenue": 100.0,
        "gross_profit": 30.0,
        "revenue_fact_id": "r",
        "gross_profit_fact_id": "c" if direct else None,
        "cost_fact_id": None if direct else "c",
        "derivation_method": "direct_gross_profit" if direct else "revenue_minus_total_cost",
        "q4_source_method": "not_applicable",
    }
    if q4:
        for original in facts[:]:
            facts.append(
                {
                    **original,
                    "fact_id": "y" + original["fact_id"],
                    "period_end": pd.Timestamp("2024-09-30"),
                    "accession_number": "prior-ytd",
                    "value": "80.0"
                    if original["fact_id"] == "r"
                    else ("24.0" if direct else "56.0"),
                    "accepted_at": pd.Timestamp("2024-11-01T12:00:00Z"),
                }
            )
        row.update(
            q4_source_method="annual_minus_original_nine_months",
            ytd_filing_accession="prior-ytd",
            ytd_available_at=pd.Timestamp("2024-11-01T12:00:00Z"),
            annual_revenue_fact_id="r",
            annual_component_fact_id="c",
            ytd_revenue_fact_id="yr",
            ytd_component_fact_id="yc",
            revenue_fact_id=_derived_fact_id("r", "yr"),
            revenue=20.0,
            gross_profit=6.0,
            period_start=pd.Timestamp("2024-10-01"),
        )
        row["gross_profit_fact_id" if direct else "cost_fact_id"] = _derived_fact_id("c", "yc")
    return pd.DataFrame([row]), pd.DataFrame(facts)


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("q4", [True, False])
def test_original_recipe_reproduced_without_scope_certification(q4: bool, direct: bool) -> None:
    result = audit_original_recipes(*recipe_fixture(q4, direct))
    assert result.original_recipe_verified.all()
    assert not result.cross_year_accounting_scope_certified.any()
    assert result.cost_recipe_scope_unverified.iloc[0] == (not direct)
    assert len(result.iloc[0].component_fact_ids) == (4 if q4 else 2)


@pytest.mark.parametrize(
    "violation", ["amendment", "future", "wrong_period", "wrong_value", "missing_id", "unit"]
)
def test_original_recipe_rejects_non_original_or_inconsistent_sources(violation: str) -> None:
    history, facts = recipe_fixture(q4=True)
    if violation == "amendment":
        facts.loc[0, "is_amendment"] = True
    elif violation == "future":
        facts.loc[2:, "accepted_at"] = pd.Timestamp("2025-03-01T12:00:00Z")
    elif violation == "wrong_period":
        facts.loc[1, "period_start"] = pd.Timestamp("2023-01-01")
    elif violation == "wrong_value":
        facts.loc[1, "value"] = "31.0"
    elif violation == "missing_id":
        facts = facts.iloc[1:]
    else:
        facts.loc[0, "unit"] = "EUR"
    with pytest.raises(ValueError):
        audit_original_recipes(history, facts)


@pytest.mark.parametrize("violation", [None, "missing", "future", "amendment", "different_value"])
def test_original_cross_tag_bridge_requires_strictly_prior_source_proof(
    violation: str | None,
) -> None:
    history, facts = recipe_fixture(q4=True, direct=False)
    facts.loc[1, "tag"] = "CostOfGoodsAndServicesSold"
    pair = pd.DataFrame(
        [
            {
                **facts.iloc[1].to_dict(),
                "fact_id": "b1",
                "value": "42.0",
                "period_start": pd.Timestamp("2023-01-01"),
                "period_end": pd.Timestamp("2023-03-31"),
                "accepted_at": pd.Timestamp("2023-05-01T12:00:00Z"),
            },
            {
                **facts.iloc[1].to_dict(),
                "fact_id": "b2",
                "tag": "CostOfRevenue",
                "value": "42.0",
                "period_start": pd.Timestamp("2023-01-01"),
                "period_end": pd.Timestamp("2023-03-31"),
                "accepted_at": pd.Timestamp("2023-05-01T12:00:00Z"),
            },
        ]
    )
    history["tag_bridge_evidence"] = json.dumps(
        [
            {
                "family": "component",
                "annual_tag": "CostOfGoodsAndServicesSold",
                "ytd_tag": "CostOfRevenue",
                "prior_evidence_fact_ids": ["b1", "b2"],
            }
        ]
    )
    if violation == "missing":
        history["tag_bridge_evidence"] = None
    elif violation == "future":
        pair["accepted_at"] = pd.Timestamp("2025-03-01T12:00:00Z")
    elif violation == "amendment":
        pair.loc[0, "is_amendment"] = True
    elif violation == "different_value":
        pair.loc[0, "value"] = "43.0"
    facts = pd.concat([facts, pair], ignore_index=True)
    if violation is None:
        result = audit_original_recipes(history, facts)
        assert result.original_recipe_verified.all()
        assert not result.cross_year_accounting_scope_certified.any()
    else:
        with pytest.raises(ValueError, match="bridge"):
            audit_original_recipes(history, facts)


def diagnostic_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identical three-case outcomes for all synthetic representation arms."""
    records = []
    for arm in ARMS:
        for i in range(3):
            records.append(
                {
                    "arm": arm,
                    "validation_event_id": str(i),
                    "cik": str(i),
                    "ticker": str(i),
                    "actual_available_at": "2025-01-01T12:00:00Z",
                    **{s: float(i) for s in SIGNALS},
                }
            )
    returns = [
        {
            "validation_event_id": str(i),
            "cik": str(i),
            "ticker": str(i),
            "announcement_at": "2025-01-01T12:00:00Z",
            "market_excess_return": i / 10,
        }
        for i in range(3)
    ]
    return pd.DataFrame(records), pd.DataFrame(returns)


def test_all_scope_representations_share_identical_event_outcome_sample() -> None:
    sample, stats, influence = scope_rank_diagnostics(*diagnostic_fixture())
    assert len(sample) == 9
    assert len(stats) == 12
    assert stats.spearman_ic.eq(1).all()
    assert not stats.inferential_test.any()
    assert len(influence) == 36


@pytest.mark.parametrize(
    "violation", ["missing_arm_case", "duplicate_return", "issuer", "clock", "nonfinite"]
)
def test_scope_diagnostic_cannot_hide_incomplete_cases_or_identity_errors(violation: str) -> None:
    signals, returns = diagnostic_fixture()
    if violation == "missing_arm_case":
        signals = signals.iloc[1:]
    elif violation == "duplicate_return":
        returns = pd.concat([returns, returns.iloc[[0]]], ignore_index=True)
    elif violation == "issuer":
        returns.loc[0, "cik"] = "wrong"
    elif violation == "clock":
        returns.loc[0, "announcement_at"] = "2025-01-02T12:00:00Z"
    else:
        signals.loc[0, SIGNALS[0]] = np.nan
    with pytest.raises(ValueError):
        scope_rank_diagnostics(signals, returns)


@pytest.mark.parametrize(
    "field",
    [
        "general_scope_gate_cleared",
        "hyperparameter_search_allowed",
        "numerical_history_repair_allowed",
        "sample_replacement_allowed",
        "network_allowed",
        "trading_admission",
        "inferential_tests_allowed",
    ],
)
def test_scope_contract_cannot_be_weakened_or_silently_promoted(field: str, tmp_path: Path) -> None:
    config = json.loads(Path("docs/margin_scope_replay_v1.json").read_bytes())
    weakened = copy.deepcopy(config)
    weakened[field] = True
    with pytest.raises(ValueError, match="weakened"):
        validate_contract(weakened)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(weakened), encoding="utf-8")
    with pytest.raises(ValueError, match="weakened"):
        build_scope_replay(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
