"""One proven scope discrepancy must not silently relabel or repair the full history."""

from __future__ import annotations

from typing import Any

import pytest

from sec_data.announcement_scope_audit import audit_margin_scope, bind_reconciliation_value


@pytest.fixture()
def scope_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history = {
        "company_id": "0001552275",
        "filing_accession": "0001552275-24-000086",
        "period_start": "2024-07-01",
        "period_end": "2024-09-30",
        "revenue": 5751000000,
        "gross_profit": 424000000,
        "derivation_method": "revenue_minus_total_cost",
        "gross_profit_fact_id": None,
    }
    cost = {
        "accn": history["filing_accession"],
        "form": "10-Q",
        "tag": "CostOfRevenue",
        "unit": "USD",
        "start": history["period_start"],
        "end": history["period_end"],
        "val": 5327000000,
    }
    return history, [cost]


def test_exact_scope_reconciliation_stops_inference_not_all_history(
    scope_evidence: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    history, costs = scope_evidence
    proof = audit_margin_scope(history, costs, 331000000, 424000000, 93000000)
    assert proof["scope_gap_USD"] == 93000000
    assert proof["margin_gap_percentage_points"] == pytest.approx(93 / 5751 * 100)
    assert proof["recipe_matches_segment_not_reported_GAAP_GP"]
    assert proof["new_model_return_diagnostic_blocked"]
    assert not proof["automatic_repair_allowed"]
    assert not proof["all_issuer_history_proven_wrong"]
    assert history["gross_profit"] == 424000000


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("accn", "0001552275-25-000001"),
        ("form", "10-Q/A"),
        ("unit", "EUR"),
        ("start", "2024-01-01"),
        ("end", "2023-09-30"),
        ("tag", "OperatingExpenses"),
    ],
)
def test_scope_comparison_requires_identical_original_quarter(
    scope_evidence: tuple[dict[str, Any], list[dict[str, Any]]], key: str, value: str
) -> None:
    history, costs = scope_evidence
    with pytest.raises(ValueError, match="same original quarter"):
        audit_margin_scope(history, [{**costs[0], key: value}], 331000000, 424000000, 93000000)


def test_scope_rejects_ambiguous_cost_nonfinite_and_wrong_recipe(
    scope_evidence: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    history, costs = scope_evidence
    with pytest.raises(ValueError, match="Ambiguous"):
        audit_margin_scope(
            history, [*costs, {**costs[0], "val": 5328000000}], 331000000, 424000000, 93000000
        )
    with pytest.raises(ValueError, match="identities"):
        audit_margin_scope(history, costs, 332000000, 424000000, 93000000)
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_margin_scope(
            {**history, "revenue": float("nan")}, costs, 331000000, 424000000, 93000000
        )
    with pytest.raises(ValueError, match="explicitly derived"):
        audit_margin_scope(
            {**history, "gross_profit_fact_id": "direct-fact"},
            costs,
            331000000,
            424000000,
            93000000,
        )
    with pytest.raises(ValueError, match="scope discrepancy"):
        audit_margin_scope(history, costs, 424000000, 424000000, 0)


def test_reconciliation_scalar_bound_to_current_quarter_cell_not_arbitrary_config() -> None:
    anchor = "Total segment profit 424 527 1,473 1,216"
    assert (
        bind_reconciliation_value(anchor, anchor, "Total segment profit", "424", 1000000)
        == 424000000
    )
    for text, value, scale in [
        (anchor, "527", 1000000),
        (anchor * 2, "424", 1000000),
        ("missing", "424", 1000000),
        (anchor, "424", 0),
    ]:
        with pytest.raises(ValueError):
            bind_reconciliation_value(text, anchor, "Total segment profit", value, scale)
