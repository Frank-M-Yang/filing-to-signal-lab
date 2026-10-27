"""A fail-closed accounting-scope gate; no history repair or performance optimization."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any


def bind_reconciliation_value(
    text: str, anchor: str, label: str, display_value: str, scale: int
) -> float:
    """Bind a reviewed current-quarter value to one exact original reconciliation row."""
    if scale <= 0 or text.count(anchor) != 1:
        raise ValueError("Reconciliation anchor is missing or ambiguous")
    if not re.match(
        rf"^{re.escape(label)}\s+{re.escape(display_value)}(?=\s|$)", anchor
    ) or not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", display_value):
        raise ValueError("Reconciliation value is not the first reviewed current-quarter cell")
    return float(Decimal(display_value.replace(",", "")) * scale)


def audit_margin_scope(
    history: dict[str, Any],
    original_cost_rows: list[dict[str, Any]],
    reported_gp: float,
    reported_segment_profit: float,
    depreciation: float,
) -> dict[str, Any]:
    """Prove one same-quarter recipe/scope discrepancy without generalizing to all years."""
    if (
        history["derivation_method"] != "revenue_minus_total_cost"
        or history.get("gross_profit_fact_id") is not None
        or not original_cost_rows
    ):
        raise ValueError("Expected an explicitly derived original gross-profit observation")
    amounts = set()
    for fact in original_cost_rows:
        if (
            fact["accn"] != history["filing_accession"]
            or fact["form"] != "10-Q"
            or fact["start"] != history["period_start"]
            or fact["end"] != history["period_end"]
            or fact["tag"] not in {"CostOfRevenue", "CostOfGoodsAndServicesSold"}
            or fact["unit"] != "USD"
        ):
            raise ValueError("Scope audit cost evidence is not the same original quarter")
        amounts.add(Decimal(str(fact["val"])))
    if len(amounts) != 1:
        raise ValueError("Ambiguous original cost value; do not choose one")
    revenue, derived = Decimal(str(history["revenue"])), Decimal(str(history["gross_profit"]))
    cost = next(iter(amounts))
    gp, segment, da = (
        Decimal(str(v)) for v in (reported_gp, reported_segment_profit, depreciation)
    )
    if not all(v.is_finite() for v in (revenue, derived, cost, gp, segment, da)) or revenue <= 0:
        raise ValueError("Nonfinite scope evidence")
    if revenue - cost != derived or derived != segment or segment - da != gp:
        raise ValueError("The stated source identities do not reproduce exactly")
    if da <= 0 or derived == gp:
        raise ValueError("Reconciliation does not prove the stated scope discrepancy")
    return {
        "company_id": history["company_id"],
        "filing_accession": history["filing_accession"],
        "period_start": history["period_start"],
        "period_end": history["period_end"],
        "derivation_method": history["derivation_method"],
        "revenue": float(revenue),
        "original_cost": float(cost),
        "history_gross_profit": float(derived),
        "reported_gaap_gross_profit": float(gp),
        "reported_segment_profit": float(segment),
        "depreciation_excluding_corporate": float(da),
        "scope_gap_USD": float(derived - gp),
        "margin_gap_percentage_points": float((derived - gp) / revenue * 100),
        "recipe_reproduces_original_cost": True,
        "recipe_matches_segment_not_reported_GAAP_GP": True,
        "all_issuer_history_proven_wrong": False,
        "automatic_repair_allowed": False,
        "new_model_return_diagnostic_blocked": True,
        "status": "observed_GAAP_margin_scope_discrepancy_requires_history_audit",
    }
