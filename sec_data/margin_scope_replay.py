"""Original recipe provenance and explicit, retrospective scope quarantine views."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_replay import SIGNALS, case_row, verify_target_reconstruction
from sec_data.clock_layer import FLAG_COLUMNS, replay_history
from sec_data.disclosure_audit import utc_timestamp
from sec_data.pilot_filing_features import (
    PilotFilingFeatureConfig,
    _derived_fact_id,
)

VERSION = "margin-scope-replay-v1"
ARMS = (
    "legacy_scope_unresolved",
    "SUN_issuer_quarantined",
    "SUN_and_all_cost_derived_quarantined",
)


def _verify_original_tag_bridge(
    event: dict[str, Any], annual: pd.Series, ytd: pd.Series, lookup: pd.DataFrame, family: str
) -> None:
    """Verify recorded, strictly prior original cross-tag numeric evidence, not scope."""
    if annual.tag == ytd.tag:
        return
    raw = event.get("tag_bridge_evidence")
    if not isinstance(raw, str):
        raise ValueError("Different Q4 concepts require recorded prior original bridge evidence")
    evidence = [
        r
        for r in json.loads(raw)
        if r["family"] == family and r["annual_tag"] == annual.tag and r["ytd_tag"] == ytd.tag
    ]
    if len(evidence) != 1 or len(set(evidence[0]["prior_evidence_fact_ids"])) != 2:
        raise ValueError("Missing or ambiguous original tag bridge")
    ids = evidence[0]["prior_evidence_fact_ids"]
    if any(fid not in lookup.index for fid in ids):
        raise ValueError("Original tag bridge component missing")
    pair = lookup.loc[ids]
    if (
        set(pair.tag) != {annual.tag, ytd.tag}
        or pair.is_amendment.any()
        or not pair.cik.eq(event["company_id"]).all()
        or not pair.unit.eq(annual.unit).all()
        or not pair.taxonomy.eq(annual.taxonomy).all()
        or pair.period_start.nunique() != 1
        or pair.period_end.nunique() != 1
        or len({Decimal(str(v)) for v in pair.value}) != 1
        or not all(utc_timestamp(t) < utc_timestamp(annual.accepted_at) for t in pair.accepted_at)
    ):
        raise ValueError("Original tag bridge is not a strictly prior equivalent numeric pair")


def audit_original_recipes(history: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Reproduce each original quarter/FY-minus-YTD recipe, not certify economic scope.

    A direct GrossProfit tag is stronger provenance than a cost recipe, but this
    audit does not certify that historical accounting presentations are comparable.
    Amendments, later comparative facts and inferred zero values are forbidden.
    """
    if history.event_id.duplicated().any() or facts.fact_id.duplicated().any():
        raise ValueError("Duplicate original recipe identity")
    lookup = facts.set_index("fact_id", drop=False)
    rows = []
    for raw_event in history.to_dict("records"):
        event = {str(k): v for k, v in raw_event.items()}
        method = event["derivation_method"]
        if method not in {"direct_gross_profit", "revenue_minus_total_cost"}:
            raise ValueError("Unsupported original margin recipe")
        direct = method == "direct_gross_profit"
        derived_q4 = event["q4_source_method"] == "annual_minus_original_nine_months"
        refs = (
            (
                "annual_revenue_fact_id",
                "annual_component_fact_id",
                "ytd_revenue_fact_id",
                "ytd_component_fact_id",
            )
            if derived_q4
            else ("revenue_fact_id", "gross_profit_fact_id" if direct else "cost_fact_id")
        )
        selected = []
        for ref in refs:
            fact_id = event[ref]
            if pd.isna(fact_id) or fact_id not in lookup.index:
                raise ValueError("Original component fact missing; never impute")
            fact = lookup.loc[fact_id]
            ytd = ref.startswith("ytd_")
            accession = event["ytd_filing_accession"] if ytd else event["filing_accession"]
            available = event["ytd_available_at"] if ytd else event["filing_accepted_at"]
            if (
                fact.cik != event["company_id"]
                or fact.accession_number != accession
                or bool(fact.is_amendment)
                or fact.unit != event["unit"]
                or fact.taxonomy != "us-gaap"
                or utc_timestamp(fact.accepted_at) != utc_timestamp(available)
            ):
                raise ValueError("Recipe evidence is not the matched original information set")
            selected.append(fact)
        revenue, component = selected[:2]
        if direct and component.tag != "GrossProfit":
            raise ValueError("Direct gross-profit provenance has a different concept")
        if not direct and component.tag not in {"CostOfRevenue", "CostOfGoodsAndServicesSold"}:
            raise ValueError("Unsupported original cost concept")
        if (
            revenue.period_start != component.period_start
            or revenue.period_end != component.period_end
        ):
            raise ValueError("Original revenue/component periods differ")

        def value(fact: pd.Series) -> Decimal:
            return Decimal(str(fact.value))  # exact SEC decimal, not a rounding plug

        quarter_revenue, quarter_component = value(revenue), value(component)
        if derived_q4:
            ytd_revenue, ytd_component = selected[2:]
            if (
                ytd_revenue.period_start != revenue.period_start
                or ytd_component.period_start != revenue.period_start
                or ytd_revenue.period_end != ytd_component.period_end
                or not utc_timestamp(ytd_revenue.accepted_at) < utc_timestamp(revenue.accepted_at)
                or pd.Timestamp(event["period_start"])
                != ytd_revenue.period_end + pd.Timedelta(days=1)
            ):
                raise ValueError("Q4 requires a strictly earlier, same-concept original YTD pair")
            _verify_original_tag_bridge(event, revenue, ytd_revenue, lookup, "revenue")
            _verify_original_tag_bridge(event, component, ytd_component, lookup, "component")
            if (
                _derived_fact_id(str(revenue.fact_id), str(ytd_revenue.fact_id))
                != event["revenue_fact_id"]
                or _derived_fact_id(str(component.fact_id), str(ytd_component.fact_id))
                != event["gross_profit_fact_id" if direct else "cost_fact_id"]
            ):
                raise ValueError("Q4 derived component identities changed")
            quarter_revenue -= value(ytd_revenue)
            quarter_component -= value(ytd_component)
        elif revenue.period_start != pd.Timestamp(event["period_start"]):
            raise ValueError("Original direct quarter start differs")
        if revenue.period_end != pd.Timestamp(event["period_end"]):
            raise ValueError("Original recipe period end differs")
        gp = quarter_component if direct else quarter_revenue - quarter_component
        if quarter_revenue != Decimal(str(event["revenue"])) or gp != Decimal(
            str(event["gross_profit"])
        ):
            raise ValueError("Original recipe does not exactly reproduce stored amounts")
        rows.append(
            {
                "event_id": event["event_id"],
                "company_id": event["company_id"],
                "filing_accession": event["filing_accession"],
                "period_start": event["period_start"],
                "period_end": event["period_end"],
                "actual_available_at": event["actual_available_at"],
                "derivation_method": method,
                "q4_source_method": event["q4_source_method"],
                "revenue": float(quarter_revenue),
                "stored_gross_profit": float(gp),
                "component_tag": component.tag,
                "component_fact_ids": [str(f.fact_id) for f in selected],
                "component_accessions": [str(f.accession_number) for f in selected],
                "component_accepted_at": [
                    utc_timestamp(f.accepted_at).isoformat() for f in selected
                ],
                "source_hashes": sorted({str(f.source_snapshot_sha256) for f in selected}),
                "original_recipe_verified": True,
                "cross_year_accounting_scope_certified": False,
                "cost_recipe_scope_unverified": not direct,
                "recorded_tag_bridge_evidence": event.get("tag_bridge_evidence"),
            }
        )
    return pd.DataFrame(rows)


def scope_quarantine_view(history: pd.DataFrame, arm: str, issuer_cik: str) -> pd.DataFrame:
    """Retain every source row; mask analytical margin values, never repair originals.

    Readiness flags alone are insufficient: old numeric deltas can otherwise
    leak through lagged company features. Mask both level and margin, then rebuild
    all lags, forecasts and strictly prior OOS errors from this view.
    """
    if arm not in ARMS or history.event_id.duplicated().any():
        raise ValueError("Unsupported scope arm or duplicate history event")
    if "source_gross_margin" in history or "scope_quarantined" in history:
        raise ValueError("Do not apply scope quarantine twice")
    result = history.copy(deep=True)
    mask = result.company_id.eq(issuer_cik) & (arm != ARMS[0])
    if arm == ARMS[2]:
        mask |= result.derivation_method.eq("revenue_minus_total_cost")
    result["source_gross_margin"] = result.gross_margin
    result["source_gm_level"] = result.gm_level
    result["scope_quarantined"] = mask
    result["scope_arm"] = arm
    result.loc[mask, ["gross_margin", "gm_level"]] = np.nan
    result.loc[mask, list(FLAG_COLUMNS)] = False
    return result


def replay_scope_view(
    history: pd.DataFrame, arm: str, issuer_cik: str, config: PilotFilingFeatureConfig
) -> pd.DataFrame:
    """Fixed-spec refit with quarantined rows barred from labels, lags and OOS sigma."""
    view = scope_quarantine_view(history, arm, issuer_cik)
    replayed = replay_history(view, config)
    source_columns = [
        "event_id",
        "company_id",
        "filing_accession",
        "filing_accepted_at",
        "actual_available_at",
        "revenue",
        "gross_profit",
        "period_start",
        "period_end",
    ]
    pd.testing.assert_frame_equal(
        history[source_columns].reset_index(drop=True),
        replayed[source_columns].reset_index(drop=True),
    )
    excluded = replayed.scope_quarantined
    if (
        replayed.loc[excluded, list(FLAG_COLUMNS)].any().any()
        or replayed.loc[excluded, ["gm_forecast", "gm_innovation_raw", "gm_innovation_std"]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("Quarantined margin entered forecasts or OOS error history")
    sources = set(replayed.loc[excluded, "event_id"])
    admitted = replayed.loc[replayed.analysis_ready_delta]
    if admitted.gm_yoy_lag_event_id.isin(sources).any():
        raise ValueError("Quarantined source entered an admitted year-ago lag")
    return replayed


def scope_case_signals(
    histories: dict[str, pd.DataFrame], cases: list[dict[str, Any]]
) -> pd.DataFrame:
    """Require all frozen cases to be complete in every arm, never select on outcomes."""
    if tuple(histories) != ARMS or len({c["validation_event_id"] for c in cases}) != len(cases):
        raise ValueError("Scope arms or common case identity changed")
    rows = []
    for arm, history in histories.items():
        for case in cases:
            target = case_row(history, case)
            if (
                not all(bool(target[f]) for f in FLAG_COLUMNS)
                or not np.isfinite(target[list(SIGNALS)].to_numpy(dtype=float)).all()
            ):
                raise ValueError("Frozen common-case forecast incomplete; do not drop or replace")
            verify_target_reconstruction(history, case)
            if utc_timestamp(target.actual_available_at) != utc_timestamp(
                case["earliest_disclosure_at"]
            ):
                raise ValueError("Scope replay changed the announcement clock")
            rows.append(
                {
                    "arm": arm,
                    "validation_event_id": case["validation_event_id"],
                    "cik": case["cik"],
                    "ticker": case["assigned_ticker"],
                    "actual_available_at": target.actual_available_at,
                    **{s: float(target[s]) for s in SIGNALS},
                    "gm_forecast": float(target.gm_forecast),
                    "forecast_sigma": float(target.forecast_sigma),
                    "training_observations": int(target.forecast_training_observations),
                    "prior_oos_errors": int(target.forecast_sigma_observations),
                }
            )
    return pd.DataFrame(rows)


def scope_rank_diagnostics(
    signals: pd.DataFrame, returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Known six-case descriptive IC and leave-one-issuer-out; no significance or Alpha."""
    if returns.validation_event_id.duplicated().any() or set(signals.arm) != set(ARMS):
        raise ValueError("Invalid same-case return identities or scope arms")
    for _, group in signals.groupby("arm", sort=False):
        if group.validation_event_id.duplicated().any() or set(group.validation_event_id) != set(
            returns.validation_event_id
        ):
            raise ValueError("All arms and representations require the same complete-case sample")
    sample = signals.merge(
        returns, on="validation_event_id", suffixes=("", "_return"), validate="many_to_one"
    )
    if (
        not sample.cik.eq(sample.cik_return).all()
        or not sample.ticker.eq(sample.ticker_return).all()
        or not np.isfinite(sample[[*SIGNALS, "market_excess_return"]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Incomplete or mismatched same-case outcomes")
    for actual, announced in zip(sample.actual_available_at, sample.announcement_at, strict=True):
        if utc_timestamp(actual) != utc_timestamp(announced):
            raise ValueError("Same-case information timestamps differ")

    def ic(group: pd.DataFrame, signal: str) -> float | None:
        x, y = group[signal], group.market_excess_return
        return (
            float(x.rank().corr(y.rank()))
            if len(group) >= 3 and x.nunique() > 1 and y.nunique() > 1
            else None
        )

    stats, influence = [], []
    for arm, group in sample.groupby("arm", sort=False):
        for signal in SIGNALS:
            stats.append(
                {
                    "arm": arm,
                    "representation": signal,
                    "events": len(group),
                    "spearman_ic": ic(group, signal),
                    "inferential_test": False,
                }
            )
            for cik in sorted(group.cik.unique()):
                reduced = group.loc[group.cik.ne(cik)]
                influence.append(
                    {
                        "arm": arm,
                        "representation": signal,
                        "removed_cik": cik,
                        "events": len(reduced),
                        "spearman_ic": ic(reduced, signal),
                    }
                )
    return sample, pd.DataFrame(stats), pd.DataFrame(influence)
