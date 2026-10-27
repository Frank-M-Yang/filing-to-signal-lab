"""Outcome-blind source expansion; partial releases never imply a complete information set."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import earnings_candidates, submission_url, utc_timestamp

VERSION = "next-disclosure-review-v1"
QUEUE_COLUMNS = (
    "validation_event_id",
    "cik",
    "assigned_ticker",
    "assigned_exchange_mic",
    "target_accession_number",
    "release_accession_number",
    "earliest_disclosure_at",
)


def freeze_sources(
    queue: list[dict[str, Any]], history: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the fixed twenty issuers, all bounded 2.02/7.01 candidates, and missing-release reports.

    Periodic reports remain explicitly labelled fallbacks, never evidence that no
    earlier announcement existed. Both candidate discovery and acquisition order
    depend only on original identities, periods and catalog clocks, not values.
    """
    cases = [{k: r[k] for k in QUEUE_COLUMNS} for r in queue]
    if (
        len(cases) != 20
        or len({r["cik"] for r in cases}) != 20
        or len({r["validation_event_id"] for r in cases}) != 20
    ):
        raise ValueError("Expected the frozen twenty distinct issuers and events")
    events: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for case in cases:
        rows = history.loc[
            history.company_id.eq(case["cik"])
            & history.filing_accession.eq(case["target_accession_number"])
        ]
        reports = catalog.loc[
            catalog.cik.eq(case["cik"])
            & catalog.accession_number.eq(case["target_accession_number"])
        ]
        if len(rows) != 1 or len(reports) != 1:
            raise ValueError("Frozen target financial/catalog identity is missing or ambiguous")
        row, report = rows.iloc[0], reports.iloc[0]
        if row.derivation_method not in {"direct_gross_profit", "revenue_minus_total_cost"}:
            raise ValueError("Unknown margin derivation requires an explicit scope review")
        if (
            row.filing_form != report.form
            or str(row.period_end)[:10] != str(report.report_date)[:10]
        ):
            raise ValueError("Frozen report form or financial period mismatch")
        event = {
            **case,
            "ticker": case["assigned_ticker"],
            "filing_accession": case["target_accession_number"],
            "accepted_at": utc_timestamp(report.accepted_at).isoformat(),
            "form": row.filing_form,
            "period_start": str(row.period_start)[:10],
            "period_end": str(row.period_end)[:10],
            "derivation_method": row.derivation_method,
            "existing_scope_quarantine": row.derivation_method == "revenue_minus_total_cost",
        }
        found = earnings_candidates(event, catalog)
        if case["release_accession_number"] is not None and case[
            "release_accession_number"
        ] not in {r["accession_number"] for r in found}:
            raise ValueError("Frozen release is outside the bounded candidate catalog")
        candidates.extend({**r, "source_role": "announcement_candidate"} for r in found)
        if case["release_accession_number"] is None:
            candidates.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "cik": case["cik"],
                    "accession_number": case["target_accession_number"],
                    "form": row.filing_form,
                    "accepted_at": event["accepted_at"],
                    "url": submission_url(case["cik"], case["target_accession_number"]),
                    "source_role": "periodic_report_fallback_not_first_public_proof",
                }
            )
        events.append(event)
    targets = []
    for number, candidate in enumerate(candidates, start=1):
        identity = candidate["cik"] + "|" + candidate["accession_number"]
        targets.append(
            {
                "guidance_document_id": hashlib.sha256(identity.encode()).hexdigest(),
                "cik": candidate["cik"],
                "prior_accession_number": candidate["accession_number"],
                "complete_submission_url": candidate["url"],
                "document_acquisition_order": number,
            }
        )
    if len({r["guidance_document_id"] for r in targets}) != len(targets) or len(targets) > 32:
        raise ValueError("Duplicate source identity or frozen request budget exceeded")
    return events, candidates, targets


def pair_dispositions(
    events: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    reasons: dict[str, str],
) -> list[dict[str, Any]]:
    """Retain all twenty; bind exact same-document GAAP pairs without erasing earlier candidates.

    Reviewer assertions are conditional source comparisons, not independent scope
    or first-public certification. A later complete release cannot backfill the
    earliest preliminary announcement's information set.
    """
    identities = {e["validation_event_id"] for e in events}
    if (
        len(events) != 20
        or len(identities) != 20
        or len({e["cik"] for e in events}) != 20
        or any(r["validation_event_id"] not in identities for r in cells + documents)
    ):
        raise ValueError("Review escaped its frozen event denominator")
    if set(reasons) - {e["ticker"] for e in events}:
        raise ValueError("Disposition reason is outside the frozen batch")
    output = []
    for event in events:
        matched = [r for r in cells if r["validation_event_id"] == event["validation_event_id"]]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in matched:
            groups.setdefault(row["release_accession"], []).append(row)
        pairs = []
        for accession, group in groups.items():
            if len(group) != 2 or {r["metric"] for r in group} != {"revenue", "gross_profit"}:
                raise ValueError(
                    "Every reviewed release needs one revenue and one gross-profit cell"
                )
            if (
                len(
                    {
                        (r["source_sha256"], r["exhibit_sha256"], r["sec_acceptance_proxy_at"])
                        for r in group
                    }
                )
                != 1
            ):
                raise ValueError("Revenue and gross profit are not the same source information set")
            sources = [
                d
                for d in documents
                if d["validation_event_id"] == event["validation_event_id"]
                and d["release_accession"] == accession
                and d["kind"] == "earnings_release"
            ]
            if len(sources) != 1 or any(
                r["cik"] != event["cik"]
                or r["source_sha256"] != sources[0]["submission_sha256"]
                or r["exhibit_sha256"] != sources[0]["exhibit_sha256"]
                for r in group
            ):
                raise ValueError("Financial cells are not bound to their reviewed source identity")
            source_clock = sources[0]["clock"]
            if any(
                r["clock_status"] != source_clock["status"]
                or r["sec_acceptance_proxy_at"] != source_clock["sec_acceptance_proxy_at"]
                for r in group
            ):
                raise ValueError("Financial cell clock differs from its audited raw source")
            consistent = all(
                r["clock_status"] == "consistent_acceptance_proxy" and r["sec_acceptance_proxy_at"]
                for r in group
            )
            exact = all(r["comparison_status"] == "exact_original_match" for r in group)
            revenue = next(r["value"] for r in group if r["metric"] == "revenue")
            gp = next(r["value"] for r in group if r["metric"] == "gross_profit")
            if (
                any(isinstance(v, bool) or not math.isfinite(float(v)) for v in (revenue, gp))
                or revenue <= 0
            ):
                raise ValueError("A gross-margin denominator must be positive")
            proxy = group[0]["sec_acceptance_proxy_at"] if consistent else None
            pairs.append(
                {
                    "release_accession": accession,
                    "submission_sha256": group[0]["source_sha256"],
                    "exhibit_sha256": group[0]["exhibit_sha256"],
                    "revenue": revenue,
                    "gross_profit": gp,
                    "gross_margin": gp / revenue,
                    "pair_acceptance_proxy_at": proxy,
                    "exact_original_pair": exact,
                    "clock_consistent": consistent,
                    "conditional_pair_verified": bool(
                        exact and consistent and not event["existing_scope_quarantine"]
                    ),
                    "later_than_frozen_earliest_proxy": utc_timestamp(proxy)
                    > utc_timestamp(event["earliest_disclosure_at"])
                    if proxy
                    else None,
                }
            )
        verified = [p for p in pairs if p["conditional_pair_verified"]]
        status = (
            "conditional_original_pair_verified_not_admitted"
            if verified
            else reasons.get(event["ticker"])
        )
        if status is None:
            raise ValueError("Every unverified issuer needs an explicit unresolved disposition")
        output.append(
            {
                **event,
                "status": status,
                "reviewed_pairs": pairs,
                "conditional_pair_verified": bool(verified),
                "pair_acceptance_proxy_at": min(
                    (p["pair_acceptance_proxy_at"] for p in verified), key=utc_timestamp
                )
                if verified
                else None,
                "first_public_certified": False,
                "financial_forecast_replayed": False,
                "model_admission": False,
                "return_admission": False,
                "trading_admission": False,
            }
        )
    return output
