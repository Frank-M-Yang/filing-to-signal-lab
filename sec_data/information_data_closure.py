"""Offline cache closure, with SEC proxy clocks kept distinct from lab receipt clocks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.announcement_expansion import attach_prior_industry, extract_header_industry
from sec_data.disclosure_audit import utc_timestamp, verify_file
from sec_data.information_complexity import lagged_industry_features

VERSION = "information-data-closure-v1"
CORE_TICKERS = ("TTC", "USNA", "UFPT")


def cached_industries(
    cache: dict[tuple[str, str], dict[str, Any]], issuers: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify original bytes before parsing; do not substitute current company profiles.

    Acceptance is an operational disclosure proxy, not a certified first-public
    time. Late lab ingestion does not invalidate a retrospective header comparison,
    but cannot prove that the lab possessed that information before an old event.
    """
    observations, rejected = [], []
    for (cik, accession), meta in sorted(cache.items()):
        if cik not in issuers:
            continue
        path = Path(meta["path"])
        verify_file(path, meta["sha256"])
        if meta["status_code"] != 200 or path.stat().st_size != meta["bytes"]:
            raise ValueError("Invalid SEC cache receipt")
        received = utc_timestamp(meta["fetched_at"])
        try:
            row = extract_header_industry(path.read_bytes(), cik, accession)
        except ValueError as exc:
            rejected.append(
                {
                    "cik": cik,
                    "accession_number": accession,
                    "source_sha256": meta["sha256"],
                    "reason": str(exc),
                }
            )
            continue
        if received < utc_timestamp(row["industry_available_at"]):
            raise ValueError("SEC receipt precedes its acceptance timestamp")
        observations.append(
            {
                **row,
                "source_sha256": meta["sha256"],
                "source_url": meta["requested_url"],
                "lab_fetched_at": received.isoformat(),
                "first_public_time_certified": False,
            }
        )
    return observations, rejected


def industry_closure(
    events: list[dict[str, Any]], history: pd.DataFrame, observations: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Reuse the frozen lagged-peer definition and expose every selected financial/header link.

    Coverage is conditional on the existing original-value history and its scope
    quarantine, not a new validation of those peer financial facts or publication
    times. Retain targets with too few peers; never widen SIC or impute a value.
    """
    features = lagged_industry_features(events, history, observations)
    ledger = []
    for event, feature in zip(events, features.to_dict("records"), strict=True):
        cutoff = utc_timestamp(event["announcement_at"])
        bindings = attach_prior_industry(
            [
                {
                    "validation_event_id": c,
                    "cik": c,
                    "assigned_ticker": c,
                    "earliest_disclosure_at": cutoff,
                }
                for c in sorted(set(history.company_id) | {event["cik"]})
            ],
            observations,
        ).set_index("cik")
        target = bindings.loc[event["cik"]]
        codes: list[str] = (
            bindings.index[
                bindings.industry_ready & bindings.sic_2digit.eq(target.sic_2digit)
            ].tolist()
            if target.industry_ready
            else []
        )
        start = pd.to_datetime(event["period_start"], utc=True)
        candidate = history.loc[
            history.company_id.isin(codes)
            & history.company_id.ne(event["cik"])
            & pd.to_datetime(history.actual_available_at, utc=True).lt(cutoff)
            & pd.to_datetime(history.period_end, utc=True).lt(start)
            & pd.to_datetime(history.period_end, utc=True).ge(start - pd.Timedelta(days=365))
        ]
        peers = []
        for identity in feature["industry_peer_events"]:
            fact = history.loc[history.event_id.eq(identity)].iloc[0]
            header = bindings.loc[fact.company_id]
            peers.append(
                {
                    "financial_event_id": identity,
                    "cik": fact.company_id,
                    "filing_accession": fact.filing_accession,
                    "gross_profit_fact_id": fact.gross_profit_fact_id,
                    "period_end": str(fact.period_end),
                    "financial_available_at_proxy": utc_timestamp(
                        fact.actual_available_at
                    ).isoformat(),
                    "gm_level": float(fact.gm_level),
                    "derivation_method": fact.derivation_method,
                    "header_accessions": header.source_accessions,
                    "header_sha256": header.source_sha256,
                    "header_available_at_proxy": header.industry_available_at,
                }
            )
        quarantined = sorted(set(candidate.loc[candidate.scope_quarantined.eq(True), "company_id"]))
        ledger.append(
            {
                "validation_event_id": event["validation_event_id"],
                "ticker": event["ticker"],
                "sic_2digit": target.sic_2digit,
                "target_header_sha256": target.source_sha256,
                "candidate_peer_issuers_before_scope_filter": int(candidate.company_id.nunique()),
                "scope_quarantined_candidate_issuers": quarantined,
                "selected_peer_count": len(peers),
                "selected_peers": peers,
                "conditional_industry_state_ready": bool(feature["industry_state_ready"]),
                "financial_peer_source_admission_certified": False,
                "first_public_time_certified": False,
            }
        )
    return features, ledger


def receipt_audit(
    required: list[dict[str, Any]], receipts: list[dict[str, Any]], allowed_roots: list[Path]
) -> dict[str, Any]:
    """Bind required source hashes to exact bytes and local ingestion metadata, not live delivery.

    Missing receipts stay missing. Duplicate receipts are bounded conservatively by
    their latest timestamp. No exit-close timestamp is fabricated as label delivery;
    even a complete price-input audit does not certify a fully constructed label.
    """
    roles = [r["role"] for r in required]
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Receipt audit requires unique, nonempty roles")
    bound, missing = [], []
    for requirement in required:
        digest = requirement["source_sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", str(digest)):
            raise ValueError("Invalid required source hash")
        matching = [r for r in receipts if r["sha256"] == digest]
        if not matching:
            missing.append(requirement["role"])
            continue
        clocks = []
        for receipt in matching:
            path = Path(receipt["path"]).resolve()
            if not any(path.is_relative_to(root.resolve()) for root in allowed_roots):
                raise ValueError("Receipt escapes approved cache roots")
            verify_file(path, digest)
            if receipt["status_code"] != 200 or path.stat().st_size != receipt["bytes"]:
                raise ValueError("Invalid market receipt status or length")
            stamp = utc_timestamp(receipt["fetched_at"])
            if requirement.get("not_before") is not None and stamp < utc_timestamp(
                requirement["not_before"]
            ):
                raise ValueError("Market receipt precedes its required price observation")
            clocks.append(stamp)
        bound.append(
            {
                "role": requirement["role"],
                "source_sha256": digest,
                "lab_fetched_at": max(clocks).isoformat(),
                "verified_receipts": len(matching),
            }
        )
    return {
        "required_roles": len(required),
        "verified_roles": len(bound),
        "missing_roles": missing,
        "complete": not missing,
        "receipts": bound,
        "lab_inputs_received_by": max(utc_timestamp(r["lab_fetched_at"]) for r in bound).isoformat()
        if bound and not missing
        else None,
        "clock_basis": "local_receipt_metadata_not_independently_certified_vendor_delivery",
        "label_available_at": None,
        "historical_delivery_certified": False,
        "source_admitted": False,
    }


def missing_market_plan(audits: list[dict[str, Any]], as_of: str) -> list[dict[str, Any]]:
    """Plan only cache gaps; an advertised two-year Basic window is not an entitlement test."""
    boundary = (pd.Timestamp(as_of) - pd.DateOffset(years=2)).date().isoformat()
    rows = []
    for row in audits:
        tasks = []
        for field, task in (
            ("missing_issuer_dates", "issuer_daily_bars"),
            ("missing_benchmark_dates", "benchmark_daily_bars"),
            ("calendar_extension_required", "exchange_calendar_extension"),
        ):
            if row[field]:
                tasks.append(task)
        if not row["action_query_interval_covered"]:
            tasks.append("historical_identity_and_action_query_extension")
        rows.append(
            {
                "validation_event_id": row["validation_event_id"],
                "ticker": row["ticker"],
                "requested_start": row["requested_start"],
                "requested_end": row["requested_end"],
                "cache_gap_tasks": tasks,
                "redownload_cached_prices_needed": False,
                "start_precedes_advertised_basic_two_year_window": row["requested_start"]
                < boundary,
                "advertised_window_boundary_approximation": boundary,
                "actual_account_entitlement": "unknown_not_probed",
                "network_request_authorized_by_this_artifact": False,
            }
        )
    return rows
