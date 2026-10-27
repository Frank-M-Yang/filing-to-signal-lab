"""Evidence-only OTC identity candidates; never impute a provider CIK or MIC."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any
from urllib.parse import urlencode

from sec_data.disclosure_review import audit_submission_clock
from sec_data.guidance_evidence import parse_complete_submission
from sec_data.massive_security_actions import massive_ticker_events_url
from sec_data.next_publication import DISARM_FLAGS

VERSION = "otc-identity-review-v1"


def review_sec_issuer_link(
    body: bytes, case: dict[str, Any], filename: str, issuer_name: str
) -> dict[str, Any]:
    """Bind one issuer/ticker assertion to an original accession and exhibit.

    An SEC acceptance timestamp remains an acceptance proxy, not first-public
    availability. This source documents a ticker, not its FIGI or venue MIC.
    """
    if hashlib.sha256(body).hexdigest() != case["release_pair_source_sha256"]:
        raise ValueError("SEC source hash changed")
    header = body.split(b"</SEC-HEADER>", 1)[0][:20000].decode("latin-1")
    if re.findall(r"CENTRAL INDEX KEY:\s*(\d+)", header) != [case["cik"]] or re.findall(
        r"COMPANY CONFORMED NAME:\s*([^\r\n]+)", header
    ) != [issuer_name]:
        raise ValueError("Ambiguous SEC issuer identity")
    clock = audit_submission_clock(
        body,
        {
            "cik": case["cik"],
            "accession_number": case["release_accession_number"],
            "form": "8-K",
            "accepted_at": case["release_accepted_at"],
        },
    )
    exhibits = [
        r
        for r in parse_complete_submission(body)
        if r["filename"] == filename
        and r["document_type"] == "EX-99.1"
        and r["document_sha256"] == case["release_pair_exhibit_sha256"]
    ]
    if len(exhibits) != 1 or clock["status"] != "consistent_acceptance_proxy":
        raise ValueError("Changed SEC exhibit or acceptance clock")
    anchor = f"(OTCQX: {case['ticker']})"
    text = re.sub(r"\s+", " ", exhibits[0]["visible_text"])
    if text.count(anchor) != 1:
        raise ValueError("Missing or ambiguous original ticker anchor")
    return {
        "issuer_cik": case["cik"],
        "ticker": case["ticker"],
        "issuer_name": issuer_name,
        "accession": case["release_accession_number"],
        "source_sha256": case["release_pair_source_sha256"],
        "exhibit_sha256": case["release_pair_exhibit_sha256"],
        "ticker_anchor": anchor,
        "sec_acceptance_proxy_at": clock["sec_acceptance_proxy_at"],
        "issuer_ticker_documented": True,
        "sec_proves_figi_or_mic": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def review_reference(
    query: dict[str, Any], body: bytes, sha256: str, target: dict[str, Any]
) -> dict[str, Any]:
    """Keep absent CIKs separate from contradictions and preserve raw venue labels.

    A matching share-class FIGI gives a diagnostic association to a frozen
    target; it never certifies a continuous historical security interval.
    """
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("Reference source hash changed")
    if (
        query["kind"] != "reference"
        or query["route"] not in {"ticker", "otc"}
        or query["ticker"] != target["selected_ticker"]
        or target["security_identity_key_kind"] != "share_class_figi"
        or not re.fullmatch(r"\d{10}", target["cik"])
        or not re.fullmatch(r"BBG[A-Z0-9]{9}", target["security_identity_key"])
    ):
        raise ValueError("Not an exact ticker reference control")
    date.fromisoformat(query["date"])
    result: dict[str, Any] = {
        "query_id": query["id"],
        "as_of_date": query["date"],
        "route": query["route"],
        "source_sha256": sha256,
        "candidate_match": False,
        "provider_cik": None,
        "provider_cik_state": "unobserved",
        "raw_primary_exchange": None,
        "observed_mic": None,
        "candidate_composite_figi": None,
        "issuer_expected_cik": target["cik"],
        "expected_ticker": target["selected_ticker"],
        "expected_mic": target["selected_exchange_mic"],
        "provider_cik_imputed": False,
        "venue_label_converted_to_mic": False,
        "historical_symbol_interval_ready": False,
        "provider_historical_view_not_vintage_evidence": True,
        "reasons": [],
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    try:
        d = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {**result, "reasons": ["malformed_json"]}
    if not isinstance(d, dict) or d.get("status") != "OK" or d.get("next_url"):
        return {**result, "reasons": ["incomplete_envelope"]}
    rows = d.get("results", [])
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return {**result, "reasons": ["empty_or_ambiguous_reference"]}
    row = rows[0]
    cik = row.get("cik")
    result["provider_cik"] = cik
    cik_state = "missing" if cik is None or cik == "" else "conflict"
    if isinstance(cik, str) and re.fullmatch(r"\d{1,10}", cik) and cik.zfill(10) == target["cik"]:
        cik_state = "match"
    result["provider_cik_state"] = cik_state
    for field, expected in (
        ("ticker", target["selected_ticker"]),
        ("share_class_figi", target["security_identity_key"]),
        ("name", target["issuer_name"]),
        ("market", "otc"),
        ("locale", "us"),
        ("type", "CS"),
    ):
        if row.get(field) != expected:
            result["reasons"].append(field + "_mismatch")
    if row.get("active") is not True:
        result["reasons"].append("active_state_missing_or_conflicting")
    if cik_state == "conflict":
        result["reasons"].append("provider_cik_conflict")
    figi = row.get("composite_figi")
    if not isinstance(figi, str) or not re.fullmatch(r"BBG[A-Z0-9]{9}", figi):
        result["reasons"].append("missing_or_invalid_composite_figi")
    result["observed_share_class_figi"] = row.get("share_class_figi")
    result["candidate_composite_figi"] = figi
    label = row.get("primary_exchange")
    result["raw_primary_exchange"] = label
    mic = label if isinstance(label, str) and re.fullmatch(r"[A-Z0-9]{4}", label) else None
    result["observed_mic"] = mic
    if mic is not None and mic != target["selected_exchange_mic"]:
        result["reasons"].append("observed_mic_conflict")
    result["candidate_match"] = not result["reasons"]
    return result


def review_consensus(rows: list[dict[str, Any]], sec: dict[str, Any]) -> dict[str, Any]:
    """Require both routes at both control dates, without extending their coverage."""
    coverage = {(r["as_of_date"], r["route"]) for r in rows}
    dates = sorted({r["as_of_date"] for r in rows})
    complete = (
        len(rows) == 4
        and len(coverage) == 4
        and len(dates) == 2
        and coverage == {(d, route) for d in dates for route in ("ticker", "otc")}
    )
    figis = {r["candidate_composite_figi"] for r in rows if r["candidate_match"]}
    candidate = (
        complete
        and all(r["candidate_match"] for r in rows)
        and len(figis) == 1
        and sec["issuer_ticker_documented"] is True
        and all(r["issuer_expected_cik"] == sec["issuer_cik"] for r in rows)
        and all(r["expected_ticker"] == sec["ticker"] for r in rows)
    )
    return {
        "candidate_association_supported": candidate,
        "control_dates": dates,
        "candidate_composite_figi": next(iter(figis)) if candidate else None,
        "historical_interval_ready": False,
        "control_dates_not_assumed_window_endpoints": True,
        "independent_sources_prove_all_links": False,
        "missing_requirements": [
            "exact_window_endpoints",
            "mic_evidence_or_reviewed_crosswalk",
            "ticker_timeline",
            "listing_outcome",
            "complete_prices",
            "historical_first_public_evidence",
        ],
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def followup_queries(
    consensus: dict[str, Any], ticker: str, start: str, end: str, cutoff: str
) -> list[dict[str, str]]:
    """Prepare at most five exact diagnostic requests; no network or price request."""
    if not consensus["candidate_association_supported"]:
        return []
    figi = consensus["candidate_composite_figi"]
    if not isinstance(figi, str) or not re.fullmatch(r"BBG[A-Z0-9]{9}", figi):
        raise ValueError("A source-observed composite FIGI is required")
    if not re.fullmatch(r"[A-Z0-9.\-/]{1,32}", ticker) or not (
        date.fromisoformat(start) <= date.fromisoformat(end) <= date.fromisoformat(cutoff)
    ):
        raise ValueError("Unsafe ticker or follow-up dates")
    jobs = []
    for kind, day, active in (
        ("reference_start", start, True),
        ("reference_end", end, True),
        ("listing_active", cutoff, True),
        ("listing_inactive", cutoff, False),
    ):
        params = {
            "ticker": ticker,
            "market": "otc",
            "date": day,
            "active": str(active).lower(),
            "order": "asc",
            "limit": 1000,
            "sort": "ticker",
        }
        jobs.append(
            {
                "kind": kind,
                "url": "https://api.massive.com/v3/reference/tickers?" + urlencode(params),
            }
        )
    jobs.append({"kind": "ticker_events", "url": massive_ticker_events_url(figi)})
    return jobs
