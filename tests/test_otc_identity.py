"""Preserve missing provider IDs, ambiguous records, independent sources and gates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_sec_clock_layer import _encode
from scripts.review_otc_identity import validate_contract
from sec_data.guidance_evidence import parse_complete_submission
from sec_data.next_publication import DISARM_FLAGS
from sec_data.otc_identity import (
    followup_queries,
    review_consensus,
    review_reference,
    review_sec_issuer_link,
)


def target() -> dict[str, Any]:
    """A synthetic frozen security, never a live market query."""
    return {
        "cik": "0000000001",
        "selected_ticker": "TEST",
        "selected_exchange_mic": "OTCM",
        "security_identity_key": "BBG000000001",
        "security_identity_key_kind": "share_class_figi",
        "issuer_name": "SYNTHETIC ISSUER",
    }


def reference() -> dict[str, Any]:
    """Deliberately omit CIK; retain a non-MIC provider venue label."""
    return {
        "status": "OK",
        "results": [
            {
                "ticker": "TEST",
                "name": "SYNTHETIC ISSUER",
                "share_class_figi": "BBG000000001",
                "composite_figi": "BBG000000002",
                "primary_exchange": "OTC Link",
                "market": "otc",
                "locale": "us",
                "type": "CS",
                "active": True,
            }
        ],
    }


def parse(data: Any, route: str = "ticker", day: str = "2024-09-17") -> dict[str, Any]:
    body = _encode(data)
    query = {"kind": "reference", "id": day + route, "ticker": "TEST", "date": day, "route": route}
    return review_reference(query, body, hashlib.sha256(body).hexdigest(), target())


def candidates() -> list[dict[str, Any]]:
    """Both frozen diagnostic routes at two control dates."""
    return [
        parse(reference(), r, d) for d in ("2024-09-17", "2025-10-08") for r in ("ticker", "otc")
    ]


def sec_link() -> dict[str, Any]:
    """Independent issuer-to-ticker assertion, not a provider CIK fill value."""
    return {"issuer_ticker_documented": True, "issuer_cik": "0000000001", "ticker": "TEST"}


def test_missing_cik_and_non_mic_are_preserved_without_authorization() -> None:
    row = parse(reference())
    assert row["candidate_match"] and row["provider_cik"] is None
    assert row["provider_cik_state"] == "missing" and row["issuer_expected_cik"] == "0000000001"
    assert row["raw_primary_exchange"] == "OTC Link" and row["observed_mic"] is None
    assert not row["provider_cik_imputed"] and not row["venue_label_converted_to_mic"]
    assert not row["historical_symbol_interval_ready"] and all(
        row[k] is False for k in DISARM_FLAGS
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticker", "OTHER"),
        ("share_class_figi", "BBG000000009"),
        ("name", "DIFFERENT ISSUER"),
        ("market", "stocks"),
        ("locale", "global"),
        ("type", "PFD"),
        ("active", None),
        ("active", 1),
        ("cik", "0000000002"),
        ("cik", "bad"),
        ("cik", 1),
        ("composite_figi", None),
        ("composite_figi", []),
        ("primary_exchange", "XNAS"),
    ],
)
def test_conflicting_or_weak_reference_rows_are_quarantined(field: str, value: Any) -> None:
    data = reference()
    data["results"][0][field] = value
    result = parse(data)
    assert not result["candidate_match"] and result["reasons"]
    assert not review_consensus([result, *candidates()[1:]], sec_link())[
        "candidate_association_supported"
    ]


@pytest.mark.parametrize("cik", [None, "", "1", "0000000001"])
def test_absence_and_agreement_remain_distinct(cik: Any) -> None:
    data = reference()
    data["results"][0]["cik"] = cik
    result = parse(data)
    assert result["candidate_match"] and result["provider_cik"] == cik
    assert result["provider_cik_state"] == ("missing" if cik in (None, "") else "match")


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "wrong_schema", "pagination", "status"])
def test_missing_ambiguous_or_paginated_reference_not_accepted(mutation: str) -> None:
    data = reference()
    if mutation == "empty":
        data["results"] = []
    elif mutation == "duplicate":
        data["results"] *= 2
    elif mutation == "wrong_schema":
        data["results"] = [{}]
    elif mutation == "pagination":
        data["next_url"] = "https://example.org/next"
    else:
        data["status"] = "NOT_AUTHORIZED"
    assert not parse(data)["candidate_match"]


def test_sources_are_hash_bound_and_wrong_query_does_not_parse() -> None:
    q = {"kind": "reference", "route": "ticker", "ticker": "TEST", "date": "2024-09-17", "id": "a"}
    raw = _encode(reference())
    with pytest.raises(ValueError, match="hash changed"):
        review_reference(q, raw, "a" * 64, target())
    q["ticker"] = "OTHER"
    with pytest.raises(ValueError, match="exact ticker"):
        review_reference(q, raw, hashlib.sha256(raw).hexdigest(), target())


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "figi_conflict", "different_issuer", "different_ticker"]
)
def test_consensus_requires_both_routes_dates_figis_and_independent_issuer(mutation: str) -> None:
    rows, sec = candidates(), sec_link()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[3] = copy.deepcopy(rows[0])
    elif mutation == "figi_conflict":
        rows[3]["candidate_composite_figi"] = "BBG000000003"
    elif mutation == "different_issuer":
        sec["issuer_cik"] = "0000000002"
    else:
        sec["ticker"] = "OTHER"
    result = review_consensus(rows, sec)
    assert not result["candidate_association_supported"]
    assert not followup_queries(result, "TEST", "2024-09-11", "2024-10-09", "2026-08-14")


def test_positive_candidate_only_prepares_bounded_secret_free_diagnostic_queries() -> None:
    result = review_consensus(candidates(), sec_link())
    assert result["candidate_association_supported"] and not result["historical_interval_ready"]
    jobs = followup_queries(result, "TEST", "2024-09-11", "2024-10-09", "2026-08-14")
    assert len(jobs) == 5 and all(
        "apiKey" not in q["url"] and "/None/" not in q["url"] for q in jobs
    )
    assert all("cik=" not in q["url"] for q in jobs)
    assert all("aggs" not in q["url"] for q in jobs)
    assert "BBG000000002" in jobs[-1]["url"]
    with pytest.raises(ValueError, match="dates"):
        followup_queries(result, "TEST", "2024-09-11", "2024-10-09", "2024-09-14")
    result["candidate_composite_figi"] = None
    with pytest.raises(ValueError, match="source-observed"):
        followup_queries(result, "TEST", "2024-09-11", "2024-10-09", "2026-08-14")


def original_sec() -> tuple[bytes, dict[str, Any]]:
    """A complete, tiny synthetic accession and independently hashed release exhibit."""
    body = (
        b"<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20240911091629\n"
        b"ACCESSION NUMBER: 0000000001-24-000001\nCONFORMED SUBMISSION TYPE: 8-K\n"
        b"COMPANY CONFORMED NAME: SYNTHETIC ISSUER\nCENTRAL INDEX KEY: 0000000001\n</SEC-HEADER>\n"
        b"<DOCUMENT>\n<TYPE>EX-99.1\n<FILENAME>release.htm\n<TEXT>"
        b"<html><p>Synthetic issuer (OTCQX: TEST) announces results.</p></html>"
        b"</TEXT>\n</DOCUMENT>"
    )
    case = {
        "release_pair_source_sha256": hashlib.sha256(body).hexdigest(),
        "cik": "0000000001",
        "ticker": "TEST",
        "release_accession_number": "0000000001-24-000001",
        "release_accepted_at": "2024-09-11T13:16:29Z",
        "release_pair_exhibit_sha256": parse_complete_submission(body)[0]["document_sha256"],
    }
    return body, case


def test_original_sec_link_checks_header_acceptance_and_ticker_without_certifying_figi() -> None:
    body, case = original_sec()
    result = review_sec_issuer_link(body, case, "release.htm", "SYNTHETIC ISSUER")
    assert result["issuer_ticker_documented"] and not result["sec_proves_figi_or_mic"]
    assert result["sec_acceptance_proxy_at"] == "2024-09-11T13:16:29+00:00"
    assert all(result[k] is False for k in DISARM_FLAGS)


@pytest.mark.parametrize(
    "mutation",
    ["raw_hash", "exhibit_hash", "cik", "clock", "ticker", "accession", "filename", "name"],
)
def test_sec_link_cannot_be_manufactured_from_frozen_labels(mutation: str) -> None:
    body, case = original_sec()
    filename, name = "release.htm", "SYNTHETIC ISSUER"
    field = {
        "raw_hash": "release_pair_source_sha256",
        "exhibit_hash": "release_pair_exhibit_sha256",
        "clock": "release_accepted_at",
        "accession": "release_accession_number",
    }.get(mutation, mutation)
    if mutation == "filename":
        filename = "different.htm"
    elif mutation == "name":
        name = "DIFFERENT ISSUER"
    else:
        case[field] = {"cik": "0000000002", "ticker": "OTHER", "clock": "2024-09-11T13:17:29Z"}.get(
            mutation, "wrong"
        )
    with pytest.raises(ValueError):
        review_sec_issuer_link(body, case, filename, name)


@pytest.mark.parametrize(
    "field",
    [
        *DISARM_FLAGS,
        "network_allowed",
        "provider_cik_imputation_allowed",
        "venue_crosswalk_imputation_allowed",
        "frozen_pair_cases",
    ],
)
def test_review_contract_never_relaxes_the_admission_or_missing_field_rules(field: str) -> None:
    c = json.loads(Path("docs/otc_identity_review_v1.json").read_bytes())
    validate_contract(c)
    c[field] = True if type(c[field]) is bool else 5
    with pytest.raises(ValueError, match="contract"):
        validate_contract(c)
