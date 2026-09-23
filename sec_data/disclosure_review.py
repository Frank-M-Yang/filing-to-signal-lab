"""Evidence-bound disclosure follow-up; SEC acceptance is NOT public availability.

Table context is a reviewer assertion. Hash/anchor/value checks bind that assertion
to evidence but cannot independently establish accounting scope or publication time.
No output from this module is admitted to a forecast, signal or trading policy.
"""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import METRIC_TAGS, utc_timestamp, verify_file
from sec_data.guidance_evidence import parse_complete_submission

VERSION = "sec-disclosure-followup-v1"
SEC_CLOCK_SOURCE = "https://www.sec.gov/about/webmaster-frequently-asked-questions"


def audit_submission_clock(content: bytes, candidate: dict[str, Any]) -> dict[str, Any]:
    """Reconcile the raw Eastern wall clock with the frozen catalog UTC clock.

    America/New_York handles DST; never apply a universal four/five-hour shift to
    the catalog. Missing, ambiguous and contradictory clocks stay quarantined.
    Even agreement certifies neither dissemination nor first public availability.
    """
    header = content.split(b"</SEC-HEADER>", 1)[0][:20000].decode("latin-1")
    accession = re.findall(r"ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})", header)
    ciks = re.findall(r"CENTRAL INDEX KEY:\s*(\d+)", header)
    forms = re.findall(r"CONFORMED SUBMISSION TYPE:\s*([^\r\n]+)", header)
    if (
        accession != [candidate["accession_number"]]
        or candidate["cik"] not in {c.zfill(10) for c in ciks}
        or forms != [candidate["form"]]
    ):
        raise ValueError("Raw submission identity differs from frozen candidate")
    raw_clocks = re.findall(r"<ACCEPTANCE-DATETIME>([^\s<]+)", header)
    result: dict[str, Any] = {
        "catalog_accepted_at": utc_timestamp(candidate["accepted_at"]).isoformat(),
        "header_acceptance_literal": raw_clocks[0] if len(raw_clocks) == 1 else None,
        "header_timezone": "America/New_York",
        "header_accepted_at_utc": None,
        "catalog_minus_header_hours": None,
        "status": "missing_or_invalid_header_clock",
        "sec_acceptance_proxy_at": None,
        "public_available_by": None,
        "first_public_at": None,
        "trading_admission": False,
    }
    if len(raw_clocks) != 1 or not re.fullmatch(r"\d{14}", raw_clocks[0]):
        return result
    naive = pd.to_datetime(raw_clocks[0], format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(naive):
        return result
    eastern = naive.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    if pd.isna(eastern):
        result["status"] = "ambiguous_or_nonexistent_header_clock"
        return result
    header_utc = eastern.tz_convert("UTC")
    delta = (utc_timestamp(candidate["accepted_at"]) - header_utc).total_seconds() / 3600
    result.update(
        header_accepted_at_utc=header_utc.isoformat(),
        catalog_minus_header_hours=delta,
        status="consistent_acceptance_proxy" if delta == 0 else "clock_conflict_quarantined",
        sec_acceptance_proxy_at=header_utc.isoformat() if delta == 0 else None,
    )
    return result


def _bound_exhibit(
    specification: dict[str, Any],
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    key = (specification["cik"], specification["release_accession"])
    matches = [
        row
        for row in candidates
        if row["validation_event_id"] == event["validation_event_id"]
        and (row["cik"], row["accession_number"]) == key
    ]
    if len(matches) != 1 or key not in cache:
        raise ValueError("Review is outside the frozen candidate set")
    source = cache[key]
    if source["sha256"] != specification["submission_sha256"]:
        raise ValueError("Reviewed submission snapshot changed")
    verify_file(Path(source["path"]), source["sha256"])
    content = Path(source["path"]).read_bytes()
    clock = audit_submission_clock(content, matches[0])
    exhibits = [
        d
        for d in parse_complete_submission(content)
        if d["filename"] == specification["exhibit_filename"]
        and d["document_sha256"] == specification["exhibit_sha256"]
        and d["document_type"].upper().startswith("EX-99")
    ]
    if len(exhibits) != 1:
        raise ValueError("Reviewed exhibit snapshot changed")
    return re.sub(r"\s+", " ", exhibits[0]["visible_text"]), clock


def review_document_kinds(
    specifications: list[dict[str, Any]],
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind document-type assertions to source anchors, not to keyword classifiers."""
    allowed = {"earnings_release", "preliminary_results", "distribution", "acquisition", "listing"}
    result = []
    seen = set()
    for spec in specifications:
        key = (spec["cik"], spec["release_accession"])
        if key in seen or spec["kind"] not in allowed:
            raise ValueError("Duplicate or unsupported document disposition")
        seen.add(key)
        event = _review_event(spec, events)
        text, clock = _bound_exhibit(spec, event, candidates, cache)
        if not spec["anchors"] or any(not a or a not in text for a in spec["anchors"]):
            raise ValueError("Document disposition anchor missing")
        result.append(
            {
                **spec,
                "validation_event_id": event["validation_event_id"],
                "ticker": event["ticker"],
                "clock": clock,
                "review_method": "reviewer_context_assertion_with_hash_and_anchor_checks",
                "first_public_at": None,
                "trading_admission": False,
            }
        )
    return result


def _review_event(spec: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        e for e in events if e["cik"] == spec["cik"] and e["period_end"] == spec["period_end"]
    ]
    if len(matches) != 1:
        raise ValueError("Review event identity is not unique")
    return matches[0]


def review_metric_cells(
    specifications: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare reviewed original-release cells without repairing missing originals.

    No tolerance/rounding plug is used. A directly reported quarter can differ
    from a difference of two rounded YTD filings; preserve both and the gap.
    """
    result = []
    seen = set()
    for spec in specifications:
        event = _review_event(spec, events)
        text, clock = _bound_exhibit(spec, event, candidates, cache)
        if spec["accounting_basis"] != "GAAP":
            raise ValueError("Only reviewed GAAP cells may be compared to original facts")
        for anchor in (spec["period_anchor"], spec["unit_anchor"]):
            if not anchor or anchor not in text:
                raise ValueError("Reviewed period/unit anchor missing")
        unit = spec["unit_anchor"].lower()
        scale = 1000000 if "million" in unit else 1000 if "thousand" in unit else 1
        if type(spec["scale"]) is not int or spec["scale"] != scale:
            raise ValueError("Scale contradicts reviewed unit anchor")
        for cell in spec["cells"]:
            key = (event["validation_event_id"], spec["release_accession"], cell["metric"])
            if key in seen or cell["metric"] not in METRIC_TAGS:
                raise ValueError("Duplicate or unsupported metric review")
            seen.add(key)
            literal, anchor = cell["display_value"], cell["text_anchor"]
            if (
                not anchor
                or anchor not in text
                or not re.search(r"(?<![\d,.])" + re.escape(literal) + r"(?![\d,.])", anchor)
            ):
                raise ValueError("Reviewed cell/literal anchor missing")
            try:
                value = Decimal(literal.replace(",", "").replace("(", "-").replace(")", ""))
            except InvalidOperation as exc:
                raise ValueError("Invalid reviewed numeric value") from exc
            if not value.is_finite():
                raise ValueError("Nonfinite reviewed numeric value")
            value *= scale
            originals = [
                f
                for f in fields
                if f["validation_event_id"] == event["validation_event_id"]
                and f["metric"] == cell["metric"]
            ]
            if len(originals) != 1:
                raise ValueError("Original field identity is not unique")
            original = originals[0]
            direct = original["quarter_or_instant_ready"]
            match = direct and value == Decimal(str(original["value"]))
            derived = original.get("derived_quarter_candidate")
            status = (
                "exact_original_match"
                if match
                else "original_value_disagrees"
                if direct
                else "release_quarter_vs_cross_filing_ytd_candidate"
                if derived is not None
                else "original_field_unresolved"
            )
            proxy = clock["sec_acceptance_proxy_at"]
            result.append(
                {
                    "validation_event_id": event["validation_event_id"],
                    "cik": event["cik"],
                    "ticker": event["ticker"],
                    "period_start": event["period_start"],
                    "period_end": event["period_end"],
                    "metric": cell["metric"],
                    "unit": "USD",
                    "value": float(value),
                    "original_value": original["value"],
                    "original_status": original["status"],
                    "comparison_status": status,
                    "derived_quarter_candidate": derived,
                    "release_minus_ytd_candidate": float(value - Decimal(str(derived)))
                    if derived is not None
                    else None,
                    "scope_note": cell.get("scope_note", "Scope compatibility not certified."),
                    "release_accession": spec["release_accession"],
                    "source_sha256": spec["submission_sha256"],
                    "exhibit_sha256": spec["exhibit_sha256"],
                    "text_anchor": anchor,
                    "sec_acceptance_proxy_at": proxy,
                    "clock_status": clock["status"],
                    "public_available_by": None,
                    "first_public_at": None,
                    "formal_feature_admitted": False,
                    "cashflow_basis_compatible": False,
                }
            )
    return result


def forecast_proxy_check(metadata: dict[str, Any], proxy: str) -> dict[str, Any]:
    """Flag timing risk against acceptance, without claiming a proven public clock."""
    timestamp = utc_timestamp(proxy)
    clocks = {}
    for key in ("forecast_feature_cutoff", "forecast_training_end", "forecast_generated_at"):
        value = metadata.get(key)
        clocks[key] = None if value is None or pd.isna(value) else utc_timestamp(value) < timestamp
    return {
        "sec_acceptance_proxy_at": timestamp.isoformat(),
        "strictly_before_proxy": clocks,
        "status": "at_or_after_acceptance_proxy"
        if any(value is False for value in clocks.values())
        else "missing_forecast_clock"
        if any(value is None for value in clocks.values())
        else "before_proxy_not_first_public_certified",
        "first_public_forecast_certified": False,
    }


def earliest_metric_proxies(cells: list[dict[str, Any]]) -> dict[str, str]:
    """Use the earliest consistent matched proxy per metric, then both GM inputs."""
    result: dict[str, str] = {}
    for row in cells:
        if row["comparison_status"] != "exact_original_match" or not row["sec_acceptance_proxy_at"]:
            continue
        metric = row["metric"]
        proxy = utc_timestamp(row["sec_acceptance_proxy_at"]).isoformat()
        if metric not in result or utc_timestamp(proxy) < utc_timestamp(result[metric]):
            result[metric] = proxy
    if "revenue" in result and "gross_profit" in result:
        result["gross_margin"] = max(result["revenue"], result["gross_profit"])
    return result


def audit_candidate_headers(
    candidates: list[dict[str, Any]], cache: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Check every frozen candidate, including unrelated and unreviewed documents."""
    result = []
    for candidate in candidates:
        key = (candidate["cik"], candidate["accession_number"])
        source = cache[key]
        verify_file(Path(source["path"]), source["sha256"])
        if source["requested_url"] != candidate["url"]:
            raise ValueError("Cached request differs from frozen candidate URL")
        content = Path(source["path"]).read_bytes()
        result.append(
            {
                **candidate,
                "source_sha256": hashlib.sha256(content).hexdigest(),
                **audit_submission_clock(content, candidate),
            }
        )
    return result
