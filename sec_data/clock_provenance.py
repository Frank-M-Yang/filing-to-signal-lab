"""Trace SEC clock conflicts to raw sources and propose evidence-bound overlays.

An overlay reconciles SEC acceptance representations only. It is not a public
timestamp, a forecast repair, or authorization to shift unreviewed timestamps.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import submission_url, utc_timestamp
from sec_data.disclosure_review import audit_submission_clock
from sec_data.filing_cover import parse_filing_index

VERSION = "sec-clock-provenance-v1"


def index_url(cik: str, accession: str) -> str:
    """Derive a SEC index URL from validated filing identifiers."""
    return submission_url(cik, accession).removesuffix(".txt") + "-index.html"


def trace_source_row(content: bytes, filing: dict[str, Any]) -> dict[str, Any]:
    """Bind a catalog row to its original submissions member/array index.

    Explicit timezone labels are interpreted literally for provenance, not assumed
    to be truthful. A timezone-less source never becomes a UTC fact by guessing.
    """
    document = json.loads(content)
    if not re.fullmatch(
        rf"CIK{re.escape(filing['cik'])}(?:-submissions-\d{{3}})?\.json", filing["source_member"]
    ):
        raise ValueError("Source member filename differs from catalog issuer")
    recent = document.get("filings", {}).get("recent", document)
    index = filing["source_row_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid source row index")
    if "cik" in document and str(document["cik"]).zfill(10) != filing["cik"]:
        raise ValueError("Source member issuer differs from catalog")
    evidence = {}
    for field, target in {
        "accessionNumber": "accession_number",
        "form": "form",
        "filingDate": "filing_date",
        "reportDate": "report_date",
    }.items():
        values = recent.get(field, [])
        if index >= len(values) or values[index] != filing[target]:
            raise ValueError("Source row identity/period differs from catalog")
        evidence[field] = values[index]
    values = recent.get("acceptanceDateTime", [])
    if index >= len(values) or not isinstance(values[index], str):
        raise ValueError("Source acceptance literal missing")
    literal = values[index]
    source_utc = None
    encoding = "timezone_unspecified"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", literal):
        source_utc = utc_timestamp(literal).isoformat()
        encoding = "explicit_timezone_label"
    return {
        "source_member": filing["source_member"],
        "source_member_sha256": hashlib.sha256(content).hexdigest(),
        "source_zip_sha256": filing["source_zip_sha256"],
        "source_row_index": index,
        "source_array_path": "filings.recent" if "filings" in document else "root",
        "source_identity": evidence,
        "raw_acceptance_literal": literal,
        "source_clock_encoding": encoding,
        "literal_interpreted_as_labelled_utc": source_utc,
        "catalog_reproduces_literal": source_utc == utc_timestamp(filing["accepted_at"]).isoformat()
        if source_utc
        else None,
    }


def parse_index_acceptance(content: bytes, filing: dict[str, Any]) -> dict[str, Any]:
    """Verify subject/accession and extract the index's Eastern acceptance clock."""
    url = index_url(filing["cik"], filing["accession_number"])
    metadata = parse_filing_index(content, index_url=url, subject_cik=filing["cik"])
    source = content.decode("utf-8", errors="replace")
    visible = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", source)))
    accessions = re.findall(r"SEC\s+Accession\s+No\.\s+(\d{10}-\d{2}-\d{6})", visible)
    if (
        metadata["subject_cik"] != filing["cik"]
        or metadata["subject_form"] != filing["form"]
        or accessions != [filing["accession_number"]]
    ):
        raise ValueError("Filing index identity differs from target")
    literals = re.findall(
        r'<div\s+class=["\']infoHead["\']>\s*Accepted\s*</div>\s*'
        r'<div\s+class=["\']info["\']>\s*([^<]+?)\s*</div>',
        source,
        re.I,
    )
    literal = literals[0] if len(literals) == 1 else None
    stamp = pd.to_datetime(literal or "", format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if stamp is not None and not pd.isna(stamp):
        stamp = stamp.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return {
        "index_acceptance_literal": literal,
        "index_accepted_at_utc": stamp.tz_convert("UTC").isoformat()
        if stamp is not None and not pd.isna(stamp)
        else None,
        "index_sha256": hashlib.sha256(content).hexdigest(),
        "index_url": url,
    }


def reconcile_acceptance(
    filing: dict[str, Any],
    source: dict[str, Any],
    submission: bytes,
    index: bytes | None = None,
) -> dict[str, Any]:
    """Reconcile three source representations, keeping unresolved cases quarantined."""
    header = audit_submission_clock(submission, filing)
    index_evidence = parse_index_acceptance(index, filing) if index is not None else {}
    corrected = None
    if source["catalog_reproduces_literal"] is False:
        status = "catalog_source_divergence"
    elif source["catalog_reproduces_literal"] is None:
        status = "source_timezone_unspecified"
    elif header["header_accepted_at_utc"] is None:
        status = "header_clock_unresolved"
    elif index is None:
        status = (
            "consistent_header_control"
            if header["status"] == "consistent_acceptance_proxy"
            else "index_evidence_needed"
        )
    elif index_evidence["index_accepted_at_utc"] != header["header_accepted_at_utc"]:
        status = "header_index_conflict"
    else:
        corrected = header["header_accepted_at_utc"]
        status = (
            "source_label_conflict_resolved_by_header_and_index"
            if header["status"] == "clock_conflict_quarantined"
            else "three_representations_consistent"
        )
    return {
        "cik": filing["cik"],
        "accession_number": filing["accession_number"],
        "form": filing["form"],
        "catalog_accepted_at": filing["accepted_at"],
        **source,
        **header,
        **index_evidence,
        "submission_sha256": hashlib.sha256(submission).hexdigest(),
        "resolution_status": status,
        "reviewed_sec_acceptance_proxy_at": corrected,
        "overlay_applied": False,
        "first_public_at": None,
        "public_available_by": None,
        "model_admission": False,
    }


def boundary_scan(
    event: dict[str, Any],
    catalog: pd.DataFrame,
    *,
    padding_days: int = 1,
) -> pd.DataFrame:
    """Inspect an expanded calendar-day envelope, not just the old UTC boundary.

    Padding covers an Eastern/UTC date-edge shift. This remains a bounded SEC
    Item 2.02/7.01 search, not proof of exhaustive public-release coverage.
    """
    if padding_days < 1:
        raise ValueError("Boundary scan must include at least one padding day")
    lower = (pd.Timestamp(event["period_end"]) - pd.Timedelta(days=padding_days)).date().isoformat()
    upper = (
        (utc_timestamp(event["accepted_at"]) + pd.Timedelta(days=padding_days)).date().isoformat()
    )
    selected = catalog.loc[
        catalog["cik"].eq(event["cik"])
        & catalog["form"].isin(["8-K", "8-K/A"])
        & catalog["filing_date"].between(lower, upper)
        & catalog["items"].fillna("").str.contains(r"(?:^|[,;])\s*(?:2\.02|7\.01)\s*(?:[,;]|$)")
    ].copy()
    return selected.sort_values(["filing_date", "accession_number"])


def audit_event_boundary(
    event: dict[str, Any],
    scan_rows: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    old_accessions: list[str],
) -> dict[str, Any]:
    """Compare old/new event membership only if all relevant clocks are reconciled."""
    rows = {r["accession_number"]: r for r in resolutions if r["cik"] == event["cik"]}
    target = rows.get(event["filing_accession"], {})
    new_target = target.get("reviewed_sec_acceptance_proxy_at")
    unresolved = [
        r["accession_number"]
        for r in scan_rows
        if not rows.get(r["accession_number"], {}).get("reviewed_sec_acceptance_proxy_at")
    ]
    if not new_target:
        unresolved.append(event["filing_accession"])
    retained = None
    if not unresolved:
        lower = pd.Timestamp(event["period_end"], tz="America/New_York").tz_convert("UTC")
        retained = sorted(
            r["accession_number"]
            for r in scan_rows
            if lower
            <= utc_timestamp(rows[r["accession_number"]]["reviewed_sec_acceptance_proxy_at"])
            <= utc_timestamp(new_target)
        )
    return {
        "validation_event_id": event["validation_event_id"],
        "ticker": event["ticker"],
        "cik": event["cik"],
        "target_accession": event["filing_accession"],
        "old_target_accepted_at": event["accepted_at"],
        "reviewed_target_proxy_at": new_target,
        "old_candidate_accessions": sorted(old_accessions),
        "reviewed_candidate_accessions": retained,
        "scan_accessions": sorted(r["accession_number"] for r in scan_rows),
        "unresolved_accessions": sorted(unresolved),
        "candidate_membership_unchanged": retained == sorted(old_accessions)
        if retained is not None
        else None,
        "primary_sample_reselected": False,
        "first_public_at": None,
    }
