"""Outcome-blind issuer expansion and strictly earlier, raw-header SIC evidence."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_replay import SIGNALS
from sec_data.disclosure_audit import utc_timestamp

VERSION = "announcement-expansion-v1"


def select_expansion_batch(
    cohort: list[dict[str, Any]], cache_keys: set[tuple[str, str]], *, size: int = 20
) -> list[dict[str, Any]]:
    """First cached candidate per new issuer, then earliest proxy, with no replacement."""
    if size <= 0 or len({r["validation_event_id"] for r in cohort}) != len(cohort):
        raise ValueError("Invalid batch size or duplicate cohort identity")
    controls = {r["cik"] for r in cohort if r.get("release_value_pair_verified") is True}
    eligible = [
        r
        for r in cohort
        if r["cik"] not in controls
        and r["clock_source"] == "item_2_02_8k"
        and (r["cik"], r["release_accession_number"]) in cache_keys
    ]
    selected, seen = [], set()
    for row in sorted(
        eligible, key=lambda r: (r["earliest_disclosure_at"], r["validation_event_id"])
    ):
        if row["cik"] not in seen:
            selected.append(row)
            seen.add(row["cik"])
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError("Insufficient new cached issuers; do not replace frozen selection")
    return selected


def extract_header_industry(content: bytes, cik: str, accession: str) -> dict[str, Any]:
    """Read the matched FILER block only; use the SEC Eastern header time, not a date."""
    header = content.split(b"</SEC-HEADER>", 1)[0].decode("latin-1")
    accessions = re.findall(r"ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})", header)
    clocks = re.findall(r"<ACCEPTANCE-DATETIME>([^\s<]+)", header)
    if accessions != [accession] or len(clocks) != 1 or not re.fullmatch(r"\d{14}", clocks[0]):
        raise ValueError("Ambiguous header accession or timestamp")
    stamp = pd.to_datetime(clocks[0], format="%Y%m%d%H%M%S").tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    if pd.isna(stamp):
        raise ValueError("Ambiguous or nonexistent header wall clock")
    matches = []
    for block in re.split(r"(?m)^[ \t]*FILER:[ \t]*$", header)[1:]:
        block = re.split(r"(?m)^[ \t]*(?:SUBJECT COMPANY|REPORTING-OWNER|ISSUER|FILED BY):", block)[
            0
        ]
        ids = re.findall(r"CENTRAL INDEX KEY:\s*(\d+)", block)
        if [v.zfill(10) for v in ids] == [cik]:
            matches.append(block)
    if len(matches) != 1:
        raise ValueError("SIC must belong to exactly one matching FILER block")
    sics = re.findall(
        r"STANDARD INDUSTRIAL CLASSIFICATION:\s*([^\r\n]*?)\s*\[(\d{4})\]", matches[0]
    )
    if len(sics) != 1:
        raise ValueError("Missing or ambiguous original filing SIC")
    return {
        "cik": cik,
        "accession_number": accession,
        "industry_available_at": stamp.tz_convert("UTC").isoformat(),
        "sic": sics[0][1],
        "sic_description": sics[0][0].strip(),
        "industry_basis": "original_SEC_FILER_header",
    }


def attach_prior_industry(
    cases: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> pd.DataFrame:
    """As-of join strictly earlier original SIC, preserving missing/conflicting evidence."""
    rows = []
    for case in cases:
        cutoff = utc_timestamp(case["earliest_disclosure_at"])
        prior = [
            r
            for r in observations
            if r["cik"] == case["cik"] and utc_timestamp(r["industry_available_at"]) < cutoff
        ]
        row: dict[str, Any] = {
            "validation_event_id": case["validation_event_id"],
            "cik": case["cik"],
            "ticker": case["assigned_ticker"],
            "announcement_at": cutoff.isoformat(),
            "sic": None,
            "sic_2digit": None,
            "industry_available_at": None,
            "source_accessions": [],
            "source_sha256": [],
            "industry_ready": False,
            "industry_status": "no_strictly_prior_header_SIC",
        }
        if prior:
            latest = max(utc_timestamp(r["industry_available_at"]) for r in prior)
            bound = [r for r in prior if utc_timestamp(r["industry_available_at"]) == latest]
            if len({r["sic"] for r in bound}) != 1:
                row["industry_status"] = "same_time_conflicting_SIC"
            elif any(
                not re.fullmatch(r"[a-f0-9]{64}", str(r.get("source_sha256", "")))
                or not re.fullmatch(r"\d{4}", str(r["sic"]))
                for r in bound
            ):
                raise ValueError("Industry evidence lacks valid code or provenance")
            else:
                sic = str(bound[0]["sic"])
                row.update(
                    sic=sic,
                    sic_2digit=sic[:2],
                    industry_available_at=latest.isoformat(),
                    source_accessions=sorted({r["accession_number"] for r in bound}),
                    source_sha256=sorted({r["source_sha256"] for r in bound}),
                    industry_ready=True,
                    industry_status="strictly_prior_original_header_SIC",
                )
        rows.append(row)
    return pd.DataFrame(rows)


def industry_rank_diagnostic(
    analysis: pd.DataFrame, industries: pd.DataFrame, *, minimum_events: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Same-sample rank residuals by prior SIC2, with a frozen 20-event adequacy gate.

    This is a composition check, not a sector ETF excess return. Every industry
    needs two distinct issuers, not just repeated quarters of a single company.
    No uncertainty, model fitting, or trading inference is performed here.
    """
    if minimum_events != 20:
        raise ValueError("Do not weaken the frozen industry sample gate")
    if (
        analysis.validation_event_id.duplicated().any()
        or industries.validation_event_id.duplicated().any()
        or set(analysis.validation_event_id) != set(industries.validation_event_id)
    ):
        raise ValueError("Industry and financial samples must have identical event identities")
    sample = analysis.merge(
        industries, on="validation_event_id", suffixes=("", "_industry"), validate="one_to_one"
    )
    if not sample.cik.eq(sample.cik_industry).all():
        raise ValueError("Industry issuer mismatch")
    if not np.isfinite(sample[[*SIGNALS, "market_excess_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete numerical industry sample")
    for row in sample.loc[sample.industry_ready].to_dict("records"):
        if (
            utc_timestamp(row["announcement_at_industry"]) != utc_timestamp(row["announcement_at"])
            or not utc_timestamp(row["industry_available_at"])
            < utc_timestamp(row["announcement_at"])
            or not re.fullmatch(r"\d{4}", str(row["sic"]))
            or str(row["sic"])[:2] != row["sic_2digit"]
        ):
            raise ValueError("Industry evidence is not strictly prior to the signal event")
    composition = (
        sample.loc[sample.industry_ready]
        .groupby("sic_2digit", as_index=False)
        .agg(events=("validation_event_id", "size"), issuers=("cik", "nunique"))
    )
    eligible = set(composition.loc[composition.issuers.ge(2), "sic_2digit"])
    sample["industry_same_sample_eligible"] = sample.industry_ready & sample.sic_2digit.isin(
        eligible
    )
    complete = sample.loc[sample.industry_same_sample_eligible]
    coverage = float(sample.industry_ready.mean()) if len(sample) else 0.0
    allowed = len(complete) >= 20 and len(eligible) >= 2 and coverage >= 0.8
    statistics = []
    for signal in SIGNALS:
        raw = adjusted = None
        if allowed:
            x = complete[signal].rank(method="average")
            y = complete.market_excess_return.rank(method="average")
            groups = complete.sic_2digit
            xr = x - x.groupby(groups).transform("mean")
            yr = y - y.groupby(groups).transform("mean")
            raw = float(x.corr(y)) if x.nunique() > 1 and y.nunique() > 1 else None
            adjusted = float(xr.corr(yr)) if xr.nunique() > 1 and yr.nunique() > 1 else None
        statistics.append(
            {
                "representation": signal,
                "events": len(complete),
                "raw_same_industry_sample_ic": raw,
                "prior_SIC2_rank_residual_ic": adjusted,
                "computed": allowed,
            }
        )
    report = {
        "prior_industry_matches": int(sample.industry_ready.sum()),
        "events": len(sample),
        "prior_industry_coverage": coverage,
        "eligible_industry_events": len(complete),
        "eligible_industries": len(eligible),
        "minimum_events": 20,
        "industry_composition_diagnostic_allowed": allowed,
        "tradable_sector_benchmark": False,
        "inferential_tests_computed": False,
    }
    return (
        sample,
        pd.DataFrame(statistics),
        {**report, "composition": composition.to_dict("records")},
    )
