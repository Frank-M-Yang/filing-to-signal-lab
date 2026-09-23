"""Outcome-blind SEC event-clock and original accounting-field coverage audit.

An accession's acceptance is an operational proxy, not a public-availability
upper bound. Text mentions never become verified actuals or public clocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from sec_data.guidance_evidence import parse_complete_submission

VERSION = "sec-disclosure-audit-v1"
METRIC_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "assets": ("Assets",),
}
METRIC_PATTERNS = {
    "revenue": r"\b(?:revenues?|net sales)\b",
    "gross_margin": r"\bgross (?:profit|margin)\b",
    "net_income": r"\bnet (?:income|loss|earnings)\b",
    "operating_cash_flow": r"\b(?:operating activities|operating cash flows?)\b",
    "assets": r"\btotal assets\b",
}


def digest_file(path: Path) -> str:
    """Hash exact bytes in bounded memory, including the preserved bulk ZIP."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Reject stale or altered evidence rather than silently refreshing it."""
    if digest_file(path) != expected:
        raise ValueError(f"Evidence hash mismatch: {path.name}")


def utc_timestamp(value: Any) -> pd.Timestamp:
    """Require an explicit timezone; do not guess an SEC or release clock."""
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Availability timestamp must include a timezone")
    return result.tz_convert("UTC")


def select_audit_sample(
    events: pd.DataFrame, *, sample_size: int = 30, salt: str = VERSION
) -> pd.DataFrame:
    """Select original 10-Q events in issuer-balanced hash order, without values.

    Every issuer's first hashed event precedes any issuer's second event. Missing
    evidence after selection never changes this order or causes replacement.
    """
    columns = ["validation_event_id", "cik", "ticker", "form", "accepted_at"]
    if sample_size < 1 or events["validation_event_id"].duplicated().any():
        raise ValueError("Invalid sample size or duplicate event identity")
    selected = events.loc[events["form"].eq("10-Q"), columns].copy()
    selected["accepted_at"] = selected["accepted_at"].map(utc_timestamp)
    selected["cik"] = selected["cik"].astype(str).str.zfill(10)
    selected["sample_hash"] = selected["validation_event_id"].map(
        lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    )
    selected = selected.sort_values(["sample_hash", "validation_event_id"])
    selected["issuer_round"] = selected.groupby("cik").cumcount()
    selected = selected.sort_values(["issuer_round", "sample_hash", "validation_event_id"])
    selected = selected.head(sample_size).reset_index(drop=True)
    selected["sample_order"] = selected.index + 1
    return selected


def earnings_candidates(event: dict[str, Any], filings: pd.DataFrame) -> list[dict[str, Any]]:
    """Find ALL Item 2.02/7.01 candidates between period end and target acceptance.

    This is a bounded search, not an exhaustive first-public-information claim.
    An 8-K's report date is an event date and is NOT used as the financial period.
    """
    accepted = utc_timestamp(event["accepted_at"])
    end = pd.Timestamp(event["period_end"], tz="UTC")
    selected = filings.loc[
        filings["cik"].eq(event["cik"]) & filings["form"].isin(["8-K", "8-K/A"])
    ].copy()
    selected["accepted_at"] = selected["accepted_at"].map(utc_timestamp)
    selected = selected.loc[
        selected["accepted_at"].between(end, accepted, inclusive="both")
        & selected["items"].fillna("").str.contains(r"(?:^|,)\s*(?:2\.02|7\.01)\s*(?:,|$)")
    ].sort_values(["accepted_at", "accession_number"])
    return [
        {
            "validation_event_id": event["validation_event_id"],
            "cik": event["cik"],
            "accession_number": row.accession_number,
            "form": row.form,
            "items": row.items,
            "accepted_at": utc_timestamp(row.accepted_at).isoformat(),
            "url": submission_url(event["cik"], str(row.accession_number)),
            "period_match_verified": False,
            "first_public_at": None,
        }
        for row in selected.itertuples(index=False)
    ]


def submission_url(cik: str, accession: str) -> str:
    """Build an allowlisted SEC full-submission URL from validated identifiers."""
    if not re.fullmatch(r"\d{10}", cik) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise ValueError("Invalid SEC identity")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession.replace('-', '')}/{accession}.txt"
    )


def inspect_earnings_document(content: bytes, accession: str) -> dict[str, Any]:
    """Expose review snippets, never infer actuals/first-public times from keywords."""
    header = content[:12000].decode("latin-1")
    match = re.search(r"ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})", header)
    if match is None or match.group(1) != accession:
        raise ValueError("Submission accession does not match candidate")
    documents = parse_complete_submission(content)
    exhibits = [d for d in documents if str(d["document_type"]).upper().startswith("EX-99")]
    text = "\n".join(str(d["visible_text"]) for d in exhibits)
    snippets = {}
    for metric, pattern in METRIC_PATTERNS.items():
        found = re.search(pattern, text, re.I)
        snippets[metric] = text[max(0, found.start() - 160) : found.end() + 260] if found else None
    return {
        "exhibit_count": len(exhibits),
        "exhibit_head": text[:1400],
        "metric_snippets": snippets,
        "metric_presence_is_actual_match": False,
        "first_public_at": None,
        "status": "exhibit_needs_period_value_and_release_clock_review"
        if text
        else "no_text_exhibit",
    }


def _metric_facts(document: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    results = []
    for tag in METRIC_TAGS[metric]:
        units = document.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
        for unit, records in units.items():
            for index, record in enumerate(records):
                results.append(
                    {
                        **record,
                        "tag": tag,
                        "unit": unit,
                        "record_path": f"facts.us-gaap.{tag}.units.{unit}[{index}]",
                    }
                )
    return results


def _unique_value(rows: list[dict[str, Any]]) -> tuple[float | None, str]:
    if not rows:
        return None, "missing"
    values = []
    for row in rows:
        if isinstance(row.get("val"), bool):
            return None, "invalid_value"
        try:
            value = float(row["val"])
        except (ValueError, TypeError, KeyError):
            return None, "invalid_value"
        if not math.isfinite(value):
            return None, "invalid_value"
        values.append(value)
    if len(set(values)) != 1 or len({r["tag"] for r in rows}) != 1:
        return None, "ambiguous_value_or_tag"
    return values[0], "unique"


def _evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in ("accn", "tag", "unit", "start", "end", "record_path")}
        for row in rows
    ]


def audit_financial_fields(
    event: dict[str, Any], document: dict[str, Any], filings: pd.DataFrame
) -> list[dict[str, Any]]:
    """Audit accession-bound USD fields without silently treating YTD as a quarter.

    Cross-filing YTD subtraction is only a candidate: comparable accounting basis
    and any intervening restatement still require review. Later filings, amendments
    and aggregate latest values cannot repair missing original observations.
    """
    if str(document.get("cik", "")).zfill(10) != event["cik"]:
        raise ValueError("Companyfacts issuer mismatch")
    accepted = utc_timestamp(event["accepted_at"])
    start, end = str(event["period_start"]), str(event["period_end"])
    if not 70 <= (pd.Timestamp(end) - pd.Timestamp(start)).days <= 110:
        raise ValueError("Target is not an exact quarter")
    target = filings.loc[
        filings["cik"].eq(event["cik"]) & filings["accession_number"].eq(event["filing_accession"])
    ]
    if (
        len(target) != 1
        or str(target.iloc[0]["form"]) != "10-Q"
        or utc_timestamp(target.iloc[0]["accepted_at"]) != accepted
        or str(target.iloc[0]["report_date"])[:10] != end
    ):
        raise ValueError("Target accession/period/acceptance mismatch")
    outputs = []
    for metric in METRIC_TAGS:
        all_rows = _metric_facts(document, metric)
        originals = [
            r
            for r in all_rows
            if r.get("accn") == event["filing_accession"]
            and r.get("form") == "10-Q"
            and r.get("end") == end
        ]
        exact = [
            r
            for r in originals
            if r["unit"] == "USD"
            and (r.get("start") == start if metric != "assets" else not r.get("start"))
        ]
        value, status = _unique_value(exact)
        row: dict[str, Any] = {
            "validation_event_id": event["validation_event_id"],
            "cik": event["cik"],
            "metric": metric,
            "period_start": None if metric == "assets" else start,
            "period_end": end,
            "unit": "USD",
            "value": value,
            "status": "original_instant"
            if status == "unique" and metric == "assets"
            else "original_quarter"
            if status == "unique"
            else status,
            "source_evidence": _evidence(exact),
            "original_accession": event["filing_accession"],
            "sec_acceptance_proxy_at": accepted.isoformat(),
            "public_available_by": None,
            "first_public_at": None,
            "quarter_or_instant_ready": status == "unique",
            "formal_feature_admitted": False,
            "ytd_value": None,
            "derived_quarter_candidate": None,
            "prior_ytd_sec_accepted_at": None,
        }
        if status == "missing" and originals and not any(r["unit"] == "USD" for r in originals):
            row["status"] = "unsupported_currency"
        if metric == "operating_cash_flow" and status == "missing":
            _audit_ytd_cash_flow(row, event, originals, all_rows, filings)
        outputs.append(row)
    return outputs


def _audit_ytd_cash_flow(
    result: dict[str, Any],
    event: dict[str, Any],
    originals: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    filings: pd.DataFrame,
) -> None:
    start = pd.Timestamp(event["period_start"])
    current = [
        r
        for r in originals
        if r["unit"] == "USD"
        and r.get("start")
        and 0 < (start - pd.Timestamp(r["start"])).days <= 230
    ]
    ytd, status = _unique_value(current)
    if status == "missing":
        return
    result["source_evidence"] = _evidence(current)
    if status != "unique" or len({r["start"] for r in current}) != 1:
        result["status"] = "ambiguous_ytd"
        return
    result["ytd_value"] = ytd
    result["status"] = "ytd_only_prior_component_missing"
    previous_end = (start - timedelta(days=1)).date().isoformat()
    priors = filings.loc[
        filings["cik"].eq(event["cik"])
        & filings["form"].eq("10-Q")
        & filings["report_date"].astype(str).str[:10].eq(previous_end)
    ].copy()
    priors["accepted_at"] = priors["accepted_at"].map(utc_timestamp)
    priors = priors.loc[priors["accepted_at"] < utc_timestamp(event["accepted_at"])]
    if priors.empty:
        return
    prior = priors.sort_values(["accepted_at", "accession_number"]).iloc[0]
    previous = [
        r
        for r in all_rows
        if r.get("accn") == prior["accession_number"]
        and r.get("form") == "10-Q"
        and r["unit"] == "USD"
        and r["tag"] == current[0]["tag"]
        and r.get("start") == current[0]["start"]
        and r.get("end") == previous_end
    ]
    prior_value, status = _unique_value(previous)
    if status != "unique":
        result["status"] = "ytd_prior_component_" + status
        return
    assert ytd is not None and prior_value is not None
    result.update(
        status="ytd_difference_requires_basis_review",
        derived_quarter_candidate=ytd - prior_value,
        prior_ytd_sec_accepted_at=prior["accepted_at"].isoformat(),
        source_evidence=_evidence(current) + _evidence(previous),
    )


def load_cached_submissions(roots: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index preserved completions, verifying bytes only when an artifact is used."""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for root in roots:
        for path in sorted((root / "completions").glob("*.json")):
            item = json.loads(path.read_bytes())
            if item.get("status_code") != 200:
                continue
            raw = (root / item["raw_path"]).resolve()
            if not raw.is_relative_to(root.resolve()):
                raise ValueError("Cached evidence escapes its source root")
            key = (str(item["cik"]).zfill(10), str(item["accession_number"]))
            if key in indexed and indexed[key]["sha256"] != item["sha256"]:
                raise ValueError("Multiple distinct snapshots require an explicit version choice")
            indexed[key] = {**item, "path": str(raw)}
    return indexed


def audit_forecast_clocks(metadata: dict[str, Any], sec_acceptance_proxy: str) -> dict[str, Any]:
    """Flag forecast timing risk against an operational SEC acceptance proxy.

    Acceptance is not a public timestamp: propagation can lag, and an IR release
    can precede acceptance. Neither passing nor failing certifies publication order.
    """
    bound = utc_timestamp(sec_acceptance_proxy)
    clocks = {}
    for name in ("forecast_feature_cutoff", "forecast_training_end", "forecast_generated_at"):
        value = metadata.get(name)
        timestamp = None if value is None or pd.isna(value) else utc_timestamp(value)
        clocks[name] = {
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "strictly_before_sec_acceptance_proxy": timestamp < bound
            if timestamp is not None
            else None,
        }
    failed = [k for k, v in clocks.items() if v["strictly_before_sec_acceptance_proxy"] is False]
    return {
        "sec_acceptance_proxy_at": bound.isoformat(),
        "public_available_by": None,
        "clocks": clocks,
        "failed_clocks": failed,
        "status": "at_or_after_sec_acceptance_proxy"
        if failed
        else "first_public_time_still_unverified",
        "first_public_forecast_certified": False,
    }


def verify_manual_reviews(
    reviews: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind human-reviewed table cells to exact release bytes and original actuals.

    Context/column/basis interpretation is a reviewer assertion, not an automatic
    extraction result. Checks bind that assertion to text, values and hashes.
    The catalog acceptance proxy is not a public-availability upper bound and is
    not header-reconciled here. Use the follow-up audit for clock quarantine.
    """
    output = []
    seen = set()
    for review in reviews:
        matches = [
            e
            for e in events
            if e["cik"] == review["cik"] and e["period_end"] == review["period_end"]
        ]
        if len(matches) != 1 or review["accounting_basis"] != "GAAP":
            raise ValueError("Manual review event/basis is not unique or eligible")
        event = matches[0]
        key = (review["cik"], review["release_accession"])
        candidate = [
            c
            for c in candidates
            if c["validation_event_id"] == event["validation_event_id"]
            and c["accession_number"] == key[1]
        ]
        if len(candidate) != 1 or key not in cache:
            raise ValueError("Manual release review is outside the frozen candidate set")
        cached = cache[key]
        if cached["sha256"] != review["submission_sha256"]:
            raise ValueError("Manual review snapshot changed")
        verify_file(Path(cached["path"]), review["submission_sha256"])
        documents = parse_complete_submission(Path(cached["path"]).read_bytes())
        exhibits = [
            d
            for d in documents
            if d["filename"] == review["exhibit_filename"]
            and d["document_sha256"] == review["exhibit_sha256"]
            and d["document_type"].upper().startswith("EX-99")
        ]
        if len(exhibits) != 1:
            raise ValueError("Reviewed exhibit identity changed")
        text = re.sub(r"\s+", " ", exhibits[0]["visible_text"])
        for anchor in (review["period_anchor"], review["unit_anchor"]):
            if not anchor or anchor not in text:
                raise ValueError("Reviewed period or scale anchor not found")
        unit_text = review["unit_anchor"].lower()
        expected_scale = (
            1000000 if "million" in unit_text else 1000 if "thousand" in unit_text else 1
        )
        if review["scale"] != expected_scale:
            raise ValueError("Reviewed scale contradicts its unit anchor")
        for cell in review["cells"]:
            identity = (event["validation_event_id"], key[1], cell["metric"])
            if identity in seen:
                raise ValueError("Duplicate metric review")
            seen.add(identity)
            rows = [
                f
                for f in fields
                if f["validation_event_id"] == event["validation_event_id"]
                and f["metric"] == cell["metric"]
                and f["quarter_or_instant_ready"]
            ]
            if len(rows) != 1 or not cell["text_anchor"] or cell["text_anchor"] not in text:
                raise ValueError("Reviewed metric does not match a unique original or source text")
            literal = cell["display_value"]
            if literal not in cell["text_anchor"] or review["scale"] not in (1, 1000, 1000000):
                raise ValueError("Reviewed numeric literal or scale is invalid")
            displayed = float(literal.replace(",", "").replace("(", "-").replace(")", ""))
            if not math.isclose(
                displayed * review["scale"], rows[0]["value"], abs_tol=0.01, rel_tol=0
            ):
                raise ValueError("Earlier disclosure and original actual differ")
            available_by = utc_timestamp(candidate[0]["accepted_at"])
            if available_by >= utc_timestamp(event["accepted_at"]):
                raise ValueError("Reviewed release is not strictly earlier than 10-Q")
            output.append(
                {
                    "validation_event_id": event["validation_event_id"],
                    "cik": event["cik"],
                    "ticker": event["ticker"],
                    "metric": cell["metric"],
                    "period_end": event["period_end"],
                    "value": rows[0]["value"],
                    "unit": "USD",
                    "release_accession": key[1],
                    "source_sha256": cached["sha256"],
                    "exhibit_sha256": review["exhibit_sha256"],
                    "sec_acceptance_proxy_at": available_by.isoformat(),
                    "public_available_by": None,
                    "header_clock_reconciled": False,
                    "hours_before_10q": (
                        utc_timestamp(event["accepted_at"]) - available_by
                    ).total_seconds()
                    / 3600,
                    "first_public_at": None,
                    "first_public_verified": False,
                    "review_method": "manual_table_context_with_hash_anchor_and_value_checks",
                    "trading_admission": False,
                }
            )
    return output
