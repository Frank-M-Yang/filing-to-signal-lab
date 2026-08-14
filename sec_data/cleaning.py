"""Pure SEC submissions/companyfacts normalization and amendment comparison."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

from sec_data.client import normalize_cik
from sec_data.models import ParseResult

ALLOWED_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")
STANDARD_TAXONOMIES = frozenset({"us-gaap", "ifrs-full", "dei", "srt", "invest"})
UNIT_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]*(?:/[A-Za-z][A-Za-z0-9._:-]*)?$")
ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")

FILING_COLUMNS = [
    "filing_id",
    "cik",
    "ticker",
    "company_name",
    "accession_number",
    "form",
    "is_amendment",
    "filing_date",
    "report_date",
    "accepted_at",
    "primary_document",
    "source_url",
    "source_snapshot_sha256",
    "source_document",
]
FACT_COLUMNS = [
    "fact_id",
    "cik",
    "taxonomy",
    "tag",
    "label",
    "description",
    "unit",
    "value",
    "value_numeric",
    "period_start",
    "period_end",
    "duration_days",
    "fiscal_year",
    "fiscal_period",
    "frame",
    "accession_number",
    "form",
    "base_form",
    "is_amendment",
    "source_filed_date",
    "accepted_at",
    "source_url",
    "source_record_path",
    "source_snapshot_sha256",
    "source_occurrence",
    "quality_status",
]
REJECTION_COLUMNS = [
    "rejection_id",
    "stage",
    "reason",
    "detail",
    "source_record_path",
    "source_snapshot_sha256",
    "source_data",
    "rejected_at",
]
AMENDMENT_COLUMNS = [
    "amendment_fact_id",
    "taxonomy",
    "tag",
    "label",
    "unit",
    "period_start",
    "period_end",
    "base_form",
    "original_accession",
    "original_accepted_at",
    "original_value",
    "amendment_accession",
    "amendment_accepted_at",
    "amendment_value",
    "absolute_difference",
    "percentage_difference",
    "comparison_status",
]


def parse_submissions_documents(
    documents: list[tuple[dict[str, Any], str, str]],
    *,
    cik: str,
    ticker: str | None,
    company_name: str,
    parsed_at: datetime | None = None,
) -> ParseResult:
    """Normalize recent and archived columnar submissions documents."""
    normalized_cik = normalize_cik(cik)
    timestamp = parsed_at or datetime.now(timezone.utc)
    filing_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    required_arrays = (
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    )
    for document, source_document, snapshot_hash in documents:
        recent = _submissions_array_object(document)
        if recent is None:
            rejection_rows.append(
                _rejection(
                    stage="submissions",
                    reason="missing_columnar_filing_data",
                    detail="Document contains neither filings.recent nor archived filing arrays",
                    path=source_document,
                    snapshot_hash=snapshot_hash,
                    source_data=document,
                    rejected_at=timestamp,
                )
            )
            continue
        arrays: dict[str, list[Any]] = {}
        invalid_field: str | None = None
        for field in required_arrays:
            value = recent.get(field)
            if not isinstance(value, list):
                invalid_field = field
                break
            arrays[field] = value
        if invalid_field is not None:
            rejection_rows.append(
                _rejection(
                    stage="submissions",
                    reason="invalid_columnar_array",
                    detail=f"{invalid_field} must be an array",
                    path=source_document,
                    snapshot_hash=snapshot_hash,
                    source_data={"field": invalid_field},
                    rejected_at=timestamp,
                )
            )
            continue
        row_count = len(arrays["accessionNumber"])
        if any(len(values) != row_count for values in arrays.values()):
            rejection_rows.append(
                _rejection(
                    stage="submissions",
                    reason="unequal_columnar_lengths",
                    detail="Required filing arrays have unequal lengths",
                    path=source_document,
                    snapshot_hash=snapshot_hash,
                    source_data={key: len(value) for key, value in arrays.items()},
                    rejected_at=timestamp,
                )
            )
            continue
        for index in range(row_count):
            row = {field: arrays[field][index] for field in required_arrays}
            if row["form"] not in ALLOWED_FORMS:
                continue
            path = f"{source_document}[{index}]"
            try:
                accession = _required_accession(row["accessionNumber"])
                accepted_at = _required_utc_timestamp(
                    row["acceptanceDateTime"], "acceptanceDateTime"
                )
                filing_date = _required_date(row["filingDate"], "filingDate")
                report_date = _required_date(row["reportDate"], "reportDate")
                primary_document = row["primaryDocument"]
                if primary_document is not None and not isinstance(primary_document, str):
                    raise ValueError("primaryDocument must be a string or null")
                archive_base = (
                    "https://www.sec.gov/Archives/edgar/data/"
                    f"{int(normalized_cik)}/{accession.replace('-', '')}"
                )
                source_url = (
                    f"{archive_base}/{primary_document}" if primary_document else archive_base
                )
                filing_rows.append(
                    {
                        "filing_id": _stable_hash(
                            {
                                "cik": normalized_cik,
                                "accession": accession,
                                "accepted_at": accepted_at.isoformat(),
                            }
                        ),
                        "cik": normalized_cik,
                        "ticker": ticker,
                        "company_name": company_name,
                        "accession_number": accession,
                        "form": str(row["form"]),
                        "is_amendment": str(row["form"]).endswith("/A"),
                        "filing_date": filing_date,
                        "report_date": report_date,
                        "accepted_at": accepted_at,
                        "primary_document": primary_document,
                        "source_url": source_url,
                        "source_snapshot_sha256": snapshot_hash,
                        "source_document": source_document,
                    }
                )
            except (TypeError, ValueError) as exc:
                rejection_rows.append(
                    _rejection(
                        stage="submissions",
                        reason="invalid_filing_metadata",
                        detail=str(exc),
                        path=path,
                        snapshot_hash=snapshot_hash,
                        source_data=row,
                        rejected_at=timestamp,
                    )
                )
    filings = pd.DataFrame(filing_rows, columns=FILING_COLUMNS)
    if not filings.empty:
        duplicate_groups = filings.loc[filings.duplicated("accession_number", keep=False)].groupby(
            "accession_number", sort=False
        )
        for duplicate_key, duplicates in duplicate_groups:
            duplicate_accession = str(duplicate_key)
            conflicts = duplicates[
                ["form", "accepted_at", "report_date", "primary_document"]
            ].drop_duplicates()
            reason = (
                "conflicting_filing_metadata" if len(conflicts) > 1 else "duplicate_filing_metadata"
            )
            for _, duplicate in duplicates.iloc[1:].iterrows():
                rejection_rows.append(
                    _rejection(
                        stage="submissions",
                        reason=reason,
                        detail=f"Accession {duplicate_accession} appears more than once",
                        path=str(duplicate["source_document"]),
                        snapshot_hash=str(duplicate["source_snapshot_sha256"]),
                        source_data={"accession_number": duplicate_accession},
                        rejected_at=timestamp,
                    )
                )
        filings = filings.drop_duplicates("accession_number", keep="first")
        filings = filings.sort_values(["accepted_at", "accession_number"]).reset_index(drop=True)
    rejections = pd.DataFrame(rejection_rows, columns=REJECTION_COLUMNS)
    return ParseResult(filings, rejections)


def parse_companyfacts(
    document: dict[str, Any],
    filings: pd.DataFrame,
    *,
    source_url: str,
    snapshot_hash: str,
    parsed_at: datetime | None = None,
) -> ParseResult:
    """Flatten standardized companyfacts with accession-level availability times."""
    timestamp = parsed_at or datetime.now(timezone.utc)
    try:
        document_cik = normalize_cik(str(document.get("cik", "")))
    except ValueError as exc:
        raise ValueError(f"Invalid companyfacts CIK: {exc}") from exc
    if filings.empty:
        raise ValueError("Companyfacts cannot be normalized without filing metadata")
    expected_ciks = set(filings["cik"].astype(str))
    if expected_ciks != {document_cik}:
        raise ValueError("Companyfacts CIK does not match submissions metadata")
    accepted_by_accession = filings.set_index("accession_number")["accepted_at"].to_dict()
    facts_root = document.get("facts")
    if not isinstance(facts_root, dict):
        raise ValueError("companyfacts payload has no facts object")

    fact_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    occurrences: defaultdict[str, int] = defaultdict(int)
    for taxonomy, concepts in facts_root.items():
        if not isinstance(taxonomy, str) or not isinstance(concepts, dict):
            continue
        if taxonomy not in STANDARD_TAXONOMIES:
            rejection_rows.append(
                _rejection(
                    stage="companyfacts",
                    reason="non_standard_taxonomy",
                    detail=f"Taxonomy {taxonomy} is outside the supported standardized set",
                    path=f"facts.{taxonomy}",
                    snapshot_hash=snapshot_hash,
                    source_data={"taxonomy": taxonomy},
                    rejected_at=timestamp,
                )
            )
            continue
        for tag, concept in concepts.items():
            if not isinstance(tag, str) or not isinstance(concept, dict):
                continue
            units = concept.get("units")
            if not isinstance(units, dict):
                rejection_rows.append(
                    _rejection(
                        stage="companyfacts",
                        reason="missing_units_mapping",
                        detail=f"{taxonomy}:{tag} has no units mapping",
                        path=f"facts.{taxonomy}.{tag}",
                        snapshot_hash=snapshot_hash,
                        source_data=concept,
                        rejected_at=timestamp,
                    )
                )
                continue
            for unit, source_rows in units.items():
                unit_path = f"facts.{taxonomy}.{tag}.units.{unit}"
                if (
                    not isinstance(unit, str)
                    or UNIT_PATTERN.fullmatch(unit) is None
                    or not isinstance(source_rows, list)
                ):
                    rejection_rows.append(
                        _rejection(
                            stage="companyfacts",
                            reason="invalid_unit_bucket",
                            detail=f"{taxonomy}:{tag} has an invalid unit bucket",
                            path=unit_path,
                            snapshot_hash=snapshot_hash,
                            source_data={"unit": unit},
                            rejected_at=timestamp,
                        )
                    )
                    continue
                for index, source_row in enumerate(source_rows):
                    path = f"{unit_path}[{index}]"
                    if not isinstance(source_row, dict):
                        rejection_rows.append(
                            _rejection(
                                stage="companyfacts",
                                reason="invalid_fact_record",
                                detail="Fact row must be an object",
                                path=path,
                                snapshot_hash=snapshot_hash,
                                source_data={"row": source_row},
                                rejected_at=timestamp,
                            )
                        )
                        continue
                    if source_row.get("form") not in ALLOWED_FORMS:
                        continue
                    try:
                        accession = _required_accession(source_row.get("accn"))
                        accepted_at = accepted_by_accession.get(accession)
                        if accepted_at is None:
                            raise MissingAcceptanceTimestamp(
                                f"No accession-level acceptance timestamp for {accession}"
                            )
                        value_decimal = _required_decimal(source_row.get("val"))
                        period_end = _required_date(source_row.get("end"), "end")
                        period_start = _optional_date(source_row.get("start"), "start")
                        source_filed_date = _required_date(source_row.get("filed"), "filed")
                        form = str(source_row.get("form"))
                        semantic = {
                            "cik": document_cik,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "source_row": source_row,
                        }
                        semantic_hash = _stable_hash(semantic)
                        occurrence = occurrences[semantic_hash]
                        occurrences[semantic_hash] += 1
                        fact_id = _stable_hash(
                            {"semantic_hash": semantic_hash, "occurrence": occurrence}
                        )
                        duration_days = (
                            (period_end - period_start).days if period_start is not None else None
                        )
                        fact_rows.append(
                            {
                                "fact_id": fact_id,
                                "cik": document_cik,
                                "taxonomy": taxonomy,
                                "tag": tag,
                                "label": _optional_text(concept.get("label")),
                                "description": _optional_text(concept.get("description")),
                                "unit": unit,
                                "value": format(value_decimal, "f"),
                                "value_numeric": float(value_decimal),
                                "period_start": period_start,
                                "period_end": period_end,
                                "duration_days": duration_days,
                                "fiscal_year": _optional_integer(source_row.get("fy"), "fy"),
                                "fiscal_period": _optional_text(source_row.get("fp")),
                                "frame": _optional_text(source_row.get("frame")),
                                "accession_number": accession,
                                "form": form,
                                "base_form": form.removesuffix("/A"),
                                "is_amendment": form.endswith("/A"),
                                "source_filed_date": source_filed_date,
                                "accepted_at": accepted_at,
                                "source_url": source_url,
                                "source_record_path": path,
                                "source_snapshot_sha256": snapshot_hash,
                                "source_occurrence": occurrence,
                                "quality_status": "accepted_standard_fact",
                            }
                        )
                    except MissingAcceptanceTimestamp as exc:
                        rejection_rows.append(
                            _rejection(
                                stage="companyfacts",
                                reason="missing_acceptance_timestamp",
                                detail=str(exc),
                                path=path,
                                snapshot_hash=snapshot_hash,
                                source_data=source_row,
                                rejected_at=timestamp,
                            )
                        )
                    except (InvalidOperation, OverflowError, TypeError, ValueError) as exc:
                        rejection_rows.append(
                            _rejection(
                                stage="companyfacts",
                                reason="invalid_financial_fact",
                                detail=str(exc),
                                path=path,
                                snapshot_hash=snapshot_hash,
                                source_data=source_row,
                                rejected_at=timestamp,
                            )
                        )
    facts = pd.DataFrame(fact_rows, columns=FACT_COLUMNS)
    if not facts.empty:
        facts = facts.sort_values(
            ["accepted_at", "accession_number", "taxonomy", "tag", "fact_id"]
        ).reset_index(drop=True)
    rejections = pd.DataFrame(rejection_rows, columns=REJECTION_COLUMNS)
    return ParseResult(facts, rejections)


def compare_amendment_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Pair amended facts with the latest comparable pre-amendment disclosure.

    Comparison keys are deliberately strict: taxonomy, tag, unit, period start,
    period end, and base form must all match. New amendment-only facts are retained
    with ``no_prior_match`` rather than being treated as zero-valued originals.
    """
    if facts.empty or not facts["is_amendment"].any():
        return pd.DataFrame(columns=AMENDMENT_COLUMNS)
    keys = ["taxonomy", "tag", "unit", "period_start", "period_end", "base_form"]
    amended = facts.loc[facts["is_amendment"]].copy()
    originals = facts.loc[~facts["is_amendment"]].copy()
    candidates = amended.merge(originals, on=keys, how="inner", suffixes=("_amend", "_original"))
    if not candidates.empty:
        candidates = candidates.loc[
            candidates["accepted_at_original"] < candidates["accepted_at_amend"]
        ].copy()
        candidates = candidates.sort_values(
            ["fact_id_amend", "accepted_at_original", "fact_id_original"]
        ).drop_duplicates("fact_id_amend", keep="last")
    matched_amendment_ids = set(candidates.get("fact_id_amend", pd.Series(dtype=str)).astype(str))
    output_rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        original_value = float(row["value_numeric_original"])
        amendment_value = float(row["value_numeric_amend"])
        difference = amendment_value - original_value
        changed = not math.isclose(original_value, amendment_value, rel_tol=1e-12, abs_tol=0.0)
        output_rows.append(
            {
                "amendment_fact_id": row["fact_id_amend"],
                "taxonomy": row["taxonomy"],
                "tag": row["tag"],
                "label": row["label_amend"],
                "unit": row["unit"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "base_form": row["base_form"],
                "original_accession": row["accession_number_original"],
                "original_accepted_at": row["accepted_at_original"],
                "original_value": original_value,
                "amendment_accession": row["accession_number_amend"],
                "amendment_accepted_at": row["accepted_at_amend"],
                "amendment_value": amendment_value,
                "absolute_difference": difference,
                "percentage_difference": (
                    difference / abs(original_value) if original_value != 0.0 else np.nan
                ),
                "comparison_status": "changed" if changed else "unchanged",
            }
        )
    unmatched = amended.loc[~amended["fact_id"].astype(str).isin(matched_amendment_ids)]
    for _, row in unmatched.iterrows():
        output_rows.append(
            {
                "amendment_fact_id": row["fact_id"],
                "taxonomy": row["taxonomy"],
                "tag": row["tag"],
                "label": row["label"],
                "unit": row["unit"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "base_form": row["base_form"],
                "original_accession": None,
                "original_accepted_at": pd.NaT,
                "original_value": np.nan,
                "amendment_accession": row["accession_number"],
                "amendment_accepted_at": row["accepted_at"],
                "amendment_value": float(row["value_numeric"]),
                "absolute_difference": np.nan,
                "percentage_difference": np.nan,
                "comparison_status": "no_prior_match",
            }
        )
    return (
        pd.DataFrame(output_rows, columns=AMENDMENT_COLUMNS)
        .sort_values(["amendment_accepted_at", "tag", "period_end"])
        .reset_index(drop=True)
    )


class MissingAcceptanceTimestamp(ValueError):
    """Internal distinction for unsafe date-only companyfacts rows."""


def _submissions_array_object(document: dict[str, Any]) -> dict[str, Any] | None:
    filings = document.get("filings")
    if isinstance(filings, dict) and isinstance(filings.get("recent"), dict):
        recent = filings["recent"]
        assert isinstance(recent, dict)
        return recent
    if "accessionNumber" in document:
        return document
    return None


def _required_accession(value: Any) -> str:
    if not isinstance(value, str) or ACCESSION_PATTERN.fullmatch(value) is None:
        raise ValueError("accessionNumber has an invalid format")
    return value


def _required_utc_timestamp(value: Any, field: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return pd.Timestamp(parsed)


def _required_date(value: Any, field: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} must be an ISO date")
    return pd.Timestamp(parsed).normalize()


def _optional_date(value: Any, field: str) -> pd.Timestamp | None:
    return None if value is None else _required_date(value, field)


def _required_decimal(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError("val must be a finite numeric value")
    decimal = Decimal(str(value))
    if not decimal.is_finite() or not math.isfinite(float(decimal)):
        raise ValueError("val must be a finite numeric value")
    return decimal


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    integer = int(value)
    if str(integer) != str(value):
        raise ValueError(f"{field} must be an integer or null")
    return integer


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string or null")
    return value


def _rejection(
    *,
    stage: str,
    reason: str,
    detail: str,
    path: str,
    snapshot_hash: str,
    source_data: Any,
    rejected_at: datetime,
) -> dict[str, Any]:
    encoded_source = json.dumps(source_data, sort_keys=True, default=str, ensure_ascii=False)
    rejection_id = _stable_hash(
        {
            "stage": stage,
            "reason": reason,
            "path": path,
            "snapshot_hash": snapshot_hash,
            "source_data": encoded_source,
        }
    )
    return {
        "rejection_id": rejection_id,
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "source_record_path": path,
        "source_snapshot_sha256": snapshot_hash,
        "source_data": encoded_source,
        "rejected_at": rejected_at,
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
