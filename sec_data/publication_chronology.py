"""Precision-aware publisher evidence, separate from certified information clocks."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.filing_text import extract_visible_filing_text

VERSION = "publication-chronology-v1"
FORECAST_CLOCKS = [
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_generated_at",
    "forecast_error_history_end",
]
PANEL_COLUMNS = [
    "event_id",
    "company_id",
    "filing_accession",
    "actual_available_at",
    *FORECAST_CLOCKS,
]


class _Metadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.times: list[str] = []
        self.graphs: list[Any] = []
        self._json = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "time" and values.get("datetime"):
            self.times.append(str(values["datetime"]))
        if tag == "script" and values.get("type") == "application/ld+json":
            self._json = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._json:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._json:
            self.graphs.append(json.loads("".join(self._parts)))
            self._json = False


def _objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value, *(r for v in value.values() for r in _objects(v))]
    if isinstance(value, list):
        return [r for v in value for r in _objects(v)]
    return []


def _clock(content: bytes, spec: dict[str, Any], text: str) -> tuple[str, pd.Timestamp | None]:
    parser = _Metadata()
    parser.feed(content.decode("utf-8", errors="strict"))
    literal, kind = spec["clock_literal"], spec["clock_kind"]
    if kind == "time_datetime":
        if parser.times != [literal]:
            raise ValueError("Missing or ambiguous time element")
    elif kind == "jsonld_datePublished":
        # Only this canonical page's publication field; never dateModified or image metadata.
        nodes = [
            n
            for graph in parser.graphs
            for n in _objects(graph)
            if n.get("@type") in ("WebPage", "NewsArticle")
            and n.get("url", n.get("@id")) == spec["url"]
            and "datePublished" in n
        ]
        if len(nodes) != 1 or nodes[0]["datePublished"] != literal:
            raise ValueError("Canonical-page publication metadata is missing or ambiguous")
    elif kind in {"visible_date", "visible_EST_minute", "visible_EDT_minute"}:
        if literal not in spec["clock_anchor"] or text.count(spec["clock_anchor"]) != 1:
            raise ValueError("Publication anchor is missing or ambiguous")
    else:
        raise ValueError("Unsupported publisher clock selector")
    if kind == "visible_date":
        parsed = datetime.strptime(literal, "%B %d, %Y").date().isoformat()
        if parsed != spec["reported_date"]:
            raise ValueError("Publication date label does not reproduce")
        return "date_only_no_timezone", None
    if kind in {"visible_EST_minute", "visible_EDT_minute"}:
        zone = "EST" if kind == "visible_EST_minute" else "EDT"
        offset = timedelta(hours=-5 if zone == "EST" else -4)
        if not literal.endswith(f" {zone}"):
            raise ValueError("Explicit timezone label differs from source selector")
        local = datetime.strptime(literal, f"%b %d, %Y %I:%M%p {zone}")
        stamp = pd.Timestamp(local.replace(tzinfo=timezone(offset)))
        if stamp.tz_convert("America/New_York").utcoffset() != offset:
            raise ValueError(f"{zone} label conflicts with local DST; do not guess")
        return f"minute_with_explicit_{zone}", stamp.tz_convert("UTC")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?", literal):
        raise ValueError("Unsupported datetime literal precision")
    parsed_stamp = pd.Timestamp(literal)
    if parsed_stamp.tzinfo is None:
        return "datetime_without_timezone", None
    return "second_with_explicit_offset", utc_timestamp(literal)


def review_publisher_source(
    content: bytes,
    spec: dict[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify exact page identity, clock provenance and financial precision independently."""
    if hashlib.sha256(content).hexdigest() != spec["snapshot_sha256"]:
        raise ValueError("Publisher snapshot hash differs")
    text = re.sub(r"\s+", " ", extract_visible_filing_text(content))
    if not spec["context_anchors"] or any(a not in text for a in spec["context_anchors"]):
        raise ValueError("Publisher context differs")
    if spec["content_scope"] not in {"full_financial_table", "rounded_summary"}:
        raise ValueError("Unsupported accounting precision")
    if spec["content_scope"] == "rounded_summary" and spec["cells"]:
        raise ValueError("Rounded summaries cannot supply exact financial cells")
    matched = []
    for item in spec["cells"]:
        if item["anchor"] not in text:
            raise ValueError("Financial cell anchor missing")
        selected = [
            c
            for c in cells
            if all(
                c[k] == spec[k] for k in ("cik", "release_accession", "period_start", "period_end")
            )
            and c["metric"] == item["metric"]
            and c["unit"] == "USD"
        ]
        if len(selected) != 1 or selected[0]["comparison_status"] != "exact_original_match":
            raise ValueError("Financial cell is not an exact original match")
        if any(selected[0][k] != item["value_usd"] for k in ("value", "original_value")):
            raise ValueError("Publisher value disagrees with reviewed SEC values")
        matched.append(item["metric"])
    if len(matched) != len(set(matched)):
        raise ValueError("Duplicate financial metric")
    full = {"revenue", "gross_profit"}.issubset(matched)
    if spec["content_scope"] == "full_financial_table" and not full:
        raise ValueError("Full-table GM review requires both exact component cells")
    precision, stamp = _clock(content, spec, text)
    return {
        "source_id": spec["source_id"],
        "ticker": spec["ticker"],
        "cik": spec["cik"],
        "release_accession": spec["release_accession"],
        "url": spec["url"],
        "snapshot_sha256": spec["snapshot_sha256"],
        "clock_kind": spec["clock_kind"],
        "clock_literal": spec["clock_literal"],
        "clock_precision": precision,
        "reported_date": spec.get("reported_date"),
        "source_reported_at_utc": None if stamp is None else stamp.isoformat(),
        "content_scope": spec["content_scope"],
        "matched_metrics": matched,
        "exact_gross_margin_components": full,
        "point_clock_and_exact_values": full and stamp is not None,
        "first_public_at": None,
        "public_available_by": None,
        "historical_capture": False,
        "model_admission": False,
        "return_admission": False,
    }


def chronology_sensitivity(
    reviewed: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_offset_hours: int = 14,
) -> dict[str, Any]:
    """Compare forecast metadata to labels; timezone scenarios are NOT evidence bounds.

    Unknown timezone is tested over [-14h,+14h], a declared conservative sensitivity
    parameter. No single timezone is imputed. Missing historical channel coverage
    prevents all first-public and return admission, even with a precise label.
    """
    if max_offset_hours != 14:
        raise ValueError("Changed sensitivity range requires a new protocol")
    precision = reviewed["clock_precision"]
    if reviewed["source_reported_at_utc"] is not None:
        lower = utc_timestamp(reviewed["source_reported_at_utc"])
        seconds = 60 if precision in {"minute_with_explicit_EST", "minute_with_explicit_EDT"} else 1
        upper = lower + pd.Timedelta(seconds=seconds)
        basis = "publisher_label_precision_not_first_public"
    else:
        if precision == "date_only_no_timezone":
            local = pd.Timestamp(reviewed["reported_date"])
            width = pd.Timedelta(days=1)
        elif precision == "datetime_without_timezone":
            local = pd.Timestamp(reviewed["clock_literal"])
            width = pd.Timedelta(seconds=1)
        else:
            raise ValueError("Unsupported uncertain clock precision")
        if local.tzinfo is not None:
            raise ValueError("Uncertain clock unexpectedly has timezone")
        lower = local.tz_localize("UTC") - pd.Timedelta(hours=max_offset_hours)
        upper = local.tz_localize("UTC") + width + pd.Timedelta(hours=max_offset_hours)
        basis = "offset_sensitivity_only_not_an_evidence_bound"
    comparisons = {}
    for field in FORECAST_CLOCKS:
        value = metadata.get(field)
        stamp = None if value is None or pd.isna(value) else utc_timestamp(value)
        comparisons[field] = {
            "at": None if stamp is None else stamp.isoformat(),
            "status": "missing"
            if stamp is None
            else "before_all_label_scenarios"
            if stamp < lower
            else "after_all_label_scenarios"
            if stamp >= upper
            else "overlaps_label_scenarios",
            "seconds_after_latest_scenario": None
            if stamp is None
            else (stamp - upper).total_seconds(),
        }
    return {
        "source_id": reviewed["source_id"],
        "basis": basis,
        "lower_inclusive": lower.isoformat(),
        "upper_exclusive": upper.isoformat(),
        "max_offset_hours_if_unknown": max_offset_hours
        if reviewed["source_reported_at_utc"] is None
        else None,
        "forecast_comparisons": comparisons,
        "applies_to_exact_gross_margin": reviewed["exact_gross_margin_components"],
        "first_public_certified": False,
        "forecast_recomputed": False,
        "return_admission": False,
    }
