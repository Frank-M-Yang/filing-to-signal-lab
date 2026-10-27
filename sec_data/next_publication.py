"""Original release labels and window metadata, never certified first-public clocks."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from sec_data.disclosure_audit import utc_timestamp
from sec_data.execution_contract import execution_window
from sec_data.filing_text import extract_visible_filing_text
from sec_data.publication_chronology import FORECAST_CLOCKS

VERSION = "next-publication-audit-v1"
PRICE_COLUMNS = ["price_target_id", "ticker", "session_date", "source_sha256"]
BENCHMARK_COLUMNS = ["ticker", "session_date", "source_sha256"]
DISARM_FLAGS = (
    "first_public_certified",
    "model_admission",
    "return_admission",
    "trading_admission",
)


class _PublicationMetadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.times: list[str] = []
        self.graphs: list[Any] = []
        self.meta: list[dict[str, str | None]] = []
        self._script = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "time" and values.get("datetime"):
            self.times.append(str(values["datetime"]))
        if tag == "meta":
            self.meta.append(values)
        if tag == "script" and values.get("type") == "application/ld+json":
            self._script, self._parts = True, []

    def handle_data(self, data: str) -> None:
        if self._script:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script:
            self.graphs.append(json.loads("".join(self._parts)))
            self._script = False


def _nodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value, *(r for v in value.values() for r in _nodes(v))]
    if isinstance(value, list):
        return [r for v in value for r in _nodes(v)]
    return []


def _clock(content: bytes, text: str, spec: dict[str, Any]) -> dict[str, Any]:
    parser = _PublicationMetadata()
    parser.feed(content.decode("utf-8", errors="strict"))
    literal, kind = spec["clock_literal"], spec["clock_kind"]
    anchor = spec["clock_anchor"]
    if text.count(anchor) != 1 or spec["visible_clock_literal"] not in anchor:
        raise ValueError("Missing or ambiguous visible publication anchor")
    visible = pd.Timestamp(
        datetime.strptime(spec["visible_clock_literal"], spec["visible_clock_format"])
    )
    if visible.date().isoformat() != spec["reported_date"]:
        raise ValueError("Visible publication date differs")
    if kind == "visible_naive_minute":
        if literal != visible.isoformat() or spec.get("visible_zone") is not None:
            raise ValueError("Unknown timezone cannot be imputed")
        labels = [
            m.get("content")
            for m in parser.meta
            if m.get("property") == "og:article:published_time"
        ]
        if (
            labels != [spec["meta_clock_literal"]]
            or pd.Timestamp(datetime.strptime(str(labels[0]), spec["meta_clock_format"])) != visible
        ):
            raise ValueError("Naive publication metadata differs")
        lower = visible.tz_localize("UTC") - pd.Timedelta(hours=14)
        upper = visible.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=1)
        stamp = None
        basis = "unknown_offset_sensitivity_only_not_an_evidence_bound"
    else:
        if kind == "time_datetime":
            if parser.times != [literal]:
                raise ValueError("Missing or ambiguous time element")
        elif kind == "jsonld_datePublished":
            nodes = []
            for graph in parser.graphs:
                for node in _nodes(graph):
                    entity = node.get("mainEntityOfPage")
                    entity = entity.get("@id") if isinstance(entity, dict) else entity
                    identity = node.get("url", node.get("@id", entity))
                    if (
                        node.get("@type") in {"NewsArticle", "WebPage"}
                        and identity == spec["canonical_url"]
                        and "datePublished" in node
                    ):
                        nodes.append(node)
            if len(nodes) != 1 or nodes[0].get("datePublished") != literal:
                raise ValueError(
                    "Canonical publication metadata differs; dateModified is forbidden"
                )
            if spec["canonical_url"].rstrip("/") != spec["url"].rstrip("/"):
                raise ValueError("Canonical page belongs to another release")
        else:
            raise ValueError("Unsupported publication clock selector")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})", literal):
            raise ValueError("Publication metadata needs an explicit offset")
        stamp = utc_timestamp(literal)
        local = stamp.tz_convert("America/New_York")
        if local.tz_localize(None).floor("min") != visible or spec["visible_zone"] not in {
            "ET",
            "EDT",
            "EST",
        }:
            raise ValueError("Visible and structured publication clocks disagree")
        if spec["visible_zone"] != "ET" and local.strftime("%Z") != spec["visible_zone"]:
            raise ValueError("Explicit timezone conflicts with DST")
        # A :00 metadata field is not proof of second-precision historical availability.
        # Use the coarser visible minute even when the CMS happens to record seconds.
        lower, upper = stamp.floor("min"), stamp.floor("min") + pd.Timedelta(minutes=1)
        basis = "current_publisher_reported_minute_not_first_public"
    return {
        "clock_kind": kind,
        "clock_literal": literal,
        "visible_clock_literal": spec["visible_clock_literal"],
        "clock_precision": "minute_explicit_offset"
        if stamp is not None
        else "minute_unknown_timezone",
        "clock_resolution_seconds": 60,
        "reported_date": spec["reported_date"],
        "source_metadata_at_utc": None if stamp is None else stamp.isoformat(),
        "source_reported_at_utc": None if stamp is None else lower.isoformat(),
        "lower_inclusive": lower.isoformat(),
        "upper_exclusive": upper.isoformat(),
        "clock_basis": basis,
    }


def review_publication(
    content: bytes, spec: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind the original quarter's first GAAP column, its display scale and source label.

    Historical SEC cells provide quarter start/identity and original-value comparison;
    the publisher's bounded income-statement region must reproduce both literal rows.
    Neither a rounded narrative nor product-only/non-GAAP rows can substitute.
    """
    if hashlib.sha256(content).hexdigest() != spec["snapshot_sha256"]:
        raise ValueError("Publisher snapshot hash differs")
    text = re.sub(r"\s+", " ", extract_visible_filing_text(content))
    if not spec["context_anchors"] or any(a not in text for a in spec["context_anchors"]):
        raise ValueError("Issuer/release context differs")
    if text.count(spec["statement_anchor"]) != 1 or not 500 <= spec["scope_max_chars"] <= 6000:
        raise ValueError("Ambiguous income statement or unbounded financial region")
    start = text.index(spec["statement_anchor"])
    region = text[start : start + spec["scope_max_chars"]]
    if any(a not in region for a in [spec["quarter_anchor"], spec["scale_anchor"]]):
        raise ValueError("Wrong quarterly GAAP column or display unit")
    if spec["display_scale"] not in {1, 1000, 1000000}:
        raise ValueError("Unsupported display scale")
    metrics = []
    for item in spec["cells"]:
        anchor, label = item["anchor"], item["row_label"]
        if region.count(anchor) != 1 or not anchor.startswith(label):
            raise ValueError("Financial row missing or ambiguous")
        tokens = re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?", anchor[len(label) :])
        if not tokens or tokens[0] != item["display_value"] or item["column_index"] != 0:
            raise ValueError("Must use the first current-quarter GAAP column")
        value = Decimal(tokens[0].replace(",", "")) * spec["display_scale"]
        if value != Decimal(str(item["value_usd"])):
            raise ValueError("Display literal/scale does not reproduce USD value")
        matching = [
            c
            for c in cells
            if all(
                c[k] == spec[k]
                for k in ("ticker", "cik", "release_accession", "period_start", "period_end")
            )
            and c["metric"] == item["metric"]
            and c["unit"] == "USD"
        ]
        if (
            len(matching) != 1
            or matching[0]["comparison_status"] != "exact_original_match"
            or any(Decimal(str(matching[0][k])) != value for k in ("value", "original_value"))
        ):
            raise ValueError("Publisher row disagrees with the exact original SEC quarter")
        metrics.append({**item, "display_scale": spec["display_scale"], "unit": "USD"})
    if len(metrics) != 2 or {m["metric"] for m in metrics} != {"revenue", "gross_profit"}:
        raise ValueError("Both exact original gross-margin components are required")
    return {
        **{
            k: spec[k]
            for k in (
                "source_id",
                "ticker",
                "cik",
                "release_accession",
                "url",
                "snapshot_sha256",
                "period_start",
                "period_end",
            )
        },
        **_clock(content, text, spec),
        "matched_cells": metrics,
        "exact_original_pair": True,
        "current_page_capture_not_historical_archive": True,
        "first_public_at": None,
        "public_available_by": None,
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def compare_forecast_clocks(source: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Test strict-prior ordering over an entire label precision/offset sensitivity range."""
    lower, upper = (utc_timestamp(source[k]) for k in ("lower_inclusive", "upper_exclusive"))
    if not lower < upper:
        raise ValueError("Invalid clock sensitivity interval")
    comparisons = {}
    for field in FORECAST_CLOCKS:
        value = metadata.get(field)
        stamp = None if value is None or pd.isna(value) else utc_timestamp(value)
        comparisons[field] = {
            "at": None if stamp is None else stamp.isoformat(),
            "status": "missing"
            if stamp is None
            else "before_all"
            if stamp < lower
            else "after_all"
            if stamp >= upper
            else "overlap",
        }
    before = all(r["status"] == "before_all" for r in comparisons.values())
    return {
        "forecast_comparisons": comparisons,
        "prior_to_all_label_scenarios": before,
        "financial_reconstruction_status": "clock_only_check_not_dependency_rebuild",
        "rebuild_required_before_innovation_use": not before,
        "forecast_recomputed": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def label_window_scenarios(
    source: dict[str, Any], exchange_mic: str, sessions: pd.DataFrame
) -> list[dict[str, Any]]:
    """Enumerate date-based schedules without silently imputing unknown timezones.

    The date endpoints enumerate every local date intersecting the sensitivity
    interval, not evidence bounds. All windows remain disarmed even if unchanged.
    """
    lower, upper = (utc_timestamp(source[k]) for k in ("lower_inclusive", "upper_exclusive"))
    if not lower < upper or upper - lower > pd.Timedelta(hours=29):
        raise ValueError("Invalid publication sensitivity interval")
    lo = lower.tz_convert("America/New_York").date()
    hi = (upper - pd.Timedelta(nanoseconds=1)).tz_convert("America/New_York").date()
    result = []
    for date in pd.date_range(lo, hi):
        # Representative noon is a scheduling device, NEVER an availability estimate.
        representative = (date + pd.Timedelta(hours=12)).tz_localize("America/New_York")
        row = {
            "scenario_local_date": date.date().isoformat(),
            "basis": source["clock_basis"],
            "representative_time_is_not_publication_time": True,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        try:
            row["schedule"] = execution_window(representative.isoformat(), exchange_mic, sessions)
            row["schedule"]["announcement_at"] = None
        except ValueError as exc:
            row["schedule"] = None
            row["unresolved_reason"] = str(exc)
        result.append(row)
    return result


def audit_price_dates(
    case: dict[str, Any],
    schedule: dict[str, Any],
    sessions: pd.DataFrame,
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict[str, Any]:
    """Check exact issuer/query/date metadata only; do not open numeric prices or returns."""
    if prices.columns.tolist() != PRICE_COLUMNS or benchmark.columns.tolist() != BENCHMARK_COLUMNS:
        raise ValueError("Numeric prices, returns or unapproved metadata columns are forbidden")
    mic = case["assigned_exchange_mic"]
    if schedule["exchange_mic"] != mic:
        raise ValueError("Cannot substitute another exchange's calendar")
    chosen = targets.loc[
        targets.cik.eq(case["cik"])
        & targets.selected_ticker.eq(case["ticker"])
        & targets.selected_exchange_mic.eq(mic)
    ]
    expected = (
        sessions.loc[
            sessions.exchange_mic.eq(mic)
            & sessions.session_date.between(
                schedule["entry_session_date"], schedule["exit_session_date"]
            ),
            "session_date",
        ]
        .sort_values()
        .tolist()
    )
    if len(expected) != 20 or len(set(expected)) != 20:
        raise ValueError("Calendar must contain exactly twenty distinct holding sessions")
    if len(chosen) != 1:
        return {
            "status": "missing_or_ambiguous_query_identity",
            "metadata_window_complete": False,
            "expected_dates": expected,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
    target = chosen.iloc[0]
    if bool(target.adjusted) or pd.isna(target.security_identity_key):
        raise ValueError("Query must retain an unadjusted identified security")
    issuer = prices.loc[
        prices.price_target_id.eq(target.price_target_id) & prices.session_date.isin(expected)
    ]
    market = benchmark.loc[benchmark.ticker.eq("SPY") & benchmark.session_date.isin(expected)]
    if not issuer.ticker.eq(case["ticker"]).all():
        raise ValueError("Price query contains another ticker")
    for frame in (issuer, market):
        if (
            frame.session_date.duplicated().any()
            or not frame.source_sha256.str.fullmatch(r"[a-f0-9]{64}").fillna(False).all()
        ):
            raise ValueError("Duplicate price dates or invalid raw snapshot provenance")
    missing_issuer = sorted(set(expected) - set(issuer.session_date))
    missing_benchmark = sorted(set(expected) - set(market.session_date))
    return {
        "status": "date_metadata_complete"
        if not missing_issuer and not missing_benchmark
        else "date_metadata_gap",
        "metadata_window_complete": not missing_issuer and not missing_benchmark,
        "price_target_id": target.price_target_id,
        "security_identity_key": target.security_identity_key,
        "security_identity_key_kind": target.security_identity_key_kind,
        "expected_dates": expected,
        "missing_issuer_dates": missing_issuer,
        "missing_benchmark_dates": missing_benchmark,
        "issuer_snapshot_sha256": sorted(issuer.source_sha256.unique()),
        "benchmark_snapshot_sha256": sorted(market.source_sha256.unique()),
        "identity_interval_and_actions_reauthorized": False,
        "price_values_loaded": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
