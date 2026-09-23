"""Outcome-blind migration of a frozen execution cohort and publication evidence."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from sec_data.clock_layer import calendar_windows
from sec_data.disclosure_audit import utc_timestamp
from sec_data.execution_contract import execution_window
from sec_data.filing_text import extract_visible_filing_text

VERSION = "sec-execution-migration-v1"
EVENT_COLUMNS = [
    "validation_event_id",
    "cik",
    "target_accession_number",
    "form",
    "accepted_at",
    "assigned_ticker",
    "assigned_exchange_mic",
    "security_component_count",
    "formal_entry_open_at",
    "formal_exit_close_at",
    "formal_entry_price_type",
    "formal_exit_price_type",
    "entry_session_offset",
    "holding_sessions",
    "selection_uses_return_values",
    "outcome_values_loaded",
]


def migrate_execution_windows(
    events: pd.DataFrame,
    layer: list[dict[str, Any]],
    sessions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Keep every event; separate convention effects from source-clock corrections.

    This is a new schedule contract, not authority to reuse old price, identity,
    corporate-action or delisting evidence over a changed holding interval.
    """
    if set(events.columns) != set(EVENT_COLUMNS):
        raise ValueError("Migration must receive projected metadata only")
    if events.validation_event_id.duplicated().any() or events.empty:
        raise ValueError("Invalid frozen event identity")
    if events.isna().any().any():
        raise ValueError("Missing execution metadata cannot pass migration")
    if events.selection_uses_return_values.any() or events.outcome_values_loaded.any():
        raise ValueError("Frozen cohort was selected using outcomes")
    if not events.entry_session_offset.eq(2).all() or not events.holding_sessions.eq(20).all():
        raise ValueError("Legacy contract is not the expected 2/20 convention")
    clocks = {(r["cik"], r["accession_number"]): r for r in layer}
    if len(clocks) != len(layer):
        raise ValueError("Duplicate clock overlay identity")
    result = []
    for event in events.to_dict("records"):
        key = (event["cik"], event["target_accession_number"])
        old = utc_timestamp(event["accepted_at"]).isoformat()
        venue = event["assigned_exchange_mic"]
        legacy = calendar_windows(old, venue, sessions)["legacy_second_future_open"]
        if (
            utc_timestamp(event["formal_entry_open_at"]) != utc_timestamp(legacy["entry_at"])
            or utc_timestamp(event["formal_exit_close_at"]) != utc_timestamp(legacy["exit_at"])
            or event["formal_entry_price_type"] != "unadjusted_open"
            or event["formal_exit_price_type"] != "unadjusted_close"
        ):
            raise ValueError("Stored legacy window does not reproduce; inspect before migration")
        clock = clocks.get(key)
        corrected = old
        status = "catalog_proxy_unreviewed"
        if clock is not None:
            if (
                utc_timestamp(clock["catalog_accepted_at"]) != utc_timestamp(old)
                or clock["form"] != event["form"]
            ):
                raise ValueError("Clock view is not an exact event match")
            corrected = clock["experimental_acceptance_proxy_at"]
            status = clock["resolution_status"]
        same_clock = execution_window(old, venue, sessions)
        corrected_window = execution_window(corrected, venue, sessions) if corrected else None
        rule_changed = (
            same_clock["entry_at"] != legacy["entry_at"]
            or same_clock["exit_at"] != legacy["exit_at"]
        )
        clock_changed = (
            None
            if corrected_window is None
            else (
                corrected_window["entry_at"] != same_clock["entry_at"]
                or corrected_window["exit_at"] != same_clock["exit_at"]
            )
        )
        final_changed = (
            None
            if corrected_window is None
            else (
                corrected_window["entry_at"] != legacy["entry_at"]
                or corrected_window["exit_at"] != legacy["exit_at"]
            )
        )
        result.append(
            {
                "validation_event_id": event["validation_event_id"],
                "cik": event["cik"],
                "accession_number": event["target_accession_number"],
                "ticker": event["assigned_ticker"],
                "form": event["form"],
                "exchange_mic": venue,
                "old_acceptance_proxy_at": old,
                "new_acceptance_proxy_at": corrected,
                "clock_review_status": status,
                "legacy_stored": legacy,
                "new_contract_old_clock": same_clock,
                "new_contract_reviewed_clock": corrected_window,
                "rule_changes_window": rule_changed,
                "clock_changes_window_under_new_rule": clock_changed,
                "final_window_differs_from_legacy": final_changed,
                "source_clock_changed": corrected is not None
                and utc_timestamp(corrected) != utc_timestamp(old),
                "security_components": event["security_component_count"],
                "evidence_revalidation_required": [
                    "publication_clock",
                    "new_window_security_identity",
                    "daily_bars",
                    "corporate_actions",
                    "delisting_outcomes",
                    "benchmark_window",
                    "forecast_information_set",
                ],
                "price_read_authorized": False,
                "return_admission": False,
                "primary_sample_reselected": False,
            }
        )
    return result


class _TimeElements(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        value = dict(attrs).get("datetime")
        if tag == "time" and value:
            self.values.append(value)


def review_release_page(
    content: bytes,
    specification: dict[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind a current publisher page to reviewed financial cells, without inventing timezone."""
    text = re.sub(r"\s+", " ", extract_visible_filing_text(content))
    if any(anchor not in text for anchor in specification["context_anchors"]):
        raise ValueError("Publisher issuer/quarter/unit context differs")
    matches = []
    for expected in specification["cells"]:
        if expected["anchor"] not in text:
            raise ValueError("Publisher financial-cell anchor missing")
        selected = [
            c
            for c in cells
            if c["cik"] == specification["cik"]
            and c["release_accession"] == specification["release_accession"]
            and c["metric"] == expected["metric"]
            and c["period_end"] == specification["period_end"]
            and c["period_start"] == specification["period_start"]
            and c["unit"] == "USD"
        ]
        if len(selected) != 1 or selected[0]["comparison_status"] != "exact_original_match":
            raise ValueError("Financial cell not matched in frozen SEC review")
        if (
            selected[0]["value"] != expected["value_usd"]
            or selected[0]["original_value"] != expected["value_usd"]
        ):
            raise ValueError("Publisher value differs from reviewed original")
        matches.append(expected["metric"])
    parser = _TimeElements()
    parser.feed(content.decode("utf-8", errors="replace"))
    if parser.values != [specification["time_literal"]]:
        raise ValueError("Publisher time element missing or ambiguous")
    literal = parser.values[0]
    aware = bool(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", literal)
    )
    return {
        "ticker": specification["ticker"],
        "cik": specification["cik"],
        "release_accession": specification["release_accession"],
        "matched_metrics": matches,
        "period_start": specification["period_start"],
        "period_end": specification["period_end"],
        "source_reported_time_literal": literal,
        "source_reported_at_utc": utc_timestamp(literal).isoformat() if aware else None,
        "clock_status": "publisher_label_only_not_first_public"
        if aware
        else "timezone_unspecified",
        "historical_capture": False,
        "first_public_at": None,
        "public_available_by": None,
        "forecast_certification": False,
        "return_admission": False,
    }


def publication_review_queue(
    events: list[dict[str, Any]],
    source_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve the full review sample and list specific unresolved admission work."""
    # This version freezes one event per issuer; never attach a page to another quarter.
    if len({r["cik"] for r in events}) != len(events):
        raise ValueError("Publication review requires one frozen event per issuer")
    if not {r["cik"] for r in source_checks}.issubset({r["cik"] for r in events}):
        raise ValueError("Publisher source outside the frozen publication review")
    outputs = []
    for event in events:
        related = [s for s in source_checks if s["cik"] == event["cik"]]
        reasons = ["first_public_search_incomplete", "earlier_disclosure_forecast_rebuild_required"]
        if event["candidate_count"] == 0:
            reasons.append("no_SEC_candidate_does_not_prove_no_release")
        if "gross_margin" not in event["matched_acceptance_proxies"]:
            reasons.append("gross_margin_quarter_and_accounting_scope_unresolved")
        if not related:
            reasons.append("publisher_clock_not_collected_in_this_risk_batch")
        else:
            reasons.extend(s["clock_status"] for s in related)
        outputs.append(
            {
                "validation_event_id": event["validation_event_id"],
                "cik": event["cik"],
                "ticker": event["ticker"],
                "target_accession": event["filing_accession"],
                "candidate_count": event["candidate_count"],
                "prior_proxy_check": event["forecast_proxy_check"],
                "source_checks": related,
                "remaining_work": reasons,
                "return_admission": False,
                "sample_replacement_allowed": False,
            }
        )
    return outputs
