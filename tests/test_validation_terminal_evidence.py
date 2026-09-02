"""Tests for primary-source validation terminal evidence."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_terminal_evidence import (
    calculate_resolved_terminal_event_outcomes,
    calculate_validation_terminal_filing_targets,
    parse_cash_terminal_filing,
)


def test_terminal_target_selection_uses_nearby_completion_and_form25() -> None:
    repair = pd.DataFrame(
        [
            {
                "outcome_target_id": "outcome-1",
                "cik": "0000000001",
                "ticker": "ONE",
                "terminal_value_required_events": 1,
                "repair_selection_uses_return_values": False,
            }
        ]
    )
    outcomes = pd.DataFrame([{"outcome_target_id": "outcome-1", "delisting_date": "2024-11-22"}])
    filings = pd.DataFrame(
        [
            _filing("8-K", "2024-11-22", "2.01,3.01,9.01", "0000000001-24-000001"),
            _filing("25-NSE", "2024-11-22", "", "0000000001-24-000002"),
            _filing("8-K", "2024-11-21", "5.07", "0000000001-24-000003"),
        ]
    )

    targets, selections = calculate_validation_terminal_filing_targets(repair, outcomes, filings)

    assert set(targets["evidence_role"]) == {
        "terminal_completion_primary",
        "delisting_form25_corroboration",
    }
    assert selections.loc[0, "selection_status"] == "completion_and_form25_selected"
    assert not targets["selection_uses_return_values"].any()


def test_cash_terminal_parser_requires_unique_completed_consideration() -> None:
    payload = b"""
    <html><body>
    <h2>Closing of the Merger</h2>
    <p>On November 22, 2024 (the &ldquo;Closing Date&rdquo;), Parent completed the acquisition.</p>
    <p>Each share was cancelled and automatically converted into the right to receive cash
    in an amount equal to $3.40 per share, payable without interest.</p>
    </body></html>
    """

    parsed = parse_cash_terminal_filing(payload)

    assert parsed["terminal_evidence_ready"]
    assert parsed["terminal_effective_date"] == "2024-11-22"
    assert parsed["terminal_value"] == 3.4
    assert parsed["terminal_value_currency"] == "USD"


def test_verified_terminal_value_resolves_only_required_event() -> None:
    events = pd.DataFrame(
        [
            {
                "outcome_target_id": "outcome-1",
                "formal_entry_session_date": "2024-11-05",
                "formal_exit_session_date": "2024-12-03",
                "terminal_value_required": True,
                "market_outcome_ready": False,
                "outcome_exception_status": "terminal_value_required",
                "returns_computed": False,
            },
            {
                "outcome_target_id": "outcome-2",
                "formal_entry_session_date": "2024-11-05",
                "formal_exit_session_date": "2024-12-03",
                "terminal_value_required": False,
                "market_outcome_ready": True,
                "outcome_exception_status": "ready",
                "returns_computed": False,
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "outcome_target_id": "outcome-1",
                "terminal_effective_date": "2024-11-22",
                "terminal_value": 3.4,
                "terminal_value_currency": "USD",
                "terminal_evidence_ready": True,
                "source_sha256": "a" * 64,
                "accession_number": "0000000001-24-000001",
            }
        ]
    )

    resolved = calculate_resolved_terminal_event_outcomes(events, evidence)

    assert resolved["resolved_market_outcome_ready"].all()
    assert resolved.loc[0, "terminal_value_observed"]
    assert resolved.loc[0, "resolved_outcome_status"] == "ready_with_sec_terminal_value"
    assert resolved.loc[0, "terminal_source_accession_number"] == "0000000001-24-000001"
    assert not resolved.loc[1, "terminal_value_observed"]
    assert not resolved["returns_computed"].any()


def _filing(form: str, filing_date: str, items: str, accession: str) -> dict[str, object]:
    return {
        "cik": "0000000001",
        "accession_number": accession,
        "form": form,
        "filing_date": filing_date,
        "accepted_at": f"{filing_date}T12:00:00+00:00",
        "primary_document": "primary.htm",
        "items": items,
        "source_url": "https://www.sec.gov/Archives/edgar/data/1/primary.htm",
    }
