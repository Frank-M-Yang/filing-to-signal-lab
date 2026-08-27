"""Outcome-blind cohort and event-registry tests for validation expansion."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_expansion import (
    ValidationExpansionConfig,
    calculate_validation_expansion,
)


def test_validation_expansion_freezes_unseen_events_without_replacement() -> None:
    scope, catalog, legacy, reference = _fixture_inputs()
    result = calculate_validation_expansion(
        scope,
        catalog,
        legacy,
        reference,
        ValidationExpansionConfig(collection_wave_issuers=2),
    )

    assert result.report["probability_sample_equity_units"] == 4
    assert result.report["cohort_issuers"] == 3
    assert result.report["candidate_events"] == 3
    assert result.report["unobserved_validation_candidate_events"] == 2
    assert result.report["outcome_values_loaded"] is False
    assert result.report["replacement_after_collection_failure_allowed"] is False
    roles = result.candidate_events.set_index("accession_number")["evidence_role"]
    assert roles["accession-1"] == "legacy_issuer_training_only"
    assert roles["accession-2"] == "unobserved_validation_candidate"
    assert roles["accession-3"] == "unobserved_validation_candidate"
    assert "amendment-2" not in set(result.candidate_events["accession_number"])
    assert "holdout-3" not in set(result.candidate_events["accession_number"])
    assert result.collection_waves["issuers"].tolist() == [2, 1]


def test_provider_success_cannot_change_selection_or_event_ids() -> None:
    scope, catalog, legacy, reference = _fixture_inputs()
    first = calculate_validation_expansion(scope, catalog, legacy, reference)
    changed = reference.copy()
    changed["provider_query_complete"] = ~changed["provider_query_complete"]
    changed["event_date_reference_candidate"] = ~changed["event_date_reference_candidate"]
    second = calculate_validation_expansion(scope, catalog, legacy, changed)

    cohort_columns = ["cik", "expansion_rank", "collection_wave", "evaluation_eligible_issuer"]
    pd.testing.assert_frame_equal(
        first.cohort_registry[cohort_columns],
        second.cohort_registry[cohort_columns],
    )
    event_columns = ["accession_number", "validation_event_id", "evidence_role"]
    pd.testing.assert_frame_equal(
        first.candidate_events[event_columns],
        second.candidate_events[event_columns],
    )


def _fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope = pd.DataFrame(
        [
            _scope_row("0000000001", 2024, "Alpha", "hash-c", True),
            _scope_row("0000000002", 2023, "Beta old", "hash-a", True),
            _scope_row("0000000002", 2025, "Beta", "hash-b", True),
            _scope_row("0000000003", 2025, "Gamma", "hash-d", True),
            _scope_row("0000000004", 2025, "Not equity", "hash-e", False),
        ]
    )
    catalog = pd.DataFrame(
        [
            _filing_row("0000000001", "accession-1", "10-Q", "2024-05-01T20:00:00Z"),
            _filing_row("0000000002", "accession-2", "10-K", "2023-03-01T21:00:00Z"),
            _filing_row("0000000002", "amendment-2", "10-K/A", "2023-04-01T21:00:00Z"),
            _filing_row("0000000003", "accession-3", "10-Q", "2025-08-01T20:00:00Z"),
            _filing_row("0000000003", "before-window", "10-Q", "2021-08-01T20:00:00Z"),
            _filing_row("0000000003", "holdout-3", "10-Q", "2026-08-28T20:00:00Z"),
        ]
    )
    legacy = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "accession-1",
                "source_study": "legacy-study",
            }
        ]
    )
    reference = pd.DataFrame(
        [
            _reference_row("target-1", "0000000001", True, True),
            _reference_row("target-2a", "0000000002", True, False),
            _reference_row("target-2b", "0000000002", False, False),
            _reference_row("target-3", "0000000003", False, False),
        ]
    )
    return scope, catalog, legacy, reference


def _scope_row(
    cik: str, calendar_year: int, company: str, sampling_hash: str, eligible: bool
) -> dict[str, object]:
    return {
        "cik": cik,
        "calendar_year": calendar_year,
        "company_name_snapshot": company,
        "sampling_hash": sampling_hash,
        "sample_size_bucket": "asset_q3",
        "stage1_inclusion_probability": 0.1,
        "stage1_sampling_weight": 10.0,
        "provisional_equity_eligible": eligible,
    }


def _filing_row(cik: str, accession: str, form: str, accepted_at: str) -> dict[str, object]:
    return {
        "cik": cik,
        "accession_number": accession,
        "form": form,
        "base_form": form.removesuffix("/A"),
        "is_amendment": form.endswith("/A"),
        "report_date": "2024-03-31",
        "accepted_at": accepted_at,
    }


def _reference_row(target_id: str, cik: str, complete: bool, candidate: bool) -> dict[str, object]:
    return {
        "target_id": target_id,
        "cik": cik,
        "provider_query_complete": complete,
        "event_date_reference_candidate": candidate,
    }
