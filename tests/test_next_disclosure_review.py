"""Synthetic source timelines test admission guards, not returns or empirical Alpha."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.review_next_disclosures import prepare_sources, review_sources
from sec_data.next_disclosure_review import freeze_sources, pair_dispositions


def source_fixture() -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Twenty distinct issuers; issuer zero intentionally lacks a frozen release."""
    queue, history, catalog = [], [], []
    for i in range(20):
        cik = str(i + 1).zfill(10)
        accession = cik + "-24-000003"
        release = cik + "-24-000002"
        queue.append(
            {
                "validation_event_id": f"event-{i}",
                "cik": cik,
                "assigned_ticker": f"T{i}",
                "assigned_exchange_mic": "XNYS",
                "target_accession_number": accession,
                "release_accession_number": release if i else None,
                "earliest_disclosure_at": "2024-10-11T13:00:00Z",
            }
        )
        history.append(
            {
                "company_id": cik,
                "filing_accession": accession,
                "period_start": "2024-07-01",
                "period_end": "2024-09-30",
                "filing_form": "10-Q",
                "derivation_method": "direct_gross_profit",
            }
        )
        catalog.extend(
            [
                {
                    "cik": cik,
                    "accession_number": accession,
                    "form": "10-Q",
                    "report_date": "2024-09-30",
                    "accepted_at": "2024-11-05T21:00:00Z",
                    "items": "",
                },
                {
                    "cik": cik,
                    "accession_number": release,
                    "form": "8-K",
                    "report_date": "2024-10-11",
                    "accepted_at": "2024-10-11T13:00:00Z",
                    "items": "2.02,9.01",
                },
            ]
        )
    return queue, pd.DataFrame(history), pd.DataFrame(catalog)


def test_freeze_retains_all_issuers_and_labels_periodic_fallback() -> None:
    queue, history, catalog = source_fixture()
    events, candidates, targets = freeze_sources(queue, history, catalog)
    assert len(events) == 20 and len(targets) == 21
    fallback = [r for r in candidates if r["source_role"].startswith("periodic_report")]
    assert len(fallback) == 1 and fallback[0]["validation_event_id"] == "event-0"
    assert events[0]["release_accession_number"] is None
    assert len(set(r["guidance_document_id"] for r in targets)) == 21
    assert all(
        r["complete_submission_url"].startswith("https://www.sec.gov/Archives/") for r in targets
    )


def test_discovery_includes_all_prior_candidates_not_future_or_other_periods() -> None:
    queue, history, catalog = source_fixture()
    original = catalog.iloc[1].to_dict()
    extra = [
        {
            **original,
            "accession_number": "0000000001-24-000004",
            "accepted_at": "2024-10-22T13:00:00Z",
        },
        {
            **original,
            "accession_number": "0000000001-24-000005",
            "accepted_at": "2024-11-06T13:00:00Z",
        },
        {
            **original,
            "accession_number": "0000000001-24-000006",
            "accepted_at": "2024-09-01T13:00:00Z",
        },
    ]
    _, candidates, _ = freeze_sources(queue, history, pd.concat([catalog, pd.DataFrame(extra)]))
    found = {r["accession_number"] for r in candidates}
    assert "0000000001-24-000004" in found
    assert "0000000001-24-000005" not in found and "0000000001-24-000006" not in found


def test_selection_does_not_read_return_or_financial_value_columns() -> None:
    queue, history, catalog = source_fixture()
    original = freeze_sources(queue, history, catalog)
    for row in queue:
        row["future_return"] = 1e10
        row["is_listed"] = False
    history["gm_level"] = float("nan")
    history["gm_latest_restated"] = 1e20
    assert original == freeze_sources(queue, history, catalog)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "period", "form", "derivation", "release"]
)
def test_frozen_metadata_cannot_be_replaced_or_silently_repaired(damage: str) -> None:
    queue, history, catalog = source_fixture()
    if damage == "missing":
        history = history.iloc[1:]
    elif damage == "duplicate":
        queue[-1] = queue[0]
    elif damage == "period":
        catalog.loc[0, "report_date"] = "2024-06-30"
    elif damage == "form":
        history.loc[0, "filing_form"] = "10-K"
    elif damage == "derivation":
        history.loc[0, "derivation_method"] = "latest_imputation"
    else:
        queue[1]["release_accession_number"] = "0000000002-24-000099"
    with pytest.raises(ValueError):
        freeze_sources(queue, history, catalog)


def pair_fixture() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]
]:
    """One exact late pair among twenty cases; all unverified cases remain explicit."""
    queue, history, catalog = source_fixture()
    events, _, _ = freeze_sources(queue, history, catalog)
    cells = [
        {
            "validation_event_id": "event-0",
            "cik": events[0]["cik"],
            "release_accession": "0000000001-24-000002",
            "metric": metric,
            "value": value,
            "comparison_status": "exact_original_match",
            "clock_status": "consistent_acceptance_proxy",
            "sec_acceptance_proxy_at": "2024-10-22T13:00:00Z",
            "source_sha256": "1" * 64,
            "exhibit_sha256": "2" * 64,
        }
        for metric, value in [("revenue", 100.0), ("gross_profit", 30.0)]
    ]
    docs = [
        {
            "validation_event_id": "event-0",
            "release_accession": cells[0]["release_accession"],
            "kind": "earnings_release",
            "submission_sha256": "1" * 64,
            "exhibit_sha256": "2" * 64,
            "clock": {
                "status": "consistent_acceptance_proxy",
                "sec_acceptance_proxy_at": cells[0]["sec_acceptance_proxy_at"],
            },
        }
    ]
    reasons = {e["ticker"]: "source_review_pending" for e in events}
    return events, cells, docs, reasons


def test_later_verified_pair_never_backfills_the_earliest_information_set() -> None:
    events, cells, docs, reasons = pair_fixture()
    rows = pair_dispositions(events, cells, docs, reasons)
    assert len(rows) == 20 and sum(r["conditional_pair_verified"] for r in rows) == 1
    assert rows[0]["earliest_disclosure_at"] == "2024-10-11T13:00:00Z"
    assert rows[0]["pair_acceptance_proxy_at"] == "2024-10-22T13:00:00Z"
    assert rows[0]["reviewed_pairs"][0]["later_than_frozen_earliest_proxy"]
    assert rows[0]["reviewed_pairs"][0]["gross_margin"] == pytest.approx(0.3)
    assert not any(
        r["first_public_certified"]
        or r["model_admission"]
        or r["return_admission"]
        or r["trading_admission"]
        for r in rows
    )


def test_cost_scope_quarantine_is_not_cured_by_equal_numbers() -> None:
    events, cells, docs, reasons = pair_fixture()
    events[0]["existing_scope_quarantine"] = True
    rows = pair_dispositions(events, cells, docs, reasons)
    assert not rows[0]["conditional_pair_verified"] and rows[0]["pair_acceptance_proxy_at"] is None
    assert rows[0]["reviewed_pairs"][0]["exact_original_pair"]


def test_exact_values_do_not_override_a_header_catalog_conflict() -> None:
    events, cells, docs, reasons = pair_fixture()
    for r in cells:
        r.update(clock_status="clock_conflict_quarantined", sec_acceptance_proxy_at=None)
    docs[0]["clock"] = {"status": "clock_conflict_quarantined", "sec_acceptance_proxy_at": None}
    rows = pair_dispositions(events, cells, docs, reasons)
    assert not rows[0]["conditional_pair_verified"]
    assert rows[0]["reviewed_pairs"][0]["exact_original_pair"]
    assert rows[0]["reviewed_pairs"][0]["pair_acceptance_proxy_at"] is None


def test_rounded_or_unresolved_original_is_not_an_exact_pair() -> None:
    events, cells, docs, reasons = pair_fixture()
    cells[1]["comparison_status"] = "original_value_disagrees"
    assert not pair_dispositions(events, cells, docs, reasons)[0]["conditional_pair_verified"]
    cells[1]["comparison_status"] = "original_field_unresolved"
    assert not pair_dispositions(events, cells, docs, reasons)[0]["conditional_pair_verified"]


@pytest.mark.parametrize(
    "damage",
    [
        "mixed_hash",
        "mixed_clock",
        "cik",
        "preliminary",
        "missing_gp",
        "nonfinite",
        "zero_revenue",
        "outside",
        "no_reason",
    ],
)
def test_corrupted_or_incomplete_review_fails_closed(damage: str) -> None:
    events, cells, docs, reasons = pair_fixture()
    if damage == "mixed_hash":
        cells[1]["source_sha256"] = "3" * 64
    elif damage == "mixed_clock":
        cells[1]["sec_acceptance_proxy_at"] = "2024-10-22T14:00:00Z"
    elif damage == "cik":
        cells[1]["cik"] = events[1]["cik"]
    elif damage == "preliminary":
        docs[0]["kind"] = "preliminary_results"
    elif damage == "missing_gp":
        cells = cells[:1]
    elif damage == "nonfinite":
        cells[1]["value"] = float("inf")
    elif damage == "zero_revenue":
        cells[0]["value"] = 0.0
    elif damage == "outside":
        cells[0]["validation_event_id"] = "not-frozen"
    else:
        reasons.pop("T1")
    with pytest.raises(ValueError):
        pair_dispositions(events, cells, docs, reasons)


def test_pair_comparison_is_independent_of_return_outcomes() -> None:
    events, cells, docs, reasons = pair_fixture()
    original = pair_dispositions(events, cells, docs, reasons)
    other_cells = copy.deepcopy(cells)
    for r in other_cells:
        r["future_return"] = 1e10
    assert original == pair_dispositions(events, other_cells, docs, reasons)


@pytest.mark.parametrize(
    "flag",
    [
        "model_fitting_allowed",
        "return_values_allowed",
        "trading_admission",
        "overwrite_originals",
        "sample_replacement_allowed",
    ],
)
def test_prepare_contract_cannot_enable_trading_or_fitting(tmp_path: Path, flag: str) -> None:
    c = {
        "version": "next-disclosure-review-v1",
        "model_fitting_allowed": False,
        "return_values_allowed": False,
        "trading_admission": False,
        "overwrite_originals": False,
        "sample_replacement_allowed": False,
        flag: True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        prepare_sources(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_review_contract_cannot_certify_first_public_or_use_rounding_plugs(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    c = {
        "version": "next-disclosure-review-v1",
        "first_public_certified": False,
        "model_fitting_allowed": False,
        "return_values_allowed": False,
        "sample_replacement_allowed": False,
        "rounding_plugs_allowed": False,
    }
    for flag in ("first_public_certified", "rounding_plugs_allowed"):
        path.write_text(json.dumps({**c, flag: True}))
        with pytest.raises(ValueError, match="contract"):
            review_sources(tmp_path, "not-read", path, tmp_path / "outputs")
