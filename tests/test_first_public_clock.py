"""Rule-based earliest SEC disclosure clock tests (synthetic catalog, no data)."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sec_data.first_public_clock import (
    FirstPublicClockConfig,
    FirstPublicClockError,
    build_first_public_clock,
    build_first_public_clock_audit,
    summarize_first_public_clock,
)


def _row(
    cik: str,
    form: str,
    report_date: str,
    accepted_at: str,
    items: str,
    accession: str,
) -> dict[str, object]:
    return {
        "cik": cik,
        "form": form,
        "base_form": form.removesuffix("/A"),
        "is_amendment": form.endswith("/A"),
        "report_date": report_date,
        "accepted_at": accepted_at,
        "items": items,
        "accession_number": accession,
    }


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # A: releases precede both reports; the Q4 release must not leak into Q1.
            _row("A", "10-K", "2024-12-31", "2025-02-20T21:00:00+00:00", "", "A-10K"),
            _row("A", "8-K", "2025-02-05", "2025-02-05T21:10:00+00:00", "2.02,9.01", "A-8K-Q4"),
            _row("A", "8-K", "2025-04-10", "2025-04-10T12:00:00+00:00", "5.02", "A-8K-HR"),
            _row("A", "8-K", "2025-04-24", "2025-04-24T20:05:00+00:00", "2.02,9.01", "A-8K-Q1"),
            _row("A", "10-Q", "2025-03-31", "2025-05-01T20:30:00+00:00", "", "A-10Q1"),
            # B: no earnings release; the 10-Q is the first SEC disclosure.
            _row("B", "10-Q", "2025-03-31", "2025-05-10T14:00:00+00:00", "", "B-10Q1"),
            # C: preliminary and final releases plus an ignored amendment.
            _row("C", "8-K", "2025-04-08", "2025-04-08T11:00:00+00:00", "2.02,7.01,9.01", "C-P"),
            _row("C", "8-K", "2025-04-29", "2025-04-29T20:00:00+00:00", "2.02,9.01", "C-F"),
            _row("C", "8-K/A", "2025-04-29", "2025-04-30T12:00:00+00:00", "2.02", "C-FA"),
            _row("C", "10-Q", "2025-03-31", "2025-04-29T21:00:00+00:00", "", "C-10Q1"),
            # D: New York date differs from the UTC date; 8-K furnished after its event.
            _row("D", "8-K", "2025-04-28", "2025-05-01T23:45:00+00:00", "2.02,5.02", "D-8K"),
            _row("D", "10-Q", "2025-03-31", "2025-05-02T00:30:00+00:00", "", "D-10Q1"),
            _row("D", "8-K", "2025-05-03", "2025-05-03T13:00:00+00:00", "2.02", "D-LATE"),
            # E: an amended periodic report is never a clock subject.
            _row("E", "10-Q/A", "2025-03-31", "2025-05-20T14:00:00+00:00", "", "E-10QA"),
        ]
    )


def _by_accession() -> pd.DataFrame:
    return build_first_public_clock(_catalog()).set_index("accession_number")


def test_only_original_periodic_reports_are_subjects() -> None:
    clocks = _by_accession()
    assert sorted(clocks.index) == ["A-10K", "A-10Q1", "B-10Q1", "C-10Q1", "D-10Q1"]
    assert not clocks["first_public_certified"].any()


def test_release_matches_its_own_fiscal_period() -> None:
    clocks = _by_accession()
    assert clocks.loc["A-10K", "release_accession_number"] == "A-8K-Q4"
    assert clocks.loc["A-10Q1", "release_accession_number"] == "A-8K-Q1"
    assert clocks.loc["A-10Q1", "release_candidate_count"] == 1
    assert clocks.loc["A-10Q1", "clock_source"] == "item_2_02_8k"
    expected_lead = (
        pd.Timestamp("2025-05-01T20:30:00Z") - pd.Timestamp("2025-04-24T20:05:00Z")
    ).total_seconds() / 86_400
    assert clocks.loc["A-10Q1", "disclosure_lead_days"] == pytest.approx(expected_lead)
    assert clocks.loc["A-10Q1", "earliest_disclosure_at"] == pd.Timestamp("2025-04-24T20:05:00Z")
    assert bool(clocks.loc["A-10Q1", "same_local_date"]) is False


def test_report_without_release_keeps_its_own_clock() -> None:
    clocks = _by_accession()
    assert clocks.loc["B-10Q1", "clock_source"] == "periodic_report"
    assert clocks.loc["B-10Q1", "disclosure_lead_days"] == 0.0
    assert (
        clocks.loc["B-10Q1", "earliest_disclosure_at"] == clocks.loc["B-10Q1", "report_accepted_at"]
    )
    assert pd.isna(clocks.loc["B-10Q1", "same_local_date"])
    assert clocks.loc["B-10Q1", "release_candidate_count"] == 0


def test_earliest_of_several_candidates_and_amendments_ignored() -> None:
    clocks = _by_accession()
    assert clocks.loc["C-10Q1", "release_accession_number"] == "C-P"
    assert clocks.loc["C-10Q1", "release_candidate_count"] == 2
    assert bool(clocks.loc["C-10Q1", "release_event_date_reliable"]) is True


def test_new_york_date_and_late_furnishing() -> None:
    clocks = _by_accession()
    # 23:45 UTC and 00:30 UTC fall on the same New York date (May 1).
    assert clocks.loc["D-10Q1", "release_accession_number"] == "D-8K"
    assert bool(clocks.loc["D-10Q1", "same_local_date"]) is True
    assert bool(clocks.loc["D-10Q1", "release_filed_after_event_date"]) is True
    # Item 5.02 may describe a separate earlier event, so the event date is not trusted.
    assert bool(clocks.loc["D-10Q1", "release_event_date_reliable"]) is False


def test_co_registrants_share_an_accession_but_not_a_release() -> None:
    # A parent and a subsidiary file one combined 10-Q; only the parent furnishes
    # an earnings 8-K, so only the parent's clock may move earlier.
    catalog = pd.DataFrame(
        [
            _row("P", "8-K", "2025-04-24", "2025-04-24T20:05:00+00:00", "2.02,9.01", "P-8K"),
            _row("P", "10-Q", "2025-03-31", "2025-05-01T20:30:00+00:00", "", "PS-10Q"),
            _row("S", "10-Q", "2025-03-31", "2025-05-01T20:30:00+00:00", "", "PS-10Q"),
        ]
    )
    clocks = build_first_public_clock(catalog).set_index("cik")
    assert clocks.loc["P", "accession_number"] == "PS-10Q"
    assert clocks.loc["S", "accession_number"] == "PS-10Q"
    assert clocks.loc["P", "clock_source"] == "item_2_02_8k"
    assert clocks.loc["S", "clock_source"] == "periodic_report"


def test_missing_period_end_keeps_the_later_report_clock() -> None:
    catalog = pd.DataFrame(
        [
            _row("M", "8-K", "2025-04-24", "2025-04-24T20:05:00+00:00", "2.02,9.01", "M-8K"),
            _row("M", "10-Q", "", "2025-05-01T20:30:00+00:00", "", "M-10Q"),
        ]
    )
    clocks = build_first_public_clock(catalog)
    assert clocks.loc[0, "clock_source"] == "periodic_report_missing_period_end"
    assert clocks.loc[0, "release_candidate_count"] == 0
    assert clocks.loc[0, "disclosure_lead_days"] == 0.0
    assert summarize_first_public_clock(clocks)["all"]["missing_period_end"] == 1


def test_summary_by_form() -> None:
    summary = summarize_first_public_clock(build_first_public_clock(_catalog()))
    assert summary["all"]["periodic_reports"] == 5
    assert summary["all"]["with_item_2_02_release"] == 4
    assert summary["by_form"]["10-Q"]["periodic_reports"] == 4
    assert summary["by_form"]["10-Q"]["share_same_local_date"] == pytest.approx(1 / 3)
    assert summary["by_form"]["10-K"]["share_lead_at_least_3_days"] == pytest.approx(1.0)


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(FirstPublicClockError):
        FirstPublicClockConfig(max_release_lead_days=0).validate()
    with pytest.raises(FirstPublicClockError):
        build_first_public_clock(_catalog().drop(columns=["items"]))
    naive = _catalog()
    naive.loc[0, "accepted_at"] = "2025-02-20T21:00:00"
    with pytest.raises(FirstPublicClockError):
        build_first_public_clock(naive)
    duplicate = pd.concat([_catalog(), _catalog().iloc[[0]]], ignore_index=True)
    with pytest.raises(FirstPublicClockError):
        build_first_public_clock(duplicate)


def test_audit_bundle_is_content_addressed_and_reused(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    table = catalog_root / "filing_catalog.parquet"
    _catalog().to_parquet(table, index=False)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    (catalog_root / "output_manifest.json").write_text(
        json.dumps({"artifacts": {"filing_catalog.parquet": {"sha256": digest}}}),
        encoding="utf-8",
    )
    first = build_first_public_clock_audit(catalog_root, output_root=tmp_path / "out")
    second = build_first_public_clock_audit(catalog_root, output_root=tmp_path / "out")
    assert first.output_root == second.output_root
    assert first.report["summary"]["all"]["periodic_reports"] == 5
    assert first.report["returns_computed"] is False
    stored = pd.read_parquet(first.output_root / "first_public_clock.parquet")
    assert len(stored) == 5
