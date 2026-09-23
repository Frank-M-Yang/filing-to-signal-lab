"""Offline adversarial tests for frozen samples, original facts and disclosure clocks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_sec_disclosure_audit import freeze_plan, run_audit, write_bundle
from sec_data.disclosure_audit import (
    VERSION,
    audit_financial_fields,
    audit_forecast_clocks,
    digest_file,
    earnings_candidates,
    inspect_earnings_document,
    load_cached_submissions,
    select_audit_sample,
    utc_timestamp,
    verify_manual_reviews,
)
from sec_data.guidance_evidence import parse_complete_submission


def _event() -> dict[str, Any]:
    return {
        "validation_event_id": "event-a",
        "cik": "0000000001",
        "ticker": "AAA",
        "form": "10-Q",
        "filing_accession": "0000000001-25-000002",
        "period_start": "2025-04-01",
        "period_end": "2025-06-30",
        "accepted_at": "2025-08-01T20:00:00Z",
    }


def _filings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "0000000001-25-000001",
                "form": "10-Q",
                "report_date": "2025-03-31",
                "accepted_at": "2025-05-01T20:00:00Z",
                "items": "",
            },
            {
                "cik": "0000000001",
                "accession_number": "0000000001-25-000002",
                "form": "10-Q",
                "report_date": "2025-06-30",
                "accepted_at": "2025-08-01T20:00:00Z",
                "items": "",
            },
            {
                "cik": "0000000001",
                "accession_number": "0000000001-25-000003",
                "form": "8-K",
                "report_date": "2025-07-25",
                "accepted_at": "2025-07-25T12:00:00Z",
                "items": "2.02,9.01",
            },
        ]
    )


def _fact(
    value: float,
    *,
    start: str | None = "2025-04-01",
    end: str = "2025-06-30",
    accn: str = "0000000001-25-000002",
    form: str = "10-Q",
) -> dict[str, Any]:
    result = {"val": value, "end": end, "accn": accn, "form": form}
    if start:
        result["start"] = start
    return result


def _document() -> dict[str, Any]:
    return {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(100)]}},
                "GrossProfit": {"units": {"USD": [_fact(40)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(-5)]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact(0)]}},
                "Assets": {"units": {"USD": [_fact(1000, start=None)]}},
            }
        },
    }


def _metric(document: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        r for r in audit_financial_fields(_event(), document, _filings()) if r["metric"] == name
    )


def test_selection_invariant_to_order_values_and_issuer_multiplicity() -> None:
    events = pd.DataFrame(
        [
            {
                **_event(),
                "validation_event_id": f"e-{issuer}-{q}",
                "cik": str(issuer).zfill(10),
                "future_return": issuer / 10,
                "actual": q,
            }
            for issuer in range(1, 5)
            for q in range(4)
        ]
    )
    result = select_audit_sample(events, sample_size=4, salt="test")
    assert result.cik.nunique() == 4
    assert "future_return" not in result and "actual" not in result
    changed = events.sample(frac=1, random_state=2).assign(future_return=-100, actual=1000)
    pd.testing.assert_frame_equal(result, select_audit_sample(changed, sample_size=4, salt="test"))
    events.loc[0, "form"] = "10-Q/A"
    assert "e-1-0" not in set(select_audit_sample(events, sample_size=99).validation_event_id)


def test_selection_rejects_duplicates_and_naive_clocks() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        select_audit_sample(pd.DataFrame([_event(), _event()]))
    with pytest.raises(ValueError, match="timezone"):
        utc_timestamp("2025-01-01")


def test_original_negative_zero_values_and_no_model_admission() -> None:
    rows = audit_financial_fields(_event(), _document(), _filings())
    assert all(r["quarter_or_instant_ready"] for r in rows)
    assert _metric(_document(), "net_income")["value"] == -5
    assert _metric(_document(), "operating_cash_flow")["value"] == 0
    assert all(not r["formal_feature_admitted"] and r["first_public_at"] is None for r in rows)


def test_later_restated_facts_do_not_fill_missing_original() -> None:
    document = _document()
    document["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = [
        _fact(123, accn="0000000001-25-000009", form="10-Q/A"),
        _fact(456, accn="0000000001-25-000010"),
    ]
    row = _metric(document, "net_income")
    assert row["status"] == "missing" and row["value"] is None


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("conflict", "ambiguous_value_or_tag"),
        ("nan", "invalid_value"),
        ("bool", "invalid_value"),
        ("currency", "unsupported_currency"),
    ],
)
def test_ambiguous_and_invalid_originals_are_not_averaged(mutation: str, status: str) -> None:
    document = _document()
    units = document["facts"]["us-gaap"]["NetIncomeLoss"]["units"]
    if mutation == "conflict":
        units["USD"].append(_fact(-6))
    elif mutation == "currency":
        units["EUR"] = units.pop("USD")
    else:
        units["USD"][0]["val"] = float("nan") if mutation == "nan" else True
    row = _metric(document, "net_income")
    assert row["status"] == status and not row["quarter_or_instant_ready"]


def test_cfo_ytd_is_not_quarter_and_cross_filing_difference_is_review_only() -> None:
    document = _document()
    rows = document["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]
    rows["USD"] = [
        _fact(70, start="2025-01-01"),
        _fact(20, start="2025-01-01", end="2025-03-31", accn="0000000001-25-000001"),
    ]
    result = _metric(document, "operating_cash_flow")
    assert result["value"] is None and result["ytd_value"] == 70
    assert result["derived_quarter_candidate"] == 50
    assert not result["quarter_or_instant_ready"]
    assert len(result["source_evidence"]) == 2
    assert result["status"] == "ytd_difference_requires_basis_review"
    rows["USD"][1]["form"] = "10-Q/A"
    assert _metric(document, "operating_cash_flow")["derived_quarter_candidate"] is None


def test_cfo_future_prior_acceptance_and_wrong_start_forbid_derivation() -> None:
    document = _document()
    rows = document["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]
    rows["USD"] = [
        _fact(70, start="2025-01-01"),
        _fact(20, start="2025-01-02", end="2025-03-31", accn="0000000001-25-000001"),
    ]
    assert _metric(document, "operating_cash_flow")["derived_quarter_candidate"] is None
    rows["USD"][1]["start"] = "2025-01-01"
    catalog = _filings()
    catalog.loc[0, "accepted_at"] = _event()["accepted_at"]
    result = audit_financial_fields(_event(), document, catalog)
    assert (
        next(r for r in result if r["metric"] == "operating_cash_flow")["derived_quarter_candidate"]
        is None
    )


@pytest.mark.parametrize("mutation", ["cik", "acceptance", "quarter", "report_date"])
def test_fact_identity_and_period_guards(mutation: str) -> None:
    event, document, catalog = _event(), _document(), _filings()
    if mutation == "cik":
        document["cik"] = 2
    elif mutation == "acceptance":
        event["accepted_at"] = "2025-08-01T21:00:00Z"
    elif mutation == "quarter":
        event["period_start"] = "2025-01-01"
    else:
        catalog.loc[1, "report_date"] = "2025-06-29"
    with pytest.raises(ValueError):
        audit_financial_fields(event, document, catalog)


def _submission() -> bytes:
    return b"""ACCESSION NUMBER: 0000000001-25-000003
<DOCUMENT><TYPE>EX-99.1
<FILENAME>release.htm
<TEXT><html><body><p>Quarter ended June 30, 2025. USD.</p>
<p>Revenue $ 100 Gross profit $ 40</p></body></html></TEXT></DOCUMENT>"""


def test_candidate_does_not_use_8k_report_date_as_financial_period() -> None:
    catalog = _filings()
    catalog = pd.concat(
        [
            catalog,
            pd.DataFrame(
                [
                    {**catalog.iloc[-1].to_dict(), "accepted_at": "2025-08-02T12:00:00Z"},
                    {
                        **catalog.iloc[-1].to_dict(),
                        "accepted_at": "2025-07-25T12:00:00Z",
                        "items": "12.02",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    candidates = earnings_candidates(_event(), catalog)
    assert len(candidates) == 1 and not candidates[0]["period_match_verified"]
    result = inspect_earnings_document(_submission(), "0000000001-25-000003")
    assert result["metric_snippets"]["revenue"]
    assert result["first_public_at"] is None and not result["metric_presence_is_actual_match"]
    with pytest.raises(ValueError, match="accession"):
        inspect_earnings_document(_submission(), "0000000001-25-000004")


def test_manual_review_is_acceptance_proxy_not_public_bound(tmp_path: Path) -> None:
    raw = tmp_path / "source.txt"
    raw.write_bytes(_submission())
    digest = digest_file(raw)
    exhibit = parse_complete_submission(_submission())[0]
    cache = {("0000000001", "0000000001-25-000003"): {"path": str(raw), "sha256": digest}}
    review = {
        "cik": "0000000001",
        "period_end": "2025-06-30",
        "release_accession": "0000000001-25-000003",
        "submission_sha256": digest,
        "exhibit_filename": "release.htm",
        "exhibit_sha256": exhibit["document_sha256"],
        "accounting_basis": "GAAP",
        "scale": 1,
        "period_anchor": "Quarter ended June 30, 2025.",
        "unit_anchor": "USD.",
        "cells": [{"metric": "revenue", "text_anchor": "Revenue $ 100", "display_value": "100"}],
    }
    fields = audit_financial_fields(_event(), _document(), _filings())
    candidates = earnings_candidates(_event(), _filings())
    result = verify_manual_reviews([review], [_event()], fields, candidates, cache)
    assert result[0]["sec_acceptance_proxy_at"] == "2025-07-25T12:00:00+00:00"
    assert result[0]["public_available_by"] is None
    assert not result[0]["header_clock_reconciled"]
    assert result[0]["first_public_at"] is None and not result[0]["trading_admission"]
    changed = copy.deepcopy(review)
    changed["cells"][0]["display_value"] = "10"
    with pytest.raises(ValueError, match="differ"):
        verify_manual_reviews([changed], [_event()], fields, candidates, cache)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        verify_manual_reviews([review], [_event()], fields, candidates, cache)


def test_cached_root_escape_and_distinct_snapshots_rejected(tmp_path: Path) -> None:
    root = tmp_path / "a"
    (root / "completions").mkdir(parents=True)
    record = {
        "status_code": 200,
        "raw_path": "../outside.txt",
        "cik": "1",
        "accession_number": "a",
        "sha256": "first",
    }
    path = root / "completions/one.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="escapes"):
        load_cached_submissions([root])
    record["raw_path"] = "raw/source.txt"
    path.write_text(json.dumps(record))
    (root / "completions/two.json").write_text(json.dumps({**record, "sha256": "second"}))
    with pytest.raises(ValueError, match="distinct"):
        load_cached_submissions([root])


def test_immutable_bundle_reuse_and_corruption(tmp_path: Path) -> None:
    first = write_bundle(tmp_path, "test", {"a.json": b"{}"})
    assert write_bundle(tmp_path, "test", {"a.json": b"{}"}) == first
    (first / "a.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        write_bundle(tmp_path, "test", {"a.json": b"{}"})


def test_forecast_clock_bound_requires_all_three_clocks_strictly_earlier() -> None:
    metadata = {
        "forecast_feature_cutoff": "2025-01-01T00:00:00Z",
        "forecast_training_end": "2025-07-25T12:00:00Z",
        "forecast_generated_at": "2025-07-26T00:00:00Z",
    }
    result = audit_forecast_clocks(metadata, "2025-07-25T12:00:00Z")
    assert result["status"] == "at_or_after_sec_acceptance_proxy"
    assert result["public_available_by"] is None
    assert result["failed_clocks"] == ["forecast_training_end", "forecast_generated_at"]
    for key in metadata:
        metadata[key] = "2025-01-01T00:00:00Z"
    result = audit_forecast_clocks(metadata, "2025-07-25T12:00:00Z")
    assert not result["first_public_forecast_certified"] and result["failed_clocks"] == []


def _fixture(tmp_path: Path) -> Path:
    import zipfile

    config: dict[str, Any] = {
        "version": VERSION,
        "sample_size": 1,
        "selection_salt": "synthetic",
        "sample_replacement_allowed": False,
        "return_values_allowed": False,
        "model_fitting_allowed": False,
        "cached_submission_roots": [],
    }
    tables = {
        "market": {"validation_market_state_features.parquet": pd.DataFrame([_event()])},
        "feature": {
            "target_filing_features.parquet": pd.DataFrame([_event()]),
            "quarterly_feature_history.parquet": pd.DataFrame(
                [
                    {
                        **_event(),
                        "company_id": "0000000001",
                        "forecast_feature_cutoff": "2025-05-01T20:00:00Z",
                        "forecast_training_end": "2025-07-31T20:00:00Z",
                        "forecast_generated_at": "2025-08-01T19:59:59Z",
                    }
                ]
            ),
        },
        "catalog": {"filing_catalog.parquet": _filings()},
    }
    for name, frames in tables.items():
        root = tmp_path / name
        root.mkdir()
        artifacts = {}
        for filename, frame in frames.items():
            frame.to_parquet(root / filename)
            artifacts[filename] = {"sha256": digest_file(root / filename)}
        manifest = root / "output_manifest.json"
        manifest.write_text(json.dumps({"artifacts": artifacts}))
        config[f"{name}_root"] = str(root)
        config[f"{name}_manifest_sha256"] = digest_file(manifest)
    source = tmp_path / "companyfacts.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("CIK0000000001.json", json.dumps(_document()))
    config.update(companyfacts_zip=str(source), companyfacts_sha256=digest_file(source))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_end_to_end_offline_no_returns_and_frozen_before_actuals(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    root, plan, catalog = freeze_plan(config, tmp_path / "output")
    assert (root / "output_manifest.json").exists()
    assert not plan["return_values_loaded"]
    assert "val" not in plan["events"][0]
    output, report = run_audit(root, plan, catalog, tmp_path / "output")
    assert report["events"] == 1 and report["direct_core_financial_complete_events"] == 1
    assert report["cached_candidate_documents"] == 0
    assert not report["model_fitting_authorized"] and not report["return_values_loaded"]
    assert (output / "clock_audit.json").exists()
    changed = copy.deepcopy(plan)
    changed["events"][0]["ticker"] = "BBB"
    with pytest.raises(ValueError, match="persisted plan"):
        run_audit(root, changed, catalog, tmp_path / "output")
