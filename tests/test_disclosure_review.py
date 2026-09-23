"""Offline checks for clock conflicts, honest public-time claims and release cells."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_sec_disclosure_audit import write_bundle
from scripts.review_sec_disclosures import run_followup
from sec_data.disclosure_audit import digest_file
from sec_data.disclosure_review import (
    VERSION,
    audit_candidate_headers,
    audit_submission_clock,
    earliest_metric_proxies,
    forecast_proxy_check,
    review_document_kinds,
    review_metric_cells,
)
from sec_data.guidance_evidence import parse_complete_submission


def _candidate() -> dict[str, Any]:
    return {
        "validation_event_id": "event",
        "cik": "0000000001",
        "form": "8-K",
        "accession_number": "0000000001-25-000001",
        "accepted_at": "2025-07-25T16:00:00Z",
        "url": "https://www.sec.gov/Archives/synthetic.txt",
    }


def _raw(literal: str = "20250725120000") -> bytes:
    return f"""<SEC-HEADER>
<ACCEPTANCE-DATETIME>{literal}
ACCESSION NUMBER: 0000000001-25-000001
CONFORMED SUBMISSION TYPE: 8-K
CENTRAL INDEX KEY: 0000000001
</SEC-HEADER>
<DOCUMENT><TYPE>EX-99.1
<FILENAME>release.htm
<TEXT><p>Quarter ended June 30, 2025 (USD millions)</p>
<p>GAAP earnings release. Revenue 100 90. Operating cash flow 14,489 10,000.</p>
</TEXT></DOCUMENT>""".encode()


@pytest.mark.parametrize(
    "literal,catalog",
    [
        ("20250725120000", "2025-07-25T16:00:00Z"),
        ("20250125120000", "2025-01-25T17:00:00Z"),
    ],
)
def test_header_dst_aware_proxy_is_not_public_time(literal: str, catalog: str) -> None:
    result = audit_submission_clock(_raw(literal), {**_candidate(), "accepted_at": catalog})
    assert result["status"] == "consistent_acceptance_proxy"
    assert result["public_available_by"] is None and result["first_public_at"] is None
    assert not result["trading_admission"]


@pytest.mark.parametrize(
    "literal,status",
    [
        ("20250725160000", "clock_conflict_quarantined"),
        ("20251102013000", "ambiguous_or_nonexistent_header_clock"),
        ("20250309023000", "ambiguous_or_nonexistent_header_clock"),
        ("20250230000000", "missing_or_invalid_header_clock"),
        ("20250725", "missing_or_invalid_header_clock"),
    ],
)
def test_clock_conflicts_and_ambiguous_times_never_admitted(literal: str, status: str) -> None:
    result = audit_submission_clock(_raw(literal), _candidate())
    assert result["status"] == status and result["sec_acceptance_proxy_at"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("cik", "0000000002"),
        ("accession_number", "0000000001-25-000002"),
        ("form", "10-Q"),
    ],
)
def test_header_identity_mismatch_rejected(key: str, value: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        audit_submission_clock(_raw(), {**_candidate(), key: value})


def _review_fixture(tmp_path: Path) -> tuple[Any, ...]:
    content = _raw()
    raw = tmp_path / "raw.txt"
    raw.write_bytes(content)
    candidate = _candidate()
    event = {
        "validation_event_id": "event",
        "cik": candidate["cik"],
        "ticker": "AAA",
        "period_start": "2025-04-01",
        "period_end": "2025-06-30",
        "filing_accession": "0000000001-25-000002",
        "accepted_at": "2025-08-01T20:00:00Z",
    }
    cache = {
        (candidate["cik"], candidate["accession_number"]): {
            "path": str(raw),
            "sha256": digest_file(raw),
            "requested_url": candidate["url"],
        }
    }
    spec = {
        "cik": event["cik"],
        "period_end": event["period_end"],
        "release_accession": candidate["accession_number"],
        "submission_sha256": digest_file(raw),
        "exhibit_filename": "release.htm",
        "exhibit_sha256": parse_complete_submission(content)[0]["document_sha256"],
        "period_anchor": "Quarter ended June 30, 2025",
        "unit_anchor": "(USD millions)",
        "accounting_basis": "GAAP",
        "scale": 1000000,
        "cells": [{"metric": "revenue", "display_value": "100", "text_anchor": "Revenue 100 90"}],
        "kind": "earnings_release",
        "anchors": ["GAAP earnings release"],
        "notes": "Synthetic",
    }
    fields = [
        {
            "validation_event_id": "event",
            "metric": "revenue",
            "value": 100000000,
            "quarter_or_instant_ready": True,
            "status": "original_quarter",
        }
    ]
    return spec, event, fields, candidate, cache


def test_matched_values_and_conflicted_clocks_are_separate(tmp_path: Path) -> None:
    spec, event, fields, candidate, cache = _review_fixture(tmp_path)
    result = review_metric_cells([spec], [event], fields, [candidate], cache)
    assert result[0]["comparison_status"] == "exact_original_match"
    assert not result[0]["formal_feature_admitted"]
    assert earliest_metric_proxies(result) == {"revenue": "2025-07-25T16:00:00+00:00"}
    candidate["accepted_at"] = "2025-07-25T12:00:00Z"
    result = review_metric_cells([spec], [event], fields, [candidate], cache)
    assert result[0]["comparison_status"] == "exact_original_match"
    assert result[0]["clock_status"] == "clock_conflict_quarantined"
    assert earliest_metric_proxies(result) == {}


def test_ytd_difference_not_silently_replaced_by_direct_release(tmp_path: Path) -> None:
    spec, event, fields, candidate, cache = _review_fixture(tmp_path)
    spec["cells"] = [
        {
            "metric": "operating_cash_flow",
            "display_value": "14,489",
            "text_anchor": "Operating cash flow 14,489 10,000",
        }
    ]
    fields[0].update(
        metric="operating_cash_flow",
        value=None,
        quarter_or_instant_ready=False,
        status="ytd_difference_requires_basis_review",
        derived_quarter_candidate=14488000000,
    )
    result = review_metric_cells([spec], [event], fields, [candidate], cache)
    assert result[0]["release_minus_ytd_candidate"] == 1000000
    assert result[0]["comparison_status"] == "release_quarter_vs_cross_filing_ytd_candidate"
    assert result[0]["original_value"] is None and not result[0]["cashflow_basis_compatible"]


@pytest.mark.parametrize("mutation", ["scale", "basis", "anchor", "exhibit", "period", "substring"])
def test_review_anchor_and_semantics_guards(tmp_path: Path, mutation: str) -> None:
    spec, event, fields, candidate, cache = _review_fixture(tmp_path)
    if mutation == "scale":
        spec["scale"] = 1000
    elif mutation == "basis":
        spec["accounting_basis"] = "non-GAAP"
    elif mutation == "anchor":
        spec["cells"][0]["text_anchor"] = "invented"
    elif mutation == "exhibit":
        spec["exhibit_sha256"] = "0" * 64
    elif mutation == "period":
        spec["period_end"] = "2025-03-31"
    else:
        spec["cells"][0]["display_value"] = "10"
    with pytest.raises(ValueError):
        review_metric_cells([spec], [event], fields, [candidate], cache)


def test_duplicates_and_corrupt_raw_cannot_supply_cells(tmp_path: Path) -> None:
    spec, event, fields, candidate, cache = _review_fixture(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        review_metric_cells([spec, copy.deepcopy(spec)], [event], fields, [candidate], cache)
    Path(next(iter(cache.values()))["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        review_metric_cells([spec], [event], fields, [candidate], cache)


def test_disposition_and_cache_url_are_evidence_bound(tmp_path: Path) -> None:
    spec, event, _, candidate, cache = _review_fixture(tmp_path)
    rows = review_document_kinds([spec], [event], [candidate], cache)
    assert rows[0]["kind"] == "earnings_release" and rows[0]["first_public_at"] is None
    spec["anchors"] = ["missing"]
    with pytest.raises(ValueError, match="anchor"):
        review_document_kinds([spec], [event], [candidate], cache)
    candidate["url"] = "https://www.sec.gov/different.txt"
    with pytest.raises(ValueError, match="URL"):
        audit_candidate_headers([candidate], cache)


def test_earliest_per_metric_then_latest_needed_input_and_forecast_risk() -> None:
    rows = [
        {"metric": m, "sec_acceptance_proxy_at": t, "comparison_status": "exact_original_match"}
        for m, t in [
            ("revenue", "2025-07-25T16:00:00Z"),
            ("revenue", "2025-07-27T16:00:00Z"),
            ("gross_profit", "2025-07-26T16:00:00Z"),
        ]
    ]
    proxies = earliest_metric_proxies(rows)
    assert proxies == earliest_metric_proxies(rows[::-1])
    assert proxies["gross_margin"] == "2025-07-26T16:00:00+00:00"
    meta = {
        key: "2025-07-26T16:00:00Z"
        for key in ["forecast_feature_cutoff", "forecast_training_end", "forecast_generated_at"]
    }
    result = forecast_proxy_check(meta, proxies["gross_margin"])
    assert result["status"] == "at_or_after_acceptance_proxy"
    assert not result["first_public_forecast_certified"]
    assert forecast_proxy_check({}, proxies["gross_margin"])["status"] == "missing_forecast_clock"


def test_end_to_end_frozen_sources_offline_and_immutable(tmp_path: Path) -> None:
    spec, event, fields, candidate, cache = _review_fixture(tmp_path)
    raw_root = tmp_path / "cache"
    (raw_root / "completions").mkdir(parents=True)
    (raw_root / "raw.txt").write_bytes(_raw())
    completion = {
        **next(iter(cache.values())),
        "cik": candidate["cik"],
        "accession_number": candidate["accession_number"],
        "status_code": 200,
        "raw_path": "raw.txt",
    }
    (raw_root / "completions/one.json").write_text(json.dumps(completion))
    feature = tmp_path / "features"
    feature.mkdir()
    filename = "quarterly_feature_history.parquet"
    pd.DataFrame(
        [
            {
                "company_id": event["cik"],
                "filing_accession": event["filing_accession"],
                "forecast_feature_cutoff": "2025-01-01T00:00:00Z",
                "forecast_training_end": "2025-07-01T00:00:00Z",
                "forecast_generated_at": "2025-07-01T00:00:00Z",
                "future_return": 999,
            }
        ]
    ).to_parquet(feature / filename)
    (feature / "output_manifest.json").write_text(
        json.dumps({"artifacts": {filename: {"sha256": digest_file(feature / filename)}}})
    )
    plan = {
        "events": [event],
        "earnings_candidates": [candidate],
        "config": {
            "cached_submission_roots": [str(raw_root)],
            "feature_root": str(feature),
            "feature_manifest_sha256": digest_file(feature / "output_manifest.json"),
        },
    }
    plan_root = write_bundle(tmp_path, "plan", {"plan.json": json.dumps(plan).encode()})
    plan_sha = digest_file(plan_root / "output_manifest.json")
    audit = write_bundle(
        tmp_path,
        "audit",
        {
            "field_audit.json": json.dumps(fields).encode(),
            "report.json": json.dumps({"plan_manifest_sha256": plan_sha}).encode(),
        },
    )
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"version": "sec-disclosure-manual-review-v1", "reviews": []}))
    config = {
        "version": VERSION,
        "selection": "synthetic",
        "plan_root": str(plan_root),
        "plan_manifest_sha256": plan_sha,
        "audit_root": str(audit),
        "audit_manifest_sha256": digest_file(audit / "output_manifest.json"),
        "prior_review_path": str(prior),
        "prior_review_sha256": digest_file(prior),
        "documents": [spec],
        "metric_reviews": [
            {"release_accession": spec["release_accession"], "cells": spec["cells"]}
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    root, report = run_followup(config_path, tmp_path / "output")
    assert report["exact_value_match_events"] == 1 and report["header_checks"] == 1
    assert not report["model_fitting_authorized"] and not report["return_values_loaded"]
    assert run_followup(config_path, tmp_path / "output")[0] == root
    assert "999" not in (root / "event_review.json").read_text()
    config["documents"][0]["kind"] = "preliminary_results"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only a reviewed earnings release"):
        run_followup(config_path, tmp_path / "output")
