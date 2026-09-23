"""Offline adversarial tests for provenance, selective overlays and candidate edges."""

from __future__ import annotations

import copy
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.audit_sec_clock_provenance import (
    collect_clock_evidence,
    freeze_clock_plan,
    run_provenance,
)
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.clock_provenance import (
    VERSION,
    audit_event_boundary,
    boundary_scan,
    index_url,
    parse_index_acceptance,
    reconcile_acceptance,
    trace_source_row,
)
from sec_data.disclosure_audit import digest_file, submission_url
from sec_data.models import SecResponse


def _filing(
    *,
    cik: str = "0000000001",
    number: str = "000001",
    form: str = "8-K",
    accepted: str = "2025-05-01T12:00:00Z",
) -> dict[str, Any]:
    return {
        "cik": cik,
        "accession_number": f"{cik}-25-{number}",
        "form": form,
        "accepted_at": accepted,
        "filing_date": accepted[:10],
        "report_date": "2025-03-31",
        "items": "2.02,9.01" if form == "8-K" else "",
        "source_member": f"CIK{cik}.json",
        "source_row_index": 0,
        "source_zip_sha256": "a" * 64,
    }


def _member(rows: list[dict[str, Any]], literal: str | None = None) -> bytes:
    fields = {
        "accessionNumber": "accession_number",
        "form": "form",
        "filingDate": "filing_date",
        "reportDate": "report_date",
        "acceptanceDateTime": "accepted_at",
    }
    recent = {key: [r[value] for r in rows] for key, value in fields.items()}
    if literal is not None:
        recent["acceptanceDateTime"][0] = literal
    return json.dumps({"cik": rows[0]["cik"], "filings": {"recent": recent}}).encode()


def _submission(row: dict[str, Any], literal: str = "20250501120000") -> bytes:
    return f"""<SEC-HEADER>
<ACCEPTANCE-DATETIME>{literal}
ACCESSION NUMBER: {row["accession_number"]}
CONFORMED SUBMISSION TYPE: {row["form"]}
CENTRAL INDEX KEY: {row["cik"]}
</SEC-HEADER><DOCUMENT><TYPE>{row["form"]}
<FILENAME>filing.htm
<TEXT><p>synthetic evidence</p></TEXT></DOCUMENT>""".encode()


def _index(row: dict[str, Any], literal: str = "2025-05-01 12:00:00") -> bytes:
    return f"""<html><div id="formName">Form {row["form"]}</div>
<div>SEC Accession No. {row["accession_number"]}</div>
<div class="infoHead">Accepted</div><div class="info">{literal}</div>
<div class="companyInfo"><span class="companyName">Synthetic Inc <acronym>CIK</acronym>
<a href="/cgi-bin/browse-edgar?CIK={row["cik"]}">{row["cik"]}</a></span>
<p>Type: <strong>{row["form"]}</strong></p></div></html>""".encode()


def test_literal_timezone_conflict_is_traced_not_blindly_relabelled() -> None:
    row = _filing()
    source = trace_source_row(_member([row]), row)
    result = reconcile_acceptance(row, source, _submission(row), _index(row))
    assert result["catalog_reproduces_literal"]
    assert result["resolution_status"] == "source_label_conflict_resolved_by_header_and_index"
    assert result["reviewed_sec_acceptance_proxy_at"] == "2025-05-01T16:00:00+00:00"
    assert result["catalog_minus_header_hours"] == -4
    assert not result["overlay_applied"] and not result["model_admission"]
    assert result["first_public_at"] is None and result["public_available_by"] is None


def test_correct_utc_control_is_not_shifted() -> None:
    row = _filing(accepted="2025-05-01T16:00:00Z")
    result = reconcile_acceptance(row, trace_source_row(_member([row]), row), _submission(row))
    assert result["resolution_status"] == "consistent_header_control"
    assert result["reviewed_sec_acceptance_proxy_at"] is None


@pytest.mark.parametrize("literal", ["20250501120000", "2025-05-01T12:00:00"])
def test_unspecified_timezone_cannot_be_invented(literal: str) -> None:
    row = _filing()
    result = reconcile_acceptance(
        row, trace_source_row(_member([row], literal), row), _submission(row), _index(row)
    )
    assert result["resolution_status"] == "source_timezone_unspecified"
    assert result["reviewed_sec_acceptance_proxy_at"] is None


@pytest.mark.parametrize("mutation", ["index", "issuer", "accession", "form", "period", "member"])
def test_source_row_identity_and_period_are_bound(mutation: str) -> None:
    row = _filing()
    content = _member([row])
    if mutation == "index":
        row["source_row_index"] = 1
    elif mutation == "issuer":
        row["cik"] = "0000000002"
    elif mutation == "accession":
        row["accession_number"] = "0000000001-25-000002"
    elif mutation == "form":
        row["form"] = "10-Q"
    elif mutation == "period":
        row["report_date"] = "2025-06-30"
    else:
        row["source_member"] = "CIK0000000002.json"
    with pytest.raises(ValueError):
        trace_source_row(content, row)


def test_archive_shard_flat_arrays_preserve_provenance() -> None:
    row = _filing()
    content = json.loads(_member([row]))["filings"]["recent"]
    row["source_member"] = "CIK0000000001-submissions-001.json"
    assert trace_source_row(json.dumps(content).encode(), row)["source_array_path"] == "root"


def test_catalog_corruption_is_not_blame_shifted_to_source() -> None:
    row = _filing()
    source = trace_source_row(_member([row], "2025-05-01T13:00:00Z"), row)
    result = reconcile_acceptance(row, source, _submission(row), _index(row))
    assert result["resolution_status"] == "catalog_source_divergence"
    assert result["reviewed_sec_acceptance_proxy_at"] is None


@pytest.mark.parametrize("mutation", ["cik", "form", "accession"])
def test_index_identity_is_not_just_an_accession_mention(mutation: str) -> None:
    row = _filing()
    other = copy.deepcopy(row)
    if mutation == "cik":
        other["cik"] = "0000000002"
    elif mutation == "form":
        other["form"] = "10-Q"
    else:
        other["accession_number"] = "0000000001-25-000002"
    content = _index(other) + f"<p>{row['accession_number']}</p>".encode()
    with pytest.raises(ValueError, match="identity"):
        parse_index_acceptance(content, row)


@pytest.mark.parametrize(
    "literal", ["", "2025-11-02 01:30:00", "2025-03-09 02:30:00", "2025-02-30 00:00:00"]
)
def test_missing_or_ambiguous_index_clock_never_resolves(literal: str) -> None:
    row = _filing()
    result = reconcile_acceptance(
        row, trace_source_row(_member([row]), row), _submission(row), _index(row, literal)
    )
    assert result["resolution_status"] == "header_index_conflict"
    assert result["reviewed_sec_acceptance_proxy_at"] is None


def test_index_required_for_conflict_and_disagreement_keeps_quarantine() -> None:
    row = _filing()
    source = trace_source_row(_member([row]), row)
    assert (
        reconcile_acceptance(row, source, _submission(row))["resolution_status"]
        == "index_evidence_needed"
    )
    result = reconcile_acceptance(row, source, _submission(row), _index(row, "2025-05-01 13:00:00"))
    assert result["resolution_status"] == "header_index_conflict"


def test_winter_uses_five_hours_without_a_global_shift() -> None:
    row = _filing(accepted="2025-01-23T12:00:00Z")
    result = reconcile_acceptance(
        row,
        trace_source_row(_member([row]), row),
        _submission(row, "20250123120000"),
        _index(row, "2025-01-23 12:00:00"),
    )
    assert result["catalog_minus_header_hours"] == -5
    assert result["reviewed_sec_acceptance_proxy_at"] == "2025-01-23T17:00:00+00:00"


def _event() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "ticker": "AAA",
        "validation_event_id": "event",
        "filing_accession": "0000000001-25-000002",
        "accepted_at": "2025-05-02T12:00:00Z",
        "period_end": "2025-03-31",
    }


def test_date_padding_catches_new_candidates_and_does_not_hide_uncertainty() -> None:
    event = _event()
    old = _filing()
    added = _filing(number="000003", accepted="2025-05-02T15:00:00Z")
    added["items"] = "7.01;9.01"
    unrelated = {**added, "accession_number": "0000000001-25-000004", "items": "12.02"}
    scan = boundary_scan(event, pd.DataFrame([old, added, unrelated]))
    assert set(scan.accession_number) == {old["accession_number"], added["accession_number"]}
    rows = [
        {"cik": event["cik"], "accession_number": acc, "reviewed_sec_acceptance_proxy_at": stamp}
        for acc, stamp in [
            (event["filing_accession"], "2025-05-02T16:00:00Z"),
            (old["accession_number"], "2025-05-01T16:00:00Z"),
            (added["accession_number"], "2025-05-02T15:00:00Z"),
        ]
    ]
    result = audit_event_boundary(
        event, scan.to_dict("records"), rows[:-1], [old["accession_number"]]
    )
    assert result["candidate_membership_unchanged"] is None
    assert result["reviewed_candidate_accessions"] is None
    result = audit_event_boundary(event, scan.to_dict("records"), rows, [old["accession_number"]])
    assert result["candidate_membership_unchanged"] is False
    assert len(result["reviewed_candidate_accessions"]) == 2
    assert not result["primary_sample_reselected"]


def _pipeline_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    rows = [
        _filing(),
        _filing(number="000002", form="10-Q", accepted="2025-05-02T12:00:00Z"),
        _filing(cik="0000000002", accepted="2025-05-01T16:00:00Z"),
    ]
    rows[1]["source_row_index"] = 1
    bulk = tmp_path / "submissions.zip"
    with zipfile.ZipFile(bulk, "w") as archive:
        archive.writestr(rows[0]["source_member"], _member(rows[:2]))
        archive.writestr(rows[2]["source_member"], _member(rows[2:]))
    for row in rows:
        row["source_zip_sha256"] = digest_file(bulk)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    pd.DataFrame(rows).assign(future_return=999).to_parquet(catalog / "filing_catalog.parquet")
    (catalog / "output_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "filing_catalog.parquet": {
                        "sha256": digest_file(catalog / "filing_catalog.parquet")
                    }
                }
            }
        )
    )
    plan = {
        "events": [_event()],
        "earnings_candidates": [{**rows[0], "validation_event_id": "event"}],
        "config": {
            "catalog_root": str(catalog),
            "catalog_manifest_sha256": digest_file(catalog / "output_manifest.json"),
            "cached_submission_roots": [],
        },
    }
    plan_root = write_bundle(tmp_path, "plan", {"plan.json": json.dumps(plan).encode()})
    plan_sha = digest_file(plan_root / "output_manifest.json")
    followup = write_bundle(
        tmp_path,
        "followup",
        {
            "report.json": json.dumps({"plan_manifest_sha256": plan_sha}).encode(),
            "header_clock_audit.json": json.dumps(
                [
                    {**rows[0], "status": "clock_conflict_quarantined"},
                    {**rows[2], "status": "consistent_acceptance_proxy"},
                ]
            ).encode(),
        },
    )
    config = {
        "version": VERSION,
        "plan_root": str(plan_root),
        "plan_manifest_sha256": plan_sha,
        "followup_root": str(followup),
        "followup_manifest_sha256": digest_file(followup / "output_manifest.json"),
        "submissions_zip": str(bulk),
        "submissions_sha256": digest_file(bulk),
        "raw_root": str(tmp_path / "raw"),
        "existing_indexes": [],
        "boundary_padding_days": 1,
        "apply_to_catalog": False,
        "model_fitting_allowed": False,
        "return_values_allowed": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    responses = {}
    for row in rows:
        literal = "20250502120000" if row["form"] == "10-Q" else "20250501120000"
        index_literal = "2025-05-02 12:00:00" if row["form"] == "10-Q" else "2025-05-01 12:00:00"
        responses[submission_url(row["cik"], row["accession_number"])] = _submission(row, literal)
        responses[index_url(row["cik"], row["accession_number"])] = _index(row, index_literal)
    return path, responses


def test_pipeline_offline_missing_then_mocked_collection_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import audit_sec_clock_provenance as runner

    config, responses = _pipeline_fixture(tmp_path)
    plan_root, plan = freeze_clock_plan(config, tmp_path / "out")
    assert len(plan["targets"]) == 3 and len(plan["focus_ciks"]) == 1
    assert all("future_return" not in r for r in plan["targets"])
    _, initial = run_provenance(plan_root, plan, tmp_path / "out")
    assert initial["proposed_clock_corrections"] == 0 and initial["missing_artifacts"] > 0
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        collect_clock_evidence(plan, limit=5)
    calls = []

    class FakeClient:
        def __init__(self, identity: str, *, request_interval_seconds: float) -> None:
            assert identity and request_interval_seconds == 0.25

        def get_filing_artifact(self, url: str) -> SecResponse:
            calls.append(url)
            return SecResponse(
                url=url,
                content=responses[url],
                status_code=200,
                fetched_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
                headers={},
            )

    monkeypatch.setattr(runner, "SecClient", FakeClient)
    monkeypatch.setenv("SEC_USER_AGENT", "Test demo@example.com")
    assert collect_clock_evidence(plan, limit=2)["completed"] == 2
    assert len(calls) == 2
    assert collect_clock_evidence(plan, limit=5)["completed"] == 3
    assert collect_clock_evidence(plan, limit=5)["selected"] == 0
    assert len(calls) == 5
    root, report = run_provenance(plan_root, plan, tmp_path / "out")
    assert report["proposed_clock_corrections"] == 2 and report["missing_artifacts"] == 0
    assert report["resolution_counts"]["consistent_header_control"] == 1
    assert report["boundary_membership_unchanged"] == 1
    assert not report["overlay_applied"] and not report["return_values_loaded"]
    assert run_provenance(plan_root, plan, tmp_path / "out")[0] == root
    altered = copy.deepcopy(plan)
    altered["targets"][0]["accepted_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="frozen"):
        run_provenance(plan_root, altered, tmp_path / "out")
    (root / "report.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        run_provenance(plan_root, plan, tmp_path / "out")
