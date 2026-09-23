"""Offline synthetic contracts: full cohorts, separate clock effects, no admission."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from scripts.build_sec_disclosure_audit import write_bundle
from scripts.build_sec_execution_migration import build_migration
from scripts.collect_release_clock_evidence import collect_sources
from sec_data.clock_layer import calendar_windows
from sec_data.disclosure_audit import digest_file
from sec_data.execution_contract import ExecutionContract, execution_window
from sec_data.execution_migration import (
    EVENT_COLUMNS,
    migrate_execution_windows,
    publication_review_queue,
    review_release_page,
)


def _sessions() -> pd.DataFrame:
    # A synthetic calendar, not a source of real exchange evidence.
    dates = pd.bdate_range("2025-03-03", "2025-07-15").difference(pd.DatetimeIndex(["2025-05-26"]))
    return pd.DataFrame(
        [
            {
                "exchange_mic": "XNYS",
                "session_date": d.date().isoformat(),
                "regular_open_at": (d + pd.Timedelta(hours=9, minutes=30))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
                .isoformat(),
                "regular_close_at": (
                    d + pd.Timedelta(hours=13 if d == pd.Timestamp("2025-05-23") else 16)
                )
                .tz_localize("America/New_York")
                .tz_convert("UTC")
                .isoformat(),
                "source_snapshot_sha256": "b" * 64,
            }
            for d in dates
        ]
    )


def _events() -> pd.DataFrame:
    rows = []
    for i, hour in enumerate([11, 21]):
        stamp = f"2025-05-02T{hour}:00:00Z"
        legacy = calendar_windows(stamp, "XNYS", _sessions())["legacy_second_future_open"]
        rows.append(
            {
                "validation_event_id": f"event-{i}",
                "cik": f"{i + 1:010d}",
                "target_accession_number": f"{i + 1:010d}-25-000001",
                "form": "10-Q",
                "accepted_at": stamp,
                "assigned_ticker": f"DEMO{i}",
                "assigned_exchange_mic": "XNYS",
                "security_component_count": 1,
                "formal_entry_open_at": legacy["entry_at"],
                "formal_exit_close_at": legacy["exit_at"],
                "formal_entry_price_type": "unadjusted_open",
                "formal_exit_price_type": "unadjusted_close",
                "entry_session_offset": 2,
                "holding_sessions": 20,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
            }
        )
    return pd.DataFrame(rows)[EVENT_COLUMNS]


@pytest.mark.parametrize(
    "stamp",
    [
        "2025-05-02T11:00:00Z",
        "2025-05-02T13:30:00Z",
        "2025-05-02T17:00:00Z",
        "2025-05-02T20:00:00Z",
        "2025-05-03T00:30:00Z",
        "2025-05-04T12:00:00Z",
    ],
)
def test_rule_is_local_date_based_and_holds_twenty_sessions(stamp: str) -> None:
    schedule = _sessions()
    window = execution_window(stamp, "XNYS", schedule)
    assert window["entry_at"] == "2025-05-06T13:30:00+00:00"
    actual = schedule.loc[
        schedule.session_date.between(window["entry_session_date"], window["exit_session_date"])
    ]
    assert len(actual) == 20
    assert window["entry_price_type"] == "unadjusted_open"
    assert window["exit_price_type"] == "unadjusted_close"
    assert not window["clock_certified"] and not window["price_readiness"]
    assert not window["return_admission"]


def test_dst_holiday_and_early_close_come_from_calendar() -> None:
    sessions = _sessions()
    dst = execution_window("2025-03-07T12:00:00Z", "XNYS", sessions)
    assert dst["entry_at"] == "2025-03-11T13:30:00+00:00"
    holiday = execution_window("2025-05-23T18:00:00Z", "XNYS", sessions)
    assert holiday["entry_session_date"] == "2025-05-28"
    early = execution_window("2025-04-24T10:00:00Z", "XNYS", sessions)
    assert early["exit_at"] == "2025-05-23T17:00:00+00:00"


@pytest.mark.parametrize(
    "problem", ["naive", "venue", "duplicate", "hash", "local_date", "end", "contract"]
)
def test_incomplete_or_changed_contract_fails_closed(problem: str) -> None:
    sessions, stamp, venue = _sessions(), "2025-05-02T12:00:00Z", "XNYS"
    contract = ExecutionContract()
    if problem == "naive":
        stamp = stamp[:-1]
    elif problem == "venue":
        venue = "XNAS"
    elif problem == "duplicate":
        sessions = pd.concat([sessions, sessions.iloc[:1]])
    elif problem == "hash":
        sessions.loc[0, "source_snapshot_sha256"] = "missing"
    elif problem == "local_date":
        sessions.loc[0, "regular_open_at"] = "2025-03-02T14:30:00Z"
    elif problem == "end":
        stamp = "2025-07-10T12:00:00Z"
    else:
        contract = replace(contract, holding_sessions_inclusive=19)
    with pytest.raises(ValueError):
        execution_window(stamp, venue, sessions, contract)


def test_migration_separates_rule_and_clock_and_preserves_unresolved() -> None:
    events = _events()
    original = events.copy(deep=True)
    first = events.iloc[0]
    clock = {
        "cik": first.cik,
        "accession_number": first.target_accession_number,
        "form": first.form,
        "catalog_accepted_at": first.accepted_at,
        "experimental_acceptance_proxy_at": "2025-05-02T15:00:00Z",
        "resolution_status": "source_label_conflict_resolved_by_header_and_index",
    }
    rows = migrate_execution_windows(events, [clock], _sessions())
    assert len(rows) == 2
    assert [r["rule_changes_window"] for r in rows] == [True, False]
    assert rows[0]["source_clock_changed"]
    assert not any(r["clock_changes_window_under_new_rule"] for r in rows)
    assert rows[1]["clock_review_status"] == "catalog_proxy_unreviewed"
    assert not any(r["return_admission"] or r["price_read_authorized"] for r in rows)
    pd.testing.assert_frame_equal(events, original)
    clock["experimental_acceptance_proxy_at"] = None
    rows = migrate_execution_windows(events, [clock], _sessions())
    assert len(rows) == 2 and rows[0]["new_contract_reviewed_clock"] is None
    assert rows[0]["clock_changes_window_under_new_rule"] is None
    assert rows[0]["final_window_differs_from_legacy"] is None


@pytest.mark.parametrize("problem", ["outcomes", "selection", "duplicate", "legacy", "identity"])
def test_migration_rejects_leakage_changed_baseline_or_overlay(problem: str) -> None:
    events, layer = _events(), []
    if problem == "outcomes":
        events["future_return"] = 999
    elif problem == "selection":
        events.loc[0, "selection_uses_return_values"] = True
    elif problem == "duplicate":
        events = pd.concat([events, events.iloc[:1]])
    elif problem == "legacy":
        events.loc[0, "formal_entry_open_at"] = "2025-05-07T13:30:00Z"
    else:
        layer = [
            {
                "cik": events.iloc[0].cik,
                "accession_number": events.iloc[0].target_accession_number,
                "catalog_accepted_at": "2025-05-01T11:00:00Z",
                "form": "10-Q",
            }
        ]
    with pytest.raises(ValueError):
        migrate_execution_windows(events, layer, _sessions())


def _page_fixture() -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    content = b'<html><time datetime="2025-04-22 16:05:00">April 22</time>'
    content += b"<p>Demo company Q1 in thousands Net sales 100 Gross profit 40</p></html>"
    spec = {
        "ticker": "DEMO0",
        "cik": "0000000001",
        "release_accession": "0000000001-25-000002",
        "snapshot_sha256": hashlib.sha256(content).hexdigest(),
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
        "context_anchors": ["Demo company", "Q1", "in thousands"],
        "time_literal": "2025-04-22 16:05:00",
        "cells": [
            {"metric": "revenue", "value_usd": 100000, "anchor": "Net sales 100"},
            {"metric": "gross_profit", "value_usd": 40000, "anchor": "Gross profit 40"},
        ],
    }
    cells = [
        {
            "cik": spec["cik"],
            "release_accession": spec["release_accession"],
            "period_start": spec["period_start"],
            "period_end": spec["period_end"],
            "unit": "USD",
            "metric": c["metric"],
            "value": c["value_usd"],
            "original_value": c["value_usd"],
            "comparison_status": "exact_original_match",
        }
        for c in spec["cells"]
    ]
    return content, spec, cells


def test_publisher_label_never_certifies_first_public_or_invents_timezone() -> None:
    content, spec, cells = _page_fixture()
    result = review_release_page(content, spec, cells)
    assert result["clock_status"] == "timezone_unspecified"
    assert result["source_reported_at_utc"] is None
    spec["time_literal"] += "-04:00"
    content = content.replace(b'16:05:00"', b'16:05:00-04:00"')
    result = review_release_page(content, spec, cells)
    assert result["source_reported_at_utc"] == "2025-04-22T20:05:00+00:00"
    assert result["first_public_at"] is None and result["public_available_by"] is None
    assert not result["return_admission"] and not result["forecast_certification"]


@pytest.mark.parametrize(
    "problem", ["time", "ambiguous", "context", "value", "original", "quarter", "unit"]
)
def test_page_cells_and_time_must_match_pinned_review(problem: str) -> None:
    content, spec, cells = _page_fixture()
    if problem == "time":
        content = content.replace(b"datetime=", b"conference-time=")
    elif problem == "ambiguous":
        content += b'<time datetime="2025-04-22 17:00:00">Call</time>'
    elif problem == "context":
        content = content.replace(b"Q1", b"Q2")
    elif problem == "value":
        cells[0]["value"] = 200000
    elif problem == "original":
        cells[0]["original_value"] = 200000
    elif problem == "quarter":
        cells[0]["period_end"] = "2025-06-30"
    else:
        cells[0]["unit"] = "EUR"
    with pytest.raises(ValueError):
        review_release_page(content, spec, cells)


def _review_events() -> list[dict[str, Any]]:
    return [
        {
            "validation_event_id": e["validation_event_id"],
            "cik": e["cik"],
            "ticker": e["assigned_ticker"],
            "filing_accession": e["target_accession_number"],
            "candidate_count": 0,
            "matched_acceptance_proxies": {},
            "forecast_proxy_check": None,
        }
        for e in _events().to_dict("records")
    ]


def test_empty_source_queue_does_not_drop_unresolved_events() -> None:
    queue = publication_review_queue(_review_events(), [])
    assert len(queue) == 2
    assert all("no_SEC_candidate_does_not_prove_no_release" in r["remaining_work"] for r in queue)
    assert not any(r["return_admission"] or r["sample_replacement_allowed"] for r in queue)


def _encode(value: Any) -> bytes:
    if isinstance(value, pd.DataFrame):
        buffer = io.BytesIO()
        value.to_parquet(buffer, index=False)
        return buffer.getvalue()
    if isinstance(value, bytes):
        return value
    return json.dumps(value).encode()


def _pipeline_fixture(tmp_path: Path) -> Path:
    config = json.loads(Path("docs/sec_execution_migration_v1.json").read_bytes())
    content, spec, cells = _page_fixture()
    config["publisher_reviews"] = [spec]
    source = {
        "ticker": "DEMO0",
        "cik": spec["cik"],
        "release_accession": spec["release_accession"],
        "url": "https://ir.usana.com/demo",
        "kind": "issuer_ir_release",
    }
    path = tmp_path / "sources.json"
    path.write_bytes(_encode({"version": "release-clock-evidence-v1", "sources": [source]}))
    config["release_sources_config"] = str(path)
    config["release_sources_config_sha256"] = digest_file(path)
    receipt = {
        **source,
        "file": "page.html",
        "sha256": spec["snapshot_sha256"],
        "fetched_at": "2026-09-20T00:00:00Z",
        "status_code": 200,
        "content_type": "text/html",
        "response_url": source["url"],
        "bytes": len(content),
    }
    events = _events()
    events["future_return"] = 99999  # Projection must prevent reading this column.
    bundles = {
        "execution": {
            "validation_execution_events.parquet": events,
            "validation_exchange_sessions.parquet": _sessions(),
        },
        "clock": {"clock_layer.json": []},
        "followup": {"event_review.json": _review_events(), "metric_cell_review.json": cells},
        "release_snapshots": {
            "page.html": content,
            "receipts.json": {
                "config_sha256": config["release_sources_config_sha256"],
                "receipts": [receipt],
            },
        },
    }
    for name, artifacts in bundles.items():
        root = write_bundle(tmp_path, name, {k: _encode(v) for k, v in artifacts.items()})
        config[f"{name}_root"] = str(root)
        config[f"{name}_manifest_sha256"] = digest_file(root / "output_manifest.json")
    path = tmp_path / "config.json"
    path.write_bytes(_encode(config))
    return path


def test_full_offline_pipeline_is_deterministic_immutable_and_projected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pipeline_fixture(tmp_path)
    real_read = pd.read_parquet

    def read_metadata(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = real_read(*args, **kwargs)
        assert "future_return" not in frame
        return frame

    monkeypatch.setattr(pd, "read_parquet", read_metadata)
    root, report = build_migration(config, tmp_path / "out")
    assert report["events_retained"] == report["legacy_windows_reproduced"] == 2
    assert report["rule_changed_windows"] == 1
    assert report["publication_review_events"] == 2
    assert report["risk_batch_raw_snapshots"] == 1
    assert report["public_times_certified"] == 0
    assert not report["return_values_loaded"] and not report["trading_admission"]
    assert build_migration(config, tmp_path / "out") == (root, report)
    (root / "report.json").write_text("broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_migration(config, tmp_path / "out")


@pytest.mark.parametrize("problem", ["source_hash", "receipt_config", "sample", "admission"])
def test_pipeline_rejects_tampering_and_scope_changes(tmp_path: Path, problem: str) -> None:
    path = _pipeline_fixture(tmp_path)
    config = json.loads(path.read_bytes())
    if problem == "source_hash":
        (Path(config["release_snapshots_root"]) / "page.html").write_bytes(b"changed")
    elif problem == "receipt_config":
        config["release_sources_config_sha256"] = "a" * 64
    elif problem == "sample":
        events = _review_events()
        events[0]["filing_accession"] = "wrong"
        _, _, cells = _page_fixture()
        root = write_bundle(
            tmp_path,
            "bad-followup",
            {"event_review.json": _encode(events), "metric_cell_review.json": _encode(cells)},
        )
        config["followup_root"] = str(root)
        config["followup_manifest_sha256"] = digest_file(root / "output_manifest.json")
    else:
        config["model_admission"] = True
    path.write_bytes(_encode(config))
    with pytest.raises(ValueError):
        build_migration(path, tmp_path / "out")


def test_collection_preserves_success_before_later_timeout_and_never_follows_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        {
            "ticker": t,
            "cik": "0000000001",
            "release_accession": "demo",
            "url": f"https://ir.usana.com/{t}",
        }
        for t in ["A", "B", "C"]
    ]
    path = tmp_path / "sources.json"
    path.write_bytes(_encode({"version": "release-clock-evidence-v1", "sources": sources}))
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/B":
            assert list((tmp_path / "raw" / "responses").glob("*/receipt.json"))
            raise httpx.ReadTimeout("fixture", request=request)
        if request.url.path == "/C":
            return httpx.Response(
                302, content=b"redirect", headers={"location": "https://invalid.example/"}
            )
        return httpx.Response(
            200, content=b"exact source bytes", headers={"content-type": "text/html"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("scripts.collect_release_clock_evidence.httpx.Client", lambda **_: client)
    monkeypatch.setattr("scripts.collect_release_clock_evidence.time.sleep", lambda _: None)
    root = collect_sources(path, tmp_path / "raw")
    receipts = json.loads((root / "receipts.json").read_bytes())["receipts"]
    assert len(seen) == 3
    assert [r["status_code"] for r in receipts] == [200, None, 302]
    assert receipts[1]["error"] == "ReadTimeout"
    assert (root / receipts[0]["file"]).read_bytes() == b"exact source bytes"
    assert digest_file(root / receipts[0]["file"]) == receipts[0]["sha256"]


def test_collector_rejects_non_allowlisted_sources_before_network(tmp_path: Path) -> None:
    config = {
        "version": "release-clock-evidence-v1",
        "sources": [{"url": "https://www.sec.gov/demo"}],
    }
    path = tmp_path / "sources.json"
    path.write_bytes(_encode(config))
    with pytest.raises(ValueError, match="allowlist"):
        collect_sources(path, tmp_path / "raw")


def test_safety_flags_cannot_be_weakened_before_any_input_reads(tmp_path: Path) -> None:
    original = json.loads(Path("docs/sec_execution_migration_v1.json").read_bytes())
    for flag in ("overwrite_originals", "return_values_allowed", "model_admission"):
        config = copy.deepcopy(original)
        config[flag] = True
        path = tmp_path / "bad.json"
        path.write_bytes(_encode(config))
        with pytest.raises(ValueError, match="contract"):
            build_migration(path, tmp_path / "out")
