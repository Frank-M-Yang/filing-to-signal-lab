"""Synthetic evidence tests: timestamps, monetary precision, fixed-cohort chronology."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_publication_chronology import build_chronology
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file
from sec_data.execution_contract import execution_window
from sec_data.publication_chronology import (
    FORECAST_CLOCKS,
    PANEL_COLUMNS,
    chronology_sensitivity,
    review_publisher_source,
)


def _fixture() -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    content = b'<html><time datetime="2025-04-22 16:05:00">April 22</time>'
    content += b"<p>DEMO GAAP Q1 in thousands Net sales 100 Gross profit 40</p></html>"
    spec = {
        "source_id": "demo_ir",
        "ticker": "DEMO",
        "cik": "0000000001",
        "release_accession": "0000000001-25-000002",
        "url": "https://issuer.example.org/release",
        "snapshot_sha256": hashlib.sha256(content).hexdigest(),
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
        "context_anchors": ["DEMO GAAP Q1 in thousands"],
        "clock_literal": "2025-04-22 16:05:00",
        "clock_kind": "time_datetime",
        "content_scope": "full_financial_table",
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


def _review(content: bytes, spec: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    spec["snapshot_sha256"] = hashlib.sha256(content).hexdigest()
    return review_publisher_source(content, spec, cells)


def test_naive_time_stays_unknown_and_sensitivity_is_not_evidence() -> None:
    content, spec, cells = _fixture()
    reviewed = _review(content, spec, cells)
    assert reviewed["source_reported_at_utc"] is None
    clocks = {k: "2025-05-06T17:42:46Z" for k in FORECAST_CLOCKS}
    clocks["forecast_feature_cutoff"] = "2025-03-12T19:36:14Z"
    scenarios = chronology_sensitivity(reviewed, clocks)
    assert scenarios["lower_inclusive"] == "2025-04-22T02:05:00+00:00"
    assert scenarios["upper_exclusive"] == "2025-04-23T06:05:01+00:00"
    assert (
        scenarios["forecast_comparisons"]["forecast_generated_at"]["status"]
        == "after_all_label_scenarios"
    )
    assert (
        scenarios["forecast_comparisons"]["forecast_feature_cutoff"]["status"]
        == "before_all_label_scenarios"
    )
    assert reviewed["first_public_at"] is None and not scenarios["first_public_certified"]
    assert "sensitivity_only" in scenarios["basis"]
    with pytest.raises(ValueError, match="protocol"):
        chronology_sensitivity(reviewed, clocks, max_offset_hours=4)


def test_date_only_cannot_answer_same_day_forecast_ordering() -> None:
    content, spec, cells = _fixture()
    content += b"<p>Release October 31, 2024 LOUDON</p>"
    spec.update(
        {
            "clock_kind": "visible_date",
            "clock_literal": "October 31, 2024",
            "reported_date": "2024-10-31",
            "clock_anchor": "Release October 31, 2024 LOUDON",
        }
    )
    result = _review(content, spec, cells)
    scenario = chronology_sensitivity(result, {k: "2024-10-31T11:25:38Z" for k in FORECAST_CLOCKS})
    assert scenario["lower_inclusive"] == "2024-10-30T10:00:00+00:00"
    assert scenario["upper_exclusive"] == "2024-11-01T14:00:00+00:00"
    assert all(
        c["status"] == "overlaps_label_scenarios" for c in scenario["forecast_comparisons"].values()
    )


def test_explicit_est_minute_and_strict_boundary_conditions() -> None:
    content, spec, cells = _fixture()
    content += b"<p>Published Nov 3, 2025 4:08pm EST Share</p>"
    spec.update(
        {
            "clock_kind": "visible_EST_minute",
            "clock_literal": "Nov 3, 2025 4:08pm EST",
            "clock_anchor": "Published Nov 3, 2025 4:08pm EST Share",
        }
    )
    result = _review(content, spec, cells)
    assert result["source_reported_at_utc"] == "2025-11-03T21:08:00+00:00"
    assert result["point_clock_and_exact_values"] and not result["model_admission"]
    for stamp, expected in [
        ("2025-11-03T21:07:59Z", "before_all_label_scenarios"),
        ("2025-11-03T21:08:00Z", "overlaps_label_scenarios"),
        ("2025-11-03T21:08:59Z", "overlaps_label_scenarios"),
        ("2025-11-03T21:09:00Z", "after_all_label_scenarios"),
    ]:
        scenario = chronology_sensitivity(result, {k: stamp for k in FORECAST_CLOCKS})
        assert scenario["forecast_comparisons"]["forecast_generated_at"]["status"] == expected
    content = content.replace(b"Nov 3, 2025", b"Jul 3, 2025")
    spec["clock_literal"] = "Jul 3, 2025 4:08pm EST"
    spec["clock_anchor"] = "Published Jul 3, 2025 4:08pm EST Share"
    with pytest.raises(ValueError, match="DST"):
        _review(content, spec, cells)


def test_explicit_edt_keeps_minute_precision_and_rejects_winter_mislabel() -> None:
    content, spec, cells = _fixture()
    content += b"<p>Published Apr 22, 2025 4:05pm EDT Share</p>"
    spec.update(
        {
            "clock_kind": "visible_EDT_minute",
            "clock_literal": "Apr 22, 2025 4:05pm EDT",
            "clock_anchor": "Published Apr 22, 2025 4:05pm EDT Share",
        }
    )
    result = _review(content, spec, cells)
    assert result["source_reported_at_utc"] == "2025-04-22T20:05:00+00:00"
    assert result["clock_precision"] == "minute_with_explicit_EDT"
    assert result["first_public_at"] is None and not result["model_admission"]
    scenario = chronology_sensitivity(
        result, dict.fromkeys(FORECAST_CLOCKS, "2025-04-22T20:05:59Z")
    )
    assert scenario["upper_exclusive"] == "2025-04-22T20:06:00+00:00"
    assert all(
        c["status"] == "overlaps_label_scenarios" for c in scenario["forecast_comparisons"].values()
    )
    content = content.replace(b"Apr 22, 2025", b"Dec 22, 2025")
    spec["clock_literal"] = "Dec 22, 2025 4:05pm EDT"
    spec["clock_anchor"] = "Published Dec 22, 2025 4:05pm EDT Share"
    with pytest.raises(ValueError, match="DST"):
        _review(content, spec, cells)
    spec["clock_kind"] = "visible_EST_minute"
    with pytest.raises(ValueError, match="timezone label"):
        _review(content, spec, cells)


def test_jsonld_binds_canonical_page_not_modified_or_image_clocks() -> None:
    content, spec, cells = _fixture()
    node = {
        "@type": "WebPage",
        "url": spec["url"],
        "datePublished": "2025-11-03T21:35:57+00:00",
        "dateModified": "2026-01-01T00:00:00+00:00",
    }
    spec.update(
        {
            "clock_kind": "jsonld_datePublished",
            "clock_literal": node["datePublished"],
            "content_scope": "rounded_summary",
            "cells": [],
        }
    )
    for alteration in ["none", "no_published", "duplicate", "wrong_url", "image"]:
        changed = copy.deepcopy(node)
        if alteration == "no_published":
            changed.pop("datePublished")
        elif alteration == "wrong_url":
            changed["url"] += "/another-quarter"
        elif alteration == "image":
            changed["@type"] = "ImageObject"
        graph = {"@graph": [changed, changed] if alteration == "duplicate" else [changed]}
        page = (
            content
            + b'<script type="application/ld+json">'
            + json.dumps(graph).encode()
            + b"</script>"
        )
        if alteration == "none":
            result = _review(page, spec, cells)
            assert result["source_reported_at_utc"] == node["datePublished"]
            assert not result["exact_gross_margin_components"]
            assert not result["point_clock_and_exact_values"]
        else:
            with pytest.raises(ValueError, match="metadata"):
                _review(page, spec, cells)


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "quarter",
        "unit",
        "value",
        "original",
        "context",
        "summary",
        "missing_cell",
        "invalid_time",
    ],
)
def test_source_or_accounting_mismatch_fails(problem: str) -> None:
    content, spec, cells = _fixture()
    if problem == "hash":
        spec["snapshot_sha256"] = "a" * 64
    elif problem == "quarter":
        cells[0]["period_end"] = "2025-06-30"
    elif problem == "unit":
        cells[0]["unit"] = "EUR"
    elif problem == "value":
        cells[0]["value"] += 1
    elif problem == "original":
        cells[0]["original_value"] += 1
    elif problem == "context":
        spec["context_anchors"] = ["Q2"]
    elif problem == "summary":
        spec["content_scope"] = "rounded_summary"
    elif problem == "missing_cell":
        spec["cells"] = spec["cells"][:1]
    else:
        content = content.replace(b"2025-04-22 16:05:00", b"NaT")
        spec["clock_literal"] = "NaT"
        spec["snapshot_sha256"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError):
        review_publisher_source(content, spec, cells)


def _encode(value: Any) -> bytes:
    if isinstance(value, pd.DataFrame):
        buffer = io.BytesIO()
        value.to_parquet(buffer, index=False)
        return buffer.getvalue()
    return value if isinstance(value, bytes) else json.dumps(value).encode()


def _pipeline_fixture(tmp_path: Path) -> Path:
    content, spec, cells = _fixture()
    config = json.loads(Path("docs/publication_chronology_v1.json").read_bytes())
    event = {
        "validation_event_id": "one",
        "cik": spec["cik"],
        "ticker": spec["ticker"],
        "filing_accession": "0000000001-25-000003",
        "period_start": spec["period_start"],
        "period_end": spec["period_end"],
        "forecast_proxy_check": None,
    }
    control = {**event, "cik": "0000000002", "validation_event_id": "two", "ticker": "CONTROL"}
    metadata = pd.DataFrame(
        [
            {
                "event_id": e["validation_event_id"],
                "company_id": e["cik"],
                "filing_accession": e["filing_accession"],
                "actual_available_at": "2025-05-06T18:00:00Z",
                **dict.fromkeys(FORECAST_CLOCKS, "2025-05-06T17:00:00Z"),
                "future_return": 999,
                "gm_forecast": 0.5,
            }
            for e in [event, control]
        ]
    )
    artifacts = {
        "calendar": {
            "validation_exchange_sessions.parquet": pd.DataFrame(
                [
                    {
                        "exchange_mic": "XNYS",
                        "session_date": d.date().isoformat(),
                        "regular_open_at": (d + pd.Timedelta(hours=13, minutes=30))
                        .tz_localize("UTC")
                        .isoformat(),
                        "regular_close_at": (d + pd.Timedelta(hours=20))
                        .tz_localize("UTC")
                        .isoformat(),
                        "source_snapshot_sha256": "b" * 64,
                    }
                    for d in pd.bdate_range("2025-04-01", "2025-07-01")
                ]
            )
        },
        "followup": {"event_review.json": [event, control], "metric_cell_review.json": cells},
        "clock": {"disarmed_replay_panel.parquet": metadata},
        "migration": {
            "publication_review_queue.json": [event, control],
            "migration_ledger.json": [
                {**e, "accession_number": e["filing_accession"]} for e in [event, control]
            ],
        },
    }
    for key, files in artifacts.items():
        root = write_bundle(tmp_path, key, {k: _encode(v) for k, v in files.items()})
        config[f"{key}_root"], config[f"{key}_manifest_sha256"] = (
            str(root),
            digest_file(root / "output_manifest.json"),
        )
    source = {k: spec[k] for k in ["ticker", "cik", "release_accession", "url"]}
    sources_path = tmp_path / "sources.json"
    sources_path.write_bytes(_encode({"sources": [source]}))
    receipt = {
        **source,
        "file": "source.html",
        "sha256": spec["snapshot_sha256"],
        "status_code": 200,
        "response_url": spec["url"],
        "content_type": "text/html",
        "bytes": len(content),
        "fetched_at": "2026-09-20T00:00:00Z",
    }
    root = write_bundle(
        tmp_path,
        "raw",
        {
            "source.html": content,
            "receipts.json": _encode(
                {"config_sha256": digest_file(sources_path), "receipts": [receipt]}
            ),
        },
    )
    config["batches"] = [
        {
            "root": str(root),
            "manifest_sha256": digest_file(root / "output_manifest.json"),
            "config": str(sources_path),
            "config_sha256": digest_file(sources_path),
        }
    ]
    config["reviews"] = [spec]
    path = tmp_path / "config.json"
    path.write_bytes(_encode(config))
    return path


@pytest.mark.parametrize("write_readout", [True, False])
def test_precise_full_table_creates_only_a_disarmed_replay_request(
    tmp_path: Path, write_readout: bool
) -> None:
    path = _pipeline_fixture(tmp_path)
    config = json.loads(path.read_bytes())
    config["write_readout"] = write_readout
    content, spec, _ = _fixture()
    content = content.replace(b"2025-04-22 16:05:00", b"2025-04-22T16:05:00+00:00")
    spec["clock_literal"] = "2025-04-22T16:05:00+00:00"
    spec["snapshot_sha256"] = hashlib.sha256(content).hexdigest()
    batch = config["batches"][0]
    receipts = json.loads((Path(batch["root"]) / "receipts.json").read_bytes())
    receipts["receipts"][0].update({"sha256": spec["snapshot_sha256"], "bytes": len(content)})
    raw = write_bundle(
        tmp_path, "aware", {"source.html": content, "receipts.json": _encode(receipts)}
    )
    batch["root"], batch["manifest_sha256"] = str(raw), digest_file(raw / "output_manifest.json")
    config["reviews"] = [spec]
    old_root = Path(config["migration_root"])
    ledger = json.loads((old_root / "migration_ledger.json").read_bytes())
    sessions = pd.read_parquet(
        Path(config["calendar_root"]) / "validation_exchange_sessions.parquet"
    )
    for event in ledger:
        event["exchange_mic"] = "XNYS"
        event["new_contract_reviewed_clock"] = execution_window(
            "2025-05-06T18:00:00Z", "XNYS", sessions
        )
    migration = write_bundle(
        tmp_path,
        "migration-aware",
        {
            "migration_ledger.json": _encode(ledger),
            "publication_review_queue.json": (
                old_root / "publication_review_queue.json"
            ).read_bytes(),
        },
    )
    config["migration_root"] = str(migration)
    config["migration_manifest_sha256"] = digest_file(migration / "output_manifest.json")
    path.write_bytes(_encode(config))
    root, report = build_chronology(path, tmp_path / "out")
    assert report["precise_label_and_full_table_events"] == report["disarmed_replay_requests"] == 1
    request = json.loads((root / "disarmed_replay_requests.json").read_bytes())[0]
    assert request["publisher_label_window"]["entry_session_date"] == "2025-04-24"
    assert request["same_contract_filing_proxy_window"]["entry_session_date"] == "2025-05-08"
    assert not request["change_filing_acceptance_timestamp"]
    assert not request["price_collection_authorized"] and not request["return_admission"]
    assert (root / "readout.md").exists() is write_readout


def test_full_pipeline_is_offline_projected_immutable_and_keeps_unreviewed_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _pipeline_fixture(tmp_path)
    read = pd.read_parquet

    def projected(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = read(*args, **kwargs)
        if "company_id" in frame:
            assert frame.columns.tolist() == PANEL_COLUMNS
        return frame

    monkeypatch.setattr(pd, "read_parquet", projected)
    root, report = build_chronology(path, tmp_path / "out")
    assert report["publication_events_retained"] == 2 and report["risk_events_reviewed"] == 1
    assert (
        report["exact_component_cells"] == 2 and report["precise_label_and_full_table_events"] == 0
    )
    assert report["forecast_after_all_label_scenarios_tickers"] == ["DEMO"]
    assert report["first_public_certified"] == 0 and not report["returns_recomputed"]
    queue = json.loads((root / "event_chronology.json").read_bytes())
    assert queue[1]["status"] == "not_in_risk_batch_retained"
    assert build_chronology(path, tmp_path / "out") == (root, report)
    (root / "report.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_chronology(path, tmp_path / "out")


@pytest.mark.parametrize("problem", ["raw", "cohort", "quarter", "link", "flags", "range"])
def test_pipeline_blocks_input_drift_and_unsafe_admission(tmp_path: Path, problem: str) -> None:
    path = _pipeline_fixture(tmp_path)
    config = json.loads(path.read_bytes())
    if problem == "raw":
        (Path(config["batches"][0]["root"]) / "source.html").write_bytes(b"changed")
    elif problem == "cohort":
        (Path(config["migration_root"]) / "publication_review_queue.json").write_bytes(b"[]")
    elif problem == "quarter":
        config["reviews"][0]["period_end"] = "2025-06-30"
    elif problem == "link":
        config["reviews"][0]["linked_from_source_id"] = "demo_ir"
    elif problem == "flags":
        config["model_admission"] = True
    else:
        config["max_offset_hours_if_unknown"] = 4
    path.write_bytes(_encode(config))
    with pytest.raises(ValueError):
        build_chronology(path, tmp_path / "out")


def test_missing_forecast_clocks_do_not_pass_as_before_publication() -> None:
    content, spec, cells = _fixture()
    result = chronology_sensitivity(_review(content, spec, cells), {})
    assert all(c["status"] == "missing" for c in result["forecast_comparisons"].values())
    assert not result["return_admission"]
