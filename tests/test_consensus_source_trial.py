"""A source trial freezes identities and requests, not results or event admission."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.build_consensus_source_trial import build_trial_request, select_trial_periods


def _document(cik: str = "0000000001", ticker: str = "AAA") -> dict[str, Any]:
    return {
        "cik": int(cik),
        "name": "Synthetic company",
        "tickers": [ticker],
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q/A", "10-Q", "10-Q", "10-Q", "10-K"],
                "reportDate": [
                    "2026-03-31",
                    "2026-03-31",
                    "2026-03-31",
                    "2025-12-31",
                    "2026-06-30",
                    "2025-09-30",
                ],
                "acceptanceDateTime": [
                    "2026-05-02T20:00:00Z",
                    "2026-06-01T20:00:00Z",
                    "2026-05-01T20:00:00Z",
                    "2026-02-01T20:00:00Z",
                    "2026-08-01T00:00:00Z",
                    "2025-11-01T20:00:00Z",
                ],
                "accessionNumber": [
                    "duplicate-later",
                    "amendment",
                    "original",
                    "prior-quarter",
                    "at-cutoff",
                    "annual",
                ],
            }
        },
    }


def test_distinct_original_quarters_strict_cutoff_and_earliest_acceptance() -> None:
    result = select_trial_periods(_document(), cik="0000000001", cutoff_at="2026-08-01T00:00:00Z")
    assert [r["reference_accession"] for r in result] == ["original", "prior-quarter"]
    assert "amendment" not in str(result)


@pytest.mark.parametrize("mutation", ["identity", "length", "naive", "future_period"])
def test_inconsistent_metadata_rejected(mutation: str) -> None:
    document = _document()
    recent = document["filings"]["recent"]
    if mutation == "identity":
        document["cik"] = 2
    elif mutation == "length":
        recent["form"].pop()
    elif mutation == "naive":
        recent["acceptanceDateTime"][0] = "2026-05-02T20:00:00"
    else:
        recent["reportDate"][0] = "2026-09-30"
    with pytest.raises(ValueError):
        select_trial_periods(document, cik="0000000001", cutoff_at="2026-08-01T00:00:00Z")


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "PSX", "MPC", "SNEX", "DELL"]
    issuers = []
    for i, ticker in enumerate(tickers, 1):
        cik = str(i).zfill(10)
        document = _document(cik, ticker)
        content = json.dumps(document).encode()
        digest = hashlib.sha256(content).hexdigest()
        path = tmp_path / "data/sec/raw" / cik / digest[:2] / f"submissions-{digest}.json"
        _write(path, document)
        issuers.append(
            {
                "ticker": ticker,
                "cik": cik,
                "role": "synthetic_fixture",
                "submissions_sha256": digest,
            }
        )
    root = tmp_path / "original-plan"
    root.mkdir()
    table = root / "alpha_vantage_live_targets.parquet"
    pd.DataFrame([{**r, "collection_order": i} for i, r in enumerate(issuers[:6])]).to_parquet(
        table
    )
    parent_sha = _write(
        root / "output_manifest.json",
        {"artifacts": {table.name: {"sha256": hashlib.sha256(table.read_bytes()).hexdigest()}}},
    )
    docs = []
    for i in range(4):
        path = tmp_path / f"public-doc-{i}.json"
        digest = _write(path, {"synthetic": i})
        manifest = tmp_path / f"public-manifest-{i}.json"
        _write(
            manifest,
            {
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "url": "https://example.com/synthetic",
                "fetched_at": "2026-09-19T00:00:00Z",
            },
        )
        docs.append(str(manifest))
    config = {
        "version": "consensus-source-trial-v1",
        "selection_cutoff_at": "2026-08-01T00:00:00Z",
        "quarters_per_issuer": 2,
        "perspective_days_before_period_end": [30, 7],
        "issuers": issuers,
        "providers": ["factset_estimates_pit_consensus", "lseg_ibes_point_in_time"],
        "selection_rule": "Synthetic test only",
        "source_plan_root": str(root),
        "source_plan_manifest_sha256": parent_sha,
        "public_document_manifests": docs,
    }
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    return config_path


def test_same_forty_requests_and_no_economic_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _fixture(tmp_path)
    report = build_trial_request(config)
    assert report == build_trial_request(config)
    assert report["resolved_requests_per_provider"] == 40
    assert report["resolved_period_slots"] == 20
    assert len({r["request_id"] for r in report["requests"]}) == 40
    for request in report["requests"]:
        assert request["requested_as_of_at"] < request["target_period_end"]
        assert not request["reference_is_first_earnings_announcement"]
    for provider in report["provider_status"]:
        assert provider["financial_sample_rows_received"] == 0
        assert provider["acceptance"] == "not_tested"
    assert not report["formal_model_admission"]
    assert not report["purchase_authorized"]
    assert not report["return_values_loaded"]


@pytest.mark.parametrize(
    "mutation", ["source_hash", "duplicate_cik", "baseline_order", "stress_replace", "public_hash"]
)
def test_frozen_inputs_cannot_be_silently_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _fixture(tmp_path)
    config = json.loads(path.read_bytes())
    if mutation == "source_hash":
        source = next(Path("data/sec/raw").glob("*/*/*.json"))
        source.write_bytes(b"modified")
    elif mutation == "duplicate_cik":
        config["issuers"][1]["cik"] = config["issuers"][0]["cik"]
    elif mutation == "baseline_order":
        config["issuers"][0], config["issuers"][1] = config["issuers"][1], config["issuers"][0]
    elif mutation == "stress_replace":
        config["issuers"][-1]["ticker"] = "REPLACEMENT"
    else:
        Path("public-doc-0.json").write_bytes(b"modified")
    _write(path, config)
    with pytest.raises(ValueError):
        build_trial_request(path)


def test_insufficient_metadata_is_not_replaced_or_filled() -> None:
    document = copy.deepcopy(_document())
    recent = document["filings"]["recent"]
    for key in recent:
        recent[key] = recent[key][:1]
    result = select_trial_periods(document, cik="0000000001", cutoff_at="2026-08-01T00:00:00Z")
    assert len(result) == 1
