"""Tests for representative/stress separation and freeze gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sec_data.universe_design import build_universe_design


def test_universe_design_integrates_size_but_waits_for_identity(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    inventory = pd.DataFrame(
        [
            _inventory_row("0000000001", 2020, True),
            _inventory_row("0000000002", 2020, False),
        ]
    )
    signals = pd.DataFrame(
        [
            _signal("0000000001", "s1", "periodic_amendment", "stress", False),
            _signal("0000000001", "s2", "formal_delisting", "exit", True),
            _signal("0000000002", "s3", "financial_statement_non_reliance", "stress", False),
        ]
    )
    archives = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "coverage_status": "resolved_from_bulk_snapshot",
            }
        ]
    )
    _write_catalog(catalog, inventory, signals, archives)
    size = tmp_path / "size"
    _write_size(catalog, size)

    result = build_universe_design(catalog, size_root=size, output_root=tmp_path / "design")
    representative = pd.read_parquet(result.output_root / "representative_candidates.parquet")
    stress = pd.read_parquet(result.output_root / "stress_membership.parquet")
    exits = pd.read_parquet(result.output_root / "exit_signal_events.parquet")

    assert representative["cik"].tolist() == ["0000000001"]
    assert representative["event_time_size_ready"].all()
    assert representative.loc[0, "size_anchor_total_assets_usd"] == 125_000_000
    assert representative.loc[0, "size_anchor_stratum"] == "asset_q4_largest"
    assert not representative["size_plausibility_review_required"].any()
    assert not representative["sample_freeze_eligible"].any()
    assert not representative["future_exit_information_allowed_for_sampling"].any()
    assert stress["cik"].nunique() == 1
    assert stress["population_weight"].isna().all()
    assert not stress["may_estimate_population_rate"].any()
    assert exits["final_exit_reason"].tolist() == ["unknown"]
    assert result.report["archive_coverage_gate"]
    assert result.report["size_asset_integrated"]
    assert result.report["event_time_size_coverage"] == 1.0
    assert not result.report["representative_sample_frozen"]


def test_universe_design_accepts_subject_matched_cover_gates(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    inventory = pd.DataFrame([_inventory_row("0000000001", 2020, True)])
    signals = pd.DataFrame(
        columns=[
            "signal_id",
            "cik",
            "accession_number",
            "form",
            "accepted_at",
            "event_family",
            "signal_type",
            "is_exit_candidate",
            "listing_status_effect",
            "reporting_status_effect",
            "operating_status_effect",
            "signal_confidence",
            "source_items",
            "source_url",
            "source_zip_sha256",
        ]
    )
    archives = pd.DataFrame(
        [{"cik": "0000000001", "coverage_status": "resolved_from_bulk_snapshot"}]
    )
    _write_catalog(catalog, inventory, signals, archives)
    size = tmp_path / "size"
    _write_size(catalog, size)
    cover = tmp_path / "cover"
    _write_cover(catalog, cover)

    result = build_universe_design(
        catalog,
        size_root=size,
        cover_root=cover,
        output_root=tmp_path / "design",
    )
    representative = pd.read_parquet(result.output_root / "representative_candidates.parquet")

    assert representative.loc[0, "filing_identity_ready"]
    assert representative.loc[0, "security_identity_observed"]
    assert representative.loc[0, "historical_industry_ready"]
    assert representative.loc[0, "filing_time_sic"] == "3571"
    assert representative.loc[0, "sample_freeze_eligible"]
    assert result.report["representative_sample_frozen"]


def _inventory_row(cik: str, year: int, representative: bool) -> dict[str, object]:
    return {
        "cik": cik,
        "calendar_year": year,
        "company_name_snapshot": f"Company {cik}",
        "entity_type_snapshot": "operating",
        "sic_snapshot": "3571",
        "original_10q_filings": 3 if representative else 0,
        "original_10k_filings": 1 if representative else 0,
        "periodic_amendments": 0,
        "late_filing_notices": 0,
        "non_reliance_notices": 0,
        "auditor_change_notices": 0,
        "listing_notices": 0,
        "formal_delistings": 0,
        "reporting_deregistrations": 0,
        "bankruptcy_notices": 0,
        "acquisition_candidates": 0,
        "representative_candidate": representative,
        "current_metadata_only": True,
    }


def _signal(
    cik: str,
    signal_id: str,
    signal_type: str,
    event_family: str,
    exit_candidate: bool,
) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "cik": cik,
        "accession_number": f"{cik}-20-00000{signal_id[-1]}",
        "form": "25" if exit_candidate else "10-Q/A",
        "accepted_at": "2020-06-01T20:00:00+00:00",
        "event_family": event_family,
        "signal_type": signal_type,
        "is_exit_candidate": exit_candidate,
        "listing_status_effect": "delisted" if exit_candidate else "unchanged",
        "reporting_status_effect": "unchanged",
        "operating_status_effect": "unknown" if exit_candidate else "unchanged",
        "signal_confidence": "formal_filing",
        "source_items": "",
        "source_url": "https://www.sec.gov/test",
        "source_zip_sha256": "a" * 64,
    }


def _write_catalog(
    root: Path,
    inventory: pd.DataFrame,
    signals: pd.DataFrame,
    archives: pd.DataFrame,
) -> None:
    frames = {
        "issuer_year_inventory.parquet": inventory,
        "filing_signals.parquet": signals,
        "archive_coverage_audit.parquet": archives,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for filename, frame in frames.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "catalog_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-catalog",
                "archive_coverage_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )


def _write_size(catalog: Path, root: Path) -> None:
    root.mkdir()
    anchors = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "calendar_year": 2020,
                "accession_number": "0000000001-20-000001",
                "form": "10-Q",
                "accepted_at": "2020-05-01T20:30:00+00:00",
                "total_assets_usd": 125_000_000.0,
                "log_total_assets": 18.643824,
                "asset_size_percentile": 1.0,
                "asset_size_stratum": "asset_q4_largest",
                "size_anchor_rule": "first_original_periodic_filing_with_assets_in_year",
                "size_measure_not_market_cap": True,
            }
        ]
    )
    path = root / "issuer_year_size_anchors.parquet"
    anchors.to_parquet(path, index=False)
    artifact = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    catalog_manifest_sha256 = hashlib.sha256(
        (catalog / "output_manifest.json").read_bytes()
    ).hexdigest()
    (root / "size_feature_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-size",
                "catalog_manifest_sha256": catalog_manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": {"issuer_year_size_anchors.parquet": artifact}}),
        encoding="utf-8",
    )


def _write_cover(catalog: Path, root: Path) -> None:
    root.mkdir()
    anchors = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "calendar_year": 2020,
                "target_id": "target-1",
                "index_artifact_preserved": True,
                "cover_artifact_preserved": True,
                "filing_identity_ready": True,
                "security_identity_observed": True,
                "tradable_security_mapping_ready": True,
                "historical_identity_ready": True,
                "historical_industry_ready": True,
                "registrant_name": "Demo Corporation",
                "identity_status": "verified_from_cover_facts",
                "sic": "3571",
                "sic_description": "Electronic Computers",
                "industry_evidence_status": "filing_index_sic",
            }
        ]
    )
    path = root / "issuer_year_identity_industry_anchors.parquet"
    anchors.to_parquet(path, index=False)
    artifact = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    catalog_manifest_sha256 = hashlib.sha256(
        (catalog / "output_manifest.json").read_bytes()
    ).hexdigest()
    (root / "filing_cover_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-cover",
                "catalog_manifest_sha256": catalog_manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": {"issuer_year_identity_industry_anchors.parquet": artifact}}),
        encoding="utf-8",
    )
