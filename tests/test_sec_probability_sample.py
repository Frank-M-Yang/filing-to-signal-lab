"""Tests for stratified inclusion probabilities and equity-scope adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sec_data.probability_sample import (
    ProbabilitySampleConfig,
    build_equity_scope_audit,
    build_probability_sample_plan,
)


def test_probability_plan_reconstructs_frame_without_scope_conditioning(tmp_path: Path) -> None:
    universe = tmp_path / "universe"
    targets = tmp_path / "targets"
    _write_probability_fixtures(universe, targets)

    result = build_probability_sample_plan(
        universe,
        targets,
        output_root=tmp_path / "sample",
        config=ProbabilitySampleConfig(sample_per_stratum=2),
    )
    selected = pd.read_parquet(result.output_root / "selected_issuer_years.parquet")
    strata = pd.read_parquet(result.output_root / "sampling_strata.parquet")

    assert len(selected) == 8
    assert len(strata) == 4
    assert (selected.groupby("sampling_stratum").size() == 2).all()
    assert np.isclose(selected["stage1_sampling_weight"].sum(), 16.0)
    assert not selected["selection_uses_identity_or_industry"].any()
    assert not selected["selection_uses_future_exit_information"].any()
    assert not selected["replacement_after_scope_review_allowed"].any()
    assert result.report["probability_weights_ready"]


def test_scope_audit_keeps_ineligible_units_and_never_replaces(tmp_path: Path) -> None:
    universe = tmp_path / "universe"
    targets = tmp_path / "targets"
    _write_probability_fixtures(universe, targets)
    plan = build_probability_sample_plan(
        universe,
        targets,
        output_root=tmp_path / "sample",
        config=ProbabilitySampleConfig(sample_per_stratum=2),
    )
    selected = pd.read_parquet(plan.output_root / "selected_issuer_years.parquet")
    cover = tmp_path / "cover"
    _write_cover_fixture(cover, selected, plan.report["catalog_manifest_sha256"])

    result = build_equity_scope_audit(
        plan.output_root,
        cover,
        output_root=tmp_path / "scope",
    )
    audit = pd.read_parquet(result.output_root / "sample_scope_audit.parquet")

    assert len(audit) == len(selected)
    assert not audit["scope_review_replacement_allowed"].any()
    assert (audit["equity_scope_status"] == "structural_non_common_equity_asset_backed").sum() == 1
    assert audit["provisional_equity_eligible"].sum() == len(audit) - 1
    assert result.report["structural_asset_backed"] == 1
    assert result.report["scope_gate"]
    assert not result.report["return_study_gate"]


def _write_probability_fixtures(universe_root: Path, target_root: Path) -> None:
    universe_root.mkdir()
    target_root.mkdir()
    universe_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    order = 0
    for year in (2020, 2021):
        for bucket in ("asset_q1_smallest", "asset_q4_largest"):
            for unit in range(4):
                order += 1
                cik = f"{year}{1 if bucket.endswith('smallest') else 4}{unit:05d}".zfill(10)
                target_id = f"target-{year}-{bucket}-{unit}"
                universe_rows.append(
                    {
                        "cik": cik,
                        "calendar_year": year,
                        "size_anchor_stratum": bucket,
                        "future_exit_information_allowed_for_sampling": False,
                    }
                )
                target_rows.append(
                    {
                        "target_id": target_id,
                        "acquisition_order": order,
                        "sampling_hash": hashlib.sha256(target_id.encode()).hexdigest(),
                        "cik": cik,
                        "calendar_year": year,
                        "filing_id": f"filing-{order}",
                        "accession_number": f"{cik}-20-{order:06d}",
                        "form": "10-Q",
                        "report_date": f"{year}-03-31",
                        "accepted_at": f"{year}-05-01T20:00:00+00:00",
                        "is_xbrl": True,
                        "is_inline_xbrl": True,
                        "primary_document": "filing.htm",
                        "primary_document_url": (
                            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/filing.htm"
                        ),
                        "index_url": (
                            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/index.html"
                        ),
                        "anchor_rule": "fixture",
                        "selection_uses_future_exit_information": False,
                    }
                )
    catalog_hash = "c" * 64
    _write_table_root(
        universe_root,
        "representative_candidates.parquet",
        pd.DataFrame(universe_rows),
        "universe_gate_report.json",
        {
            "dataset_version": "fixture-universe",
            "catalog_manifest_sha256": catalog_hash,
            "size_feature_version": "fixture-size",
        },
    )
    _write_table_root(
        target_root,
        "filing_cover_targets.parquet",
        pd.DataFrame(target_rows),
        "target_report.json",
        {
            "dataset_version": "fixture-targets",
            "catalog_manifest_sha256": catalog_hash,
        },
    )


def _write_cover_fixture(root: Path, selected: pd.DataFrame, catalog_hash: str) -> None:
    root.mkdir()
    rows: list[dict[str, object]] = []
    security_rows: list[dict[str, object]] = []
    for index, row in enumerate(selected.itertuples(index=False)):
        rows.append(
            {
                "target_id": row.target_id,
                "index_artifact_preserved": True,
                "cover_artifact_preserved": True,
                "filing_identity_ready": True,
                "security_identity_observed": True,
                "tradable_security_mapping_ready": index != 0,
                "explicit_no_trading_symbol": False,
                "historical_industry_ready": True,
                "registrant_name": f"Issuer {index}",
                "identity_status": "verified_from_cover_facts",
                "sic": "6189" if index == 0 else "3571",
                "sic_description": (
                    "Asset-Backed Securities" if index == 0 else "Electronic Computers"
                ),
                "industry_evidence_status": "filing_index_sic",
            }
        )
        security_rows.append(
            {
                "target_id": row.target_id,
                "security_subject_cik_match": True,
                "security_title": "9% Asset-Backed Notes" if index == 0 else "Common Stock",
                "trading_symbol": None if index == 0 else f"T{index}",
                "security_identity_observed": True,
                "tradable_security_mapping_ready": index != 0,
            }
        )
    _write_table_root(
        root,
        "issuer_year_identity_industry_anchors.parquet",
        pd.DataFrame(rows),
        "filing_cover_report.json",
        {
            "dataset_version": "fixture-cover",
            "catalog_manifest_sha256": catalog_hash,
        },
    )
    security_path = root / "filing_security_observations.parquet"
    pd.DataFrame(security_rows).to_parquet(security_path, index=False)
    manifest_path = root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["filing_security_observations.parquet"] = {
        "sha256": hashlib.sha256(security_path.read_bytes()).hexdigest()
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_table_root(
    root: Path,
    filename: str,
    frame: pd.DataFrame,
    report_filename: str,
    report: dict[str, object],
) -> None:
    path = root / filename
    frame.to_parquet(path, index=False)
    artifact = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / report_filename).write_text(json.dumps(report), encoding="utf-8")
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": {filename: artifact}}), encoding="utf-8"
    )
