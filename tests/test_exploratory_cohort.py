"""Tests for outcome-blind calendar-year exploratory cohorts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sec_data.exploratory_cohort import (
    CalendarYearExploratoryCohortConfig,
    build_calendar_year_exploratory_cohort,
)


def test_calendar_year_selection_does_not_follow_price_coverage(tmp_path: Path) -> None:
    reference_root, price_root, calendar_root = _write_upstreams(tmp_path)

    result = build_calendar_year_exploratory_cohort(
        reference_root,
        price_root,
        calendar_root,
        output_root=tmp_path / "cohort",
        config=CalendarYearExploratoryCohortConfig(event_year=2025),
    )
    registry = pd.read_parquet(result.output_root / "exploratory_cohort_registry.parquet")
    selected = pd.read_parquet(result.output_root / "selected_exploratory_events.parquet")

    assert selected["price_target_id"].tolist() == ["price-2025"]
    assert registry.set_index("price_target_id").loc["price-2024", "provider_price_query_complete"]
    assert not registry.set_index("price_target_id").loc[
        "price-2025", "provider_price_query_complete"
    ]
    assert result.report["selection_uses_provider_price_coverage"] is False
    assert result.report["selection_uses_returns"] is False
    assert result.report["exploratory_only"] is True
    assert result.report["primary_effectiveness_eligible"] is False
    assert result.report["downstream_evidence_collection_gate"] is False


def _write_upstreams(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference_root = tmp_path / "reference"
    price_root = tmp_path / "price"
    calendar_root = tmp_path / "calendar"
    for root in (reference_root, price_root, calendar_root):
        root.mkdir()
    (reference_root / "massive_reference_report.json").write_text(
        json.dumps({"dataset_version": "reference-fixture-v1"}), encoding="utf-8"
    )
    _write_manifest(reference_root)
    reference_hash = _sha256(reference_root / "output_manifest.json")
    targets = pd.DataFrame(
        [
            {
                "price_target_id": "price-2024",
                "target_id": "event-2024",
                "cik": "0000000001",
                "accepted_at": "2024-03-01T21:00:00+00:00",
                "query_date": "2024-03-01",
                "form": "10-K",
                "accession_number": "0000000001-24-000001",
                "selected_ticker": "OLD",
                "selected_exchange_mic": "XNAS",
            },
            {
                "price_target_id": "price-2025",
                "target_id": "event-2025",
                "cik": "0000000002",
                "accepted_at": "2025-03-01T21:00:00+00:00",
                "query_date": "2025-03-01",
                "form": "10-K",
                "accession_number": "0000000002-25-000001",
                "selected_ticker": "NEW",
                "selected_exchange_mic": "XNYS",
            },
        ]
    )
    targets.to_parquet(price_root / "massive_daily_price_targets.parquet", index=False)
    pd.DataFrame(
        [
            {
                "price_target_id": "price-2024",
                "provider_price_query_complete": True,
                "observed_event_window_candidate": True,
            },
            {
                "price_target_id": "price-2025",
                "provider_price_query_complete": False,
                "observed_event_window_candidate": False,
            },
        ]
    ).to_parquet(price_root / "sample_massive_price_audit.parquet", index=False)
    (price_root / "massive_price_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "price-fixture-v1",
                "reference_manifest_sha256": reference_hash,
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(price_root)
    price_hash = _sha256(price_root / "output_manifest.json")
    pd.DataFrame(
        [
            {"price_target_id": "price-2024", "formal_event_window_ready": True},
            {"price_target_id": "price-2025", "formal_event_window_ready": False},
        ]
    ).to_parquet(calendar_root / "event_calendar_alignment.parquet", index=False)
    (calendar_root / "exchange_calendar_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "calendar-fixture-v1",
                "price_manifest_sha256": price_hash,
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(calendar_root)
    return reference_root, price_root, calendar_root


def _write_manifest(root: Path) -> None:
    artifacts = {path.name: {"sha256": _sha256(path)} for path in root.glob("*.parquet")}
    (root / "output_manifest.json").write_text(
        json.dumps({"dataset_version": root.name, "artifacts": artifacts}), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
