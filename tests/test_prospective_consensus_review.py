from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from prospective_consensus_quality import REVIEW_REQUIRED, calculate_prospective_consensus_quality
from prospective_consensus_review import (
    ConsensusReviewError,
    build_prospective_consensus_review,
    calculate_consensus_membership_bounds,
    calculate_prospective_consensus_review,
)


def test_added_high_forecast_can_explain_mean_change_without_revisions() -> None:
    result = calculate_consensus_membership_bounds(2, 3, 100, 320 / 3, 90, 110, 90, 120)

    assert result["required_added_or_removed_mean"] == pytest.approx(120)
    assert result["composition_not_ruled_out_by_necessary_bounds"]
    assert result["retained_mean_revision_lower_pct"] == pytest.approx(0)
    assert not result["membership_identity_verified"]


def test_removing_low_forecast_can_explain_mean_increase() -> None:
    result = calculate_consensus_membership_bounds(3, 2, 100, 110, 80, 120, 100, 120)

    assert result["required_added_or_removed_mean"] == pytest.approx(80)
    assert result["composition_not_ruled_out_by_necessary_bounds"]
    assert result["retained_mean_revision_lower_pct"] == pytest.approx(0)


def test_impossible_composition_yields_conditional_not_empirical_revision_bounds() -> None:
    result = calculate_consensus_membership_bounds(2, 3, 100, 120, 90, 110, 90, 140)

    assert not result["mean_only_composition_not_ruled_out"]
    assert not result["nested_set_range_contradiction"]
    assert result["retained_mean_revision_lower_pct"] == pytest.approx(0.1)
    assert result["retained_mean_revision_upper_pct"] == pytest.approx(0.35)
    assert not result["membership_identity_verified"]


def test_contracting_range_rejects_nested_addition_even_when_mean_is_compatible() -> None:
    result = calculate_consensus_membership_bounds(2, 3, 100, 100, 90, 110, 95, 105)

    assert result["mean_only_composition_not_ruled_out"]
    assert result["nested_set_range_contradiction"]
    assert not result["composition_not_ruled_out_by_necessary_bounds"]


def test_equal_count_is_not_verified_same_analyst_revision() -> None:
    result = calculate_consensus_membership_bounds(2, 2, 100, 102, 90, 110, 92, 112)

    assert result["counterfactual_status"] == "equal_count_membership_still_unknown"
    assert result["mean_only_composition_not_ruled_out"] is None
    assert result["retained_mean_revision_lower_pct"] is None


def test_review_keeps_admission_and_records_unobserved_dates() -> None:
    snapshots = _snapshots()
    quality = _quality(snapshots)
    before = quality.copy(deep=True)

    result = calculate_prospective_consensus_review(quality, snapshots)
    event = result.event_review.iloc[0]
    pair = result.pair_evidence.iloc[0]

    pd.testing.assert_frame_equal(quality, before)
    assert event["quality_status_after_review"] == REVIEW_REQUIRED
    assert not event["downstream_admission_after_review"]
    assert not event["promotion_authorized"]
    assert pair["uncaptured_calendar_dates"] == 12
    assert pair["counterfactual_available_at"] == pd.Timestamp("2026-09-14T12:00:00Z")
    assert result.report["quality_promotions"] == 0
    assert result.report["review_latest_mean_only_composition_not_ruled_out"] == 1
    assert result.report["downstream_admitted_issuers"] == 0
    assert result.report["formal_minimum_complete_events"] == 30
    assert result.report["formal_minimum_complete_issuers"] == 15
    assert not result.report["current_admission_capacity_sufficient"]


def test_single_snapshot_has_no_attribution_or_release() -> None:
    snapshots = _snapshots().iloc[:1].copy()

    result = calculate_prospective_consensus_review(_quality(snapshots), snapshots)

    assert result.pair_evidence.empty
    assert result.event_review.iloc[0]["latest_counterfactual_status"] == "no_pair"
    assert not result.event_review.iloc[0]["downstream_admission_after_review"]
    assert result.report["membership_attribution_identified_events"] == 0


def test_latest_stable_count_does_not_erase_historical_quality_warning() -> None:
    snapshots = _snapshots()
    later = snapshots.iloc[-1:].copy()
    later["captured_at"] = pd.Timestamp("2026-09-15T12:00:00Z")
    later["capture_date"] = pd.Timestamp("2026-09-15T00:00:00Z")
    later["source_sha256"] = hashlib.sha256(b"later").hexdigest()
    snapshots = pd.concat([snapshots, later], ignore_index=True)

    result = calculate_prospective_consensus_review(_quality(snapshots), snapshots)

    assert result.event_review.iloc[0]["quality_status_after_review"] == REVIEW_REQUIRED
    assert not result.event_review.iloc[0]["downstream_admission_after_review"]
    assert result.event_review.iloc[0]["latest_counterfactual_status"] == (
        "equal_count_membership_still_unknown"
    )


@pytest.mark.parametrize("column", ["actual_revenue", "revenue_innovation_std", "net_return_20d"])
def test_outcome_contamination_is_rejected(column: str) -> None:
    snapshots = _snapshots()
    quality = _quality(snapshots)
    snapshots[column] = 1.0

    with pytest.raises(ConsensusReviewError, match="Outcome value"):
        calculate_prospective_consensus_review(quality, snapshots)


def test_acceptance_cutoff_is_enforced_even_when_snapshot_clock_column_is_absent() -> None:
    snapshots = _snapshots()
    quality = _quality(snapshots)
    quality["accepted_at"] = pd.Timestamp("2026-09-14T12:00:00Z")

    with pytest.raises(ConsensusReviewError, match="acceptance cutoff"):
        calculate_prospective_consensus_review(quality, snapshots)


@pytest.mark.parametrize("alteration", ["capture_date", "ticker", "source_sha256", "count"])
def test_backdating_remapping_bad_hash_or_fractional_count_is_rejected(alteration: str) -> None:
    snapshots = _snapshots()
    quality = _quality(snapshots)
    if alteration == "capture_date":
        snapshots.loc[1, "capture_date"] = pd.Timestamp("2026-09-02T00:00:00Z")
    elif alteration == "ticker":
        snapshots.loc[1, "ticker"] = "WRONG"
    elif alteration == "source_sha256":
        snapshots.loc[1, "source_sha256"] = "not-a-hash"
    else:
        snapshots.loc[1, "revenue_estimate_analyst_count"] = 10.5

    with pytest.raises(ConsensusReviewError):
        calculate_prospective_consensus_review(quality, snapshots)


def test_build_is_reusable_immutable_and_rejects_tampered_parent(tmp_path: Path) -> None:
    quality_root, readiness_root = _write_parents(tmp_path)
    parent_bytes = (quality_root / "prospective_consensus_event_quality.parquet").read_bytes()
    first = build_prospective_consensus_review(
        quality_root, readiness_root, output_root=tmp_path / "output"
    )
    second = build_prospective_consensus_review(
        quality_root, readiness_root, output_root=tmp_path / "output"
    )

    assert first.output_root == second.output_root
    assert first.report == second.report
    assert (first.output_root / "consensus_membership_review.md").is_file()
    assert (
        quality_root / "prospective_consensus_event_quality.parquet"
    ).read_bytes() == parent_bytes
    (quality_root / "prospective_consensus_event_quality.parquet").write_bytes(b"tampered")
    with pytest.raises(ConsensusReviewError, match="changed parent artifact"):
        build_prospective_consensus_review(
            quality_root, readiness_root, output_root=tmp_path / "output"
        )


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prospective_event_id": "event-AAA",
                "ticker": "AAA",
                "target_period_end": pd.Timestamp("2026-09-30T00:00:00Z"),
                "captured_at": pd.Timestamp(f"{day}T12:00:00Z"),
                "capture_date": pd.Timestamp(f"{day}T00:00:00Z"),
                "revenue_estimate_average": average,
                "revenue_estimate_low": 90.0,
                "revenue_estimate_high": 110.0,
                "revenue_estimate_analyst_count": count,
                "source_occurrence": 1,
                "source_sha256": hashlib.sha256(day.encode()).hexdigest(),
                "horizon": "fiscal quarter",
                "historical_backtest_admission": False,
            }
            for day, average, count in [("2026-09-01", 100.0, 10.0), ("2026-09-14", 100.5, 11.0)]
        ]
    )


def _quality(snapshots: pd.DataFrame) -> pd.DataFrame:
    events = pd.DataFrame(
        [
            {
                "prospective_event_id": "event-AAA",
                "registration_order": 1,
                "cik": "0000000001",
                "ticker": "AAA",
                "target_period_end": pd.Timestamp("2026-09-30T00:00:00Z"),
                "accepted_at": pd.NaT,
                "future_filing_observed": False,
            }
        ]
    )
    return calculate_prospective_consensus_quality(events, snapshots).event_quality


def _write_parents(task_root: Path) -> tuple[Path, Path]:
    snapshots = _snapshots()
    readiness_root = task_root / "readiness"
    quality_root = task_root / "quality"
    readiness_root.mkdir()
    quality_root.mkdir()
    name = "prospective_market_innovation_consensus_snapshots.parquet"
    snapshots.to_parquet(readiness_root / name, index=False)
    readiness_manifest = {"artifacts": {name: {"sha256": _digest(readiness_root / name)}}}
    (readiness_root / "output_manifest.json").write_text(
        json.dumps(readiness_manifest), encoding="utf-8"
    )
    quality_name = "prospective_consensus_event_quality.parquet"
    _quality(snapshots).to_parquet(quality_root / quality_name, index=False)
    quality_report = {"readiness_manifest_sha256": _digest(readiness_root / "output_manifest.json")}
    report_path = quality_root / "prospective_consensus_quality_report.json"
    report_path.write_text(json.dumps(quality_report), encoding="utf-8")
    manifest = {
        "artifacts": {quality_name: {"sha256": _digest(quality_root / quality_name)}},
        "report_sha256": _digest(report_path),
    }
    (quality_root / "output_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return quality_root, readiness_root


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
