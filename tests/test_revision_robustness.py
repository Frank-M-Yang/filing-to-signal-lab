"""Tests for point-in-time revision-risk buffers and decision diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revision_robustness import (
    RevisionRobustnessConfig,
    RevisionRobustnessError,
    build_revision_robustness,
    validate_revision_robustness,
)


def test_revision_buffers_use_only_resolved_prior_outcomes() -> None:
    result = build_revision_robustness(_events(), _config())
    eligible = result.panel.loc[result.panel["revision_buffer_eligible"]]

    assert not eligible.empty
    assert (eligible["revision_history_end"] < eligible["actual_available_at"]).all()
    assert (eligible["revision_history_observations"] >= _config().min_resolved_history).all()
    assert (
        eligible["revision_history_changed_observations"] >= _config().min_revised_history
    ).all()
    assert result.report["right_censored_outcomes"] == 1
    assert result.report["primary_decision_events"] == len(eligible)


def test_unresolved_future_revision_cannot_change_past_buffers() -> None:
    events = _events()
    baseline = build_revision_robustness(events, _config()).panel
    changed = events.copy()
    changed.loc[4, "revision_size"] = 0.08
    changed.loc[4, "gross_margin_latest"] = changed.loc[4, "gross_margin"] + 0.08
    changed.loc[4, "was_revised"] = True
    alternative = build_revision_robustness(changed, _config()).panel
    resolution_time = pd.to_datetime(events.loc[4, "latest_available_at"], utc=True)
    before_resolution = baseline["actual_available_at"] <= resolution_time
    columns = [
        "revision_history_observations",
        "revision_history_changed_observations",
        "revision_at_risk_900",
        "revision_at_risk_950",
    ]

    pd.testing.assert_frame_equal(
        baseline.loc[before_resolution, columns].reset_index(drop=True),
        alternative.loc[before_resolution, columns].reset_index(drop=True),
    )


def test_buffers_are_monotonic_in_confidence() -> None:
    result = build_revision_robustness(_events(), _config())
    eligible = result.panel.loc[result.panel["revision_buffer_eligible"]]
    buffers = eligible[
        ["revision_at_risk_800", "revision_at_risk_900", "revision_at_risk_950"]
    ].to_numpy(dtype=float)

    assert (np.diff(buffers, axis=1) >= 0.0).all()
    assert result.confidence_calibration["confidence"].tolist() == [0.80, 0.90, 0.95]


def test_semantic_collision_audit_restores_original_value() -> None:
    events = _events()
    event_id = str(events.loc[5, "event_id"])
    reviews = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "audit_classification": "xbrl_semantic_collision",
                "audit_family": "data_pipeline",
                "audit_action": "set_latest_equal_original",
                "economic_error_correction": False,
                "event_time_structural_context": False,
                "review_status": "reviewed",
            }
        ]
    )

    result = build_revision_robustness(events, _config(), audit_reviews=reviews)
    audited = result.panel.loc[result.panel["event_id"].eq(event_id)].iloc[0]

    assert audited["revision_size_unreviewed"] == pytest.approx(0.03)
    assert audited["revision_size"] == 0.0
    assert audited["gross_margin_latest"] == audited["gross_margin"]
    assert not bool(audited["was_revised"])
    assert bool(audited["tail_audit_reviewed"])
    assert result.report["semantic_collision_corrections"] == 1
    assert str(result.report["revision_robustness_version"]).startswith("revision-robustness-v2-")


def test_validation_rejects_same_time_history_cutoff() -> None:
    config = _config()
    result = build_revision_robustness(_events(), config)
    broken = result.panel.copy()
    row = broken.index[broken["revision_buffer_eligible"]][0]
    broken.loc[row, "revision_history_end"] = broken.loc[row, "actual_available_at"]

    with pytest.raises(RevisionRobustnessError, match="same-time or future"):
        validate_revision_robustness(broken, config)


def _config() -> RevisionRobustnessConfig:
    return RevisionRobustnessConfig(
        primary_confidence=0.90,
        confidence_grid=(0.80, 0.90, 0.95),
        min_resolved_history=4,
        min_revised_history=1,
    )


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2020-01-15 21:00:00", tz="UTC")
    revisions = [0.00, 0.01, 0.00, -0.02, 0.00, 0.03, 0.00, 0.00, -0.01, 0.00, 0.02, 0.00]
    for index, revision in enumerate(revisions):
        actual_time = base + pd.Timedelta(days=90 * index)
        source_type = "original_same_accession" if index == 1 else "later_comparative_filing"
        latest_time = actual_time if index == 1 else actual_time + pd.Timedelta(days=180)
        gross_margin = 0.30 + index / 1_000
        rows.append(
            {
                "event_id": f"event-{index:02d}",
                "company_id": f"{index % 3 + 1:010d}",
                "company_name": f"Company {index % 3 + 1}",
                "ticker": f"T{index % 3 + 1}",
                "fiscal_year": 2020 + index // 4,
                "fiscal_quarter": f"Q{index % 4 + 1}",
                "period_end": actual_time.tz_localize(None) - pd.Timedelta(days=45),
                "actual_available_at": actual_time,
                "gross_margin": gross_margin,
                "gross_margin_latest": gross_margin + revision,
                "latest_available_at": latest_time,
                "revision_source_type": source_type,
                "was_revised": revision != 0.0,
                "revision_size": revision,
                "filing_accession": f"initial-{index:02d}",
                "latest_filing_accession": f"latest-{index:02d}",
                "M2_innovation_raw": (-1.0 if index % 2 else 1.0) * (0.005 + index / 1_000),
            }
        )
    return pd.DataFrame(rows)
