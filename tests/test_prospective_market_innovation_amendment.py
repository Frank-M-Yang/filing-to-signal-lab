"""v3.1 amendment record tests: it can only be frozen before any outcome exists."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from prospective_market_innovation_amendment import (
    AMENDMENT_RULES,
    ProspectiveMarketInnovationAmendmentError,
    freeze_prospective_market_innovation_amendment,
    verify_pre_outcome_state,
)

IDS = ["evt-1", "evt-2"]


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": IDS,
            "cik": ["1", "2"],
            "actual_value_loaded_at_registration": [False, False],
            "return_value_loaded_at_registration": [False, False],
            "selection_uses_return_values": [False, False],
        }
    )


def _readiness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": IDS,
            "actual_value_loaded": [False, False],
            "return_value_loaded": [False, False],
            "selection_uses_return_values": [False, False],
        }
    )


def _states(second: str = "consensus_revision_ready") -> pd.DataFrame:
    return pd.DataFrame(
        {"prospective_event_id": IDS, "current_state": ["consensus_revision_ready", second]}
    )


def _bundle(root: Path, tables: dict[str, pd.DataFrame], report: dict | None = None) -> Path:
    root.mkdir(parents=True)
    artifacts = {}
    for name, frame in tables.items():
        frame.to_parquet(root / name, index=False)
        artifacts[name] = {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
    manifest: dict = {"artifacts": artifacts}
    if report is not None:
        name = next(key for key in report if key.endswith("_report_name"))
        report_path = root / report[name]
        payload = {k: v for k, v in report.items() if k != name}
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (root / "output_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _roots(tmp_path: Path, second_state: str = "consensus_revision_ready") -> list[Path]:
    return [
        _bundle(tmp_path / "prereg", {"prospective_market_innovation_events.parquet": _registry()}),
        _bundle(
            tmp_path / "ready",
            {"prospective_market_innovation_event_readiness.parquet": _readiness()},
        ),
        _bundle(
            tmp_path / "states",
            {"prospective_event_states.parquet": _states(second_state)},
        ),
        _bundle(
            tmp_path / "clock",
            {},
            {
                "_report_name": "forward_disclosure_clock_report.json",
                "dataset_version": "clock-x",
                "summary": {"disclosed_events": 0},
            },
        ),
        _bundle(
            tmp_path / "quality",
            {},
            {
                "_report_name": "prospective_disclosure_quality_report.json",
                "dataset_version": "quality-x",
                "status_counts": {"awaiting_disclosure": 2},
                "actual_values_loaded": False,
                "return_values_loaded": False,
            },
        ),
    ]


def test_verification_passes_only_before_outcomes() -> None:
    summary = verify_pre_outcome_state(_registry(), _readiness(), _states())
    assert summary == {
        "events": 2,
        "state_counts": {"consensus_revision_ready": 2},
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "paper_positions": 0,
    }
    with pytest.raises(ProspectiveMarketInnovationAmendmentError):
        verify_pre_outcome_state(_registry(), _readiness(), _states("actual_ready"))
    loaded = _readiness()
    loaded.loc[0, "return_value_loaded"] = True
    with pytest.raises(ProspectiveMarketInnovationAmendmentError):
        verify_pre_outcome_state(_registry(), loaded, _states())
    with pytest.raises(ProspectiveMarketInnovationAmendmentError):
        verify_pre_outcome_state(_registry(), _readiness().iloc[[0]], _states())


def test_freeze_is_content_addressed_and_binds_the_cohort(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    first = freeze_prospective_market_innovation_amendment(*roots, output_root=tmp_path / "out")
    second = freeze_prospective_market_innovation_amendment(*roots, output_root=tmp_path / "out")
    assert first.output_root == second.output_root
    report = first.report
    assert report["registered_events"] == 2
    assert report["cohort_changed"] is False and report["thresholds_changed"] is False
    assert report["rules"] == AMENDMENT_RULES
    assert report["clock_dataset_version"] == "clock-x"
    assert report["verification_at_declaration"]["paper_positions"] == 0
    stored_ids = json.loads((first.output_root / "amended_event_ids.json").read_text())
    assert stored_ids == sorted(IDS)


def test_freeze_fails_closed_once_an_event_has_an_actual(tmp_path: Path) -> None:
    roots = _roots(tmp_path, second_state="actual_ready")
    with pytest.raises(ProspectiveMarketInnovationAmendmentError):
        freeze_prospective_market_innovation_amendment(*roots, output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()
