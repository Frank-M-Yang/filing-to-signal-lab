from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from prospective_cohort_feasibility import (
    CORE_COLUMNS,
    CohortFeasibilityConfig,
    CohortFeasibilityError,
    build_prospective_cohort_feasibility,
    calculate_prospective_cohort_feasibility,
)
from scripts.build_prospective_cohort_feasibility import _latest_report_root, _linked_root


def test_capacity_is_conditional_and_registry_unchanged() -> None:
    inputs = _inputs()
    originals = [frame.copy(deep=True) for frame in inputs]
    result = calculate_prospective_cohort_feasibility(*inputs)

    for original, frame in zip(originals, inputs, strict=True):
        pd.testing.assert_frame_equal(original, frame)
    report = result.report
    assert report["registered_events"] == 37
    assert report["registered_issuers"] == 20
    assert report["clean_events"] == 25
    assert report["issuers_with_any_clean_event"] == 14
    assert report["clean_event_shortfall"] == 5
    assert report["clean_issuer_shortfall"] == 1
    assert not report["current_quality_capacity_sufficient"]
    assert not report["new_cohort_registered"]
    assert not report["retention_is_end_to_end"]
    assert not report["statistical_power_estimated"]
    optimistic = result.capacity_scenarios.query(
        "assumed_additional_event_retention == 1 and assumed_positive_share == 0.3"
    ).iloc[0]
    assert optimistic["raw_events_for_expected_counts"] == 45
    assert optimistic["raw_issuers_for_expected_counts"] == 22
    conservative = result.capacity_scenarios.query(
        "assumed_additional_event_retention == 0.8 and assumed_positive_share == 0.2"
    ).iloc[0]
    assert conservative["raw_events_for_expected_counts"] == 74
    assert conservative["raw_issuers_for_expected_counts"] == 27
    assert not result.capacity_scenarios["tail_shares_are_observed"].any()
    assert not result.capacity_scenarios["additional_retention_is_observed"].any()


def test_unprobed_is_unknown_and_missing_rank_has_explicit_fallback() -> None:
    inputs = _inputs()
    inputs[2].loc[2, "selection_order"] = None
    result = calculate_prospective_cohort_feasibility(*inputs)

    assert result.probe_queue["sec_ticker"].tolist() == ["NEW_B", "NEW_A"]
    assert pd.isna(result.probe_queue.iloc[-1]["selection_order"])
    assert result.report["core_missing_selection_order"] == 1
    unknown = result.core_coverage.query("coverage_status == 'unprobed'")
    assert unknown["strict_forward_revenue_consensus_observed"].isna().all()
    assert not result.probe_queue["queue_is_registration"].any()
    assert not result.probe_queue["probe_authorized"].any()
    budget25 = result.request_budget_scenarios.query("assumed_daily_request_budget == 25").iloc[0]
    assert budget25["available_new_probe_requests"] == 4
    budget20 = result.request_budget_scenarios.query("assumed_daily_request_budget == 20").iloc[0]
    assert budget20["available_new_probe_requests"] == 0
    assert pd.isna(budget20["days_for_one_pass_of_unprobed_queue"])


def test_zero_clean_capacity_is_unknown_not_a_false_finite_estimate() -> None:
    inputs = _inputs()
    inputs[1]["quality_status"] = "review_required"
    inputs[1]["downstream_admission"] = False
    result = calculate_prospective_cohort_feasibility(*inputs)

    assert result.capacity_scenarios["raw_events_for_expected_counts"].isna().all()
    assert result.capacity_scenarios["raw_issuers_for_expected_counts"].isna().all()


@pytest.mark.parametrize("column", ["actual_revenue", "innovation_raw", "net_return_20d"])
def test_populated_outcomes_are_forbidden(column: str) -> None:
    inputs = _inputs()
    inputs[1][column] = 1.0
    with pytest.raises(CohortFeasibilityError, match="Outcome value"):
        calculate_prospective_cohort_feasibility(*inputs)


@pytest.mark.parametrize("column", ["return_values_loaded", "replacement_allowed"])
def test_unknown_or_enabled_forbidden_flags_are_rejected(column: str) -> None:
    inputs = _inputs()
    inputs[0][column] = None
    with pytest.raises(CohortFeasibilityError, match="Forbidden or unknown flag"):
        calculate_prospective_cohort_feasibility(*inputs)


@pytest.mark.parametrize("change", ["event_set", "ticker", "admission", "duplicate"])
def test_no_registry_replacement_remapping_or_promotion(change: str) -> None:
    inputs = _inputs()
    if change == "event_set":
        inputs[1].loc[0, "prospective_event_id"] = "replacement"
    elif change == "ticker":
        inputs[1].loc[0, "ticker"] = "WRONG"
    elif change == "admission":
        inputs[1].loc[36, "downstream_admission"] = True
    else:
        inputs[0].loc[1, ["cik", "target_period_end"]] = inputs[0].loc[
            0, ["cik", "target_period_end"]
        ]
    with pytest.raises(CohortFeasibilityError):
        calculate_prospective_cohort_feasibility(*inputs)


@pytest.mark.parametrize(
    "cfg",
    [
        CohortFeasibilityConfig(additional_retention_scenarios=(0,)),
        CohortFeasibilityConfig(additional_retention_scenarios=(float("nan"),)),
        CohortFeasibilityConfig(tail_share_scenarios=(0.6,)),
        CohortFeasibilityConfig(daily_budget_scenarios=(0,)),
    ],
)
def test_invalid_planning_assumptions_are_rejected(cfg: CohortFeasibilityConfig) -> None:
    with pytest.raises(CohortFeasibilityError):
        calculate_prospective_cohort_feasibility(*_inputs(), config=cfg)


def test_bundle_reuse_and_tampering_fail_closed(tmp_path: Path) -> None:
    parents = _write_parents(tmp_path)
    registry_bytes = (parents[2] / "prospective_market_innovation_events.parquet").read_bytes()
    result = build_prospective_cohort_feasibility(*parents, output_root=tmp_path / "output")
    again = build_prospective_cohort_feasibility(*parents, output_root=tmp_path / "output")

    assert result.output_root == again.output_root
    assert result.report == again.report
    assert "不是功效分析" in (result.output_root / "cohort_feasibility.md").read_text(
        encoding="utf-8"
    )
    assert (
        parents[2] / "prospective_market_innovation_events.parquet"
    ).read_bytes() == registry_bytes
    (result.output_root / "unprobed_company_queue.parquet").write_bytes(b"tampered")
    with pytest.raises(CohortFeasibilityError, match="Changed existing"):
        build_prospective_cohort_feasibility(*parents, output_root=tmp_path / "output")
    (parents[3] / "core_asset_82.csv").write_bytes(b"tampered")
    with pytest.raises(CohortFeasibilityError, match="changed core CSV"):
        build_prospective_cohort_feasibility(*parents, output_root=tmp_path / "output")


def test_generation_time_not_mtime_and_parent_hash_resolution(tmp_path: Path) -> None:
    for folder, day in [("z_old", "2026-09-01"), ("a_new", "2026-09-14")]:
        root = tmp_path / folder
        root.mkdir()
        (root / "report.json").write_text(json.dumps({"generated_at": day + "T12:00:00Z"}))
        (root / "output_manifest.json").write_text(json.dumps({"folder": folder}))
    assert _latest_report_root(tmp_path, "report.json").name == "a_new"
    digest = _digest(tmp_path / "z_old" / "output_manifest.json")
    assert _linked_root(tmp_path, digest).name == "z_old"
    with pytest.raises(CohortFeasibilityError, match="exact hash-linked"):
        _linked_root(tmp_path, "0" * 64)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issuer_numbers = list(range(1, 15)) + list(range(1, 12)) + list(range(15, 21)) * 2
    registry = pd.DataFrame(
        {
            "prospective_event_id": [f"e{i}" for i in range(37)],
            "registration_order": range(1, 38),
            "cik": [f"{i:010d}" for i in issuer_numbers],
            "ticker": [f"T{i}" for i in issuer_numbers],
            "target_period_end": ["2026-09-30"] * 14
            + ["2026-12-31"] * 11
            + ["2026-09-30"] * 6
            + ["2026-12-31"] * 6,
            "replacement_allowed": False,
            "actual_value_loaded_at_registration": False,
            "return_value_loaded_at_registration": False,
            "selection_uses_return_values": False,
        }
    )
    quality = registry.drop(
        columns=[
            "replacement_allowed",
            "actual_value_loaded_at_registration",
            "return_value_loaded_at_registration",
        ]
    ).copy()
    quality["quality_status"] = ["clean"] * 25 + ["review_required"] * 9 + ["quarantined"] * 3
    quality["downstream_admission"] = quality["quality_status"].eq("clean")
    quality["actual_values_loaded"] = False
    quality["return_values_loaded"] = False
    core = pd.DataFrame(
        {
            "core_asset_id": ["1", "2", "3", "4"],
            "core_asset_version": "frozen-core",
            "core_asset_status": "active",
            "company_name": ["Known A", "Known B", "New A", "New B"],
            "sec_cik": [f"{i:010d}" for i in [1, 2, 21, 22]],
            "sec_ticker": ["T1", "T2", "NEW_A", "NEW_B"],
            "mapping_status": "accepted",
            "selection_order": [1, 2, 3, 4],
        }
    )
    coverage = pd.DataFrame(
        {
            "cik": [f"{i:010d}" for i in range(1, 21)],
            "ticker": [f"T{i}" for i in range(1, 21)],
            "current_revenue_consensus_observed": True,
            "strict_forward_revenue_consensus_observed": True,
        }
    )
    return registry, quality, core, coverage


def _write_parents(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    registry, quality, core, coverage = _inputs()
    roots = {
        name: root / name
        for name in ["quality", "readiness", "registry", "core", "consensus", "provider_audit"]
    }
    for path in roots.values():
        path.mkdir()
    _write_bundle(
        roots["registry"],
        "prospective_market_innovation_report.json",
        {},
        {"prospective_market_innovation_events.parquet": registry},
    )
    _write_bundle(
        roots["consensus"],
        "alpha_vantage_live_audit_report.json",
        {},
        {"alpha_vantage_live_coverage.parquet": coverage},
    )
    _write_bundle(
        roots["readiness"],
        "prospective_market_innovation_readiness_report.json",
        {
            "preregistration_manifest_sha256": _digest(roots["registry"] / "output_manifest.json"),
            "consensus_audit_manifest_sha256": _digest(roots["consensus"] / "output_manifest.json"),
        },
        {},
    )
    _write_bundle(
        roots["quality"],
        "prospective_consensus_quality_report.json",
        {
            "readiness_manifest_sha256": _digest(roots["readiness"] / "output_manifest.json"),
        },
        {"prospective_consensus_event_quality.parquet": quality},
    )
    _write_bundle(
        roots["provider_audit"],
        "consensus_provider_audit_report.json",
        {},
        {
            "consensus_provider_decisions.parquet": pd.DataFrame(
                [
                    {
                        "provider": "probe",
                        "decision": "rejected",
                        "reason": "no historical revenue",
                        "selection_uses_return_values": False,
                    }
                ]
            ),
        },
    )
    path = roots["core"] / "core_asset_82.csv"
    core[CORE_COLUMNS].to_csv(path, index=False)
    (roots["core"] / "output_manifest.json").write_text(
        json.dumps({"artifacts": {path.name: {"sha256": _digest(path)}}})
    )
    return (
        roots["quality"],
        roots["readiness"],
        roots["registry"],
        roots["core"],
        roots["consensus"],
        roots["provider_audit"],
    )


def _write_bundle(
    root: Path, report_name: str, report: dict[str, object], tables: dict[str, pd.DataFrame]
) -> None:
    artifacts = {}
    for name, table in tables.items():
        table.to_parquet(root / name, index=False)
        artifacts[name] = {"sha256": _digest(root / name)}
    report["generated_at"] = "2026-09-14T12:00:00Z"
    (root / report_name).write_text(json.dumps(report))
    (root / "output_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "report_sha256": _digest(root / report_name),
            }
        )
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
