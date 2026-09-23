"""Synthetic, offline tests for selective clocks and whole-chain replay safety."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_sec_clock_layer import build_layer
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    HISTORY_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    build_clock_layer,
    calendar_windows,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file
from sec_data.pilot_filing_features import PilotFilingFeatureConfig


def _evidence() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    for i in range(2):
        stamp = "2025-05-02T11:56:15Z"
        records.append(
            {
                "cik": f"{i + 1:010d}",
                "accession_number": f"{i + 1:010d}-25-000030",
                "form": "10-Q",
                "accepted_at": stamp,
                "source_zip_sha256": "a" * 64,
                "source_member": f"CIK{i + 1:010d}.json",
                "source_row_index": 3,
            }
        )
    evidence = []
    for i, row in enumerate(records):
        new = "2025-05-02T15:56:15Z" if i == 0 else row["accepted_at"]
        evidence.append(
            {
                **row,
                "catalog_accepted_at": row["accepted_at"],
                "raw_acceptance_literal": row["accepted_at"],
                "literal_interpreted_as_labelled_utc": row["accepted_at"],
                "header_accepted_at_utc": new,
                "index_accepted_at_utc": new,
                "reviewed_sec_acceptance_proxy_at": new,
                "catalog_reproduces_literal": True,
                "resolution_status": "source_label_conflict_resolved_by_header_and_index"
                if i == 0
                else "consistent_header_control",
            }
        )
    return pd.DataFrame(records), evidence


def _history() -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(123)
    for q, period in enumerate(pd.date_range("2021-03-31", periods=16, freq="QE")):
        for company in range(3):
            stamp = (period + pd.Timedelta(days=40, hours=10 + company)).tz_localize("UTC")
            margin = 0.3 + 0.03 * np.sin(q + company) + company * 0.02 + rng.normal(0, 0.01)
            row: dict[str, Any] = {key: np.nan for key in NUMERIC_COLUMNS}
            row.update({key: pd.NaT for key in TIME_COLUMNS})
            row.update({key: False for key in FLAG_COLUMNS})
            row.update(
                {
                    "event_id": f"{company}-{q}",
                    "company_id": f"{company + 1:010d}",
                    "ticker": f"S{company}",
                    "filing_accession": f"{company + 1:010d}-{period.year % 100:02d}-{q:06d}",
                    "fiscal_year": period.year,
                    "fiscal_quarter": f"Q{q % 4 + 1}",
                    "period_start": period - pd.DateOffset(months=3) + pd.Timedelta(days=1),
                    "period_end": period,
                    "filing_form": "10-Q",
                    "ytd_filing_accession": None,
                    "q4_source_method": "not_applicable",
                    "gm_yoy_lag_event_id": None,
                    "forecast_model_version": None,
                    "revenue": 100 + company + q,
                    "gross_profit": (100 + company + q) * margin,
                    "gross_margin": margin,
                    "gm_level": margin,
                    "analysis_ready_level": True,
                    "filing_accepted_at": stamp,
                    "actual_available_at": stamp,
                }
            )
            records.append(row)
    result = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    for column in TIME_COLUMNS:
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def _config() -> PilotFilingFeatureConfig:
    return PilotFilingFeatureConfig(minimum_forecast_training_events=8, minimum_prior_oos_errors=2)


def _sessions(year: int = 2025) -> pd.DataFrame:
    # Synthetic schedule, deliberately with an early close and an omitted holiday.
    dates = pd.bdate_range(f"{year}-05-01", f"{year}-07-15").difference(
        pd.DatetimeIndex([f"{year}-05-26"])
    )
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
                    d + pd.Timedelta(hours=13 if d.day == 23 and d.month == 5 else 16)
                )
                .tz_localize("America/New_York")
                .tz_convert("UTC")
                .isoformat(),
                "source_snapshot_sha256": "b" * 64,
            }
            for d in dates
        ]
    )


def test_only_evidence_bound_row_changes_and_input_is_untouched() -> None:
    catalog, evidence = _evidence()
    original = catalog.copy(deep=True)
    layer = build_clock_layer(catalog, evidence)
    assert [r["clock_changed"] for r in layer] == [True, False]
    assert all(r["first_public_at"] is None and not r["model_admission"] for r in layer)
    pd.testing.assert_frame_equal(catalog, original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("form", "10-K"),
        ("source_member", "wrong.json"),
        ("source_row_index", 4),
        ("source_zip_sha256", "bad"),
        ("catalog_accepted_at", "2025-05-02T12:00Z"),
        ("raw_acceptance_literal", "2025-05-02T11:56:15"),
        ("index_accepted_at_utc", "2025-05-02T17:00Z"),
    ],
)
def test_overlay_identity_and_clock_conflicts_fail(field: str, value: Any) -> None:
    catalog, evidence = _evidence()
    evidence[0][field] = value
    with pytest.raises(ValueError):
        build_clock_layer(catalog, evidence)


def test_duplicates_and_unresolved_clocks_never_become_approved() -> None:
    catalog, evidence = _evidence()
    with pytest.raises(ValueError, match="Duplicate"):
        build_clock_layer(catalog, [*evidence, evidence[0]])
    evidence[0]["resolution_status"] = "header_index_conflict"
    layer = build_clock_layer(catalog, evidence)
    assert layer[0]["experimental_acceptance_proxy_at"] is None
    assert not layer[0]["overlay_applied"]


def test_exact_history_alias_is_required_and_no_blind_timestamp_shift() -> None:
    history = _history()
    first = history.iloc[0]
    stamp = first.actual_available_at.isoformat()
    layer = [
        {
            "cik": first.company_id,
            "accession_number": first.filing_accession,
            "catalog_accepted_at": stamp,
            "experimental_acceptance_proxy_at": (
                first.actual_available_at + pd.Timedelta(hours=4)
            ).isoformat(),
        }
    ]
    changed = apply_history_clocks(history, layer)
    assert changed.iloc[0].actual_available_at == first.actual_available_at + pd.Timedelta(hours=4)
    assert changed.iloc[1].actual_available_at == history.iloc[1].actual_available_at
    bad = history.copy()
    bad.loc[0, "actual_available_at"] = first.actual_available_at - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="alias"):
        apply_history_clocks(bad, layer)
    layer[0]["experimental_acceptance_proxy_at"] = None
    with pytest.raises(ValueError, match="Unresolved"):
        apply_history_clocks(history, layer)


def test_q4_source_updates_by_accession_but_future_dependency_stops() -> None:
    history = _history()
    source = history.iloc[0]
    history.loc[3, "ytd_filing_accession"] = source.filing_accession
    history.loc[3, "ytd_available_at"] = source.actual_available_at
    layer = [
        {
            "cik": source.company_id,
            "accession_number": source.filing_accession,
            "catalog_accepted_at": source.actual_available_at.isoformat(),
            "experimental_acceptance_proxy_at": (
                source.actual_available_at + pd.Timedelta(hours=4)
            ).isoformat(),
        }
    ]
    revised = apply_history_clocks(history, layer)
    assert revised.loc[3, "ytd_available_at"] == revised.loc[0, "actual_available_at"]
    layer[0]["experimental_acceptance_proxy_at"] = history.loc[3, "actual_available_at"].isoformat()
    with pytest.raises(ValueError, match="Q4"):
        apply_history_clocks(history, layer)


def test_replay_resets_stale_joins_and_rebuilds_training_and_sigma() -> None:
    history = _history()
    before = replay_history(history, _config())
    source = history.loc[history.event_id.eq("0-12")].iloc[0]
    layer = [
        {
            "cik": source.company_id,
            "accession_number": source.filing_accession,
            "catalog_accepted_at": source.actual_available_at.isoformat(),
            "experimental_acceptance_proxy_at": (
                source.actual_available_at + pd.Timedelta(hours=4)
            ).isoformat(),
        }
    ]
    shifted = apply_history_clocks(before, layer)
    shifted["gm_yoy_lag"] = 9000  # Must not survive recomputation.
    shifted["forecast_sigma"] = 9000
    after = replay_history(shifted, _config())
    counts, changes = compare_histories(before, after)
    assert counts["actual_available_at"] == 1
    assert counts["forecast_training_observations"] > 0
    assert counts["gm_forecast"] > 0 and counts["forecast_sigma"] > 0
    assert not changes.empty and after.gm_yoy_lag.max() < 1
    assert audit_replay_chronology(after)["violations"] == 0
    second = replay_history(after, _config())
    assert compare_histories(after, second)[1].empty


@pytest.mark.parametrize(
    "field",
    [
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_generated_at",
        "forecast_error_history_end",
    ],
)
def test_same_time_and_future_dependencies_rejected(field: str) -> None:
    history = replay_history(_history(), _config())
    index = history.index[history.gm_innovation_std.notna()][0]
    history.loc[index, field] = history.loc[index, "actual_available_at"]
    with pytest.raises(ValueError, match="Non-prior"):
        audit_replay_chronology(history)


def test_comparison_rejects_changed_sample_and_tracks_nan_transitions() -> None:
    before = replay_history(_history(), _config())
    with pytest.raises(ValueError, match="sample"):
        compare_histories(before, before.iloc[:-1])
    after = before.copy()
    after.loc[0, "gm_forecast"] = 0.2
    counts, _ = compare_histories(before, after)
    assert counts["gm_forecast"] == 1


def test_two_entry_conventions_and_after_close_reaction_bracket() -> None:
    sessions = _sessions()
    pre = calendar_windows("2025-05-02T11:56:15Z", "XNYS", sessions)
    intra = calendar_windows("2025-05-02T15:56:15Z", "XNYS", sessions)
    assert pre["session_relation"] == "pre_open" and intra["session_relation"] == "in_session"
    assert pre["legacy_second_future_open"]["entry_at"].startswith("2025-05-05")
    assert intra["legacy_second_future_open"]["entry_at"].startswith("2025-05-06")
    assert pre["second_session_after_local_date"] == intra["second_session_after_local_date"]
    close = calendar_windows("2025-05-02T20:00:00Z", "XNYS", sessions)
    assert close["session_relation"] == "at_or_after_close"
    assert close["daily_close_bracket"]["end_at"].startswith("2025-05-05")


def test_local_date_not_utc_date_and_early_close_holidays() -> None:
    sessions = _sessions()
    night = calendar_windows("2025-05-03T00:30:00Z", "XNYS", sessions)
    assert night["local_date"] == "2025-05-02"
    assert night["session_relation"] == "at_or_after_close"
    early = calendar_windows("2025-05-23T17:01:00Z", "XNYS", sessions)
    assert early["session_relation"] == "at_or_after_close"
    assert early["second_session_after_local_date"]["entry_at"].startswith("2025-05-28")
    assert early["second_session_after_local_date"]["holding_sessions_inclusive"] == 20


@pytest.mark.parametrize(
    "stamp,exchange",
    [
        ("2025-01-01T00:00Z", "XNYS"),
        ("2025-05-02T12:00Z", "XNAS"),
        ("2025-07-14T12:00Z", "XNYS"),
        ("2025-05-02T12:00", "XNYS"),
    ],
)
def test_incomplete_or_naive_calendar_inputs_fail(stamp: str, exchange: str) -> None:
    with pytest.raises(ValueError):
        calendar_windows(stamp, exchange, _sessions())


def test_pipeline_rejects_weakened_contract_before_accessing_sources(tmp_path: Path) -> None:
    config = json.loads(Path("docs/sec_clock_layer_v1.json").read_bytes())
    for field in ("overwrite_originals", "return_values_allowed", "model_admission"):
        bad = copy.deepcopy(config)
        bad[field] = True
        path = tmp_path / "config.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="contract"):
            build_layer(path, tmp_path / "output")


def test_no_hyperparameter_changes_in_config_roundtrip() -> None:
    cfg = _config()
    assert PilotFilingFeatureConfig(**asdict(cfg)) == cfg


def _bundle(root: Path, files: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    root.mkdir(parents=True)
    manifest: dict[str, Any] = {"artifacts": {}}
    for name, content in files.items():
        path = root / name
        if isinstance(content, pd.DataFrame):
            content.to_parquet(path, index=False)
        else:
            path.write_text(json.dumps(content), encoding="utf-8")
        manifest["artifacts"][name] = {"sha256": digest_file(path)}
    if report is not None:
        path = root / "validation_feature_report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        manifest["report_sha256"] = digest_file(path)
    path = root / "output_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return digest_file(path)


def _pipeline_fixture(tmp_path: Path) -> Path:
    baseline = replay_history(_history(), _config())
    event = baseline.loc[baseline.event_id.eq("0-12")].iloc[0]
    catalog, evidence = _evidence()
    catalog = catalog.iloc[:1].copy()
    evidence = evidence[:1]
    old = event.actual_available_at.isoformat()
    new = (event.actual_available_at + pd.Timedelta(hours=4)).isoformat()
    catalog.loc[0, "accession_number"] = event.filing_accession
    catalog.loc[0, "accepted_at"] = old
    evidence[0].update(
        {
            "accession_number": event.filing_accession,
            "catalog_accepted_at": old,
            "raw_acceptance_literal": old,
            "literal_interpreted_as_labelled_utc": old,
            "header_accepted_at_utc": new,
            "index_accepted_at_utc": new,
            "reviewed_sec_acceptance_proxy_at": new,
        }
    )
    source = {}
    root = tmp_path / "catalog"
    source["catalog_root"] = str(root)
    source["catalog_manifest_sha256"] = _bundle(root, {"filing_catalog.parquet": catalog})
    root = tmp_path / "features"
    source["feature_root"] = str(root)
    source["feature_manifest_sha256"] = _bundle(
        root,
        {"quarterly_feature_history.parquet": baseline},
        {
            "catalog_manifest_sha256": source["catalog_manifest_sha256"],
            "config": asdict(_config()),
        },
    )
    root = tmp_path / "market"
    source["market_root"] = str(root)
    source["market_manifest_sha256"] = _bundle(
        root,
        {
            "validation_market_state_features.parquet": pd.DataFrame(
                [
                    {
                        "validation_event_id": "sample",
                        "cik": event.company_id,
                        "exchange_mic": "XNYS",
                        "market_feature_available_at": event.actual_available_at
                        - pd.Timedelta(days=1),
                        "future_return": 9999,
                    }
                ]
            )
        },
    )
    plan = {
        "config": source,
        "events": [
            {
                "validation_event_id": "sample",
                "cik": event.company_id,
                "ticker": event.ticker,
                "filing_accession": event.filing_accession,
                "accepted_at": old,
            }
        ],
    }
    root = tmp_path / "sample_plan"
    plan_hash = _bundle(root, {"plan.json": plan})
    provenance_plan = {"config": {"plan_root": str(root), "plan_manifest_sha256": plan_hash}}
    config = json.loads(Path("docs/sec_clock_layer_v1.json").read_bytes())
    root = tmp_path / "provenance_plan"
    config["provenance_plan_root"] = str(root)
    config["provenance_plan_manifest_sha256"] = _bundle(root, {"plan.json": provenance_plan})
    root = tmp_path / "provenance"
    config["provenance_root"] = str(root)
    config["provenance_manifest_sha256"] = _bundle(
        root,
        {
            "report.json": {"plan_manifest_sha256": config["provenance_plan_manifest_sha256"]},
            "source_clock_provenance.json": evidence,
            "proposed_acceptance_overlay.json": evidence,
        },
    )
    root = tmp_path / "calendar"
    config["calendar_root"] = str(root)
    config["calendar_manifest_sha256"] = _bundle(
        root, {"historical_exchange_sessions.parquet": _sessions(2024)}
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_full_pipeline_is_offline_immutable_disarmed_and_outcome_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _pipeline_fixture(tmp_path)
    original_read = pd.read_parquet
    reads = []

    def projected_read(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original_read(*args, **kwargs)
        reads.append(frame.columns.tolist())
        assert "future_return" not in frame
        return frame

    monkeypatch.setattr(pd, "read_parquet", projected_read)
    output, report = build_layer(path, tmp_path / "out")
    assert report["baseline_reproduced"] and report["fixed_sample_events"] == 1
    assert report["replay_difference_counts"]["forecast_training_observations"] > 0
    assert not report["model_admission"] and not report["return_values_loaded"]
    panel = pd.read_parquet(output / "disarmed_replay_panel.parquet")
    assert not panel.model_admission.any() and not panel.public_time_certified.any()
    assert len(panel) == 48 and reads
    second, report2 = build_layer(path, tmp_path / "out")
    assert second == output and report2 == report
    (output / "report.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_layer(path, tmp_path / "out")


def test_source_drift_blocks_corrected_replay(tmp_path: Path) -> None:
    path = _pipeline_fixture(tmp_path)
    table = tmp_path / "features" / "quarterly_feature_history.parquet"
    frame = pd.read_parquet(table)
    frame.loc[frame.gm_forecast.notna(), "gm_forecast"] += 0.1
    frame.to_parquet(table, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        build_layer(path, tmp_path / "out")


def test_future_actual_values_do_not_change_earlier_replay_forecasts() -> None:
    history = _history()
    before = replay_history(history, _config())
    mutated = history.copy()
    last = mutated.index[-1]
    mutated.loc[last, ["gm_level", "gross_margin"]] = 0.9
    after = replay_history(mutated, _config())
    earlier = before.actual_available_at < before.loc[last, "actual_available_at"]
    pd.testing.assert_frame_equal(
        before.loc[earlier, ["gm_forecast", "forecast_sigma"]],
        after.loc[earlier, ["gm_forecast", "forecast_sigma"]],
    )
