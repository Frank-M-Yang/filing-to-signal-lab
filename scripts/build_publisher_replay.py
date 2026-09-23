"""Offline single-event publisher-minute replay and outcome-blind readiness."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_publication_chronology import _content, _receipts
from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.clock_layer import (
    FLAG_COLUMNS,
    HISTORY_COLUMNS,
    NUMERIC_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.pilot_filing_features import PilotFilingFeatureConfig
from sec_data.publication_chronology import review_publisher_source
from sec_data.publisher_replay import (
    EVENT_COLUMNS,
    MARKET_COLUMNS,
    VERSION,
    apply_publisher_clock,
    market_review_requests,
    target_dependencies,
    three_arm_readiness,
    verify_baseline_projection,
)


def build_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Reproduce the pinned baseline, replay both minute endpoints, and retain all events."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or config["numeric_absolute_tolerance"] != 1e-10
        or config.get("market_review_policy", "late_cutoffs")
        not in {"late_cutoffs", "selected_target"}
        or any(
            config[k] is not False
            for k in ("return_values_allowed", "model_admission", "overwrite_originals")
        )
    ):
        raise ValueError("Unsupported or weakened replay contract")

    def load(group: str, name: str) -> Any:
        return _json(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name)

    chronology_report = load("chronology", "report.json")
    if chronology_report["plan_manifest_sha256"] != config["chronology_plan_manifest_sha256"]:
        raise ValueError("Chronology plan/output mismatch")
    chronology = load("chronology_plan", "plan.json")["config"]
    requests = load("chronology", "disarmed_replay_requests.json")
    selected = [r for r in requests if r["source_id"] == config["source_id"]]
    if len(selected) != 1:
        raise ValueError("Expected one authorized publisher replay")
    request = selected[0]
    specs = [s for s in chronology["reviews"] if s["source_id"] == request["source_id"]]
    if len(specs) != 1:
        raise ValueError("Missing unique publisher review specification")
    spec = specs[0]
    cells = _json(
        Path(chronology["followup_root"]),
        chronology["followup_manifest_sha256"],
        "metric_cell_review.json",
    )
    receipts = [r for r in _receipts(chronology) if r.get("sha256") == spec["snapshot_sha256"]]
    if len(receipts) != 1:
        raise ValueError("Missing unique publisher snapshot receipt")
    reviewed = review_publisher_source(_content(receipts[0]), spec, cells)
    saved_reviews = load("chronology", "source_reviews.json")
    saved = [r for r in saved_reviews if r["source_id"] == request["source_id"]]
    if (
        len(saved) != 1
        or any(saved[0][k] != v for k, v in reviewed.items())
        or (
            request["source_reported_at_utc"] != reviewed["source_reported_at_utc"]
            or saved[0]["validation_event_id"] != request["validation_event_id"]
        )
    ):
        raise ValueError("Pinned source review no longer reproduces")
    clock_root = Path(chronology["clock_root"])
    clock_hash = chronology["clock_manifest_sha256"]
    clock_report = _json(clock_root, clock_hash, "report.json")
    if clock_report["plan_manifest_sha256"] != config["clock_plan_manifest_sha256"]:
        raise ValueError("Clock plan/output mismatch")
    clock_plan = load("clock_plan", "plan.json")
    source = clock_plan["source"]
    feature_root = Path(source["feature_root"])
    feature_hash = source["feature_manifest_sha256"]
    verify_file(feature_root / "output_manifest.json", feature_hash)
    manifest = json.loads((feature_root / "output_manifest.json").read_bytes())
    verify_file(feature_root / "validation_feature_report.json", manifest["report_sha256"])
    feature_report = json.loads((feature_root / "validation_feature_report.json").read_bytes())
    if feature_report["config"] != clock_plan["forecast_config"]:
        raise ValueError("Frozen forecast specification changed")
    cfg = PilotFilingFeatureConfig(**clock_plan["forecast_config"])
    history = pd.read_parquet(
        _artifact(feature_root, feature_hash, "quarterly_feature_history.parquet"),
        columns=HISTORY_COLUMNS,
    )
    layer = _json(clock_root, clock_hash, "clock_layer.json")
    implementations = {
        p: digest_file(Path(p))
        for p in [
            "scripts/build_publisher_replay.py",
            "sec_data/publisher_replay.py",
            "scripts/build_publication_chronology.py",
            "sec_data/publication_chronology.py",
            "scripts/build_sec_clock_layer.py",
            "sec_data/clock_layer.py",
            "event_panel.py",
            "sec_data/event_cleaning.py",
            "sec_data/pilot_filing_features.py",
            "sec_data/disclosure_audit.py",
            "sec_data/filing_text.py",
            "scripts/build_sec_disclosure_audit.py",
        ]
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "source": source,
        "forecast_config": clock_plan["forecast_config"],
        "implementation_sha256": implementations,
        "request": request,
        "spec": spec,
        "sample_reselected": False,
        "return_values_loaded": False,
    }
    frozen = write_bundle(output_root, "publisher-replay-plan-v1", {"plan.json": _encode(plan)})
    baseline = replay_history(apply_history_clocks(history, layer), cfg)
    verify_baseline_projection(
        baseline,
        pd.read_parquet(_artifact(clock_root, clock_hash, "disarmed_replay_panel.parquet")),
    )
    print(f"PUBLISHER BASELINE reproduced=True history={len(baseline)}", flush=True)
    lower = utc_timestamp(request["source_reported_at_utc"])
    # Endpoint sensitivity, not a claim of second-level clock precision.
    upper = lower + pd.Timedelta(minutes=1) - pd.Timedelta(microseconds=1)
    target_mask = baseline.company_id.eq(request["cik"]) & baseline.filing_accession.eq(
        request["target_filing_accession"]
    )
    if target_mask.sum() != 1:
        raise ValueError("Replay target identity missing")
    target_id = str(baseline.loc[target_mask, "event_id"].iloc[0])
    others = baseline.loc[~target_mask]
    intraminute = others.actual_available_at.ge(lower) & others.actual_available_at.le(upper)
    if intraminute.any():
        raise ValueError(
            "Other event inside uncertain publisher minute; interval-aware replay required"
        )
    results, differences, dependency_tables, scenario_reports = {}, {}, {}, {}
    old_dependencies = target_dependencies(baseline, target_id)
    old_ids = set(old_dependencies.loc[old_dependencies.in_training, "event_id"])
    for name, stamp in (("minute_start", lower), ("minute_end", upper)):
        revised = replay_history(apply_publisher_clock(baseline, request, spec, stamp), cfg)
        counts, diff = compare_histories(baseline, revised)
        earlier = baseline.actual_available_at.lt(lower)
        _, earlier_diff = compare_histories(baseline.loc[earlier], revised.loc[earlier])
        if not earlier_diff.empty:
            raise ValueError("Publisher replay changed an earlier event")
        target = revised.loc[revised.event_id.eq(target_id)]
        if any(
            pd.to_datetime(target[field], utc=True).ge(lower).any()
            for field in (
                "forecast_feature_cutoff",
                "forecast_training_end",
                "forecast_generated_at",
                "forecast_error_history_end",
            )
        ):
            raise ValueError("Target forecast uses information from inside/after uncertain minute")
        deps = target_dependencies(revised, target_id)
        new_ids = set(deps.loc[deps.in_training, "event_id"])
        scenario_reports[name] = {
            "timestamp": stamp.isoformat(),
            "difference_counts": counts,
            "chronology": audit_replay_chronology(revised),
            "training_removed": sorted(old_ids - new_ids),
            "training_added": sorted(new_ids - old_ids),
            "earlier_events_unchanged": int(earlier.sum()),
            "target": _records(target)[0],
        }
        results[name], differences[name], dependency_tables[name] = revised, diff, deps
        print(
            f"PUBLISHER REPLAY scenario={name} forecast_changes={counts['gm_forecast']} "
            f"training_removed={len(old_ids - new_ids)}",
            flush=True,
        )
    interval_counts, interval_diff = compare_histories(
        results["minute_start"], results["minute_end"]
    )
    invariant_fields = (*NUMERIC_COLUMNS, *FLAG_COLUMNS, "gm_yoy_lag_event_id")
    if any(interval_counts[k] for k in invariant_fields):
        raise ValueError("Minute endpoints change signal values; do not collapse uncertainty")
    execution_root = Path(chronology["calendar_root"])
    events = pd.read_parquet(
        _artifact(
            execution_root,
            chronology["calendar_manifest_sha256"],
            "validation_execution_events.parquet",
        ),
        columns=EVENT_COLUMNS,
    )
    target_event = events.loc[events.validation_event_id.eq(request["validation_event_id"])]
    if (
        len(target_event) != 1
        or target_event.iloc[0].cik != request["cik"]
        or (target_event.iloc[0].target_accession_number != request["target_filing_accession"])
    ):
        raise ValueError("Execution/replay identity mismatch")
    market = pd.read_parquet(
        _artifact(
            Path(source["market_root"]),
            source["market_manifest_sha256"],
            "validation_market_state_features.parquet",
        ),
        columns=MARKET_COLUMNS,
    )
    readiness = three_arm_readiness(
        events, results["minute_start"], market, request["validation_event_id"]
    )
    readiness_summary = {
        "frozen_events": len(events),
        "issuers": int(events.cik.nunique()),
        "market_rows": len(market),
        "financial_representation_coverage": int(readiness.financial_representation_coverage.sum()),
        "market_metadata_coverage": int(readiness.market_metadata_coverage.sum()),
        "joint_metadata_coverage": int(readiness.joint_metadata_coverage.sum()),
        "cutoff_status_counts": readiness.market_cutoff_status.value_counts().to_dict(),
        "admitted_events": 0,
        "return_values_loaded": False,
    }
    report = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "baseline_reproduced": True,
        "history_events": len(history),
        "target_event_id": target_id,
        "original_target": _records(baseline.loc[target_mask])[0],
        "scenarios": scenario_reports,
        "minute_endpoint_difference_counts": interval_counts,
        "minute_endpoint_signals_equal": True,
        "minute_endpoints_are_sensitivity_not_first_public_bounds": True,
        "readiness": readiness_summary,
        "originals_overwritten": False,
        "ytd_clocks_changed": False,
        "hyperparameters_changed": False,
        "market_features_rebuilt": False,
        "return_values_loaded": False,
        "first_public_certified": False,
        "model_admission": False,
        "trading_admission": False,
        "forecast_is_retrospective_reconstruction_not_live_log": True,
        "windows": {
            k: request[k] for k in ("same_contract_filing_proxy_window", "publisher_label_window")
        },
    }
    files = {
        "report.json": _encode(report),
        "three_arm_readiness.json": _encode(_records(readiness)),
        "market_rebuild_requests.json": _encode(
            _records(
                market_review_requests(
                    readiness,
                    request["validation_event_id"],
                    config.get("market_review_policy", "late_cutoffs"),
                )
            )
        ),
        "source_review.json": _encode(reviewed),
        "baseline_target_dependencies.json": _encode(_records(old_dependencies)),
        "minute_endpoint_differences.json": _encode(_records(interval_diff)),
    }
    if config.get("write_readout", True):
        files["readout.md"] = render_readout(report).encode()
    for name, revised in results.items():
        revised = revised.copy()
        for field in (
            "model_admission",
            "return_admission",
            "trading_admission",
            "first_public_certified",
        ):
            revised[field] = False
        revised["clock_basis"] = "mixed_acceptance_proxies_with_one_publisher_label_sensitivity"
        buffer = io.BytesIO()
        revised.to_parquet(buffer, index=False)
        files[f"{name}_disarmed_history.parquet"] = buffer.getvalue()
        files[f"{name}_differences.json"] = _encode(_records(differences[name]))
        files[f"{name}_target_dependencies.json"] = _encode(_records(dependency_tables[name]))
    return write_bundle(output_root, VERSION, files), report


def render_readout(report: dict[str, Any]) -> str:
    """Summarize forecast sensitivity without a performance or first-public claim."""
    old = report["original_target"]
    new = report["scenarios"]["minute_start"]["target"]
    readiness = report["readiness"]
    return "\n".join(
        [
            "# Publisher-time sensitivity / 发布时点敏感性重放",
            "",
            "## English",
            "",
            f"Pinned baseline reproduced: {report['history_events']} history events; "
            "no parameter changes.",
            "One exact-quarter publisher label moves availability earlier; "
            "SEC acceptance and YTD clocks are unchanged.",
            "Both minute endpoints give the same signal values within absolute tolerance 1e-10.",
            "",
            "|UFPT quantity|Filing proxy|Publisher-label scenario|",
            "|---|---:|---:|",
            f"|Training events|{old['forecast_training_observations']}|"
            f"{new['forecast_training_observations']}|",
            f"|Forecast gross margin|{old['gm_forecast']:.6%}|{new['gm_forecast']:.6%}|",
            f"|Raw innovation (percentage points)|{old['gm_innovation_raw'] * 100:.6f}|"
            f"{new['gm_innovation_raw'] * 100:.6f}|",
            f"|Standardized innovation|{old['gm_innovation_std']:.6f}|"
            f"{new['gm_innovation_std']:.6f}|",
            "",
            f"All {readiness['frozen_events']} execution events remain in the readiness ledger; "
            f"{readiness['joint_metadata_coverage']} have joint metadata coverage, "
            "not study admission.",
            "Publisher labels are not certified first-public times. "
            "Forecasts are retrospective reconstructions, not live logs.",
            "No return values read, no returns recomputed, "
            "no market features rebuilt, no trading/model admission.",
            "",
            "## 中文",
            "",
            "这次只检验信息时点如何影响预测，没有检验赚钱能力。",
            f"旧训练集剔除{len(report['scenarios']['minute_start']['training_removed'])}条记录；"
            "同一模型的后续预测及历史误差也必须重算。",
            "市场特征不能沿用11月6日行情来解释11月3日的信息冲击。财报原值、SEC接收时间与YTD来源时间均保留。",
            "下一步先重建事件前市场状态、核对公开时点及新版收益窗口，再冻结同样本三组对照；不增添政客交易特征。",
            "",
        ]
    )


def main() -> None:
    """Run locally without credentials, network requests or return-value access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/publisher_replay_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/publisher_replay"))
    args = parser.parse_args()
    root, report = build_replay(args.config, args.output_root)
    print(
        f"PUBLISHER REPLAY DONE baseline=True history={report['history_events']} "
        f"retained={report['readiness']['frozen_events']} admitted=0 output={root}"
    )


if __name__ == "__main__":
    main()
