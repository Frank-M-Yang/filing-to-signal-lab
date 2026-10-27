"""Replay frozen financial cases at reviewed 8-K acceptance proxies, offline."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_announcement_admission import _config, _recheck_cells
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_replay import (
    STRESS_VERSION,
    VERSION,
    apply_release_proxies,
    case_row,
    company_dependencies,
    compare_case_signals,
    select_replay_cases,
    verify_prefix_causality,
    verify_target_reconstruction,
)
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.joint_publisher_replay import assert_originals_preserved, compare_error_histories
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig
from sec_data.publisher_replay import verify_baseline_projection


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=15)
    )
    return result


def build_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Replay a fixed evidence-selected category and preserve every source cohort identity."""
    c = json.loads(config_path.read_bytes())
    stress = c["version"] == STRESS_VERSION
    expected_policy = "reviewed_values_clock_failures" if stress else "ready_cases"
    if (
        c["version"] not in {VERSION, STRESS_VERSION}
        or c.get("case_selection_policy", "ready_cases") != expected_policy
        or c["frozen_event_count"] != 316
        or c["numeric_absolute_tolerance"] != 1e-10
        or c["fixed_specification_refit_allowed"] is not True
        or len(c["case_event_ids"]) != (3 if stress else 2)
        or len(set(c["case_event_ids"])) != len(c["case_event_ids"])
        or any(
            c[k] is not False
            for k in (
                "hyperparameter_search_allowed",
                "market_features_in_forecast",
                "network_allowed",
                "post_event_prices_allowed",
                "returns_allowed",
                "first_public_certified",
                "trading_admission",
                "sample_replacement_allowed",
                "overwrite_originals",
            )
        )
    ):
        raise ValueError("Unsupported or weakened announcement replay contract")

    def load(group: str, name: str) -> Any:
        return _json(Path(c[f"{group}_root"]), c[f"{group}_manifest_sha256"], name)

    admission_summary = load("admission", "summary.json")
    admission_plan = load("admission_plan", "plan.json")
    if admission_summary["plan_manifest_sha256"] != c["admission_plan_manifest_sha256"]:
        raise ValueError("Admission plan/output mismatch")
    cohort = load("admission", "event_admission_ledger.json")
    if len(cohort) != c["frozen_event_count"] or len(
        {r["validation_event_id"] for r in cohort}
    ) != len(cohort):
        raise ValueError("Frozen cohort changed")
    cases = select_replay_cases(cohort, c["case_event_ids"], expected_policy)
    review = _config(
        admission_plan["config"]["review_config"], admission_plan["config"]["review_config_sha256"]
    )
    cells, _, source = _recheck_cells(review)
    if cells != load("admission", "rechecked_metric_cells.json"):
        raise ValueError("Raw source table reviews no longer reproduce")
    clock_plan = load("clock_plan", "plan.json")
    clock_report = load("clock", "report.json")
    if clock_report["plan_manifest_sha256"] != c["clock_plan_manifest_sha256"]:
        raise ValueError("Clock plan/output mismatch")
    if source != clock_plan["source"] or source != admission_plan["financial_sources"]:
        raise ValueError("Clock, admission and financial sources differ")
    source_root, source_hash = Path(source["feature_root"]), source["feature_manifest_sha256"]
    verify_file(source_root / "output_manifest.json", source_hash)
    manifest = json.loads((source_root / "output_manifest.json").read_bytes())
    verify_file(source_root / "validation_feature_report.json", manifest["report_sha256"])
    feature_report = json.loads((source_root / "validation_feature_report.json").read_bytes())
    if feature_report["config"] != clock_plan["forecast_config"]:
        raise ValueError("Fixed model specification changed")
    cfg = PilotFilingFeatureConfig(**clock_plan["forecast_config"])
    cfg.validate()
    history_path = _artifact(source_root, source_hash, "quarterly_feature_history.parquet")
    history = pd.read_parquet(history_path, columns=HISTORY_COLUMNS)
    layer = load("clock", "clock_layer.json")
    # These states were rebuilt only from the 61 exact pre-proxy sessions. They are
    # context, NOT extra regressors in the fixed company-history OLS experiment.
    market_states = load("admission", "candidate_market_states.json")
    market_context = []
    for case in cases:
        state = [
            r for r in market_states if r["validation_event_id"] == case["validation_event_id"]
        ]
        if not state and stress:
            market_context.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "ticker": case["assigned_ticker"],
                    "market_context_status": "missing_pre_proxy_history",
                    "remaining_upstream_gaps": case["blocking_reasons"],
                    "market_features_used_as_regressors": False,
                }
            )
            continue
        if len(state) != 1 or utc_timestamp(state[0]["announcement_at"]) != utc_timestamp(
            case["earliest_disclosure_at"]
        ):
            raise ValueError("Pre-event market context is not bound to the case clock")
        market_context.append(state[0])
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": cases,
        "financial_sources": source,
        "forecast_config": clock_plan["forecast_config"],
        "forecast_features": list(FORECAST_FEATURES),
        "history_events": len(history),
        "market_features_used_as_regressors": False,
        "industry_features_used": False,
        "arms": [
            "legacy",
            "reconciled",
            *[f"only_{r['assigned_ticker']}" for r in cases],
            *[f"prefix_{size}" for size in range(2, len(cases))],
            "joint",
        ],
        "first_public_certified": False,
        "return_values_loaded": False,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/replay_announcement_cases.py",
                "sec_data/announcement_replay.py",
                "scripts/audit_announcement_admission.py",
                "sec_data/announcement_admission.py",
                "sec_data/disclosure_review.py",
                "sec_data/clock_layer.py",
                "sec_data/publisher_replay.py",
                "sec_data/joint_publisher_replay.py",
                "sec_data/pilot_filing_features.py",
                "sec_data/event_cleaning.py",
                "event_panel.py",
            )
        },
    }
    frozen = write_bundle(
        output_root / "plans", c["version"] + "-plan", {"plan.json": _encode(plan)}
    )
    print(f"REPLAY PLAN cases={len(cases)} history={len(history)} output={frozen}", flush=True)
    histories = {"legacy": replay_history(history, cfg)}
    verify_baseline_projection(histories["legacy"], history)
    print("REPLAY legacy reproduced=True", flush=True)
    baseline = replay_history(apply_history_clocks(history, layer), cfg)
    saved_projection = pd.read_parquet(
        _artifact(
            Path(c["clock_root"]), c["clock_manifest_sha256"], "disarmed_replay_panel.parquet"
        )
    )
    verify_baseline_projection(baseline, saved_projection)
    histories["reconciled"] = baseline
    print("REPLAY reconciled reproduced=True", flush=True)
    for case in cases:
        name = f"only_{case['assigned_ticker']}"
        histories[name] = replay_history(apply_release_proxies(baseline, [case], cells), cfg)
        print(f"REPLAY completed={name}", flush=True)
    prefixes = {1: histories[f"only_{cases[0]['assigned_ticker']}"]}
    for size in range(2, len(cases)):
        name = f"prefix_{size}"
        histories[name] = replay_history(apply_release_proxies(baseline, cases[:size], cells), cfg)
        prefixes[size] = histories[name]
        print(f"REPLAY completed={name}", flush=True)
    joint_input = apply_release_proxies(baseline, cases, cells)
    pd.testing.assert_frame_equal(joint_input, apply_release_proxies(baseline, cases[::-1], cells))
    histories["joint"] = replay_history(joint_input, cfg)
    causal_counts = verify_prefix_causality(baseline, histories["joint"], cases, prefixes)
    outputs: dict[str, bytes] = {
        "rechecked_metric_cells.json": _encode(cells),
        "pre_event_market_context.json": _encode(market_context),
    }
    arm_reports = {}
    for name, frame in histories.items():
        if name != "legacy":
            assert_originals_preserved(baseline, frame)
        target_reports = []
        for case in cases:
            target = case_row(frame, case)
            deps = verify_target_reconstruction(frame, case)
            company = company_dependencies(frame, target, cfg)
            ticker = case["assigned_ticker"]
            outputs[f"{name}_{ticker}_training_and_errors.json"] = _encode(_rows(deps))
            outputs[f"{name}_{ticker}_company_dependencies.json"] = _encode(_rows(company))
            target_reports.append(
                {
                    **_rows(pd.DataFrame([target]))[0],
                    "validation_event_id": case["validation_event_id"],
                    "assigned_ticker": case["assigned_ticker"],
                }
            )
        arm_reports[name] = {
            "chronology": audit_replay_chronology(frame),
            "targets": target_reports,
            "change_counts_from_reconciled": compare_histories(baseline, frame)[0],
        }
    comparisons = compare_case_signals(histories, cases)
    dependency_changes: dict[str, Any] = {}
    for case in cases:
        event_id = str(case_row(baseline, case).event_id)
        dependency_changes[case["assigned_ticker"]] = {}
        for name in ("legacy", "reconciled", f"only_{case['assigned_ticker']}"):
            counts, diff = compare_error_histories(histories[name], histories["joint"], event_id)
            dependency_changes[case["assigned_ticker"]][name] = counts
            outputs[f"{name}_to_joint_{case['assigned_ticker']}_dependency_changes.json"] = _encode(
                _rows(diff)
            )
    # Preserve all 316 cohort identities alongside the selected-case experiment.
    for row in cohort:
        selected = row["validation_event_id"] in c["case_event_ids"]
        row["case_replay_completed"] = selected
        row["replay_clock_basis"] = (
            "reviewed_8k_acceptance_proxy" if selected else "not_case_replayed"
        )
        row["model_admission"] = row["return_admission"] = row["first_public_certified"] = False
    report = {
        "version": c["version"],
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_events_retained": len(cohort),
        "history_events": len(history),
        "case_tickers": [r["assigned_ticker"] for r in cases],
        "same_case_sample": True,
        "legacy_baseline_reproduced": True,
        "reconciled_baseline_reproduced": True,
        "arms": arm_reports,
        "dependency_changes": dependency_changes,
        "signal_comparison": _rows(comparisons),
        "hyperparameters_changed": False,
        "market_features_used_as_regressors": False,
        "network_requests": 0,
        "post_event_prices_loaded": False,
        "returns_computed": False,
        "first_public_certified": False,
        "trading_admission": False,
        "model_admission": False,
        "forecast_is_retrospective_reconstruction_not_live_log": True,
        "earlier_events_unchanged": causal_counts["before_first_correction"],
        "causal_prefix_checks": causal_counts,
        "financial_only_case_tickers": [
            r["assigned_ticker"] for r in cases if not r["market_history_evidence_ready"]
        ],
        "originals_overwritten": False,
        "limitations": [
            "Only selected quarter availability clocks move; other historical events retain "
            "their existing SEC acceptance proxies.",
            "The forecast generation field is a reconstructed dependency boundary, "
            "not a contemporaneous saved prediction.",
            "SEC acceptance is not certified first-public time; "
            "this is a conditional clock-sensitivity experiment.",
            "The cases were selected from a frozen evidence category, "
            "not by return performance or as a representative sample.",
            "Company-history OLS uses six frozen inputs; "
            "no industry or market regressor is added in this replay.",
            "No outcome test, alpha claim or trading authorization follows from this replay; "
            "market gaps and first-public certification requirements remain.",
        ],
    }
    outputs.update(
        {
            "summary.json": _encode(report),
            "signal_comparison.json": _encode(_rows(comparisons)),
            "cohort_replay_status.json": _encode(cohort),
        }
    )
    frame = histories["joint"].copy()
    for flag in (
        "model_admission",
        "trading_admission",
        "return_admission",
        "first_public_certified",
    ):
        frame[flag] = False
    frame["clock_basis"] = f"mixed_acceptance_proxies_with_{len(cases)}_reviewed_release_proxies"
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    outputs["joint_disarmed_history.parquet"] = buffer.getvalue()
    root = write_bundle(output_root, c["version"], outputs)
    return root, report


def main() -> None:
    """Run the frozen offline case experiment without downloads or performance evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/announcement_replay_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_replay")
    )
    args = parser.parse_args()
    root, report = build_replay(args.config, args.output_root)
    print(
        f"ANNOUNCEMENT REPLAY DONE cases={','.join(report['case_tickers'])} "
        f"history={report['history_events']} returns=False output={root}"
    )


if __name__ == "__main__":
    main()
