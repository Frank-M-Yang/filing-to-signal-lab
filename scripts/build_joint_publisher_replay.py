"""Replay USNA/UFPT together, audit error propagation, and retain the full cohort."""

from __future__ import annotations

import argparse
import io
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_publication_chronology import _content, _receipts
from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.event_admission import REPRESENTATIONS, build_admission_ledger
from sec_data.execution_contract import execution_window
from sec_data.joint_publisher_replay import (
    VERSION,
    apply_joint_clocks,
    assert_originals_preserved,
    compare_error_histories,
    consolidate_joint_admission,
    require_endpoint_invariance,
    validate_joint_intervals,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig
from sec_data.publication_chronology import FORECAST_CLOCKS, PANEL_COLUMNS, review_publisher_source
from sec_data.publisher_replay import (
    EVENT_COLUMNS,
    MARKET_COLUMNS,
    apply_publisher_clock,
    target_dependencies,
    three_arm_readiness,
    verify_baseline_projection,
)


def _load(config: dict[str, Any], group: str, name: str) -> Any:
    return _json(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name)


def _table(config: dict[str, Any], group: str, name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(
        _artifact(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name),
        columns=columns,
    )


def _pinned_config(path: str, digest: str) -> dict[str, Any]:
    verify_file(Path(path), digest)
    result: dict[str, Any] = json.loads(Path(path).read_bytes())
    return result


def _target(history: pd.DataFrame, request: dict[str, Any]) -> pd.DataFrame:
    rows = history.loc[
        history.company_id.eq(request["cik"])
        & history.filing_accession.eq(request["target_filing_accession"])
    ]
    if len(rows) != 1:
        raise ValueError("Expected one accession-bound target")
    return rows


def build_joint_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Run the fixed two-event, four-corner diagnostic with no outcomes or network."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or config["frozen_event_count"] != 316
        or config["numeric_absolute_tolerance"] != 1e-10
        or config["endpoint_policy"] != "all_four_corners_two_publisher_minutes"
        or len(config["cases"]) != 2
        or {c["source_id"] for c in config["cases"]} != {"usna_wirecopy", "ufpt_wirecopy"}
        or any(
            config[k] is not False
            for k in (
                "network_allowed",
                "forward_price_values_allowed",
                "returns_allowed",
                "model_admission",
                "trading_admission",
                "overwrite_originals",
                "sample_reselection_allowed",
            )
        )
    ):
        raise ValueError("Unsupported or weakened joint-replay contract")
    base = _pinned_config(config["base_replay_config"], config["base_replay_config_sha256"])
    chronology_report = _load(base, "chronology", "report.json")
    if chronology_report["plan_manifest_sha256"] != base["chronology_plan_manifest_sha256"]:
        raise ValueError("Chronology plan/output mismatch")
    chronology = _load(base, "chronology_plan", "plan.json")["config"]
    clock_plan = _load(base, "clock_plan", "plan.json")
    if (
        _load(chronology, "clock", "report.json")["plan_manifest_sha256"]
        != base["clock_plan_manifest_sha256"]
    ):
        raise ValueError("Clock plan/output mismatch")
    source = clock_plan["source"]
    cfg = PilotFilingFeatureConfig(**clock_plan["forecast_config"])
    feature_root = Path(source["feature_root"])
    verify_file(feature_root / "output_manifest.json", source["feature_manifest_sha256"])
    feature_manifest = json.loads((feature_root / "output_manifest.json").read_bytes())
    verify_file(feature_root / "validation_feature_report.json", feature_manifest["report_sha256"])
    if (
        json.loads((feature_root / "validation_feature_report.json").read_bytes())["config"]
        != clock_plan["forecast_config"]
    ):
        raise ValueError("Forecast specification changed")
    history = _table(source, "feature", "quarterly_feature_history.parquet", HISTORY_COLUMNS)
    requests = _load(base, "chronology", "disarmed_replay_requests.json")
    saved_reviews = _load(base, "chronology", "source_reviews.json")
    cells = _load(chronology, "followup", "metric_cell_review.json")
    receipts = _receipts(chronology)
    cases = []
    market_config: dict[str, Any] | None = None
    for pinned in config["cases"]:
        matches = [r for r in requests if r["source_id"] == pinned["source_id"]]
        specs = [s for s in chronology["reviews"] if s["source_id"] == pinned["source_id"]]
        if len(matches) != 1 or len(specs) != 1:
            raise ValueError("Missing unique joint source/request")
        request, spec = matches[0], specs[0]
        raw = [
            r
            for r in receipts
            if r.get("sha256") == spec["snapshot_sha256"]
            and r["status_code"] == 200
            and all(r.get(k) == spec[k] for k in ("cik", "ticker", "release_accession", "url"))
        ]
        if len(raw) != 1:
            raise ValueError("Missing unique successful publisher receipt")
        if raw[0].get("response_url") != spec["url"] or "text/html" not in raw[0].get(
            "content_type", ""
        ):
            raise ValueError("Publisher response URL or content type differs")
        reviewed = review_publisher_source(_content(raw[0]), spec, cells)
        saved = [r for r in saved_reviews if r["source_id"] == spec["source_id"]]
        if (
            len(saved) != 1
            or any(saved[0][k] != v for k, v in reviewed.items())
            or request["source_reported_at_utc"] != reviewed["source_reported_at_utc"]
            or request["validation_event_id"] != saved[0]["validation_event_id"]
        ):
            raise ValueError("Source review no longer reproduces")
        admission = _pinned_config(pinned["admission_config"], pinned["admission_config_sha256"])
        admission_report = _load(pinned, "admission", "summary.json")
        old_plan = _json(
            Path(admission_report["plan_root"]),
            admission_report["plan_manifest_sha256"],
            "plan.json",
        )
        if old_plan["config_sha256"] != pinned["admission_config_sha256"]:
            raise ValueError("Cached admission evidence belongs to another plan")
        market = _pinned_config(admission["market_config"], admission["market_config_sha256"])
        if market_config is None:
            market_config = market
        elif {k: v for k, v in market.items() if not k.startswith("publisher_")} != {
            k: v for k, v in market_config.items() if not k.startswith("publisher_")
        }:
            raise ValueError("Cases use different frozen market evidence")
        state_rows = _load(admission, "rebuild", "market_state.json")
        if len(state_rows) != 1:
            raise ValueError("Expected one rebuilt market state per case")
        cases.append(
            {
                "request": request,
                "spec": spec,
                "source_review": reviewed,
                "single_config": market,
                "market_state": state_rows[0],
                "event": _load(admission, "rebuild", "source_evidence.json")["event"],
                "liquidity": _load(pinned, "admission", "liquidity_audit.json"),
                "window": _load(admission, "rebuild", "window_evidence.json"),
            }
        )
    assert market_config is not None
    cases.sort(key=lambda c: utc_timestamp(c["request"]["source_reported_at_utc"]))
    events = _table(
        market_config,
        "execution",
        "validation_execution_events.parquet",
        [*EVENT_COLUMNS, "assigned_ticker"],
    ).rename(columns={"assigned_ticker": "ticker"})
    if len(events) != config["frozen_event_count"]:
        raise ValueError("Frozen cohort size changed")
    sessions = _table(
        market_config,
        "execution",
        "validation_exchange_sessions.parquet",
        [
            "exchange_mic",
            "session_date",
            "regular_open_at",
            "regular_close_at",
            "source_snapshot_sha256",
        ],
    )
    for case in cases:
        event, req = case["event"], case["request"]
        row = events.loc[events.validation_event_id.eq(req["validation_event_id"])]
        if (
            len(row) != 1
            or row.iloc[0].cik != req["cik"]
            or row.iloc[0].target_accession_number != req["target_filing_accession"]
            or event["validation_event_id"] != req["validation_event_id"]
        ):
            raise ValueError("Frozen execution/source identity differs")
        if utc_timestamp(event["announcement_at"]) != utc_timestamp(req["source_reported_at_utc"]):
            raise ValueError("Cached market evidence uses a different clock")
        expected = execution_window(req["source_reported_at_utc"], event["exchange_mic"], sessions)
        if expected != case["window"]["window"] or expected != req["publisher_label_window"]:
            raise ValueError("Cached window differs from publisher/execution contract")
        for name in ("model_admission", "return_admission", "first_public_certified"):
            if any(case[k][name] is not False for k in ("market_state", "window")):
                raise ValueError("Cached sensitivity must remain disarmed")
    implementations = {
        p: digest_file(Path(p))
        for p in (
            "scripts/build_joint_publisher_replay.py",
            "sec_data/joint_publisher_replay.py",
            "sec_data/clock_layer.py",
            "sec_data/publisher_replay.py",
            "sec_data/publication_chronology.py",
            "sec_data/event_admission.py",
            "sec_data/disclosure_audit.py",
            "sec_data/filing_text.py",
            "sec_data/pilot_filing_features.py",
            "sec_data/event_cleaning.py",
            "event_panel.py",
            "sec_data/execution_contract.py",
            "scripts/build_publication_chronology.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
        )
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "implementation_sha256": implementations,
        "forecast_config": clock_plan["forecast_config"],
        "requests": [c["request"] for c in cases],
        "source_reviews": [c["source_review"] for c in cases],
        "sample_reselected": False,
        "market_evidence_reused_not_refetched": True,
        "return_values_loaded": False,
    }
    frozen = write_bundle(output_root, "joint-publisher-plan-v1", {"plan.json": _encode(plan)})
    baseline = replay_history(
        apply_history_clocks(history, _load(chronology, "clock", "clock_layer.json")), cfg
    )
    verify_baseline_projection(
        baseline,
        _table(
            chronology,
            "clock",
            "disarmed_replay_panel.parquet",
            [
                *PANEL_COLUMNS,
                "gm_forecast",
                "forecast_sigma",
                "gm_innovation_raw",
                "gm_innovation_std",
            ],
        ),
    )
    validate_joint_intervals(baseline, cases)
    print(f"JOINT BASELINE reproduced=True history={len(baseline)}", flush=True)
    singles = {}
    for case in cases:
        req = case["request"]
        single = replay_history(
            apply_publisher_clock(
                baseline, req, case["spec"], utc_timestamp(req["source_reported_at_utc"])
            ),
            cfg,
        )
        verify_baseline_projection(
            single,
            _table(
                case["single_config"],
                "publisher",
                "minute_start_disarmed_history.parquet",
                HISTORY_COLUMNS,
            ),
        )
        singles[req["source_id"]] = single
        print(f"JOINT INDEPENDENT reproduced={req['source_id']}", flush=True)
    corner_reports: dict[str, Any] = {}
    outputs: dict[str, pd.DataFrame] = {}
    source_ids = [c["request"]["source_id"] for c in cases]
    for ends in product(("start", "end"), repeat=2):
        name = "_".join(ends)
        endpoints = dict(zip(source_ids, ends, strict=True))
        overlay = apply_joint_clocks(baseline, cases, endpoints)
        pd.testing.assert_frame_equal(overlay, apply_joint_clocks(baseline, cases[::-1], endpoints))
        joint = replay_history(overlay, cfg)
        assert_originals_preserved(baseline, joint)
        targets = {}
        for case in cases:
            req = case["request"]
            target = _target(joint, req)
            lower = utc_timestamp(req["source_reported_at_utc"])
            if any(pd.to_datetime(target[k], utc=True).ge(lower).any() for k in FORECAST_CLOCKS):
                raise ValueError(
                    "Joint target depends on information inside/after publisher minute"
                )
            target_dependencies(joint, str(target.event_id.iloc[0]))
            targets[req["source_id"]] = _records(target)[0]
        if outputs:
            require_endpoint_invariance(outputs["start_start"], joint)
        corner_reports[name] = {
            "endpoints": endpoints,
            "targets": targets,
            "chronology": audit_replay_chronology(joint),
            "changes_from_baseline": compare_histories(baseline, joint)[0],
        }
        outputs[name] = joint
        print(f"JOINT CORNER completed={name}", flush=True)
    joint = outputs["start_start"]
    # A later availability correction cannot alter any earlier reconstructed event.
    first_at = utc_timestamp(cases[0]["request"]["source_reported_at_utc"])
    later_at = utc_timestamp(cases[1]["request"]["source_reported_at_utc"])
    earlier = joint.actual_available_at.lt(first_at)
    if not compare_histories(baseline.loc[earlier], joint.loc[earlier])[1].empty:
        raise ValueError("Joint replay changed history before the first correction")
    before_later = joint.actual_available_at.lt(later_at)
    first_single = singles[source_ids[0]]
    if not compare_histories(first_single.loc[before_later], joint.loc[before_later])[1].empty:
        raise ValueError("Later correction changed earlier information")
    comparisons, dependencies = [], {}
    for case in cases:
        req = case["request"]
        event_id = str(_target(joint, req).event_id.iloc[0])
        counts, dependency = compare_error_histories(singles[req["source_id"]], joint, event_id)
        dependencies[req["source_id"]] = dependency
        comparisons.append(
            {
                "source_id": req["source_id"],
                "ticker": req["ticker"],
                "validation_event_id": req["validation_event_id"],
                "baseline": _records(_target(baseline, req))[0],
                "independent": _records(_target(singles[req["source_id"]], req))[0],
                "joint": _records(_target(joint, req))[0],
                "dependency_changes": counts,
                "target_changes_from_independent": compare_histories(
                    _target(singles[req["source_id"]], req), _target(joint, req)
                )[0],
            }
        )
    ledger = _joint_admission(
        events,
        joint,
        outputs["end_end"],
        cases,
        market_config,
        _load(base, "chronology", "event_chronology.json"),
    )
    if list(ledger.validation_event_id) != list(events.validation_event_id):
        raise ValueError("Joint ledger dropped/reordered frozen events")
    counts, differences = compare_histories(baseline, joint)
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "history_events": len(joint),
        "frozen_events_retained": len(ledger),
        "joint_label_scenario_inputs_complete": int(ledger.label_scenario_inputs_complete.sum()),
        "joint_metadata_coverage_not_admission": int(ledger.joint_metadata_coverage.sum()),
        "first_public_certified_events": 0,
        "admitted_events": 0,
        "changes_from_baseline": counts,
        "case_dependency_changes": [
            {"ticker": r["ticker"], **r["dependency_changes"]} for r in comparisons
        ],
        "all_four_corners_numerically_equal": True,
        "application_order_invariant": True,
        "later_correction_cannot_change_earlier_events": True,
        "baseline_and_both_independent_replays_reproduced": True,
        "market_evidence_reused_at_identical_clocks": True,
        "mbuu_clock_unchanged_and_unresolved": True,
        "remaining_unrebuilt_events": len(ledger)
        - int(ledger.label_scenario_inputs_complete.sum()),
        "blocker_counts_overlap": ledger.blockers.explode().value_counts().to_dict(),
        "sample_reselected": False,
        "hyperparameters_changed": False,
        "originals_overwritten": False,
        "network_requests": 0,
        "forward_price_values_loaded": False,
        "returns_computed": False,
        "first_public_certified": False,
        "model_admission": False,
        "trading_admission": False,
        "forecast_is_retrospective_reconstruction_not_live_log": True,
    }
    files = {
        "summary.json": _encode(summary),
        "case_comparisons.json": _encode(comparisons),
        "corner_checks.json": _encode(corner_reports),
        "admission_ledger.json": _encode(_records(ledger)),
        "joint_differences.json": _encode(_records(differences)),
    }
    frames = {
        "admission_ledger": ledger,
        **{f"{k}_dependencies": v for k, v in dependencies.items()},
    }
    for name, frame in outputs.items():
        frame = frame.copy()
        for field in (
            "model_admission",
            "return_admission",
            "trading_admission",
            "first_public_certified",
        ):
            frame[field] = False
        frame["clock_basis"] = "mixed_acceptance_proxies_with_two_joint_publisher_labels"
        frames[f"{name}_disarmed_history"] = frame
    for name, frame in frames.items():
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        files[f"{name}.parquet"] = buffer.getvalue()
    return write_bundle(output_root, VERSION, files), summary


def _joint_admission(
    events: pd.DataFrame,
    start: pd.DataFrame,
    end: pd.DataFrame,
    cases: list[dict[str, Any]],
    market_config: dict[str, Any],
    chronology: list[dict[str, Any]],
) -> pd.DataFrame:
    market = _table(
        market_config, "market", "validation_market_state_features.parquet", MARKET_COLUMNS
    )
    for case in cases:
        state = case["market_state"]
        match = market.validation_event_id.eq(state["validation_event_id"])
        if match.sum() != 1 or market.loc[match, "cik"].iloc[0] != state["cik"]:
            raise ValueError("Rebuilt market row is not in frozen cohort")
        market.loc[match, "market_feature_available_at"] = state["market_feature_available_at"]
        market.loc[match, "market_state_ready"] = state["market_state_ready"]
    readiness = three_arm_readiness(events[EVENT_COLUMNS], start, market, "no_single_target")
    for case in cases:
        match = readiness.validation_event_id.eq(case["request"]["validation_event_id"])
        readiness.loc[match, "clock_basis"] = "joint_publisher_minute_sensitivity"
        readiness.loc[match, "new_return_window_verified"] = case["window"]["window_evidence_ready"]
    gates = _table(
        market_config,
        "old_gate",
        "validation_return_gate_universe.parquet",
        [
            "validation_event_id",
            "cik",
            "liquidity_eligible",
            "liquidity_status",
            "return_gate_status",
        ],
    )
    readiness = readiness.merge(
        gates[["validation_event_id", "return_gate_status"]].rename(
            columns={"return_gate_status": "legacy_gate_status_not_new_authorization"}
        ),
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    financial = start[["company_id", "filing_accession", *REPRESENTATIONS]]
    case_ledgers = []
    for case in cases:
        req = case["request"]
        # This adapter validates both target projections from the JOINT history;
        # it does not reuse either independently reconstructed target prediction.
        joint_check = {
            "baseline_reproduced": True,
            "minute_endpoint_signals_equal": True,
            "hyperparameters_changed": False,
            "scenarios": {
                name: {"target": _records(_target(frame, req))[0]}
                for name, frame in (("minute_start", start), ("minute_end", end))
            },
        }
        case_ledgers.append(
            (
                req["validation_event_id"],
                build_admission_ledger(
                    readiness,
                    events,
                    gates,
                    financial,
                    chronology,
                    joint_check,
                    case["event"],
                    case["liquidity"],
                    case["market_state"],
                    case["window"],
                ),
            )
        )
    return consolidate_joint_admission(case_ledgers)


def main() -> None:
    """Build immutable local artifacts without generating a bilingual narrative report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/joint_publisher_replay_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/joint_publisher_replay")
    )
    args = parser.parse_args()
    root, result = build_joint_replay(args.config, args.output_root)
    print(
        f"JOINT PUBLISHER DONE retained={result['frozen_events_retained']} "
        f"joint_prepared={result['joint_label_scenario_inputs_complete']} "
        f"admitted=0 corners=4 output={root}"
    )


if __name__ == "__main__":
    main()
