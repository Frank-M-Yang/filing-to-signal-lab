"""Fixed-specification joint interval replay; no price values, returns or network access."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_next_publication_audit import build_audit
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.review_next_disclosures import review_sources
from sec_data.announcement_replay import SIGNALS, case_row
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    audit_replay_chronology,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.joint_publisher_replay import compare_error_histories
from sec_data.next_disclosure_replay import verify_prefix_outputs
from sec_data.next_publication import DISARM_FLAGS
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig
from sec_data.publication_interval_replay import (
    COMPARISONS,
    VERSION,
    audit_interval_cases,
    build_ordering_grid,
    interval_signal_summary,
    overlay_intervals,
    require_stable_endpoint_equivalence,
)


def validate_contract(c: dict[str, Any]) -> None:
    """Reject outcomes, model search, weaker sigma thresholds and sample replacement."""
    if (
        c.get("version") != VERSION
        or c.get("frozen_history_events") != 3254
        or c.get("frozen_queue_issuers") != 20
        or c.get("frozen_pair_cases") != 6
        or c.get("scope_quarantined_records") != 792
        or c.get("max_joint_scenarios") != 32
        or c.get("minimum_forecast_training_events") != 100
        or c.get("minimum_prior_oos_errors") != 50
        or c.get("fixed_specification_refit_allowed") is not True
        or any(
            c.get(k) is not False
            for k in (
                "network_allowed",
                "price_values_allowed",
                "returns_allowed",
                "market_features_in_forecast",
                "hyperparameter_search_allowed",
                "threshold_relaxation_allowed",
                "sample_replacement_allowed",
                "overwrite_originals",
                *DISARM_FLAGS,
            )
        )
    ):
        raise ValueError("Unsupported or weakened publication interval replay contract")


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _error_comparison(
    independent: pd.DataFrame, joint: pd.DataFrame, case: dict[str, Any]
) -> dict[str, Any]:
    counts, changes = compare_error_histories(
        independent, joint, str(case_row(joint, case).event_id)
    )
    return {
        "ticker": case["ticker"],
        "validation_event_id": case["validation_event_id"],
        **counts,
        "shared_prior_error_events_changed": changes.loc[
            changes.shared_prior_error_changed, "event_id"
        ].tolist(),
        "first_public_certified": False,
        "return_admission": False,
    }


def build_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Rebuild one shared history per arm, including arrival ties inside label intervals."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    publication_summary = load("publication", "summary.json")
    publication_plan = load("publication_plan", "plan.json")
    if publication_summary["plan_manifest_sha256"] != c["publication_plan_manifest_sha256"]:
        raise ValueError("Publication summary/plan mismatch")
    verify_file(Path(c["publication_config"]), publication_plan["config_sha256"])
    rechecked, _ = build_audit(Path(c["publication_config"]), Path(c["publication_root"]).parent)
    if digest_file(rechecked / "output_manifest.json") != c["publication_manifest_sha256"]:
        raise ValueError("Publisher bytes, exact original cells or clock checks do not reproduce")
    old = load("replay_plan", "plan.json")
    if (
        old["forecast_features"] != list(FORECAST_FEATURES)
        or publication_plan["config"]["replay_plan_manifest_sha256"]
        != c["replay_plan_manifest_sha256"]
        or publication_plan["config"]["replay_manifest_sha256"] != c["replay_manifest_sha256"]
    ):
        raise ValueError("Frozen financial forecast or replay source chain changed")
    review_cfg = old["config"]
    review = _json(
        Path(review_cfg["review_root"]), review_cfg["review_manifest_sha256"], "summary.json"
    )
    verify_file(Path(review_cfg["review_config"]), review["review_config_sha256"])
    sec_rechecked, _ = review_sources(
        Path(review["prepared_root"]),
        review["prepared_manifest_sha256"],
        Path(review_cfg["review_config"]),
        output_root / "sec_source_rechecks",
    )
    if digest_file(sec_rechecked / "output_manifest.json") != review_cfg["review_manifest_sha256"]:
        raise ValueError("Exact original SEC snapshot/cell/timestamp review changed")
    cases = old["cases"]
    sources = load("publication", "source_reviews.json")
    dispositions = load("publication", "case_dispositions.json")
    if (
        len(dispositions) != 20
        or [r["validation_event_id"] for r in dispositions] != old["retained_disposition_ids"]
        or any(r.get(k) is not False for r in dispositions for k in DISARM_FLAGS)
    ):
        raise ValueError("Twenty-issuer frozen disposition ledger changed")
    cfg = PilotFilingFeatureConfig(**old["forecast_config"])
    cfg.validate()
    if cfg.minimum_forecast_training_events != 100 or cfg.minimum_prior_oos_errors != 50:
        raise ValueError("Frozen forecast/sigma thresholds changed")
    history_path = _artifact(
        Path(c["replay_root"]), c["replay_manifest_sha256"], "shared_disarmed_history.parquet"
    )
    columns = list(
        dict.fromkeys([*HISTORY_COLUMNS, "scope_quarantined", "scope_arm", "derivation_method"])
    )
    history = pd.read_parquet(history_path, columns=columns)
    if (
        len(history) != 3254
        or int(history.scope_quarantined.sum()) != 792
        or history.scope_arm.ne("SUN_and_all_cost_derived_quarantined").any()
    ):
        raise ValueError("Frozen history sample or conservative accounting-scope mask changed")
    scenarios, crossings = build_ordering_grid(history, cases, sources)
    implementations = {
        p: digest_file(Path(p))
        for p in (
            "scripts/replay_publication_intervals.py",
            "sec_data/publication_interval_replay.py",
            "sec_data/announcement_replay.py",
            "sec_data/next_disclosure_replay.py",
            "sec_data/clock_layer.py",
            "sec_data/pilot_filing_features.py",
            "event_panel.py",
            "sec_data/event_cleaning.py",
            "sec_data/publisher_replay.py",
            "sec_data/joint_publisher_replay.py",
        )
    }
    frozen = write_bundle(
        output_root / "plans",
        VERSION + "-plan",
        {
            "plan.json": _encode(
                {
                    "config": c,
                    "config_sha256": digest_file(config_path),
                    "cases": cases,
                    "sources": sources,
                    "scenarios": scenarios,
                    "crossing_proxy_arrivals": crossings,
                    "implementation_sha256": implementations,
                    "forecast_config": old["forecast_config"],
                    "forecast_features": list(FORECAST_FEATURES),
                    "history_sha256": digest_file(history_path),
                    "retained_disposition_ids": old["retained_disposition_ids"],
                    "selection_uses_outcomes": False,
                    "historical_information_arrivals_complete": False,
                }
            )
        },
    )
    print(
        f"INTERVAL PLAN scenarios={len(scenarios)} crossings={len(crossings)} output={frozen}",
        flush=True,
    )
    baseline = replay_history(history, cfg)
    if not compare_histories(history, baseline)[1].empty:
        raise ValueError("Saved complete-pair proxy baseline no longer reproduces")
    rows, dependency_frames, company_frames, arm_audits = [], [], [], []
    joint_lower: pd.DataFrame | None = None
    equivalent: pd.DataFrame | None = None
    selected_id = scenarios[-1]["equivalent_numeric_scenario"]
    for index, scenario in enumerate(scenarios):
        print(f"INTERVAL REPLAY {index + 1}/{len(scenarios)} {scenario['scenario_id']}", flush=True)
        overlaid = overlay_intervals(baseline, cases, sources, scenario["cutoffs"])
        pd.testing.assert_frame_equal(
            overlaid, overlay_intervals(baseline, cases[::-1], sources, scenario["cutoffs"])
        )
        replayed = replay_history(overlaid, cfg)
        audit, deps, own = audit_interval_cases(baseline, replayed, cases, scenario, cfg)
        if scenario["scenario_id"] == scenarios[0]["scenario_id"]:
            joint_lower = replayed
        if scenario["scenario_id"] == selected_id:
            equivalent = replayed
        if scenario["scenario_id"] == "all_interval_ends":
            if equivalent is None:
                raise ValueError("Missing stable-endpoint comparison arm")
            require_stable_endpoint_equivalence(equivalent, replayed)
        rows.extend(audit)
        dependency_frames.append(deps)
        company_frames.append(own)
        counts, _ = compare_histories(baseline, replayed)
        arm_audits.append(
            {
                "scenario_id": scenario["scenario_id"],
                "chronology": audit_replay_chronology(replayed),
                "changed_fields": counts,
                "same_six_common_gate": all(r["same_six_common_financial_gate"] for r in audit),
            }
        )
    if joint_lower is None:
        raise ValueError("Missing joint lower sensitivity arm")
    ranges = interval_signal_summary(rows)
    lower = scenarios[0]["cutoffs"]
    sorted_cases = sorted(cases, key=lambda case: utc_timestamp(lower[case["ticker"]]))
    causal = [
        {
            "before": lower[sorted_cases[0]["ticker"]],
            "unchanged_rows": verify_prefix_outputs(
                baseline, joint_lower, lower[sorted_cases[0]["ticker"]]
            ),
        }
    ]
    for size in range(1, 6):
        print(f"INTERVAL CAUSAL PREFIX {size}/5", flush=True)
        tickers = {case["ticker"] for case in sorted_cases[:size]}
        prefix = replay_history(
            overlay_intervals(baseline, cases, sources, lower, selected_tickers=tickers), cfg
        )
        cutoff = lower[sorted_cases[size]["ticker"]]
        causal.append(
            {"before": cutoff, "unchanged_rows": verify_prefix_outputs(prefix, joint_lower, cutoff)}
        )
    # Fixed own-only counterfactuals isolate shared-history propagation, not a model tournament.
    propagation, own_comparisons = [], []
    for case in cases:
        print(f"INTERVAL PROPAGATION {case['ticker']}", flush=True)
        own_history = replay_history(
            overlay_intervals(baseline, cases, sources, lower, selected_tickers={case["ticker"]}),
            cfg,
        )
        propagation.append(_error_comparison(own_history, joint_lower, case))
        own_row, joint_row = case_row(own_history, case), case_row(joint_lower, case)
        for field in COMPARISONS:
            a, b = own_row[field], joint_row[field]
            own_comparisons.append(
                {
                    "ticker": case["ticker"],
                    "representation": field,
                    "own_clock_only": None if pd.isna(a) else float(a),
                    "joint_shared_history": None if pd.isna(b) else float(b),
                    "joint_minus_own_only": None if pd.isna(a) or pd.isna(b) else float(b - a),
                    "changed": None if pd.isna(a) or pd.isna(b) else bool(abs(b - a) > 1e-10),
                    "unit": "prior_OOS_sigma_units"
                    if field == "gm_innovation_std"
                    else "margin_fraction",
                }
            )
    # Crossed proxy event membership is retained, including exact-time ties.
    deps_all = pd.concat(dependency_frames, ignore_index=True)
    boundary_membership = []
    for crossing in crossings:
        for scenario in scenarios:
            match = deps_all.loc[
                deps_all.scenario_id.eq(scenario["scenario_id"])
                & deps_all.target_ticker.eq(crossing["target_ticker"])
                & deps_all.event_id.eq(crossing["event_id"])
            ]
            boundary_membership.append(
                {
                    **crossing,
                    "scenario_id": scenario["scenario_id"],
                    "target_at": scenario["cutoffs"][crossing["target_ticker"]],
                    "in_training": bool(not match.empty and match.in_training.any()),
                    "in_prior_oos_errors": bool(
                        not match.empty and match.in_prior_oos_errors.any()
                    ),
                }
            )
    ledger = [
        {
            **d,
            "publication_interval_scenarios_replayed": len(scenarios)
            if d["validation_event_id"] in {case["validation_event_id"] for case in cases}
            else 0,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        for d in dispositions
    ]
    result = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "history_events_retained": len(history),
        "scope_quarantined_records": 792,
        "frozen_queue_issuers_retained": len(ledger),
        "pair_cases": len(cases),
        "joint_interval_scenarios": len(scenarios),
        "crossing_proxy_arrivals": len(crossings),
        "crossing_target_tickers": sorted({r["target_ticker"] for r in crossings}),
        "common_financial_gate_all_scenarios": all(
            r["same_six_common_financial_gate"] for r in rows
        ),
        "causal_prefix_checks": len(causal),
        "stable_endpoint_equivalence_passed": True,
        "numerically_interval_sensitive_tickers": sorted(
            {
                r["ticker"]
                for r in ranges
                if r["representation"] in SIGNALS and r["changed_across_clock_scenarios"] is True
            }
        ),
        "innovation_direction_changes_within_uncertainty": sorted(
            {
                r["ticker"]
                for r in ranges
                if r["representation"] in ("gm_innovation_raw", "gm_innovation_std")
                and r["direction_changes_within_uncertainty"] is True
            }
        ),
        "max_raw_innovation_clock_range_pp": max(
            (
                r["range"] * 100
                for r in ranges
                if r["representation"] == "gm_innovation_raw" and r["range"] is not None
            ),
            default=None,
        ),
        "max_standardized_innovation_clock_range": max(
            (
                r["range"]
                for r in ranges
                if r["representation"] == "gm_innovation_std" and r["range"] is not None
            ),
            default=None,
        ),
        "shared_history_propagation_tickers": sorted(
            {
                r["ticker"]
                for r in own_comparisons
                if r["representation"] in SIGNALS and r["changed"] is True
            }
        ),
        "all_chronology_violations": sum(a["chronology"]["violations"] for a in arm_audits),
        "forecasts_reconstructed_not_live": True,
        "forecast_specification": "fixed_pooled_OLS_six_company_history_features",
        "return_models_fitted": 0,
        "return_values_read": False,
        "price_values_read": False,
        "historical_information_arrivals_complete": False,
        "network_requests": 0,
        "keys_read": False,
        "old_artifacts_overwritten": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    files = {
        "summary.json": _encode(result),
        "case_dispositions.json": _encode(ledger),
        "scenario_case_audit.json": _encode(rows),
        "interval_signal_ranges.json": _encode(ranges),
        "shared_history_propagation.json": _encode(propagation),
        "joint_vs_own_only.json": _encode(own_comparisons),
        "boundary_dependency_membership.json": _encode(boundary_membership),
        "scenario_history_audit.json": _encode(arm_audits),
        "causal_prefix_audit.json": _encode(causal),
        "forecast_dependencies.parquet": _parquet(deps_all),
        "company_feature_dependencies.parquet": _parquet(
            pd.concat(company_frames, ignore_index=True)
        ),
        "joint_lower_disarmed_history.parquet": _parquet(
            joint_lower.assign(**dict.fromkeys(DISARM_FLAGS, False))
        ),
        "readout.txt": render_readout(result, ranges).encode("utf-8"),
    }
    return write_bundle(output_root, VERSION, files), result


def render_readout(summary: dict[str, Any], ranges: list[dict[str, Any]]) -> str:
    """Explain sensitivity, strict-prior error propagation, and the absence of an Alpha test."""
    lines = [
        "发布时间精度与共享信息集重放（非收益实验）",
        "",
        "保留20家公司、6个原始财务完整案例、3254条财务历史记录；792条口径可疑记录继续隔离。",
        f"固定同一个6特征OLS规格，重放{summary['joint_interval_scenarios']}个时间情景。",
        "分别检查早于其他信息、精确同时、紧接其他信息及区间末端；同一时刻的事件不得互相训练。",
        "RAMP是明确时区的1分钟标签敏感性；NOBH是未知时区的±14小时情景，不是认证的历史公开时间上下界。",
        "",
        "公司 | 原始Innovation区间（百分点） | 标准化Innovation区间（历史误差sigma单位）",
    ]
    for ticker in sorted({r["ticker"] for r in ranges}):
        raw = next(
            r
            for r in ranges
            if r["ticker"] == ticker and r["representation"] == "gm_innovation_raw"
        )
        std = next(
            r
            for r in ranges
            if r["ticker"] == ticker and r["representation"] == "gm_innovation_std"
        )
        if raw["minimum"] is None or std["minimum"] is None:
            lines.append(f"{ticker} | 共同财务样本不完整，不删除案例 | 不报告可比区间")
        else:
            lines.append(
                f"{ticker} | {raw['minimum'] * 100:.6f} ~ {raw['maximum'] * 100:.6f} | "
                f"{std['minimum']:.6f} ~ {std['maximum']:.6f}"
            )
    lines += [
        "",
        "这些区间只是固定代理历史中的时间顺序敏感性，不是置信区间或收益预测区间。",
        "单独重放各家公司后直接拼接，会丢掉它们对其他公司样本外预测误差历史的影响；因此必须重放共享历史。",
        "财务预期模型不是股票收益模型。没有读价格或收益，没有选最赚钱的时间、换样本或调超参数。",
        "本轮只量化时钟不确定性与信息集传播；真实首次公开认证、完整公告覆盖及收益准入仍未建立。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Reproduce from immutable local disclosure caches with no credentials or requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/publication_interval_replay_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publication_interval_replay")
    )
    args = parser.parse_args()
    root, summary = build_replay(args.config, args.output_root)
    print(f"INTERVAL REPLAY DONE summary={json.dumps(summary)} output={root}")


if __name__ == "__main__":
    main()
