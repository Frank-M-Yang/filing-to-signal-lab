"""Replay fourteen reviewed announcements and audit their cached, proxy-clock windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_announcement_admission import _config
from scripts.build_announcement_return_diagnostic import _one, _parquet, _rows
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.review_announcement_expansion import review_batch
from sec_data.announcement_expanded_returns import (
    VERSION,
    all_case_return_gate,
    freeze_reviewed_expansion,
    prior_industry_summary,
    reauthorize_cached_continuity,
    supplementation_plan,
    validate_contract,
)
from sec_data.announcement_replay import apply_release_proxies, case_row
from sec_data.announcement_returns import (
    audit_return_window,
    calculate_case_return,
    normalize_holding_actions,
)
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    compare_histories,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.margin_scope_replay import (
    ARMS,
    replay_scope_view,
    scope_case_signals,
    scope_quarantine_view,
    scope_rank_diagnostics,
)
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig
from sec_data.publisher_replay import target_dependencies


def _market_metadata(
    market: dict[str, Any],
    continuity_config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    """Read identities, query reports and price-date projections, never numeric OHLC."""

    def table(group: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            _artifact(
                Path(market[group + "_root"]),
                market[group + "_manifest_sha256"],
                name,
            ),
            columns=columns,
        )

    tables = {
        "targets": table("prices", "validation_price_batch_targets.parquet"),
        "price_audit": table("prices", "validation_price_batch_target_audit.parquet"),
        "actions": table("actions", "sample_security_actions_audit.parquet"),
        "identities": table("actions", "security_reference_interval_observations.parquet"),
        "ticker_events": table("actions", "ticker_change_observations.parquet"),
        "listing": table("listing", "validation_security_listing_outcomes.parquet"),
        "sessions": table("execution", "validation_exchange_sessions.parquet"),
        "queries": table("benchmark", "massive_benchmark_snapshots.parquet"),
        "metadata": table(
            "prices",
            "validation_price_batch_bars.parquet",
            ["price_target_id", "ticker", "session_date", "source_sha256"],
        ),
        "benchmark_metadata": table(
            "benchmark",
            "benchmark_daily_price_observations.parquet",
            ["ticker", "session_date", "source_sha256"],
        ),
    }
    cc = continuity_config
    tables["continuity"] = pd.read_parquet(
        _artifact(
            Path(cc["continuity_root"]),
            cc["continuity_manifest_sha256"],
            "validation_action_evidence_status.parquet",
        )
    )
    cm = json.loads((Path(cc["continuity_root"]) / "output_manifest.json").read_bytes())
    if any(
        cm[k] != market[g + "_manifest_sha256"]
        for k, g in (
            ("execution_plan_manifest_sha256", "execution"),
            ("price_audit_manifest_sha256", "prices"),
            ("security_actions_manifest_sha256", "actions"),
        )
    ):
        raise ValueError("Cached continuity belongs to another source chain")
    root = Path(market["benchmark_root"])
    verify_file(root / "output_manifest.json", market["benchmark_manifest_sha256"])
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    report_path = root / "massive_benchmark_report.json"
    verify_file(report_path, manifest["report_sha256"])
    report = json.loads(report_path.read_bytes())
    if tables["queries"].benchmark_target_id.nunique() != 1:
        raise ValueError("Ambiguous benchmark identity")
    benchmark_id = str(tables["queries"].benchmark_target_id.iloc[0])
    events = []
    for case in cases:
        tid = case["price_target_id"]
        target = _one(tables["targets"].loc[tables["targets"].price_target_id.eq(tid)])
        action = _one(tables["actions"].loc[tables["actions"].price_target_id.eq(tid)])
        if (
            target["adjusted"] is not False
            or target["selection_uses_return_values"] is not False
            or (
                any(
                    target[k] != case[v]
                    for k, v in (
                        ("cik", "cik"),
                        ("selected_ticker", "assigned_ticker"),
                        ("selected_exchange_mic", "assigned_exchange_mic"),
                    )
                )
            )
        ):
            raise ValueError("Frozen outcome-blind raw-price identity changed")
        events.append(
            {
                "validation_event_id": case["validation_event_id"],
                "cik": case["cik"],
                "ticker": case["assigned_ticker"],
                "exchange_mic": case["assigned_exchange_mic"],
                "price_target_id": tid,
                "benchmark_target_id": benchmark_id,
                "security_id": target["security_identity_key"],
                "security_id_kind": target["security_identity_key_kind"],
                "action_target_id": action["action_target_id"],
                "announcement_at": case["earliest_disclosure_at"],
            }
        )
    return tables, report, events


def _audit_windows(
    tables: dict[str, pd.DataFrame],
    benchmark_report: dict[str, Any],
    events: list[dict[str, Any]],
    continuity: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Retain every case and every missing check under the unchanged twenty-session audit."""
    windows, audits = [], {}
    for event in events:
        tid, aid = event["price_target_id"], event["action_target_id"]
        outcome = tables["listing"]
        audit, dates = audit_return_window(
            event,
            tables["sessions"],
            tables["metadata"]
            .loc[tables["metadata"].price_target_id.eq(tid)]
            .drop(columns="price_target_id"),
            tables["benchmark_metadata"],
            _one(continuity.loc[continuity.action_target_id.eq(aid)]),
            _one(tables["actions"].loc[tables["actions"].action_target_id.eq(aid)]),
            _one(tables["price_audit"].loc[tables["price_audit"].price_target_id.eq(tid)]),
            _one(
                outcome.loc[
                    outcome.cik.eq(event["cik"]) & outcome.security_id.eq(event["security_id"])
                ]
            ),
            tables["queries"],
            benchmark_report,
        )
        windows.append(audit)
        audits[event["ticker"]] = dates
    return windows, audits


def _complete_returns(
    market: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    events: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Only read numeric prices after ALL fourteen metadata windows and signals are frozen."""
    if not all_case_return_gate(windows, [e["validation_event_id"] for e in events])[
        "all_case_return_gate"
    ]:
        raise ValueError("No numeric price read permitted for a partial frozen sample")

    def path(group: str, name: str) -> Path:
        return _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], name)

    splits = pd.read_parquet(path("actions", "split_observations.parquet"))
    dividends = pd.read_parquet(path("actions", "dividend_observations.parquet"))
    bs = pd.read_parquet(path("benchmark", "benchmark_split_observations.parquet"))
    bd = pd.read_parquet(path("benchmark", "benchmark_dividend_observations.parquet"))
    results, cashflows = [], []
    for event, audit in zip(events, windows, strict=True):
        w = audit["window"]
        filters = [
            ("session_date", ">=", w["entry_session_date"]),
            ("session_date", "<=", w["exit_session_date"]),
        ]
        columns = ["ticker", "session_date", "open_unadjusted", "close_unadjusted", "source_sha256"]
        issuer = pd.read_parquet(
            path("prices", "validation_price_batch_bars.parquet"),
            columns=["price_target_id", *columns],
            filters=[("price_target_id", "==", event["price_target_id"]), *filters],
        )
        benchmark = pd.read_parquet(
            path("benchmark", "benchmark_daily_price_observations.parquet"),
            columns=["benchmark_target_id", *columns],
            filters=[("benchmark_target_id", "==", event["benchmark_target_id"]), *filters],
        )
        stock_actions = normalize_holding_actions(
            splits.loc[splits.action_target_id.eq(event["action_target_id"])],
            dividends.loc[dividends.action_target_id.eq(event["action_target_id"])],
            tables["sessions"],
            w,
            event["ticker"],
        )
        benchmark_actions = normalize_holding_actions(bs, bd, tables["sessions"], w, "SPY")
        result, flows = calculate_case_return(
            event,
            audit,
            issuer,
            benchmark,
            stock_actions,
            benchmark_actions,
        )
        results.append(result)
        cashflows.extend(flows)
    return pd.DataFrame(results), cashflows


def build_expanded_returns(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze selection/masks, recheck sources, refit OOS signals and publish complete gaps."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    expansion = load("expansion_plan", "plan.json")
    source_plan = expansion["source_plan"]
    source = source_plan["financial_sources"]
    review = load("review", "summary.json")
    scope = load("scope", "summary.json")
    if review["plan_manifest_sha256"] != c["expansion_plan_manifest_sha256"] or (
        not scope["baseline_IC_reproduced"] or scope["general_scope_gate_cleared"] is not False
    ):
        raise ValueError("Source review or conservative scope audit identity changed")
    cases = freeze_reviewed_expansion(
        load("review", "financial_pair_candidates.json"),
        expansion["selected_cases"],
        expansion["controls"],
        load("review", "new_case_dispositions.json"),
        expansion["all_cohort_event_ids"],
        c["case_event_ids"],
    )
    market = source_plan["market_config"]
    tables, benchmark_report, events = _market_metadata(
        market,
        source_plan["continuity_config"],
        cases,
    )
    old_windows, _ = _audit_windows(tables, benchmark_report, events, tables["continuity"])
    replay_config = _config(
        expansion["base_config"]["source_replay_config"],
        expansion["base_config"]["source_replay_config_sha256"],
    )
    clock_plan = _json(
        Path(replay_config["clock_plan_root"]),
        replay_config["clock_plan_manifest_sha256"],
        "plan.json",
    )
    if clock_plan["source"] != source or clock_plan["forecast_config"] != scope["forecast_config"]:
        raise ValueError("Fixed model or financial sources differ from audited scope baseline")
    cfg = PilotFilingFeatureConfig(**clock_plan["forecast_config"])
    cfg.validate()
    original = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        )
    )
    if len(original) != 3254 or original.event_id.duplicated().any():
        raise ValueError("Original financial history denominator or identity changed")
    history = original[HISTORY_COLUMNS].copy()
    history["derivation_method"] = original.derivation_method
    history["gross_profit_fact_id"] = original.gross_profit_fact_id
    history = apply_history_clocks(
        history,
        _json(
            Path(replay_config["clock_root"]),
            replay_config["clock_manifest_sha256"],
            "clock_layer.json",
        ),
    )
    masks = {a: scope_quarantine_view(history, a, c["issuer_cik"]) for a in ARMS}
    excluded = {a: f.loc[f.scope_quarantined, "event_id"].tolist() for a, f in masks.items()}
    if {a: len(ids) for a, ids in excluded.items()} != scope["excluded_margin_records_by_arm"]:
        raise ValueError("Conservative accounting-scope masks changed")
    implementation = {
        p: digest_file(Path(p))
        for p in (
            "scripts/build_announcement_expanded_returns.py",
            "sec_data/announcement_expanded_returns.py",
            "sec_data/announcement_returns.py",
            "sec_data/margin_scope_replay.py",
            "sec_data/announcement_replay.py",
            "sec_data/validation_action_continuity.py",
            "sec_data/pilot_filing_features.py",
            "sec_data/clock_layer.py",
            "sec_data/execution_contract.py",
            "scripts/review_announcement_expansion.py",
            "scripts/build_announcement_return_diagnostic.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
            "scripts/audit_announcement_admission.py",
            "sec_data/disclosure_review.py",
            "sec_data/event_returns.py",
            "sec_data/event_cleaning.py",
            "sec_data/publisher_replay.py",
            "event_panel.py",
        )
    }
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": cases,
        "all_cohort_event_ids": expansion["all_cohort_event_ids"],
        "all_twenty_new_case_dispositions": load("review", "new_case_dispositions.json"),
        "market_config": market,
        "continuity_config": source_plan["continuity_config"],
        "financial_sources": source,
        "excluded_event_ids": excluded,
        "forecast_config": clock_plan["forecast_config"],
        "forecast_features": list(FORECAST_FEATURES),
        "initial_window_audit": old_windows,
        "requested_execution_events": events,
        "implementation_sha256": implementation,
        "forward_numeric_prices_read": False,
        "selection_uses_return_values": False,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "first_public_certified": False,
        "trading_admission": False,
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    print(
        f"EXPANDED PLAN cases=14 "
        f"initial_ready={sum(w['conditional_return_diagnostic_ready'] for w in old_windows)} "
        f"output={frozen}",
        flush=True,
    )
    rebound, reuse_audit = reauthorize_cached_continuity(
        events,
        old_windows,
        tables["actions"],
        tables["identities"],
        tables["ticker_events"],
        tables["metadata"],
        tables["sessions"],
    )
    continuity = tables["continuity"].loc[
        ~tables["continuity"].action_target_id.isin(rebound.get("action_target_id", []))
    ]
    continuity = pd.concat([continuity, rebound], ignore_index=True)
    windows, session_audits = _audit_windows(tables, benchmark_report, events, continuity)
    gate = all_case_return_gate(windows, c["case_event_ids"])
    print(
        f"EXPANDED WINDOWS ready={gate['metadata_ready_events']}/14 "
        f"blocked={gate['blocked_tickers']}",
        flush=True,
    )
    # Re-execute the original raw-byte reviews, rather than trusting numbers in an export.
    _config(c["review_config"], c["review_config_sha256"])
    rechecked_root, _ = review_batch(Path(c["review_config"]), output_root / "source_rechecks")
    if digest_file(rechecked_root / "output_manifest.json") != c["review_manifest_sha256"]:
        raise ValueError("Fourteen original-value source checks no longer reproduce")
    cells = load("review", "all_old_and_new_cells.json")
    overlaid = apply_release_proxies(history, cases, cells)
    pd.testing.assert_frame_equal(overlaid, apply_release_proxies(history, cases[::-1], cells))
    histories = {}
    for arm in ARMS:
        print(f"EXPANDED FORECAST start={arm}", flush=True)
        histories[arm] = replay_scope_view(overlaid, arm, c["issuer_cik"], cfg)
    signals = scope_case_signals(histories, cases)
    dependencies: list[dict[str, Any]] = []
    for arm, frame in histories.items():
        for case in cases:
            deps = target_dependencies(frame, str(case_row(frame, case).event_id))
            if set(deps.event_id) & set(excluded[arm]):
                raise ValueError("Quarantined label or OOS error entered target dependencies")
            dependencies.extend(
                {"arm": arm, "ticker": case["assigned_ticker"], **r} for r in _rows(deps)
            )
    baseline = expansion["base_config"]
    old_summary = _json(
        Path(baseline["admission_root"]), baseline["admission_manifest_sha256"], "summary.json"
    )
    if old_summary["events_retained"] != 316:
        raise ValueError("Original admission denominator changed")
    # The previously frozen six-case replay must remain unchanged before the first NEW overlay.
    old_history = pd.read_parquet(
        _artifact(
            Path(c["baseline_root"]),
            c["baseline_manifest_sha256"],
            "joint_history.parquet",
        )
    )
    controls = {r["validation_event_id"] for r in expansion["controls"]}
    cutoff = min(
        utc_timestamp(r["earliest_disclosure_at"])
        for r in cases
        if r["validation_event_id"] not in controls
    )
    joint = histories[ARMS[0]]
    if not compare_histories(
        old_history.loc[old_history.actual_available_at.lt(cutoff)],
        joint.loc[joint.actual_available_at.lt(cutoff)],
    )[1].empty:
        raise ValueError("Later announcement overlays changed earlier history")
    freeze = write_bundle(
        output_root / "plans",
        VERSION + "-signal-freeze",
        {
            "signal_panel.json": _encode(_rows(signals)),
            "contract.json": _encode(
                {
                    "selection_plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
                    "same_fourteen_cases_complete_in_all_arms": True,
                    "returns_read": False,
                    "primary_arm": ARMS[2],
                    "forecast_config": clock_plan["forecast_config"],
                }
            ),
        },
    )
    print(
        f"EXPANDED SIGNALS rows={len(signals)} cases=14 all_arms_complete=True output={freeze}",
        flush=True,
    )
    industry = prior_industry_summary(cases, load("review", "prior_industry.json"))
    files = {
        "signal_panel.json": _encode(_rows(signals)),
        "window_evidence.json": _encode(windows),
        "initial_window_evidence.json": _encode(old_windows),
        "cached_identity_reuse_audit.json": _encode(reuse_audit),
        "conditional_identity_continuity.parquet": _parquet(continuity),
        "supplementation_plan.json": _encode(supplementation_plan(events, windows, session_audits)),
        "forecast_dependency_audit.json": _encode(dependencies),
        "prior_industry_summary.json": _encode(industry),
        "source_evidence.json": _encode(
            {
                "market_config": market,
                "continuity_config": source_plan["continuity_config"],
                "source_review_manifest_sha256": c["review_manifest_sha256"],
                "source_scope_manifest_sha256": c["scope_manifest_sha256"],
            }
        ),
        **{a + "_history.parquet": _parquet(f) for a, f in histories.items()},
        **{
            ticker + "_session_audit.parquet": _parquet(frame)
            for ticker, frame in session_audits.items()
        },
    }
    if gate["all_case_return_gate"]:
        returns, flows = _complete_returns(market, tables, events, windows)
        sample, stats, influence = scope_rank_diagnostics(signals, returns)
        files.update(
            {
                "case_returns.json": _encode(_rows(returns)),
                "cashflow_evidence.json": _encode(flows),
                "same_case_analysis.json": _encode(_rows(sample)),
                "descriptive_ic.json": _encode(_rows(stats)),
                "leave_one_issuer_out.json": _encode(_rows(influence)),
            }
        )
    summary = {
        "version": VERSION,
        **gate,
        "signal_events_per_arm": 14,
        "signal_rows": len(signals),
        "frozen_cohort_events_retained": 316,
        "frozen_new_selections_retained": 20,
        "source_blocked_new_cases_retained": 12,
        "financial_history_events_retained": len(history),
        "primary_arm": ARMS[2],
        "case_tickers": [r["assigned_ticker"] for r in cases],
        "excluded_margin_records_by_arm": {a: len(ids) for a, ids in excluded.items()},
        "chronology_by_arm": {a: audit_replay_chronology(f) for a, f in histories.items()},
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "signal_freeze_manifest_sha256": digest_file(freeze / "output_manifest.json"),
        "raw_original_source_reviews_reproduced": True,
        "history_before_first_new_overlay_unchanged": True,
        "conditional_returns_computed": gate["all_case_return_gate"],
        "descriptive_ic_computed": gate["all_case_return_gate"],
        "forward_numeric_prices_read": gate["all_case_return_gate"],
        "industry_adjusted_ic_computed": False,
        "minimum_industry_events_unchanged": 20,
        "general_scope_gate_cleared": False,
        "policy_basis": c["policy_basis"],
        "retrospective_replayed_forecasts_not_live": True,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "first_public_certified": False,
        "trading_admission": False,
        "inferential_tests_computed": False,
        "strategy_pnl_computed": False,
        "hyperparameters_retuned": False,
        "network_requests": 0,
        "originals_overwritten": False,
        "verdict": "fourteen_case_pipeline_verified_effectiveness_not_established"
        if gate["all_case_return_gate"]
        else "fourteen_signals_frozen_market_supplement_required",
        "limitations": [
            "hand_reviewed_nonrepresentative_sample",
            "SEC_acceptance_not_first_public",
            "remaining_historical_scope_and_clock_uncertainty",
            "no_costs_or_sector_risk_controls",
            "historical_price_delivery_times_not_certified",
            "quarantine_not_a_known_live_filter",
        ],
    }
    cohort = _json(
        Path(baseline["admission_root"]),
        baseline["admission_manifest_sha256"],
        "event_admission_ledger.json",
    )
    for row in cohort:
        row["selected_for_expanded_conditional_diagnostic"] = (
            row["validation_event_id"] in c["case_event_ids"]
        )
        row["return_admission"] = row["trading_admission"] = False
    files["cohort_status.json"] = _encode(cohort)
    files["summary.json"] = _encode(summary)
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Run offline; blocked market cases produce a saved plan rather than a smaller IC sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/announcement_expanded_returns_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_expanded_returns")
    )
    args = parser.parse_args()
    root, report = build_expanded_returns(args.config, args.output_root)
    print(
        f"EXPANDED RETURNS DONE signals=14 market_ready={report['metadata_ready_events']}/14 "
        f"gate={report['all_case_return_gate']} verdict={report['verdict']} output={root}"
    )


if __name__ == "__main__":
    main()
