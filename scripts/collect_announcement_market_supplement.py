"""Collect the two frozen gaps, reuse frozen signals, then audit the identical fourteen events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_announcement_expanded_returns import _audit_windows, _market_metadata
from scripts.build_announcement_return_diagnostic import _one, _parquet, _rows
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_expanded_returns import (
    all_case_return_gate,
    reauthorize_cached_continuity,
)
from sec_data.announcement_market_supplement import (
    VERSION,
    collection_targets,
    reconcile_overlap,
    replace_target_rows,
    validate_contract,
    verify_control_returns,
    write_typed_bundle,
)
from sec_data.announcement_returns import calculate_case_return, normalize_holding_actions
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.margin_scope_replay import ARMS, scope_rank_diagnostics
from sec_data.massive_prices import (
    collect_massive_price_artifacts,
    load_collected_massive_daily_bars,
)
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.massive_security_actions import (
    build_massive_security_actions_audit,
    collect_massive_security_actions_artifacts,
    collect_massive_security_session_fallbacks,
)


def prepare_supplement(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze two source-bound windows and the optional forty-session identity grid BEFORE HTTP."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    source_plan = load("source_plan", "plan.json")
    summary = load("source", "summary.json")
    contract = load("signal_freeze", "contract.json")
    if (
        summary["plan_manifest_sha256"] != c["source_plan_manifest_sha256"]
        or summary["signal_freeze_manifest_sha256"] != c["signal_freeze_manifest_sha256"]
        or contract["selection_plan_manifest_sha256"] != c["source_plan_manifest_sha256"]
        or contract["same_fourteen_cases_complete_in_all_arms"] is not True
        or contract["returns_read"] is not False
    ):
        raise ValueError("Source cohort and pre-collection signal freeze disagree")
    signals = load("source", "signal_panel.json")
    if signals != load("signal_freeze", "signal_panel.json") or len(signals) != 42:
        raise ValueError("Do not change the previously frozen fourteen-case, three-arm forecasts")
    for path, digest in source_plan["implementation_sha256"].items():
        verify_file(Path(path), digest)
    supplement = load("source", "supplementation_plan.json")
    jobs = supplement["jobs"]
    market = source_plan["market_config"]
    tables, _, _ = _market_metadata(market, source_plan["continuity_config"], source_plan["cases"])
    prices, actions = collection_targets(
        jobs, load("source", "window_evidence.json"), tables["targets"], tables["actions"]
    )
    calendar = write_typed_bundle(
        output_root / "calendars",
        VERSION + "-calendar",
        {
            "historical_exchange_sessions.parquet": _parquet(tables["sessions"]),
            "exchange_calendar_report.json": _encode(
                {
                    "dataset_version": VERSION + "-calendar",
                    "source_execution_manifest_sha256": market["execution_manifest_sha256"],
                }
            ),
        },
        report_name="exchange_calendar_report.json",
    )
    old_actions_manifest = json.loads(
        (Path(market["actions_root"]) / "output_manifest.json").read_bytes()
    )
    old_report_path = Path(market["actions_root"]) / "massive_security_actions_report.json"
    verify_file(old_report_path, old_actions_manifest["report_sha256"])
    old_report = json.loads(old_report_path.read_bytes())
    common = {
        "reference_version": old_report["reference_version"],
        "reference_manifest_sha256": old_report["reference_manifest_sha256"],
    }
    price_plan = write_typed_bundle(
        output_root / "prices",
        VERSION + "-prices",
        {
            "massive_daily_price_targets.parquet": _parquet(prices),
            "target_report.json": _encode({**common, "dataset_version": VERSION + "-prices"}),
        },
        report_name="target_report.json",
    )
    action_plan = write_typed_bundle(
        output_root / "actions",
        VERSION + "-actions",
        {
            "massive_security_actions_targets.parquet": _parquet(actions),
            "target_report.json": _encode(
                {
                    **common,
                    "dataset_version": VERSION + "-actions",
                    "price_version": VERSION + "-prices",
                    "price_manifest_sha256": digest_file(price_plan / "output_manifest.json"),
                    "calendar_version": VERSION + "-calendar",
                    "calendar_manifest_sha256": digest_file(calendar / "output_manifest.json"),
                }
            ),
        },
        report_name="target_report.json",
    )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "jobs": jobs,
        "source_selection": source_plan,
        "signal_count": 42,
        "price_plan_root": price_plan.as_posix(),
        "action_plan_root": action_plan.as_posix(),
        "calendar_root": calendar.as_posix(),
        "child_manifest_sha256": {
            "prices": digest_file(price_plan / "output_manifest.json"),
            "actions": digest_file(action_plan / "output_manifest.json"),
            "calendar": digest_file(calendar / "output_manifest.json"),
        },
        "base_queries": 12,
        "maximum_session_identity_fallback_queries": 40,
        "signals_refitted": False,
        "same_fourteen_cases_retained": True,
        "network_requests_made": 0,
        "return_values_computed": False,
        "raw_store_scoped_to_request_and_signal_contract": True,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/collect_announcement_market_supplement.py",
                "sec_data/announcement_market_supplement.py",
                "sec_data/massive_prices.py",
                "sec_data/massive_security_actions.py",
                "sec_data/massive_reference.py",
            )
        },
    }
    # Keep source bytes reusable after a code-only repair. Never share completion
    # directories across different windows, target tables, or frozen signal sets.
    plan["raw_store_scope_sha256"] = hashlib.sha256(
        _encode(
            {
                "child_manifest_sha256": plan["child_manifest_sha256"],
                "signal_freeze_manifest_sha256": c["signal_freeze_manifest_sha256"],
            }
        )
    ).hexdigest()
    root = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    return root, plan


def finish_diagnostic(
    plan_root: Path,
    plan: dict[str, Any],
    price_root: Path,
    action_root: Path,
    output_root: Path,
    collection: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Audit every identical window before any return calculation; never select a ready subset."""
    c, source = plan["config"], plan["source_selection"]
    sr, sh = Path(c["source_root"]), c["source_manifest_sha256"]
    market = source["market_config"]
    tables, benchmark_report, events = _market_metadata(
        market, source["continuity_config"], source["cases"]
    )
    ph, ah = (
        digest_file(price_root / "output_manifest.json"),
        digest_file(action_root / "output_manifest.json"),
    )

    def fresh(root: Path, digest: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(_artifact(root, digest, name), columns=columns)

    new_prices = fresh(price_root, ph, "sample_massive_price_audit.parquet")
    new_actions = fresh(action_root, ah, "sample_security_actions_audit.parquet")
    new_metadata = fresh(
        price_root,
        ph,
        "massive_daily_price_observations.parquet",
        ["price_target_id", "ticker", "session_date", "source_sha256"],
    )
    # Generic audit column names are adapted only after their source hashes are pinned.
    price_audits = tables["price_audit"].copy()
    for _, row in new_prices.iterrows():
        target = _one(new_actions.loc[new_actions.price_target_id.eq(row.price_target_id)])
        mask = price_audits.price_target_id.eq(row.price_target_id)
        price_audits.loc[mask, "provider_query_complete"] = bool(row.provider_price_query_complete)
        price_audits.loc[mask, "parser_rejection_rows"] = int(row.rejected_bar_count)
        price_audits.loc[mask, "price_window_start"] = target["price_window_start"]
        price_audits.loc[mask, "price_window_end"] = target["price_window_end"]
    tables["price_audit"] = price_audits
    tables["actions"] = replace_target_rows(tables["actions"], new_actions, "price_target_id")
    for key, name in (
        ("identities", "security_reference_interval_observations.parquet"),
        ("ticker_events", "ticker_change_observations.parquet"),
    ):
        rows = fresh(action_root, ah, name)
        base = tables[key].loc[~tables[key].action_target_id.isin(new_actions.action_target_id)]
        tables[key] = pd.concat([base, rows], ignore_index=True)
    tables["metadata"] = replace_target_rows(tables["metadata"], new_metadata, "price_target_id")
    prior_windows = _json(sr, sh, "window_evidence.json")
    rebound, reuse = reauthorize_cached_continuity(
        events,
        prior_windows,
        tables["actions"],
        tables["identities"],
        tables["ticker_events"],
        tables["metadata"],
        tables["sessions"],
    )
    prior_continuity = pd.read_parquet(_artifact(sr, sh, "conditional_identity_continuity.parquet"))
    continuity = replace_target_rows(prior_continuity, rebound, "action_target_id")
    windows, dates = _audit_windows(tables, benchmark_report, events, continuity)
    gate = all_case_return_gate(windows, [e["validation_event_id"] for e in events])
    files = {
        "window_evidence.json": _encode(windows),
        "cached_identity_reuse_audit.json": _encode(reuse),
        "conditional_identity_continuity.parquet": _parquet(continuity),
        "collection.json": _encode(collection),
        **{ticker + "_session_audit.parquet": _parquet(frame) for ticker, frame in dates.items()},
    }
    action_manifest = json.loads((action_root / "output_manifest.json").read_bytes())
    action_report_path = action_root / "massive_security_actions_report.json"
    verify_file(action_report_path, action_manifest["report_sha256"])
    action_report = json.loads(action_report_path.read_bytes())
    report = {
        "version": VERSION,
        **gate,
        "source_manifest_sha256": sh,
        "supplement_plan_root": plan_root.as_posix(),
        "supplement_plan_manifest_sha256": digest_file(plan_root / "output_manifest.json"),
        "price_root": price_root.as_posix(),
        "price_manifest_sha256": ph,
        "actions_root": action_root.as_posix(),
        "actions_manifest_sha256": ah,
        "signals_refitted": False,
        "signals_frozen_before_collection": True,
        "first_public_certified": False,
        "trading_admission": False,
        "inferential_tests_computed": False,
        "strategy_pnl_computed": False,
        "general_scope_gate_cleared": False,
        "industry_adjusted_ic_computed": False,
        "minimum_industry_events": 20,
        "originals_overwritten": False,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "retrospective_replayed_forecasts_not_live": True,
        "transaction_costs_included": False,
        "sector_neutralization_included": False,
        "limitations": [
            "fourteen_hand_reviewed_events_not_representative",
            "SEC_acceptance_proxy_not_certified_first_public",
            "accounting_scope_quarantine_not_a_known_live_filter",
            "remaining_unreviewed_historical_release_clocks_and_accounting_scopes",
            "historical_price_delivery_times_not_certified",
            "no_costs_or_sector_risk_controls",
            "not_a_new_holdout_or_inferential_test",
        ],
        "network_requests": sum(v.get("requests_made", 0) for v in collection.values()),
        "network_requests_describe_this_replay_only": True,
        "source_successful_response_pages": int(
            action_report["raw_snapshots"]
            + fresh(price_root, ph, "massive_daily_price_completions.parquet").page_count.sum()
        ),
        "conditional_returns_computed": False,
        "verdict": "market_supplement_incomplete",
    }
    if gate["all_case_return_gate"]:
        # Both forecasts and every window were frozen/audited before these numeric reads.
        new_bars = fresh(price_root, ph, "massive_daily_price_observations.parquet")
        old_bars_path = _artifact(
            Path(market["prices_root"]),
            market["prices_manifest_sha256"],
            "validation_price_batch_bars.parquet",
        )
        old_overlap = pd.read_parquet(
            old_bars_path,
            filters=[("price_target_id", "in", new_bars.price_target_id.unique().tolist())],
        )
        overlap = reconcile_overlap(old_overlap, new_bars)
        files["overlap_reconciliation.json"] = _encode(_rows(overlap))
        splits = fresh(action_root, ah, "split_observations.parquet")
        dividends = fresh(action_root, ah, "dividend_observations.parquet")

        def table(group: str, name: str) -> pd.DataFrame:
            return pd.read_parquet(
                _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], name)
            )

        old_splits, old_dividends = (
            table("actions", "split_observations.parquet"),
            table("actions", "dividend_observations.parquet"),
        )
        replaced = set(new_actions.action_target_id)
        splits = pd.concat([old_splits.loc[~old_splits.action_target_id.isin(replaced)], splits])
        dividends = pd.concat(
            [old_dividends.loc[~old_dividends.action_target_id.isin(replaced)], dividends]
        )
        bs, bd = (
            table("benchmark", "benchmark_split_observations.parquet"),
            table("benchmark", "benchmark_dividend_observations.parquet"),
        )
        benchmark_path = _artifact(
            Path(market["benchmark_root"]),
            market["benchmark_manifest_sha256"],
            "benchmark_daily_price_observations.parquet",
        )
        results, cashflows = [], []
        for event, audit in zip(events, windows, strict=True):
            w = audit["window"]
            filters = [
                ("session_date", ">=", w["entry_session_date"]),
                ("session_date", "<=", w["exit_session_date"]),
            ]
            columns = [
                "ticker",
                "session_date",
                "open_unadjusted",
                "close_unadjusted",
                "source_sha256",
            ]
            bars = (
                new_bars.loc[new_bars.price_target_id.eq(event["price_target_id"])]
                if event["price_target_id"] in set(new_bars.price_target_id)
                else pd.read_parquet(
                    old_bars_path,
                    columns=["price_target_id", *columns],
                    filters=[("price_target_id", "==", event["price_target_id"]), *filters],
                )
            )
            benchmark = pd.read_parquet(
                benchmark_path,
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
            market_actions = normalize_holding_actions(bs, bd, tables["sessions"], w, "SPY")
            result, flows = calculate_case_return(
                event, audit, bars, benchmark, stock_actions, market_actions
            )
            results.append(result)
            cashflows.extend(flows)
        signals = pd.DataFrame(_json(sr, sh, "signal_panel.json"))
        sample, stats, influence = scope_rank_diagnostics(signals, pd.DataFrame(results))
        baseline = source["config"]
        controls = _json(
            Path(baseline["baseline_root"]),
            baseline["baseline_manifest_sha256"],
            "case_returns.json",
        )
        verify_control_returns(controls, results)
        files.update(
            {
                "signal_panel.json": _artifact(sr, sh, "signal_panel.json").read_bytes(),
                "case_returns.json": _encode(results),
                "cashflow_evidence.json": _encode(cashflows),
                "same_case_analysis.json": _encode(_rows(sample)),
                "descriptive_ic.json": _encode(_rows(stats)),
                "leave_one_issuer_out.json": _encode(_rows(influence)),
            }
        )
        report.update(
            conditional_returns_computed=True,
            overlap_rows_verified=len(overlap),
            complete_case_events=14,
            issuers=int(sample.cik.nunique()),
            primary_arm=ARMS[2],
            old_six_control_returns_reproduced=True,
            frozen_signal_bytes_preserved=True,
            later_delisted_cases_retained=[
                r["ticker"] for r in results if r["later_delisted_after_window"]
            ],
            verdict="fourteen_case_diagnostic_effectiveness_not_established",
        )
        report["primary_descriptive_ic"] = _rows(stats.loc[stats.arm.eq(ARMS[2])])
        report["primary_leave_one_issuer_out_ic_range"] = _rows(
            influence.loc[influence.arm.eq(ARMS[2])]
            .groupby("representation", as_index=False)
            .agg(minimum_ic=("spearman_ic", "min"), maximum_ic=("spearman_ic", "max"))
        )
    files["summary.json"] = _encode(report)
    return write_bundle(output_root, VERSION + "-diagnostic", files), report


def audit_supplement_prices(
    targets: pd.DataFrame,
    raw_root: Path,
    output_root: Path,
    plan: dict[str, Any],
) -> Path:
    """Audit the EXACT frozen windows, not the generic observed-bar entry estimator.

    The legacy pilot audit estimates entry from a filing-date label and therefore
    requires query_date. This source adapter deliberately does not estimate entry:
    the fourteen-case official-calendar window audit remains authoritative.
    """
    bars, rejected, completions = load_collected_massive_daily_bars(targets, raw_root=raw_root)
    audit = targets.copy()
    audit["provider_price_query_complete"] = audit.price_target_id.isin(completions.price_target_id)
    audit["rejected_bar_count"] = (
        audit.price_target_id.map(rejected.groupby("price_target_id").size()).fillna(0).astype(int)
    )
    audit["daily_bar_count"] = (
        audit.price_target_id.map(bars.groupby("price_target_id").size()).fillna(0).astype(int)
    )
    return write_typed_bundle(
        output_root,
        VERSION + "-price-source-audit",
        {
            "sample_massive_price_audit.parquet": _parquet(audit),
            "massive_daily_price_observations.parquet": _parquet(bars),
            "massive_daily_price_rejections.parquet": _parquet(rejected),
            "massive_daily_price_completions.parquet": _parquet(completions),
            "source_contract.json": _encode(
                {
                    "price_plan_manifest_sha256": plan["child_manifest_sha256"]["prices"],
                    "raw_store_scope_sha256": plan["raw_store_scope_sha256"],
                    "entry_inferred_from_observed_bars": False,
                    "entry_defined_by_frozen_official_calendar": True,
                    "returns_computed": False,
                    "query_adjusted": False,
                }
            ),
        },
    )


def main() -> None:
    """Collect using an environment-only credential; never replace the previous raw store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/announcement_market_supplement_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_market_supplement")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/market/announcement_market_supplement")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--collection-evidence-root", type=Path, default=None)
    parser.add_argument("--collection-evidence-manifest-sha256", default=None)
    args = parser.parse_args()
    root, plan = prepare_supplement(args.config, args.output_root)
    scope = plan["raw_store_scope_sha256"]
    raw = args.raw_root / scope
    prices = pd.read_parquet(
        _artifact(
            Path(plan["price_plan_root"]),
            plan["child_manifest_sha256"]["prices"],
            "massive_daily_price_targets.parquet",
        )
    )
    actions = pd.read_parquet(
        _artifact(
            Path(plan["action_plan_root"]),
            plan["child_manifest_sha256"]["actions"],
            "massive_security_actions_targets.parquet",
        )
    )
    print(f"SUPPLEMENT PLAN targets=2 queries=12 fallback_max=40 output={root}", flush=True)
    collection: dict[str, Any] = {}
    if args.collection_evidence_root is not None:
        if args.collect or not args.collection_evidence_manifest_sha256:
            parser.error("Pinned collection evidence is only supported for an offline replay")
        collection = _json(
            args.collection_evidence_root,
            args.collection_evidence_manifest_sha256,
            "collection.json",
        )
        evidence_summary = _json(
            args.collection_evidence_root, args.collection_evidence_manifest_sha256, "summary.json"
        )
        if evidence_summary["source_manifest_sha256"] != plan["config"]["source_manifest_sha256"]:
            raise ValueError("Collection receipt belongs to a different frozen research sample")
    if args.collect:
        key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not key:
            parser.error("MASSIVE_API_KEY is required with --collect; never put it in a file")
        client = MassiveReferenceClient(key, request_interval_seconds=12.1)

        def progress(done: int, total: int, tid: str) -> None:
            print(f"SUPPLEMENT COLLECT progress={done}/{total} target={tid[:12]}", flush=True)

        ac = collect_massive_security_actions_artifacts(
            actions, client, raw_root=raw / "actions", limit=2, progress=progress
        )
        collection["actions"] = asdict(ac)
        print(
            f"SUPPLEMENT ACTIONS completed={ac.targets_completed} failed={ac.targets_failed} "
            f"requests={ac.requests_made} blocked={ac.blocking_reason}",
            flush=True,
        )
        if ac.blocking_reason:
            blocked = write_bundle(
                args.output_root,
                VERSION + "-blocked",
                {
                    "collection.json": _encode(collection),
                    "plan_root.json": _encode(root.as_posix()),
                },
            )
            print(f"SUPPLEMENT STOP blocked={ac.blocking_reason} output={blocked}", flush=True)
            return
        sessions = pd.read_parquet(
            _artifact(
                Path(plan["calendar_root"]),
                plan["child_manifest_sha256"]["calendar"],
                "historical_exchange_sessions.parquet",
            )
        )
        fallback = collect_massive_security_session_fallbacks(
            actions, sessions, client, raw_root=raw / "actions", limit=2, progress=progress
        )
        collection["session_identity_fallback"] = asdict(fallback)
        if fallback.blocking_reason:
            print(f"SUPPLEMENT STOP blocked={fallback.blocking_reason}", flush=True)
            return
        pc = collect_massive_price_artifacts(
            prices, client, raw_root=raw / "prices", limit=2, progress=progress
        )
        collection["prices"] = asdict(pc)
        print(
            f"SUPPLEMENT PRICES completed={pc.targets_completed} failed={pc.targets_failed} "
            f"requests={pc.requests_made} blocked={pc.blocking_reason}",
            flush=True,
        )
    price_root = audit_supplement_prices(prices, raw / "prices", args.output_root / "prices", plan)
    action = build_massive_security_actions_audit(
        Path(plan["action_plan_root"]),
        Path(plan["calendar_root"]),
        raw_root=raw / "actions",
        output_root=args.output_root / "actions",
    )
    output, report = finish_diagnostic(
        root, plan, price_root, action.output_root, args.output_root, collection
    )
    print(
        f"SUPPLEMENT DONE ready={report['metadata_ready_events']}/14 "
        f"gate={report['all_case_return_gate']} verdict={report['verdict']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
