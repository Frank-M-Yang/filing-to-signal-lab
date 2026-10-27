"""Freeze six reviewed announcements, replay the fixed model, then read cached returns."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from scripts.audit_announcement_admission import _config, _recheck_cells
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_replay import SIGNALS, apply_release_proxies
from sec_data.announcement_returns import (
    VERSION,
    audit_return_window,
    calculate_case_return,
    case_signal_panel,
    descriptive_same_case_diagnostics,
    normalize_holding_actions,
    select_reviewed_cases,
)
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    apply_history_clocks,
    audit_replay_chronology,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.joint_publisher_replay import assert_originals_preserved
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig
from sec_data.publisher_replay import verify_baseline_projection


def validate_contract(c: dict[str, Any]) -> None:
    """Allow conditional descriptive returns only, never silently weaken prior experiments."""
    expected = {
        "version": VERSION,
        "selection_policy": "all_reviewed_original_GM_pairs_no_market_feature_requirement",
        "frozen_event_count": 316,
        "case_count": 6,
        "clock_policy": "reviewed_8k_acceptance_proxy_not_certified_first_public",
        "entry_session_offset": 2,
        "holding_sessions_inclusive": 20,
        "benchmark": "SPY",
        "fixed_financial_specification_only": True,
        "conditional_return_diagnostics_allowed": True,
        **dict.fromkeys(
            (
                "network_allowed",
                "market_features_in_forecast",
                "hyperparameter_search_allowed",
                "sample_replacement_allowed",
                "first_public_certified",
                "trading_admission",
                "inferential_tests_allowed",
                "strategy_pnl_allowed",
                "overwrite_originals",
            ),
            False,
        ),
    }
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported or weakened conditional return contract")
    ids = c.get("case_event_ids", [])
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Six unique frozen cases required")


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=15)
    )
    return result


def _one(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 1:
        raise ValueError("Missing or ambiguous case evidence")
    return _rows(frame)[0]


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _figure(sample: pd.DataFrame) -> bytes:
    titles = ("Level (margin %)", "Delta (pp)", "Raw Innovation (pp)", "Standardized Innovation")
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    for i, signal in enumerate(SIGNALS):
        scale = 1 if signal == "gm_innovation_std" else 100
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=sample[signal] * scale,
                y=sample.market_excess_return * 100,
                text=sample.ticker,
                mode="markers+text",
                textposition="top center",
                marker={"size": 10, "color": "#127c72"},
                showlegend=False,
                hovertemplate="%{text}<br>Signal %{x:.4f}<br>SPY excess %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text="20-session SPY excess (%)", row=row, col=col)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=col)
    fig.update_layout(
        title="Six reviewed announcements | conditional 8-K clock diagnostic | descriptive only",
        template="plotly_white",
        height=850,
        margin={"t": 110, "b": 100},
    )
    fig.add_annotation(
        x=0,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text="Same six events in every panel. No p-values, strategy P&L, "
        "or first-public certification.<br>Original values; +2 session open to "
        "holding-session 20 close; explicit splits/cash dividends.",
    )
    html: str = fig.to_html(include_plotlyjs=True, full_html=True)
    return html.encode("utf-8")


def build_diagnostic(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Pin a metadata-only plan before reading any new forward numeric price values."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)
    replay = _config(c["source_replay_config"], c["source_replay_config_sha256"])
    for key in (
        "admission_root",
        "admission_manifest_sha256",
        "admission_plan_root",
        "admission_plan_manifest_sha256",
    ):
        if c[key] != replay[key]:
            raise ValueError("Replay and return diagnostic cohorts differ")

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    cohort = load("admission", "event_admission_ledger.json")
    source_plan = load("admission_plan", "plan.json")
    if (
        len(cohort) != 316
        or load("admission", "summary.json")["plan_manifest_sha256"]
        != c["admission_plan_manifest_sha256"]
    ):
        raise ValueError("Frozen admission plan or denominator changed")
    cases = select_reviewed_cases(cohort, c["case_event_ids"])
    market = source_plan["market_config"]
    for name, expected in (("market", market), ("continuity", source_plan["continuity_config"])):
        ac = source_plan["config"]
        if _config(ac[name + "_config"], ac[name + "_config_sha256"]) != expected:
            raise ValueError("Evidence configuration changed")

    def path(group: str, name: str) -> Path:
        return _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], name)

    def table(group: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(path(group, name), columns=columns)

    def report(group: str, name: str) -> dict[str, Any]:
        root = Path(market[group + "_root"])
        verify_file(root / "output_manifest.json", market[group + "_manifest_sha256"])
        m = json.loads((root / "output_manifest.json").read_bytes())
        verify_file(root / name, m["report_sha256"])
        value: dict[str, Any] = json.loads((root / name).read_bytes())
        return value

    cc = source_plan["continuity_config"]
    cm = json.loads((Path(cc["continuity_root"]) / "output_manifest.json").read_bytes())
    if any(
        cm[k] != market[g + "_manifest_sha256"]
        for k, g in (
            ("execution_plan_manifest_sha256", "execution"),
            ("price_audit_manifest_sha256", "prices"),
            ("security_actions_manifest_sha256", "actions"),
        )
    ):
        raise ValueError("Continuity evidence belongs to another market chain")
    continuity = pd.read_parquet(
        _artifact(
            Path(cc["continuity_root"]),
            cc["continuity_manifest_sha256"],
            "validation_action_evidence_status.parquet",
        )
    )
    targets = table("prices", "validation_price_batch_targets.parquet")
    price_audits = table("prices", "validation_price_batch_target_audit.parquet")
    actions = table("actions", "sample_security_actions_audit.parquet")
    listing = table("listing", "validation_security_listing_outcomes.parquet")
    sessions = table("execution", "validation_exchange_sessions.parquet")
    queries = table("benchmark", "massive_benchmark_snapshots.parquet")
    bench_report = report("benchmark", "massive_benchmark_report.json")
    if queries.benchmark_target_id.nunique() != 1:
        raise ValueError("Ambiguous benchmark identity")
    benchmark_id = str(queries.benchmark_target_id.iloc[0])
    # Metadata projections cannot accidentally expose forward OHLC fields.
    meta = table(
        "prices",
        "validation_price_batch_bars.parquet",
        ["price_target_id", "ticker", "session_date", "source_sha256"],
    )
    bmeta = table(
        "benchmark",
        "benchmark_daily_price_observations.parquet",
        ["ticker", "session_date", "source_sha256"],
    )
    events, windows, session_audits, source_evidence = [], [], {}, []
    for case in cases:
        tid = case["price_target_id"]
        target = _one(targets.loc[targets.price_target_id.eq(tid)])
        if (
            target["adjusted"] is not False
            or target["selection_uses_return_values"] is not False
            or any(
                target[k] != case[v]
                for k, v in (
                    ("cik", "cik"),
                    ("selected_ticker", "assigned_ticker"),
                    ("selected_exchange_mic", "assigned_exchange_mic"),
                )
            )
        ):
            raise ValueError("Outcome-blind raw-price identity changed")
        action = _one(actions.loc[actions.price_target_id.eq(tid)])
        event = {
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
        cont = _one(continuity.loc[continuity.action_target_id.eq(action["action_target_id"])])
        prices = _one(price_audits.loc[price_audits.price_target_id.eq(tid)])
        outcome = _one(
            listing.loc[listing.cik.eq(event["cik"]) & listing.security_id.eq(event["security_id"])]
        )
        audit, dates = audit_return_window(
            event,
            sessions,
            meta.loc[meta.price_target_id.eq(tid)].drop(columns="price_target_id"),
            bmeta,
            cont,
            action,
            prices,
            outcome,
            queries,
            bench_report,
        )
        events.append(event)
        windows.append(audit)
        session_audits[event["ticker"]] = dates
        source_evidence.append(
            {
                "event": event,
                "continuity": cont,
                "actions": action,
                "prices": prices,
                "listing": outcome,
            }
        )
    implementations = {
        name: digest_file(Path(name))
        for name in (
            "scripts/build_announcement_return_diagnostic.py",
            "sec_data/announcement_returns.py",
            "sec_data/announcement_replay.py",
            "sec_data/clock_layer.py",
            "sec_data/event_returns.py",
            "sec_data/execution_contract.py",
            "sec_data/pilot_filing_features.py",
            "sec_data/event_cleaning.py",
            "event_panel.py",
            "sec_data/disclosure_review.py",
            "scripts/audit_announcement_admission.py",
            "scripts/build_sec_disclosure_audit.py",
            "scripts/build_sec_clock_layer.py",
            "sec_data/publisher_replay.py",
            "sec_data/joint_publisher_replay.py",
        )
    }
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "source_replay_config": replay,
        "cases": cases,
        "windows": windows,
        "market_config": market,
        "continuity_config": cc,
        "financial_sources": source_plan["financial_sources"],
        "implementation_sha256": implementations,
        "selection_uses_return_values": False,
        "forward_numeric_prices_read": False,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "all_six_reviewed_cases_retained": True,
        "analysis_policy": "same_case_descriptive_Spearman_and_leave_one_issuer_out_only",
        "source_evidence": source_evidence,
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    ready = sum(w["conditional_return_diagnostic_ready"] for w in windows)
    print(
        f"RETURN PLAN cases={len(cases)} metadata_ready={ready} output={frozen}",
        flush=True,
    )
    if not all(w["conditional_return_diagnostic_ready"] for w in windows):
        raise ValueError(f"Frozen cases blocked; no outcomes read. Inspect {frozen}")
    review = _config(
        source_plan["config"]["review_config"], source_plan["config"]["review_config_sha256"]
    )
    cells, _, source = _recheck_cells(review)
    if (
        cells != load("admission", "rechecked_metric_cells.json")
        or source != source_plan["financial_sources"]
    ):
        raise ValueError("Raw original values do not reproduce")
    cp = _json(Path(replay["clock_plan_root"]), replay["clock_plan_manifest_sha256"], "plan.json")
    cr = _json(Path(replay["clock_root"]), replay["clock_manifest_sha256"], "report.json")
    if cr["plan_manifest_sha256"] != replay["clock_plan_manifest_sha256"] or cp["source"] != source:
        raise ValueError("Clock/model/source identity changed")
    cfg = PilotFilingFeatureConfig(**cp["forecast_config"])
    cfg.validate()
    history = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        ),
        columns=HISTORY_COLUMNS,
    )
    legacy = replay_history(history, cfg)
    verify_baseline_projection(legacy, history)
    print("RETURN REPLAY legacy reproduced=True", flush=True)
    baseline = replay_history(
        apply_history_clocks(
            history,
            _json(Path(replay["clock_root"]), replay["clock_manifest_sha256"], "clock_layer.json"),
        ),
        cfg,
    )
    saved = pd.read_parquet(
        _artifact(
            Path(replay["clock_root"]),
            replay["clock_manifest_sha256"],
            "disarmed_replay_panel.parquet",
        )
    )
    verify_baseline_projection(baseline, saved)
    joint = replay_history(apply_release_proxies(baseline, cases, cells), cfg)
    assert_originals_preserved(baseline, joint)
    pd.testing.assert_frame_equal(
        apply_release_proxies(baseline, cases, cells),
        apply_release_proxies(baseline, cases[::-1], cells),
    )
    earlier = baseline.actual_available_at.lt(utc_timestamp(cases[0]["earliest_disclosure_at"]))
    if not compare_histories(baseline.loc[earlier], joint.loc[earlier])[1].empty:
        raise ValueError("Later overlays changed earlier history")
    signals = case_signal_panel(baseline, joint, cases)
    forecast_plan = write_bundle(
        output_root / "plans",
        VERSION + "-signal-freeze",
        {
            "signal_panel.json": _encode(_rows(signals)),
            "forecast_contract.json": _encode(
                {
                    "forecast_config": cp["forecast_config"],
                    "forecast_features": list(FORECAST_FEATURES),
                    "selection_plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
                    "forward_numeric_prices_read": False,
                }
            ),
        },
    )
    print(
        f"RETURN SIGNALS frozen={len(signals)} chronology=passed output={forecast_plan}", flush=True
    )
    issuer_splits = table("actions", "split_observations.parquet")
    issuer_dividends = table("actions", "dividend_observations.parquet")
    benchmark_splits = table("benchmark", "benchmark_split_observations.parquet")
    benchmark_dividends = table("benchmark", "benchmark_dividend_observations.parquet")
    returns, cashflows = [], []
    for event, evidence in zip(events, windows, strict=True):
        window = evidence["window"]
        filters = [
            ("session_date", ">=", window["entry_session_date"]),
            ("session_date", "<=", window["exit_session_date"]),
        ]
        columns = ["ticker", "session_date", "open_unadjusted", "close_unadjusted", "source_sha256"]
        bars = pd.read_parquet(
            path("prices", "validation_price_batch_bars.parquet"),
            columns=["price_target_id", *columns],
            filters=[("price_target_id", "==", event["price_target_id"]), *filters],
        )
        bench = pd.read_parquet(
            path("benchmark", "benchmark_daily_price_observations.parquet"),
            columns=["benchmark_target_id", *columns],
            filters=[("benchmark_target_id", "==", benchmark_id), *filters],
        )

        stock_actions = normalize_holding_actions(
            issuer_splits.loc[issuer_splits.action_target_id.eq(event["action_target_id"])],
            issuer_dividends.loc[issuer_dividends.action_target_id.eq(event["action_target_id"])],
            sessions,
            window,
            event["ticker"],
        )
        market_actions = normalize_holding_actions(
            benchmark_splits, benchmark_dividends, sessions, window, "SPY"
        )
        result, flows = calculate_case_return(
            event, evidence, bars, bench, stock_actions, market_actions
        )
        returns.append(result)
        cashflows.extend(flows)
    sample, stats, influence = descriptive_same_case_diagnostics(signals, pd.DataFrame(returns))
    report_out = {
        "version": VERSION,
        "selection_plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "signal_freeze_manifest_sha256": digest_file(forecast_plan / "output_manifest.json"),
        "frozen_cohort_events_retained": len(cohort),
        "reviewed_cases": len(cases),
        "complete_case_events": len(sample),
        "issuers": int(sample.cik.nunique()),
        "tickers": sample.ticker.tolist(),
        "conditional_window_ready": len(windows),
        "financial_history_events": len(joint),
        "forecast_config": cp["forecast_config"],
        "chronology": audit_replay_chronology(joint),
        "history_before_first_overlay_unchanged": True,
        "later_delisted_cases_retained": sample.loc[
            sample.later_delisted_after_window, "ticker"
        ].tolist(),
        "market_features_in_forecast": False,
        "retrospective_replayed_forecasts_not_live": True,
        "event_set_frozen_before_this_run_outcome_read": True,
        "signals_frozen_before_this_run_outcome_read": True,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "first_public_certified": False,
        "trading_admission": False,
        "inferential_tests_computed": False,
        "strategy_pnl_computed": False,
        "transaction_costs_included": False,
        "sector_neutralization_included": False,
        "network_requests": 0,
        "originals_overwritten": False,
        "verdict": "six_case_pipeline_verified_effectiveness_not_established",
        "limitations": [
            "six_hand_reviewed_cases_not_representative",
            "SEC_acceptance_is_not_certified_first_public",
            "some_training_actuals_may_have_earlier_unreviewed_public_releases",
            "historical_price_delivery_times_not_certified",
            "AMWD_alternative_identity_route_not_official_exchange_certification",
            "no_costs_or_sector_risk_controls",
            "historical_outcomes_have_been_examined_in_prior_work",
        ],
    }
    for row in cohort:
        row["selected_for_conditional_return_diagnostic"] = (
            row["validation_event_id"] in c["case_event_ids"]
        )
        row["return_admission"] = row["trading_admission"] = False
    files = {
        "summary.json": _encode(report_out),
        "case_returns.json": _encode(returns),
        "signal_panel.json": _encode(_rows(signals)),
        "same_case_analysis.json": _encode(_rows(sample)),
        "descriptive_ic.json": _encode(_rows(stats)),
        "leave_one_issuer_out.json": _encode(_rows(influence)),
        "window_evidence.json": _encode(windows),
        "cashflow_evidence.json": _encode(cashflows),
        "cohort_status.json": _encode(cohort),
        "rechecked_metric_cells.json": _encode(cells),
        "joint_history.parquet": _parquet(joint),
        "diagnostic.html": _figure(sample),
        **{
            ticker + "_session_audit.parquet": _parquet(frame)
            for ticker, frame in session_audits.items()
        },
    }
    return write_bundle(output_root, VERSION, files), report_out


def main() -> None:
    """Run the fixed offline diagnostic; report neither strategy performance nor Alpha."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/announcement_return_diagnostic_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_return_diagnostic")
    )
    args = parser.parse_args()
    root, report = build_diagnostic(args.config, args.output_root)
    print(
        f"ANNOUNCEMENT RETURNS DONE events={report['complete_case_events']} "
        f"issuers={report['issuers']} verdict={report['verdict']} output={root}"
    )


if __name__ == "__main__":
    main()
