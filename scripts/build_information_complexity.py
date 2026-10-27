"""Freeze four return-model arms and audit inputs offline; never retune fourteen cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_replay import case_row
from sec_data.disclosure_audit import digest_file
from sec_data.information_complexity import (
    ARMS,
    FINANCIAL,
    FINANCIAL_CLOCKS,
    STATE,
    VERSION,
    ExperimentConfig,
    audit_prior_history,
    lagged_industry_features,
    run_experiment,
    select_review_batch,
)
from sec_data.validation_market_state import (
    RAW_PRICE_BASIS,
    ValidationMarketStateConfig,
    _feature_row,
    _prepare_market_inputs,
)


def build_experiment(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Preserve source bytes, freeze parameters, expose gaps and refuse premature training."""
    c = json.loads(config_path.read_bytes())
    if c.get("version") != VERSION or any(
        c.get(k) is not False
        for k in (
            "network_allowed",
            "financial_model_refitting_allowed",
            "overwrite_originals",
            "hyperparameter_search_allowed",
            "trading_admission",
        )
    ):
        raise ValueError("Unsupported or weakened information-complexity contract")
    sr, sh = Path(c["source_root"]), c["source_manifest_sha256"]
    summary = _json(sr, sh, "summary.json")
    supplement = _json(
        Path(summary["supplement_plan_root"]),
        summary["supplement_plan_manifest_sha256"],
        "plan.json",
    )
    sp = supplement["config"]
    p = _json(Path(sp["source_plan_root"]), sp["source_plan_manifest_sha256"], "plan.json")
    old = Path(sp["source_root"])
    oh = sp["source_manifest_sha256"]
    signal_path = _artifact(sr, sh, "signal_panel.json")
    old_signal = _artifact(old, oh, "signal_panel.json")
    if (
        summary["all_case_return_gate"] is not True
        or summary["complete_case_events"] != 14
        or summary["primary_arm"] != p["config"]["primary_arm"]
        or signal_path.read_bytes() != old_signal.read_bytes()
    ):
        raise ValueError("Frozen fourteen-case source chain changed")
    primary = summary["primary_arm"]
    # Freeze the identity-only next batch before reading any return labels in this run.
    expansion = _json(
        Path(p["config"]["expansion_plan_root"]),
        p["config"]["expansion_plan_manifest_sha256"],
        "plan.json",
    )
    base = expansion["base_config"]
    cohort = _json(
        Path(base["admission_root"]),
        base["admission_manifest_sha256"],
        "event_admission_ledger.json",
    )
    if len(cohort) != 316 or {r["validation_event_id"] for r in cohort} != set(
        p["all_cohort_event_ids"]
    ):
        raise ValueError("Original cohort denominator changed")
    prior = {r["cik"] for r in [*expansion["controls"], *expansion["selected_cases"]]}
    queue, candidate_issuers = select_review_batch(cohort, prior)
    protocol = {
        "version": VERSION,
        "config": c,
        "config_sha256": digest_file(config_path),
        "parameters": vars(ExperimentConfig()),
        "arms": ARMS,
        "target": "20_session_issuer_total_return_minus_SPY_total_return",
        "entry": "second_trading_day_after_announcement_unadjusted_open",
        "exit": "twentieth_session_inclusive_unadjusted_close",
        "decision": "one_minute_before_entry_open_with_pre_announcement_market_state",
        "financial_forecast_cutoff": "strictly_before_announcement",
        "return_prediction_cutoff": "after_announcement_strictly_before_entry",
        "industry_feature": "median_other_issuers_latest_known_strictly_prior_quarter_GM",
        "financial_signal_sha256": digest_file(signal_path),
        "implementation_sha256": {
            n: digest_file(Path(n))
            for n in (
                "scripts/build_information_complexity.py",
                "sec_data/information_complexity.py",
                "sec_data/validation_market_state.py",
                "sec_data/announcement_expansion.py",
            )
        },
        "fourteen_cases_for_pipeline_audit_only": True,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "live_forecast_generation_certified": False,
        "strategy_pnl_allowed": False,
        "next_review_batch": queue,
        "cohort_events_retained": len(cohort),
        "previous_batch_issuers_not_replaced": sorted(prior),
    }
    frozen = write_bundle(
        output_root / "plans", VERSION + "-protocol", {"plan.json": _encode(protocol)}
    )
    # Freeze specifications before reading labels or feature values in this run.
    signals = pd.DataFrame(_json(sr, sh, "signal_panel.json"))
    signals = signals.loc[signals.arm.eq(primary)].copy()
    returns = pd.DataFrame(_json(sr, sh, "case_returns.json"))
    panel = signals.merge(
        returns, on=["validation_event_id", "cik", "ticker"], validate="one_to_one"
    )
    if len(panel) != 14 or set(panel.validation_event_id) != set(p["config"]["case_event_ids"]):
        raise ValueError("Do not drop, substitute or duplicate frozen cases")
    history = pd.read_parquet(_artifact(old, oh, primary + "_history.parquet"))
    events = cast(list[dict[str, Any]], panel.to_dict("records"))
    cases = {r["validation_event_id"]: r for r in p["cases"]}
    for event in events:
        target = case_row(history, cases[event["validation_event_id"]])
        if not np.allclose(
            target[list(FINANCIAL)].to_numpy(float),
            [event[k] for k in FINANCIAL],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Financial signals differ from frozen history; do not refit")
        if pd.Timestamp(target.actual_available_at) != pd.Timestamp(event["announcement_at"]):
            raise ValueError("Financial signals no longer match the announcement information set")
        event.update({k: target[k] for k in FINANCIAL_CLOCKS})
        event["period_start"] = target.period_start
    market = p["market_config"]

    def table(group: str, name: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(
            _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], name),
            **kwargs,
        )

    ids = panel.price_target_id.tolist()
    dates = table(
        "prices",
        "validation_price_batch_bars.parquet",
        columns=["price_target_id", "session_date"],
        filters=[("price_target_id", "in", ids)],
    )
    benchmark_dates = table(
        "benchmark", "benchmark_daily_price_observations.parquet", columns=["session_date"]
    )
    sessions = table("execution", "validation_exchange_sessions.parquet")
    actions = table("actions", "sample_security_actions_audit.parquet")
    audit = audit_prior_history(events, dates, benchmark_dates, sessions, actions)
    industry_rows = _json(
        Path(c["industry_root"]), c["industry_manifest_sha256"], "industry_observations.json"
    )
    industry = lagged_industry_features(events, history, industry_rows)
    state_rows = []
    for event, check in zip(events, audit, strict=True):
        state = {
            "validation_event_id": event["validation_event_id"],
            **dict.fromkeys(STATE[:3], np.nan),
            "market_feature_available_at": pd.NaT,
        }
        if check["conditional_feature_calculation_allowed"]:
            filters = [
                ("session_date", ">=", check["requested_start"]),
                ("session_date", "<=", check["requested_end"]),
            ]
            bars = table(
                "prices",
                "validation_price_batch_bars.parquet",
                filters=[("price_target_id", "==", event["price_target_id"]), *filters],
            )
            benchmark = table(
                "benchmark", "benchmark_daily_price_observations.parquet", filters=filters
            )
            groups, bm, splits, closes = _prepare_market_inputs(
                bars,
                benchmark,
                table("actions", "split_observations.parquet"),
                actions,
                sessions,
                issuer_price_basis=RAW_PRICE_BASIS,
                benchmark_price_basis=RAW_PRICE_BASIS,
            )
            values = _feature_row(
                SimpleNamespace(**{**event, "accepted_at": event["announcement_at"]}),
                groups[event["price_target_id"]],
                bm,
                splits.get(event["price_target_id"], pd.DataFrame()),
                closes,
                ValidationMarketStateConfig(),
            )
            if values["market_state_ready"]:
                state.update({k: values[k] for k in (*STATE[:3], "market_feature_available_at")})
        state_rows.append(state)
    panel = pd.DataFrame(events).merge(
        pd.DataFrame(state_rows), on="validation_event_id", validate="one_to_one"
    )
    panel = panel.merge(industry, on="validation_event_id", validate="one_to_one")
    panel["decision_at"] = pd.to_datetime(panel.entry_at, utc=True) - pd.Timedelta(minutes=1)
    # Do NOT invent delivery timestamps from session dates or label old sources as admitted.
    panel["label_available_at"] = pd.NaT
    panel["source_admitted"] = False
    panel["information_set"] = "original_release"
    predictions, metrics, report = run_experiment(panel)
    report.update(
        protocol_root=frozen.as_posix(),
        protocol_manifest_sha256=digest_file(frozen / "output_manifest.json"),
        source_signal_bytes_preserved=True,
        existing_returns_recomputed=False,
        conditional_market_state_events=int(panel[list(STATE[:3])].notna().all(axis=1).sum()),
        lagged_industry_state_events=int(panel.industry_state_ready.sum()),
        prior_history_audited_events=14,
        next_review_issuers=len(queue),
        unreviewed_candidate_issuers=candidate_issuers,
        label_delivery_clocks_certified=False,
        originals_overwritten=False,
        network_requests=0,
    )
    output = write_bundle(
        output_root,
        VERSION + "-audit",
        {
            "summary.json": _encode(report),
            "protocol.json": _encode(protocol),
            "model_input.json": _encode(_records(panel)),
            "history_audit.json": _encode(audit),
            "lagged_industry_state.json": _encode(_records(industry)),
            "next_review_batch.json": _encode(queue),
            "predictions.json": _encode(_records(predictions)),
            "metrics.json": _encode(_records(metrics)),
        },
    )
    return output, report


def main() -> None:
    """Build immutable offline audit artifacts; no API key, download or broker integration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/information_complexity_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/information_complexity")
    )
    args = parser.parse_args()
    output, report = build_experiment(args.config, args.output_root)
    print(
        f"INFORMATION COMPLEXITY DONE events={report['input_events']} "
        f"market_state={report['conditional_market_state_events']} "
        f"industry_state={report['lagged_industry_state_events']} "
        f"fits={report['models_fitted']} next_review={report['next_review_issuers']} "
        f"verdict={report['verdict']} output={output}"
    )


if __name__ == "__main__":
    main()
