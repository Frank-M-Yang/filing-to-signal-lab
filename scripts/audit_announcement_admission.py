"""Offline 316-event release-value, forecast-clock and market-window readiness ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_admission import (
    FORECAST_CLOCKS,
    VERSION,
    audit_forecast_metadata,
    exact_session_coverage,
    gap_records,
    join_candidate_clocks,
    price_query_covers_history,
    review_candidate_pair,
)
from sec_data.disclosure_audit import (
    digest_file,
    load_cached_submissions,
    utc_timestamp,
    verify_file,
)
from sec_data.disclosure_review import review_document_kinds, review_metric_cells
from sec_data.execution_contract import execution_window
from sec_data.publisher_market_rebuild import BAR_VALUE_COLUMNS, rebuild_pre_event_state
from sec_data.validation_market_state import RAW_PRICE_BASIS

EVENT_COLUMNS = [
    "validation_event_id",
    "cik",
    "target_accession_number",
    "report_date",
    "accepted_at",
    "assigned_ticker",
    "assigned_exchange_mic",
]
CLOCK_COLUMNS = [
    "cik",
    "accession_number",
    "period_end",
    "report_accepted_at",
    "release_accession_number",
    "release_accepted_at",
    "earliest_disclosure_at",
    "clock_source",
    "release_candidate_count",
    "release_event_date",
    "release_event_date_reliable",
]
FLAGS = [
    "analysis_ready_level",
    "analysis_ready_delta",
    "analysis_ready_innovation_raw",
    "analysis_ready_innovation_std",
]


def _config(path: str | Path, digest: str) -> dict[str, Any]:
    verify_file(Path(path), digest)
    result: dict[str, Any] = json.loads(Path(path).read_bytes())
    return result


def _table(c: dict[str, Any], group: str, name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(
        _artifact(Path(c[f"{group}_root"]), c[f"{group}_manifest_sha256"], name), columns=columns
    )


def _recheck_cells(
    c: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Revalidate the existing assertions against exact raw submissions, exhibits and anchors."""
    plan = _json(Path(c["plan_root"]), c["plan_manifest_sha256"], "plan.json")
    fields = _json(Path(c["audit_root"]), c["audit_manifest_sha256"], "field_audit.json")
    prior = _config(c["prior_review_path"], c["prior_review_sha256"])
    cache = load_cached_submissions([Path(p) for p in plan["config"]["cached_submission_roots"]])
    dispositions = review_document_kinds(
        c["documents"], plan["events"], plan["earnings_candidates"], cache
    )
    specs = list(prior["reviews"])
    for spec in c["metric_reviews"]:
        documents = [
            r for r in c["documents"] if r["release_accession"] == spec["release_accession"]
        ]
        if len(documents) != 1 or documents[0]["kind"] != "earnings_release":
            raise ValueError("Only reviewed final earnings releases may supply the actual pair")
        specs.append({**documents[0], **spec})
    cells = review_metric_cells(specs, plan["events"], fields, plan["earnings_candidates"], cache)
    return cells, dispositions, plan["config"]


def _date_interval(evidence: dict[str, Any], start: str, end: str) -> bool:
    return bool(
        evidence
        and all(
            evidence.get(k) is True
            for k in (
                "resolved_identity_ready",
                "historical_symbol_interval_ready",
                "corporate_actions_ready",
            )
        )
        and evidence["authorized_entry_start"] <= start <= end <= evidence["authorized_exit_end"]
    )


def build_audit(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Preserve all 316 events and expose gaps without certifying proxy clocks."""
    c = json.loads(config_path.read_bytes())
    if (
        c["version"] != VERSION
        or c["frozen_event_count"] != 316
        or c["required_sessions"] != 61
        or c["metric_pair"] != ["revenue", "gross_profit"]
        or any(
            c[k] is not False
            for k in (
                "network_allowed",
                "post_event_numeric_prices_allowed",
                "returns_allowed",
                "model_fitting_allowed",
                "sample_replacement_allowed",
                "overwrite_originals",
            )
        )
    ):
        raise ValueError("Unsupported announcement admission contract")
    market = _config(c["market_config"], c["market_config_sha256"])
    review = _config(c["review_config"], c["review_config_sha256"])
    continuity_config = _config(c["continuity_config"], c["continuity_config_sha256"])
    events = _table(market, "execution", "validation_execution_events.parquet", EVENT_COLUMNS)
    clocks = _table(c, "clock", "first_public_clock.parquet", CLOCK_COLUMNS)
    cohort = join_candidate_clocks(events, clocks)
    if len(cohort) != c["frozen_event_count"]:
        raise ValueError("Frozen event denominator changed")
    cells, dispositions, sources = _recheck_cells(review)
    clock_manifest = json.loads((Path(c["clock_root"]) / "output_manifest.json").read_bytes())
    clock_report_path = Path(c["clock_root"]) / "first_public_clock_report.json"
    verify_file(clock_report_path, clock_manifest["report_sha256"])
    clock_report = json.loads(clock_report_path.read_bytes())
    if clock_report["catalog_manifest_sha256"] != sources["catalog_manifest_sha256"]:
        raise ValueError("Clock catalog and financial evidence catalog differ")
    history = _table(
        sources,
        "feature",
        "quarterly_feature_history.parquet",
        ["company_id", "filing_accession", *FORECAST_CLOCKS, *FLAGS],
    )
    if history.duplicated(["company_id", "filing_accession"]).any():
        raise ValueError("Ambiguous existing forecast metadata")
    endpoints = _table(
        market,
        "prices",
        "validation_price_batch_event_endpoints.parquet",
        ["validation_event_id", "price_target_id", "cik", "ticker", "exchange_mic"],
    )
    targets = _table(
        market,
        "prices",
        "validation_price_batch_targets.parquet",
        [
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "adjusted",
            "price_window_start",
            "price_window_end",
        ],
    )
    price_audits = _table(
        market,
        "prices",
        "validation_price_batch_target_audit.parquet",
        [
            "price_target_id",
            "cik",
            "ticker",
            "price_window_start",
            "price_window_end",
            "provider_query_complete",
            "parser_rejection_rows",
        ],
    )
    actions = _table(
        market,
        "actions",
        "sample_security_actions_audit.parquet",
        [
            "price_target_id",
            "action_target_id",
            "cik",
            "selected_ticker",
            "price_window_start",
            "price_window_end",
            "splits_query_complete",
            "dividends_query_complete",
        ],
    )
    continuity = _table(
        continuity_config,
        "continuity",
        "validation_action_evidence_status.parquet",
        [
            "action_target_id",
            "cik",
            "ticker",
            "resolved_identity_ready",
            "historical_symbol_interval_ready",
            "corporate_actions_ready",
            "authorized_entry_start",
            "authorized_exit_end",
            "identity_continuity_route",
        ],
    )
    sessions = _table(
        market,
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
    price_path = _artifact(
        Path(market["prices_root"]),
        market["prices_manifest_sha256"],
        "validation_price_batch_bars.parquet",
    )
    benchmark_path = _artifact(
        Path(market["benchmark_root"]),
        market["benchmark_manifest_sha256"],
        "benchmark_daily_price_observations.parquet",
    )
    # Metadata only, including possible execution-window dates. No forward OHLCV fields here.
    price_dates = pd.read_parquet(
        price_path, columns=["price_target_id", "ticker", "session_date", "source_sha256"]
    )
    benchmark_dates = pd.read_parquet(
        benchmark_path, columns=["benchmark_target_id", "ticker", "session_date", "source_sha256"]
    )
    if (
        benchmark_dates.benchmark_target_id.nunique() != 1
        or not benchmark_dates.ticker.eq("SPY").all()
    ):
        raise ValueError("Ambiguous benchmark identity")
    benchmark_id = str(benchmark_dates.benchmark_target_id.iloc[0])
    bench_manifest = json.loads(
        (Path(market["benchmark_root"]) / "output_manifest.json").read_bytes()
    )
    bench_report_path = Path(market["benchmark_root"]) / "massive_benchmark_report.json"
    verify_file(bench_report_path, bench_manifest["report_sha256"])
    bench_report = json.loads(bench_report_path.read_bytes())
    benchmark_raw = (
        bench_report.get("query_adjusted") is False
        and bench_report.get("queries_complete") == bench_report.get("queries_required") == 3
        and bench_report.get("rejections") == bench_report.get("action_normalization_failures") == 0
        and bench_report.get("splits_explicit") is True
        and bench_report.get("dividends_explicit") is True
    )
    bench_queries = _table(
        market,
        "benchmark",
        "massive_benchmark_snapshots.parquet",
        [
            "window_start",
            "window_end",
            "query_kind",
            "status_code",
            "ticker",
            "benchmark_target_id",
            "sha256",
        ],
    )
    benchmark_raw = bool(
        benchmark_raw
        and set(bench_queries.query_kind) == {"daily_bars", "splits", "dividends"}
        and bench_queries.status_code.eq(200).all()
        and bench_queries.ticker.eq("SPY").all()
        and bench_queries.benchmark_target_id.eq(benchmark_id).all()
        and bench_queries.sha256.notna().all()
        and bench_queries.sha256.str.fullmatch(r"[a-f0-9]{64}").all()
    )
    benchmark_splits = _table(
        market, "benchmark", "benchmark_split_observations.parquet", ["execution_date"]
    )
    splits = _table(
        market,
        "actions",
        "split_observations.parquet",
        ["action_target_id", "ticker", "execution_date", "split_from", "split_to", "source_sha256"],
    )
    listing = _table(
        market,
        "listing",
        "validation_security_listing_outcomes.parquet",
        [
            "cik",
            "ticker_at_event",
            "listing_outcome_status",
            "delisting_date",
            "terminal_value_observed",
        ],
    )
    if (
        targets.price_target_id.duplicated().any()
        or price_audits.price_target_id.duplicated().any()
        or actions.price_target_id.duplicated().any()
        or continuity.action_target_id.duplicated().any()
        or listing.duplicated(["cik", "ticker_at_event"]).any()
    ):
        raise ValueError("Ambiguous market evidence identity")
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "market_config": market,
        "review_config": review,
        "financial_sources": sources,
        "continuity_config": continuity_config,
        "events": _records(cohort),
        "required_history_sessions": 61,
        "same_sample_no_replacement": True,
        "returns_allowed": False,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/audit_announcement_admission.py",
                "sec_data/announcement_admission.py",
                "sec_data/disclosure_review.py",
                "sec_data/publisher_market_rebuild.py",
                "sec_data/validation_market_state.py",
                "sec_data/execution_contract.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    print(
        f"ANNOUNCEMENT PLAN events={len(cohort)} cells_rechecked={len(cells)} output={frozen}",
        flush=True,
    )
    rows, states, gaps = [], [], []
    for e in _records(cohort):
        event_id = e["validation_event_id"]
        row = {
            **e,
            **review_candidate_pair(e, cells),
            "market_features_calculable": False,
            "market_history_evidence_ready": False,
            "existing_forecast_clocks_before_proxy": False,
            "legacy_all_representations_ready": False,
            "issuer_history_dates_complete": False,
            "benchmark_history_dates_complete": False,
            "execution_dates_present_only": False,
            "local_replay_candidate": False,
            "first_public_certified": False,
            "model_admission": False,
            "return_admission": False,
        }
        reasons: list[str] = []
        proxy = e["earliest_disclosure_at"]
        if not proxy or not e["filing_clock_matches_catalog"]:
            row["blocking_reasons"] = [
                "missing_or_conflicting_catalog_clock",
                "first_public_not_certified",
                "financial_history_replay_not_completed",
            ]
            rows.append(row)
            gaps.extend(gap_records(row))
            continue
        selected_docs = [
            r
            for r in dispositions
            if r["cik"] == e["cik"] and r["release_accession"] == e["release_accession_number"]
        ]
        row["selected_document_kind"] = (
            selected_docs[0]["kind"] if len(selected_docs) == 1 else "not_kind_reviewed"
        )
        old_forecast = history.loc[
            history.company_id.eq(e["cik"])
            & history.filing_accession.eq(e["target_accession_number"])
        ]
        meta = _records(old_forecast)[0] if len(old_forecast) == 1 else {}
        row.update(audit_forecast_metadata(meta, proxy))
        row["legacy_all_representations_ready"] = all(meta.get(k) is True for k in FLAGS)
        row["multiple_release_candidates"] = bool(
            e["release_candidate_count"] and e["release_candidate_count"] > 1
        )
        if not row["release_value_pair_verified"]:
            reasons.append(row["candidate_value_status"])
        if not row["existing_forecast_clocks_before_proxy"]:
            reasons.append("forecast_dependencies_not_all_strictly_before_proxy")
        if not row["legacy_all_representations_ready"]:
            reasons.append("legacy_financial_representations_incomplete")
        calendar = sessions.loc[sessions.exchange_mic.eq(e["assigned_exchange_mic"])].copy()
        component = endpoints.loc[endpoints.validation_event_id.eq(event_id)]
        target_rows = targets.loc[targets.price_target_id.isin(component.price_target_id)]
        identity_ok = len(component) == len(target_rows) == 1
        if identity_ok:
            t = _records(target_rows)[0]
            component_row = _records(component)[0]
            identity_ok = (
                t["cik"] == component_row["cik"] == e["cik"]
                and t["selected_ticker"] == component_row["ticker"] == e["assigned_ticker"]
                and t["selected_exchange_mic"]
                == component_row["exchange_mic"]
                == e["assigned_exchange_mic"]
            )
        pdates = (
            price_dates.loc[price_dates.price_target_id.eq(t["price_target_id"])]
            if identity_ok
            else price_dates.iloc[:0]
        )
        issuer_coverage = exact_session_coverage(calendar, pdates.session_date.tolist(), proxy)
        bench_coverage = exact_session_coverage(
            calendar, benchmark_dates.session_date.tolist(), proxy
        )
        row.update(
            price_target_resolved=identity_ok,
            price_target_id=t["price_target_id"] if identity_ok else None,
            calendar_ready=issuer_coverage["calendar_ready"],
            issuer_history_dates_complete=issuer_coverage["history_dates_complete"],
            benchmark_history_dates_complete=bench_coverage["history_dates_complete"],
            missing_issuer_dates=issuer_coverage["missing_dates"],
            missing_benchmark_dates=bench_coverage["missing_dates"],
        )
        if not identity_ok:
            reasons.append("missing_or_ambiguous_price_target")
        required = issuer_coverage["required_dates"]
        evidence: dict[str, Any] = {}
        action: dict[str, Any] = {}
        price_audit: dict[str, Any] = {}
        if identity_ok:
            prows = price_audits.loc[price_audits.price_target_id.eq(t["price_target_id"])]
            if len(prows) == 1:
                price_audit = _records(prows)[0]
            arows = actions.loc[actions.price_target_id.eq(t["price_target_id"])]
            if len(arows) == 1:
                action = _records(arows)[0]
                erows = continuity.loc[continuity.action_target_id.eq(action["action_target_id"])]
                if len(erows) == 1:
                    evidence = _records(erows)[0]
        query_ready = identity_ok and price_query_covers_history(
            {
                "price_target_id": t["price_target_id"],
                "cik": e["cik"],
                "ticker": e["assigned_ticker"],
            },
            price_audit,
            required,
        )
        numeric_ready = (
            issuer_coverage["history_dates_complete"]
            and bench_coverage["history_dates_complete"]
            and identity_ok
            and t["adjusted"] is False
            and benchmark_raw
        )
        interval_ready = bool(
            required
            and identity_ok
            and action
            and evidence
            and action["cik"] == evidence["cik"] == e["cik"]
            and action["selected_ticker"] == evidence["ticker"] == e["assigned_ticker"]
            and _date_interval(evidence, required[0], required[-1])
            and action["splits_query_complete"] is True
            and action["dividends_query_complete"] is True
            and action["price_window_start"]
            <= required[0]
            <= required[-1]
            <= action["price_window_end"]
            and t["price_window_start"] <= required[0] <= required[-1] <= t["price_window_end"]
            and bench_queries.window_start.le(required[0]).all()
            and bench_queries.window_end.ge(required[-1]).all()
        )
        row["history_identity_action_interval_ready"] = interval_ready
        row["history_price_query_ready"] = query_ready
        if not query_ready:
            reasons.append("issuer_price_query_incomplete_or_outside_interval")
        if not benchmark_raw:
            reasons.append("benchmark_query_metadata_not_ready")
        if not issuer_coverage["history_dates_complete"]:
            reasons.append("issuer_missing_exact_61_session_history")
        if not bench_coverage["history_dates_complete"]:
            reasons.append("benchmark_missing_exact_61_session_history")
        if not interval_ready:
            reasons.append("historical_identity_or_action_interval_not_covered")
        if numeric_ready and action:
            try:
                event = {
                    "validation_event_id": event_id,
                    "cik": e["cik"],
                    "ticker": e["assigned_ticker"],
                    "exchange_mic": e["assigned_exchange_mic"],
                    "price_target_id": t["price_target_id"],
                    "benchmark_target_id": benchmark_id,
                    "announcement_at": proxy,
                }
                frames = []
                for path, key, value in (
                    (price_path, "price_target_id", t["price_target_id"]),
                    (benchmark_path, "benchmark_target_id", benchmark_id),
                ):
                    frames.append(
                        pd.read_parquet(
                            path,
                            columns=BAR_VALUE_COLUMNS,
                            filters=[
                                (key, "==", value),
                                ("session_date", ">=", required[0]),
                                ("session_date", "<=", required[-1]),
                            ],
                        )
                    )
                if not benchmark_splits.loc[
                    benchmark_splits.execution_date.between(required[0], required[-1])
                ].empty:
                    raise ValueError("benchmark_splits_require_separate_adapter")
                candidate_day = (
                    utc_timestamp(proxy).tz_convert("America/New_York").date().isoformat()
                )
                eligible_splits = splits.loc[
                    splits.action_target_id.eq(action["action_target_id"])
                    & splits.execution_date.lt(candidate_day)
                ]
                state, _ = rebuild_pre_event_state(
                    event,
                    frames[0],
                    frames[1],
                    eligible_splits,
                    sessions,
                    price_basis=RAW_PRICE_BASIS,
                )
                states.append(state)
                row["market_features_calculable"] = bool(state["market_state_ready"])
                row["market_history_evidence_ready"] = bool(
                    state["market_state_ready"] and interval_ready and query_ready
                )
                row["market_feature_status"] = state["market_state_status"]
            except ValueError as exc:
                row["market_feature_status"] = str(exc)
                reasons.append("invalid_market_feature_inputs")
        if not row["market_features_calculable"]:
            reasons.append("market_features_not_calculable")
        # Execution dates are reviewed using dates and listing metadata only, never future prices.
        try:
            window = execution_window(proxy, e["assigned_exchange_mic"], sessions)
            holding = calendar.loc[
                calendar.session_date.between(
                    window["entry_session_date"], window["exit_session_date"]
                )
            ].session_date.tolist()
            row["entry_session_date"], row["exit_session_date"] = (
                window["entry_session_date"],
                window["exit_session_date"],
            )
            row["missing_execution_issuer_dates"] = sorted(set(holding) - set(pdates.session_date))
            row["missing_execution_benchmark_dates"] = sorted(
                set(holding) - set(benchmark_dates.session_date)
            )
            row["execution_dates_present_only"] = (
                len(holding) == 20
                and not row["missing_execution_issuer_dates"]
                and not row["missing_execution_benchmark_dates"]
            )
            if row["missing_execution_issuer_dates"]:
                reasons.append("execution_issuer_dates_missing")
            if row["missing_execution_benchmark_dates"]:
                reasons.append("execution_benchmark_dates_missing")
            outcomes = listing.loc[
                listing.cik.eq(e["cik"]) & listing.ticker_at_event.eq(e["assigned_ticker"])
            ]
            outcome = _records(outcomes)[0] if len(outcomes) == 1 else {}
            delisted_at = outcome.get("delisting_date")
            row["delisting_inside_candidate_window"] = bool(
                delisted_at
                and window["entry_session_date"] <= delisted_at <= window["exit_session_date"]
            )
            row["terminal_value_gap"] = (
                row["delisting_inside_candidate_window"]
                and outcome.get("terminal_value_observed") is not True
            )
            if row["terminal_value_gap"]:
                reasons.append("terminal_value_missing")
        except ValueError:
            row["execution_dates_present_only"] = False
            reasons.append("execution_calendar_unresolved")
        row["local_replay_candidate"] = bool(
            row["release_value_pair_verified"]
            and row["existing_forecast_clocks_before_proxy"]
            and row["legacy_all_representations_ready"]
            and row["market_history_evidence_ready"]
        )
        reasons.extend(["first_public_not_certified", "financial_history_replay_not_completed"])
        row["blocking_reasons"] = reasons
        rows.append(row)
        gaps.extend(gap_records(row))
    frame = pd.DataFrame(rows)
    counts = {
        k: int(frame[k].fillna(False).sum())
        for k in (
            "release_value_pair_verified",
            "existing_forecast_clocks_before_proxy",
            "legacy_all_representations_ready",
            "issuer_history_dates_complete",
            "benchmark_history_dates_complete",
            "market_features_calculable",
            "market_history_evidence_ready",
            "execution_dates_present_only",
            "local_replay_candidate",
        )
    }
    summary = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "events_retained": len(rows),
        "issuers": int(frame.cik.nunique()),
        "earlier_release_candidates": int(frame.clock_source.eq("item_2_02_8k").sum()),
        "cells_rechecked_against_raw": len(cells),
        "stage_counts": counts,
        "reviewed_pair_tickers": frame.loc[
            frame.release_value_pair_verified, "assigned_ticker"
        ].tolist(),
        "local_replay_candidate_tickers": frame.loc[
            frame.local_replay_candidate, "assigned_ticker"
        ].tolist(),
        "nonexclusive_gap_counts": dict(Counter(g["gap"] for g in gaps)),
        "first_public_certified": False,
        "network_requests": 0,
        "post_event_numeric_prices_loaded": False,
        "returns_computed": False,
        "model_fitted": False,
        "model_admission": False,
        "trading_admission": False,
        "old_forecasts_promoted": False,
        "originals_overwritten": False,
        "limitation": (
            "Local replay candidates are debugging priorities, not a validated trading sample. "
            "Missing Item 2.02 candidates do not certify first publication. "
            "Complete price dates do not certify executable prices. "
            "Existing forecast metadata is only a precheck; history, model and sigma "
            "must be rebuilt before any return study."
        ),
    }
    payloads = {
        "summary.json": summary,
        "event_admission_ledger.json": rows,
        "data_gap_ledger.json": gaps,
        "rechecked_metric_cells.json": cells,
        "rechecked_document_kinds.json": dispositions,
        "candidate_market_states.json": _records(pd.DataFrame(states)),
    }
    return write_bundle(
        output_root,
        VERSION,
        {
            name: _encode(_records(pd.DataFrame(value)) if isinstance(value, list) else value)
            for name, value in payloads.items()
        },
    ), summary


def main() -> None:
    """Run the cached-evidence-only audit; do not collect data or retrain a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/announcement_admission_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_admission")
    )
    args = parser.parse_args()
    root, report = build_audit(args.config, args.output_root)
    print(
        f"ANNOUNCEMENT AUDIT events={report['events_retained']} "
        f"pairs={report['stage_counts']['release_value_pair_verified']} "
        f"market={report['stage_counts']['market_features_calculable']} "
        f"replay_candidates={report['stage_counts']['local_replay_candidate']} "
        f"admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
