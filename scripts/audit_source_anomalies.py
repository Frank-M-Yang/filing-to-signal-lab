"""Frozen split/reverse-split unit checks and metadata-only execution-boundary audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.replicate_source_sensitivity import collect_plan
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.market_source_sensitivity import (
    calculate_quote_feature_proxy,
    compare_feature_proxies,
)
from sec_data.pre_event_source_audit import compare_quote_overlap, parse_pre_event_quotes
from sec_data.source_anomaly_audit import (
    classify_execution_exceptions,
    split_boundary_changes,
    split_to_cutoff_units,
)
from sec_data.validation_liquidity_fallback import yahoo_liquidity_chart_url

VERSION = "source-anomaly-audit-v1"
EXCEPTION_COLUMNS = [
    "validation_event_id",
    "cik",
    "ticker",
    "formal_entry_session_date",
    "formal_exit_session_date",
    "entry_exception_code",
    "exit_exception_code",
    "execution_exception_route",
    "formal_price_status",
]
LISTING_COLUMNS = [
    "cik",
    "ticker_at_event",
    "listing_outcome_status",
    "delisting_date",
    "terminal_value_observed",
]


def _table(
    c: dict[str, Any], group: str, filename: str, columns: list[str], filters: Any = None
) -> pd.DataFrame:
    return pd.read_parquet(
        _artifact(Path(c[f"{group}_root"]), c[f"{group}_manifest_sha256"], filename),
        columns=columns,
        filters=filters,
    )


def freeze_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Select all frozen splits and all exceptions before opening any price values."""
    c = json.loads(config_path.read_bytes())
    if (
        c["version"] != VERSION
        or c["split_tickers"] != ["PANW", "EAF"]
        or c["maximum_requests"] != 4
        or c["required_sessions"] != 61
        or c["yahoo_scenarios"] != ["as_supplied_quote", "apply_split_as_if_raw_quote"]
        or any(
            c[k] is not False
            for k in (
                "scenario_winner_selection",
                "sample_replacement",
                "provider_splicing",
                "forward_price_values_allowed",
                "post_event_returns_allowed",
                "model_admission",
                "trading_admission",
                "official_halt_evidence_available",
            )
        )
    ):
        raise ValueError("Unsupported anomaly audit contract")
    verify_file(Path(c["market_config"]), c["market_config_sha256"])
    market = json.loads(Path(c["market_config"]).read_bytes())
    splits = _table(
        market,
        "actions",
        "split_observations.parquet",
        ["action_target_id", "ticker", "execution_date", "split_from", "split_to", "source_sha256"],
    )
    if len(splits) != 2 or set(splits.ticker) != set(c["split_tickers"]):
        raise ValueError("Frozen split population changed")
    targets = _table(
        market,
        "actions",
        "massive_security_actions_targets.parquet",
        [
            "action_target_id",
            "price_target_id",
            "cik",
            "selected_ticker",
            "selected_exchange_mic",
            "price_window_start",
            "price_window_end",
        ],
    )
    events = _table(
        market,
        "execution",
        "validation_execution_events.parquet",
        ["validation_event_id", "accepted_at", "assigned_ticker", "cik", "assigned_exchange_mic"],
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
    continuity = _table(
        c,
        "continuity",
        "validation_action_evidence_status.parquet",
        [
            "action_target_id",
            "ticker",
            "cik",
            "generic_identity_ready",
            "resolved_identity_ready",
            "historical_symbol_interval_ready",
            "corporate_actions_ready",
            "identity_continuity_route",
            "authorized_entry_start",
            "authorized_exit_end",
        ],
    )
    exceptions = _table(
        c, "exceptions", "validation_execution_exceptions.parquet", EXCEPTION_COLUMNS
    )
    if len(exceptions) != 8:
        raise ValueError("Frozen exception population changed")
    cases, queries = [], []
    for ticker in c["split_tickers"]:
        a = splits.loc[splits.ticker.eq(ticker)].iloc[0].to_dict()
        target_rows = targets.loc[targets.action_target_id.eq(a["action_target_id"])]
        evidence_rows = continuity.loc[continuity.action_target_id.eq(a["action_target_id"])]
        if len(target_rows) != 1 or len(evidence_rows) != 1:
            raise ValueError("Ambiguous action target or continuity evidence")
        target, identity = target_rows.iloc[0].to_dict(), evidence_rows.iloc[0].to_dict()
        selected = events.loc[
            events.cik.eq(target["cik"]) & events.assigned_ticker.eq(ticker)
        ].sort_values(["accepted_at", "validation_event_id"])
        selected = selected.loc[
            selected.accepted_at.map(
                lambda t: utc_timestamp(t).tz_convert("America/New_York").date().isoformat()
            ).gt(a["execution_date"])
        ]
        if selected.empty:
            raise ValueError("No post-split frozen filing; do not replace")
        event = selected.iloc[0].to_dict()
        cutoff = utc_timestamp(event["accepted_at"])
        day = cutoff.tz_convert("America/New_York").date().isoformat()
        cal = (
            sessions.loc[
                sessions.exchange_mic.eq(target["selected_exchange_mic"])
                & sessions.session_date.lt(day)
            ]
            .sort_values("session_date")
            .tail(61)
        )
        dates = cal.session_date.tolist()
        if (
            len(dates) != 61
            or dates != sorted(set(dates))
            or a["execution_date"] not in dates
            or not cal.regular_close_at.map(utc_timestamp).lt(cutoff).all()
        ):
            raise ValueError("Split is not inside the full strictly prior history")
        event["accepted_at"] = cutoff.isoformat()
        case = {
            "event": event,
            "target": target,
            "action": a,
            "identity_evidence": identity,
            "sessions": dates,
        }
        cases.append(case)
        for symbol in (ticker, "SPY"):
            queries.append(
                {
                    "id": f"{ticker}_{symbol}",
                    "case_ticker": ticker,
                    "ticker": symbol,
                    "start": dates[0],
                    "end": dates[-1],
                    "cutoff_at": cutoff.isoformat(),
                    "exchange_mic": target["selected_exchange_mic"] if symbol != "SPY" else "ARCX",
                    "url": yahoo_liquidity_chart_url(symbol, dates[0], dates[-1]),
                }
            )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "market": market,
        "cases": cases,
        "queries": queries,
        "exception_event_ids": exceptions.validation_event_id.tolist(),
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/audit_source_anomalies.py",
                "sec_data/source_anomaly_audit.py",
                "scripts/replicate_source_sensitivity.py",
                "sec_data/market_source_sensitivity.py",
                "sec_data/pre_event_source_audit.py",
                "scripts/build_sec_clock_layer.py",
                "scripts/build_sec_disclosure_audit.py",
            )
        },
    }
    return write_bundle(output_root, "source-anomaly-plan-v1", {"plan.json": _encode(plan)}), plan


def _numeric_case(
    case: dict[str, Any], market: dict[str, Any], yahoo: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, Any]]:
    t, event, dates = case["target"], case["event"], case["sessions"]
    ticker = t["selected_ticker"]
    i = case["identity_evidence"]
    identity_ready = bool(
        i["resolved_identity_ready"]
        and i["historical_symbol_interval_ready"]
        and i["corporate_actions_ready"]
        and i["authorized_entry_start"] <= dates[0]
        and i["authorized_exit_end"] >= dates[-1]
    )
    if not identity_ready:
        raise ValueError("Continuity evidence does not cover the selected interval")
    action_audit = _table(
        market,
        "actions",
        "sample_security_actions_audit.parquet",
        ["action_target_id", "splits_query_complete", "dividends_query_complete"],
        [("action_target_id", "==", t["action_target_id"])],
    )
    if len(action_audit) != 1 or not action_audit[
        ["splits_query_complete", "dividends_query_complete"]
    ].all(axis=None):
        raise ValueError("Corporate-action queries incomplete")
    benchmark_splits = _table(
        market,
        "benchmark",
        "benchmark_split_observations.parquet",
        ["execution_date"],
        [("execution_date", ">=", dates[0]), ("execution_date", "<=", dates[-1])],
    )
    if not benchmark_splits.empty:
        raise ValueError("Benchmark splits require a separate specification")
    primary = {}
    for symbol in (ticker, "SPY"):
        group, filename, filters = (
            (
                "prices",
                "validation_price_batch_bars.parquet",
                [("price_target_id", "==", t["price_target_id"])],
            )
            if symbol == ticker
            else (
                "benchmark",
                "benchmark_daily_price_observations.parquet",
                [("ticker", "==", "SPY")],
            )
        )
        primary[symbol] = (
            _table(
                market,
                group,
                filename,
                ["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"],
                [*filters, ("session_date", ">=", dates[0]), ("session_date", "<=", dates[-1])],
            )
            .rename(columns={"close_unadjusted": "quote_close"})
            .sort_values("session_date")
            .reset_index(drop=True)
        )
        if (
            primary[symbol].session_date.tolist() != dates
            or yahoo[symbol].session_date.tolist() != dates
        ):
            raise ValueError("Incomplete official history; no filling or shortened windows")
    actions = pd.DataFrame([case["action"]])
    normalized, ledger = split_to_cutoff_units(primary[ticker], actions, event["accepted_at"])
    alternative, _ = split_to_cutoff_units(yahoo[ticker], actions, event["accepted_at"])
    series = {
        "massive_raw": primary[ticker],
        "massive_cutoff_share_units": normalized,
        "yahoo_as_supplied": yahoo[ticker],
        "yahoo_split_applied_again_hypothesis": alternative,
    }
    features = {
        name: calculate_quote_feature_proxy(
            frame,
            yahoo["SPY"] if name.startswith("yahoo") else primary["SPY"],
            dates,
            event["accepted_at"],
            issuer_ticker=ticker,
        )
        for name, frame in series.items()
    }
    comparisons = {
        name: compare_feature_proxies(features["massive_cutoff_share_units"], f)
        for name, f in features.items()
        if name != "massive_cutoff_share_units"
    }
    overlap, _ = compare_quote_overlap(normalized, yahoo[ticker], minimum_sessions=61)
    ratios = primary[ticker][["session_date", "quote_close", "volume"]].merge(
        yahoo[ticker][["session_date", "quote_close", "volume"]],
        on="session_date",
        suffixes=("_primary", "_yahoo"),
        validate="one_to_one",
    )
    ratios["price_ratio_yahoo_to_raw"] = ratios.quote_close_yahoo / ratios.quote_close_primary
    ratios["volume_ratio_yahoo_to_raw"] = ratios.volume_yahoo / ratios.volume_primary.where(
        ratios.volume_primary.gt(0)
    )
    ratios["side"] = ratios.session_date.map(
        lambda d: "before_split" if d < case["action"]["execution_date"] else "from_split_onward"
    )
    medians = (
        ratios.groupby("side", sort=True)[["price_ratio_yahoo_to_raw", "volume_ratio_yahoo_to_raw"]]
        .median()
        .reset_index()
    )
    boundary = split_boundary_changes(series, actions, dates)
    report = {
        "ticker": ticker,
        "status": "compared",
        "validation_event_id": event["validation_event_id"],
        "cutoff_at": event["accepted_at"],
        "required_sessions": 61,
        "source_comparison_is_retrospective": True,
        "first_public_certified": False,
        "identity_route": i["identity_continuity_route"],
        "generic_identity_ready": bool(i["generic_identity_ready"]),
        "interval_continuity_corroborated": identity_ready,
        "split_ledger": ledger,
        "boundary_close_changes": boundary,
        "share_unit_ratios": _records(medians),
        "normalized_primary_vs_yahoo_overlap": overlap,
        "all_scenarios_reported_no_winner_selected": True,
        "model_admission": False,
    }
    details = {
        "features": {name: _records(f) for name, f in features.items()},
        "comparisons": {name: _records(f) for name, f in comparisons.items()},
        "daily_unit_ratios": _records(ratios),
    }
    return report, details


def audit_capture(
    capture_root: Path, capture_hash: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Replay immutable quotes, retaining blocked cases and refusing missing-price inference."""
    capture = _json(capture_root, capture_hash, "receipts.json")
    plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    if [r["query_id"] for r in capture["receipts"]] != [q["id"] for q in plan["queries"]]:
        raise ValueError("Capture query list changed")
    frames: dict[str, dict[str, pd.DataFrame]] = {t: {} for t in plan["config"]["split_tickers"]}
    failures: dict[str, list[str]] = {t: [] for t in frames}
    for q, r in zip(plan["queries"], capture["receipts"], strict=True):
        if q["url"] != r["requested_url"]:
            raise ValueError("Capture URL changed")
        if r["status_code"] != 200:
            failures[q["case_ticker"]].append(f"{q['id']}: response_not_available")
            continue
        path = _artifact(capture_root, capture_hash, r["file"])
        verify_file(path, r["sha256"])
        try:
            frames[q["case_ticker"]][q["ticker"]] = parse_pre_event_quotes(
                path.read_bytes(),
                q["ticker"],
                q["start"],
                q["end"],
                q["cutoff_at"],
                expected_exchange_mic=q["exchange_mic"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures[q["case_ticker"]].append(f"{q['id']}: invalid_quote_{type(exc).__name__}")
    cases, details = [], {}
    for case in plan["cases"]:
        ticker = case["target"]["selected_ticker"]
        report = {
            "ticker": ticker,
            "status": "blocked",
            "errors": failures[ticker],
            "model_admission": False,
        }
        if not failures[ticker]:
            try:
                report, details[ticker] = _numeric_case(case, plan["market"], frames[ticker])
            except ValueError as exc:
                report["errors"] = [str(exc)]
        cases.append(report)
    exceptions = _table(
        plan["config"], "exceptions", "validation_execution_exceptions.parquet", EXCEPTION_COLUMNS
    )
    if exceptions.validation_event_id.tolist() != plan["exception_event_ids"]:
        raise ValueError("Exception denominator changed")
    listing = _table(
        plan["market"], "listing", "validation_security_listing_outcomes.parquet", LISTING_COLUMNS
    )
    boundary = classify_execution_exceptions(exceptions, listing)
    summary = {
        "version": VERSION,
        "capture_root": capture_root.as_posix(),
        "capture_manifest_sha256": capture_hash,
        "plan_manifest_sha256": capture["plan_manifest_sha256"],
        "split_cases_selected": len(cases),
        "split_cases_compared": sum(r["status"] == "compared" for r in cases),
        "cases": cases,
        "exceptions_retained": len(boundary),
        "dated_delistings_inside_execution_window": int(
            boundary.dated_delisting_inside_window.sum()
        ),
        "terminal_values_missing_when_required": int(
            boundary.terminal_value_required_but_unobserved.sum()
        ),
        "official_halts_verified": 0,
        "cutoff_basis": plan["config"]["cutoff_basis"],
        "data_quality_diagnostic_not_filing_predictor": True,
        "post_cutoff_numeric_prices_loaded": False,
        "post_event_returns_computed": False,
        "sample_replaced": False,
        "provider_splicing": False,
        "model_fitted": False,
        "main_sample_changed": False,
        "model_admission": False,
        "trading_admission": False,
        "limitation": (
            "Historical share-unit diagnostics only. All Yahoo adjustment hypotheses retained; "
            "no automatic basis selection. Missing or zero-volume prices do not prove a halt. "
            "A dated delisting does not provide a terminal payoff."
        ),
    }
    files = {
        "summary.json": _encode(summary),
        "split_feature_details.json": _encode(details),
        "execution_boundary_audit.json": _encode(_records(boundary)),
    }
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Freeze by default; use explicit collection or a paired cached capture root/hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/source_anomaly_audit_v1.json"))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest")
    parser.add_argument("--raw-root", type=Path, default=Path("data/market/source_anomaly_audit"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/source_anomaly_audit")
    )
    args = parser.parse_args()
    if (args.collect and args.capture_root) or (args.capture_root is None) != (
        args.capture_manifest is None
    ):
        parser.error("Choose collection or a paired offline capture root/hash")
    if args.collect or args.capture_root is None:
        root, _ = freeze_plan(args.config, args.raw_root / "plans")
        digest = digest_file(root / "output_manifest.json")
        print(f"ANOMALY PLAN output={root} manifest={digest}")
        if not args.collect:
            return
        args.capture_root = collect_plan(root, digest, args.raw_root)
        args.capture_manifest = digest_file(args.capture_root / "output_manifest.json")
        print(f"ANOMALY CAPTURE output={args.capture_root} manifest={args.capture_manifest}")
    output, report = audit_capture(args.capture_root, args.capture_manifest, args.output_root)
    print(
        f"ANOMALY DONE splits={report['split_cases_compared']}/{report['split_cases_selected']} "
        f"exceptions={report['exceptions_retained']} "
        f"missing_terminal={report['terminal_values_missing_when_required']} "
        f"admitted=False output={output}"
    )


if __name__ == "__main__":
    main()
