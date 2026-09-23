"""Two fixed publisher-clock cases: bounded public collection and offline source comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.market_source_sensitivity import (
    calculate_quote_feature_proxy,
    compare_feature_proxies,
)
from sec_data.pre_event_source_audit import compare_quote_overlap, parse_pre_event_quotes
from sec_data.publisher_market_rebuild import rebuild_pre_event_state
from sec_data.validation_liquidity_fallback import yahoo_liquidity_chart_url
from sec_data.validation_market_state import MARKET_STATE_FEATURES

VERSION = "source-sensitivity-replication-v1"


def freeze_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze the two specified cases using identity/calendar metadata, before prices."""
    c = json.loads(config_path.read_bytes())
    if (
        c["version"] != VERSION
        or [x["ticker"] for x in c["cases"]] != ["USNA", "UFPT"]
        or c["required_sessions"] != 61
        or c["maximum_requests"] != 4
        or c["price_absolute_tolerance_usd"] != 0.01
        or c["volume_absolute_tolerance"] != 0
        or c["economic_acceptance_threshold"] is not None
        or c["retries"] != 0
        or c["collection_requires_explicit_flag"] is not True
        or any(
            c[k] is not False
            for k in (
                "control_in_new_case_denominator",
                "follow_redirects",
                "sample_replacement",
                "provider_splicing",
                "shorten_lookbacks",
                "post_event_returns_allowed",
                "model_fitting_allowed",
                "model_admission",
                "trading_admission",
            )
        )
    ):
        raise ValueError("Unsupported replication contract")
    cases, queries = [], []
    for selected in c["cases"]:
        verify_file(Path(selected["admission_config"]), selected["admission_config_sha256"])
        admission = json.loads(Path(selected["admission_config"]).read_bytes())
        verify_file(Path(admission["market_config"]), admission["market_config_sha256"])
        market = json.loads(Path(admission["market_config"]).read_bytes())
        root, digest = Path(admission["rebuild_root"]), admission["rebuild_manifest_sha256"]
        event = _json(root, digest, "source_evidence.json")["event"]
        if event["ticker"] != selected["ticker"]:
            raise ValueError("Selected ticker and reviewed event differ")
        calendar = pd.read_parquet(
            _artifact(root, digest, "market_history_session_audit.parquet"),
            columns=[
                "session_date",
                "regular_open_at",
                "regular_close_at",
                "source_snapshot_sha256",
            ],
        )
        dates = calendar.session_date.tolist()
        boundary = utc_timestamp(event["announcement_at"])
        if (
            len(dates) != 61
            or dates != sorted(set(dates))
            or dates[-1] >= boundary.tz_convert("America/New_York").date().isoformat()
            or not calendar.regular_close_at.map(utc_timestamp).lt(boundary).all()
        ):
            raise ValueError("Expected 61 ordered pre-event official sessions")
        case = {
            "event": event,
            "market": market,
            "rebuild_root": root.as_posix(),
            "rebuild_manifest_sha256": digest,
            "sessions": dates,
        }
        cases.append(case)
        for ticker in (event["ticker"], "SPY"):
            queries.append(
                {
                    "id": f"{event['ticker']}_{ticker}",
                    "case_ticker": event["ticker"],
                    "ticker": ticker,
                    "start": dates[0],
                    "end": dates[-1],
                    "announcement_at": event["announcement_at"],
                    "exchange_mic": event["exchange_mic"] if ticker != "SPY" else "ARCX",
                    "url": yahoo_liquidity_chart_url(ticker, dates[0], dates[-1]),
                }
            )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": cases,
        "queries": queries,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/replicate_source_sensitivity.py",
                "sec_data/market_source_sensitivity.py",
                "sec_data/pre_event_source_audit.py",
                "sec_data/publisher_market_rebuild.py",
                "sec_data/validation_market_state.py",
                "sec_data/validation_liquidity_fallback.py",
                "scripts/build_sec_clock_layer.py",
                "scripts/build_sec_disclosure_audit.py",
            )
        },
    }
    return write_bundle(
        output_root, "source-replication-plan-v1", {"plan.json": _encode(plan)}
    ), plan


def collect_plan(plan_root: Path, plan_hash: str, output_root: Path) -> Path:
    """Capture four fixed public responses; no keys, retries, redirects or access workaround."""
    plan = _json(plan_root, plan_hash, "plan.json")
    queries = plan["queries"]
    if len(queries) != 4 or any(
        q["url"] != yahoo_liquidity_chart_url(q["ticker"], q["start"], q["end"]) for q in queries
    ):
        raise ValueError("Unexpected collection plan")
    files: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    stop = None
    with httpx.Client(
        timeout=25,
        follow_redirects=False,
        headers={"User-Agent": "Filing-to-Signal-Lab/1.0", "Accept": "application/json"},
    ) as client:
        for i, q in enumerate(queries):
            r: dict[str, Any] = {
                "query_id": q["id"],
                "requested_url": q["url"],
                "status_code": None,
            }
            if stop:
                r["skipped"] = stop
            else:
                if i:
                    time.sleep(1)
                r["fetched_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    response = client.get(q["url"])
                except httpx.RequestError as exc:
                    r["error"] = type(exc).__name__
                else:
                    filename = f"{q['id']}.json"
                    files[filename] = response.content
                    r.update(
                        file=filename,
                        status_code=response.status_code,
                        sha256=hashlib.sha256(response.content).hexdigest(),
                        bytes=len(response.content),
                    )
                    write_bundle(
                        output_root / "responses",
                        "source-replication-response-v1",
                        {filename: response.content, "receipt.json": _encode(r)},
                    )
                    if response.status_code in {401, 403, 429} or 300 <= response.status_code < 400:
                        stop = f"http_{response.status_code}"
            receipts.append(r)
            print(
                f"REPLICATION COLLECT {q['id']} http={r['status_code']} "
                f"error={r.get('error')} skipped={r.get('skipped')}",
                flush=True,
            )
    files["receipts.json"] = _encode(
        {"plan_root": plan_root.as_posix(), "plan_manifest_sha256": plan_hash, "receipts": receipts}
    )
    return write_bundle(output_root, "source-replication-capture-v1", files)


def _case_result(
    case: dict[str, Any], quotes: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, Any]]:
    event, dates, market = case["event"], case["sessions"], case["market"]
    ticker = event["ticker"]
    root, digest = Path(case["rebuild_root"]), case["rebuild_manifest_sha256"]
    old_state = _json(root, digest, "market_state.json")[0]
    checks = old_state["history_evidence_checks"]
    if set(checks) != {
        "identity_interval",
        "issuer_actions_interval",
        "issuer_price_query_interval",
        "benchmark_query_interval",
    } or not all(v is True for v in checks.values()):
        raise ValueError("Primary identity/action/query interval is incomplete")
    source = _json(root, digest, "source_evidence.json")
    for group, filename, filters in (
        (
            "actions",
            "split_observations.parquet",
            [("action_target_id", "==", source["action_audit"]["action_target_id"])],
        ),
        ("benchmark", "benchmark_split_observations.parquet", []),
    ):
        splits = pd.read_parquet(
            _artifact(Path(market[f"{group}_root"]), market[f"{group}_manifest_sha256"], filename),
            columns=["execution_date"],
            filters=[
                *filters,
                ("execution_date", ">=", dates[0]),
                ("execution_date", "<=", dates[-1]),
            ],
        )
        if not splits.empty:
            raise ValueError("Split adjustment requires a separately frozen experiment")
    primary = {}
    overlaps = {}
    for symbol in (ticker, "SPY"):
        group, filename, key = (
            ("prices", "validation_price_batch_bars.parquet", "price_target_id")
            if symbol == ticker
            else ("benchmark", "benchmark_daily_price_observations.parquet", "benchmark_target_id")
        )
        primary[symbol] = (
            pd.read_parquet(
                _artifact(
                    Path(market[f"{group}_root"]), market[f"{group}_manifest_sha256"], filename
                ),
                columns=["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"],
                filters=[
                    (key, "==", event[key]),
                    ("session_date", ">=", dates[0]),
                    ("session_date", "<=", dates[-1]),
                ],
            )
            .rename(columns={"close_unadjusted": "quote_close"})
            .sort_values("session_date")
            .reset_index(drop=True)
        )
        overlaps[symbol], _ = compare_quote_overlap(
            primary[symbol], quotes[symbol], minimum_sessions=61
        )
    features = {
        name: calculate_quote_feature_proxy(
            pair[ticker], pair["SPY"], dates, event["announcement_at"], issuer_ticker=ticker
        )
        for name, pair in (("massive", primary), ("yahoo", quotes))
    }
    # Check unchanged equations on complete histories; do not export the kernel's
    # temporary ready flag as approval of Yahoo's adjustment or delivery semantics.
    calendar = pd.read_parquet(
        _artifact(root, digest, "market_history_session_audit.parquet"),
        columns=["session_date", "regular_open_at", "regular_close_at", "source_snapshot_sha256"],
    ).assign(exchange_mic=event["exchange_mic"])
    for name, pair in (("massive", primary), ("yahoo", quotes)):
        if all(pair[s].session_date.tolist() == dates for s in (ticker, "SPY")):
            kernel, _ = rebuild_pre_event_state(
                event,
                pair[ticker].rename(columns={"quote_close": "close_unadjusted"}),
                pair["SPY"].rename(columns={"quote_close": "close_unadjusted"}),
                pd.DataFrame(),
                calendar,
                price_basis="diagnostic_quote_no_splits",
            )
            if not np.allclose(
                features[name].value.to_numpy(float),
                [kernel[f] for f in MARKET_STATE_FEATURES],
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            ):
                raise ValueError("Math differs from the frozen kernel")
    compared = compare_feature_proxies(features["massive"], features["yahoo"])

    def maximum(column: str) -> float | None:
        values = compared[column].dropna()
        return float(values.abs().max()) if not values.empty else None

    report = {
        "ticker": ticker,
        "validation_event_id": event["validation_event_id"],
        "status": "compared",
        "source_rows": {
            "massive": {s: len(f) for s, f in primary.items()},
            "yahoo": {s: len(f) for s, f in quotes.items()},
        },
        "comparable_features": int(compared.comparable.sum()),
        "unverified_features": compared.loc[~compared.comparable, "feature"].tolist(),
        "max_absolute_price_feature_difference_bp": maximum("difference_basis_points"),
        "max_absolute_dollar_volume_proxy_difference_pct": maximum(
            "implied_dollar_volume_difference_pct"
        ),
        "volume_ratio_difference": float(
            compared.loc[
                compared.feature.eq("pre_volume_ratio_5_20"), "candidate_minus_primary"
            ].iloc[0]
        ),
        "beta_difference": float(
            compared.loc[
                compared.feature.eq("pre_market_beta_60d"), "candidate_minus_primary"
            ].iloc[0]
        ),
        "raw_quote_overlap": overlaps,
        "model_admission": False,
    }
    details = {
        "features": {k: _records(f) for k, f in features.items()},
        "comparison": _records(compared),
    }
    return _records(pd.DataFrame([report]))[0], details


def audit_capture(
    capture_root: Path, capture_hash: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Keep every selected case in the denominator, including failed downloads/parses."""
    capture = _json(capture_root, capture_hash, "receipts.json")
    plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    if [r["query_id"] for r in capture["receipts"]] != [q["id"] for q in plan["queries"]]:
        raise ValueError("Capture and plan query lists differ")
    quotes: dict[str, dict[str, pd.DataFrame]] = {c["event"]["ticker"]: {} for c in plan["cases"]}
    errors: dict[str, list[str]] = {t: [] for t in quotes}
    for q, r in zip(plan["queries"], capture["receipts"], strict=True):
        if r["requested_url"] != q["url"]:
            raise ValueError("Capture request mismatch")
        if r["status_code"] != 200:
            errors[q["case_ticker"]].append(
                f"{q['id']}: {r.get('skipped') or r.get('error') or r['status_code']}"
            )
            continue
        path = _artifact(capture_root, capture_hash, r["file"])
        verify_file(path, r["sha256"])
        try:
            quotes[q["case_ticker"]][q["ticker"]] = parse_pre_event_quotes(
                path.read_bytes(),
                q["ticker"],
                q["start"],
                q["end"],
                q["announcement_at"],
                expected_exchange_mic=q["exchange_mic"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            errors[q["case_ticker"]].append(f"{q['id']}: invalid_quote_{type(exc).__name__}")
    reports, details = [], {}
    for case in plan["cases"]:
        ticker = case["event"]["ticker"]
        report = {
            "ticker": ticker,
            "status": "blocked",
            "comparable_features": 0,
            "errors": errors[ticker],
            "model_admission": False,
        }
        if not errors[ticker]:
            try:
                report, details[ticker] = _case_result(case, quotes[ticker])
            except ValueError as exc:
                report["errors"] = [str(exc)]
        reports.append(report)
    cfg = plan["config"]
    prior = _json(
        Path(cfg["prior_control_root"]), cfg["prior_control_manifest_sha256"], "summary.json"
    )
    report = {
        "version": VERSION,
        "capture_root": capture_root.as_posix(),
        "capture_manifest_sha256": capture_hash,
        "plan_manifest_sha256": capture["plan_manifest_sha256"],
        "audit_implementation_sha256": {
            p: digest_file(Path(p)) for p in plan["implementation_sha256"]
        },
        "new_cases_selected": len(reports),
        "new_cases_compared": sum(r["status"] == "compared" for r in reports),
        "new_cases_full_13_feature_comparison": sum(
            r["comparable_features"] == 13 for r in reports
        ),
        "cases": reports,
        "prior_mbuu_control": {
            "already_seen": True,
            "included_in_new_case_denominator": False,
            "root": cfg["prior_control_root"],
            "manifest_sha256": cfg["prior_control_manifest_sha256"],
            "comparable_features": prior["comparable_features"]["yahoo_vs_cached"],
        },
        "selection_reason": cfg["selection_reason"],
        "economic_acceptance_threshold": None,
        "sample_replaced": False,
        "provider_splicing": False,
        "main_sample_changed": False,
        "first_public_certified": False,
        "vintage_adjustment_semantics_certified": False,
        "dividends_reinvested": False,
        "post_event_prices_loaded": False,
        "post_event_returns_computed": False,
        "model_fitted": False,
        "model_admission": False,
        "trading_admission": False,
        "limitation": (
            "Two purposively selected engineering cases, no pooled significance or source "
            "endorsement. Price-only quote proxies, not certified historical unadjusted "
            "prices or total returns. Shared SPY data and related features are not "
            "independent trials."
        ),
    }
    return write_bundle(
        output_root,
        VERSION,
        {"summary.json": _encode(report), "feature_details.json": _encode(details)},
    ), report


def main() -> None:
    """Freeze by default, explicitly collect four public requests, or replay a pinned capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/source_sensitivity_replication_v1.json")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest")
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/market/source_sensitivity_replication")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/source_sensitivity_replication")
    )
    args = parser.parse_args()
    if (args.collect and args.capture_root) or (args.capture_root is None) != (
        args.capture_manifest is None
    ):
        parser.error("Choose collection or a paired offline capture root/hash")
    if args.collect or args.capture_root is None:
        root, _ = freeze_plan(args.config, args.raw_root / "plans")
        digest = digest_file(root / "output_manifest.json")
        print(f"REPLICATION PLAN output={root} manifest={digest}")
        if not args.collect:
            return
        args.capture_root = collect_plan(root, digest, args.raw_root)
        args.capture_manifest = digest_file(args.capture_root / "output_manifest.json")
        print(f"REPLICATION CAPTURE output={args.capture_root} manifest={args.capture_manifest}")
    output, report = audit_capture(args.capture_root, args.capture_manifest, args.output_root)
    print(
        f"REPLICATION DONE compared={report['new_cases_compared']}/{report['new_cases_selected']} "
        f"full13={report['new_cases_full_13_feature_comparison']} admitted=False output={output}"
    )


if __name__ == "__main__":
    main()
