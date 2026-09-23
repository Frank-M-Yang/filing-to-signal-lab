"""Offline same-window market-feature sensitivity, separate from the canonical sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.market_source_sensitivity import (
    WINDOW_CLOSES,
    calculate_quote_feature_proxy,
    compare_feature_proxies,
)
from sec_data.pre_event_primary_evidence import (
    END,
    START,
    audit_cached_timeline,
    audit_primary_response,
)
from sec_data.pre_event_source_audit import QUOTE_COLUMNS, parse_pre_event_quotes
from sec_data.publisher_market_rebuild import rebuild_pre_event_state
from sec_data.validation_market_state import MARKET_STATE_FEATURES

VERSION = "mbuu-source-sensitivity-v1"


def _load(root: str, digest: str, name: str) -> Any:
    return _json(Path(root), digest, name)


def _config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c = json.loads(path.read_bytes())
    if (
        c["version"] != VERSION
        or c["required_sessions"] != 61
        or c["economic_acceptance_threshold"] is not None
        or c["price_basis"]
        != "quote_close_conditional_proxy_not_total_return_or_unadjusted_certification"
        or c["comparison_policy"] != "same_event_same_formula_same_exact_session_window_per_feature"
        or any(
            c[k] is not False
            for k in (
                "network_allowed",
                "provider_splicing",
                "shorten_feature_lookbacks",
                "returns_allowed",
                "model_fitting_allowed",
                "model_admission",
                "trading_admission",
            )
        )
    ):
        raise ValueError("Unsupported source-sensitivity contract")
    for group in ("repair_plan", "market_config"):
        verify_file(Path(c[group]), c[f"{group}_sha256"])
    return (
        c,
        json.loads(Path(c["repair_plan"]).read_bytes()),
        json.loads(Path(c["market_config"]).read_bytes()),
    )


def _primary_evidence(c: dict[str, Any], dates: list[str]) -> dict[str, Any]:
    root, digest = c["primary_capture_root"], c["primary_capture_manifest_sha256"]
    capture = _load(root, digest, "receipts.json")
    plan = _load(capture["plan_root"], capture["plan_manifest_sha256"], "plan.json")
    expected = [
        "MBUU_bars",
        "SPY_bars",
        "MBUU_reference_start",
        "MBUU_reference_end",
        "MBUU_splits",
        "MBUU_dividends",
        "SPY_splits",
        "SPY_dividends",
    ]
    if (
        plan["required_sessions"] != dates
        or [q["id"] for q in plan["queries"]] != expected
        or [r["query_id"] for r in capture["receipts"]] != expected
    ):
        raise ValueError("Primary capture/calendar/query mismatch")
    parsed = {}
    for query, receipt in zip(plan["queries"], capture["receipts"], strict=True):
        if receipt["status_code"] != 200 or receipt["requested_url"] != query["url"]:
            raise ValueError("Primary evidence query did not succeed as planned")
        path = _artifact(Path(root), digest, receipt["file"])
        verify_file(path, receipt["sha256"])
        result = audit_primary_response(path.read_bytes(), query, dates)
        if result["status"] != "parsed" or result["rejections"] or result["pagination_remaining"]:
            raise ValueError("Primary evidence contains unparsed/rejected/incomplete rows")
        if query["kind"] != "bars" and not result["evidence_complete"]:
            raise ValueError("Identity or corporate-action evidence incomplete")
        if query["kind"] == "splits" and result["observed_rows"] != 0:
            raise ValueError("This fixed quote sensitivity requires a split-free interval")
        parsed[query["id"]] = result
    cfg = plan["config"]
    verify_file(Path(cfg["ticker_events_raw"]), cfg["ticker_events_sha256"])
    timeline = audit_cached_timeline(Path(cfg["ticker_events_raw"]).read_bytes())
    if not timeline["consistent"]:
        raise ValueError("Primary ticker timeline does not corroborate identity")
    return parsed


def build_sensitivity(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Use pinned cached pre-event data only; never patch an existing market feature store."""
    c, repair, market = _config(config_path)
    root, digest = Path(repair["rebuild_root"]), repair["rebuild_manifest_sha256"]
    event = _json(root, digest, "source_evidence.json")["event"]
    if (
        event["ticker"] != "MBUU"
        or event["cik"] != "0001590976"
        or event["validation_event_id"] != repair["validation_event_id"]
        or utc_timestamp(event["announcement_at"]) != utc_timestamp("2024-10-31T11:00Z")
    ):
        raise ValueError("Frozen MBUU event identity/clock mismatch")
    calendar = pd.read_parquet(
        _artifact(root, digest, "market_history_session_audit.parquet"),
        columns=["session_date", "regular_open_at", "regular_close_at", "source_snapshot_sha256"],
    )
    calendar["exchange_mic"] = event["exchange_mic"]
    dates = calendar.session_date.tolist()
    if len(dates) != 61 or dates != sorted(set(dates)) or (dates[0], dates[-1]) != (START, END):
        raise ValueError("Frozen calendar is not the full 61-session requirement")
    if (
        not calendar.regular_close_at.map(utc_timestamp)
        .lt(utc_timestamp(event["announcement_at"]))
        .all()
    ):
        raise ValueError("Non-prior session close")
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "event": event,
        "required_sessions": dates,
        "feature_required_closes": WINDOW_CLOSES,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/build_mbuu_source_sensitivity.py",
                "sec_data/market_source_sensitivity.py",
                "sec_data/pre_event_source_audit.py",
                "sec_data/pre_event_primary_evidence.py",
                "sec_data/publisher_market_rebuild.py",
                "sec_data/validation_market_state.py",
                "scripts/build_sec_clock_layer.py",
                "scripts/build_sec_disclosure_audit.py",
                "sec_data/massive_prices.py",
                "sec_data/massive_reference.py",
                "sec_data/massive_security_actions.py",
            )
        },
        "feature_math_parity_tolerance": {"absolute": 1e-12, "relative": 0},
        "parity_tolerance_is_not_provider_acceptance_threshold": True,
    }
    frozen = write_bundle(
        output_root / "plans", "mbuu-source-sensitivity-plan-v1", {"plan.json": _encode(plan)}
    )
    evidence = _primary_evidence(c, dates)
    yahoo_capture = _load(
        c["yahoo_capture_root"], c["yahoo_capture_manifest_sha256"], "receipts.json"
    )
    if [r["ticker"] for r in yahoo_capture["receipts"]] != ["MBUU", "SPY"]:
        raise ValueError("Ambiguous Yahoo capture")
    frames: dict[str, dict[str, pd.DataFrame]] = {
        name: {} for name in ("yahoo_only", "massive_cached", "massive_refreshed")
    }
    for ticker, receipt in zip(("MBUU", "SPY"), yahoo_capture["receipts"], strict=True):
        path = _artifact(
            Path(c["yahoo_capture_root"]), c["yahoo_capture_manifest_sha256"], receipt["file"]
        )
        verify_file(path, receipt["sha256"])
        if receipt["status_code"] != 200:
            raise ValueError("Yahoo response did not succeed")
        quotes = parse_pre_event_quotes(
            path.read_bytes(), ticker, START, END, event["announcement_at"]
        )
        if quotes.session_date.tolist() != dates:
            raise ValueError("Yahoo lacks exact official sessions; no fill or splice is allowed")
        frames["yahoo_only"][ticker] = quotes
        group, filename, key = (
            ("prices", "validation_price_batch_bars.parquet", "price_target_id")
            if ticker == "MBUU"
            else ("benchmark", "benchmark_daily_price_observations.parquet", "benchmark_target_id")
        )
        # Filter and project before loading numeric values. Never open outcomes.
        frames["massive_cached"][ticker] = (
            pd.read_parquet(
                _artifact(
                    Path(market[f"{group}_root"]), market[f"{group}_manifest_sha256"], filename
                ),
                columns=["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"],
                filters=[
                    (key, "==", event[key]),
                    ("session_date", ">=", START),
                    ("session_date", "<=", END),
                ],
            )
            .rename(columns={"close_unadjusted": "quote_close"})
            .sort_values("session_date")
            .reset_index(drop=True)
        )
        refreshed = pd.DataFrame(evidence[f"{ticker}_bars"]["rows"]).rename(
            columns={"close_unadjusted": "quote_close"}
        )
        frames["massive_refreshed"][ticker] = (
            refreshed[QUOTE_COLUMNS].sort_values("session_date").reset_index(drop=True)
        )
    features = {
        name: calculate_quote_feature_proxy(
            pair["MBUU"], pair["SPY"], dates, event["announcement_at"]
        )
        for name, pair in frames.items()
    }
    # Private, temporary adapter checks algebra only. It does not certify Yahoo's
    # price semantics; no canonical-ready flag is exported from this check.
    kernel, _ = rebuild_pre_event_state(
        event,
        frames["yahoo_only"]["MBUU"].rename(columns={"quote_close": "close_unadjusted"}),
        frames["yahoo_only"]["SPY"].rename(columns={"quote_close": "close_unadjusted"}),
        pd.DataFrame(),
        calendar,
        price_basis="diagnostic_quote_no_splits",
    )
    if not np.allclose(
        features["yahoo_only"].value.to_numpy(float),
        [kernel[f] for f in MARKET_STATE_FEATURES],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Proxy equations differ from the frozen 13-feature kernel")
    comparisons = {
        "yahoo_vs_cached": compare_feature_proxies(
            features["massive_cached"], features["yahoo_only"]
        ),
        "yahoo_vs_refreshed": compare_feature_proxies(
            features["massive_refreshed"], features["yahoo_only"]
        ),
        "refreshed_vs_cached": compare_feature_proxies(
            features["massive_cached"], features["massive_refreshed"]
        ),
    }
    all_compared = pd.concat(comparisons.values(), ignore_index=True)
    report = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "event": event,
        "events_tested": 1,
        "required_sessions": 61,
        "source_rows": {
            name: {t: len(f) for t, f in pair.items()} for name, pair in frames.items()
        },
        "proxy_features_available": {
            name: int(f.value.notna().sum()) for name, f in features.items()
        },
        "comparable_features": {name: int(f.comparable.sum()) for name, f in comparisons.items()},
        "primary_unverified_features": features["massive_cached"]
        .loc[features["massive_cached"].value.isna(), "feature"]
        .tolist(),
        "maximum_absolute_price_feature_difference_basis_points": float(
            all_compared.difference_basis_points.abs().max()
        ),
        "maximum_absolute_implied_dollar_volume_difference_pct": float(
            all_compared.implied_dollar_volume_difference_pct.abs().max()
        ),
        "volume_ratio_differences": {
            name: float(
                f.loc[f.feature.eq("pre_volume_ratio_5_20"), "candidate_minus_primary"].iloc[0]
            )
            for name, f in comparisons.items()
        },
        "corporate_action_counts": {
            q: p["observed_rows"]
            for q, p in evidence.items()
            if q.endswith(("_splits", "_dividends"))
        },
        "identity_corroboration": (
            "MBUU historical endpoints and experimental ticker timeline agree; "
            "not exchange-certified continuity"
        ),
        "price_semantics": c["price_basis"],
        "dividends_reinvested": False,
        "math_matches_frozen_kernel": True,
        "economic_acceptance_threshold": None,
        "first_public_certified": False,
        "vendor_delivery_times_certified": False,
        "historical_sources_are_vintage_snapshots": False,
        "main_sample_changed": False,
        "provider_splicing": False,
        "lookbacks_shortened": False,
        "forward_price_values_loaded": False,
        "post_event_returns_computed": False,
        "model_fitted": False,
        "market_feature_admission": False,
        "model_admission": False,
        "trading_admission": False,
        "network_requests": 0,
        "limitations": [
            "One event is a source diagnostic, not statistical or economic effectiveness evidence.",
            "Seven available short-window comparisons cannot verify the six long-window features.",
            "Yahoo quote.close is a conditional proxy, not certified historically unadjusted data.",
            "The frozen kernel uses price changes, not dividend-reinvested total returns.",
            "No provider was selected or approved on the basis of these differences.",
        ],
    }
    files = {
        "summary.json": _encode(report),
        "feature_values.json": _encode({k: _records(v) for k, v in features.items()}),
        "feature_comparisons.json": _encode({k: _records(v) for k, v in comparisons.items()}),
        "corporate_action_evidence.json": _encode(
            {k: v for k, v in evidence.items() if k.endswith(("_splits", "_dividends"))}
        ),
    }
    return write_bundle(output_root, VERSION, files), report


def main() -> None:
    """Run the bounded offline experiment with no key, network flag or return access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/mbuu_source_sensitivity_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/mbuu_source_sensitivity")
    )
    args = parser.parse_args()
    root, report = build_sensitivity(args.config, args.output_root)
    print(
        f"SOURCE SENSITIVITY proxy_features={report['proxy_features_available']} "
        f"comparable={report['comparable_features']} admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
