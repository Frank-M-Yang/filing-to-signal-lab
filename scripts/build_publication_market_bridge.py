"""Close publication-window evidence and freeze exact requests without HTTP or returns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from scripts.build_announcement_expanded_returns import _market_metadata
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_prices import massive_daily_bars_url
from sec_data.massive_reference import MASSIVE_TICKERS_URL
from sec_data.massive_security_actions import (
    massive_dividends_url,
    massive_splits_url,
    massive_ticker_events_url,
)
from sec_data.next_publication import DISARM_FLAGS
from sec_data.publication_market_bridge import (
    VERSION,
    audit_case_windows,
    close_common_gate,
    validate_contract,
)


def build_requests(
    audits: list[dict[str, Any]], evidence: pd.DataFrame, listing: pd.DataFrame
) -> list[dict[str, Any]]:
    """Freeze only observed gaps, preserving exact CIK, share class and window bounds."""
    requests = []
    for ticker in sorted({r["ticker"] for r in audits}):
        rows = [r for r in audits if r["ticker"] == ticker]
        if all(r["conditional_return_diagnostic_ready"] for r in rows):
            continue
        matches = evidence.loc[evidence.selected_ticker.eq(ticker)]
        if len(matches) != 1:
            raise ValueError("Collection needs one source-pinned selected security")
        target = {str(k): v for k, v in matches.to_dict("records")[0].items()}
        start = min(r["window"]["entry_session_date"] for r in rows)
        end = max(r["window"]["exit_session_date"] for r in rows)
        if target["security_identity_key_kind"] != "share_class_figi" or any(
            target[k] is not False for k in ("selection_uses_return_values", "returns_computed")
        ):
            raise ValueError("Changed or outcome-selected frozen collection identity")
        if any(
            target[k] != r[v]
            for r in rows
            for k, v in (
                ("cik", "cik"),
                ("price_target_id", "price_target_id"),
                ("selected_exchange_mic", "exchange_mic"),
                ("security_identity_key", "security_id"),
                ("security_identity_key_kind", "security_id_kind"),
            )
        ):
            raise ValueError("Collection identity differs from the audited window")
        missing = {k for r in rows for k in r["missing_checks"]}
        jobs = []
        if {"issuer_all_20_parsed_sessions", "issuer_price_query_interval"} & missing:
            jobs.append(("daily_bars", massive_daily_bars_url(ticker, start, end)))
        if {"provider_identity_interval", "issuer_action_query_interval"} & missing:
            for kind, date in (("reference_start", start), ("reference_end", end)):
                jobs.append(
                    (
                        kind,
                        MASSIVE_TICKERS_URL
                        + "?"
                        + urlencode(
                            {
                                "cik": target["cik"],
                                "date": date,
                                "active": "true",
                                "order": "asc",
                                "limit": 1000,
                                "sort": "ticker",
                            }
                        ),
                    )
                )
            if pd.notna(target["selected_composite_figi"]):
                jobs.append(
                    ("ticker_events", massive_ticker_events_url(target["selected_composite_figi"]))
                )
            jobs.extend(
                [
                    ("splits", massive_splits_url(ticker, start, end)),
                    ("dividends", massive_dividends_url(ticker, start, end)),
                ]
            )
        if "listing_outcome_interval" in missing:
            selected = listing.loc[
                listing.cik.eq(target["cik"])
                & listing.security_id.eq(target["security_identity_key"])
            ]
            if len(selected) != 1:
                raise ValueError("Ambiguous frozen outcome cutoff")
            cutoff = selected.outcome_cutoff_date.iloc[0]
            for active in (True, False):
                jobs.append(
                    (
                        "listing_active" if active else "listing_inactive",
                        MASSIVE_TICKERS_URL
                        + "?"
                        + urlencode(
                            {
                                "cik": target["cik"],
                                "date": cutoff,
                                "active": str(active).lower(),
                                "order": "asc",
                                "limit": 1000,
                                "sort": "ticker",
                            }
                        ),
                    )
                )
        for kind, url in jobs:
            row = {
                "ticker": ticker,
                "cik": target["cik"],
                "validation_event_id": rows[0]["validation_event_id"],
                "price_target_id": target["price_target_id"],
                "exchange_mic": target["selected_exchange_mic"],
                "security_id": target["security_identity_key"],
                "security_id_kind": target["security_identity_key_kind"],
                "selected_composite_figi": target["selected_composite_figi"]
                if pd.notna(target["selected_composite_figi"])
                else None,
                "listing_outcome_cutoff": cutoff if kind.startswith("listing_") else None,
                "query_kind": kind,
                "window_start": start,
                "window_end": end,
                "url": url,
                "adjusted": False,
                "selection_uses_return_values": False,
                **dict.fromkeys(DISARM_FLAGS, False),
            }
            row["query_id"] = hashlib.sha256(_encode(row)).hexdigest()
            requests.append(row)
    if len(requests) > 12:
        raise ValueError("Unexpected collection growth; freeze a new bounded protocol")
    return requests


def build_bridge(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Reuse immutable signals; audit seven date windows before any numerical price read."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    publication_plan = load("publication_plan", "plan.json")
    interval_summary = load("interval", "summary.json")
    interval_plan = _json(
        Path(interval_summary["plan_root"]), interval_summary["plan_manifest_sha256"], "plan.json"
    )
    cases = interval_plan["cases"]
    if (
        publication_plan["cases"] != cases
        or interval_summary["joint_interval_scenarios"] != 17
        or interval_summary["all_chronology_violations"] != 0
        or any(interval_summary[k] is not False for k in DISARM_FLAGS)
    ):
        raise ValueError("The frozen publication/financial chain changed")
    for name, sha in interval_plan["implementation_sha256"].items():
        verify_file(Path(name), sha)
    financial = load("interval", "scenario_case_audit.json")
    sources = load("publication", "source_reviews.json")
    dispositions = load("interval", "case_dispositions.json")
    if (
        len(dispositions) != 20
        or [r["validation_event_id"] for r in dispositions]
        != interval_plan["retained_disposition_ids"]
    ):
        raise ValueError("Twenty-issuer disposition ledger changed")
    expanded = load("expanded_plan", "plan.json")
    market = expanded["market_config"]
    if publication_plan["market_config"] != market:
        raise ValueError("Market source chain changed")
    tables, report, _ = _market_metadata(market, expanded["continuity_config"], [])
    evidence = pd.read_parquet(
        _artifact(
            Path(c["evidence_root"]),
            c["evidence_manifest_sha256"],
            "validation_action_evidence_targets.parquet",
        )
    )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": cases,
        "market_config": market,
        "retained_disposition_ids": interval_plan["retained_disposition_ids"],
        "financial_forecasts_reused_not_refitted": True,
        "return_values_read": False,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/build_publication_market_bridge.py",
                "sec_data/publication_market_bridge.py",
                "sec_data/announcement_returns.py",
                "sec_data/announcement_expanded_returns.py",
                "sec_data/validation_action_continuity.py",
                "sec_data/execution_contract.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    audits, dates, gaps = [], [], []
    for case in cases:
        source = next(s for s in sources if s["ticker"] == case["ticker"])
        a, d, g = audit_case_windows(case, source, tables, report)
        audits.extend(a)
        dates.append(d)
        gaps.extend(g)
    gate = close_common_gate(audits, financial)
    queries = build_requests(audits, evidence, tables["listing"])
    query_plan = {
        "requests": queries,
        "maximum_initial_requests": 12,
        "request_interval_seconds": 12.1,
        "pagination_requires_a_new_plan": True,
        "historical_first_public_not_resolved_by_market_API": True,
        "NOBH_missing_composite_figi_must_not_be_imputed": True,
        "session_identity_grid_requires_a_separate_explicit_followup": True,
        "signal_freeze_manifest_sha256": c["interval_manifest_sha256"],
        "signals_refitted": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_queue_issuers_retained": 20,
        "frozen_pair_cases_retained": 6,
        "date_windows": len(audits),
        "price_gap_rows_across_scenarios": len(gaps),
        "distinct_missing_price_dates": len({(r["ticker"], r["session_date"]) for r in gaps}),
        "distinct_gaps_outside_old_query": len(
            {
                (r["ticker"], r["session_date"])
                for r in gaps
                if r["classification"] == "outside_cached_query_interval"
            }
        ),
        "missing_action_target_tickers": sorted(
            {r["ticker"] for r in audits if r["missing_cached_action_target"]}
        ),
        "planned_initial_requests": len(queries),
        "financial_forecasts_refitted": False,
        "numerical_price_values_loaded": False,
        "return_values_computed": False,
        "network_requests": 0,
        "old_artifacts_overwritten": False,
        **gate,
    }
    buffer = io.BytesIO()
    pd.concat(dates, ignore_index=True).to_parquet(buffer, index=False)
    readout = (
        "公告窗口与市场证据闭合（非收益实验）\n\n"
        f"保留20家公司、6个财务案例、17个财务时间情景，匹配{len(audits)}个日期窗口。\n"
        f"完整窗口证据：{', '.join(gate['metadata_window_ready_tickers'])}。\n"
        f"仍阻塞：{', '.join(gate['metadata_window_blocked_tickers'])}。\n"
        f"跨日期情景共{len(gaps)}条缺价记录，去重后{summary['distinct_missing_price_dates']}个日期；"
        f"其中{summary['distinct_gaps_outside_old_query']}个日期在旧请求范围以外。\n"
        "范围以外是尚未抓取，不是当天没有交易；范围内无解析记录也不能证明无成交、停牌或零收益。\n"
        "缺公司行动证据不等于拆股或分红为零。NOBH的Composite FIGI仍缺失，不能造一个来请求。\n"
        f"已冻结{len(queries)}个精确初始补抓请求，未发送网络请求。\n"
        "保留不完整案例，不单独用完整子样本计算效果。历史首次公开与渠道覆盖仍需独立核验。\n"
    )
    files = {
        "summary.json": _encode(summary),
        "window_audits.json": _encode(audits),
        "price_gap_classification.json": _encode(gaps),
        "collection_plan.json": _encode(query_plan),
        "case_dispositions.json": _encode(dispositions),
        "required_window_sessions.parquet": buffer.getvalue(),
        "readout.txt": readout.encode("utf-8"),
    }
    root = write_bundle(output_root, VERSION, files)
    return root, summary


def main() -> None:
    """Run only the offline evidence bridge; no credentials or API calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/publication_market_bridge_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publication_market_bridge")
    )
    args = parser.parse_args()
    root, summary = build_bridge(args.config, args.output_root)
    print(f"PUBLICATION MARKET BRIDGE DONE summary={json.dumps(summary)} output={root}")


if __name__ == "__main__":
    main()
