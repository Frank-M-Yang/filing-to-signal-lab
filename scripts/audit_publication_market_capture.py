"""Offline reaudit of captured market gaps; keep coverage discrepancies quarantined."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_announcement_expanded_returns import _market_metadata
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.collect_publication_market_gaps import load_requests
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_security_actions import (
    _dividend_columns,
    _identity_columns,
    _split_columns,
    _ticker_event_columns,
)
from sec_data.next_publication import DISARM_FLAGS, PRICE_COLUMNS
from sec_data.publication_market_bridge import audit_case_windows, close_common_gate
from sec_data.publication_market_capture import (
    action_audit,
    history_access_sensitivity,
    overlay_price_dates,
    parse_query,
)

VERSION = "publication-market-capture-audit-v1"


def validate_contract(c: dict[str, Any]) -> None:
    """Allow payload validation, never financial refits, replacements or returns."""
    expected = {
        "version": VERSION,
        "market_payload_validation_allowed": True,
        "frozen_pair_cases": 6,
        "frozen_queue_issuers": 20,
        "frozen_financial_scenarios": 17,
        "entry_session_offset": 2,
        "holding_sessions_inclusive": 20,
        "assumed_basic_history_days": 730,
        "actual_key_plan_verified": False,
        "history_horizon_rule_verified": False,
        **dict.fromkeys(
            (
                "network_allowed",
                "forecast_refit_allowed",
                "returns_allowed",
                "sample_replacement_allowed",
                "partial_sample_returns_allowed",
                "missing_price_imputation_allowed",
                "overwrite_originals",
                *DISARM_FLAGS,
            ),
            False,
        ),
    }
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported capture-audit contract")


def _table_bytes(frame: pd.DataFrame) -> bytes:
    result = io.BytesIO()
    frame.to_parquet(result, index=False)
    return result.getvalue()


def build_audit(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Bind all twelve receipts, normalize evidence, and rerun the same six-case gate."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)
    bridge = Path(c["bridge_root"])
    requests = load_requests(bridge, c["bridge_manifest_sha256"])
    old_summary = _json(bridge, c["bridge_manifest_sha256"], "summary.json")
    old_plan = _json(
        Path(old_summary["plan_root"]), old_summary["plan_manifest_sha256"], "plan.json"
    )
    capture = Path(c["capture_root"])
    receipts = _json(capture, c["capture_manifest_sha256"], "receipts.json")
    captured_summary = _json(capture, c["capture_manifest_sha256"], "summary.json")
    if (
        len(receipts) != len(requests)
        or [r["request"] for r in receipts] != requests
        or any(r.get("source_manifest_sha256") != c["bridge_manifest_sha256"] for r in receipts)
        or any(captured_summary.get(k) is not False for k in DISARM_FLAGS)
    ):
        raise ValueError("An incomplete or rebound capture cannot replace the frozen request batch")
    source = old_plan["config"]
    expanded = _json(
        Path(source["expanded_plan_root"]), source["expanded_plan_manifest_sha256"], "plan.json"
    )
    tables, report, _ = _market_metadata(
        old_plan["market_config"], expanded["continuity_config"], []
    )
    templates = pd.read_parquet(
        _artifact(
            Path(c["action_template_root"]),
            c["action_template_manifest_sha256"],
            "sample_security_actions_audit.parquet",
        )
    )
    witnesses = []
    for w in c["cached_price_witnesses"]:
        verify_file(Path(w["receipt_path"]), w["receipt_sha256"])
        verify_file(Path(w["raw_path"]), w["raw_sha256"])
        receipt = json.loads(Path(w["receipt_path"]).read_bytes())
        if any(
            receipt[k] != w[v]
            for k, v in (
                ("ticker", "ticker"),
                ("price_target_id", "price_target_id"),
                ("sha256", "raw_sha256"),
            )
        ):
            raise ValueError("Cached price witness identity changed")
        witnesses.append(
            {
                **w,
                "original_query_start": receipt["window_start"],
                "original_query_end": receipt["window_end"],
                "original_fetched_at": receipt["fetched_at"],
            }
        )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": old_plan["cases"],
        "requests": requests,
        "cached_price_witnesses": witnesses,
        "financial_forecasts_reused_not_refitted": True,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/audit_publication_market_capture.py",
                "sec_data/publication_market_capture.py",
                "sec_data/publication_market_bridge.py",
                "sec_data/massive_security_actions.py",
                "sec_data/massive_reference.py",
                "sec_data/massive_prices.py",
                "sec_data/announcement_expanded_returns.py",
                "sec_data/validation_action_continuity.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    targets = []
    for ticker in sorted({r["ticker"] for r in requests}):
        group = [r for r in requests if r["ticker"] == ticker]
        t = templates.loc[templates.price_target_id.eq(group[0]["price_target_id"])]
        if len(t) != 1 or ticker in set(tables["actions"].selected_ticker):
            raise ValueError("Capture overlay requires one missing original action target")
        target = {str(k): v for k, v in t.to_dict("records")[0].items()}
        if any(
            target[k] != r[v]
            for r in group
            for k, v in (
                ("cik", "cik"),
                ("selected_ticker", "ticker"),
                ("selected_exchange_mic", "exchange_mic"),
                ("security_identity_key", "security_id"),
                ("security_identity_key_kind", "security_id_kind"),
            )
        ) or any(
            target[k] is not False
            for k in ("selection_uses_return_values", "outcome_values_loaded", "returns_computed")
        ):
            raise ValueError("Template security identity or outcome-selection flags changed")
        target["action_target_id"] = hashlib.sha256(
            _encode(
                {
                    "parent_action_target_id": target["action_target_id"],
                    "query_ids": [r["query_id"] for r in group],
                }
            )
        ).hexdigest()
        target.update(
            price_window_start=group[0]["window_start"], price_window_end=group[0]["window_end"]
        )
        targets.append(target)
    target_table = pd.DataFrame(targets)
    statuses, rejected, parsed = [], [], {}
    for receipt in receipts:
        request = receipt["request"]
        folder = Path(c["response_root"]) / request["query_id"]
        found = list(
            folder.glob("publication-market-gap-capture-v1-response-*/output_manifest.json")
        )
        if len(found) != 1:
            raise ValueError("Missing or ambiguous raw response bundle")
        manifest = found[0]
        bound = _json(manifest.parent, digest_file(manifest), "receipt.json")
        if bound != receipt:
            raise ValueError("Raw receipt differs from the captured batch")
        payload = _artifact(manifest.parent, digest_file(manifest), "response.json").read_bytes()
        aid = str(
            target_table.loc[
                target_table.selected_ticker.eq(request["ticker"]), "action_target_id"
            ].iloc[0]
        )
        status, rows, bad = parse_query(request, payload, receipt["sha256"], aid)
        statuses.append(status)
        rejected.extend(bad)
        parsed[request["query_id"]] = rows

    def rows_for(kinds: set[str]) -> list[dict[str, Any]]:
        return [row for r in requests if r["query_kind"] in kinds for row in parsed[r["query_id"]]]

    identities = pd.DataFrame(
        rows_for({"reference_start", "reference_end"}), columns=_identity_columns()
    )
    timeline = pd.DataFrame(rows_for({"ticker_events"}), columns=_ticker_event_columns())
    splits = pd.DataFrame(rows_for({"splits"}), columns=_split_columns())
    dividends = pd.DataFrame(rows_for({"dividends"}), columns=_dividend_columns())
    actions = action_audit(
        target_table, identities, timeline, splits, dividends, statuses, tables["sessions"]
    )
    tables["actions"] = pd.concat([tables["actions"], actions], ignore_index=True)
    tables["identities"] = pd.concat([tables["identities"], identities], ignore_index=True)
    tables["ticker_events"] = pd.concat([tables["ticker_events"], timeline], ignore_index=True)
    conflicts = []
    horizon_audits = []
    new_prices = pd.DataFrame(rows_for({"daily_bars"}), columns=PRICE_COLUMNS)
    for request in [r for r in requests if r["query_kind"] == "daily_bars"]:
        status = next(s for s in statuses if s["query_id"] == request["query_id"])
        tables["metadata"], comparison = overlay_price_dates(
            tables["metadata"],
            new_prices.loc[new_prices.price_target_id.eq(request["price_target_id"])],
            request,
            status,
        )
        comparison["cached_raw_witnesses"] = [
            w for w in witnesses if w["price_target_id"] == request["price_target_id"]
        ]
        if comparison["cached_overlap_source_hashes"] != sorted(
            w["raw_sha256"] for w in comparison["cached_raw_witnesses"]
        ):
            raise ValueError("Every cached overlap source requires an exact raw witness")
        conflicts.append(comparison)
        fresh_receipt = next(r for r in receipts if r["request"]["query_id"] == request["query_id"])
        for witness in comparison["cached_raw_witnesses"]:
            horizon_audits.append(
                {
                    "ticker": request["ticker"],
                    "query_id": request["query_id"],
                    "source_url": c["history_horizon_source_url"],
                    **history_access_sensitivity(
                        comparison["cached_dates_absent_from_new_response"],
                        witness["original_fetched_at"],
                        fresh_receipt["fetched_at"],
                        history_days=c["assumed_basic_history_days"],
                    ),
                }
            )
        if comparison["new_query_reauthorized"]:
            match = tables["price_audit"].price_target_id.eq(request["price_target_id"])
            tables["price_audit"].loc[match, "price_window_start"] = request["window_start"]
            tables["price_audit"].loc[match, "price_window_end"] = request["window_end"]
            tables["price_audit"].loc[match, "provider_query_complete"] = True
            tables["price_audit"].loc[match, "parser_rejection_rows"] = 0
    listing_rows = rows_for({"listing_active", "listing_inactive"})
    # Do not silently promote a newly encountered positive/ambiguous delisting
    # candidate. This batch's empty references remain an explicitly unresolved
    # outcome; a positive candidate requires its own semantics/terminal audit.
    listing_audit = {
        "parsed_candidate_rows": len(listing_rows),
        "existing_listing_evidence_unchanged": True,
        "empty_reference_is_not_a_delisting_or_no_trade_claim": True,
        "new_positive_candidates_require_separate_listing_review": bool(listing_rows),
    }
    audits, dates, gaps = [], [], []
    sources = _json(
        Path(source["publication_root"]),
        source["publication_manifest_sha256"],
        "source_reviews.json",
    )
    for case in old_plan["cases"]:
        a, d, g = audit_case_windows(
            case, next(s for s in sources if s["ticker"] == case["ticker"]), tables, report
        )
        audits.extend(a)
        dates.append(d)
        gaps.extend(g)
    financial = _json(
        Path(source["interval_root"]),
        source["interval_manifest_sha256"],
        "scenario_case_audit.json",
    )
    gate = close_common_gate(audits, financial)
    dispositions = _json(bridge, c["bridge_manifest_sha256"], "case_dispositions.json")
    if len(dispositions) != 20:
        raise ValueError("The original twenty-issuer disposition ledger changed")
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_queue_issuers_retained": 20,
        "frozen_pair_cases_retained": 6,
        "date_windows": len(audits),
        "captured_queries": len(receipts),
        "parser_complete_queries": sum(s["query_complete"] for s in statuses),
        "previous_metadata_ready_tickers": old_summary["metadata_window_ready_tickers"],
        "new_metadata_ready_tickers": sorted(
            set(gate["metadata_window_ready_tickers"])
            - set(old_summary["metadata_window_ready_tickers"])
        ),
        "overlap_coverage_discrepancy_tickers": [
            r["ticker"] for r in conflicts if r["overlap_coverage_discrepancy"]
        ],
        "cached_dates_absent_from_new_responses": sum(
            len(r["cached_dates_absent_from_new_response"]) for r in conflicts
        ),
        "new_price_metadata_dates_added": sum(r["new_dates_added"] for r in conflicts),
        "history_expiry_hypothesis_fits_tickers": sorted(
            {r["ticker"] for r in horizon_audits if r["all_discrepant_dates_fit_expiry_hypothesis"]}
        ),
        "actual_key_plan_verified": False,
        "history_expiry_cause_confirmed": False,
        "new_raw_price_payload_validated_then_ohlc_discarded": True,
        "return_values_computed": False,
        "financial_forecasts_refitted": False,
        "network_requests": 0,
        "old_artifacts_overwritten": False,
        **gate,
    }
    readout = (
        "市场补抓证据复核（不是收益或Alpha实验）\n\n"
        f"12个冻结请求全部保存；解析完成{summary['parser_complete_queries']}个。\n"
        f"市场元数据完整：{', '.join(gate['metadata_window_ready_tickers'])}。"
        f"仍阻塞：{', '.join(gate['metadata_window_blocked_tickers'])}。\n"
        "NOBH新价格响应为空，与旧快照重叠窗口里的5个日期记录不一致。两次查询范围不同，不能据此确认数据被修订，更不能推断无成交或零收益。\n"
        "新空响应已隔离，旧记录和原始字节不覆盖。NOBH的历史身份、首次公开时刻和可成交窗口仍需要独立证据。\n"
        "两年滚动历史窗口是可检验的原因假设：这些旧日期8月采集时未超过730天，10月采集时已超过。实际密钥套餐和供应商边界规则尚未确认，不将假设当成原因。\n"
        f"套餐历史长度参考：{c['history_horizon_source_url']}。\n"
        "CTSO身份/拆股/分红查询已按同一证券和窗口核验；未观察到窗口内拆股、现金分红。\n"
        "保留全部20家公司、6个财务案例、102个财务时间情景；不选择完整子样本单独报告效果，不改变模型，不计算收益。\n"
    )
    files = {
        "summary.json": _encode(summary),
        "query_parser_audit.json": _encode(statuses),
        "parser_rejections.json": _encode(rejected),
        "price_overlap_audit.json": _encode(conflicts),
        "history_access_sensitivity.json": _encode(horizon_audits),
        "listing_reference_audit.json": _encode(listing_audit),
        "window_audits.json": _encode(audits),
        "case_dispositions.json": _encode(dispositions),
        "price_gap_classification.json": _encode(gaps),
        "sample_security_actions_audit.parquet": _table_bytes(actions),
        "security_reference_interval_observations.parquet": _table_bytes(identities),
        "ticker_change_observations.parquet": _table_bytes(timeline),
        "split_observations.parquet": _table_bytes(splits),
        "dividend_observations.parquet": _table_bytes(dividends),
        "new_price_metadata.parquet": _table_bytes(new_prices),
        "required_window_sessions.parquet": _table_bytes(pd.concat(dates, ignore_index=True)),
        "readout.txt": readout.encode("utf-8"),
    }
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Run the source-pinned offline audit; never read keys or make requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/publication_market_capture_audit_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publication_market_capture_audit")
    )
    args = parser.parse_args()
    root, summary = build_audit(args.config, args.output_root)
    print(f"PUBLICATION MARKET CAPTURE AUDIT DONE summary={json.dumps(summary)} output={root}")


if __name__ == "__main__":
    main()
