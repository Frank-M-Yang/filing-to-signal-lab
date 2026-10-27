"""Offline publisher-label chronology and price-date coverage for six fixed original pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_publication_chronology import _content, _receipts
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, utc_timestamp
from sec_data.execution_contract import execution_window
from sec_data.next_publication import (
    BENCHMARK_COLUMNS,
    DISARM_FLAGS,
    PRICE_COLUMNS,
    VERSION,
    audit_price_dates,
    compare_forecast_clocks,
    label_window_scenarios,
    review_publication,
)
from sec_data.publication_chronology import PANEL_COLUMNS


def validate_contract(config: dict[str, Any]) -> None:
    """Block sample substitution, implicit networking, price values and admission."""
    if (
        config.get("version") != VERSION
        or config.get("frozen_queue_issuers") != 20
        or config.get("frozen_pair_cases") != 6
        or any(
            config.get(k) is not False
            for k in (
                "network_allowed",
                "price_values_allowed",
                "returns_allowed",
                "forecast_refit_allowed",
                "overwrite_originals",
                "sample_replacement_allowed",
                *DISARM_FLAGS,
            )
        )
    ):
        raise ValueError("Unsupported or weakened next-publication contract")


def build_audit(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Preserve all twenty dispositions; publication labels never open a return gate."""
    config = json.loads(config_path.read_bytes())
    validate_contract(config)

    def load(group: str, name: str) -> Any:
        return _json(Path(config[group + "_root"]), config[group + "_manifest_sha256"], name)

    summary = load("replay", "summary.json")
    plan = load("replay_plan", "plan.json")
    cases = plan["cases"]
    dispositions = load("replay", "case_dispositions.json")
    cells = load("review", "reviewed_metric_cells.json")
    if (
        summary["plan_manifest_sha256"] != config["replay_plan_manifest_sha256"]
        or plan["config"]["review_manifest_sha256"] != config["review_manifest_sha256"]
        or len(dispositions) != 20
        or len({r["validation_event_id"] for r in dispositions}) != 20
        or [r["validation_event_id"] for r in dispositions] != plan["retained_disposition_ids"]
        or len(cases) != 6
        or {r["ticker"] for r in cases} != {"NOBH", "AOS", "ROCK", "RAMP", "ORA", "CTSO"}
        or any(r.get(k) is not False for r in dispositions for k in DISARM_FLAGS)
        or any(r.get(k) is not False for r in cases for k in DISARM_FLAGS)
    ):
        raise ValueError("Frozen replay/source cohort changed")
    metadata = pd.read_parquet(
        _artifact(
            Path(config["replay_root"]),
            config["replay_manifest_sha256"],
            "shared_disarmed_history.parquet",
        ),
        columns=PANEL_COLUMNS,
    )
    if metadata.duplicated(["company_id", "filing_accession"]).any():
        raise ValueError("Duplicated financial forecast metadata")
    receipts = _receipts(config)
    market = load("expanded_plan", "plan.json")["market_config"]

    def table(group: str, filename: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], filename),
            columns=columns,
        )

    sessions = table("execution", "validation_exchange_sessions.parquet")
    targets = table("prices", "validation_price_batch_targets.parquet")
    prices = table("prices", "validation_price_batch_bars.parquet", PRICE_COLUMNS)
    benchmark = table("benchmark", "benchmark_daily_price_observations.parquet", BENCHMARK_COLUMNS)
    specs = config["reviews"]
    if (
        len(specs) != 6
        or len({s["source_id"] for s in specs}) != 6
        or {s["ticker"] for s in specs} != {c["ticker"] for c in cases}
    ):
        raise ValueError("Exactly one reviewed source for each frozen pair is required")
    implementations = {
        p: digest_file(Path(p))
        for p in (
            "scripts/build_next_publication_audit.py",
            "sec_data/next_publication.py",
            "scripts/build_publication_chronology.py",
            "sec_data/execution_contract.py",
            "sec_data/filing_text.py",
            "sec_data/disclosure_audit.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
        )
    }
    frozen = write_bundle(
        output_root / "plans",
        VERSION + "-plan",
        {
            "plan.json": _encode(
                {
                    "config": config,
                    "config_sha256": digest_file(config_path),
                    "implementation_sha256": implementations,
                    "retained_disposition_ids": plan["retained_disposition_ids"],
                    "cases": cases,
                    "market_config": market,
                    "return_values_loaded": False,
                    "forecast_values_loaded": False,
                    "price_values_loaded": False,
                }
            )
        },
    )
    reviewed, case_audits, requests = [], [], []
    for case in cases:
        spec = next(s for s in specs if s["ticker"] == case["ticker"])
        if (
            any(spec[k] != case[k] for k in ("ticker", "cik", "period_start", "period_end"))
            or spec["release_accession"] != case["release_accession_number"]
        ):
            raise ValueError("Publisher source does not belong to the complete original pair")
        matching = [
            r
            for r in receipts
            if all(r.get(k) == spec[k] for k in ("ticker", "cik", "release_accession", "url"))
            and r.get("sha256") == spec["snapshot_sha256"]
            and r["status_code"] == 200
        ]
        if (
            len(matching) != 1
            or matching[0].get("response_url") != spec["url"]
            or "text/html" not in matching[0].get("content_type", "")
        ):
            raise ValueError("No unique successful raw HTML receipt for source review")
        receipt = matching[0]
        source = review_publication(_content(receipt), spec, cells)
        source.update(
            {
                "fetched_at": receipt["fetched_at"],
                "validation_event_id": case["validation_event_id"],
            }
        )
        rows = metadata.loc[
            metadata.company_id.eq(case["cik"])
            & metadata.filing_accession.eq(case["target_accession_number"])
        ]
        if len(rows) != 1 or utc_timestamp(rows.iloc[0].actual_available_at) != utc_timestamp(
            case["complete_pair_replay_cutoff_at"]
        ):
            raise ValueError("Source case lacks its exact reconstructed forecast metadata")
        clocks = compare_forecast_clocks(source, {str(k): v for k, v in rows.iloc[0].items()})
        scenarios = label_window_scenarios(source, case["assigned_exchange_mic"], sessions)
        for scenario in scenarios:
            if scenario["schedule"] is not None:
                scenario["price_date_audit"] = audit_price_dates(
                    case, scenario["schedule"], sessions, targets, prices, benchmark
                )
        try:
            previous = execution_window(
                case["complete_pair_replay_cutoff_at"], case["assigned_exchange_mic"], sessions
            )
        except ValueError as exc:
            previous = {"schedule_resolved": False, "unresolved_reason": str(exc)}
        signatures = {
            (s["schedule"]["entry_session_date"], s["schedule"]["exit_session_date"])
            for s in scenarios
            if s["schedule"] is not None
        }
        ready = bool(
            len(signatures) == 1
            and all(
                s.get("price_date_audit", {}).get("metadata_window_complete", False)
                for s in scenarios
            )
        )
        signatures_differ = (
            None
            if not previous["schedule_resolved"]
            else any(
                s != (previous["entry_session_date"], previous["exit_session_date"])
                for s in signatures
            )
        )
        row = {
            "validation_event_id": case["validation_event_id"],
            "ticker": case["ticker"],
            "cik": case["cik"],
            "complete_pair_acceptance_proxy_at": case["complete_pair_replay_cutoff_at"],
            "publisher_source": source,
            "forecast_label_check": clocks,
            "prior_acceptance_proxy_schedule": previous,
            "publisher_label_scenarios": scenarios,
            "window_invariant_across_label_scenarios": len(signatures) == 1
            and all(s["schedule"] is not None for s in scenarios),
            "any_window_differs_from_acceptance_proxy": signatures_differ,
            "single_metadata_window_ready": ready,
            "historical_first_public_and_earlier_channels_verified": False,
            "identity_actions_listing_and_liquidity_reauthorized": False,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        requests.append(
            {
                "validation_event_id": case["validation_event_id"],
                "ticker": case["ticker"],
                "publisher_source_id": source["source_id"],
                "snapshot_sha256": source["snapshot_sha256"],
                "source_reported_at_utc": source["source_reported_at_utc"],
                "financial_as_of_dependency_rebuild_required": True,
                "label_clock_order_status": clocks["forecast_comparisons"],
                "historical_publication_verification_required": True,
                "market_windows_for_reauthorization": scenarios,
                "price_collection_authorized": False,
                "network_authorized": False,
                **dict.fromkeys(DISARM_FLAGS, False),
            }
        )
        reviewed.append(source)
        case_audits.append(row)
    indexed = {r["validation_event_id"]: r for r in case_audits}
    retained = [
        {
            **r,
            "publication_audit": indexed.get(r["validation_event_id"]),
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        for r in dispositions
    ]
    after = [
        r["ticker"]
        for r in case_audits
        if any(
            v["status"] == "after_all"
            for v in r["forecast_label_check"]["forecast_comparisons"].values()
        )
    ]
    overlap = [
        r["ticker"]
        for r in case_audits
        if any(
            v["status"] == "overlap"
            for v in r["forecast_label_check"]["forecast_comparisons"].values()
        )
    ]
    result = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_queue_issuers_retained": len(retained),
        "complete_pair_sources_reviewed": len(reviewed),
        "exact_original_component_cells": sum(len(s["matched_cells"]) for s in reviewed),
        "explicit_offset_minute_labels": sum(
            s["source_reported_at_utc"] is not None for s in reviewed
        ),
        "unknown_timezone_tickers": [
            s["ticker"] for s in reviewed if s["source_reported_at_utc"] is None
        ],
        "forecast_after_all_label_scenarios_tickers": after,
        "forecast_overlaps_label_precision_or_offset_tickers": overlap,
        "forecast_strictly_prior_all_label_scenarios_tickers": [
            r["ticker"]
            for r in case_audits
            if r["forecast_label_check"]["prior_to_all_label_scenarios"]
        ],
        "invariant_date_windows": sum(
            r["window_invariant_across_label_scenarios"] for r in case_audits
        ),
        "single_complete_metadata_windows": sum(
            r["single_metadata_window_ready"] for r in case_audits
        ),
        "price_date_gap_tickers": [
            r["ticker"]
            for r in case_audits
            if any(
                s.get("price_date_audit", {}).get("metadata_window_complete") is False
                for s in r["publisher_label_scenarios"]
            )
        ],
        "changed_or_sensitive_window_tickers": [
            r["ticker"] for r in case_audits if r["any_window_differs_from_acceptance_proxy"]
        ],
        "pinned_receipts": len(receipts),
        "failed_pinned_receipts_preserved": sum(r["status_code"] != 200 for r in receipts),
        "first_public_certified_events": 0,
        "common_return_gate": False,
        "price_values_loaded": False,
        "forecast_values_loaded": False,
        "returns_recomputed": False,
        "forecasts_recomputed": False,
        "network_requests": 0,
        "keys_read": False,
        "sample_reselected": False,
        "old_artifacts_overwritten": False,
        **dict.fromkeys(DISARM_FLAGS, False),
        "next_action": (
            "publisher_label_shared_asof_financial_replay_then_"
            "identity_actions_window_reauthorization"
        ),
    }
    files = {
        "summary.json": _encode(result),
        "source_reviews.json": _encode(reviewed),
        "case_publication_audit.json": _encode(case_audits),
        "case_dispositions.json": _encode(retained),
        "disarmed_rebuild_requests.json": _encode(requests),
        "request_receipts.json": _encode(receipts),
        "readout.txt": render_readout(result, case_audits).encode("utf-8"),
    }
    return write_bundle(output_root, VERSION, files), result


def render_readout(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    """Explain the audit in Chinese without converting clock checks into an Alpha claim."""
    lines = [
        "六家公司发布来源与价格窗口审查（非收益实验）",
        "",
        "保留原20家公司；核验6份完整财务表，12个原始营业收入/毛利润单元格。",
        f"5份有明确时区的分钟级标签；未知时区：{', '.join(summary['unknown_timezone_tickers'])}。",
        "结构化字段里的秒数不自动证明历史秒级可用性，按更保守的可见分钟精度处理。",
        "",
    ]
    for case in cases:
        source = case["publisher_source"]
        status = case["forecast_label_check"]["forecast_comparisons"]["forecast_generated_at"][
            "status"
        ]
        windows = "; ".join(
            f"{s['schedule']['entry_session_date']}开盘至{s['schedule']['exit_session_date']}收盘"
            if s["schedule"] is not None
            else "日历未解析"
            for s in case["publisher_label_scenarios"]
        )
        lines.append(
            f"{case['ticker']}：{source['visible_clock_literal']}；旧重建预测相对标签={status}；{windows}。"
        )
    lines += [
        "",
        "after_all表示现有重建预测晚于整段发布标签；overlap表示分钟精度或时区信息不足，不能证明严格先于。",
        "这不是模型已读入目标值的独立证据，但这些预测不能直接用于更早的标签时钟实验。",
        "before_all也只是时间字段检查：所有6家公司仍需重建共享信息集、训练样本和严格更早的样本外误差历史。",
        "价格检查只读取日期、证券查询身份和哈希。没有读取开收盘价格或计算收益，也没有检验Alpha。",
        "NOBH两种候选窗口均有行情日期缺口；不能直接认定只是漏下载，也可能无成交。未前填价格或伪造零收益。",
        "当前发布页不是历史档案，不能证明当时已含相同表格，更不能排除更早渠道；首次公开认证仍为0。",
        "后续先做独立、禁用交易的发布标签敏感性重放，再重授权身份、公司行动、上市状态与流动性证据。",
        "NOBH时区问题未解决前不改变6家共同样本，更不因时间或行情缺失替换公司。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run deterministically from pinned local bytes without credentials or downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/next_publication_audit_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/next_publication_audit")
    )
    args = parser.parse_args()
    root, summary = build_audit(args.config, args.output_root)
    print(f"NEXT PUBLICATION DONE summary={json.dumps(summary)} output={root}")


if __name__ == "__main__":
    main()
