"""Offline follow-up of publisher clocks and financial-table precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.execution_contract import execution_window
from sec_data.publication_chronology import (
    PANEL_COLUMNS,
    VERSION,
    chronology_sensitivity,
    review_publisher_source,
)


def _receipts(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for batch in config["batches"]:
        verify_file(Path(batch["config"]), batch["config_sha256"])
        sources = json.loads(Path(batch["config"]).read_bytes())["sources"]
        data = _json(Path(batch["root"]), batch["manifest_sha256"], "receipts.json")
        if data["config_sha256"] != batch["config_sha256"] or len(data["receipts"]) != len(sources):
            raise ValueError("Receipt batch differs from requested sources")
        for source, row in zip(sources, data["receipts"], strict=True):
            if any(row.get(k) != v for k, v in source.items()):
                raise ValueError("Receipt identity differs from source request")
            result.append({**row, "batch": batch})
    return result


def _content(receipt: dict[str, Any]) -> bytes:
    filename = receipt["file"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Unsafe source filename")
    batch = receipt["batch"]
    path = _artifact(Path(batch["root"]), batch["manifest_sha256"], filename)
    verify_file(path, receipt["sha256"])
    content = path.read_bytes()
    if len(content) != receipt["bytes"]:
        raise ValueError("Source byte length differs")
    return content


def build_chronology(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Keep the fixed 30-event queue; never promote publication labels into first-public facts."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or config["max_offset_hours_if_unknown"] != 14
        or any(
            config[k] is not False
            for k in ["overwrite_originals", "return_values_allowed", "model_admission"]
        )
    ):
        raise ValueError("Unsupported or weakened publication protocol")

    def load(group: str, name: str) -> Any:
        return _json(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name)

    events = load("followup", "event_review.json")
    cells = load("followup", "metric_cell_review.json")
    migration = load("migration", "migration_ledger.json")
    prior_queue = load("migration", "publication_review_queue.json")
    if [e["validation_event_id"] for e in events] != [
        e["validation_event_id"] for e in prior_queue
    ]:
        raise ValueError("Frozen publication cohort changed")
    if len({e["validation_event_id"] for e in events}) != len(events) or len(
        {e["cik"] for e in events}
    ) != len(events):
        raise ValueError("Duplicate publication event or issuer")
    for event in events:
        matches = [
            r
            for r in migration
            if r["validation_event_id"] == event["validation_event_id"]
            and r["cik"] == event["cik"]
            and r["accession_number"] == event["filing_accession"]
        ]
        if len(matches) != 1:
            raise ValueError("Publication cohort is not bound to frozen execution cohort")
    metadata = pd.read_parquet(
        _artifact(
            Path(config["clock_root"]),
            config["clock_manifest_sha256"],
            "disarmed_replay_panel.parquet",
        ),
        columns=PANEL_COLUMNS,
    )
    if metadata.duplicated(["company_id", "filing_accession"]).any():
        raise ValueError("Forecast metadata identity duplicated")
    receipts = _receipts(config)
    sessions = pd.read_parquet(
        _artifact(
            Path(config["calendar_root"]),
            config["calendar_manifest_sha256"],
            "validation_exchange_sessions.parquet",
        )
    )
    specs = config["reviews"]
    if len({s["source_id"] for s in specs}) != len(specs):
        raise ValueError("Duplicate source review")
    implementations = {
        p: digest_file(Path(p))
        for p in [
            "scripts/build_publication_chronology.py",
            "sec_data/publication_chronology.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
            "sec_data/disclosure_audit.py",
            "sec_data/filing_text.py",
            "sec_data/execution_contract.py",
        ]
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "implementation_sha256": implementations,
        "event_ids": [e["validation_event_id"] for e in events],
        "return_values_loaded": False,
        "sample_reselected": False,
    }
    frozen = write_bundle(
        output_root, "publication-chronology-plan-v1", {"plan.json": _encode(plan)}
    )
    reviews, contents = [], {}
    for spec in specs:
        matches = [
            r
            for r in receipts
            if all(r.get(k) == spec[k] for k in ("ticker", "cik", "release_accession", "url"))
            and r.get("sha256") == spec["snapshot_sha256"]
            and r["status_code"] == 200
        ]
        if len(matches) != 1:
            raise ValueError("Publisher review lacks a unique successful raw receipt")
        receipt = matches[0]
        if receipt.get("response_url") != spec["url"] or "text/html" not in receipt.get(
            "content_type", ""
        ):
            raise ValueError("Wrong source URL or response type")
        target = [
            e
            for e in events
            if all(e[k] == spec[k] for k in ("cik", "ticker", "period_start", "period_end"))
        ]
        if len(target) != 1:
            raise ValueError("Publisher source does not match a frozen quarter")
        content = _content(receipt)
        contents[spec["source_id"]] = content
        reviewed = review_publisher_source(content, spec, cells)
        reviewed.update(
            {
                "fetched_at": receipt["fetched_at"],
                "validation_event_id": target[0]["validation_event_id"],
            }
        )
        reviews.append(reviewed)
    for spec in specs:
        if "linked_from_source_id" in spec:
            parent = contents[spec["linked_from_source_id"]].decode("utf-8")
            if 'href="' + spec["url"] + '"' not in parent:
                raise ValueError("Declared issuer link is absent from saved source")
    queue = []
    for event in events:
        rows = metadata.loc[
            metadata.company_id.eq(event["cik"])
            & metadata.filing_accession.eq(event["filing_accession"])
        ]
        if len(rows) != 1:
            raise ValueError("Frozen event has no unique replay forecast metadata")
        event_sources = [
            s for s in reviews if s["validation_event_id"] == event["validation_event_id"]
        ]
        scenarios = [
            chronology_sensitivity(s, cast(dict[str, Any], rows.iloc[0].to_dict()))
            for s in event_sources
        ]
        queue.append(
            {
                "validation_event_id": event["validation_event_id"],
                "cik": event["cik"],
                "ticker": event["ticker"],
                "filing_accession": event["filing_accession"],
                "sources": event_sources,
                "chronology_scenarios": scenarios,
                "prior_acceptance_proxy_check": event["forecast_proxy_check"],
                "precise_label_replay_candidates": [
                    s["source_id"] for s in event_sources if s["point_clock_and_exact_values"]
                ],
                "status": "publisher_evidence_reviewed_not_first_public"
                if event_sources
                else "not_in_risk_batch_retained",
                "remaining_work": [
                    "earlier_channel_search",
                    "historical_publication_verification",
                    "as_of_forecast_and_error_history_rebuild",
                    "new_window_market_evidence",
                ],
                "first_public_certified": False,
                "model_admission": False,
                "return_admission": False,
            }
        )
    risk = [e for e in queue if e["sources"]]
    replay_requests = []
    for event in risk:
        schedule_event = next(
            r for r in migration if r["validation_event_id"] == event["validation_event_id"]
        )
        for source in event["sources"]:
            if not source["point_clock_and_exact_values"]:
                continue
            proposed = execution_window(
                source["source_reported_at_utc"], schedule_event["exchange_mic"], sessions
            )
            replay_requests.append(
                {
                    "validation_event_id": event["validation_event_id"],
                    "cik": event["cik"],
                    "ticker": event["ticker"],
                    "target_filing_accession": event["filing_accession"],
                    "source_id": source["source_id"],
                    "snapshot_sha256": source["snapshot_sha256"],
                    "source_reported_at_utc": source["source_reported_at_utc"],
                    "purpose": "disarmed_publisher_label_sensitivity_only",
                    "same_contract_filing_proxy_window": schedule_event[
                        "new_contract_reviewed_clock"
                    ],
                    "publisher_label_window": proposed,
                    "change_filing_acceptance_timestamp": False,
                    "rebuild_required": [
                        "actual_availability",
                        "lag_features",
                        "forecast_cutoff",
                        "training_end",
                        "prior_oos_errors",
                        "forecast_sigma",
                    ],
                    "first_public_certified": False,
                    "forecast_recomputed": False,
                    "price_collection_authorized": False,
                    "return_admission": False,
                }
            )
    after = [
        e["ticker"]
        for e in risk
        if any(
            s["applies_to_exact_gross_margin"]
            and s["forecast_comparisons"]["forecast_generated_at"]["status"]
            == "after_all_label_scenarios"
            for s in e["chronology_scenarios"]
        )
    ]
    report = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "implementation_sha256": implementations,
        "execution_events_retained_upstream": len(migration),
        "publication_events_retained": len(queue),
        "risk_events_reviewed": len(risk),
        "pinned_receipts": len(receipts),
        "failed_receipts_preserved": sum(r["status_code"] != 200 for r in receipts),
        "reviewed_snapshots": len(reviews),
        "exact_component_cells": sum(len(r["matched_metrics"]) for r in reviews),
        "full_table_events": len({r["cik"] for r in reviews if r["exact_gross_margin_components"]}),
        "precise_label_and_full_table_events": len(
            {r["cik"] for r in reviews if r["point_clock_and_exact_values"]}
        ),
        "forecast_after_all_label_scenarios_tickers": after,
        "first_public_certified": 0,
        "disarmed_replay_requests": len(replay_requests),
        "forecast_recomputed": False,
        "returns_recomputed": False,
        "model_admission": False,
        "trading_admission": False,
        "originals_overwritten": False,
    }
    files = {
        "report.json": _encode(report),
        "source_reviews.json": _encode(reviews),
        "event_chronology.json": _encode(queue),
        "request_receipts.json": _encode(receipts),
        "disarmed_replay_requests.json": _encode(replay_requests),
    }
    if config.get("write_readout", True):
        files["readout.md"] = render_readout(report, risk).encode()
    return write_bundle(output_root, VERSION, files), report


def render_readout(report: dict[str, Any], risk: list[dict[str, Any]]) -> str:
    """Explain precision limits and the distinction between source labels and first-public time."""
    lines = [
        "# 发布来源与预测时序核验",
        "",
        f"保留 {report['publication_events_retained']} 个事件；"
        f"优先核验 {report['risk_events_reviewed']} 个。",
        f"已审查原文 {report['reviewed_snapshots']} 份，"
        f"精确财务单元格 {report['exact_component_cells']} 个。",
        "",
        "|公司/来源|时间精度|生成预测与发布时间标签的关系|精确毛利率两项原值|",
        "|---|---|---|---|",
    ]
    for event in risk:
        for source, scenario in zip(event["sources"], event["chronology_scenarios"], strict=True):
            status = scenario["forecast_comparisons"]["forecast_generated_at"]["status"]
            lines.append(
                f"|{event['ticker']} / {source['source_id']}|{source['clock_precision']}|"
                f"{status}|{source['exact_gross_margin_components']}|"
            )
    lines += [
        "",
        "时间不完整时使用明确的±14小时UTC偏移压力区间；日期精度还覆盖整天。",
        "这些是敏感性情景，不是已认证的公开时间上下界。精确标签也只是该发布渠道报告的时间。",
        "不把CMS更新时间当发布时间，不把摘要四舍五入数值当精确财务表，不把网页抓取时间当历史发布时间。",
        "预测时间来自历史重建，不是当年的实际生成日志。after标签提示需重建，不单独证明模型已使用目标值。",
        "首次公开认证仍为0，没有改预测、收益、样本或交易政策。下一步仅可做独立、禁用交易的发布标签时钟重放；",
        "真实收益准入仍需更早披露渠道核验、信息集重建和市场窗口证据。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Build from existing immutable evidence, with no implicit network requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/publication_chronology_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publication_chronology")
    )
    args = parser.parse_args()
    root, report = build_chronology(args.config, args.output_root)
    print(
        f"PUBLICATION CHRONOLOGY DONE events={report['publication_events_retained']} "
        f"risk={report['risk_events_reviewed']} pages={report['reviewed_snapshots']} "
        f"precise_full_table={report['precise_label_and_full_table_events']} "
        f"admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
