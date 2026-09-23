"""Offline execution-contract migration and release-clock evidence gap ledger."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.execution_contract import ExecutionContract
from sec_data.execution_migration import (
    EVENT_COLUMNS,
    VERSION,
    migrate_execution_windows,
    publication_review_queue,
    review_release_page,
)


def _source_checks(config: dict[str, Any], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = Path(config["release_snapshots_root"])
    digest = config["release_snapshots_manifest_sha256"]
    metadata = _json(root, digest, "receipts.json")
    source_config = Path(config["release_sources_config"])
    verify_file(source_config, config["release_sources_config_sha256"])
    if metadata["config_sha256"] != config["release_sources_config_sha256"]:
        raise ValueError("Release receipt/config binding differs")
    sources = json.loads(source_config.read_bytes())["sources"]
    receipts = metadata["receipts"]
    if len(sources) != len(receipts) or len({s["url"] for s in sources}) != len(sources):
        raise ValueError("Incomplete or duplicated source receipts")
    reviews = {(r["cik"], r["release_accession"]): r for r in config["publisher_reviews"]}
    if len(reviews) != len(config["publisher_reviews"]):
        raise ValueError("Duplicate publisher review")
    outputs = []
    applied = set()
    for source, receipt in zip(sources, receipts, strict=True):
        if any(receipt.get(k) != v for k, v in source.items()):
            raise ValueError("Release source identity differs from frozen request")
        content = None
        if "file" in receipt:
            filename = receipt["file"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("Unsafe snapshot filename")
            path = _artifact(root, digest, filename)
            verify_file(path, receipt["sha256"])
            content = path.read_bytes()
            if len(content) != receipt["bytes"]:
                raise ValueError("Response byte length differs")
        key = (source["cik"], source["release_accession"])
        status = "raw_snapshot_missing_" + receipt.get("error", "unknown")
        result: dict[str, Any] = {
            **source,
            "snapshot_sha256": receipt.get("sha256"),
            "fetched_at": receipt["fetched_at"],
            "status_code": receipt["status_code"],
            "clock_status": status,
            "source_reported_at_utc": None,
            "first_public_at": None,
            "public_available_by": None,
            "historical_capture": False,
            "forecast_certification": False,
            "return_admission": False,
        }
        if content is not None:
            result["clock_status"] = "response_not_reviewable"
        if receipt["status_code"] == 200 and content is not None:
            if receipt.get("response_url") != source["url"] or "text/html" not in receipt.get(
                "content_type", ""
            ):
                raise ValueError("Unexpected response URL or media type")
            result["clock_status"] = "snapshot_present_not_cell_reviewed"
            if key in reviews:
                spec = reviews[key]
                if (
                    spec["snapshot_sha256"] != receipt["sha256"]
                    or spec["ticker"] != source["ticker"]
                ):
                    raise ValueError("Publisher review not bound to snapshot")
                result.update(review_release_page(content, spec, cells))
                applied.add(key)
        outputs.append(result)
    if applied != set(reviews):
        raise ValueError("A declared publisher review could not be applied")
    return outputs


def build_migration(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Preserve the whole frozen execution cohort; never read or recompute returns."""
    config = json.loads(config_path.read_bytes())
    if config["version"] != VERSION or any(
        config[k] is not False
        for k in ("overwrite_originals", "return_values_allowed", "model_admission")
    ):
        raise ValueError("Unsupported or weakened migration contract")
    root, digest = Path(config["execution_root"]), config["execution_manifest_sha256"]
    events = pd.read_parquet(
        _artifact(root, digest, "validation_execution_events.parquet"), columns=EVENT_COLUMNS
    )
    sessions = pd.read_parquet(_artifact(root, digest, "validation_exchange_sessions.parquet"))
    layer = _json(Path(config["clock_root"]), config["clock_manifest_sha256"], "clock_layer.json")
    followup_root, followup_hash = Path(config["followup_root"]), config["followup_manifest_sha256"]
    review_events = _json(followup_root, followup_hash, "event_review.json")
    cells = _json(followup_root, followup_hash, "metric_cell_review.json")
    by_id = {r["validation_event_id"]: r for r in events.to_dict("records")}
    if len({r["validation_event_id"] for r in review_events}) != len(review_events):
        raise ValueError("Duplicate publication review event")
    for row in review_events:
        target = by_id.get(row["validation_event_id"])
        if (
            target is None
            or target["cik"] != row["cik"]
            or target["target_accession_number"] != row["filing_accession"]
        ):
            raise ValueError("Publication review is not within the frozen execution cohort")
    implementation = {
        p: digest_file(Path(p))
        for p in (
            "scripts/build_sec_execution_migration.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
            "sec_data/execution_contract.py",
            "sec_data/execution_migration.py",
            "sec_data/clock_layer.py",
            "sec_data/disclosure_audit.py",
            "sec_data/filing_text.py",
        )
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "execution_contract": asdict(ExecutionContract()),
        "implementation_sha256": implementation,
        "event_ids": events.validation_event_id.tolist(),
        "publication_review_event_ids": [r["validation_event_id"] for r in review_events],
        "sample_reselection": False,
        "return_values_loaded": False,
    }
    plan_root = write_bundle(
        output_root, "sec-execution-migration-plan-v1", {"plan.json": _encode(plan)}
    )
    ledger = migrate_execution_windows(events, layer, sessions)
    checks = _source_checks(config, cells)
    queue = publication_review_queue(review_events, checks)
    # The changed-window queue is a schedule-only request, never a price-read authorization.
    requests = [
        {
            "validation_event_id": r["validation_event_id"],
            "cik": r["cik"],
            "ticker": r["ticker"],
            "provisional_window": r["new_contract_reviewed_clock"],
            "required_checks": r["evidence_revalidation_required"],
            "clock_basis": "acceptance_proxy_not_first_public",
            "collection_authorized": False,
        }
        for r in ledger
        if r["final_window_differs_from_legacy"] is not False
    ]
    report = {
        "version": VERSION,
        "execution_contract": asdict(ExecutionContract()),
        "plan_manifest_sha256": digest_file(plan_root / "output_manifest.json"),
        "implementation_sha256": implementation,
        "frozen_events": len(events),
        "events_retained": len(ledger),
        "issuers": int(events.cik.nunique()),
        "legacy_windows_reproduced": len(ledger),
        "rule_changed_windows": sum(r["rule_changes_window"] is True for r in ledger),
        "reviewed_target_clocks": sum(
            r["clock_review_status"] != "catalog_proxy_unreviewed" for r in ledger
        ),
        "source_clock_changes": sum(r["source_clock_changed"] for r in ledger),
        "clock_changed_windows_under_new_rule": sum(
            r["clock_changes_window_under_new_rule"] is True for r in ledger
        ),
        "final_changed_windows": sum(r["final_window_differs_from_legacy"] is True for r in ledger),
        "unresolved_windows": sum(r["new_contract_reviewed_clock"] is None for r in ledger),
        "changed_window_evidence_requests": len(requests),
        "publication_review_events": len(queue),
        "risk_batch_sources": len(checks),
        "risk_batch_raw_snapshots": sum(
            r["status_code"] == 200 and bool(r["snapshot_sha256"]) for r in checks
        ),
        "timezone_unspecified_sources": sum(
            r["clock_status"] == "timezone_unspecified" for r in checks
        ),
        "source_checks": checks,
        "public_times_certified": 0,
        "publication_review_admitted": 0,
        "return_values_loaded": False,
        "prices_loaded": False,
        "originals_overwritten": False,
        "sample_reselected": False,
        "model_admission": False,
        "trading_admission": False,
        "scope": (
            "schedule migration using mixed acceptance proxies; "
            "not a first-public or return validation"
        ),
    }
    files = {
        "report.json": _encode(report),
        "migration_ledger.json": _encode(ledger),
        "publication_review_queue.json": _encode(queue),
        "release_source_checks.json": _encode(checks),
        "changed_window_evidence_requests.json": _encode(requests),
        "readout.md": render_readout(report).encode(),
    }
    return write_bundle(output_root, VERSION, files), report


def render_readout(report: dict[str, Any]) -> str:
    """Summarize operational differences without new performance claims."""
    lines = [
        "# 执行约定迁移与公开时间证据",
        "",
        "统一约定：公告纽约当地日期之后的第二个交易日开盘，入场日计为持有第1日，第20日收盘退出。",
        "入场/退出价格类型为未复权开盘/收盘；本轮只定日程，不读取价格。除权、分红与退市仍需另行验证。",
        "",
        f"固定执行事件 {report['frozen_events']}，保留 {report['events_retained']}，"
        f"发行人 {report['issuers']}。",
        f"原入场/退出窗口全部复现：{report['legacy_windows_reproduced']}。",
        f"只改日期规则：{report['rule_changed_windows']} 个窗口改变。",
        f"执行样本中已核验接收时钟 {report['reviewed_target_clocks']} 个；"
        f"其中 {report['source_clock_changes']} 个有源时钟修正。",
        "在新日期规则下，修正时钟额外改变 "
        f"{report['clock_changed_windows_under_new_rule']} 个窗口。",
        f"新窗口相对旧窗口改变 {report['final_changed_windows']} 个；"
        f"未解决窗口 {report['unresolved_windows']} 个。",
        "",
        "注意：这是同一批接收时间代理上的执行约定差异，不是已认证的首次公告日交易计划。",
        "未通过审核的其他接收时钟仍保留为未审代理；未获准沿用旧收益或跳过新窗口市场证据核验。",
        "",
        f"保留全部 {report['publication_review_events']} 个公开时间审查事件；"
        f"优先抓取 {report['risk_batch_sources']} 个既有风险案例，",
        f"本地留存有效原文 {report['risk_batch_raw_snapshots']} 份。"
        f"首次公开时间认证数 {report['public_times_certified']}。",
        "",
        "|公司|原文证据状态|公开时间状态|",
        "|---|---|---|",
    ]
    for row in report["source_checks"]:
        lines.append(f"|{row['ticker']}|HTTP {row['status_code']}|{row['clock_status']}|")
    lines += [
        "",
        "当前网页快照不是当年的抓取日志；即使带时区的发布标签，也不能单独证明全市场首次公开。",
        "无时区字段不猜成美东或UTC；下载超时不使用搜索摘要替代原文哈希证据。",
        "没有改原数据、模型、预测参数、收益或政策，没有删除不完整事件，没有批准交易。",
        "",
        "下一步：补齐发布时区/原文留存及更早披露检索，按真实可用时间重建信息集；",
        "对受影响事件重新核对证券身份、价格、公司行动、退市与基准窗口，通过后才建立新版收益账本。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run offline from pinned inputs with no implicit network collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/sec_execution_migration_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/sec_execution_migration")
    )
    args = parser.parse_args()
    root, report = build_migration(args.config, args.output_root)
    print(
        f"EXECUTION MIGRATION DONE events={report['events_retained']}/{report['frozen_events']} "
        f"rule_changes={report['rule_changed_windows']} "
        f"clock_changes={report['source_clock_changes']} "
        f"public_certified={report['public_times_certified']} admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
