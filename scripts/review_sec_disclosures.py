"""Offline follow-up of frozen SEC disclosures; no prices, forecasts or policies are changed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, load_cached_submissions, verify_file
from sec_data.disclosure_review import (
    SEC_CLOCK_SOURCE,
    VERSION,
    audit_candidate_headers,
    earliest_metric_proxies,
    forecast_proxy_check,
    review_document_kinds,
    review_metric_cells,
)


def _read_artifact(root: Path, manifest_sha256: str, filename: str) -> Any:
    verify_file(root / "output_manifest.json", manifest_sha256)
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    verify_file(root / filename, manifest["artifacts"][filename])
    return json.loads((root / filename).read_bytes())


def run_followup(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Validate pinned evidence and reviewer assertions in the unchanged sample."""
    config = json.loads(config_path.read_bytes())
    if config["version"] != VERSION:
        raise ValueError("Unsupported follow-up schema")
    plan = _read_artifact(Path(config["plan_root"]), config["plan_manifest_sha256"], "plan.json")
    fields = _read_artifact(
        Path(config["audit_root"]), config["audit_manifest_sha256"], "field_audit.json"
    )
    prior_report = _read_artifact(
        Path(config["audit_root"]), config["audit_manifest_sha256"], "report.json"
    )
    if prior_report["plan_manifest_sha256"] != config["plan_manifest_sha256"]:
        raise ValueError("Field audit belongs to a different frozen sample")
    prior_path = Path(config["prior_review_path"])
    verify_file(prior_path, config["prior_review_sha256"])
    prior = json.loads(prior_path.read_bytes())
    if prior["version"] != "sec-disclosure-manual-review-v1":
        raise ValueError("Unsupported prior review schema")
    events, candidates = plan["events"], plan["earnings_candidates"]
    cache = load_cached_submissions([Path(p) for p in plan["config"]["cached_submission_roots"]])
    headers = audit_candidate_headers(candidates, cache)
    dispositions = review_document_kinds(config["documents"], events, candidates, cache)
    new_reviews = []
    for cell_spec in config["metric_reviews"]:
        documents = [
            d
            for d in config["documents"]
            if d["release_accession"] == cell_spec["release_accession"]
        ]
        if len(documents) != 1 or documents[0]["kind"] != "earnings_release":
            raise ValueError("Only a reviewed earnings release can supply final table cells")
        new_reviews.append({**documents[0], **cell_spec})
    cells = review_metric_cells(prior["reviews"] + new_reviews, events, fields, candidates, cache)

    feature_config = plan["config"]
    feature_root = Path(feature_config["feature_root"])
    verify_file(feature_root / "output_manifest.json", feature_config["feature_manifest_sha256"])
    feature_manifest = json.loads((feature_root / "output_manifest.json").read_bytes())
    filename = "quarterly_feature_history.parquet"
    verify_file(feature_root / filename, feature_manifest["artifacts"][filename]["sha256"])
    metadata = pd.read_parquet(
        feature_root / filename,
        columns=[
            "company_id",
            "filing_accession",
            "forecast_feature_cutoff",
            "forecast_training_end",
            "forecast_generated_at",
        ],
    )
    event_reviews = []
    for event in events:
        event_id = event["validation_event_id"]
        event_cells = [r for r in cells if r["validation_event_id"] == event_id]
        proxies = earliest_metric_proxies(event_cells)
        check = None
        if "gross_margin" in proxies:
            rows = metadata.loc[
                metadata["company_id"].eq(event["cik"])
                & metadata["filing_accession"].eq(event["filing_accession"])
            ]
            if len(rows) != 1:
                raise ValueError("Forecast metadata identity is not unique")
            check = forecast_proxy_check(
                cast(dict[str, Any], rows.iloc[0].to_dict()), proxies["gross_margin"]
            )
        related_headers = [r for r in headers if r["validation_event_id"] == event_id]
        conflicts = [r for r in related_headers if r["status"] != "consistent_acceptance_proxy"]
        event_reviews.append(
            {
                **event,
                "candidate_count": len(related_headers),
                "clock_quarantined_documents": len(conflicts),
                "reviewed_metric_count": len(event_cells),
                "matched_acceptance_proxies": proxies,
                "forecast_proxy_check": check,
                "first_public_at": None,
                "public_available_by": None,
                "status": "clock_conflict_quarantined"
                if conflicts
                else "partial_metric_review"
                if event_cells
                else "candidate_content_unreviewed"
                if related_headers
                else "no_candidate_not_proof_of_absence",
                "formal_feature_admitted": False,
            }
        )
    report = {
        "version": VERSION,
        "config_sha256": digest_file(config_path),
        "plan_manifest_sha256": config["plan_manifest_sha256"],
        "prior_audit_manifest_sha256": config["audit_manifest_sha256"],
        "prior_review_sha256": config["prior_review_sha256"],
        "implementation_sha256": {
            "pipeline": digest_file(Path(__file__)),
            "research": digest_file(Path("sec_data/disclosure_review.py")),
            "shared_evidence": digest_file(Path("sec_data/disclosure_audit.py")),
            "submission_parser": digest_file(Path("sec_data/guidance_evidence.py")),
            "bundle_writer": digest_file(Path("scripts/build_sec_disclosure_audit.py")),
        },
        "events": len(events),
        "header_checks": len(headers),
        "header_status_counts": dict(Counter(r["status"] for r in headers)),
        "document_dispositions": len(dispositions),
        "document_kind_counts": dict(Counter(r["kind"] for r in dispositions)),
        "reviewed_metric_cells": len(cells),
        "metric_comparison_counts": dict(Counter(r["comparison_status"] for r in cells)),
        "exact_value_match_events": len(
            {
                r["validation_event_id"]
                for r in cells
                if r["comparison_status"] == "exact_original_match"
            }
        ),
        "gross_margin_proxy_check_events": sum(
            r["forecast_proxy_check"] is not None for r in event_reviews
        ),
        "forecast_proxy_status_counts": dict(
            Counter(
                r["forecast_proxy_check"]["status"]
                for r in event_reviews
                if r["forecast_proxy_check"] is not None
            )
        ),
        "reviewed_document_scope": config["selection"],
        "first_public_verified_events": 0,
        "model_fitting_authorized": False,
        "return_values_loaded": False,
        "current_policies_modified": False,
        "historical_outputs_overwritten": False,
        "source_clock_semantics": SEC_CLOCK_SOURCE,
        "corrects_prior_claim": (
            "SEC acceptance is an operational proxy, not a proven public-availability upper bound. "
            "Prior after-bound labels become timing-risk flags, not proof of post-publication "
            "forecasts."
        ),
    }
    payloads = {
        "report.json": report,
        "header_clock_audit.json": headers,
        "document_dispositions.json": dispositions,
        "metric_cell_review.json": cells,
        "event_review.json": event_reviews,
    }
    files = {
        name: json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        for name, value in payloads.items()
    }
    files["readout.md"] = render_followup(
        report, headers, dispositions, cells, event_reviews
    ).encode()
    return write_bundle(output_root, VERSION, files), report


def render_followup(
    report: dict[str, Any],
    headers: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    """Render audit findings and remaining work without an investment conclusion."""
    tickers = {e["validation_event_id"]: e["ticker"] for e in events}
    lines = [
        "# SEC附件复核：信息时钟与会计口径",
        "",
        "固定30事件；复核全部32份候选的原文头部时钟，以及新下载12份的文档类型。",
        "未读取收益、未训练模型、未修改原历史产物或交易规则。",
        "",
        "## 重要更正",
        "",
        "SEC接收时间是代理时钟，不是已证实的公开可得时间上界。此前报告的上界措辞过强，予以更正；",
        "先后关系检查只能指出时点风险，不能单凭接收时间证明预测已看到了公开结果。",
        f"[SEC说明]({SEC_CLOCK_SOURCE})：文件公开可能晚于系统接收，且不存在保证的公开延迟上限。",
        "没有人为补上固定几分钟；首次公开时间仍未认证。",
        "",
        "## 时钟冲突（隔离，不自动平移全库）",
        "",
        "|公司|accession|目录UTC|原文按美东解释的UTC|差值小时|",
        "|---|---|---|---|---:|",
    ]
    for row in headers:
        if row["status"] != "consistent_acceptance_proxy":
            lines.append(
                f"|{tickers[row['validation_event_id']]}|{row['accession_number']}|{row['catalog_accepted_at']}|{row['header_accepted_at_utc']}|{row['catalog_minus_header_hours']}|"
            )
    lines += [
        "",
        "## 新下载文档逐份审查",
        "",
        "|公司|accession|类型|口径/处置|",
        "|---|---|---|---|",
    ]
    for row in documents:
        lines.append(f"|{row['ticker']}|{row['release_accession']}|{row['kind']}|{row['notes']}|")
    lines += [
        "",
        "## 数值核验",
        "",
        f"共复核{report['reviewed_metric_cells']}个单元格（含上轮3个事件），状态：{report['metric_comparison_counts']}。",
        "匹配数值不等于会计口径兼容，更不等于信号有效。",
        "",
    ]
    for row in cells:
        if row["comparison_status"] != "exact_original_match":
            lines.append(
                f"- {row['ticker']} {row['metric']}：公告直接值 {row['value']:,.0f} USD；"
                f"跨文件累计值之差 {row['derived_quarter_candidate']}；"
                f"差额 {row['release_minus_ytd_candidate']} USD。原因未确认，不能静默修补。"
            )
    lines += [
        "",
        "## 固定事件的复核状态",
        "",
        "|公司|数值核对数|时钟隔离数|毛利率预测代理时钟检查|",
        "|---|---:|---:|---|",
    ]
    for event in events:
        check = event["forecast_proxy_check"]
        lines.append(
            f"|{event['ticker']}|{event['reviewed_metric_count']}|"
            f"{event['clock_quarantined_documents']}|{check['status'] if check else 'not_checked'}|"
        )
    lines += [
        "",
        "## 下一步",
        "",
        "先核对X、OCUL原始10-Q及上游目录时间来源，再完成其余候选的数值/期间/口径复核。",
        "EA预告作为独立信息阶段保留；NVIDIA现金流差异核实同一财年、舍入精度及是否存在更正。",
        "需要IR/新闻稿或历史传播证据才能加强公开时间主张。当前所有模型、训练和交易准入闸门仍关闭。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the deterministic local-only follow-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/sec_disclosure_followup_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/sec_disclosure_followup")
    )
    args = parser.parse_args()
    root, report = run_followup(args.config, args.output_root)
    print(
        f"DISCLOSURE FOLLOWUP headers={report['header_checks']} "
        f"documents={report['document_dispositions']} cells={report['reviewed_metric_cells']} "
        f"model_gate=False output={root}"
    )


if __name__ == "__main__":
    main()
