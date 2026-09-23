"""Freeze and audit a small SEC disclosure-clock/financial-field sample, without returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

from sec_data.client import SecClient
from sec_data.disclosure_audit import (
    VERSION,
    audit_financial_fields,
    audit_forecast_clocks,
    digest_file,
    earnings_candidates,
    inspect_earnings_document,
    load_cached_submissions,
    select_audit_sample,
    utc_timestamp,
    verify_file,
    verify_manual_reviews,
)
from sec_data.guidance_evidence import collect_guidance_documents


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode()


def _table(config: dict[str, Any], name: str, filename: str, columns: list[str]) -> pd.DataFrame:
    root = Path(config[f"{name}_root"])
    path = root / "output_manifest.json"
    verify_file(path, config[f"{name}_manifest_sha256"])
    manifest = json.loads(path.read_bytes())
    verify_file(root / filename, manifest["artifacts"][filename]["sha256"])
    # Column projection prevents opening any financial forecast or return values.
    return pd.read_parquet(root / filename, columns=columns)


def write_bundle(output_root: Path, kind: str, files: dict[str, bytes]) -> Path:
    """Persist content-addressed outputs; reject partial or changed prior bundles."""
    artifacts = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    identity = hashlib.sha256(_json_bytes(artifacts)).hexdigest()[:16]
    root = output_root / f"{kind}-{identity}"
    manifest = _json_bytes({"version": kind, "artifacts": artifacts})
    if root.exists():
        if not (root / "output_manifest.json").exists():
            raise ValueError("Partial output bundle exists; inspect before retrying")
        if (root / "output_manifest.json").read_bytes() != manifest:
            raise ValueError("Output manifest changed")
        for name, digest in artifacts.items():
            verify_file(root / name, digest)
        return root
    root.mkdir(parents=True)
    for name, content in files.items():
        (root / name).write_bytes(content)
    (root / "output_manifest.json").write_bytes(manifest)
    return root


def freeze_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    """Freeze identity-only selection before reading any actuals or release text."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or config["sample_replacement_allowed"]
        or config["return_values_allowed"]
        or config["model_fitting_allowed"]
    ):
        raise ValueError("Unsupported or weakened audit contract")
    market = _table(
        config,
        "market",
        "validation_market_state_features.parquet",
        ["validation_event_id", "cik", "ticker", "form", "accepted_at"],
    )
    sample = select_audit_sample(
        market, sample_size=config["sample_size"], salt=config["selection_salt"]
    )
    features = _table(
        config,
        "feature",
        "target_filing_features.parquet",
        ["validation_event_id", "cik", "filing_accession", "period_end", "accepted_at"],
    ).drop_duplicates()
    sample = sample.merge(
        features.rename(columns={"accepted_at": "feature_accepted_at"}),
        on=["validation_event_id", "cik"],
        how="left",
        validate="one_to_one",
    )
    if sample[["filing_accession", "period_end", "feature_accepted_at"]].isna().any().any():
        raise ValueError("Frozen source metadata is incomplete; sample may not be replaced")
    if not sample["accepted_at"].eq(sample["feature_accepted_at"].map(utc_timestamp)).all():
        raise ValueError("Source acceptance clocks differ")
    history = (
        _table(
            config,
            "feature",
            "quarterly_feature_history.parquet",
            ["company_id", "filing_accession", "period_start", "period_end"],
        )
        .rename(columns={"company_id": "cik"})
        .drop_duplicates()
    )
    sample = sample.merge(
        history, on=["cik", "filing_accession", "period_end"], how="left", validate="one_to_one"
    )
    if sample["period_start"].isna().any():
        raise ValueError("Target quarterly start missing")
    catalog = _table(
        config,
        "catalog",
        "filing_catalog.parquet",
        ["cik", "accession_number", "form", "report_date", "accepted_at", "items"],
    )
    catalog = catalog.loc[catalog["cik"].isin(sample["cik"])].copy()
    records = []
    candidates = []
    for raw_event in sample.to_dict("records"):
        event = cast(dict[str, Any], raw_event)
        event["accepted_at"] = utc_timestamp(event["accepted_at"]).isoformat()
        event.pop("feature_accepted_at")
        for field in ("period_start", "period_end"):
            event[field] = pd.Timestamp(event[field]).date().isoformat()
        records.append(event)
        candidates.extend(earnings_candidates(event, catalog))
    plan = {
        "version": VERSION,
        "config_sha256": digest_file(config_path),
        "config": config,
        "source_events": len(market),
        "source_original_10q_events": int(market["form"].eq("10-Q").sum()),
        "selection_rule": "original_10q_then_issuer_round_then_salted_event_hash",
        "events": records,
        "earnings_candidates": candidates,
        "selection_uses_values_or_returns": False,
        "return_values_loaded": False,
        "missing_evidence_replacement_allowed": False,
        "independent_test_sample": False,
        "representative_market_sample": False,
    }
    root = write_bundle(output_root, "sec-disclosure-plan-v1", {"plan.json": _json_bytes(plan)})
    return root, plan, catalog


def collect_missing(plan: dict[str, Any], raw_root: Path, limit: int) -> None:
    """Collect only frozen SEC candidates after compliant contact identity is supplied."""
    identity = os.environ.get("SEC_USER_AGENT", "").strip()
    if not identity:
        raise ValueError("SEC_USER_AGENT is required for collection; do not put it in source files")
    cache = load_cached_submissions([Path(p) for p in plan["config"]["cached_submission_roots"]])
    targets: list[dict[str, Any]] = []
    seen = set()
    for candidate in plan["earnings_candidates"]:
        key = (candidate["cik"], candidate["accession_number"])
        if key in seen:
            continue
        seen.add(key)
        if key in cache:
            verify_file(Path(cache[key]["path"]), cache[key]["sha256"])
            continue
        targets.append(
            {
                "guidance_document_id": hashlib.sha256(candidate["url"].encode()).hexdigest(),
                "cik": candidate["cik"],
                "prior_accession_number": candidate["accession_number"],
                "complete_submission_url": candidate["url"],
                "document_acquisition_order": len(targets) + 1,
            }
        )
    if not targets:
        print("DISCLOSURE COLLECT pending=0", flush=True)
        return
    result = collect_guidance_documents(
        pd.DataFrame(targets),
        SecClient(identity, request_interval_seconds=0.25),
        raw_root=raw_root,
        limit=limit,
        progress=lambda done, total, doc: print(
            f"DISCLOSURE COLLECT progress={done}/{total} document={doc[:12]}", flush=True
        ),
    )
    print(
        f"DISCLOSURE COLLECT completed={result.completed_documents} "
        f"failed={result.failed_documents} requests={result.requests_made}",
        flush=True,
    )


def run_audit(
    plan_root: Path,
    plan: dict[str, Any],
    catalog: pd.DataFrame,
    output_root: Path,
    review_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Audit frozen events using pinned companyfacts and hash-verified cached releases."""
    manifest = json.loads((plan_root / "output_manifest.json").read_bytes())
    verify_file(plan_root / "plan.json", manifest["artifacts"]["plan.json"])
    if (plan_root / "plan.json").read_bytes() != _json_bytes(plan):
        raise ValueError("In-memory sample differs from persisted plan")
    config = plan["config"]
    zip_path = Path(config["companyfacts_zip"])
    verify_file(zip_path, config["companyfacts_sha256"])
    fields = []
    with zipfile.ZipFile(zip_path) as archive:
        for event in plan["events"]:
            member = f"CIK{event['cik']}.json"
            try:
                content = archive.read(member)
            except KeyError:
                fields.append(
                    {
                        "validation_event_id": event["validation_event_id"],
                        "cik": event["cik"],
                        "metric": "all",
                        "status": "companyfacts_member_missing",
                        "quarter_or_instant_ready": False,
                    }
                )
                continue
            for row in audit_financial_fields(event, json.loads(content), catalog):
                fields.append(
                    {
                        **row,
                        "source_zip_sha256": config["companyfacts_sha256"],
                        "source_member": member,
                        "source_member_sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    cache = load_cached_submissions([Path(p) for p in config["cached_submission_roots"]])
    documents = []
    for candidate in plan["earnings_candidates"]:
        key = (candidate["cik"], candidate["accession_number"])
        row = {**candidate, "source_sha256": None, "status": "not_cached"}
        if key in cache:
            item = cache[key]
            verify_file(Path(item["path"]), item["sha256"])
            if item["requested_url"] != candidate["url"]:
                raise ValueError("Cached document request differs from frozen target")
            row.update(inspect_earnings_document(Path(item["path"]).read_bytes(), key[1]))
            row["source_sha256"] = item["sha256"]
        documents.append(row)
    reviews = []
    if review_path is not None:
        review = json.loads(review_path.read_bytes())
        if review["version"] != "sec-disclosure-manual-review-v1":
            raise ValueError("Unsupported manual review schema")
        reviews = verify_manual_reviews(
            review["reviews"], plan["events"], fields, plan["earnings_candidates"], cache
        )
    clocks = []
    forecast_metadata = _table(
        config,
        "feature",
        "quarterly_feature_history.parquet",
        [
            "company_id",
            "filing_accession",
            "forecast_feature_cutoff",
            "forecast_training_end",
            "forecast_generated_at",
        ],
    )
    for event in plan["events"]:
        matches = [r for r in documents if r["validation_event_id"] == event["validation_event_id"]]
        cached = [r for r in matches if r["source_sha256"]]
        reviewed = [r for r in reviews if r["validation_event_id"] == event["validation_event_id"]]
        metric_bounds: dict[str, str] = {}
        for row in reviewed:
            metric = row["metric"]
            proxy = row["sec_acceptance_proxy_at"]
            metric_bounds[metric] = min(proxy, metric_bounds.get(metric, proxy))
        if "revenue" in metric_bounds and "gross_profit" in metric_bounds:
            metric_bounds["gross_margin"] = max(
                metric_bounds["revenue"], metric_bounds["gross_profit"]
            )
        forecast_check = None
        if "gross_margin" in metric_bounds:
            metadata = forecast_metadata.loc[
                forecast_metadata["company_id"].eq(event["cik"])
                & forecast_metadata["filing_accession"].eq(event["filing_accession"])
            ]
            if len(metadata) != 1:
                raise ValueError("Forecast clock metadata is not uniquely bound to event")
            forecast_check = audit_forecast_clocks(
                cast(dict[str, Any], metadata.iloc[0].to_dict()), metric_bounds["gross_margin"]
            )
        clocks.append(
            {
                **event,
                "earlier_8k_candidates": len(matches),
                "cached_candidate_documents": len(cached),
                "earliest_candidate_sec_acceptance": min(
                    (r["accepted_at"] for r in matches), default=None
                ),
                "first_public_at": None,
                "first_public_verified": False,
                "reviewed_acceptance_proxies": metric_bounds,
                "public_available_by": None,
                "forecast_clock_check": forecast_check,
                "status": "candidate_documents_need_review"
                if cached
                else "candidate_download_needed"
                if matches
                else "no_candidate_not_proof_of_no_announcement",
                "metric_clock_status": {
                    metric: "unverified"
                    for metric in [
                        "revenue",
                        "gross_margin",
                        "net_income",
                        "operating_cash_flow",
                        "assets",
                    ]
                },
            }
        )
    metric_summary = {}
    for metric in ["revenue", "gross_profit", "net_income", "operating_cash_flow", "assets"]:
        rows = [r for r in fields if r["metric"] == metric]
        metric_summary[metric] = {
            "direct_original_ready": sum(bool(r["quarter_or_instant_ready"]) for r in rows),
            "status_counts": {
                s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
            },
        }
    complete = sum(
        all(
            any(
                r["validation_event_id"] == e["validation_event_id"]
                and r["metric"] == m
                and r["quarter_or_instant_ready"]
                for r in fields
            )
            for m in ("net_income", "operating_cash_flow", "assets")
        )
        for e in plan["events"]
    )
    report = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(plan_root / "output_manifest.json"),
        "implementation_sha256": {
            "pipeline": digest_file(Path(__file__)),
            "research": digest_file(Path("sec_data/disclosure_audit.py")),
        },
        "events": len(plan["events"]),
        "issuers": len({r["cik"] for r in plan["events"]}),
        "events_with_earlier_release_candidates": sum(
            bool(r["earlier_8k_candidates"]) for r in clocks
        ),
        "candidate_documents": len({r["url"] for r in documents}),
        "cached_candidate_documents": len({r["url"] for r in documents if r["source_sha256"]}),
        "verified_first_public_events": 0,
        "verified_earlier_value_events": len({r["validation_event_id"] for r in reviews}),
        "verified_earlier_value_matches": len(reviews),
        "review_sha256": digest_file(review_path) if review_path is not None else None,
        "forecasts_at_or_after_acceptance_proxy": sum(
            r["forecast_clock_check"] is not None
            and r["forecast_clock_check"]["status"] == "at_or_after_sec_acceptance_proxy"
            for r in clocks
        ),
        "metric_summary": metric_summary,
        "direct_core_financial_complete_events": complete,
        "model_ready_events": 0,
        "model_fitting_authorized": False,
        "return_values_loaded": False,
        "returns_computed": False,
        "first_public_review_required": True,
        "survivorship_bias_free_claim": False,
        "historical_sample_is_new_holdout": False,
        "cashflow_basis_review_required": True,
        "current_policies_modified": False,
        "acceptance_is_public_availability_bound": False,
        "header_clock_followup_required": True,
        "audit_semantics_version": "sec-acceptance-proxy-v2",
    }
    files = {
        "report.json": _json_bytes(report),
        "field_audit.json": _json_bytes(fields),
        "clock_audit.json": _json_bytes(clocks),
        "document_review.json": _json_bytes(documents),
        "verified_earlier_values.json": _json_bytes(reviews),
        "readout.md": render_readout(report, clocks, fields).encode(),
    }
    root = write_bundle(output_root, VERSION, files)
    return root, report


def render_readout(
    report: dict[str, Any], clocks: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> str:
    """Render a plain-language review table without financial-return claims."""
    lines = [
        "# SEC 披露时点与财务字段审计 v1",
        "",
        f"固定样本：{report['events']} 个原始 10-Q 事件 / {report['issuers']} 家公司。",
        "样本来自既有价格证据样本，不是全市场代表样本，也不是新的独立留出样本。",
        "",
        "## 结论",
        "",
        f"{report['events_with_earlier_release_candidates']} 个事件存在披露前的 8-K 候选；"
        f"{report['cached_candidate_documents']}/{report['candidate_documents']} "
        "份候选原文已缓存。",
        "候选、关键词命中和 SEC 接收时间均不等于已确认的首次公开时间。当前首次公开时点均待核实。",
        f"其中 {report['verified_earlier_value_events']} 个事件经表格核对，相同数值出现在"
        "更早接收的8-K中；接收时间只是代理时钟，不是已证实的公开可得时间上界。",
        f"{report['forecasts_at_or_after_acceptance_proxy']} 个已复核事件存在预测训练/生成时钟"
        "不早于接收代理时钟的风险，公开传播顺序仍待核实。原历史产物未改写。",
        "目录时间尚需原文头部核对；请继续运行 python -m scripts.review_sec_disclosures。",
        "净利润、经营现金流和资产可直接取得的共同样本："
        f"{report['direct_core_financial_complete_events']} 个；不代表会计合并口径已兼容。",
        "跨文件累计现金流之差只作候选，不能自动进入模型。未加载收益、未训练模型、未改交易规则。",
        "",
        "## 字段覆盖",
        "",
        "| 字段 | 直接原始季度/时点值 | 其他状态 |",
        "|---|---:|---|",
    ]
    for metric, summary in report["metric_summary"].items():
        lines.append(
            f"| {metric} | {summary['direct_original_ready']} | {summary['status_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 固定事件逐项核对",
            "",
            "| Ticker | 财季结束 | SEC 接收时间 | 8-K 候选/已缓存 | 现金流状态 |",
            "|---|---|---|---:|---|",
        ]
    )
    for event in clocks:
        cfo = next(
            (
                r["status"]
                for r in fields
                if r["validation_event_id"] == event["validation_event_id"]
                and r["metric"] == "operating_cash_flow"
            ),
            "missing",
        )
        lines.append(
            f"| {event['ticker']} | {event['period_end']} | {event['accepted_at']} | "
            f"{event['earlier_8k_candidates']}/{event['cached_candidate_documents']} | {cfo} |"
        )
    lines.extend(
        [
            "",
            "## 数值匹配的较早文件（时间仍是代理）",
            "",
            "| Ticker | 财季结束 | 数值匹配指标及SEC接收代理时钟 |",
            "|---|---|---|",
        ]
    )
    for event in clocks:
        if event["reviewed_acceptance_proxies"]:
            lines.append(
                f"| {event['ticker']} | {event['period_end']} | "
                f"{event['reviewed_acceptance_proxies']} |"
            )
    lines.extend(
        [
            "",
            "## 下一道门槛",
            "",
            "逐项匹配新闻稿中的财季、数值、单位和会计口径，并寻找带时区的首次公开时间；"
            "只找到日期的保留为日期精度，不补造时分秒。现金流累计值拆季需核实同一财年及口径。",
            "本轮不检验收益。后续假设必须先冻结，再在新留出样本上比较，不得以本轮旧样本当作新验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run offline by default; collection requires explicit flag and SEC identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/sec_disclosure_audit_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/sec_disclosure_audit")
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--review-file", type=Path, default=Path("docs/sec_disclosure_review_v1.json")
    )
    args = parser.parse_args()
    if args.limit < 1 or (args.plan_only and args.collect):
        parser.error("Invalid collection limit or incompatible modes")
    plan_root, plan, catalog = freeze_plan(args.config, args.output_root)
    print(f"DISCLOSURE PLAN events={len(plan['events'])} output={plan_root}", flush=True)
    if args.plan_only:
        return
    if args.collect:
        collect_missing(plan, Path("data/sec/disclosure_audit"), args.limit)
    root, report = run_audit(plan_root, plan, catalog, args.output_root, args.review_file)
    print(
        f"DISCLOSURE AUDIT events={report['events']} issuers={report['issuers']} "
        f"cached={report['cached_candidate_documents']}/{report['candidate_documents']} "
        f"financial_complete={report['direct_core_financial_complete_events']} "
        f"model_gate={report['model_fitting_authorized']} output={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
