"""Trace frozen SEC clock conflicts; optional bounded collection, no global repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.client import SecClient
from sec_data.clock_provenance import (
    VERSION,
    audit_event_boundary,
    boundary_scan,
    index_url,
    reconcile_acceptance,
    trace_source_row,
)
from sec_data.disclosure_audit import (
    digest_file,
    load_cached_submissions,
    submission_url,
    verify_file,
)
from sec_data.guidance_evidence import collect_guidance_documents


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()


def _read_bundle(root: Path, sha256: str, filename: str) -> Any:
    verify_file(root / "output_manifest.json", sha256)
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    verify_file(root / filename, manifest["artifacts"][filename])
    return json.loads((root / filename).read_bytes())


def freeze_clock_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze conflict issuers, nearby candidates and controls before collecting."""
    config = json.loads(config_path.read_bytes())
    if config["version"] != VERSION or any(
        config[k] for k in ["apply_to_catalog", "model_fitting_allowed", "return_values_allowed"]
    ):
        raise ValueError("Unsupported or weakened clock-audit contract")
    plan = _read_bundle(Path(config["plan_root"]), config["plan_manifest_sha256"], "plan.json")
    followup_root = Path(config["followup_root"])
    previous = _read_bundle(followup_root, config["followup_manifest_sha256"], "report.json")
    if previous["plan_manifest_sha256"] != config["plan_manifest_sha256"]:
        raise ValueError("Follow-up does not belong to frozen event sample")
    headers = _read_bundle(
        followup_root, config["followup_manifest_sha256"], "header_clock_audit.json"
    )
    focus = sorted({r["cik"] for r in headers if r["status"] != "consistent_acceptance_proxy"})
    events = [e for e in plan["events"] if e["cik"] in focus]
    source = plan["config"]
    root = Path(source["catalog_root"])
    verify_file(root / "output_manifest.json", source["catalog_manifest_sha256"])
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    verify_file(
        root / "filing_catalog.parquet", manifest["artifacts"]["filing_catalog.parquet"]["sha256"]
    )
    catalog = pd.read_parquet(
        root / "filing_catalog.parquet",
        columns=[
            "cik",
            "accession_number",
            "form",
            "accepted_at",
            "filing_date",
            "report_date",
            "items",
            "source_zip_sha256",
            "source_member",
            "source_row_index",
        ],
    )
    keys = {(h["cik"], h["accession_number"]) for h in headers}
    focus_keys = set()
    scans = []
    for event in events:
        scan = boundary_scan(event, catalog, padding_days=config["boundary_padding_days"])
        rows = scan.to_dict("records")
        old = [
            c["accession_number"]
            for c in plan["earnings_candidates"]
            if c["validation_event_id"] == event["validation_event_id"]
        ]
        scans.append({"event": event, "rows": rows, "old_accessions": old})
        focus_keys.add((event["cik"], event["filing_accession"]))
        focus_keys.update((r["cik"], r["accession_number"]) for r in rows)
    keys.update(focus_keys)
    selected = catalog.loc[
        pd.MultiIndex.from_frame(catalog[["cik", "accession_number"]]).isin(list(keys))
    ]
    if len(selected) != len(keys):
        raise ValueError("Clock target identity missing or duplicated in catalog")
    targets = []
    for row in selected.sort_values(["cik", "accepted_at", "accession_number"]).to_dict("records"):
        if row["source_zip_sha256"] != config["submissions_sha256"]:
            raise ValueError("A different submissions snapshot needs an explicit version")
        targets.append(
            {
                **row,
                "needs_index": (row["cik"], row["accession_number"]) in focus_keys,
                "submission_url": submission_url(row["cik"], row["accession_number"]),
                "index_url": index_url(row["cik"], row["accession_number"]),
            }
        )
    frozen = {
        "version": VERSION,
        "config": config,
        "config_sha256": digest_file(config_path),
        "focus_ciks": focus,
        "events": events,
        "scans": scans,
        "targets": targets,
        "cached_submission_roots": source["cached_submission_roots"],
        "selection": (
            "prior clock conflicts plus their target 10-Q and date-padded candidates; "
            "all prior candidate headers are controls"
        ),
        "primary_sample_reselected": False,
        "return_values_loaded": False,
    }
    root = write_bundle(
        output_root, "sec-clock-provenance-plan-v1", {"plan.json": _json_bytes(frozen)}
    )
    return root, frozen


def _caches(
    plan: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    config = plan["config"]
    raw_root = Path(config["raw_root"])
    submissions = load_cached_submissions(
        [Path(p) for p in plan["cached_submission_roots"]] + [raw_root / "submissions"]
    )
    indexes = load_cached_submissions([raw_root / "indexes"])
    by_url = {t["index_url"]: t for t in plan["targets"] if t["needs_index"]}
    for spec in config["existing_indexes"]:
        manifest_path = Path(spec["manifest"])
        verify_file(manifest_path, spec["sha256"])
        item = json.loads(manifest_path.read_bytes())
        if (
            item["status_code"] != 200
            or item["artifact_kind"] != "index"
            or item["requested_url"] not in by_url
        ):
            raise ValueError("Pinned index snapshot does not match a frozen target")
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(Path(config["existing_index_root"]).resolve()):
            raise ValueError("Existing index path escapes its source root")
        verify_file(path, item["sha256"])
        target = by_url[item["requested_url"]]
        key = (target["cik"], target["accession_number"])
        if key in indexes and indexes[key]["sha256"] != item["sha256"]:
            raise ValueError("Distinct index snapshots require an explicit choice")
        indexes[key] = {**item, "path": str(path), "receipt_sha256": spec["sha256"]}
    return submissions, indexes


def collect_clock_evidence(plan: dict[str, Any], *, limit: int) -> dict[str, int]:
    """Fetch only missing frozen artifacts with one shared identified SEC limiter."""
    identity = os.environ.get("SEC_USER_AGENT", "").strip()
    if not identity or limit < 1:
        raise ValueError("Explicit collection requires SEC_USER_AGENT and positive limit")
    submissions, indexes = _caches(plan)
    client = SecClient(identity, request_interval_seconds=0.25)
    counts = {"selected": 0, "completed": 0, "failed": 0, "requests": 0}
    for kind, cache, url_field in [
        ("submissions", submissions, "submission_url"),
        ("indexes", indexes, "index_url"),
    ]:
        targets: list[dict[str, Any]] = []
        for row in plan["targets"]:
            if kind == "indexes" and not row["needs_index"]:
                continue
            key = (row["cik"], row["accession_number"])
            if key in cache:
                verify_file(Path(cache[key]["path"]), cache[key]["sha256"])
                continue
            targets.append(
                {
                    "guidance_document_id": hashlib.sha256(row[url_field].encode()).hexdigest(),
                    "cik": row["cik"],
                    "prior_accession_number": row["accession_number"],
                    "complete_submission_url": row[url_field],
                    "document_acquisition_order": len(targets) + 1,
                }
            )
        remaining = limit - counts["selected"]
        if not targets or remaining <= 0:
            continue
        result = collect_guidance_documents(
            pd.DataFrame(targets),
            client,
            raw_root=Path(plan["config"]["raw_root"]) / kind,
            limit=remaining,
            progress=lambda done, total, doc: print(
                f"CLOCK COLLECT progress={done}/{total} artifact={doc[:12]}", flush=True
            ),
        )
        counts["selected"] += result.selected_documents
        counts["completed"] += result.completed_documents
        counts["failed"] += result.failed_documents
        counts["requests"] += result.requests_made
    return counts


def run_provenance(
    plan_root: Path, plan: dict[str, Any], output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Audit sources and emit a separate unapplied acceptance-clock overlay."""
    manifest = json.loads((plan_root / "output_manifest.json").read_bytes())
    verify_file(plan_root / "plan.json", manifest["artifacts"]["plan.json"])
    if (plan_root / "plan.json").read_bytes() != _json_bytes(plan):
        raise ValueError("In-memory plan differs from frozen targets")
    config = plan["config"]
    zip_path = Path(config["submissions_zip"])
    verify_file(zip_path, config["submissions_sha256"])
    submissions, indexes = _caches(plan)
    resolutions = []
    missing = []
    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for row in plan["targets"]:
            key = (row["cik"], row["accession_number"])
            member = row["source_member"]
            if member not in members:
                members[member] = archive.read(member)
            source = trace_source_row(members[member], row)
            if row["needs_index"] and key not in indexes:
                missing.append({"cik": key[0], "accession_number": key[1], "kind": "index"})
            if key not in submissions:
                missing.append({"cik": key[0], "accession_number": key[1], "kind": "submission"})
                continue
            cached = submissions[key]
            if cached["requested_url"] != row["submission_url"]:
                raise ValueError("Submission URL differs from frozen target")
            verify_file(Path(cached["path"]), cached["sha256"])
            index_bytes = None
            if row["needs_index"]:
                if key in indexes:
                    item = indexes[key]
                    if item["requested_url"] != row["index_url"]:
                        raise ValueError("Index URL differs from frozen target")
                    verify_file(Path(item["path"]), item["sha256"])
                    index_bytes = Path(item["path"]).read_bytes()
            resolutions.append(
                reconcile_acceptance(row, source, Path(cached["path"]).read_bytes(), index_bytes)
            )
    boundaries = [
        audit_event_boundary(s["event"], s["rows"], resolutions, s["old_accessions"])
        for s in plan["scans"]
    ]
    overlay = [
        r
        for r in resolutions
        if r["resolution_status"] == "source_label_conflict_resolved_by_header_and_index"
    ]
    report = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(plan_root / "output_manifest.json"),
        "source_config_sha256": plan["config_sha256"],
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in [
                "scripts/audit_sec_clock_provenance.py",
                "sec_data/clock_provenance.py",
                "sec_data/disclosure_review.py",
                "sec_data/disclosure_audit.py",
                "sec_data/filing_cover.py",
                "sec_data/guidance_evidence.py",
                "scripts/build_sec_disclosure_audit.py",
            ]
        },
        "target_filings": len(plan["targets"]),
        "traced_filings": len(resolutions),
        "focus_events": len(plan["events"]),
        "resolution_counts": dict(Counter(r["resolution_status"] for r in resolutions)),
        "proposed_clock_corrections": len(overlay),
        "missing_artifacts": len(missing),
        "boundary_reviews_complete": sum(not r["unresolved_accessions"] for r in boundaries),
        "boundary_membership_unchanged": sum(
            r["candidate_membership_unchanged"] is True for r in boundaries
        ),
        "overlay_applied": False,
        "catalog_rebuilt": False,
        "forecasts_rebuilt": False,
        "return_values_loaded": False,
        "model_admission": False,
        "public_times_certified": 0,
    }
    payloads = {
        "report.json": report,
        "source_clock_provenance.json": resolutions,
        "proposed_acceptance_overlay.json": overlay,
        "candidate_boundary_audit.json": boundaries,
        "missing_artifacts.json": missing,
    }
    files = {name: _json_bytes(value) for name, value in payloads.items()}
    files["readout.md"] = render_provenance(report, overlay, boundaries).encode()
    return write_bundle(output_root, VERSION, files), report


def render_provenance(
    report: dict[str, Any], overlay: list[dict[str, Any]], boundaries: list[dict[str, Any]]
) -> str:
    """Describe provenance and limited clock corrections without performance claims."""
    lines = [
        "# SEC时钟溯源与局部修正提案",
        "",
        f"追溯 {report['traced_filings']}/{report['target_filings']} 份文件；"
        f"提出 {report['proposed_clock_corrections']} 条逐文件接收时钟修正。",
        "该提案尚未写入主库；接收代理时钟不是已认证的市场公开时间。未训练、未读取收益、未改交易规则。",
        "",
        "## 三层证据",
        "",
        "|CIK|Accession|原始JSON时间字面值|目录UTC|原文+索引一致的接收代理UTC|",
        "|---|---|---|---|---|",
    ]
    for row in overlay:
        lines.append(
            f"|{row['cik']}|{row['accession_number']}|{row['raw_acceptance_literal']}|{row['catalog_accepted_at']}|{row['reviewed_sec_acceptance_proxy_at']}|"
        )
    lines += [
        "",
        "表中提案的目录忠实保留JSON显式时区；差异已存在于上游表示，不能靠把全库一律平移修复。",
        "同一SEC来源的两个文件表示相互印证，不等于独立新闻传播证据。",
        "",
        "## 候选边界重检",
        "",
        "|公司|旧10-Q时钟|复核10-Q代理时钟|旧/复核候选数|候选集合未变|",
        "|---|---|---|---|---|",
    ]
    for row in boundaries:
        new = row["reviewed_candidate_accessions"]
        lines.append(
            f"|{row['ticker']}|{row['old_target_accepted_at']}|{row['reviewed_target_proxy_at']}|"
            f"{len(row['old_candidate_accessions'])}/{len(new) if new is not None else '待核实'}|"
            f"{row['candidate_membership_unchanged']}|"
        )
    lines += [
        "",
        "边界检查包括财季结束至目标10-Q接收日、两端各加一天的Item 2.02/7.01候选。",
        "这不是全网首次公告搜索，也不证明未复核公司无时间问题。",
        "",
        "## 下一步",
        "",
        "将原始时间字面值、时区语义和原文校验状态纳入下一版统一时钟层，再逐事件重建受影响特征和执行窗口。",
        "不能只改目标公告时间，却沿用旧预测生成、训练截止和价格窗口。公开时间证据、会计口径审查仍需继续。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run offline unless bounded collection is explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/sec_clock_provenance_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/sec_clock_provenance")
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1 or (args.plan_only and args.collect):
        parser.error("Invalid collection limit or incompatible modes")
    plan_root, plan = freeze_clock_plan(args.config, args.output_root)
    print(
        f"CLOCK PLAN targets={len(plan['targets'])} "
        f"focus_events={len(plan['events'])} output={plan_root}",
        flush=True,
    )
    if args.plan_only:
        return
    if args.collect:
        print(f"CLOCK COLLECT DONE {collect_clock_evidence(plan, limit=args.limit)}", flush=True)
    root, report = run_provenance(plan_root, plan, args.output_root)
    print(
        f"CLOCK PROVENANCE traced={report['traced_filings']}/{report['target_filings']} "
        f"proposals={report['proposed_clock_corrections']} "
        f"missing={report['missing_artifacts']} applied=False output={root}"
    )


if __name__ == "__main__":
    main()
