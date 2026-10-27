"""Review cached OTC identity evidence without changing frozen parsers or samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.collect_publication_market_gaps import load_requests
from scripts.probe_publication_market_access import audit_probe, build_queries
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.next_publication import DISARM_FLAGS
from sec_data.otc_identity import (
    VERSION,
    followup_queries,
    review_consensus,
    review_reference,
    review_sec_issuer_link,
)


def validate_contract(c: dict[str, Any]) -> None:
    """Keep this an offline candidate review, not a new source or return admission."""
    expected = {
        "version": VERSION,
        "frozen_pair_cases": 6,
        "frozen_queue_issuers": 20,
        **dict.fromkeys(
            (
                "network_allowed",
                "forecast_refit_allowed",
                "returns_allowed",
                "overwrite_originals",
                "sample_replacement_allowed",
                "provider_cik_imputation_allowed",
                "venue_crosswalk_imputation_allowed",
                *DISARM_FLAGS,
            ),
            False,
        ),
    }
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in expected.items()):
        raise ValueError("Changed or weakened OTC review contract")


def build_review(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Pin all receipts, an SEC exhibit and the unchanged six-case price-gap ledger."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)
    market_root = Path(c["market_audit_root"])
    market_sha = c["market_audit_manifest_sha256"]
    market = _json(market_root, market_sha, "summary.json")
    prior = _json(Path(market["plan_root"]), market["plan_manifest_sha256"], "plan.json")
    bridge = Path(prior["config"]["bridge_root"])
    bridge_sha = prior["config"]["bridge_manifest_sha256"]
    original_requests = load_requests(bridge, bridge_sha)
    bridge_summary = _json(bridge, bridge_sha, "summary.json")
    bridge_plan = _json(
        Path(bridge_summary["plan_root"]), bridge_summary["plan_manifest_sha256"], "plan.json"
    )
    cases = bridge_plan["cases"]
    (case,) = [r for r in cases if r["ticker"] == "NOBH"]
    if c["sec_raw_sha256"] != case["release_pair_source_sha256"]:
        raise ValueError("SEC identity review is outside the frozen filing event")
    verify_file(Path(c["sec_raw_path"]), c["sec_raw_sha256"])
    source = bridge_plan["config"]
    targets = pd.read_parquet(
        _artifact(
            Path(source["evidence_root"]),
            source["evidence_manifest_sha256"],
            "validation_action_evidence_targets.parquet",
        )
    )
    target_rows = targets.loc[targets.selected_ticker.eq("NOBH")].to_dict("records")
    if len(target_rows) != 1:
        raise ValueError("Ambiguous original OTC target")
    target = {str(k): v for k, v in target_rows[0].items()}
    target["issuer_name"] = c["issuer_name"]
    if (
        target["cik"] != case["cik"]
        or target["security_identity_key_kind"] != "share_class_figi"
        or pd.notna(target["selected_composite_figi"])
    ):
        raise ValueError("The original missing-FIGI target changed")
    raw_root, raw_sha = Path(c["capture_root"]), c["capture_manifest_sha256"]
    capture = _json(raw_root, raw_sha, "receipts.json")
    access_root, access_sha = Path(c["access_audit_root"]), c["access_audit_manifest_sha256"]
    access = _json(access_root, access_sha, "summary.json")
    if access["capture_manifest_sha256"] != raw_sha:
        raise ValueError("Access audit refers to a different capture")
    access_plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    for name, sha in access_plan["implementation_sha256"].items():
        verify_file(Path(name), sha)
    if (
        access_plan["queries"] != build_queries()
        or len(cases) != 6
        or any(market[k] is not False or access[k] is not False for k in DISARM_FLAGS)
    ):
        raise ValueError("Changed cohort, access controls or prior gate")
    # Reproduce the prior normalization in an immutable bundle, never overwrite it.
    replayed, replay_summary = audit_probe(raw_root, raw_sha, output_root / "source_rechecks")
    if digest_file(replayed / "output_manifest.json") != access_sha or replay_summary != access:
        raise ValueError("Access audit cannot be reproduced from original bytes")
    dispositions = _json(market_root, market_sha, "case_dispositions.json")
    gaps = _json(market_root, market_sha, "price_gap_classification.json")
    windows = _json(market_root, market_sha, "window_audits.json")
    if len(dispositions) != 20 or len(windows) != 7:
        raise ValueError("Changed disposition or date-window ledger")
    plan = {
        "config": c,
        "cases": cases,
        "target": {
            k: target[k]
            for k in (
                "cik",
                "selected_ticker",
                "selected_exchange_mic",
                "security_identity_key",
                "security_identity_key_kind",
            )
        },
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/review_otc_identity.py",
                "sec_data/otc_identity.py",
                "sec_data/disclosure_review.py",
                "sec_data/guidance_evidence.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    sec = review_sec_issuer_link(
        Path(c["sec_raw_path"]).read_bytes(), case, c["exhibit_filename"], c["issuer_name"]
    )
    receipts = {r["query_id"]: r for r in capture["receipts"]}
    rows = []
    for query in build_queries():
        if query["kind"] != "reference" or query["route"] == "cik":
            continue
        r = receipts[query["id"]]
        if (
            r["requested_url"] != query["url"]
            or r["status_code"] != 200
            or r["file"] != query["id"] + ".json"
        ):
            raise ValueError("OTC response rebound or unsuccessful")
        body = _artifact(raw_root, raw_sha, r["file"]).read_bytes()
        if len(body) != r["bytes"]:
            raise ValueError("Raw response length changed")
        row = review_reference(query, body, r["sha256"], target)
        row["source_fetched_at"] = r["fetched_at"]
        rows.append(row)
    consensus = review_consensus(rows, sec)
    (q,) = [
        r
        for r in original_requests
        if r["ticker"] == "NOBH" and r["query_kind"] == "listing_active"
    ]
    followups = followup_queries(
        consensus, "NOBH", q["window_start"], q["window_end"], q["listing_outcome_cutoff"]
    )
    followup_plan = {
        "requests": followups,
        "maximum_initial_requests": 5,
        "request_interval_seconds": 12.1,
        "diagnostic_only_not_admitted": True,
        "network_execution_authorized": False,
        "cik_filter_removed_not_cik_imputed": True,
        "all_unresolved_requirements_retained": True,
        "source_reference_sha256": sorted({r["source_sha256"] for r in rows}),
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "issuer_ticker_documented_in_original_sec": sec["issuer_ticker_documented"],
        "candidate_association_supported": consensus["candidate_association_supported"],
        "provider_cik_missing_rows": sum(r["provider_cik_state"] == "missing" for r in rows),
        "candidate_composite_figi": consensus["candidate_composite_figi"],
        "reference_candidate_rows": len(rows),
        "network_requests": 0,
        "followup_requests_prepared": len(followups),
        "frozen_queue_issuers_retained": len(dispositions),
        "frozen_pair_cases_retained": len(cases),
        "price_gap_rows_retained": len(gaps),
        "metadata_ready_tickers_unchanged": market["metadata_window_ready_tickers"],
        "original_provider_cik_and_figi_overwritten": False,
        "mic_mapping_invented": False,
        "financial_forecasts_refitted": False,
        "returns_computed": False,
        "historical_security_interval_ready": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    readout = (
        "OTC身份证据复核（不是Alpha或收益实验）\n\n"
        "原始SEC公告的CIK、公司名及正文(OTCQX: NOBH)关联已核对。\n"
        f"供应商候选匹配{sum(r['candidate_match'] for r in rows)}/{len(rows)}条；"
        f"候选关联支持={consensus['candidate_association_supported']}。CIK缺失或冲突均原样保留。\n"
        "这是可追溯候选关联，不是认证的历史证券区间。OTC Link标签未自动映射为OTCM。\n"
        "控制日期不是持有窗口端点，区间连续性、退市状态、缺日和历史首次公开时刻仍未通过。\n"
        f"保留20家公司、6个案例和{len(gaps)}条缺日记录，行情完整名单不变。\n"
        f"已准备{len(followups)}个端点/时间线/上市状态查询（上限5个），尚未执行；"
        "本轮不联网、不改模型、不填零、不算部分样本收益。\n"
    )
    return write_bundle(
        output_root,
        VERSION,
        {
            "summary.json": _encode(summary),
            "sec_issuer_ticker_link.json": _encode(sec),
            "reference_candidates.json": _encode(rows),
            "candidate_consensus.json": _encode(consensus),
            "followup_plan.json": _encode(followup_plan),
            "case_dispositions.json": _encode(dispositions),
            "price_gap_classification.json": _encode(gaps),
            "readout.txt": readout.encode("utf-8"),
        },
    ), summary


def main() -> None:
    """Run offline only; credentials are neither read nor needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/otc_identity_review_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/otc_identity_review")
    )
    args = parser.parse_args()
    root, summary = build_review(args.config, args.output_root)
    print(f"OTC IDENTITY REVIEW DONE output={root} summary={json.dumps(summary)}")


if __name__ == "__main__":
    main()
