"""Capture five frozen OTC identity queries; no prices, refits or return admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.review_otc_identity import build_review
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_reference import MassiveRequestError
from sec_data.massive_security_actions import parse_massive_ticker_events
from sec_data.next_publication import DISARM_FLAGS
from sec_data.otc_identity import followup_queries

VERSION = "otc-identity-followup-v1"
REVIEW = Path("data/processed/otc_identity_review/otc-identity-review-v1-d27c000d78e060d3")
REVIEW_SHA = "8a5803c13a7465e5009ccecaa7b10226251a844fb029a33d0d466014d594627b"


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject query growth, destination changes, invented IDs and relaxed gates."""
    expected = followup_queries(
        {"candidate_association_supported": True, "candidate_composite_figi": "BBG000BQ2104"},
        "NOBH",
        "2024-09-11",
        "2024-10-09",
        "2026-08-14",
    )
    if (
        plan["queries"] != expected
        or type(plan["maximum_requests"]) is not int
        or plan["maximum_requests"] != 5
        or plan["request_interval_seconds"] != 12.1
        or type(plan["retries"]) is not int
        or plan["retries"] != 0
        or plan["requires_explicit_collect_invocation"] is not True
        or plan["old_offline_preparation_authorization_not_overwritten"] is not True
        or plan["target"]["security_identity_key"] != "BBG001S5TP08"
        or plan["target"]["cik"] != "0000072205"
        or plan["target"]["selected_ticker"] != "NOBH"
        or plan["target"]["selected_exchange_mic"] != "OTCM"
        or plan["candidate_composite_figi"] != "BBG000BQ2104"
        or plan["window_start"] != "2024-09-11"
        or plan["window_end"] != "2024-10-09"
        or plan["listing_cutoff"] != "2026-08-14"
        or any(
            plan.get(k) is not False
            for k in (
                *DISARM_FLAGS,
                "follow_redirects",
                "follow_pagination",
                "returns_allowed",
                "forecast_refit_allowed",
            )
        )
    ):
        raise ValueError("Changed or unsafe five-query OTC follow-up plan")


def freeze_plan(
    review_root: Path, review_sha: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Reproduce prior evidence and bind exact queries before any credential read."""
    summary = _json(review_root, review_sha, "summary.json")
    prior = _json(Path(summary["plan_root"]), summary["plan_manifest_sha256"], "plan.json")
    for path, sha in prior["implementation_sha256"].items():
        verify_file(Path(path), sha)
    config = Path("docs/otc_identity_review_v1.json")
    if json.loads(config.read_bytes()) != prior["config"]:
        raise ValueError("The source review configuration changed")
    repeated, _ = build_review(config, review_root.parent)
    if repeated != review_root or digest_file(repeated / "output_manifest.json") != review_sha:
        raise ValueError("Original OTC candidate review cannot be reproduced")
    prepared = _json(review_root, review_sha, "followup_plan.json")
    consensus = _json(review_root, review_sha, "candidate_consensus.json")
    if (
        prepared["network_execution_authorized"] is not False
        or consensus["candidate_association_supported"] is not True
    ):
        raise ValueError("Unexpected old preparation or unresolved candidate")
    target = {**prior["target"], "issuer_name": prior["config"]["issuer_name"]}
    plan = {
        "source_review_root": review_root.as_posix(),
        "source_review_manifest_sha256": review_sha,
        "queries": prepared["requests"],
        "target": target,
        "candidate_composite_figi": consensus["candidate_composite_figi"],
        "window_start": "2024-09-11",
        "window_end": "2024-10-09",
        "listing_cutoff": "2026-08-14",
        "maximum_requests": 5,
        "request_interval_seconds": 12.1,
        "retries": 0,
        "requires_explicit_collect_invocation": True,
        "old_offline_preparation_authorization_not_overwritten": True,
        "follow_redirects": False,
        "follow_pagination": False,
        "returns_allowed": False,
        "forecast_refit_allowed": False,
        **dict.fromkeys(DISARM_FLAGS, False),
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/collect_otc_identity_followup.py",
                "scripts/review_otc_identity.py",
                "sec_data/otc_identity.py",
                "sec_data/massive_security_actions.py",
                "scripts/build_sec_clock_layer.py",
                "scripts/build_sec_disclosure_audit.py",
            )
        },
    }
    validate_plan(plan)
    return write_bundle(output_root, VERSION + "-plan", {"plan.json": _encode(plan)}), plan


def collect_queries(
    plan_root: Path, plan_sha: str, raw_root: Path, key: str, sender: httpx.Client
) -> Path:
    """Preserve exact successful/error bytes before parsing; never retry or redirect.

    Stop on authentication, rate limits, redirects, server/network failures or
    reflected credentials. Other planned queries can run after a 403 or 404;
    no fallback or pagination requests are invented.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{32}", key):
        raise ValueError("Missing or malformed Massive key")
    plan = _json(plan_root, plan_sha, "plan.json")
    validate_plan(plan)
    if sender.follow_redirects:
        raise ValueError("Redirect following is forbidden")
    for path, sha in plan["implementation_sha256"].items():
        verify_file(Path(path), sha)
    files: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    last_start, stop = None, None
    for query in plan["queries"]:
        r: dict[str, Any] = {
            "kind": query["kind"],
            "requested_url": query["url"],
            "status_code": None,
        }
        if stop:
            r["skipped"] = stop
        else:
            if last_start is not None:
                time.sleep(max(0, 12.1 - (time.monotonic() - last_start)))
            last_start = time.monotonic()
            r["fetched_at"] = datetime.now(timezone.utc).isoformat()
            try:
                response = sender.get(
                    query["url"],
                    headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                )
            except httpx.RequestError:
                stop = r["error"] = "network_failure"
            else:
                body, code = response.content, response.status_code
                r.update(status_code=code, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                if key.encode() in body:
                    stop = r["error"] = "credential_reflected_body_withheld"
                else:
                    name = query["kind"] + ".json"
                    r["file"] = name
                    write_bundle(
                        raw_root / "responses",
                        VERSION + "-response",
                        {name: body, "receipt.json": _encode(r)},
                    )
                    files[name] = body
                    if code in {401, 429} or 300 <= code < 400 or code >= 500:
                        stop = f"http_{code}"
        receipts.append(r)
        print(
            f"OTC FOLLOWUP kind={query['kind']} http={r['status_code']} skipped={r.get('skipped')}",
            flush=True,
        )
    files["receipts.json"] = _encode(
        {"plan_root": plan_root.as_posix(), "plan_manifest_sha256": plan_sha, "receipts": receipts}
    )
    return write_bundle(raw_root, VERSION + "-capture", files)


def inspect_query(
    plan: dict[str, Any], query: dict[str, str], body: bytes, sha: str, code: int
) -> dict[str, Any]:
    """Parse IDs and dated ticker events, never infer zero returns or terminal prices."""
    if hashlib.sha256(body).hexdigest() != sha:
        raise ValueError("Follow-up response hash changed")
    kind, target = query["kind"], plan["target"]
    result: dict[str, Any] = {
        "kind": kind,
        "http_status": code,
        "source_sha256": sha,
        "provider_status": None,
        "query_complete": False,
        "identity_candidate_match": False,
        "identities": [],
        "events": [],
        "rejections": [],
        "empty_not_delisting_or_no_trade": True,
        "historical_security_interval_ready": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    try:
        d = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {**result, "parser_error": "malformed_json"}
    if not isinstance(d, dict):
        return {**result, "parser_error": "invalid_envelope"}
    status = d.get("status")
    result["provider_status"] = (
        status
        if isinstance(status, str) and status in {"OK", "NOT_AUTHORIZED", "ERROR"}
        else "unrecognized"
    )
    if code != 200 or status != "OK" or d.get("next_url"):
        return result
    if kind == "ticker_events":
        timeline = d.get("results")
        if (
            not isinstance(timeline, dict)
            or timeline.get("composite_figi") != plan["candidate_composite_figi"]
            or not isinstance(timeline.get("events"), list)
        ):
            return {**result, "parser_error": "timeline_identity_or_schema_mismatch"}
        try:
            rows, bad, _ = parse_massive_ticker_events(
                body, action_target_id="otc-diagnostic-only", source_sha256=sha
            )
        except MassiveRequestError as exc:
            return {**result, "parser_error": exc.reason}
        result.update(query_complete=not bad, events=rows, rejections=bad)
        result["in_window_events"] = [
            r for r in rows if plan["window_start"] <= r["event_date"] <= plan["window_end"]
        ]
        result["outside_window_events_not_used"] = len(rows) - len(result["in_window_events"])
        return result
    rows = d.get("results", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return {**result, "parser_error": "invalid_reference_schema"}
    active = kind != "listing_inactive"
    day = (
        plan["listing_cutoff"]
        if kind.startswith("listing_")
        else plan["window_start" if kind == "reference_start" else "window_end"]
    )
    for row in rows:
        cik = row.get("cik")
        state = "missing" if cik is None or cik == "" else "conflict"
        if (
            isinstance(cik, str)
            and re.fullmatch(r"\d{1,10}", cik)
            and cik.zfill(10) == target["cik"]
        ):
            state = "match"
        reasons = [
            field + "_mismatch"
            for field, value in (
                ("ticker", target["selected_ticker"]),
                ("name", target["issuer_name"]),
                ("market", "otc"),
                ("locale", "us"),
                ("type", "CS"),
                ("share_class_figi", target["security_identity_key"]),
                ("composite_figi", plan["candidate_composite_figi"]),
            )
            if row.get(field) != value
        ]
        if row.get("active") is not active:
            reasons.append("active_state_mismatch")
        if state == "conflict":
            reasons.append("provider_cik_conflict")
        label = row.get("primary_exchange")
        mic = label if isinstance(label, str) and re.fullmatch(r"[A-Z0-9]{4}", label) else None
        if mic is not None and mic != target["selected_exchange_mic"]:
            reasons.append("observed_mic_conflict")
        result["identities"].append(
            {
                "as_of_date": day,
                "ticker": row.get("ticker"),
                "provider_cik": cik,
                "provider_cik_state": state,
                "share_class_figi": row.get("share_class_figi"),
                "composite_figi": row.get("composite_figi"),
                "provider_active": row.get("active"),
                "raw_primary_exchange": label,
                "observed_mic": mic,
                "provider_delisted_utc": row.get("delisted_utc"),
                "last_updated_utc": row.get("last_updated_utc"),
                "candidate_match": not reasons,
                "reasons": reasons,
                "provider_cik_imputed": False,
                "venue_label_converted_to_mic": False,
            }
        )
    result["query_complete"] = True
    result["identity_candidate_match"] = (
        len(rows) == 1 and result["identities"][0]["candidate_match"]
    )
    return result


def audit_capture(
    capture_root: Path, capture_sha: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Audit all five receipts offline, keeping every original case and gap record."""
    capture = _json(capture_root, capture_sha, "receipts.json")
    plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    validate_plan(plan)
    for path, sha in plan["implementation_sha256"].items():
        verify_file(Path(path), sha)
    if [r["kind"] for r in capture["receipts"]] != [q["kind"] for q in plan["queries"]]:
        raise ValueError("Capture differs from the five planned queries")
    audits = {}
    for q, r in zip(plan["queries"], capture["receipts"], strict=True):
        if r["requested_url"] != q["url"]:
            raise ValueError("Follow-up request URL rebound")
        if "file" not in r:
            audits[q["kind"]] = {
                "query_complete": False,
                "identity_candidate_match": False,
                "untested": True,
                "reason": r.get("skipped") or r.get("error"),
            }
            continue
        if r["file"] != q["kind"] + ".json":
            raise ValueError("Follow-up response file rebound")
        body = _artifact(capture_root, capture_sha, r["file"]).read_bytes()
        if len(body) != r["bytes"]:
            raise ValueError("Response length changed")
        audits[q["kind"]] = inspect_query(plan, q, body, r["sha256"], r["status_code"])
    root, sha = Path(plan["source_review_root"]), plan["source_review_manifest_sha256"]
    gaps = _json(root, sha, "price_gap_classification.json")
    dispositions = _json(root, sha, "case_dispositions.json")
    active, inactive = audits["listing_active"], audits["listing_inactive"]
    listing_queries_complete = active["query_complete"] and inactive["query_complete"]
    summary = {
        "version": VERSION,
        "capture_manifest_sha256": capture_sha,
        "plan_manifest_sha256": capture["plan_manifest_sha256"],
        "requests_made": sum("fetched_at" in r for r in capture["receipts"]),
        "http_responses_saved": sum("file" in r for r in capture["receipts"]),
        "queries_complete": sum(r["query_complete"] for r in audits.values()),
        "exact_window_endpoint_candidates_match": all(
            audits[k]["identity_candidate_match"] for k in ("reference_start", "reference_end")
        ),
        "timeline_query_complete": audits["ticker_events"]["query_complete"],
        "in_window_ticker_events": len(audits["ticker_events"]["in_window_events"])
        if audits["ticker_events"]["query_complete"]
        else None,
        "observed_active_candidate_at_cutoff": bool(
            listing_queries_complete
            and active["identity_candidate_match"]
            and not inactive.get("identities")
        ),
        "observed_inactive_candidate_at_cutoff": bool(
            listing_queries_complete
            and inactive["identity_candidate_match"]
            and not active.get("identities")
        ),
        "listing_outcome_formally_ready": False,
        "historical_security_interval_ready": False,
        "frozen_queue_issuers_retained": len(dispositions),
        "frozen_pair_cases_retained": 6,
        "price_gap_rows_retained": len(gaps),
        "distinct_price_gaps_retained": len({r["session_date"] for r in gaps}),
        "original_snapshots_overwritten": False,
        "provider_cik_imputed": False,
        "mic_crosswalk_invented": False,
        "prices_requested": False,
        "returns_computed": False,
        "forecasts_refitted": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    readout = (
        "OTC身份补查（不是收益或Alpha实验）\n\n"
        f"HTTP响应已保存{summary['http_responses_saved']}/5，解析完整{summary['queries_complete']}/5。\n"
        f"持有窗口端点候选一致={summary['exact_window_endpoint_candidates_match']}；时间线查询完整={summary['timeline_query_complete']}。\n"
        f"窗口内ticker事件数={summary['in_window_ticker_events']}（未完成时未知，不填零）。\n"
        f"截止日观察到唯一活跃候选={summary['observed_active_candidate_at_cutoff']}；唯一非活跃候选={summary['observed_inactive_candidate_at_cutoff']}。\n"
        "以上是供应商历史重建视图，不认证首次公开时刻、交易场所或正式退市收益。\n"
        f"保留20家公司、6个案例和{summary['distinct_price_gaps_retained']}个不同缺价日期；不补CIK/MIC、不改模型、不算收益。\n"
    )
    return write_bundle(
        output_root,
        VERSION + "-audit",
        {
            "summary.json": _encode(summary),
            "query_audits.json": _encode(audits),
            "case_dispositions.json": _encode(dispositions),
            "price_gap_classification.json": _encode(gaps),
            "readout.txt": readout.encode("utf-8"),
        },
    ), summary


def main() -> None:
    """Default to offline freezing; --collect requires a process-local environment key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=REVIEW)
    parser.add_argument("--review-manifest-sha256", default=REVIEW_SHA)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest-sha256")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/otc_identity_followup"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/otc_identity_followup")
    )
    args = parser.parse_args()
    if (args.collect and args.capture_root) or (args.capture_root is None) != (
        args.capture_manifest_sha256 is None
    ):
        parser.error("Choose explicit collection or a complete offline capture binding")
    if args.capture_root is None:
        root, _ = freeze_plan(
            args.review_root, args.review_manifest_sha256, args.raw_root / "plans"
        )
        sha = digest_file(root / "output_manifest.json")
        print(f"OTC FOLLOWUP PLAN output={root} manifest={sha}")
        if not args.collect:
            return
        with httpx.Client(timeout=25, follow_redirects=False) as sender:
            args.capture_root = collect_queries(
                root, sha, args.raw_root, os.environ.get("MASSIVE_API_KEY", "").strip(), sender
            )
        args.capture_manifest_sha256 = digest_file(args.capture_root / "output_manifest.json")
    root, summary = audit_capture(args.capture_root, args.capture_manifest_sha256, args.output_root)
    print(f"OTC FOLLOWUP AUDIT DONE output={root} summary={json.dumps(summary)}")


if __name__ == "__main__":
    main()
