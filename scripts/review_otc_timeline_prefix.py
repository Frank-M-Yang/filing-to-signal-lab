"""Offline consistency check: a pre-window ticker event is not irrelevant history."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from scripts.build_sec_clock_layer import _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.collect_otc_identity_followup import audit_capture
from sec_data.disclosure_audit import digest_file
from sec_data.next_publication import DISARM_FLAGS

VERSION = "otc-timeline-prefix-review-v1"
AUDIT = Path("data/processed/otc_identity_followup/otc-identity-followup-v1-audit-5d9224bbb9ef21bd")
AUDIT_SHA = "d957cf4f9c43e96871737db3d8cd4a5ee8f1134e198d13297f3668a554e0c8f8"
CAPTURE = Path("data/raw/otc_identity_followup/otc-identity-followup-v1-capture-f2bcdf59a2bb7515")
DOC_URL = "https://massive.com/docs/rest/stocks/corporate-actions/ticker-events"


def assess_prefix(
    audits: dict[str, Any], ticker: str, window_start: str, window_end: str
) -> dict[str, Any]:
    """Compare conditional ticker states with both endpoint candidates.

    Date-only events at a boundary cannot establish an intraday state. A later
    event is never carried backwards. Even agreement cannot certify timeline
    completeness, venue scope, first-public availability or trade eligibility.
    """
    start, end = date.fromisoformat(window_start), date.fromisoformat(window_end)
    if end < start:
        raise ValueError("Reversed timeline review window")
    timeline = audits["ticker_events"]
    endpoints = all(
        audits[k].get("query_complete") is True
        and audits[k].get("identity_candidate_match") is True
        and len(audits[k].get("identities", [])) == 1
        and audits[k]["identities"][0].get("ticker") == ticker
        for k in ("reference_start", "reference_end")
    )
    result: dict[str, Any] = {
        "expected_endpoint_ticker": ticker,
        "endpoint_candidates_match": endpoints,
        "timeline_query_complete": timeline.get("query_complete") is True,
        "conditional_boundary_states": [],
        "prefix_conflict_detected": None,
        "assessment": "insufficient_evidence",
        "scope_and_completeness_not_certified": True,
        "date_only_events_not_intraday_certified": True,
        "historical_security_interval_ready": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    if not result["timeline_query_complete"]:
        return result
    groups: dict[date, set[str]] = {}
    future_count = 0
    for row in timeline["events"]:
        day = date.fromisoformat(row["event_date"])
        symbol = row["new_ticker"]
        if row["event_type"] != "ticker_change" or not isinstance(symbol, str) or not symbol:
            raise ValueError("Malformed normalized ticker event")
        if day > end:
            future_count += 1
            continue
        groups.setdefault(day, set()).add(symbol)
    result["events_after_window_ignored"] = future_count
    conflicts, unknown = [], False
    for name, boundary in (("start", start), ("end", end)):
        earlier = [d for d in groups if d < boundary]
        latest = max(earlier) if earlier else None
        symbols = groups[latest] if latest is not None else set()
        ambiguous = boundary in groups or len(symbols) != 1
        symbol = next(iter(symbols)) if not ambiguous else None
        mismatch = symbol != ticker if symbol is not None else None
        conflicts.append(mismatch is True)
        unknown |= ambiguous
        result["conditional_boundary_states"].append(
            {
                "boundary": name,
                "date": boundary.isoformat(),
                "latest_strictly_prior_event_date": latest.isoformat() if latest else None,
                "conditional_ticker": symbol,
                "same_date_or_missing_state_ambiguous": ambiguous,
                "conflicts_with_endpoint_ticker": mismatch,
            }
        )
    in_window = [symbols for day, symbols in groups.items() if start <= day <= end]
    result["in_window_nonmatching_symbols"] = sorted(
        {s for g in in_window for s in g if s != ticker}
    )
    unknown |= any(len(g) != 1 for g in in_window)
    conflict = any(conflicts) or bool(result["in_window_nonmatching_symbols"])
    # A negative prefix result is diagnostic only: experimental endpoint scope
    # may differ from US-OTC reference scope. Do not assert a real-world rename.
    result["prefix_conflict_detected"] = (
        (True if conflict else None if unknown else False) if endpoints else None
    )
    if endpoints and conflict:
        result["assessment"] = "conditional_timeline_reference_conflict"
    elif endpoints and not unknown:
        result["assessment"] = "candidate_consistent_not_certified"
    return result


def build_review(
    audit_root: Path, audit_sha: str, capture_root: Path, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Reproduce the raw-bound capture audit before a separate immutable review."""
    summary = _json(audit_root, audit_sha, "summary.json")
    repeated, same = audit_capture(
        capture_root, summary["capture_manifest_sha256"], audit_root.parent
    )
    if repeated != audit_root or same != summary:
        raise ValueError("Source capture audit cannot be reproduced")
    audits = _json(audit_root, audit_sha, "query_audits.json")
    assessment = assess_prefix(audits, "NOBH", "2024-09-11", "2024-10-09")
    plan = {
        "source_audit_root": audit_root.as_posix(),
        "source_audit_manifest_sha256": audit_sha,
        "source_capture_root": capture_root.as_posix(),
        "source_capture_manifest_sha256": summary["capture_manifest_sha256"],
        "ticker": "NOBH",
        "window_start": "2024-09-11",
        "window_end": "2024-10-09",
        "method": "Strictly-prior date prefix; same-date intraday state is unknown",
        "provider_documentation_url": DOC_URL,
        "endpoint_documented_experimental": True,
        "implementation_sha256": {
            "scripts/review_otc_timeline_prefix.py": digest_file(Path(__file__))
        },
        "network_allowed": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    plan_root = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    result = {
        **assessment,
        "version": VERSION,
        "plan_manifest_sha256": digest_file(plan_root / "output_manifest.json"),
        "source_timeline_sha256": audits["ticker_events"]["source_sha256"],
        "frozen_queue_issuers_retained": summary["frozen_queue_issuers_retained"],
        "frozen_pair_cases_retained": summary["frozen_pair_cases_retained"],
        "distinct_price_gaps_retained": summary["distinct_price_gaps_retained"],
        "network_requests": 0,
        "returns_computed": False,
        "forecasts_refitted": False,
        "original_snapshots_overwritten": False,
    }
    states = result["conditional_boundary_states"]
    readout = (
        "OTC时间线一致性复核（离线，不是Alpha实验）\n\n"
        f"结果：{result['assessment']}。\n"
        + "".join(
            f"{s['date']}端点查询=NOBH；此前最近事件日期={s['latest_strictly_prior_event_date']}，"
            f"按ticker_change状态语义推演={s['conditional_ticker']}。\n"
            for s in states
        )
        + "窗口内没有改名记录，不代表窗口开始时的历史状态一致。\n"
        "这是两个返回口径间的条件性矛盾，不是已证实的真实改名、退市、欺诈或无交易。\n"
        "该事件端点被官方标为experimental，场所范围和历史完整性尚未核实；不采用S200替换价格代码。\n"
        "保留原始响应和之前的结果，另建复核记录；"
        f"保留{result['frozen_queue_issuers_retained']}家公司、{result['frozen_pair_cases_retained']}个案例"
        f"及{result['distinct_price_gaps_retained']}个缺价日期。\n"
        "下一步需要核实时间线/场所口径，并取得缺失价格或可审计的缺失原因；正式收益与模型准入仍关闭。\n"
        f"接口说明：{DOC_URL}\n"
    )
    files = {
        "summary.json": _encode(result),
        "readout.txt": readout.encode("utf-8"),
        "case_dispositions.json": _encode(_json(audit_root, audit_sha, "case_dispositions.json")),
        "price_gap_classification.json": _encode(
            _json(audit_root, audit_sha, "price_gap_classification.json")
        ),
    }
    return write_bundle(output_root, VERSION, files), result


def main() -> None:
    """Run a source-bound offline review; this command has no collection option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument("--audit-manifest-sha256", default=AUDIT_SHA)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/otc_timeline_prefix")
    )
    args = parser.parse_args()
    root, summary = build_review(
        args.audit_root, args.audit_manifest_sha256, args.capture_root, args.output_root
    )
    print(f"OTC PREFIX REVIEW DONE assessment={summary['assessment']} gate=False output={root}")


if __name__ == "__main__":
    main()
