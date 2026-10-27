"""Explicit, bounded source capture for the frozen publication-market gap plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd

from scripts.build_publication_market_bridge import build_requests
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_prices import massive_daily_bars_url
from sec_data.massive_reference import (
    MASSIVE_TICKERS_URL,
    MassiveReferenceClient,
    MassiveRequestError,
)
from sec_data.massive_security_actions import (
    massive_dividends_url,
    massive_splits_url,
    massive_ticker_events_url,
)
from sec_data.next_publication import DISARM_FLAGS
from sec_data.publication_market_bridge import close_common_gate, validate_contract

VERSION = "publication-market-gap-capture-v1"


def validate_requests(requests: list[dict[str, Any]]) -> None:
    """Reject expanded budgets, secret-bearing URLs, invented FIGIs or changed bounds."""
    if not 0 < len(requests) <= 12 or len({r["query_id"] for r in requests}) != len(requests):
        raise ValueError("One bounded, non-duplicate request plan is required")
    for r in requests:
        start, end = (date.fromisoformat(r[k]) for k in ("window_start", "window_end"))
        if (
            start > end
            or r["ticker"] not in {"CTSO", "NOBH"}
            or not re.fullmatch(r"\d{10}", r["cik"])
            or not re.fullmatch(r"BBG[A-Z0-9]{9}", r["security_id"])
            or r["security_id_kind"] != "share_class_figi"
            or r["adjusted"] is not False
            or r["selection_uses_return_values"] is not False
            or any(r.get(k) is not False for k in DISARM_FLAGS)
            or hashlib.sha256(_encode({k: v for k, v in r.items() if k != "query_id"})).hexdigest()
            != r["query_id"]
        ):
            raise ValueError("Unsafe request identity or weakened outcome gate")
        kind = r["query_kind"]
        if kind == "daily_bars":
            expected = massive_daily_bars_url(r["ticker"], start.isoformat(), end.isoformat())
        elif kind in {"splits", "dividends"}:
            builder = massive_splits_url if kind == "splits" else massive_dividends_url
            expected = builder(r["ticker"], start.isoformat(), end.isoformat())
        elif kind == "ticker_events":
            figi = r["selected_composite_figi"]
            if not isinstance(figi, str) or not re.fullmatch(r"BBG[A-Z0-9]{9}", figi):
                raise ValueError("Ticker events require an explicit original Composite FIGI")
            expected = massive_ticker_events_url(figi)
        elif kind in {"reference_start", "reference_end", "listing_active", "listing_inactive"}:
            day = (
                r["listing_outcome_cutoff"]
                if kind.startswith("listing_")
                else r["window_start"]
                if kind == "reference_start"
                else r["window_end"]
            )
            if date.fromisoformat(day) < end and kind.startswith("listing_"):
                raise ValueError("Listing cutoff precedes the retained return window")
            expected = (
                MASSIVE_TICKERS_URL
                + "?"
                + urlencode(
                    {
                        "cik": r["cik"],
                        "date": day,
                        "active": "false" if kind == "listing_inactive" else "true",
                        "order": "asc",
                        "limit": 1000,
                        "sort": "ticker",
                    }
                )
            )
        else:
            raise ValueError("Query outside the frozen endpoint allowlist")
        if r["url"] != expected:
            raise ValueError("URL differs from exact secret-free query specification")


def load_requests(root: Path, manifest_sha256: str) -> list[dict[str, Any]]:
    """Verify the financial/source chain and reproduce queries before any HTTP request."""
    summary = _json(root, manifest_sha256, "summary.json")
    collection = _json(root, manifest_sha256, "collection_plan.json")
    audits = _json(root, manifest_sha256, "window_audits.json")
    plan = _json(Path(summary["plan_root"]), summary["plan_manifest_sha256"], "plan.json")
    c = plan["config"]
    validate_contract(c)
    for name, sha in plan["implementation_sha256"].items():
        verify_file(Path(name), sha)
    financial = _json(
        Path(c["interval_root"]), c["interval_manifest_sha256"], "scenario_case_audit.json"
    )
    gate = close_common_gate(audits, financial)
    if any(summary[k] != v for k, v in gate.items()) or any(
        collection.get(k) is not False for k in DISARM_FLAGS
    ):
        raise ValueError("Bridge gate differs from retained six-case evidence")
    if (
        collection["maximum_initial_requests"] != 12
        or collection["request_interval_seconds"] != 12.1
        or collection["pagination_requires_a_new_plan"] is not True
        or collection["signals_refitted"] is not False
        or collection["signal_freeze_manifest_sha256"] != c["interval_manifest_sha256"]
    ):
        raise ValueError("The bounded capture contract changed")
    evidence = pd.read_parquet(
        _artifact(
            Path(c["evidence_root"]),
            c["evidence_manifest_sha256"],
            "validation_action_evidence_targets.parquet",
        )
    )
    m = plan["market_config"]
    listing = pd.read_parquet(
        _artifact(
            Path(m["listing_root"]),
            m["listing_manifest_sha256"],
            "validation_security_listing_outcomes.parquet",
        )
    )
    expected = build_requests(audits, evidence, listing)
    if expected != collection["requests"]:
        raise ValueError("Requests differ from the source-pinned observed gaps")
    validate_requests(expected)
    return expected


def collect_requests(
    requests: list[dict[str, Any]],
    raw_root: Path,
    client: MassiveReferenceClient,
    *,
    source_manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Save exact bytes before JSON inspection; do not parse bars or admit returns.

    A resumed run reuses only hash-verified successful captures. Pagination is
    recorded but never followed. Authentication or entitlement failure stops the
    batch. Empty successful responses remain evidence, not a zero-return claim.
    """
    validate_requests(requests)
    if client.max_retries != 0 or client.request_interval_seconds < 12.1:
        raise ValueError("Capture requires no retries and at least 12.1 seconds between requests")
    if not re.fullmatch(r"[a-f0-9]{64}", source_manifest_sha256):
        raise ValueError("A source manifest hash is required")
    capture_plan = {
        "requests": requests,
        "source_manifest_sha256": source_manifest_sha256,
        "collector_sha256": digest_file(Path(__file__)),
        "maximum_requests_per_run": 12,
        "request_interval_seconds": client.request_interval_seconds,
        "retries": 0,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    plan_root = write_bundle(
        raw_root / "plans", VERSION + "-plan", {"plan.json": _encode(capture_plan)}
    )
    receipts: list[dict[str, Any]] = []
    requests_made, reused, blocked = 0, 0, None
    for r in requests:
        folder = raw_root / "responses" / r["query_id"]
        cached = sorted(folder.glob(VERSION + "-response-*/output_manifest.json"))
        if len(cached) > 1:
            raise ValueError("Ambiguous cached capture; manual review is required")
        if cached:
            saved = _json(cached[0].parent, digest_file(cached[0]), "receipt.json")
            _artifact(cached[0].parent, digest_file(cached[0]), "response.json")
            if saved["request"] != r or saved["source_manifest_sha256"] != source_manifest_sha256:
                raise ValueError("Cached request or source binding changed")
            receipts.append(saved)
            reused += 1
            continue
        try:
            response = client.get(r["url"])
        except MassiveRequestError as exc:
            requests_made += exc.requests_made
            failure = {
                "request": r,
                "reason": exc.reason,
                "raw_error_response_not_available_from_client": True,
            }
            receipts.append(failure)
            write_bundle(
                raw_root / "failures", VERSION + "-failure", {"failure.json": _encode(failure)}
            )
            if exc.reason in {"authentication_failed", "entitlement_denied"}:
                blocked = exc.reason
                break
            continue
        requests_made += response.requests_made
        receipt: dict[str, Any] = {
            "request": r,
            "source_manifest_sha256": source_manifest_sha256,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
            "status_code": response.status_code,
            "fetched_at": response.fetched_at.isoformat(),
            "response_url": response.response_url,
            "content_type": response.headers.get("content-type"),
            "response_not_yet_interpreted": True,
        }
        # Persist every successful HTTP payload before inspecting even its envelope.
        saved_root = write_bundle(
            folder,
            VERSION + "-response",
            {"response.json": response.content, "receipt.json": _encode(receipt)},
        )
        receipts.append(receipt)
        print(
            f"PUBLICATION GAP CAPTURE ticker={r['ticker']} kind={r['query_kind']} "
            f"saved={saved_root.name}",
            flush=True,
        )
    envelopes = []
    for receipt in receipts:
        if "sha256" not in receipt:
            continue
        r = receipt["request"]
        folder = raw_root / "responses" / r["query_id"]
        manifest = next(folder.glob(VERSION + "-response-*/output_manifest.json"))
        payload = _artifact(manifest.parent, digest_file(manifest), "response.json").read_bytes()
        try:
            data = json.loads(payload)
            valid = isinstance(data, dict) and data.get("status") in {"OK", "DELAYED"}
            paginated = isinstance(data, dict) and bool(data.get("next_url"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            valid, paginated = False, False
        envelopes.append(
            {
                "query_id": r["query_id"],
                "provider_envelope_ok": valid,
                "pagination_present": paginated,
            }
        )
    summary = {
        "version": VERSION,
        "capture_plan_root": plan_root.as_posix(),
        "planned_queries": len(requests),
        "responses_saved": len(envelopes),
        "verified_cache_reused": reused,
        "requests_made": requests_made,
        "blocking_reason": blocked,
        "pagination_followed": False,
        "failed_queries": sum("reason" in r for r in receipts),
        "raw_json_envelopes_decoded_after_capture": True,
        "bars_or_actions_normalized": False,
        "numeric_prices_used_for_selection_or_returns": False,
        "financial_forecasts_refitted": False,
        "all_case_return_gate": False,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    root = write_bundle(
        raw_root,
        VERSION,
        {
            "summary.json": _encode(summary),
            "receipts.json": _encode(receipts),
            "envelopes.json": _encode(envelopes),
        },
    )
    return root, summary


def main() -> None:
    """Require explicit source hash, --collect and an environment key; never use clipboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--bridge-manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/publication_market_gaps"))
    args = parser.parse_args()
    if not args.collect:
        parser.error("Explicit --collect is required; no requests have been sent")
    requests = load_requests(args.bridge_root, args.bridge_manifest_sha256)
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{32}", key):
        parser.error("MASSIVE_API_KEY is absent or malformed; no requests have been sent")
    with httpx.Client(follow_redirects=False) as sender:
        client = MassiveReferenceClient(
            key, sender=sender, max_retries=0, request_interval_seconds=12.1
        )
        root, summary = collect_requests(
            requests, args.raw_root, client, source_manifest_sha256=args.bridge_manifest_sha256
        )
    print(f"PUBLICATION GAP CAPTURE DONE summary={json.dumps(summary)} output={root}")


if __name__ == "__main__":
    main()
