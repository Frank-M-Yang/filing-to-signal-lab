"""Freeze, collect and audit at most eight pre-event Massive requests; no return access."""

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
from urllib.parse import urlencode, urlparse

import httpx
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_prices import massive_daily_bars_url
from sec_data.massive_security_actions import massive_dividends_url, massive_splits_url
from sec_data.pre_event_primary_evidence import (
    END,
    START,
    audit_cached_timeline,
    audit_primary_response,
)

VERSION = "mbuu-primary-history-probe-v1"


def freeze_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Pin queries and the prior calendar/evidence before any authenticated request."""
    c = json.loads(config_path.read_bytes())
    if (
        c["version"] != VERSION
        or c["maximum_requests"] != 8
        or c["request_interval_seconds"] != 12.1
        or c["retries"] != 0
        or any(
            c[k] is not False
            for k in (
                "follow_redirects",
                "follow_pagination",
                "automatic_splicing",
                "returns_allowed",
                "model_admission",
                "trading_admission",
            )
        )
    ):
        raise ValueError("Unsupported probe contract")
    verify_file(Path(c["repair_plan"]), c["repair_plan_sha256"])
    verify_file(Path(c["ticker_events_raw"]), c["ticker_events_sha256"])
    repair = json.loads(Path(c["repair_plan"]).read_bytes())
    if (repair["first_required_session"], repair["last_required_session"]) != (START, END):
        raise ValueError("Frozen repair window mismatch")
    root, digest = Path(repair["rebuild_root"]), repair["rebuild_manifest_sha256"]
    sessions = pd.read_parquet(
        _artifact(root, digest, "market_history_session_audit.parquet"), columns=["session_date"]
    ).session_date.tolist()
    if (
        len(sessions) != 61
        or sorted(set(sessions)) != sessions
        or (sessions[0], sessions[-1]) != (START, END)
    ):
        raise ValueError("Frozen calendar mismatch")
    queries = [
        {
            "id": f"{t}_bars",
            "kind": "bars",
            "ticker": t,
            "url": massive_daily_bars_url(t, START, END),
        }
        for t in ("MBUU", "SPY")
    ]
    for label, day in (("start", START), ("end", END)):
        params = urlencode(
            {
                "ticker": "MBUU",
                "date": day,
                "active": "true",
                "limit": 1000,
                "sort": "ticker",
                "order": "asc",
            }
        )
        queries.append(
            {
                "id": f"MBUU_reference_{label}",
                "kind": "reference",
                "ticker": "MBUU",
                "date": day,
                "url": f"https://api.massive.com/v3/reference/tickers?{params}",
            }
        )
    for t in ("MBUU", "SPY"):
        for kind, builder in (("splits", massive_splits_url), ("dividends", massive_dividends_url)):
            queries.append(
                {"id": f"{t}_{kind}", "kind": kind, "ticker": t, "url": builder(t, START, END)}
            )
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "required_sessions": sessions,
        "queries": queries,
        "ticker_timeline": audit_cached_timeline(Path(c["ticker_events_raw"]).read_bytes()),
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/probe_mbuu_primary_history.py",
                "sec_data/pre_event_primary_evidence.py",
                "sec_data/massive_prices.py",
                "sec_data/massive_reference.py",
                "sec_data/massive_security_actions.py",
                "scripts/build_sec_disclosure_audit.py",
                "scripts/build_sec_clock_layer.py",
            )
        },
    }
    return write_bundle(
        output_root, "mbuu-primary-history-plan-v1", {"plan.json": _encode(plan)}
    ), plan


def collect_probe(plan_root: Path, plan_hash: str, raw_root: Path, api_key: str) -> Path:
    """Use header-only credentials, preserve errors, never redirect/retry/buy access.

    On a 403, skip the remaining queries of that endpoint family, recording that
    they were NOT tested. On auth/rate-limit/redirect failures stop all requests.
    A response reflecting the credential is hashed but withheld, never persisted.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", api_key):
        raise ValueError("MASSIVE_API_KEY is missing or malformed")
    plan = _json(plan_root, plan_hash, "plan.json")
    queries = plan["queries"]
    if len(queries) != 8 or plan["config"]["maximum_requests"] != 8:
        raise ValueError("Unexpected request budget")
    if any(
        urlparse(q["url"]).netloc != "api.massive.com" or not q["url"].startswith("https://")
        for q in queries
    ):
        raise ValueError("Untrusted credential destination")
    files: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    blocked_families: set[str] = set()
    stop_reason: str | None = None
    last_start: float | None = None
    with httpx.Client(timeout=25, follow_redirects=False) as client:
        for query in queries:
            receipt: dict[str, Any] = {
                "query_id": query["id"],
                "requested_url": query["url"],
                "status_code": None,
            }
            if stop_reason or query["kind"] in blocked_families:
                receipt["skipped"] = stop_reason or "endpoint_family_denied_not_tested"
            else:
                if last_start is not None:
                    time.sleep(max(0, 12.1 - (time.monotonic() - last_start)))
                last_start = time.monotonic()
                receipt["fetched_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    response = client.get(
                        query["url"],
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "application/json",
                            "User-Agent": "Filing-to-Signal-Lab/1.0",
                        },
                    )
                except httpx.RequestError as exc:
                    receipt["error"] = type(exc).__name__
                    stop_reason = "network_failure"
                else:
                    body = response.content
                    receipt.update(
                        status_code=response.status_code,
                        bytes=len(body),
                        sha256=hashlib.sha256(body).hexdigest(),
                    )
                    if api_key.encode() in body:
                        receipt["error"] = "credential_reflected_body_withheld"
                        stop_reason = "privacy_guard"
                    else:
                        name = f"{query['id']}.json"
                        files[name] = body
                        receipt["file"] = name
                        # Persist before parsing, including failed HTTP responses.
                        write_bundle(
                            raw_root / "responses",
                            "primary-response-v1",
                            {name: body, "receipt.json": _encode(receipt)},
                        )
                    code = response.status_code
                    if code == 403:
                        blocked_families.add(query["kind"])
                    elif code in {401, 429} or 300 <= code < 400:
                        stop_reason = f"http_{code}"
            receipts.append(receipt)
            print(
                f"PRIMARY PROBE {query['id']} http={receipt['status_code']} "
                f"skipped={receipt.get('skipped')} error={receipt.get('error')}",
                flush=True,
            )
    files["receipts.json"] = _encode(
        {"plan_root": plan_root.as_posix(), "plan_manifest_sha256": plan_hash, "receipts": receipts}
    )
    return write_bundle(raw_root, "mbuu-primary-history-capture-v1", files)


def audit_probe(
    capture_root: Path, capture_hash: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Replay a hash-pinned capture offline, keeping all downstream gates closed."""
    capture = _json(capture_root, capture_hash, "receipts.json")
    plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    queries = plan["queries"]
    receipts = capture["receipts"]
    if [r["query_id"] for r in receipts] != [q["id"] for q in queries]:
        raise ValueError("Capture does not match the frozen query plan")
    audits = {}
    for query, receipt in zip(queries, receipts, strict=True):
        if receipt["requested_url"] != query["url"]:
            raise ValueError("Capture request differs from planned request")
        body = None
        if "file" in receipt:
            body = _artifact(capture_root, capture_hash, receipt["file"]).read_bytes()
            if (
                hashlib.sha256(body).hexdigest() != receipt["sha256"]
                or len(body) != receipt["bytes"]
            ):
                raise ValueError("Response receipt digest/length mismatch")
        if receipt["status_code"] == 200 and body is not None:
            result = audit_primary_response(body, query, plan["required_sessions"])
        else:
            result = {
                "status": receipt.get("skipped")
                or receipt.get("error")
                or f"http_{receipt['status_code']}",
                "evidence_complete": False,
                "observed_rows": None,
            }
        audits[query["id"]] = result
    ready = {k: v["evidence_complete"] for k, v in audits.items()}
    summary = {
        "version": VERSION,
        "capture_root": capture_root.as_posix(),
        "capture_manifest_sha256": capture_hash,
        "plan_manifest_sha256": capture["plan_manifest_sha256"],
        "requests_made": sum("fetched_at" in r for r in receipts),
        "http_statuses": {r["query_id"]: r["status_code"] for r in receipts},
        "evidence_complete": ready,
        "identity_endpoints_and_experimental_timeline_consistent": bool(
            ready["MBUU_reference_start"]
            and ready["MBUU_reference_end"]
            and plan["ticker_timeline"]["consistent"]
        ),
        "issuer_actions_complete": bool(ready["MBUU_splits"] and ready["MBUU_dividends"]),
        "benchmark_actions_complete": bool(ready["SPY_splits"] and ready["SPY_dividends"]),
        "both_price_histories_complete": bool(ready["MBUU_bars"] and ready["SPY_bars"]),
        "first_public_certified": False,
        "market_features_rebuilt": False,
        "automatic_splicing": False,
        "returns_allowed": False,
        "model_admission": False,
        "trading_admission": False,
        "limitation": (
            "Historical provider reconstruction, not vintage prices or exchange-certified "
            "continuity. Endpoint-family skips are untested, not verified denial. "
            "No change to the sample, clocks, lookbacks, forecasts or returns."
        ),
    }
    files = {"summary.json": _encode(summary), "query_audits.json": _encode(audits)}
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Default to freezing only; collection requires an explicit flag and process key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/mbuu_primary_history_probe_v1.json")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest")
    parser.add_argument("--raw-root", type=Path, default=Path("data/market/mbuu_primary_history"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/mbuu_primary_history")
    )
    args = parser.parse_args()
    if args.collect and args.capture_root:
        parser.error("Choose collection or offline replay, not both")
    if (args.capture_root is None) != (args.capture_manifest is None):
        parser.error("Capture root and manifest hash must be supplied together")
    if args.collect or args.capture_root is None:
        frozen, _ = freeze_plan(args.config, args.raw_root / "plans")
        print(
            f"PRIMARY PLAN output={frozen} manifest={digest_file(frozen / 'output_manifest.json')}"
        )
        if not args.collect:
            return
        args.capture_root = collect_probe(
            frozen,
            digest_file(frozen / "output_manifest.json"),
            args.raw_root,
            os.environ.get("MASSIVE_API_KEY", "").strip(),
        )
        args.capture_manifest = digest_file(args.capture_root / "output_manifest.json")
        print(f"PRIMARY CAPTURE output={args.capture_root} manifest={args.capture_manifest}")
    root, report = audit_probe(args.capture_root, args.capture_manifest, args.output_root)
    print(
        f"PRIMARY AUDIT requests={report['requests_made']} "
        f"identity={report['identity_endpoints_and_experimental_timeline_consistent']} "
        f"issuer_actions={report['issuer_actions_complete']} "
        f"benchmark_actions={report['benchmark_actions_complete']} "
        f"prices={report['both_price_histories_complete']} admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
