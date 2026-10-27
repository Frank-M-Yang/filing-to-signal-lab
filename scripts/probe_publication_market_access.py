"""Frozen Basic access controls, not a market-data repair or a return experiment."""

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
from urllib.parse import urlencode

import httpx
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.massive_prices import massive_daily_bars_url, parse_massive_daily_bars
from sec_data.massive_reference import MassiveRequestError
from sec_data.next_publication import DISARM_FLAGS

VERSION = "publication-market-access-probe-v1"
CONFIG = Path("docs/publication_market_access_probe_v1.json")


def build_queries() -> list[dict[str, Any]]:
    """Fix six range controls and six one-filter reference controls, before capture.

    The long range reproduces the original NOBH request exactly. Other dates
    are the first/last cached trading observations, not chosen using prices.
    SPY is only an access control, not a substitute research observation.
    """
    queries = []
    for label, start, end in (
        ("original", "2024-09-16", "2025-10-16"),
        ("old", "2024-09-17", "2024-09-27"),
        ("inside", "2025-09-17", "2025-10-08"),
    ):
        for ticker in ("NOBH", "SPY"):
            queries.append(
                {
                    "id": f"{ticker}_{label}",
                    "kind": "bars",
                    "ticker": ticker,
                    "start": start,
                    "end": end,
                    "url": massive_daily_bars_url(ticker, start, end),
                }
            )
    for label, day in (("old", "2024-09-17"), ("inside", "2025-10-08")):
        for route, filters in (
            ("cik", {"cik": "0000072205"}),
            ("ticker", {"ticker": "NOBH"}),
            ("otc", {"ticker": "NOBH", "market": "otc"}),
        ):
            params = {
                **filters,
                "date": day,
                "active": "true",
                "order": "asc",
                "limit": 1000,
                "sort": "ticker",
            }
            queries.append(
                {
                    "id": f"NOBH_{label}_{route}",
                    "kind": "reference",
                    "ticker": "NOBH",
                    "date": day,
                    "route": route,
                    "url": "https://api.massive.com/v3/reference/tickers?" + urlencode(params),
                }
            )
    return queries


def validate_contract(c: dict[str, Any]) -> None:
    """Keep the user-reported plan distinct from provider-verified permissions."""
    expected = {
        "version": VERSION,
        "user_reported_plan": "Basic",
        "actual_key_plan_verified": False,
        "maximum_requests": 12,
        "request_interval_seconds": 12.1,
        "retries": 0,
        **dict.fromkeys(
            (
                "follow_redirects",
                "follow_pagination",
                "automatic_splicing",
                "returns_allowed",
                "forecast_refit_allowed",
                "sample_replacement_allowed",
                "missing_price_imputation_allowed",
                *DISARM_FLAGS,
            ),
            False,
        ),
    }
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in expected.items()):
        raise ValueError("Changed or weakened access-probe contract")


def freeze_plan(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Verify immutable old/captured evidence and freeze controls without reading prices."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)
    prior_config = _json(Path(c["audit_plan_root"]), c["audit_plan_manifest_sha256"], "plan.json")[
        "config"
    ]
    audit = _json(Path(c["audit_root"]), c["audit_manifest_sha256"], "summary.json")
    if audit["plan_manifest_sha256"] != c["audit_plan_manifest_sha256"] or any(
        audit[k] is not False for k in DISARM_FLAGS
    ):
        raise ValueError("Changed prior gate or capture-audit plan")
    (witness,) = prior_config["cached_price_witnesses"]
    for field in ("receipt", "raw"):
        verify_file(Path(witness[field + "_path"]), witness[field + "_sha256"])
    receipt = json.loads(Path(witness["receipt_path"]).read_bytes())
    raw = json.loads(Path(witness["raw_path"]).read_bytes())
    queries = build_queries()
    if (
        receipt["requested_url"] != queries[0]["url"]
        or receipt["sha256"] != witness["raw_sha256"]
        or receipt["status_code"] != 200
        or raw.get("ticker") != "NOBH"
        or raw.get("status") != "OK"
        or raw.get("next_url")
    ):
        raise ValueError("Old source is not the exact complete NOBH range witness")
    dates = sorted(
        {
            pd.Timestamp(r["t"], unit="ms", tz="UTC")
            .tz_convert("America/New_York")
            .date()
            .isoformat()
            for r in raw["results"]
        }
    )
    if not {"2024-09-17", "2024-09-27", "2025-09-17", "2025-10-08"}.issubset(dates):
        raise ValueError("Control bounds are not original observed session dates")
    plan = {
        "config": c,
        "queries": queries,
        "cached_dates": dates,
        "cached_raw_sha256": witness["raw_sha256"],
        "old_query_replayed_exactly": True,
        "expected_cik": "0000072205",
        "expected_share_class_figi": "BBG001S5TP08",
        "reference_market_default": "all_markets_per_official_docs_not_assumed_stocks_only",
        "controls_do_not_certify_a_security_interval": True,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/probe_publication_market_access.py",
                "sec_data/massive_prices.py",
                "scripts/build_sec_disclosure_audit.py",
                "scripts/build_sec_clock_layer.py",
            )
        },
    }
    return write_bundle(output_root, VERSION + "-plan", {"plan.json": _encode(plan)}), plan


def inspect_response(query: dict[str, Any], body: bytes, sha256: str, code: int) -> dict[str, Any]:
    """Classify preserved HTTP errors and dates/IDs only; an empty result is ambiguous."""
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("Response hash changed")
    result: dict[str, Any] = {
        "http_status": code,
        "provider_status": None,
        "complete": False,
        "permission_denied": code == 403,
        "session_dates": [],
        "identities": [],
        "empty_not_no_trade": True,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {**result, "parser_error": "malformed_json"}
    if not isinstance(data, dict):
        return {**result, "parser_error": "invalid_envelope"}
    # Do not print or copy free-form error messages or arbitrary response text.
    status = data.get("status")
    result["provider_status"] = (
        status
        if isinstance(status, str) and status in {"OK", "ERROR", "NOT_AUTHORIZED", "DELAYED"}
        else "unrecognized"
    )
    result["permission_denied"] = code == 403 or status == "NOT_AUTHORIZED"
    if code != 200 or status != "OK" or data.get("next_url"):
        return result
    if query["kind"] == "bars":
        try:
            rows, bad, _ = parse_massive_daily_bars(
                body,
                expected_ticker=query["ticker"],
                price_target_id="access-control-only",
                window_start=query["start"],
                window_end=query["end"],
                source_sha256=sha256,
            )
        except MassiveRequestError as exc:
            return {**result, "parser_error": exc.reason}
        result["complete"] = not bad
        result["rejected_rows"] = len(bad)
        result["session_dates"] = sorted({r["session_date"] for r in rows})
        if len(rows) != len(result["session_dates"]):
            result["complete"] = False
    else:
        rows = data.get("results", [])
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            return {**result, "parser_error": "invalid_reference_rows"}
        fields = (
            "ticker",
            "cik",
            "share_class_figi",
            "composite_figi",
            "market",
            "primary_exchange",
            "type",
            "active",
        )
        result["identities"] = [{k: r.get(k) for k in fields} for r in rows]
        result["complete"] = True
    return result


def collect_probe(
    plan_root: Path, plan_hash: str, raw_root: Path, key: str, sender: httpx.Client
) -> Path:
    """Preserve every HTTP body before parsing; no retries, redirects or pagination.

    A 403 is a planned comparison outcome, so later fixed controls still run.
    Authentication, rate limits, redirects, network errors and reflected secrets
    stop the batch. No request is invented in response to an outcome.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{32}", key):
        raise ValueError("Missing or malformed Massive key")
    plan = _json(plan_root, plan_hash, "plan.json")
    validate_contract(plan["config"])
    if plan["queries"] != build_queries() or sender.follow_redirects:
        raise ValueError("Changed queries or unsafe redirect setting")
    for path, sha in plan["implementation_sha256"].items():
        verify_file(Path(path), sha)
    receipts, last_start, stop = [], None, None
    files: dict[str, bytes] = {}
    for q in plan["queries"]:
        receipt: dict[str, Any] = {
            "query_id": q["id"],
            "requested_url": q["url"],
            "status_code": None,
        }
        if stop:
            receipt["skipped"] = stop
        else:
            if last_start is not None:
                time.sleep(max(0, 12.1 - (time.monotonic() - last_start)))
            last_start = time.monotonic()
            receipt["fetched_at"] = datetime.now(timezone.utc).isoformat()
            try:
                response = sender.get(
                    q["url"],
                    headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                )
            except httpx.RequestError:
                stop = receipt["error"] = "network_error"
            else:
                body, code = response.content, response.status_code
                receipt.update(
                    status_code=code, sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)
                )
                if key.encode() in body:
                    stop = receipt["error"] = "credential_reflected_body_withheld"
                else:
                    name = q["id"] + ".json"
                    receipt["file"] = name
                    write_bundle(
                        raw_root / "responses",
                        VERSION + "-response",
                        {name: body, "receipt.json": _encode(receipt)},
                    )
                    files[name] = body
                    if code in {401, 429} or 300 <= code < 400 or code >= 500:
                        stop = f"http_{code}"
        receipts.append(receipt)
        print(
            f"ACCESS CONTROL id={q['id']} http={receipt['status_code']} "
            f"skipped={receipt.get('skipped')}",
            flush=True,
        )
    files["receipts.json"] = _encode(
        {"plan_root": plan_root.as_posix(), "plan_manifest_sha256": plan_hash, "receipts": receipts}
    )
    return write_bundle(raw_root, VERSION + "-capture", files)


def audit_probe(
    capture_root: Path, capture_hash: str, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Replay dates and identifiers offline; never splice or promote the six-case sample."""
    capture = _json(capture_root, capture_hash, "receipts.json")
    plan = _json(Path(capture["plan_root"]), capture["plan_manifest_sha256"], "plan.json")
    validate_contract(plan["config"])
    if plan["queries"] != build_queries() or [r["query_id"] for r in capture["receipts"]] != [
        q["id"] for q in plan["queries"]
    ]:
        raise ValueError("Capture does not match all frozen controls")
    audits = {}
    for q, r in zip(plan["queries"], capture["receipts"], strict=True):
        if r["requested_url"] != q["url"]:
            raise ValueError("Request URL changed")
        if "file" in r:
            if r["file"] != q["id"] + ".json":
                raise ValueError("Response file rebound")
            body = _artifact(capture_root, capture_hash, r["file"]).read_bytes()
            if len(body) != r["bytes"]:
                raise ValueError("Response length changed")
            audits[q["id"]] = inspect_response(q, body, r["sha256"], r["status_code"])
        else:
            audits[q["id"]] = {
                "complete": False,
                "untested": True,
                "reason": r.get("skipped") or r.get("error"),
            }
    replay = audits["NOBH_original"]
    old = set(plan["cached_dates"])
    summary = {
        "version": VERSION,
        "plan_manifest_sha256": capture["plan_manifest_sha256"],
        "capture_manifest_sha256": capture_hash,
        "user_reported_plan": "Basic",
        "actual_key_plan_verified": False,
        "requests_made": sum("fetched_at" in r for r in capture["receipts"]),
        "controls_complete": sum(r["complete"] for r in audits.values()),
        "exact_old_query_replay_complete": replay["complete"],
        "exact_replay_cached_dates_not_returned": sorted(old - set(replay.get("session_dates", [])))
        if replay["complete"]
        else None,
        "expiry_cause_confirmed": False,
        "history_boundary_certified": False,
        "old_snapshots_replaced": False,
        "sample_replaced": False,
        "returns_computed": False,
        "forecasts_refitted": False,
        "reference_rows_are_diagnostic_not_security_authorization": True,
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    return write_bundle(
        output_root,
        VERSION + "-audit",
        {"summary.json": _encode(summary), "query_audits.json": _encode(audits)},
    ), summary


def main() -> None:
    """Freeze offline by default; require --collect and process-local credentials online."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest-sha256")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/publication_market_access"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publication_market_access")
    )
    args = parser.parse_args()
    if (args.collect and args.capture_root) or (args.capture_root is None) != (
        args.capture_manifest_sha256 is None
    ):
        parser.error("Choose explicit collection or a complete offline capture binding")
    if args.capture_root is None:
        root, _ = freeze_plan(args.config, args.raw_root / "plans")
        sha = digest_file(root / "output_manifest.json")
        print(f"ACCESS PLAN output={root} manifest={sha}")
        if not args.collect:
            return
        with httpx.Client(timeout=25, follow_redirects=False) as sender:
            args.capture_root = collect_probe(
                root, sha, args.raw_root, os.environ.get("MASSIVE_API_KEY", "").strip(), sender
            )
        args.capture_manifest_sha256 = digest_file(args.capture_root / "output_manifest.json")
    output, summary = audit_probe(args.capture_root, args.capture_manifest_sha256, args.output_root)
    print(f"ACCESS AUDIT DONE output={output} summary={json.dumps(summary)}")


if __name__ == "__main__":
    main()
