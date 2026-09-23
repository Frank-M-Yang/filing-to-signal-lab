"""Bounded pre-event source collection and offline overlap audit; no provider splicing."""

from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.pre_event_source_audit import (
    QUOTE_COLUMNS,
    VERSION,
    compare_quote_overlap,
    parse_pre_event_quotes,
)
from sec_data.validation_liquidity_fallback import yahoo_liquidity_chart_url


def _config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c = json.loads(path.read_bytes())
    if (
        c["version"] != VERSION
        or c["tickers"] != ["MBUU", "SPY"]
        or c["first_session"] != "2024-08-06"
        or c["last_session"] != "2024-10-30"
        or c["announcement_at"] != "2024-10-31T11:00:00Z"
        or c["price_absolute_tolerance_usd"] != 0.01
        or c["price_relative_tolerance"] != 0
        or c["volume_absolute_tolerance"] != 0
        or c["minimum_primary_overlap_sessions"] != 45
        or c["maximum_network_requests"] != 2
        or c["collection_requires_explicit_flag"] is not True
        or any(
            c[k] is not False
            for k in (
                "automatic_splicing",
                "current_quote_metadata_used",
                "adjusted_close_used",
                "forward_event_bars_allowed",
                "returns_allowed",
                "model_admission",
                "trading_admission",
                "sample_reselection_allowed",
            )
        )
    ):
        raise ValueError("Unsupported or weakened source-audit contract")
    for key in ("repair_plan", "market_config"):
        verify_file(Path(c[key]), c[f"{key}_sha256"])
    repair = json.loads(Path(c["repair_plan"]).read_bytes())
    market = json.loads(Path(c["market_config"]).read_bytes())
    if (repair["first_required_session"], repair["last_required_session"]) != (
        c["first_session"],
        c["last_session"],
    ):
        raise ValueError("Repair interval differs from frozen request")
    return c, repair, market


def _load(source: dict[str, Any], group: str, name: str) -> Any:
    return _json(Path(source[f"{group}_root"]), source[f"{group}_manifest_sha256"], name)


def _freeze(config_path: Path, root: Path) -> tuple[Path, dict[str, Any]]:
    config, _, _ = _config(config_path)
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "urls": {
            t: yahoo_liquidity_chart_url(t, config["first_session"], config["last_session"])
            for t in config["tickers"]
        },
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/audit_mbuu_source_overlap.py",
                "sec_data/pre_event_source_audit.py",
                "scripts/build_sec_clock_layer.py",
                "scripts/build_sec_disclosure_audit.py",
                "sec_data/validation_liquidity_fallback.py",
            )
        },
        "network_authorizes_only_two_public_pre_event_queries": True,
        "primary_outputs_overwritten": False,
    }
    return write_bundle(root, "pre-event-source-plan-v1", {"plan.json": _encode(plan)}), plan


def collect_quotes(config_path: Path, raw_root: Path) -> Path:
    """Preserve every response before parsing, without redirects, keys, retries or outcomes."""
    frozen, plan = _freeze(config_path, raw_root / "plans")
    files: dict[str, bytes] = {}
    receipts = []
    with httpx.Client(
        timeout=20,
        follow_redirects=False,
        headers={
            "User-Agent": "Filing-to-Signal-Lab/0.4 educational-research",
            "Accept": "application/json",
        },
    ) as client:
        for index, (ticker, url) in enumerate(plan["urls"].items()):
            if index:
                time.sleep(1)
            receipt: dict[str, Any] = {
                "ticker": ticker,
                "requested_url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                response = client.get(url)
            except httpx.RequestError as exc:
                receipt.update(status_code=None, error=type(exc).__name__)
            else:
                name = f"{ticker}.json"
                files[name] = response.content
                receipt.update(
                    file=name,
                    status_code=response.status_code,
                    response_url=str(response.url),
                    bytes=len(response.content),
                    content_type=response.headers.get("content-type", ""),
                )
                one = write_bundle(
                    raw_root / "responses",
                    "quote-response-v1",
                    {
                        name: response.content,
                        "receipt.json": _encode(receipt),
                    },
                )
                receipt["sha256"] = digest_file(one / name)
            receipts.append(receipt)
            print(
                f"QUOTE COLLECT {ticker} status={receipt['status_code']} "
                f"error={receipt.get('error')}",
                flush=True,
            )
    files["receipts.json"] = _encode(
        {
            "config_sha256": plan["config_sha256"],
            "plan_root": frozen.as_posix(),
            "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
            "receipts": receipts,
        }
    )
    return write_bundle(raw_root, "pre-event-quote-capture-v1", files)


def audit_sources(
    config_path: Path,
    output_root: Path,
    capture_root: Path | None = None,
    capture_manifest: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Reproduce cached quotes and compare pre-event prices only; all gates stay closed."""
    if (capture_root is None) != (capture_manifest is None):
        raise ValueError("Capture root and manifest hash must be supplied together")
    config, repair, market = _config(config_path)
    frozen, plan = _freeze(config_path, output_root / "plans")
    source = _load(repair, "rebuild", "source_evidence.json")
    history = pd.read_parquet(
        _artifact(
            Path(repair["rebuild_root"]),
            repair["rebuild_manifest_sha256"],
            "market_history_session_audit.parquet",
        )
    )
    expected = list(history.session_date)
    if (
        len(expected) != 61
        or expected != sorted(set(expected))
        or (expected[0], expected[-1]) != (config["first_session"], config["last_session"])
    ):
        raise ValueError("Required official sessions changed")

    verify_file(Path(config["legacy_completion"]), config["legacy_completion_sha256"])
    verify_file(Path(config["legacy_raw_path"]), config["legacy_raw_sha256"])
    completion = json.loads(Path(config["legacy_completion"]).read_bytes())
    receipt = completion["snapshot"]
    raw = Path(config["legacy_raw_path"]).read_bytes()
    if (
        receipt["sha256"] != config["legacy_raw_sha256"]
        or receipt["status_code"] != 200
        or receipt["bytes"] != len(raw)
        or receipt["ticker"] != "MBUU"
        or (receipt["window_start"], receipt["window_end"]) != ("2024-08-01", "2024-08-27")
        or Path(receipt["path"]).resolve() != Path(config["legacy_raw_path"]).resolve()
        or receipt["requested_url"] != yahoo_liquidity_chart_url("MBUU", "2024-08-01", "2024-08-27")
        or receipt["response_url"] != receipt["requested_url"]
    ):
        raise ValueError("Legacy response does not match frozen pre-event request")
    parsed = parse_pre_event_quotes(
        raw, "MBUU", "2024-08-01", "2024-08-27", config["announcement_at"]
    )
    cached = repair["cache_discovery"]["secondary_candidate"]
    path = _artifact(Path(cached["root"]), cached["manifest_sha256"], cached["artifact"])
    verify_file(path, cached["artifact_sha256"])
    saved = pd.read_parquet(
        path,
        columns=["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"],
        filters=[("fallback_target_id", "==", completion["fallback_target_id"])],
    )
    saved = saved.rename(columns={"close_unadjusted": "quote_close"})
    pd.testing.assert_frame_equal(
        parsed.sort_values("session_date").reset_index(drop=True),
        saved[QUOTE_COLUMNS].sort_values("session_date").reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    missing = set(repair["missing_canonical_sessions"]["MBUU"])
    if not missing.issubset(set(parsed.session_date)):
        raise ValueError("Legacy cache does not cover frozen gaps")
    report: dict[str, Any] = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "legacy_cache_raw_and_parquet_reproduced": True,
        "legacy_cache_pre_event_rows": len(parsed),
        "legacy_missing_dates_covered": len(missing),
        "legacy_target_id_is_not_current_event_identity": True,
        "comparisons": {},
        "fresh_source_status": {},
        "candidate_missing_dates_covered": {},
        "raw_quote_fields_not_unadjusted_certification": True,
        "current_quote_metadata_used": False,
        "adjusted_close_used": False,
        "first_public_certified": False,
        "historical_identity_extended": False,
        "corporate_action_interval_extended": False,
        "automatic_splicing": False,
        "forward_event_bars_used": False,
        "returns_computed": False,
        "model_admission": False,
        "trading_admission": False,
        "market_feature_admission": False,
        "primary_outputs_overwritten": False,
        "remaining_primary_interval_checks": repair["failed_interval_checks"],
    }
    frames = {"legacy_mbuu_quotes": parsed}
    if capture_root is not None and capture_manifest is not None:
        capture = _json(capture_root, capture_manifest, "receipts.json")
        report["capture_root"], report["capture_manifest_sha256"] = (
            capture_root.as_posix(),
            capture_manifest,
        )
        if (
            capture["config_sha256"] != plan["config_sha256"]
            or [r["ticker"] for r in capture["receipts"]] != config["tickers"]
        ):
            raise ValueError("Capture belongs to another plan or target list")
        for row in capture["receipts"]:
            ticker = row["ticker"]
            if row["requested_url"] != plan["urls"][ticker]:
                raise ValueError("Capture query interval changed")
            if row["status_code"] != 200:
                report["fresh_source_status"][ticker] = row.get(
                    "error", f"http_{row['status_code']}"
                )
                continue
            if (
                row["response_url"] != row["requested_url"]
                or "application/json" not in row["content_type"]
                or row["file"] != f"{ticker}.json"
            ):
                raise ValueError("Unexpected quote response identity/type")
            path = _artifact(capture_root, capture_manifest, row["file"])
            verify_file(path, row["sha256"])
            content = path.read_bytes()
            if len(content) != row["bytes"]:
                raise ValueError("Capture byte length changed")
            candidate = parse_pre_event_quotes(
                content,
                ticker,
                config["first_session"],
                config["last_session"],
                config["announcement_at"],
            )
            if list(candidate.session_date) != expected:
                raise ValueError("Candidate does not contain exactly the 61 official sessions")
            report["fresh_source_status"][ticker] = "61_pre_event_quote_rows_not_admitted"
            report["candidate_missing_dates_covered"][ticker] = len(
                set(candidate.session_date) & set(repair["missing_canonical_sessions"][ticker])
            )
            group, filename, key = (
                ("prices", "validation_price_batch_bars.parquet", "price_target_id")
                if ticker == "MBUU"
                else (
                    "benchmark",
                    "benchmark_daily_price_observations.parquet",
                    "benchmark_target_id",
                )
            )
            primary = pd.read_parquet(
                _artifact(
                    Path(market[f"{group}_root"]), market[f"{group}_manifest_sha256"], filename
                ),
                columns=["ticker", "session_date", "close_unadjusted", "volume", "source_sha256"],
                filters=[
                    (key, "==", source["event"][key]),
                    ("session_date", ">=", config["first_session"]),
                    ("session_date", "<=", config["last_session"]),
                ],
            )
            primary = primary.rename(columns={"close_unadjusted": "quote_close"})
            summary, pairs = compare_quote_overlap(
                primary, candidate, minimum_sessions=config["minimum_primary_overlap_sessions"]
            )
            report["comparisons"][ticker] = summary
            frames[f"{ticker}_candidate_quotes"], frames[f"{ticker}_primary_overlap"] = (
                candidate,
                pairs,
            )
            if ticker == "MBUU":
                summary, pairs = compare_quote_overlap(
                    parsed, candidate, price_tolerance=0, minimum_sessions=16
                )
                report["legacy_vs_refreshed_mbuu"] = summary
                frames["MBUU_legacy_overlap"] = pairs
    files = {"summary.json": _encode(report)}
    for name, frame in frames.items():
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        files[f"{name}.parquet"] = buffer.getvalue()
        if name.endswith("overlap"):
            files[f"{name}.json"] = _encode(_records(frame))
    return write_bundle(output_root, VERSION, files), report


def main() -> None:
    """Collect explicitly, or audit locally with a hash-pinned optional capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/mbuu_source_overlap_v1.json"))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/market/mbuu_source_overlap"))
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--capture-manifest")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/mbuu_source_overlap")
    )
    args = parser.parse_args()
    if args.collect:
        root = collect_quotes(args.config, args.raw_root)
        print(f"QUOTE CAPTURE output={root} manifest={digest_file(root / 'output_manifest.json')}")
    else:
        root, report = audit_sources(
            args.config, args.output_root, args.capture_root, args.capture_manifest
        )
        print(
            f"SOURCE AUDIT legacy_rows={report['legacy_cache_pre_event_rows']} "
            f"admitted=False output={root}"
        )


if __name__ == "__main__":
    main()
