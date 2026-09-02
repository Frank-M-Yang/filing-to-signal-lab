"""Preserve and summarize one Eulerpool point-in-time estimates response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

EULERPOOL_PIT_URL = "https://api.eulerpool.com/api/1/equity/pit/estimates/{ticker}"


def main() -> int:
    """Fetch one non-secret URL, preserve exact bytes, then print only its schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="RAMP")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/market/eulerpool_consensus/probes"),
    )
    args = parser.parse_args()
    ticker = str(args.ticker).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-/]{0,31}", ticker):
        parser.error("--ticker is invalid")
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    api_key = os.environ.get("EULERPOOL_API_KEY", "").strip()
    if not api_key:
        parser.error("EULERPOOL_API_KEY is required")
    url = EULERPOOL_PIT_URL.format(ticker=ticker)
    fetched_at = datetime.now(timezone.utc)
    response = httpx.get(
        url,
        params={"limit": args.limit},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "Filing-to-Signal-Lab/1.0",
        },
        follow_redirects=True,
        timeout=30.0,
    )
    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    target = args.output_root / ticker / sha256
    target.mkdir(parents=True, exist_ok=True)
    raw_path = target / "response.bin"
    if raw_path.exists() and hashlib.sha256(raw_path.read_bytes()).hexdigest() != sha256:
        raise RuntimeError(f"Existing probe snapshot hash mismatch: {raw_path}")
    if not raw_path.exists():
        raw_path.write_bytes(content)
    manifest = {
        "provider": "eulerpool",
        "endpoint": url,
        "ticker": ticker,
        "limit": args.limit,
        "status_code": response.status_code,
        "fetched_at": fetched_at.isoformat(),
        "sha256": sha256,
        "bytes": len(content),
        "path": str(raw_path.resolve()),
        "response_headers": {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "date", "etag", "last-modified"}
        },
    }
    manifest_path = target / "manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"Immutable probe manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(payload, encoding="utf-8")
    schema: Any
    try:
        schema = _schema(json.loads(content))
    except (UnicodeDecodeError, json.JSONDecodeError):
        schema = {"response": "non_json"}
    print(
        "EULERPOOL CONSENSUS PROBE "
        f"ticker={ticker} status={response.status_code} bytes={len(content)} "
        f"sha256={sha256} schema={json.dumps(schema, sort_keys=True)} "
        f"output={target.resolve()}",
        flush=True,
    )
    return 0 if 200 <= response.status_code < 300 else 1


def _schema(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): _schema(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item": _schema(value[0], depth=depth + 1) if value else None,
        }
    if value is None:
        return "null"
    return type(value).__name__


if __name__ == "__main__":
    sys.exit(main())
