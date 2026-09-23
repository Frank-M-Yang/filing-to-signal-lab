"""Bounded public HTML collection, immutable bytes/receipts, offline by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file


def collect_sources(config_path: Path, raw_root: Path) -> Path:
    """Fetch only frozen allowlisted public releases; no credentials/SEC endpoints."""
    config = json.loads(config_path.read_bytes())
    if config["version"] != "release-clock-evidence-v1" or len(config["sources"]) > 3:
        raise ValueError("Unsupported collection contract")
    for item in config["sources"]:
        url = urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.hostname
            not in {
                "ir.usana.com",
                "www.globenewswire.com",
                "www.businesswire.com",
                "malibuboatsinc.com",
                "ufpt.com",
                "www.nasdaq.com",
            }
            or url.username
            or url.password
            or url.port not in (None, 443)
            or url.query
            or url.fragment
        ):
            raise ValueError("Source outside explicit collection allowlist")
        if not re.fullmatch(r"[A-Z0-9.-]{1,12}", item["ticker"]):
            raise ValueError("Invalid source ticker for immutable snapshot filename")
    files: dict[str, bytes] = {}
    receipts = []
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        for index, item in enumerate(config["sources"]):
            if index:
                time.sleep(0.5)
            try:
                response = client.get(
                    item["url"],
                    headers={
                        "User-Agent": "Filing-to-Signal-Lab public-disclosure-research/1.0",
                        "Accept": "text/html",
                    },
                )
            except httpx.RequestError as exc:
                receipts.append(
                    {
                        **item,
                        "status_code": None,
                        "error": type(exc).__name__,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"RELEASE COLLECT {item['ticker']} error={type(exc).__name__}", flush=True)
                continue
            # Preserve failed/redirect responses too, but never parse them as releases.
            content = response.content
            digest = hashlib.sha256(content).hexdigest()
            filename = f"{item['ticker']}-{digest}.html"
            files[filename] = content
            receipts.append(
                {
                    **item,
                    "file": filename,
                    "sha256": digest,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "bytes": len(content),
                    "location": response.headers.get("location"),
                    "response_url": str(response.url),
                    "historical_capture": False,
                }
            )
            # Persist each response before the next request can fail.
            write_bundle(
                raw_root / "responses",
                "release-response-v1",
                {
                    filename: content,
                    "receipt.json": json.dumps(receipts[-1], indent=2, sort_keys=True).encode(),
                },
            )
            print(
                f"RELEASE COLLECT {item['ticker']} status={response.status_code} "
                f"bytes={len(content)}",
                flush=True,
            )
    metadata: dict[str, Any] = {"config_sha256": digest_file(config_path), "receipts": receipts}
    files["receipts.json"] = json.dumps(metadata, indent=2, sort_keys=True).encode()
    return write_bundle(raw_root, "release-clock-snapshots-v1", files)


def main() -> None:
    """Require explicit collection; never refresh pinned evidence implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("docs/release_clock_sources_v1.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/release_clock"))
    args = parser.parse_args()
    if not args.collect:
        parser.error("Use --collect explicitly to fetch these three public pages")
    print(f"RELEASE SNAPSHOTS output={collect_sources(args.config, args.raw_root)}")


if __name__ == "__main__":
    main()
