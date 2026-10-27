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


def validate_source_config(config: dict[str, Any]) -> None:
    """Keep the original three-page contract or the separate frozen six-issuer batch."""
    next_batch = config.get("version") == "next-publication-sources-v1"
    if config.get("version") not in {
        "release-clock-evidence-v1",
        "next-publication-sources-v1",
    } or len(config["sources"]) > (6 if next_batch else 3):
        raise ValueError("Unsupported collection contract")
    identities = {
        "NOBH": ("0000072205", "0001193125-24-216991"),
        "AOS": ("0000091142", "0000091142-24-000169"),
        "ROCK": ("0000912562", "0000912562-24-000059"),
        "RAMP": ("0000733269", "0000733269-24-000037"),
        "ORA": ("0001296445", "0001437749-24-033671"),
        "CTSO": ("0001175151", "0001104659-24-115296"),
    }
    if next_batch and (
        len(config["sources"]) != 6
        or {s["ticker"] for s in config["sources"]} != set(identities)
        or any(
            (s["cik"], s["release_accession"]) != identities[s["ticker"]] for s in config["sources"]
        )
    ):
        raise ValueError("Six original issuer/release identities must remain frozen")
    hosts = (
        {
            "www.accessnewswire.com",
            "www.prnewswire.com",
            "ir.gibraltar1.com",
            "www.globenewswire.com",
        }
        if next_batch
        else {
            "ir.usana.com",
            "www.globenewswire.com",
            "www.businesswire.com",
            "malibuboatsinc.com",
            "ufpt.com",
            "www.nasdaq.com",
        }
    )
    for item in config["sources"]:
        url = urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.hostname not in hosts
            or url.username
            or url.password
            or url.port not in (None, 443)
            or url.query
            or url.fragment
        ):
            raise ValueError("Source outside explicit collection allowlist")
        if not re.fullmatch(r"[A-Z0-9.-]{1,12}", item["ticker"]):
            raise ValueError("Invalid source ticker for immutable snapshot filename")


def collect_sources(config_path: Path, raw_root: Path, *, transport: str = "httpx") -> Path:
    """Fetch only frozen allowlisted public releases; no credentials/SEC endpoints."""
    config = json.loads(config_path.read_bytes())
    validate_source_config(config)
    if transport not in {"httpx", "httpx-direct"}:
        raise ValueError("Unsupported public HTML transport")
    files: dict[str, bytes] = {}
    receipts = []
    with httpx.Client(
        timeout=15, follow_redirects=False, trust_env=transport != "httpx-direct"
    ) as client:
        for index, item in enumerate(config["sources"]):
            if index:
                time.sleep(0.5)
            try:
                headers = {"Accept": "text/html"}
                if transport == "httpx":
                    headers["User-Agent"] = "Filing-to-Signal-Lab public-disclosure-research/1.0"
                response = client.get(item["url"], headers=headers)
                content = response.content
                response_meta = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "location": response.headers.get("location"),
                    "response_url": str(response.url),
                }
            except httpx.RequestError as exc:
                receipts.append(
                    {
                        **item,
                        "status_code": None,
                        "error": type(exc).__name__,
                        "transport": transport,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"RELEASE COLLECT {item['ticker']} error={type(exc).__name__}", flush=True)
                continue
            # Preserve failed/redirect responses too, but never parse them as releases.
            digest = hashlib.sha256(content).hexdigest()
            filename = f"{item['ticker']}-{digest}.html"
            files[filename] = content
            receipts.append(
                {
                    **item,
                    "file": filename,
                    "sha256": digest,
                    **response_meta,
                    "transport": transport,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "bytes": len(content),
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
                f"RELEASE COLLECT {item['ticker']} status={response_meta['status_code']} "
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
    parser.add_argument("--transport", choices=("httpx", "httpx-direct"), default="httpx")
    args = parser.parse_args()
    if not args.collect:
        parser.error("Use --collect explicitly to fetch the bounded public-page batch")
    root = collect_sources(args.config, args.raw_root, transport=args.transport)
    print(f"RELEASE SNAPSHOTS output={root}")


if __name__ == "__main__":
    main()
