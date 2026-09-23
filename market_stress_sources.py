"""Public academic-return adapter. Latest-vintage data are NOT point-in-time data."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from scripts.build_sec_disclosure_audit import write_bundle

INDUSTRIES = (
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "HiTec",
    "Telcm",
    "Shops",
    "Hlth",
    "Utils",
    "Other",
)
SOURCE_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SOURCE_URLS = {
    "market": SOURCE_BASE + "F-F_Research_Data_Factors_daily_CSV.zip",
    "industries": SOURCE_BASE + "10_Industry_Portfolios_daily_CSV.zip",
}


def json_bytes(value: Any) -> bytes:
    """Serialize finite JSON consistently for immutable local artifacts."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_bundle(root: Path) -> dict[str, bytes]:
    """Verify every file before parsing, rejecting unsafe manifest paths."""
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    files: dict[str, bytes] = {}
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Unsafe artifact path")
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
        files[name] = content
    identity = hashlib.sha256(
        json.dumps(manifest["artifacts"], sort_keys=True, indent=2, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    if root.name != f"{manifest['version']}-{identity}":
        raise ValueError("Bundle identity does not match its immutable directory")
    return files


def freeze_protocol(config_path: Path, root: Path) -> Path:
    """Reuse an exact frozen protocol, or timestamp it before any source is read."""
    content = config_path.read_bytes()
    config = json.loads(content)
    if config["sources"] != SOURCE_URLS or config["version"] != "market-stress-pilot-v1":
        raise ValueError("Unexpected protocol or source URLs")
    for old in sorted(root.glob("market-stress-protocol-*")):
        files = verify_bundle(old)
        if files["protocol.json"] == content:
            return old
    return write_bundle(
        root,
        "market-stress-protocol",
        {
            "protocol.json": content,
            "freeze_record.json": json_bytes(
                {
                    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_sha256": hashlib.sha256(content).hexdigest(),
                    "numeric_test_outcomes_read_at_freeze": False,
                    "known_historical_crisis_not_untouched": True,
                }
            ),
        },
    )


def collect_sources(root: Path, protocol: Path) -> Path:
    """Download two public ZIPs, persisting exact bodies before opening either ZIP."""
    frozen = verify_bundle(protocol)
    config = json.loads(frozen["protocol.json"])
    if config["sources"] != SOURCE_URLS:
        raise ValueError("Only the two public source URLs are allowed")
    files = {
        "protocol.json": frozen["protocol.json"],
        "freeze_record.json": frozen["freeze_record.json"],
    }
    receipts: dict[str, Any] = {}
    with httpx.Client(timeout=90, follow_redirects=False) as client:
        for name, url in SOURCE_URLS.items():
            response = client.get(url)
            response.raise_for_status()
            body = response.content
            if len(body) > 20_000_000:
                raise ValueError("Unexpectedly large public source")
            receipt = {
                "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "content_type": response.headers.get("content-type"),
                "last_modified": response.headers.get("last-modified"),
                "vintage": "latest_reconstructed_not_point_in_time",
            }
            # Preserve a successful first response even if the second request fails.
            write_bundle(
                root / "responses",
                name,
                {"response.zip": body, "receipt.json": json_bytes(receipt)},
            )
            files[f"{name}.zip"] = body
            receipts[name] = receipt
    files["receipts.json"] = json_bytes(receipts)
    return write_bundle(root, "market-stress-sources", files)


def parse_daily_zip(body: bytes, kind: str, start: str, end: str) -> pd.DataFrame:
    """Read the first daily return table only; preserve missing codes, convert percent."""
    expected = ["Mkt-RF", "SMB", "HML", "RF"] if kind == "market" else list(INDUSTRIES)
    if kind not in SOURCE_URLS:
        raise ValueError("Unknown return source")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise ValueError("Expected a single CSV inside the source ZIP")
        if members[0].file_size > 30_000_000:
            raise ValueError("Unexpectedly large uncompressed source")
        lines = archive.read(members[0]).decode("utf-8-sig").splitlines()
    header = next(
        (i for i, line in enumerate(lines) if [x.strip() for x in line.split(",")][1:] == expected),
        None,
    )
    if header is None:
        raise ValueError("Daily return table header missing")
    if kind == "industries" and not any(
        "Average Value Weighted Returns -- Daily" in x for x in lines[:header]
    ):
        raise ValueError("Cannot establish value-weighted daily industry table")
    records: list[list[str]] = []
    lower, upper = start.replace("-", ""), end.replace("-", "")
    for fields in csv.reader(lines[header + 1 :]):
        fields = [x.strip() for x in fields]
        if not fields or not re.fullmatch(r"\d{8}", fields[0]):
            if records:
                break  # Never mix value/equal weighted or monthly/annual tables.
            continue
        if len(fields) != len(expected) + 1:
            raise ValueError("Malformed daily return row")
        if lower <= fields[0] <= upper:
            records.append(fields)
    if not records:
        raise ValueError("No daily rows in frozen input period")
    frame = pd.DataFrame(records, columns=["session", *expected])
    frame["session"] = pd.to_datetime(frame.session, format="%Y%m%d")
    if frame.session.duplicated().any() or not frame.session.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered source sessions")
    frame = frame.set_index("session").astype(float).replace([-99.99, -999], np.nan) / 100.0
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Nonfinite source returns")
    if kind == "market":
        frame = (frame["Mkt-RF"] + frame["RF"]).to_frame("market_return")
    else:
        frame.columns = pd.Index([f"industry_{name}" for name in INDUSTRIES])
    if (frame < -1).any().any():
        raise ValueError("Simple portfolio returns cannot be below -100%")
    return frame


def load_sources(root: Path, protocol: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join on the UNION of dates; never fill or drop an interior missing session."""
    files, plan = verify_bundle(root), verify_bundle(protocol)
    if files["protocol.json"] != plan["protocol.json"]:
        raise ValueError("Source snapshot was collected for a different frozen protocol")
    if files["freeze_record.json"] != plan["freeze_record.json"]:
        raise ValueError("Source snapshot does not match the original freeze record")
    config = json.loads(plan["protocol.json"])
    receipts = json.loads(files["receipts.json"])
    frozen_at = pd.Timestamp(json.loads(plan["freeze_record.json"])["frozen_at_utc"])
    for kind in SOURCE_URLS:
        if receipts[kind]["url"] != SOURCE_URLS[kind]:
            raise ValueError("Unexpected source receipt URL")
        if pd.Timestamp(receipts[kind]["retrieved_at_utc"]) < frozen_at:
            raise ValueError("Source acquisition precedes the frozen protocol")
        if hashlib.sha256(files[f"{kind}.zip"]).hexdigest() != receipts[kind]["response_sha256"]:
            raise ValueError("Response receipt hash mismatch")
    frame = parse_return_panel(files, config["input_start"], config["input_end"])
    evidence = {
        "source_receipts": receipts,
        "joined_rows": len(frame),
        "first_session": str(frame.index.min().date()),
        "last_session": str(frame.index.max().date()),
        "incomplete_source_rows": int(frame.isna().any(axis=1).sum()),
        "independent_calendar_certified": False,
        "historical_release_vintages_available": False,
        "live_availability_certified": False,
    }
    return frame, evidence


def parse_return_panel(files: dict[str, bytes], start: str, end: str) -> pd.DataFrame:
    """Parse a previously verified pair without filling or compressing source dates.

    Verification of acquisition/freeze lineage belongs to the caller. This also
    supports explicit reuse of an older immutable source snapshot for a new plan.
    """
    frames = [parse_daily_zip(files[f"{kind}.zip"], kind, start, end) for kind in SOURCE_URLS]
    frame = frames[0].join(frames[1], how="outer").sort_index()
    if frame.index.to_series().diff().dt.days.max() > 7:
        raise ValueError("Suspicious gap in joint source calendar")
    if frame.index.min() > pd.Timestamp(start) + pd.Timedelta(
        days=7
    ) or frame.index.max() < pd.Timestamp(end) - pd.Timedelta(days=3):
        raise ValueError("Sources do not cover the frozen input interval")
    return frame
