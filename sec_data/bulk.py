"""Streaming, content-addressed preservation of official SEC bulk archives."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import httpx

from sec_data.client import SecConfigurationError, SecRequestError, validate_user_agent

BulkArchiveKind = Literal["submissions", "companyfacts"]

BULK_ARCHIVE_URLS: Final[dict[BulkArchiveKind, str]] = {
    "submissions": ("https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"),
    "companyfacts": ("https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"),
}


@dataclass(frozen=True)
class SecBulkSnapshot:
    """Exact preserved SEC bulk response and its acquisition metadata."""

    kind: BulkArchiveKind
    sha256: str
    source_url: str
    final_url: str
    fetched_at: str
    path: str
    bytes: int
    status_code: int
    content_type: str | None
    content_length_header: int | None
    etag: str | None
    last_modified: str | None


class SecBulkDownloader:
    """Download large SEC ZIP files without loading them into memory."""

    retryable_statuses = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        user_agent: str,
        *,
        root: Path = Path("data/sec/bulk"),
        request_interval_seconds: float = 0.20,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
        chunk_bytes: int = 1024 * 1024,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.user_agent = validate_user_agent(user_agent)
        if request_interval_seconds < 0.20:
            raise SecConfigurationError("Bulk SEC interval must remain at or below five requests/s")
        if timeout_seconds <= 0 or not 0 <= max_retries <= 8 or chunk_bytes <= 0:
            raise SecConfigurationError("Bulk download settings are invalid")
        self.root = root
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.chunk_bytes = chunk_bytes
        self.client = client or httpx.Client(follow_redirects=True)
        self.sleep = sleep
        self.now = now

    def download(self, kind: BulkArchiveKind) -> SecBulkSnapshot:
        """Stream one official archive, hash it, validate ZIP structure, and preserve it."""
        url = BULK_ARCHIVE_URLS[kind]
        last_detail = "request did not start"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.sleep(max(self.request_interval_seconds, min(0.5 * (2 ** (attempt - 1)), 8.0)))
            part_path = self._temporary_path(kind)
            try:
                snapshot = self._download_once(kind, url, part_path)
            except (httpx.HTTPError, OSError, zipfile.BadZipFile, SecRequestError) as exc:
                last_detail = str(exc)
                part_path.unlink(missing_ok=True)
                if attempt == self.max_retries:
                    break
                continue
            return snapshot
        raise SecRequestError(f"SEC bulk download failed for {url}: {last_detail}")

    def _download_once(
        self,
        kind: BulkArchiveKind,
        url: str,
        part_path: Path,
    ) -> SecBulkSnapshot:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/zip",
            "Accept-Encoding": "identity",
        }
        digest = hashlib.sha256()
        byte_count = 0
        fetched_at = self.now()
        with self.client.stream(
            "GET",
            url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        ) as response:
            if not 200 <= response.status_code < 300:
                detail = f"HTTP {response.status_code}"
                if response.status_code in self.retryable_statuses:
                    raise SecRequestError(detail)
                raise SecRequestError(f"Non-retryable {detail}")
            part_path.parent.mkdir(parents=True, exist_ok=True)
            with part_path.open("xb") as stream:
                for chunk in response.iter_bytes(self.chunk_bytes):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            headers_lower = {key.lower(): value for key, value in response.headers.items()}
            final_url = str(response.url)
            status_code = response.status_code
        expected_length = _optional_int(headers_lower.get("content-length"))
        if expected_length is not None and byte_count != expected_length:
            raise SecRequestError(
                f"SEC bulk byte count mismatch: expected {expected_length}, received {byte_count}"
            )
        sha256 = digest.hexdigest()
        target_directory = self.root / "raw" / kind / sha256[:2]
        target_directory.mkdir(parents=True, exist_ok=True)
        target = target_directory / f"{kind}-{sha256}.zip"
        if target.exists():
            if _sha256_file(target) != sha256:
                raise SecRequestError(f"Existing bulk snapshot hash mismatch: {target}")
            part_path.unlink(missing_ok=True)
        else:
            part_path.replace(target)
        _validate_zip_directory(target)
        snapshot = SecBulkSnapshot(
            kind=kind,
            sha256=sha256,
            source_url=url,
            final_url=final_url,
            fetched_at=fetched_at.isoformat(),
            path=str(target.resolve()),
            bytes=byte_count,
            status_code=status_code,
            content_type=headers_lower.get("content-type"),
            content_length_header=expected_length,
            etag=headers_lower.get("etag"),
            last_modified=headers_lower.get("last-modified"),
        )
        self._write_manifest(snapshot)
        return snapshot

    def _temporary_path(self, kind: BulkArchiveKind) -> Path:
        return self.root / "tmp" / f"{kind}-{uuid.uuid4().hex}.part"

    def _write_manifest(self, snapshot: SecBulkSnapshot) -> None:
        directory = self.root / "snapshots" / snapshot.kind / snapshot.sha256
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        payload = json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("sha256") != snapshot.sha256:
                raise SecRequestError(f"Bulk manifest collision: {path}")
            return
        path.write_text(payload, encoding="utf-8")


def load_bulk_snapshot_manifest(path: Path) -> SecBulkSnapshot:
    """Load and verify a previously written SEC bulk snapshot manifest."""
    document = json.loads(path.read_text(encoding="utf-8"))
    snapshot = SecBulkSnapshot(**document)
    source = Path(snapshot.path)
    if not source.is_file() or source.stat().st_size != snapshot.bytes:
        raise SecRequestError(f"Bulk snapshot file is missing or truncated: {source}")
    if _sha256_file(source) != snapshot.sha256:
        raise SecRequestError(f"Bulk snapshot hash mismatch: {source}")
    _validate_zip_directory(source)
    return snapshot


def _validate_zip_directory(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members:
            raise zipfile.BadZipFile("SEC bulk ZIP is empty")
        for member in members:
            normalized = member.filename.replace("\\", "/")
            if normalized.startswith("/") or "../" in f"/{normalized}":
                raise zipfile.BadZipFile(f"Unsafe ZIP member path: {member.filename}")


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
