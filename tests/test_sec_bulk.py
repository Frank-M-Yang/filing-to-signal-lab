"""Tests for streaming, content-addressed SEC bulk preservation."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

from sec_data.bulk import SecBulkDownloader, load_bulk_snapshot_manifest


def test_bulk_downloader_streams_hashes_and_reloads_manifest(tmp_path: Path) -> None:
    payload = _zip_bytes({"CIK0000000001.json": b'{"cik": 1}'})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == "Filing Lab research@example.org"
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            content=payload,
            headers={
                "Content-Type": "application/zip",
                "ETag": '"fixture"',
                "Last-Modified": "Thu, 13 Aug 2026 04:00:00 GMT",
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    downloader = SecBulkDownloader(
        "Filing Lab research@example.org",
        root=tmp_path / "bulk",
        client=client,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
    )

    snapshot = downloader.download("submissions")
    manifest_path = (
        tmp_path / "bulk" / "snapshots" / "submissions" / snapshot.sha256 / "manifest.json"
    )
    reloaded = load_bulk_snapshot_manifest(manifest_path)

    assert Path(snapshot.path).read_bytes() == payload
    assert snapshot.bytes == len(payload)
    assert snapshot.etag == '"fixture"'
    assert reloaded == snapshot
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"] == snapshot.sha256


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
