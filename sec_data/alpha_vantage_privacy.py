"""Prevent reflected API credentials from becoming durable response evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

_KEY_ECHO = re.compile(rb"(api[ _-]?key\s+(?:as|is)\s+)[A-Za-z0-9_-]{8,128}", re.IGNORECASE)


@dataclass(frozen=True)
class SafeAlphaVantageBody:
    """Stored bytes and wire fingerprint; redacted bytes are never called exact."""

    content: bytes
    wire_sha256: str
    wire_bytes: int
    credentials_redacted: bool


def sanitize_alpha_vantage_body(content: bytes, api_key: str | None = None) -> SafeAlphaVantageBody:
    """Remove echoed credentials before persistence, retaining only wire fingerprints.

    Error bodies can echo the account key even when the URL is sanitized. If an
    apparently successful response needs redaction, withhold its whole payload
    instead of promoting altered financial data to research evidence.
    """
    safe = content
    if api_key:
        for token in (
            api_key.encode(),
            json.dumps(api_key)[1:-1].encode(),
            quote(api_key, safe="").encode(),
        ):
            safe = safe.replace(token, b"[REDACTED]")
    safe = _KEY_ECHO.sub(rb"\1[REDACTED]", safe)
    redacted = safe != content
    if redacted:
        try:
            document = json.loads(safe)
        except (ValueError, UnicodeDecodeError):
            document = None
        if not isinstance(document, dict) or "estimates" in document:
            safe = b'{"Error Message":"Response withheld because it echoed a credential"}'
    return SafeAlphaVantageBody(
        content=safe,
        wire_sha256=hashlib.sha256(content).hexdigest(),
        wire_bytes=len(content),
        credentials_redacted=redacted,
    )


def redact_existing_alpha_vantage_error_snapshot(
    manifest_path: Path, *, raw_root: Path
) -> dict[str, Any]:
    """Redact one verified local error snapshot, explicitly retiring its unsafe bytes.

    This privacy repair is intentionally narrow and irreversible for the original
    error body. Keep the original safe manifest, wire hash, byte count and a dated
    redaction record. Never alter a successful financial snapshot or capture time.
    No directory deletion, recursive rewrite or network request is performed.
    """
    manifest_path = manifest_path.resolve()
    root = raw_root.resolve()
    if not manifest_path.is_relative_to(root) or manifest_path.name != "manifest.json":
        raise ValueError("Privacy repair manifest is outside the requested raw root")
    manifest_bytes = manifest_path.read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_bytes)
    if any(
        "apikey=" in str(manifest.get(field, "")).lower()
        for field in ("requested_url", "response_url")
    ):
        raise ValueError("Privacy repair requires already sanitized transport metadata")
    original_path = manifest_path.parent / "response.bin"
    record_path = manifest_path.parent / "privacy_redaction.json"
    if manifest.get("credentials_redacted") is True:
        record: dict[str, Any] = json.loads(record_path.read_bytes())
        if original_path.exists():
            if hashlib.sha256(original_path.read_bytes()).hexdigest() != record["wire_sha256"]:
                raise ValueError("Unexpected remaining error body")
            original_path.unlink()
        return record
    if Path(manifest["path"]).resolve() != original_path or not original_path.is_relative_to(root):
        raise ValueError("Privacy repair response is not the exact sibling response.bin")
    original = original_path.read_bytes()
    if (
        hashlib.sha256(original).hexdigest() != manifest["sha256"]
        or len(original) != manifest["bytes"]
    ):
        raise ValueError("Privacy repair refuses changed raw bytes")
    document = json.loads(original)
    if (
        not isinstance(document, dict)
        or "estimates" in document
        or not any(key in document for key in ("Information", "Note", "Error Message"))
    ):
        raise ValueError("Privacy repair only accepts a provider error body")
    safe = sanitize_alpha_vantage_body(original)
    if not safe.credentials_redacted:
        raise ValueError("No recognized credential echo to redact")
    redacted_path = manifest_path.parent / "response.redacted.bin"
    backup = manifest_path.parent / "manifest.before-credential-redaction.json"
    _write_or_verify(backup, manifest_bytes)
    _write_or_verify(redacted_path, safe.content)
    record = {
        "reason": "provider_error_echoed_credential",
        "redacted_at": datetime.now(timezone.utc).isoformat(),
        "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "wire_sha256": safe.wire_sha256,
        "wire_bytes": safe.wire_bytes,
        "stored_sha256": hashlib.sha256(safe.content).hexdigest(),
        "stored_bytes": len(safe.content),
        "original_wire_body_retained": False,
        "credentials_redacted": True,
    }
    if record_path.exists():
        existing = json.loads(record_path.read_bytes())
        if any(existing.get(key) != value for key, value in record.items() if key != "redacted_at"):
            raise ValueError("Conflicting privacy repair record")
        record = existing
    else:
        _write_or_verify(record_path, json.dumps(record, indent=2, sort_keys=True).encode())
    manifest.update(
        path=str(redacted_path),
        sha256=record["stored_sha256"],
        bytes=record["stored_bytes"],
        wire_sha256=record["wire_sha256"],
        wire_bytes=record["wire_bytes"],
        credentials_redacted=True,
        original_wire_body_retained=False,
        privacy_redaction_record=record_path.name,
    )
    pending = manifest_path.with_name("manifest.redaction.pending")
    _write_or_verify(pending, json.dumps(manifest, indent=2, sort_keys=True).encode())
    pending.replace(manifest_path)
    # Only this verified, newly created error response is retired. No financial
    # response is deleted, and the replacement is explicitly labelled redacted.
    original_path.unlink()
    return record


def _write_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Privacy repair refuses to overwrite a distinct artifact")
    else:
        with path.open("xb") as stream:
            stream.write(content)
