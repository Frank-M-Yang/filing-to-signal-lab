"""Reflected credentials must not survive storage or contaminate research data."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from sec_data.alpha_vantage_consensus import (
    AlphaVantageConsensusClient,
    alpha_vantage_provider_reason,
    alpha_vantage_response_status,
    collect_alpha_vantage_live_snapshots,
)
from sec_data.alpha_vantage_privacy import (
    redact_existing_alpha_vantage_error_snapshot,
    sanitize_alpha_vantage_body,
)

_TEST_KEY = "test-secret-key-123"


def _echo_body() -> bytes:
    return json.dumps(
        {
            "Information": f"We have detected your API key as {_TEST_KEY} "
            "and our API rate limit is 25 requests per day. Upgrade to premium."
        }
    ).encode()


def test_quota_information_is_not_invalid_key_and_body_has_no_secret() -> None:
    original = _echo_body()
    safe = sanitize_alpha_vantage_body(original, _TEST_KEY)
    assert _TEST_KEY.encode() not in safe.content
    assert safe.credentials_redacted
    assert safe.wire_sha256 == hashlib.sha256(original).hexdigest()
    assert safe.wire_bytes == len(original)
    assert alpha_vantage_provider_reason(safe.content, 200) == "rate_limited"
    # Historical envelope labels are retained, not rewritten in old manifests.
    assert alpha_vantage_response_status(safe.content, 200) == "premium_endpoint_or_invalid_key"


@pytest.mark.parametrize("key", ['abcde"fghij', "key/with+special", "ordinary-key-123"])
def test_known_key_is_removed_even_with_json_escaping(key: str) -> None:
    original = json.dumps({"Error Message": "Credential rejected: " + key}).encode()
    safe = sanitize_alpha_vantage_body(original, key)
    assert safe.credentials_redacted
    assert key not in json.loads(safe.content)["Error Message"]


def test_clean_data_are_verbatim_but_changed_data_are_withheld() -> None:
    original = b'{"symbol":"AAA","estimates":[]}'
    safe = sanitize_alpha_vantage_body(original, _TEST_KEY)
    assert safe.content == original and not safe.credentials_redacted
    contaminated = json.dumps({"symbol": "AAA", "estimates": [], "echo": _TEST_KEY}).encode()
    safe = sanitize_alpha_vantage_body(contaminated, _TEST_KEY)
    assert "estimates" not in json.loads(safe.content)
    assert alpha_vantage_response_status(safe.content, 200) == "provider_error"


def test_client_and_original_collector_never_write_echoed_key(tmp_path: Path) -> None:
    class Sender:
        def get(
            self, url: str, *, params: dict[str, str], headers: dict[str, str], timeout: float
        ) -> httpx.Response:
            del headers, timeout
            return httpx.Response(
                200,
                request=httpx.Request("GET", url, params=params),
                content=_echo_body(),
                headers={"etag": _TEST_KEY},
            )

    client = AlphaVantageConsensusClient(_TEST_KEY, sender=Sender())
    targets = pd.DataFrame(
        [
            {
                "live_consensus_target_id": "one",
                "collection_order": 1,
                "cik": "0000000001",
                "ticker": "AAA",
            }
        ]
    )
    result = collect_alpha_vantage_live_snapshots(
        targets, client, raw_root=tmp_path, capture_date=date.today(), limit=1
    )
    assert result.failed_targets == 1
    assert result.blocking_reason == "rate_limited"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert _TEST_KEY.encode() not in path.read_bytes()
    manifest = json.loads(next(tmp_path.glob("snapshots/*/*/*/manifest.json")).read_bytes())
    assert manifest["credentials_redacted"]
    assert not manifest["original_wire_body_retained"]
    assert manifest["sha256"] != manifest["wire_sha256"]


def _snapshot(root: Path, content: bytes) -> Path:
    root.mkdir(parents=True)
    raw = root / "response.bin"
    raw.write_bytes(content)
    path = root / "manifest.json"
    manifest: dict[str, Any] = {
        "path": str(raw.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "status_code": 200,
        "fetched_at": "2026-09-15T00:00:00Z",
    }
    path.write_text(json.dumps(manifest))
    return path


def test_existing_error_repair_is_explicit_idempotent_and_removes_only_unsafe_body(
    tmp_path: Path,
) -> None:
    path = _snapshot(tmp_path / "error", _echo_body())
    old_manifest = path.read_bytes()
    record = redact_existing_alpha_vantage_error_snapshot(path, raw_root=tmp_path)
    assert record["wire_sha256"] == hashlib.sha256(_echo_body()).hexdigest()
    assert not record["original_wire_body_retained"]
    assert not (path.parent / "response.bin").exists()
    assert (path.parent / "manifest.before-credential-redaction.json").read_bytes() == old_manifest
    assert redact_existing_alpha_vantage_error_snapshot(path, raw_root=tmp_path) == record
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert _TEST_KEY.encode() not in artifact.read_bytes()


def test_repair_rejects_financial_data_tampering_and_outside_root(tmp_path: Path) -> None:
    financial = _snapshot(tmp_path / "data", b'{"estimates":[]}')
    with pytest.raises(ValueError, match="only accepts a provider error"):
        redact_existing_alpha_vantage_error_snapshot(financial, raw_root=tmp_path)
    path = _snapshot(tmp_path / "error", _echo_body())
    with pytest.raises(ValueError, match="outside"):
        redact_existing_alpha_vantage_error_snapshot(path, raw_root=tmp_path / "data")
    (path.parent / "response.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed raw bytes"):
        redact_existing_alpha_vantage_error_snapshot(path, raw_root=tmp_path)
