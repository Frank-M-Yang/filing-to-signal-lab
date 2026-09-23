"""Definition evidence is stricter than metadata hints or collection timestamps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.audit_consensus_provider_definitions import (
    audit_saved_response,
    build_definition_audit,
)


def _saved(tmp_path: Path, **changes: Any) -> Path:
    directory = tmp_path / "snapshots/2026-09-19/AAA/capture"
    directory.mkdir(parents=True)
    payload = {
        "symbol": "AAA",
        "estimates": [
            {
                "date": "2026-10-31",
                "horizon": "fiscal quarter",
                "revenue_estimate_average": 0,
                "eps_estimate_average_7_days_ago": 3.2,
            },
        ],
        **changes,
    }
    content = json.dumps(payload).encode()
    path = directory / "response.bin"
    path.write_bytes(content)
    manifest = {
        "ticker": "AAA",
        "cik": "0000000001",
        "capture_date": "2026-09-19",
        "fetched_at": "2026-09-19T22:00:00Z",
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "path": str(path),
        "status_code": 200,
        "response_headers": {
            "date": "Sat, 19 Sep 2026 22:00:00 GMT",
            "last-modified": "Sat, 19 Sep 2026 21:00:00 GMT",
        },
    }
    target = directory / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def _review(root: Path) -> Path:
    source = root / "public-doc.html"
    source.write_bytes(b"Test documentation evidence only")
    review = {
        "review_version": "alpha-vantage-definition-review-v1",
        "endpoint": "EARNINGS_ESTIMATES",
        "vendor_response_received": False,
        "formal_model_admission": False,
        "unresolved_definitions": ["currency"],
        "sources": [
            {
                "local_path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    path = root / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    return path


def test_http_date_period_date_and_eps_history_are_not_revenue_update_metadata(
    tmp_path: Path,
) -> None:
    result = audit_saved_response(_saved(tmp_path), raw_root=tmp_path)
    assert result["metadata_inventory_performed"]
    assert not any(result["metadata_hint_paths"].values())
    assert not result["http_timestamp_is_estimate_timestamp"]
    assert not result["formal_model_admission"]
    # A zero estimate counts as an observed row in a schema inventory, not an admitted candidate.
    assert result["future_quarterly_rows"] == 1


def test_present_nested_metadata_is_only_a_hint_not_verification(tmp_path: Path) -> None:
    result = audit_saved_response(
        _saved(tmp_path, metadata={"currency": "USD", "last_updated": "2099-01-01"}),
        raw_root=tmp_path,
    )
    assert result["metadata_hint_paths"]["currency"] == ["payload.metadata.currency"]
    assert result["metadata_hint_paths"]["update_time"] == ["payload.metadata.last_updated"]
    assert not result["provider_definition_verified"]


@pytest.mark.parametrize(
    "kind",
    [
        "changed_bytes",
        "changed_size",
        "naive_time",
        "wrong_date",
        "wrong_symbol",
        "escape",
        "relabelled",
    ],
)
def test_invalid_snapshot_fails_closed(tmp_path: Path, kind: str) -> None:
    path = _saved(tmp_path)
    manifest = json.loads(path.read_bytes())
    if kind == "changed_bytes":
        (path.parent / "response.bin").write_bytes(b"tampered")
    elif kind == "changed_size":
        manifest["bytes"] += 1
    elif kind == "naive_time":
        manifest["fetched_at"] = "2026-09-19T22:00:00"
    elif kind == "wrong_date":
        manifest["capture_date"] = "2026-09-18"
    elif kind == "wrong_symbol":
        manifest["ticker"] = "BBB"
    elif kind == "relabelled":
        manifest["status"] = "authentication_failed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_saved_response(
            path, raw_root=tmp_path / "unrelated" if kind == "escape" else tmp_path
        )


def test_redacted_snapshot_never_supplies_definition_evidence(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    manifest = json.loads(path.read_bytes())
    manifest["credentials_redacted"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_saved_response(path, raw_root=tmp_path)
    assert not result["metadata_inventory_performed"]
    assert not result["estimate_row_fields"]


def test_error_body_is_not_copied_to_output(tmp_path: Path) -> None:
    result = audit_saved_response(
        _saved(tmp_path, Information="request limit reached"), raw_root=tmp_path
    )
    assert not result["metadata_inventory_performed"]
    assert "request limit" not in json.dumps(result)


def test_document_lineage_and_deterministic_report(tmp_path: Path) -> None:
    _saved(tmp_path)
    review = _review(tmp_path)
    first = build_definition_audit([tmp_path], review_path=review)
    assert first == build_definition_audit([tmp_path], review_path=review)
    assert first["saved_response_count"] == first["issuer_count"] == 1
    assert not first["changes_frozen_policy"]
    assert not first["absence_proves_provider_lacks_information"]
    with pytest.raises(ValueError, match="double count"):
        build_definition_audit([tmp_path, tmp_path], review_path=review)
    (tmp_path / "public-doc.html").write_bytes(b"Changed documentation")
    with pytest.raises(ValueError, match="documentation"):
        build_definition_audit([tmp_path], review_path=review)


def test_empty_capture_set_is_not_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No saved"):
        build_definition_audit([tmp_path], review_path=_review(tmp_path))
