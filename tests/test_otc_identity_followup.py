"""Five bounded metadata requests never promote unknown OTC data into returns."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts.build_sec_clock_layer import _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.collect_otc_identity_followup import (
    audit_capture,
    collect_queries,
    inspect_query,
    validate_plan,
)
from sec_data.disclosure_audit import digest_file
from sec_data.next_publication import DISARM_FLAGS
from sec_data.otc_identity import followup_queries


def plan() -> dict[str, Any]:
    """Use the frozen public identifiers but only synthetic responses and sources."""
    return {
        "queries": followup_queries(
            {"candidate_association_supported": True, "candidate_composite_figi": "BBG000BQ2104"},
            "NOBH",
            "2024-09-11",
            "2024-10-09",
            "2026-08-14",
        ),
        "target": {
            "security_identity_key": "BBG001S5TP08",
            "cik": "0000072205",
            "selected_ticker": "NOBH",
            "selected_exchange_mic": "OTCM",
            "issuer_name": "NOBILITY HOMES INC",
        },
        "candidate_composite_figi": "BBG000BQ2104",
        "window_start": "2024-09-11",
        "window_end": "2024-10-09",
        "listing_cutoff": "2026-08-14",
        "maximum_requests": 5,
        "request_interval_seconds": 12.1,
        "retries": 0,
        "requires_explicit_collect_invocation": True,
        "old_offline_preparation_authorization_not_overwritten": True,
        "follow_redirects": False,
        "follow_pagination": False,
        "returns_allowed": False,
        "forecast_refit_allowed": False,
        "implementation_sha256": {},
        **dict.fromkeys(DISARM_FLAGS, False),
    }


def reference(active: bool = True) -> dict[str, Any]:
    """CIK is absent and the raw venue is deliberately not a MIC."""
    return {
        "status": "OK",
        "results": [
            {
                "ticker": "NOBH",
                "name": "NOBILITY HOMES INC",
                "market": "otc",
                "locale": "us",
                "type": "CS",
                "active": active,
                "share_class_figi": "BBG001S5TP08",
                "composite_figi": "BBG000BQ2104",
                "primary_exchange": "OTC Link",
            }
        ],
    }


def inspect(data: Any, index: int = 0, code: int = 200) -> dict[str, Any]:
    """Inspect synthetic bytes without credentials or network access."""
    p, body = plan(), _encode(data)
    return inspect_query(p, p["queries"][index], body, hashlib.sha256(body).hexdigest(), code)


@pytest.fixture()
def frozen(tmp_path: Path) -> tuple[Path, str]:
    """Bind a synthetic twenty-issuer queue and thirty gaps on sixteen dates."""
    source = write_bundle(
        tmp_path / "source",
        "synthetic",
        {
            "case_dispositions.json": _encode([{"issuer": str(i)} for i in range(20)]),
            "price_gap_classification.json": _encode(
                [{"session_date": f"2024-09-{i % 16 + 11:02d}"} for i in range(30)]
            ),
        },
    )
    p = plan()
    p.update(
        source_review_root=source.as_posix(),
        source_review_manifest_sha256=digest_file(source / "output_manifest.json"),
    )
    root = write_bundle(tmp_path / "plans", "synthetic", {"plan.json": _encode(p)})
    return root, digest_file(root / "output_manifest.json")


@pytest.mark.parametrize(
    "field",
    [
        *DISARM_FLAGS,
        "follow_redirects",
        "follow_pagination",
        "returns_allowed",
        "forecast_refit_allowed",
        "maximum_requests",
        "retries",
        "requires_explicit_collect_invocation",
        "old_offline_preparation_authorization_not_overwritten",
    ],
)
def test_contract_changes_fail_closed(field: str) -> None:
    p = plan()
    validate_plan(p)
    p[field] = not p[field] if type(p[field]) is bool else p[field] + 1
    with pytest.raises(ValueError, match="unsafe"):
        validate_plan(p)


def test_bool_retry_and_unobserved_figi_rejected() -> None:
    p = plan()
    p["retries"] = False
    with pytest.raises(ValueError):
        validate_plan(p)
    p = plan()
    p["candidate_composite_figi"] = None
    with pytest.raises(ValueError):
        validate_plan(p)


@pytest.mark.parametrize("failure", [403, 404, 401, 429, 302, 500, "reflection", "network"])
def test_error_capture_preserves_bytes_without_retry_or_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: tuple[Path, str],
    failure: int | str,
) -> None:
    monkeypatch.setattr("scripts.collect_otc_identity_followup.time.sleep", lambda seconds: None)
    key, calls = "Z" * 32, []  # Synthetic test token, never a real credential.

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert key not in str(request.url)
        assert request.headers["Authorization"] == "Bearer " + key
        if failure == "network":
            raise httpx.ConnectError("not persisted", request=request)
        if failure == "reflection":
            return httpx.Response(200, content=key.encode())
        return httpx.Response(int(failure), content=b'{"status":"NOT_AUTHORIZED"}')

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as sender:
        root = collect_queries(*frozen, tmp_path / "raw", key, sender)
    sha = digest_file(root / "output_manifest.json")
    receipts = _json(root, sha, "receipts.json")["receipts"]
    assert len(receipts) == 5 and len(calls) == (5 if failure in (403, 404) else 1)
    if isinstance(failure, int):
        assert (root / "reference_start.json").read_bytes() == b'{"status":"NOT_AUTHORIZED"}'
        assert len(list((tmp_path / "raw" / "responses").glob("*/receipt.json"))) == len(calls)
    else:
        assert not (root / "reference_start.json").exists()
    assert all(
        key.encode() not in f.read_bytes() for f in (tmp_path / "raw").rglob("*") if f.is_file()
    )
    _, summary = audit_capture(root, sha, tmp_path / "audit")
    assert not summary["timeline_query_complete"] and summary["in_window_ticker_events"] is None
    assert not summary["observed_inactive_candidate_at_cutoff"]
    assert summary["distinct_price_gaps_retained"] == 16
    assert all(summary[k] is False for k in DISARM_FLAGS)


def test_destination_changes_redirects_and_invalid_keys_fail_before_request(
    tmp_path: Path,
    frozen: tuple[Path, str],
) -> None:
    with httpx.Client(follow_redirects=True) as sender, pytest.raises(ValueError, match="Redirect"):
        collect_queries(*frozen, tmp_path / "raw", "Z" * 32, sender)
    p = _json(*frozen, "plan.json")
    p["queries"][0]["url"] = "https://example.org/unsafe"
    root = write_bundle(tmp_path / "other", "synthetic", {"plan.json": _encode(p)})
    with httpx.Client(follow_redirects=False) as sender:
        with pytest.raises(ValueError, match="unsafe"):
            collect_queries(
                root, digest_file(root / "output_manifest.json"), tmp_path / "raw", "Z" * 32, sender
            )
        with pytest.raises(ValueError, match="malformed"):
            collect_queries(*frozen, tmp_path / "raw", "wrong", sender)


def test_missing_cik_and_venue_labels_are_not_filled() -> None:
    r = inspect(reference())
    assert r["query_complete"] and r["identity_candidate_match"]
    row = r["identities"][0]
    assert row["provider_cik"] is None and row["provider_cik_state"] == "missing"
    assert row["raw_primary_exchange"] == "OTC Link" and row["observed_mic"] is None
    assert not row["provider_cik_imputed"] and not row["venue_label_converted_to_mic"]
    assert not r["historical_security_interval_ready"]
    assert inspect(reference(False), index=3)["identity_candidate_match"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cik", "wrong"),
        ("cik", "0000000001"),
        ("cik", 72205),
        ("active", 1),
        ("active", None),
        ("share_class_figi", "BBG000000009"),
        ("composite_figi", None),
        ("type", "PFD"),
        ("primary_exchange", "XNAS"),
    ],
)
def test_conflicts_and_weak_types_are_not_candidates(field: str, value: Any) -> None:
    d = reference()
    d["results"][0][field] = value
    r = inspect(d)
    assert r["query_complete"] and not r["identity_candidate_match"]
    assert r["identities"][0]["reasons"]


def test_duplicate_reference_and_bad_envelope_are_not_candidates() -> None:
    d = reference()
    d["results"] *= 2
    assert not inspect(d)["identity_candidate_match"]
    for d in ([], {"status": "OK", "results": {}}, {"status": "OK", "next_url": "private"}):
        assert not inspect(d)["query_complete"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_figi", "invalid_event", "pagination"])
def test_timeline_needs_explicit_events_and_matching_composite(mutation: str) -> None:
    d: dict[str, Any] = {
        "status": "OK",
        "results": {"composite_figi": "BBG000BQ2104", "events": []},
    }
    if mutation == "missing":
        del d["results"]["events"]
    elif mutation == "wrong_figi":
        d["results"]["composite_figi"] = "BBG000000009"
    elif mutation == "invalid_event":
        d["results"]["events"] = [{}]
    else:
        d["next_url"] = "https://example.org/next"
    assert not inspect(d, index=4)["query_complete"]


def test_outside_window_ticker_change_not_used_and_hash_checked_first() -> None:
    d = {
        "status": "OK",
        "results": {
            "composite_figi": "BBG000BQ2104",
            "events": [
                {
                    "date": "2026-09-08",
                    "type": "ticker_change",
                    "ticker_change": {"ticker": "NOBH"},
                },
            ],
        },
    }
    r = inspect(d, index=4)
    assert r["query_complete"] and r["in_window_events"] == []
    assert r["outside_window_events_not_used"] == 1
    with pytest.raises(ValueError, match="hash changed"):
        inspect_query(plan(), plan()["queries"][0], b"bad", "a" * 64, 200)


@pytest.mark.parametrize("listing_state", ["active", "inactive", "empty", "contradictory"])
def test_complete_capture_preserves_queue_gaps_and_closed_gates(
    tmp_path: Path,
    frozen: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    listing_state: str,
) -> None:
    monkeypatch.setattr("scripts.collect_otc_identity_followup.time.sleep", lambda seconds: None)
    p = plan()
    bodies = [
        reference(),
        reference(),
        {"status": "OK", "results": []},
        {"status": "OK", "results": []},
        {"status": "OK", "results": {"composite_figi": "BBG000BQ2104", "events": []}},
    ]
    if listing_state in ("active", "contradictory"):
        bodies[2] = reference()
    if listing_state in ("inactive", "contradictory"):
        bodies[3] = reference(False)
    responses = {q["url"]: _encode(d) for q, d in zip(p["queries"], bodies, strict=True)}
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=responses[str(r.url)]))
    ) as sender:
        root = collect_queries(*frozen, tmp_path / "raw", "Z" * 32, sender)
    sha = digest_file(root / "output_manifest.json")
    result, summary = audit_capture(root, sha, tmp_path / "audit")
    assert summary["queries_complete"] == 5 and summary["exact_window_endpoint_candidates_match"]
    assert summary["observed_active_candidate_at_cutoff"] == (listing_state == "active")
    assert summary["observed_inactive_candidate_at_cutoff"] == (listing_state == "inactive")
    assert (
        summary["price_gap_rows_retained"] == 30 and summary["distinct_price_gaps_retained"] == 16
    )
    assert (
        summary["frozen_queue_issuers_retained"] == 20
        and summary["frozen_pair_cases_retained"] == 6
    )
    assert (
        not summary["listing_outcome_formally_ready"]
        and not summary["historical_security_interval_ready"]
    )
    assert all(summary[k] is False for k in DISARM_FLAGS)
    repeated, same = audit_capture(root, sha, tmp_path / "audit")
    assert repeated == result and same == summary
    capture = _json(root, sha, "receipts.json")
    capture["receipts"][0]["requested_url"] = "https://example.org/rebound"
    rebound = write_bundle(
        tmp_path / "rebound",
        "synthetic",
        {
            **{q["kind"] + ".json": responses[q["url"]] for q in p["queries"]},
            "receipts.json": _encode(capture),
        },
    )
    with pytest.raises(ValueError, match="URL rebound"):
        audit_capture(rebound, digest_file(rebound / "output_manifest.json"), tmp_path / "audit")
