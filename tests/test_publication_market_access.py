"""Fixed access controls cannot repair prices or authorize trading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
import pytest

from scripts.build_sec_clock_layer import _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.probe_publication_market_access import (
    CONFIG,
    audit_probe,
    build_queries,
    collect_probe,
    inspect_response,
    validate_contract,
)
from sec_data.disclosure_audit import digest_file
from sec_data.next_publication import DISARM_FLAGS


def response(data: Any, code: int = 200, index: int = 0) -> dict[str, Any]:
    """Inspect only synthetic JSON, never a provider request."""
    body = _encode(data)
    return inspect_response(build_queries()[index], body, hashlib.sha256(body).hexdigest(), code)


def test_exact_fixed_controls_and_single_filter_pairs() -> None:
    queries = build_queries()
    assert len(queries) == 12 and len({q["id"] for q in queries}) == 12
    assert queries[0]["url"] == (
        "https://api.massive.com/v2/aggs/ticker/NOBH/range/1/day/2024-09-16/2025-10-16"
        "?adjusted=false&sort=asc&limit=50000"
    )
    for a, b in zip(queries[:6:2], queries[1:6:2], strict=True):
        assert a["start"] == b["start"] and a["end"] == b["end"]
        assert (a["ticker"], b["ticker"]) == ("NOBH", "SPY")
    for start in (6, 9):
        cik, ticker, otc = (parse_qs(urlparse(q["url"]).query) for q in queries[start : start + 3])
        assert "cik" in cik and "ticker" not in cik and "market" not in cik
        assert "ticker" in ticker and "cik" not in ticker and "market" not in ticker
        assert {**ticker, "market": ["otc"]} == otc
    assert all(urlparse(q["url"]).netloc == "api.massive.com" for q in queries)
    assert all("apiKey" not in q["url"] for q in queries)


@pytest.mark.parametrize(
    "field",
    [
        *DISARM_FLAGS,
        "maximum_requests",
        "retries",
        "actual_key_plan_verified",
        "follow_redirects",
        "returns_allowed",
    ],
)
def test_weakened_or_expanded_contract_rejected(field: str) -> None:
    c = json.loads(CONFIG.read_bytes())
    validate_contract(c)
    c[field] = True if type(c[field]) is bool else c[field] + 1
    with pytest.raises(ValueError, match="contract"):
        validate_contract(c)


@pytest.mark.parametrize(
    "status,code,denied",
    [
        ("NOT_AUTHORIZED", 403, True),
        ("NOT_AUTHORIZED", 200, True),
        ("ERROR", 401, False),
        ("OK", 500, False),
    ],
)
def test_errors_preserved_but_never_promoted(status: str, code: int, denied: bool) -> None:
    result = response({"status": status, "message": "synthetic private message"}, code)
    assert not result["complete"] and result["permission_denied"] == denied
    assert "private message" not in json.dumps(result)
    assert all(result[k] is False for k in DISARM_FLAGS)


def test_valid_bars_are_reduced_to_dates_and_duplicates_rejected() -> None:
    bar = {
        "o": 10,
        "c": 11,
        "h": 12,
        "l": 9,
        "v": 20,
        "t": int(pd.Timestamp("2024-09-17T04:00Z").timestamp() * 1000),
    }
    data = {"status": "OK", "ticker": "NOBH", "adjusted": False, "results": [bar]}
    result = response(data)
    assert result["complete"] and result["session_dates"] == ["2024-09-17"]
    assert not any(k in result for k in ("o", "c", "returns"))
    data["results"] = [bar, bar]
    assert not response(data)["complete"]


def test_empty_200_and_missing_identity_do_not_imply_no_trade_or_cik() -> None:
    assert response({"status": "OK", "ticker": "NOBH", "adjusted": False})["empty_not_no_trade"]
    result = response(
        {
            "status": "OK",
            "results": [{"ticker": "NOBH", "market": "otc", "share_class_figi": "BBG001S5TP08"}],
        },
        index=8,
    )
    assert result["complete"] and result["identities"][0]["cik"] is None
    assert not result["return_admission"]


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"status": []},
        {"status": "OK", "next_url": "https://example.org/next"},
        {"status": "OK", "results": {}},
    ],
)
def test_bad_or_incomplete_reference_envelopes_remain_incomplete(data: Any) -> None:
    assert not response(data, index=8)["complete"]


def test_hash_checked_before_json() -> None:
    with pytest.raises(ValueError, match="hash changed"):
        inspect_response(build_queries()[0], b"bad", "a" * 64, 200)


@pytest.fixture()
def frozen(tmp_path: Path) -> tuple[Path, str]:
    """Synthetic source-pinned plan; collection still validates queries and code hashes."""
    plan = {
        "config": json.loads(CONFIG.read_bytes()),
        "queries": build_queries(),
        "cached_dates": ["2024-09-17"],
        "implementation_sha256": {},
    }
    root = write_bundle(tmp_path / "plans", "synthetic", {"plan.json": _encode(plan)})
    return root, digest_file(root / "output_manifest.json")


@pytest.mark.parametrize("failure", [403, 401, 429, 302, 500, "reflection", "network"])
def test_mock_capture_preserves_error_bytes_and_stops_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: tuple[Path, str],
    failure: int | str,
) -> None:
    monkeypatch.setattr("scripts.probe_publication_market_access.time.sleep", lambda duration: None)
    key = "Z" * 32  # Synthetic, never a real credential.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert key not in str(request.url) and request.headers["Authorization"] == "Bearer " + key
        if failure == "network":
            raise httpx.ConnectError("not logged", request=request)
        if failure == "reflection":
            return httpx.Response(200, content=key.encode())
        return httpx.Response(int(failure), json={"status": "NOT_AUTHORIZED"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as sender:
        root = collect_probe(*frozen, tmp_path / "raw", key, sender)
    capture = _json(root, digest_file(root / "output_manifest.json"), "receipts.json")
    assert len(capture["receipts"]) == 12
    assert len(calls) == (12 if failure == 403 else 1)
    if type(failure) is int:
        assert (root / "NOBH_original.json").read_bytes() == b'{"status":"NOT_AUTHORIZED"}'
    else:
        assert not (root / "NOBH_original.json").exists()
    if failure == "reflection":
        assert all(
            key.encode() not in path.read_bytes() for path in root.iterdir() if path.is_file()
        )
    audit_root, summary = audit_probe(
        root, digest_file(root / "output_manifest.json"), tmp_path / "audit"
    )
    assert not summary["exact_old_query_replay_complete"]
    assert summary["exact_replay_cached_dates_not_returned"] is None
    assert not summary["expiry_cause_confirmed"] and not summary["returns_computed"]
    assert audit_root.exists()


def test_collection_rejects_redirects_and_query_tampering_before_network(
    tmp_path: Path,
    frozen: tuple[Path, str],
) -> None:
    with httpx.Client(follow_redirects=True) as sender, pytest.raises(ValueError, match="unsafe"):
        collect_probe(*frozen, tmp_path / "raw", "Z" * 32, sender)
    plan = _json(*frozen, "plan.json")
    plan["queries"][0]["url"] = "https://example.org/unsafe"
    root = write_bundle(tmp_path / "other", "synthetic", {"plan.json": _encode(plan)})
    with (
        httpx.Client(follow_redirects=False) as sender,
        pytest.raises(ValueError, match="Changed queries"),
    ):
        collect_probe(
            root, digest_file(root / "output_manifest.json"), tmp_path / "raw", "Z" * 32, sender
        )
