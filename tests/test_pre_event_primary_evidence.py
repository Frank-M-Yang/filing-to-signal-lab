"""Permission boundaries, missingness, immutable receipts and pre-event evidence checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from scripts.build_sec_disclosure_audit import write_bundle
from scripts.probe_mbuu_primary_history import audit_probe, collect_probe, freeze_plan
from sec_data.disclosure_audit import digest_file
from sec_data.pre_event_primary_evidence import (
    CIK,
    COMPOSITE_FIGI,
    SHARE_FIGI,
    audit_cached_timeline,
    audit_primary_response,
)


def _bytes(value: Any) -> bytes:
    return json.dumps(value).encode()


def _query(kind: str = "bars") -> dict[str, str]:
    return {"id": f"MBUU_{kind}", "ticker": "MBUU", "kind": kind, "date": "2024-08-06"}


def _bars() -> dict[str, Any]:
    return {
        "status": "OK",
        "ticker": "MBUU",
        "adjusted": False,
        "results": [
            {
                "t": int(pd.Timestamp("2024-08-06T04:00Z").timestamp() * 1000),
                "o": 30,
                "h": 32,
                "l": 29,
                "c": 31,
                "v": 100,
            }
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "partial",
        "future",
        "adjusted",
        "duplicate",
        "pagination",
        "missing_results",
        "bad_status",
    ],
)
def test_price_coverage_is_not_http_success(change: str) -> None:
    d = _bars()
    sessions = ["2024-08-06"]
    if change == "partial":
        sessions.append("2024-08-07")
    elif change == "future":
        d["results"][0]["t"] = int(pd.Timestamp("2024-10-31T04:00Z").timestamp() * 1000)
    elif change == "adjusted":
        d["adjusted"] = True
    elif change == "duplicate":
        d["results"] *= 2
    elif change == "pagination":
        d["next_url"] = "https://api.massive.com/next"
    elif change == "missing_results":
        d.pop("results")
    elif change == "bad_status":
        d["status"] = "NOT_AUTHORIZED"
    report = audit_primary_response(_bytes(d), _query(), sessions)
    assert report["evidence_complete"] == (change == "none")


@pytest.mark.parametrize("kind", ["splits", "dividends"])
def test_no_actions_requires_explicit_complete_empty_response(kind: str) -> None:
    d = {"status": "OK", "results": []}
    good = audit_primary_response(_bytes(d), _query(kind), [])
    assert good["evidence_complete"] and good["observed_rows"] == 0
    del d["results"]
    unknown = audit_primary_response(_bytes(d), _query(kind), [])
    assert not unknown["evidence_complete"] and unknown["observed_rows"] is None


@pytest.mark.parametrize(
    "changed_field",
    [
        None,
        "share_class_figi",
        "composite_figi",
        "primary_exchange",
        "cik",
        "active",
        "currency_name",
    ],
)
def test_exact_historical_identity(changed_field: str | None) -> None:
    item = {
        "ticker": "MBUU",
        "cik": CIK,
        "share_class_figi": SHARE_FIGI,
        "composite_figi": COMPOSITE_FIGI,
        "primary_exchange": "XNAS",
        "currency_name": "usd",
        "market": "stocks",
        "locale": "us",
        "type": "CS",
        "active": True,
    }
    if changed_field:
        item[changed_field] = "wrong"
    report = audit_primary_response(
        _bytes({"status": "OK", "results": [item]}), _query("reference"), []
    )
    assert report["evidence_complete"] == (changed_field is None)


@pytest.mark.parametrize("bad", [None, "missing", "change", "identity", "duplicate"])
def test_timeline_is_corroboration_not_certification(bad: str | None) -> None:
    result = {
        "cik": CIK,
        "composite_figi": COMPOSITE_FIGI,
        "events": [
            {"date": "2014-01-31", "type": "ticker_change", "ticker_change": {"ticker": "MBUU"}}
        ],
    }
    if bad == "missing":
        result.pop("events")
    elif bad == "change":
        result["events"].append(
            {"date": "2024-09-01", "type": "ticker_change", "ticker_change": {"ticker": "OTHER"}}
        )
    elif bad == "identity":
        result["composite_figi"] = "OTHER"
    elif bad == "duplicate":
        result["events"] *= 2
    r = audit_cached_timeline(_bytes({"status": "OK", "results": result}))
    assert r["consistent"] == (bad is None)
    assert r["experimental_endpoint"]


def _plan(tmp_path: Path) -> Path:
    qs = []
    for name, kind in [
        ("MBUU_bars", "bars"),
        ("SPY_bars", "bars"),
        ("MBUU_reference_start", "reference"),
        ("MBUU_reference_end", "reference"),
        ("MBUU_splits", "splits"),
        ("MBUU_dividends", "dividends"),
        ("SPY_splits", "splits"),
        ("SPY_dividends", "dividends"),
    ]:
        qs.append(
            {
                "id": name,
                "kind": kind,
                "ticker": name.split("_")[0],
                "date": "2024-08-06",
                "url": f"https://api.massive.com/{name}",
            }
        )
    return write_bundle(
        tmp_path / "plans",
        "test",
        {
            "plan.json": _bytes(
                {
                    "queries": qs,
                    "required_sessions": ["2024-08-06"],
                    "config": {"maximum_requests": 8},
                    "ticker_timeline": {"consistent": True},
                }
            )
        },
    )


@pytest.mark.parametrize(
    "first_status, expected_requests", [(401, 1), (429, 1), (302, 1), (403, 7), (200, 8)]
)
def test_collection_preserves_denials_and_stops_correct_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_status: int, expected_requests: int
) -> None:
    key = "synthetic_credential_only"
    requested = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {key}"
        assert key not in str(request.url)
        requested.append(str(request.url))
        return httpx.Response(
            first_status if len(requested) == 1 else 200,
            content=b'{"status":"OK","results":[]}',
            headers={"location": "https://other.example/"},
        )

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("scripts.probe_mbuu_primary_history.httpx.Client", client)
    monkeypatch.setattr("scripts.probe_mbuu_primary_history.time.sleep", lambda _: None)
    plan = _plan(tmp_path)
    captured = collect_probe(
        plan, digest_file(plan / "output_manifest.json"), tmp_path / "raw", key
    )
    assert len(requested) == expected_requests
    assert (captured / "MBUU_bars.json").read_bytes() == b'{"status":"OK","results":[]}'
    receipts = json.loads((captured / "receipts.json").read_bytes())["receipts"]
    if first_status == 403:
        assert receipts[1]["skipped"] == "endpoint_family_denied_not_tested"
        assert receipts[2]["status_code"] == 200
    result, summary = audit_probe(
        captured, digest_file(captured / "output_manifest.json"), tmp_path / "out"
    )
    assert summary["requests_made"] == expected_requests
    assert not summary["model_admission"] and not summary["returns_allowed"]
    assert not summary["both_price_histories_complete"]
    again, _ = audit_probe(
        captured, digest_file(captured / "output_manifest.json"), tmp_path / "out"
    )
    assert again == result
    assert all(key.encode() not in p.read_bytes() for p in tmp_path.rglob("*.json"))


def test_reflected_credential_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "synthetic_credential_only"
    real_client = httpx.Client
    monkeypatch.setattr(
        "scripts.probe_mbuu_primary_history.httpx.Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(403, content=key.encode())),
            **kw,
        ),
    )
    plan = _plan(tmp_path)
    captured = collect_probe(
        plan, digest_file(plan / "output_manifest.json"), tmp_path / "raw", key
    )
    assert not (captured / "MBUU_bars.json").exists()
    assert all(key.encode() not in p.read_bytes() for p in tmp_path.rglob("*.json"))
    receipts = json.loads((captured / "receipts.json").read_bytes())["receipts"]
    assert receipts[0]["error"] == "credential_reflected_body_withheld"
    assert receipts[1]["skipped"] == "privacy_guard"


def test_contract_rejects_mutations_before_network(tmp_path: Path) -> None:
    c = json.loads(Path("docs/mbuu_primary_history_probe_v1.json").read_bytes())
    for k, v in [("returns_allowed", True), ("maximum_requests", 9), ("follow_redirects", True)]:
        altered = copy.deepcopy(c)
        altered[k] = v
        path = tmp_path / "config.json"
        path.write_bytes(_bytes(altered))
        with pytest.raises(ValueError, match="contract"):
            freeze_plan(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
