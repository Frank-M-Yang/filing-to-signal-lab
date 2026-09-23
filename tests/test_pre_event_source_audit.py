"""Synthetic quote schemas, contamination and immutable failed-response controls."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from scripts.audit_mbuu_source_overlap import audit_sources, collect_quotes
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.pre_event_source_audit import compare_quote_overlap, parse_pre_event_quotes


def _chart() -> dict[str, Any]:
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "MBUU",
                        "currency": "USD",
                        "exchangeTimezoneName": "America/New_York",
                        "instrumentType": "EQUITY",
                        "exchangeName": "NGM",
                        "regularMarketPrice": 999999,
                    },
                    "timestamp": [
                        int(pd.Timestamp(d).timestamp())
                        for d in ("2024-08-06T13:30Z", "2024-08-07T13:30Z")
                    ],
                    "indicators": {
                        "quote": [{"close": [20.5, 21.1], "volume": [1000, 2000]}],
                        "adjclose": [{"adjclose": [100000, 200000]}],
                    },
                }
            ],
        }
    }


def _parse(body: dict[str, Any]) -> pd.DataFrame:
    return parse_pre_event_quotes(
        json.dumps(body).encode(), "MBUU", "2024-08-06", "2024-08-07", "2024-10-31T11:00Z"
    )


def test_current_quote_and_adjusted_fields_cannot_enter_historical_prices() -> None:
    result = _parse(_chart())
    assert list(result.quote_close) == [20.5, 21.1]
    assert list(result.session_date) == ["2024-08-06", "2024-08-07"]
    altered = _chart()
    altered["chart"]["result"][0]["meta"]["regularMarketPrice"] = -999999
    altered["chart"]["result"][0]["indicators"]["adjclose"] = []
    pd.testing.assert_frame_equal(
        result.drop(columns="source_sha256"), _parse(altered).drop(columns="source_sha256")
    )


@pytest.mark.parametrize(
    "bad",
    [
        "symbol",
        "currency",
        "timezone",
        "instrument",
        "exchange",
        "future",
        "outside",
        "duplicate",
        "unordered",
        "missing",
        "negative",
        "fraction",
        "array",
        "error",
    ],
)
def test_quote_rejects_mismatched_or_contaminated_data(bad: str) -> None:
    body = _chart()
    item = body["chart"]["result"][0]
    keys = {
        "symbol": "symbol",
        "currency": "currency",
        "timezone": "exchangeTimezoneName",
        "instrument": "instrumentType",
        "exchange": "exchangeName",
    }
    if bad in keys:
        item["meta"][keys[bad]] = "unknown"
    elif bad in {"future", "outside"}:
        item["timestamp"][1] = int(
            pd.Timestamp(
                "2024-10-31T13:30Z" if bad == "future" else "2024-08-08T13:30Z"
            ).timestamp()
        )
    elif bad == "duplicate":
        item["timestamp"][1] = item["timestamp"][0]
    elif bad == "unordered":
        item["timestamp"].reverse()
    elif bad == "error":
        body["chart"]["error"] = {"code": "unavailable"}
    else:
        quote = item["indicators"]["quote"][0]
        if bad == "missing":
            quote["close"][0] = None
        elif bad == "array":
            quote["volume"] = [1]
        else:
            quote["volume"][0] = -1 if bad == "negative" else 1.1
    with pytest.raises(ValueError):
        _parse(body)


def test_requested_forward_interval_rejected_before_parsing() -> None:
    with pytest.raises(ValueError, match="strictly before"):
        parse_pre_event_quotes(b"not JSON", "MBUU", "2024-08-06", "2024-10-31", "2024-10-31T11:00Z")


def test_overlap_prices_do_not_override_volume_mismatches_or_grant_admission() -> None:
    a = _parse(_chart())
    b = a.copy()
    b.loc[0, "quote_close"] += 0.005
    b.loc[1, "volume"] += 1
    report, rows = compare_quote_overlap(a, b, minimum_sessions=2)
    assert report["close_disagreements"] == 0
    assert report["volume_disagreements"] == 1
    assert report["maximum_absolute_volume_relative_difference"] == pytest.approx(1 / 2000)
    assert not report["overlap_consistent"]
    assert len(rows) == 2
    same, _ = compare_quote_overlap(a, a, minimum_sessions=2)
    assert same["overlap_consistent"]
    assert not same["automatic_splicing"] and not same["model_admission"]
    assert not same["unadjusted_semantics_certified"]
    insufficient, _ = compare_quote_overlap(a, a)
    assert not insufficient["overlap_consistent"]


def test_zero_volume_does_not_create_infinite_relative_difference() -> None:
    a = _parse(_chart())
    a["volume"] = 0
    b = a.copy()
    b["volume"] = 1
    report, _ = compare_quote_overlap(a, b, minimum_sessions=2)
    assert report["zero_primary_volume_sessions"] == 2
    assert report["maximum_absolute_volume_relative_difference"] is None
    assert report["volume_disagreements"] == 2
    assert not report["overlap_consistent"]


@pytest.mark.parametrize("bad", ["identity", "duplicate", "nan", "provenance", "extra"])
def test_overlap_input_guard(bad: str) -> None:
    a = _parse(_chart())
    b = a.copy()
    if bad == "identity":
        b["ticker"] = "SPY"
    elif bad == "duplicate":
        b = pd.concat([b, b])
    elif bad == "nan":
        b.loc[0, "quote_close"] = float("nan")
    elif bad == "provenance":
        b["source_sha256"] = "missing"
    else:
        b["forward_return"] = 1
    with pytest.raises(ValueError):
        compare_quote_overlap(a, b)


def test_failed_and_redirect_responses_are_preserved_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = {
        "MBUU": "https://query1.finance.yahoo.com/one",
        "SPY": "https://query1.finance.yahoo.com/two",
    }
    frozen = write_bundle(tmp_path / "plan", "test", {"plan.json": b"{}"})
    monkeypatch.setattr(
        "scripts.audit_mbuu_source_overlap._freeze",
        lambda *args: (frozen, {"config_sha256": "a" * 64, "urls": urls}),
    )
    requested = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["follow_redirects"] is False

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str) -> httpx.Response:
            requested.append(url)
            if len(requested) == 1:
                return httpx.Response(
                    302,
                    content=b"redirect body",
                    request=httpx.Request("GET", url),
                    headers={"location": "https://other.example/"},
                )
            raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr("scripts.audit_mbuu_source_overlap.httpx.Client", Client)
    monkeypatch.setattr("scripts.audit_mbuu_source_overlap.time.sleep", lambda _: None)
    root = collect_quotes(tmp_path / "config.json", tmp_path / "raw")
    assert requested == list(urls.values())
    assert (root / "MBUU.json").read_bytes() == b"redirect body"
    receipts = json.loads((root / "receipts.json").read_bytes())["receipts"]
    assert receipts[0]["status_code"] == 302
    assert receipts[1]["error"] == "ReadTimeout"


def test_contract_rejects_outcome_or_tolerance_changes_before_io(tmp_path: Path) -> None:
    original = json.loads(Path("docs/mbuu_source_overlap_v1.json").read_bytes())
    for key, value in [
        ("returns_allowed", True),
        ("price_absolute_tolerance_usd", 10),
        ("tickers", ["SPY"]),
    ]:
        c = copy.deepcopy(original)
        c[key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError, match="contract"):
            audit_sources(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
