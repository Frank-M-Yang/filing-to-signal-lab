"""Fixed-case sampling, exchange identity, bounded requests and failed-case retention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from scripts.build_sec_disclosure_audit import write_bundle
from scripts.replicate_source_sensitivity import audit_capture, collect_plan, freeze_plan
from sec_data.disclosure_audit import digest_file
from sec_data.market_source_sensitivity import calculate_quote_feature_proxy
from sec_data.pre_event_source_audit import parse_pre_event_quotes
from sec_data.validation_liquidity_fallback import yahoo_liquidity_chart_url


def _body(ticker: str, exchange: str) -> bytes:
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": ticker,
                            "currency": "USD",
                            "exchangeTimezoneName": "America/New_York",
                            "instrumentType": "ETF" if ticker == "SPY" else "EQUITY",
                            "exchangeName": exchange,
                        },
                        "timestamp": [int(pd.Timestamp("2025-04-21T13:30Z").timestamp())],
                        "indicators": {
                            "quote": [{"close": [30], "volume": [100]}],
                            "adjclose": [{"adjclose": [-999]}],
                        },
                    }
                ],
            }
        }
    ).encode()


@pytest.mark.parametrize(
    "ticker,mic,exchange",
    [("USNA", "XNYS", "NYQ"), ("UFPT", "XNAS", "NMS"), ("SPY", "ARCX", "PCX")],
)
def test_explicit_exchange_and_ticker_routing(ticker: str, mic: str, exchange: str) -> None:
    result = parse_pre_event_quotes(
        _body(ticker, exchange),
        ticker,
        "2025-04-21",
        "2025-04-21",
        "2025-04-22T20:05Z",
        expected_exchange_mic=mic,
    )
    assert result.ticker.tolist() == [ticker] and result.quote_close.tolist() == [30]
    if ticker != "SPY":
        with pytest.raises(ValueError, match="mismatch"):
            parse_pre_event_quotes(
                _body(ticker, "PCX"),
                ticker,
                "2025-04-21",
                "2025-04-21",
                "2025-04-22T20:05Z",
                expected_exchange_mic=mic,
            )


def test_feature_math_accepts_only_explicit_issuer() -> None:
    dates = pd.bdate_range("2025-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    s = pd.DataFrame(
        {
            "ticker": "USNA",
            "session_date": dates,
            "quote_close": [30 + i * 0.1 for i in range(61)],
            "volume": 100,
            "source_sha256": "a" * 64,
        }
    )
    b = s.assign(ticker="SPY")
    good = calculate_quote_feature_proxy(s, b, dates, "2025-04-22T20:05Z", issuer_ticker="USNA")
    assert good.value.notna().all() and not good.model_admission.any()
    with pytest.raises(ValueError):
        calculate_quote_feature_proxy(s, b, dates, "2025-04-22T20:05Z")


def _plan(root: Path) -> Path:
    control = write_bundle(
        root / "control", "test", {"summary.json": b'{"comparable_features":{"yahoo_vs_cached":7}}'}
    )
    qs = [
        {
            "id": f"{case}_{t}",
            "case_ticker": case,
            "ticker": t,
            "start": "2025-04-21",
            "end": "2025-04-21",
            "announcement_at": "2025-04-22T20:05Z",
            "exchange_mic": "XNYS",
            "url": yahoo_liquidity_chart_url(t, "2025-04-21", "2025-04-21"),
        }
        for case in ("USNA", "UFPT")
        for t in (case, "SPY")
    ]
    plan = {
        "queries": qs,
        "cases": [{"event": {"ticker": t}} for t in ("USNA", "UFPT")],
        "config": {
            "prior_control_root": control.as_posix(),
            "prior_control_manifest_sha256": digest_file(control / "output_manifest.json"),
            "selection_reason": "synthetic fixed test",
        },
        "implementation_sha256": {},
    }
    return write_bundle(root / "plan", "test", {"plan.json": json.dumps(plan).encode()})


@pytest.mark.parametrize("status", [403, 429, 302])
def test_access_denial_stops_without_replacing_failed_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    real_client = httpx.Client
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert "Authorization" not in request.headers
        return httpx.Response(
            status, content=b"denial", headers={"location": "https://other.example/"}
        )

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr("scripts.replicate_source_sensitivity.httpx.Client", client)
    plan = _plan(tmp_path)
    capture = collect_plan(plan, digest_file(plan / "output_manifest.json"), tmp_path / "raw")
    assert len(seen) == 1
    assert (capture / "USNA_USNA.json").read_bytes() == b"denial"
    output, report = audit_capture(
        capture, digest_file(capture / "output_manifest.json"), tmp_path / "output"
    )
    assert report["new_cases_selected"] == 2 and report["new_cases_compared"] == 0
    assert report["new_cases_full_13_feature_comparison"] == 0
    assert [r["ticker"] for r in report["cases"]] == ["USNA", "UFPT"]
    assert all(r["errors"] for r in report["cases"])
    assert not report["prior_mbuu_control"]["included_in_new_case_denominator"]
    assert not report["model_admission"] and not report["sample_replaced"]
    again, _ = audit_capture(
        capture, digest_file(capture / "output_manifest.json"), tmp_path / "output"
    )
    assert again == output


def test_unsafe_url_rejected_before_client_construction(tmp_path: Path) -> None:
    original = _plan(tmp_path)
    plan = json.loads((original / "plan.json").read_bytes())
    plan["queries"][0]["url"] = "https://other.example/"
    bad = write_bundle(tmp_path / "bad", "test", {"plan.json": json.dumps(plan).encode()})
    with pytest.raises(ValueError, match="plan"):
        collect_plan(bad, digest_file(bad / "output_manifest.json"), tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("sample_replacement", True),
        ("post_event_returns_allowed", True),
        ("model_admission", True),
        ("maximum_requests", 100),
        ("required_sessions", 45),
        ("control_in_new_case_denominator", True),
    ],
)
def test_weakened_selection_contract_rejected(tmp_path: Path, key: str, value: Any) -> None:
    c = json.loads(Path("docs/source_sensitivity_replication_v1.json").read_bytes())
    c[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        freeze_plan(path, tmp_path / "plans")
    assert not (tmp_path / "plans").exists()


def test_case_order_cannot_be_changed_after_results(tmp_path: Path) -> None:
    c = json.loads(Path("docs/source_sensitivity_replication_v1.json").read_bytes())
    c["cases"].reverse()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        freeze_plan(path, tmp_path / "plans")
