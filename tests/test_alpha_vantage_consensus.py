"""Tests for the forward-only Alpha Vantage consensus ledger."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from sec_data.alpha_vantage_consensus import (
    AlphaVantageConsensusClient,
    AlphaVantageLiveConfig,
    AlphaVantageResponse,
    alpha_vantage_response_status,
    build_alpha_vantage_live_audit,
    build_alpha_vantage_live_plan,
    calculate_alpha_vantage_live_plan,
    collect_alpha_vantage_live_snapshots,
    parse_alpha_vantage_live_response,
)


def _core_assets(rows: int = 22) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "core_asset_id": number,
                "core_asset_version": "core-v1",
                "core_asset_status": "active",
                "company_name": f"Issuer {number}",
                "sec_cik": str(100000 + number).zfill(10),
                "sec_ticker": f"T{number}",
                "mapping_status": "accepted",
                "selection_order": number,
            }
            for number in range(1, rows + 1)
        ]
    )


def test_live_plan_is_deterministic_forward_only_and_return_free() -> None:
    assets = _core_assets().sample(frac=1.0, random_state=7)

    plan = calculate_alpha_vantage_live_plan(
        assets, AlphaVantageLiveConfig(universe_size=20, daily_request_cap=20)
    )

    assert len(plan) == 20
    assert plan["collection_order"].tolist() == list(range(1, 21))
    assert plan["core_asset_id"].tolist() == list(range(1, 21))
    assert plan["forward_only"].all()
    assert not plan["historical_backfill_allowed"].any()
    assert not plan["selection_uses_return_values"].any()
    assert not plan["return_values_loaded"].any()


def test_parser_keeps_quarterly_revenue_and_capture_clock() -> None:
    payload = {
        "symbol": "MSFT",
        "estimates": [
            {
                "date": "2026-09-30",
                "horizon": "fiscal quarter",
                "revenue_estimate_average": "70000000000",
                "revenue_estimate_high": "71000000000",
                "revenue_estimate_low": "69000000000",
                "revenue_estimate_analyst_count": "31",
            },
            {
                "date": "2026-12-31",
                "horizon": "fiscal year",
                "revenue_estimate_average": "300000000000",
            },
        ],
    }

    rows = parse_alpha_vantage_live_response(
        json.dumps(payload).encode(),
        expected_ticker="msft",
        captured_at="2026-08-29T10:00:00+00:00",
        source_sha256="abc123",
    )

    assert len(rows) == 1
    assert rows[0]["target_period_end"] == "2026-09-30"
    assert rows[0]["revenue_estimate_average"] == 70_000_000_000.0
    assert rows[0]["revenue_estimate_analyst_count"] == 31.0
    assert rows[0]["availability_clock"] == "collector_fetched_at"
    assert rows[0]["forward_only_admission"] is True
    assert rows[0]["historical_backtest_admission"] is False
    assert rows[0]["target_period_after_capture"] is True
    assert rows[0]["forward_event_candidate"] is True


def test_response_status_distinguishes_provider_failures() -> None:
    assert alpha_vantage_response_status(b'{"Note":"slow down"}', 200) == "rate_limited"
    assert (
        alpha_vantage_response_status(b'{"Information":"premium only"}', 200)
        == "premium_endpoint_or_invalid_key"
    )
    assert alpha_vantage_response_status(b'{"estimates":[]}', 200) == "ok"
    assert alpha_vantage_response_status(b"not-json", 200) == "malformed_json"


def test_client_redacts_api_key_from_transport_metadata() -> None:
    class Sender:
        def get(
            self,
            url: str,
            *,
            params: dict[str, str],
            headers: dict[str, str],
            timeout: float,
        ) -> httpx.Response:
            del headers, timeout
            request = httpx.Request("GET", url, params=params)
            return httpx.Response(
                200,
                request=request,
                content=b'{"symbol":"MSFT","estimates":[]}',
            )

    client = AlphaVantageConsensusClient(
        "secret-key-123",
        sender=Sender(),
        request_interval_seconds=12.0,
        now=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
    )

    response = client.get_earnings_estimates("MSFT")

    assert "apikey" not in response.requested_url.lower()
    assert "apikey" not in response.response_url.lower()
    assert "secret-key-123" not in response.requested_url
    assert "secret-key-123" not in response.response_url


def test_live_snapshot_round_trip_preserves_forward_only_admission(tmp_path: Path) -> None:
    core_root = tmp_path / "core"
    core_root.mkdir()
    _core_assets(1).to_csv(core_root / "core_asset_82.csv", index=False)
    (core_root / "output_manifest.json").write_text('{"version":"test"}', encoding="utf-8")
    plan = build_alpha_vantage_live_plan(
        core_root,
        output_root=tmp_path / "targets",
        config=AlphaVantageLiveConfig(universe_size=1, daily_request_cap=1),
    )
    payload = json.dumps(
        {
            "symbol": "T1",
            "estimates": [
                {
                    "date": "2026-09-30",
                    "horizon": "fiscal quarter",
                    "revenue_estimate_average": "1000000",
                    "revenue_estimate_high": "1100000",
                    "revenue_estimate_low": "900000",
                    "revenue_estimate_analyst_count": "4",
                }
            ],
        }
    ).encode()

    class Client:
        def get_earnings_estimates(self, ticker: str) -> AlphaVantageResponse:
            assert ticker == "T1"
            return AlphaVantageResponse(
                requested_url=(
                    "https://www.alphavantage.co/query?function=EARNINGS_ESTIMATES&symbol=T1"
                ),
                response_url=(
                    "https://www.alphavantage.co/query?function=EARNINGS_ESTIMATES&symbol=T1"
                ),
                content=payload,
                fetched_at=datetime(2026, 8, 29, 10, tzinfo=timezone.utc),
                status_code=200,
                headers={"content-type": "application/json"},
            )

    raw_root = tmp_path / "raw"
    collection = collect_alpha_vantage_live_snapshots(
        pd.read_parquet(plan.output_root / "alpha_vantage_live_targets.parquet"),
        Client(),  # type: ignore[arg-type]
        raw_root=raw_root,
        capture_date=date(2026, 8, 29),
        limit=1,
    )
    audit = build_alpha_vantage_live_audit(
        plan.output_root,
        raw_root=raw_root,
        output_root=tmp_path / "processed",
    )

    assert collection.completed_targets == 1
    assert audit.report["live_snapshot_gate"] is True
    assert audit.report["strict_forward_snapshot_gate"] is True
    assert audit.report["strict_forward_observations"] == 1
    assert audit.report["historical_backtest_admission"] is False
    observations = pd.read_parquet(
        audit.output_root / "alpha_vantage_live_revenue_consensus.parquet"
    )
    assert observations["forward_only_admission"].all()
    assert not observations["historical_backtest_admission"].any()
