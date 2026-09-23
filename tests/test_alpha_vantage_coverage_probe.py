from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from sec_data.alpha_vantage_budget import AlphaVantageBudgetError, AlphaVantageRequestBudget
from sec_data.alpha_vantage_consensus import AlphaVantageConsensusClient, AlphaVantageConsensusError
from sec_data.alpha_vantage_coverage_probe import (
    build_alpha_vantage_coverage_probe_audit,
    build_alpha_vantage_coverage_probe_plan,
    collect_alpha_vantage_coverage_probe,
)


def test_budget_preserves_twenty_original_requests_and_one_retry(tmp_path: Path) -> None:
    budget = AlphaVantageRequestBudget(
        state_root=tmp_path / "budget",
        legacy_raw_root=tmp_path / "original",
        protected_target_ids=tuple(f"target{i}" for i in range(20)),
        retry_reserve=1,
        now=lambda: datetime(2026, 9, 14, tzinfo=timezone.utc),
        sleep=lambda _: None,
    )
    assert budget.summary()["available_requests"] == 4
    for i in range(4):
        with budget.request(f"NEW{i}"):
            pass  # Uncertain/failed attempts still consume allowance.
    assert budget.summary()["available_requests"] == 0
    with pytest.raises(AlphaVantageBudgetError, match="exhausted_or_protected"):
        with budget.request("FIFTH"):
            pytest.fail("Fifth request must never dispatch")


def test_legacy_responses_and_new_receipts_are_not_double_counted(tmp_path: Path) -> None:
    instant = datetime(2026, 9, 14, tzinfo=timezone.utc)
    budget = AlphaVantageRequestBudget(
        state_root=tmp_path / "budget",
        legacy_raw_root=tmp_path / "original",
        now=lambda: instant,
        sleep=lambda _: None,
    )
    content = b'{"symbol":"AAA","estimates":[]}'
    with budget.request("AAA") as reservation:
        budget.record_response(reservation, content)
    _legacy_response(budget.legacy_raw_root, "AAA", content, instant)
    _legacy_response(budget.legacy_raw_root, "BBB", b"older response", instant)
    summary = budget.summary()
    assert summary["local_attempts_recorded"] == 1
    assert summary["legacy_response_attempts_not_in_ledger"] == 1
    assert summary["available_requests"] == 23


def test_shared_lock_and_duplicate_guard_fail_closed(tmp_path: Path) -> None:
    budget = AlphaVantageRequestBudget(
        state_root=tmp_path / "budget",
        legacy_raw_root=tmp_path / "original",
        now=lambda: datetime(2026, 9, 14, tzinfo=timezone.utc),
        sleep=lambda _: None,
        single_attempt_per_ticker_day=True,
    )
    with budget.request("AAA"):
        with pytest.raises(AlphaVantageBudgetError, match="shared_collector_busy"):
            with budget.request("BBB"):
                pytest.fail("Concurrent request must not dispatch")
    with pytest.raises(AlphaVantageBudgetError, match="already_attempted_today"):
        with budget.request("AAA"):
            pytest.fail("Same-day duplicate must not dispatch")
    assert budget.summary()["local_attempts_recorded"] == 1
    assert not (budget.state_root / "collector.lock").exists()


def test_throttle_recomputes_utc_day_after_midnight(tmp_path: Path) -> None:
    clock = [datetime(2026, 9, 14, 23, 59, 55, tzinfo=timezone.utc)]
    budget = AlphaVantageRequestBudget(
        state_root=tmp_path / "budget",
        legacy_raw_root=tmp_path / "original",
        now=lambda: clock[0],
        sleep=lambda delay: clock.__setitem__(0, clock[0] + timedelta(seconds=delay)),
    )
    _legacy_response(budget.legacy_raw_root, "AAA", b"earlier", clock[0] - timedelta(seconds=1))
    with budget.request("BBB") as reservation:
        assert reservation.parent.name == "2026-09-15"
    assert budget.summary()["utc_day"] == "2026-09-15"


def test_first_fixed_batch_is_isolated_repeatable_and_not_formally_admitted(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    targets = pd.read_parquet(plan / "coverage_probe_targets.parquet")
    sender, client = _client(tmp_path)
    raw = tmp_path / "raw"
    result = collect_alpha_vantage_coverage_probe(targets, client, raw_root=raw)
    audit = build_alpha_vantage_coverage_probe_audit(
        plan, raw_root=raw, output_root=tmp_path / "audit"
    )
    assert sender.symbols == ["NEW1", "NEW2", "NEW3", "NEW4"]
    assert result["requests_made"] == 4
    assert audit.report["candidate_issuers"] == 4
    assert audit.report["unprobed_issuers"] == 2
    assert audit.report["repeated_candidate_issuers"] == 0
    assert not audit.report["formal_model_admission"]
    rows = pd.read_parquet(audit.output_root / "coverage_probe_future_candidates.parquet")
    assert rows["currency"].isna().all() and rows["unit"].isna().all()
    assert not rows["sec_acceptance_verified"].any()
    again = collect_alpha_vantage_coverage_probe(targets, client, raw_root=raw)
    assert again["already_attempted_today"] == 4 and again["requests_made"] == 0
    assert len(sender.symbols) == 4
    sender.clock += timedelta(days=1)
    collect_alpha_vantage_coverage_probe(targets, client, raw_root=raw)
    second = build_alpha_vantage_coverage_probe_audit(
        plan, raw_root=raw, output_root=tmp_path / "audit"
    )
    assert second.report["repeated_candidate_issuers"] == 4
    assert second.report["adjacent_snapshot_pairs"] == 4
    assert second.report["comparable_quarter_pairs"] == 4
    assert second.report["identical_byte_snapshot_pairs"] == 4
    assert second.report["snapshot_pairs_spanning_24_hours"] == 4
    assert second.report["quarter_comparison_status_counts"] == {"aggregate_fields_unchanged": 4}
    assert not second.report["new_cohort_registered"]
    assert not second.report["analyst_membership_identity_verified"]
    assert (second.output_root / "output_manifest.json").read_bytes() != (
        audit.output_root / "output_manifest.json"
    ).read_bytes()


def test_rate_limit_preserves_exact_error_and_does_not_replace_company(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    targets = pd.read_parquet(plan / "coverage_probe_targets.parquet")
    sender, client = _client(tmp_path, mode="rate_limited")
    result = collect_alpha_vantage_coverage_probe(targets, client, raw_root=tmp_path / "raw")
    assert result["requests_made"] == 1
    assert result["blocking_reason"] == "rate_limited"
    assert sender.symbols == ["NEW1"]
    path = next((tmp_path / "raw").glob("snapshots/*/*/*/response.bin"))
    assert path.read_bytes() == b'{"Note":"quota exhausted"}'
    audit = build_alpha_vantage_coverage_probe_audit(
        plan, raw_root=tmp_path / "raw", output_root=tmp_path / "audit"
    )
    assert audit.report["candidate_issuers"] == 0
    assert audit.report["attempted_issuers"] == 1


@pytest.mark.parametrize("mode", ["invalid_fields", "no_future", "wrong_symbol", "zero_coverage"])
def test_bad_fields_past_periods_and_wrong_symbols_are_not_candidates(
    tmp_path: Path, mode: str
) -> None:
    plan = _plan(tmp_path)
    _, client = _client(tmp_path, mode=mode)
    collect_alpha_vantage_coverage_probe(
        pd.read_parquet(plan / "coverage_probe_targets.parquet"),
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    audit = build_alpha_vantage_coverage_probe_audit(
        plan, raw_root=tmp_path / "raw", output_root=tmp_path / "audit"
    )
    assert audit.report["candidate_issuers"] == 0
    assert not audit.report["formal_model_admission"]
    if mode == "zero_coverage":
        rejections = pd.read_parquet(audit.output_root / "coverage_probe_field_rejections.parquet")
        assert rejections["revenue_estimate_average"].eq(0).all()
        assert rejections["revenue_estimate_analyst_count"].eq(0).all()
        assert (
            rejections["field_rejection_reason"]
            .eq("zero_revenue_and_zero_analyst_count_no_valid_coverage")
            .all()
        )


def test_transport_failure_consumes_attempt_without_leaking_exception_key(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _, client = _client(tmp_path, mode="transport")
    result = collect_alpha_vantage_coverage_probe(
        pd.read_parquet(plan / "coverage_probe_targets.parquet"), client, raw_root=tmp_path / "raw"
    )
    assert result["blocking_reason"] == "transport_failure"
    assert client.request_budget is not None
    assert client.request_budget.summary()["local_attempts_recorded"] == 1
    for path in (tmp_path / "raw").rglob("*.json"):
        assert "secret-test-key" not in path.read_text()


def test_quota_information_stops_probe_and_redacts_reflected_key(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    sender, client = _client(tmp_path, mode="quota_echo")
    result = collect_alpha_vantage_coverage_probe(
        pd.read_parquet(plan / "coverage_probe_targets.parquet"), client, raw_root=tmp_path / "raw"
    )
    assert result["blocking_reason"] == "rate_limited"
    assert result["requests_made"] == 1 and sender.symbols == ["NEW1"]
    audit = build_alpha_vantage_coverage_probe_audit(
        plan, raw_root=tmp_path / "raw", output_root=tmp_path / "audit"
    )
    assert audit.report["provider_reason_counts"] == {"rate_limited": 1}
    assert audit.report["credential_redacted_response_snapshots"] == 1
    assert audit.report["candidate_issuers"] == 0
    for path in (tmp_path / "raw").rglob("*"):
        if path.is_file():
            assert b"secret-test-key" not in path.read_bytes()


def test_response_tampering_and_backdating_are_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _, client = _client(tmp_path)
    collect_alpha_vantage_coverage_probe(
        pd.read_parquet(plan / "coverage_probe_targets.parquet"),
        client,
        raw_root=tmp_path / "raw",
        limit=1,
    )
    path = next((tmp_path / "raw").glob("snapshots/*/*/*/manifest.json"))
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["capture_date"] = "2026-09-13"
    path.write_text(json.dumps(manifest))
    with pytest.raises(AlphaVantageConsensusError, match="backdated"):
        build_alpha_vantage_coverage_probe_audit(plan, raw_root=tmp_path / "raw")
    path.write_bytes(original)
    (path.parent / "response.bin").write_bytes(b"changed")
    with pytest.raises(AlphaVantageConsensusError, match="Changed probe response"):
        build_alpha_vantage_coverage_probe_audit(plan, raw_root=tmp_path / "raw")


def _plan(root: Path) -> Path:
    parent = root / "feasibility"
    parent.mkdir()
    queue = pd.DataFrame(
        [
            {
                "probe_order": i,
                "core_asset_id": str(i),
                "company_name": f"New issuer {i}",
                "sec_cik": f"{i:010d}",
                "sec_ticker": f"NEW{i}",
                "coverage_status": "unprobed",
                "mapping_eligible": True,
                "queue_is_registration": False,
                "selection_uses_return_values": False,
            }
            for i in range(1, 7)
        ]
    )
    path = parent / "unprobed_company_queue.parquet"
    queue.to_parquet(path, index=False)
    report = parent / "prospective_cohort_feasibility_report.json"
    report.write_text(json.dumps({"actual_values_loaded": False, "return_values_loaded": False}))
    (parent / "output_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {path.name: {"sha256": _digest(path)}},
                "report_sha256": _digest(report),
            }
        )
    )
    return build_alpha_vantage_coverage_probe_plan(parent, output_root=root / "plans").output_root


class _Sender:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.symbols: list[str] = []
        self.clock = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)

    def get(
        self, url: str, *, params: dict[str, str], headers: dict[str, str], timeout: float
    ) -> httpx.Response:
        del headers, timeout
        self.symbols.append(params["symbol"])
        self.clock += timedelta(seconds=20)
        if self.mode == "transport":
            raise httpx.ConnectError("secret-test-key must never appear in output")
        payload = {
            "symbol": "WRONG" if self.mode == "wrong_symbol" else params["symbol"],
            "estimates": [
                {
                    "horizon": "fiscal quarter",
                    "date": "2026-06-30" if self.mode == "no_future" else "2026-12-31",
                    "revenue_estimate_average": "0" if self.mode == "zero_coverage" else "100",
                    "revenue_estimate_low": "90",
                    "revenue_estimate_high": "110",
                    "revenue_estimate_analyst_count": (
                        "0"
                        if self.mode == "zero_coverage"
                        else "3.5"
                        if self.mode == "invalid_fields"
                        else "4"
                    ),
                }
            ],
        }
        content = (
            b'{"Information":"Your API key as secret-test-key: '
            b'API rate limit is 25 requests per day"}'
            if self.mode == "quota_echo"
            else b'{"Note":"quota exhausted"}'
            if self.mode == "rate_limited"
            else json.dumps(payload).encode()
        )
        return httpx.Response(
            200, request=httpx.Request("GET", url, params=params), content=content
        )


def _client(root: Path, mode: str = "ok") -> tuple[_Sender, AlphaVantageConsensusClient]:
    sender = _Sender(mode)
    budget = AlphaVantageRequestBudget(
        state_root=root / "budget",
        legacy_raw_root=root / "original",
        now=lambda: sender.clock,
        single_attempt_per_ticker_day=True,
        sleep=lambda seconds: setattr(sender, "clock", sender.clock + timedelta(seconds=seconds)),
    )
    client = AlphaVantageConsensusClient(
        "secret-test-key",
        sender=sender,
        request_budget=budget,
        now=lambda: sender.clock,
        sleep=lambda _: None,
    )
    return sender, client


def _legacy_response(root: Path, ticker: str, content: bytes, instant: datetime) -> None:
    directory = (
        root
        / "snapshots"
        / instant.date().isoformat()
        / ticker
        / hashlib.sha256(content).hexdigest()
    )
    directory.mkdir(parents=True)
    path = directory / "response.bin"
    path.write_bytes(content)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "sha256": _digest(path),
                "path": str(path.resolve()),
                "fetched_at": instant.isoformat(),
            }
        )
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
