"""Tests for the forward SEC filing monitor."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from sec_data.forward_filing_monitor import (
    build_forward_filing_monitor_audit,
    build_forward_filing_monitor_plan,
    calculate_forward_filing_monitor_plan,
    collect_forward_filing_snapshots,
    parse_forward_filing_snapshot,
)
from sec_data.models import SecResponse


def test_monitor_plan_is_forward_only_and_return_free() -> None:
    plan = calculate_forward_filing_monitor_plan(_next_events())

    assert len(plan) == 2
    assert plan["cik"].tolist() == ["0000000001", "0000000002"]
    assert not plan["historical_backfill_allowed"].any()
    assert not plan["actual_values_loaded"].any()
    assert not plan["return_values_loaded"].any()
    assert not plan["selection_uses_return_values"].any()
    assert plan["submissions_url"].str.startswith("https://data.sec.gov/submissions/").all()


def test_snapshot_parser_uses_consensus_and_sec_acceptance_clocks() -> None:
    rows = parse_forward_filing_snapshot(
        _submissions_payload("1"),
        target=_target_record(),
        observed_at="2026-10-25T12:00:00+00:00",
        source_sha256="snapshot-hash",
    )

    assert len(rows) == 2
    by_accession = {row["accession_number"]: row for row in rows}
    future = by_accession["0000000001-26-000002"]
    prior = by_accession["0000000001-26-000001"]
    assert future["period_match_status"] == "exact"
    assert future["accepted_after_consensus"] is True
    assert future["observed_after_acceptance"] is True
    assert future["future_filing_candidate"] is True
    assert prior["accepted_after_consensus"] is False
    assert prior["future_filing_candidate"] is False
    assert all(row["historical_backtest_admission"] is False for row in rows)


def test_monitor_round_trip_preserves_baseline_and_first_seen_candidate(tmp_path: Path) -> None:
    consensus_root = tmp_path / "consensus"
    consensus_root.mkdir()
    _next_events().iloc[[0]].to_parquet(
        consensus_root / "forward_consensus_next_events.parquet", index=False
    )
    (consensus_root / "output_manifest.json").write_text(
        '{"dataset_version":"test"}', encoding="utf-8"
    )
    plan = build_forward_filing_monitor_plan(consensus_root, output_root=tmp_path / "targets")

    class Client:
        def get_submissions(self, cik: str) -> SecResponse:
            assert cik == "0000000001"
            return SecResponse(
                url="https://data.sec.gov/submissions/CIK0000000001.json",
                content=_submissions_payload("1"),
                fetched_at=datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
                status_code=200,
                headers={"content-type": "application/json"},
            )

    raw_root = tmp_path / "raw"
    collection = collect_forward_filing_snapshots(
        pd.read_parquet(plan.output_root / "forward_filing_monitor_targets.parquet"),
        Client(),  # type: ignore[arg-type]
        raw_root=raw_root,
        capture_date=date(2026, 10, 25),
        limit=1,
    )
    audit = build_forward_filing_monitor_audit(
        plan.output_root,
        raw_root=raw_root,
        output_root=tmp_path / "processed",
    )

    assert collection.completed_targets == 1
    assert audit.report["monitor_baseline_gate_passed"] is True
    assert audit.report["future_filing_candidates"] == 1
    assert audit.report["exact_period_candidates"] == 1
    assert audit.report["latest_prior_baseline_filings"] == 1
    assert audit.report["actual_revenue_join_gate_passed"] is False
    candidates = pd.read_parquet(audit.output_root / "forward_filing_candidates.parquet")
    assert candidates.iloc[0]["candidate_status"] == ("ready_for_accession_xbrl_actual_collection")
    baseline = pd.read_parquet(audit.output_root / "forward_filing_latest_prior_baseline.parquet")
    assert baseline.iloc[0]["accession_number"] == "0000000001-26-000001"
    assert baseline.iloc[0]["consensus_to_prior_filing_days"] > 0.0


def _next_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "live_consensus_target_id": "consensus-aaa",
                "cik": "0000000001",
                "ticker": "AAA",
                "target_period_end": pd.Timestamp("2026-09-30", tz="UTC"),
                "captured_at": pd.Timestamp("2026-08-29T12:00:00", tz="UTC"),
                "consensus_revenue_level": 100.0,
                "source_sha256": "source-aaa",
                "innovation_status": "pending_future_sec_actual",
                "future_actual_join_rule": (
                    "same_cik_and_fiscal_period_latest_capture_strictly_before_sec_accepted_at"
                ),
            },
            {
                "live_consensus_target_id": "consensus-bbb",
                "cik": "0000000002",
                "ticker": "BBB",
                "target_period_end": pd.Timestamp("2026-10-31", tz="UTC"),
                "captured_at": pd.Timestamp("2026-08-29T12:05:00", tz="UTC"),
                "consensus_revenue_level": 200.0,
                "source_sha256": "source-bbb",
                "innovation_status": "pending_future_sec_actual",
                "future_actual_join_rule": (
                    "same_cik_and_fiscal_period_latest_capture_strictly_before_sec_accepted_at"
                ),
            },
        ]
    )


def _target_record() -> dict[str, object]:
    return {
        "sec_monitor_target_id": "monitor-aaa",
        "live_consensus_target_id": "consensus-aaa",
        "cik": "0000000001",
        "ticker": "AAA",
        "target_period_end": "2026-09-30T00:00:00+00:00",
        "consensus_captured_at": "2026-08-29T12:00:00+00:00",
        "consensus_revenue_level": 100.0,
    }


def _submissions_payload(cik: str) -> bytes:
    return json.dumps(
        {
            "cik": cik,
            "filings": {
                "recent": {
                    "accessionNumber": [
                        "0000000001-26-000002",
                        "0000000001-26-000001",
                        "0000000001-26-000003",
                    ],
                    "form": ["10-Q", "10-Q", "8-K"],
                    "filingDate": ["2026-10-24", "2026-08-01", "2026-10-24"],
                    "reportDate": ["2026-09-30", "2026-09-30", "2026-09-30"],
                    "acceptanceDateTime": [
                        "2026-10-24T20:00:00.000Z",
                        "2026-08-01T20:00:00.000Z",
                        "2026-10-24T21:00:00.000Z",
                    ],
                    "primaryDocument": ["q3.htm", "q2.htm", "event.htm"],
                }
            },
        }
    ).encode()
