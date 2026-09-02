"""Tests for the forward Level/Revision Delta/Innovation ledger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sec_data.forward_innovation_ledger import (
    ForwardInnovationLedgerConfig,
    build_forward_innovation_ledger,
    calculate_forward_innovation_ledger,
)


def test_join_uses_latest_pre_acceptance_consensus_and_prior_date_delta() -> None:
    snapshots = _consensus_snapshots()
    actuals = _actuals()

    tables = calculate_forward_innovation_ledger(
        snapshots,
        actuals,
        ForwardInnovationLedgerConfig(
            minimum_raw_innovation_events=2,
            minimum_revision_events=2,
            minimum_prior_errors_for_sigma=2,
        ),
    )
    first = tables.events.iloc[0]

    assert first["consensus_revenue_level"] == 103.0
    assert first["previous_consensus_revenue"] == 100.0
    assert first["consensus_revision_delta"] == 3.0
    assert first["revenue_innovation"] == 7.0
    assert first["consensus_captured_at"] < first["sec_accepted_at"]
    assert first["consensus_captured_at"] != pd.Timestamp("2026-01-11T12:00:00Z")


def test_standardization_uses_only_strictly_earlier_event_errors() -> None:
    tables = calculate_forward_innovation_ledger(
        _consensus_snapshots(),
        _actuals(),
        ForwardInnovationLedgerConfig(
            minimum_raw_innovation_events=2,
            minimum_revision_events=2,
            minimum_prior_errors_for_sigma=2,
        ),
    )
    events = tables.events.sort_values("sec_accepted_at").reset_index(drop=True)

    assert events.loc[0, "prior_innovation_error_count"] == 0
    assert events.loc[1, "prior_innovation_error_count"] == 1
    assert not events.loc[1, "standardized_innovation_ready"]
    assert events.loc[2, "prior_innovation_error_count"] == 2
    assert events.loc[2, "standardized_innovation_ready"]
    assert pd.notna(events.loc[2, "prior_innovation_error_sigma"])


def test_persisted_empty_ledger_reports_honest_gates(tmp_path: Path) -> None:
    consensus_root = tmp_path / "consensus"
    actual_root = tmp_path / "actual"
    consensus_root.mkdir()
    actual_root.mkdir()
    _consensus_snapshots().to_parquet(
        consensus_root / "forward_consensus_snapshots.parquet", index=False
    )
    pd.DataFrame(columns=_actuals().columns).to_parquet(
        actual_root / "forward_actual_revenue_observations.parquet", index=False
    )
    (consensus_root / "output_manifest.json").write_text("{}", encoding="utf-8")
    (actual_root / "output_manifest.json").write_text("{}", encoding="utf-8")

    result = build_forward_innovation_ledger(
        consensus_root,
        actual_root,
        output_root=tmp_path / "output",
    )

    assert result.report["event_rows"] == 0
    assert result.report["feature_research_gate_passed"] is False
    assert result.report["return_research_gate_passed"] is False


def _consensus_snapshots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = [
        ("AAA", "2026-01-10T20:00:00Z", 100.0, 103.0, 999.0),
        ("BBB", "2026-02-10T20:00:00Z", 200.0, 204.0, 999.0),
        ("CCC", "2026-03-10T20:00:00Z", 300.0, 305.0, 999.0),
    ]
    for ticker, accepted_at, old, new, post in events:
        accepted = pd.Timestamp(accepted_at)
        period = accepted.normalize() - pd.Timedelta(days=10)
        for captured_at, value, suffix in [
            (accepted - pd.Timedelta(days=20), old, "old"),
            (accepted - pd.Timedelta(days=5), new, "new"),
            (accepted + pd.Timedelta(days=1), post, "post"),
        ]:
            rows.append(
                {
                    "ticker": ticker,
                    "target_period_end": period,
                    "captured_at": captured_at,
                    "consensus_revenue_level": value,
                    "source_sha256": f"{ticker}-{suffix}",
                    "historical_backtest_admission": False,
                }
            )
    return pd.DataFrame(rows)


def _actuals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = [
        ("AAA", "2026-01-10T20:00:00Z", 110.0),
        ("BBB", "2026-02-10T20:00:00Z", 220.0),
        ("CCC", "2026-03-10T20:00:00Z", 320.0),
    ]
    for number, (ticker, accepted_at, actual) in enumerate(events, start=1):
        accepted = pd.Timestamp(accepted_at)
        rows.append(
            {
                "forward_actual_target_id": f"actual-{ticker}",
                "cik": str(number).zfill(10),
                "ticker": ticker,
                "target_period_end": accepted.normalize() - pd.Timedelta(days=10),
                "accession_number": f"{number:010d}-26-000001",
                "accepted_at": accepted,
                "actual_available_at": accepted,
                "actual_status": "direct_quarterly_fact_ready",
                "actual_revenue": actual,
                "source_snapshot_sha256": f"actual-{ticker}",
                "historical_backfill_admission": False,
                "return_values_loaded": False,
            }
        )
    return pd.DataFrame(rows)
