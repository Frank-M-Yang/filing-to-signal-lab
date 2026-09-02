from __future__ import annotations

import pandas as pd

from prospective_market_innovation_readiness import (
    calculate_prospective_market_innovation_readiness,
)


def test_registered_period_continues_after_period_end_until_sec_acceptance() -> None:
    registry = _registry()
    observations = _observations()
    filings = _empty_filings()

    tables = calculate_prospective_market_innovation_readiness(registry, observations, filings)

    assert tables.report["registered_consensus_events"] == 30
    assert tables.report["capture_dates"] == 2
    assert tables.report["revision_ready_events"] == 30
    assert tables.report["period_end_is_collection_stop"] is False
    assert tables.report["consensus_collection_stop"] == (
        "sec_acceptance_timestamp_per_registered_event"
    )
    assert tables.report["revision_gate_passed"] is True
    assert tables.report["actual_values_loaded"] is False
    assert tables.report["return_values_loaded"] is False
    first = tables.revision_state.iloc[0]
    assert first["consensus_capture_dates"] == 2
    assert first["consensus_revision_pct"] == 0.01


def test_snapshots_at_or_after_sec_acceptance_are_excluded() -> None:
    registry = _registry()
    observations = _observations()
    event = registry.iloc[0]
    filings = _empty_filings()
    filings.loc[0] = {
        "prospective_event_id": event["prospective_event_id"],
        "cik": event["cik"],
        "ticker": event["ticker"],
        "target_period_end": event["target_period_end"],
        "accession_number": "0000000001-26-000001",
        "form": "10-Q",
        "accepted_at": pd.Timestamp("2026-10-01 11:00:00", tz="UTC"),
        "observed_at": pd.Timestamp("2026-10-01 11:01:00", tz="UTC"),
        "future_filing_candidate": True,
        "amendment_form": False,
        "source_sha256": "sec-sha",
    }

    tables = calculate_prospective_market_innovation_readiness(registry, observations, filings)

    selected = tables.consensus_snapshots.loc[
        tables.consensus_snapshots["prospective_event_id"].eq(event["prospective_event_id"])
    ]
    assert len(selected) == 1
    assert selected["captured_at"].max() < pd.Timestamp("2026-10-01 11:00:00", tz="UTC")
    event_state = tables.event_readiness.loc[
        tables.event_readiness["prospective_event_id"].eq(event["prospective_event_id"])
    ].iloc[0]
    assert event_state["future_filing_observed"]
    assert event_state["readiness_status"] == "future_filing_observed_actual_not_loaded"


def _registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(30):
        issuer = index % 15
        period = pd.Timestamp("2026-09-30", tz="UTC") + pd.DateOffset(months=3 * (index // 15))
        rows.append(
            {
                "prospective_event_id": f"event-{index}",
                "live_consensus_target_id": f"target-{issuer}",
                "cik": f"{issuer + 1:010d}",
                "ticker": f"T{issuer:02d}",
                "target_period_end": period,
                "captured_at": pd.Timestamp("2026-08-29 10:00:00", tz="UTC"),
                "consensus_revenue_level": float(1_000 + index),
                "actual_value_loaded_at_registration": False,
                "return_value_loaded_at_registration": False,
                "replacement_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in _registry().itertuples(index=False):
        for day, multiplier in [("2026-08-29", 1.0), ("2026-10-01", 1.01)]:
            captured = pd.Timestamp(f"{day} 12:00:00", tz="UTC")
            average = float(event.consensus_revenue_level) * multiplier
            rows.append(
                {
                    "ticker": event.ticker,
                    "target_period_end": event.target_period_end,
                    "captured_at": captured,
                    "availability_clock": "collector_fetched_at",
                    "forward_only_admission": True,
                    "historical_backtest_admission": False,
                    "revenue_estimate_average": average,
                    "revenue_estimate_high": average * 1.05,
                    "revenue_estimate_low": average * 0.95,
                    "revenue_estimate_analyst_count": 10.0,
                    "source_sha256": f"sha-{event.prospective_event_id}-{day}",
                }
            )
    return pd.DataFrame(rows)


def _empty_filings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prospective_event_id",
            "cik",
            "ticker",
            "target_period_end",
            "accession_number",
            "form",
            "accepted_at",
            "observed_at",
            "future_filing_candidate",
            "amendment_form",
            "source_sha256",
        ]
    )
