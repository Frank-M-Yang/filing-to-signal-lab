from __future__ import annotations

import pandas as pd

from prospective_consensus_quality import (
    CLEAN,
    INSUFFICIENT_HISTORY,
    QUARANTINED,
    REVIEW_REQUIRED,
    calculate_prospective_consensus_quality,
)


def test_structural_changes_are_separated_from_clean_numeric_revisions() -> None:
    events = _events(["CLEAN", "REVIEW", "QUAR", "SHORT"])
    snapshots = pd.concat(
        [
            _pair("CLEAN", analysts=(20, 20), occurrence=(2, 2), average=(100.0, 101.0)),
            _pair("REVIEW", analysts=(12, 14), occurrence=(2, 2), average=(100.0, 100.2)),
            _pair("QUAR", analysts=(28, 27), occurrence=(2, 3), average=(100.0, 99.3)),
            _pair("SHORT", analysts=(10, 10), occurrence=(2, 2), average=(100.0, 100.0)).iloc[:1],
        ],
        ignore_index=True,
    )

    tables = calculate_prospective_consensus_quality(events, snapshots)
    quality = tables.event_quality.set_index("ticker")

    assert quality.loc["CLEAN", "quality_status"] == CLEAN
    assert quality.loc["REVIEW", "quality_status"] == REVIEW_REQUIRED
    assert quality.loc["QUAR", "quality_status"] == QUARANTINED
    assert quality.loc["SHORT", "quality_status"] == INSUFFICIENT_HISTORY
    assert quality.loc["CLEAN", "downstream_admission"]
    assert not quality.loc["REVIEW", "downstream_admission"]
    assert "analyst_coverage_changed" in quality.loc["REVIEW", "quality_reasons"]
    assert "provider_source_occurrence_changed" in quality.loc["QUAR", "quality_reasons"]
    assert tables.report["downstream_admitted_events"] == 1


def test_large_coverage_and_range_change_is_quarantined() -> None:
    events = _events(["AAA"])
    snapshots = _pair(
        "AAA",
        analysts=(40, 20),
        occurrence=(2, 2),
        average=(100.0, 100.5),
        low=(80.0, 98.0),
        high=(120.0, 103.0),
    )

    tables = calculate_prospective_consensus_quality(events, snapshots)
    result = tables.event_quality.iloc[0]

    assert result["quality_status"] == QUARANTINED
    assert "large_analyst_coverage_change" in result["quality_reasons"]
    assert "range_shock_with_coverage_change" in result["quality_reasons"]


def test_stale_preacceptance_snapshot_is_quarantined() -> None:
    events = _events(["AAA"])
    events.loc[0, "future_filing_observed"] = True
    events["accepted_at"] = pd.Series(
        [pd.Timestamp("2026-05-20T20:00:00Z")], dtype="datetime64[ns, UTC]"
    )
    snapshots = _pair("AAA", analysts=(10, 10), occurrence=(2, 2), average=(100.0, 100.0))

    tables = calculate_prospective_consensus_quality(events, snapshots)
    result = tables.event_quality.iloc[0]

    assert result["stale_at_filing"]
    assert result["quality_status"] == QUARANTINED
    assert "stale_preacceptance_consensus" in result["quality_reasons"]


def _events(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for order, ticker in enumerate(tickers, start=1):
        rows.append(
            {
                "prospective_event_id": f"event-{ticker}",
                "registration_order": order,
                "cik": str(order).zfill(10),
                "ticker": ticker,
                "target_period_end": pd.Timestamp("2026-06-30T00:00:00Z"),
                "accepted_at": pd.NaT,
                "future_filing_observed": False,
            }
        )
    return pd.DataFrame(rows)


def _pair(
    ticker: str,
    *,
    analysts: tuple[int, int],
    occurrence: tuple[int, int],
    average: tuple[float, float],
    low: tuple[float, float] = (90.0, 90.0),
    high: tuple[float, float] = (110.0, 110.0),
) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(("2026-05-01", "2026-05-02")):
        rows.append(
            {
                "prospective_event_id": f"event-{ticker}",
                "ticker": ticker,
                "target_period_end": pd.Timestamp("2026-06-30T00:00:00Z"),
                "horizon": "fiscal quarter",
                "revenue_estimate_average": average[index],
                "revenue_estimate_high": high[index],
                "revenue_estimate_low": low[index],
                "revenue_estimate_analyst_count": analysts[index],
                "captured_at": pd.Timestamp(f"{day}T12:00:00Z"),
                "capture_date": pd.Timestamp(f"{day}T00:00:00Z"),
                "source_occurrence": occurrence[index],
                "source_sha256": f"sha-{ticker}-{index}",
                "historical_backtest_admission": False,
            }
        )
    return pd.DataFrame(rows)
