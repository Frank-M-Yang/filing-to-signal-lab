"""Tests for the final longitudinal market-data plan."""

from __future__ import annotations

import pandas as pd

from sec_data.longitudinal_market import calculate_longitudinal_market_plan


def test_market_plan_batches_prices_and_limits_fallback_to_event_windows() -> None:
    events = _events()
    issuers = _issuers()
    prior_targets = _prior_targets()
    prior_audit = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000002"],
            "ticker_events_unavailable_fallback_used": [False, True],
            "historical_symbol_interval_ready": [True, True],
        }
    )

    result = calculate_longitudinal_market_plan(
        events,
        issuers,
        prior_targets,
        prior_audit,
        _sessions(),
    )

    assert len(result.price_targets) == 2
    assert len(result.action_targets) == 3
    assert result.report["fallback_event_window_targets"] == 2
    assert result.report["event_horizon_rows"] == 12
    assert len(result.event_market_map) == 4
    assert result.event_market_map["price_target_id"].nunique() == 2
    fallback = result.action_targets[result.action_targets["fallback_event_window_target"]]
    assert fallback["price_window_start"].tolist() == ["2025-01-06", "2025-02-05"]
    primary = result.event_horizons[result.event_horizons["horizon_sessions"].eq(20)]
    assert primary["exit_session_date"].tolist() == [
        "2025-01-31",
        "2025-01-31",
        "2025-03-04",
        "2025-03-04",
    ]
    assert result.report["selection_uses_market_returns"] is False


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cik, ticker in [("0000000001", "ONE"), ("0000000002", "TWO")]:
        for number, entry in enumerate(["2025-01-06", "2025-02-05"], start=1):
            entry_timestamp = pd.Timestamp(entry)
            exit_date = (entry_timestamp + pd.offsets.BDay(19)).date().isoformat()
            rows.append(
                {
                    "longitudinal_event_id": f"{cik}-{number}",
                    "cik": cik,
                    "accession_number": f"accession-{cik}-{number}",
                    "accepted_at": entry_timestamp - pd.Timedelta(days=4),
                    "form": "10-Q",
                    "selected_ticker": ticker,
                    "selected_exchange_mic": "XNAS",
                    "feature_dense_issuer_eligible": True,
                    "representation_features_complete": True,
                    "formal_entry_session_date": entry,
                    "formal_entry_open_at": f"{entry}T14:30:00+00:00",
                    "formal_exit_session_date": exit_date,
                    "formal_exit_close_at": f"{exit_date}T21:00:00+00:00",
                }
            )
    return pd.DataFrame(rows)


def _issuers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_batch_id": ["batch-1", "batch-2"],
            "cik": ["0000000001", "0000000002"],
            "selected_ticker": ["ONE", "TWO"],
            "selected_exchange_mic": ["XNAS", "XNAS"],
            "selected_provider_security_id": ["figi-1", "figi-2"],
            "selected_provider_security_id_kind": ["share_class_figi", "share_class_figi"],
            "selected_reference_source_sha256": ["source-1", "source-2"],
            "currency_name": ["usd", "usd"],
            "market_window_start": ["2024-12-20", "2024-12-20"],
            "market_window_end": ["2025-03-12", "2025-03-12"],
            "planned_next_batch": [True, True],
            "collection_order": [1, 2],
        }
    )


def _prior_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": ["0000000001", "0000000002"],
            "selected_share_class_figi": ["BBG000000001", "BBG000000002"],
            "selected_composite_figi": ["BBG000000003", "BBG000000004"],
            "selected_reference_source_sha256": ["source-1", "source-2"],
        }
    )


def _sessions() -> pd.DataFrame:
    dates = pd.bdate_range("2024-12-20", "2025-03-31")
    return pd.DataFrame(
        {
            "exchange_mic": ["XNAS"] * len(dates),
            "session_date": dates.strftime("%Y-%m-%d"),
            "regular_open_at": dates.strftime("%Y-%m-%dT14:30:00+00:00"),
            "regular_close_at": dates.strftime("%Y-%m-%dT21:00:00+00:00"),
        }
    )
