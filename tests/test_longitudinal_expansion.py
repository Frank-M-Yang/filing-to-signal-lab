"""Tests for the pre-return longitudinal expansion design."""

from __future__ import annotations

import pandas as pd

from sec_data.longitudinal_expansion import (
    LongitudinalExpansionConfig,
    calculate_longitudinal_expansion,
)


def test_feature_dense_eligibility_uses_only_pre_cutoff_history() -> None:
    readiness = _readiness()
    actions = _action_targets()
    ticker_changes = pd.DataFrame(
        {
            "action_target_id": ["action-1", "action-2"],
            "event_date": ["2020-01-01", "2020-01-01"],
            "new_ticker": ["ONE", "TWO"],
        }
    )
    history = pd.DataFrame(
        [
            *[_history_row("0000000001", f"one-prior-{index}", index) for index in range(8)],
            *[_history_row("0000000002", f"two-prior-{index}", index) for index in range(7)],
            _history_row("0000000001", "one-current", 20, current=True),
            _history_row("0000000002", "two-current", 20, current=True),
        ]
    )
    catalog = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000001", "0000000002"],
            "accession_number": ["one-current", "one-missing", "two-current"],
            "form": ["10-Q", "10-Q", "10-Q"],
            "report_date": ["2024-12-31", "2025-03-31", "2024-12-31"],
            "accepted_at": [
                "2025-01-02T21:00:00Z",
                "2025-01-03T21:00:00Z",
                "2025-01-02T21:00:00Z",
            ],
        }
    )
    config = LongitudinalExpansionConfig(
        feature_eligibility_cutoff="2025-01-01T00:00:00+00:00",
        minimum_prior_level_events=8,
        minimum_prior_delta_events=4,
        minimum_prior_standardized_innovations=4,
    )

    result = calculate_longitudinal_expansion(
        readiness,
        catalog,
        history,
        _sessions(),
        actions,
        ticker_changes,
        config,
    )

    eligibility = result.issuer_eligibility.set_index("cik")
    assert bool(eligibility.at["0000000001", "feature_dense_issuer_eligible"])
    assert not bool(eligibility.at["0000000002", "feature_dense_issuer_eligible"])
    assert eligibility.at["0000000002", "prior_level_events"] == 7
    assert len(result.representative_events) == 3
    assert len(result.feature_dense_events) == 2
    assert result.feature_dense_events["feature_status"].tolist() == [
        "complete_case_ready",
        "no_quarterly_level_event",
    ]
    assert result.report["selection_uses_event_returns"] is False
    assert "market_excess_return_arithmetic" not in result.feature_dense_events


def test_formal_window_uses_second_future_open_and_twenty_sessions() -> None:
    result = calculate_longitudinal_expansion(
        _readiness().iloc[[0]],
        pd.DataFrame(
            {
                "cik": ["0000000001"],
                "accession_number": ["one-current"],
                "form": ["10-Q"],
                "report_date": ["2024-12-31"],
                "accepted_at": ["2025-01-02T21:00:00Z"],
            }
        ),
        pd.DataFrame(
            [
                *[_history_row("0000000001", f"one-prior-{index}", index) for index in range(8)],
                _history_row("0000000001", "one-current", 20, current=True),
            ]
        ),
        _sessions(),
        _action_targets().iloc[[0]],
        pd.DataFrame(
            {
                "action_target_id": ["action-1"],
                "event_date": ["2020-01-01"],
                "new_ticker": ["ONE"],
            }
        ),
        LongitudinalExpansionConfig(feature_eligibility_cutoff="2025-01-01T00:00:00+00:00"),
    )

    event = result.feature_dense_events.iloc[0]
    assert event["formal_entry_session_date"] == "2025-01-06"
    assert event["formal_exit_session_date"] == "2025-01-31"
    target = result.market_collection_targets.iloc[0]
    assert bool(target["historical_ticker_query_ready"])
    assert not bool(target["identity_discovery_required"])
    assert target["daily_bar_queries_planned"] == 1


def _readiness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": ["target-1", "target-2"],
            "cik": ["0000000001", "0000000002"],
            "sampling_hash": ["hash-1", "hash-2"],
            "stage1_inclusion_probability": [0.1, 0.1],
            "stage1_sampling_weight": [10.0, 10.0],
            "sic": [2000, 3000],
            "sic_description": ["Industry one", "Industry two"],
            "return_study_eligible": [True, True],
        }
    )


def _action_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_target_id": ["action-1", "action-2"],
            "target_id": ["target-1", "target-2"],
            "cik": ["0000000001", "0000000002"],
            "selected_ticker": ["ONE", "TWO"],
            "selected_exchange_mic": ["XNAS", "XNAS"],
            "selected_provider_security_id": ["security-1", "security-2"],
            "selected_provider_security_id_kind": ["share_class_figi", "share_class_figi"],
            "selected_reference_source_sha256": ["source-1", "source-2"],
            "currency_name": ["usd", "usd"],
        }
    )


def _history_row(
    company_id: str,
    accession: str,
    index: int,
    *,
    current: bool = False,
) -> dict[str, object]:
    available_at = (
        pd.Timestamp("2025-01-02T21:00:00Z")
        if current
        else pd.Timestamp("2023-01-01T21:00:00Z") + pd.Timedelta(days=30 * index)
    )
    delta_ready = index >= 4
    innovation_ready = index >= 4
    return {
        "company_id": company_id,
        "filing_accession": accession,
        "actual_available_at": available_at,
        "gm_level": 0.40,
        "gm_delta_yoy": 0.01 if delta_ready else float("nan"),
        "gm_innovation_raw": 0.02 if innovation_ready else float("nan"),
        "gm_innovation_std": 1.0 if innovation_ready else float("nan"),
        "analysis_ready_level": True,
        "analysis_ready_delta": delta_ready,
        "analysis_ready_innovation_raw": innovation_ready,
        "analysis_ready_innovation_std": innovation_ready,
        "forecast_feature_cutoff": available_at - pd.Timedelta(days=1),
        "forecast_training_end": available_at - pd.Timedelta(days=2),
        "forecast_training_observations": 100,
        "forecast_error_history_end": available_at - pd.Timedelta(days=3),
        "forecast_sigma_observations": 50,
    }


def _sessions() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-03", "2025-02-28")
    return pd.DataFrame(
        {
            "exchange_mic": ["XNAS"] * len(dates),
            "session_date": dates.strftime("%Y-%m-%d"),
            "regular_open_at": [
                timestamp.strftime("%Y-%m-%dT14:30:00+00:00") for timestamp in dates
            ],
            "regular_close_at": [
                timestamp.strftime("%Y-%m-%dT21:00:00+00:00") for timestamp in dates
            ],
        }
    )
