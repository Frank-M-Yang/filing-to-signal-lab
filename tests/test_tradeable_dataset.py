"""Point-in-time contracts for the first model-ready event feature store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from tradeable_dataset import (
    COMBINED_FEATURE_COLUMNS,
    calculate_tradeable_dataset,
    validate_tradeable_dataset,
)


def test_tradeable_dataset_uses_only_prior_market_sessions() -> None:
    ledger, bars, benchmark, actions = _fixture_inputs()
    first = calculate_tradeable_dataset(ledger, bars, benchmark, actions, {"event-1"})

    future = bars.copy()
    future.loc[future["session_date"].ge("2024-03-09"), "close_unadjusted"] *= 100.0
    second = calculate_tradeable_dataset(ledger, future, benchmark, actions, {"event-1"})

    columns = ["longitudinal_event_id", *COMBINED_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        first.event_dataset.loc[:, columns],
        second.event_dataset.loc[:, columns],
    )
    assert first.report["events"] == 2
    assert first.report["market_feature_complete_events"] == 2
    assert first.report["legacy_observed_events"] == 1
    assert first.report["unobserved_evaluation_candidates"] == 1
    roles = first.event_dataset.set_index("longitudinal_event_id")["evidence_role"]
    assert roles["event-1"] == "legacy_observed_training_only"
    assert roles["event-2"] == "unobserved_evaluation_candidate"


def test_tradeable_dataset_rejects_future_market_clock() -> None:
    ledger, bars, benchmark, actions = _fixture_inputs()
    dataset = calculate_tradeable_dataset(ledger, bars, benchmark, actions, set()).event_dataset
    dataset.loc[0, "market_feature_available_at"] = dataset.loc[0, "execution_entry_at"]

    with pytest.raises(FilingCatalogError, match="market feature"):
        validate_tradeable_dataset(dataset)


def test_preregistered_accession_roles_control_evaluation() -> None:
    ledger, bars, benchmark, actions = _fixture_inputs()
    roles = pd.DataFrame(
        {
            "accession_number": ["accession-1", "accession-2"],
            "evidence_role": [
                "legacy_issuer_training_only",
                "unobserved_validation_candidate",
            ],
        }
    )
    result = calculate_tradeable_dataset(
        ledger,
        bars,
        benchmark,
        actions,
        set(),
        preregistered_event_roles=roles,
    )
    events = result.event_dataset.set_index("accession_number")

    assert bool(events.loc["accession-1", "prior_outcome_observed"])
    assert not bool(events.loc["accession-1", "model_evaluation_candidate"])
    assert not bool(events.loc["accession-2", "prior_outcome_observed"])
    assert bool(events.loc["accession-2", "model_evaluation_candidate"])
    assert result.report["external_event_role_registry_used"] is True


def test_unregistered_return_event_is_rejected() -> None:
    ledger, bars, benchmark, actions = _fixture_inputs()
    incomplete_registry = pd.DataFrame(
        {
            "accession_number": ["accession-1"],
            "evidence_role": ["legacy_issuer_training_only"],
        }
    )

    with pytest.raises(FilingCatalogError, match="not frozen before outcome"):
        calculate_tradeable_dataset(
            ledger,
            bars,
            benchmark,
            actions,
            set(),
            preregistered_event_roles=incomplete_registry,
        )


def test_split_adjustment_avoids_false_momentum_jump() -> None:
    ledger, bars, benchmark, actions = _fixture_inputs()
    baseline = calculate_tradeable_dataset(ledger, bars, benchmark, actions, set())
    baseline_event = baseline.event_dataset.set_index("longitudinal_event_id").loc["event-1"]
    target = bars["price_target_id"].eq("price-1") & bars["session_date"].ge("2024-02-15")
    bars.loc[target, "close_unadjusted"] /= 2.0
    actions = pd.DataFrame(
        [
            {
                "price_target_id": "price-1",
                "effective_at": "2024-02-15T14:30:00Z",
                "action_type": "split",
                "split_from": 1.0,
                "split_to": 2.0,
            }
        ]
    )
    result = calculate_tradeable_dataset(ledger, bars, benchmark, actions, set())
    event = result.event_dataset.set_index("longitudinal_event_id").loc["event-1"]

    assert event["issuer_momentum_20d"] == pytest.approx(
        baseline_event["issuer_momentum_20d"], rel=1e-10
    )


def _fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_dates = pd.bdate_range("2024-01-15", "2024-04-15")
    bar_rows: list[dict[str, object]] = []
    for target_index, target_id in enumerate(("price-1", "price-2")):
        for position, session_date in enumerate(session_dates):
            bar_rows.append(
                {
                    "price_target_id": target_id,
                    "session_date": session_date,
                    "close_unadjusted": 100.0 + position + target_index,
                    "volume": 1_000.0 + position * 10.0,
                }
            )
    bars = pd.DataFrame(bar_rows)
    benchmark = pd.DataFrame(
        {
            "session_date": session_dates,
            "close_unadjusted": 400.0 + np.arange(len(session_dates), dtype=float) * 0.5,
        }
    )
    actions = pd.DataFrame(
        columns=["price_target_id", "effective_at", "action_type", "split_from", "split_to"]
    )
    event_rows: list[dict[str, object]] = []
    for event_index, (event_id, target_id, accepted_at, entry_at) in enumerate(
        (
            (
                "event-1",
                "price-1",
                "2024-03-01T21:30:00Z",
                "2024-03-05T14:30:00Z",
            ),
            (
                "event-2",
                "price-2",
                "2024-03-08T21:30:00Z",
                "2024-03-12T13:30:00Z",
            ),
        )
    ):
        exits = (
            ("2024-03-12T20:00:00Z", "2024-04-02T20:00:00Z")
            if event_index == 0
            else ("2024-03-19T20:00:00Z", "2024-04-09T20:00:00Z")
        )
        for horizon, exit_at, excess_return in (
            (5, exits[0], 0.01 + event_index * 0.01),
            (20, exits[1], 0.03 + event_index * 0.01),
        ):
            event_rows.append(
                {
                    "longitudinal_event_id": event_id,
                    "cik": f"{event_index + 1:010d}",
                    "accession_number": f"accession-{event_index + 1}",
                    "accepted_at": accepted_at,
                    "form": "10-Q" if event_index == 0 else "10-K",
                    "horizon_sessions": horizon,
                    "price_target_id": target_id,
                    "ticker": f"T{event_index + 1}",
                    "security_id": f"security-{event_index + 1}",
                    "entry_at": entry_at,
                    "exit_at": exit_at,
                    "market_excess_return_arithmetic": excess_return,
                    "market_excess_return_ready": True,
                    "later_delisted_after_formal_exit": event_index == 1,
                    "analysis_sample": True,
                    "gm_level": 0.40 + event_index * 0.01,
                    "gm_delta_yoy": 0.01 + event_index * 0.005,
                    "gm_innovation_raw": 0.02 + event_index * 0.005,
                    "gm_innovation_std": 0.5 + event_index * 0.2,
                }
            )
    return pd.DataFrame(event_rows), bars, benchmark, actions
