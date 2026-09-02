from __future__ import annotations

import pandas as pd
import pytest

from shadow_market_evidence import (
    ShadowMarketEvidenceError,
    calculate_shadow_market_evidence_plan,
)


def test_acceptance_creates_five_future_evidence_queries() -> None:
    states = pd.DataFrame(
        [
            _state("accepted", 1, "2026-08-31T20:00:00Z", "2026-08-31T20:01:00Z"),
            _state("waiting", 2, None, None),
        ]
    )

    tables = calculate_shadow_market_evidence_plan(
        states,
        _policy_events(["accepted", "waiting"]),
        _core_assets(),
        "2026-08-31T21:00:00Z",
    )

    assert tables.report["security_candidates_ready"] == 2
    assert tables.report["accepted_events"] == 1
    assert tables.report["market_queries_planned"] == 5
    assert set(tables.queries["query_type"]) == {
        "point_in_time_identity",
        "preacceptance_cost_bars",
        "execution_window_bars",
        "split_actions",
        "cash_dividend_actions",
    }
    assert (
        pd.to_datetime(tables.queries["query_not_before"], utc=True)
        >= pd.to_datetime(tables.queries["sec_accepted_at"], utc=True)
    ).all()
    statuses = tables.events.set_index("prospective_event_id")["market_evidence_status"]
    assert statuses["accepted"] == "queries_planned_evidence_not_collected"
    assert statuses["waiting"] == "awaiting_sec_acceptance"
    assert not tables.events["order_evidence_ready"].any()
    assert tables.gates["passed"].all()


def test_current_ticker_is_only_an_identity_candidate() -> None:
    tables = calculate_shadow_market_evidence_plan(
        pd.DataFrame([_state("waiting", 1, None, None)]),
        _policy_events(["waiting"]),
        _core_assets().iloc[[0]],
        "2026-08-31T21:00:00Z",
    )

    event = tables.events.iloc[0]
    assert event["security_candidate_ready"]
    assert not event["point_in_time_identity_verified"]
    assert "sec_acceptance_pending" in event["market_evidence_blockers"]


def test_market_plan_rejects_return_values() -> None:
    states = pd.DataFrame([_state("waiting", 1, None, None)])
    states.loc[0, "return_values_loaded"] = True

    with pytest.raises(ShadowMarketEvidenceError):
        calculate_shadow_market_evidence_plan(
            states,
            _policy_events(["waiting"]),
            _core_assets().iloc[[0]],
            "2026-08-31T21:00:00Z",
        )


def _state(
    event_id: str,
    order: int,
    accepted_at: str | None,
    observed_at: str | None,
) -> dict[str, object]:
    ticker = "AAA" if order == 1 else "BBB"
    return {
        "prospective_event_id": event_id,
        "registration_order": order,
        "cik": str(order).zfill(10),
        "ticker": ticker,
        "target_period_end": "2026-06-30",
        "current_state": "consensus_revision_ready",
        "sec_accepted_at": accepted_at,
        "filing_first_observed_at": observed_at,
        "innovation_ready": False,
        "consensus_quality_status": "clean",
        "downstream_admission": True,
        "return_values_loaded": False,
        "live_trading_authorized": False,
    }


def _policy_events(event_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospective_event_id": event_ids,
            "shadow_policy_admitted": True,
            "return_values_loaded": False,
            "live_trading_authorized": False,
        }
    )


def _core_assets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sec_cik": ["1", "2"],
            "sec_ticker": ["AAA", "BBB"],
            "stock_quote": ["NASDAQ", "NYSE"],
            "mapping_status": ["accepted", "accepted"],
            "core_asset_version": ["v1", "v1"],
            "core_asset_frozen_at": ["2026-08-01", "2026-08-01"],
        }
    )
