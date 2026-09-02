"""Tests for the revenue-guidance return-readiness audit."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_return_audit import (
    RevenueGuidanceReturnAuditConfig,
    calculate_revenue_guidance_return_audit,
)


def test_return_statistics_are_withheld_when_breadth_gate_fails() -> None:
    forecasts, authorization, ledger = _fixtures(events=6, issuers=3)
    result = calculate_revenue_guidance_return_audit(
        forecasts,
        authorization,
        ledger,
        forecast_gate_passed=True,
        config=RevenueGuidanceReturnAuditConfig(
            minimum_events=10,
            minimum_issuers=3,
            minimum_positive_innovations=3,
            minimum_negative_innovations=3,
            bootstrap_repetitions=1_000,
        ),
    )

    assert result.report["return_diagnostic_performed"] is False
    assert result.report["return_inference_status"] == "withheld_until_breadth_gates_pass"
    assert result.report["return_descriptives_performed"] is True
    assert len(result.descriptives) == 1
    assert not bool(result.descriptives.iloc[0]["inference_allowed"])
    assert result.descriptives.iloc[0]["events"] == 6
    assert result.statistics.empty


def test_return_statistics_use_only_authorized_complete_cases() -> None:
    forecasts, authorization, ledger = _fixtures(events=12, issuers=3)
    authorization.loc[0, "return_outcome_read_authorized"] = False
    result = calculate_revenue_guidance_return_audit(
        forecasts,
        authorization,
        ledger,
        forecast_gate_passed=True,
        config=RevenueGuidanceReturnAuditConfig(
            minimum_events=10,
            minimum_issuers=3,
            minimum_positive_innovations=3,
            minimum_negative_innovations=3,
            bootstrap_repetitions=1_000,
            bootstrap_seed=7,
        ),
    )

    assert len(result.complete_cases) == 11
    assert result.report["return_diagnostic_performed"] is True
    assert bool(result.descriptives.iloc[0]["inference_allowed"])
    assert len(result.statistics) == 1
    assert result.complete_cases["entry_price_type"].eq("unadjusted_open").all()


def _fixtures(events: int, issuers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows = []
    authorization_rows = []
    ledger_rows = []
    for index in range(events):
        event_id = f"event-{index}"
        accepted = pd.Timestamp("2025-01-02 21:00:00", tz="UTC") + pd.Timedelta(days=index)
        positive = index % 2 == 0
        forecast = 100.0
        actual = 110.0 if positive else 90.0
        forecast_rows.append(
            {
                "validation_event_id": event_id,
                "cik": f"{index % issuers + 1:010d}",
                "guidance_accepted_at": accepted - pd.Timedelta(days=80),
                "target_actual_available_at": accepted,
                "target_revenue": actual,
                "management_guidance_forecast": forecast,
                "selection_uses_return_values": False,
            }
        )
        authorization_rows.append(
            {
                "validation_event_id": event_id,
                "return_outcome_read_authorized": True,
                "selection_uses_return_values": False,
                "authorization_uses_return_values": False,
            }
        )
        ledger_rows.append(
            {
                "validation_event_id": event_id,
                "accepted_at": accepted,
                "entry_at": accepted + pd.Timedelta(days=2),
                "entry_price_type": "unadjusted_open",
                "exit_at": accepted + pd.Timedelta(days=30),
                "exit_price_type": "unadjusted_close",
                "holding_sessions": 20,
                "issuer_total_return": 0.05 if positive else -0.05,
                "market_total_return": 0.0,
                "market_excess_return_arithmetic": 0.05 if positive else -0.05,
                "market_excess_return_ready": True,
                "return_outcome_read_authorized": True,
                "selection_uses_return_values": False,
                "returns_computed": True,
            }
        )
    return pd.DataFrame(forecast_rows), pd.DataFrame(authorization_rows), pd.DataFrame(ledger_rows)
