"""Tests for the authorized formal event-return ledger."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.validation_event_returns import (
    ValidationEventReturnError,
    calculate_validation_event_return_tables,
)


def test_calculate_authorized_cash_retained_excess_return() -> None:
    tables = _tables()
    ledger, cashflows, evidence = calculate_validation_event_return_tables(
        *tables,
        action_manifest_sha256="a" * 64,
        benchmark_manifest_sha256="b" * 64,
    )
    row = ledger.iloc[0]
    assert row["issuer_total_return"] == pytest.approx(0.11)
    assert row["market_total_return"] == pytest.approx(0.025)
    assert row["market_excess_return_arithmetic"] == pytest.approx(0.085)
    assert row["market_relative_wealth_return"] == pytest.approx(1.11 / 1.025 - 1.0)
    assert row["return_outcome_read_authorized"]
    assert len(cashflows) == 6
    assert set(evidence["evidence_type"]) >= {
        "issuer_entry_price",
        "issuer_exit_price",
        "issuer_action_bundle",
        "market_action_bundle",
    }


def test_rejects_incomplete_action_evidence() -> None:
    tables = list(_tables())
    tables[5].loc[:, "corporate_actions_ready"] = False
    with pytest.raises(ValidationEventReturnError, match="action evidence"):
        calculate_validation_event_return_tables(
            *tables,
            action_manifest_sha256="a" * 64,
            benchmark_manifest_sha256="b" * 64,
        )


def test_calculate_accepts_frozen_secondary_issuer_endpoints() -> None:
    tables = list(_tables())
    tables[2].loc[:, "reconciled_price_source_tier"] = "secondary_yahoo"

    ledger, _, evidence = calculate_validation_event_return_tables(
        *tables,
        action_manifest_sha256="a" * 64,
        benchmark_manifest_sha256="b" * 64,
    )

    assert ledger.loc[0, "issuer_total_return"] == pytest.approx(0.11)
    issuer_hashes = evidence.loc[
        evidence["evidence_type"].isin(["issuer_entry_price", "issuer_exit_price"]),
        "source_sha256",
    ]
    assert set(issuer_hashes) == {"i" * 64, "j" * 64}


def _tables() -> tuple[pd.DataFrame, ...]:
    authorized = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "return_outcome_read_authorized": [True],
            "authorization_uses_return_values": [False],
            "returns_computed": [False],
        }
    )
    components = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "market_event_id": ["m1"],
            "validation_event_id": ["v1"],
            "cik": ["1"],
            "ticker": ["ABC"],
            "exchange_mic": ["XNAS"],
            "accepted_at": ["2025-01-07T21:00:00+00:00"],
            "form": ["10-Q"],
            "accession_number": ["0001"],
            "formal_entry_session_date": ["2025-01-09"],
            "formal_entry_open_at": ["2025-01-09T14:30:00+00:00"],
            "formal_entry_price_type": ["unadjusted_open"],
            "formal_exit_session_date": ["2025-02-06"],
            "formal_exit_close_at": ["2025-02-06T21:00:00+00:00"],
            "formal_exit_price_type": ["unadjusted_close"],
        }
    )
    prices = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "price_target_id": ["p1"],
            "reconciled_formal_price_ready": [True],
            "reconciled_price_source_tier": ["primary_massive"],
            "returns_computed": [False],
        }
    )
    outcomes = pd.DataFrame(
        {
            "security_event_component_id": ["e1"],
            "resolved_market_outcome_ready": [True],
            "resolved_outcome_status": ["ready"],
            "delisted_on_or_before_formal_exit": [False],
            "later_delisted_after_formal_exit": [False],
            "terminal_value_required": [False],
            "source_snapshot_sha256": ["o" * 64],
            "returns_computed": [False],
        }
    )
    bars = pd.DataFrame(
        {
            "price_target_id": ["p1", "p1"],
            "session_date": ["2025-01-09", "2025-02-06"],
            "open_unadjusted": [100.0, 108.0],
            "close_unadjusted": [101.0, 110.0],
            "source_sha256": ["i" * 64, "j" * 64],
        }
    )
    action_audit = pd.DataFrame(
        {
            "price_target_id": ["p1"],
            "action_target_id": ["a1"],
            "historical_symbol_interval_ready": [True],
            "corporate_actions_ready": [True],
        }
    )
    issuer_actions = pd.DataFrame(
        {
            "price_target_id": ["p1"],
            "provider_event_id": ["d1"],
            "effective_at": ["2025-01-20T14:30:00+00:00"],
            "action_type": ["cash_dividend"],
            "cash_amount": [1.0],
            "currency": ["USD"],
            "split_from": [float("nan")],
            "split_to": [float("nan")],
            "source_snapshot_sha256": ["d" * 64],
        }
    )
    benchmark = pd.DataFrame(
        {
            "target_id": ["e1", "e1"],
            "required_session_date": ["2025-01-09", "2025-02-06"],
            "open_unadjusted": [400.0, 408.0],
            "close_unadjusted": [402.0, 408.0],
            "source_sha256": ["s" * 64, "t" * 64],
            "fallback_source_sha256": [None, None],
            "reconciled_benchmark_price_ready": [True, True],
            "reconciled_price_source_tier": ["primary_massive", "primary_massive"],
            "returns_computed": [False, False],
        }
    )
    benchmark_actions = pd.DataFrame(
        {
            "provider_event_id": ["sd1"],
            "effective_at": ["2025-01-24T14:30:00+00:00"],
            "action_type": ["cash_dividend"],
            "cash_amount": [2.0],
            "currency": ["USD"],
            "split_from": [float("nan")],
            "split_to": [float("nan")],
            "source_snapshot_sha256": ["q" * 64],
        }
    )
    return (
        authorized,
        components,
        prices,
        outcomes,
        bars,
        action_audit,
        issuer_actions,
        benchmark,
        benchmark_actions,
    )
