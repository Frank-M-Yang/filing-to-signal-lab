"""Tests for the point-in-time price research contract."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.price_research import (
    point_in_time_price_contract,
    provider_capability_matrix,
    validate_security_mapping_frame,
)


def test_provider_matrix_does_not_claim_a_ready_low_cost_source() -> None:
    matrix = provider_capability_matrix()

    assert not matrix["contract_ready_without_external_data"].any()
    assert (
        matrix.loc[
            matrix["provider"].eq("CRSP US Stock Databases"),
            "delisting_return_or_terminal_value",
        ].item()
        == "verified_delret"
    )
    assert (
        matrix.loc[
            matrix["provider"].str.startswith("Massive"),
            "delisting_return_or_terminal_value",
        ].item()
        == "not_documented"
    )


def test_price_contract_forbids_current_ticker_join() -> None:
    contract = point_in_time_price_contract()

    assert "current_ticker_join" in contract["forbidden_mapping_methods"]
    assert "source_snapshot_sha256" in contract["daily_price_required_columns"]


def test_mapping_validator_accepts_historical_interval() -> None:
    mapping = pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "cik": "0000000001",
                "security_id": "stable-1",
                "ticker": "DEMO",
                "exchange_mic": "XNAS",
                "valid_from_inclusive": "2019-01-01T00:00:00Z",
                "valid_to_exclusive": "2021-01-01T00:00:00Z",
                "mapping_method": "provider_stable_id_history",
                "source_snapshot_sha256": hashlib.sha256(b"source").hexdigest(),
            }
        ]
    )

    validate_security_mapping_frame(mapping)


def test_mapping_validator_rejects_current_ticker_join() -> None:
    mapping = pd.DataFrame(
        [
            {
                "target_id": "event-1",
                "cik": "0000000001",
                "security_id": "stable-1",
                "ticker": "DEMO",
                "exchange_mic": "XNAS",
                "valid_from_inclusive": "2019-01-01T00:00:00Z",
                "valid_to_exclusive": "2021-01-01T00:00:00Z",
                "mapping_method": "current_ticker_join",
                "source_snapshot_sha256": hashlib.sha256(b"source").hexdigest(),
            }
        ]
    )

    with pytest.raises(FilingCatalogError, match="Forbidden"):
        validate_security_mapping_frame(mapping)
