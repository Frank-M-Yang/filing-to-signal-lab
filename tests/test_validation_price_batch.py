"""Tests for the outcome-blind formal validation price batch."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.validation_price_batch import (
    ValidationPriceBatchError,
    calculate_validation_price_batch_targets,
)


def test_formal_price_batch_freezes_all_targets_in_acquisition_order() -> None:
    targets = pd.DataFrame(
        [
            _target("later", 2, "TWO"),
            _target("first", 1, "ONE"),
        ]
    )

    selected = calculate_validation_price_batch_targets(targets)

    assert selected["price_target_id"].tolist() == ["first", "later"]
    assert selected["target_id"].tolist() == ["first", "later"]
    assert not selected["batch_selection_uses_price_coverage"].any()
    assert not selected["batch_selection_uses_return_values"].any()
    assert not selected["batch_replacement_after_failure_allowed"].any()


def test_formal_price_batch_rejects_outcome_informed_targets() -> None:
    targets = pd.DataFrame([_target("bad", 1, "BAD")])
    targets.loc[0, "selection_uses_return_values"] = True

    with pytest.raises(ValidationPriceBatchError, match="used return values"):
        calculate_validation_price_batch_targets(targets)


def _target(target_id: str, order: int, ticker: str) -> dict[str, object]:
    return {
        "price_target_id": target_id,
        "cik": f"{order:010d}",
        "selected_ticker": ticker,
        "selected_exchange_mic": "XNAS",
        "security_identity_key": f"ID-{ticker}",
        "security_identity_key_kind": "share_class_figi",
        "selected_provider_security_id": f"ID-{ticker}",
        "selected_provider_security_id_kind": "share_class_figi",
        "price_window_start": "2024-01-01",
        "price_window_end": "2025-01-01",
        "adjusted": False,
        "price_collection_authorized": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_acquisition_order": order,
    }
