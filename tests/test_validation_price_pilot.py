"""Tests for structural validation-price pilot selection."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_price_pilot import calculate_validation_price_pilot_targets


def test_price_pilot_freezes_listed_otc_and_complete_multiclass_strata() -> None:
    targets = pd.DataFrame(
        [
            _target("listed-z", "1", "ONE", "XNAS", "ID-ONE"),
            _target("listed-a", "2", "TWO", "XNYS", "ID-TWO"),
            _target("otc-z", "3", "OTCZ", "OTCM", "ID-OTCZ"),
            _target("otc-a", "4", "OTCA", "OTCM", "ID-OTCA"),
            _target("multi-b", "5", "DUALB", "XNAS", "ID-B"),
            _target("multi-a", "5", "DUALA", "XNAS", "ID-A"),
        ]
    )
    components = pd.DataFrame(
        [
            {
                "cik": "5",
                "ticker": ticker,
                "identity_key": identity,
                "component_count": 2,
                "identity_assignment_method": ("identity_repair_equal_weighted_share_class_basket"),
                "outcome_values_loaded": False,
            }
            for ticker, identity in (("DUALA", "ID-A"), ("DUALB", "ID-B"))
        ]
    )

    selected = calculate_validation_price_pilot_targets(targets, components)

    assert set(selected["price_target_id"]) == {"listed-a", "otc-a", "multi-a", "multi-b"}
    assert selected["pilot_stratum"].value_counts().to_dict() == {
        "complete_multiclass_basket": 2,
        "listed_single_security": 1,
        "otc_single_security": 1,
    }
    assert not selected["pilot_selection_uses_price_coverage"].any()
    assert not selected["pilot_selection_uses_return_values"].any()


def test_price_pilot_accepts_pre_event_primary_share_class_resolution() -> None:
    targets = pd.DataFrame(
        [
            _target("listed", "1", "ONE", "XNAS", "ID-ONE"),
            _target("otc", "2", "OTC", "OTCM", "ID-OTC"),
            _target("primary", "3", "DUALA", "XNAS", "ID-A"),
        ]
    )
    components = pd.DataFrame(
        [
            {
                "cik": "3",
                "ticker": "DUALA",
                "identity_key": "ID-A",
                "component_count": 1,
                "identity_assignment_method": ("primary_share_class_by_largest_disclosed_shares"),
                "outcome_values_loaded": False,
            }
        ]
    )

    selected = calculate_validation_price_pilot_targets(targets, components)

    assert set(selected["price_target_id"]) == {"listed", "otc", "primary"}
    assert selected["pilot_stratum"].value_counts().to_dict() == {
        "listed_single_security": 1,
        "otc_single_security": 1,
        "resolved_multiclass_primary": 1,
    }


def _target(
    target_id: str, cik: str, ticker: str, exchange: str, identity: str
) -> dict[str, object]:
    return {
        "price_target_id": target_id,
        "cik": cik,
        "selected_ticker": ticker,
        "selected_exchange_mic": exchange,
        "security_identity_key": identity,
        "security_identity_key_kind": "share_class_figi",
        "price_window_start": "2025-01-02",
        "price_window_end": "2025-03-01",
        "adjusted": False,
        "price_collection_authorized": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }
