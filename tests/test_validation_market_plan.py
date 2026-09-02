"""Tests for the outcome-blind validation market-data plan."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.validation_market_plan import (
    ValidationMarketPlanConfig,
    calculate_validation_market_plan,
)


def test_market_plan_uses_pre_event_applicability_and_endpoint_anchors() -> None:
    features, catalog = _inputs()

    result = calculate_validation_market_plan(features, catalog)

    assert result.report["unobserved_basic_window_history_eligible_events"] == 5
    assert result.report["unobserved_basic_window_complete_price_candidates"] == 4
    assert result.report["price_candidate_issuers"] == 2
    assert result.report["identity_anchor_queries"] == 4
    assert result.report["filing_index_evidence_targets"] == 5
    assert result.report["outcome_values_loaded"] is False
    assert not result.event_targets["return_outcome_read_authorized"].any()
    assert not result.issuer_price_spans["price_query_authorized"].any()
    assert set(result.identity_anchors["anchor_role"]) == {
        "first_candidate",
        "last_candidate",
    }


def test_incomplete_current_event_keeps_evidence_but_not_price_authorization() -> None:
    features, catalog = _inputs()
    features.loc[0, "gross_margin_applicability_complete_case"] = False

    result = calculate_validation_market_plan(features, catalog)

    event = result.event_targets.loc[result.event_targets["target_id"].eq("target-0")].iloc[0]
    assert event["identity_evidence_candidate"]
    assert not event["price_collection_candidate"]
    assert event["market_plan_status"] == "history_eligible_but_current_features_incomplete"


def test_market_plan_rejects_any_loaded_outcome_column() -> None:
    features, catalog = _inputs()
    features["market_excess_return_arithmetic"] = 0.0

    with pytest.raises(FilingCatalogError, match="outcome columns"):
        calculate_validation_market_plan(features, catalog)


def test_catalog_acceptance_clock_mismatch_keeps_sec_evidence_gate_closed() -> None:
    features, catalog = _inputs()
    catalog.loc[0, "accepted_at"] = "2025-01-01T12:00:01Z"

    result = calculate_validation_market_plan(
        features,
        catalog,
        ValidationMarketPlanConfig(),
    )

    event = result.filing_evidence_targets.loc[
        result.filing_evidence_targets["validation_event_id"].eq("event-0")
    ].iloc[0]
    assert not event["catalog_clock_match"]
    assert not event["sec_index_collection_required"]


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = pd.to_datetime(
        [
            "2025-01-01T12:00:00Z",
            "2025-02-01T12:00:00Z",
            "2025-03-01T12:00:00Z",
            "2025-01-15T12:00:00Z",
            "2025-02-15T12:00:00Z",
            "2025-03-15T12:00:00Z",
        ],
        utc=True,
    )
    features = pd.DataFrame(
        {
            "target_id": [f"target-{index}" for index in range(6)],
            "validation_event_id": [f"event-{index}" for index in range(6)],
            "cik": ["0000000001"] * 3 + ["0000000002"] * 3,
            "target_accession_number": [f"accession-{index}" for index in range(6)],
            "form": ["10-Q"] * 6,
            "report_date": ["2024-12-31"] * 6,
            "accepted_at": accepted,
            "evidence_role": ["unobserved_validation_candidate"] * 5
            + ["legacy_issuer_training_only"],
            "within_documented_massive_basic_window": [True] * 6,
            "gross_margin_history_eligible": [True] * 6,
            "gross_margin_applicability_complete_case": [
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            "collection_wave": [1] * 3 + [2] * 3,
        }
    )
    catalog = pd.DataFrame(
        {
            "cik": features["cik"],
            "accession_number": features["target_accession_number"],
            "accepted_at": accepted,
            "primary_document": [f"{index}.htm" for index in range(6)],
            "source_url": [f"https://sec.example/{index}.htm" for index in range(6)],
        }
    )
    return features, catalog
