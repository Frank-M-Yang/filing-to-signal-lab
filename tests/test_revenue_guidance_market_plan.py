"""Tests for the outcome-blind revenue-guidance market expansion."""

from __future__ import annotations

import pandas as pd

from sec_data.revenue_guidance_market_plan import calculate_revenue_guidance_market_plan


def test_market_plan_freezes_every_uncovered_event_without_return_values() -> None:
    forecasts, features, authorization, catalog = _fixtures()
    result = calculate_revenue_guidance_market_plan(
        forecasts,
        features,
        authorization,
        catalog,
        forecast_gate_passed=True,
    )

    assert result.report["forecast_events"] == 5
    assert result.report["existing_authorized_return_events"] == 2
    assert result.report["market_evidence_collection_events"] == 3
    assert result.report["identity_anchor_targets"] == 3
    assert result.events["selection_uses_return_values"].eq(False).all()
    assert result.events["outcome_values_loaded"].eq(False).all()
    assert (
        result.events.loc[
            result.events["market_evidence_collection_required"],
            "replacement_after_collection_failure_allowed",
        ]
        .eq(False)
        .all()
    )
    assert set(result.identity_anchors["cik"]) == {"0000000001", "0000000002"}


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows = []
    feature_rows = []
    catalog_rows = []
    for index in range(5):
        event_id = f"event-{index}"
        cik = f"{index % 2 + 1:010d}"
        accession = f"{index % 2 + 1:010d}-25-{index + 1:06d}"
        accepted = pd.Timestamp("2025-01-02 21:00:00", tz="UTC") + pd.Timedelta(days=index)
        forecast_rows.append(
            {
                "validation_event_id": event_id,
                "cik": cik,
                "target_actual_available_at": accepted,
                "selection_uses_return_values": False,
            }
        )
        feature_rows.append(
            {
                "validation_event_id": event_id,
                "target_id": f"target-{index}",
                "cik": cik,
                "target_accession_number": accession,
                "form": "10-Q",
                "report_date": "2024-12-31",
                "accepted_at": accepted,
            }
        )
        catalog_rows.append(
            {
                "cik": cik,
                "accession_number": accession,
                "accepted_at": accepted,
                "primary_document": f"event-{index}.htm",
                "source_url": f"https://www.sec.gov/example/event-{index}.htm",
            }
        )
    authorization = pd.DataFrame(
        [
            {
                "validation_event_id": f"event-{index}",
                "return_outcome_read_authorized": True,
                "selection_uses_return_values": False,
                "authorization_uses_return_values": False,
            }
            for index in (0, 1)
        ]
    )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(feature_rows),
        authorization,
        pd.DataFrame(catalog_rows),
    )
