"""Tests for the pre-return preliminary effectiveness contract."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_data.filing_catalog import FilingCatalogError
from sec_data.preliminary_effectiveness import (
    PreliminaryEffectivenessConfig,
    calculate_preliminary_effectiveness_preregistration,
)


def test_preregistration_freezes_same_sample_and_primary_contrast() -> None:
    events = _events(25)
    config = PreliminaryEffectivenessConfig(
        minimum_complete_events=20,
        minimum_complete_coverage=0.80,
    )

    tables = calculate_preliminary_effectiveness_preregistration(events, config)

    assert tables.report["candidate_events"] == 25
    assert tables.report["complete_feature_events"] == 20
    assert tables.report["minimum_combined_feature_return_events"] == 20
    assert tables.report["outcomes_observed_at_preregistration"] is False
    assert tables.report["selection_uses_event_returns"] is False
    assert tables.report["primary_contrast"] == ("IC(gm_innovation_std,R20)-IC(gm_delta_yoy,R20)")
    assert tables.event_registry["outcome_observed_at_preregistration"].eq(False).all()
    assert "market_excess_return_arithmetic" not in tables.event_registry
    assert tables.hypothesis_registry.iloc[0]["hypothesis_id"] == "H1_PRIMARY"


def test_preregistration_rejects_observed_outcome_columns() -> None:
    events = _events(25)
    events["market_excess_return_arithmetic"] = 0.01

    with pytest.raises(FilingCatalogError, match="observed outcome columns"):
        calculate_preliminary_effectiveness_preregistration(
            events,
            PreliminaryEffectivenessConfig(
                minimum_complete_events=20,
                minimum_complete_coverage=0.80,
            ),
        )


def _events(count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        complete = index < 20
        accepted = pd.Timestamp("2025-01-01T21:00:00Z") + pd.Timedelta(days=index)
        rows.append(
            {
                "longitudinal_event_id": f"event-{index}",
                "cik": f"{index % 3 + 1:010d}",
                "accession_number": f"accession-{index}",
                "accepted_at": accepted,
                "report_date": "2024-12-31",
                "form": "10-Q",
                "formal_entry_session_date": "2025-01-06",
                "formal_entry_open_at": "2025-01-06T14:30:00+00:00",
                "formal_exit_session_date": "2025-01-31",
                "formal_exit_close_at": "2025-01-31T21:00:00+00:00",
                "forecast_feature_cutoff": accepted - pd.Timedelta(days=1),
                "forecast_training_end": accepted - pd.Timedelta(days=2),
                "forecast_error_history_end": accepted - pd.Timedelta(days=3),
                "gm_level": 0.40 if complete else float("nan"),
                "gm_delta_yoy": 0.01 if complete else float("nan"),
                "gm_innovation_raw": 0.02 if complete else float("nan"),
                "gm_innovation_std": 1.0 if complete else float("nan"),
                "representation_features_complete": complete,
                "feature_status": "complete_case_ready" if complete else "delta_unavailable",
            }
        )
    return pd.DataFrame(rows)
