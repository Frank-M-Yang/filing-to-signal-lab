"""Tests for the independent formal validation signal diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sec_data.preliminary_effectiveness_diagnostics import EffectivenessDiagnosticConfig
from sec_data.validation_effectiveness_diagnostics import (
    calculate_validation_effectiveness_diagnostics,
)


def test_calculates_same_complete_case_cluster_inference() -> None:
    rows = 24
    event_ids = [f"v{index:02d}" for index in range(rows)]
    ciks = [f"{index % 6 + 1:010d}" for index in range(rows)]
    innovation = np.linspace(-2.0, 2.0, rows)
    delta = np.sin(np.linspace(0.0, 4.0, rows))
    returns = pd.DataFrame(
        {
            "security_event_component_id": [f"e{index:02d}" for index in range(rows)],
            "validation_event_id": event_ids,
            "cik": ciks,
            "accepted_at": pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC").astype(
                str
            ),
            "form": ["10-Q"] * rows,
            "accession_number": [f"a{index:02d}" for index in range(rows)],
            "issuer_total_return": innovation * 0.02,
            "market_total_return": [0.01] * rows,
            "market_excess_return_arithmetic": innovation * 0.02 - 0.01,
            "market_excess_return_ready": [True] * rows,
            "later_delisted_after_formal_exit": [False] * rows,
            "return_outcome_read_authorized": [True] * rows,
            "selection_uses_return_values": [False] * rows,
            "returns_computed": [True] * rows,
        }
    )
    features = pd.DataFrame(
        {
            "validation_event_id": event_ids,
            "cik": ciks,
            "target_accession_number": [f"a{index:02d}" for index in range(rows)],
            "accepted_at": returns["accepted_at"],
            "actual_available_at": returns["accepted_at"],
            "gross_margin_history_eligible": [True] * rows,
            "gross_margin_applicability_complete_case": [True] * rows,
            "feature_status": ["complete_case_ready"] * rows,
            "gm_level": np.linspace(0.2, 0.5, rows),
            "gm_delta_yoy": delta,
            "gm_innovation_raw": innovation + 0.1 * delta,
            "gm_innovation_std": innovation,
        }
    )
    config = EffectivenessDiagnosticConfig(
        minimum_complete_events=20,
        bootstrap_replications=1000,
    )
    result = calculate_validation_effectiveness_diagnostics(returns, features, config)
    assert len(result.analysis_sample) == rows
    assert result.report["complete_case_coverage"] == 1.0
    assert result.report["innovation_ic"] > 0.9
    assert len(result.leave_one_issuer_out) == 6
    assert len(result.quantile_returns) == 12
