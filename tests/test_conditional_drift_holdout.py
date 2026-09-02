from __future__ import annotations

import numpy as np
import pandas as pd

from conditional_drift_holdout import (
    CANDIDATE_SCORE,
    CHALLENGER_SCORE,
    CLIPPED_CANDIDATE_SCORE,
    calculate_conditional_drift_holdout_diagnostics,
)


def test_holdout_diagnostic_uses_exact_same_event_sample() -> None:
    event_ids = [f"event-{index:02d}" for index in range(16)]
    ciks = [f"{index:010d}" for index in range(14)] + ["0000000000", "0000000001"]
    actual = np.linspace(-0.08, 0.07, 16)
    preregistered = pd.DataFrame(
        {
            "validation_event_id": event_ids,
            "cik": ciks,
            CHALLENGER_SCORE: actual[::-1],
            CANDIDATE_SCORE: actual,
            CLIPPED_CANDIDATE_SCORE: actual * 0.9,
            "alignment_candidate_within_training_support": [True] * 8 + [False] * 8,
            "alignment_candidate_severe_extrapolation": [False] * 14 + [True] * 2,
        }
    )
    outcomes = pd.DataFrame(
        {
            "validation_event_id": event_ids,
            "net_market_excess_return_20d": actual,
            "outcome_ready": [True] * 16,
        }
    )

    tables = calculate_conditional_drift_holdout_diagnostics(preregistered, outcomes)

    assert len(tables.analysis_sample) == 16
    assert tables.report["candidate_spearman_ic"] == 1.0
    assert tables.report["challenger_spearman_ic"] == -1.0
    assert tables.report["candidate_minus_challenger_spearman"] == 2.0
    assert tables.report["within_training_support_events"] == 8
    assert tables.report["verdict"] == "candidate_effectiveness_established"
    assert tables.verdict_checks["passed"].all()
    assert set(tables.quantile_returns["quantile"]) == {"bottom", "middle", "top"}
