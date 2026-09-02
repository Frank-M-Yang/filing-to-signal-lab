from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conditional_drift_forward import (
    ALIGNMENT_FEATURES,
    PRIMARY_TARGET,
    PRIMARY_TARGET_AVAILABLE_AT,
    calculate_conditional_drift_forward_candidate,
    freeze_conditional_drift_models,
    score_forward_candidates,
)
from sec_data.filing_catalog import FilingCatalogError


def _development() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(30):
        row: dict[str, object] = {
            "cik": f"{index % 10:010d}",
            "evidence_role": "legacy_observed_training_only",
            "v2_development_candidate": True,
            PRIMARY_TARGET: 0.002 * index - 0.02,
            PRIMARY_TARGET_AVAILABLE_AT: pd.Timestamp("2024-01-01", tz="UTC")
            + pd.Timedelta(days=index),
        }
        for feature_index, feature in enumerate(ALIGNMENT_FEATURES, start=1):
            row[feature] = np.sin((index + 1) * feature_index / 17.0) + index / 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def _score_row() -> pd.DataFrame:
    row: dict[str, object] = {
        "validation_event_id": "event-1",
        "cik": "0000000001",
        "ticker": "TEST",
        "accepted_at": pd.Timestamp("2024-03-01 21:00:00", tz="UTC"),
        "reaction_feature_available_at": pd.Timestamp("2024-03-04 21:00:00", tz="UTC"),
        "execution_entry_at": pd.Timestamp("2024-03-05 14:30:00", tz="UTC"),
        "evidence_role": "unobserved_validation_candidate",
        "prior_outcome_observed": False,
        "outcome_materialized_locally": False,
    }
    for index, feature in enumerate(ALIGNMENT_FEATURES, start=1):
        row[feature] = index / 10.0
    return pd.DataFrame([row])


def test_frozen_model_scores_without_outcome_columns() -> None:
    models = freeze_conditional_drift_models(_development())
    first = score_forward_candidates(_score_row(), models)
    second = score_forward_candidates(_score_row(), models)

    assert len(models) == 2
    assert models[0].training_events == 30
    assert first.equals(second)
    assert first["outcome_columns_loaded"].eq(False).all()
    assert not any(column.startswith("target_") for column in first.columns)
    assert np.isfinite(first["F2_alignment_candidate_research_score"]).all()


def test_forward_scorer_rejects_outcome_join_and_observed_event() -> None:
    models = freeze_conditional_drift_models(_development())
    with_target = _score_row().assign(target_market_excess_return_20d=0.2)
    with pytest.raises(FilingCatalogError, match="forbidden outcome columns"):
        score_forward_candidates(with_target, models)

    observed = _score_row().assign(prior_outcome_observed=True)
    with pytest.raises(FilingCatalogError, match="observed outcome event"):
        score_forward_candidates(observed, models)


def test_forward_scorer_rejects_late_reaction_clock() -> None:
    models = freeze_conditional_drift_models(_development())
    late = _score_row().assign(
        reaction_feature_available_at=pd.Timestamp("2024-03-05 15:00:00", tz="UTC")
    )
    with pytest.raises(FilingCatalogError, match="pre-entry reaction clock"):
        score_forward_candidates(late, models)


def test_forward_queue_quarantines_locally_materialized_outcomes() -> None:
    development = _development()
    accepted = [
        pd.Timestamp("2024-03-01", tz="UTC"),
        pd.Timestamp("2024-03-02", tz="UTC"),
        pd.Timestamp("2024-03-03", tz="UTC"),
    ]
    registry = pd.DataFrame(
        {
            "validation_event_id": ["event-1", "event-2", "event-3"],
            "cik": ["1", "2", "3"],
            "accession_number": ["a1", "a2", "a3"],
            "accepted_at": accepted,
            "form": ["10-Q", "10-Q", "10-K"],
            "company_name_snapshot": ["A", "B", "C"],
            "collection_wave": [1, 1, 1],
            "evidence_role": ["unobserved_validation_candidate"] * 3,
            "exact_outcome_previously_observed": [False] * 3,
            "return_outcome_read_authorized": [False] * 3,
        }
    )
    filing = pd.DataFrame(
        {
            "validation_event_id": ["event-1", "event-2", "event-3"],
            "feature_status": ["complete_case_ready"] * 3,
            "gross_margin_applicability_complete_case": [True] * 3,
            "evidence_role": ["unobserved_validation_candidate"] * 3,
        }
    )
    identity = pd.DataFrame(
        {
            "validation_event_id": ["event-1", "event-2", "event-3"],
            "event_security_identity_ready": [True, True, False],
            "assigned_ticker": ["A", "B", pd.NA],
            "assigned_exchange_mic": ["XNYS", "XNAS", pd.NA],
        }
    )
    tables = calculate_conditional_drift_forward_candidate(
        development,
        registry,
        filing,
        identity,
        {"event-2"},
    )

    status = tables.scoring_queue.set_index("validation_event_id")["queue_status"].to_dict()
    assert status == {
        "event-1": "awaiting_outcome_blind_market_reaction_features",
        "event-2": "quarantined_local_outcome_exists",
        "event-3": "awaiting_security_identity",
    }
    assert tables.report["clean_temporal_queue_events"] == 2
    assert tables.report["locally_materialized_outcome_quarantine_events"] == 1
    assert tables.report["scored_candidate_events"] == 0
