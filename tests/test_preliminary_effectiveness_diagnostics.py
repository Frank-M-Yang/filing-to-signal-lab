"""Tests for the pre-registered filing-signal diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sec_data.preliminary_effectiveness_diagnostics import (
    EffectivenessDiagnosticConfig,
    calculate_effectiveness_diagnostics,
)


def test_diagnostic_runs_same_sample_cluster_bootstrap() -> None:
    ledger = _ledger()
    config = EffectivenessDiagnosticConfig(
        minimum_complete_events=20,
        bootstrap_replications=1_000,
        bootstrap_seed=7,
    )

    result = calculate_effectiveness_diagnostics(ledger, config)

    assert result.report["primary_analysis_gate"] is True
    assert result.report["primary_analysis_events"] == 25
    assert len(result.representation_ic) == 4
    innovation = result.representation_ic.set_index("feature").loc["gm_innovation_std"]
    assert innovation["spearman_ic"] > 0.9
    assert len(result.leave_one_issuer_out) == 5
    assert set(result.secondary_multiplicity["horizon_sessions"]) == {1, 5}
    assert result.verdict_checks.iloc[0]["passed"]


def test_diagnostic_blocks_below_precommitted_coverage() -> None:
    ledger = _ledger().loc[lambda frame: ~frame["cik"].isin(["0000000004", "0000000005"])]
    result = calculate_effectiveness_diagnostics(
        ledger,
        EffectivenessDiagnosticConfig(
            minimum_complete_events=20,
            bootstrap_replications=1_000,
            bootstrap_seed=7,
        ),
    )

    assert result.report["primary_analysis_gate"] is False
    assert result.report["verdict"] == "insufficient_data_quality_or_coverage"
    assert result.representation_ic.empty


def _ledger() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for issuer_index in range(5):
        cik = f"{issuer_index + 1:010d}"
        for event_index in range(5):
            event_id = f"event-{issuer_index}-{event_index}"
            innovation = issuer_index * 0.8 + event_index * 0.4
            delta = rng.normal()
            level = 0.40 + rng.normal(scale=0.02)
            for horizon in (1, 5, 20):
                outcome = 0.03 * innovation + rng.normal(scale=0.005 + 0.0002 * horizon)
                rows.append(
                    {
                        "longitudinal_event_id": event_id,
                        "cik": cik,
                        "accepted_at": (
                            pd.Timestamp("2025-01-01T21:00:00Z")
                            + pd.Timedelta(days=issuer_index * 20 + event_index)
                        ),
                        "form": "10-Q",
                        "horizon_sessions": horizon,
                        "analysis_sample": True,
                        "market_excess_return_ready": True,
                        "market_excess_return_arithmetic": outcome,
                        "later_delisted_after_formal_exit": cik == "0000000001",
                        "gm_level": level,
                        "gm_delta_yoy": delta,
                        "gm_innovation_raw": innovation * 0.01,
                        "gm_innovation_std": innovation,
                    }
                )
    return pd.DataFrame(rows)
