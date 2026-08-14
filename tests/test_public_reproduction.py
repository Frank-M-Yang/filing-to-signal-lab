"""Offline reproducibility contract for the public repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_public_reproduction import build_public_reproduction


def test_public_reproduction_is_deterministic_and_honest(tmp_path: Path) -> None:
    first_path = build_public_reproduction(tmp_path)
    first_bytes = first_path.read_bytes()
    second_path = build_public_reproduction(tmp_path)

    assert second_path.read_bytes() == first_bytes
    document: dict[str, Any] = json.loads(first_bytes)
    assert document["checks"] == {
        "apple_signal_inflation_reproduced": True,
        "empirical_effectiveness_not_overclaimed": True,
        "simulation_bias_gap_pass": True,
    }

    metrics = document["simulation"]["scenario_metrics"]
    assert 0.10 <= metrics["原始回测（全部偏差）"]["annual_return"] <= 0.20
    assert metrics["修正全部偏差后"]["annual_return"] < 0.03
    assert document["apple_2009"]["fiscal_2009"]["signal_inflation"] == pytest.approx(0.032903320)
    empirical = document["empirical_longitudinal"]
    assert empirical["evidence_type"] == "frozen_aggregate_snapshot_not_recomputed"
    assert empirical["complete_events"] == 49
    assert empirical["verdict"] == "effectiveness_not_established"
