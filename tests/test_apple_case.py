"""Audits for the repository's first real-filing case study."""

from pathlib import Path

import pandas as pd
import pytest

from sec_data.apple_case import CASE_FACTS_PATH, load_apple_amendment_case


def test_apple_case_reproduces_signal_distortion() -> None:
    case = load_apple_amendment_case()

    assert len(case.facts) == 13
    assert case.metadata["comparable_fact_count"] == 210
    assert case.metadata["changed_fact_count"] == 77
    assert case.information_delay_days == pytest.approx(90.0469, rel=1e-4)
    assert not case.facts["amendment_fact_id"].duplicated().any()

    fiscal_2009 = case.gross_margin.loc[case.gross_margin["fiscal_year"].eq(2009)].iloc[0]
    assert fiscal_2009["original_margin"] == pytest.approx(0.3596354380)
    assert fiscal_2009["amendment_margin"] == pytest.approx(0.4013984384)
    assert fiscal_2009["original_yoy_change"] == pytest.approx(0.016490637)
    assert fiscal_2009["amendment_yoy_change"] == pytest.approx(0.049393957)
    assert fiscal_2009["signal_inflation"] == pytest.approx(0.032903320)


def test_apple_case_rejects_invalid_point_in_time_order(tmp_path: Path) -> None:
    facts = pd.read_csv(CASE_FACTS_PATH)
    facts.loc[0, "amendment_accepted_at"] = facts.loc[0, "original_accepted_at"]
    invalid_path = tmp_path / "invalid-case.csv"
    facts.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="point-in-time"):
        load_apple_amendment_case(facts_path=invalid_path)
