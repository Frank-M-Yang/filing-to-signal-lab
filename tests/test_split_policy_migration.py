"""The split migration must retain event identity, missing data and the fixed denominator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_split_policy_migration import compare_same_event_states
from sec_data.validation_market_state import MARKET_STATE_FEATURES


def _states() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation_event_id": str(i),
                "cik": "1",
                "ticker": "TEST",
                "exchange_mic": "XNYS",
                "price_target_id": "p",
                "accepted_at": pd.Timestamp("2025-01-06T20:00Z"),
                **dict.fromkeys(MARKET_STATE_FEATURES, 0.1 if i == 0 else np.nan),
            }
            for i in range(2)
        ]
    )


def test_missing_features_stay_missing_and_changes_are_same_event() -> None:
    old, new = _states(), _states()
    new.loc[0, MARKET_STATE_FEATURES[0]] = 0.2
    out = compare_same_event_states(old, new)
    assert len(out) == 26
    assert out.numerically_changed.sum() == 1
    assert not out.availability_changed.any()
    assert out.loc[out.validation_event_id.eq("1"), "difference"].isna().all()
    new.loc[1, MARKET_STATE_FEATURES[0]] = 0.3
    out = compare_same_event_states(old, new)
    assert out.availability_changed.sum() == 1
    assert out.loc[out.availability_changed, "difference"].isna().all()


@pytest.mark.parametrize(
    "field",
    ["ticker", "cik", "accepted_at", "price_target_id", "exchange_mic", "validation_event_id"],
)
def test_changed_event_identity_or_clock_is_not_a_model_improvement(field: str) -> None:
    old, new = _states(), _states()
    if field == "accepted_at":
        new.loc[0, field] += pd.Timedelta(days=1)
    else:
        new.loc[0, field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        compare_same_event_states(old, new)


def test_missing_duplicate_or_reordered_events() -> None:
    old = _states()
    with pytest.raises(ValueError):
        compare_same_event_states(old, old.iloc[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_same_event_states(old, pd.concat([old, old.iloc[:1]]))
    result = compare_same_event_states(old, old.iloc[::-1])
    assert not result.numerically_changed.any()
