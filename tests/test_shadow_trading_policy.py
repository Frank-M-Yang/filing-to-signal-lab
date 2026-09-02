from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shadow_trading_policy import (
    ShadowTradingPolicyConfig,
    ShadowTradingPolicyError,
    build_shadow_trading_policy,
    calculate_shadow_trading_policy,
)


def test_policy_is_cash_only_outcome_blind_and_complete() -> None:
    events = pd.DataFrame([_event("one", 1), _event("two", 2)])

    tables = calculate_shadow_trading_policy(events, _report(2))

    assert len(tables.event_registry) == 2
    assert tables.gates["passed"].all()
    assert tables.report["policy_gate_passed"] is True
    assert tables.report["shadow_intent_generation_authorized"] is True
    assert tables.report["shadow_order_execution_authorized"] is False
    assert tables.report["return_values_loaded"] is False
    assert tables.protocol["missing_evidence_action"] == "block_without_imputation"
    config = tables.report["config"]
    assert config["buy_threshold"] == pytest.approx(0.01)
    assert config["exit_or_avoid_threshold"] == pytest.approx(-0.01)
    assert config["maximum_gross_weight"] == pytest.approx(0.20)
    assert config["short_selling_allowed"] is False


@pytest.mark.parametrize(
    "flag",
    [
        "actual_values_loaded",
        "return_values_loaded",
        "selection_uses_return_values",
        "external_validation_metrics_computed",
    ],
)
def test_policy_rejects_observed_upstream_outcomes(flag: str) -> None:
    report = _report(1)
    report[flag] = True

    with pytest.raises(ShadowTradingPolicyError):
        calculate_shadow_trading_policy(pd.DataFrame([_event("one", 1)]), report)


def test_policy_config_cannot_be_tuned() -> None:
    with pytest.raises(ValueError):
        ShadowTradingPolicyConfig(buy_threshold=0.02).validate()


def test_policy_build_is_immutable(tmp_path: Path) -> None:
    preregistration = tmp_path / "preregistration"
    preregistration.mkdir()
    pd.DataFrame([_event("one", 1)]).to_parquet(
        preregistration / "prospective_market_innovation_events.parquet", index=False
    )
    (preregistration / "prospective_market_innovation_report.json").write_text(
        __import__("json").dumps(_report(1)), encoding="utf-8"
    )
    (preregistration / "output_manifest.json").write_text("{}", encoding="utf-8")

    first = build_shadow_trading_policy(preregistration, output_root=tmp_path / "policy")
    second = build_shadow_trading_policy(preregistration, output_root=tmp_path / "policy")

    assert first.output_root == second.output_root
    assert (first.output_root / "shadow_trading_policy_protocol.json").is_file()


def _event(event_id: str, order: int) -> dict[str, object]:
    return {
        "prospective_event_id": event_id,
        "registration_order": order,
        "cik": str(order),
        "ticker": f"T{order}",
        "target_period_end": "2026-09-30",
        "replacement_allowed": False,
        "actual_value_loaded_at_registration": False,
        "return_value_loaded_at_registration": False,
        "selection_uses_return_values": False,
    }


def _report(events: int) -> dict[str, object]:
    return {
        "registered_events": events,
        "preregistration_gate_passed": True,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "external_validation_metrics_computed": False,
        "portfolio_backtest_computed": False,
        "live_trading_authorized": False,
    }
