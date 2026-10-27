"""Same-case price admission, explicit cash flows, and honest six-event diagnostics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_announcement_return_diagnostic import build_diagnostic, validate_contract
from sec_data.announcement_returns import (
    ACTION_COLUMNS,
    audit_return_window,
    calculate_case_return,
    descriptive_same_case_diagnostics,
    normalize_holding_actions,
    select_reviewed_cases,
)
from sec_data.execution_contract import execution_window


@pytest.fixture()
def evidence() -> dict[str, Any]:
    dates = pd.bdate_range("2025-03-03", periods=32)
    sessions = pd.DataFrame(
        [
            {
                "exchange_mic": "XNAS",
                "session_date": d.date().isoformat(),
                "regular_open_at": (d + pd.Timedelta(hours=9, minutes=30))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
                .isoformat(),
                "regular_close_at": (d + pd.Timedelta(hours=16))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
                .isoformat(),
                "source_snapshot_sha256": "a" * 64,
            }
            for d in dates
        ]
    )
    event = {
        "validation_event_id": "event",
        "cik": "0000000001",
        "ticker": "TEST",
        "exchange_mic": "XNAS",
        "price_target_id": "price",
        "benchmark_target_id": "bench",
        "security_id": "security",
        "security_id_kind": "share_class_figi",
        "announcement_at": "2025-03-03T12:00:00Z",
    }
    window = execution_window(event["announcement_at"], "XNAS", sessions)
    meta = pd.DataFrame(
        {"ticker": "TEST", "session_date": dates.strftime("%Y-%m-%d"), "source_sha256": "b" * 64}
    )
    continuity = {
        **{k: event[k] for k in ("cik", "ticker", "exchange_mic", "price_target_id")},
        "action_target_id": "actions",
        "resolved_provider_security_id": "security",
        "authorized_entry_start": "2025-03-03",
        "authorized_exit_end": "2025-04-30",
        "resolved_identity_ready": True,
        "historical_symbol_interval_ready": True,
        "corporate_actions_ready": True,
        "formal_action_evidence_ready": True,
        "identity_continuity_route": "stable_reference_endpoints_plus_official_session_daily_bars",
        "endpoint_identity_source_hashes": "c" * 64 + "|" + "d" * 64,
        "selection_uses_return_values": False,
        "returns_computed": False,
    }
    action = {
        "action_target_id": "actions",
        "price_target_id": "price",
        "cik": event["cik"],
        "selected_ticker": "TEST",
        "selected_exchange_mic": "XNAS",
        "security_identity_key": "security",
        "security_identity_key_kind": "share_class_figi",
        "splits_query_complete": True,
        "dividends_query_complete": True,
        "price_window_start": "2025-03-03",
        "price_window_end": "2025-04-30",
        "rejected_action_rows": 0,
        "selection_uses_return_values": False,
    }
    price = {
        "price_target_id": "price",
        "cik": event["cik"],
        "ticker": "TEST",
        "provider_query_complete": True,
        "parser_rejection_rows": 0,
        "price_window_start": "2025-03-03",
        "price_window_end": "2025-04-30",
    }
    listing = {
        **{k: event[k] for k in ("cik", "exchange_mic", "security_id", "security_id_kind")},
        "listing_outcome_ready": True,
        "listing_outcome_status": "inactive_with_delisting_date",
        "delisting_date": "2026-01-01",
        "outcome_cutoff_date": "2026-02-01",
        "source_snapshot_sha256": "e" * 64,
    }
    queries = pd.DataFrame(
        [
            {
                "query_kind": k,
                "ticker": "SPY",
                "benchmark_target_id": "bench",
                "status_code": 200,
                "sha256": "f" * 64,
                "window_start": "2025-03-03",
                "window_end": "2025-04-30",
            }
            for k in ("daily_bars", "splits", "dividends")
        ]
    )
    report = {
        "queries_complete": 3,
        "queries_required": 3,
        "rejections": 0,
        "action_normalization_failures": 0,
        "query_adjusted": False,
        "splits_explicit": True,
        "dividends_explicit": True,
    }
    return {
        "event": event,
        "sessions": sessions,
        "issuer_metadata": meta,
        "benchmark_metadata": meta.assign(ticker="SPY"),
        "continuity": continuity,
        "action": action,
        "price": price,
        "listing": listing,
        "queries": queries,
        "benchmark_report": report,
        "window": window,
    }


def _audit(e: dict[str, Any]) -> dict[str, Any]:
    return audit_return_window(**{k: v for k, v in e.items() if k != "window"})[0]


def test_all_reviewed_cases_retained_without_market_history() -> None:
    rows = [
        {
            "validation_event_id": "a",
            "release_value_pair_verified": True,
            "earliest_disclosure_at": "2024-01-01",
            "market_features_calculable": False,
        },
        {
            "validation_event_id": "b",
            "release_value_pair_verified": True,
            "earliest_disclosure_at": "2024-02-01",
        },
        {"validation_event_id": "c", "release_value_pair_verified": False},
    ]
    assert [r["validation_event_id"] for r in select_reviewed_cases(rows, ["b", "a"])] == ["a", "b"]
    for ids in (["b"], ["a", "a"], ["a", "b", "c"]):
        with pytest.raises(ValueError, match="entire"):
            select_reviewed_cases(rows, ids)
    with pytest.raises(ValueError, match="Duplicate"):
        select_reviewed_cases([*rows, rows[0]], ["a", "b"])


def test_metadata_admission_keeps_later_delisted_alternative_identity(
    evidence: dict[str, Any],
) -> None:
    result = _audit(evidence)
    assert result["conditional_return_diagnostic_ready"]
    assert result["later_delisted_after_window"]
    assert result["window"]["entry_session_date"] == "2025-03-05"
    assert not result["first_public_certified"] and not result["trading_admission"]
    assert not result["forward_numeric_price_values_loaded"]
    assert (
        result["identity_continuity_route"] == evidence["continuity"]["identity_continuity_route"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "interior_gap",
        "in_window_delisting",
        "wrong_security",
        "short_interval",
        "unqueried_actions",
        "adjusted_benchmark",
        "wrong_price_target",
    ],
)
def test_metadata_failures_block_without_replacement(
    evidence: dict[str, Any], mutation: str
) -> None:
    if mutation == "interior_gap":
        evidence["issuer_metadata"] = evidence["issuer_metadata"].drop(index=10)
    elif mutation == "in_window_delisting":
        evidence["listing"]["delisting_date"] = evidence["window"]["exit_session_date"]
    elif mutation == "wrong_security":
        evidence["continuity"]["resolved_provider_security_id"] = "other"
    elif mutation == "short_interval":
        evidence["continuity"]["authorized_exit_end"] = "2025-03-10"
    elif mutation == "unqueried_actions":
        evidence["action"]["dividends_query_complete"] = False
    elif mutation == "adjusted_benchmark":
        evidence["benchmark_report"]["query_adjusted"] = True
    else:
        evidence["price"]["price_target_id"] = "other"
    assert not _audit(evidence)["conditional_return_diagnostic_ready"]


def test_metadata_rejects_numeric_prices_and_duplicate_dates(evidence: dict[str, Any]) -> None:
    evidence["issuer_metadata"]["close_unadjusted"] = 100
    with pytest.raises(ValueError, match="metadata only"):
        _audit(evidence)
    evidence["issuer_metadata"] = evidence["issuer_metadata"].drop(columns="close_unadjusted")
    evidence["issuer_metadata"] = pd.concat(
        [evidence["issuer_metadata"], evidence["issuer_metadata"].iloc[:1]]
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        _audit(evidence)


def test_splits_dividends_and_exact_cash_retained_returns(evidence: dict[str, Any]) -> None:
    window = evidence["window"]
    split = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "execution_date": "2025-03-05",
                "split_from": 1,
                "split_to": 2,
                "historical_adjustment_factor": 999,
                "provider_event_id": "entry_split",
                "source_sha256": "c" * 64,
            },
            {
                "ticker": "TEST",
                "execution_date": "2025-03-06",
                "split_from": 1,
                "split_to": 2,
                "historical_adjustment_factor": 999,
                "provider_event_id": "held_split",
                "source_sha256": "c" * 64,
            },
        ]
    )
    dividend = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "ex_dividend_date": "2025-03-06",
                "cash_amount": 1,
                "currency": "usd",
                "provider_event_id": "dividend",
                "source_sha256": "d" * 64,
            }
        ]
    )
    holding = normalize_holding_actions(split, dividend, evidence["sessions"], window, "TEST")
    assert set(holding.provider_event_id) == {"held_split", "dividend"}
    bars = evidence["issuer_metadata"].assign(
        price_target_id="price", open_unadjusted=100, close_unadjusted=55
    )
    benchmark = evidence["benchmark_metadata"].assign(
        benchmark_target_id="bench", open_unadjusted=100, close_unadjusted=105
    )
    row, cash = calculate_case_return(
        evidence["event"],
        _audit(evidence),
        bars,
        benchmark,
        holding,
        pd.DataFrame(columns=ACTION_COLUMNS),
    )
    assert row["issuer_total_return"] == pytest.approx(0.12)
    assert row["benchmark_total_return"] == pytest.approx(0.05)
    assert row["market_excess_return"] == pytest.approx(0.07)
    assert row["market_relative_wealth_return"] == pytest.approx(1.12 / 1.05 - 1)
    assert row["issuer_split_factor"] == 2 and row["issuer_cash_dividend"] == 2
    assert [r["action_type"] for r in cash] == ["split", "cash_dividend"]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_holding_actions(
            pd.concat([split, split.iloc[1:]]), dividend, evidence["sessions"], window, "TEST"
        )


def test_blocked_return_never_reads_endpoint_values(evidence: dict[str, Any]) -> None:
    audit = _audit(evidence)
    audit["conditional_return_diagnostic_ready"] = False
    with pytest.raises(ValueError, match="authorized"):
        calculate_case_return(
            evidence["event"], audit, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )


def _analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = pd.DataFrame(
        [
            {
                "validation_event_id": str(i),
                "cik": str(i),
                "ticker": str(i),
                "actual_available_at": "2025-01-01T12:00:00Z",
                "gm_level": i,
                "gm_delta_yoy": -i,
                "gm_innovation_raw": i,
                "gm_innovation_std": i / 2,
                "filing_clock_gm_level": i,
                "filing_clock_gm_delta_yoy": -i,
                "filing_clock_gm_innovation_raw": i + 0.001,
                "filing_clock_gm_innovation_std": i / 2 + 0.001,
            }
            for i in range(6)
        ]
    )
    returns = signals[["validation_event_id", "cik", "ticker"]].assign(
        announcement_at="2025-01-01T12:00:00Z", market_excess_return=np.arange(6) / 100
    )
    return signals, returns


def test_same_case_descriptive_statistics_and_leave_one_out() -> None:
    signals, returns = _analysis()
    sample, stats, influence = descriptive_same_case_diagnostics(signals, returns)
    assert len(sample) == 6 and len(stats) == 8 and len(influence) == 48
    assert not stats.inferential_test.any()
    assert set(stats.loc[stats.representation.eq("gm_delta_yoy"), "spearman_ic"]) == {-1}
    assert influence.events.eq(5).all()
    # A small numeric correction need not change the cross-sectional ranking.
    assert stats.groupby("representation").spearman_ic.nunique().eq(1).all()


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "nan", "wrong_clock", "wrong_issuer"])
def test_diagnostics_reject_changing_denominator_or_information_set(mutation: str) -> None:
    signals, returns = _analysis()
    if mutation == "drop":
        returns = returns.iloc[1:]
    elif mutation == "duplicate":
        returns = pd.concat([returns, returns.iloc[:1]])
    elif mutation == "nan":
        signals.loc[0, "gm_innovation_std"] = np.nan
    elif mutation == "wrong_clock":
        returns.loc[0, "announcement_at"] = "2025-01-02T12:00:00Z"
    else:
        returns.loc[0, "cik"] = "other"
    with pytest.raises(ValueError):
        descriptive_same_case_diagnostics(signals, returns)


def test_constant_signal_produces_null_not_zero_ic() -> None:
    signals, returns = _analysis()
    signals["gm_level"] = 0.3
    _, stats, _ = descriptive_same_case_diagnostics(signals, returns)
    assert pd.isna(
        stats.loc[
            (stats.arm.eq("proxy_joint")) & stats.representation.eq("gm_level"), "spearman_ic"
        ]
    ).all()


def test_contract_cannot_enable_trading_or_retune_model(tmp_path: Path) -> None:
    c = json.loads(Path("docs/announcement_return_diagnostic_v1.json").read_bytes())
    validate_contract(c)
    for key in (
        "network_allowed",
        "trading_admission",
        "first_public_certified",
        "market_features_in_forecast",
        "hyperparameter_search_allowed",
        "inferential_tests_allowed",
    ):
        changed = copy.deepcopy(c)
        changed[key] = True
        with pytest.raises(ValueError, match="contract"):
            validate_contract(changed)
    c["holding_sessions_inclusive"] = 21
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="contract"):
        build_diagnostic(path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()
