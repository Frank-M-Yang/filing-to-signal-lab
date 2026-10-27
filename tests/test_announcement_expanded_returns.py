"""No replacement, metadata-only continuity reuse and all-fourteen outcome gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.build_announcement_expanded_returns import _complete_returns, build_expanded_returns
from sec_data.announcement_expanded_returns import (
    all_case_return_gate,
    freeze_reviewed_expansion,
    prior_industry_summary,
    reauthorize_cached_continuity,
    supplementation_plan,
    validate_contract,
)


def selection_fixture() -> tuple[Any, ...]:
    """A synthetic six-control/twenty-new-issuer cohort, qualified before outcomes."""
    rows = [
        {
            "validation_event_id": str(i),
            "cik": str(i),
            "assigned_ticker": f"T{i}",
            "assigned_exchange_mic": "XNAS",
            "target_accession_number": f"acc{i}",
            "price_target_id": f"price{i}",
            "earliest_disclosure_at": "2025-03-03T12:00:00Z",
            "release_value_pair_verified": i < 14,
            "first_public_certified": False,
            "model_admission": False,
            "return_admission": False,
        }
        for i in range(26)
    ]
    return (
        copy.deepcopy(rows[:14]),
        copy.deepcopy(rows[6:]),
        copy.deepcopy(rows[:6]),
        copy.deepcopy(rows[6:]),
        [str(i) for i in range(316)],
        [str(i) for i in range(14)],
    )


def test_entire_reviewed_category_retained_without_market_coverage_selection() -> None:
    args = selection_fixture()
    args[0][0]["blocking_reasons"] = ["stale_execution_issuer_dates_missing"]
    assert len(freeze_reviewed_expansion(*args)) == 14


@pytest.mark.parametrize(
    "violation",
    [
        "duplicate",
        "replace",
        "denominator",
        "missing_disposition",
        "changed_clock",
        "promote",
        "same_issuer",
    ],
)
def test_expansion_sample_and_information_set_cannot_be_changed(violation: str) -> None:
    args = list(selection_fixture())
    if violation == "duplicate":
        args[0][0] = copy.deepcopy(args[0][1])
    elif violation == "replace":
        args[0][-1] = copy.deepcopy(args[1][-1])
        args[0][-1]["release_value_pair_verified"] = True
    elif violation == "denominator":
        args[4].pop()
    elif violation == "missing_disposition":
        args[3].pop()
    elif violation == "changed_clock":
        args[0][0]["earliest_disclosure_at"] = "2025-03-04T12:00:00Z"
    elif violation == "promote":
        args[0][0]["return_admission"] = True
    else:
        args[1][-1]["cik"] = args[2][0]["cik"]
    with pytest.raises(ValueError):
        freeze_reviewed_expansion(*args)


def continuity_fixture() -> tuple[Any, ...]:
    """Exact cached provider identity bounds; synthetic dates are not market evidence."""
    dates = pd.bdate_range("2025-03-03", periods=20).strftime("%Y-%m-%d")
    event = {
        "validation_event_id": "event",
        "cik": "1",
        "ticker": "TEST",
        "exchange_mic": "XNAS",
        "action_target_id": "actions",
        "price_target_id": "price",
        "security_id": "share",
        "security_id_kind": "share_class_figi",
    }
    window = {
        "validation_event_id": "event",
        "window": {"entry_session_date": str(dates[0]), "exit_session_date": str(dates[-1])},
    }
    actions = pd.DataFrame(
        [
            {
                "action_target_id": "actions",
                "action_acquisition_order": 1,
                "price_target_id": "price",
                "cik": "1",
                "selected_ticker": "TEST",
                "selected_exchange_mic": "XNAS",
                "selected_provider_security_id": "share",
                "selected_share_class_figi": "share",
                "selected_composite_figi": "composite",
                "security_identity_key": "share",
                "reference_start_query_complete": True,
                "reference_end_query_complete": True,
                "ticker_events_query_complete": False,
                "historical_symbol_interval_ready": False,
                "corporate_actions_ready": True,
                "selection_uses_return_values": False,
                "returns_computed": False,
                "price_window_start": "2025-03-01",
                "price_window_end": "2025-04-01",
            }
        ]
    )
    identities = pd.DataFrame(
        [
            {
                "action_target_id": "actions",
                "cik": "1",
                "query_kind": kind,
                "as_of_date": day,
                "ticker": "TEST",
                "exchange_mic": "XNAS",
                "security_type": "CS",
                "provider_security_id": "share",
                "share_class_figi": "share",
                "composite_figi": "composite",
                "name": "Test company",
                "source_sha256": "a" * 64,
            }
            for kind, day in [("reference_start", "2025-03-01"), ("reference_end", "2025-04-01")]
        ]
    )
    ticker_events = pd.DataFrame(columns=["action_target_id", "event_date", "new_ticker"])
    metadata = pd.DataFrame(
        {
            "price_target_id": "price",
            "ticker": "TEST",
            "session_date": dates,
            "source_sha256": "b" * 64,
        }
    )
    sessions = pd.DataFrame({"exchange_mic": "XNAS", "session_date": dates})
    return [event], [window], actions, identities, ticker_events, metadata, sessions


def test_cached_identity_can_cover_new_authorization_without_changing_source_data() -> None:
    args = continuity_fixture()
    before = args[2].copy(deep=True)
    result, audit = reauthorize_cached_continuity(*args)
    assert result.formal_action_evidence_ready.all()
    assert result.authorized_entry_start.iloc[0] == "2025-03-03"
    assert not result.price_values_loaded.any()
    assert audit[0]["cached_endpoint_dates_verified"]
    assert not audit[0]["trading_admission"]
    pd.testing.assert_frame_equal(before, args[2])


def test_same_issuer_other_share_class_does_not_create_selected_class_ambiguity() -> None:
    args = list(continuity_fixture())
    other = args[3].copy()
    other["ticker"], other["share_class_figi"] = "OTHER", "other-share"
    args[3] = pd.concat([args[3], other], ignore_index=True)
    result, _ = reauthorize_cached_continuity(*args)
    assert result.resolved_identity_ready.all()
    assert result.resolved_share_class_figi.eq("share").all()


@pytest.mark.parametrize(
    "violation",
    [
        "early_entry",
        "late_exit",
        "wrong_endpoint_date",
        "duplicate_endpoint",
        "wrong_share_class",
        "wrong_cik",
    ],
)
def test_reauthorization_never_extrapolates_cached_query_intervals(violation: str) -> None:
    args = list(continuity_fixture())
    if violation == "early_entry":
        args[1][0]["window"]["entry_session_date"] = "2025-02-28"
    elif violation == "late_exit":
        args[1][0]["window"]["exit_session_date"] = "2025-04-02"
    elif violation == "wrong_endpoint_date":
        args[3].loc[0, "as_of_date"] = "2025-03-04"
    elif violation == "duplicate_endpoint":
        args[3] = pd.concat([args[3], args[3].iloc[[0]]], ignore_index=True)
    elif violation == "wrong_share_class":
        args[3]["share_class_figi"] = "wrong"
    else:
        args[3]["cik"] = "wrong"
    result, audit = reauthorize_cached_continuity(*args)
    assert result.empty
    assert not audit[0]["reauthorization_attempted"]


def test_no_numeric_prices_and_no_outcome_based_identity_inputs() -> None:
    args = list(continuity_fixture())
    args[5]["close_unadjusted"] = 999.0
    with pytest.raises(ValueError, match="metadata only"):
        reauthorize_cached_continuity(*args)
    args[5] = args[5].drop(columns="close_unadjusted")
    args[2]["selection_uses_return_values"] = True
    with pytest.raises(ValueError, match="used returns"):
        reauthorize_cached_continuity(*args)


def test_missing_daily_boundary_cannot_pass_existing_fallback() -> None:
    args = list(continuity_fixture())
    args[5] = args[5].iloc[1:]
    result, _ = reauthorize_cached_continuity(*args)
    assert not result.daily_bar_boundaries_ready.any()
    assert not result.resolved_identity_ready.any()


def window_fixture() -> tuple[list[dict[str, Any]], list[str]]:
    """All fourteen return checks, one explicit incomplete identity interval."""
    ids = [str(i) for i in range(14)]
    windows = [
        {
            "validation_event_id": i,
            "ticker": f"T{i}",
            "conditional_return_diagnostic_ready": i != "13",
            "missing_checks": ["provider_identity_interval"] if i == "13" else [],
            "window": {"entry_session_date": "2025-03-03", "exit_session_date": "2025-03-28"},
        }
        for i in ids
    ]
    return windows, ids


def test_missing_case_blocks_entire_sample_and_numeric_price_loader() -> None:
    windows, ids = window_fixture()
    gate = all_case_return_gate(windows, ids)
    assert gate["metadata_ready_events"] == 13
    assert not gate["all_case_return_gate"]
    assert not gate["partial_sample_returns_allowed"]
    events = [{"validation_event_id": i} for i in ids]
    with patch("scripts.build_announcement_expanded_returns.pd.read_parquet") as read:
        with pytest.raises(ValueError, match="partial"):
            _complete_returns({}, {}, events, windows)
        read.assert_not_called()
    with pytest.raises(ValueError, match="no readiness subset"):
        all_case_return_gate(windows[:-1], ids[:-1])


def test_full_same_sample_gate_passes_only_when_every_case_is_ready() -> None:
    windows, ids = window_fixture()
    windows[-1]["conditional_return_diagnostic_ready"] = True
    windows[-1]["missing_checks"] = []
    assert all_case_return_gate(windows, ids)["all_case_return_gate"]


def test_readiness_label_alone_cannot_override_an_explicit_evidence_gap() -> None:
    windows, ids = window_fixture()
    windows[-1]["conditional_return_diagnostic_ready"] = True
    with pytest.raises(ValueError, match="explicit missing checks"):
        all_case_return_gate(windows, ids)


def test_supplement_queries_exact_missing_window_no_adjusted_prices_or_replacements() -> None:
    windows, ids = window_fixture()
    events = [{"validation_event_id": i, "ticker": f"T{i}"} for i in ids]
    audits = {
        "T13": pd.DataFrame(
            {"session_date": ["2025-03-03", "2025-03-04"], "issuer_source_sha256": [None, "a" * 64]}
        )
    }
    plan = supplementation_plan(events, windows, audits)
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert job["missing_issuer_sessions"] == ["2025-03-03"]
    assert job["identity_query_dates"] == ["2025-03-03", "2025-03-28"]
    assert job["price_query"]["adjusted"] is False
    assert plan["network_requests_made"] == 0
    assert not plan["sample_replacements_allowed"]


def test_fourteen_case_industry_metadata_cannot_weaken_twenty_event_gate() -> None:
    cases = selection_fixture()[0]
    rows = [
        {
            "validation_event_id": c["validation_event_id"],
            "cik": c["cik"],
            "industry_ready": True,
            "sic_2digit": "35",
            "industry_available_at": "2025-03-02T12:00:00Z",
        }
        for c in cases
    ]
    result = prior_industry_summary(cases, rows)
    assert result["prior_industry_matches"] == 14
    assert result["minimum_events"] == 20
    assert not result["industry_adjusted_ic_computed"]
    rows[-1]["industry_available_at"] = cases[-1]["earliest_disclosure_at"]
    with pytest.raises(ValueError, match="strictly prior"):
        prior_industry_summary(cases, rows)


@pytest.mark.parametrize(
    "field",
    [
        "network_allowed",
        "all_frozen_cases_required_for_returns",
        "general_scope_gate_cleared",
        "sample_replacement_allowed",
        "hyperparameter_search_allowed",
        "market_features_in_forecast",
        "first_public_certified",
        "trading_admission",
        "inferential_tests_allowed",
        "strategy_pnl_allowed",
        "overwrite_originals",
    ],
)
def test_unsupported_contract_fails_before_any_private_source_read(
    field: str, tmp_path: Path
) -> None:
    config = json.loads(Path("docs/announcement_expanded_returns_v1.json").read_bytes())
    config[field] = not config[field]
    with pytest.raises(ValueError, match="weakened"):
        validate_contract(config)
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="weakened"):
        build_expanded_returns(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
