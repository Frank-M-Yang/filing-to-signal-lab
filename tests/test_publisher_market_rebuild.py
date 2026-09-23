"""Offline controls for pre-release features and metadata-only window coverage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.build_publisher_market_rebuild import build_rebuild, read_pre_event_bars
from sec_data.execution_contract import execution_window
from sec_data.publisher_market_rebuild import (
    BAR_METADATA_COLUMNS,
    BAR_VALUE_COLUMNS,
    audit_revised_window,
    compare_market_features,
    interval_evidence_checks,
    rebuild_pre_event_state,
)
from sec_data.validation_market_state import MARKET_STATE_FEATURES


def test_missing_market_features_remain_missing_not_zero() -> None:
    old = dict.fromkeys(MARKET_STATE_FEATURES, None)
    new = dict.fromkeys(MARKET_STATE_FEATURES, np.nan)
    old[MARKET_STATE_FEATURES[0]] = 0.2
    new[MARKET_STATE_FEATURES[0]] = 0.3
    new[MARKET_STATE_FEATURES[1]] = 0.4
    old[MARKET_STATE_FEATURES[2]] = 0.5
    result = compare_market_features(old, new)
    assert result.change.iloc[0] == pytest.approx(0.1)
    assert result.change.iloc[1:].isna().all()
    assert pd.isna(result.old_filing_proxy_value.iloc[1])
    assert result.rebuilt_publisher_value.iloc[1] == 0.4
    assert pd.isna(result.rebuilt_publisher_value.iloc[2])
    assert old[MARKET_STATE_FEATURES[1]] is None


def test_nonnumeric_market_feature_is_not_silently_made_missing() -> None:
    values = dict.fromkeys(MARKET_STATE_FEATURES, "not_a_number")
    with pytest.raises(ValueError):
        compare_market_features(values, values)


@pytest.fixture()
def inputs() -> dict[str, Any]:
    dates = pd.bdate_range("2025-06-02", "2025-12-31").difference(
        pd.DatetimeIndex(["2025-07-04", "2025-11-27"])
    )
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
        "price_target_id": "p",
        "benchmark_target_id": "b",
        "security_id": "stable",
        "security_id_kind": "share_class_figi",
        "announcement_at": "2025-11-03T21:08:00+00:00",
    }
    bars = []
    for ticker in ("TEST", "SPY"):
        n = np.arange(len(dates))
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "session_date": dates.strftime("%Y-%m-%d"),
                "close_unadjusted": 100 + n * 0.1 + np.sin(n * (0.3 if ticker == "TEST" else 0.4)),
                "volume": 1000 + n,
                "source_sha256": "b" * 64,
            }
        )
        bars.append(frame)
    window = execution_window(event["announcement_at"], event["exchange_mic"], sessions)
    mapping = {
        **{
            k: event[k]
            for k in (
                "cik",
                "ticker",
                "exchange_mic",
                "price_target_id",
                "security_id",
                "security_id_kind",
            )
        },
        "valid_from_inclusive": "2025-06-01T00:00:00-04:00",
        "valid_to_exclusive": "2026-01-01T00:00:00-05:00",
        "mapping_method": "synthetic",
        "ticker_events_endpoint_experimental": True,
        **{
            k: "c" * 64
            for k in (
                "source_snapshot_sha256",
                "end_source_snapshot_sha256",
                "ticker_events_source_snapshot_sha256",
            )
        },
    }
    action = {
        "price_target_id": "p",
        "cik": event["cik"],
        "selected_ticker": "TEST",
        "selected_exchange_mic": "XNAS",
        "stable_security_id": "stable",
        "price_window_start": "2025-06-01",
        "price_window_end": "2025-12-31",
        **{
            k: True
            for k in (
                "splits_query_complete",
                "dividends_query_complete",
                "corporate_actions_ready",
                "historical_symbol_interval_ready",
            )
        },
    }
    price = {
        "price_target_id": "p",
        "cik": event["cik"],
        "ticker": "TEST",
        "price_window_start": "2025-06-01",
        "price_window_end": "2025-12-31",
        "provider_query_complete": True,
        "parser_rejection_rows": 0,
    }
    listing = {
        **{k: event[k] for k in ("cik", "exchange_mic", "security_id", "security_id_kind")},
        "outcome_cutoff_date": "2026-01-01",
        "listing_outcome_status": "active_at_outcome_cutoff",
        "delisting_date": None,
        "listing_outcome_ready": True,
        "source_snapshot_sha256": "d" * 64,
    }
    queries = pd.DataFrame(
        [
            {
                "query_kind": kind,
                "ticker": "SPY",
                "benchmark_target_id": "b",
                "window_start": "2025-06-01",
                "window_end": "2025-12-31",
                "status_code": 200,
                "sha256": "e" * 64,
            }
            for kind in ("daily_bars", "splits", "dividends")
        ]
    )
    return {
        "event": event,
        "window": window,
        "sessions": sessions,
        "stock": bars[0],
        "benchmark": bars[1],
        "mapping": mapping,
        "actions": action,
        "price_audit": price,
        "listing": listing,
        "queries": queries,
        "benchmark_report": {
            "queries_complete": 3,
            "queries_required": 3,
            "rejections": 0,
            "action_normalization_failures": 0,
            "query_adjusted": False,
            "splits_explicit": True,
            "dividends_explicit": True,
        },
    }


def _state(inputs: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    return rebuild_pre_event_state(
        inputs["event"],
        inputs["stock"].loc[inputs["stock"].session_date.lt("2025-11-03")],
        inputs["benchmark"].loc[inputs["benchmark"].session_date.lt("2025-11-03")],
        pd.DataFrame(),
        inputs["sessions"],
        price_basis="raw_as_traded",
    )


def _window(inputs: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    return audit_revised_window(
        inputs["event"],
        inputs["window"],
        inputs["sessions"],
        inputs["stock"][BAR_METADATA_COLUMNS],
        inputs["benchmark"][BAR_METADATA_COLUMNS],
        inputs["mapping"],
        inputs["actions"],
        inputs["price_audit"],
        inputs["listing"],
        inputs["queries"],
        inputs["benchmark_report"],
    )


def test_market_state_uses_61_consecutive_sessions_and_unchanged_math(inputs: Any) -> None:
    state, audit = _state(inputs)
    assert state["market_state_ready"] and len(audit) == 61
    assert state["market_feature_available_at"] == pd.Timestamp("2025-10-31T20:00:00Z")
    assert state["prior_common_sessions"] == 61
    history = inputs["stock"].set_index("session_date").loc[audit.session_date]
    assert state["pre_price_momentum_5d"] == pytest.approx(
        history.close_unadjusted.iloc[-1] / history.close_unadjusted.iloc[-6] - 1
    )
    assert all(np.isfinite(state[k]) for k in MARKET_STATE_FEATURES)
    original = copy.deepcopy(inputs)
    inputs["stock"].loc[inputs["stock"].session_date.ge("2025-11-03"), "close_unadjusted"] = 1e12
    after, after_audit = _state(inputs)
    assert after == state
    pd.testing.assert_frame_equal(after_audit, audit)
    assert not state["return_admission"]
    assert not original["stock"].equals(inputs["stock"])


def test_release_day_and_utc_midnight_policy(inputs: Any) -> None:
    inputs["event"]["announcement_at"] = "2025-11-04T01:08:00Z"  # Still November 3 in NY.
    state, _ = _state(inputs)
    assert state["market_last_required_session"] == "2025-10-31"
    with pytest.raises(ValueError, match="release-day"):
        rebuild_pre_event_state(
            inputs["event"],
            inputs["stock"],
            inputs["benchmark"],
            pd.DataFrame(),
            inputs["sessions"],
            price_basis="raw_as_traded",
        )


def test_missing_interior_bar_is_not_replaced_by_older_history(inputs: Any) -> None:
    inputs["stock"] = inputs["stock"].loc[inputs["stock"].session_date.ne("2025-10-15")]
    state, audit = _state(inputs)
    assert not state["market_state_ready"]
    assert audit.required_session_present.sum() == 60
    assert all(pd.isna(state[k]) for k in MARKET_STATE_FEATURES)


def test_window_complete_but_not_return_authorized(inputs: Any) -> None:
    result, audit = _window(inputs)
    assert result["window_evidence_ready"] and len(audit) == 20
    assert not result["return_admission"] and not result["forward_price_values_loaded"]
    assert not result["terminal_value_imputed"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_price",
        "wrong_identity",
        "wrong_action_target",
        "incomplete_actions",
        "action_range",
        "mapping_end",
        "benchmark_query",
        "listing_early",
        "delisting_inside",
    ],
)
def test_bad_window_evidence_fails_closed(inputs: Any, defect: str) -> None:
    if defect == "missing_price":
        inputs["stock"] = inputs["stock"].loc[
            inputs["stock"].session_date.ne(inputs["window"]["entry_session_date"])
        ]
    elif defect == "wrong_identity":
        inputs["mapping"]["security_id"] = "different-share-class"
    elif defect == "wrong_action_target":
        inputs["actions"]["price_target_id"] = "other"
    elif defect == "incomplete_actions":
        inputs["actions"]["dividends_query_complete"] = False
    elif defect == "action_range":
        inputs["actions"]["price_window_start"] = "2025-11-10"
    elif defect == "mapping_end":
        inputs["mapping"]["valid_to_exclusive"] = inputs["window"]["exit_at"]
    elif defect == "benchmark_query":
        inputs["queries"] = inputs["queries"].iloc[:-1]
    elif defect == "listing_early":
        inputs["listing"]["outcome_cutoff_date"] = "2025-10-31"
    else:
        inputs["listing"].update(
            listing_outcome_status="inactive_with_delisting_date", delisting_date="2025-11-20"
        )
    result, _ = _window(inputs)
    assert not result["window_evidence_ready"]
    assert result["missing_checks"] and not result["return_admission"]


def test_later_delisting_is_retained_not_survivor_filtered(inputs: Any) -> None:
    inputs["listing"].update(
        listing_outcome_status="inactive_with_delisting_date", delisting_date="2025-12-20"
    )
    assert _window(inputs)[0]["window_evidence_ready"]


def test_forward_numeric_projection_and_changed_schedule_rejected(inputs: Any) -> None:
    with pytest.raises(ValueError, match="metadata only"):
        audit_revised_window(
            inputs["event"],
            inputs["window"],
            inputs["sessions"],
            inputs["stock"],
            inputs["benchmark"][BAR_METADATA_COLUMNS],
            inputs["mapping"],
            inputs["actions"],
            inputs["price_audit"],
            inputs["listing"],
            inputs["queries"],
            inputs["benchmark_report"],
        )
    inputs["window"]["entry_price_type"] = "close"
    with pytest.raises(ValueError, match="execution contract"):
        _window(inputs)


def test_pre_history_needs_its_own_action_coverage(inputs: Any) -> None:
    inputs["actions"]["price_window_start"] = "2025-11-01"
    assert _window(inputs)[0]["window_evidence_ready"]
    checks = interval_evidence_checks(
        inputs["event"],
        "2025-08-01T20:00:00Z",
        "2025-10-31T20:00:00Z",
        inputs["mapping"],
        inputs["actions"],
        inputs["price_audit"],
        inputs["queries"],
        inputs["benchmark_report"],
    )
    assert not checks["issuer_actions_interval"]


def test_parquet_reader_pushes_date_and_target_filters(tmp_path: Path, inputs: Any) -> None:
    path = tmp_path / "bars.parquet"
    frame = inputs["stock"].assign(price_target_id="p")
    frame.loc[frame.session_date.ge("2025-11-03"), "close_unadjusted"] = np.inf
    frame.to_parquet(path, index=False)
    loaded = read_pre_event_bars(path, "price_target_id", "p", "2025-11-03")
    assert set(loaded) == set(BAR_VALUE_COLUMNS)
    assert loaded.session_date.lt("2025-11-03").all()
    assert np.isfinite(loaded.close_unadjusted).all()


def test_tampered_upstream_manifest_stops_before_rebuild(tmp_path: Path) -> None:
    config = json.loads(Path("docs/publisher_market_rebuild_v1.json").read_text())
    config["publisher_root"] = str(tmp_path)
    (tmp_path / "output_manifest.json").write_text("{}")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        build_rebuild(path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_publisher_adapter_uses_local_ratios_and_refuses_unknown_quotes(inputs: Any) -> None:
    raw = inputs["stock"].loc[inputs["stock"].session_date.lt("2025-11-03")].copy()
    benchmark = inputs["benchmark"].loc[inputs["benchmark"].session_date.lt("2025-11-03")]
    expected, _ = _state(inputs)
    actions = pd.DataFrame(
        {
            "execution_date": ["2025-09-03", "2025-10-03"],
            "split_from": [1.0, 1.0],
            "split_to": [2.0, 3.0],
            "historical_adjustment_factor": [0.00001, 0.00003],
        }
    )
    for day, ratio in (("2025-09-03", 2), ("2025-10-03", 3)):
        mask = raw.session_date.lt(day)
        raw.loc[mask, "close_unadjusted"] *= ratio
        raw["volume"] = raw.volume.astype(float)
        raw.loc[mask, "volume"] /= ratio
    rebuilt, _ = rebuild_pre_event_state(
        inputs["event"], raw, benchmark, actions, inputs["sessions"], price_basis="raw_as_traded"
    )
    np.testing.assert_allclose(
        [rebuilt[f] for f in MARKET_STATE_FEATURES],
        [expected[f] for f in MARKET_STATE_FEATURES],
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="Unknown quote"):
        rebuild_pre_event_state(
            inputs["event"],
            raw,
            benchmark,
            actions,
            inputs["sessions"],
            price_basis="diagnostic_quote_no_splits",
        )
