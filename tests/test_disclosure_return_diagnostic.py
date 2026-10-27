"""Two-clock return diagnostic tests (synthetic bars, calendar, and features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sec_data.disclosure_return_diagnostic import (
    DisclosureReturnDiagnosticConfig,
    DisclosureReturnDiagnosticError,
    calculate_disclosure_return_diagnostic,
)

SHA = "b" * 64
DATES = pd.bdate_range("2025-01-02", "2025-12-31")
ISSUERS = ["0000000001", "0000000002", "0000000003"]
EVENTS_PER_ISSUER = 8
CONFIG = DisclosureReturnDiagnosticConfig(minimum_complete_events=20, bootstrap_replications=1_000)


def _sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange_mic": "XNAS",
            "session_date": [d.strftime("%Y-%m-%d") for d in DATES],
            "regular_open_at": [f"{d:%Y-%m-%d}T14:30:00+00:00" for d in DATES],
            "regular_close_at": [f"{d:%Y-%m-%d}T21:00:00+00:00" for d in DATES],
            "source_snapshot_sha256": SHA,
        }
    )


def _bars(seed: int, target: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(DATES))))
    return pd.DataFrame(
        {
            "price_target_id": target,
            "session_date": DATES,
            "open_unadjusted": close * (1.0 + rng.normal(0.0, 0.002, len(DATES))),
            "close_unadjusted": close,
        }
    )


def _cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    ledger_rows, clock_rows, feature_rows = [], [], []
    for issuer_index, cik in enumerate(ISSUERS):
        for event_index in range(EVENTS_PER_ISSUER):
            position = 20 + event_index * 28 + issuer_index * 3
            filing_day = DATES[position]
            # Even events have a release five sessions before the filing; odd events do not.
            lead_sessions = 5 if event_index % 2 == 0 else 0
            disclosure_day = DATES[position - lead_sessions]
            accepted_at = pd.Timestamp(f"{filing_day:%Y-%m-%d}T21:30:00Z")
            disclosed_at = pd.Timestamp(f"{disclosure_day:%Y-%m-%d}T21:10:00Z")
            event_id = f"{cik}-{event_index}"
            accession = f"ACC-{cik}-{event_index}"
            ledger_rows.append(
                {
                    "validation_event_id": event_id,
                    "cik": cik,
                    "accession_number": accession,
                    "ticker": f"T{issuer_index}",
                    "exchange_mic": "XNAS",
                    "accepted_at": accepted_at,
                    "price_target_id": f"price-{cik}",
                    "entry_session_date": DATES[position + 2].strftime("%Y-%m-%d"),
                    "exit_session_date": DATES[position + 21].strftime("%Y-%m-%d"),
                    "market_excess_return_arithmetic": np.nan,
                }
            )
            clock_rows.append(
                {
                    "cik": cik,
                    "accession_number": accession,
                    "report_accepted_at": accepted_at,
                    "earliest_disclosure_at": disclosed_at,
                    "clock_source": "item_2_02_8k" if lead_sessions else "periodic_report",
                }
            )
            # Forecast dependencies: known before the release for even events only.
            dependency = disclosed_at - pd.Timedelta(days=1 if event_index % 2 == 0 else -1)
            feature_rows.append(
                {
                    "validation_event_id": event_id,
                    "gm_level": float(rng.normal(0.4, 0.05)),
                    "gm_delta_yoy": float(rng.normal(0.0, 0.02)),
                    "gm_innovation_raw": float(rng.normal(0.0, 0.01)),
                    "gm_innovation_std": float(rng.normal(0.0, 1.0)),
                    "forecast_feature_cutoff": dependency - pd.Timedelta(days=30),
                    "forecast_training_end": dependency,
                    "forecast_error_history_end": dependency - pd.Timedelta(days=2),
                }
            )
    return pd.DataFrame(ledger_rows), pd.DataFrame(clock_rows), pd.DataFrame(feature_rows)


def _inputs() -> dict[str, pd.DataFrame]:
    ledger, clocks, features = _cohort()
    bars = pd.concat([_bars(10 + i, f"price-{cik}") for i, cik in enumerate(ISSUERS)])
    return {
        "ledger": ledger,
        "clocks": clocks,
        "features": features,
        "sessions": _sessions(),
        "issuer_bars": bars,
        "issuer_actions": pd.DataFrame(
            columns=[
                "price_target_id",
                "effective_at",
                "action_type",
                "cash_amount",
                "currency",
                "split_from",
                "split_to",
                "provider_event_id",
                "source_snapshot_sha256",
            ]
        ),
        "benchmark_bars": _bars(99, "benchmark").drop(columns=["price_target_id"]),
        "benchmark_actions": pd.DataFrame(
            columns=[
                "effective_at",
                "action_type",
                "cash_amount",
                "currency",
                "split_from",
                "split_to",
                "provider_event_id",
                "source_snapshot_sha256",
            ]
        ),
    }


def test_two_clocks_differ_only_where_the_release_is_earlier() -> None:
    inputs = _inputs()
    tables = calculate_disclosure_return_diagnostic(**inputs, config=CONFIG)
    events = tables.events.set_index("validation_event_id")
    even = events.loc["0000000001-0"]
    odd = events.loc["0000000001-1"]
    assert even["filing_status"] == "ready" and even["disclosure_status"] == "ready"
    assert bool(even["window_changed"]) is True and even["entry_sessions_earlier"] == 5
    assert bool(odd["window_changed"]) is False and odd["entry_sessions_earlier"] == 0
    assert odd["disclosure_market_excess_return_arithmetic"] == pytest.approx(
        odd["filing_market_excess_return_arithmetic"]
    )
    assert even["disclosure_market_excess_return_arithmetic"] != pytest.approx(
        even["filing_market_excess_return_arithmetic"]
    )
    # Manual check of one filing-window return: open at entry, close at exit, no actions.
    bars = inputs["issuer_bars"].set_index(["price_target_id", "session_date"])
    entry = bars.loc[("price-0000000001", pd.Timestamp(even["filing_entry_session_date"]))]
    exit_bar = bars.loc[("price-0000000001", pd.Timestamp(even["filing_exit_session_date"]))]
    expected = exit_bar["close_unadjusted"] / entry["open_unadjusted"] - 1.0
    assert even["filing_issuer_total_return"] == pytest.approx(expected)
    assert bool(even["innovation_knowable_at_disclosure"]) is True
    assert bool(odd["innovation_knowable_at_disclosure"]) is False
    report = tables.report
    assert report["same_sample_events"] == 24 and report["window_changed_events"] == 12
    assert report["innovation_knowable_at_disclosure"] == 12
    assert set(report["by_analysis"]) == {"same_sample", "innovation_knowable", "window_changed"}
    assert report["by_analysis"]["innovation_knowable"]["status"] == "below_minimum_sample"
    same = report["by_analysis"]["same_sample"]
    assert same["events"] == 24
    assert "disclosure_minus_filing_ic" in same["gm_innovation_std"]
    assert report["validation_evidence"] is False and report["release_value_assumption"] is True


def test_ledger_consistency_is_enforced() -> None:
    inputs = _inputs()
    first = calculate_disclosure_return_diagnostic(**inputs, config=CONFIG).events
    # Fill the ledger with the recomputed filing-window returns: everything must match.
    filled = inputs["ledger"].merge(
        first[["validation_event_id", "filing_market_excess_return_arithmetic"]],
        on="validation_event_id",
    )
    filled["market_excess_return_arithmetic"] = filled.pop("filing_market_excess_return_arithmetic")
    consistent = calculate_disclosure_return_diagnostic(
        **{**inputs, "ledger": filled}, config=CONFIG
    )
    assert consistent.report["ledger_consistency"]["events_matching_ledger_return"] == 24
    perturbed = filled.copy()
    perturbed.loc[0, "market_excess_return_arithmetic"] += 1e-6
    with pytest.raises(DisclosureReturnDiagnosticError):
        calculate_disclosure_return_diagnostic(**{**inputs, "ledger": perturbed}, config=CONFIG)


def test_missing_bar_excludes_the_event_from_the_same_sample() -> None:
    inputs = _inputs()
    bars = inputs["issuer_bars"]
    events = calculate_disclosure_return_diagnostic(**inputs, config=CONFIG).events
    target = events.set_index("validation_event_id").loc["0000000002-2"]
    trimmed = bars.loc[
        ~(
            bars["price_target_id"].eq("price-0000000002")
            & bars["session_date"].eq(pd.Timestamp(target["disclosure_entry_session_date"]))
        )
    ]
    tables = calculate_disclosure_return_diagnostic(
        **{**inputs, "issuer_bars": trimmed}, config=CONFIG
    )
    row = tables.events.set_index("validation_event_id").loc["0000000002-2"]
    assert row["disclosure_status"] == "issuer_entry_bar_missing_or_duplicated"
    assert bool(row["same_sample_ready"]) is False
    assert tables.report["same_sample_events"] == 23


def test_rejects_clock_table_that_disagrees_with_the_ledger() -> None:
    inputs = _inputs()
    with pytest.raises(DisclosureReturnDiagnosticError):
        DisclosureReturnDiagnosticConfig(minimum_complete_events=10).validate()
    bad = inputs["clocks"].copy()
    bad.loc[0, "report_accepted_at"] = bad.loc[0, "report_accepted_at"] + pd.Timedelta(hours=1)
    with pytest.raises(DisclosureReturnDiagnosticError):
        calculate_disclosure_return_diagnostic(**{**inputs, "clocks": bad}, config=CONFIG)
    later = inputs["clocks"].copy()
    later.loc[0, "earliest_disclosure_at"] = later.loc[0, "report_accepted_at"] + pd.Timedelta(
        hours=1
    )
    with pytest.raises(DisclosureReturnDiagnosticError):
        calculate_disclosure_return_diagnostic(**{**inputs, "clocks": later}, config=CONFIG)
