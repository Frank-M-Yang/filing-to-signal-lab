"""Synthetic evidence gaps, exact calendar denominators and bounded API capture."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pandas as pd
import pytest

from scripts.build_publication_market_bridge import build_requests
from scripts.build_sec_clock_layer import _encode
from scripts.collect_publication_market_gaps import collect_requests, validate_requests
from sec_data.massive_reference import MassiveReferenceClient
from sec_data.next_publication import DISARM_FLAGS
from sec_data.publication_market_bridge import (
    TICKERS,
    audit_case_windows,
    close_common_gate,
    validate_contract,
)
from tests.test_announcement_returns import evidence  # noqa: F401


@pytest.fixture()
def fixture(evidence: dict[str, Any]) -> tuple[Any, ...]:  # noqa: F811
    """One synthetic RAMP case with both matching provider endpoints and full dates."""
    e = evidence
    ticker, security = "RAMP", "BBG000000001"
    case = {
        "ticker": ticker,
        "cik": e["event"]["cik"],
        "validation_event_id": "event",
        "assigned_exchange_mic": "XNAS",
    }
    source = {
        **case,
        "lower_inclusive": "2025-03-03T12:00:00Z",
        "upper_exclusive": "2025-03-03T12:01:00Z",
        "clock_basis": "explicit_minute_label",
        **dict.fromkeys(DISARM_FLAGS, False),
    }
    target = {
        "price_target_id": "price",
        "cik": case["cik"],
        "selected_ticker": ticker,
        "selected_exchange_mic": "XNAS",
        "security_identity_key": security,
        "security_identity_key_kind": "share_class_figi",
        "adjusted": False,
        "selection_uses_return_values": False,
    }
    action = {
        **e["action"],
        "selected_ticker": ticker,
        "security_identity_key": security,
        "selected_provider_security_id": security,
        "selected_share_class_figi": security,
        "selected_composite_figi": "BBG000000002",
        "action_acquisition_order": 1,
        "reference_start_query_complete": True,
        "reference_end_query_complete": True,
        "ticker_events_query_complete": False,
        "historical_symbol_interval_ready": False,
        "corporate_actions_ready": True,
        "returns_computed": False,
    }
    identities = pd.DataFrame(
        [
            {
                "action_target_id": "actions",
                "cik": case["cik"],
                "query_kind": kind,
                "as_of_date": day,
                "ticker": ticker,
                "exchange_mic": "XNAS",
                "security_type": "CS",
                "provider_security_id": security,
                "share_class_figi": security,
                "composite_figi": "BBG000000002",
                "name": "Synthetic",
                "source_sha256": "a" * 64,
            }
            for kind, day in [("reference_start", "2025-03-03"), ("reference_end", "2025-04-30")]
        ]
    )
    tables = {
        "targets": pd.DataFrame([target]),
        "metadata": e["issuer_metadata"].assign(ticker=ticker, price_target_id="price")[
            ["price_target_id", "ticker", "session_date", "source_sha256"]
        ],
        "benchmark_metadata": e["benchmark_metadata"],
        "actions": pd.DataFrame([action]),
        "identities": identities,
        "ticker_events": pd.DataFrame(columns=["action_target_id", "event_date", "new_ticker"]),
        "price_audit": pd.DataFrame([{**e["price"], "ticker": ticker}]),
        "listing": pd.DataFrame([{**e["listing"], "security_id": security}]),
        "queries": e["queries"],
        "sessions": e["sessions"],
    }
    return case, source, tables, e["benchmark_report"]


def test_case_without_price_id_resolves_exact_cik_ticker_mic_and_reuses_no_values(
    fixture: tuple[Any, ...],
) -> None:
    before = copy.deepcopy(fixture[2])
    a, dates, gaps = audit_case_windows(*fixture)
    assert len(a) == 1 and len(dates) == 20 and not gaps
    assert a[0]["conditional_return_diagnostic_ready"]
    assert a[0]["price_target_id"] == "price"
    assert a[0]["window"]["announcement_at"] is None
    assert a[0]["window"]["entry_session_date"] == "2025-03-05"
    assert all(a[0][k] is False for k in DISARM_FLAGS)
    for name, table in before.items():
        pd.testing.assert_frame_equal(fixture[2][name], table)


@pytest.mark.parametrize(
    "mutation", ["numeric", "duplicate", "wrong_id", "wrong_mic", "adjusted", "source_flag"]
)
def test_exact_parent_and_metadata_projections_cannot_be_weakened(
    fixture: tuple[Any, ...], mutation: str
) -> None:
    case, source, tables, _ = fixture
    if mutation == "numeric":
        tables["metadata"]["close"] = 123.0
    elif mutation == "duplicate":
        tables["targets"] = pd.concat([tables["targets"]] * 2)
    elif mutation == "wrong_id":
        case["price_target_id"] = "other"
    elif mutation == "wrong_mic":
        case["assigned_exchange_mic"] = "XNYS"
    elif mutation == "adjusted":
        tables["targets"]["adjusted"] = True
    else:
        source["return_admission"] = True
    with pytest.raises(ValueError):
        audit_case_windows(*fixture)


def test_unknown_zone_retains_two_windows_and_outside_query_is_not_no_trade(
    fixture: tuple[Any, ...],
) -> None:
    _, source, tables, _ = fixture
    source.update(lower_inclusive="2025-03-03T20:00:00Z", upper_exclusive="2025-03-05T00:01:00Z")
    tables["actions"] = tables["actions"].iloc[:0]
    tables["price_audit"]["price_window_start"] = "2025-03-10"
    tables["metadata"] = tables["metadata"].loc[tables["metadata"].session_date.ge("2025-03-10")]
    a, dates, gaps = audit_case_windows(*fixture)
    assert [r["scenario_local_date"] for r in a] == ["2025-03-03", "2025-03-04"]
    assert len(dates) == 40 and len(gaps) == 5
    assert all(r["classification"] == "outside_cached_query_interval" for r in gaps)
    assert all(r["no_trading_halt_or_zero_return_inferred"] for r in gaps)
    assert all(r["missing_cached_action_target"] for r in a)
    assert all(not r["checks"]["issuer_action_query_interval"] for r in a)
    assert all(not r["conditional_return_diagnostic_ready"] for r in a)


@pytest.mark.parametrize(
    ("complete", "rejected", "expected"),
    [
        (True, 0, "no_parsed_bar_despite_complete_query"),
        (False, 0, "incomplete_price_query"),
        (True, 1, "parser_rejection_present"),
    ],
)
def test_inside_query_gaps_are_not_silently_imputed(
    fixture: tuple[Any, ...], complete: bool, rejected: int, expected: str
) -> None:
    tables = fixture[2]
    tables["metadata"] = tables["metadata"].loc[tables["metadata"].session_date.ne("2025-03-10")]
    tables["price_audit"]["provider_query_complete"] = complete
    tables["price_audit"]["parser_rejection_rows"] = rejected
    a, _, gaps = audit_case_windows(*fixture)
    assert not a[0]["conditional_return_diagnostic_ready"]
    assert len(gaps) == 1 and gaps[0]["classification"] == expected


def gate_fixture(fixture: tuple[Any, ...]) -> tuple[Any, ...]:
    """Exactly seventeen arms and six retained synthetic identities, with one blocked case."""
    audit = audit_case_windows(*fixture)[0][0]
    audits, financial = [], []
    for t in sorted(TICKERS):
        row = copy.deepcopy(audit)
        row.update(ticker=t, validation_event_id=t)
        if t == "NOBH":
            row["conditional_return_diagnostic_ready"] = False
            row["checks"]["listing_outcome_interval"] = False
            row["missing_checks"] = ["listing_outcome_interval"]
        audits.append(row)
        for i in range(17):
            financial.append(
                {
                    "ticker": t,
                    "cik": row["cik"],
                    "validation_event_id": t,
                    "scenario_id": str(i),
                    "actual_available_at": "2025-03-03T12:00:00Z",
                    "same_six_common_financial_gate": True,
                    "complete_financial_representations": True,
                    **dict.fromkeys(DISARM_FLAGS, False),
                }
            )
    return audits, financial


def test_all_financial_arms_share_one_gate_not_the_ready_subset(fixture: tuple[Any, ...]) -> None:
    audits, financial = gate_fixture(fixture)
    gate = close_common_gate(audits, financial)
    assert gate["metadata_window_blocked_tickers"] == ["NOBH"]
    assert not gate["all_six_metadata_windows_ready"]
    assert gate["all_102_financial_arms_matched_to_calendar_date"]
    assert not gate["partial_sample_returns_allowed"]
    # JSON canonical sorting must not turn an identical missing-check set into
    # a different research gate simply because dict insertion order changed.
    decoded = [json.loads(_encode(rows)) for rows in (audits, financial)]
    assert close_common_gate(*decoded) == gate


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_case",
        "drop_arm",
        "duplicate_arm",
        "date",
        "cik",
        "naive",
        "promote",
        "readiness",
        "fabricated_clock",
    ],
)
def test_common_gate_rejects_denominator_clock_or_admission_drift(
    fixture: tuple[Any, ...], mutation: str
) -> None:
    audits, financial = gate_fixture(fixture)
    if mutation == "drop_case":
        audits.pop()
    elif mutation == "drop_arm":
        financial.pop()
    elif mutation == "duplicate_arm":
        financial[-1] = financial[-2]
    elif mutation == "date":
        financial[0]["actual_available_at"] = "2025-03-04T12:00:00Z"
    elif mutation == "cik":
        financial[0]["cik"] = "other"
    elif mutation == "naive":
        financial[0]["actual_available_at"] = "2025-03-03T12:00:00"
    elif mutation == "promote":
        financial[0]["model_admission"] = True
    elif mutation == "readiness":
        audits[0]["conditional_return_diagnostic_ready"] = not audits[0][
            "conditional_return_diagnostic_ready"
        ]
    else:
        audits[0]["window"]["announcement_at"] = "2025-03-03T12:00:00Z"
    with pytest.raises(ValueError):
        close_common_gate(audits, financial)


def request_fixture(fixture: tuple[Any, ...]) -> tuple[Any, ...]:
    """Two blocked cases: missing OTC FIGI remains None, not a made-up events URL."""
    audits, evidence_rows, listing_rows = [], [], []
    original = audit_case_windows(*fixture)[0][0]
    for i, ticker in enumerate(["CTSO", "NOBH"]):
        r = copy.deepcopy(original)
        r.update(
            ticker=ticker, validation_event_id=ticker, conditional_return_diagnostic_ready=False
        )
        r["missing_checks"] = ["provider_identity_interval", "issuer_action_query_interval"]
        if ticker == "NOBH":
            r["missing_checks"] += [
                "issuer_all_20_parsed_sessions",
                "issuer_price_query_interval",
                "listing_outcome_interval",
            ]
        audits.append(r)
        evidence_rows.append(
            {
                "cik": r["cik"],
                "selected_ticker": ticker,
                "selected_exchange_mic": r["exchange_mic"],
                "price_target_id": r["price_target_id"],
                "security_identity_key": r["security_id"],
                "security_identity_key_kind": "share_class_figi",
                "selected_composite_figi": "BBG000000002" if i == 0 else None,
                "selection_uses_return_values": False,
                "returns_computed": False,
            }
        )
        # Two synthetic issuers need distinct identities to bind one listing each.
        r["cik"] = evidence_rows[-1]["cik"] = str(i + 1).zfill(10)
        listing_rows.append(
            {"cik": r["cik"], "security_id": r["security_id"], "outcome_cutoff_date": "2026-02-01"}
        )
    return audits, pd.DataFrame(evidence_rows), pd.DataFrame(listing_rows)


def test_twelve_requests_preserve_bounds_and_skip_missing_composite_figi(
    fixture: tuple[Any, ...],
) -> None:
    requests = build_requests(*request_fixture(fixture))
    validate_requests(requests)
    assert len(requests) == 12 and len({r["query_id"] for r in requests}) == 12
    assert not any(r["ticker"] == "NOBH" and r["query_kind"] == "ticker_events" for r in requests)
    assert all(
        r["window_start"] == "2025-03-05" and r["window_end"] == "2025-04-01" for r in requests
    )
    assert all("None" not in r["url"] and "apiKey" not in r["url"] for r in requests)


def test_requests_reject_other_share_class_and_budget_growth(fixture: tuple[Any, ...]) -> None:
    args = request_fixture(fixture)
    args[1].loc[0, "security_identity_key"] = "BBG000000003"
    with pytest.raises(ValueError, match="identity differs"):
        build_requests(*args)
    requests = build_requests(*request_fixture(fixture))
    with pytest.raises(ValueError, match="bounded"):
        validate_requests([*requests, requests[0]])


@pytest.mark.parametrize(
    "mutation", ["url_key", "foreign_url", "invent_figi", "promote", "hash", "adjusted"]
)
def test_capture_validation_runs_before_any_network(
    fixture: tuple[Any, ...], mutation: str
) -> None:
    requests = build_requests(*request_fixture(fixture))
    r = requests[0]
    if mutation == "url_key":
        r["url"] += "&apiKey=not-a-real-key"
    elif mutation == "foreign_url":
        r["url"] = "https://example.org/"
    elif mutation == "invent_figi":
        r["query_kind"] = "ticker_events"
        r["selected_composite_figi"] = None
    elif mutation == "promote":
        r["return_admission"] = True
    elif mutation == "hash":
        r["query_id"] = "a" * 64
    else:
        r["adjusted"] = True
    if mutation != "hash":
        r["query_id"] = hashlib.sha256(
            _encode({k: v for k, v in r.items() if k != "query_id"})
        ).hexdigest()
    with pytest.raises(ValueError):
        validate_requests(requests)


def capture_client(sender: Any, clock: dict[str, float]) -> MassiveReferenceClient:
    """An injected fixed clock permits fast tests of the real client throttle."""
    return MassiveReferenceClient(
        "test-not-real-credential",
        sender=sender,
        max_retries=0,
        request_interval_seconds=12.1,
        monotonic=lambda: clock["time"],
        sleep=lambda seconds: clock.update(time=clock["time"] + seconds),
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_exact_bytes_saved_before_decode_and_verified_resume_without_http(
    fixture: tuple[Any, ...],
    tmp_path: Path,
) -> None:
    requests = build_requests(*request_fixture(fixture))[:2]
    content = b'{ "status": "OK", "results": [], "next_url": "https://api.massive.com/next" }\n'
    call_times: list[float] = []
    clock = {"time": 0.0}

    def respond(request: httpx.Request) -> httpx.Response:
        scheme, credential = request.headers["authorization"].split()
        assert scheme == "Bearer" and credential == "test-not-real-credential"
        assert "credential" not in str(request.url)
        call_times.append(clock["time"])
        return httpx.Response(200, content=content, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as sender:
        client = capture_client(sender, clock)
        root, s = collect_requests(requests, tmp_path, client, source_manifest_sha256="a" * 64)
        assert s["requests_made"] == 2 and s["responses_saved"] == 2
        assert call_times[1] - call_times[0] >= 12.1
        assert not s["all_case_return_gate"] and not s["pagination_followed"]
        assert all(
            r["pagination_present"] for r in json.loads((root / "envelopes.json").read_bytes())
        )
        assert all(
            p.read_bytes() == content for p in (tmp_path / "responses").glob("*/*/response.json")
        )
        with patch.object(client, "get", side_effect=AssertionError("no network on resume")):
            _, resumed = collect_requests(
                requests, tmp_path, client, source_manifest_sha256="a" * 64
            )
        assert resumed["verified_cache_reused"] == 2 and resumed["requests_made"] == 0
        assert "test-not-real-credential" not in "".join(
            p.read_text() for p in tmp_path.rglob("*.json")
        )


@pytest.mark.parametrize(
    ("status", "reason"), [(401, "authentication_failed"), (403, "entitlement_denied")]
)
def test_auth_or_entitlement_stops_batch_without_fake_success(
    fixture: tuple[Any, ...], tmp_path: Path, status: int, reason: str
) -> None:
    requests = build_requests(*request_fixture(fixture))[:2]
    clock = {"time": 0.0}
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, request=r))
    ) as sender:
        root, summary = collect_requests(
            requests, tmp_path, capture_client(sender, clock), source_manifest_sha256="a" * 64
        )
    assert summary["requests_made"] == 1 and summary["responses_saved"] == 0
    assert summary["blocking_reason"] == reason and summary["failed_queries"] == 1
    assert json.loads((root / "receipts.json").read_bytes())[0][
        "raw_error_response_not_available_from_client"
    ]


def test_changed_cache_is_not_overwritten_or_reused(
    fixture: tuple[Any, ...], tmp_path: Path
) -> None:
    requests = build_requests(*request_fixture(fixture))[:1]
    clock = {"time": 0.0}
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"status": "OK", "results": []}, request=r)
        )
    ) as sender:
        client = capture_client(sender, clock)
        collect_requests(requests, tmp_path, client, source_manifest_sha256="a" * 64)
        path = next((tmp_path / "responses").glob("*/*/response.json"))
        path.write_bytes(b"changed-for-test")
        with patch.object(client, "get", side_effect=AssertionError("must verify before HTTP")):
            with pytest.raises(ValueError):
                collect_requests(requests, tmp_path, client, source_manifest_sha256="a" * 64)
        assert path.read_bytes() == b"changed-for-test"


@pytest.mark.parametrize(
    "flag",
    [
        "returns_allowed",
        "price_values_allowed",
        "network_allowed",
        "forecast_refit_allowed",
        "partial_sample_returns_allowed",
        "model_admission",
    ],
)
def test_contract_remains_offline_and_disarmed(flag: str) -> None:
    config = json.loads(Path("docs/publication_market_bridge_v1.json").read_bytes())
    validate_contract(config)
    config[flag] = True
    with pytest.raises(ValueError):
        validate_contract(config)
