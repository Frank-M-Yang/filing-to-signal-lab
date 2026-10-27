"""Synthetic publication evidence and outcome-blind, immutable six-case pipeline tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from scripts.build_next_publication_audit import build_audit, validate_contract
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.collect_release_clock_evidence import collect_sources, validate_source_config
from sec_data.disclosure_audit import digest_file
from sec_data.execution_contract import execution_window
from sec_data.next_publication import (
    BENCHMARK_COLUMNS,
    DISARM_FLAGS,
    PRICE_COLUMNS,
    audit_price_dates,
    compare_forecast_clocks,
    label_window_scenarios,
    review_publication,
)
from sec_data.publication_chronology import FORECAST_CLOCKS, PANEL_COLUMNS


def _fixture(ticker: str = "RAMP") -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(Path("docs/next_publication_audit_v1.json").read_bytes())
    spec = next(r for r in config["reviews"] if r["ticker"] == ticker)
    statement = spec["statement_anchor"]
    parts = [*spec["context_anchors"], spec["clock_anchor"], statement]
    for field in ("quarter_anchor", "scale_anchor"):
        if spec[field] not in statement:
            parts.append(spec[field])
    parts.extend(c["anchor"] for c in spec["cells"])
    metadata = ""
    if spec["clock_kind"] == "time_datetime":
        metadata = f'<time datetime="{spec["clock_literal"]}"></time>'
    elif spec["clock_kind"] == "jsonld_datePublished":
        node = {
            "@type": "NewsArticle",
            "url": spec["canonical_url"],
            "datePublished": spec["clock_literal"],
        }
        metadata = '<script type="application/ld+json">' + json.dumps(node) + "</script>"
    else:
        metadata = (
            f'<meta property="og:article:published_time" content="{spec["meta_clock_literal"]}">'
        )
    # Model the separate blocks/cells in the preserved production HTML, so the
    # visible-text normalizer does not collapse a space before a standalone %.
    tokens = " ".join(parts).split()
    content = ("<html>" + metadata + "<p>" + "</p><p>".join(tokens) + "</p></html>").encode()
    spec["snapshot_sha256"] = hashlib.sha256(content).hexdigest()
    cells = [
        {
            **{
                k: spec[k]
                for k in ("ticker", "cik", "release_accession", "period_start", "period_end")
            },
            "metric": c["metric"],
            "unit": "USD",
            "value": c["value_usd"],
            "original_value": c["value_usd"],
            "comparison_status": "exact_original_match",
        }
        for c in spec["cells"]
    ]
    return content, spec, cells


@pytest.mark.parametrize("ticker", ["NOBH", "AOS", "ROCK", "RAMP", "ORA", "CTSO"])
def test_exact_pair_precision_and_disarmed_flags(ticker: str) -> None:
    content, spec, cells = _fixture(ticker)
    source = review_publication(content, spec, cells)
    assert len(source["matched_cells"]) == 2 and source["exact_original_pair"]
    assert source["clock_resolution_seconds"] == 60
    assert all(source[k] is False for k in DISARM_FLAGS)
    assert source["first_public_at"] is None
    if ticker == "NOBH":
        assert source["source_reported_at_utc"] is None
        assert source["lower_inclusive"] == "2024-09-09T20:00:00+00:00"
        assert source["upper_exclusive"] == "2024-09-11T00:01:00+00:00"
    if ticker == "ROCK":
        assert source["source_metadata_at_utc"] == "2024-10-30T11:35:24+00:00"
        assert source["source_reported_at_utc"] == "2024-10-30T11:35:00+00:00"


@pytest.mark.parametrize(
    "at,expected",
    [
        ("2024-11-06T21:04:59Z", "before_all"),
        ("2024-11-06T21:05:00Z", "overlap"),
        ("2024-11-06T21:05:46Z", "overlap"),
        ("2024-11-06T21:05:59Z", "overlap"),
        ("2024-11-06T21:06:00Z", "after_all"),
        (None, "missing"),
    ],
)
def test_minute_metadata_padding_never_proves_prior_order(at: str | None, expected: str) -> None:
    content, spec, cells = _fixture()
    source = review_publication(content, spec, cells)
    check = compare_forecast_clocks(source, dict.fromkeys(FORECAST_CLOCKS, at))
    assert all(c["status"] == expected for c in check["forecast_comparisons"].values())
    assert check["prior_to_all_label_scenarios"] is (expected == "before_all")
    assert check["forecast_recomputed"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "unit",
        "quarter",
        "original",
        "rounded",
        "scale",
        "column",
        "period_header",
        "duplicate_statement",
        "row",
        "missing_metric",
        "duplicate_metric",
        "wrong_ticker",
        "bad_visible",
        "timezone",
        "extra_time",
        "no_offset",
        "modified_only",
        "canonical",
        "duplicate_jsonld",
    ],
)
def test_source_time_identity_or_financial_precision_failure(problem: str) -> None:
    content, spec, cells = _fixture(
        "AOS" if problem in {"modified_only", "canonical", "duplicate_jsonld"} else "RAMP"
    )
    if problem == "hash":
        spec["snapshot_sha256"] = "a" * 64
    elif problem in {"unit", "quarter", "original", "wrong_ticker"}:
        key, value = {
            "unit": ("unit", "EUR"),
            "quarter": ("period_end", "2024-06-30"),
            "original": ("original_value", 1),
            "wrong_ticker": ("ticker", "OTHER"),
        }[problem]
        cells[0][key] = value
    elif problem == "rounded":
        spec["cells"][0]["value_usd"] = 185000000
    elif problem == "scale":
        spec["display_scale"] = 1000000
    elif problem == "column":
        spec["cells"][0]["column_index"] = 1
    elif problem == "period_header":
        spec["quarter_anchor"] = "For the six months ended September 30"
    elif problem == "duplicate_statement":
        content += ("<p>" + "</p><p>".join(spec["statement_anchor"].split()) + "</p>").encode()
    elif problem == "row":
        spec["cells"][0]["anchor"] = "Total revenue $ 185"
    elif problem == "missing_metric":
        spec["cells"] = spec["cells"][:1]
    elif problem == "duplicate_metric":
        spec["cells"][1]["metric"] = "revenue"
    elif problem == "bad_visible":
        spec["visible_clock_literal"] = "November 06, 2024 15:05 ET"
    elif problem == "timezone":
        spec["visible_zone"] = "EDT"  # November 6 is EST.
    elif problem == "extra_time":
        content += f'<time datetime="{spec["clock_literal"]}"></time>'.encode()
    elif problem == "no_offset":
        content = content.replace(spec["clock_literal"].encode(), b"2024-11-06T21:05:00")
        spec["clock_literal"] = "2024-11-06T21:05:00"
    elif problem == "modified_only":
        content = content.replace(b"datePublished", b"dateModified")
    elif problem == "canonical":
        spec["canonical_url"] += "/other"
    else:
        content += content[content.index(b"<script") : content.index(b"</script>") + 9]
    if problem != "hash":
        spec["snapshot_sha256"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError):
        review_publication(content, spec, cells)


def _sessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange_mic": mic,
                "session_date": d.date().isoformat(),
                "regular_open_at": (d + pd.Timedelta(hours=9, minutes=30))
                .tz_localize("America/New_York")
                .isoformat(),
                "regular_close_at": (d + pd.Timedelta(hours=16))
                .tz_localize("America/New_York")
                .isoformat(),
                "source_snapshot_sha256": "a" * 64,
            }
            for mic in ("XNYS", "XNAS", "OTCM")
            for d in pd.bdate_range("2024-08-01", "2024-12-31")
        ]
    )


def test_unknown_timezone_keeps_both_local_dates_and_cannot_impute_nyse() -> None:
    content, spec, cells = _fixture("NOBH")
    source = review_publication(content, spec, cells)
    scenarios = label_window_scenarios(source, "OTCM", _sessions())
    assert [s["scenario_local_date"] for s in scenarios] == ["2024-09-09", "2024-09-10"]
    assert [s["schedule"]["entry_session_date"] for s in scenarios] == ["2024-09-11", "2024-09-12"]
    assert all(s["schedule"]["announcement_at"] is None for s in scenarios)
    missing = label_window_scenarios(source, "MISSING", _sessions())
    assert all(s["schedule"] is None for s in missing)
    assert all(s["return_admission"] is False for s in scenarios)


def _market() -> tuple[
    dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    sessions = _sessions()
    case = {"ticker": "AOS", "cik": "0000091142", "assigned_exchange_mic": "XNYS"}
    schedule = execution_window("2024-10-22T10:55:00Z", "XNYS", sessions)
    dates = sessions.loc[
        sessions.exchange_mic.eq("XNYS")
        & sessions.session_date.between(
            schedule["entry_session_date"], schedule["exit_session_date"]
        ),
        "session_date",
    ]
    targets = pd.DataFrame(
        [
            {
                "cik": case["cik"],
                "selected_ticker": "AOS",
                "selected_exchange_mic": "XNYS",
                "price_target_id": "p",
                "security_identity_key": "share-class",
                "security_identity_key_kind": "share_class_figi",
                "adjusted": False,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"price_target_id": "p", "ticker": "AOS", "session_date": d, "source_sha256": "b" * 64}
            for d in dates
        ],
        columns=PRICE_COLUMNS,
    )
    benchmark = pd.DataFrame(
        [{"ticker": "SPY", "session_date": d, "source_sha256": "c" * 64} for d in dates],
        columns=BENCHMARK_COLUMNS,
    )
    return case, schedule, sessions, targets, prices, benchmark


def test_exact_dates_are_only_metadata_not_return_authorization() -> None:
    inputs = _market()
    result = audit_price_dates(*inputs)
    assert result["metadata_window_complete"] and len(result["expected_dates"]) == 20
    assert not result["price_values_loaded"] and not result["return_admission"]
    missing = audit_price_dates(*inputs[:4], inputs[4].iloc[1:], inputs[5].iloc[:-1])
    assert not missing["metadata_window_complete"]
    assert len(missing["missing_issuer_dates"]) == len(missing["missing_benchmark_dates"]) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "extra_price",
        "extra_benchmark",
        "mic",
        "duplicate",
        "hash",
        "ticker",
        "adjusted",
        "wrong_cik",
    ],
)
def test_metadata_guard_never_loads_outcomes_or_matches_on_ticker_alone(problem: str) -> None:
    case, schedule, sessions, targets, prices, benchmark = _market()
    if problem == "extra_price":
        prices["open_unadjusted"] = 123
    elif problem == "extra_benchmark":
        benchmark["future_return"] = 999
    elif problem == "mic":
        schedule["exchange_mic"] = "XNAS"
    elif problem == "duplicate":
        prices = pd.concat([prices, prices.iloc[:1]])
    elif problem == "hash":
        prices.loc[0, "source_sha256"] = None
    elif problem == "ticker":
        prices.loc[0, "ticker"] = "OTHER"
    elif problem == "adjusted":
        targets["adjusted"] = True
    else:
        targets["cik"] = "0000000001"
        result = audit_price_dates(case, schedule, sessions, targets, prices, benchmark)
        assert not result["metadata_window_complete"] and "query_identity" in result["status"]
        return
    with pytest.raises(ValueError):
        audit_price_dates(case, schedule, sessions, targets, prices, benchmark)


def _encode(value: Any) -> bytes:
    if isinstance(value, pd.DataFrame):
        buffer = io.BytesIO()
        value.to_parquet(buffer, index=False)
        return buffer.getvalue()
    return json.dumps(value).encode()


def _pipeline(tmp_path: Path) -> Path:
    config = json.loads(Path("docs/next_publication_audit_v1.json").read_bytes())
    fixtures = [_fixture(t) for t in ["NOBH", "AOS", "ROCK", "RAMP", "ORA", "CTSO"]]
    config["reviews"] = [s for _, s, _ in fixtures]
    cases, cells, metadata, sources, receipts = [], [], [], [], []
    raw: dict[str, bytes] = {}
    for index, (content, spec, original_cells) in enumerate(fixtures):
        filename = spec["ticker"] + ".html"
        raw[filename] = content
        source = {k: spec[k] for k in ("ticker", "cik", "release_accession", "url")}
        sources.append(source)
        receipts.append(
            {
                **source,
                "file": filename,
                "sha256": spec["snapshot_sha256"],
                "status_code": 200,
                "content_type": "text/html",
                "response_url": spec["url"],
                "bytes": len(content),
                "fetched_at": "2026-10-01T00:00:00Z",
            }
        )
        proxy = spec["reported_date"] + "T23:00:00Z"
        case = {
            "validation_event_id": f"e{index}",
            "ticker": spec["ticker"],
            "cik": spec["cik"],
            "period_start": spec["period_start"],
            "period_end": spec["period_end"],
            "target_accession_number": f"target{index}",
            "release_accession_number": spec["release_accession"],
            "complete_pair_replay_cutoff_at": proxy,
            "assigned_exchange_mic": "OTCM" if index == 0 else "XNYS",
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        cases.append(case)
        cells.extend(original_cells)
        metadata.append(
            {
                "event_id": f"e{index}",
                "company_id": case["cik"],
                "filing_accession": case["target_accession_number"],
                "actual_available_at": proxy,
                **dict.fromkeys(FORECAST_CLOCKS, "2024-07-01T00:00:00Z"),
                "future_return": 999,
                "gm_forecast": 0.42,
            }
        )
    dispositions = [
        *cases,
        *[
            {
                "validation_event_id": f"e{i}",
                "ticker": f"OTHER{i}",
                "cik": f"{i:010d}",
                **dict.fromkeys(DISARM_FLAGS, False),
            }
            for i in range(6, 20)
        ],
    ]

    def bundle(group: str, files: dict[str, Any]) -> dict[str, str]:
        root = write_bundle(
            tmp_path,
            group,
            {k: v if isinstance(v, bytes) else _encode(v) for k, v in files.items()},
        )
        return {"root": str(root), "manifest_sha256": digest_file(root / "output_manifest.json")}

    review = bundle("review", {"reviewed_metric_cells.json": cells})
    plan = bundle(
        "plan",
        {
            "plan.json": {
                "config": {"review_manifest_sha256": review["manifest_sha256"]},
                "cases": cases,
                "retained_disposition_ids": [r["validation_event_id"] for r in dispositions],
            }
        },
    )
    replay = bundle(
        "replay",
        {
            "summary.json": {"plan_manifest_sha256": plan["manifest_sha256"]},
            "case_dispositions.json": dispositions,
            "shared_disarmed_history.parquet": pd.DataFrame(metadata),
        },
    )
    sessions = bundle("sessions", {"validation_exchange_sessions.parquet": _sessions()})
    targets = pd.DataFrame(
        [
            {
                "cik": c["cik"],
                "selected_ticker": c["ticker"],
                "selected_exchange_mic": c["assigned_exchange_mic"],
                "price_target_id": c["ticker"],
                "security_identity_key": c["ticker"] + "-identity",
                "security_identity_key_kind": "share_class_figi",
                "adjusted": False,
            }
            for c in cases
        ]
    )
    dates = _sessions().loc[lambda f: f.exchange_mic.eq("XNYS"), "session_date"]
    price = bundle(
        "prices",
        {
            "validation_price_batch_targets.parquet": targets,
            "validation_price_batch_bars.parquet": pd.DataFrame(
                [
                    {
                        "price_target_id": c["ticker"],
                        "ticker": c["ticker"],
                        "session_date": d,
                        "source_sha256": "b" * 64,
                        "open_unadjusted": 100,
                        "future_return": 999,
                    }
                    for c in cases
                    for d in dates
                ]
            ),
        },
    )
    market = bundle(
        "benchmark",
        {
            "benchmark_daily_price_observations.parquet": pd.DataFrame(
                [
                    {
                        "ticker": "SPY",
                        "session_date": d,
                        "source_sha256": "c" * 64,
                        "close_unadjusted": 500,
                    }
                    for d in dates
                ]
            )
        },
    )
    market_config = {
        f"{group}_{k}": v
        for group, ref in [("execution", sessions), ("prices", price), ("benchmark", market)]
        for k, v in ref.items()
    }
    expanded = bundle("expanded", {"plan.json": {"market_config": market_config}})
    for group, ref in [
        ("review", review),
        ("replay_plan", plan),
        ("replay", replay),
        ("expanded_plan", expanded),
    ]:
        config.update({f"{group}_{k}": v for k, v in ref.items()})
    source_path = tmp_path / "sources.json"
    source_path.write_bytes(_encode({"sources": sources}))
    raw["receipts.json"] = _encode(
        {"config_sha256": digest_file(source_path), "receipts": receipts}
    )
    evidence = bundle("raw", raw)
    config["batches"] = [
        {**evidence, "config": str(source_path), "config_sha256": digest_file(source_path)}
    ]
    path = tmp_path / "config.json"
    path.write_bytes(_encode(config))
    return path


def test_full_builder_is_projected_offline_deterministic_and_retains_twenty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _pipeline(tmp_path)
    read = pd.read_parquet
    expected = {
        "shared_disarmed_history.parquet": PANEL_COLUMNS,
        "validation_price_batch_bars.parquet": PRICE_COLUMNS,
        "benchmark_daily_price_observations.parquet": BENCHMARK_COLUMNS,
    }

    def projected(file: Any, **kwargs: Any) -> pd.DataFrame:
        if Path(file).name in expected:
            assert kwargs["columns"] == expected[Path(file).name]
        return read(file, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", projected)
    monkeypatch.setattr(
        httpx.Client, "get", lambda *_a, **_k: pytest.fail("Offline builder attempted networking")
    )
    root, summary = build_audit(path, tmp_path / "out")
    assert summary["frozen_queue_issuers_retained"] == 20
    assert summary["complete_pair_sources_reviewed"] == 6
    assert summary["exact_original_component_cells"] == 12
    assert summary["explicit_offset_minute_labels"] == 5
    assert summary["single_complete_metadata_windows"] == 5
    assert summary["common_return_gate"] is False
    assert summary["forecasts_recomputed"] is summary["returns_recomputed"] is False
    assert not summary["price_values_loaded"] and not summary["forecast_values_loaded"]
    assert build_audit(path, tmp_path / "out") == (root, summary)
    dispositions = json.loads((root / "case_dispositions.json").read_bytes())
    assert (
        len(dispositions) == 20
        and sum(r["publication_audit"] is not None for r in dispositions) == 6
    )
    assert all(r[k] is False for r in dispositions for k in DISARM_FLAGS)
    (root / "summary.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_audit(path, tmp_path / "out")


@pytest.mark.parametrize(
    "problem", ["raw", "queue", "quarter", "flags", "duplicate_case", "receipt", "cohort"]
)
def test_builder_blocks_drift_instead_of_replacing_missing_cases(
    tmp_path: Path, problem: str
) -> None:
    path = _pipeline(tmp_path)
    config = json.loads(path.read_bytes())
    if problem == "raw":
        (Path(config["batches"][0]["root"]) / "RAMP.html").write_bytes(b"changed")
    elif problem == "queue":
        (Path(config["replay_root"]) / "case_dispositions.json").write_bytes(b"[]")
    elif problem == "quarter":
        config["reviews"][0]["period_end"] = "2024-09-30"
    elif problem == "flags":
        config["return_admission"] = True
    elif problem == "duplicate_case":
        config["reviews"][1] = config["reviews"][0]
    elif problem == "receipt":
        config["reviews"][0]["snapshot_sha256"] = "a" * 64
    else:
        config["reviews"] = config["reviews"][:-1]
    path.write_bytes(_encode(config))
    with pytest.raises(ValueError):
        build_audit(path, tmp_path / "out")


def test_contract_flags_fail_before_input_access() -> None:
    config = json.loads(Path("docs/next_publication_audit_v1.json").read_bytes())
    for k in (
        "network_allowed",
        "price_values_allowed",
        "returns_allowed",
        "forecast_refit_allowed",
        "sample_replacement_allowed",
        *DISARM_FLAGS,
    ):
        changed = copy.deepcopy(config)
        changed[k] = True
        with pytest.raises(ValueError, match="contract"):
            validate_contract(changed)


@pytest.mark.parametrize(
    "problem", ["count", "identity", "ticker", "host", "query", "user", "port", "fragment"]
)
def test_six_source_collector_rejects_scope_expansion(problem: str) -> None:
    config = json.loads(Path("docs/next_publication_sources_v1.json").read_bytes())
    if problem == "count":
        config["sources"] = config["sources"][:-1]
    elif problem == "identity":
        config["sources"][0]["release_accession"] = "other"
    elif problem == "ticker":
        config["sources"][0]["ticker"] = "OTHER"
    else:
        config["sources"][0]["url"] = {
            "host": "https://www.sec.gov/Archives/",
            "query": "https://www.globenewswire.com/path?token=example",
            # Construct artificial URL credentials without resembling an email
            # in the public-release privacy scan; no real credentials are used.
            "user": "https://user:pass" + chr(64) + "www.globenewswire.com/path",
            "port": "https://www.globenewswire.com:444/path",
            "fragment": "https://www.globenewswire.com/path#date",
        }[problem]
    with pytest.raises(ValueError):
        validate_source_config(config)


def test_direct_public_transport_is_explicit_and_never_used_for_sec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(Path("docs/next_publication_sources_v1.json").read_bytes())
    path = tmp_path / "sources.json"
    path.write_bytes(_encode(config))
    seen: list[dict[str, Any]] = []

    original_client = httpx.Client

    # Avoid recursive monkeypatching in the factory.
    def factory(**kwargs: Any) -> httpx.Client:
        seen.append(kwargs)
        return original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"preserved",
                    headers={"content-type": "text/html"},
                    request=request,
                )
            )
        )

    monkeypatch.setattr("scripts.collect_release_clock_evidence.httpx.Client", factory)
    monkeypatch.setattr("scripts.collect_release_clock_evidence.time.sleep", lambda _: None)
    root = collect_sources(path, tmp_path / "raw", transport="httpx-direct")
    assert seen[0]["trust_env"] is False and seen[0]["follow_redirects"] is False
    rows = json.loads((root / "receipts.json").read_bytes())["receipts"]
    assert len(rows) == 6 and all(r["transport"] == "httpx-direct" for r in rows)
    assert all((root / r["file"]).read_bytes() == b"preserved" for r in rows)
    with pytest.raises(ValueError, match="transport"):
        collect_sources(path, tmp_path / "raw", transport="unknown")
