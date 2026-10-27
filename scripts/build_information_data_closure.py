"""Reuse preserved SEC headers and bind three cached market cases; no download or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, load_cached_submissions, verify_file
from sec_data.information_complexity import FINANCIAL, FINANCIAL_CLOCKS, STATE, run_experiment
from sec_data.information_data_closure import (
    CORE_TICKERS,
    VERSION,
    cached_industries,
    industry_closure,
    missing_market_plan,
    receipt_audit,
)


def build_closure(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Produce a separate immutable cache-repair artifact without promoting source admission."""
    c = json.loads(config_path.read_bytes())
    if (
        c.get("version") != VERSION
        or tuple(c["core_tickers"]) != CORE_TICKERS
        or any(
            c.get(k) is not False
            for k in (
                "network_allowed",
                "model_fitting_allowed",
                "overwrite_originals",
                "trading_admission",
            )
        )
    ):
        raise ValueError("Unsupported or weakened offline closure contract")
    root, sha = Path(c["source_root"]), c["source_manifest_sha256"]
    original_path = _artifact(root, sha, "model_input.json")
    source = _json(root, sha, "protocol.json")
    before = _json(root, sha, "summary.json")
    audits = _json(root, sha, "history_audit.json")
    events = json.loads(original_path.read_bytes())
    if len(events) != 14 or before["models_fitted"] != 0 or before["training_gate"] is not False:
        raise ValueError("Expected the frozen, untrained fourteen-case audit")
    core = [next(e for e in events if e["ticker"] == ticker) for ticker in CORE_TICKERS]
    if len({e["validation_event_id"] for e in events}) != 14 or any(
        e["source_admitted"] is not False or e["label_available_at"] is not None for e in events
    ):
        raise ValueError("Do not silently certify sources or invent label clocks")
    upstream = _json(Path(c["expanded_plan_root"]), c["expanded_plan_manifest_sha256"], "plan.json")
    if source["config"]["source_root"] != c["return_source_root"] or set(
        upstream["config"]["case_event_ids"]
    ) != {e["validation_event_id"] for e in events}:
        raise ValueError("Frozen experiment and expanded return source do not match")
    history_path = _artifact(
        Path(c["history_root"]),
        c["history_manifest_sha256"],
        "SUN_and_all_cost_derived_quarantined_history.parquet",
    )
    # Freeze identity/source denominators before opening any peer financial values.
    ids = pd.read_parquet(history_path, columns=["company_id"])
    cache = load_cached_submissions([Path(p) for p in c["sec_cache_roots"]])
    cache = {k: v for k, v in cache.items() if k[0] in set(ids.company_id)}
    protocol = {
        "version": VERSION,
        "config": c,
        "config_sha256": digest_file(config_path),
        "source_model_input_sha256": digest_file(original_path),
        "history_sha256": digest_file(history_path),
        "financial_history_issuers": int(ids.company_id.nunique()),
        "case_event_ids": [e["validation_event_id"] for e in events],
        "core_event_ids": [e["validation_event_id"] for e in core],
        "cached_sec_sources": [
            {
                "cik": cik,
                "accession_number": acc,
                "source_sha256": meta["sha256"],
                "fetched_at": meta["fetched_at"],
            }
            for (cik, acc), meta in sorted(cache.items())
        ],
        "implementation_sha256": {
            name: digest_file(Path(name))
            for name in (
                "scripts/build_information_data_closure.py",
                "sec_data/information_data_closure.py",
                "sec_data/information_complexity.py",
                "sec_data/announcement_expansion.py",
            )
        },
        "sample_replacement_allowed": False,
        "returns_used_for_selection": False,
        "label_receipts_backdated_to_exit_allowed": False,
        "historical_clock_certification_allowed": False,
    }
    frozen = write_bundle(
        output_root / "plans", VERSION + "-plan", {"plan.json": _encode(protocol)}
    )
    observations, rejections = cached_industries(cache, set(ids.company_id))
    previous = _json(
        Path(source["config"]["industry_root"]),
        source["config"]["industry_manifest_sha256"],
        "industry_observations.json",
    )
    indexed = {(r["cik"], r["accession_number"]): r for r in observations}
    if any(
        k not in indexed.get((r["cik"], r["accession_number"]), {})
        or indexed[(r["cik"], r["accession_number"])][k] != v
        for r in previous
        for k, v in r.items()
    ):
        raise ValueError("Previous industry evidence changed or disappeared")
    history = pd.read_parquet(history_path)
    features, peers = industry_closure(events, history, observations)
    repaired = []
    for event, feature in zip(events, _records(features), strict=True):
        repaired.append({**event, **feature})
    for old, new in zip(events, repaired, strict=True):
        if any(old[k] != new[k] for k in (*FINANCIAL, *FINANCIAL_CLOCKS, "market_excess_return")):
            raise ValueError("Financial signals, clocks or returns were changed")
    # Reuse existing conditional market features. No historical admission is granted.
    predictions, metrics, report = run_experiment(pd.DataFrame(repaired))
    if len(predictions) or len(metrics) or report["models_fitted"]:
        raise ValueError("Offline cache closure must never train a model")
    market = upstream["market_config"]

    def table(group: str, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            _artifact(Path(market[group + "_root"]), market[group + "_manifest_sha256"], name),
            columns=columns,
        )

    price_root = Path(market["prices_root"])
    verify_file(price_root / "output_manifest.json", market["prices_manifest_sha256"])
    price_manifest = json.loads((price_root / "output_manifest.json").read_bytes())
    report_path = price_root / "validation_price_batch_report.json"
    verify_file(report_path, price_manifest["report_sha256"])
    price_report = json.loads(report_path.read_bytes())
    price_raw = Path(price_report["raw_store"])
    action_receipts = _records(table("actions", "massive_security_actions_snapshots.parquet"))
    benchmark_receipts = _records(table("benchmark", "massive_benchmark_snapshots.parquet"))
    stock_lineage = table(
        "prices",
        "validation_price_batch_bars.parquet",
        ["price_target_id", "session_date", "source_sha256"],
    )
    benchmark_lineage = table(
        "benchmark",
        "benchmark_daily_price_observations.parquet",
        ["session_date", "source_sha256"],
    )
    receipts, closure = [], []
    for event in core:
        completion_path = price_raw / "completed" / (event["price_target_id"] + ".json")
        completion = json.loads(completion_path.read_bytes())
        query = completion["query"]
        if (
            completion["price_target_id"] != event["price_target_id"]
            or query["adjusted"] is not False
            or query["pagination_complete"] is not True
        ):
            raise ValueError("Unadjusted cached price query is not complete or matched")
        stock_receipts = []
        for path in query["snapshots"]:
            meta_path = Path(path).resolve()
            if not meta_path.is_relative_to(price_raw.resolve()):
                raise ValueError("Price snapshot metadata escapes its cache")
            meta = json.loads(meta_path.read_bytes())
            if (
                meta["price_target_id"] != event["price_target_id"]
                or meta["ticker"] != event["ticker"]
            ):
                raise ValueError("Mismatched price snapshot identity")
            stock_receipts.append({**meta, "metadata_sha256": digest_file(meta_path)})
        price_requirements = [
            {
                "role": side + "_" + endpoint,
                "source_sha256": event[side + "_" + endpoint + "_unadjusted_source_sha256"],
                "not_before": event["entry_at" if endpoint == "open" else "exit_at"],
            }
            for side in ("issuer", "benchmark")
            for endpoint in ("open", "close")
        ]
        check = next(r for r in audits if r["validation_event_id"] == event["validation_event_id"])
        dates = check["expected_session_dates"]
        if len(dates) != 61 or check["conditional_feature_calculation_allowed"] is not True:
            raise ValueError("Frozen core case lacks its exact 61-session pre-announcement grid")
        for side, frame in (
            (
                "issuer",
                stock_lineage.loc[stock_lineage.price_target_id.eq(event["price_target_id"])],
            ),
            ("benchmark", benchmark_lineage),
        ):
            prior = frame.loc[frame.session_date.isin(dates)]
            if len(prior) != 61 or set(prior.session_date) != set(dates):
                raise ValueError("Prior market feature lineage has missing or duplicate dates")
            price_requirements.extend(
                {
                    "role": side + "_prior61_" + digest,
                    "source_sha256": digest,
                    "not_before": event["market_feature_available_at"],
                }
                for digest in sorted(set(prior.source_sha256))
            )
        price_check = receipt_audit(
            price_requirements,
            stock_receipts + benchmark_receipts,
            [price_raw, Path(c["benchmark_raw_root"])],
        )
        issuer_supports = [
            r for r in action_receipts if r["action_target_id"] == event["action_target_id"]
        ]
        benchmark_supports = [
            r for r in benchmark_receipts if r["query_kind"] in ("splits", "dividends")
        ]
        if not {"reference_start", "reference_end", "ticker_events", "splits", "dividends"} <= {
            r["query_kind"] for r in issuer_supports
        } or not {"splits", "dividends"} <= {r["query_kind"] for r in benchmark_supports}:
            raise ValueError("Missing corporate-action or identity query receipts")
        supports = issuer_supports + benchmark_supports
        support_check = receipt_audit(
            [
                {"role": r["query_id"] + "_" + str(r["page_number"]), "source_sha256": r["sha256"]}
                for r in supports
            ],
            supports,
            [Path(c["actions_raw_root"]), Path(c["benchmark_raw_root"])],
        )
        feature = next(
            r for r in repaired if r["validation_event_id"] == event["validation_event_id"]
        )
        numeric = np.isfinite(
            [feature[k] if feature[k] is not None else np.nan for k in (*FINANCIAL, *STATE)]
        ).all()
        closure.append(
            {
                "validation_event_id": event["validation_event_id"],
                "ticker": event["ticker"],
                "conditional_numeric_features_complete": bool(numeric),
                "industry_peer_count": feature["industry_peer_count"],
                "price_input_receipts": price_check,
                "identity_action_receipts": support_check,
                "price_completion_sha256": digest_file(completion_path),
                "historical_label_available_at": None,
                "source_admitted": False,
                "unresolved": [
                    "first_public_disclosure_time",
                    "peer_financial_scope_and_source_admission",
                    "historical_vendor_delivery_and_label_generation_time",
                    "pre_event_identity_continuity_not_certified_by_query_bounds",
                ]
                + (
                    []
                    if feature["industry_state_ready"]
                    else ["fewer_than_two_eligible_lagged_peers"]
                ),
            }
        )
        receipts.extend(stock_receipts)
    receipts.extend(benchmark_receipts)
    report.update(
        version=VERSION,
        original_experiment_version=before["version"],
        protocol_root=frozen.as_posix(),
        protocol_manifest_sha256=digest_file(frozen / "output_manifest.json"),
        financial_history_issuers=int(ids.company_id.nunique()),
        cached_header_observations=len(observations),
        cached_header_issuers=len({r["cik"] for r in observations}),
        header_parse_rejections=len(rejections),
        previous_header_observations=len(previous),
        previous_header_evidence_preserved=True,
        lagged_industry_state_events_before=before["lagged_industry_state_events"],
        lagged_industry_state_events_after=int(features.industry_state_ready.sum()),
        core_events=len(core),
        core_numeric_complete_events=sum(
            r["conditional_numeric_features_complete"] for r in closure
        ),
        core_price_receipts_complete=sum(r["price_input_receipts"]["complete"] for r in closure),
        core_support_receipts_complete=sum(
            r["identity_action_receipts"]["complete"] for r in closure
        ),
        historical_label_clocks_certified=False,
        financial_signals_unchanged=True,
        existing_returns_recomputed=False,
        network_requests=0,
        keys_read=False,
        source_model_input_bytes_preserved=True,
        originals_overwritten=False,
        user_account_history_entitlement="unknown_not_probed",
        next_action="expand_original_peer_financial_and_disclosure_admission_not_retune_fourteen_cases",
    )
    output = write_bundle(
        output_root,
        VERSION + "-audit",
        {
            "summary.json": _encode(report),
            "protocol.json": _encode(protocol),
            "source_model_input.json": original_path.read_bytes(),
            "candidate_model_input.json": _encode(repaired),
            "industry_observations.json": _encode(observations),
            "industry_rejections.json": _encode(rejections),
            "industry_peer_ledger.json": _encode(peers),
            "core_case_closure.json": _encode(closure),
            "price_receipt_metadata.json": _encode(receipts),
            "market_gap_plan.json": _encode(
                missing_market_plan(audits, c["permission_review_as_of"])
            ),
            "next_review_batch.json": _artifact(root, sha, "next_review_batch.json").read_bytes(),
        },
    )
    return output, report


def main() -> None:
    """Run a deterministic offline audit; no credentials, downloads, broker calls or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/information_data_closure_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/information_data_closure")
    )
    args = parser.parse_args()
    output, report = build_closure(args.config, args.output_root)
    print(
        f"INFORMATION DATA CLOSURE DONE headers={report['cached_header_observations']} "
        f"issuers={report['cached_header_issuers']} "
        f"industry={report['lagged_industry_state_events_after']}/14 "
        f"core_features={report['core_numeric_complete_events']}/3 "
        f"core_price_receipts={report['core_price_receipts_complete']}/3 "
        f"fits={report['models_fitted']} gate={report['training_gate']} output={output}"
    )


if __name__ == "__main__":
    main()
