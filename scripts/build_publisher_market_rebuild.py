"""Offline publisher-clock market rebuild; JSON/Parquet artifacts, no narrative report."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.execution_contract import execution_window
from sec_data.publisher_market_rebuild import (
    BAR_METADATA_COLUMNS,
    BAR_VALUE_COLUMNS,
    VERSION,
    audit_revised_window,
    compare_market_features,
    interval_evidence_checks,
    rebuild_pre_event_state,
)
from sec_data.publisher_replay import EVENT_COLUMNS, MARKET_COLUMNS, three_arm_readiness
from sec_data.validation_market_state import MARKET_STATE_FEATURES, RAW_PRICE_BASIS, SPLIT_POLICY


def _one(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if len(frame) != 1:
        raise ValueError(f"Expected one {label}")
    return _records(frame)[0]


def read_pre_event_bars(path: Path, key: str, target: str, cutoff_date: str) -> pd.DataFrame:
    """Filter dates in the reader before materializing any numeric price values."""
    return pd.read_parquet(
        path,
        columns=BAR_VALUE_COLUMNS,
        filters=[(key, "==", target), ("session_date", "<", cutoff_date)],
    )


def build_rebuild(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Rebuild the pinned queue only, preserve all cohort rows, keep trading disabled."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or config["market_clock_policy"] != "previous_New_York_date_sessions_only"
        or any(
            config[k] is not False
            for k in (
                "network_allowed",
                "forward_price_values_allowed",
                "returns_allowed",
                "overwrite_originals",
                "model_admission",
            )
        )
    ):
        raise ValueError("Unsupported or weakened market-rebuild contract")

    def artifact(group: str, name: str) -> Path:
        return _artifact(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name)

    def load(group: str, name: str) -> Any:
        return _json(Path(config[f"{group}_root"]), config[f"{group}_manifest_sha256"], name)

    def table(group: str, name: str, columns: list[str], filters: Any = None) -> pd.DataFrame:
        return pd.read_parquet(artifact(group, name), columns=columns, filters=filters)

    def report(group: str, name: str) -> dict[str, Any]:
        root = Path(config[f"{group}_root"])
        verify_file(root / "output_manifest.json", config[f"{group}_manifest_sha256"])
        manifest = json.loads((root / "output_manifest.json").read_bytes())
        verify_file(root / name, manifest["report_sha256"])
        value: dict[str, Any] = json.loads((root / name).read_bytes())
        return value

    upstream = load("publisher", "report.json")
    queue = load("publisher", "market_rebuild_requests.json")
    if len(queue) != 1 or queue[0]["return_admission"] is not False:
        raise ValueError("Expected one disarmed publisher-market request")
    request = queue[0]
    scenario = upstream["scenarios"]["minute_start"]["target"]
    if (
        request["cik"] != scenario["company_id"]
        or request["target_accession_number"] != scenario["filing_accession"]
    ):
        raise ValueError("Publisher history/queue identity differs")
    market_manifest = json.loads(
        artifact("market", "validation_market_state_features.parquet")
        .parent.joinpath("output_manifest.json")
        .read_bytes()
    )
    for current, old_group in (
        ("prices", "price_audit"),
        ("actions", "security_actions"),
        ("benchmark", "benchmark"),
        ("execution", "execution_plan"),
        ("old_gate", "return_gate"),
    ):
        if (
            config[f"{current}_manifest_sha256"]
            != market_manifest["input_manifest_sha256"][old_group]
        ):
            raise ValueError("Market rebuild inputs differ from frozen evidence chain")
    market = table(
        "market",
        "validation_market_state_features.parquet",
        list(
            dict.fromkeys(
                [
                    *MARKET_COLUMNS,
                    "ticker",
                    "exchange_mic",
                    "price_target_id",
                    *MARKET_STATE_FEATURES,
                ]
            )
        ),
    )
    old = _one(
        market.loc[market.validation_event_id.eq(request["validation_event_id"])], "market event"
    )
    target = old["price_target_id"]
    filter_target = [("price_target_id", "==", target)]
    prices = _one(
        table(
            "prices",
            "validation_price_batch_targets.parquet",
            [
                "price_target_id",
                "cik",
                "selected_ticker",
                "selected_exchange_mic",
                "security_identity_key",
                "security_identity_key_kind",
                "adjusted",
                "selection_uses_return_values",
            ],
            filter_target,
        ),
        "price target",
    )
    if (
        prices["adjusted"] is not False
        or prices["selection_uses_return_values"] is not False
        or any(
            prices[k] != old[v]
            for k, v in (
                ("cik", "cik"),
                ("selected_ticker", "ticker"),
                ("selected_exchange_mic", "exchange_mic"),
            )
        )
    ):
        raise ValueError("Price identity, adjustment or selection mismatch")
    bench_queries = table(
        "benchmark",
        "massive_benchmark_snapshots.parquet",
        [
            "benchmark_target_id",
            "query_kind",
            "ticker",
            "window_start",
            "window_end",
            "status_code",
            "sha256",
        ],
    )
    if bench_queries.benchmark_target_id.nunique() != 1:
        raise ValueError("Ambiguous benchmark instrument")
    event = {
        "validation_event_id": request["validation_event_id"],
        "cik": request["cik"],
        "ticker": old["ticker"],
        "exchange_mic": old["exchange_mic"],
        "price_target_id": target,
        "security_id": prices["security_identity_key"],
        "security_id_kind": prices["security_identity_key_kind"],
        "benchmark_target_id": str(bench_queries.benchmark_target_id.iloc[0]),
        "announcement_at": request["actual_available_at"],
    }
    stamp = utc_timestamp(event["announcement_at"])
    if stamp != utc_timestamp(upstream["windows"]["publisher_label_window"]["announcement_at"]):
        raise ValueError("Publisher window/feature time mismatch")
    date = stamp.tz_convert("America/New_York").date().isoformat()
    implementations = {
        p: digest_file(Path(p))
        for p in [
            "scripts/build_publisher_market_rebuild.py",
            "sec_data/publisher_market_rebuild.py",
            "sec_data/validation_market_state.py",
            "sec_data/execution_contract.py",
            "sec_data/publisher_replay.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
            "sec_data/disclosure_audit.py",
        ]
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "event": event,
        "implementation_sha256": implementations,
        "sample_reselected": False,
        "feature_columns": list(MARKET_STATE_FEATURES),
        "same_day_prices_excluded": True,
    }
    frozen = write_bundle(
        output_root, "publisher-market-rebuild-plan-v1", {"plan.json": _encode(plan)}
    )
    sessions = table(
        "execution",
        "validation_exchange_sessions.parquet",
        [
            "exchange_mic",
            "session_date",
            "regular_open_at",
            "regular_close_at",
            "source_snapshot_sha256",
        ],
    )
    window = execution_window(event["announcement_at"], event["exchange_mic"], sessions)
    if any(
        window[k] != upstream["windows"]["publisher_label_window"][k]
        for k in ("entry_at", "exit_at", "entry_price_type", "exit_price_type")
    ):
        raise ValueError("Revised execution schedule does not reproduce")
    action_audit = _one(
        table(
            "actions",
            "sample_security_actions_audit.parquet",
            [
                "action_target_id",
                "price_target_id",
                "cik",
                "selected_ticker",
                "selected_exchange_mic",
                "price_window_start",
                "price_window_end",
                "splits_query_complete",
                "dividends_query_complete",
                "corporate_actions_ready",
                "historical_symbol_interval_ready",
                "stable_security_id",
                "split_count",
                "dividend_count",
            ],
            filter_target,
        ),
        "action audit",
    )
    mapping = _one(
        table(
            "actions",
            "validated_security_mappings.parquet",
            [
                "price_target_id",
                "cik",
                "security_id",
                "security_id_kind",
                "ticker",
                "exchange_mic",
                "valid_from_inclusive",
                "valid_to_exclusive",
                "mapping_method",
                "ticker_events_endpoint_experimental",
                "source_snapshot_sha256",
                "end_source_snapshot_sha256",
                "ticker_events_source_snapshot_sha256",
            ],
            filter_target,
        ),
        "security interval",
    )
    split_filter = [
        ("action_target_id", "==", action_audit["action_target_id"]),
        ("execution_date", "<", date),
    ]
    splits = table(
        "actions",
        "split_observations.parquet",
        ["execution_date", "split_from", "split_to", "source_sha256"],
        split_filter,
    )
    benchmark_splits = table(
        "benchmark",
        "benchmark_split_observations.parquet",
        ["execution_date"],
        [("execution_date", "<", date)],
    )
    if not benchmark_splits.empty:
        raise ValueError(
            "Frozen market-feature math does not support benchmark splits; extend explicitly"
        )
    issuer_path = artifact("prices", "validation_price_batch_bars.parquet")
    benchmark_path = artifact("benchmark", "benchmark_daily_price_observations.parquet")
    state, state_audit = rebuild_pre_event_state(
        event,
        read_pre_event_bars(issuer_path, "price_target_id", target, date),
        read_pre_event_bars(
            benchmark_path, "benchmark_target_id", event["benchmark_target_id"], date
        ),
        splits,
        sessions,
        price_basis=RAW_PRICE_BASIS,
    )
    metadata = pd.read_parquet(issuer_path, columns=BAR_METADATA_COLUMNS, filters=filter_target)
    benchmark_metadata = pd.read_parquet(
        benchmark_path,
        columns=BAR_METADATA_COLUMNS,
        filters=[("benchmark_target_id", "==", event["benchmark_target_id"])],
    )
    price_audit = _one(
        table(
            "prices",
            "validation_price_batch_target_audit.parquet",
            [
                "price_target_id",
                "cik",
                "ticker",
                "price_window_start",
                "price_window_end",
                "provider_query_complete",
                "parser_rejection_rows",
            ],
            filter_target,
        ),
        "price audit",
    )
    listing = _one(
        table(
            "listing",
            "validation_security_listing_outcomes.parquet",
            [
                "cik",
                "exchange_mic",
                "security_id",
                "security_id_kind",
                "outcome_cutoff_date",
                "listing_outcome_status",
                "delisting_date",
                "listing_outcome_ready",
                "source_snapshot_sha256",
            ],
            [("cik", "==", event["cik"]), ("security_id", "==", event["security_id"])],
        ),
        "listing evidence",
    )
    benchmark_report = report("benchmark", "massive_benchmark_report.json")
    history_checks = interval_evidence_checks(
        event,
        utc_timestamp(state_audit.regular_open_at.iloc[0]).isoformat(),
        utc_timestamp(state_audit.regular_close_at.iloc[-1]).isoformat(),
        mapping,
        action_audit,
        price_audit,
        bench_queries,
        benchmark_report,
    )
    state["history_evidence_checks"] = history_checks
    if not all(history_checks.values()):
        state["market_state_ready"] = False
        state["market_state_status"] = "history_identity_or_action_query_incomplete"
        for feature in MARKET_STATE_FEATURES:
            state[feature] = np.nan
    window_audit, window_sessions = audit_revised_window(
        event,
        window,
        sessions,
        metadata,
        benchmark_metadata,
        mapping,
        action_audit,
        price_audit,
        listing,
        bench_queries,
        benchmark_report,
    )
    market_metadata = market[MARKET_COLUMNS].copy()
    match = market_metadata.validation_event_id.eq(event["validation_event_id"])
    market_metadata.loc[match, "market_feature_available_at"] = state["market_feature_available_at"]
    market_metadata.loc[match, "market_state_ready"] = state["market_state_ready"]
    history = table(
        "publisher",
        "minute_start_disarmed_history.parquet",
        [
            "company_id",
            "filing_accession",
            "actual_available_at",
            "analysis_ready_level",
            "analysis_ready_delta",
            "analysis_ready_innovation_std",
        ],
    )
    events = table("execution", "validation_execution_events.parquet", EVENT_COLUMNS)
    readiness = three_arm_readiness(events, history, market_metadata, event["validation_event_id"])
    readiness.loc[
        readiness.validation_event_id.eq(event["validation_event_id"]), "new_return_window_verified"
    ] = window_audit["window_evidence_ready"]
    # Preserve the cause of absent market rows: most were excluded by the old
    # liquidity gate, not necessarily missing from the underlying price store.
    gates = table(
        "old_gate",
        "validation_return_gate_universe.parquet",
        ["validation_event_id", "return_gate_status"],
    ).rename(columns={"return_gate_status": "legacy_gate_status_not_new_authorization"})
    readiness = readiness.merge(gates, on="validation_event_id", how="left", validate="one_to_one")
    before = pd.DataFrame(load("publisher", "three_arm_readiness.json"))
    if list(readiness.validation_event_id) != list(before.validation_event_id):
        raise ValueError("Frozen cohort changed")
    unchanged = ~readiness.validation_event_id.eq(event["validation_event_id"])
    for field in (
        "joint_metadata_coverage",
        "market_metadata_coverage",
        "financial_representation_coverage",
    ):
        if not readiness.loc[unchanged, field].eq(before.loc[unchanged, field]).all():
            raise ValueError("Rebuild unexpectedly changed another event's coverage")
    differences = compare_market_features(old, state)
    source_evidence = {
        "event": event,
        "mapping": mapping,
        "action_audit": action_audit,
        "price_audit": price_audit,
        "listing": listing,
        "benchmark_queries": _records(bench_queries),
    }
    comparable = np.isfinite(differences.old_filing_proxy_value) & np.isfinite(
        differences.rebuilt_publisher_value
    )
    summary = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "event": event,
        "market_state_ready": state["market_state_ready"],
        "previous_market_cutoff": old["market_feature_available_at"],
        "rebuilt_market_cutoff": state["market_feature_available_at"],
        "market_history_sessions_required": 61,
        "feature_math": "unchanged_13_equations_with_v2_local_split_ratios",
        "split_adjustment_policy": SPLIT_POLICY,
        "same_day_market_data_included": False,
        "vendor_delivery_times_certified": False,
        "historical_prices_collected_retrospectively": True,
        "features_unavailable": int((~np.isfinite(differences.rebuilt_publisher_value)).sum()),
        "numeric_features_changed": int(
            (
                comparable
                & ~np.isclose(
                    differences.old_filing_proxy_value,
                    differences.rebuilt_publisher_value,
                    atol=1e-10,
                    rtol=0,
                )
            ).sum()
        ),
        "joint_metadata_coverage_before": int(before.joint_metadata_coverage.sum()),
        "joint_metadata_coverage_after": int(readiness.joint_metadata_coverage.sum()),
        "frozen_events_retained": len(readiness),
        "window_evidence_ready": window_audit["window_evidence_ready"],
        "window_checks": window_audit["checks"],
        "first_public_certified": False,
        "forward_price_values_loaded": False,
        "returns_computed": False,
        "model_admission": False,
        "trading_admission": False,
        "originals_overwritten": False,
        "network_requests": 0,
        "missing_market_row_legacy_reasons": readiness.loc[
            readiness.market_cutoff_status.eq("missing_market_row"),
            "legacy_gate_status_not_new_authorization",
        ]
        .value_counts()
        .to_dict(),
        "remaining_tasks": [
            "resolve_first_public_clock_evidence",
            "freeze_same_sample_return_experiment",
            "rebuild_event_time_liquidity_and_other_time_dependent_controls",
        ],
    }
    # Normalize timestamps/nullable values without formatting a prose report.
    summary = _records(pd.DataFrame([summary]))[0]
    files = {
        "summary.json": _encode(summary),
        "market_state.json": _encode(_records(pd.DataFrame([state]))),
        "feature_changes.json": _encode(_records(differences)),
        "source_evidence.json": _encode(source_evidence),
        "window_evidence.json": _encode(window_audit),
        "three_arm_readiness.json": _encode(_records(readiness)),
    }
    for name, frame in (
        ("market_history_session_audit", state_audit),
        ("window_session_audit", window_sessions),
    ):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        files[f"{name}.parquet"] = buffer.getvalue()
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Run the isolated cached-data diagnostic; never fetch or write a bilingual report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/publisher_market_rebuild_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/publisher_market_rebuild")
    )
    args = parser.parse_args()
    root, summary = build_rebuild(args.config, args.output_root)
    print(
        f"PUBLISHER MARKET DONE market={summary['market_state_ready']} "
        f"window_evidence={summary['window_evidence_ready']} "
        f"joint={summary['joint_metadata_coverage_after']}/{summary['frozen_events_retained']} "
        f"admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
