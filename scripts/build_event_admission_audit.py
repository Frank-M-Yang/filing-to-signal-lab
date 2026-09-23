"""Build the frozen-cohort admission checklist without reading forward price values."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.event_admission import (
    LIQUIDITY_COLUMNS,
    REPRESENTATIONS,
    VERSION,
    audit_label_liquidity,
    build_admission_ledger,
    prior_liquidity_sessions,
)
from sec_data.execution_contract import execution_window
from sec_data.publisher_market_rebuild import interval_evidence_checks
from sec_data.validation_liquidity import ValidationLiquidityConfig


def read_liquidity_bars(path: Path, target: str, dates: list[str]) -> pd.DataFrame:
    """Restrict rows and columns in Parquet before materializing numeric values."""
    if not dates:
        return pd.DataFrame(columns=LIQUIDITY_COLUMNS)
    return pd.read_parquet(
        path,
        columns=LIQUIDITY_COLUMNS,
        filters=[("price_target_id", "==", target), ("session_date", "in", dates)],
    )


def build_audit(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Audit all 316 events and rebuild one label-clock liquidity window offline."""
    config = json.loads(config_path.read_bytes())
    cfg = asdict(ValidationLiquidityConfig())
    if (
        config["version"] != VERSION
        or config["frozen_event_count"] != 316
        or config["market_clock_policy"] != "previous_New_York_date_sessions_only"
        or config["liquidity"]
        != {
            k: cfg[k]
            for k in (
                "lookback_sessions",
                "minimum_print_coverage",
                "reference_position_notional_usd",
                "maximum_median_dollar_volume_participation",
            )
        }
        or any(
            config[k] is not False
            for k in (
                "network_allowed",
                "forward_price_values_allowed",
                "returns_allowed",
                "model_admission",
                "trading_admission",
                "sample_reselection_allowed",
            )
        )
        or any(
            lead["eligible_for_clock_upgrade"] is not False
            or lead["raw_response_preserved"] is not False
            for lead in config["unverified_discovery_leads"]
        )
    ):
        raise ValueError("Unsupported or weakened admission contract")
    verify_file(Path(config["market_config"]), config["market_config_sha256"])
    market = json.loads(Path(config["market_config"]).read_bytes())

    def artifact(group: str, name: str) -> Path:
        source = config if f"{group}_root" in config else market
        return _artifact(Path(source[f"{group}_root"]), source[f"{group}_manifest_sha256"], name)

    def load(group: str, name: str) -> Any:
        source = config if f"{group}_root" in config else market
        return _json(Path(source[f"{group}_root"]), source[f"{group}_manifest_sha256"], name)

    def table(group: str, name: str, columns: list[str]) -> pd.DataFrame:
        return pd.read_parquet(artifact(group, name), columns=columns)

    readiness = pd.DataFrame(load("rebuild", "three_arm_readiness.json"))
    if len(readiness) != config["frozen_event_count"]:
        raise ValueError("Frozen event count changed")
    chronology = load("chronology", "event_chronology.json")
    publisher = load("publisher", "report.json")
    source = load("rebuild", "source_evidence.json")
    event = source["event"]
    state_rows = load("rebuild", "market_state.json")
    if len(state_rows) != 1:
        raise ValueError("Expected one previously rebuilt market state")
    state = state_rows[0]
    window = load("rebuild", "window_evidence.json")
    if len(config["priority_events"]) != 3 or len(set(config["priority_events"])) != 3:
        raise ValueError("Expected three distinct priority cases")
    if not set(config["priority_events"]).issubset(set(readiness.validation_event_id)):
        raise ValueError("Priority event outside frozen cohort")
    if config["priority_events"][0] != event["validation_event_id"]:
        raise ValueError("Priority and rebuilt event differ")
    identities = table(
        "execution",
        "validation_execution_events.parquet",
        [
            "validation_event_id",
            "cik",
            "target_accession_number",
            "assigned_ticker",
        ],
    ).rename(columns={"assigned_ticker": "ticker"})
    pairs = readiness.merge(identities, on=["validation_event_id", "cik"], validate="one_to_one")
    if (
        len(pairs) != len(readiness)
        or not pairs.target_accession_number_x.eq(pairs.target_accession_number_y).all()
    ):
        raise ValueError("Readiness/execution accession mismatch")
    gates = table(
        "old_gate",
        "validation_return_gate_universe.parquet",
        [
            "validation_event_id",
            "cik",
            "liquidity_eligible",
            "liquidity_status",
            "return_gate_status",
        ],
    )
    old_status = readiness[
        ["validation_event_id", "legacy_gate_status_not_new_authorization"]
    ].merge(
        gates,
        on="validation_event_id",
        validate="one_to_one",
    )
    if not old_status.legacy_gate_status_not_new_authorization.eq(
        old_status.return_gate_status
    ).all():
        raise ValueError("Legacy gate metadata differs from frozen predecessor")
    financial = table(
        "publisher",
        "minute_start_disarmed_history.parquet",
        [
            "company_id",
            "filing_accession",
            *REPRESENTATIONS,
        ],
    )
    calendar = table(
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
    if (
        execution_window(event["announcement_at"], event["exchange_mic"], calendar)
        != window["window"]
    ):
        raise ValueError("Revised window does not reproduce from frozen execution contract")
    sessions = prior_liquidity_sessions(event, calendar)
    if len(sessions) != cfg["lookback_sessions"]:
        raise ValueError("Pinned official calendar lacks the required prior sessions")
    implementations = {
        p: digest_file(Path(p))
        for p in (
            "scripts/build_event_admission_audit.py",
            "sec_data/event_admission.py",
            "sec_data/validation_liquidity.py",
            "sec_data/publisher_market_rebuild.py",
            "sec_data/execution_contract.py",
            "scripts/build_sec_clock_layer.py",
            "scripts/build_sec_disclosure_audit.py",
            "sec_data/disclosure_audit.py",
        )
    }
    plan = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "implementation_sha256": implementations,
        "event": event,
        "price_value_columns": LIQUIDITY_COLUMNS,
        "price_value_dates": list(sessions.session_date),
        "sample_reselected": False,
        "thresholds_retuned": False,
        "legacy_authorizations_are_not_new_authorizations": True,
    }
    frozen = write_bundle(output_root, "event-admission-plan-v1", {"plan.json": _encode(plan)})
    # Freeze the exact selection before opening numeric price columns. No post-event
    # prices, benchmark outcomes or return values are materialized anywhere here.
    bars = read_liquidity_bars(
        artifact("prices", "validation_price_batch_bars.parquet"),
        event["price_target_id"],
        list(sessions.session_date),
    )
    benchmark_root = Path(market["benchmark_root"])
    verify_file(benchmark_root / "output_manifest.json", market["benchmark_manifest_sha256"])
    manifest = json.loads((benchmark_root / "output_manifest.json").read_bytes())
    report_path = benchmark_root / "massive_benchmark_report.json"
    verify_file(report_path, manifest["report_sha256"])
    checks = interval_evidence_checks(
        event,
        str(sessions.regular_open_at.iloc[0]),
        str(sessions.regular_close_at.iloc[-1]),
        source["mapping"],
        source["action_audit"],
        source["price_audit"],
        pd.DataFrame(source["benchmark_queries"]),
        json.loads(report_path.read_bytes()),
    )
    liquidity, liquidity_sessions = audit_label_liquidity(event, sessions, bars, checks)
    ledger = build_admission_ledger(
        readiness,
        identities,
        gates,
        financial,
        chronology,
        publisher,
        event,
        liquidity,
        state,
        window,
    )
    if list(ledger.validation_event_id) != list(readiness.validation_event_id):
        raise ValueError("Frozen cohort was dropped, reordered or reselected")
    reviews = {r["validation_event_id"]: r for r in chronology}
    priorities = []
    for event_id in config["priority_events"]:
        record = _records(ledger.loc[ledger.validation_event_id.eq(event_id)])[0]
        review = reviews[event_id]
        record["publisher_sources"] = review["sources"]
        record["legacy_forecast_chronology_not_current_authorization"] = review[
            "chronology_scenarios"
        ]
        record["next_actions"] = (
            [
                "verify_historical_first_public_clock_or_bound_and_search_earlier_channels",
                "freeze_same_sample_outcome_experiment_before_reading_returns",
            ]
            if record["label_scenario_inputs_complete"]
            else [
                "preserve_and_verify_primary_release_clock_with_timezone_and_exact_values",
                "rebuild_forecast_features_training_and_prior_oos_sigma_at_release_clock",
                "rebuild_market_liquidity_and_execution_window_evidence",
            ]
        )
        priorities.append(record)
    blockers = ledger[["validation_event_id", "blockers"]].explode("blockers")
    blocker_counts = blockers.blockers.value_counts().to_dict()
    capture = load("capture", "receipts.json")
    receipts = capture["receipts"]
    summary = {
        "version": VERSION,
        "rebuild_scope": "single_event_publisher_sensitivity_not_joint_history",
        "rebuilt_validation_event_id": event["validation_event_id"],
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_events_retained": len(ledger),
        "sample_reselected": False,
        "financial_four_representation_coverage": int(
            ledger.financial_four_representation_coverage.sum()
        ),
        "legacy_joint_metadata_coverage_not_admission": int(ledger.joint_metadata_coverage.sum()),
        "publisher_sources_reviewed_events": int(
            ledger.reviewed_publisher_source_count.gt(0).sum()
        ),
        "clock_review_queue_events": len(chronology),
        "label_scenario_inputs_complete": int(ledger.label_scenario_inputs_complete.sum()),
        "first_public_certified_events": 0,
        "admitted_events": 0,
        "legacy_gate_counts_not_new_authorization": (
            gates.return_gate_status.value_counts().to_dict()
        ),
        "blocker_counts_overlap": blocker_counts,
        "blocker_counts_are_not_disjoint": True,
        "rebuilt_liquidity": liquidity,
        "new_source_capture_attempts": len(receipts),
        "new_source_capture_failures": sum(bool(r.get("error")) for r in receipts),
        "build_network_requests": 0,
        "forward_price_values_loaded": False,
        "returns_computed": False,
        "model_admission": False,
        "trading_admission": False,
        "forecast_is_retrospective_reconstruction_not_live_log": True,
        "originals_overwritten": False,
    }
    files = {
        "summary.json": _encode(summary),
        "admission_ledger.json": _encode(_records(ledger)),
        "priority_cases.json": _encode(priorities),
        "liquidity_audit.json": _encode(liquidity),
        "source_capture_receipts.json": _encode(capture),
        "unverified_discovery_leads.json": _encode(config["unverified_discovery_leads"]),
    }
    for name, frame in (("admission_ledger", ledger), ("liquidity_sessions", liquidity_sessions)):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        files[f"{name}.parquet"] = buffer.getvalue()
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Create an immutable local checklist without granting any return access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/event_admission_audit_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/event_admission"))
    args = parser.parse_args()
    root, result = build_audit(args.config, args.output_root)
    print(
        f"EVENT ADMISSION DONE retained={result['frozen_events_retained']} "
        f"label_prepared={result['label_scenario_inputs_complete']} "
        f"admitted={result['admitted_events']} "
        f"liquidity={result['rebuilt_liquidity']['status']} output={root}"
    )


if __name__ == "__main__":
    main()
