"""Offline same-event impact audit for local split ratios; no outcomes or model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_publisher_market_rebuild import read_pre_event_bars
from scripts.build_sec_clock_layer import _artifact, _encode, _records
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.validation_market_state import (
    MARKET_STATE_FEATURES,
    RAW_PRICE_BASIS,
    SPLIT_POLICY,
    ValidationMarketStateConfig,
    _feature_row,
    _prepare_bars,
    _prepare_splits,
)

VERSION = "split-policy-migration-v1"
CONFIG = Path("docs/mbuu_publisher_market_rebuild_v1.json")
CONFIG_SHA256 = "4cdcd85aaf7b0b09869c1cd1a29a94214a058b2970598bdd0924cced6618da58"
IDENTITY = [
    "validation_event_id",
    "cik",
    "ticker",
    "exchange_mic",
    "price_target_id",
    "accepted_at",
]
STATE = [
    "market_state_ready",
    "market_state_status",
    "market_feature_available_at",
    "market_last_session_date",
    "split_adjustments_applied",
]


def compare_same_event_states(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Compare the same 13 quantities and event identities, retaining nulls."""
    if old.validation_event_id.duplicated().any() or new.validation_event_id.duplicated().any():
        raise ValueError("Duplicate migration event")
    a = old.sort_values("validation_event_id").reset_index(drop=True)
    b = new.sort_values("validation_event_id").reset_index(drop=True)
    if not a[IDENTITY].equals(b[IDENTITY]):
        raise ValueError("Migration identity, clock or denominator changed")
    left = a.melt(
        id_vars=IDENTITY,
        value_vars=list(MARKET_STATE_FEATURES),
        var_name="feature",
        value_name="old_value",
    )
    right = b.melt(
        id_vars=["validation_event_id"],
        value_vars=list(MARKET_STATE_FEATURES),
        var_name="feature",
        value_name="new_value",
    )
    result = left.merge(right, on=["validation_event_id", "feature"], validate="one_to_one")
    result["both_defined"] = np.isfinite(result.old_value) & np.isfinite(result.new_value)
    result["availability_changed"] = np.isfinite(result.old_value) != np.isfinite(result.new_value)
    result["difference"] = (result.new_value - result.old_value).where(result.both_defined)
    result["numerically_changed"] = result.both_defined & ~np.isclose(
        result.old_value, result.new_value, atol=1e-10, rtol=0
    )
    return result


def run_migration(output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Audit every one of the 227 pinned events; never replace cached artifacts or advance gates."""
    verify_file(CONFIG, CONFIG_SHA256)
    c = json.loads(CONFIG.read_bytes())

    def artifact(group: str, name: str) -> Path:
        return _artifact(Path(c[f"{group}_root"]), c[f"{group}_manifest_sha256"], name)

    old = pd.read_parquet(
        artifact("market", "validation_market_state_features.parquet"),
        columns=[*IDENTITY, *STATE, *MARKET_STATE_FEATURES],
    )
    if len(old) != 227 or old.validation_event_id.duplicated().any():
        raise ValueError("Frozen 227-event denominator changed")
    targets = pd.read_parquet(
        artifact("prices", "validation_price_batch_targets.parquet"),
        columns=["price_target_id", "selected_ticker", "cik", "adjusted"],
    )
    if targets.price_target_id.duplicated().any() or not targets.adjusted.eq(False).all():
        raise ValueError("Ambiguous targets or non-raw provider basis")
    benchmark_manifest = json.loads(
        (Path(c["benchmark_root"]) / "output_manifest.json").read_bytes()
    )
    verify_file(Path(c["benchmark_root"]) / "output_manifest.json", c["benchmark_manifest_sha256"])
    bench_report_path = Path(c["benchmark_root"]) / "massive_benchmark_report.json"
    verify_file(bench_report_path, benchmark_manifest["report_sha256"])
    if json.loads(bench_report_path.read_bytes()).get("query_adjusted") is not False:
        raise ValueError("Benchmark basis not raw")
    if not pd.read_parquet(
        artifact("benchmark", "benchmark_split_observations.parquet"), columns=["execution_date"]
    ).empty:
        raise ValueError("Benchmark split history requires its own adapter")
    actions = pd.read_parquet(
        artifact("actions", "split_observations.parquet"),
        columns=[
            "action_target_id",
            "execution_date",
            "split_from",
            "split_to",
            "historical_adjustment_factor",
            "source_sha256",
        ],
    )
    action_targets = pd.read_parquet(
        artifact("actions", "massive_security_actions_targets.parquet"),
        columns=["action_target_id", "price_target_id"],
    )
    calendars = pd.read_parquet(
        artifact("execution", "validation_exchange_sessions.parquet"),
        columns=["exchange_mic", "session_date", "regular_close_at"],
    )
    if calendars.duplicated(["exchange_mic", "session_date"]).any():
        raise ValueError("Ambiguous exchange calendar")
    clock_lookup = {
        (r["exchange_mic"], pd.Timestamp(r["session_date"]).date()): utc_timestamp(
            r["regular_close_at"]
        )
        for r in _records(calendars)
    }
    plan = {
        "version": VERSION,
        "config_sha256": CONFIG_SHA256,
        "source_config": c,
        "event_ids": old.validation_event_id.tolist(),
        "features": list(MARKET_STATE_FEATURES),
        "split_policy": SPLIT_POLICY,
        "clock_policy": "prior_New_York_date",
        "numeric_comparison_atol": 1e-10,
        "numeric_comparison_rtol": 0,
        "model_admission": False,
        "post_event_returns_allowed": False,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/audit_split_policy_migration.py",
                "sec_data/validation_market_state.py",
                "sec_data/publisher_market_rebuild.py",
                "scripts/build_publisher_market_rebuild.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    # Only now open numeric histories; predicate pushdown excludes event-date and future prices.
    price_path = artifact("prices", "validation_price_batch_bars.parquet")
    benchmark_path = artifact("benchmark", "benchmark_daily_price_observations.parquet")
    benchmark_ids = pd.read_parquet(
        benchmark_path, columns=["benchmark_target_id"]
    ).benchmark_target_id.unique()
    if len(benchmark_ids) != 1:
        raise ValueError("Ambiguous benchmark target")
    split_map = _prepare_splits(actions, action_targets)
    rows, clocks = [], []
    for e_raw in old[IDENTITY].to_dict("records"):
        e: dict[str, Any] = {str(k): v for k, v in e_raw.items()}
        target = targets.loc[targets.price_target_id.eq(e["price_target_id"])]
        if (
            len(target) != 1
            or target.iloc[0].selected_ticker != e["ticker"]
            or target.iloc[0].cik != e["cik"]
        ):
            raise ValueError("Event and raw price target identity mismatch")
        stamp = utc_timestamp(e["accepted_at"])
        date = stamp.tz_convert("America/New_York").date().isoformat()
        prepared = []
        for path, key, value in (
            (price_path, "price_target_id", e["price_target_id"]),
            (benchmark_path, "benchmark_target_id", benchmark_ids[0]),
        ):
            frame = read_pre_event_bars(path, key, value, date).assign(**{key: value})
            prepared.append(_prepare_bars(frame, key, key, price_basis=RAW_PRICE_BASIS))
        state = _feature_row(
            SimpleNamespace(**e),
            prepared[0],
            prepared[1],
            split_map.get(e["price_target_id"], pd.DataFrame()),
            clock_lookup,
            ValidationMarketStateConfig(),
        )
        rows.append({**e, **state})
        clocks.append(
            {
                "validation_event_id": e["validation_event_id"],
                "ticker": e["ticker"],
                "new_york_date": date,
                "legacy_utc_date": stamp.date().isoformat(),
                "date_boundary_changed": date != stamp.date().isoformat(),
            }
        )
    new = pd.DataFrame(rows)
    changes = compare_same_event_states(old, new)
    matched = old[["validation_event_id", *STATE]].merge(
        new[["validation_event_id", *STATE]],
        on="validation_event_id",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    factor_audit = actions.copy()
    factor_audit["local_factor"] = actions.split_from / actions.split_to
    factor_audit["vendor_factor_differs"] = ~np.isclose(
        factor_audit.local_factor, actions.historical_adjustment_factor, atol=1e-12, rtol=0
    )
    changed_ids = set(changes.loc[changes.numerically_changed, "validation_event_id"])
    boundary_ids = {r["validation_event_id"] for r in clocks if r["date_boundary_changed"]}
    summary = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "events_retained": len(new),
        "issuers": int(old.cik.nunique()),
        "old_ready": int(old.market_state_ready.sum()),
        "new_ready": int(new.market_state_ready.sum()),
        "changed_feature_cells": int(changes.numerically_changed.sum()),
        "changed_events": int(
            changes.loc[changes.numerically_changed, "validation_event_id"].nunique()
        ),
        "feature_availability_changes": int(changes.availability_changed.sum()),
        "readiness_changes": int(
            matched.market_state_ready_old.ne(matched.market_state_ready_new).sum()
        ),
        "new_york_vs_utc_date_boundaries": sum(r["date_boundary_changed"] for r in clocks),
        "changed_events_with_date_boundary_change": len(changed_ids & boundary_ids),
        "changed_events_without_date_boundary_change": len(changed_ids - boundary_ids),
        "status_reason_changes": int(
            matched.market_state_status_old.ne(matched.market_state_status_new).sum()
        ),
        "events_with_applied_split_adjustments": int(new.split_adjustments_applied.gt(0).sum()),
        "split_records": len(actions),
        "vendor_vs_local_factor_mismatches": int(factor_audit.vendor_factor_differs.sum()),
        "split_policy": SPLIT_POLICY,
        "network_requests": 0,
        "old_outputs_overwritten": False,
        "main_sample_reselected": False,
        "post_event_prices_loaded": False,
        "returns_computed": False,
        "model_fitted": False,
        "model_admission": False,
        "trading_admission": False,
        "limitation": (
            "Same-event feature migration only; legacy common-session history selection retained. "
            "No downstream training or first-public/vendor-delivery certification. "
            "Retrospective source collection remains unchanged."
        ),
    }
    files = {
        "summary.json": _encode(summary),
        "feature_changes.json": _encode(_records(changes)),
        "new_market_states.json": _encode(_records(new)),
        "clock_and_readiness.json": _encode(
            _records(
                pd.DataFrame(clocks).merge(matched, on="validation_event_id", validate="one_to_one")
            )
        ),
        "factor_audit.json": _encode(_records(factor_audit)),
    }
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Run the fixed offline migration audit; outputs are separate from the production store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/split_policy_migration")
    )
    args = parser.parse_args()
    root, report = run_migration(args.output_root)
    print(
        f"SPLIT MIGRATION DONE events={report['events_retained']} ready={report['new_ready']} "
        f"changed={report['changed_events']} cells={report['changed_feature_cells']} "
        f"admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
