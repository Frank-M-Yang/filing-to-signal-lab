"""Offline six-case financial clock reconstruction; never read returns or fit return models."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from scripts.review_next_disclosures import review_sources
from sec_data.announcement_replay import SIGNALS, apply_release_proxies
from sec_data.clock_layer import audit_replay_chronology, compare_histories, replay_history
from sec_data.disclosure_audit import digest_file, utc_timestamp, verify_file
from sec_data.next_disclosure_replay import (
    VERSION,
    audit_replayed_cases,
    overlay_complete_pairs,
    select_complete_pairs,
    verify_prefix_outputs,
)
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig


def validate_contract(c: dict[str, Any]) -> None:
    """Reject sample replacement, outcome access, threshold tuning or trading authorization."""
    if (
        c.get("version") != VERSION
        or c.get("frozen_queue_issuers") != 20
        or c.get("frozen_pair_cases") != 6
        or c.get("frozen_history_events") != 3254
        or c.get("fixed_specification_refit_allowed") is not True
        or c.get("primary_scope_arm") != "SUN_and_all_cost_derived_quarantined"
        or c.get("minimum_forecast_training_events") != 100
        or c.get("minimum_prior_oos_errors") != 50
        or any(
            c.get(k) is not False
            for k in (
                "hyperparameter_search_allowed",
                "threshold_relaxation_allowed",
                "sample_replacement_allowed",
                "network_allowed",
                "market_features_in_forecast",
                "returns_allowed",
                "first_public_certified",
                "model_admission",
                "return_admission",
                "trading_admission",
                "overwrite_originals",
            )
        )
    ):
        raise ValueError("Unsupported or weakened next-disclosure replay contract")


def _rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads(
        pd.DataFrame(rows).to_json(orient="records", date_format="iso", double_precision=15)
    )
    return result


def build_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Pin all six original pairs before values, then rebuild one shared conservative history."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[group + "_root"]), c[group + "_manifest_sha256"], name)

    reviewed = load("review", "summary.json")
    source_plan = _json(
        Path(reviewed["prepared_root"]), reviewed["prepared_manifest_sha256"], "plan.json"
    )
    verify_file(Path(c["review_config"]), reviewed["review_config_sha256"])
    rechecked, _ = review_sources(
        Path(reviewed["prepared_root"]),
        reviewed["prepared_manifest_sha256"],
        Path(c["review_config"]),
        output_root / "source_rechecks",
    )
    if digest_file(rechecked / "output_manifest.json") != c["review_manifest_sha256"]:
        raise ValueError("Exact original SEC byte/clock/cell checks no longer reproduce")
    dispositions = load("review", "case_dispositions.json")
    cells = load("review", "reviewed_metric_cells.json")
    cases = select_complete_pairs(dispositions, cells)
    old_plan = load("baseline_plan", "plan.json")
    old_summary = load("baseline", "summary.json")
    if (
        old_summary["plan_manifest_sha256"] != c["baseline_plan_manifest_sha256"]
        or source_plan["financial_sources"] != old_plan["financial_sources"]
        or old_plan["forecast_features"] != list(FORECAST_FEATURES)
    ):
        raise ValueError("Source chain or frozen financial forecast specification changed")
    cfg = PilotFilingFeatureConfig(**old_plan["forecast_config"])
    cfg.validate()
    if (
        cfg.minimum_forecast_training_events != c["minimum_forecast_training_events"]
        or cfg.minimum_prior_oos_errors != c["minimum_prior_oos_errors"]
    ):
        raise ValueError("Pinned forecast/sigma thresholds changed")
    history_path = _artifact(
        Path(c["baseline_root"]),
        c["baseline_manifest_sha256"],
        c["primary_scope_arm"] + "_history.parquet",
    )
    # Financial history only: no case returns, OHLC, market features or labels are opened.
    history = pd.read_parquet(history_path)
    excluded = set(old_plan["excluded_event_ids"][c["primary_scope_arm"]])
    if (
        len(history) != c["frozen_history_events"]
        or history.event_id.duplicated().any()
        or set(history.loc[history.scope_quarantined, "event_id"]) != excluded
        or history.scope_arm.ne(c["primary_scope_arm"]).any()
    ):
        raise ValueError("Frozen financial history or conservative scope mask changed")
    protocol = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "cases": cases,
        "retained_disposition_ids": [r["validation_event_id"] for r in dispositions],
        "history_sha256": digest_file(history_path),
        "forecast_config": old_plan["forecast_config"],
        "forecast_features": list(FORECAST_FEATURES),
        "excluded_margin_records": len(excluded),
        "return_values_read": False,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/replay_next_disclosures.py",
                "sec_data/next_disclosure_replay.py",
                "sec_data/announcement_replay.py",
                "sec_data/clock_layer.py",
                "sec_data/pilot_filing_features.py",
                "sec_data/publisher_replay.py",
                "sec_data/event_cleaning.py",
                "event_panel.py",
            )
        },
    }
    frozen = write_bundle(
        output_root / "plans", VERSION + "-plan", {"plan.json": _encode(protocol)}
    )
    print(f"NEXT REPLAY PLAN cases=6 history={len(history)} output={frozen}", flush=True)
    baseline = replay_history(history, cfg)
    if not compare_histories(history, baseline)[1].empty:
        raise ValueError("Pinned conservative financial baseline no longer reproduces")
    overlaid = overlay_complete_pairs(baseline, cases, cells)
    pd.testing.assert_frame_equal(overlaid, overlay_complete_pairs(baseline, cases[::-1], cells))
    print("NEXT REPLAY shared_history", flush=True)
    replayed = replay_history(overlaid, cfg)
    causal = [
        {
            "before": cases[0]["complete_pair_replay_cutoff_at"],
            "unchanged_rows": verify_prefix_outputs(
                baseline, replayed, cases[0]["complete_pair_replay_cutoff_at"]
            ),
        }
    ]
    for size in range(1, len(cases)):
        print(f"NEXT REPLAY causal_prefix={size}/5", flush=True)
        boundary = cases[size]["complete_pair_replay_cutoff_at"]
        prefix = replay_history(apply_release_proxies(baseline, cases[:size], cells), cfg)
        causal.append(
            {
                "before": boundary,
                "unchanged_rows": verify_prefix_outputs(prefix, replayed, boundary),
            }
        )
    audit, comparisons, deps, company = audit_replayed_cases(baseline, replayed, cases, cfg)
    common_ready = all(r["all_six_common_financial_cases_ready"] for r in audit)
    sample = []
    if common_ready:
        for case in cases:
            row = replayed.loc[
                replayed.company_id.eq(case["cik"])
                & replayed.filing_accession.eq(case["target_accession_number"])
            ].iloc[0]
            sample.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "cik": case["cik"],
                    "ticker": case["ticker"],
                    "information_set_accession": case["release_accession_number"],
                    "information_set_sha256": case["release_pair_source_sha256"],
                    "actual_available_at": row.actual_available_at,
                    "gm_forecast": float(row.gm_forecast),
                    "forecast_sigma": float(row.forecast_sigma),
                    "forecast_generated_at": row.forecast_generated_at,
                    "forecast_training_end": row.forecast_training_end,
                    "forecast_error_history_end": row.forecast_error_history_end,
                    "forecasts_reconstructed_not_live": True,
                    "first_public_certified": False,
                    "model_admission": False,
                    "return_admission": False,
                    "trading_admission": False,
                    **{s: float(row[s]) for s in SIGNALS},
                }
            )
    ledger = []
    indexed = {r["validation_event_id"]: r for r in _rows(audit)}
    for source in dispositions:
        ledger.append(
            {
                **source,
                "new_financial_replay_audit": indexed.get(source["validation_event_id"]),
                "financial_forecast_replayed": source["validation_event_id"] in indexed,
                "model_admission": False,
                "return_admission": False,
                "trading_admission": False,
            }
        )
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_queue_issuers_retained": len(ledger),
        "frozen_pair_cases": len(cases),
        "history_events_retained": len(history),
        "scope_quarantined_records": len(excluded),
        "old_forecasts_not_prior_to_complete_pair": sum(
            not r["old_forecast_clock_check"]["existing_forecast_clocks_before_proxy"]
            for r in audit
        ),
        "replayed_complete_financial_cases": sum(
            r["complete_financial_representations"] for r in audit
        ),
        "common_financial_comparison_gate": common_ready,
        "common_financial_sample_events": len(sample),
        "later_complete_pairs_not_backfilled": sum(
            utc_timestamp(r["complete_pair_replay_cutoff_at"])
            > utc_timestamp(r["frozen_earliest_candidate_at"])
            for r in cases
        ),
        "signals_changed_by_representation": {
            s: sum(r["changed"] is True for r in comparisons if r["representation"] == s)
            for s in SIGNALS
        },
        "chronology_audit": audit_replay_chronology(replayed),
        "causal_prefix_checks": len(causal),
        "financial_forecast_specification": "fixed_company_history_OLS_six_features",
        "forecasts_reconstructed_not_live": True,
        "return_models_fitted": 0,
        "return_values_read": False,
        "first_public_certified": False,
        "model_admission": False,
        "return_admission": False,
        "trading_admission": False,
        "network_requests": 0,
        "keys_read": False,
        "old_artifacts_overwritten": False,
        "next_action": "verify_earlier_publication_evidence_then_rebuild_exact_market_windows",
    }
    buffer = io.BytesIO()
    saved = replayed.assign(
        first_public_certified=False,
        model_admission=False,
        return_admission=False,
        trading_admission=False,
        forecasts_reconstructed_not_live=True,
    )
    saved.to_parquet(buffer, index=False)
    files = {
        "summary.json": _encode(summary),
        "case_dispositions.json": _encode(ledger),
        "forecast_case_audit.json": _encode(_rows(audit)),
        "signal_clock_comparison.json": _encode(comparisons),
        "common_financial_sample.json": _encode(_rows(sample)),
        "forecast_dependency_audit.json": _encode(_rows(deps)),
        "company_feature_dependencies.json": _encode(_rows(company)),
        "causal_prefix_audit.json": _encode(causal),
        "shared_disarmed_history.parquet": buffer.getvalue(),
    }
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Reproduce conditional financial signals from pinned private caches without credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/next_disclosure_replay_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/next_disclosure_replay")
    )
    args = parser.parse_args()
    root, report = build_replay(args.config, args.output_root)
    print(f"NEXT REPLAY DONE report={json.dumps(report)} output={root}")


if __name__ == "__main__":
    main()
