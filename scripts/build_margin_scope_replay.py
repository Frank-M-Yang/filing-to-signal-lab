"""Audit all SUN recipes and run frozen, offline scope-quarantine sensitivity arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_announcement_admission import _config
from scripts.build_announcement_return_diagnostic import _parquet, _rows
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_replay import SIGNALS
from sec_data.announcement_scope_audit import audit_margin_scope, bind_reconciliation_value
from sec_data.cleaning import parse_companyfacts
from sec_data.clock_layer import HISTORY_COLUMNS, audit_replay_chronology
from sec_data.disclosure_audit import (
    digest_file,
    load_cached_submissions,
    utc_timestamp,
    verify_file,
)
from sec_data.disclosure_review import audit_submission_clock
from sec_data.guidance_evidence import parse_complete_submission
from sec_data.margin_scope_replay import (
    ARMS,
    VERSION,
    audit_original_recipes,
    replay_scope_view,
    scope_case_signals,
    scope_quarantine_view,
    scope_rank_diagnostics,
)
from sec_data.pilot_filing_features import PilotFilingFeatureConfig
from sec_data.publisher_replay import target_dependencies


def validate_contract(config: dict[str, Any]) -> None:
    """Permit only the frozen three-arm sensitivity protocol, not a repaired-data claim."""
    expected = {
        "version": VERSION,
        "issuer_cik": "0001552275",
        "frozen_history_events": 3254,
        "frozen_issuer_events": 35,
        "frozen_case_count": 6,
        "arms": list(ARMS),
        "policy_basis": "retrospective_accounting_scope_sensitivity_not_a_historical_live_filter",
        "fixed_specification_refit_allowed": True,
        "known_same_case_returns_allowed_after_signal_freeze": True,
        **dict.fromkeys(
            (
                "general_scope_gate_cleared",
                "hyperparameter_search_allowed",
                "sample_replacement_allowed",
                "numerical_history_repair_allowed",
                "network_allowed",
                "first_public_certified",
                "trading_admission",
                "inferential_tests_allowed",
                "strategy_pnl_allowed",
                "overwrite_originals",
            ),
            False,
        ),
    }
    if any(type(config.get(k)) is not type(v) or config[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported or weakened accounting-scope sensitivity contract")


def _additional_proofs(
    config: dict[str, Any],
    history: pd.DataFrame,
    facts: pd.DataFrame,
    catalog: pd.DataFrame,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    proofs = []
    for spec in config["additional_original_reconciliations"]:
        targets = history.loc[history.filing_accession.eq(spec["filing_accession"])]
        if len(targets) != 1:
            raise ValueError("Reconciliation target must be one original issuer quarter")
        target = _rows(targets)[0]
        for key in ("period_start", "period_end"):
            target[key] = target[key][:10]
            if target[key] != spec[key]:
                raise ValueError("Reviewed reconciliation quarter differs from original target")
        source = cache[(config["issuer_cik"], spec["release_accession"])]
        verify_file(Path(source["path"]), spec["submission_sha256"])
        content = Path(source["path"]).read_bytes()
        candidates = catalog.loc[catalog.accession_number.eq(spec["release_accession"])]
        if len(candidates) != 1 or candidates.iloc[0].form != "8-K":
            raise ValueError("Reconciliation must bind to an original 8-K accession")
        clock = audit_submission_clock(content, {str(k): v for k, v in candidates.iloc[0].items()})
        if clock["status"] != "consistent_acceptance_proxy" or not utc_timestamp(
            clock["sec_acceptance_proxy_at"]
        ) < utc_timestamp(target["actual_available_at"]):
            raise ValueError("Source reconciliation is not an earlier, clock-verified release")
        documents = [
            d
            for d in parse_complete_submission(content)
            if d["filename"] == spec["exhibit_filename"]
            and d["document_type"].upper().startswith("EX-99")
            and d["document_sha256"] == spec["exhibit_sha256"]
        ]
        if len(documents) != 1:
            raise ValueError("Original reconciliation exhibit is missing or ambiguous")
        text = re.sub(r"\s+", " ", documents[0]["visible_text"])
        if any(
            anchor not in text
            for anchor in (spec["document_anchor"], spec["period_anchor"], spec["unit_anchor"])
        ):
            raise ValueError("Original current-quarter context, accounting units or title missing")
        revenue = bind_reconciliation_value(
            text,
            spec["revenue_anchor"],
            spec["revenue_label"],
            spec["revenue_display"],
            spec["scale"],
        )
        if revenue != target["revenue"]:
            raise ValueError("Reconciliation denominator differs from original recipe")
        gp = bind_reconciliation_value(
            text, spec["gp_anchor"], "Gross profit $", spec["gp_display"], spec["scale"]
        )
        segment = bind_reconciliation_value(
            text,
            spec["segment_anchor"],
            "Total segment profit",
            spec["segment_display"],
            spec["scale"],
        )
        depreciation = bind_reconciliation_value(
            text,
            spec["depreciation_anchor"],
            "Depreciation, amortization and accretion, excluding corporate and other",
            spec["depreciation_display"],
            spec["scale"],
        )
        matching_cost = facts.loc[facts.fact_id.eq(target["cost_fact_id"])]
        if len(matching_cost) != 1:
            raise ValueError("Original cost source must match its exact fact identity")
        fact = matching_cost.iloc[0]
        costs = [
            {
                "accn": fact.accession_number,
                "form": fact.form,
                "tag": fact.tag,
                "unit": fact.unit,
                "start": fact.period_start.date().isoformat(),
                "end": fact.period_end.date().isoformat(),
                "val": float(fact.value_numeric),
            }
        ]
        proof = audit_margin_scope(target, costs, gp, segment, depreciation)
        proof.update(
            source_submission_sha256=spec["submission_sha256"],
            source_exhibit_sha256=spec["exhibit_sha256"],
            source_url=source["requested_url"],
            release_accession=spec["release_accession"],
            release_acceptance_proxy_at=clock["sec_acceptance_proxy_at"],
            original_cost_available_at=utc_timestamp(fact.accepted_at).isoformat(),
            source_companyfacts_zip_sha256=fact.source_snapshot_sha256,
            segment_profit_anchor=spec["segment_anchor"],
            depreciation_anchor=spec["depreciation_anchor"],
            reported_gp_anchor=spec["gp_anchor"],
            period_anchor=spec["period_anchor"],
            later_comparative_columns_used=False,
            numerical_repair_applied=False,
        )
        proofs.append(proof)
    return proofs


def build_scope_replay(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Freeze masks before refitting, preserve all rows/cases and use only known returns."""
    c = json.loads(config_path.read_bytes())
    validate_contract(c)

    def load(group: str, name: str) -> Any:
        return _json(Path(c[f"{group}_root"]), c[f"{group}_manifest_sha256"], name)

    expansion = load("expansion_plan", "plan.json")
    review = load("review", "summary.json")
    old_summary = load("baseline", "summary.json")
    if (
        review["plan_manifest_sha256"] != c["expansion_plan_manifest_sha256"]
        or not review["new_return_diagnostic_blocked"]
        or old_summary["complete_case_events"] != 6
    ):
        raise ValueError("Scope finding or baseline sample identity changed")
    source = expansion["source_plan"]["financial_sources"]
    cases = expansion["controls"]
    if len(cases) != c["frozen_case_count"]:
        raise ValueError("Do not replace frozen baseline cases")
    original = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        )
    )
    if len(original) != c["frozen_history_events"]:
        raise ValueError("Financial history denominator changed")
    sun = original.loc[original.company_id.eq(c["issuer_cik"])]
    if len(sun) != c["frozen_issuer_events"]:
        raise ValueError("SUN history denominator changed")
    config = _config(
        expansion["base_config"]["source_replay_config"],
        expansion["base_config"]["source_replay_config_sha256"],
    )
    clock_plan = _json(
        Path(config["clock_plan_root"]), config["clock_plan_manifest_sha256"], "plan.json"
    )
    cfg = PilotFilingFeatureConfig(**clock_plan["forecast_config"])
    cfg.validate()
    if old_summary["forecast_config"] != clock_plan["forecast_config"]:
        raise ValueError("Frozen model thresholds/features changed")
    joint = pd.read_parquet(
        _artifact(Path(c["baseline_root"]), c["baseline_manifest_sha256"], "joint_history.parquet")
    )
    if joint.event_id.duplicated().any() or set(joint.event_id) != set(original.event_id):
        raise ValueError("Baseline source history identity changed")
    original = original.set_index("event_id").loc[joint.event_id].reset_index()
    for field in (
        "company_id",
        "filing_accession",
        "revenue",
        "gross_profit",
        "gross_margin",
        "period_start",
        "period_end",
    ):
        pd.testing.assert_series_equal(original[field], joint[field], check_names=False)
    history = joint[HISTORY_COLUMNS].copy()
    history["derivation_method"] = original.derivation_method
    history["gross_profit_fact_id"] = original.gross_profit_fact_id
    masks = {arm: scope_quarantine_view(history, arm, c["issuer_cik"]) for arm in ARMS}
    masked_ids = {
        arm: frame.loc[frame.scope_quarantined, "event_id"].tolist() for arm, frame in masks.items()
    }
    plan = {
        "config": c,
        "config_sha256": digest_file(config_path),
        "financial_sources": source,
        "cases": cases,
        "forecast_config": clock_plan["forecast_config"],
        "arms": list(ARMS),
        "excluded_event_ids": masked_ids,
        "all_history_event_ids": history.event_id.tolist(),
        "all_original_case_ids": [r["validation_event_id"] for r in cases],
        "forward_returns_read": False,
        "forecasts_refitted": False,
        "previous_outcomes_already_known": True,
        "confirmatory_preregistration": False,
        "no_training_filter_was_known_live_claim": True,
        "general_scope_gate_cleared": False,
        "source_values_not_overwritten": True,
        "implementation_sha256": {
            p: digest_file(Path(p))
            for p in (
                "scripts/build_margin_scope_replay.py",
                "sec_data/margin_scope_replay.py",
                "sec_data/announcement_scope_audit.py",
                "sec_data/pilot_filing_features.py",
                "sec_data/event_cleaning.py",
                "sec_data/clock_layer.py",
                "event_panel.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    print(
        f"SCOPE PLAN frozen=True excluded={ {a: len(ids) for a, ids in masked_ids.items()} } "
        f"output={frozen}",
        flush=True,
    )
    # Reproduce every original SUN component from the same immutable snapshot.
    catalog = pd.read_parquet(
        _artifact(
            Path(source["catalog_root"]),
            source["catalog_manifest_sha256"],
            "filing_catalog.parquet",
        ),
        filters=[("cik", "==", c["issuer_cik"])],
    )
    verify_file(Path(source["companyfacts_zip"]), source["companyfacts_sha256"])
    with zipfile.ZipFile(source["companyfacts_zip"]) as archive:
        member = archive.read(f"CIK{c['issuer_cik']}.json")
    facts = parse_companyfacts(
        json.loads(member),
        catalog,
        source_url="https://data.sec.gov/api/xbrl/companyfacts/" + f"CIK{c['issuer_cik']}.json",
        snapshot_hash=source["companyfacts_sha256"],
    ).data
    recipes = audit_original_recipes(sun, facts)
    cache = load_cached_submissions([Path(p) for p in source["cached_submission_roots"]])
    proofs = [
        load("review", "SUN_scope_discrepancy.json"),
        *_additional_proofs(c, sun, facts, catalog, cache),
    ]
    if len({(p["company_id"], p["filing_accession"]) for p in proofs}) != len(proofs):
        raise ValueError("Duplicate source discrepancy observations")
    source_audit = write_bundle(
        output_root,
        VERSION + "-source-audit",
        {
            "SUN_original_recipe_audit.json": _encode(_rows(recipes)),
            "confirmed_original_scope_discrepancies.json": _encode(proofs),
            "summary.json": _encode(
                {
                    "SUN_history_events": len(sun),
                    "original_recipes_verified": len(recipes),
                    "direct_GrossProfit_recipes": int(
                        sun.derivation_method.eq("direct_gross_profit").sum()
                    ),
                    "cost_derived_recipes": int(
                        sun.derivation_method.eq("revenue_minus_total_cost").sum()
                    ),
                    "same_original_quarter_discrepancies": len(proofs),
                    "companyfacts_member_sha256": hashlib.sha256(member).hexdigest(),
                    "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
                    "all_SUN_history_proven_wrong": False,
                    "cross_year_accounting_scope_certified": False,
                    "numerical_repairs_applied": False,
                    "network_requests": 0,
                }
            ),
        },
    )
    print(
        f"SCOPE AUDIT recipes={len(recipes)} discrepancies={len(proofs)} output={source_audit}",
        flush=True,
    )
    histories = {ARMS[0]: masks[ARMS[0]]}
    for arm in ARMS[1:]:
        print(f"SCOPE REPLAY start={arm}", flush=True)
        histories[arm] = replay_scope_view(history, arm, c["issuer_cik"], cfg)
        print(
            f"SCOPE REPLAY done={arm} chronology={audit_replay_chronology(histories[arm])}",
            flush=True,
        )
    signals = scope_case_signals(histories, cases)
    # Independently reconstructed training/error membership must contain no excluded labels.
    dependencies: list[dict[str, Any]] = []
    for arm, frame in histories.items():
        for case in cases:
            row = frame.loc[
                frame.company_id.eq(case["cik"])
                & frame.filing_accession.eq(case["target_accession_number"])
            ].iloc[0]
            deps = target_dependencies(frame, str(row.event_id))
            if set(deps.event_id) & set(masked_ids[arm]):
                raise ValueError("Quarantined event entered target training or prior OOS errors")
            dependencies.extend(
                {"arm": arm, "ticker": case["assigned_ticker"], **r} for r in _rows(deps)
            )
    signal_freeze = write_bundle(
        output_root / "plans",
        VERSION + "-signal-freeze",
        {
            "signals.json": _encode(_rows(signals)),
            "contract.json": _encode(
                {
                    "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
                    "returns_read": False,
                    "same_six_cases_complete_in_all_arms": True,
                    "fixed_forecast_config": clock_plan["forecast_config"],
                }
            ),
        },
    )
    returns = pd.DataFrame(load("baseline", "case_returns.json"))
    sample, stats, influence = scope_rank_diagnostics(signals, returns)
    old_stats = load("baseline", "descriptive_ic.json")
    for signal in SIGNALS:
        expected = next(
            r["spearman_ic"]
            for r in old_stats
            if r["arm"] == "proxy_joint" and r["representation"] == signal
        )
        actual = stats.loc[
            stats.arm.eq(ARMS[0]) & stats.representation.eq(signal), "spearman_ic"
        ].iloc[0]
        if abs(actual - expected) > 1e-12:
            raise ValueError("Pinned baseline descriptive result does not reproduce")
    base_signals = signals.loc[signals.arm.eq(ARMS[0])].set_index("validation_event_id")
    changes = []
    for arm in ARMS[1:]:
        after = signals.loc[signals.arm.eq(arm)].set_index("validation_event_id")
        for signal in SIGNALS:
            difference = after[signal] - base_signals[signal]
            changes.append(
                {
                    "arm": arm,
                    "representation": signal,
                    "changed_cases_above_1e_minus_10": int(difference.abs().gt(1e-10).sum()),
                    "max_absolute_signal_change": float(difference.abs().max()),
                    "same_six_case_ranks_unchanged": bool(
                        after[signal].rank().eq(base_signals[signal].rank()).all()
                    ),
                    "signal_signs_unchanged": bool(
                        after[signal].gt(0).eq(base_signals[signal].gt(0)).all()
                    ),
                    "unit": "prior_OOS_sigma_units"
                    if signal == "gm_innovation_std"
                    else "margin_fraction",
                }
            )
    summary = {
        "version": VERSION,
        "frozen_cohort_events_retained": 316,
        "history_events_retained": len(history),
        "complete_case_events": 6,
        "issuers": 6,
        "SUN_original_recipes_verified": len(recipes),
        "confirmed_same_quarter_scope_discrepancies": len(proofs),
        "excluded_margin_records_by_arm": {a: len(ids) for a, ids in masked_ids.items()},
        "case_tickers": [r["assigned_ticker"] for r in cases],
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "source_audit_manifest_sha256": digest_file(source_audit / "output_manifest.json"),
        "source_audit_root": str(source_audit),
        "signal_freeze_manifest_sha256": digest_file(signal_freeze / "output_manifest.json"),
        "forecast_config": clock_plan["forecast_config"],
        "baseline_IC_reproduced": True,
        "signal_change_summary": changes,
        "chronology_by_arm": {a: audit_replay_chronology(f) for a, f in histories.items()},
        "general_scope_gate_cleared": False,
        "expanded_fourteen_case_returns_computed": False,
        "policy_basis": c["policy_basis"],
        "quarantine_is_not_numerical_repair": True,
        "historical_returns_are_not_new_holdout": True,
        "inferential_tests_computed": False,
        "first_public_certified": False,
        "trading_admission": False,
        "strategy_pnl_computed": False,
        "network_requests": 0,
        "originals_overwritten": False,
        "hyperparameters_retuned": False,
        "verdict": "accounting_scope_sensitivity_only_effectiveness_not_established",
        "limitations": [
            "six_unrepresentative_previously_examined_cases",
            "ex_post_scope_filter_not_known_live",
            "direct_GrossProfit_concept_not_cross_year_scope_certification",
            "unreviewed_cost_derived_values_not_all_proven_wrong",
            "prior_announcement_clock_and_price_delivery_limits_remain",
        ],
    }
    files = {
        "summary.json": _encode(summary),
        "signals.json": _encode(_rows(signals)),
        "same_case_analysis.json": _encode(_rows(sample)),
        "descriptive_ic.json": _encode(_rows(stats)),
        "leave_one_issuer_out.json": _encode(_rows(influence)),
        "dependency_membership.json": _encode(dependencies),
        "confirmed_original_scope_discrepancies.json": _encode(proofs),
        "SUN_original_recipe_audit.json": _encode(_rows(recipes)),
        **{a + "_history.parquet": _parquet(f) for a, f in histories.items()},
    }
    return write_bundle(output_root, VERSION, files), summary


def main() -> None:
    """Run source-provenance checks and only the predeclared quarantine sensitivity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/margin_scope_replay_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/margin_scope_replay")
    )
    args = parser.parse_args()
    root, report = build_scope_replay(args.config, args.output_root)
    print(
        f"SCOPE REPLAY DONE cases={report['complete_case_events']} "
        f"verdict={report['verdict']} output={root}"
    )


if __name__ == "__main__":
    main()
