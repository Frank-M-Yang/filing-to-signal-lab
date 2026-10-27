"""Review all twenty frozen issuers and stop new inference on proven scope discrepancies."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_announcement_admission import _config, _recheck_cells
from scripts.build_announcement_return_diagnostic import _rows
from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.announcement_admission import review_candidate_pair
from sec_data.announcement_scope_audit import audit_margin_scope, bind_reconciliation_value
from sec_data.disclosure_audit import digest_file, load_cached_submissions, verify_file
from sec_data.disclosure_review import _bound_exhibit, review_document_kinds, review_metric_cells
from sec_data.publisher_replay import target_dependencies


def review_batch(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Recheck original eight GAAP pairs and isolate SUN's mismatch before model replay."""
    c = json.loads(config_path.read_bytes())
    if c["version"] != "announcement-expansion-review-v1" or any(
        c[k] is not False
        for k in (
            "returns_used_for_review",
            "first_public_certified",
            "trading_admission",
            "model_retuning_allowed",
            "rounding_plugs_allowed",
            "sample_replacement_allowed",
        )
    ):
        raise ValueError("Unsupported or weakened statement review")

    def prepared(name: str) -> Any:
        return _json(Path(c["prepared_root"]), c["prepared_manifest_sha256"], name)

    plan = _json(Path(c["plan_root"]), c["plan_manifest_sha256"], "plan.json")
    if prepared("summary.json")["selection_plan_manifest_sha256"] != c["plan_manifest_sha256"]:
        raise ValueError("Prepared evidence and frozen selection differ")
    selected = plan["selected_cases"]
    wanted = {r["assigned_ticker"] for r in selected}
    reviewed = [r["ticker"] for r in c["reviews"]]
    blocked = [r["ticker"] for r in c["blocked_cases"]]
    if (
        len(selected) != 20
        or len(set(reviewed + blocked)) != len(reviewed + blocked)
        or set(reviewed + blocked) != wanted
    ):
        raise ValueError("Every frozen issuer needs exactly one disposition; no replacements")
    source = plan["source_plan"]["financial_sources"]
    cache = load_cached_submissions([Path(p) for p in source["cached_submission_roots"]])
    docs = prepared("documents.json")
    events = plan["events"]
    candidates = prepared("selected_candidates.json")
    fields = prepared("field_audit.json")

    def specification(review: dict[str, Any]) -> dict[str, Any]:
        matching = [
            d
            for d in docs
            if d["ticker"] == review["ticker"]
            and d["exhibit_filename"] == review["exhibit_filename"]
        ]
        if len(matching) != 1:
            raise ValueError("Review exhibit is missing or ambiguous")
        return {
            **matching[0],
            **review,
            "kind": "earnings_release",
            "anchors": [review["document_anchor"]],
            "accounting_basis": "GAAP",
        }

    specs = [specification(r) for r in c["reviews"]]
    dispositions = review_document_kinds(specs, events, candidates, cache)
    cells = review_metric_cells(specs, events, fields, candidates, cache)
    source_plan = plan["source_plan"]
    prior_review = _config(
        source_plan["config"]["review_config"], source_plan["config"]["review_config_sha256"]
    )
    old_cells, _, old_source = _recheck_cells(prior_review)
    if old_source != source:
        raise ValueError("Expansion changed the financial evidence chain")
    all_cells = [*old_cells, *cells]
    financial_cases, ledger = list(plan["controls"]), []
    for case in selected:
        if case["assigned_ticker"] in reviewed:
            result = review_candidate_pair(case, cells)
            if not result["release_value_pair_verified"]:
                raise ValueError("Reviewed pair fails exact original or proxy checks")
            financial_cases.append({**case, **result})
            status = "original_GAAP_quarter_pair_rechecked"
        else:
            result = {"release_value_pair_verified": False}
            status = next(
                r["reason"] for r in c["blocked_cases"] if r["ticker"] == case["assigned_ticker"]
            )
        ledger.append({**case, **result, "expansion_status": status, "return_admission": False})
    # Bind the observed discrepancy to exactly the reviewed source and original cost facts.
    scope_spec = specification(c["scope_review"])
    sun_event = next(r for r in events if r["ticker"] == "SUN")
    text, clock = _bound_exhibit(scope_spec, sun_event, candidates, cache)
    if clock["status"] != "consistent_acceptance_proxy":
        raise ValueError("Scope evidence clock is quarantined")
    segment = bind_reconciliation_value(
        text,
        scope_spec["segment_profit_anchor"],
        "Total segment profit",
        scope_spec["segment_profit_display_value"],
        scope_spec["scale"],
    )
    depreciation = bind_reconciliation_value(
        text,
        scope_spec["depreciation_anchor"],
        "Depreciation, amortization and accretion, excluding corporate and other",
        scope_spec["depreciation_display_value"],
        scope_spec["scale"],
    )
    if (
        segment != scope_spec["segment_profit"]
        or depreciation != scope_spec["depreciation_excluding_corporate"]
    ):
        raise ValueError("Configured scope values differ from original source cells")
    review_document_kinds([scope_spec], events, candidates, cache)
    scope_cells = review_metric_cells([scope_spec], events, fields, candidates, cache)
    revenue_cell = next(r for r in scope_cells if r["metric"] == "revenue")
    if revenue_cell["comparison_status"] != "exact_original_match":
        raise ValueError("Scope proof requires an exact original revenue denominator")
    gp = next(r for r in scope_cells if r["metric"] == "gross_profit")
    if gp["value"] != scope_spec["reported_gaap_gross_profit"]:
        raise ValueError("Scope proof value disagrees with the reviewed original source")
    history = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        )
    )
    sun = history.loc[
        history.company_id.eq(sun_event["cik"])
        & history.filing_accession.eq(sun_event["filing_accession"])
    ]
    if len(sun) != 1:
        raise ValueError("Scope observation is not unique in frozen model history")
    target = _rows(sun)[0]
    if target["revenue"] != revenue_cell["value"]:
        raise ValueError("Model history revenue differs from the reviewed original denominator")
    for key in ("period_start", "period_end"):
        target[key] = target[key][:10]
    verify_file(Path(source["companyfacts_zip"]), source["companyfacts_sha256"])
    with zipfile.ZipFile(source["companyfacts_zip"]) as archive:
        raw = archive.read(f"CIK{sun_event['cik']}.json")
    companyfacts = json.loads(raw)
    costs: list[dict[str, Any]] = []
    for tag in ("CostOfRevenue", "CostOfGoodsAndServicesSold"):
        rows = (
            companyfacts.get("facts", {})
            .get("us-gaap", {})
            .get(tag, {})
            .get("units", {})
            .get("USD", [])
        )
        costs.extend(
            {**r, "tag": tag, "unit": "USD"}
            for r in rows
            if r.get("accn") == target["filing_accession"]
            and r.get("start") == target["period_start"]
            and r.get("end") == target["period_end"]
        )
    scope = audit_margin_scope(target, costs, gp["value"], segment, depreciation)
    scope.update(
        source_submission_sha256=scope_spec["submission_sha256"],
        source_exhibit_sha256=scope_spec["exhibit_sha256"],
        release_accession=scope_spec["release_accession"],
        release_acceptance_proxy_at=gp["sec_acceptance_proxy_at"],
        filing_accepted_at=sun_event["accepted_at"],
        original_cost_available_at=sun_event["accepted_at"],
        original_cost_rows=costs,
        source_companyfacts_zip_sha256=source["companyfacts_sha256"],
        source_revenue_fact_id=target["revenue_fact_id"],
        source_cost_fact_id=target["cost_fact_id"],
        segment_profit_anchor=scope_spec["segment_profit_anchor"],
        depreciation_anchor=scope_spec["depreciation_anchor"],
        reported_gp_anchor=gp["text_anchor"],
    )
    # Read dependencies, not returns; determine whether the observed row entered old forecasts.
    replay = _config(
        plan["base_config"]["source_replay_config"],
        plan["base_config"]["source_replay_config_sha256"],
    )
    clock_history = pd.read_parquet(
        _artifact(
            Path(replay["clock_root"]),
            replay["clock_manifest_sha256"],
            "disarmed_replay_panel.parquet",
        )
    )
    deps = []
    for case in plan["controls"]:
        row = history.loc[
            history.company_id.eq(case["cik"])
            & history.filing_accession.eq(case["target_accession_number"])
        ].iloc[0]
        members = target_dependencies(history, str(row.event_id))
        matched = members.loc[members.event_id.eq(target["event_id"])]
        deps.append(
            {
                "ticker": case["assigned_ticker"],
                "validation_event_id": case["validation_event_id"],
                "observed_scope_row_in_training": bool(
                    not matched.empty and matched.in_training.any()
                ),
                "observed_scope_row_in_sigma_history": bool(
                    not matched.empty and matched.in_prior_oos_errors.any()
                ),
            }
        )
    # Keep the source snapshot/overlay relationship explicit; do not recompute compromised models.
    if len(clock_history) != len(history):
        raise ValueError("Clock/history denominator changed")
    derived = history.loc[
        history.derivation_method.eq("revenue_minus_total_cost"),
        [
            "event_id",
            "company_id",
            "filing_accession",
            "period_start",
            "period_end",
            "revenue",
            "gross_profit",
            "revenue_fact_id",
            "cost_fact_id",
            "derivation_method",
        ],
    ]
    prior = pd.DataFrame(prepared("prior_industry.json"))
    summary = {
        "version": "announcement-expansion-reviewed-v1",
        "review_config_sha256": digest_file(config_path),
        "prepared_manifest_sha256": c["prepared_manifest_sha256"],
        "plan_manifest_sha256": c["plan_manifest_sha256"],
        "frozen_cohort_events_retained": 316,
        "new_issuers_reviewed": 20,
        "new_original_pairs_verified": len(cells) // 2,
        "blocked_new_cases_retained": len(blocked),
        "financial_pair_candidates_including_controls": len(financial_cases),
        "prior_industry_matches": int(prior.industry_ready.sum()),
        "industry_cases": len(prior),
        "derived_margin_history_review_queue": len(derived),
        "SUN_history_events_for_scope_review": int(history.company_id.eq(sun_event["cik"]).sum()),
        "control_forecasts_directly_using_observed_scope_row": [
            r["ticker"] for r in deps if r["observed_scope_row_in_training"]
        ],
        "new_return_diagnostic_blocked": True,
        "block_reason": scope["status"],
        "existing_results_require_scope_recheck": True,
        "prior_results_overwritten": False,
        "first_public_certified": False,
        "trading_admission": False,
        "network_requests": 0,
        "return_values_read": False,
        "models_refitted": False,
        "industry_adjusted_IC_computed": False,
        "industry_minimum_events_unchanged": 20,
    }
    files = {
        "summary.json": _encode(summary),
        "reviewed_metric_cells.json": _encode(cells),
        "all_old_and_new_cells.json": _encode(all_cells),
        "reviewed_document_kinds.json": _encode(dispositions),
        "new_case_dispositions.json": _encode(ledger),
        "financial_pair_candidates.json": _encode(financial_cases),
        "scope_cells.json": _encode(scope_cells),
        "SUN_scope_discrepancy.json": _encode(scope),
        "original_control_dependency_audit.json": _encode(deps),
        "derived_margin_review_queue.json": _encode(_rows(derived)),
        "prior_industry.json": _encode(_rows(prior)),
        "implementation_sha256.json": _encode(
            {
                p: digest_file(Path(p))
                for p in (
                    "scripts/review_announcement_expansion.py",
                    "sec_data/announcement_scope_audit.py",
                    "sec_data/announcement_expansion.py",
                    "sec_data/disclosure_review.py",
                    "sec_data/announcement_admission.py",
                )
            }
        ),
    }
    return write_bundle(output_root, "announcement-expansion-reviewed-v1", files), summary


def main() -> None:
    """Finish source checks and report the fail-closed margin-scope gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/announcement_expansion_review_v1.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/announcement_expansion")
    )
    args = parser.parse_args()
    root, report = review_batch(args.config, args.output_root)
    print(
        f"EXPANSION REVIEWED new_pairs={report['new_original_pairs_verified']} "
        f"candidates={report['financial_pair_candidates_including_controls']} "
        f"return_diagnostic_blocked=True output={root}"
    )


if __name__ == "__main__":
    main()
