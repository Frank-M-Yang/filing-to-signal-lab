"""Complete-pair acceptance-proxy replays, never preliminary-date backfills or Alpha tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_admission import audit_forecast_metadata, review_candidate_pair
from sec_data.announcement_replay import (
    SIGNALS,
    apply_release_proxies,
    case_row,
    company_dependencies,
    verify_target_reconstruction,
)
from sec_data.clock_layer import FLAG_COLUMNS, compare_histories
from sec_data.disclosure_audit import utc_timestamp
from sec_data.pilot_filing_features import PilotFilingFeatureConfig

VERSION = "next-disclosure-replay-v1"
COMPARISON_FIELDS = (*SIGNALS, "gm_forecast", "forecast_sigma")
DISARM_FLAGS = (
    "first_public_certified",
    "model_admission",
    "return_admission",
    "trading_admission",
)


def select_complete_pairs(
    dispositions: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep all six verified pairs from the twenty-source queue, not six chosen outcomes.

    The compatibility alias earliest_disclosure_at refers ONLY to the complete
    pair's replay cutoff. The frozen earlier candidate clock is retained under
    frozen_earliest_candidate_at; neither prior artifacts nor partial sets change.
    """
    if (
        len(dispositions) != 20
        or len({r["validation_event_id"] for r in dispositions}) != 20
        or len({r["cik"] for r in dispositions}) != 20
        or any(r.get(k) is not False for r in dispositions for k in DISARM_FLAGS)
    ):
        raise ValueError("Frozen denominator or disarmed source dispositions changed")
    selected = [r for r in dispositions if r["conditional_pair_verified"] is True]
    if len(selected) != 6:
        raise ValueError("Replay requires the entire frozen six-pair evidence category")
    result = []
    for source in selected:
        pairs = [p for p in source["reviewed_pairs"] if p["conditional_pair_verified"] is True]
        if len(pairs) != 1 or source["existing_scope_quarantine"] is not False:
            raise ValueError("Ambiguous complete pair or existing scope quarantine")
        pair = pairs[0]
        cutoff = utc_timestamp(pair["pair_acceptance_proxy_at"])
        early, report = (
            utc_timestamp(source["earliest_disclosure_at"]),
            utc_timestamp(source["accepted_at"]),
        )
        if (
            not pd.Timestamp(source["period_end"], tz="UTC") < cutoff < report
            or cutoff < early
            or cutoff != utc_timestamp(source["pair_acceptance_proxy_at"])
            or pair["exact_original_pair"] is not True
            or pair["clock_consistent"] is not True
            or pair["later_than_frozen_earliest_proxy"] != (cutoff > early)
        ):
            raise ValueError("Complete pair clock cannot backfill the earlier candidate")
        case = {
            **source,
            "frozen_earliest_candidate_at": source["earliest_disclosure_at"],
            "frozen_earliest_candidate_accession": source["release_accession_number"],
            "release_accession_number": pair["release_accession"],
            "earliest_disclosure_at": cutoff.isoformat(),
            "release_accepted_at": cutoff.isoformat(),
            "complete_pair_replay_cutoff_at": cutoff.isoformat(),
            "clock_source": "item_2_02_8k",
            "report_date": source["period_end"],
            "release_pair_source_sha256": pair["submission_sha256"],
            "release_pair_exhibit_sha256": pair["exhibit_sha256"],
            "earlier_candidate_backfill_allowed": False,
        }
        proof = review_candidate_pair(case, cells)
        if (
            proof["release_value_pair_verified"] is not True
            or proof["release_pair_source_sha256"] != pair["submission_sha256"]
            or proof["release_pair_exhibit_sha256"] != pair["exhibit_sha256"]
            or not np.isclose(
                pair["gross_margin"], proof["release_gross_margin"], rtol=0, atol=1e-12
            )
        ):
            raise ValueError("Complete-pair source, exact original cells or margin changed")
        result.append(case)
    return sorted(result, key=lambda r: utc_timestamp(r["complete_pair_replay_cutoff_at"]))


def overlay_complete_pairs(
    history: pd.DataFrame, cases: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> pd.DataFrame:
    """Change only six quarter clocks, preserving the conservative scope mask and all originals."""
    if len(cases) != 6 or any(
        utc_timestamp(c["complete_pair_replay_cutoff_at"])
        != utc_timestamp(c["earliest_disclosure_at"])
        for c in cases
    ):
        raise ValueError("All six complete-pair cutoffs are required")
    for case in cases:
        row = case_row(history, case)
        if bool(row.scope_quarantined) or row.derivation_method != "direct_gross_profit":
            raise ValueError("Quarantined or cost-derived target cannot enter this replay")
    result = apply_release_proxies(history, cases, cells)
    pd.testing.assert_frame_equal(
        history.drop(columns="actual_available_at"), result.drop(columns="actual_available_at")
    )
    return result


def verify_prefix_outputs(before: pd.DataFrame, after: pd.DataFrame, cutoff: str) -> int:
    """A later source overlay must leave every earlier reconstructed output unchanged."""
    boundary = utc_timestamp(cutoff)
    a = before.loc[before.actual_available_at.lt(boundary)]
    b = after.loc[after.actual_available_at.lt(boundary)]
    if not compare_histories(a, b)[1].empty:
        raise ValueError("Later complete-pair evidence changed earlier replay outputs")
    return len(a)


def audit_replayed_cases(
    baseline: pd.DataFrame,
    replayed: pd.DataFrame,
    cases: list[dict[str, Any]],
    config: PilotFilingFeatureConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce each prior-only OLS/sigma membership and expose common-case financial signals.

    Missing forecasts stay in the case ledger. No representation gets a different
    case subset: a single incomplete case closes the six-case comparison gate.
    Predictions are retrospective reconstructions, not contemporaneous saved forecasts.
    """
    if len(cases) != 6 or len({r["validation_event_id"] for r in cases}) != 6:
        raise ValueError("The frozen six-case denominator changed")
    audit: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    company: list[dict[str, Any]] = []
    for case in cases:
        old, new = case_row(baseline, case), case_row(replayed, case)
        cutoff = case["complete_pair_replay_cutoff_at"]
        if utc_timestamp(new.actual_available_at) != utc_timestamp(cutoff):
            raise ValueError("Replayed target clock is not the complete-pair cutoff")
        clocks = audit_forecast_metadata({str(k): v for k, v in new.items()}, cutoff)
        prior = audit_forecast_metadata({str(k): v for k, v in old.items()}, cutoff)
        complete = bool(
            all(bool(new[f]) for f in FLAG_COLUMNS)
            and np.isfinite(new[list(COMPARISON_FIELDS)].to_numpy(dtype=float)).all()
            and clocks["existing_forecast_clocks_before_proxy"]
        )
        if complete:
            deps = verify_target_reconstruction(replayed, case)
            if set(deps.event_id) & set(replayed.loc[replayed.scope_quarantined, "event_id"]):
                raise ValueError("Quarantined training label or OOS error entered replay")
            if (
                int(new.forecast_training_observations) < config.minimum_forecast_training_events
                or int(new.forecast_sigma_observations) < config.minimum_prior_oos_errors
            ):
                raise ValueError("Fixed prior-training or prior-OOS sigma threshold weakened")
            dependencies.extend(
                {
                    "validation_event_id": case["validation_event_id"],
                    "ticker": case["ticker"],
                    **{str(k): v for k, v in r.items()},
                }
                for r in deps.to_dict("records")
            )
            company.extend(
                {
                    "validation_event_id": case["validation_event_id"],
                    "ticker": case["ticker"],
                    **{str(k): v for k, v in r.items()},
                }
                for r in company_dependencies(replayed, new, config).to_dict("records")
            )
            clocks["history_replay_completed"] = True
        audit.append(
            {
                "validation_event_id": case["validation_event_id"],
                "cik": case["cik"],
                "ticker": case["ticker"],
                "frozen_earliest_candidate_at": case["frozen_earliest_candidate_at"],
                "complete_pair_replay_cutoff_at": cutoff,
                "prior_periodic_report_actual_at": old.actual_available_at,
                "old_forecast_clock_check": prior,
                "replayed_forecast_clock_check": clocks,
                "complete_financial_representations": complete,
                "forecasts_reconstructed_not_live": True,
                "reconstructed_generation_clock": new.forecast_generated_at,
                "training_observations": new.forecast_training_observations,
                "prior_oos_errors": new.forecast_sigma_observations,
                "missing_representations": [s for s in SIGNALS if pd.isna(new[s])],
                **dict.fromkeys(DISARM_FLAGS, False),
                "return_window_verified": False,
                "earlier_candidate_backfill_allowed": False,
            }
        )
        for field in COMPARISON_FIELDS:
            a, b = (float(v) if pd.notna(v) else np.nan for v in (old[field], new[field]))
            finite = np.isfinite([a, b]).all()
            comparisons.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "ticker": case["ticker"],
                    "representation": field,
                    "periodic_report_clock_value": a if np.isfinite(a) else None,
                    "complete_pair_clock_value": b if np.isfinite(b) else None,
                    "change": b - a if finite else None,
                    "changed": bool(abs(b - a) > 1e-10) if finite else None,
                    "direction_flipped": bool(a * b < 0 and abs(a) > 1e-10 and abs(b) > 1e-10)
                    if finite
                    else None,
                    "unit": "prior_OOS_sigma_units"
                    if field == "gm_innovation_std"
                    else "margin_fraction",
                    "old_value_is_valid_at_complete_pair_cutoff": prior[
                        "existing_forecast_clocks_before_proxy"
                    ],
                }
            )
    common = all(r["complete_financial_representations"] for r in audit)
    for row in audit:
        row["all_six_common_financial_cases_ready"] = common
    return audit, comparisons, dependencies, company
