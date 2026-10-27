"""Finite ordering regimes on a frozen proxy history, not historical publication certification."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_admission import audit_forecast_metadata
from sec_data.announcement_replay import (
    SIGNALS,
    case_row,
    company_dependencies,
    verify_target_reconstruction,
)
from sec_data.clock_layer import FLAG_COLUMNS, NUMERIC_COLUMNS, compare_histories
from sec_data.disclosure_audit import utc_timestamp
from sec_data.next_publication import DISARM_FLAGS
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig

VERSION = "publication-interval-replay-v1"
COMPARISONS = (*SIGNALS, "gm_forecast", "forecast_sigma")
HISTORY_FORBIDDEN = {"open_unadjusted", "close_unadjusted", "future_return", "market_excess_return"}


def validate_sources(
    history: pd.DataFrame, cases: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require all six original pairs and retain explicit minute/unknown-offset uncertainty."""
    if (
        len(cases) != 6
        or len(sources) != 6
        or len({c["validation_event_id"] for c in cases}) != 6
        or len({(c["cik"], c["target_accession_number"]) for c in cases}) != 6
        or len({s["source_id"] for s in sources}) != 6
        or {s["ticker"] for s in sources} != {c["ticker"] for c in cases}
        or history.event_id.duplicated().any()
        or HISTORY_FORBIDDEN.intersection(history.columns)
        or any(c.get(k) is not False for c in cases for k in DISARM_FLAGS)
    ):
        raise ValueError("Frozen six-source cohort, financial-only history or disarm flags changed")
    result = []
    for case in cases:
        selected = [s for s in sources if s["ticker"] == case["ticker"]]
        if len(selected) != 1:
            raise ValueError("Ambiguous publication source")
        source = selected[0]
        if (
            any(
                source[k] != case[k]
                for k in ("cik", "ticker", "period_start", "period_end", "validation_event_id")
            )
            or source["release_accession"] != case["release_accession_number"]
            or source["exact_original_pair"] is not True
            or any(source.get(k) is not False for k in DISARM_FLAGS)
            or source["clock_resolution_seconds"] != 60
        ):
            raise ValueError("Source is not the same disarmed exact original quarter")
        lower, upper = (utc_timestamp(source[k]) for k in ("lower_inclusive", "upper_exclusive"))
        width = upper - lower
        if source["clock_precision"] == "minute_explicit_offset":
            if (
                width != pd.Timedelta(minutes=1)
                or utc_timestamp(source["source_reported_at_utc"]) != lower
            ):
                raise ValueError("Changed explicit-minute interval")
        elif source["clock_precision"] == "minute_unknown_timezone":
            if source["source_reported_at_utc"] is not None or width != pd.Timedelta(
                hours=28, minutes=1
            ):
                raise ValueError("Unknown timezone must retain the frozen offset sensitivity range")
        else:
            raise ValueError("Unsupported publication precision")
        row = case_row(history, case)
        if (
            bool(row.scope_quarantined)
            or row.derivation_method != "direct_gross_profit"
            or pd.notna(row.ytd_filing_accession)
            or utc_timestamp(row.actual_available_at)
            != utc_timestamp(case["complete_pair_replay_cutoff_at"])
            or utc_timestamp(row.filing_accepted_at) != utc_timestamp(case["accepted_at"])
            or not pd.Timestamp(case["period_end"], tz="UTC")
            < lower
            < upper
            <= utc_timestamp(row.actual_available_at)
        ):
            raise ValueError("Target scope, original acceptance or baseline availability changed")
        values = {c["metric"]: c["value_usd"] for c in source["matched_cells"]}
        if (
            len(source["matched_cells"]) != 2
            or set(values) != {"revenue", "gross_profit"}
            or values["revenue"] <= 0
        ):
            raise ValueError("Exactly two positive-revenue original component cells required")
        if any(row[k] != v for k, v in values.items()) or not np.isclose(
            row.gross_margin, values["gross_profit"] / values["revenue"], rtol=0, atol=1e-12
        ):
            raise ValueError("Original financial values changed")
        result.append(source)
    intervals = sorted(
        (utc_timestamp(s["lower_inclusive"]), utc_timestamp(s["upper_exclusive"])) for s in result
    )
    if any(b[0] < a[1] for a, b in itertools.pairwise(intervals)):
        raise ValueError("Overlapping target intervals need a different ordering protocol")
    return result


def build_ordering_grid(
    history: pd.DataFrame,
    cases: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    max_scenarios: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate start, exact tie, immediately-after and end, without outcome selection.

    No finite sweep certifies the missing historical information arrivals. This
    sweep covers only ordering changes visible in the frozen financial proxy history.
    Stable minute intervals use their start plus one global all-end equivalence check.
    If the cartesian grid exceeds its cap, fail rather than drop cases or crossings.
    """
    if max_scenarios != 32:
        raise ValueError("Changed scenario cap requires a new protocol")
    reviewed = validate_sources(history, cases, sources)
    targets = {str(case_row(history, c).event_id) for c in cases}
    actual = pd.to_datetime(history.actual_available_at, utc=True)
    options, crossings = [], []
    for source in reviewed:
        lower, upper = (utc_timestamp(source[k]) for k in ("lower_inclusive", "upper_exclusive"))
        frame = history.loc[~history.event_id.isin(targets) & actual.ge(lower) & actual.lt(upper)]
        stamps = sorted({utc_timestamp(v) for v in frame.actual_available_at})
        points = {lower: "start"}
        if stamps:
            points[upper - pd.Timedelta(nanoseconds=1)] = "end"
            for i, stamp in enumerate(stamps):
                points[stamp] = f"arrival_{i + 1}_same_time"
                if stamp + pd.Timedelta(nanoseconds=1) < upper:
                    points[stamp + pd.Timedelta(nanoseconds=1)] = f"arrival_{i + 1}_after"
        options.append(
            [
                (source["ticker"], stamp.isoformat(), label)
                for stamp, label in sorted(points.items())
            ]
        )
        for _, row in frame.iterrows():
            crossings.append(
                {
                    "target_ticker": source["ticker"],
                    "source_id": source["source_id"],
                    "event_id": str(row.event_id),
                    "cik": row.company_id,
                    "filing_accession": row.filing_accession,
                    "proxy_arrival_at": utc_timestamp(row.actual_available_at).isoformat(),
                    "scope_quarantined": bool(row.scope_quarantined),
                    "baseline_training_eligible": bool(
                        row.analysis_ready_delta
                        and row[list(FORECAST_FEATURES)].notna().all()
                        and pd.notna(row.forecast_feature_cutoff)
                    ),
                    "baseline_prior_oos_error_available": bool(pd.notna(row.gm_innovation_raw)),
                    "historical_first_public_certified": False,
                }
            )
    size = int(np.prod([len(o) for o in options]))
    if size + 1 > max_scenarios:
        raise ValueError("Ordering grid exceeds cap; no crossings or issuers may be dropped")
    scenarios: list[dict[str, Any]] = [
        {
            "scenario_id": f"ordering_{index:02d}",
            "kind": "frozen_proxy_arrival_ordering",
            "cutoffs": {t: stamp for t, stamp, _ in choice},
            "positions": {t: label for t, _, label in choice},
            **dict.fromkeys(DISARM_FLAGS, False),
        }
        for index, choice in enumerate(itertools.product(*options))
    ]
    all_end = {
        s["ticker"]: (utc_timestamp(s["upper_exclusive"]) - pd.Timedelta(nanoseconds=1)).isoformat()
        for s in reviewed
    }
    equivalent = next(
        s["scenario_id"]
        for s in scenarios
        if all(
            utc_timestamp(s["cutoffs"][t]) == utc_timestamp(all_end[t])
            for t in {r["target_ticker"] for r in crossings}
        )
    )
    scenarios.append(
        {
            "scenario_id": "all_interval_ends",
            "kind": "no_crossing_endpoint_equivalence_check",
            "cutoffs": all_end,
            "equivalent_numeric_scenario": equivalent,
            **dict.fromkeys(DISARM_FLAGS, False),
        }
    )
    return scenarios, crossings


def overlay_intervals(
    history: pd.DataFrame,
    cases: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    cutoffs: dict[str, str],
    *,
    selected_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """Move verified original-quarter availability only, on a copy of one shared history."""
    reviewed = validate_sources(history, cases, sources)
    tickers = {c["ticker"] for c in cases}
    selected = tickers if selected_tickers is None else selected_tickers
    if set(cutoffs) != tickers or not selected or not selected.issubset(tickers):
        raise ValueError("All six explicit cutoffs are required; selected overlay must be nonempty")
    result = history.copy(deep=True)
    # Ordering probes use one nanosecond to separate strict ties. Parquet inputs
    # may have microsecond storage; promote before assignment, never round a tie.
    result["actual_available_at"] = pd.to_datetime(result["actual_available_at"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    for case, source in zip(cases, reviewed, strict=True):
        stamp = utc_timestamp(cutoffs[case["ticker"]])
        if (
            not utc_timestamp(source["lower_inclusive"])
            <= stamp
            < utc_timestamp(source["upper_exclusive"])
        ):
            raise ValueError("Scenario outside declared publisher-label sensitivity interval")
        if case["ticker"] in selected:
            row = case_row(result, case)
            result.loc[result.event_id.eq(row.event_id), "actual_available_at"] = stamp
    pd.testing.assert_frame_equal(
        history.drop(columns="actual_available_at"), result.drop(columns="actual_available_at")
    )
    return result


def audit_interval_cases(
    baseline: pd.DataFrame,
    replayed: pd.DataFrame,
    cases: list[dict[str, Any]],
    scenario: dict[str, Any],
    config: PilotFilingFeatureConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Independently reproduce six OLS forecasts and strictly prior OOS-error sigmas."""
    rows, dependencies, company = [], [], []
    if len(cases) != 6:
        raise ValueError("Six-case common denominator required")
    for case in cases:
        old, row = case_row(baseline, case), case_row(replayed, case)
        cutoff = utc_timestamp(scenario["cutoffs"][case["ticker"]])
        if utc_timestamp(row.actual_available_at) != cutoff:
            raise ValueError("Target does not use the requested scenario clock")
        clocks = audit_forecast_metadata({str(k): v for k, v in row.items()}, cutoff.isoformat())
        complete = bool(
            all(bool(row[f]) for f in FLAG_COLUMNS)
            and np.isfinite(row[list(COMPARISONS)].to_numpy(dtype=float)).all()
            and clocks["existing_forecast_clocks_before_proxy"]
        )
        if complete:
            deps = verify_target_reconstruction(replayed, case)
            own = company_dependencies(replayed, row, config)
            if (
                set(deps.event_id) & set(replayed.loc[replayed.scope_quarantined, "event_id"])
                or int(row.forecast_training_observations) < config.minimum_forecast_training_events
                or int(row.forecast_sigma_observations) < config.minimum_prior_oos_errors
            ):
                raise ValueError(
                    "Quarantined dependency or weakened fixed training/sigma thresholds"
                )
            dependencies.append(
                deps.assign(
                    scenario_id=scenario["scenario_id"],
                    target_event_id=case["validation_event_id"],
                    target_ticker=case["ticker"],
                )
            )
            company.append(
                own.assign(
                    scenario_id=scenario["scenario_id"],
                    target_event_id=case["validation_event_id"],
                    target_ticker=case["ticker"],
                )
            )
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "validation_event_id": case["validation_event_id"],
                "ticker": case["ticker"],
                "cik": case["cik"],
                "actual_available_at": cutoff.isoformat(),
                "complete_financial_representations": complete,
                "prior_forecast_valid_at_scenario_cutoff": audit_forecast_metadata(
                    {str(k): v for k, v in old.items()}, cutoff.isoformat()
                )["existing_forecast_clocks_before_proxy"],
                "forecast_clocks": {
                    k: None if pd.isna(row[k]) else utc_timestamp(row[k]).isoformat()
                    for k in (
                        "forecast_feature_cutoff",
                        "forecast_training_end",
                        "forecast_generated_at",
                        "forecast_error_history_end",
                    )
                },
                "training_observations": None
                if pd.isna(row.forecast_training_observations)
                else int(row.forecast_training_observations),
                "prior_oos_errors": None
                if pd.isna(row.forecast_sigma_observations)
                else int(row.forecast_sigma_observations),
                "values": {f: None if pd.isna(row[f]) else float(row[f]) for f in COMPARISONS},
                "baseline_values": {
                    f: None if pd.isna(old[f]) else float(old[f]) for f in COMPARISONS
                },
                "forecasts_reconstructed_not_live": True,
                **dict.fromkeys(DISARM_FLAGS, False),
            }
        )
    common = all(r["complete_financial_representations"] for r in rows)
    for row_dict in rows:
        row_dict["same_six_common_financial_gate"] = common
    return (
        rows,
        pd.concat(dependencies, ignore_index=True) if dependencies else pd.DataFrame(),
        pd.concat(company, ignore_index=True) if company else pd.DataFrame(),
    )


def require_stable_endpoint_equivalence(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Require no-arrival shifts to preserve values, after auditing interior arrivals."""
    counts, _ = compare_histories(a, b)
    if any(counts[k] for k in (*NUMERIC_COLUMNS, *FLAG_COLUMNS, "gm_yoy_lag_event_id")):
        raise ValueError("No-crossing endpoint equivalence failed; expand the ordering protocol")


def interval_signal_summary(
    rows: list[dict[str, Any]], *, tolerance: float = 1e-10
) -> list[dict[str, Any]]:
    """Report all six ranges on one common sample, not whichever cutoff looks best."""
    if tolerance != 1e-10 or not rows:
        raise ValueError("Fixed comparison tolerance and nonempty scenario output required")
    groups = sorted({r["scenario_id"] for r in rows})
    identities = {r["validation_event_id"] for r in rows}
    if len(identities) != 6 or any(
        len([r for r in rows if r["scenario_id"] == group]) != 6 for group in groups
    ):
        raise ValueError("Every scenario must retain the same six complete-case identities")
    common = all(r["same_six_common_financial_gate"] for r in rows)
    result = []
    for event_id in sorted(identities):
        selected = [r for r in rows if r["validation_event_id"] == event_id]
        if {r["scenario_id"] for r in selected} != set(groups) or len(selected) != len(groups):
            raise ValueError("Scenario/event identity duplicated or omitted")
        for field in COMPARISONS:
            values = [r["values"][field] for r in selected]
            valid = common and all(v is not None and np.isfinite(v) for v in values)
            lo, hi = (float(min(values)), float(max(values))) if valid else (None, None)
            baseline = selected[0]["baseline_values"][field]
            result.append(
                {
                    "validation_event_id": event_id,
                    "ticker": selected[0]["ticker"],
                    "representation": field,
                    "minimum": lo,
                    "maximum": hi,
                    "range": None if lo is None or hi is None else hi - lo,
                    "changed_across_clock_scenarios": None
                    if lo is None or hi is None
                    else hi - lo > tolerance,
                    "direction_changes_within_uncertainty": None
                    if lo is None or hi is None
                    else lo < -tolerance and hi > tolerance,
                    "baseline_value": baseline,
                    "max_abs_change_from_acceptance_proxy": None
                    if lo is None or hi is None or baseline is None
                    else max(abs(lo - baseline), abs(hi - baseline)),
                    "unit": "prior_OOS_sigma_units"
                    if field == "gm_innovation_std"
                    else "margin_fraction",
                    "all_scenarios_same_six_common_gate": common,
                    "range_is_sensitivity_not_a_confidence_interval": True,
                    **dict.fromkeys(DISARM_FLAGS, False),
                }
            )
    return result
