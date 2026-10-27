"""Fixed-specification release-proxy replays, not live forecasts or first-public claims."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sec_data.announcement_admission import audit_forecast_metadata, review_candidate_pair
from sec_data.clock_layer import compare_histories
from sec_data.disclosure_audit import utc_timestamp
from sec_data.joint_publisher_replay import assert_originals_preserved
from sec_data.pilot_filing_features import FORECAST_FEATURES, PilotFilingFeatureConfig
from sec_data.publisher_replay import target_dependencies

VERSION = "announcement-replay-v1"
STRESS_VERSION = "announcement-replay-v2-clock-failures"
SIGNALS = ("gm_level", "gm_delta_yoy", "gm_innovation_raw", "gm_innovation_std")


def select_replay_cases(
    cohort: list[dict[str, Any]], event_ids: list[str], policy: str
) -> list[dict[str, Any]]:
    """Select every member of a frozen evidence category, independently of prices or returns."""
    if len({r["validation_event_id"] for r in cohort}) != len(cohort):
        raise ValueError("Duplicate frozen cohort identity")
    if policy == "ready_cases":
        candidates = [r for r in cohort if r.get("local_replay_candidate") is True]
    elif policy == "reviewed_values_clock_failures":
        candidates = [
            r
            for r in cohort
            if r.get("release_value_pair_verified") is True
            and r.get("existing_forecast_clocks_before_proxy") is False
        ]
    else:
        raise ValueError("Unsupported frozen case-selection policy")
    if (
        not candidates
        or len(set(event_ids)) != len(event_ids)
        or {r["validation_event_id"] for r in candidates} != set(event_ids)
    ):
        raise ValueError("Cases must equal the entire previously frozen evidence category")
    return sorted(candidates, key=lambda r: r["earliest_disclosure_at"])


def verify_prefix_causality(
    baseline: pd.DataFrame,
    joint: pd.DataFrame,
    cases: list[dict[str, Any]],
    prefixes: dict[int, pd.DataFrame],
) -> dict[str, int]:
    """Require each later correction to leave all earlier prefix outputs unchanged."""
    cutoffs = [utc_timestamp(r["earliest_disclosure_at"]) for r in cases]
    if not cutoffs or cutoffs != sorted(cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Cases require distinct chronological release proxies")
    if set(prefixes) != set(range(1, len(cases))):
        raise ValueError("Missing causal prefix replay")
    earlier = baseline.actual_available_at.lt(cutoffs[0])
    if not compare_histories(baseline.loc[earlier], joint.loc[earlier])[1].empty:
        raise ValueError("Release replay changed history before the first correction")
    counts = {"before_first_correction": int(earlier.sum())}
    for size, prefix in prefixes.items():
        before_later = joint.actual_available_at.lt(cutoffs[size])
        if not compare_histories(prefix.loc[before_later], joint.loc[before_later])[1].empty:
            raise ValueError("Later correction changed earlier prefix information")
        counts[f"before_correction_{size + 1}"] = int(before_later.sum())
    return counts


def case_row(history: pd.DataFrame, case: dict[str, Any]) -> pd.Series:
    """Resolve exactly one issuer/event pair; accession alone is not an identity."""
    rows = history.loc[
        history.company_id.eq(case["cik"])
        & history.filing_accession.eq(case["target_accession_number"])
    ]
    if len(rows) != 1:
        raise ValueError("Missing or ambiguous case history identity")
    return rows.iloc[0]


def apply_release_proxies(
    history: pd.DataFrame, cases: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> pd.DataFrame:
    """Move only evidence-matched quarter availability, never acceptance, YTD or values."""
    if (
        not cases
        or len({c["validation_event_id"] for c in cases}) != len(cases)
        or len({(c["cik"], c["target_accession_number"]) for c in cases}) != len(cases)
        or history.event_id.duplicated().any()
    ):
        raise ValueError("Missing or duplicate replay case identity")
    result = history.copy(deep=True)
    for case in cases:
        if any(
            case.get(k) is not False
            for k in ("first_public_certified", "model_admission", "return_admission")
        ):
            raise ValueError("Release-proxy replay cannot promote admission")
        proof = review_candidate_pair(case, cells)
        if not proof["release_value_pair_verified"]:
            raise ValueError("Missing exact selected-release quarter-value proof")
        if any(
            case[k] != proof[k]
            for k in ("release_pair_source_sha256", "release_pair_exhibit_sha256")
        ):
            raise ValueError("Release proof hash changed")
        row = case_row(result, case)
        pair = [
            c
            for c in cells
            if c["validation_event_id"] == case["validation_event_id"]
            and c["cik"] == case["cik"]
            and c["release_accession"] == case["release_accession_number"]
            and c["metric"] in ("revenue", "gross_profit")
        ]
        for cell in pair:
            if row[cell["metric"]] != cell["value"] or any(
                pd.Timestamp(row[k]).date().isoformat() != cell[k]
                for k in ("period_start", "period_end")
            ):
                raise ValueError("Release values or quarter differ from frozen history")
        if not np.isclose(row.gross_margin, proof["release_gross_margin"], rtol=0, atol=1e-12):
            raise ValueError("Gross margin does not reproduce the reviewed component pair")
        old, new = utc_timestamp(case["accepted_at"]), utc_timestamp(case["earliest_disclosure_at"])
        if (
            utc_timestamp(row.actual_available_at) != old
            or utc_timestamp(row.filing_accepted_at) != old
            or utc_timestamp(case["release_accepted_at"]) != new
            or not pd.Timestamp(case["period_end"], tz="UTC") < new < old
        ):
            raise ValueError("Release clock ordering or baseline alias changed")
        if pd.notna(row.ytd_filing_accession):
            raise ValueError("Direct-quarter evidence cannot backdate a YTD-derived actual")
        result.loc[result.event_id.eq(row.event_id), "actual_available_at"] = new
    assert_originals_preserved(history, result)
    return result


def company_dependencies(
    history: pd.DataFrame, target: pd.Series, config: PilotFilingFeatureConfig
) -> pd.DataFrame:
    """Expose the exact lag/trailing rows used by the frozen six-feature specification."""
    prior = history.loc[
        history.company_id.eq(target.company_id)
        & history.period_end.lt(target.period_end)
        & history.actual_available_at.lt(target.actual_available_at)
        & history.analysis_ready_level.astype(bool)
    ].copy()
    gap = (target.period_end - prior.period_end).dt.days
    quarter = prior.loc[gap.between(45, 150)].sort_values(
        ["period_end", "actual_available_at", "event_id"], kind="mergesort"
    )
    roles: dict[str, list[str]] = {}

    def add(event_id: Any, role: str) -> None:
        if pd.notna(event_id):
            roles.setdefault(str(event_id), []).append(role)

    add(target.gm_yoy_lag_event_id, "target_year_ago_margin")
    if not quarter.empty:
        lag = quarter.iloc[-1]
        add(lag.event_id, "previous_quarter_margin_delta_revenue")
        add(lag.gm_yoy_lag_event_id, "previous_quarter_year_ago_margin_revenue")
    trailing = prior.sort_values(["actual_available_at", "event_id"], kind="mergesort").tail(
        config.trailing_history_events
    )
    if len(trailing) >= config.minimum_trailing_events:
        for event_id in trailing.event_id:
            add(event_id, "trailing_margin_mean_and_volatility")
    columns = [
        "event_id",
        "company_id",
        "filing_accession",
        "period_end",
        "actual_available_at",
        "revenue",
        "gross_profit",
        "gross_margin",
    ]
    result = history.loc[history.event_id.isin(roles), columns].copy()
    if (
        len(result) != len(roles)
        or not result.actual_available_at.lt(target.actual_available_at).all()
    ):
        raise ValueError("Missing or non-prior company feature dependency")
    result["roles"] = result.event_id.map(roles)
    return result.sort_values(["actual_available_at", "event_id"]).reset_index(drop=True)


def verify_target_reconstruction(history: pd.DataFrame, case: dict[str, Any]) -> pd.DataFrame:
    """Reconstruct OLS and prior-OOS sigma independently from the recorded membership."""
    row = case_row(history, case)
    if not audit_forecast_metadata(
        {str(k): v for k, v in row.items()}, row.actual_available_at.isoformat()
    )["existing_forecast_clocks_before_proxy"]:
        raise ValueError("Target dependency is not strictly before availability")
    deps = target_dependencies(history, str(row.event_id))
    if not deps.actual_available_at.lt(row.actual_available_at).all():
        raise ValueError("Non-prior training or error membership")
    training = history.set_index("event_id").loc[deps.loc[deps.in_training, "event_id"]]
    x = training[list(FORECAST_FEATURES)].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]),
        training.gross_margin.to_numpy(dtype=float),
        rcond=None,
    )
    expected = float(np.r_[1.0, row[list(FORECAST_FEATURES)].to_numpy(dtype=float)] @ beta)
    if not np.isclose(row.gm_forecast, expected, rtol=0, atol=1e-10):
        raise ValueError("Target forecast does not reproduce from prior training rows")
    if any(
        not np.isclose(a, b, rtol=0, atol=1e-10)
        for a, b in (
            (row.gm_level, row.gross_margin),
            (row.gm_delta_yoy, row.gross_margin - row.gm_yoy_lag),
            (row.gm_innovation_raw, row.gross_margin - row.gm_forecast),
            (row.gm_innovation_std, row.gm_innovation_raw / row.forecast_sigma),
        )
    ):
        raise ValueError("Target representation identity does not reproduce")
    return deps


def compare_case_signals(
    histories: dict[str, pd.DataFrame], cases: list[dict[str, Any]], *, tolerance: float = 1e-10
) -> pd.DataFrame:
    """Compare identical cases across legacy, reconciled, individual and shared replay arms."""
    if tolerance < 0:
        raise ValueError("Comparison tolerance must be nonnegative")
    rows = []
    for case in cases:
        ticker = case["assigned_ticker"]
        arms = {name: case_row(history, case) for name, history in histories.items()}
        for field in (*SIGNALS, "gm_forecast", "forecast_sigma"):
            values = {name: float(row[field]) for name, row in arms.items()}
            if not np.isfinite(list(values.values())).all():
                raise ValueError("Incomplete common-case signal comparison; do not replace case")
            old, new = values["legacy"], values["joint"]

            def direction(value: float) -> int:
                return 0 if abs(value) <= tolerance else 1 if value > 0 else -1

            rows.append(
                {
                    "validation_event_id": case["validation_event_id"],
                    "ticker": ticker,
                    "representation": field,
                    **values,
                    "clock_reconciliation_change": values["reconciled"] - old,
                    "own_release_shift_change": values[f"only_{ticker}"] - values["reconciled"],
                    "joint_minus_own_only": new - values[f"only_{ticker}"],
                    "total_change": new - old,
                    "changed": abs(new - old) > tolerance,
                    "direction_flipped": direction(old) * direction(new) == -1,
                    "unit": "prior_OOS_sigma_units"
                    if field == "gm_innovation_std"
                    else "margin_fraction",
                }
            )
    return pd.DataFrame(rows)
