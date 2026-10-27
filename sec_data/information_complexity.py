"""Bounded information x complexity experiment; financial and return forecasts stay separate."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-untyped]

from sec_data.announcement_expansion import attach_prior_industry
from sec_data.disclosure_audit import utc_timestamp

VERSION = "information-complexity-v1"
FINANCIAL = ("gm_level", "gm_delta_yoy", "gm_innovation_raw", "gm_innovation_std")
STATE = (
    "pre_relative_momentum_20d",
    "pre_price_volatility_20d",
    "pre_volume_ratio_5_20",
    "lagged_industry_gm",
)
ARMS = {
    "A_financial_ridge": (FINANCIAL, "ridge"),
    "B_state_ridge": ((*FINANCIAL, *STATE), "ridge"),
    "C_financial_tree": (FINANCIAL, "tree"),
    "D_state_tree": ((*FINANCIAL, *STATE), "tree"),
}
FINANCIAL_CLOCKS = (
    "forecast_feature_cutoff",
    "forecast_training_end",
    "forecast_generated_at",
    "forecast_error_history_end",
)


def select_review_batch(
    cohort: list[dict[str, Any]], excluded_issuers: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Freeze up to twenty new issuers on clocks/identities, never returns or listing survival.

    An independent batch does not replace failed prior selections. Missing release
    accessions remain missing and require source resolution before admission.
    """
    if len({r["validation_event_id"] for r in cohort}) != len(cohort):
        raise ValueError("Duplicate cohort identity")
    columns = (
        "validation_event_id",
        "cik",
        "assigned_ticker",
        "assigned_exchange_mic",
        "target_accession_number",
        "release_accession_number",
        "earliest_disclosure_at",
    )
    metadata = [
        {k: row.get(k) for k in columns}
        for row in cohort
        if row["cik"] not in excluded_issuers and row.get("earliest_disclosure_at")
    ]
    metadata.sort(
        key=lambda r: (
            utc_timestamp(r["earliest_disclosure_at"]),
            r["cik"],
            r["validation_event_id"],
        )
    )
    queue, seen = [], set()
    for row in metadata:
        if row["cik"] not in seen:
            seen.add(row["cik"])
            queue.append(row)
        if len(queue) == 20:
            break
    return queue, len({r["cik"] for r in metadata})


@dataclass(frozen=True)
class ExperimentConfig:
    """Engineering floors, not evidence of statistical power or profitability.

    Parameters are fixed, with no tuning API. A different specification needs a
    new version, rather than lowering these gates to accommodate fourteen cases.
    """

    minimum_common_events: int = 200
    minimum_common_issuers: int = 20
    minimum_common_coverage: float = 0.80
    minimum_training_events: int = 100
    minimum_training_issuers: int = 10
    minimum_evaluation_events: int = 50
    minimum_evaluation_issuers: int = 20
    minimum_evaluation_span_days: int = 180
    label_embargo_days: int = 1
    ridge_alpha: float = 10.0
    tree_max_depth: int = 2
    tree_min_samples_leaf: int = 20
    seed: int = 20261002

    def validate(self) -> None:
        """Reject silent retuning of a frozen four-arm experiment."""
        if asdict(self) != asdict(ExperimentConfig()):
            raise ValueError("Changed parameters require a new experiment version")


def audit_prior_history(
    events: list[dict[str, Any]],
    issuer_dates: pd.DataFrame,
    benchmark_dates: pd.DataFrame,
    sessions: pd.DataFrame,
    action_targets: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Audit exact 61-session pre-announcement dates without opening numeric prices.

    Never bridge a missing day with an older observation, or use a filing-time
    feature whose last observation follows the earlier earnings announcement.
    Query bounds alone do not certify identity continuity or vendor delivery.
    """
    if len({e["validation_event_id"] for e in events}) != len(events):
        raise ValueError("Duplicate event identity")
    if issuer_dates.duplicated(["price_target_id", "session_date"]).any():
        raise ValueError("Duplicate issuer price date")
    rows = []
    for event in events:
        cutoff = utc_timestamp(event["announcement_at"])
        local = cutoff.tz_convert("America/New_York").date().isoformat()
        grid = sessions.loc[sessions.exchange_mic.eq(event["exchange_mic"])].copy()
        if grid.session_date.duplicated().any():
            raise ValueError("Ambiguous official session grid")
        grid = grid.loc[grid.session_date.astype(str).lt(local)].sort_values("session_date")
        grid = grid.tail(61)
        dates = grid.session_date.astype(str).tolist()
        closes = [utc_timestamp(v) for v in grid.regular_close_at]
        if any(t >= cutoff for t in closes):
            raise ValueError("A prior market session does not close before announcement")
        stock = set(
            issuer_dates.loc[
                issuer_dates.price_target_id.eq(event["price_target_id"]), "session_date"
            ].astype(str)
        )
        market = set(benchmark_dates.session_date.astype(str))
        missing_stock, missing_market = sorted(set(dates) - stock), sorted(set(dates) - market)
        target = action_targets.loc[action_targets.action_target_id.eq(event["action_target_id"])]
        query_covered = bool(
            len(target) == 1
            and len(dates) == 61
            and str(target.iloc[0].price_window_start) <= dates[0]
            and dates[-1] <= str(target.iloc[0].price_window_end)
            and target.iloc[0].all_security_action_queries_complete == True  # noqa: E712
        )
        numeric_ready = len(dates) == 61 and not missing_stock and not missing_market
        rows.append(
            {
                "validation_event_id": event["validation_event_id"],
                "ticker": event["ticker"],
                "price_target_id": event["price_target_id"],
                "action_target_id": event["action_target_id"],
                "security_id": event["security_id"],
                "exchange_mic": event["exchange_mic"],
                "announcement_at": cutoff.isoformat(),
                "required_sessions": 61,
                "official_prior_sessions": len(dates),
                "expected_session_dates": dates,
                "missing_issuer_dates": missing_stock,
                "missing_benchmark_dates": missing_market,
                "numeric_history_ready": numeric_ready,
                "action_query_interval_covered": query_covered,
                "history_identity_continuity_certified": False,
                "vendor_delivery_clock_certified": False,
                "conditional_feature_calculation_allowed": numeric_ready and query_covered,
                "requested_start": (
                    dates[0]
                    if len(dates) == 61
                    else (pd.Timestamp(local) - pd.Timedelta(days=120)).date().isoformat()
                ),
                "requested_end": (pd.Timestamp(local) - pd.Timedelta(days=1)).date().isoformat(),
                "market_feature_available_at": closes[-1].isoformat() if closes else None,
                "calendar_extension_required": len(dates) != 61,
            }
        )
    return rows


def lagged_industry_features(
    events: list[dict[str, Any]], history: pd.DataFrame, industries: list[dict[str, Any]]
) -> pd.DataFrame:
    """Median of at least two OTHER issuers' latest known, strictly prior-quarter margins.

    Both original-header SIC and quarter availability are joined as of the target
    announcement. Current-quarter, same-time, stale (>365 days), quarantined and
    conflicting quarter values are excluded. Keep every target, even if missing.
    """
    if history.event_id.duplicated().any():
        raise ValueError("Duplicate financial history event")
    h = history.copy()
    h["actual_available_at"] = h.actual_available_at.map(utc_timestamp)
    h["period_end"] = pd.to_datetime(h.period_end, utc=True)
    rows = []
    for event in events:
        cutoff = utc_timestamp(event["announcement_at"])
        start = pd.Timestamp(event["period_start"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        issuers = sorted(set(h.company_id.astype(str)) | {event["cik"]})
        bindings = attach_prior_industry(
            [
                {
                    "validation_event_id": cik,
                    "cik": cik,
                    "assigned_ticker": cik,
                    "earliest_disclosure_at": cutoff,
                }
                for cik in issuers
            ],
            industries,
        ).set_index("cik")
        target = bindings.loc[event["cik"]]
        peer_codes = bindings.loc[
            bindings.industry_ready & bindings.sic_2digit.eq(target.sic_2digit)
        ].index
        eligible = h.loc[
            h.company_id.isin(peer_codes)
            & h.company_id.ne(event["cik"])
            & h.analysis_ready_level.eq(True)
            & h.scope_quarantined.eq(False)
            & h.actual_available_at.lt(cutoff)
            & h.period_end.lt(start)
            & h.period_end.ge(start - pd.Timedelta(days=365))
            & np.isfinite(pd.to_numeric(h.gm_level, errors="coerce"))
        ].sort_values(["company_id", "period_end", "actual_available_at", "event_id"])
        ambiguous = eligible.groupby(["company_id", "period_end"]).gm_level.nunique().gt(1)
        conflicting = set(ambiguous.loc[ambiguous].index.get_level_values(0))
        peers = eligible.loc[~eligible.company_id.isin(conflicting)].drop_duplicates(
            "company_id", keep="last"
        )
        ready = bool(target.industry_ready and len(peers) >= 2)
        clocks = [
            *peers.actual_available_at.tolist(),
            *bindings.loc[peers.company_id, "industry_available_at"].map(utc_timestamp).tolist(),
        ]
        if target.industry_ready:
            clocks.append(utc_timestamp(target.industry_available_at))
        rows.append(
            {
                "validation_event_id": event["validation_event_id"],
                "lagged_industry_gm": float(peers.gm_level.median()) if ready else np.nan,
                "industry_feature_available_at": max(clocks) if ready else pd.NaT,
                "industry_peer_events": peers.event_id.tolist(),
                "industry_peer_issuers": peers.company_id.tolist(),
                "industry_peer_count": len(peers),
                "industry_state_ready": ready,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Walk forward on mature return labels; never fit a financial expectation model.

    All four arms AND all four original representations share one complete-case
    sample. A same-time decision batch cannot use any member's outcome. Labels
    must be available for one full day before training; overlapping unfinished
    holding periods therefore never enter training. This is exploratory replay,
    not a new holdout, live forecast, strategy P&L or Alpha certification.
    """
    cfg = ExperimentConfig()
    cfg.validate()
    clocks = (
        "announcement_at",
        "decision_at",
        "entry_at",
        "exit_at",
        "label_available_at",
        "market_feature_available_at",
        "industry_feature_available_at",
        *FINANCIAL_CLOCKS,
    )
    required = {
        "validation_event_id",
        "cik",
        "source_admitted",
        "information_set",
        "entry_session_offset",
        "holding_sessions_inclusive",
        "entry_price_type",
        "exit_price_type",
        "market_excess_return",
        *clocks,
        *FINANCIAL,
        *STATE,
    }
    if required - set(panel) or panel.validation_event_id.duplicated().any():
        raise ValueError("Incomplete schema or duplicate company-event-information-set identity")
    data = panel.copy()
    if data.cik.isna().any() or data.validation_event_id.isna().any():
        raise ValueError("Missing event/issuer identity")
    expected_contract: dict[str, str | int] = {
        "information_set": "original_release",
        "entry_session_offset": 2,
        "holding_sessions_inclusive": 20,
        "entry_price_type": "unadjusted_open",
        "exit_price_type": "unadjusted_close",
    }
    for name, expected in expected_contract.items():
        if not data[name].eq(expected).all():
            raise ValueError(f"Changed information or execution contract: {name}")
    if any(type(v) is not bool for v in data.source_admitted.tolist()):
        raise ValueError("Admission must be explicit booleans, not strings or truthy flags")
    for name in clocks:
        data[name] = pd.to_datetime(
            pd.Series(
                [utc_timestamp(v) if pd.notna(v) else pd.NaT for v in data[name]], index=data.index
            ),
            utc=True,
        )
    for left, right in (
        ("announcement_at", "decision_at"),
        ("decision_at", "entry_at"),
        ("entry_at", "exit_at"),
        ("market_feature_available_at", "announcement_at"),
        ("industry_feature_available_at", "announcement_at"),
        *((c, "announcement_at") for c in FINANCIAL_CLOCKS),
    ):
        present = data[left].notna() & data[right].notna()
        if not data.loc[present, left].lt(data.loc[present, right]).all():
            raise ValueError(f"Leaking or ambiguous clock: {left} must precede {right}")
    if (data.label_available_at < data.exit_at).any():
        raise ValueError("Return labels cannot be available before the exit price")
    numeric = [*FINANCIAL, *STATE, "market_excess_return"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(data[numeric].to_numpy(float)).all(axis=1) & data[
        list(clocks)
    ].notna().all(axis=1)
    admitted = complete & data.source_admitted
    common = data.loc[admitted].sort_values(["decision_at", "validation_event_id"])
    report: dict[str, Any] = {
        "version": VERSION,
        "parameters": asdict(cfg),
        "input_events": len(data),
        "numeric_complete_events": int(complete.sum()),
        "source_admitted_events": int(data.source_admitted.sum()),
        "common_events": len(common),
        "common_issuers": int(common.cik.nunique()),
        "missing_feature_counts": {k: int(data[k].isna().sum()) for k in (*FINANCIAL, *STATE)},
        "missing_clock_counts": {k: int(data[k].isna().sum()) for k in clocks},
        "financial_expectation_refitted": False,
        "hyperparameters_searched": False,
        "inferential_tests_computed": False,
        "strategy_pnl_computed": False,
        "trading_admission": False,
        "alpha_established": False,
        "previously_unseen_outcomes": False,
        "confirmatory_preregistration": False,
        "minimum_counts_are_power_guarantees": False,
        "models_fitted": 0,
        "target": "20_inclusive_sessions_issuer_total_return_minus_SPY_total_return",
        "verdict": "blocked_insufficient_admitted_common_sample",
        "evaluation_gate": False,
        "limitations": [
            "retrospective_not_live",
            "no_costs_or_sector_neutralization",
            "dependent_events_no_inferential_test",
            "not_a_new_untouched_holdout",
        ],
    }
    gate = (
        len(common) >= cfg.minimum_common_events
        and common.cik.nunique() >= cfg.minimum_common_issuers
        and len(common) / max(len(data), 1) >= cfg.minimum_common_coverage
    )
    report["training_gate"] = bool(gate)
    report["training_blockers"] = [
        name
        for name, passed in {
            "fewer_than_200_admitted_common_events": len(common) >= cfg.minimum_common_events,
            "fewer_than_20_admitted_common_issuers": common.cik.nunique()
            >= cfg.minimum_common_issuers,
            "less_than_80_percent_common_coverage": len(common) / max(len(data), 1)
            >= cfg.minimum_common_coverage,
        }.items()
        if not passed
    ]
    if not gate:
        return pd.DataFrame(), pd.DataFrame(), report
    rows = []
    for decision, targets in common.groupby("decision_at", sort=True):
        training = common.loc[
            common.label_available_at.lt(decision - pd.Timedelta(days=cfg.label_embargo_days))
        ]
        if (
            len(training) < cfg.minimum_training_events
            or training.cik.nunique() < cfg.minimum_training_issuers
        ):
            continue
        if set(training.validation_event_id) & set(targets.validation_event_id):
            raise ValueError("Target label entered training")
        output = targets[
            ["validation_event_id", "cik", "decision_at", "market_excess_return", *FINANCIAL]
        ].copy()
        output["training_events"] = len(training)
        output["training_issuers"] = training.cik.nunique()
        output["training_label_available_end"] = training.label_available_at.max()
        output["training_event_ids_sha256"] = hashlib.sha256(
            "\n".join(sorted(training.validation_event_id)).encode()
        ).hexdigest()
        output["training_mean_control"] = float(training.market_excess_return.mean())
        for name, (features, algorithm) in ARMS.items():
            model = (
                make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
                if algorithm == "ridge"
                else DecisionTreeRegressor(
                    max_depth=cfg.tree_max_depth,
                    min_samples_leaf=cfg.tree_min_samples_leaf,
                    random_state=cfg.seed,
                )
            )
            model.fit(training[list(features)], training.market_excess_return)
            output[name] = model.predict(targets[list(features)])
            report["models_fitted"] += 1
        rows.append(output)
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    span = (
        (predictions.decision_at.max() - predictions.decision_at.min()).days
        if len(predictions)
        else 0
    )
    evaluation = (
        (
            len(predictions) >= cfg.minimum_evaluation_events
            and predictions.cik.nunique() >= cfg.minimum_evaluation_issuers
            and span >= cfg.minimum_evaluation_span_days
        )
        if len(predictions)
        else False
    )
    report.update(
        evaluation_events=len(predictions),
        evaluation_span_days=span,
        evaluation_gate=bool(evaluation),
        verdict="exploratory_replay_not_alpha"
        if evaluation
        else "blocked_insufficient_forward_evaluation",
    )
    if not evaluation:
        return predictions, pd.DataFrame(), report
    y = predictions.market_excess_return.to_numpy(float)
    control_sse = float(np.sum((y - predictions.training_mean_control.to_numpy(float)) ** 2))
    metrics = []
    for name in (*ARMS, "training_mean_control", *FINANCIAL):
        values = predictions[name]
        rank_ic = (
            float(values.rank().corr(predictions.market_excess_return.rank()))
            if values.nunique() > 1 and predictions.market_excess_return.nunique() > 1
            else None
        )
        predictive = name not in FINANCIAL
        metrics.append(
            {
                "model_or_representation": name,
                "events": len(predictions),
                "rank_ic": rank_ic,
                "mae": float(np.mean(np.abs(y - values))) if predictive else None,
                "oos_r2_vs_training_mean": (1.0 - float(np.sum((y - values) ** 2)) / control_sse)
                if predictive and control_sse > 0
                else None,
            }
        )
    comparison = pd.DataFrame(metrics).set_index("model_or_representation")
    report["paired_contrasts"] = [
        {
            "contrast": f"{challenger}_minus_{baseline}",
            "mae_reduction": float(
                cast(float, comparison.loc[baseline, "mae"])
                - cast(float, comparison.loc[challenger, "mae"])
            ),
            "interpretation": interpretation,
            "inferential_test": False,
        }
        for challenger, baseline, interpretation in (
            ("B_state_ridge", "A_financial_ridge", "information_increment_with_ridge_fixed"),
            (
                "C_financial_tree",
                "A_financial_ridge",
                "complexity_increment_with_financial_inputs_fixed",
            ),
            ("D_state_tree", "B_state_ridge", "complexity_increment_with_state_inputs_fixed"),
            ("D_state_tree", "C_financial_tree", "information_increment_with_tree_fixed"),
        )
    ]
    return predictions, pd.DataFrame(metrics), report
