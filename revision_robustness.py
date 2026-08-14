"""Point-in-time revision-risk buffers for filing-event decisions.

The module answers a narrow decision question: given the originally reported
gross-margin Innovation, how large was a historically defensible revision buffer
at that instant?  Revision outcomes enter the empirical distribution only after
their later comparative filing or amendment was available.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

CONFIDENCE_GRID: Final[tuple[float, ...]] = (0.80, 0.90, 0.95, 0.975, 0.99)
RESOLVED_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"later_comparative_filing", "amendment"})
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "company_id",
    "company_name",
    "ticker",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    "actual_available_at",
    "gross_margin",
    "gross_margin_latest",
    "latest_available_at",
    "revision_source_type",
    "was_revised",
    "revision_size",
    "filing_accession",
    "latest_filing_accession",
    "M2_innovation_raw",
)


class RevisionRobustnessError(ValueError):
    """Raised when a revision-robustness panel violates its PIT contract."""


@dataclass(frozen=True)
class RevisionRobustnessConfig:
    """Frozen first-pass rules for empirical revision-risk buffers."""

    selected_model_id: str = "M2"
    primary_confidence: float = 0.95
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID
    min_resolved_history: int = 100
    min_revised_history: int = 10
    revision_tolerance: float = 1e-12
    history_scope: str = "pooled_gross_margin"
    quantile_method: Literal["higher"] = "higher"

    def __post_init__(self) -> None:
        if self.selected_model_id != "M2":
            raise ValueError("Revision robustness v1 is frozen to selected model M2")
        if not 0.5 < self.primary_confidence < 1.0:
            raise ValueError("primary_confidence must be between 0.5 and 1")
        if self.primary_confidence not in self.confidence_grid:
            raise ValueError("primary_confidence must be present in confidence_grid")
        if tuple(sorted(set(self.confidence_grid))) != self.confidence_grid:
            raise ValueError("confidence_grid must be unique and increasing")
        if any(not 0.5 < value < 1.0 for value in self.confidence_grid):
            raise ValueError("Every confidence level must be between 0.5 and 1")
        if self.min_resolved_history <= 0 or self.min_revised_history <= 0:
            raise ValueError("History thresholds must be positive")
        if self.min_revised_history >= self.min_resolved_history:
            raise ValueError("min_revised_history must be below min_resolved_history")
        if self.revision_tolerance < 0:
            raise ValueError("revision_tolerance must be non-negative")
        if self.quantile_method != "higher":
            raise ValueError("v1 freezes the conservative 'higher' quantile method")


@dataclass(frozen=True)
class RevisionRobustnessResult:
    """Event decisions and diagnostics from the walk-forward revision experiment."""

    panel: pd.DataFrame
    primary_decisions: pd.DataFrame
    confidence_calibration: pd.DataFrame
    diagnostics_by_year: pd.DataFrame
    source_audit: pd.DataFrame
    tail_events: pd.DataFrame
    report: dict[str, Any]


def build_revision_robustness(
    events: pd.DataFrame,
    config: RevisionRobustnessConfig | None = None,
    audit_reviews: pd.DataFrame | None = None,
) -> RevisionRobustnessResult:
    """Estimate revision buffers using only outcomes resolved before each event.

    ``latest_available_at`` is treated as an outcome-resolution clock only for a
    later comparative filing or an amendment.  An event whose latest fact still
    comes from its original accession is right-censored: hindsight says that its
    value has not changed in the current snapshot, but that was not knowable at
    the original announcement.
    """
    cfg = config or RevisionRobustnessConfig()
    panel = _prepare_events(events, cfg, audit_reviews)
    resolved_history = panel.loc[panel["revision_outcome_resolved"]].sort_values(
        ["revision_outcome_available_at", "event_id"]
    )
    history_times = resolved_history["revision_outcome_available_at"].reset_index(drop=True)
    history_revisions = resolved_history["revision_size"].to_numpy(dtype=float)
    history_changed = resolved_history["was_revised"].to_numpy(dtype=bool)

    decision_rows: list[dict[str, Any]] = []
    for event in panel.itertuples(index=False):
        history_count = int(history_times.searchsorted(event.actual_available_at, side="left"))
        prior_revisions = history_revisions[:history_count]
        prior_changed = history_changed[:history_count]
        revised_count = int(prior_changed.sum())
        eligible = (
            history_count >= cfg.min_resolved_history and revised_count >= cfg.min_revised_history
        )
        initial_innovation = float(cast(float, event.M2_innovation_raw))
        row: dict[str, Any] = {
            "revision_history_scope": cfg.history_scope,
            "revision_history_observations": history_count,
            "revision_history_changed_observations": revised_count,
            "revision_history_rate": revised_count / history_count if history_count else np.nan,
            "revision_history_end": (
                resolved_history.iloc[history_count - 1]["revision_outcome_available_at"]
                if history_count
                else pd.NaT
            ),
            "revision_buffer_eligible": eligible,
        }
        for confidence in cfg.confidence_grid:
            suffix = _confidence_suffix(confidence)
            row[f"revision_at_risk_{suffix}"] = (
                float(
                    np.quantile(
                        np.abs(prior_revisions),
                        confidence,
                        method=cfg.quantile_method,
                    )
                )
                if eligible
                else np.nan
            )
        if eligible:
            primary_buffer = float(row[_primary_buffer_column(cfg)])
            sign = float(np.sign(initial_innovation))
            survival = float(np.mean(np.sign(initial_innovation + prior_revisions) == sign))
            robust = abs(initial_innovation) > primary_buffer
            row.update(
                {
                    "sign_survival_probability": survival,
                    "robust_innovation": sign * max(abs(initial_innovation) - primary_buffer, 0.0),
                    "robust_direction": robust,
                    "decision_status": (
                        "robust_positive"
                        if robust and initial_innovation > 0
                        else "robust_negative"
                        if robust and initial_innovation < 0
                        else "revision_sensitive"
                    ),
                    "signal_to_buffer": (
                        abs(initial_innovation) / primary_buffer if primary_buffer > 0 else np.inf
                    ),
                }
            )
        else:
            row.update(
                {
                    "sign_survival_probability": np.nan,
                    "robust_innovation": np.nan,
                    "robust_direction": False,
                    "decision_status": "insufficient_history",
                    "signal_to_buffer": np.nan,
                }
            )
        decision_rows.append(row)

    decisions = pd.DataFrame(decision_rows, index=panel.index)
    panel = pd.concat([panel, decisions], axis="columns")
    panel["latest_innovation_hindsight"] = panel["M2_innovation_raw"] + panel["revision_size"]
    panel["realized_revision_within_primary_buffer"] = pd.Series(
        pd.NA, index=panel.index, dtype="boolean"
    )
    panel["realized_direction_flip"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    evaluable = panel["revision_buffer_eligible"] & panel["revision_outcome_resolved"]
    primary_buffer_column = _primary_buffer_column(cfg)
    panel.loc[evaluable, "realized_revision_within_primary_buffer"] = (
        panel.loc[evaluable, "revision_size"].abs() <= panel.loc[evaluable, primary_buffer_column]
    ).to_numpy()
    panel.loc[evaluable, "realized_direction_flip"] = (
        np.sign(panel.loc[evaluable, "M2_innovation_raw"])
        != np.sign(panel.loc[evaluable, "latest_innovation_hindsight"])
    ).to_numpy()
    panel["revision_robustness_version"] = _research_version(panel, cfg)

    validate_revision_robustness(panel, cfg)
    primary_decisions = panel.loc[panel["revision_buffer_eligible"]].copy()
    confidence_calibration = _confidence_calibration(panel, cfg)
    diagnostics_by_year = _diagnostics_by_year(panel, cfg)
    source_audit = _source_audit(panel)
    tail_events = _tail_events(panel, cfg)
    report = _research_report(panel, confidence_calibration, cfg)
    return RevisionRobustnessResult(
        panel=panel,
        primary_decisions=primary_decisions,
        confidence_calibration=confidence_calibration,
        diagnostics_by_year=diagnostics_by_year,
        source_audit=source_audit,
        tail_events=tail_events,
        report=report,
    )


def validate_revision_robustness(
    panel: pd.DataFrame,
    config: RevisionRobustnessConfig | None = None,
) -> None:
    """Validate revision identities, clocks, monotonic buffers, and decisions."""
    cfg = config or RevisionRobustnessConfig()
    _require_columns(panel, REQUIRED_COLUMNS)
    _require_columns(
        panel,
        (
            "revision_outcome_resolved",
            "revision_outcome_available_at",
            "revision_buffer_eligible",
            "revision_history_end",
            "latest_innovation_hindsight",
            "robust_innovation",
        ),
    )
    if panel.empty or panel["event_id"].duplicated().any():
        raise RevisionRobustnessError("Revision panel must be non-empty and event-unique")
    expected_revision = panel["gross_margin_latest"] - panel["gross_margin"]
    if not np.allclose(
        expected_revision,
        panel["revision_size"],
        rtol=1e-10,
        atol=cfg.revision_tolerance,
    ):
        raise RevisionRobustnessError("revision_size does not tie to latest minus initial GM")
    expected_changed = panel["revision_size"].abs() > cfg.revision_tolerance
    if not expected_changed.equals(panel["was_revised"]):
        raise RevisionRobustnessError("was_revised does not tie to revision_size")
    resolved = panel["revision_outcome_resolved"]
    if panel.loc[resolved, "revision_outcome_available_at"].isna().any():
        raise RevisionRobustnessError("Resolved outcomes require an availability timestamp")
    if (
        panel.loc[resolved, "revision_outcome_available_at"]
        < panel.loc[resolved, "actual_available_at"]
    ).any():
        raise RevisionRobustnessError("A revision outcome predates the original actual")
    eligible = panel["revision_buffer_eligible"]
    history_end = panel.loc[eligible, "revision_history_end"]
    if (
        history_end.isna().any()
        or (history_end >= panel.loc[eligible, "actual_available_at"]).any()
    ):
        raise RevisionRobustnessError("Revision history contains same-time or future outcomes")
    buffer_columns = [_buffer_column(value) for value in cfg.confidence_grid]
    buffers = panel.loc[eligible, buffer_columns].to_numpy(dtype=float)
    if np.isnan(buffers).any() or (np.diff(buffers, axis=1) < -cfg.revision_tolerance).any():
        raise RevisionRobustnessError("Revision buffers must be present and non-decreasing")
    expected_latest = panel["M2_innovation_raw"] + panel["revision_size"]
    if not np.allclose(
        expected_latest,
        panel["latest_innovation_hindsight"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RevisionRobustnessError("Latest Innovation identity does not tie")
    primary_buffer = panel.loc[eligible, _primary_buffer_column(cfg)]
    expected_robust = np.sign(panel.loc[eligible, "M2_innovation_raw"]) * (
        panel.loc[eligible, "M2_innovation_raw"].abs() - primary_buffer
    ).clip(lower=0.0)
    if not np.allclose(
        expected_robust,
        panel.loc[eligible, "robust_innovation"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RevisionRobustnessError("Robust Innovation does not tie to the primary buffer")


def _prepare_events(
    events: pd.DataFrame,
    config: RevisionRobustnessConfig,
    audit_reviews: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_columns(events, REQUIRED_COLUMNS)
    result = events.copy()
    if result.empty or result["event_id"].duplicated().any():
        raise RevisionRobustnessError("Input must be non-empty and event-unique")
    for column in ("actual_available_at", "latest_available_at"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    for column in (
        "gross_margin",
        "gross_margin_latest",
        "revision_size",
        f"{config.selected_model_id}_innovation_raw",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (
        result[
            [
                "actual_available_at",
                "period_end",
                "gross_margin",
                "gross_margin_latest",
                "revision_size",
                f"{config.selected_model_id}_innovation_raw",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RevisionRobustnessError("Input contains missing required dates or numeric values")
    result["was_revised"] = _as_bool(result["was_revised"])
    result["company_id"] = result["company_id"].astype("string").str.zfill(10)
    result["revision_source_type"] = result["revision_source_type"].astype("string")
    result = _apply_revision_audit(result, audit_reviews)
    result["revision_outcome_resolved"] = (
        result["revision_source_type"].isin(RESOLVED_SOURCE_TYPES)
        & result["latest_available_at"].notna()
    )
    result["revision_outcome_status"] = np.where(
        result["revision_outcome_resolved"],
        "resolved_by_later_disclosure",
        "right_censored_at_snapshot",
    )
    result["revision_outcome_available_at"] = result["latest_available_at"].where(
        result["revision_outcome_resolved"]
    )
    result["resolution_days"] = (
        result["revision_outcome_available_at"] - result["actual_available_at"]
    ).dt.total_seconds() / 86_400.0
    return result.sort_values(["actual_available_at", "company_id", "event_id"]).reset_index(
        drop=True
    )


def _apply_revision_audit(
    events: pd.DataFrame,
    audit_reviews: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply source-traceable corrections for manually verified semantic collisions."""
    result = events.copy()
    result["gross_margin_latest_unreviewed"] = result["gross_margin_latest"]
    result["revision_size_unreviewed"] = result["revision_size"]
    result["was_revised_unreviewed"] = result["was_revised"]
    result["tail_audit_reviewed"] = False
    result["tail_audit_classification"] = "not_tail_audited"
    result["tail_audit_family"] = "not_tail_audited"
    result["tail_audit_action"] = "retain_unreviewed"
    result["economic_error_correction"] = False
    result["event_time_structural_context"] = False
    result["revision_audit_version"] = "none"
    if audit_reviews is None:
        return result

    required = (
        "event_id",
        "audit_classification",
        "audit_family",
        "audit_action",
        "economic_error_correction",
        "event_time_structural_context",
        "review_status",
    )
    _require_columns(audit_reviews, required)
    reviews = audit_reviews.copy()
    if reviews.empty or reviews["event_id"].duplicated().any():
        raise RevisionRobustnessError("Audit reviews must be non-empty and event-unique")
    if not reviews["event_id"].isin(result["event_id"]).all():
        raise RevisionRobustnessError("Audit reviews contain event IDs outside the input sample")
    if not reviews["review_status"].astype("string").eq("reviewed").all():
        raise RevisionRobustnessError("Every applied audit review must have reviewed status")
    allowed_actions = {"set_latest_equal_original", "retain_published_history_redraw"}
    if not reviews["audit_action"].astype("string").isin(allowed_actions).all():
        raise RevisionRobustnessError("Audit reviews contain an unsupported action")
    reviews["economic_error_correction"] = _as_bool(reviews["economic_error_correction"])
    reviews["event_time_structural_context"] = _as_bool(reviews["event_time_structural_context"])
    audit_hash = hashlib.sha256(
        reviews.sort_values("event_id").astype("string").to_csv(index=False).encode("utf-8")
    ).hexdigest()[:16]
    rename = {
        "audit_classification": "tail_audit_classification_review",
        "audit_family": "tail_audit_family_review",
        "audit_action": "tail_audit_action_review",
        "economic_error_correction": "economic_error_correction_review",
        "event_time_structural_context": "event_time_structural_context_review",
    }
    merged = result.merge(
        reviews.loc[:, list(required)].rename(columns=rename),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    reviewed = merged["review_status"].eq("reviewed")
    merged.loc[reviewed, "tail_audit_reviewed"] = True
    for target, source in (
        ("tail_audit_classification", "tail_audit_classification_review"),
        ("tail_audit_family", "tail_audit_family_review"),
        ("tail_audit_action", "tail_audit_action_review"),
        ("economic_error_correction", "economic_error_correction_review"),
        ("event_time_structural_context", "event_time_structural_context_review"),
    ):
        merged.loc[reviewed, target] = merged.loc[reviewed, source].to_numpy()
    semantic_collision = merged["tail_audit_action"].eq("set_latest_equal_original")
    merged.loc[semantic_collision, "gross_margin_latest"] = merged.loc[
        semantic_collision, "gross_margin"
    ].to_numpy()
    merged.loc[semantic_collision, "revision_size"] = 0.0
    merged.loc[semantic_collision, "was_revised"] = False
    merged.loc[reviewed, "revision_audit_version"] = f"revision-tail-audit-v1-{audit_hash}"
    return merged.drop(columns=[*rename.values(), "review_status"])


def _confidence_calibration(
    panel: pd.DataFrame,
    config: RevisionRobustnessConfig,
) -> pd.DataFrame:
    decision_eligible = panel["revision_buffer_eligible"]
    evaluable = decision_eligible & panel["revision_outcome_resolved"]
    rows: list[dict[str, Any]] = []
    for confidence in config.confidence_grid:
        buffer_column = _buffer_column(confidence)
        buffer = panel.loc[decision_eligible, buffer_column]
        evaluation_buffer = panel.loc[evaluable, buffer_column]
        within = panel.loc[evaluable, "revision_size"].abs() <= evaluation_buffer
        robust_decision = panel.loc[decision_eligible, "M2_innovation_raw"].abs() > buffer
        robust_evaluation = panel.loc[evaluable, "M2_innovation_raw"].abs() > evaluation_buffer
        flips = panel.loc[evaluable, "realized_direction_flip"].astype(bool)
        rows.append(
            {
                "confidence": confidence,
                "tail_probability": 1.0 - confidence,
                "decision_events": int(decision_eligible.sum()),
                "resolved_evaluation_events": int(evaluable.sum()),
                "median_revision_at_risk": float(buffer.median()),
                "mean_revision_at_risk": float(buffer.mean()),
                "empirical_coverage": float(within.mean()),
                "robust_decision_share": float(robust_decision.mean()),
                "robust_resolved_share": float(robust_evaluation.mean()),
                "realized_direction_flips": int(flips.sum()),
                "robust_direction_flips": int((robust_evaluation & flips).sum()),
                "false_robust_rate": float(
                    (robust_evaluation & flips).sum() / max(int(robust_evaluation.sum()), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _diagnostics_by_year(
    panel: pd.DataFrame,
    config: RevisionRobustnessConfig,
) -> pd.DataFrame:
    primary_buffer = _primary_buffer_column(config)
    eligible = panel.loc[panel["revision_buffer_eligible"]].copy()
    eligible["availability_year"] = eligible["actual_available_at"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in eligible.groupby("availability_year", sort=True):
        evaluable = group.loc[group["revision_outcome_resolved"]]
        rows.append(
            {
                "availability_year": int(str(year)),
                "decision_events": len(group),
                "resolved_evaluation_events": len(evaluable),
                "median_resolved_history": float(group["revision_history_observations"].median()),
                "median_revised_history": float(
                    group["revision_history_changed_observations"].median()
                ),
                "median_revision_at_risk": float(group[primary_buffer].median()),
                "resolved_revision_rate": (
                    float(evaluable["was_revised"].mean()) if not evaluable.empty else np.nan
                ),
                "empirical_coverage": (
                    float(evaluable["realized_revision_within_primary_buffer"].mean())
                    if not evaluable.empty
                    else np.nan
                ),
                "robust_decision_share": float(group["robust_direction"].mean()),
                "realized_direction_flips": (
                    int(evaluable["realized_direction_flip"].sum()) if not evaluable.empty else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_audit(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby(
            ["revision_source_type", "revision_outcome_status"],
            dropna=False,
            observed=True,
        )
        .agg(
            events=("event_id", "size"),
            companies=("company_id", "nunique"),
            changed_events=("was_revised", "sum"),
            median_resolution_days=(
                "resolution_days",
                "median",
            ),
        )
        .reset_index()
    )


def _tail_events(
    panel: pd.DataFrame,
    config: RevisionRobustnessConfig,
) -> pd.DataFrame:
    resolved = panel.loc[panel["revision_outcome_resolved"]].copy()
    resolved["absolute_revision"] = resolved["revision_size"].abs()
    buffer_column = _primary_buffer_column(config)
    resolved["exceeded_primary_buffer"] = resolved["revision_buffer_eligible"] & (
        resolved["absolute_revision"] > resolved[buffer_column]
    )
    columns = [
        "event_id",
        "company_id",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "actual_available_at",
        "revision_outcome_available_at",
        "revision_source_type",
        "M2_innovation_raw",
        "revision_size",
        "absolute_revision",
        "latest_innovation_hindsight",
        buffer_column,
        "exceeded_primary_buffer",
        "realized_direction_flip",
        "filing_accession",
        "latest_filing_accession",
    ]
    audit_columns = [
        "revision_size_unreviewed",
        "tail_audit_reviewed",
        "tail_audit_classification",
        "tail_audit_family",
        "tail_audit_action",
        "economic_error_correction",
        "event_time_structural_context",
        "revision_audit_version",
    ]
    columns.extend(column for column in audit_columns if column in resolved.columns)
    return resolved.loc[:, columns].sort_values("absolute_revision", ascending=False)


def _research_report(
    panel: pd.DataFrame,
    calibration: pd.DataFrame,
    config: RevisionRobustnessConfig,
) -> dict[str, Any]:
    primary = calibration.loc[calibration["confidence"].eq(config.primary_confidence)].iloc[0]
    resolved = panel["revision_outcome_resolved"]
    eligible = panel["revision_buffer_eligible"]
    evaluable = resolved & eligible
    reviewed = panel["tail_audit_reviewed"] if "tail_audit_reviewed" in panel else False
    semantic_corrections = (
        panel["tail_audit_action"].eq("set_latest_equal_original")
        if "tail_audit_action" in panel
        else pd.Series(False, index=panel.index)
    )
    return {
        "revision_robustness_version": str(panel["revision_robustness_version"].iloc[0]),
        "selected_model_id": config.selected_model_id,
        "config": asdict(config),
        "input_events": len(panel),
        "input_companies": int(panel["company_id"].nunique()),
        "resolved_revision_outcomes": int(resolved.sum()),
        "right_censored_outcomes": int((~resolved).sum()),
        "changed_resolved_outcomes": int(panel.loc[resolved, "was_revised"].sum()),
        "primary_decision_events": int(eligible.sum()),
        "primary_resolved_evaluation_events": int(evaluable.sum()),
        "primary_confidence": config.primary_confidence,
        "primary_median_revision_at_risk": float(primary["median_revision_at_risk"]),
        "primary_empirical_coverage": float(primary["empirical_coverage"]),
        "primary_robust_decision_share": float(primary["robust_decision_share"]),
        "primary_robust_resolved_share": float(primary["robust_resolved_share"]),
        "primary_realized_direction_flips": int(primary["realized_direction_flips"]),
        "primary_robust_direction_flips": int(primary["robust_direction_flips"]),
        "formal_amendment_events": int(panel["revision_source_type"].eq("amendment").sum()),
        "tail_audit_reviewed_events": int(reviewed.sum()) if isinstance(reviewed, pd.Series) else 0,
        "semantic_collision_corrections": int(semantic_corrections.sum()),
        "economic_error_corrections_in_reviewed_tail": int(
            panel.loc[panel["tail_audit_reviewed"], "economic_error_correction"].sum()
        ),
        "evidence_boundary": (
            "Revision means a source-audited change in the reconstructed published history. "
            "It can reflect accounting scope or presentation rather than an error correction, "
            "and this experiment contains no return evidence."
        ),
    }


def _research_version(panel: pd.DataFrame, config: RevisionRobustnessConfig) -> str:
    columns = [
        "event_id",
        "actual_available_at",
        "latest_available_at",
        "revision_source_type",
        "revision_size",
        "M2_innovation_raw",
        "tail_audit_action",
        "revision_audit_version",
    ]
    payload = panel.loc[:, columns].astype("string").to_csv(index=False).encode("utf-8")
    payload += json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    version = "v2" if panel["tail_audit_reviewed"].any() else "v1"
    return f"revision-robustness-{version}-" + hashlib.sha256(payload).hexdigest()[:16]


def _primary_buffer_column(config: RevisionRobustnessConfig) -> str:
    return _buffer_column(config.primary_confidence)


def _buffer_column(confidence: float) -> str:
    return f"revision_at_risk_{_confidence_suffix(confidence)}"


def _confidence_suffix(confidence: float) -> str:
    return str(round(confidence * 1_000)).zfill(3)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RevisionRobustnessError("Boolean columns contain invalid values")
    return normalized.eq("true")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RevisionRobustnessError(f"Missing required columns: {missing}")
