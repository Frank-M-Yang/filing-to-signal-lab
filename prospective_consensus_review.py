"""Outcome-free, non-promoting review of consensus membership contamination."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from prospective_consensus_quality import CLEAN, INSUFFICIENT_HISTORY, QUARANTINED, REVIEW_REQUIRED
from prospective_market_innovation import ProspectiveMarketInnovationConfig
from sec_data.filing_catalog import FilingCatalogError

CONSENSUS_REVIEW_VERSION: Final[str] = "prospective-consensus-review-v1"
ARITHMETIC_RELATIVE_TOLERANCE: Final[float] = 1e-10


class ConsensusReviewError(FilingCatalogError):
    """Raised when review inputs violate lineage or outcome-free constraints."""


@dataclass(frozen=True)
class ConsensusReviewTables:
    """Consecutive-pair counterfactuals and unchanged event admission decisions."""

    pair_evidence: pd.DataFrame
    event_review: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ConsensusReviewResult:
    """One immutable review artifact, separate from the admission authority."""

    output_root: Path
    report: dict[str, Any]


def calculate_consensus_membership_bounds(
    previous_count: int,
    current_count: int,
    previous_mean: float,
    current_mean: float,
    previous_low: float,
    previous_high: float,
    current_low: float,
    current_high: float,
) -> dict[str, Any]:
    """Test necessary conditions for composition-only changes, not actual attribution.

    Assumptions: means are unweighted arithmetic means over the reported counts;
    high/low bound that same panel; the smaller panel is a subset of the larger
    panel; retained analysts have unchanged estimates. These assumptions are
    NOT verified by aggregate data. Count-preserving churn is unobservable.
    A compatible result is only a failure to reject this counterfactual.
    """
    values = [previous_mean, current_mean, previous_low, previous_high, current_low, current_high]
    if (
        previous_count < 1
        or current_count < 1
        or not all(math.isfinite(value) for value in values)
        or min(previous_mean, current_mean) <= 0.0
    ):
        raise ConsensusReviewError("Invalid count or aggregate value in membership bounds")
    tolerance = ARITHMETIC_RELATIVE_TOLERANCE * max(previous_mean, current_mean, 1.0)
    if not (
        previous_low <= previous_mean + tolerance
        and previous_mean <= previous_high + tolerance
        and current_low <= current_mean + tolerance
        and current_mean <= current_high + tolerance
        and previous_low <= previous_high
        and current_low <= current_high
    ):
        raise ConsensusReviewError("Consensus mean lies outside its reported range")
    delta_count = current_count - previous_count
    base: dict[str, Any] = {
        "net_analyst_count_delta": delta_count,
        "membership_identity_verified": False,
        "required_added_or_removed_mean": None,
        "changed_panel_bound_low": None,
        "changed_panel_bound_high": None,
        "mean_only_composition_not_ruled_out": None,
        "nested_set_range_contradiction": None,
        "composition_not_ruled_out_by_necessary_bounds": None,
        "retained_mean_revision_lower_pct": None,
        "retained_mean_revision_upper_pct": None,
        "counterfactual_status": "equal_count_membership_still_unknown",
    }
    if delta_count == 0:
        return base
    sum_change = current_count * current_mean - previous_count * previous_mean
    required_mean = sum_change / delta_count
    if delta_count > 0:
        low, high = current_low, current_high
        lower_sum = sum_change - delta_count * high
        upper_sum = sum_change - delta_count * low
        retained_count = previous_count
        range_contradiction = bool(
            current_low > previous_low + tolerance or current_high < previous_high - tolerance
        )
    else:
        removed_count = -delta_count
        low, high = previous_low, previous_high
        lower_sum = sum_change + removed_count * low
        upper_sum = sum_change + removed_count * high
        retained_count = current_count
        range_contradiction = bool(
            current_low < previous_low - tolerance or current_high > previous_high + tolerance
        )
    mean_compatible = bool(low - tolerance <= required_mean <= high + tolerance)
    compatible = mean_compatible and not range_contradiction
    base.update(
        {
            "required_added_or_removed_mean": required_mean,
            "changed_panel_bound_low": low,
            "changed_panel_bound_high": high,
            "mean_only_composition_not_ruled_out": mean_compatible,
            "nested_set_range_contradiction": range_contradiction,
            "composition_not_ruled_out_by_necessary_bounds": compatible,
            "retained_mean_revision_lower_pct": lower_sum / retained_count / previous_mean,
            "retained_mean_revision_upper_pct": upper_sum / retained_count / previous_mean,
            "counterfactual_status": (
                "composition_not_ruled_out_by_necessary_bounds"
                if compatible
                else "simple_nested_composition_rejected_not_attribution_verified"
            ),
        }
    )
    return base


def calculate_prospective_consensus_review(
    event_quality: pd.DataFrame, consensus_snapshots: pd.DataFrame
) -> ConsensusReviewTables:
    """Review all frozen events without changing quality statuses or admission."""
    events, snapshots = _prepare_inputs(event_quality, consensus_snapshots)
    value_columns = [
        "captured_at",
        "source_sha256",
        "source_occurrence",
        "horizon",
        "revenue_estimate_average",
        "revenue_estimate_low",
        "revenue_estimate_high",
        "revenue_estimate_analyst_count",
    ]
    groups = snapshots.groupby("prospective_event_id", sort=False)
    for column in value_columns:
        snapshots[f"previous_{column}"] = groups[column].shift(1)
    paired = snapshots.loc[snapshots["previous_captured_at"].notna()].copy()
    paired = paired.merge(
        events[["prospective_event_id", "quality_status", "registration_order"]],
        on="prospective_event_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for raw in paired.to_dict("records"):
        row = {str(key): value for key, value in raw.items()}
        bounds = calculate_consensus_membership_bounds(
            int(row["previous_revenue_estimate_analyst_count"]),
            int(row["revenue_estimate_analyst_count"]),
            float(row["previous_revenue_estimate_average"]),
            float(row["revenue_estimate_average"]),
            float(row["previous_revenue_estimate_low"]),
            float(row["previous_revenue_estimate_high"]),
            float(row["revenue_estimate_low"]),
            float(row["revenue_estimate_high"]),
        )
        previous_at = pd.Timestamp(row["previous_captured_at"])
        current_at = pd.Timestamp(row["captured_at"])
        gap_days = (current_at.normalize() - previous_at.normalize()).days
        if current_at <= previous_at:
            raise ConsensusReviewError("A review pair has a non-prior observation clock")
        row.update(bounds)
        row.update(
            {
                "consensus_revision_pct": (
                    float(row["revenue_estimate_average"])
                    / float(row["previous_revenue_estimate_average"])
                    - 1.0
                ),
                "uncaptured_calendar_dates": max(gap_days - 1, 0),
                "intercapture_elapsed_days": (current_at - previous_at).total_seconds() / 86400,
                "change_timing_is_interval_censored": True,
                "counterfactual_available_at": current_at,
                "provider_structure_changed": bool(
                    row["horizon"] != row["previous_horizon"]
                    or row["source_occurrence"] != row["previous_source_occurrence"]
                ),
                "attribution_identified": False,
                "admission_override_authorized": False,
                "return_values_loaded": False,
            }
        )
        rows.append(row)
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        pairs = pd.DataFrame(columns=[*paired.columns, *_pair_extra_columns()])
    for column in (
        "mean_only_composition_not_ruled_out",
        "nested_set_range_contradiction",
        "composition_not_ruled_out_by_necessary_bounds",
    ):
        pairs[column] = pairs[column].astype("boolean")
    pairs = pairs.sort_values(["registration_order", "captured_at"], kind="mergesort")
    event_rows: list[dict[str, Any]] = []
    for raw in events.to_dict("records"):
        event = {str(key): value for key, value in raw.items()}
        changes = pairs.loc[pairs["prospective_event_id"].eq(event["prospective_event_id"])]
        latest = changes.iloc[-1].to_dict() if not changes.empty else {}
        focus = event["quality_status"] in {REVIEW_REQUIRED, QUARANTINED}
        event.update(
            {
                "review_focus": focus,
                "review_status": (
                    "aggregate_review_complete_external_evidence_pending"
                    if focus
                    else "not_flagged_by_frozen_rules_not_identity_proof"
                ),
                "quality_status_after_review": event["quality_status"],
                "downstream_admission_after_review": bool(event["downstream_admission"]),
                "membership_attribution_identified": False,
                "promotion_authorized": False,
                "pair_rows": len(changes),
                "latest_previous_captured_at": latest.get("previous_captured_at"),
                "latest_captured_at": latest.get("captured_at"),
                "latest_consensus_revision_pct": latest.get("consensus_revision_pct"),
                "latest_previous_analyst_count": latest.get(
                    "previous_revenue_estimate_analyst_count"
                ),
                "latest_analyst_count": latest.get("revenue_estimate_analyst_count"),
                "latest_mean_only_composition_not_ruled_out": latest.get(
                    "mean_only_composition_not_ruled_out"
                ),
                "latest_nested_set_range_contradiction": latest.get(
                    "nested_set_range_contradiction"
                ),
                "latest_counterfactual_status": latest.get("counterfactual_status", "no_pair"),
                "latest_uncaptured_calendar_dates": latest.get("uncaptured_calendar_dates", 0),
                "external_evidence_needed": (
                    "analyst_id_estimate_timestamp_panel_and_provider_remapping_history"
                    if focus
                    else "analyst_ids_would_be_required_to_verify_membership"
                ),
            }
        )
        event_rows.append(event)
    reviews = pd.DataFrame(event_rows)
    for column in (
        "latest_mean_only_composition_not_ruled_out",
        "latest_nested_set_range_contradiction",
    ):
        reviews[column] = reviews[column].astype("boolean")
    review_focus = reviews.loc[reviews["quality_status"].eq(REVIEW_REQUIRED)]
    protocol = ProspectiveMarketInnovationConfig()
    protocol.validate()
    admitted_events = int(reviews["downstream_admission_after_review"].sum())
    admitted_issuers = int(
        reviews.loc[reviews["downstream_admission_after_review"], "cik"].nunique()
    )
    report: dict[str, Any] = {
        "contract_version": CONSENSUS_REVIEW_VERSION,
        "events": len(reviews),
        "issuers": int(reviews["cik"].nunique()),
        "pair_rows": len(pairs),
        "review_required_events": int(reviews["quality_status"].eq(REVIEW_REQUIRED).sum()),
        "quarantined_events": int(reviews["quality_status"].eq(QUARANTINED).sum()),
        "review_latest_mean_only_composition_not_ruled_out": int(
            review_focus["latest_mean_only_composition_not_ruled_out"].fillna(False).sum()
        ),
        "review_latest_nested_set_range_contradictions": int(
            review_focus["latest_nested_set_range_contradiction"].fillna(False).sum()
        ),
        "pairs_with_uncaptured_calendar_dates": int(pairs["uncaptured_calendar_dates"].gt(0).sum()),
        "quality_promotions": 0,
        "membership_attribution_identified_events": 0,
        "downstream_admitted_events_before_review": int(reviews["downstream_admission"].sum()),
        "downstream_admitted_events_after_review": int(
            reviews["downstream_admission_after_review"].sum()
        ),
        "downstream_admitted_issuers": admitted_issuers,
        "formal_minimum_complete_events": protocol.minimum_complete_events,
        "formal_minimum_complete_issuers": protocol.minimum_complete_issuers,
        "current_admission_capacity_sufficient": bool(
            admitted_events >= protocol.minimum_complete_events
            and admitted_issuers >= protocol.minimum_complete_issuers
        ),
        "capacity_is_not_return_readiness": True,
        "return_gate_passed": False,
        "arithmetic_relative_tolerance": ARITHMETIC_RELATIVE_TOLERANCE,
        "assumptions_verified_from_aggregate_data": False,
        "bounds_are_necessary_not_sufficient": True,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "model_fitting_performed": False,
        "live_trading_authorized": False,
        "review_integrity_gate": True,
    }
    return ConsensusReviewTables(pairs.reset_index(drop=True), reviews, report)


def build_prospective_consensus_review(
    quality_root: Path,
    readiness_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_consensus_review"),
) -> ConsensusReviewResult:
    """Preserve a hash-verified, immutable aggregate review and readable report."""
    readiness_hash = _sha256_file(readiness_root / "output_manifest.json")
    quality_manifest = _read_json(quality_root / "output_manifest.json")
    if quality_manifest.get("report_sha256") != _sha256_file(
        quality_root / "prospective_consensus_quality_report.json"
    ):
        raise ConsensusReviewError("The parent quality report was changed")
    quality_report = _read_json(quality_root / "prospective_consensus_quality_report.json")
    if quality_report.get("readiness_manifest_sha256") != readiness_hash:
        raise ConsensusReviewError("Quality and readiness refer to different frozen inputs")
    lineage = {
        "quality_manifest_sha256": _sha256_file(quality_root / "output_manifest.json"),
        "readiness_manifest_sha256": readiness_hash,
        "implementation_sha256": _sha256_file(Path(__file__)),
        "protocol_implementation_sha256": _sha256_file(
            Path(__file__).with_name("prospective_market_innovation.py")
        ),
    }
    tables = calculate_prospective_consensus_review(
        _load_verified_table(quality_root, "prospective_consensus_event_quality.parquet"),
        _load_verified_table(
            readiness_root, "prospective_market_innovation_consensus_snapshots.parquet"
        ),
    )
    payload = json.dumps({"version": CONSENSUS_REVIEW_VERSION, **lineage}, sort_keys=True).encode()
    version = CONSENSUS_REVIEW_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / version
    if (target / "output_manifest.json").is_file():
        return ConsensusReviewResult(
            target.resolve(), _read_json(target / "prospective_consensus_review_report.json")
        )
    if target.exists():
        raise ConsensusReviewError(f"Incomplete consensus review exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in {
        "prospective_consensus_pair_review.parquet": tables.pair_evidence,
        "prospective_consensus_event_review.parquet": tables.event_review,
    }.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
    report: dict[str, Any] = {
        **tables.report,
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **lineage,
    }
    readable = staging / "consensus_membership_review.md"
    readable.write_text(_render_review(tables, report), encoding="utf-8")
    artifacts[readable.name] = {"sha256": _sha256_file(readable), "bytes": readable.stat().st_size}
    report["artifacts"] = artifacts
    report_path = staging / "prospective_consensus_review_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": version,
                **lineage,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return ConsensusReviewResult(target.resolve(), report)


def _prepare_inputs(
    event_quality: pd.DataFrame, consensus_snapshots: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = event_quality.copy()
    snapshots = consensus_snapshots.copy()
    _require_columns(
        events,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "quality_status",
            "downstream_admission",
            "actual_values_loaded",
            "return_values_loaded",
            "selection_uses_return_values",
        },
    )
    _require_columns(
        snapshots,
        {
            "prospective_event_id",
            "ticker",
            "target_period_end",
            "captured_at",
            "capture_date",
            "revenue_estimate_average",
            "revenue_estimate_high",
            "revenue_estimate_low",
            "revenue_estimate_analyst_count",
            "horizon",
            "source_occurrence",
            "source_sha256",
            "historical_backtest_admission",
        },
    )
    for frame in (events, snapshots):
        for flag in (
            "actual_values_loaded",
            "return_values_loaded",
            "selection_uses_return_values",
            "historical_backtest_admission",
            "live_trading_authorized",
        ):
            if flag in frame and frame[flag].fillna(False).astype(bool).any():
                raise ConsensusReviewError(f"Forbidden admission or outcome flag: {flag}")
        metadata = {"actual_values_loaded", "return_values_loaded", "selection_uses_return_values"}
        for column in frame.columns:
            is_outcome = (
                column.startswith(("actual_", "revenue_innovation")) or "return" in column
            ) and column not in metadata
            if is_outcome and frame[column].notna().any():
                raise ConsensusReviewError(f"Outcome value entered aggregate review: {column}")
    if events.empty or events["prospective_event_id"].duplicated().any():
        raise ConsensusReviewError("The event review registry is empty or non-atomic")
    if (
        not events["quality_status"]
        .isin([CLEAN, REVIEW_REQUIRED, QUARANTINED, INSUFFICIENT_HISTORY])
        .all()
    ):
        raise ConsensusReviewError("Unknown frozen quality status")
    if events["downstream_admission"].isna().any():
        raise ConsensusReviewError("Frozen admission is unknown")
    if not events["downstream_admission"].astype(bool).eq(events["quality_status"].eq(CLEAN)).all():
        raise ConsensusReviewError("Frozen quality status and admission disagree")
    if set(snapshots["prospective_event_id"]) - set(events["prospective_event_id"]):
        raise ConsensusReviewError("An unregistered event entered review")
    for column in ("captured_at", "capture_date", "target_period_end"):
        snapshots[column] = pd.to_datetime(snapshots[column], utc=True, errors="coerce")
        if snapshots[column].isna().any():
            raise ConsensusReviewError(f"Invalid snapshot timestamp: {column}")
    if not snapshots["capture_date"].eq(snapshots["captured_at"].dt.normalize()).all():
        raise ConsensusReviewError("Snapshot capture date was backdated or relabelled")
    identity = snapshots.merge(
        events[["prospective_event_id", "ticker", "target_period_end"]],
        on="prospective_event_id",
        suffixes=("", "_event"),
        validate="many_to_one",
    )
    if (
        identity["ticker"].ne(identity["ticker_event"]).any()
        or not pd.to_datetime(identity["target_period_end_event"], utc=True)
        .eq(identity["target_period_end"])
        .all()
    ):
        raise ConsensusReviewError("Snapshot ticker/period differs from the frozen event")
    if "accepted_at" in events:
        accepted = pd.to_datetime(
            snapshots["prospective_event_id"].map(
                events.set_index("prospective_event_id")["accepted_at"]
            ),
            utc=True,
            errors="coerce",
        )
        if (accepted.notna() & snapshots["captured_at"].ge(accepted)).any():
            raise ConsensusReviewError("A snapshot reaches or exceeds its event acceptance cutoff")
    for column in ("accepted_at", "registered_at"):
        if column not in snapshots:
            continue
        clock = pd.to_datetime(snapshots[column], utc=True, errors="coerce")
        invalid = clock.notna() & (
            snapshots["captured_at"].ge(clock)
            if column == "accepted_at"
            else snapshots["captured_at"].lt(clock)
        )
        if invalid.any():
            raise ConsensusReviewError(f"Snapshot violates its {column} clock")
    numeric = [
        "revenue_estimate_average",
        "revenue_estimate_high",
        "revenue_estimate_low",
        "revenue_estimate_analyst_count",
        "source_occurrence",
    ]
    for column in numeric:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")
    if not np.isfinite(snapshots[numeric].to_numpy(float)).all():
        raise ConsensusReviewError("Snapshot aggregate values must be finite")
    counts = snapshots["revenue_estimate_analyst_count"]
    if counts.isna().any() or counts.lt(1).any() or counts.ne(np.floor(counts)).any():
        raise ConsensusReviewError("Analyst counts must be positive integers")
    occurrence = snapshots["source_occurrence"]
    if occurrence.lt(0).any() or occurrence.ne(np.floor(occurrence)).any():
        raise ConsensusReviewError("Provider row occurrences must be nonnegative integers")
    means = snapshots["revenue_estimate_average"]
    tolerance = means.clip(lower=1.0) * ARITHMETIC_RELATIVE_TOLERANCE
    if (
        means.le(0).any()
        or snapshots["revenue_estimate_low"].gt(means + tolerance).any()
        or snapshots["revenue_estimate_high"].lt(means - tolerance).any()
        or snapshots["revenue_estimate_low"].gt(snapshots["revenue_estimate_high"]).any()
    ):
        raise ConsensusReviewError("Snapshot mean lies outside its aggregate range")
    if not snapshots["source_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ConsensusReviewError("Source snapshot hashes must be complete SHA-256 digests")
    snapshots = snapshots.sort_values(
        ["prospective_event_id", "captured_at", "source_sha256"], kind="mergesort"
    ).drop_duplicates(["prospective_event_id", "capture_date"], keep="last")
    return events.sort_values("registration_order", kind="mergesort"), snapshots.copy()


def _pair_extra_columns() -> list[str]:
    bounds = calculate_consensus_membership_bounds(1, 1, 100, 100, 90, 110, 90, 110)
    return [
        *bounds,
        "consensus_revision_pct",
        "uncaptured_calendar_dates",
        "intercapture_elapsed_days",
        "change_timing_is_interval_censored",
        "counterfactual_available_at",
        "provider_structure_changed",
        "attribution_identified",
        "admission_override_authorized",
        "return_values_loaded",
    ]


def _render_review(tables: ConsensusReviewTables, report: dict[str, Any]) -> str:
    focus = tables.event_review.loc[tables.event_review["review_focus"]]
    lines = [
        "# Consensus membership review / 一致预期成员变化复核",
        "",
        "## English",
        "",
        f"Generated: {report['generated_at']}. Events: {report['events']}; "
        f"review-required: {report['review_required_events']}; "
        f"quarantined: {report['quarantined_events']}.",
        "",
        "All frozen quality statuses and admission decisions remain unchanged. "
        "No event is released.",
        "",
        "The test assumes unweighted arithmetic means and a nested analyst panel. "
        "Neither assumption is verified by this provider's aggregate snapshots. "
        "High/low must describe the same panel.",
        "",
        "For additions, implied entering mean = (n1*m1 - n0*m0)/(n1-n0), "
        "tested against the current range. For removals the same expression "
        "is tested against the previous range. Range monotonicity is "
        "a separate necessary condition. Passing these tests does not identify the true cause.",
        "",
        "Even unchanged counts do not prove unchanged membership. A rejected simple counterfactual "
        "also does not prove genuine same-analyst revisions: churn, weighting or "
        "provider remapping may matter.",
        "",
        "## 中文",
        "",
        "这是一份聚合数据的算术复核，不是解除待审、预测收益或授权交易。原质量规则与准入状态全部保留。",
        "",
        "假设平均值是人数对应的简单算术平均，高低值属于同一分析师集合，且只发生净加入或净退出、"
        "没有同时换人。在这些未被验证的假设下，检验仅靠成员变化能否解释平均值变化。",
        "",
        "“未排除”不等于成员变化就是真因；“不兼容”也不等于已经证实同一分析师上修/下修。"
        "要解决归因，仍需分析师ID、逐人预测及修订时点、供应商映射记录。人数不变也可能存在换人。",
        "",
        "以下表格使用每个事件的最新比较。均值检验之外还单独列出高低范围是否违反嵌套集合假设。",
        "",
        "| 股票 | 目标财季 | 冻结状态 | 收入预期变化 | 人数 | "
        "均值层面仅成员变化 | 范围单调性矛盾 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for raw in focus.to_dict("records"):
        row = {str(key): value for key, value in raw.items()}
        revision = row.get("latest_consensus_revision_pct")
        pct = "n/a" if pd.isna(revision) else f"{float(revision) * 100:+.4f}%"
        old_count, new_count = (
            row.get("latest_previous_analyst_count"),
            row.get("latest_analyst_count"),
        )
        count = (
            "n/a"
            if pd.isna(old_count) or pd.isna(new_count)
            else f"{int(old_count)} → {int(new_count)}"
        )
        compatible = row.get("latest_mean_only_composition_not_ruled_out")
        label = "不适用" if pd.isna(compatible) else "未排除" if compatible else "不兼容"
        contradiction = row.get("latest_nested_set_range_contradiction")
        range_label = "不适用" if pd.isna(contradiction) else "有" if contradiction else "无"
        period = pd.Timestamp(row["target_period_end"]).date().isoformat()
        lines.append(
            f"| {row['ticker']} | {period} | {row['quality_status']} | {pct} | "
            f"{count} | {label} | {range_label} |"
        )
    lines.extend(
        [
            "",
            "变化发生时点只能定位在两次真实采集之间。未采日期不能用今天的供应商数据倒填。"
            "配对表保留两端时间、source SHA-256及条件性区间，以支持逐条检查。",
            "",
            "本次解除事件数：0。下游可准入事件数仍为 "
            f"{report['downstream_admitted_events_after_review']}。",
            "",
            "当前可准入队列容量为 "
            f"{report['downstream_admitted_events_after_review']} 个事件 / "
            f"{report['downstream_admitted_issuers']} 家发行人；"
            "冻结正式检验门槛为 "
            f"{report['formal_minimum_complete_events']} 个完整事件 / "
            f"{report['formal_minimum_complete_issuers']} 家发行人。"
            "若容量不足且问题未被解决，原队列无法达到门槛；"
            "容量达到也不代表实际值、收益或正负尾部已齐备。",
            "",
            "No actual values, returns, model fitting, broker submission or "
            "live authorization are used.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_verified_table(root: Path, name: str) -> pd.DataFrame:
    manifest = _read_json(root / "output_manifest.json")
    artifact = manifest.get("artifacts", {}).get(name)
    if not artifact or artifact.get("sha256") != _sha256_file(root / name):
        raise ConsensusReviewError(f"Missing or changed parent artifact: {name}")
    return pd.read_parquet(root / name)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ConsensusReviewError(f"Review evidence lacks columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
