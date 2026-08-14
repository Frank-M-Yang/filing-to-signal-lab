"""Pre-register the first longitudinal filing-signal effectiveness study.

This module deliberately runs before longitudinal prices are collected.  It
freezes the event registry, estimands, windows, diagnostics, and verdict rules
so the eventual report cannot select a favorable horizon or sample after
observing returns.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

PRELIMINARY_EFFECTIVENESS_VERSION: Final[str] = "preliminary-effectiveness-preregistration-v1"
REPRESENTATION_FEATURES: Final[tuple[str, ...]] = (
    "gm_level",
    "gm_delta_yoy",
    "gm_innovation_raw",
    "gm_innovation_std",
)
FORBIDDEN_OUTCOME_FRAGMENTS: Final[tuple[str, ...]] = (
    "issuer_return",
    "market_return",
    "market_excess_return",
    "sector_excess_return",
    "entry_price",
    "exit_price",
)


@dataclass(frozen=True)
class PreliminaryEffectivenessConfig:
    """Frozen design choices for the first effectiveness report."""

    primary_signal: str = "gm_innovation_std"
    primary_comparator: str = "gm_delta_yoy"
    reference_representation: str = "gm_level"
    primary_horizon_sessions: int = 20
    secondary_horizons_sessions: tuple[int, ...] = (1, 5)
    entry_rule: str = "second_future_regular_session_open_after_filing_acceptance"
    exit_rule: str = "close_of_the_20th_holding_session"
    primary_benchmark: str = "SPY_point_in_time_total_return"
    secondary_benchmark: str = "point_in_time_sector_total_return_if_gate_passes"
    minimum_complete_events: int = 20
    minimum_complete_coverage: float = 0.80
    bootstrap_replications: int = 10_000
    bootstrap_seed: int = 20260814
    confidence_level: float = 0.95
    minimum_leave_one_issuer_out_sign_share: float = 0.80
    maximum_single_issuer_effect_share: float = 0.25
    version: str = PRELIMINARY_EFFECTIVENESS_VERSION

    def validate(self) -> None:
        """Reject a design that weakens the repository research invariants."""
        if self.primary_signal != "gm_innovation_std":
            raise ValueError("The primary signal must remain standardized OOS Innovation")
        if self.primary_comparator != "gm_delta_yoy":
            raise ValueError("Innovation must be compared with Delta")
        if self.primary_horizon_sessions != 20:
            raise ValueError("The primary holding horizon is frozen at 20 sessions")
        if tuple(self.secondary_horizons_sessions) != (1, 5):
            raise ValueError("Secondary horizons are frozen at one and five sessions")
        if self.minimum_complete_events < 20:
            raise ValueError("At least 20 complete events are required")
        if not 0.80 <= self.minimum_complete_coverage <= 1.0:
            raise ValueError("Complete-case coverage cannot be below 80%")
        if self.bootstrap_replications < 1_000:
            raise ValueError("Issuer-cluster bootstrap requires at least 1,000 replications")
        if not 0.90 <= self.confidence_level < 1.0:
            raise ValueError("Confidence level must be in [0.90, 1.0)")
        if not 0.50 <= self.minimum_leave_one_issuer_out_sign_share <= 1.0:
            raise ValueError("Leave-one-issuer-out sign share must be in [0.50, 1.0]")
        if not 0.0 < self.maximum_single_issuer_effect_share <= 0.50:
            raise ValueError("Single-issuer influence limit must be in (0, 0.50]")


@dataclass(frozen=True)
class PreliminaryEffectivenessTables:
    """Pure pre-registration tables before immutable persistence."""

    event_registry: pd.DataFrame
    hypothesis_registry: pd.DataFrame
    verdict_rules: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class PreliminaryEffectivenessResult:
    """Immutable pre-registration bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_preliminary_effectiveness_preregistration(
    feature_dense_events: pd.DataFrame,
    config: PreliminaryEffectivenessConfig | None = None,
) -> PreliminaryEffectivenessTables:
    """Freeze the analysis registry without accepting any observed outcomes."""
    cfg = config or PreliminaryEffectivenessConfig()
    cfg.validate()
    _reject_observed_outcomes(feature_dense_events)
    required = {
        "longitudinal_event_id",
        "cik",
        "accession_number",
        "accepted_at",
        "form",
        "formal_entry_session_date",
        "formal_exit_session_date",
        "representation_features_complete",
        *REPRESENTATION_FEATURES,
    }
    _require_columns(feature_dense_events, required, "feature-dense event panel")
    if feature_dense_events["longitudinal_event_id"].duplicated().any():
        raise FilingCatalogError("Longitudinal event ids must be unique")
    if feature_dense_events["accession_number"].duplicated().any():
        raise FilingCatalogError("Filing accessions must be unique")

    registry_columns = [
        "longitudinal_event_id",
        "cik",
        "accession_number",
        "accepted_at",
        "report_date",
        "form",
        "formal_entry_session_date",
        "formal_entry_open_at",
        "formal_exit_session_date",
        "formal_exit_close_at",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_error_history_end",
        *REPRESENTATION_FEATURES,
        "representation_features_complete",
        "feature_status",
    ]
    _require_columns(feature_dense_events, set(registry_columns), "feature-dense event panel")
    event_registry = feature_dense_events[registry_columns].copy()
    event_registry["cik"] = event_registry["cik"].astype(str).str.zfill(10)
    event_registry = event_registry.sort_values(
        ["accepted_at", "cik", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)
    event_registry["analysis_sample"] = event_registry["representation_features_complete"].astype(
        bool
    )
    event_registry["outcome_observed_at_preregistration"] = False

    total_events = len(event_registry)
    complete_events = int(event_registry["analysis_sample"].sum())
    complete_coverage = complete_events / total_events if total_events else 0.0
    required_by_coverage = math.ceil(cfg.minimum_complete_coverage * total_events)
    minimum_required = max(cfg.minimum_complete_events, required_by_coverage)
    hypotheses = _hypothesis_registry(cfg)
    verdict_rules = _verdict_rules(cfg, minimum_required)
    event_set_sha256 = _event_set_sha256(event_registry)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "study_title": "Level vs Delta vs Innovation：财报信号初步有效性检验",
        "study_status": "preregistered_returns_not_observed",
        "estimand": (
            "Does standardized out-of-sample gross-margin Innovation explain "
            "20-session post-filing market-excess returns better than Delta on "
            "the same point-in-time complete-case events?"
        ),
        "expected_direction": "higher_innovation_higher_market_excess_return",
        "candidate_events": total_events,
        "candidate_issuers": int(event_registry["cik"].nunique()),
        "complete_feature_events": complete_events,
        "complete_feature_coverage": complete_coverage,
        "minimum_combined_feature_return_events": minimum_required,
        "prospective_feature_gate": complete_events >= minimum_required,
        "event_set_sha256": event_set_sha256,
        "same_complete_case_sample_required": True,
        "selection_uses_event_returns": False,
        "outcomes_observed_at_preregistration": False,
        "annualized_performance_in_scope": False,
        "sharpe_ratio_in_scope": False,
        "strategy_pnl_in_scope": False,
        "investment_recommendation_in_scope": False,
        "primary_statistic": "issuer_cluster_bootstrap_spearman_ic_difference",
        "primary_contrast": "IC(gm_innovation_std,R20)-IC(gm_delta_yoy,R20)",
        "incremental_model": ("rank(R20) ~ rank(gm_delta_yoy) + rank(gm_innovation_std)"),
        "mandatory_sensitivities": [
            "leave_one_issuer_out",
            "exclude_ASNS_tail_case",
            "10-Q_only",
            "calendar_quarter_block",
            "secondary_1_and_5_session_horizons",
        ],
        "verdict_vocabulary": [
            "preliminary_supportive_evidence",
            "effectiveness_not_established",
            "innovation_not_distinct_from_delta",
            "issuer_concentrated_or_tail_fragile",
            "insufficient_data_quality_or_coverage",
        ],
        "config": asdict(cfg),
    }
    return PreliminaryEffectivenessTables(
        event_registry=event_registry,
        hypothesis_registry=hypotheses,
        verdict_rules=verdict_rules,
        report=report,
    )


def build_preliminary_effectiveness_preregistration(
    longitudinal_root: Path,
    *,
    output_root: Path = Path("data/processed/preliminary_effectiveness"),
    config: PreliminaryEffectivenessConfig | None = None,
) -> PreliminaryEffectivenessResult:
    """Build an immutable contract tied to the frozen longitudinal event set."""
    cfg = config or PreliminaryEffectivenessConfig()
    cfg.validate()
    longitudinal_manifest_sha256 = _sha256_file(longitudinal_root / "output_manifest.json")
    longitudinal_report = _read_json(longitudinal_root / "longitudinal_expansion_report.json")
    events = load_catalog_table(longitudinal_root, "feature_dense_event_panel.parquet")
    tables = calculate_preliminary_effectiveness_preregistration(events, cfg)
    payload = json.dumps(
        {
            "longitudinal_manifest_sha256": longitudinal_manifest_sha256,
            "event_set_sha256": tables.report["event_set_sha256"],
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return PreliminaryEffectivenessResult(
            target.resolve(), _read_json(target / "preregistration_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete preregistration directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "event_registry.parquet": tables.event_registry,
        "hypothesis_registry.parquet": tables.hypothesis_registry,
        "verdict_rules.parquet": tables.verdict_rules,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "analysis_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    shell_path = staging / "report_shell.md"
    shell_path.write_text(_report_shell(tables.report), encoding="utf-8")
    artifacts[shell_path.name] = _artifact_record(shell_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "longitudinal_version": longitudinal_report["dataset_version"],
        "longitudinal_manifest_sha256": longitudinal_manifest_sha256,
        "artifacts": artifacts,
    }
    report_path = staging / "preregistration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = staging / "output_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "longitudinal_manifest_sha256": longitudinal_manifest_sha256,
                "event_set_sha256": tables.report["event_set_sha256"],
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.rename(target)
    return PreliminaryEffectivenessResult(target.resolve(), report)


def _hypothesis_registry(config: PreliminaryEffectivenessConfig) -> pd.DataFrame:
    rows = [
        {
            "hypothesis_id": "H1_PRIMARY",
            "status": "primary_confirmatory",
            "horizon_sessions": config.primary_horizon_sessions,
            "feature": config.primary_signal,
            "comparator": config.primary_comparator,
            "outcome": "market_excess_total_return",
            "statistic": "issuer_cluster_bootstrap_delta_spearman_ic",
            "expected_direction": "positive",
            "multiplicity_role": "single_primary_test",
        },
        {
            "hypothesis_id": "H2_INCREMENTAL",
            "status": "required_supporting",
            "horizon_sessions": config.primary_horizon_sessions,
            "feature": config.primary_signal,
            "comparator": config.primary_comparator,
            "outcome": "market_excess_total_return",
            "statistic": "rank_regression_incremental_coefficient",
            "expected_direction": "positive",
            "multiplicity_role": "supporting_not_separate_discovery",
        },
        {
            "hypothesis_id": "H3_LEVEL_COMPARISON",
            "status": "descriptive",
            "horizon_sessions": config.primary_horizon_sessions,
            "feature": config.reference_representation,
            "comparator": config.primary_signal,
            "outcome": "market_excess_total_return",
            "statistic": "same_sample_spearman_ic",
            "expected_direction": "not_precommitted",
            "multiplicity_role": "descriptive",
        },
    ]
    for horizon in config.secondary_horizons_sessions:
        rows.append(
            {
                "hypothesis_id": f"S{horizon}_HORIZON",
                "status": "secondary_sensitivity",
                "horizon_sessions": horizon,
                "feature": config.primary_signal,
                "comparator": config.primary_comparator,
                "outcome": "market_excess_total_return",
                "statistic": "issuer_cluster_bootstrap_delta_spearman_ic",
                "expected_direction": "positive",
                "multiplicity_role": "secondary_BH_family",
            }
        )
    return pd.DataFrame(rows)


def _verdict_rules(config: PreliminaryEffectivenessConfig, minimum_required: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_order": 1,
                "verdict": "insufficient_data_quality_or_coverage",
                "condition": (
                    f"combined complete feature-return events < {minimum_required}, "
                    "or identity/action/delisting/benchmark gate fails"
                ),
                "terminal": True,
            },
            {
                "rule_order": 2,
                "verdict": "effectiveness_not_established",
                "condition": (
                    f"primary {config.primary_signal} IC is non-positive or its "
                    f"{config.confidence_level:.0%} issuer-cluster bootstrap CI includes zero"
                ),
                "terminal": True,
            },
            {
                "rule_order": 3,
                "verdict": "innovation_not_distinct_from_delta",
                "condition": (
                    "CI for IC(Innovation)-IC(Delta) includes zero, or incremental "
                    "Innovation rank coefficient is non-positive"
                ),
                "terminal": True,
            },
            {
                "rule_order": 4,
                "verdict": "issuer_concentrated_or_tail_fragile",
                "condition": (
                    "Innovation sign survives fewer than "
                    f"{config.minimum_leave_one_issuer_out_sign_share:.0%} of leave-one-issuer-out "
                    "runs, reverses without ASNS, or one issuer explains more than "
                    f"{config.maximum_single_issuer_effect_share:.0%} of the effect"
                ),
                "terminal": True,
            },
            {
                "rule_order": 5,
                "verdict": "preliminary_supportive_evidence",
                "condition": (
                    "all prior gates pass; wording remains preliminary and is not an "
                    "investment recommendation"
                ),
                "terminal": True,
            },
        ]
    )


def _report_shell(report: dict[str, Any]) -> str:
    return f"""# Level vs Delta vs Innovation：财报信号初步有效性报告

状态：**分析已预注册，纵向收益尚未读取**。

## 冻结样本

- 候选事件：{report["candidate_events"]}
- 候选公司：{report["candidate_issuers"]}
- 完整 Level/Delta/Innovation 事件：{report["complete_feature_events"]}
- 最低特征+收益完整事件：{report["minimum_combined_feature_return_events"]}
- 事件集 SHA-256：`{report["event_set_sha256"]}`

## 主问题

在完全相同的点时完整样本上，标准化样本外 Innovation 对公告后 20 个交易日的
市场超额总收益，是否比 Delta 具有更强、且不由单一公司驱动的解释力？

## 预先承诺

- 主要窗口：公告后第二个未来正常交易日开盘建仓，持有第 20 个交易日收盘退出。
- 主要对照：SPY 点时总收益；行业对照只在历史行业映射通过后作为次要结果。
- 主要统计量：按公司整簇重采样的 Spearman IC 差值。
- 必做稳健性：逐公司剔除、剔除 ASNS 尾部案例、仅 10-Q、日历季度块、1/5 日次要窗口。
- 不报告：年化收益、夏普、策略净值、买卖建议。

## 结果

待市场数据、公司行动、退市结局与基准门槛通过后自动填入。
"""


def _reject_observed_outcomes(frame: pd.DataFrame) -> None:
    observed = [
        column
        for column in frame.columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_OUTCOME_FRAGMENTS)
    ]
    if observed:
        raise FilingCatalogError(
            "Pre-registration input contains observed outcome columns: " + ", ".join(observed)
        )


def _event_set_sha256(frame: pd.DataFrame) -> str:
    values = (
        frame[["longitudinal_event_id", "cik", "accession_number", "accepted_at"]]
        .astype(str)
        .sort_values(["accepted_at", "cik", "accession_number"], kind="mergesort")
        .to_dict(orient="records")
    )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
