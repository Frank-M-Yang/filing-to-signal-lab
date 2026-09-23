"""Outcome-free capacity planning; this module cannot register or admit events."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prospective_consensus_quality import CLEAN, INSUFFICIENT_HISTORY, QUARANTINED, REVIEW_REQUIRED
from prospective_market_innovation import ProspectiveMarketInnovationConfig
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

FEASIBILITY_VERSION = "prospective-cohort-feasibility-v1"
CORE_COLUMNS = [
    "core_asset_id",
    "core_asset_version",
    "core_asset_status",
    "company_name",
    "sec_cik",
    "sec_ticker",
    "mapping_status",
    "selection_order",
]


class CohortFeasibilityError(FilingCatalogError):
    """The capacity audit received inconsistent, contaminated, or changed inputs."""


@dataclass(frozen=True)
class CohortFeasibilityConfig:
    """Planning assumptions, not changes to the frozen research specification."""

    additional_retention_scenarios: tuple[float, ...] = (1.0, 0.9, 0.8, 0.6)
    tail_share_scenarios: tuple[float, ...] = (0.2, 0.3, 0.5)
    daily_budget_scenarios: tuple[int, ...] = (20, 25, 40)
    retry_reserve_requests: int = 1

    def validate(self) -> None:
        """Require meaningful sensitivity assumptions without admitting any cohort."""
        if not self.additional_retention_scenarios or any(
            not math.isfinite(p) or not 0 < p <= 1 for p in self.additional_retention_scenarios
        ):
            raise CohortFeasibilityError("Additional retention must be in (0, 1]")
        if not self.tail_share_scenarios or any(
            not math.isfinite(p) or not 0 < p <= 0.5 for p in self.tail_share_scenarios
        ):
            raise CohortFeasibilityError("Each hypothetical sign share must be in (0, 0.5]")
        if not self.daily_budget_scenarios or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in self.daily_budget_scenarios
        ):
            raise CohortFeasibilityError("Daily budgets must be positive integers")
        if self.retry_reserve_requests < 0:
            raise CohortFeasibilityError("Retry reserve must not be negative")


@dataclass(frozen=True)
class CohortFeasibilityTables:
    """Read-only coverage, conditional scenarios, and a non-registered probe queue."""

    issuer_quality: pd.DataFrame
    core_coverage: pd.DataFrame
    probe_queue: pd.DataFrame
    capacity_scenarios: pd.DataFrame
    request_budget_scenarios: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class CohortFeasibilityResult:
    """Immutable audit bundle and its manifest-linked report."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_cohort_feasibility(
    registry: pd.DataFrame,
    event_quality: pd.DataFrame,
    core_assets: pd.DataFrame,
    consensus_coverage: pd.DataFrame,
    *,
    config: CohortFeasibilityConfig | None = None,
) -> CohortFeasibilityTables:
    """Estimate conditional capacity using only identities and disclosure-quality metadata.

    Retention is an observed fraction in a selected sample, not a general probability.
    Additional attrition and sign balance are assumptions, not measured outcomes.
    Expected counts do not imply statistical power or probability of passing a gate.
    """
    cfg = config or CohortFeasibilityConfig()
    cfg.validate()
    protocol = ProspectiveMarketInnovationConfig()
    protocol.validate()
    events, quality, core, coverage = (
        frame.copy(deep=True)
        for frame in (registry, event_quality, core_assets, consensus_coverage)
    )
    for frame in (events, quality, core, coverage):
        _reject_outcomes(frame)
    identity = {"prospective_event_id", "cik", "ticker", "target_period_end"}
    _require_columns(events, identity | {"registration_order"})
    _require_columns(quality, identity | {"quality_status", "downstream_admission"})
    _require_columns(core, set(CORE_COLUMNS))
    _require_columns(
        coverage,
        {
            "cik",
            "ticker",
            "current_revenue_consensus_observed",
            "strict_forward_revenue_consensus_observed",
        },
    )
    if events.empty or events["prospective_event_id"].duplicated().any():
        raise CohortFeasibilityError("Empty or duplicate frozen registry")
    if quality["prospective_event_id"].duplicated().any() or set(
        events["prospective_event_id"]
    ) != set(quality["prospective_event_id"]):
        raise CohortFeasibilityError("Quality must cover exactly the frozen event registry")
    for frame, cik, ticker in (
        (events, "cik", "ticker"),
        (quality, "cik", "ticker"),
        (core, "sec_cik", "sec_ticker"),
        (coverage, "cik", "ticker"),
    ):
        frame[cik] = frame[cik].astype("string").str.strip().str.zfill(10)
        frame[ticker] = frame[ticker].astype("string").str.strip().str.upper()
    for frame in (events, quality):
        frame["target_period_end"] = pd.to_datetime(frame["target_period_end"], utc=True)
        if (
            frame[list(identity)].isna().any().any()
            or not frame["cik"].str.fullmatch(r"[0-9]{10}").all()
        ):
            raise CohortFeasibilityError("Invalid event identity")
    if events.duplicated(["cik", "target_period_end"]).any():
        raise CohortFeasibilityError("Multiple frozen events for one issuer and quarter")
    joined = events[list(identity)].merge(
        quality[list(identity)],
        on="prospective_event_id",
        suffixes=("", "_quality"),
        validate="one_to_one",
    )
    if any(
        joined[c].ne(joined[c + "_quality"]).any() for c in ("cik", "ticker", "target_period_end")
    ):
        raise CohortFeasibilityError("Frozen event identity changed in quality audit")
    statuses = [CLEAN, REVIEW_REQUIRED, QUARANTINED, INSUFFICIENT_HISTORY]
    if (
        not quality["quality_status"].isin(statuses).all()
        or not quality["downstream_admission"].isin([True, False]).all()
        or not quality["downstream_admission"].eq(quality["quality_status"].eq(CLEAN)).all()
    ):
        raise CohortFeasibilityError("Frozen quality and admission disagree")
    if core["core_asset_id"].isna().any() or core["core_asset_id"].duplicated().any():
        raise CohortFeasibilityError("Core identifiers must be unique and present")
    core["selection_order"] = pd.to_numeric(core["selection_order"], errors="coerce")
    if core["selection_order"].dropna().duplicated().any():
        raise CohortFeasibilityError("The original core selection order is ambiguous")
    # The last 12 frozen core issuers lack a selection_order. Preserve that
    # absence; use their original CSV row positions only as an explicit tie-break.
    core["core_source_row_order"] = range(1, len(core) + 1)
    core["selection_order_observed"] = core["selection_order"].notna()
    if (
        coverage.duplicated(["cik", "ticker"]).any()
        or not coverage[
            ["current_revenue_consensus_observed", "strict_forward_revenue_consensus_observed"]
        ]
        .isin([True, False])
        .all()
        .all()
    ):
        raise CohortFeasibilityError("Provider coverage is ambiguous or unknown")
    if coverage[["cik", "ticker"]].isna().any().any():
        raise CohortFeasibilityError("Provider identity is missing")
    issuer = (
        pd.crosstab(quality["cik"], quality["quality_status"])
        .reindex(columns=statuses, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    issuer["registered_events"] = issuer[statuses].sum(axis=1)
    issuer["has_clean_event"] = issuer[CLEAN].gt(0)
    issuer["clean_event_fraction"] = issuer[CLEAN] / issuer["registered_events"]
    core = core.sort_values(["selection_order", "core_source_row_order"], kind="mergesort")
    core["mapping_eligible"] = (
        core["core_asset_status"].eq("active")
        & core["mapping_status"].eq("accepted")
        & core["sec_cik"].str.fullmatch(r"[0-9]{10}").fillna(False)
        & core["sec_ticker"].notna()
        & core["sec_ticker"].ne("")
    )
    core = core.merge(
        coverage,
        left_on=["sec_cik", "sec_ticker"],
        right_on=["cik", "ticker"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    core["coverage_status"] = "unprobed"
    core.loc[core["_merge"].eq("both"), "coverage_status"] = "probed_no_strict_forward"
    core.loc[core["strict_forward_revenue_consensus_observed"].eq(True), "coverage_status"] = (
        "strict_forward_observed"
    )
    core["in_existing_registry"] = core["sec_cik"].isin(events["cik"])
    core = core.drop(columns=["cik", "ticker", "_merge"]).reset_index(drop=True)
    queue = core.loc[core["mapping_eligible"] & core["coverage_status"].eq("unprobed")].copy()
    queue = queue.drop_duplicates(["sec_cik", "sec_ticker"]).reset_index(drop=True)
    queue.insert(0, "probe_order", range(1, len(queue) + 1))
    queue["queue_is_registration"] = False
    queue["probe_authorized"] = False
    queue["selection_uses_return_values"] = False
    clean_events = int(quality["quality_status"].eq(CLEAN).sum())
    clean_issuers = int(issuer["has_clean_event"].sum())
    event_fraction = clean_events / len(events)
    issuer_fraction = clean_issuers / len(issuer)
    scenarios: list[dict[str, Any]] = []
    for retention in cfg.additional_retention_scenarios:
        for share in cfg.tail_share_scenarios:
            required_complete = max(
                protocol.minimum_complete_events,
                math.ceil(protocol.minimum_positive_innovations / share),
                math.ceil(protocol.minimum_negative_innovations / share),
            )
            scenarios.append(
                {
                    "assumed_additional_event_retention": retention,
                    "assumed_additional_issuer_retention": retention,
                    "assumed_positive_share": share,
                    "assumed_negative_share": share,
                    "observed_quality_event_fraction": event_fraction,
                    "observed_any_clean_issuer_fraction": issuer_fraction,
                    "required_complete_events_for_expected_counts": required_complete,
                    "raw_events_for_expected_counts": (
                        math.ceil(required_complete / (event_fraction * retention))
                        if event_fraction
                        else None
                    ),
                    "raw_issuers_for_expected_counts": (
                        math.ceil(protocol.minimum_complete_issuers / (issuer_fraction * retention))
                        if issuer_fraction
                        else None
                    ),
                    "counts_are_expectations_not_gate_guarantees": True,
                    "additional_retention_is_observed": False,
                    "tail_shares_are_observed": False,
                    "new_cohort_registered": False,
                }
            )
    budgets: list[dict[str, Any]] = []
    existing_daily = len(coverage)
    for budget in cfg.daily_budget_scenarios:
        spare = max(0, budget - existing_daily - cfg.retry_reserve_requests)
        budgets.append(
            {
                "assumed_daily_request_budget": budget,
                "existing_daily_consensus_requests": existing_daily,
                "assumed_retry_reserve": cfg.retry_reserve_requests,
                "available_new_probe_requests": spare,
                "days_for_one_pass_of_unprobed_queue": math.ceil(len(queue) / spare)
                if spare
                else None,
                "can_maintain_existing_with_reserve": budget
                >= existing_daily + cfg.retry_reserve_requests,
                "budget_entitlement_verified_for_this_key": False,
                "collection_authorized": False,
            }
        )
    report: dict[str, Any] = {
        "contract_version": FEASIBILITY_VERSION,
        "registered_events": len(events),
        "registered_issuers": len(issuer),
        "clean_events": clean_events,
        "issuers_with_any_clean_event": clean_issuers,
        "quality_status_counts": quality["quality_status"].value_counts().to_dict(),
        "observed_quality_event_fraction": event_fraction,
        "observed_any_clean_issuer_fraction": issuer_fraction,
        "minimum_complete_events": protocol.minimum_complete_events,
        "minimum_complete_issuers": protocol.minimum_complete_issuers,
        "minimum_positive_innovations": protocol.minimum_positive_innovations,
        "minimum_negative_innovations": protocol.minimum_negative_innovations,
        "current_quality_capacity_sufficient": (
            clean_events >= protocol.minimum_complete_events
            and clean_issuers >= protocol.minimum_complete_issuers
        ),
        "clean_event_shortfall": max(0, protocol.minimum_complete_events - clean_events),
        "clean_issuer_shortfall": max(0, protocol.minimum_complete_issuers - clean_issuers),
        "core_companies": len(core),
        "core_missing_selection_order": int(core["selection_order"].isna().sum()),
        "provider_probed_issuers": len(coverage),
        "core_strict_forward_observed": int(
            core["coverage_status"].eq("strict_forward_observed").sum()
        ),
        "core_unprobed": int(core["coverage_status"].eq("unprobed").sum()),
        "eligible_unprobed_queue": len(queue),
        "issuer_fraction_denominator_is_existing_selected_issuers": True,
        "retention_is_end_to_end": False,
        "statistical_power_estimated": False,
        "capture_time_is_analyst_revision_time": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "model_fitting_performed": False,
        "new_cohort_registered": False,
        "original_registry_changed": False,
        "quality_promotions": 0,
        "selection_uses_return_values": False,
        "collection_authorized": False,
        "live_trading_authorized": False,
        "planning_config": json.loads(json.dumps(asdict(cfg))),
    }
    capacity = pd.DataFrame(scenarios).astype(
        {
            "raw_events_for_expected_counts": "Int64",
            "raw_issuers_for_expected_counts": "Int64",
        }
    )
    requests = pd.DataFrame(budgets).astype({"days_for_one_pass_of_unprobed_queue": "Int64"})
    return CohortFeasibilityTables(issuer, core, queue, capacity, requests, report)


def build_prospective_cohort_feasibility(
    quality_root: Path,
    readiness_root: Path,
    registry_root: Path,
    core_root: Path,
    consensus_root: Path,
    provider_audit_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_cohort_feasibility"),
    config: CohortFeasibilityConfig | None = None,
) -> CohortFeasibilityResult:
    """Verify parent lineage and preserve an immutable audit without network requests."""
    cfg = config or CohortFeasibilityConfig()
    cfg.validate()
    roots = {
        "quality": quality_root,
        "readiness": readiness_root,
        "registry": registry_root,
        "core": core_root,
        "consensus": consensus_root,
        "provider_audit": provider_audit_root,
    }
    hashes = {name: _digest(root / "output_manifest.json") for name, root in roots.items()}
    quality_report = _verified_report(quality_root, "prospective_consensus_quality_report.json")
    readiness_report = _verified_report(
        readiness_root, "prospective_market_innovation_readiness_report.json"
    )
    _verified_report(registry_root, "prospective_market_innovation_report.json")
    consensus_report = _verified_report(consensus_root, "alpha_vantage_live_audit_report.json")
    provider_report = _verified_report(provider_audit_root, "consensus_provider_audit_report.json")
    if quality_report.get("readiness_manifest_sha256") != hashes["readiness"] or (
        readiness_report.get("preregistration_manifest_sha256") != hashes["registry"]
        or readiness_report.get("consensus_audit_manifest_sha256") != hashes["consensus"]
    ):
        raise CohortFeasibilityError(
            "Parent lineage does not match quality, registry and consensus"
        )
    core_manifest = _read_json(core_root / "output_manifest.json")
    core_path = core_root / "core_asset_82.csv"
    if core_manifest.get("artifacts", {}).get(core_path.name, {}).get("sha256") != _digest(
        core_path
    ):
        raise CohortFeasibilityError("Missing or changed core CSV artifact")
    tables = calculate_prospective_cohort_feasibility(
        load_catalog_table(registry_root, "prospective_market_innovation_events.parquet"),
        load_catalog_table(quality_root, "prospective_consensus_event_quality.parquet"),
        pd.read_csv(core_path, usecols=CORE_COLUMNS, dtype="string"),
        load_catalog_table(consensus_root, "alpha_vantage_live_coverage.parquet"),
        config=cfg,
    )
    providers = load_catalog_table(provider_audit_root, "consensus_provider_decisions.parquet")
    _reject_outcomes(providers)
    lineage = {
        "input_manifest_sha256": hashes,
        "implementation_sha256": _digest(Path(__file__)),
        "protocol_implementation_sha256": _digest(
            Path(__file__).with_name("prospective_market_innovation.py")
        ),
    }
    payload = {"version": FEASIBILITY_VERSION, **lineage, "config": asdict(cfg)}
    version = (
        FEASIBILITY_VERSION
        + "-"
        + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    )
    target = output_root / version
    if (target / "output_manifest.json").is_file():
        report = _verified_report(target, "prospective_cohort_feasibility_report.json")
        for name, artifact in _read_json(target / "output_manifest.json")["artifacts"].items():
            if _digest(target / name) != artifact["sha256"]:
                raise CohortFeasibilityError("Changed existing feasibility artifact")
        return CohortFeasibilityResult(target.resolve(), report)
    if target.exists():
        raise CohortFeasibilityError("An incomplete feasibility output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "issuer_quality_capacity.parquet": tables.issuer_quality,
        "core_consensus_coverage.parquet": tables.core_coverage,
        "unprobed_company_queue.parquet": tables.probe_queue,
        "capacity_scenarios.parquet": tables.capacity_scenarios,
        "request_budget_scenarios.parquet": tables.request_budget_scenarios,
        "provider_evidence_decisions.parquet": providers,
    }
    artifacts: dict[str, Any] = {}
    for name, frame in frames.items():
        path = staging / name
        frame.to_parquet(path, index=False)
        artifacts[name] = {
            "sha256": _digest(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
    report = {
        **tables.report,
        **lineage,
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quality_evidence_generated_at": quality_report.get("generated_at"),
        "consensus_evidence_generated_at": consensus_report.get("generated_at"),
        "provider_evidence_generated_at": provider_report.get("generated_at"),
        "capture_dates": readiness_report.get("capture_dates"),
        "provider_access_was_retested": False,
    }
    readable = staging / "cohort_feasibility.md"
    readable.write_text(_render_report(tables, report, providers), encoding="utf-8")
    artifacts[readable.name] = {"sha256": _digest(readable), "bytes": readable.stat().st_size}
    report["artifacts"] = artifacts
    report_path = staging / "prospective_cohort_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": version,
                **lineage,
                "artifacts": artifacts,
                "report_sha256": _digest(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if any(_digest(root / "output_manifest.json") != hashes[name] for name, root in roots.items()):
        raise CohortFeasibilityError("Parent changed during capacity audit; output not published")
    staging.rename(target)
    return CohortFeasibilityResult(target.resolve(), report)


def _render_report(
    tables: CohortFeasibilityTables, report: dict[str, Any], providers: pd.DataFrame
) -> str:
    lines = [
        "# 新前瞻队列：覆盖与容量审计 / Prospective cohort feasibility",
        "",
        f"版本：{report['dataset_version']}",
        f"质量证据时间：{report['quality_evidence_generated_at']}；这是本地证据，不是新抓取。",
        "",
        "## 已知事实",
        "",
        f"原注册 {report['registered_events']} 个事件、{report['registered_issuers']} 家公司；"
        f"其中 {report['clean_events']} 个 clean 事件、"
        f"{report['issuers_with_any_clean_event']} 家至少有一个 clean 事件。",
        f"状态计数：{report['quality_status_counts']}。"
        "原门槛仍是 30 个完整事件、15 家公司、正负创新各至少 8 个。",
        f"原核心库 {report['core_companies']} 家；"
        f"{report['core_strict_forward_observed']} 家曾观察到严格前瞻收入一致预期，"
        f"{report['core_unprobed']} 家尚未探测；未探测不等于缺数据。"
        f"候选探测队列 {len(tables.probe_queue)} 家。",
        f"{report['core_missing_selection_order']} 家缺失原筛选序号；缺失保留，"
        "这些公司排在尾部，按冻结 CSV 行序排列，不补造排名。",
        "",
        "## 条件容量（不是功效分析）",
        "",
        "当前质量事件比例和至少一个 clean 事件的公司比例，只描述已选的公司和已观察的抓取日。",
        "不是全球股票总体的合格概率，也不是经过实际值、时效性、价格、公司行动、退市和成本检查后的端到端通过率。",
        "同一公司多个季度不独立；公司比例会随每家公司观察季度数改变，不能机械外推。",
        "下面额外保留率及正负创新占比均为人为敏感性假设；未加载 actual、innovation 或收益。",
        "每个正负占比均针对所有完整事件，余下事件允许为零创新；它们不是信号出现频率的估计。",
        "",
        "| 额外保留率假设 | 正/负各占比假设 | 原始事件需求（期望值） | 原始公司需求（期望值） |",
        "| --- | --- | --- | --- |",
    ]
    for row in tables.capacity_scenarios.to_dict("records"):
        lines.append(
            f"| {row['assumed_additional_event_retention']:.0%} | "
            f"{row['assumed_positive_share']:.0%} | {row['raw_events_for_expected_counts']} | "
            f"{row['raw_issuers_for_expected_counts']} |"
        )
    lines += [
        "",
        "None 表示当前零 clean 样本无法外推。即使期望计数超过门槛，"
        "也不保证实际通过，更不证明赚钱。",
        "建议先评估约 30 家公司、60–90 个不重叠的新事件的操作容量；这是预算目标，不是正式注册。",
        "30 家 × 3 个季度只是 90 个事件的假想上限；"
        "季度和公告日期必须由真实 provider/SEC 证据确定，不凭空补 2027 年。",
        "更大样本仍需发行人聚类、跨时间验证及真正的效应量/功效分析，"
        "不能用 minimum gate 代替这些工作。",
        "",
        "## 请求预算（未变更现有采集计划）",
        "",
        "| 每日预算假设 | 原有公司每日请求 | 重试预留 | 可用于新探测 | 一遍候选队列所需天数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in tables.request_budget_scenarios.to_dict("records"):
        days = row["days_for_one_pass_of_unprobed_queue"]
        displayed_days = "无法安排" if days is None else str(days)
        lines.append(
            f"| {row['assumed_daily_request_budget']} | "
            f"{row['existing_daily_consensus_requests']} | "
            f"{row['assumed_retry_reserve']} | {row['available_new_probe_requests']} | "
            f"{displayed_days} |"
        )
    lines += [
        "",
        "天数按每日全额预算、一家公司一次探测、无其他调用计算，不含重复观察和失败请求。",
        "20 是当前软件计划；25 是官方通用免费额度，40 仅作扩展预算示例，"
        "均不表示已核验当前密钥权限。",
        "官方资料（查阅 2026-09-14）：[额度说明](https://www.alphavantage.co/support/)、"
        "[EARNINGS_ESTIMATES](https://www.alphavantage.co/documentation/#earnings-estimates)。",
        "官方端点提供收入/EPS 估计及分析师数量，"
        "但不能因此推断逐分析师身份、逐条修订时钟或历史版本已可用。",
        "每日抓取时间不是供应商更新时刻，更不是市场首次可得时刻；不允许把今日抓取回填到过去。",
        "若持续每日维护 30 家，一家公司一次请求已超过通用 25 次预算；必须先批准更高预算，"
        "或单独设计并注册较低频/错峰队列，验证事件前 7 天时效性；不能默默牺牲原 20 家的观察。",
        "",
        "## 既有供应商证据（没有重新测试权限）",
        "",
    ]
    for row in providers.to_dict("records"):
        lines.append(f"- {row['provider']}: {row['decision']} — {row['reason']}")
    lines += [
        "",
        "## 下一步与隔离边界",
        "",
        "1. 按原核心名单顺序探测未知公司，保留成功、失败、字段缺失及精确响应哈希；不看收益排序。",
        "2. 验证收入口径、币种/单位、财政季度、分析师数量、上下界、源行结构及至少两个不同抓取日。",
        "3. 明确预算和观察频率后，另行冻结新事件 ID、目标季度、首次抓取、"
        "SEC acceptance 截止及排除规则。",
        "4. 与原 37 个事件及已看过的事件 ID 做交集检查；任何重叠须归原队列，不得宣称新增独立样本。",
        "5. 最后才匹配原始 actual 和可交易收益；正负样本不足时记为结果，不追加挑选公司来凑门槛。",
        "",
        "原 37 个事件、质量状态与门槛均未修改；无新注册、质量升级、采集、拟合、收益评估或订单。",
        "核心库是大型公司且历史数据较完整的条件样本，仍有选择偏差；扩大这份名单并不解决全市场代表性。",
        "此报告是研究基础设施可行性成果，不是财报信号有效性或盈利报告。",
        "",
        "完整来源 SHA-256 链保存在同目录 output_manifest.json 与 JSON 报告中。",
        "",
    ]
    return "\n".join(lines)


def _reject_outcomes(frame: pd.DataFrame) -> None:
    flags = {
        "actual_values_loaded",
        "return_values_loaded",
        "selection_uses_return_values",
        "actual_value_loaded_at_registration",
        "return_value_loaded_at_registration",
        "replacement_allowed",
        "historical_backtest_admission",
        "live_trading_authorized",
    }
    for label in frame:
        column = str(label)
        if column in flags:
            if not frame[column].isin([False]).all():
                raise CohortFeasibilityError(f"Forbidden or unknown flag: {column}")
        elif (
            column.startswith(("actual_", "revenue_innovation", "innovation_"))
            or "return" in column
        ):
            if frame[column].notna().any():
                raise CohortFeasibilityError(f"Outcome value entered capacity audit: {column}")


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise CohortFeasibilityError(f"Missing capacity metadata: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _verified_report(root: Path, name: str) -> dict[str, Any]:
    if _read_json(root / "output_manifest.json").get("report_sha256") != _digest(root / name):
        raise CohortFeasibilityError(f"Changed parent report: {name}")
    return _read_json(root / name)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
