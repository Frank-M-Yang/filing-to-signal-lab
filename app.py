"""Streamlit application for the Bias-Aware Backtesting Framework."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import (
    STANDARD_SCENARIOS,
    BacktestConfig,
    BacktestResult,
    BiasConfig,
    calculate_bias_impact,
    compare_standard_scenarios,
    run_backtest,
)
from data_generator import ResearchData, SimulatedDataSource
from sec_data import SecConfigurationError, SecIngestionResult, SecIngestionService, SecRequestError
from sec_data.apple_case import AppleAmendmentCase, load_apple_amendment_case
from signals import SIGNAL_LABELS, SignalName

PAGE_TITLE = "财报量化信号验证框架"
REPRESENTATION_ROOT = Path("data/processed/global500_2026/representation_research_v1")
TOURNAMENT_ROOT = Path("data/processed/global500_2026/forecast_tournament_v2")
REVISION_ROBUSTNESS_ROOT = Path("data/processed/global500_2026/revision_robustness_v2")
EVENT_RETURN_DIAGNOSTIC_ROOT = Path("data/processed/event_return_diagnostics")
LONGITUDINAL_EXPANSION_ROOT = Path("data/processed/longitudinal_expansion")
PUBLIC_RESEARCH_PREVIEW_PATH = Path("data/examples/research_preview_summary.json")
COLORS = {
    "ink": "#132A27",
    "teal": "#0F766E",
    "orange": "#D97706",
    "slate": "#64748B",
    "red": "#B42318",
    "grid": "#DCE5E2",
}

st.set_page_config(page_title=PAGE_TITLE, page_icon="◫", layout="wide")


@dataclass(frozen=True)
class RepresentationStudyAssets:
    """Persisted real-data representation research used by the Streamlit view."""

    panel: pd.DataFrame
    common_sample: pd.DataFrame
    coverage: pd.DataFrame
    forecast_diagnostics: pd.DataFrame
    correlations: pd.DataFrame
    distributions: pd.DataFrame
    company_coverage: pd.DataFrame
    revisions: pd.DataFrame
    expectation_adjustment_diagnostics: pd.DataFrame
    rank_movements: pd.DataFrame
    forecast_loss_events: pd.DataFrame
    forecast_diagnostics_by_group: pd.DataFrame
    forecast_bootstrap_intervals: pd.DataFrame
    sigma_diagnostics: pd.DataFrame
    universe_timepoint_audit: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ForecastTournamentAssets:
    """Persisted M0-M4 forecast tournament outputs used by Streamlit."""

    common_sample: pd.DataFrame
    metrics: pd.DataFrame
    metrics_by_year: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    calibration: pd.DataFrame
    eligibility_gate: pd.DataFrame
    representation_diagnostics: pd.DataFrame
    subgroup_diagnostics: pd.DataFrame
    feature_clock_audit: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class RevisionRobustnessAssets:
    """Persisted walk-forward revision-risk outputs used by Streamlit."""

    panel: pd.DataFrame
    primary_decisions: pd.DataFrame
    confidence_calibration: pd.DataFrame
    diagnostics_by_year: pd.DataFrame
    source_audit: pd.DataFrame
    tail_events: pd.DataFrame
    reviewed_tail_events: pd.DataFrame
    tail_classification_summary: pd.DataFrame
    event_time_gate: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class EventReturnDiagnosticAssets:
    """Persisted real event-return pilot and its same-sample gates."""

    events: pd.DataFrame
    survivorship_counterfactual: pd.DataFrame
    form_diagnostics: pd.DataFrame
    feature_join_audit: pd.DataFrame
    representation_return_sample: pd.DataFrame
    representation_ic: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class LongitudinalExpansionAssets:
    """Frozen no-return expansion design and market collection batches."""

    issuer_eligibility: pd.DataFrame
    representative_events: pd.DataFrame
    feature_dense_events: pd.DataFrame
    market_collection_targets: pd.DataFrame
    report: dict[str, Any]


@st.cache_data(show_spinner="正在生成可复现的模拟研究数据…")
def load_research_data() -> ResearchData:
    """Load the deterministic built-in data source once per Streamlit process."""
    return SimulatedDataSource().load()


@st.cache_data(show_spinner=False)
def load_first_case_study() -> AppleAmendmentCase:
    """Load the repository's offline, audited first real-data result."""
    return load_apple_amendment_case()


@st.cache_data(show_spinner=False)
def load_public_research_preview() -> dict[str, Any]:
    """Load the aggregate-only fallback shipped with the public repository."""
    return json.loads(PUBLIC_RESEARCH_PREVIEW_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_representation_study() -> RepresentationStudyAssets:
    """Load the frozen Level/Delta/Innovation research outputs."""
    root = REPRESENTATION_ROOT
    report = json.loads((root / "research_report.json").read_text(encoding="utf-8"))
    csv_options = {"dtype": {"company_id": "string", "sic": "string"}}
    return RepresentationStudyAssets(
        panel=pd.read_csv(root / "representation_panel.csv", **csv_options),
        common_sample=pd.read_csv(root / "common_representation_sample.csv", **csv_options),
        coverage=pd.read_csv(root / "coverage_by_year.csv"),
        forecast_diagnostics=pd.read_csv(root / "forecast_diagnostics_by_year.csv"),
        correlations=pd.read_csv(root / "representation_correlations.csv"),
        distributions=pd.read_csv(root / "representation_distributions.csv"),
        company_coverage=pd.read_csv(root / "company_coverage.csv", **csv_options),
        revisions=pd.read_csv(root / "revision_diagnostics.csv", **csv_options),
        expectation_adjustment_diagnostics=pd.read_csv(
            root / "expectation_adjustment_diagnostics.csv"
        ),
        rank_movements=pd.read_csv(root / "representation_rank_movements.csv", **csv_options),
        forecast_loss_events=pd.read_csv(root / "forecast_loss_events.csv", **csv_options),
        forecast_diagnostics_by_group=pd.read_csv(root / "forecast_diagnostics_by_group.csv"),
        forecast_bootstrap_intervals=pd.read_csv(root / "forecast_bootstrap_intervals.csv"),
        sigma_diagnostics=pd.read_csv(root / "sigma_diagnostics.csv"),
        universe_timepoint_audit=pd.read_csv(root / "universe_timepoint_audit.csv", **csv_options),
        report=report,
    )


@st.cache_data(show_spinner=False)
def load_forecast_tournament() -> ForecastTournamentAssets:
    """Load the frozen M0-M4 common-sample tournament outputs."""
    root = TOURNAMENT_ROOT
    report = json.loads((root / "tournament_report.json").read_text(encoding="utf-8"))
    csv_options = {"dtype": {"company_id": "string", "sic": "string"}}
    return ForecastTournamentAssets(
        common_sample=pd.read_csv(root / "tournament_common_sample.csv", **csv_options),
        metrics=pd.read_csv(root / "model_metrics.csv"),
        metrics_by_year=pd.read_csv(root / "model_metrics_by_year.csv"),
        bootstrap_intervals=pd.read_csv(root / "clustered_bootstrap_intervals.csv"),
        calibration=pd.read_csv(root / "calibration_diagnostics.csv"),
        eligibility_gate=pd.read_csv(root / "eligibility_gate.csv"),
        representation_diagnostics=pd.read_csv(root / "representation_diagnostics.csv"),
        subgroup_diagnostics=pd.read_csv(root / "subgroup_diagnostics.csv"),
        feature_clock_audit=pd.read_csv(root / "feature_clock_audit.csv"),
        report=report,
    )


@st.cache_data(show_spinner=False)
def load_revision_robustness() -> RevisionRobustnessAssets:
    """Load the frozen M2 revision-risk and safety-margin experiment."""
    root = REVISION_ROBUSTNESS_ROOT
    report = json.loads((root / "revision_robustness_report.json").read_text(encoding="utf-8"))
    csv_options = {"dtype": {"company_id": "string", "sic": "string"}}
    return RevisionRobustnessAssets(
        panel=pd.read_csv(root / "revision_robustness_panel.csv", **csv_options),
        primary_decisions=pd.read_csv(root / "primary_decisions.csv", **csv_options),
        confidence_calibration=pd.read_csv(root / "confidence_calibration.csv"),
        diagnostics_by_year=pd.read_csv(root / "diagnostics_by_year.csv"),
        source_audit=pd.read_csv(root / "revision_source_audit.csv"),
        tail_events=pd.read_csv(root / "revision_tail_events.csv", **csv_options),
        reviewed_tail_events=pd.read_csv(root / "reviewed_tail_events.csv", **csv_options),
        tail_classification_summary=pd.read_csv(root / "tail_classification_summary.csv"),
        event_time_gate=pd.read_csv(root / "event_time_gate_diagnostic.csv"),
        report=report,
    )


@st.cache_data(show_spinner=False)
def load_event_return_diagnostics() -> EventReturnDiagnosticAssets:
    """Load the latest immutable real event-return diagnostic."""
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for report_path in EVENT_RETURN_DIAGNOSTIC_ROOT.glob("*/event_return_diagnostic_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent, report))
    if not candidates:
        raise FileNotFoundError(
            "No event-return diagnostic exists; run scripts.build_event_return_diagnostics"
        )
    _, root, report = max(candidates, key=lambda item: item[0])
    return EventReturnDiagnosticAssets(
        events=pd.read_parquet(root / "event_return_diagnostics.parquet"),
        survivorship_counterfactual=pd.read_parquet(root / "survivorship_counterfactual.parquet"),
        form_diagnostics=pd.read_parquet(root / "form_diagnostics.parquet"),
        feature_join_audit=pd.read_parquet(root / "feature_join_audit.parquet"),
        representation_return_sample=pd.read_parquet(root / "representation_return_sample.parquet"),
        representation_ic=pd.read_parquet(root / "representation_ic.parquet"),
        gates=pd.read_parquet(root / "diagnostic_gates.parquet"),
        report=report,
    )


@st.cache_data(show_spinner=False)
def load_longitudinal_expansion() -> LongitudinalExpansionAssets:
    """Load the latest immutable longitudinal event-expansion design."""
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for report_path in LONGITUDINAL_EXPANSION_ROOT.glob("*/longitudinal_expansion_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.append((str(report.get("generated_at", "")), report_path.parent, report))
    if not candidates:
        raise FileNotFoundError(
            "No longitudinal expansion exists; run scripts.build_longitudinal_expansion"
        )
    _, root, report = max(candidates, key=lambda item: item[0])
    return LongitudinalExpansionAssets(
        issuer_eligibility=pd.read_parquet(root / "issuer_eligibility_audit.parquet"),
        representative_events=pd.read_parquet(root / "representative_event_frame.parquet"),
        feature_dense_events=pd.read_parquet(root / "feature_dense_event_panel.parquet"),
        market_collection_targets=pd.read_parquet(
            root / "issuer_market_collection_targets.parquet"
        ),
        report=report,
    )


def main() -> None:
    """Render the interactive bias audit."""
    _inject_styles()
    workspace = st.sidebar.radio(
        "工作区",
        [
            "偏差回测实验室",
            "成果 01 · Apple 修订案例",
            "成果 02 · Level / Delta / Innovation",
            "成果 03 · Forecast Tournament v2",
            "成果 04 · 修订风险与安全边际",
            "成果 05 · 真实事件收益诊断",
            "SEC 真实数据试验台",
        ],
        index=0,
    )
    if workspace == "成果 01 · Apple 修订案例":
        _render_apple_case_study()
        return
    if workspace == "SEC 真实数据试验台":
        _render_sec_workbench()
        return
    if workspace == "成果 02 · Level / Delta / Innovation":
        _render_representation_study()
        return
    if workspace == "成果 03 · Forecast Tournament v2":
        _render_forecast_tournament()
        return
    if workspace == "成果 04 · 修订风险与安全边际":
        _render_revision_robustness()
        return
    if workspace == "成果 05 · 真实事件收益诊断":
        _render_event_return_diagnostics()
        return
    data = load_research_data()
    signal_name, config, current_biases = _sidebar(data)
    current = run_backtest(data, signal_name, config, current_biases)
    standards = compare_standard_scenarios(data, signal_name, config)
    impact = calculate_bias_impact(data, signal_name, config)

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">BIAS-AWARE BACKTESTING FRAMEWORK</div>
          <h1>财报信号，还是数据偏差？</h1>
          <p>用同一信号、同一模拟市场，逐项移除前视、幸存者与公告时点偏差。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _bias_badges(current_biases)
    _metric_cards(current)

    left, right = st.columns([1.62, 1.0], gap="large")
    with left:
        st.subheader("净值曲线：修正偏差的全过程")
        st.caption("固定比较三种审计情景；若当前侧边栏选择不同，则以虚线叠加。")
        st.plotly_chart(
            _equity_figure(standards, current, current_biases),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.subheader("当前情景的五分位收益")
        st.caption("Q1 为最低信号，Q5 为最高信号；数值为平均 20 日超额收益。")
        st.plotly_chart(
            _quantile_figure(current),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.subheader("偏差影响拆解")
    st.caption("每一项均从“全部偏差”基准出发单独修正；正的削弱值表示原偏差夸大了收益。")
    display_impact = impact.copy()
    display_impact["年化收益"] *= 100.0
    display_impact["相对带偏差基准的收益削弱"] *= 100.0
    st.dataframe(
        display_impact,
        hide_index=True,
        width="stretch",
        column_config={
            "年化收益": st.column_config.NumberColumn(format="%.1f%%"),
            "相对带偏差基准的收益削弱": st.column_config.NumberColumn(format="%.1f%%"),
            "夏普比率": st.column_config.NumberColumn(format="%.2f"),
            "IC": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.info(_research_summary(impact, standards, signal_name), icon="🔎")

    with st.expander("查看点时数据与执行规则"):
        _data_audit_panel(data, current)
    st.caption("模拟研究演示 · 固定随机种子 42 · 不构成真实市场证据、投资建议或可交易策略")


def _render_apple_case_study() -> None:
    """Render the project's first audited real-filing case study."""
    case = load_first_case_study()
    metadata = case.metadata
    latest_margin = case.gross_margin.loc[case.gross_margin["fiscal_year"].eq(2009)].iloc[0]
    changed_count = int(metadata["changed_fact_count"])
    comparable_count = int(metadata["comparable_fact_count"])

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 01 · REAL SEC FILING</div>
          <h1>90 天后，财报里的信号变了</h1>
          <p>Apple 2009 年 10-K 与 10-K/A：同一经济期间、两套当时可得性完全不同的财务数字。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="bias-badge off">真实 SEC 数据</span>'
        '<span class="bias-badge off">Point-in-Time 审计</span>'
        '<span class="bias-badge on">不是欺诈或错误更正案例</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "这次变化来自收入确认准则的追溯采用。修订数字可以更具可比性，但在 2009 年 10 月 27 日"
        "尚不存在；把它们用于当天的历史信号，就是前视偏差。",
        icon="⚠️",
    )

    cards = st.columns(4)
    cards[0].metric("信息延迟", f"{case.information_delay_days:.0f} 天")
    cards[1].metric("严格可比 facts", f"{comparable_count:,}")
    cards[2].metric("数值发生变化", f"{changed_count:,} / {comparable_count:,}")
    cards[3].metric(
        "2009 毛利率信号放大",
        _percentage_points(float(latest_margin["signal_inflation"])),
    )

    story_tab, data_tab, source_tab = st.tabs(["看见前视偏差", "关键事实", "来源与边界"])
    with story_tab:
        st.subheader("市场何时真正知道这些数字？")
        st.plotly_chart(
            _case_timeline_figure(case),
            width="stretch",
            config={"displayModeBar": False},
        )

        left, right = st.columns([1.05, 1.0], gap="large")
        with left:
            st.subheader("同一指标，两种历史")
            label_to_tag = {
                "营业收入": "SalesRevenueNet",
                "毛利润": "GrossProfit",
                "净利润": "NetIncomeLoss",
                "稀释每股收益": "EarningsPerShareDiluted",
                "基本每股收益": "EarningsPerShareBasic",
            }
            selected_label = st.selectbox("选择指标", list(label_to_tag), index=0)
            st.plotly_chart(
                _case_metric_figure(case, label_to_tag[selected_label]),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.subheader("毛利率水平被重新描绘")
            st.caption("毛利率 = 毛利润 ÷ 营业收入；百分点差异按同一财年比较。")
            st.plotly_chart(
                _gross_margin_level_figure(case),
                width="stretch",
                config={"displayModeBar": False},
            )

        st.subheader("真正进入策略的信号：毛利率同比变化")
        signal_columns = st.columns(3)
        signal_columns[0].metric(
            "2009 当时可得信号",
            _percentage_points(float(latest_margin["original_yoy_change"])),
        )
        signal_columns[1].metric(
            "90 天后修订信号",
            _percentage_points(float(latest_margin["amendment_yoy_change"])),
        )
        signal_columns[2].metric(
            "错误回填造成的放大",
            _percentage_points(float(latest_margin["signal_inflation"])),
        )
        st.plotly_chart(
            _gross_margin_signal_figure(case),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.info(
            "2009 年 10 月 27 日，严格点时回测只能得到 +1.65 个百分点的毛利率同比信号；"
            "若错误回填 2010 年 1 月 25 日的修订数字，信号会变成 +4.94 个百分点，"
            "被放大 3.29 个百分点。",
            icon="🔎",
        )

    with data_tab:
        st.subheader("进入案例的 13 条信号相关事实")
        st.caption(
            "从完整的 210 组严格比较中，只保留收入、毛利润、净利润和 EPS；"
            "原始值与修订值仍保留 accession、UTC 接受时间、单位和期间。"
        )
        display = case.facts[
            [
                "metric_label",
                "fiscal_year",
                "unit",
                "original_value",
                "amendment_value",
                "percentage_difference",
                "original_accession",
                "amendment_accession",
            ]
        ].copy()
        display["percentage_difference"] *= 100.0
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "metric_label": "指标",
                "fiscal_year": "财年",
                "unit": "单位",
                "original_value": st.column_config.NumberColumn("原始值", format="%.2f"),
                "amendment_value": st.column_config.NumberColumn("修订值", format="%.2f"),
                "percentage_difference": st.column_config.NumberColumn(
                    "修订幅度", format="%+.2f%%"
                ),
                "original_accession": "原始 accession",
                "amendment_accession": "修订 accession",
            },
        )
        _download_csv("下载案例明细 CSV", case.facts, "apple-2009-signal-case.csv")

    with source_tab:
        st.subheader("为什么数字发生变化？")
        st.markdown(
            "Apple 在 2010 财年第一季度追溯采用了有关多重交付安排与软件要素的修订收入确认准则。"
            "旧会计处理对 iPhone 和 Apple TV 使用订阅会计、递延收入和成本；新处理通常在销售时确认"
            "绝大部分硬件收入和产品成本。因此，这是一例**会计准则追溯采用**，不是简单的录入错误。"
        )
        st.markdown(
            f"- [原始 2009 10-K（{metadata['original_accession']}）]"
            f"({metadata['original_filing_url']})：{metadata['original_accepted_at']}\n"
            f"- [修订 10-K/A（{metadata['amendment_accession']}）]"
            f"({metadata['amendment_filing_url']})：{metadata['amendment_accepted_at']}\n"
            f"- 完整比较导出 SHA-256：`{metadata['source_export_sha256']}`"
        )
        st.markdown("**研究边界**")
        st.markdown(
            "- 这是单公司、单次修订的机制验证，不代表全市场重述频率。\n"
            "- 当前案例证明财务信号会被未来版本改变，但不估计交易收益。\n"
            "- SUE 仍需要连续八季度 EPS 和当时可得的预期基准；"
            "真实回测还需要价格、交易日历和历史股票池。"
        )


def _render_forecast_tournament() -> None:
    """Render the preregistered M0-M4 expectation-model tournament."""
    try:
        study = load_forecast_tournament()
    except (OSError, ValueError, json.JSONDecodeError):
        _render_public_tournament_preview()
        return

    report = study.report
    metrics = study.metrics.set_index("model_id")
    selected_id = str(report["selected_model_id"])
    selected = metrics.loc[selected_id]
    selected_gate = study.eligibility_gate.loc[
        study.eligibility_gate["model_id"].eq(selected_id)
    ].iloc[0]
    selected_mae_interval = study.bootstrap_intervals.loc[
        study.bootstrap_intervals["model_id"].eq(selected_id)
        & study.bootstrap_intervals["metric"].eq("mae_improvement_vs_naive")
    ].iloc[0]

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 03 · PREREGISTERED FORECAST TOURNAMENT</div>
          <h1>什么才算真正的 Innovation？</h1>
          <p>在同一批 SEC 点时事件上，让 M0–M4 先竞争预测准确性；表示差异只在选模之后报告。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="bias-badge off">共同样本</span>'
        '<span class="bias-badge off">严格 Expanding OOS</span>'
        '<span class="bias-badge off">MAE 预注册主指标</span>'
        '<span class="bias-badge on">收益研究仍被数据 Gate 阻止</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "M2 通过的是财务预测闸门，不是收益闸门。当前股票池和 SIC 都来自 2026 快照，"
        "尚无历史成分、退市证券和点时价格，因此这里不能声称发现了可交易 alpha。",
        icon="⚠️",
    )

    cards = st.columns(4)
    cards[0].metric(
        "共同样本",
        f"{int(report['common_sample_events']):,}",
        f"{int(report['common_sample_companies'])} 家公司",
        delta_color="off",
    )
    cards[1].metric("胜出模型", selected_id)
    cards[2].metric("MAE 改善", f"{float(selected['mae_improvement_vs_naive']):+.1%}")
    cards[3].metric(
        "稳定年份",
        f"{int(selected_gate['positive_mae_skill_years'])} / "
        f"{int(selected_gate['eligible_years'])}",
    )

    result_tab, representation_tab, audit_tab = st.tabs(
        ["模型竞赛", "选模后的表示诊断", "时点审计与证据边界"]
    )
    with result_tab:
        left, right = st.columns([1.05, 1.25], gap="large")
        with left:
            st.subheader("共同样本 MAE skill")
            st.caption("零线是去年同季 M0；误差线为公司×年份双向聚类 bootstrap 95% 区间。")
            st.plotly_chart(
                _tournament_mae_figure(study.metrics, study.bootstrap_intervals),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.subheader("年度稳定性")
            st.caption("闸门只统计至少 25 个共同事件的年份；2019 和 2026 仍展示但不参与多数判断。")
            st.plotly_chart(
                _tournament_year_figure(study.metrics_by_year),
                width="stretch",
                config={"displayModeBar": False},
            )

        st.success(
            f"{selected_id}（{selected['model_name']}）按预注册 MAE 规则胜出：共同样本改善 "
            f"{float(selected['mae_improvement_vs_naive']):.1%}，聚类区间为 "
            f"[{float(selected_mae_interval['ci_lower_2_5']):.1%}, "
            f"{float(selected_mae_interval['ci_upper_97_5']):.1%}]，并在全部 "
            f"{int(selected_gate['eligible_years'])} 个合格年份改善 MAE。",
            icon="✅",
        )
        st.info(
            "M3 和 M4 也通过预测闸门，但没有超过更简单的 M2。这意味着本轮最有价值的增量来自"
            "公司自身的最近状态与轨迹，而不是更复杂的 peer 特征或非线性展开。",
            icon="ℹ️",
        )

        st.subheader("冻结闸门逐项结果")
        gate_display = study.eligibility_gate.copy()
        gate_display["aggregate_mae_skill"] *= 100.0
        st.dataframe(
            gate_display[
                [
                    "model_id",
                    "model_name",
                    "aggregate_mae_skill",
                    "positive_mae_skill_years",
                    "eligible_years",
                    "calibration_pass",
                    "forecast_gate_pass",
                    "primary_selection_candidate",
                    "eligible_for_downstream_returns",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "model_id": "模型",
                "model_name": "定义",
                "aggregate_mae_skill": st.column_config.NumberColumn(
                    "总体 MAE skill", format="%+.1f%%"
                ),
                "positive_mae_skill_years": "改善年份",
                "eligible_years": "合格年份",
                "calibration_pass": "校准通过",
                "forecast_gate_pass": "预测 Gate",
                "primary_selection_candidate": "主选模候选",
                "eligible_for_downstream_returns": "收益 Gate",
            },
        )

        with st.expander("查看校准与分组损失诊断"):
            calibration_display = study.calibration.copy()
            calibration_display["bias"] *= 100.0
            calibration_display["absolute_bias_limit"] *= 100.0
            st.dataframe(
                calibration_display,
                hide_index=True,
                width="stretch",
                column_config={
                    "bias": st.column_config.NumberColumn("Bias", format="%+.2f%%"),
                    "absolute_bias_limit": st.column_config.NumberColumn(
                        "Bias 上限", format="%.2f%%"
                    ),
                    "calibration_slope": st.column_config.NumberColumn("校准斜率", format="%.3f"),
                },
            )
            selected_group = st.selectbox(
                "损失分组",
                ["availability_year", "provisional_sic2", "company"],
                format_func=lambda value: {
                    "availability_year": "年份",
                    "provisional_sic2": "当前 SIC2（临时）",
                    "company": "公司",
                }[value],
                key="tournament_group",
            )
            subgroup = study.subgroup_diagnostics.loc[
                study.subgroup_diagnostics["model_id"].eq(selected_id)
                & study.subgroup_diagnostics["group_type"].eq(selected_group)
            ]
            st.dataframe(subgroup, hide_index=True, width="stretch")

    with representation_tab:
        st.subheader("预测胜出之后，再问 Innovation 与 Delta 有多不同")
        st.caption(
            "这些结果没有参与模型选择。它们说明状态条件预期改变了财务 surprise 的横截面排序。"
        )
        st.plotly_chart(
            _tournament_representation_figure(study.representation_diagnostics),
            width="stretch",
            config={"displayModeBar": False},
        )
        selected_representation = study.representation_diagnostics.loc[
            study.representation_diagnostics["model_id"].eq(selected_id)
        ].iloc[0]
        representation_cards = st.columns(4)
        representation_cards[0].metric(
            "Delta–Innovation 秩相关",
            f"{float(selected_representation['spearman_delta_innovation']):.3f}",
        )
        representation_cards[1].metric(
            "Adjustment / Delta 波动",
            f"{float(selected_representation['std_adjustment_to_delta_ratio']):.1%}",
        )
        representation_cards[2].metric(
            "中位秩移动",
            f"{float(selected_representation['median_abs_percentile_rank_movement']):.1%}",
        )
        representation_cards[3].metric(
            "跨五分位",
            f"{float(selected_representation['observations_changing_quintile']):.1%}",
        )
        st.dataframe(
            study.representation_diagnostics.assign(
                std_expectation_adjustment=lambda frame: (
                    frame["std_expectation_adjustment"] * 100.0
                ),
                std_adjustment_to_delta_ratio=lambda frame: (
                    frame["std_adjustment_to_delta_ratio"] * 100.0
                ),
                median_abs_percentile_rank_movement=lambda frame: (
                    frame["median_abs_percentile_rank_movement"] * 100.0
                ),
                observations_changing_quintile=lambda frame: (
                    frame["observations_changing_quintile"] * 100.0
                ),
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "std_expectation_adjustment": st.column_config.NumberColumn(
                    "Adjustment 波动", format="%.2f%%"
                ),
                "std_adjustment_to_delta_ratio": st.column_config.NumberColumn(
                    "Adjustment / Delta", format="%.1f%%"
                ),
                "median_abs_percentile_rank_movement": st.column_config.NumberColumn(
                    "中位秩移动", format="%.1f%%"
                ),
                "observations_changing_quintile": st.column_config.NumberColumn(
                    "跨五分位", format="%.1f%%"
                ),
            },
        )

    with audit_tab:
        st.subheader("五个候选的点时审计")
        st.dataframe(
            study.feature_clock_audit,
            hide_index=True,
            width="stretch",
        )
        st.code(
            "M0: GM(t-4)\n"
            "M1: frozen pooled OLS v1\n"
            "M2: t-1 state + t-4 seasonal state + company trajectory + prior revenue/industry\n"
            "M3: M2 + strictly earlier same-period peer information, ridge α=10\n"
            "M4: M3 z-scores + squares + five frozen interactions, ridge α=25",
            language=None,
        )
        st.error(
            "预测 Gate 通过不等于收益 Gate 通过。历史 security master、历史成分、历史行业归属、"
            "交易所日历、点时价格、公司行动和退市收益仍是硬性缺口。"
        )
        _download_csv(
            "下载 M0–M4 共同样本",
            study.common_sample,
            "forecast-tournament-v2-common-sample.csv",
        )
        _download_csv(
            "下载预测闸门结果",
            study.eligibility_gate,
            "forecast-tournament-v2-eligibility-gate.csv",
        )
        st.caption(f"研究版本：{report['forecast_tournament_version']}")

    st.caption("真实 SEC 财务预测竞赛 · 不构成投资建议或收益证据")


def _render_revision_robustness() -> None:
    """Render the walk-forward M2 revision-risk and safety-margin experiment."""
    try:
        study = load_revision_robustness()
    except (OSError, ValueError, json.JSONDecodeError):
        _render_public_revision_preview()
        return

    report = study.report
    confidence_options = study.confidence_calibration["confidence"].astype(float).tolist()
    selected_confidence = st.sidebar.select_slider(
        "修订风险置信度",
        options=confidence_options,
        value=float(report["primary_confidence"]),
        format_func=lambda value: f"{value:.1%}",
        help="置信度越高，缓冲越大，能够保留为方向稳健的信号越少。",
    )
    selected = study.confidence_calibration.loc[
        np.isclose(study.confidence_calibration["confidence"], selected_confidence)
    ].iloc[0]

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 04 · REVISION-AWARE DECISION ROBUSTNESS</div>
          <h1>公告当下，这个信号值得相信多少？</h1>
          <p>只用当时已经解决的历史修订，给原始 M2 Innovation 留出可审计的安全边际。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="bias-badge off">M2 原始公告信号</span>'
        '<span class="bias-badge off">严格 Walk-forward</span>'
        '<span class="bias-badge off">右删失处理</span>'
        '<span class="bias-badge on">无收益证据</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "这里的“变化”是后续比较财报对历史毛利率的重新描绘。它可能来自业务范围或列报口径，"
        "不等于错误或欺诈；经来源复核确认的 XBRL 抽取错误已先行纠正。安全边际衡量财务信号稳定性，"
        "不是买卖阈值。",
        icon="⚠️",
    )

    cards = st.columns(5)
    cards[0].metric("可形成决策", f"{int(selected['decision_events']):,}")
    cards[1].metric("已验证结果", f"{int(selected['resolved_evaluation_events']):,}")
    cards[2].metric(
        f"{selected_confidence:.1%} 中位安全边际",
        f"{float(selected['median_revision_at_risk']) * 100:.3f} pp",
    )
    cards[3].metric("实际覆盖率", f"{float(selected['empirical_coverage']):.1%}")
    cards[4].metric("方向稳健信号", f"{float(selected['robust_decision_share']):.1%}")

    tradeoff_tab, event_tab, tail_audit_tab, audit_tab = st.tabs(
        ["安全与可行动性的取舍", "事件级决策审计", "15 个尾部事件复核", "数据来源与方法边界"]
    )
    with tradeoff_tab:
        left, right = st.columns([1.08, 1.0], gap="large")
        with left:
            st.subheader("要求更安全，需要放弃多少信号？")
            st.caption(
                "柱表示历史修订绝对值的事件时点分位数；折线表示原始 Innovation "
                "扣除该缓冲后仍保留方向的比例。"
            )
            st.plotly_chart(
                _revision_confidence_tradeoff_figure(study.confidence_calibration),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.subheader("原始信号与后来信号")
            st.caption(
                "仅显示安全边际已可计算、且后来结果已经解决的 "
                f"{int(selected['resolved_evaluation_events']):,} 个事件。"
            )
            st.plotly_chart(
                _revision_initial_latest_figure(study.panel),
                width="stretch",
                config={"displayModeBar": False},
            )

        if selected_confidence <= 0.90:
            st.info(
                "80% 和 90% 缓冲的中位数为零，是因为大多数比较期数字没有变化；"
                "它不代表修订风险为零。少数变化形成了明显的厚尾。",
                icon="ℹ️",
            )
        elif selected_confidence == 0.95:
            median_buffer_pp = float(selected["median_revision_at_risk"]) * 100.0
            st.success(
                f"95% 主规格的中位缓冲为 {median_buffer_pp:.3f} pp，"
                f"走步覆盖率为 {float(selected['empirical_coverage']):.1%}。但仍有 "
                f"{int(selected['robust_direction_flips'])} 个被判为方向稳健的已解决事件最终翻转，"
                "所以这是概率缓冲，不是保证。",
                icon="✅",
            )
        else:
            st.info(
                f"提高到 {selected_confidence:.1%} 后，中位缓冲扩大到 "
                f"{float(selected['median_revision_at_risk']) * 100:.2f} pp，只有 "
                f"{float(selected['robust_decision_share']):.1%} 的信号仍保留方向。"
                "尾部保护的代价是大部分事件失去可行动性。",
                icon="ℹ️",
            )

        calibration = study.confidence_calibration.copy()
        for column in ("median_revision_at_risk", "mean_revision_at_risk"):
            calibration[column] *= 100.0
        st.subheader("全部冻结置信度规格")
        st.dataframe(
            calibration,
            hide_index=True,
            width="stretch",
            column_config={
                "confidence": st.column_config.NumberColumn("置信度", format="%.1f%%"),
                "tail_probability": st.column_config.NumberColumn("尾部概率", format="%.1f%%"),
                "decision_events": "可形成决策",
                "resolved_evaluation_events": "已验证结果",
                "median_revision_at_risk": st.column_config.NumberColumn(
                    "中位缓冲（pp）", format="%.3f"
                ),
                "mean_revision_at_risk": st.column_config.NumberColumn(
                    "平均缓冲（pp）", format="%.3f"
                ),
                "empirical_coverage": st.column_config.NumberColumn("实际覆盖", format="%.1f%%"),
                "robust_decision_share": st.column_config.NumberColumn(
                    "方向稳健比例", format="%.1f%%"
                ),
                "robust_direction_flips": "稳健后仍翻转",
                "false_robust_rate": st.column_config.NumberColumn(
                    "稳健判断失败率", format="%.2f%%"
                ),
            },
        )

    with event_tab:
        buffer_column = f"revision_at_risk_{round(selected_confidence * 1_000):03d}"
        decisions = study.primary_decisions.copy()
        decisions["selected_robust_direction"] = (
            decisions["M2_innovation_raw"].abs() > decisions[buffer_column]
        )
        decisions["selected_robust_innovation"] = np.sign(decisions["M2_innovation_raw"]) * (
            decisions["M2_innovation_raw"].abs() - decisions[buffer_column]
        ).clip(lower=0.0)
        tickers = sorted(decisions["ticker"].dropna().astype(str).unique())
        selected_tickers = st.multiselect(
            "筛选公司",
            tickers,
            default=[],
            placeholder="留空显示全部公司",
            key="revision_robustness_tickers",
        )
        if selected_tickers:
            decisions = decisions.loc[decisions["ticker"].isin(selected_tickers)]
        event_columns = [
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "actual_available_at",
            "M2_innovation_raw",
            buffer_column,
            "selected_robust_innovation",
            "sign_survival_probability",
            "selected_robust_direction",
            "revision_history_observations",
            "revision_history_changed_observations",
            "revision_outcome_status",
            "revision_size",
            "latest_innovation_hindsight",
            "realized_direction_flip",
            "filing_accession",
        ]
        st.dataframe(
            decisions[event_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker",
                "actual_available_at": st.column_config.DatetimeColumn(
                    "公告可得时间", format="YYYY-MM-DD HH:mm:ss Z"
                ),
                "M2_innovation_raw": st.column_config.NumberColumn(
                    "原始 Innovation", format="%+.3f%%"
                ),
                buffer_column: st.column_config.NumberColumn("安全边际", format="%.3f%%"),
                "selected_robust_innovation": st.column_config.NumberColumn(
                    "扣除边际后", format="%+.3f%%"
                ),
                "sign_survival_probability": st.column_config.NumberColumn(
                    "历史方向存活率", format="%.1f%%"
                ),
                "selected_robust_direction": "方向稳健",
                "revision_size": st.column_config.NumberColumn("后来数值变化", format="%+.3f%%"),
                "latest_innovation_hindsight": st.column_config.NumberColumn(
                    "后来 Innovation", format="%+.3f%%"
                ),
                "realized_direction_flip": "后来翻转",
            },
        )
        _download_csv(
            "下载事件级安全边际 CSV",
            decisions,
            f"revision-robustness-{selected_confidence:.3f}.csv",
        )

        st.subheader("厚尾事件：必须逐份财报复核")
        st.caption(
            "以下是后来数值变化最大的事件。它们可能来自业务口径、重分类、并购或事实选择问题，"
            "不能未经核验就解释成经济性重述。"
        )
        tail_columns = [
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "M2_innovation_raw",
            "revision_size",
            "revision_size_unreviewed",
            "latest_innovation_hindsight",
            "revision_at_risk_950",
            "exceeded_primary_buffer",
            "realized_direction_flip",
            "revision_source_type",
            "tail_audit_classification",
            "tail_audit_action",
            "filing_accession",
            "latest_filing_accession",
        ]
        st.dataframe(
            study.tail_events[tail_columns].head(30),
            hide_index=True,
            width="stretch",
        )

    with tail_audit_tab:
        audit_report = report["tail_audit"]
        st.subheader("把数据错误与真正的历史口径重绘分开")
        st.markdown(
            "我们逐份核对了原始财报和后续比较财报，并把每个判断绑定到 SEC URL、"
            "accession 与本地原始文件 SHA-256。这里的‘重绘’不等于财务差错或欺诈。"
        )
        audit_cards = st.columns(5)
        audit_cards[0].metric("逐份复核", f"{int(audit_report['reviewed_tail_events'])}")
        audit_cards[1].metric(
            "XBRL 抽取错误", f"{int(audit_report['semantic_collision_corrections'])}"
        )
        audit_cards[2].metric("业务范围重绘", f"{int(audit_report['accounting_scope_redraws'])}")
        audit_cards[3].metric(
            "列报重分类", f"{int(audit_report['accounting_presentation_redraws'])}"
        )
        audit_cards[4].metric(
            "确认的经济差错", f"{int(audit_report['economic_error_corrections'])}"
        )
        st.success(
            "复核后，4 个 Companyfacts 语义碰撞已按原始合并报表数值纠正为零变化。"
            "剩余 11 个是业务范围或列报口径的后来重绘；本轮没有确认到经济事实被纠错的案例。",
            icon="✅",
        )

        summary = study.tail_classification_summary.copy()
        summary["max_unreviewed_revision"] *= 100.0
        audit_left, audit_right = st.columns([0.9, 1.45], gap="large")
        with audit_left:
            st.plotly_chart(
                _revision_tail_audit_figure(summary),
                width="stretch",
                config={"displayModeBar": False},
            )
        with audit_right:
            st.dataframe(
                summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "audit_family": "风险层",
                    "audit_classification": "复核分类",
                    "reviewed_events": "事件数",
                    "max_unreviewed_revision": st.column_config.NumberColumn(
                        "最大表观变化（pp）", format="%.3f"
                    ),
                    "direction_flips_after_audit": "复核后方向翻转",
                    "event_time_structural_flags": "事件当时已有结构变化线索",
                },
            )

        st.subheader("第二阶段过滤器：候选规则没有通过验证")
        gate = study.event_time_gate.iloc[0]
        st.markdown(
            "候选规则只看原始公告当时已经公开的分拆或结构变化线索。它标记了 "
            f"**{int(gate['redraws_flagged'])}/{int(gate['reviewed_published_history_redraws'])}** "
            "个历史口径重绘，却没有抓到复核后仍存在的 "
            f"**{int(gate['direction_flips_after_audit'])}** 个方向翻转。"
        )
        st.error(
            "因此当前不能把它作为决策过滤器上线：召回率不足，而且尚无广泛的非尾部对照样本可估计误报率。"
            "现阶段保留第一阶段的走步安全边际；第二阶段只保留为待检验研究假设。",
            icon="🛑",
        )

        reviewed_columns = [
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "unreviewed_revision_size",
            "audited_revision_size",
            "audit_family",
            "audit_classification",
            "mechanism",
            "event_time_structural_context",
            "direction_flip_after_audit",
            "filing_accession",
            "latest_filing_accession",
            "original_source_url",
            "later_source_url",
        ]
        st.dataframe(
            study.reviewed_tail_events[reviewed_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "unreviewed_revision_size": st.column_config.NumberColumn(
                    "复核前表观变化", format="%+.3f%%"
                ),
                "audited_revision_size": st.column_config.NumberColumn(
                    "复核后变化", format="%+.3f%%"
                ),
                "original_source_url": st.column_config.LinkColumn("原始 SEC 财报"),
                "later_source_url": st.column_config.LinkColumn("后续 SEC 财报"),
            },
        )
        _download_csv(
            "下载 15 个事件的来源审计 CSV",
            study.reviewed_tail_events,
            "revision-tail-source-audit-v1.csv",
        )

    with audit_tab:
        st.subheader("为什么 88 个零变化不能进入历史分布？")
        st.markdown(
            "`original_same_accession` 只表示当前快照里没有找到后续版本。公告当时并不知道未来"
            "不会出现新版本，所以这些事件标记为右删失，而不是把零修订提前泄漏给模型。"
        )
        st.dataframe(study.source_audit, hide_index=True, width="stretch")
        st.subheader("走步约束")
        st.code(
            "revision_outcome_available_at < target_actual_available_at\n"
            "history >= 100 resolved outcomes and >= 10 changed outcomes\n"
            "RevisionAtRisk(c) = higher_quantile_c(abs(prior resolved revisions))\n"
            "RobustInnovation = sign(Initial) × max(abs(Initial) - RevisionAtRisk, 0)",
            language=None,
        )
        st.markdown(
            "- M2 预测冻结，不用后来值重新训练。\n"
            "- v1 使用 pooled 毛利率修订分布；当前 SIC 是 2026 快照，不做行业条件化。\n"
            "- 95% 是主规格；其他置信度只做预先固定的敏感性展示。\n"
            "- 当前只检验会计信号方向，不检验收益、仓位或交易安全边际。\n"
            f"- 正式 amendment 事件只有 **{int(report['formal_amendment_events'])}** 个。"
        )
        st.dataframe(study.diagnostics_by_year, hide_index=True, width="stretch")
        _download_csv(
            "下载完整修订稳健性面板",
            study.panel,
            "revision-robustness-v1-panel.csv",
        )
        st.caption(f"研究版本：{report['revision_robustness_version']}")

    st.caption("真实 SEC 数值修订稳健性研究 · 不构成投资建议或收益证据")


def _render_representation_study() -> None:
    """Render the first multi-company real-data representation experiment."""
    try:
        study = load_representation_study()
    except (OSError, ValueError, json.JSONDecodeError):
        _render_public_representation_preview()
        return
    report = study.report
    diagnostics = report["forecast_diagnostics"]
    distinctness = report["representation_distinctness"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 02 · REAL PIT FILINGS</div>
          <h1>Level、Delta，还是 Innovation？</h1>
          <p>82 家核心资产；同一批真实 SEC 事件；预测、误差和标准化全部遵守事件前信息边界。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="bias-badge off">真实 SEC 数据</span>'
        '<span class="bias-badge off">Expanding OOS</span>'
        '<span class="bias-badge off">共同表示样本</span>'
        '<span class="bias-badge on">尚未接入收益</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "这是财务表示与预测诊断，不是收益研究。尚无点时价格、交易所日历、退市收益和交易成本，"
        "因此不能回答哪一种表示能预测市场收益。",
        icon="⚠️",
    )
    cards = st.columns(5)
    cards[0].metric("核心资产", f"{report['core_companies']:,} 家")
    cards[1].metric("严格 Delta", f"{report['delta_events']:,}")
    cards[2].metric("OOS Innovation", f"{report['raw_innovation_events']:,}")
    cards[3].metric("共同表示样本", f"{report['common_representation_events']:,}")
    cards[4].metric(
        "Innovation 覆盖",
        f"{report['companies_with_minimum_innovation_history']:,} 家",
    )

    result_tab, event_tab, method_tab = st.tabs(
        ["表示与预测诊断", "事件、覆盖与修订", "方法与证据边界"]
    )
    with result_tab:
        left, right = st.columns([1.18, 1.0], gap="large")
        with left:
            st.subheader("从 Level 到 Innovation 的样本漏斗")
            st.caption("前两年用于构造同比与滞后特征；随后依次满足训练和 OOS sigma 热身期。")
            st.plotly_chart(
                _representation_coverage_figure(study.coverage),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.subheader("OOS 预测误差：模型与去年同季基准")
            st.caption("RMSE 越低越好；2026 只有少量事件，不应用于年度比较。")
            st.plotly_chart(
                _forecast_diagnostics_figure(study.forecast_diagnostics),
                width="stretch",
                config={"displayModeBar": False},
            )
        st.info(
            _representation_summary(diagnostics, distinctness),
            icon="🔎",
        )
        st.subheader("Innovation 到底有没有重新排序？")
        st.code(
            "Innovation(t) = Delta(t) + ExpectationAdjustment(t)\n"
            "ExpectationAdjustment(t) = GM(t-4) - Forecast(t-)",
            language=None,
        )
        pooled_adjustment = study.expectation_adjustment_diagnostics.loc[
            study.expectation_adjustment_diagnostics["rank_scope"].eq("pooled_common_sample")
        ].iloc[0]
        adjustment_cards = st.columns(4)
        adjustment_cards[0].metric(
            "Adjustment 波动",
            f"{float(pooled_adjustment['std_expectation_adjustment']):.2%}",
            help="Expectation Adjustment 的样本标准差。",
        )
        adjustment_cards[1].metric(
            "相对 Delta 波动",
            f"{float(pooled_adjustment['std_adjustment_to_delta_ratio']):.1%}",
        )
        adjustment_cards[2].metric(
            "中位秩移动",
            f"{float(pooled_adjustment['median_abs_percentile_rank_movement']):.1%}",
        )
        adjustment_cards[3].metric(
            "跨五分位边界",
            f"{float(pooled_adjustment['observations_changing_quintile']):.1%}",
        )
        st.caption(
            "共同样本的 Top quintile 重合 "
            f"{float(pooled_adjustment['top_quintile_overlap']):.1%}，Bottom quintile 重合 "
            f"{float(pooled_adjustment['bottom_quintile_overlap']):.1%}；"
            f"只有 {float(pooled_adjustment['observations_moving_two_or_more_quintiles']):.1%} "
            "移动两个或更多分位。因此存在重新排序，但主要是相邻分位调整。"
        )
        st.plotly_chart(
            _rank_movement_figure(study.rank_movements),
            width="stretch",
            config={"displayModeBar": False},
        )

        st.subheader("预测技能是否稳定？")
        st.caption("MAE 是 v2 预注册的主指标；RMSE 和 OOS R² 是次要指标。零线以上才优于去年同季。")
        st.plotly_chart(
            _forecast_skill_figure(study.forecast_diagnostics),
            width="stretch",
            config={"displayModeBar": False},
        )
        bootstrap = study.forecast_bootstrap_intervals.copy()
        st.dataframe(
            bootstrap[
                ["metric", "point_estimate", "ci_lower_2_5", "ci_upper_97_5", "cluster_dimensions"]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "metric": "预测技能指标",
                "point_estimate": st.column_config.NumberColumn("点估计", format="%+.2f%%"),
                "ci_lower_2_5": st.column_config.NumberColumn("95% 下界", format="%+.2f%%"),
                "ci_upper_97_5": st.column_config.NumberColumn("95% 上界", format="%+.2f%%"),
                "cluster_dimensions": "聚类维度",
            },
        )
        with st.expander("查看预测损失分组诊断"):
            group_labels = {
                "availability_year": "年份",
                "provisional_sic2": "当前 SIC2（临时）",
                "company": "公司",
                "prior_gm_volatility": "公告前毛利率波动",
                "revenue_size_proxy": "收入规模代理",
                "filing_lag": "期末至公告时间",
            }
            selected_group = st.selectbox(
                "分组维度",
                list(group_labels),
                format_func=lambda value: group_labels[value],
            )
            grouped = study.forecast_diagnostics_by_group.loc[
                study.forecast_diagnostics_by_group["group_type"].eq(selected_group)
            ].copy()
            st.dataframe(
                grouped,
                hide_index=True,
                width="stretch",
                column_config={
                    "group_value": "分组",
                    "forecast_events": "事件数",
                    "mae_improvement_vs_naive": st.column_config.NumberColumn(
                        "MAE 技能", format="%+.1%%"
                    ),
                    "rmse_improvement_vs_naive": st.column_config.NumberColumn(
                        "RMSE 技能", format="%+.1%%"
                    ),
                    "oos_r2_vs_naive": st.column_config.NumberColumn("OOS R²", format="%+.1%%"),
                },
            )

        st.subheader("四种表示在共同样本中的秩相关")
        st.caption(
            "这里衡量表示是否彼此独立，不是与收益的 IC。高相关意味着当前 Innovation "
            "仍主要是 Delta 的重表达。"
        )
        st.plotly_chart(
            _representation_correlation_figure(study.correlations),
            width="stretch",
            config={"displayModeBar": False},
        )
        distribution = study.distributions.copy()
        distribution["mean"] *= 100.0
        distribution["std"] *= 100.0
        distribution["p05"] *= 100.0
        distribution["median"] *= 100.0
        distribution["p95"] *= 100.0
        st.dataframe(
            distribution[["representation", "observations", "mean", "std", "p05", "median", "p95"]],
            hide_index=True,
            width="stretch",
            column_config={
                "representation": "表示",
                "observations": "共同样本",
                "mean": st.column_config.NumberColumn("均值", format="%+.2f"),
                "std": st.column_config.NumberColumn("标准差", format="%.2f"),
                "p05": st.column_config.NumberColumn("P05", format="%+.2f"),
                "median": st.column_config.NumberColumn("中位数", format="%+.2f"),
                "p95": st.column_config.NumberColumn("P95", format="%+.2f"),
            },
        )
        sigma = study.sigma_diagnostics.iloc[0]
        st.info(
            f"Raw 与 Std Innovation 的 Spearman 为 "
            f"{float(sigma['spearman_raw_standardized']):.3f}。当前 sigma 来自全局 expanding "
            "OOS 误差历史，不按公司、行业或事件条件化；因此它能表达 surprise 的历史尺度，"
            "但几乎不提供新的横截面排序。",
            icon="ℹ️",
        )

    with event_tab:
        coverage_tab, panel_tab, revision_tab = st.tabs(
            ["公司覆盖", "共同样本事件", "PIT 与后来值"]
        )
        with coverage_tab:
            coverage = study.company_coverage.copy()
            st.dataframe(
                coverage,
                hide_index=True,
                width="stretch",
                column_config={
                    "company_id": "CIK",
                    "ticker": "Ticker",
                    "company_name": "公司",
                    "standardized_innovation_events": "标准化 Innovation",
                    "innovation_coverage_status": "覆盖状态",
                },
            )
            _download_csv(
                "下载公司覆盖 CSV",
                coverage,
                "level-delta-innovation-company-coverage.csv",
            )
        with panel_tab:
            tickers = sorted(study.common_sample["ticker"].dropna().astype(str).unique())
            selected = st.multiselect("选择公司", tickers, default=tickers[:3])
            display = study.common_sample.loc[study.common_sample["ticker"].isin(selected)].copy()
            columns = [
                "ticker",
                "fiscal_year",
                "fiscal_quarter",
                "actual_available_at",
                "gm_level",
                "gm_delta_yoy",
                "gm_forecast",
                "gm_innovation_raw",
                "gm_innovation_std",
                "forecast_training_observations",
                "forecast_sigma_observations",
                "filing_accession",
            ]
            st.dataframe(
                display[columns],
                hide_index=True,
                width="stretch",
                column_config={
                    "actual_available_at": st.column_config.DatetimeColumn(
                        "实际可得时间", format="YYYY-MM-DD HH:mm:ss Z"
                    ),
                    "gm_level": st.column_config.NumberColumn("Level", format="%.2f%%"),
                    "gm_delta_yoy": st.column_config.NumberColumn("Delta", format="%+.2f%%"),
                    "gm_forecast": st.column_config.NumberColumn("Forecast", format="%.2f%%"),
                    "gm_innovation_raw": st.column_config.NumberColumn(
                        "Raw Innovation", format="%+.2f%%"
                    ),
                    "gm_innovation_std": st.column_config.NumberColumn(
                        "Std Innovation", format="%+.2f"
                    ),
                },
            )
            _download_csv(
                "下载共同表示样本 CSV",
                study.common_sample,
                "level-delta-innovation-common-sample.csv",
            )
        with revision_tab:
            st.caption(
                "后来值只用于 hindsight 诊断，不参与 PIT 预测。数值变化不等同于 SEC "
                "修正申报或会计错误。"
            )
            revision_columns = [
                "ticker",
                "fiscal_year",
                "fiscal_quarter",
                "gm_level",
                "gm_level_latest",
                "gm_delta_yoy",
                "gm_delta_yoy_latest",
                "gm_innovation_raw",
                "gm_innovation_raw_latest_hindsight",
                "innovation_revision_effect",
                "filing_accession",
            ]
            st.dataframe(
                study.revisions[revision_columns].head(100),
                hide_index=True,
                width="stretch",
            )
            _download_csv(
                "下载修订诊断 CSV",
                study.revisions,
                "level-delta-innovation-revision-diagnostics.csv",
            )

    with method_tab:
        st.subheader("固定的第一版预测式")
        st.code(
            "GM(i,t) = β0 + β1·GM(i,t-4) + β2·DeltaGM(i,t-4)\n"
            "          + β3·RevenueGrowth(i,t-1) + β4·IndustryGM(i,t-) + ε(i,t)",
            language=None,
        )
        st.markdown(
            f"""
- 训练方式：expanding pooled OLS；至少
  **{report["config"]["forecast"]["min_train_observations"]}** 个更早事件。
- 标准化分母：至少 **{report["config"]["forecast"]["min_oos_errors"]}** 个严格更早的
  OOS 误差。
- 行业特征：目标公告前已披露的同行最新毛利率中位数；至少
  **{report["config"]["min_industry_peers"]}** 家。
- 同时公告：按同一时间批次预测，不能互相使用实际值。
- 当前行业归属使用 2026 SIC 快照，标记为 provisional；正式收益研究前需接历史行业分类。
- 当前股票池来自 2026 Global 500 快照，存在明确的事后存续与选择条件；不能用于历史 alpha 结论。
- v2 主选择指标已冻结为 MAE；RMSE 和 OOS R² 为次要指标。Delta–Innovation distinctness
  只报告，不参与选模。
- 研究版本：`{report["representation_research_version"]}`。
"""
        )
        gate = report["v2_eligibility_gate"]
        positive_years = report["forecast_stability"]["years_with_positive_primary_skill"]
        eligible_years = report["forecast_stability"]["eligible_years_minimum_25_events"]
        st.error(
            "当前 v1 不具备下游收益测试资格：总体 MAE 未优于去年同季，且至少25个事件的"
            f"年度中，主指标改善年份为 {positive_years} / {eligible_years}。"
            f"状态：{gate['status']}。"
        )
        _download_csv(
            "下载股票池与行业时点审计",
            study.universe_timepoint_audit,
            "universe-and-classification-timepoint-audit.csv",
        )
        st.markdown(
            "**仍未实现**：点时价格、正式交易日历、历史 security master、退市收益、异常收益窗口。"
        )
    st.caption("真实 SEC 财务表示研究 · 不构成投资建议或收益证据")


def _render_event_return_diagnostics() -> None:
    """Render the first real-price event-return diagnostic."""
    try:
        study = load_event_return_diagnostics()
        expansion = load_longitudinal_expansion()
    except (OSError, ValueError, json.JSONDecodeError):
        _render_public_effectiveness_preview()
        return
    report = study.report
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">REAL EVENT RETURNS · PILOT DIAGNOSTIC</div>
          <h1>真实事件收益：先审计，再谈信号</h1>
          <p>同一批 SEC filing，公告后第二个交易日开盘进入，持有 20 个正式交易日，
          与同期 SPY 总收益比较。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "这是 28 个事件的行情与偏差诊断，不是策略回测。事件并非全市场代表样本，"
        "窗口彼此重叠，因此不年化、不计算夏普，也不宣称财报信号有效。",
        icon="⚠️",
    )
    all_sample = study.survivorship_counterfactual.loc[
        study.survivorship_counterfactual["sample_id"].eq("all_events")
    ].iloc[0]
    active_sample = study.survivorship_counterfactual.loc[
        study.survivorship_counterfactual["sample_id"].eq("active_at_outcome_cutoff_only")
    ].iloc[0]
    first, second, third, fourth = st.columns(4)
    first.metric("全样本平均超额", f"{float(all_sample['mean_market_excess_return']):+.2%}")
    second.metric(
        "只留后来仍活跃",
        f"{float(active_sample['mean_market_excess_return']):+.2%}",
    )
    third.metric(
        "幸存者偏差抬高均值",
        f"{float(report['survivors_only_mean_inflation_pp']):+.2f} pp",
    )
    fourth.metric(
        "跑赢 SPY 的事件",
        f"{int(report['positive_market_excess_events'])} / {int(report['events'])}",
    )

    left, right = st.columns([1.58, 1.0], gap="large")
    with left:
        st.subheader("每个事件的 20 日市场超额收益")
        st.caption("红色只表示该证券在正式退出日以后又退市；窗口内没有退市事件。")
        st.plotly_chart(
            _event_excess_return_figure(study.events),
            width="stretch",
        )
    with right:
        st.subheader("故意删除后来退市公司，会发生什么")
        st.caption("这是使用未来状态的偏差反事实，不是推荐的样本筛选。")
        st.plotly_chart(
            _survivorship_counterfactual_figure(study.survivorship_counterfactual),
            width="stretch",
        )
        st.caption(
            "均值被抬高 "
            f"{float(report['survivors_only_mean_inflation_pp']):.2f} 个百分点；"
            "中位数只抬高 "
            f"{float(report['survivors_only_median_inflation_pp']):.2f} 个百分点。"
            "说明这一轮主要由少数尾部事件驱动。"
        )

    left, right = st.columns([1.18, 1.0], gap="large")
    with left:
        st.subheader("公司收益与同期 SPY")
        st.caption("虚线是公司与 SPY 收益相同；点离虚线越远，个股特异反应越强。")
        st.plotly_chart(_issuer_market_scatter(study.events), width="stretch")
    with right:
        st.subheader("当前研究闸门")
        st.caption("通过数据完整性不等于通过信号有效性。")
        st.plotly_chart(_event_return_gate_figure(study.gates), width="stretch")

    st.subheader("Level / Delta / Innovation × 收益：共同样本闸门")
    total_events = int(report["events"])
    complete_events = int(report["representation_complete_events"])
    issuer_overlap = int(report["representation_issuer_overlap"])
    accession_overlap = int(report["representation_accession_overlap"])
    level_events = int(report.get("representation_level_events", complete_events))
    delta_events = int(report.get("representation_delta_events", complete_events))
    innovation_events = int(
        report.get("representation_standardized_innovation_events", complete_events)
    )
    target, level, delta, innovation = st.columns(4)
    target.metric("目标 filing 事件", f"{total_events}")
    level.metric("Level 可用", f"{level_events} / {total_events}")
    delta.metric("Delta 可用", f"{delta_events} / {total_events}")
    innovation.metric("Innovation 可用", f"{innovation_events} / {total_events}")
    if issuer_overlap == total_events and accession_overlap == total_events:
        coverage_requirement = int(
            np.ceil(total_events * float(report["config"]["minimum_feature_coverage_ratio"]))
        )
        count_requirement = int(report["config"]["minimum_feature_events_for_ic"])
        st.info(
            f"{total_events} / {total_events} 个事件已经按同一 filing accession 与 "
            "accepted_at 精确连接。"
            f"严格完整样本为 {complete_events} / {total_events}；预先承诺的 IC 闸门同时要求"
            f"至少 {count_requirement} 个事件和 80% 覆盖（本轮即 {coverage_requirement} 个）。"
            "当前未过门，因此不计算 IC、不显示信号—收益散点，也不对信号有效性下结论。"
            "缺失事件仍保留在审计表中，没有填零或从母样本删除。",
            icon="ℹ️",
        )
    else:
        st.info(
            f"当前只有 {issuer_overlap} / {total_events} 家公司、{accession_overlap} / "
            f"{total_events} 个 filing accession 能严格连接，完整特征覆盖为 "
            f"{complete_events} / {total_events}。共同样本门槛尚未通过，因此关闭 IC。",
            icon="ℹ️",
        )
    status = (
        study.feature_join_audit.groupby("feature_join_status", as_index=False)
        .size()
        .rename(columns={"feature_join_status": "连接状态", "size": "事件数"})
    )
    st.dataframe(status, hide_index=True, width="stretch")

    expansion_report = expansion.report
    st.subheader("下一轮：纵向扩样已经冻结，尚未使用收益")
    representative, issuers, dense, coverage = st.columns(4)
    representative.metric("代表性事件母框", int(expansion_report["representative_events"]))
    issuers.metric("预先合格发行人", int(expansion_report["feature_dense_issuers"]))
    dense.metric(
        "特征密集事件",
        f"{int(expansion_report['feature_dense_complete_events'])} / "
        f"{int(expansion_report['feature_dense_events'])}",
    )
    coverage.metric(
        "预计完整覆盖",
        f"{float(expansion_report['feature_dense_complete_coverage']):.1%}",
    )
    st.info(
        "发行人资格只使用 2024-10-30 之前的 PIT 历史：至少 8 个 Level、4 个 Delta、"
        "4 个既往标准化 OOS Innovation；之后的全部原始 10-Q/10-K 都进入事件框，"
        "没有查看当期特征或收益。预计 49 / 58 个完整事件可越过样本数和80%覆盖门，"
        "但这只是特征准备度，行情和新窗口证券身份尚未抓取，所以仍不计算 IC。"
    )
    direct = int(expansion_report["planned_direct_price_issuers"])
    identity_first = int(expansion_report["planned_identity_first_issuers"])
    planned_batches = expansion.market_collection_targets.loc[
        expansion.market_collection_targets["planned_next_batch"]
    ].copy()
    identity_names = "、".join(
        planned_batches.loc[
            planned_batches["identity_discovery_required"], "selected_ticker"
        ].astype(str)
    )
    st.caption(
        f"下一批共{len(planned_batches)}家：{direct}家可直接进入整段行情采集；"
        f"{identity_first}家（{identity_names}）必须先按历史日期重新确认 ticker/FIGI，"
        "禁止用后来 ticker 回填。"
    )
    with st.expander(f"查看{len(planned_batches)}家纵向行情采集批次"):
        batches = planned_batches[
            [
                "selected_ticker",
                "cik",
                "event_count",
                "complete_feature_events",
                "market_window_start",
                "market_window_end",
                "historical_ticker_query_ready",
                "identity_discovery_required",
            ]
        ].copy()
        batches = batches.rename(
            columns={
                "selected_ticker": "当前候选Ticker",
                "cik": "CIK",
                "event_count": "事件数",
                "complete_feature_events": "完整特征事件",
                "market_window_start": "行情起点",
                "market_window_end": "行情终点",
                "historical_ticker_query_ready": "历史Ticker已验证",
                "identity_discovery_required": "需先身份发现",
            }
        )
        st.dataframe(batches, hide_index=True, width="stretch")

    with st.expander("查看 28 个事件的可追溯明细"):
        detail = study.events[
            [
                "ticker",
                "cik",
                "accession_number",
                "form",
                "accepted_at",
                "entry_at",
                "exit_at",
                "issuer_total_return",
                "market_total_return",
                "market_excess_return_arithmetic",
                "later_delisted_after_formal_exit",
            ]
        ].copy()
        detail = detail.rename(
            columns={
                "ticker": "Ticker",
                "cik": "CIK",
                "accession_number": "Accession",
                "form": "Form",
                "accepted_at": "SEC accepted_at",
                "entry_at": "入场时点",
                "exit_at": "退出时点",
                "issuer_total_return": "公司总收益",
                "market_total_return": "SPY 总收益",
                "market_excess_return_arithmetic": "市场超额收益",
                "later_delisted_after_formal_exit": "退出后后来退市",
            }
        )
        st.dataframe(
            detail,
            hide_index=True,
            width="stretch",
            column_config={
                "公司总收益": st.column_config.NumberColumn(format="%.2f%%"),
                "SPY 总收益": st.column_config.NumberColumn(format="%.2f%%"),
                "市场超额收益": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        _download_csv(
            "下载真实事件收益诊断",
            study.events,
            "event-return-diagnostics-v1.csv",
        )
    st.caption(
        f"研究版本：{report['dataset_version']} · 真实 SEC + Massive/Polygon + SPY · "
        "不构成投资建议或信号有效性证据"
    )


def _event_excess_return_figure(events: pd.DataFrame) -> go.Figure:
    """Rank all pilot events by arithmetic market-excess return."""
    frame = events.sort_values("market_excess_return_arithmetic", kind="mergesort").copy()
    later_delisted = frame["later_delisted_after_formal_exit"].astype(bool)
    colors = np.where(later_delisted, COLORS["red"], COLORS["teal"])
    customdata = np.column_stack(
        [
            frame["form"],
            frame["issuer_total_return"],
            frame["market_total_return"],
            np.where(later_delisted, "正式退出后后来退市", "结果截止日仍活跃"),
        ]
    )
    figure = go.Figure(
        go.Bar(
            x=frame["ticker"],
            y=frame["market_excess_return_arithmetic"],
            marker={"color": colors, "line": {"width": 0}},
            customdata=customdata,
            hovertemplate=(
                "%{x} · %{customdata[0]}<br>市场超额 %{y:+.2%}"
                "<br>公司总收益 %{customdata[1]:+.2%}"
                "<br>SPY 总收益 %{customdata[2]:+.2%}"
                "<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0.0, line={"color": COLORS["ink"], "width": 1})
    figure.update_layout(**_chart_layout(height=470, y_title="20 日市场超额收益"))
    figure.update_yaxes(tickformat="+.0%")
    figure.update_xaxes(title="Ticker（从最差到最好）", tickangle=-55)
    return figure


def _survivorship_counterfactual_figure(counterfactual: pd.DataFrame) -> go.Figure:
    """Compare the unbiased sample with a future-status-filtered sample."""
    order = [
        "all_events",
        "active_at_outcome_cutoff_only",
        "later_delisted_after_exit",
    ]
    labels = {
        "all_events": "全部事件",
        "active_at_outcome_cutoff_only": "只留后来仍活跃",
        "later_delisted_after_exit": "退出后后来退市",
    }
    frame = counterfactual.set_index("sample_id").loc[order].reset_index()
    values = frame["mean_market_excess_return"].astype(float)
    colors = [COLORS["slate"], COLORS["orange"], COLORS["red"]]
    figure = go.Figure(
        go.Bar(
            x=[labels[value] for value in frame["sample_id"]],
            y=values,
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:+.2%}" for value in values],
            textposition="outside",
            customdata=frame[["events", "median_market_excess_return"]],
            hovertemplate=(
                "%{x}<br>事件 %{customdata[0]}<br>平均超额 %{y:+.2%}"
                "<br>中位超额 %{customdata[1]:+.2%}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0.0, line={"color": COLORS["ink"], "width": 1})
    figure.update_layout(**_chart_layout(height=340, y_title="平均市场超额收益"))
    figure.update_yaxes(tickformat="+.0%")
    return figure


def _issuer_market_scatter(events: pd.DataFrame) -> go.Figure:
    """Compare each issuer return with its same-window SPY return."""
    frame = events.copy()
    finite = frame[["market_total_return", "issuer_total_return"]].astype(float)
    lower = float(finite.min().min())
    upper = float(finite.max().max())
    padding = max((upper - lower) * 0.06, 0.01)
    figure = go.Figure()
    for flag, label, color, symbol in (
        (False, "结果截止日仍活跃", COLORS["teal"], "circle"),
        (True, "正式退出后后来退市", COLORS["red"], "diamond"),
    ):
        group = frame.loc[frame["later_delisted_after_formal_exit"].astype(bool).eq(flag)]
        figure.add_trace(
            go.Scatter(
                x=group["market_total_return"],
                y=group["issuer_total_return"],
                mode="markers",
                name=label,
                marker={"color": color, "size": 10, "symbol": symbol},
                customdata=np.column_stack(
                    [group["ticker"], group["form"], group["market_excess_return_arithmetic"]]
                ),
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[1]}"
                    "<br>SPY %{x:+.2%}<br>公司 %{y:+.2%}"
                    "<br>市场超额 %{customdata[2]:+.2%}<extra></extra>"
                ),
            )
        )
    figure.add_shape(
        type="line",
        x0=lower - padding,
        y0=lower - padding,
        x1=upper + padding,
        y1=upper + padding,
        line={"color": COLORS["slate"], "width": 1, "dash": "dot"},
    )
    figure.update_layout(**_chart_layout(height=430, y_title="公司 20 日总收益"))
    figure.update_xaxes(
        title="同期 SPY 总收益",
        tickformat="+.0%",
        range=[lower - padding, upper + padding],
    )
    figure.update_yaxes(tickformat="+.0%", range=[lower - padding, upper + padding])
    return figure


def _event_return_gate_figure(gates: pd.DataFrame) -> go.Figure:
    """Plot completed and blocked data gates without implying model quality."""
    labels = {
        "issuer_return_coverage": "公司收益",
        "market_excess_coverage": "SPY 超额",
        "delisting_outcome_coverage": "退市结果",
        "sector_benchmark_coverage": "行业基准",
        "ldi_complete_case_coverage": "L/D/I 同事件",
        "ldi_minimum_event_count": "L/D/I 最低样本",
    }
    frame = gates.copy()
    frame["label"] = frame["gate"].map(labels).fillna(frame["gate"])
    frame["coverage_percent"] = 100.0 * frame["coverage"].astype(float)
    frame["display"] = frame.apply(
        lambda row: f"{int(row['numerator'])}/{int(row['denominator'])}", axis="columns"
    )
    colors = np.where(frame["passed"].astype(bool), COLORS["teal"], COLORS["red"])
    figure = go.Figure(
        go.Bar(
            x=frame["coverage_percent"],
            y=frame["label"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=frame["display"],
            textposition="outside",
            customdata=frame[["meaning", "passed"]],
            hovertemplate=(
                "%{y}<br>%{text} · %{x:.0f}%<br>%{customdata[0]}"
                "<br>通过：%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(**_chart_layout(height=430, y_title=""), showlegend=False)
    figure.update_xaxes(title="覆盖率", ticksuffix="%", range=[0, 112])
    figure.update_yaxes(title=None, autorange="reversed")
    return figure


def _revision_tail_audit_figure(summary: pd.DataFrame) -> go.Figure:
    """Plot the source-audited explanation of the original 95% tail."""
    labels = {
        "xbrl_semantic_collision": "XBRL 语义碰撞",
        "discontinued_operations_scope_recast": "终止经营范围重绘",
        "cost_presentation_reclassification": "成本列报重分类",
        "revenue_presentation_reclassification": "收入列报重分类",
        "expense_presentation_reclassification": "费用列报重分类",
    }
    family_colors = {
        "data_pipeline": COLORS["red"],
        "accounting_scope": COLORS["orange"],
        "accounting_presentation": COLORS["teal"],
    }
    frame = summary.sort_values("reviewed_events", ascending=True).copy()
    frame["label"] = frame["audit_classification"].map(labels).fillna(frame["audit_classification"])
    figure = go.Figure(
        go.Bar(
            x=frame["reviewed_events"],
            y=frame["label"],
            orientation="h",
            marker_color=frame["audit_family"].map(family_colors),
            text=frame["reviewed_events"],
            textposition="outside",
            cliponaxis=False,
            customdata=frame[["max_unreviewed_revision", "audit_family"]],
            hovertemplate=(
                "%{y}<br>事件数 %{x}<br>最大表观变化 %{customdata[0]:.3f} pp"
                "<br>风险层 %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 8, "r": 32, "t": 20, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "事件数", "dtick": 1, "gridcolor": COLORS["grid"]},
        yaxis={"title": None},
        showlegend=False,
        font={"family": "Inter, sans-serif", "color": COLORS["ink"]},
    )
    return figure


def _revision_confidence_tradeoff_figure(calibration: pd.DataFrame) -> go.Figure:
    """Plot the frozen confidence-buffer-actionability frontier."""
    frame = calibration.copy()
    labels = frame["confidence"].map(lambda value: f"{float(value):.1%}")
    buffer_pp = frame["median_revision_at_risk"].astype(float) * 100.0
    robust_share = frame["robust_decision_share"].astype(float) * 100.0
    coverage = frame["empirical_coverage"].astype(float) * 100.0
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=buffer_pp,
            name="中位安全边际",
            marker_color=COLORS["orange"],
            text=[f"{value:.3f}" for value in buffer_pp],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="置信度 %{x}<br>中位缓冲 %{y:.3f} pp<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=robust_share,
            name="方向稳健信号",
            mode="lines+markers",
            line={"color": COLORS["teal"], "width": 3},
            marker={"size": 9, "symbol": "diamond"},
            yaxis="y2",
            hovertemplate="置信度 %{x}<br>方向稳健 %{y:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=coverage,
            name="实际覆盖率",
            mode="lines+markers",
            line={"color": COLORS["slate"], "width": 2, "dash": "dot"},
            marker={"size": 7, "symbol": "circle-open"},
            yaxis="y2",
            hovertemplate="置信度 %{x}<br>实际覆盖 %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 8, "r": 12, "t": 40, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.14, "x": 0},
        xaxis={"title": "目标置信度", "showgrid": False},
        yaxis={
            "title": "中位安全边际（毛利率百分点）",
            "gridcolor": COLORS["grid"],
            "rangemode": "tozero",
        },
        yaxis2={
            "title": "事件比例（%）",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "showgrid": False,
        },
    )
    return figure


def _revision_initial_latest_figure(panel: pd.DataFrame) -> go.Figure:
    """Compare filing-time and later M2 Innovations on resolved decisions."""
    eligible = panel["revision_buffer_eligible"].astype("string").str.lower().eq("true")
    resolved = panel["revision_outcome_resolved"].astype("string").str.lower().eq("true")
    frame = panel.loc[eligible & resolved].copy()
    frame["is_flip"] = frame["realized_direction_flip"].astype("string").str.lower().eq("true")
    frame["initial_pp"] = pd.to_numeric(frame["M2_innovation_raw"], errors="coerce") * 100.0
    frame["latest_pp"] = (
        pd.to_numeric(frame["latest_innovation_hindsight"], errors="coerce") * 100.0
    )
    finite = frame[["initial_pp", "latest_pp"]].replace([np.inf, -np.inf], np.nan).dropna()
    lower = float(finite.min().min())
    upper = float(finite.max().max())
    padding = max((upper - lower) * 0.05, 0.1)
    figure = go.Figure()
    for is_flip, label, color, symbol in (
        (False, "方向未翻转", COLORS["teal"], "circle"),
        (True, "方向翻转", COLORS["red"], "x"),
    ):
        group = frame.loc[frame["is_flip"].eq(is_flip)]
        customdata = np.column_stack(
            [
                group["ticker"],
                group["fiscal_year"],
                group["fiscal_quarter"],
                pd.to_numeric(group["revision_size"], errors="coerce") * 100.0,
            ]
        )
        figure.add_trace(
            go.Scattergl(
                x=group["initial_pp"],
                y=group["latest_pp"],
                mode="markers",
                name=label,
                marker={"color": color, "size": 8 if is_flip else 6, "symbol": symbol},
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} FY%{customdata[1]} %{customdata[2]}"
                    "<br>原始 %{x:+.3f} pp"
                    "<br>后来 %{y:+.3f} pp"
                    "<br>数值变化 %{customdata[3]:+.3f} pp<extra></extra>"
                ),
            )
        )
    figure.add_shape(
        type="line",
        x0=lower - padding,
        y0=lower - padding,
        x1=upper + padding,
        y1=upper + padding,
        line={"color": COLORS["slate"], "width": 1, "dash": "dot"},
    )
    figure.add_hline(y=0.0, line={"color": COLORS["grid"], "width": 1})
    figure.add_vline(x=0.0, line={"color": COLORS["grid"], "width": 1})
    figure.update_layout(
        height=420,
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.14, "x": 0},
        xaxis={
            "title": "公告时 M2 Innovation（毛利率百分点）",
            "gridcolor": COLORS["grid"],
        },
        yaxis={
            "title": "后来 M2 Innovation（毛利率百分点）",
            "gridcolor": COLORS["grid"],
        },
    )
    return figure


def _tournament_mae_figure(
    metrics: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
) -> go.Figure:
    frame = metrics.merge(
        bootstrap_intervals.loc[
            bootstrap_intervals["metric"].eq("mae_improvement_vs_naive"),
            ["model_id", "ci_lower_2_5", "ci_upper_97_5"],
        ],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    values = frame["mae_improvement_vs_naive"].astype(float)
    colors = [
        COLORS["slate"] if model_id in {"M0", "M1"} else COLORS["teal"]
        for model_id in frame["model_id"]
    ]
    figure = go.Figure(
        go.Bar(
            x=frame["model_id"],
            y=values,
            marker_color=colors,
            customdata=np.column_stack([frame["model_name"], frame["forecast_events"]]),
            error_y={
                "type": "data",
                "array": (frame["ci_upper_97_5"] - values).clip(lower=0.0),
                "arrayminus": (values - frame["ci_lower_2_5"]).clip(lower=0.0),
                "color": COLORS["ink"],
                "thickness": 1.3,
            },
            hovertemplate=(
                "%{x} · %{customdata[0]}<br>MAE skill %{y:+.1%}"
                "<br>共同事件 %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0.0, line_color=COLORS["ink"], line_width=1)
    figure.update_layout(**_chart_layout(height=390, y_title="MAE skill"))
    figure.update_yaxes(tickformat="+.0%", range=[-0.22, 0.52])
    figure.update_xaxes(title="模型")
    return figure


def _tournament_year_figure(metrics_by_year: pd.DataFrame) -> go.Figure:
    frame = metrics_by_year.copy()
    frame["availability_year"] = frame["availability_year"].astype(int).astype(str)
    figure = go.Figure()
    model_colors = {
        "M1": COLORS["slate"],
        "M2": COLORS["teal"],
        "M3": COLORS["orange"],
        "M4": COLORS["ink"],
    }
    for model_id in ("M1", "M2", "M3", "M4"):
        model = frame.loc[frame["model_id"].eq(model_id)]
        figure.add_trace(
            go.Scatter(
                x=model["availability_year"],
                y=model["mae_improvement_vs_naive"],
                mode="lines+markers",
                name=model_id,
                line={"color": model_colors[model_id], "width": 2.3},
                customdata=np.column_stack([model["model_name"], model["forecast_events"]]),
                hovertemplate=(
                    "%{x} · %{customdata[0]}<br>MAE skill %{y:+.1%}"
                    "<br>共同事件 %{customdata[1]:,}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0.0, line_color=COLORS["grid"], line_width=1)
    figure.update_layout(**_chart_layout(height=390, y_title="MAE skill"))
    figure.update_yaxes(tickformat="+.0%")
    figure.update_xaxes(title="可得年份", type="category")
    return figure


def _tournament_representation_figure(diagnostics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, label, color in (
        ("spearman_delta_innovation", "Delta–Innovation 秩相关", COLORS["slate"]),
        (
            "median_abs_percentile_rank_movement",
            "中位绝对秩移动",
            COLORS["orange"],
        ),
        ("observations_changing_quintile", "跨五分位", COLORS["teal"]),
    ):
        figure.add_trace(
            go.Bar(
                x=diagnostics["model_id"],
                y=diagnostics[column],
                name=label,
                marker_color=color,
                hovertemplate=f"%{{x}}<br>{label} %{{y:.1%}}<extra></extra>",
            )
        )
    figure.update_layout(
        **_chart_layout(height=430, y_title="比例 / 秩相关"),
        barmode="group",
    )
    figure.update_yaxes(tickformat=".0%", range=[0.0, 1.05])
    figure.update_xaxes(title="模型")
    return figure


def _representation_coverage_figure(coverage: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, name, color in (
        ("level_events", "Level", COLORS["slate"]),
        ("delta_events", "Delta", COLORS["orange"]),
        ("raw_innovation_events", "Raw Innovation", COLORS["teal"]),
        ("standardized_innovation_events", "Std Innovation", COLORS["ink"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=coverage["fiscal_year"].astype(str),
                y=coverage[column],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2.5},
                hovertemplate=f"财年 %{{x}}<br>{name}: %{{y:,}}<extra></extra>",
            )
        )
    figure.update_layout(**_chart_layout(height=390, y_title="事件数"))
    figure.update_xaxes(title="财年", type="category")
    return figure


def _forecast_diagnostics_figure(diagnostics: pd.DataFrame) -> go.Figure:
    frame = diagnostics.loc[diagnostics["availability_year"].astype(str).ne("all")].copy()
    figure = go.Figure()
    for column, name, color in (
        ("model_rmse", "Pooled OLS", COLORS["teal"]),
        ("naive_seasonal_rmse", "去年同季", COLORS["orange"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame["availability_year"].astype(str),
                y=frame[column],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2.5},
                hovertemplate=f"可得年份 %{{x}}<br>{name} RMSE: %{{y:.2%}}<extra></extra>",
            )
        )
    figure.update_layout(**_chart_layout(height=390, y_title="毛利率预测 RMSE"))
    figure.update_yaxes(tickformat=".1%", rangemode="tozero")
    figure.update_xaxes(title="实际可得年份", type="category")
    return figure


def _rank_movement_figure(rank_movements: pd.DataFrame) -> go.Figure:
    frame = rank_movements.copy()
    frame["pooled_abs_rank_movement_pp"] = (
        pd.to_numeric(frame["pooled_abs_rank_movement"], errors="coerce") * 100.0
    )
    figure = go.Figure(
        go.Histogram(
            x=frame["pooled_abs_rank_movement_pp"],
            xbins={"start": 0, "end": 60, "size": 5},
            marker={"color": COLORS["teal"]},
            hovertemplate="绝对秩移动约 %{x:.0f} 百分点<br>事件 %{y}<extra></extra>",
        )
    )
    median = float(frame["pooled_abs_rank_movement_pp"].median())
    figure.add_vline(
        x=median,
        line={"color": COLORS["orange"], "width": 2},
        annotation_text=f"中位数 {median:.1f}pp",
        annotation_position="top right",
    )
    figure.update_layout(**_chart_layout(height=350, y_title="事件数"))
    figure.update_xaxes(title="|Innovation percentile rank − Delta percentile rank|（百分点）")
    return figure


def _forecast_skill_figure(diagnostics: pd.DataFrame) -> go.Figure:
    frame = diagnostics.loc[diagnostics["availability_year"].astype(str).ne("all")].copy()
    figure = go.Figure()
    for column, name, color in (
        ("mae_improvement_vs_naive", "MAE 技能（主指标）", COLORS["teal"]),
        ("rmse_improvement_vs_naive", "RMSE 技能", COLORS["orange"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame["availability_year"].astype(str),
                y=frame[column],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2.5},
                customdata=frame["forecast_events"],
                hovertemplate=(
                    f"可得年份 %{{x}}<br>{name}: %{{y:+.1%}}<br>事件 %{{customdata}}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0.0, line={"color": COLORS["slate"], "width": 1})
    figure.update_layout(**_chart_layout(height=370, y_title="相对去年同季的预测技能"))
    figure.update_yaxes(tickformat="+.0%")
    figure.update_xaxes(title="实际可得年份", type="category")
    return figure


def _representation_correlation_figure(correlations: pd.DataFrame) -> go.Figure:
    labels = {
        "gm_level": "Level",
        "gm_delta_yoy": "Delta",
        "gm_innovation_raw": "Raw Innovation",
        "gm_innovation_std": "Std Innovation",
    }
    columns = list(labels)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    spearman = correlations.loc[correlations["method"].eq("spearman")]
    for row in spearman.itertuples(index=False):
        matrix.loc[row.representation_1, row.representation_2] = row.correlation
        matrix.loc[row.representation_2, row.representation_1] = row.correlation
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=[labels[column] for column in columns],
            y=[labels[column] for column in columns],
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#B42318"],
                [0.5, "#F6F8F7"],
                [1.0, COLORS["teal"]],
            ],
            text=np.vectorize(lambda value: f"{value:.2f}")(matrix.to_numpy()),
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}<br>Spearman %{z:.3f}<extra></extra>",
            colorbar={"title": "ρ"},
        )
    )
    figure.update_layout(**_chart_layout(height=430, y_title=""))
    figure.update_xaxes(side="bottom")
    return figure


def _representation_summary(diagnostics: dict[str, Any], distinctness: dict[str, Any]) -> str:
    rmse_skill = float(diagnostics["rmse_improvement_vs_naive"])
    mae_skill = float(diagnostics["mae_improvement_vs_naive"])
    correlation = float(distinctness["spearman_delta_raw_innovation"])
    changed_quintile = float(distinctness["observations_changing_quintile"])
    large_move = float(distinctness["observations_moving_two_or_more_quintiles"])
    return (
        f"第一版模型相对去年同季基准的总体 RMSE 改善 {rmse_skill:+.1%}，"
        f"但 MAE 改善为 {mae_skill:+.1%}；Raw Innovation 与 Delta 的秩相关为 "
        f"{correlation:.2f}。尽管如此，{changed_quintile:.1%} 的事件跨了五分位边界，"
        f"但只有 {large_move:.1%} 移动两个或更多分位。结论是：当前模型会做一些相邻排序"
        "调整，却还没有形成强而稳定的新表示。"
    )


def _case_timeline_figure(case: AppleAmendmentCase) -> go.Figure:
    metadata = case.metadata
    original = pd.Timestamp(metadata["original_accepted_at"])
    amendment = pd.Timestamp(metadata["amendment_accepted_at"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[original, amendment],
            y=[0, 0],
            mode="lines+markers+text",
            line={"color": COLORS["grid"], "width": 5},
            marker={
                "color": [COLORS["orange"], COLORS["teal"]],
                "size": [18, 18],
                "line": {"color": "white", "width": 3},
            },
            text=["原始 10-K<br>市场当时可得", "10-K/A<br>未来修订版本"],
            textposition=["top center", "top center"],
            customdata=[str(metadata["original_accession"]), str(metadata["amendment_accession"])],
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_annotation(
        x=original + (amendment - original) / 2,
        y=-0.04,
        text=f"相隔 {case.information_delay_days:.0f} 天",
        showarrow=False,
        font={"color": COLORS["slate"], "size": 13},
    )
    figure.update_layout(
        height=250,
        margin={"l": 25, "r": 25, "t": 48, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["ink"], "family": "Arial, sans-serif"},
        xaxis={
            "showgrid": False,
            "tickformat": "%Y-%m-%d",
            "range": [original - pd.Timedelta(days=12), amendment + pd.Timedelta(days=12)],
        },
        yaxis={"visible": False, "range": [-0.12, 0.12]},
        hovermode="closest",
    )
    return figure


def _case_metric_figure(case: AppleAmendmentCase, tag: str) -> go.Figure:
    frame = case.facts.loc[case.facts["tag"].eq(tag)].sort_values("fiscal_year").copy()
    is_per_share = frame["unit"].eq("USD/shares").all()
    scale = 1.0 if is_per_share else 1_000_000_000.0
    suffix = "" if is_per_share else "B"
    y_title = "美元 / 股" if is_per_share else "十亿美元"
    figure = go.Figure()
    for column, name, color in (
        ("original_value", "原始 10-K（当时可得）", COLORS["orange"]),
        ("amendment_value", "10-K/A（90 天后）", COLORS["teal"]),
    ):
        values = frame[column] / scale
        figure.add_trace(
            go.Bar(
                x=frame["fiscal_year"].astype(str),
                y=values,
                name=name,
                marker_color=color,
                text=[f"${value:,.2f}{suffix}" for value in values],
                textposition="outside",
                hovertemplate=f"财年 %{{x}}<br>{name} $%{{y:,.2f}}{suffix}<extra></extra>",
            )
        )
    figure.update_layout(**_chart_layout(height=410, y_title=y_title), barmode="group")
    figure.update_xaxes(title="财年", type="category")
    return figure


def _gross_margin_level_figure(case: AppleAmendmentCase) -> go.Figure:
    frame = case.gross_margin
    figure = go.Figure()
    for column, name, color, symbol in (
        ("original_margin", "原始 10-K", COLORS["orange"], "circle"),
        ("amendment_margin", "10-K/A", COLORS["teal"], "diamond"),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame["fiscal_year"].astype(str),
                y=frame[column],
                mode="lines+markers+text",
                name=name,
                line={"color": color, "width": 3},
                marker={"color": color, "size": 10, "symbol": symbol},
                text=[f"{value:.2%}" for value in frame[column]],
                textposition="top center" if column == "amendment_margin" else "bottom center",
                hovertemplate=f"财年 %{{x}}<br>{name} %{{y:.2%}}<extra></extra>",
            )
        )
    figure.update_layout(**_chart_layout(height=410, y_title="毛利率"))
    figure.update_yaxes(tickformat=".0%", range=[0.30, 0.42])
    figure.update_xaxes(title="财年", type="category")
    return figure


def _gross_margin_signal_figure(case: AppleAmendmentCase) -> go.Figure:
    frame = case.gross_margin.dropna(subset=["original_yoy_change"]).copy()
    figure = go.Figure()
    for column, name, color in (
        ("original_yoy_change", "点时信号：只用当时可得值", COLORS["orange"]),
        ("amendment_yoy_change", "前视信号：回填未来修订值", COLORS["teal"]),
    ):
        figure.add_trace(
            go.Bar(
                x=frame["fiscal_year"].astype(str),
                y=frame[column],
                name=name,
                marker_color=color,
                text=[_percentage_points(value) for value in frame[column]],
                textposition="outside",
                hovertemplate=f"财年 %{{x}}<br>{name} %{{y:+.2%}}<extra></extra>",
            )
        )
    figure.update_layout(
        **_chart_layout(height=350, y_title="毛利率同比变化（百分点）"),
        barmode="group",
    )
    figure.update_yaxes(tickformat="+.1%", zeroline=True, zerolinecolor=COLORS["grid"])
    figure.update_xaxes(title="财年", type="category")
    return figure


def _render_sec_workbench() -> None:
    """Render the live SEC download, cleaning, and amendment-audit workbench."""
    st.sidebar.markdown("## SEC 抓取设置")
    company = st.sidebar.text_input("Ticker 或 CIK", value="AAPL").strip()
    user_agent = st.sidebar.text_input(
        "SEC User-Agent",
        value=os.environ.get("SEC_USER_AGENT", ""),
        placeholder="Your Name your.email@domain.com",
        help="SEC 要求自动客户端声明身份和真实联系邮箱；它不是 API Key。",
    )
    include_archives = st.sidebar.checkbox(
        "下载旧 submissions 分片",
        value=False,
        help="公司 recent 区块不足时再启用；会增加请求数和清洗时间。",
    )
    fetch = st.sidebar.button("抓取并清洗", type="primary", width="stretch")
    st.sidebar.caption("默认限速 5 次/秒；只抓取当前公司需要的 JSON。")

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">SEC EDGAR · DISCLOSURE AUDIT</div>
          <h1>真实披露数据试验台</h1>
          <p>从原始响应到修订差异，每一步都保留时间戳、单位、accession 和拒绝原因。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "这里验证财报数据，不生成真实收益曲线。接入历史价格、交易所日历和退市收益之前，"
        "把 SEC facts 直接送入回测会制造新的偏差。",
        icon="⚠️",
    )
    if fetch:
        try:
            with st.spinner("正在下载 SEC submissions 与 companyfacts，并执行点时清洗…"):
                service = SecIngestionService(user_agent, data_root=Path("data/sec"))
                st.session_state["sec_ingestion_result"] = service.ingest(
                    company, include_archives=include_archives
                )
        except (SecConfigurationError, SecRequestError, OSError, ValueError) as exc:
            st.error(f"抓取未完成：{exc}")

    result = st.session_state.get("sec_ingestion_result")
    if not isinstance(result, SecIngestionResult):
        st.markdown("### 抓取流程")
        first, second, third, fourth = st.columns(4)
        first.markdown("**01 · 原始快照**\n\n保存 SEC JSON 原始字节与 SHA-256。")
        second.markdown("**02 · 点时清洗**\n\n用 accession 匹配 UTC 接受时间。")
        third.markdown("**03 · 修订比较**\n\n严格匹配同标签、单位和期间。")
        fourth.markdown("**04 · 可审计输出**\n\n导出事实、差异、拒绝记录和 manifest。")
        st.info("填写真实联系邮箱后，输入 AAPL 或十位 CIK，点击“抓取并清洗”。")
        st.markdown(
            "官方说明：[SEC Data APIs](https://www.sec.gov/search-filings/"
            "edgar-application-programming-interfaces) · "
            "[Fair Access](https://www.sec.gov/search-filings/edgar-search-assistance/"
            "accessing-edgar-data)"
        )
        return

    _render_sec_result(result)


def _render_sec_result(result: SecIngestionResult) -> None:
    """Show provenance, amendment comparisons, cleaned facts, and exports."""
    amendments = result.filings.loc[result.filings["is_amendment"]]
    changed = result.amendment_comparisons.loc[
        result.amendment_comparisons["comparison_status"] == "changed"
    ]
    columns = st.columns(5)
    columns[0].metric("公司", result.ticker or result.cik)
    columns[1].metric("10-K / 10-Q", f"{len(result.filings):,}")
    columns[2].metric("标准 XBRL facts", f"{len(result.facts):,}")
    columns[3].metric("修正文件", f"{len(amendments):,}")
    columns[4].metric("数值变化", f"{len(changed):,}")
    st.caption(
        f"{result.company_name} · CIK {result.cik} · 数据版本 "
        f"{str(result.metadata['dataset_version'])[:16]} · 原始快照 {len(result.snapshots)} 个"
    )

    comparison_tab, facts_tab, filings_tab, rejection_tab, provenance_tab = st.tabs(
        ["修订差异", "标准化 facts", "公告清单", "拒绝记录", "数据血缘"]
    )
    with comparison_tab:
        if result.amendment_comparisons.empty:
            st.info(
                "当前抓取范围没有可比较的 10-K/A 或 10-Q/A XBRL facts。"
                "这不代表公司从未修订；可启用旧 submissions 分片或查看原始 filing。"
            )
        else:
            statuses = st.multiselect(
                "比较状态",
                ["changed", "unchanged", "no_prior_match"],
                default=["changed"],
            )
            comparison_view = result.amendment_comparisons.loc[
                result.amendment_comparisons["comparison_status"].isin(statuses)
            ].copy()
            comparison_view["percentage_difference"] *= 100.0
            st.dataframe(
                comparison_view,
                hide_index=True,
                width="stretch",
                column_config={
                    "percentage_difference": st.column_config.NumberColumn(format="%.2f%%"),
                    "original_accepted_at": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss Z"
                    ),
                    "amendment_accepted_at": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss Z"
                    ),
                },
            )
            _download_csv(
                "下载修订比较 CSV",
                result.amendment_comparisons,
                f"{result.cik}-amendment-comparisons.csv",
            )
    with facts_tab:
        tags = sorted(result.facts["tag"].dropna().astype(str).unique())
        preferred = [
            tag
            for tag in (
                "EarningsPerShareDiluted",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "GrossProfit",
                "NetIncomeLoss",
            )
            if tag in tags
        ]
        selected_tags = st.multiselect(
            "XBRL tags",
            tags,
            default=preferred or tags[: min(4, len(tags))],
        )
        fact_view = result.facts.loc[result.facts["tag"].isin(selected_tags)].tail(2_000)
        st.dataframe(
            fact_view,
            hide_index=True,
            width="stretch",
            column_config={
                "accepted_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss Z"),
            },
        )
        _download_csv("下载全部 facts CSV", result.facts, f"{result.cik}-facts.csv")
    with filings_tab:
        st.dataframe(
            result.filings,
            hide_index=True,
            width="stretch",
            column_config={
                "source_url": st.column_config.LinkColumn(display_text="打开 SEC filing"),
                "accepted_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss Z"),
            },
        )
        _download_csv("下载 filings CSV", result.filings, f"{result.cik}-filings.csv")
    with rejection_tab:
        if result.rejections.empty:
            st.success("本次抓取没有拒绝记录。")
        else:
            rejection_summary = (
                result.rejections.groupby(["stage", "reason"], as_index=False)
                .size()
                .sort_values("size", ascending=False)
            )
            st.dataframe(rejection_summary, hide_index=True, width="stretch")
            with st.expander("查看拒绝明细"):
                st.dataframe(result.rejections, hide_index=True, width="stretch")
            _download_csv("下载拒绝记录 CSV", result.rejections, f"{result.cik}-rejections.csv")
    with provenance_tab:
        snapshot_frame = pd.DataFrame(
            [
                {
                    "kind": snapshot.kind,
                    "sha256": snapshot.sha256,
                    "fetched_at": snapshot.fetched_at,
                    "source_url": snapshot.source_url,
                    "local_path": str(snapshot.path),
                }
                for snapshot in result.snapshots
            ]
        )
        st.dataframe(
            snapshot_frame,
            hide_index=True,
            width="stretch",
            column_config={"source_url": st.column_config.LinkColumn(display_text="SEC source")},
        )
        st.code(str(result.metadata["normalized_directory"]), language=None)

    st.caption(
        "SEC companyfacts 是当前聚合视图。修正表单可能只补充文字或新事实；"
        "changed 才表示严格可比的数值发生变化。"
    )


def _download_csv(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


def _sidebar(data: ResearchData) -> tuple[SignalName, BacktestConfig, BiasConfig]:
    st.sidebar.markdown("## 研究控制台")
    label_to_signal: dict[str, SignalName] = {label: key for key, label in SIGNAL_LABELS.items()}
    selected_label = st.sidebar.radio("财报信号", list(label_to_signal), index=0)
    signal_name = label_to_signal[selected_label]
    st.sidebar.markdown("### 回测参数")
    holding_days = st.sidebar.slider("持有期（交易日）", 10, 40, 20, 5)
    quantile = st.sidebar.slider("多空分位阈值", 0.10, 0.30, 0.20, 0.05, format="%.0f%%")
    st.sidebar.markdown("### 引入偏差")
    st.sidebar.caption("勾选代表故意使用有偏做法，便于观察结果被抬高的程度。")
    lookahead = st.sidebar.checkbox("前视偏差：使用重述值", value=True)
    survivorship = st.sidebar.checkbox("幸存者偏差：仅保留存续股票", value=True)
    timing = st.sidebar.checkbox("公告延迟偏差：公告日收盘成交", value=True)
    st.sidebar.divider()
    st.sidebar.caption(
        f"{data.metadata['n_stocks']} 只股票 · {data.metadata['n_quarters']} 个季度 · "
        f"{data.metadata['delisted_stocks']} 只最终退市"
    )
    return (
        signal_name,
        BacktestConfig(holding_days=holding_days, quantile_fraction=quantile),
        BiasConfig(lookahead, survivorship, timing),
    )


def _bias_badges(biases: BiasConfig) -> None:
    states = [
        ("前视", biases.lookahead),
        ("幸存者", biases.survivorship),
        ("公告时点", biases.announcement_timing),
    ]
    badges = "".join(
        f'<span class="bias-badge {"on" if enabled else "off"}">'
        f"{name}偏差 · {'已引入' if enabled else '已修正'}</span>"
        for name, enabled in states
    )
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def _metric_cards(result: BacktestResult) -> None:
    metrics = result.metrics
    columns = st.columns(4)
    columns[0].metric("年化收益", _percent(metrics.annual_return))
    columns[1].metric("夏普比率", f"{metrics.sharpe_ratio:.2f}")
    columns[2].metric("最大回撤", _percent(metrics.max_drawdown))
    columns[3].metric("平均 IC", f"{metrics.mean_ic:.3f}")


def _equity_figure(
    standards: dict[str, BacktestResult],
    current: BacktestResult,
    current_biases: BiasConfig,
) -> go.Figure:
    figure = go.Figure()
    scenario_styles = {
        "原始回测（全部偏差）": (COLORS["orange"], "solid"),
        "修正前视偏差后": (COLORS["slate"], "solid"),
        "修正全部偏差后": (COLORS["teal"], "solid"),
    }
    for name, result in standards.items():
        color, dash = scenario_styles[name]
        figure.add_trace(
            go.Scatter(
                x=result.equity_curve["date"],
                y=result.equity_curve["equity"],
                mode="lines",
                name=name,
                line={"color": color, "width": 3, "dash": dash},
                hovertemplate="%{x|%Y-%m}<br>净值 %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    if current_biases not in STANDARD_SCENARIOS.values():
        figure.add_trace(
            go.Scatter(
                x=current.equity_curve["date"],
                y=current.equity_curve["equity"],
                mode="lines",
                name="当前自定义情景",
                line={"color": COLORS["ink"], "width": 2, "dash": "dot"},
                hovertemplate="%{x|%Y-%m}<br>净值 %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(**_chart_layout(height=430, y_title="累计净值"))
    figure.update_yaxes(tickformat=".2f")
    return figure


def _quantile_figure(result: BacktestResult) -> go.Figure:
    frame = result.quantile_returns
    values = frame["mean_excess_return"]
    colors = [COLORS["red"] if value < 0 else COLORS["teal"] for value in values]
    figure = go.Figure(
        go.Bar(
            x=[f"Q{value}" for value in frame["quantile"]],
            y=values,
            marker_color=colors,
            text=[f"{value:.2%}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>平均超额收益 %{y:.2%}<extra></extra>",
        )
    )
    figure.update_layout(**_chart_layout(height=430, y_title="平均超额收益"))
    figure.update_yaxes(tickformat=".1%", zeroline=True, zerolinecolor=COLORS["grid"])
    return figure


def _preview_bar_figure(
    title: str,
    values: dict[str, float | int],
    *,
    y_title: str,
    value_format: str = "number",
) -> go.Figure:
    """Build a compact chart for the aggregate-only public research preview."""
    labels = list(values)
    numeric = [float(values[label]) for label in labels]
    if value_format == "percent":
        text = [f"{value:+.1%}" for value in numeric]
    elif value_format == "pp":
        text = [f"{value:.3f} pp" for value in numeric]
    else:
        text = [f"{value:,.0f}" for value in numeric]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=numeric,
            text=text,
            textposition="outside",
            cliponaxis=False,
            marker_color=[COLORS["teal"] if value >= 0 else COLORS["red"] for value in numeric],
        )
    )
    figure.update_layout(
        **_chart_layout(height=340, y_title=y_title),
        title=title,
        showlegend=False,
    )
    return figure


def _render_public_representation_preview() -> None:
    """Render aggregate frozen results when event-level local assets are absent."""
    preview = load_public_research_preview()["representation"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 02 · PUBLIC AGGREGATE SNAPSHOT</div>
          <h1>Level、Delta，还是 Innovation？</h1>
          <p>公开仓库仅分发冻结汇总值；事件级 SEC 研究资产留在本地并可按管线重建。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "当前显示的是经过哈希版本锁定的聚合快照，不包含可下载的事件级数据。"
        "它完整保留主要研究结论，但不能替代本地原始证据审计。",
        icon="ℹ️",
    )
    cards = st.columns(5)
    cards[0].metric("核心资产", f"{int(preview['core_companies'])} 家")
    cards[1].metric("严格 Delta", f"{int(preview['delta_events']):,}")
    cards[2].metric("OOS Innovation", f"{int(preview['raw_innovation_events']):,}")
    cards[3].metric("共同表示样本", f"{int(preview['common_representation_events']):,}")
    cards[4].metric(
        "Innovation 覆盖",
        f"{int(preview['companies_with_minimum_innovation_history'])} 家",
    )
    movement = preview["rank_movement"]
    movement_cards = st.columns(3)
    movement_cards[0].metric("跨五分位", f"{float(movement['changes_quintile']):.1%}")
    movement_cards[1].metric("中位秩移动", f"{float(movement['median_abs_percentile']):.1%}")
    movement_cards[2].metric(
        "跨两个以上分位", f"{float(movement['moves_two_or_more_quintiles']):.1%}"
    )

    charts = [
        _preview_bar_figure(
            "表示变量的样本漏斗",
            {
                "Level": preview["level_events"],
                "Delta": preview["delta_events"],
                "Raw Innovation": preview["raw_innovation_events"],
                "Standardized": preview["common_representation_events"],
            },
            y_title="事件数",
        ),
        _preview_bar_figure(
            "第一版 pooled OLS 相对去年同季的预测技能",
            preview["forecast_skill"],
            y_title="技能",
            value_format="percent",
        ),
        _preview_bar_figure(
            "Innovation 对 Delta 排序的改变",
            {
                "中位秩移动": movement["median_abs_percentile"],
                "跨五分位": movement["changes_quintile"],
                "跨两个以上分位": movement["moves_two_or_more_quintiles"],
            },
            y_title="事件占比",
            value_format="percent",
        ),
        _preview_bar_figure(
            "共同样本的秩相关",
            preview["rank_correlations"],
            y_title="Spearman 相关",
            value_format="percent",
        ),
        _preview_bar_figure(
            "第一版研究 Gate",
            {"点时/OOS": 1, "预测稳定性": 0, "历史收益": 0},
            y_title="通过 = 1",
        ),
    ]
    for figure in charts:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.warning(
        "结果是 Innovation 会重新排序部分事件，但仍主要是 Delta 的重新表达；这一阶段没有收益证据。",
        icon="⚠️",
    )
    st.caption(
        f"公开聚合版本：{load_public_research_preview()['generated_from']['representation']}"
    )


def _render_public_tournament_preview() -> None:
    """Render the aggregate forecast-tournament snapshot for a fresh clone."""
    preview = load_public_research_preview()["forecast_tournament"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 03 · PUBLIC AGGREGATE SNAPSHOT</div>
          <h1>什么才算真正的 Innovation？</h1>
          <p>M0–M4 在同一组点时事件上竞争预测准确性；公开版不分发事件级训练面板。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    cards[0].metric(
        "共同样本",
        f"{int(preview['common_sample_events']):,}",
        f"{int(preview['common_sample_companies'])} 家公司",
        delta_color="off",
    )
    cards[1].metric("胜出模型", str(preview["selected_model_id"]))
    selected_skill = float(preview["model_mae_skill"][str(preview["selected_model_id"])])
    cards[2].metric("MAE 改善", f"{selected_skill:+.1%}")
    cards[3].metric(
        "稳定年份",
        f"{int(preview['positive_mae_skill_years'])} / {int(preview['eligible_years'])}",
    )
    charts = [
        _preview_bar_figure(
            "M0–M4 共同样本 MAE skill",
            preview["model_mae_skill"],
            y_title="MAE skill",
            value_format="percent",
        ),
        _preview_bar_figure(
            "M2 的年度 MAE skill",
            preview["m2_year_mae_skill"],
            y_title="MAE skill",
            value_format="percent",
        ),
        _preview_bar_figure(
            "研究闸门",
            {"预测 Gate": 1, "历史股票池": 0, "收益 Gate": 0},
            y_title="通过 = 1",
        ),
    ]
    for figure in charts:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.success(
        "M2 在 812 个共同事件上将 MAE 改善 35.4%，并在六个合格年份全部改善。"
        "这是财务预测结果，不是买卖或收益证据。",
        icon="✅",
    )
    st.caption(
        f"公开聚合版本：{load_public_research_preview()['generated_from']['forecast_tournament']}"
    )


def _render_public_revision_preview() -> None:
    """Render aggregate revision-risk diagnostics without redistributing events."""
    preview = load_public_research_preview()["revision_robustness"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 04 · PUBLIC AGGREGATE SNAPSHOT</div>
          <h1>公告当下，这个信号值得相信多少？</h1>
          <p>使用当时已经解决的历史修订，估计原始 Innovation 所需的决策安全边际。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = st.columns(6)
    cards[0].metric("可形成决策", f"{int(preview['primary_decision_events']):,}")
    cards[1].metric("已验证结果", f"{int(preview['primary_resolved_evaluation_events']):,}")
    cards[2].metric(
        "95% 中位安全边际",
        f"{float(preview['primary_median_revision_at_risk']) * 100:.3f} pp",
    )
    cards[3].metric("实际覆盖率", f"{float(preview['primary_empirical_coverage']):.1%}")
    cards[4].metric("方向稳健信号", f"{float(preview['primary_robust_decision_share']):.1%}")
    cards[5].metric(
        "确认的经济差错", str(int(preview["economic_error_corrections_in_reviewed_tail"]))
    )
    confidence = [f"{float(value):.1%}" for value in preview["confidence"]]
    margin_pp = {
        label: float(value) * 100
        for label, value in zip(confidence, preview["median_revision_at_risk"], strict=True)
    }
    coverage = {
        label: float(value)
        for label, value in zip(confidence, preview["empirical_coverage"], strict=True)
    }
    charts = [
        _preview_bar_figure(
            "不同置信度的中位安全边际",
            margin_pp,
            y_title="毛利率百分点",
            value_format="pp",
        ),
        _preview_bar_figure(
            "走步修订覆盖率",
            coverage,
            y_title="覆盖率",
            value_format="percent",
        ),
        _preview_bar_figure(
            "15 个尾部事件的来源复核分类",
            preview["tail_classifications"],
            y_title="事件数",
        ),
    ]
    for figure in charts:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.warning(
        "0.166 pp 是财务信号的历史修订缓冲，不是买卖阈值。15 个尾部事件中没有确认到"
        "经济差错；变化主要来自列报、业务范围和 XBRL 语义问题。",
        icon="⚠️",
    )
    st.caption(
        f"公开聚合版本：{load_public_research_preview()['generated_from']['revision_robustness']}"
    )


def _render_public_effectiveness_preview() -> None:
    """Render the final frozen effectiveness result in a data-light public clone."""
    preview = load_public_research_preview()["longitudinal_effectiveness"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">RESULT 05 · FROZEN EFFECTIVENESS DIAGNOSTIC</div>
          <h1>信号尚未成立，偏差已经可以计量</h1>
          <p>49 个完整点时财报事件；同样本比较 Level、Delta 与 Innovation；
          保留后来退市证券和证券代码迁移。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "正式判定为 effectiveness_not_established：本页是横截面解释力与偏差诊断，"
        "不是组合回测、年化收益或买卖建议。",
        icon="⚠️",
    )
    cards = st.columns(6)
    cards[0].metric("完整事件", str(int(preview["complete_events"])))
    cards[1].metric("发行人", str(int(preview["issuers"])))
    innovation_ic = float(preview["representation_ic"]["Innovation std"])
    cards[2].metric("Innovation IC", f"{innovation_ic:+.3f}")
    interval = preview["innovation_ci"]
    cards[3].metric("Innovation 95% CI", f"[{float(interval[0]):+.3f}, {float(interval[1]):+.3f}]")
    cards[4].metric("幸存者样本抬升", f"{float(preview['survivor_only_inflation_pp']):+.2f} pp")
    cards[5].metric("冻结判定", "尚未建立")

    horizon_values: dict[str, float] = {}
    for horizon, signals in preview["horizon_ic"].items():
        horizon_values[f"{horizon} Delta"] = float(signals["Delta"])
        horizon_values[f"{horizon} Innovation"] = float(signals["Innovation"])
    charts = [
        _preview_bar_figure(
            "20 日主样本 Spearman IC",
            preview["representation_ic"],
            y_title="IC",
            value_format="percent",
        ),
        _preview_bar_figure(
            "不同事件窗口的 Delta 与 Innovation IC",
            horizon_values,
            y_title="IC",
            value_format="percent",
        ),
        _preview_bar_figure(
            "幸存者筛选如何改变样本平均收益",
            preview["survivorship_mean_return"],
            y_title="20 日市场超额收益",
            value_format="percent",
        ),
        _preview_bar_figure(
            "补回 OB→TEAD 负向事件后的 Innovation IC",
            preview["repair_ic"],
            y_title="IC",
            value_format="percent",
        ),
    ]
    for figure in charts:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.info(
        "正面结果是研究系统拒绝了一个证据不足的信号，同时量化出仅保留存活公司会把"
        "平均超额收益抬高 1.81 个百分点。下一步应扩大独立样本，而不是继续在这 49 个"
        "事件上调参。",
        icon="🔎",
    )
    st.caption(
        f"公开聚合版本："
        f"{load_public_research_preview()['generated_from']['longitudinal_effectiveness']}"
    )


def _chart_layout(*, height: int, y_title: str) -> dict[str, object]:
    return {
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 20, "b": 10},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["ink"], "family": "Arial, sans-serif"},
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "yaxis": {"title": y_title, "gridcolor": COLORS["grid"]},
        "xaxis": {"gridcolor": "rgba(0,0,0,0)"},
    }


def _research_summary(
    impact: pd.DataFrame,
    standards: dict[str, BacktestResult],
    signal_name: SignalName,
) -> str:
    individual = impact.iloc[1:4].copy()
    largest = individual.loc[individual["相对带偏差基准的收益削弱"].idxmax()]
    biased = standards["原始回测（全部偏差）"].metrics
    corrected = standards["修正全部偏差后"].metrics
    conclusion = (
        "修正后只剩弱相关，不能据此宣称信号可交易。"
        if corrected.annual_return < 0.03 or corrected.mean_ic < 0.03
        else "修正后仍有正向关系，但仍需真实点时数据和样本外检验。"
    )
    return (
        f"{SIGNAL_LABELS[signal_name]}：带全部偏差时年化收益为 {biased.annual_return:.1%}，"
        f"修正全部偏差后为 {corrected.annual_return:.1%}。单项影响最大的是“{largest['情景']}”，"
        f"年化收益削弱 {float(largest['相对带偏差基准的收益削弱']):.1%}。"
        f"完全修正后的平均 IC 为 {corrected.mean_ic:.3f}；{conclusion}"
    )


def _data_audit_panel(data: ResearchData, result: BacktestResult) -> None:
    first, second = st.columns([1.2, 1.0], gap="large")
    with first:
        st.markdown("**模拟财报样例**")
        sample_columns = [
            "ticker",
            "report_date",
            "announce_date",
            "eps",
            "eps_restated",
            "has_restatement",
            "is_listed",
        ]
        sample = data.fundamentals.loc[data.fundamentals["has_restatement"], sample_columns].head(8)
        st.dataframe(sample, hide_index=True, width="stretch")
    with second:
        st.markdown("**当前执行审计**")
        execution = (
            "公告日收盘（故意带偏差）"
            if result.biases.announcement_timing
            else "公告后第二个交易日开盘"
        )
        values = "重述后数值（故意前视）" if result.biases.lookahead else "公告时原始披露值"
        universe = "仅最终存续股票" if result.biases.survivorship else "包含最终退市股票"
        st.markdown(
            f"- 财报版本：{values}\n"
            f"- 股票池：{universe}\n"
            f"- 建仓价格：{execution}\n"
            f"- 有效观察：{result.metrics.observations:,}\n"
            f"- 横截面批次：{result.metrics.cohorts}"
        )


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: linear-gradient(180deg, #F1F5F3 0, #FAFBFA 420px); }
          .block-container { max-width: 1320px; padding-top: 2.4rem; padding-bottom: 3rem; }
          .hero { max-width: 820px; margin-bottom: 0.8rem; }
          .hero .eyebrow { color: #0F766E; font-size: .74rem; font-weight: 800;
            letter-spacing: .14em; margin-bottom: .45rem; }
          .hero h1 { color: #132A27; font-size: clamp(2rem, 4vw, 3.45rem);
            letter-spacing: -.045em; line-height: 1.05; margin: 0; }
          .hero p { color: #526762; font-size: 1.05rem; margin-top: .8rem; }
          .badge-row { display: flex; flex-wrap: wrap; gap: .5rem; margin: .7rem 0 1.25rem; }
          .bias-badge { border-radius: 999px; padding: .32rem .7rem; font-size: .78rem;
            font-weight: 700; border: 1px solid; }
          .bias-badge.on { color: #9A3412; background: #FFF7ED; border-color: #FDBA74; }
          .bias-badge.off { color: #0F766E; background: #ECFDF5; border-color: #6EE7B7; }
          [data-testid="stMetric"] { background: rgba(255,255,255,.82); border: 1px solid #DCE5E2;
            border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 8px 24px rgba(19,42,39,.04); }
          [data-testid="stMetricValue"] { color: #132A27; font-weight: 750; }
          [data-testid="stSidebar"] { border-right: 1px solid #DCE5E2; }
          h2, h3 { color: #193B35; letter-spacing: -.02em; }
          [data-testid="stDataFrame"] { border: 1px solid #DCE5E2;
            border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
