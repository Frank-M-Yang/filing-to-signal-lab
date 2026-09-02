"""Streamlit smoke test."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exception() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert len(app.metric) == 4

    biased_annual_return = app.metric[0].value
    for checkbox in app.checkbox:
        checkbox.set_value(False)
    app.run()

    assert not app.exception
    assert app.metric[0].value != biased_annual_return

    app.radio[0].set_value("成果 01 · Apple 修订案例")
    app.run()

    assert not app.exception
    assert any("90 天后，财报里的信号变了" in markdown.value for markdown in app.markdown)
    assert any(metric.value == "90 天" for metric in app.metric)
    assert len(app.get("plotly_chart")) == 4

    app.radio[0].set_value("成果 02 · Level / Delta / Innovation")
    app.run()

    assert not app.exception
    assert any("Level、Delta，还是 Innovation" in markdown.value for markdown in app.markdown)
    assert any(metric.value == "713" for metric in app.metric)
    assert len(app.get("plotly_chart")) == 5
    assert any(metric.value == "42.9%" for metric in app.metric)

    app.radio[0].set_value("成果 03 · Forecast Tournament v2")
    app.run()

    assert not app.exception
    assert any("什么才算真正的 Innovation" in markdown.value for markdown in app.markdown)
    assert any(metric.value == "M2" for metric in app.metric)
    assert any(metric.value == "+35.4%" for metric in app.metric)
    assert len(app.get("plotly_chart")) == 3

    app.radio[0].set_value("成果 04 · 修订风险与安全边际")
    app.run()

    assert not app.exception
    assert any("公告当下，这个信号值得相信多少" in markdown.value for markdown in app.markdown)
    assert any(metric.value == "0.166 pp" for metric in app.metric)
    assert any(metric.value == "97.4%" for metric in app.metric)
    assert any(metric.value == "0" and metric.label == "确认的经济差错" for metric in app.metric)
    assert len(app.get("plotly_chart")) == 3

    app.radio[0].set_value("成果 05 · 真实事件收益诊断")
    app.run()

    assert not app.exception
    empirical_root = app_path.parent / "data" / "processed" / "event_return_diagnostics"
    if empirical_root.exists():
        assert any("真实事件收益：先审计，再谈信号" in item.value for item in app.markdown)
    else:
        assert any("信号尚未成立，偏差已经可以计量" in item.value for item in app.markdown)
        assert any(metric.value == "49" and metric.label == "完整事件" for metric in app.metric)
    event_return_charts = len(app.get("plotly_chart"))
    assert event_return_charts in {4, 5, 6}
    if event_return_charts >= 5:
        assert any(
            metric.value == "32 / 38" and metric.label == "同事件完整 L/D/I"
            for metric in app.metric
        )

    app.radio[0].set_value("成果 06 · 管理层指引 Innovation")
    app.run()

    assert not app.exception
    assert any("市场预期，不再只是去年同期" in item.value for item in app.markdown)
    assert any(metric.value == "79.5%" and metric.label == "误差改善" for metric in app.metric)
    assert any(
        "诚实的描述性观察" in item.value
        or "收益诊断尚未执行" in item.value
        or "收益 Gate 未通过" in item.value
        for item in app.warning
    )

    app.radio[0].set_value("成果 07 · 前向一致预期账本")
    app.run()

    assert not app.exception
    assert any("从今天开始，保存市场当时知道什么" in item.value for item in app.markdown)
    forward_panel_root = app_path.parent / "data" / "processed" / "forward_consensus_panel"
    if forward_panel_root.exists():
        assert any(metric.value == "37" and metric.label == "前向财季快照" for metric in app.metric)
        assert any(
            metric.value == "通过" and metric.label == "Revision Gate" for metric in app.metric
        )
        prospective_root = (
            app_path.parent / "data" / "processed" / "prospective_market_innovation_preregistration"
        )
        if prospective_root.exists():
            assert any(
                metric.value == "37" and metric.label == "v3 注册事件" for metric in app.metric
            )
            assert any(
                metric.value == "37/37" and metric.label == "v3 快照覆盖" for metric in app.metric
            )
            assert any("不再回看 O3" in item.value for item in app.success)
        assert len(app.get("plotly_chart")) == 1
    else:
        assert any("本地尚无前向一致预期面板" in item.value for item in app.warning)

    app.radio[0].set_value("成果 08 · 前向 Innovation 流水线")
    app.run()

    assert not app.exception
    assert any("从“市场预期”走到“真正意外”" in item.value for item in app.markdown)
    forward_pipeline_root = app_path.parent / "data" / "processed" / "forward_innovation_ledger"
    if forward_pipeline_root.exists():
        assert any(
            metric.value == "0" and metric.label == "Raw Innovation" for metric in app.metric
        )
        assert any("时间顺序正确" in item.value for item in app.info)
        assert len(app.get("plotly_chart")) == 1
    else:
        assert any("本地尚无完整前向流水线" in item.value for item in app.warning)

    app.radio[0].set_value("成果 09 · Conditional Drift")
    app.run()

    assert not app.exception
    assert any("信息出现以后，市场反应够不够" in item.value for item in app.markdown)
    conditional_root = app_path.parent / "data" / "processed" / "conditional_drift_study"
    if conditional_root.exists():
        assert any(
            metric.value == "35" and metric.label == "完整同事件样本" for metric in app.metric
        )
        assert len(app.get("plotly_chart")) == 4
        assert any("历史开发诊断" in item.value for item in app.warning)
        assert any("完整 v2 被淘汰" in item.value for item in app.error)
        forward_root = (
            app_path.parent / "data" / "processed" / "conditional_drift_forward_candidate"
        )
        if forward_root.exists():
            assert any(
                metric.value == "16" and metric.label == "盲态预先评分" for metric in app.metric
            )
            assert any("没有读取结果收益" in item.value for item in app.success)
        holdout_root = (
            app_path.parent / "data" / "processed" / "conditional_drift_holdout_evaluation"
        )
        if holdout_root.exists():
            assert any(
                metric.value == "-0.106" and metric.label == "O3 留出 IC" for metric in app.metric
            )
            assert any("外部结果否定了当前候选" in item.value for item in app.error)
    else:
        assert any("本地尚无 Conditional Drift 产物" in item.value for item in app.info)

    app.radio[0].set_value("SEC 真实数据试验台")
    app.run()

    assert not app.exception
    assert any("真实披露数据试验台" in markdown.value for markdown in app.markdown)
