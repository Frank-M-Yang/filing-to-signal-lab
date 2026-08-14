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
    assert len(app.get("plotly_chart")) == 4

    app.radio[0].set_value("SEC 真实数据试验台")
    app.run()

    assert not app.exception
    assert any("真实披露数据试验台" in markdown.value for markdown in app.markdown)
