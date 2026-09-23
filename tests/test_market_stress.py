"""Leakage, source-vintage and deterministic negative-control tests."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_stress import (
    FEATURES,
    MODELS,
    StressConfig,
    forecast_panel,
    make_panel,
    paired_block_interval,
    score_forecasts,
)
from market_stress_sources import (
    INDUSTRIES,
    freeze_protocol,
    parse_daily_zip,
    verify_bundle,
)
from scripts.build_sec_disclosure_audit import write_bundle


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    dates = pd.bdate_range("2015-01-01", "2022-01-10")
    market = rng.normal(0, 0.01, len(dates))
    frame = pd.DataFrame({"market_return": market}, index=dates)
    for name in INDUSTRIES:
        frame[f"industry_{name}"] = 0.7 * market + rng.normal(0, 0.008, len(dates))
    return frame


def _config() -> StressConfig:
    return replace(
        StressConfig(), train_start="2015-01-01", minimum_train_rows=200, bootstrap_repetitions=50
    )


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("test.csv", text)
    return buffer.getvalue()


def test_target_excludes_origin_and_uses_exact_next_five() -> None:
    frame, config = _returns(), _config()
    panel = make_panel(frame, config)
    position = 100
    actual = frame.market_return.iloc[position + 1 : position + 6]
    assert panel.target_variance.iloc[position] == pytest.approx(252 * np.mean(actual**2))
    assert panel.future_return.iloc[position] == pytest.approx(np.prod(1 + actual) - 1)
    assert panel.target_start.iloc[position] == frame.index[position + 1]
    assert panel.target_end.iloc[position] == frame.index[position + 5]
    expected_log = np.log(
        252 * np.mean(frame.market_return.iloc[position - 19 : position + 1] ** 2)
    )
    assert panel.log_rv20.iloc[position] == pytest.approx(expected_log)


def test_all_45_correlations_and_no_diagonals() -> None:
    frame = _returns()
    panel = make_panel(frame, _config())
    matrix = frame.iloc[81:101, 1:].corr().to_numpy()
    assert panel.corr20.iloc[100] == pytest.approx(matrix[np.triu_indices(10, k=1)].mean())


def test_future_mutation_cannot_change_earlier_features_or_forecasts() -> None:
    frame, config = _returns(), _config()
    panel = make_panel(frame, config)
    forecasts, _ = forecast_panel(panel, config)
    changed = frame.copy()
    changed.loc[changed.index > "2020-03-02"] *= 2
    new_panel = make_panel(changed, config)
    new_forecasts, _ = forecast_panel(new_panel, config)
    pd.testing.assert_frame_equal(
        panel.loc[:"2020-03-02", list(FEATURES)], new_panel.loc[:"2020-03-02", list(FEATURES)]
    )
    pd.testing.assert_frame_equal(forecasts.loc[:"2020-03-02"], new_forecasts.loc[:"2020-03-02"])


def test_current_origin_mutation_cannot_change_its_monthly_fit() -> None:
    frame, config = _returns(), _config()
    _, before = forecast_panel(make_panel(frame, config), config)
    frame.loc[frame.index >= "2020-03-02"] *= 3
    _, after = forecast_panel(make_panel(frame, config), config)
    assert [fit for fit in before if fit["month"] <= "2020-03"] == [
        fit for fit in after if fit["month"] <= "2020-03"
    ]


def test_training_labels_strictly_matured_and_identical_scored_sample() -> None:
    config = _config()
    panel = make_panel(_returns(), config)
    forecasts, fits = forecast_panel(panel, config)
    assert (forecasts.training_label_end < forecasts.model_fit_origin).all()
    assert len(fits) == 36
    ledger, summary = score_forecasts(panel, forecasts, config)
    assert ledger[list(MODELS)].notna().all().all()
    assert summary["scored_origins"] == len(ledger)
    assert summary["trading_admission"] is False
    assert summary["live_availability_certified"] is False
    assert summary["manifold_model_used"] is False


def test_missing_values_do_not_compress_windows_or_change_other_arm_sample() -> None:
    frame = _returns()
    frame.iloc[110, 1] = np.nan
    panel = make_panel(frame, _config())
    assert len(panel) == len(frame)
    assert panel.corr60.iloc[110:170].isna().all()
    assert panel.corr60.iloc[170:].notna().all()
    assert panel.log_rv60.iloc[110:170].notna().all()
    assert not panel.feature_complete.iloc[110:170].any()
    frame.iloc[110, 0] = np.nan
    panel = make_panel(frame, _config())
    assert not panel.target_complete.iloc[105:110].any()
    assert panel.target_complete.iloc[110]


def test_missing_test_labels_do_not_suppress_forecast_generation() -> None:
    config = _config()
    frame = _returns().loc[:"2021-12-31"]
    panel = make_panel(frame, config)
    forecasts, _ = forecast_panel(panel, config)
    ledger, summary = score_forecasts(panel, forecasts, config)
    assert forecasts.index.max() == pd.Timestamp("2021-12-31")
    assert len(forecasts) - len(ledger) == 5
    assert summary["forecasts_without_complete_outcomes"] == 5


def test_constant_industry_is_not_silently_dropped() -> None:
    frame = _returns()
    frame["industry_Other"] = 0
    panel = make_panel(frame, _config())
    assert not panel.feature_complete.any()
    with pytest.raises(ValueError, match="No forecast"):
        forecast_panel(panel, _config())


def test_training_only_standardization() -> None:
    panel = make_panel(_returns(), _config())
    _, fits = forecast_panel(panel, _config())
    first = fits[0]
    train = panel.loc[
        (panel.index >= "2015-01-01")
        & panel.feature_complete
        & panel.target_complete
        & (panel.target_end < first["first_forecast_origin"])
    ]
    fit = first["models"][MODELS[2]]
    np.testing.assert_allclose(fit["training_mean"], train[list(FEATURES)].mean())
    np.testing.assert_allclose(fit["training_std"], train[list(FEATURES)].std(ddof=0))


def test_block_interval_deterministic_and_preserves_pairing() -> None:
    baseline = np.arange(1, 121, dtype=float)
    same = paired_block_interval(baseline, baseline, _config())
    assert same["candidate_minus_baseline_95pct"] == [0.0, 0.0]
    assert same == paired_block_interval(baseline, baseline, _config())
    less = paired_block_interval(baseline * 0.9, baseline, _config())
    np.testing.assert_allclose(less["relative_loss_change_95pct"], [-0.1, -0.1])
    assert paired_block_interval(baseline[:10], baseline[:10], _config())["available"] is False


def test_percent_conversion_market_plus_rf_and_missing_not_zero() -> None:
    body = _zip(
        "description\n,Mkt-RF,SMB,HML,RF\n20200102,1,0,0,0.02\n20200103,-99.99,0,0,0.02\n\nAnnual\n2020,3,3,3,3"
    )
    frame = parse_daily_zip(body, "market", "2020-01-01", "2020-01-10")
    assert frame.iloc[0, 0] == pytest.approx(0.0102)
    assert np.isnan(frame.iloc[1, 0])
    assert len(frame) == 2


def test_only_value_weighted_daily_table_is_read() -> None:
    header = "," + ",".join(INDUSTRIES)
    body = _zip(
        "Average Value Weighted Returns -- Daily\n"
        + header
        + "\n20200102,"
        + ",".join(["2"] * 10)
        + "\n\nAverage Equal Weighted Returns -- Daily\n"
        + header
        + "\n20200102,"
        + ",".join(["99"] * 10)
    )
    frame = parse_daily_zip(body, "industries", "2020-01-01", "2020-01-10")
    np.testing.assert_allclose(frame, 0.02)


def test_duplicate_dates_rejected() -> None:
    body = _zip(",Mkt-RF,SMB,HML,RF\n20200102,1,0,0,0\n20200102,1,0,0,0")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_daily_zip(body, "market", "2020-01-01", "2020-01-10")
    frame = _returns()
    with pytest.raises(ValueError, match="unique"):
        make_panel(pd.concat([frame.iloc[:1], frame]), _config())


def test_frozen_protocol_reused_and_tampering_rejected(tmp_path: Path) -> None:
    source = Path("docs/market_stress_pilot_v1.json")
    frozen = freeze_protocol(source, tmp_path)
    assert freeze_protocol(source, tmp_path) == frozen
    record = json.loads(verify_bundle(frozen)["freeze_record.json"])
    assert not record["numeric_test_outcomes_read_at_freeze"]
    (frozen / "protocol.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(frozen)


def test_insufficient_training_blocks_instead_of_shortening_history() -> None:
    config = replace(_config(), minimum_train_rows=10000)
    with pytest.raises(ValueError, match="Insufficient"):
        forecast_panel(make_panel(_returns(), config), config)


def test_manifest_and_content_edit_cannot_silently_rewrite_bundle(tmp_path: Path) -> None:
    import hashlib

    root = write_bundle(tmp_path, "fixture", {"value.json": b"original"})
    (root / "value.json").write_bytes(b"replacement")
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    manifest["artifacts"]["value.json"] = hashlib.sha256(b"replacement").hexdigest()
    (root / "output_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable directory"):
        verify_bundle(root)


def test_parser_ignores_numeric_outcomes_outside_frozen_period() -> None:
    body = _zip(
        ",Mkt-RF,SMB,HML,RF\n20081231,not_read,0,0,0\n20200102,1,0,0,0\n20230102,not_read,0,0,0"
    )
    frame = parse_daily_zip(body, "market", "2009-01-01", "2022-01-10")
    assert len(frame) == 1


def test_no_high_risk_denominator_is_not_a_successful_zero_rate() -> None:
    config = _config()
    panel = make_panel(_returns(), config)
    predictions, _ = forecast_panel(panel, config)
    predictions["alarm_threshold_variance"] = 1e6
    _, summary = score_forecasts(panel, predictions, config)
    for name in MODELS:
        stats = summary["slices"]["all"]["models"][name]
        assert stats["recall_high_risk"] is None
        assert stats["false_discovery_rate"] is None
