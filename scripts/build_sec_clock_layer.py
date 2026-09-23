"""Offline, immutable acceptance-clock view and fixed-spec downstream replay."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.clock_layer import (
    HISTORY_COLUMNS,
    VERSION,
    apply_history_clocks,
    audit_replay_chronology,
    build_clock_layer,
    calendar_windows,
    compare_histories,
    replay_history,
)
from sec_data.disclosure_audit import digest_file, verify_file
from sec_data.pilot_filing_features import PilotFilingFeatureConfig


def _artifact(root: Path, manifest_hash: str, name: str) -> Path:
    verify_file(root / "output_manifest.json", manifest_hash)
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    value = manifest["artifacts"][name]
    verify_file(root / name, value if isinstance(value, str) else value["sha256"])
    return root / name


def _json(root: Path, digest: str, name: str) -> Any:
    return json.loads(_artifact(root, digest, name).read_bytes())


def _encode(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # JSON serialization normalizes NaT, nullable scalars and UTC timestamps.
    result: list[dict[str, Any]] = json.loads(frame.to_json(orient="records", date_format="iso"))
    return result


def build_layer(config_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    """Replay identical events/values/parameters; expose separate proxy-only outputs."""
    config = json.loads(config_path.read_bytes())
    if (
        config["version"] != VERSION
        or not config["replay_fixed_specification"]
        or any(
            config[k] for k in ["overwrite_originals", "return_values_allowed", "model_admission"]
        )
        or config["numeric_absolute_tolerance"] != 1e-10
        or config["holding_sessions"] != 20
    ):
        raise ValueError("Unsupported or weakened clock replay contract")
    root = Path(config["provenance_root"])
    digest = config["provenance_manifest_sha256"]
    report_before = _json(root, digest, "report.json")
    if report_before["plan_manifest_sha256"] != config["provenance_plan_manifest_sha256"]:
        raise ValueError("Provenance audit/plan mismatch")
    provenance_plan = _json(
        Path(config["provenance_plan_root"]), config["provenance_plan_manifest_sha256"], "plan.json"
    )
    plan_config = provenance_plan["config"]
    plan = _json(Path(plan_config["plan_root"]), plan_config["plan_manifest_sha256"], "plan.json")
    source = plan["config"]
    evidence = _json(root, digest, "source_clock_provenance.json")
    proposed = _json(root, digest, "proposed_acceptance_overlay.json")
    if proposed != [
        r
        for r in evidence
        if r["resolution_status"] == "source_label_conflict_resolved_by_header_and_index"
    ]:
        raise ValueError("Proposed overlay differs from provenance")
    catalog_path = _artifact(
        Path(source["catalog_root"]), source["catalog_manifest_sha256"], "filing_catalog.parquet"
    )
    catalog = pd.read_parquet(
        catalog_path,
        columns=[
            "cik",
            "accession_number",
            "form",
            "accepted_at",
            "source_member",
            "source_zip_sha256",
            "source_row_index",
        ],
    )
    keys = {(r["cik"], r["accession_number"]) for r in evidence}
    catalog = catalog.loc[
        pd.MultiIndex.from_frame(catalog[["cik", "accession_number"]]).isin(list(keys))
    ]
    layer = build_clock_layer(catalog, evidence)

    feature_root = Path(source["feature_root"])
    feature_hash = source["feature_manifest_sha256"]
    verify_file(feature_root / "output_manifest.json", feature_hash)
    feature_manifest = json.loads((feature_root / "output_manifest.json").read_bytes())
    verify_file(feature_root / "validation_feature_report.json", feature_manifest["report_sha256"])
    feature_report = json.loads((feature_root / "validation_feature_report.json").read_bytes())
    if feature_report["catalog_manifest_sha256"] != source["catalog_manifest_sha256"]:
        raise ValueError("Feature source is not the pinned catalog")
    cfg = PilotFilingFeatureConfig(**feature_report["config"])
    history = pd.read_parquet(
        _artifact(feature_root, feature_hash, "quarterly_feature_history.parquet"),
        columns=HISTORY_COLUMNS,
    )
    implementation = {
        p: digest_file(Path(p))
        for p in [
            "scripts/build_sec_clock_layer.py",
            "sec_data/clock_layer.py",
            "event_panel.py",
            "sec_data/event_cleaning.py",
            "sec_data/pilot_filing_features.py",
            "sec_data/disclosure_audit.py",
            "scripts/build_sec_disclosure_audit.py",
        ]
    }
    frozen = {
        "config": config,
        "config_sha256": digest_file(config_path),
        "source": source,
        "forecast_config": feature_report["config"],
        "implementation_sha256": implementation,
        "history_events": len(history),
        "fixed_sample_events": len(plan["events"]),
        "sample_reselection": False,
        "return_values_loaded": False,
    }
    frozen_root = write_bundle(
        output_root, "sec-clock-layer-plan-v1", {"plan.json": _encode(frozen)}
    )
    print(
        f"CLOCK LAYER PLAN history={len(history)} sample={len(plan['events'])} "
        f"output={frozen_root}",
        flush=True,
    )
    baseline = replay_history(history, cfg)
    baseline_counts, baseline_diff = compare_histories(history, baseline)
    if not baseline_diff.empty:
        failure = {"status": "baseline_not_reproduced", "counts": baseline_counts, **frozen}
        failure_root = write_bundle(
            output_root,
            "sec-clock-layer-replay-failure-v1",
            {
                "report.json": _encode(failure),
                "differences.json": _encode(_records(baseline_diff)),
            },
        )
        raise ValueError(
            f"Original baseline not reproduced; no corrected result admitted: {failure_root}"
        )
    print("CLOCK BASELINE reproduced=True", flush=True)
    revised = replay_history(apply_history_clocks(history, layer), cfg)
    changes, difference = compare_histories(baseline, revised)
    chronology = audit_replay_chronology(revised)
    clock_fields = [
        "actual_available_at",
        "forecast_feature_cutoff",
        "forecast_training_end",
        "forecast_generated_at",
        "forecast_error_history_end",
    ]
    panel_cols = [
        "event_id",
        "company_id",
        "filing_accession",
        *clock_fields,
        "gm_forecast",
        "forecast_sigma",
        "gm_innovation_raw",
        "gm_innovation_std",
    ]
    panel = revised[panel_cols].copy()
    panel["clock_layer_version"] = VERSION
    panel["model_admission"] = False
    panel["trading_admission"] = False
    panel["public_time_certified"] = False
    panel["clock_basis"] = "mixed_reviewed_and_unreviewed_acceptance_proxies"
    # Keep every frozen sample event, including those not forecast-ready.
    sample = pd.DataFrame(plan["events"])
    sample = sample.merge(
        panel.rename(columns={"company_id": "cik"}),
        on=["cik", "filing_accession"],
        how="left",
        validate="one_to_one",
    )
    if sample.event_id.isna().any():
        raise ValueError("Frozen sample event missing from replay")

    market = pd.read_parquet(
        _artifact(
            Path(source["market_root"]),
            source["market_manifest_sha256"],
            "validation_market_state_features.parquet",
        ),
        columns=["validation_event_id", "cik", "exchange_mic", "market_feature_available_at"],
    )
    sessions = pd.read_parquet(
        _artifact(
            Path(config["calendar_root"]),
            config["calendar_manifest_sha256"],
            "historical_exchange_sessions.parquet",
        )
    )
    windows = []
    for row in layer:
        if not row["clock_changed"]:
            continue
        event = [e for e in plan["events"] if e["cik"] == row["cik"]]
        if len(event) != 1:
            raise ValueError("Clock issuer has ambiguous frozen event identity")
        metadata = market.loc[
            market.validation_event_id.eq(event[0]["validation_event_id"])
            & market.cik.eq(row["cik"])
        ]
        if len(metadata) != 1:
            raise ValueError("Market venue mapping is not unique")
        venue = str(metadata.iloc[0].exchange_mic)
        old = calendar_windows(row["catalog_accepted_at"], venue, sessions)
        new = calendar_windows(row["experimental_acceptance_proxy_at"], venue, sessions)
        windows.append(
            {
                "cik": row["cik"],
                "ticker": event[0]["ticker"],
                "accession_number": row["accession_number"],
                "form": row["form"],
                "target_filing": row["accession_number"] == event[0]["filing_accession"],
                "exchange_mic": venue,
                "venue_basis": "frozen_target_event_venue_for_clock_sensitivity_only",
                "old": old,
                "new": new,
                "session_relation_changed": old["session_relation"] != new["session_relation"],
                "legacy_entry_changed": old["legacy_second_future_open"]["entry_at"]
                != new["legacy_second_future_open"]["entry_at"],
                "conservative_entry_changed": old["second_session_after_local_date"]["entry_at"]
                != new["second_session_after_local_date"]["entry_at"],
                "daily_bracket_changed": old["daily_close_bracket"] != new["daily_close_bracket"],
                "public_time_certified": False,
                "trading_admission": False,
            }
        )
    # Test existing pre-event market cutoff; do not pretend market features were rebuilt.
    sample = sample.merge(
        market, on=["validation_event_id", "cik"], validate="one_to_one", how="left"
    )
    sample["market_cutoff_before_replayed_filing_proxy"] = pd.to_datetime(
        sample.market_feature_available_at, utc=True
    ) < pd.to_datetime(sample.actual_available_at, utc=True)
    sample["market_cutoff_status"] = sample["market_cutoff_before_replayed_filing_proxy"].map(
        {True: "strictly_before_filing_proxy_only", False: "at_or_after_filing_proxy"}
    )
    sample.loc[sample.market_feature_available_at.isna(), "market_cutoff_status"] = "missing"
    result = {
        "version": VERSION,
        "plan_manifest_sha256": digest_file(frozen_root / "output_manifest.json"),
        "implementation_sha256": implementation,
        "clock_layer_rows": len(layer),
        "overlay_rows_applied_in_isolated_view": sum(r["clock_changed"] for r in layer),
        "baseline_reproduced": True,
        "baseline_difference_counts": baseline_counts,
        "history_events": len(history),
        "fixed_sample_events": len(sample),
        "numeric_absolute_tolerance": config["numeric_absolute_tolerance"],
        "replay_difference_counts": changes,
        "chronology": chronology,
        "session_class_changes": sum(r["session_relation_changed"] for r in windows),
        "legacy_entry_changes": sum(r["legacy_entry_changed"] for r in windows),
        "conservative_entry_changes": sum(r["conservative_entry_changed"] for r in windows),
        "daily_bracket_changes": sum(r["daily_bracket_changed"] for r in windows),
        "original_catalog_overwritten": False,
        "market_features_rebuilt": False,
        "return_values_loaded": False,
        "hyperparameters_changed": False,
        "public_times_certified": 0,
        "model_admission": False,
        "trading_admission": False,
        "scope": (
            "fixed accounting selection; retrospective acceptance-proxy sensitivity, "
            "not first-public validation"
        ),
    }
    files = {
        "report.json": _encode(result),
        "clock_layer.json": _encode(layer),
        "replay_differences.json": _encode(_records(difference)),
        "sample_chronology.json": _encode(_records(sample)),
        "window_sensitivity.json": _encode(windows),
        "readout.md": render_readout(result, windows).encode(),
    }
    buffer = io.BytesIO()
    panel.to_parquet(buffer, index=False)
    files["disarmed_replay_panel.parquet"] = buffer.getvalue()
    return write_bundle(output_root, VERSION, files), result


def render_readout(report: dict[str, Any], windows: list[dict[str, Any]]) -> str:
    """Explain clock-only effects without presenting repaired proxies as public times."""
    counts = report["replay_difference_counts"]
    lines = [
        "# 独立时间层与下游重放",
        "",
        f"原基线复现：{report['baseline_reproduced']}；历史事件 {report['history_events']}；"
        f"固定审查样本 {report['fixed_sample_events']}。",
        f"独立视图应用 {report['overlay_rows_applied_in_isolated_view']} 条修正，原主库不变。",
        f"实际值代理时钟改变 {counts['actual_available_at']} 条；"
        f"特征截止改变 {counts['forecast_feature_cutoff']} 条；"
        f"预测生成改变 {counts['forecast_generated_at']} 条。",
        f"超过绝对容差1e-10的预测值变化 {counts['gm_forecast']} 条；"
        f"sigma变化 {counts['forecast_sigma']} 条；"
        f"标准化Innovation变化 {counts['gm_innovation_std']} 条。",
        "",
        "|公司/表单|旧/新时段|旧算法第二个未来开盘|按公告当地日期之后第二个交易日|",
        "|---|---|---|---|",
    ]
    for row in windows:
        old, new = row["old"], row["new"]
        lines.append(
            f"|{row['ticker']} {row['form']} {row['accession_number']}|"
            f"{old['session_relation']} → {new['session_relation']}|"
            f"{old['legacy_second_future_open']['entry_at']} → "
            f"{new['legacy_second_future_open']['entry_at']}|"
            f"{old['second_session_after_local_date']['entry_at']} → "
            f"{new['second_session_after_local_date']['entry_at']}|"
        )
    lines += [
        "",
        "所有表内时间为UTC。持有20个交易日（入场日计入），未读取价格或收益。",
        "旧算法对盘前公告把当天开盘计作第一个未来开盘；严格按公告当地日期之后的第二个交易日则不计当天。两种规则分别列示，未暗改生产策略。",
        "日收盘括号不是纯事件反应：盘中公告之前的同日行情也可能包含其中。",
        "SEC接收时间仍不是首次公开时间。其余历史记录仍沿用未逐份审核的接收代理；旧财务字段选择固定，不等于完成全量原文/会计复核。",
        "重放预测只是固定规格的历史重建，不是当年实际生成的预测日志。未改超参数、未换样本、未认证Alpha、未批准模型或交易。",
        "下一步：明确统一执行约定、补齐会计口径/首次公开证据，再决定是否开启新版收益账本；不得沿用旧收益声称新模型有效。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Build the frozen clock-only comparison without networking or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/sec_clock_layer_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/sec_clock_layer"))
    args = parser.parse_args()
    root, report = build_layer(args.config, args.output_root)
    print(
        f"CLOCK LAYER DONE baseline={report['baseline_reproduced']} "
        f"history={report['history_events']} sample={report['fixed_sample_events']} "
        f"legacy_entry_changes={report['legacy_entry_changes']} admitted=False output={root}"
    )


if __name__ == "__main__":
    main()
