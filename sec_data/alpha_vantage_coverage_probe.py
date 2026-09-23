"""Isolated, outcome-free issuer coverage probes; no event admission authority."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from sec_data.alpha_vantage_budget import AlphaVantageBudgetError
from sec_data.alpha_vantage_consensus import (
    AlphaVantageConsensusClient,
    AlphaVantageConsensusError,
    AlphaVantageConsensusResult,
    alpha_vantage_provider_reason,
    alpha_vantage_response_status,
    parse_alpha_vantage_live_response,
)
from sec_data.consensus_probe_continuity import compare_coverage_probe_snapshots
from sec_data.filing_catalog import load_catalog_table

PROBE_VERSION = "alpha-vantage-coverage-probe-v1"
OBSERVATION_COLUMNS = [
    "coverage_probe_target_id",
    "cik",
    "ticker",
    "target_period_end",
    "captured_at",
    "capture_date",
    "revenue_estimate_average",
    "revenue_estimate_low",
    "revenue_estimate_high",
    "revenue_estimate_analyst_count",
    "source_occurrence",
    "source_sha256",
    "aggregate_fields_valid",
    "candidate_rejection_reason",
    "currency",
    "unit",
    "metric_scope_verified",
    "sec_acceptance_verified",
    "formal_model_admission",
]


def build_alpha_vantage_coverage_probe_plan(
    feasibility_root: Path,
    *,
    output_root: Path = Path("data/market/alpha_vantage_coverage_probe/targets"),
) -> AlphaVantageConsensusResult:
    """Freeze the unknown-company queue without changing any registered filing event."""
    report = _verified_report(feasibility_root, "prospective_cohort_feasibility_report.json")
    if (
        report.get("actual_values_loaded") is not False
        or report.get("return_values_loaded") is not False
    ):
        raise AlphaVantageConsensusError("Feasibility parent loaded outcome values")
    queue = load_catalog_table(feasibility_root, "unprobed_company_queue.parquet")
    required = {
        "probe_order",
        "core_asset_id",
        "company_name",
        "sec_cik",
        "sec_ticker",
        "coverage_status",
        "mapping_eligible",
        "queue_is_registration",
        "selection_uses_return_values",
    }
    if required - set(queue):
        raise AlphaVantageConsensusError("Probe queue lacks frozen metadata")
    if (
        not queue["coverage_status"].eq("unprobed").all()
        or not queue["mapping_eligible"].eq(True).all()
    ):
        raise AlphaVantageConsensusError("Probe queue is not exclusively eligible unknown issuers")
    if not queue[["queue_is_registration", "selection_uses_return_values"]].eq(False).all().all():
        raise AlphaVantageConsensusError("Outcome-selected or already registered queue")
    if (
        queue["probe_order"].isna().any()
        or queue["probe_order"].duplicated().any()
        or queue["sec_cik"].duplicated().any()
    ):
        raise AlphaVantageConsensusError("Ambiguous issuer or probe order")
    source_hash = _digest(feasibility_root / "output_manifest.json")
    targets = (
        queue.sort_values("probe_order", kind="mergesort")[
            ["probe_order", "core_asset_id", "company_name", "sec_cik", "sec_ticker"]
        ]
        .rename(columns={"sec_cik": "cik", "sec_ticker": "ticker"})
        .reset_index(drop=True)
    )
    targets["coverage_probe_target_id"] = [
        hashlib.sha256(
            json.dumps(
                {"parent": source_hash, "cik": row.cik, "ticker": row.ticker}, sort_keys=True
            ).encode()
        ).hexdigest()
        for row in targets.itertuples(index=False)
    ]
    targets["new_cohort_registered"] = False
    targets["formal_model_admission"] = False
    return _publish(
        output_root,
        "plan",
        {"feasibility_manifest_sha256": source_hash},
        {"coverage_probe_targets.parquet": targets},
        {
            "probe_targets": len(targets),
            "selection_uses_return_values": False,
            "actual_values_loaded": False,
            "return_values_loaded": False,
            "new_cohort_registered": False,
            "formal_model_admission": False,
        },
    )


def collect_alpha_vantage_coverage_probe(
    targets: pd.DataFrame,
    client: AlphaVantageConsensusClient,
    *,
    raw_root: Path = Path("data/market/alpha_vantage_coverage_probe"),
    offset: int = 0,
    limit: int = 4,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Capture a fixed slice once per UTC day, never replace or retry failed names.

    A shared request budget is mandatory. Transport exceptions are recorded only
    by safe category, never with a potentially credential-bearing exception URL.
    Ordinary response bytes are exact. Credential-echoing errors are redacted
    before persistence and labelled with both stored and original wire hashes.
    """
    if not 1 <= limit <= 4 or offset < 0:
        raise ValueError("Coverage probe requires nonnegative offset and a limit of 1-4")
    if client.request_budget is None:
        raise ValueError("Coverage probe requires a shared request budget")
    batch = targets.sort_values("probe_order", kind="mergesort").iloc[offset : offset + limit]
    result: dict[str, Any] = {
        "selected_targets": len(batch),
        "responses_preserved": 0,
        "failed_targets": 0,
        "requests_made": 0,
        "already_attempted_today": 0,
        "blocking_reason": None,
        "new_cohort_registered": False,
    }
    for number, row in enumerate(batch.itertuples(index=False), start=1):
        now = client.now()
        if now.tzinfo is None:
            raise AlphaVantageConsensusError("Probe request clock is not timezone-aware")
        now = now.astimezone(timezone.utc)
        day = now.date().isoformat()
        previous = list((raw_root / "snapshots" / day / str(row.ticker)).glob("*/manifest.json"))
        previous += list((raw_root / "transport_failures" / day / str(row.ticker)).glob("*.json"))
        if previous:
            result["already_attempted_today"] += 1
            continue
        try:
            response = client.get_earnings_estimates(str(row.ticker))
        except AlphaVantageBudgetError as exc:
            if str(exc) == "already_attempted_today_by_shared_ledger":
                result["already_attempted_today"] += 1
                continue
            result["blocking_reason"] = str(exc)
            break
        except httpx.HTTPError:
            result["requests_made"] += 1
            result["failed_targets"] += 1
            result["blocking_reason"] = "transport_failure"
            recorded = client.now().astimezone(timezone.utc)
            directory = (
                raw_root / "transport_failures" / recorded.date().isoformat() / str(row.ticker)
            )
            directory.mkdir(parents=True, exist_ok=True)
            _write_new_json(
                directory / f"{uuid.uuid4().hex}.json",
                {
                    "coverage_probe_target_id": row.coverage_probe_target_id,
                    "ticker": row.ticker,
                    "cik": row.cik,
                    "recorded_at": recorded.isoformat(),
                    "reason": "transport_failure",
                    "response_bytes_available": False,
                },
            )
            break
        result["requests_made"] += 1
        fetched = response.fetched_at
        if fetched.tzinfo is None:
            raise AlphaVantageConsensusError("Probe response availability clock is not aware")
        captured = fetched.astimezone(timezone.utc)
        digest = hashlib.sha256(response.content).hexdigest()
        directory = (
            raw_root
            / "snapshots"
            / captured.date().isoformat()
            / str(row.ticker)
            / uuid.uuid4().hex
        )
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "response.bin"
        path.write_bytes(response.content)
        status = alpha_vantage_response_status(response.content, response.status_code)
        _write_new_json(
            directory / "manifest.json",
            {
                "coverage_probe_target_id": row.coverage_probe_target_id,
                "ticker": row.ticker,
                "cik": row.cik,
                "fetched_at": captured.isoformat(),
                "capture_date": captured.date().isoformat(),
                "sha256": digest,
                "bytes": len(response.content),
                "wire_sha256": response.wire_sha256 or digest,
                "wire_bytes": response.wire_bytes
                if response.wire_bytes is not None
                else len(response.content),
                "credentials_redacted": response.credentials_redacted,
                "original_wire_body_retained": not response.credentials_redacted,
                "path": str(path.resolve()),
                "status": status,
                "status_code": response.status_code,
                "requested_url": response.requested_url,
                "response_url": response.response_url,
            },
        )
        result["responses_preserved"] += 1
        if status != "ok":
            result["failed_targets"] += 1
        if progress:
            progress(number, len(batch), str(row.coverage_probe_target_id))
        if status in {
            "rate_limited",
            "premium_endpoint_or_invalid_key",
            "http_401",
            "http_403",
            "http_429",
        }:
            result["blocking_reason"] = alpha_vantage_provider_reason(
                response.content, response.status_code
            )
            break
        if captured.date().isoformat() != day:
            result["blocking_reason"] = "utc_day_changed_restart_collection"
            break
    return result


def build_alpha_vantage_coverage_probe_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/market/alpha_vantage_coverage_probe"),
    output_root: Path = Path("data/processed/alpha_vantage_coverage_probe"),
) -> AlphaVantageConsensusResult:
    """Audit candidate fields and repeated-capture coverage, without SEC or actual joins."""
    _verified_report(plan_root, "coverage_probe_plan_report.json")
    targets = load_catalog_table(plan_root, "coverage_probe_targets.parquet")
    observations: list[dict[str, Any]] = []
    field_rejections: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    sources: list[str] = []
    identity = targets.set_index("coverage_probe_target_id")
    for path in sorted(raw_root.glob("snapshots/*/*/*/manifest.json")):
        manifest = _read_json(path)
        target_id = manifest["coverage_probe_target_id"]
        if target_id not in identity.index:
            raise AlphaVantageConsensusError("Unplanned issuer entered coverage raw evidence")
        target = identity.loc[target_id]
        if manifest["ticker"] != target["ticker"] or manifest["cik"] != target["cik"]:
            raise AlphaVantageConsensusError("Probe issuer remapped in raw evidence")
        content_path = Path(manifest["path"]).resolve()
        if not content_path.is_relative_to(raw_root.resolve()) or not content_path.is_file():
            raise AlphaVantageConsensusError("Probe response escaped its raw root")
        content = content_path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != manifest["sha256"]
            or len(content) != manifest["bytes"]
        ):
            raise AlphaVantageConsensusError("Changed probe response bytes")
        captured = pd.Timestamp(manifest["fetched_at"])
        if (
            captured.tzinfo is None
            or captured.tz_convert("UTC").date().isoformat() != manifest["capture_date"]
        ):
            raise AlphaVantageConsensusError("Probe capture date is backdated")
        if (
            path.parents[2].name != manifest["capture_date"]
            or path.parents[1].name != manifest["ticker"]
        ):
            raise AlphaVantageConsensusError("Probe storage date or ticker is inconsistent")
        if any(
            "apikey=" in str(manifest.get(field, "")).lower()
            for field in ("requested_url", "response_url")
        ):
            raise AlphaVantageConsensusError("Probe metadata contains a query-string key")
        status = alpha_vantage_response_status(content, manifest["status_code"])
        if status != manifest["status"]:
            raise AlphaVantageConsensusError("Probe response status relabelled")
        rows: list[dict[str, Any]] = []
        parser_status = status
        if status == "ok":
            try:
                rows = parse_alpha_vantage_live_response(
                    content,
                    expected_ticker=manifest["ticker"],
                    captured_at=manifest["fetched_at"],
                    source_sha256=manifest["sha256"],
                )
            except AlphaVantageConsensusError:
                parser_status = "symbol_or_schema_mismatch"
            if parser_status == "ok":
                field_rejections.extend(_future_quarter_field_rejections(content, manifest))
        future = [row for row in rows if row["forward_event_candidate"]]
        for row in future:
            numeric = [
                row[c]
                for c in (
                    "revenue_estimate_average",
                    "revenue_estimate_low",
                    "revenue_estimate_high",
                    "revenue_estimate_analyst_count",
                )
            ]
            valid = all(n is not None and math.isfinite(n) for n in numeric)
            if valid:
                mean, low, high, count = numeric
                valid = 0 < low <= mean <= high and count >= 1 and count == math.floor(count)
            observations.append(
                {
                    **{
                        c: row[c]
                        for c in (
                            "ticker",
                            "target_period_end",
                            "captured_at",
                            "revenue_estimate_average",
                            "revenue_estimate_low",
                            "revenue_estimate_high",
                            "revenue_estimate_analyst_count",
                            "source_occurrence",
                            "source_sha256",
                        )
                    },
                    "coverage_probe_target_id": target_id,
                    "cik": manifest["cik"],
                    "capture_date": manifest["capture_date"],
                    "aggregate_fields_valid": valid,
                    "candidate_rejection_reason": "none"
                    if valid
                    else "invalid_range_or_analyst_count",
                    "currency": None,
                    "unit": None,
                    "metric_scope_verified": False,
                    "sec_acceptance_verified": False,
                    "formal_model_admission": False,
                }
            )
        attempts.append(
            {
                "coverage_probe_target_id": target_id,
                "ticker": manifest["ticker"],
                "captured_at": manifest["fetched_at"],
                "capture_date": manifest["capture_date"],
                "source_sha256": manifest["sha256"],
                "status": parser_status,
                "provider_reason": alpha_vantage_provider_reason(content, manifest["status_code"]),
                "credentials_redacted": manifest.get("credentials_redacted", False),
                "quarterly_revenue_rows": len(rows),
                "future_quarter_rows": len(future),
            }
        )
        sources.append(_digest(path))
    for path in sorted(raw_root.glob("transport_failures/*/*/*.json")):
        manifest = _read_json(path)
        target_id = manifest["coverage_probe_target_id"]
        if (
            target_id not in identity.index
            or manifest["ticker"] != identity.loc[target_id, "ticker"]
        ):
            raise AlphaVantageConsensusError("Unplanned transport failure")
        attempts.append(
            {
                "coverage_probe_target_id": target_id,
                "ticker": manifest["ticker"],
                "captured_at": manifest["recorded_at"],
                "capture_date": path.parents[1].name,
                "source_sha256": None,
                "status": "transport_failure",
                "provider_reason": "transport_failure",
                "credentials_redacted": False,
                "quarterly_revenue_rows": 0,
                "future_quarter_rows": 0,
            }
        )
        sources.append(_digest(path))
    attempt_frame = pd.DataFrame(
        attempts,
        columns=[
            "coverage_probe_target_id",
            "ticker",
            "captured_at",
            "capture_date",
            "source_sha256",
            "status",
            "provider_reason",
            "credentials_redacted",
            "quarterly_revenue_rows",
            "future_quarter_rows",
        ],
    )
    observation_frame = pd.DataFrame(observations, columns=OBSERVATION_COLUMNS)
    rejection_frame = pd.DataFrame(
        field_rejections,
        columns=[
            "coverage_probe_target_id",
            "cik",
            "ticker",
            "target_period_end",
            "captured_at",
            "capture_date",
            "source_occurrence",
            "source_sha256",
            "revenue_estimate_average",
            "revenue_estimate_low",
            "revenue_estimate_high",
            "revenue_estimate_analyst_count",
            "field_rejection_reason",
        ],
    )
    coverage = targets.copy()
    coverage["coverage_status"] = "unprobed"
    coverage["valid_future_candidate_rows"] = 0
    coverage["max_valid_capture_dates_same_quarter"] = 0
    coverage["source_occurrence_stable_across_captures"] = False
    for index, target in enumerate(coverage.to_dict("records")):
        evidence = attempt_frame.loc[
            attempt_frame["coverage_probe_target_id"].eq(target["coverage_probe_target_id"])
        ]
        valid_candidates = observation_frame.loc[
            observation_frame["coverage_probe_target_id"].eq(target["coverage_probe_target_id"])
            & observation_frame["aggregate_fields_valid"].eq(True)
        ]
        if evidence.empty:
            continue
        coverage.loc[index, "coverage_status"] = "probed_no_valid_future_candidate"
        if not valid_candidates.empty:
            dates = valid_candidates.groupby("target_period_end")["capture_date"].nunique()
            coverage.loc[index, "valid_future_candidate_rows"] = len(valid_candidates)
            coverage.loc[index, "max_valid_capture_dates_same_quarter"] = int(dates.max())
            coverage.loc[index, "coverage_status"] = (
                "candidate_repeated" if dates.max() >= 2 else "candidate_single_capture"
            )
            occurrences = valid_candidates.groupby("target_period_end")[
                "source_occurrence"
            ].nunique()
            coverage.loc[index, "source_occurrence_stable_across_captures"] = bool(
                dates.ge(2).any() and occurrences.loc[dates.ge(2)].eq(1).all()
            )
    coverage["currency_unit_and_scope_verified"] = False
    coverage["analyst_membership_identity_verified"] = False
    coverage["new_cohort_registered"] = False
    coverage["formal_model_admission"] = False
    continuity = compare_coverage_probe_snapshots(attempt_frame, observation_frame, rejection_frame)
    snapshot_pairs = continuity.drop_duplicates("snapshot_pair_id")
    lineage = {
        "plan_manifest_sha256": _digest(plan_root / "output_manifest.json"),
        "source_manifest_sha256": sorted(sources),
        "parser_implementation_sha256": _digest(
            Path(__file__).with_name("alpha_vantage_consensus.py")
        ),
        "continuity_implementation_sha256": _digest(
            Path(__file__).with_name("consensus_probe_continuity.py")
        ),
    }
    report: dict[str, Any] = {
        "probe_targets": len(targets),
        "attempted_issuers": int(attempt_frame["ticker"].nunique()),
        "response_snapshots": int(attempt_frame["source_sha256"].notna().sum()),
        "candidate_issuers": int(coverage["valid_future_candidate_rows"].gt(0).sum()),
        "repeated_candidate_issuers": int(
            coverage["max_valid_capture_dates_same_quarter"].ge(2).sum()
        ),
        "unprobed_issuers": int(coverage["coverage_status"].eq("unprobed").sum()),
        "valid_future_candidate_rows": int(
            observation_frame["aggregate_fields_valid"].eq(True).sum()
        ),
        "attempt_status_counts": attempt_frame["status"].value_counts().to_dict(),
        "provider_reason_counts": attempt_frame["provider_reason"].value_counts().to_dict(),
        "credential_redacted_response_snapshots": int(
            attempt_frame["credentials_redacted"].eq(True).sum()
        ),
        "rejected_future_provider_field_rows": len(rejection_frame),
        "field_rejection_reason_counts": rejection_frame["field_rejection_reason"]
        .value_counts()
        .to_dict(),
        "adjacent_snapshot_pairs": len(snapshot_pairs),
        "snapshot_pairs_spanning_24_hours": int(snapshot_pairs["spans_24_hours"].eq(True).sum()),
        "minimum_adjacent_capture_hours": float(snapshot_pairs["elapsed_hours"].min())
        if not snapshot_pairs.empty
        else None,
        "maximum_adjacent_capture_hours": float(snapshot_pairs["elapsed_hours"].max())
        if not snapshot_pairs.empty
        else None,
        "identical_byte_snapshot_pairs": int(
            snapshot_pairs["response_hash_changed"].eq(False).sum()
        ),
        "comparable_quarter_pairs": int(continuity["aggregate_fields_comparable"].eq(True).sum()),
        "quarter_comparison_status_counts": continuity["comparison_status"]
        .value_counts()
        .to_dict(),
        "independent_information_verified": False,
        "currency_unit_and_scope_verified": False,
        "sec_acceptance_verified": False,
        "analyst_membership_identity_verified": False,
        "new_cohort_registered": False,
        "formal_model_admission": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
    }
    return _publish(
        output_root,
        "audit",
        lineage,
        {
            "coverage_probe_attempts.parquet": attempt_frame,
            "coverage_probe_future_candidates.parquet": observation_frame,
            "coverage_probe_field_rejections.parquet": rejection_frame,
            "coverage_probe_issuer_coverage.parquet": coverage,
            "coverage_probe_continuity.parquet": continuity,
        },
        report,
    )


def _future_quarter_field_rejections(
    content: bytes, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    # Keep explicit provider zeros as observed zeros in this rejection table.
    # They are never promoted to a positive-revenue analyst consensus.
    captured = pd.Timestamp(manifest["fetched_at"]).tz_convert("UTC")
    records: list[dict[str, Any]] = []
    document = json.loads(content)
    for occurrence, item in enumerate(document["estimates"]):
        if not isinstance(item, dict) or str(item.get("horizon", "")).lower() != "fiscal quarter":
            continue
        source_date = item.get("date")
        if not isinstance(source_date, str):
            continue
        period = pd.to_datetime(source_date, utc=True, errors="coerce")
        if not isinstance(period, pd.Timestamp) or period.normalize() < captured.normalize():
            continue
        mean, low, high, count = (
            _optional_number(item.get(field))
            for field in (
                "revenue_estimate_average",
                "revenue_estimate_low",
                "revenue_estimate_high",
                "revenue_estimate_analyst_count",
            )
        )
        if mean == 0 and count == 0:
            reason = "zero_revenue_and_zero_analyst_count_no_valid_coverage"
        elif mean is None or mean <= 0:
            reason = "positive_revenue_consensus_missing"
        elif count is None or count < 1 or count != math.floor(count):
            reason = "positive_integer_analyst_count_missing"
        elif low is None or high is None or not 0 < low <= mean <= high:
            reason = "valid_revenue_range_missing"
        else:
            continue
        records.append(
            {
                "coverage_probe_target_id": manifest["coverage_probe_target_id"],
                "cik": manifest["cik"],
                "ticker": manifest["ticker"],
                "target_period_end": period.date().isoformat(),
                "captured_at": manifest["fetched_at"],
                "capture_date": manifest["capture_date"],
                "source_occurrence": occurrence,
                "source_sha256": manifest["sha256"],
                "revenue_estimate_average": mean,
                "revenue_estimate_low": low,
                "revenue_estimate_high": high,
                "revenue_estimate_analyst_count": count,
                "field_rejection_reason": reason,
            }
        )
    return records


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _publish(
    output_root: Path,
    kind: str,
    lineage: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    details: dict[str, Any],
) -> AlphaVantageConsensusResult:
    implementation = _digest(Path(__file__))
    payload = {
        "kind": kind,
        "version": PROBE_VERSION,
        "implementation_sha256": implementation,
        **lineage,
    }
    version = (
        PROBE_VERSION
        + f"-{kind}-"
        + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    )
    target = output_root / version
    report_name = f"coverage_probe_{kind}_report.json"
    if (target / "output_manifest.json").exists():
        report = _verified_report(target, report_name)
        for name, artifact in _read_json(target / "output_manifest.json")["artifacts"].items():
            if _digest(target / name) != artifact["sha256"]:
                raise AlphaVantageConsensusError("Changed existing coverage output artifact")
        return AlphaVantageConsensusResult(target.resolve(), report)
    if target.exists():
        raise AlphaVantageConsensusError("Incomplete coverage probe bundle exists")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts = {}
    for name, frame in frames.items():
        frame.to_parquet(staging / name, index=False)
        artifacts[name] = {"sha256": _digest(staging / name), "rows": len(frame)}
    report = {
        **details,
        **payload,
        "dataset_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    if kind == "audit":
        readable = staging / "coverage_probe_findings.md"
        readable.write_text(
            _render_findings(report, frames["coverage_probe_issuer_coverage.parquet"]),
            encoding="utf-8",
        )
        artifacts[readable.name] = {"sha256": _digest(readable), "bytes": readable.stat().st_size}
    _write_new_json(staging / report_name, report)
    _write_new_json(
        staging / "output_manifest.json",
        {
            **payload,
            "dataset_version": version,
            "artifacts": artifacts,
            "report_sha256": _digest(staging / report_name),
        },
    )
    staging.rename(target)
    return AlphaVantageConsensusResult(target.resolve(), report)


def _render_findings(report: dict[str, Any], coverage: pd.DataFrame) -> str:
    lines = [
        "# 收入一致预期覆盖与连续性探测 / Revenue consensus coverage and continuity probe",
        "",
        f"版本：{report['dataset_version']}",
        f"报告生成时间：{report['generated_at']}",
        "",
        f"候选名单 {report['probe_targets']} 家，已尝试 {report['attempted_issuers']} 家；"
        f"{report['candidate_issuers']} 家观察到字段通过基础检查的未来财季收入候选值，"
        f"{report['unprobed_issuers']} 家尚未探测。",
        f"同一财季至少两个 UTC 抓取日的候选公司：{report['repeated_candidate_issuers']}。",
        "",
        "| 公司 | Ticker | 状态 | 有效未来候选行 | 同一财季最大抓取日数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in coverage.loc[coverage["coverage_status"].ne("unprobed")].to_dict("records"):
        lines.append(
            f"| {row['company_name']} | {row['ticker']} | {row['coverage_status']} | "
            f"{row['valid_future_candidate_rows']} | "
            f"{row['max_valid_capture_dates_same_quarter']} |"
        )
    lines += [
        "",
        "## 重复捕获是否增加了信息",
        "",
        f"相邻公司快照对：{report['adjacent_snapshot_pairs']}；其中相隔至少 24 小时："
        f"{report['snapshot_pairs_spanning_24_hours']}；完全相同响应字节的快照对："
        f"{report['identical_byte_snapshot_pairs']}。",
        f"相邻捕获最短 / 最长间隔（小时）：{report['minimum_adjacent_capture_hours']} / "
        f"{report['maximum_adjacent_capture_hours']}。",
        f"同一财季字段可比较对：{report['comparable_quarter_pairs']}；比较结果："
        f"{report['quarter_comparison_status_counts']}。",
        "逐对证据见 coverage_probe_continuity.parquet。跨 UTC 日期不等于经过一天；"
        "24 小时只是描述性标记，不是新研究门槛，也不证明信息独立。",
        "整份响应字节改变可能仅来自 EPS 或其他财季；收入均值、区间、分析师数量"
        "和来源行位置分别比较。失败、缺行或重复财季会阻断该对，不跳过中间坏快照。",
        "均值变化只是供应商聚合字段变化；分析师人数不变，也不能确认成员不变。",
        "",
        "## 这证明什么、不证明什么",
        "",
        "这证明所尝试公司的 provider 接口和未来财政季度收入预测字段具有一定可采集性。",
        "基础检查仅包括 future period、正收入均值、上下界顺序和正整数分析师数量。",
        f"未来财季字段拒绝行数：{report['rejected_future_provider_field_rows']}；"
        "逐行原因与原值见 coverage_probe_field_rejections.parquet。",
        "收入预测与分析师数量同时为 0 的供应商行视为没有有效覆盖，"
        "不当成公司真实收入或有效零收入预测；保留原始零值，不补造分析师或改填数据。",
        "它不验证币种、单位、收入会计口径、SEC 公告时钟或分析师成员身份。",
        "供应商响应没有明确提供币种和单位：保留缺失，不擅自推断美元或补造口径。",
        "一次成功响应不等于合格研究事件；重复响应也不证明分析师成员不变或存在可交易 alpha。",
        "捕获时间仅是本系统收到响应的时间，不是市场首次获知信息的时间。",
        f"供应商拒绝原因（细分）：{report['provider_reason_counts']}；"
        f"凭据脱敏响应：{report['credential_redacted_response_snapshots']}。",
        "供应商若在错误正文回显密钥，只保留脱敏正文和原响应指纹；"
        "脱敏正文不标为原始逐字节副本，也不作为财务数据候选。",
        "目标财季使用响应中实际出现的 date；没有凭空构造 2027 年财季或回填过去未捕获日期。",
        "",
        "## 保护边界",
        "",
        "全部新响应与失败保存在隔离目录；原 37 个注册事件、质量状态和门槛未修改。",
        "没有 actual、Innovation、价格、收益、模型训练、订单或新队列注册。",
        "共享额度统计包括本仓库新请求尝试及可追溯的旧响应；不声称知道账户在其他地方的调用。",
        "名单每批固定，不因缺字段、错误或收益表现替换公司；默认同 UTC 日不重复调用同公司。",
        "",
        "## 下一步",
        "",
        "在不挤占原 20 家观察的预算内，继续本批的真实时间观察；"
        "同时查看实际间隔与字段变化，不能把跨日重复抓取当作新增独立事件。",
        "然后明确核验币种、单位、收入定义、财政周期和首次可得截止，才考虑另行冻结新事件。",
        "继续探测下一固定切片需要显式 --offset；失败记录保留，不能为凑样本自动换公司。",
        "本结果是数据覆盖可行性成果，不是信号有效性或赚钱证据。",
        "",
    ]
    return "\n".join(lines)


def _verified_report(root: Path, name: str) -> dict[str, Any]:
    if _read_json(root / "output_manifest.json").get("report_sha256") != _digest(root / name):
        raise AlphaVantageConsensusError("Changed coverage parent report")
    return _read_json(root / name)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
