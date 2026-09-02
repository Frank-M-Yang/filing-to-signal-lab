"""Plan point-in-time market evidence for prospective shadow intentions.

This layer maps every frozen event to a registered security candidate and, only
after SEC acceptance exists, creates timestamped identity, price, split, and
dividend query plans.  A current ticker is never treated as a verified future
point-in-time identity, and this planning layer never reads returns.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError

SHADOW_MARKET_EVIDENCE_VERSION: Final[str] = "shadow-market-evidence-plan-v1"
_EXCHANGE_MAP: Final[dict[str, str]] = {"NASDAQ": "XNAS", "NYSE": "XNYS"}


class ShadowMarketEvidenceError(FilingCatalogError):
    """Raised when a forward market-evidence plan violates its clocks or scope."""


@dataclass(frozen=True)
class ShadowMarketEvidenceTables:
    """Event readiness, future queries, security candidates, gates, and report."""

    events: pd.DataFrame
    queries: pd.DataFrame
    security_candidates: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ShadowMarketEvidenceResult:
    """One immutable point-in-time market-evidence planning artifact."""

    output_root: Path
    report: dict[str, Any]


def calculate_shadow_market_evidence_plan(
    event_states: pd.DataFrame,
    policy_events: pd.DataFrame,
    core_assets: pd.DataFrame,
    decision_as_of: str | pd.Timestamp,
) -> ShadowMarketEvidenceTables:
    """Create evidence blockers and post-acceptance query targets without outcomes."""
    _require_columns(
        event_states,
        {
            "prospective_event_id",
            "registration_order",
            "cik",
            "ticker",
            "target_period_end",
            "current_state",
            "sec_accepted_at",
            "filing_first_observed_at",
            "innovation_ready",
            "consensus_quality_status",
            "downstream_admission",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "prospective event states",
    )
    _require_columns(
        policy_events,
        {
            "prospective_event_id",
            "shadow_policy_admitted",
            "return_values_loaded",
            "live_trading_authorized",
        },
        "shadow policy events",
    )
    _require_columns(
        core_assets,
        {
            "sec_cik",
            "sec_ticker",
            "stock_quote",
            "mapping_status",
            "core_asset_version",
            "core_asset_frozen_at",
        },
        "core assets",
    )
    states = event_states.copy()
    policy = policy_events.copy()
    if states.empty or states.duplicated("prospective_event_id").any():
        raise ShadowMarketEvidenceError("Event states are empty or non-atomic")
    if policy.duplicated("prospective_event_id").any():
        raise ShadowMarketEvidenceError("Policy events are non-atomic")
    if set(states["prospective_event_id"].astype(str)) != set(
        policy["prospective_event_id"].astype(str)
    ):
        raise ShadowMarketEvidenceError("State and policy event sets differ")
    if not policy["shadow_policy_admitted"].astype(bool).all():
        raise ShadowMarketEvidenceError("A frozen event is outside the shadow policy")
    for frame in (states, policy):
        if frame["return_values_loaded"].fillna(False).astype(bool).any():
            raise ShadowMarketEvidenceError("Return values entered market-evidence planning")
        if frame["live_trading_authorized"].fillna(False).astype(bool).any():
            raise ShadowMarketEvidenceError("Live authorization entered market-evidence planning")
    as_of = _timestamp(decision_as_of, "decision_as_of")
    states["cik"] = states["cik"].astype(str).str.zfill(10)
    states["ticker"] = states["ticker"].astype(str).str.strip().str.upper()
    states["sec_accepted_at"] = pd.to_datetime(states["sec_accepted_at"], utc=True, errors="coerce")
    states["filing_first_observed_at"] = pd.to_datetime(
        states["filing_first_observed_at"], utc=True, errors="coerce"
    )

    candidates = _security_candidates(core_assets)
    candidates = candidates.loc[candidates["cik"].isin(set(states["cik"]))].reset_index(drop=True)
    events = states.merge(candidates, on="cik", how="left", validate="many_to_one")
    ticker_mismatch = events["registered_ticker"].notna() & events["registered_ticker"].ne(
        events["ticker"]
    )
    if ticker_mismatch.any():
        raise ShadowMarketEvidenceError("Core-asset and prospective ticker candidates differ")

    query_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for raw in events.sort_values("registration_order", kind="mergesort").to_dict("records"):
        event = {str(key): value for key, value in raw.items()}
        accepted = _optional_timestamp(event.get("sec_accepted_at"))
        observed = _optional_timestamp(event.get("filing_first_observed_at"))
        candidate_ready = bool(
            pd.notna(event.get("registered_ticker"))
            and pd.notna(event.get("registered_exchange_mic"))
            and event.get("mapping_status") == "accepted"
        )
        blockers: list[str] = []
        if not candidate_ready:
            blockers.append("registered_security_candidate_missing")
        if accepted is None:
            blockers.append("sec_acceptance_pending")
        if accepted is not None and observed is None:
            blockers.append("filing_observation_clock_missing")
        if accepted is not None and observed is not None and observed < accepted:
            raise ShadowMarketEvidenceError("A filing was observed before SEC acceptance")
        if observed is not None and observed > as_of:
            raise ShadowMarketEvidenceError("A filing observation exceeds the evidence cutoff")
        event_queries: list[dict[str, Any]] = []
        if accepted is not None and candidate_ready:
            event_queries = _event_queries(event, accepted)
            query_rows.extend(event_queries)
            blockers.extend(
                [
                    "point_in_time_identity_not_collected",
                    "official_execution_calendar_not_resolved",
                    "preacceptance_cost_bars_not_collected",
                    "sizing_reference_price_not_collected",
                    "corporate_action_evidence_not_collected",
                ]
            )
        event_rows.append(
            {
                "prospective_event_id": str(event["prospective_event_id"]),
                "registration_order": int(event["registration_order"]),
                "cik": str(event["cik"]),
                "ticker": str(event["ticker"]),
                "target_period_end": event["target_period_end"],
                "decision_as_of": as_of,
                "current_state": str(event["current_state"]),
                "sec_accepted_at": accepted,
                "filing_first_observed_at": observed,
                "registered_ticker": event.get("registered_ticker"),
                "registered_exchange_mic": event.get("registered_exchange_mic"),
                "security_candidate_ready": candidate_ready,
                "point_in_time_identity_verified": False,
                "official_calendar_window_ready": False,
                "preacceptance_cost_ready": False,
                "sizing_reference_price_ready": False,
                "corporate_action_evidence_ready": False,
                "portfolio_state_ready": False,
                "planned_entry_at": pd.NaT,
                "planned_exit_at": pd.NaT,
                "sizing_reference_price": None,
                "sizing_reference_observed_at": pd.NaT,
                "round_trip_cost_bps": None,
                "cost_evidence_observed_at": pd.NaT,
                "market_query_count": len(event_queries),
                "market_evidence_status": (
                    "awaiting_sec_acceptance"
                    if accepted is None
                    else "queries_planned_evidence_not_collected"
                ),
                "market_evidence_blockers": "|".join(blockers),
                "order_evidence_ready": False,
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        )
    event_frame = pd.DataFrame(event_rows)
    query_frame = pd.DataFrame(query_rows, columns=_query_columns())
    coverage = event_frame["security_candidate_ready"].astype(bool)
    timing_valid = bool(
        query_frame.empty
        or (
            pd.to_datetime(query_frame["query_not_before"], utc=True)
            >= pd.to_datetime(query_frame["sec_accepted_at"], utc=True)
        ).all()
    )
    gates = pd.DataFrame(
        [
            _gate(
                "frozen_event_coverage",
                len(event_frame),
                len(states),
                len(event_frame) == len(states),
            ),
            _gate(
                "registered_security_candidate",
                int(coverage.sum()),
                len(event_frame),
                bool(coverage.all()),
            ),
            _gate("post_acceptance_query_timing", len(query_frame), len(query_frame), timing_valid),
            _gate("no_return_values_loaded", 0, 0, True),
            _gate("broker_submission_disabled", 0, 0, True),
            _gate("live_trading_not_authorized", 0, 0, True),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": SHADOW_MARKET_EVIDENCE_VERSION,
        "decision_as_of": as_of.isoformat(),
        "events": len(event_frame),
        "issuers": int(event_frame["cik"].nunique()),
        "security_candidates_ready": int(coverage.sum()),
        "accepted_events": int(event_frame["sec_accepted_at"].notna().sum()),
        "market_queries_planned": len(query_frame),
        "order_evidence_ready_events": 0,
        "market_evidence_status_counts": {
            str(key): int(value)
            for key, value in event_frame["market_evidence_status"].value_counts().items()
        },
        "operational_gates_passed": bool(gates["passed"].all()),
        "return_values_loaded": False,
        "broker_submission_authorized": False,
        "live_trading_authorized": False,
    }
    return ShadowMarketEvidenceTables(event_frame, query_frame, candidates, gates, report)


def build_shadow_market_evidence_plan(
    event_state_root: Path,
    trading_policy_root: Path,
    core_asset_root: Path,
    *,
    output_root: Path = Path("data/processed/shadow_market_evidence"),
) -> ShadowMarketEvidenceResult:
    """Persist an immutable forward-only market-evidence plan."""
    state_report = _read_json(event_state_root / "prospective_event_state_machine_report.json")
    decision_as_of = state_report.get("state_as_of")
    if not decision_as_of:
        raise ShadowMarketEvidenceError("State report lacks an evidence cutoff")
    lineage = {
        "event_state_manifest_sha256": _sha256_file(event_state_root / "output_manifest.json"),
        "trading_policy_manifest_sha256": _sha256_file(
            trading_policy_root / "output_manifest.json"
        ),
        "core_asset_manifest_sha256": _sha256_file(core_asset_root / "output_manifest.json"),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    tables = calculate_shadow_market_evidence_plan(
        pd.read_parquet(event_state_root / "prospective_event_states.parquet"),
        pd.read_parquet(trading_policy_root / "shadow_trading_policy_events.parquet"),
        pd.read_csv(core_asset_root / "core_asset_82.csv", dtype={"sec_cik": "string"}),
        str(decision_as_of),
    )
    payload = json.dumps(
        {"version": SHADOW_MARKET_EVIDENCE_VERSION, **lineage, "decision_as_of": decision_as_of},
        sort_keys=True,
    ).encode()
    dataset_version = (
        SHADOW_MARKET_EVIDENCE_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ShadowMarketEvidenceResult(
            target.resolve(), _read_json(target / "shadow_market_evidence_report.json")
        )
    if target.exists():
        raise ShadowMarketEvidenceError(f"Incomplete market-evidence plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "shadow_market_evidence_events.parquet": tables.events,
        "shadow_market_evidence_queries.parquet": tables.queries,
        "shadow_market_security_candidates.parquet": tables.security_candidates,
        "shadow_market_evidence_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact(path, len(frame))
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **lineage,
        "artifacts": artifacts,
    }
    report_path = staging / "shadow_market_evidence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
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
    return ShadowMarketEvidenceResult(target.resolve(), report)


def _security_candidates(core_assets: pd.DataFrame) -> pd.DataFrame:
    core = core_assets.copy()
    core["cik"] = core["sec_cik"].astype(str).str.zfill(10)
    core["registered_ticker"] = core["sec_ticker"].astype(str).str.strip().str.upper()
    core["registered_exchange_mic"] = core["stock_quote"].astype(str).str.upper().map(_EXCHANGE_MAP)
    selected = core.loc[
        core["mapping_status"].astype(str).eq("accepted"),
        [
            "cik",
            "registered_ticker",
            "registered_exchange_mic",
            "mapping_status",
            "core_asset_version",
            "core_asset_frozen_at",
        ],
    ].copy()
    if selected.duplicated("cik").any():
        raise ShadowMarketEvidenceError("Core assets contain duplicate accepted CIKs")
    selected["identity_role"] = "registration_time_candidate_requires_acceptance_revalidation"
    selected["point_in_time_identity_verified"] = False
    selected["return_values_loaded"] = False
    return selected.reset_index(drop=True)


def _event_queries(event: dict[str, Any], accepted: pd.Timestamp) -> list[dict[str, Any]]:
    event_id = str(event["prospective_event_id"])
    ticker = str(event["registered_ticker"])
    exchange = str(event["registered_exchange_mic"])
    start = (accepted.date() - timedelta(days=45)).isoformat()
    end = (accepted.date() + timedelta(days=45)).isoformat()
    specs = [
        (
            "point_in_time_identity",
            "massive_ticker_details_as_of_acceptance",
            accepted.date().isoformat(),
            accepted.date().isoformat(),
        ),
        (
            "preacceptance_cost_bars",
            "massive_unadjusted_daily_bars",
            start,
            accepted.date().isoformat(),
        ),
        (
            "execution_window_bars",
            "massive_unadjusted_daily_bars",
            accepted.date().isoformat(),
            end,
        ),
        ("split_actions", "massive_splits", start, end),
        ("cash_dividend_actions", "massive_dividends", start, end),
    ]
    rows: list[dict[str, Any]] = []
    for query_type, endpoint, window_start, window_end in specs:
        query_id = _stable_id(event_id, query_type, window_start, window_end)
        rows.append(
            {
                "market_evidence_query_id": query_id,
                "prospective_event_id": event_id,
                "cik": str(event["cik"]),
                "ticker": ticker,
                "exchange_mic": exchange,
                "sec_accepted_at": accepted,
                "query_not_before": accepted,
                "query_type": query_type,
                "provider_endpoint": endpoint,
                "window_start": window_start,
                "window_end": window_end,
                "adjusted_prices": False,
                "query_status": "planned_not_collected",
                "return_values_loaded": False,
                "live_trading_authorized": False,
            }
        )
    return rows


def _query_columns() -> list[str]:
    return [
        "market_evidence_query_id",
        "prospective_event_id",
        "cik",
        "ticker",
        "exchange_mic",
        "sec_accepted_at",
        "query_not_before",
        "query_type",
        "provider_endpoint",
        "window_start",
        "window_end",
        "adjusted_prices",
        "query_status",
        "return_values_loaded",
        "live_trading_authorized",
    ]


def _timestamp(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ShadowMarketEvidenceError(f"Invalid {label}")
    return pd.Timestamp(result)


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join((SHADOW_MARKET_EVIDENCE_VERSION, *parts)).encode()).hexdigest()[
        :24
    ]


def _gate(name: str, observed: int, required: int, passed: bool) -> dict[str, Any]:
    return {"gate": name, "observed": observed, "required": required, "passed": bool(passed)}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ShadowMarketEvidenceError(f"{label} lacks required columns: {missing}")


def _artifact(path: Path, rows: int) -> dict[str, Any]:
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
