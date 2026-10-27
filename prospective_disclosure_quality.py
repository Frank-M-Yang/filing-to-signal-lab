"""Consensus eligibility and staleness measured at the earliest SEC disclosure.

This is the v3.1 amendment view of the frozen prospective Market-Innovation
cohort (docs/prospective_market_innovation_v3_1.md).  The v3 quality audit
measures consensus staleness at the 10-Q/10-K acceptance and treats every
capture before that acceptance as pre-event.  When an Item 2.02 earnings
release came earlier, captures between the release and the periodic report
are post-announcement data.  This module re-reads the same registry, the same
consensus snapshots, and the v3 quality table, and re-times each event at the
forward disclosure clock:

- a capture is *pre-disclosure* only if it precedes the earliest disclosure
  acceptance **and** its New York date is no later than the issuer-reported
  release event date (the stricter of the two clocks);
- staleness is the calendar gap between the last pre-disclosure capture and the
  disclosure date; the frozen seven-day limit is unchanged;
- captures on or after the disclosure are counted and excluded, never silently
  reused as pre-event consensus.

Sample thresholds are carried verbatim from v3 and may not be changed here.
The v3 tables are read, not modified.  Nothing here loads an actual value or a
return, and an outcome firewall stops the build if any input says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Hashable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

PROSPECTIVE_DISCLOSURE_QUALITY_VERSION: Final[str] = "prospective-disclosure-quality-v1"
PROTOCOL_VERSION: Final[str] = "prospective-market-innovation-v3.1"
NEW_YORK: Final[str] = "America/New_York"

AWAITING_DISCLOSURE: Final[str] = "awaiting_disclosure"
CLEAN_PRE_DISCLOSURE: Final[str] = "clean_pre_disclosure"
STALE_AT_DISCLOSURE: Final[str] = "stale_at_disclosure"
NO_PRE_DISCLOSURE_CONSENSUS: Final[str] = "no_pre_disclosure_consensus"
V3_STALENESS_REASON: Final[str] = "stale_preacceptance_consensus"

REGISTRY_COLUMNS: Final[tuple[str, ...]] = (
    "prospective_event_id",
    "registration_order",
    "cik",
    "ticker",
    "target_period_end",
    "accepted_at",
    "actual_value_loaded",
    "return_value_loaded",
    "selection_uses_return_values",
)
SNAPSHOT_COLUMNS: Final[tuple[str, ...]] = (
    "prospective_event_id",
    "captured_at",
    "capture_date",
    "revenue_estimate_average",
    "revenue_estimate_analyst_count",
    "source_sha256",
    "historical_backtest_admission",
)
CLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "prospective_event_id",
    "clock_source",
    "earliest_disclosure_at",
    "release_event_date",
)
V3_QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "prospective_event_id",
    "quality_status",
    "quality_reasons",
    "last_snapshot_at",
    "actual_values_loaded",
    "return_values_loaded",
)


class ProspectiveDisclosureQualityError(FilingCatalogError):
    """Raised when the amendment view would violate the frozen protocol."""


@dataclass(frozen=True)
class ProspectiveDisclosureQualityConfig:
    """Frozen v3 thresholds re-timed at the disclosure clock; no value may change."""

    stale_pre_disclosure_calendar_days: int = 7
    minimum_complete_events: int = 30
    minimum_issuers: int = 15
    minimum_positive_innovations: int = 8
    minimum_negative_innovations: int = 8
    version: str = PROSPECTIVE_DISCLOSURE_QUALITY_VERSION

    def validate(self) -> None:
        """The amendment changes the clock only; any threshold change needs a new protocol."""
        if asdict(self) != asdict(ProspectiveDisclosureQualityConfig()):
            raise ProspectiveDisclosureQualityError(
                "v3.1 may not change staleness or sample thresholds; register a new protocol"
            )


@dataclass(frozen=True)
class ProspectiveDisclosureQualityTables:
    """Pure amendment tables before persistence."""

    event_quality: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveDisclosureQualityResult:
    """Immutable bundle location and report."""

    output_root: Path
    report: dict[str, Any]


def calculate_prospective_disclosure_quality(
    registry: pd.DataFrame,
    snapshots: pd.DataFrame,
    clocks: pd.DataFrame,
    v3_quality: pd.DataFrame,
    config: ProspectiveDisclosureQualityConfig | None = None,
) -> ProspectiveDisclosureQualityTables:
    """Re-time every registered event at its earliest disclosure and compare with v3."""
    cfg = config or ProspectiveDisclosureQualityConfig()
    cfg.validate()
    events = _prepare_registry(registry)
    snaps = _prepare_snapshots(snapshots)
    clock_table = _prepare_clocks(clocks)
    v3 = _prepare_v3_quality(v3_quality)
    _enforce_outcome_firewall(events, snaps, v3)
    for label, frame in (("clock", clock_table), ("v3 quality", v3)):
        if set(frame["prospective_event_id"]) != set(events["prospective_event_id"]):
            raise ProspectiveDisclosureQualityError(
                f"The {label} table does not cover the registry"
            )

    joined = events.merge(clock_table, on="prospective_event_id", validate="one_to_one").merge(
        v3, on="prospective_event_id", validate="one_to_one"
    )
    rows = [_event_row(event, snaps, cfg) for event in joined.to_dict("records")]
    quality = pd.DataFrame(rows)
    quality["protocol_version"] = PROTOCOL_VERSION
    quality["actual_values_loaded"] = False
    quality["return_values_loaded"] = False
    quality["selection_uses_return_values"] = False
    quality = quality.sort_values(
        ["target_period_end", "ticker", "prospective_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    gates = _gates(quality, cfg)
    return ProspectiveDisclosureQualityTables(quality, gates, _report(quality, gates, cfg))


def build_prospective_disclosure_quality(
    readiness_root: Path,
    clock_root: Path,
    v3_quality_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_disclosure_quality"),
    config: ProspectiveDisclosureQualityConfig | None = None,
) -> ProspectiveDisclosureQualityResult:
    """Persist a content-addressed amendment view; v3 bundles are only read."""
    cfg = config or ProspectiveDisclosureQualityConfig()
    cfg.validate()
    input_hashes = {
        "readiness": _sha256_file(readiness_root / "output_manifest.json"),
        "forward_disclosure_clock": _sha256_file(clock_root / "output_manifest.json"),
        "v3_consensus_quality": _sha256_file(v3_quality_root / "output_manifest.json"),
    }
    implementation_sha256 = _sha256_file(Path(__file__))
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "implementation_sha256": implementation_sha256,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = f"{cfg.version}-{hashlib.sha256(payload).hexdigest()[:16]}"
    target = output_root / dataset_version
    report_name = "prospective_disclosure_quality_report.json"
    if (target / "output_manifest.json").is_file():
        stored: dict[str, Any] = json.loads((target / report_name).read_text(encoding="utf-8"))
        return ProspectiveDisclosureQualityResult(target.resolve(), stored)
    if target.exists():
        raise ProspectiveDisclosureQualityError(f"Incomplete amendment bundle: {target}")

    tables = calculate_prospective_disclosure_quality(
        load_catalog_table(readiness_root, "prospective_market_innovation_event_readiness.parquet"),
        load_catalog_table(
            readiness_root, "prospective_market_innovation_consensus_snapshots.parquet"
        ),
        load_catalog_table(clock_root, "forward_disclosure_clocks.parquet"),
        load_catalog_table(v3_quality_root, "prospective_consensus_event_quality.parquet"),
        cfg,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "prospective_disclosure_event_quality.parquet": tables.event_quality,
        "prospective_disclosure_quality_gates.parquet": tables.gates,
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"rows": len(frame), "sha256": _sha256_file(path)}
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "implementation_sha256": implementation_sha256,
        "artifacts": artifacts,
    }
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ProspectiveDisclosureQualityResult(target.resolve(), report)


def _event_row(
    event: Mapping[Hashable, Any],
    snapshots: pd.DataFrame,
    config: ProspectiveDisclosureQualityConfig,
) -> dict[str, Any]:
    daily = snapshots.loc[
        snapshots["prospective_event_id"].eq(event["prospective_event_id"])
    ].copy()
    disclosure_at = event["earliest_disclosure_at"]
    event_date = event["release_event_date"]
    disclosed = bool(pd.notna(disclosure_at))
    capture_local_date = (
        daily["captured_at"].dt.tz_convert(NEW_YORK).dt.normalize().dt.tz_localize(None)
    )
    if disclosed:
        pre = daily["captured_at"].lt(disclosure_at)
        if pd.notna(event_date):
            pre &= capture_local_date.le(pd.Timestamp(event_date))
    else:
        pre = pd.Series(True, index=daily.index, dtype=bool)
    pre_rows = daily.loc[pre]
    post_rows = daily.loc[~pre]
    last_pre = pre_rows.sort_values("captured_at").iloc[-1] if not pre_rows.empty else None

    staleness: float = float("nan")
    if disclosed and last_pre is not None:
        disclosure_local = pd.Timestamp(disclosure_at).tz_convert(NEW_YORK).normalize()
        last_local = pd.Timestamp(last_pre["captured_at"]).tz_convert(NEW_YORK).normalize()
        staleness = float((disclosure_local - last_local).days)
    stale = bool(np.isfinite(staleness) and staleness > config.stale_pre_disclosure_calendar_days)

    accepted_at = event["accepted_at"]
    v3_pool = daily.loc[daily["captured_at"].lt(accepted_at)] if pd.notna(accepted_at) else daily
    v3_candidate = v3_pool["captured_at"].max() if not v3_pool.empty else pd.NaT
    v3_candidate_post = bool(
        disclosed
        and pd.notna(v3_candidate)
        and not bool(pre.loc[daily["captured_at"].eq(v3_candidate)].any())
    )

    if not disclosed:
        status = AWAITING_DISCLOSURE
    elif last_pre is None:
        status = NO_PRE_DISCLOSURE_CONSENSUS
    elif stale:
        status = STALE_AT_DISCLOSURE
    else:
        status = CLEAN_PRE_DISCLOSURE
    v3_status = str(event["quality_status"])
    v3_reasons = str(event["quality_reasons"] or "")
    v3_structural_clean = v3_status == "clean" or (
        v3_status == "quarantined" and v3_reasons == V3_STALENESS_REASON
    )
    return {
        "prospective_event_id": event["prospective_event_id"],
        "registration_order": event["registration_order"],
        "cik": event["cik"],
        "ticker": event["ticker"],
        "target_period_end": event["target_period_end"],
        "clock_source": event["clock_source"],
        "earliest_disclosure_at": disclosure_at,
        "release_event_date": event_date,
        "disclosure_observed": disclosed,
        "snapshot_capture_dates": len(daily),
        "pre_disclosure_capture_dates": len(pre_rows),
        "post_disclosure_capture_dates": len(post_rows),
        "last_pre_disclosure_snapshot_at": (
            last_pre["captured_at"] if last_pre is not None else pd.NaT
        ),
        "last_pre_disclosure_consensus_revenue": (
            float(last_pre["revenue_estimate_average"]) if last_pre is not None else np.nan
        ),
        "staleness_calendar_days_at_disclosure": staleness,
        "stale_at_disclosure": stale,
        "v3_quality_status": v3_status,
        "v3_quality_reasons": v3_reasons,
        "v3_structural_clean": v3_structural_clean,
        "v3_candidate_snapshot_at": v3_candidate,
        "v3_candidate_post_disclosure": v3_candidate_post,
        "v31_status": status,
        "downstream_admission": status == CLEAN_PRE_DISCLOSURE and v3_structural_clean,
    }


def _gates(quality: pd.DataFrame, config: ProspectiveDisclosureQualityConfig) -> pd.DataFrame:
    # Complete events need a registered actual; the firewall guarantees none exist here.
    complete = 0
    return pd.DataFrame(
        [
            _gate("minimum_complete_events", complete, config.minimum_complete_events),
            _gate("minimum_issuers", 0, config.minimum_issuers),
            _gate("minimum_positive_innovations", 0, config.minimum_positive_innovations),
            _gate("minimum_negative_innovations", 0, config.minimum_negative_innovations),
        ]
    )


def _report(
    quality: pd.DataFrame, gates: pd.DataFrame, config: ProspectiveDisclosureQualityConfig
) -> dict[str, Any]:
    admissible = quality["downstream_admission"].astype(bool)
    by_period: dict[str, dict[str, int]] = {}
    for period, group in quality.groupby(quality["target_period_end"].dt.strftime("%Y-%m-%d")):
        by_period[str(period)] = {
            str(k): int(v) for k, v in group["v31_status"].value_counts().items()
        }
    return {
        "contract_version": config.version,
        "protocol_version": PROTOCOL_VERSION,
        "events": len(quality),
        "issuers": int(quality["cik"].nunique()),
        "disclosed_events": int(quality["disclosure_observed"].sum()),
        "status_counts": {str(k): int(v) for k, v in quality["v31_status"].value_counts().items()},
        "by_target_period_end": by_period,
        "stale_at_disclosure_events": int(quality["stale_at_disclosure"].sum()),
        "events_with_excluded_post_disclosure_captures": int(
            quality["post_disclosure_capture_dates"].gt(0).sum()
        ),
        "excluded_post_disclosure_captures": int(quality["post_disclosure_capture_dates"].sum()),
        "v3_candidate_post_disclosure_events": int(quality["v3_candidate_post_disclosure"].sum()),
        "admissible_pre_disclosure_events": int(admissible.sum()),
        "admissible_pre_disclosure_issuers": int(quality.loc[admissible, "cik"].nunique()),
        "gates_passed": int(gates["passed"].sum()),
        "gates_total": len(gates),
        "thresholds": asdict(config),
        "v3_tables_modified": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "selection_uses_return_values": False,
        "model_admission": False,
        "trading_admission": False,
    }


def _prepare_registry(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REGISTRY_COLUMNS, "registry")
    events = frame.loc[:, list(REGISTRY_COLUMNS)].copy()
    if events.empty or events["prospective_event_id"].duplicated().any():
        raise ProspectiveDisclosureQualityError("The registry is empty or non-atomic")
    events["cik"] = events["cik"].astype(str).str.zfill(10)
    events["ticker"] = events["ticker"].astype(str).str.strip().str.upper()
    events["target_period_end"] = pd.to_datetime(events["target_period_end"], utc=True)
    events["accepted_at"] = pd.to_datetime(events["accepted_at"], utc=True, errors="coerce")
    return events


def _prepare_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, SNAPSHOT_COLUMNS, "consensus snapshots")
    snaps = frame.loc[:, list(SNAPSHOT_COLUMNS)].copy()
    snaps["captured_at"] = pd.to_datetime(snaps["captured_at"], utc=True, errors="coerce")
    snaps["capture_date"] = pd.to_datetime(snaps["capture_date"], utc=True, errors="coerce")
    if snaps["captured_at"].isna().any() or snaps["capture_date"].isna().any():
        raise ProspectiveDisclosureQualityError("A consensus snapshot lacks a capture clock")
    snaps["capture_date"] = snaps["capture_date"].dt.normalize()
    # One observation per event and capture date, as in the v3 quality audit.
    return (
        snaps.sort_values(
            ["prospective_event_id", "capture_date", "captured_at", "source_sha256"],
            kind="mergesort",
        )
        .drop_duplicates(["prospective_event_id", "capture_date"], keep="last")
        .reset_index(drop=True)
    )


def _prepare_clocks(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CLOCK_COLUMNS, "forward disclosure clocks")
    clocks = frame.loc[:, list(CLOCK_COLUMNS)].copy()
    if clocks["prospective_event_id"].duplicated().any():
        raise ProspectiveDisclosureQualityError("Clock rows are duplicated")
    clocks["earliest_disclosure_at"] = pd.to_datetime(
        clocks["earliest_disclosure_at"], utc=True, errors="coerce"
    )
    clocks["release_event_date"] = pd.to_datetime(
        clocks["release_event_date"], errors="coerce"
    ).dt.normalize()
    return clocks


def _prepare_v3_quality(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, V3_QUALITY_COLUMNS, "v3 consensus quality")
    v3 = frame.loc[:, list(V3_QUALITY_COLUMNS)].copy()
    if v3["prospective_event_id"].duplicated().any():
        raise ProspectiveDisclosureQualityError("v3 quality rows are duplicated")
    v3["quality_reasons"] = v3["quality_reasons"].fillna("").astype(str)
    return v3


def _enforce_outcome_firewall(
    registry: pd.DataFrame, snapshots: pd.DataFrame, v3: pd.DataFrame
) -> None:
    if (
        registry["actual_value_loaded"].astype(bool).any()
        or registry["return_value_loaded"].astype(bool).any()
        or registry["selection_uses_return_values"].astype(bool).any()
        or snapshots["historical_backtest_admission"].astype(bool).any()
        or v3["actual_values_loaded"].astype(bool).any()
        or v3["return_values_loaded"].astype(bool).any()
    ):
        raise ProspectiveDisclosureQualityError(
            "The amendment view must be built before any outcome is loaded"
        )


def _gate(name: str, observed: int, required: int) -> dict[str, Any]:
    return {
        "gate": name,
        "observed": observed,
        "required": required,
        "passed": observed >= required,
    }


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ProspectiveDisclosureQualityError(f"{label} lack columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
