"""Frozen record of the v3.1 clock amendment, registered before any outcome exists.

The amendment changes one thing about the prospective Market-Innovation
protocol: the event clock.  Under v3 an event is timed at the SEC acceptance
of its 10-Q/10-K; under v3.1 it is timed at the earliest SEC-evidenced
disclosure, usually an Item 2.02 earnings 8-K.  The cohort, the thresholds,
the estimand, the firewall, and the retired models are untouched.

Freezing the amendment is only allowed while the registry, the readiness
ledger, and the event state machine all show zero actual values, zero
innovations, and zero paper positions.  The record binds the exact registry
manifest, the 37 event identities, the rule text, and the clock and quality
bundle versions that existed at declaration, so a later reader can verify
that the clock was changed before -- not after -- outcomes were seen.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from prospective_event_state_machine import (
    CONSENSUS_FROZEN,
    CONSENSUS_LEVEL_READY,
    CONSENSUS_REVISION_READY,
    FILING_OBSERVED,
    REGISTERED,
    REJECTED,
)
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

AMENDMENT_VERSION: Final[str] = "prospective-market-innovation-v3.1-amendment-v1"
PROTOCOL_VERSION: Final[str] = "prospective-market-innovation-v3.1"
PRE_OUTCOME_STATES: Final[frozenset[str]] = frozenset(
    {
        REGISTERED,
        CONSENSUS_LEVEL_READY,
        CONSENSUS_REVISION_READY,
        FILING_OBSERVED,
        CONSENSUS_FROZEN,
        REJECTED,
    }
)
AMENDMENT_RULES: Final[dict[str, str]] = {
    "event_clock": (
        "earliest SEC-evidenced disclosure: the earlier of the first qualifying original "
        "Item 2.02 8-K acceptance and the original 10-Q/10-K acceptance"
    ),
    "release_qualification": (
        "accepted after the issuer's previous original periodic report, on or after the New "
        "York day following the fiscal period end, within 120 days of the period end, and no "
        "later than the period's periodic report"
    ),
    "pre_event_consensus": (
        "captures strictly before the earliest disclosure acceptance whose New York date is "
        "no later than the issuer-reported release event date"
    ),
    "staleness": (
        "unchanged seven calendar days, measured between the last pre-disclosure capture and "
        "the disclosure, on New York dates"
    ),
    "post_disclosure_captures": "counted and excluded; never reused as pre-event consensus",
    "filing_monitor_scope": (
        "original 8-K Item 2.02 filings are read from the preserved SEC submissions snapshots "
        "already captured for 10-Q/10-K monitoring; no additional request"
    ),
    "entry": (
        "second exchange session after the New York date of the earliest disclosure under the "
        "frozen date-based execution contract; exit at the twentieth session close"
    ),
    "thresholds": (
        "unchanged: 30 complete events, 15 issuers, eight positive and eight negative "
        "innovations; no replacement event"
    ),
    "v3_reporting": (
        "v3 acceptance-clock outputs continue to be computed and reported; v3.1 is the "
        "primary clock for any return evaluation"
    ),
    "pre_declared_handling": (
        "an event whose last pre-disclosure capture is stale, or that has no pre-disclosure "
        "capture, is quarantined by this rule at declaration time regardless of its later "
        "actual or return; the rule is not revisited after outcomes exist"
    ),
}


class ProspectiveMarketInnovationAmendmentError(FilingCatalogError):
    """Raised when the amendment cannot be proven to precede every outcome."""


@dataclass(frozen=True)
class ProspectiveMarketInnovationAmendmentResult:
    """Immutable amendment record location and report."""

    output_root: Path
    report: dict[str, Any]


def verify_pre_outcome_state(
    registry: pd.DataFrame,
    readiness: pd.DataFrame,
    states: pd.DataFrame,
) -> dict[str, Any]:
    """Prove that no registered event has an actual, an innovation, or a position."""
    for frame, label in ((registry, "registry"), (readiness, "readiness")):
        if "prospective_event_id" not in frame or frame["prospective_event_id"].duplicated().any():
            raise ProspectiveMarketInnovationAmendmentError(f"The {label} is not atomic")
    ids = set(registry["prospective_event_id"].astype(str))
    if ids != set(readiness["prospective_event_id"].astype(str)) or ids != set(
        states["prospective_event_id"].astype(str)
    ):
        raise ProspectiveMarketInnovationAmendmentError(
            "Registry, readiness, and state machine do not cover the same events"
        )
    for frame, column in (
        (registry, "actual_value_loaded_at_registration"),
        (registry, "return_value_loaded_at_registration"),
        (registry, "selection_uses_return_values"),
        (readiness, "actual_value_loaded"),
        (readiness, "return_value_loaded"),
        (readiness, "selection_uses_return_values"),
    ):
        if column not in frame or frame[column].astype(bool).any():
            raise ProspectiveMarketInnovationAmendmentError(
                f"Outcome evidence exists or is unverifiable: {column}"
            )
    if "current_state" not in states:
        raise ProspectiveMarketInnovationAmendmentError("State catalog lacks current_state")
    beyond = states.loc[~states["current_state"].isin(PRE_OUTCOME_STATES), "current_state"]
    if not beyond.empty:
        raise ProspectiveMarketInnovationAmendmentError(
            f"An event has progressed past consensus: {sorted(set(beyond.astype(str)))}"
        )
    return {
        "events": len(ids),
        "state_counts": {str(k): int(v) for k, v in states["current_state"].value_counts().items()},
        "actual_values_loaded": False,
        "return_values_loaded": False,
        "paper_positions": 0,
    }


def freeze_prospective_market_innovation_amendment(
    preregistration_root: Path,
    readiness_root: Path,
    state_machine_root: Path,
    clock_root: Path,
    quality_root: Path,
    *,
    output_root: Path = Path("data/processed/prospective_market_innovation_amendment"),
) -> ProspectiveMarketInnovationAmendmentResult:
    """Persist the content-addressed amendment record, or reuse an identical one."""
    registry = load_catalog_table(
        preregistration_root, "prospective_market_innovation_events.parquet"
    )
    readiness = load_catalog_table(
        readiness_root, "prospective_market_innovation_event_readiness.parquet"
    )
    states = load_catalog_table(state_machine_root, "prospective_event_states.parquet")
    verification = verify_pre_outcome_state(registry, readiness, states)
    clock_report = _verified_report(clock_root, "forward_disclosure_clock_report.json")
    quality_report = _verified_report(quality_root, "prospective_disclosure_quality_report.json")
    if quality_report.get("actual_values_loaded") is not False or (
        quality_report.get("return_values_loaded") is not False
    ):
        raise ProspectiveMarketInnovationAmendmentError("The quality view reports outcomes")

    event_ids = sorted(registry["prospective_event_id"].astype(str))
    registry_manifest_sha256 = _sha256_file(preregistration_root / "output_manifest.json")
    implementation_sha256 = _sha256_file(Path(__file__))
    identity = json.dumps(
        {
            "amendment_version": AMENDMENT_VERSION,
            "registry_manifest_sha256": registry_manifest_sha256,
            "event_ids": event_ids,
            "rules": AMENDMENT_RULES,
            "implementation_sha256": implementation_sha256,
        },
        sort_keys=True,
    ).encode()
    dataset_version = f"{AMENDMENT_VERSION}-{hashlib.sha256(identity).hexdigest()[:16]}"
    target = output_root / dataset_version
    report_name = "prospective_market_innovation_amendment_report.json"
    if (target / "output_manifest.json").is_file():
        stored: dict[str, Any] = json.loads((target / report_name).read_text(encoding="utf-8"))
        return ProspectiveMarketInnovationAmendmentResult(target.resolve(), stored)
    if target.exists():
        raise ProspectiveMarketInnovationAmendmentError(f"Incomplete amendment record: {target}")

    report = {
        "amendment_version": AMENDMENT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_version": dataset_version,
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "registry_manifest_sha256": registry_manifest_sha256,
        "registered_events": len(event_ids),
        "registered_issuers": int(registry["cik"].nunique()),
        "event_ids_sha256": hashlib.sha256("\n".join(event_ids).encode()).hexdigest(),
        "cohort_changed": False,
        "thresholds_changed": False,
        "estimand_changed": False,
        "rules": AMENDMENT_RULES,
        "verification_at_declaration": verification,
        "clock_dataset_version": clock_report.get("dataset_version"),
        "clock_summary": clock_report.get("summary"),
        "quality_dataset_version": quality_report.get("dataset_version"),
        "quality_status_counts": quality_report.get("status_counts"),
        "implementation_sha256": implementation_sha256,
        "first_public_certified": False,
        "model_admission": False,
        "trading_admission": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    ids_path = staging / "amended_event_ids.json"
    ids_path.write_text(json.dumps(event_ids, indent=2), encoding="utf-8")
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "artifacts": {
                    "amended_event_ids.json": {"sha256": _sha256_file(ids_path)},
                },
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ProspectiveMarketInnovationAmendmentResult(target.resolve(), report)


def _verified_report(root: Path, name: str) -> dict[str, Any]:
    manifest = json.loads((root / "output_manifest.json").read_text(encoding="utf-8"))
    path = root / name
    if not path.is_file() or _sha256_file(path) != manifest.get("report_sha256"):
        raise ProspectiveMarketInnovationAmendmentError(f"Report hash mismatch: {path}")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
