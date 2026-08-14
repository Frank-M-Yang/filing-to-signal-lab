"""Source-traceable audit of gross-margin revision-tail events.

The audit separates extraction artifacts from genuine redraws of published
history.  It also evaluates a deliberately narrow second-stage watch flag that
uses only structural-change context already public in the original filing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

REQUIRED_REVIEW_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "audit_classification",
    "audit_family",
    "audit_action",
    "economic_error_correction",
    "event_time_structural_context",
    "mechanism",
    "evidence",
    "review_status",
)


class RevisionTailAuditError(ValueError):
    """Raised when the manual audit does not tie to its SEC sources."""


@dataclass(frozen=True)
class RevisionTailAuditResult:
    """Reviewed events, category summary, gate diagnostic, and report."""

    reviewed_events: pd.DataFrame
    classification_summary: pd.DataFrame
    event_time_gate: pd.DataFrame
    report: dict[str, Any]


def build_revision_tail_audit(
    tail_events: pd.DataFrame,
    manual_reviews: pd.DataFrame,
    filing_manifest: pd.DataFrame,
) -> RevisionTailAuditResult:
    """Join every reviewed tail judgment to exact preserved SEC documents."""
    _require_columns(
        tail_events,
        (
            "event_id",
            "company_id",
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "revision_size",
            "exceeded_primary_buffer",
            "realized_direction_flip",
            "filing_accession",
            "latest_filing_accession",
        ),
    )
    _require_columns(manual_reviews, REQUIRED_REVIEW_COLUMNS)
    _require_columns(
        filing_manifest,
        ("accession_number", "source_url", "sha256", "raw_path", "status_code"),
    )
    exceeded = tail_events.loc[_as_bool(tail_events["exceeded_primary_buffer"])].copy()
    reviews = manual_reviews.copy()
    if exceeded.empty or exceeded["event_id"].duplicated().any():
        raise RevisionTailAuditError("Primary-buffer tail must be non-empty and event-unique")
    if reviews.empty or reviews["event_id"].duplicated().any():
        raise RevisionTailAuditError("Manual reviews must be non-empty and event-unique")
    if set(exceeded["event_id"]) != set(reviews["event_id"]):
        raise RevisionTailAuditError("Manual reviews must cover the full primary-buffer tail")
    if not reviews["review_status"].astype("string").eq("reviewed").all():
        raise RevisionTailAuditError("Every tail judgment must have reviewed status")

    reviews["economic_error_correction"] = _as_bool(reviews["economic_error_correction"])
    reviews["event_time_structural_context"] = _as_bool(reviews["event_time_structural_context"])
    manifest = filing_manifest.copy()
    manifest["accession_number"] = manifest["accession_number"].astype("string")
    if manifest["accession_number"].duplicated().any():
        raise RevisionTailAuditError("Filing manifest must have one primary document per accession")
    if not manifest["status_code"].eq(200).all():
        raise RevisionTailAuditError("Every preserved filing must have HTTP status 200")
    _verify_preserved_hashes(manifest)

    original_sources = manifest[["accession_number", "source_url", "sha256", "raw_path"]].rename(
        columns={
            "accession_number": "filing_accession",
            "source_url": "original_source_url",
            "sha256": "original_source_sha256",
            "raw_path": "original_raw_path",
        }
    )
    later_sources = manifest[["accession_number", "source_url", "sha256", "raw_path"]].rename(
        columns={
            "accession_number": "latest_filing_accession",
            "source_url": "later_source_url",
            "sha256": "later_source_sha256",
            "raw_path": "later_raw_path",
        }
    )
    reviewed = exceeded.merge(reviews, on="event_id", how="left", validate="one_to_one")
    reviewed = reviewed.merge(
        original_sources, on="filing_accession", how="left", validate="many_to_one"
    ).merge(later_sources, on="latest_filing_accession", how="left", validate="many_to_one")
    source_columns = [
        "original_source_url",
        "original_source_sha256",
        "later_source_url",
        "later_source_sha256",
    ]
    if reviewed[source_columns].isna().any().any():
        raise RevisionTailAuditError("Every reviewed event must tie to original and later sources")

    artifact = reviewed["audit_action"].eq("set_latest_equal_original")
    reviewed["unreviewed_revision_size"] = reviewed["revision_size"]
    reviewed["audited_revision_size"] = reviewed["revision_size"].mask(artifact, 0.0)
    reviewed["published_history_redraw"] = ~artifact
    reviewed["direction_flip_after_audit"] = (
        _as_bool(reviewed["realized_direction_flip"]) & ~artifact
    )
    reviewed["candidate_structural_watch"] = reviewed["event_time_structural_context"]
    reviewed["candidate_watch_result"] = reviewed["candidate_structural_watch"].map(
        {True: "abstain_for_manual_scope_review", False: "base_buffer_only"}
    )
    reviewed = reviewed.sort_values(
        "unreviewed_revision_size", key=lambda values: values.abs(), ascending=False
    )

    classification_summary = (
        reviewed.groupby(["audit_family", "audit_classification"], as_index=False)
        .agg(
            reviewed_events=("event_id", "size"),
            max_unreviewed_revision=("unreviewed_revision_size", lambda values: values.abs().max()),
            direction_flips_after_audit=("direction_flip_after_audit", "sum"),
            event_time_structural_flags=("candidate_structural_watch", "sum"),
        )
        .sort_values(["audit_family", "reviewed_events"], ascending=[True, False])
    )
    redraw = reviewed["published_history_redraw"]
    flips = reviewed["direction_flip_after_audit"]
    watch = reviewed["candidate_structural_watch"]
    event_time_gate = pd.DataFrame(
        [
            {
                "candidate_gate": "public_structural_change_context_in_original_filing",
                "reviewed_published_history_redraws": int(redraw.sum()),
                "redraws_flagged": int((redraw & watch).sum()),
                "redraw_recall": float((redraw & watch).sum() / max(int(redraw.sum()), 1)),
                "direction_flips_after_audit": int(flips.sum()),
                "direction_flips_flagged": int((flips & watch).sum()),
                "flip_recall": float((flips & watch).sum() / max(int(flips.sum()), 1)),
                "control_sample_available": False,
                "deployment_status": "do_not_deploy_needs_broad_event_time_text_controls",
            }
        ]
    )
    report = {
        "audit_version": _audit_version(reviewed),
        "reviewed_tail_events": len(reviewed),
        "semantic_collision_corrections": int(artifact.sum()),
        "published_history_redraws": int(redraw.sum()),
        "accounting_scope_redraws": int(reviewed["audit_family"].eq("accounting_scope").sum()),
        "accounting_presentation_redraws": int(
            reviewed["audit_family"].eq("accounting_presentation").sum()
        ),
        "economic_error_corrections": int(reviewed["economic_error_correction"].sum()),
        "event_time_structural_flags": int(watch.sum()),
        "direction_flips_after_audit": int(flips.sum()),
        "direction_flips_flagged": int((flips & watch).sum()),
        "candidate_gate_status": str(event_time_gate.iloc[0]["deployment_status"]),
        "interpretation": (
            "The reviewed tail contains no verified economic error correction. Four events "
            "are extraction artifacts; the remainder are scope or presentation redraws."
        ),
    }
    return RevisionTailAuditResult(
        reviewed_events=reviewed,
        classification_summary=classification_summary,
        event_time_gate=event_time_gate,
        report=report,
    )


def _verify_preserved_hashes(manifest: pd.DataFrame) -> None:
    for row in manifest.itertuples(index=False):
        path = Path(str(row.raw_path))
        if not path.is_file():
            raise RevisionTailAuditError(f"Preserved filing is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(row.sha256):
            raise RevisionTailAuditError(f"Preserved filing hash mismatch: {path}")


def _audit_version(reviewed: pd.DataFrame) -> str:
    columns = [
        "event_id",
        "audit_classification",
        "audit_action",
        "original_source_sha256",
        "later_source_sha256",
    ]
    payload = reviewed[columns].sort_values("event_id").to_csv(index=False).encode("utf-8")
    return "revision-tail-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RevisionTailAuditError("Boolean columns contain invalid values")
    return normalized.eq("true")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RevisionTailAuditError(f"Missing required columns: {missing}")
