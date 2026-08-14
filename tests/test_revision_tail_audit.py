"""Tests for the source-traceable revision-tail review."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from revision_tail_audit import build_revision_tail_audit


def test_tail_audit_separates_artifact_from_history_redraw(tmp_path: Path) -> None:
    tail = pd.DataFrame(
        [
            _tail_row("event-artifact", "initial-a", "later-a", 0.20, True),
            _tail_row("event-redraw", "initial-b", "later-b", -0.02, True),
        ]
    )
    reviews = pd.DataFrame(
        [
            _review("event-artifact", "xbrl_semantic_collision", "data_pipeline", True, False),
            _review(
                "event-redraw",
                "discontinued_operations_scope_recast",
                "accounting_scope",
                False,
                True,
            ),
        ]
    )
    manifest_rows = []
    for accession in ("initial-a", "later-a", "initial-b", "later-b"):
        path = tmp_path / f"{accession}.html"
        content = f"<html>{accession}</html>".encode()
        path.write_bytes(content)
        manifest_rows.append(
            {
                "accession_number": accession,
                "source_url": f"https://www.sec.gov/{accession}",
                "sha256": hashlib.sha256(content).hexdigest(),
                "raw_path": str(path),
                "status_code": 200,
            }
        )

    result = build_revision_tail_audit(tail, reviews, pd.DataFrame(manifest_rows))

    events = result.reviewed_events.set_index("event_id")
    assert events.loc["event-artifact", "audited_revision_size"] == 0.0
    assert bool(events.loc["event-redraw", "published_history_redraw"])
    assert result.report["semantic_collision_corrections"] == 1
    assert result.report["published_history_redraws"] == 1
    assert result.report["economic_error_corrections"] == 0
    assert result.event_time_gate.loc[0, "deployment_status"].startswith("do_not_deploy")


def _tail_row(
    event_id: str,
    initial_accession: str,
    latest_accession: str,
    revision: float,
    flip: bool,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "company_id": "0000000001",
        "ticker": "TEST",
        "fiscal_year": 2024,
        "fiscal_quarter": "Q2",
        "revision_size": revision,
        "exceeded_primary_buffer": True,
        "realized_direction_flip": flip,
        "filing_accession": initial_accession,
        "latest_filing_accession": latest_accession,
    }


def _review(
    event_id: str,
    classification: str,
    family: str,
    artifact: bool,
    structural_context: bool,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "audit_classification": classification,
        "audit_family": family,
        "audit_action": (
            "set_latest_equal_original" if artifact else "retain_published_history_redraw"
        ),
        "economic_error_correction": False,
        "event_time_structural_context": structural_context,
        "mechanism": "test mechanism",
        "evidence": "test evidence",
        "review_status": "reviewed",
    }
