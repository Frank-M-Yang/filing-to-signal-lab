"""Primary-source terminal consideration evidence for formal validation events."""

from __future__ import annotations

import hashlib
import html
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.filing_cover import load_filing_artifact_snapshots
from sec_data.validation_delisting import calculate_validation_outcome_repair_queue

VALIDATION_TERMINAL_VERSION: Final[str] = "validation-terminal-evidence-v2-source-namespaced"


class ValidationTerminalEvidenceError(FilingCatalogError):
    """Raised when terminal-evidence inputs violate the frozen research contract."""


@dataclass(frozen=True)
class ValidationTerminalEvidenceResult:
    """Immutable terminal-evidence plan or audit."""

    output_root: Path
    report: dict[str, Any]


def calculate_validation_terminal_filing_targets(
    repair_queue: pd.DataFrame,
    security_outcomes: pd.DataFrame,
    filings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select nearby SEC completion and Form 25 filings without reading returns."""
    _require_columns(
        repair_queue,
        {
            "outcome_target_id",
            "cik",
            "ticker",
            "terminal_value_required_events",
            "repair_selection_uses_return_values",
        },
        "terminal repair queue",
    )
    _require_columns(
        security_outcomes,
        {"outcome_target_id", "delisting_date"},
        "security listing outcomes",
    )
    _require_columns(
        filings,
        {
            "cik",
            "accession_number",
            "form",
            "filing_date",
            "accepted_at",
            "primary_document",
            "items",
            "source_url",
        },
        "SEC filing catalog",
    )
    if repair_queue["repair_selection_uses_return_values"].astype(bool).any():
        raise ValidationTerminalEvidenceError("Terminal repair selection used return values")
    required = repair_queue.loc[repair_queue["terminal_value_required_events"].gt(0)].copy()
    required = required.merge(
        security_outcomes[["outcome_target_id", "delisting_date"]],
        on="outcome_target_id",
        how="left",
        validate="one_to_one",
    )
    if required.empty:
        raise ValidationTerminalEvidenceError("No terminal-value repair target exists")
    if required["delisting_date"].isna().any():
        raise ValidationTerminalEvidenceError("A terminal target lacks an evidenced delisting date")
    filing_dates = filings.copy()
    filing_dates["_filing_date"] = pd.to_datetime(filing_dates["filing_date"], errors="coerce")
    target_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for repair in required.itertuples(index=False):
        delisting_date = pd.Timestamp(str(repair.delisting_date))
        issuer_filings = filing_dates.loc[
            filing_dates["cik"].eq(str(repair.cik))
            & filing_dates["_filing_date"].between(
                delisting_date - pd.Timedelta(days=3),
                delisting_date + pd.Timedelta(days=7),
            )
        ].copy()
        completion = issuer_filings.loc[
            issuer_filings["form"].isin(["8-K", "8-K/A"])
            & issuer_filings["items"].fillna("").str.contains(r"(?:^|,)2\.01(?:,|$)")
            & issuer_filings["items"].fillna("").str.contains(r"(?:^|,)3\.01(?:,|$)")
        ].copy()
        form25 = issuer_filings.loc[issuer_filings["form"].isin(["25", "25-NSE"])].copy()
        chosen_completion = _nearest_filing(completion, delisting_date)
        chosen_form25 = _nearest_filing(form25, delisting_date)
        selection_status = (
            "completion_and_form25_selected"
            if chosen_completion is not None and chosen_form25 is not None
            else "completion_selected_form25_missing"
            if chosen_completion is not None
            else "completion_filing_missing"
        )
        selection_rows.append(
            {
                "outcome_target_id": str(repair.outcome_target_id),
                "cik": str(repair.cik),
                "ticker": str(repair.ticker),
                "delisting_date": str(repair.delisting_date),
                "completion_candidate_count": len(completion),
                "form25_candidate_count": len(form25),
                "selection_status": selection_status,
                "selection_uses_return_values": False,
                "terminal_values_loaded": False,
            }
        )
        for role, selected in (
            ("terminal_completion_primary", chosen_completion),
            ("delisting_form25_corroboration", chosen_form25),
        ):
            if selected is None:
                continue
            accession = str(selected["accession_number"])
            target_rows.append(
                {
                    "target_id": _stable_hash(
                        {
                            "outcome_target_id": str(repair.outcome_target_id),
                            "accession_number": accession,
                            "evidence_role": role,
                            "version": VALIDATION_TERMINAL_VERSION,
                        }
                    ),
                    "outcome_target_id": str(repair.outcome_target_id),
                    "cik": str(repair.cik),
                    "ticker": str(repair.ticker),
                    "delisting_date": str(repair.delisting_date),
                    "evidence_role": role,
                    "accession_number": accession,
                    "form": str(selected["form"]),
                    "filing_date": str(selected["filing_date"]),
                    "accepted_at": str(selected["accepted_at"]),
                    "primary_document": str(selected["primary_document"]),
                    "primary_document_url": str(selected["source_url"]),
                    "index_url": _filing_index_url(str(repair.cik), accession),
                    "selection_uses_return_values": False,
                    "terminal_values_loaded": False,
                }
            )
    targets = pd.DataFrame(target_rows)
    if targets.empty:
        raise ValidationTerminalEvidenceError("No SEC terminal filing target could be selected")
    targets = targets.sort_values(
        ["delisting_date", "cik", "evidence_role", "accession_number"], kind="stable"
    ).reset_index(drop=True)
    targets["acquisition_order"] = np.arange(1, len(targets) + 1)
    selections = pd.DataFrame(selection_rows).sort_values(["ticker", "cik"]).reset_index(drop=True)
    return targets, selections


def parse_cash_terminal_filing(payload: bytes) -> dict[str, Any]:
    """Extract a completed all-cash common-share terminal value from SEC filing bytes."""
    text = _visible_text(payload)
    normalized = text.replace("“", '"').replace("”", '"').replace("’", "'")
    completion_observed = bool(
        re.search(r"Closing of the Merger", normalized, flags=re.IGNORECASE)
        and re.search(r"completed the acquisition", normalized, flags=re.IGNORECASE)
    )
    date_matches = re.findall(
        r"On\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s+\(the\s+[\"']Closing Date[\"']\)",
        normalized,
    )
    parsed_dates = sorted(
        {datetime.strptime(value, "%B %d, %Y").date().isoformat() for value in date_matches}
    )
    price_matches = re.findall(
        r"(?:converted into the right to receive|right to receive)"
        r"[^$]{0,220}\$\s*([0-9][0-9,]*(?:\.\d+)?)\s+per\s+share",
        normalized,
        flags=re.IGNORECASE,
    )
    prices = sorted({float(value.replace(",", "")) for value in price_matches})
    cash_only_observed = bool(
        re.search(
            r"converted into the right to receive cash[^$]{0,160}\$",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    without_interest_observed = bool(
        re.search(r"without interest", normalized, flags=re.IGNORECASE)
    )
    ready = bool(
        completion_observed
        and len(parsed_dates) == 1
        and len(prices) == 1
        and cash_only_observed
        and without_interest_observed
    )
    return {
        "completion_observed": completion_observed,
        "terminal_effective_date": parsed_dates[0] if len(parsed_dates) == 1 else None,
        "terminal_value": prices[0] if len(prices) == 1 else np.nan,
        "terminal_value_currency": "USD" if len(prices) == 1 else None,
        "cash_only_observed": cash_only_observed,
        "without_interest_observed": without_interest_observed,
        "date_candidate_count": len(parsed_dates),
        "terminal_value_candidate_count": len(prices),
        "terminal_evidence_ready": ready,
        "applicable_security_scope": "non_excluded_common_shares",
        "parser_version": VALIDATION_TERMINAL_VERSION,
    }


def calculate_resolved_terminal_event_outcomes(
    events: pd.DataFrame,
    terminal_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Apply verified terminal values only to events that explicitly require them."""
    _require_columns(
        events,
        {
            "outcome_target_id",
            "formal_entry_session_date",
            "formal_exit_session_date",
            "terminal_value_required",
            "market_outcome_ready",
            "returns_computed",
        },
        "event outcome ledger",
    )
    _require_columns(
        terminal_evidence,
        {
            "outcome_target_id",
            "terminal_effective_date",
            "terminal_value",
            "terminal_value_currency",
            "terminal_evidence_ready",
            "source_sha256",
            "accession_number",
        },
        "terminal evidence",
    )
    if events["returns_computed"].astype(bool).any():
        raise ValidationTerminalEvidenceError("Terminal resolution received computed returns")
    ready = terminal_evidence.loc[terminal_evidence["terminal_evidence_ready"].astype(bool)].copy()
    if ready["outcome_target_id"].duplicated().any():
        raise ValidationTerminalEvidenceError("Multiple ready terminal values exist for one target")
    columns = [
        "outcome_target_id",
        "terminal_effective_date",
        "terminal_value",
        "terminal_value_currency",
        "source_sha256",
        "accession_number",
    ]
    evidence_columns = ready[columns].rename(
        columns={
            "source_sha256": "terminal_source_sha256",
            "accession_number": "terminal_source_accession_number",
        }
    )
    resolved = events.merge(
        evidence_columns, on="outcome_target_id", how="left", validate="many_to_one"
    )
    entry_dates = pd.to_datetime(resolved["formal_entry_session_date"], errors="coerce")
    exit_dates = pd.to_datetime(resolved["formal_exit_session_date"], errors="coerce")
    terminal_dates = pd.to_datetime(resolved["terminal_effective_date"], errors="coerce")
    applicable = (
        resolved["terminal_value_required"].astype(bool)
        & terminal_dates.notna()
        & terminal_dates.ge(entry_dates)
        & terminal_dates.le(exit_dates)
        & resolved["terminal_value"].notna()
    )
    resolved["terminal_value_observed"] = applicable
    resolved["resolved_market_outcome_ready"] = (
        resolved["market_outcome_ready"].astype(bool) | applicable
    )
    resolved["resolved_outcome_status"] = resolved["outcome_exception_status"]
    resolved.loc[applicable, "resolved_outcome_status"] = "ready_with_sec_terminal_value"
    resolved["returns_computed"] = False
    return resolved


def build_validation_terminal_evidence_plan(
    delisting_audit_root: Path,
    catalog_root: Path,
    *,
    output_root: Path = Path("data/sec/validation_terminal_evidence/targets"),
) -> ValidationTerminalEvidenceResult:
    """Freeze exact SEC filing targets for terminal-value repair."""
    delisting_manifest = _sha256_file(delisting_audit_root / "output_manifest.json")
    catalog_manifest = _sha256_file(catalog_root / "output_manifest.json")
    targets, selections = calculate_validation_terminal_filing_targets(
        load_catalog_table(delisting_audit_root, "validation_outcome_repair_queue.parquet"),
        load_catalog_table(delisting_audit_root, "validation_security_listing_outcomes.parquet"),
        load_catalog_table(catalog_root, "filing_catalog.parquet"),
    )
    payload = json.dumps(
        {
            "version": VALIDATION_TERMINAL_VERSION,
            "delisting_manifest_sha256": delisting_manifest,
            "catalog_manifest_sha256": catalog_manifest,
            "targets": targets["target_id"].tolist(),
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_TERMINAL_VERSION + "-targets-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationTerminalEvidenceResult(
            target.resolve(), _read_json(target / "target_report.json")
        )
    if target.exists():
        raise ValidationTerminalEvidenceError(f"Incomplete terminal plan exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_terminal_filing_targets.parquet": targets,
        "validation_terminal_selection_audit.parquet": selections,
    }
    artifacts = _write_frames(staging, outputs)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "delisting_manifest_sha256": delisting_manifest,
        "catalog_manifest_sha256": catalog_manifest,
        "terminal_securities": selections["outcome_target_id"].nunique(),
        "filing_targets": len(targets),
        "completion_filings": int(targets["evidence_role"].eq("terminal_completion_primary").sum()),
        "form25_filings": int(targets["evidence_role"].eq("delisting_form25_corroboration").sum()),
        "selection_uses_return_values": False,
        "terminal_values_loaded": False,
        "returns_computed": False,
        "artifacts": artifacts,
    }
    _commit(staging, target, report, "target_report.json", artifacts)
    return ValidationTerminalEvidenceResult(target.resolve(), report)


def build_validation_terminal_evidence_audit(
    plan_root: Path,
    delisting_audit_root: Path,
    *,
    raw_root: Path = Path("data/sec/validation_terminal_evidence"),
    output_root: Path = Path("data/processed/validation_terminal_evidence"),
) -> ValidationTerminalEvidenceResult:
    """Parse preserved SEC filings and resolve eligible event terminal outcomes."""
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    delisting_manifest = _sha256_file(delisting_audit_root / "output_manifest.json")
    plan_report = _read_json(plan_root / "target_report.json")
    if plan_report.get("delisting_manifest_sha256") != delisting_manifest:
        raise ValidationTerminalEvidenceError("Terminal plan does not tie to delisting audit")
    targets = load_catalog_table(plan_root, "validation_terminal_filing_targets.parquet")
    snapshots = {
        (snapshot.target_id, snapshot.artifact_kind): snapshot
        for snapshot in load_filing_artifact_snapshots(raw_root)
    }
    observation_rows: list[dict[str, Any]] = []
    for filing_target in targets.itertuples(index=False):
        primary = snapshots.get((str(filing_target.target_id), "primary"))
        common = {
            "target_id": str(filing_target.target_id),
            "outcome_target_id": str(filing_target.outcome_target_id),
            "cik": str(filing_target.cik),
            "ticker": str(filing_target.ticker),
            "evidence_role": str(filing_target.evidence_role),
            "accession_number": str(filing_target.accession_number),
            "form": str(filing_target.form),
            "accepted_at": str(filing_target.accepted_at),
            "artifact_preserved": primary is not None,
            "source_sha256": primary.sha256 if primary else None,
            "source_path": primary.path if primary else None,
        }
        if primary is None:
            parsed = _empty_terminal_parse()
        elif str(filing_target.evidence_role) == "terminal_completion_primary":
            parsed = parse_cash_terminal_filing(Path(primary.path).read_bytes())
        else:
            parsed = _empty_terminal_parse()
            parsed["parser_version"] = "form25-presence-corroboration-v1"
        observation_rows.append({**common, **parsed})
    observations = pd.DataFrame(observation_rows)
    terminal = observations.loc[
        observations["evidence_role"].eq("terminal_completion_primary")
    ].copy()
    events = calculate_resolved_terminal_event_outcomes(
        load_catalog_table(delisting_audit_root, "validation_event_listing_outcomes.parquet"),
        terminal,
    )
    remaining = events.loc[~events["resolved_market_outcome_ready"].astype(bool)].copy()
    queue_input = events.copy()
    queue_input["market_outcome_ready"] = queue_input["resolved_market_outcome_ready"]
    queue = calculate_validation_outcome_repair_queue(queue_input)
    source_hashes = sorted(observations["source_sha256"].dropna().astype(str).unique().tolist())
    payload = json.dumps(
        {
            "version": VALIDATION_TERMINAL_VERSION,
            "plan_manifest_sha256": plan_manifest,
            "delisting_manifest_sha256": delisting_manifest,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_TERMINAL_VERSION + "-audit-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationTerminalEvidenceResult(
            target.resolve(), _read_json(target / "validation_terminal_evidence_report.json")
        )
    if target.exists():
        raise ValidationTerminalEvidenceError(f"Incomplete terminal audit exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    gates = pd.DataFrame(
        [
            _gate(
                "terminal_filing_collection",
                int(observations["artifact_preserved"].sum()),
                len(targets),
                bool(observations["artifact_preserved"].all()),
                "Every selected primary SEC artifact is preserved and hash-verified",
            ),
            _gate(
                "terminal_value_evidence",
                int(terminal["terminal_evidence_ready"].sum()),
                len(terminal),
                bool(terminal["terminal_evidence_ready"].all()),
                "Each required terminal value has an unambiguous completed cash consideration",
            ),
            _gate(
                "resolved_market_outcome",
                int(events["resolved_market_outcome_ready"].sum()),
                len(events),
                bool(events["resolved_market_outcome_ready"].all()),
                "Each event has a formal price outcome or evidenced terminal value",
            ),
            _gate(
                "return_computation",
                0,
                len(events),
                False,
                "This evidence phase does not compute returns",
            ),
        ]
    )
    outputs = {
        "validation_terminal_filing_targets.parquet": targets,
        "validation_terminal_evidence_observations.parquet": observations,
        "validation_resolved_event_outcomes.parquet": events,
        "validation_remaining_outcome_exceptions.parquet": remaining,
        "validation_remaining_outcome_repair_queue.parquet": queue,
        "validation_terminal_evidence_gates.parquet": gates,
    }
    artifacts = _write_frames(staging, outputs)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_plan_manifest_sha256": plan_manifest,
        "delisting_manifest_sha256": delisting_manifest,
        "filing_targets": len(targets),
        "filing_artifacts_preserved": int(observations["artifact_preserved"].sum()),
        "terminal_values_required": len(terminal),
        "terminal_values_ready": int(terminal["terminal_evidence_ready"].sum()),
        "event_components": len(events),
        "resolved_market_outcomes": int(events["resolved_market_outcome_ready"].sum()),
        "remaining_event_exceptions": len(remaining),
        "remaining_repair_securities": len(queue),
        "terminal_values_observed": int(events["terminal_value_observed"].sum()),
        "zero_terminal_value_imputed": False,
        "selection_uses_return_values": False,
        "returns_computed": False,
        "return_outcome_read_authorized": False,
        "artifacts": artifacts,
    }
    _commit(
        staging,
        target,
        report,
        "validation_terminal_evidence_report.json",
        artifacts,
    )
    return ValidationTerminalEvidenceResult(target.resolve(), report)


def _nearest_filing(candidates: pd.DataFrame, event_date: pd.Timestamp) -> pd.Series[Any] | None:
    if candidates.empty:
        return None
    ranked = candidates.copy()
    ranked["_distance"] = (ranked["_filing_date"] - event_date).abs().dt.days
    return ranked.sort_values(["_distance", "accepted_at", "accession_number"], kind="stable").iloc[
        0
    ]


def _visible_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    without_scripts = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", decoded, flags=re.IGNORECASE | re.DOTALL
    )
    stripped = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", html.unescape(stripped)).strip()


def _empty_terminal_parse() -> dict[str, Any]:
    return {
        "completion_observed": False,
        "terminal_effective_date": None,
        "terminal_value": np.nan,
        "terminal_value_currency": None,
        "cash_only_observed": False,
        "without_interest_observed": False,
        "date_candidate_count": 0,
        "terminal_value_candidate_count": 0,
        "terminal_evidence_ready": False,
        "applicable_security_scope": None,
        "parser_version": VALIDATION_TERMINAL_VERSION,
    }


def _filing_index_url(cik: str, accession_number: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number.replace('-', '')}/{accession_number}-index.html"
    )


def _gate(
    gate: str, numerator: int, denominator: int, passed: bool, meaning: str
) -> dict[str, Any]:
    return {
        "gate": gate,
        "numerator": numerator,
        "denominator": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
        "passed": passed,
        "meaning": meaning,
    }


def _write_frames(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    return artifacts


def _commit(
    staging: Path,
    target: Path,
    report: dict[str, Any],
    report_name: str,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationTerminalEvidenceError(f"{label} lacks required columns: {missing}")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
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
