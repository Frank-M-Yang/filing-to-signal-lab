"""Point-in-time prior-guidance evidence from preserved SEC 8-K submissions."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from sec_data.client import SecClient, SecRequestError, normalize_cik
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.filing_text import extract_visible_filing_text

GUIDANCE_PLAN_VERSION: Final[str] = "prior-guidance-plan-v1"
GUIDANCE_EVIDENCE_VERSION: Final[str] = "prior-guidance-evidence-v4"
COMPLETE_SUBMISSION_URL: Final[str] = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_dir}/{accession}.txt"
)
_FORWARD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:guidance|outlook|expect(?:s|ed|ing)?|forecast(?:s|ed|ing)?|"
    r"project(?:ed|ing|ion|ions)|anticipat(?:e|es|ed|ing)|estimate(?:s|d)?|"
    r"will\s+be|should\s+be|target(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_GROSS_MARGIN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bgross\s+(?:profit\s+)?margins?\b", re.IGNORECASE
)
_PERCENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent(?:age)?)(?!\w)",
    re.IGNORECASE,
)
_DIRECTION_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "raised": re.compile(r"\b(?:raise[ds]?|increas(?:e|ed|es|ing)|improv(?:e|ed|es|ing))\b", re.I),
    "lowered": re.compile(
        r"\b(?:lower(?:ed|s|ing)?|reduc(?:e|ed|es|ing)|decreas(?:e|ed|es|ing))\b", re.I
    ),
    "reaffirmed": re.compile(r"\b(?:reaffirm(?:ed|s|ing)?|maintain(?:ed|s|ing)?)\b", re.I),
    "withdrawn": re.compile(r"\b(?:withdraw(?:n|s|ing)?|suspend(?:ed|s|ing)?)\b", re.I),
}


class GuidanceEvidenceError(FilingCatalogError):
    """Raised when guidance evidence violates the frozen as-of contract."""


@dataclass(frozen=True)
class GuidanceEvidenceConfig:
    """Frozen prior-filing window, collection, and coverage gates."""

    minimum_age_days: int = 30
    maximum_age_days: int = 240
    minimum_document_coverage: float = 0.95
    minimum_exhibit_text_coverage: float = 0.90
    minimum_numeric_gross_margin_coverage: float = 0.50
    maximum_numeric_midpoint_spread: float = 0.02
    plan_version: str = GUIDANCE_PLAN_VERSION
    evidence_version: str = GUIDANCE_EVIDENCE_VERSION

    def validate(self) -> None:
        """Reject windows or gates that weaken the intended prior-information test."""
        if self.minimum_age_days < 1 or self.maximum_age_days <= self.minimum_age_days:
            raise ValueError("Prior-guidance age bounds are invalid")
        for value in (
            self.minimum_document_coverage,
            self.minimum_exhibit_text_coverage,
            self.minimum_numeric_gross_margin_coverage,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError("Guidance coverage gates must be in (0, 1]")
        if not 0.0 < self.maximum_numeric_midpoint_spread <= 0.10:
            raise ValueError("Gross-margin ambiguity threshold is invalid")


@dataclass(frozen=True)
class GuidancePlanTables:
    """Pure event links and unique document targets before collection."""

    event_links: pd.DataFrame
    document_targets: pd.DataFrame
    sample_funnel: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class GuidanceCollectionResult:
    """Counts from one resumable SEC complete-submission batch."""

    selected_documents: int
    completed_documents: int
    failed_documents: int
    requests_made: int


@dataclass(frozen=True)
class GuidanceEvidenceTables:
    """Parsed document, sentence, and event-level guidance evidence."""

    document_audit: pd.DataFrame
    sentence_evidence: pd.DataFrame
    event_features: pd.DataFrame
    coverage_gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class GuidanceEvidenceResult:
    """Immutable guidance plan or parsed evidence bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_guidance_plan(
    market_state: pd.DataFrame,
    filing_catalog: pd.DataFrame,
    config: GuidanceEvidenceConfig | None = None,
) -> GuidancePlanTables:
    """Select the latest prior Item 2.02 8-K without reading current filings or returns."""
    cfg = config or GuidanceEvidenceConfig()
    cfg.validate()
    _require_columns(
        market_state,
        {"validation_event_id", "cik", "accepted_at", "market_state_ready"},
        "market-state events",
    )
    _require_columns(
        filing_catalog,
        {
            "cik",
            "accession_number",
            "form",
            "accepted_at",
            "items",
            "primary_document",
            "source_url",
        },
        "SEC filing catalog",
    )
    events = market_state.loc[
        market_state["market_state_ready"].astype(bool),
        ["validation_event_id", "cik", "accepted_at"],
    ].copy()
    events["accepted_at"] = _utc(events["accepted_at"], "event accepted_at")
    events["cik"] = events["cik"].astype(str).map(normalize_cik)
    filings = filing_catalog.loc[
        filing_catalog["cik"].astype(str).isin(events["cik"])
        & filing_catalog["form"].isin(["8-K", "8-K/A"])
    ].copy()
    filings["cik"] = filings["cik"].astype(str).map(normalize_cik)
    filings["accepted_at"] = _utc(filings["accepted_at"], "8-K accepted_at")
    filings["item_202"] = filings["items"].map(lambda value: _has_item(value, "2.02"))
    filings["item_701"] = filings["items"].map(lambda value: _has_item(value, "7.01"))
    link_rows = [
        _select_prior_guidance(event, filings, cfg) for event in events.itertuples(index=False)
    ]
    links = pd.DataFrame(link_rows)
    selected = links.loc[links["prior_guidance_candidate_ready"].astype(bool)].copy()
    document_columns = [
        "guidance_document_id",
        "cik",
        "prior_accession_number",
        "prior_form",
        "prior_accepted_at",
        "prior_items",
        "primary_document",
        "complete_submission_url",
    ]
    documents = (
        selected[document_columns]
        .drop_duplicates("guidance_document_id")
        .sort_values(["prior_accepted_at", "cik", "guidance_document_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    documents["document_acquisition_order"] = np.arange(1, len(documents) + 1)
    documents["selection_uses_return_values"] = False
    documents["outcome_values_loaded"] = False
    funnel = pd.DataFrame(
        [
            _funnel("market_state_ready_events", len(events), events["cik"].nunique()),
            _funnel(
                "prior_item_202_candidate",
                int(links["prior_item_202_ready"].sum()),
                links.loc[links["prior_item_202_ready"], "cik"].nunique(),
            ),
            _funnel(
                "prior_guidance_document_selected",
                len(selected),
                selected["cik"].nunique(),
            ),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.plan_version,
        "market_state_ready_events": len(events),
        "market_state_ready_issuers": int(events["cik"].nunique()),
        "prior_item_202_events": int(links["prior_item_202_ready"].sum()),
        "selected_guidance_events": len(selected),
        "selected_guidance_issuers": int(selected["cik"].nunique()),
        "unique_guidance_documents": len(documents),
        "candidate_coverage": float(len(selected) / len(events)) if len(events) else 0.0,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "current_filing_text_loaded": False,
        "documents_collected": False,
        "config": _plan_config(cfg),
    }
    return GuidancePlanTables(links, documents, funnel, report)


def build_guidance_plan(
    market_state_root: Path,
    filing_catalog_root: Path,
    *,
    output_root: Path = Path("data/sec/guidance_evidence/targets"),
    config: GuidanceEvidenceConfig | None = None,
) -> GuidanceEvidenceResult:
    """Persist prior-guidance targets before any filing document is downloaded."""
    cfg = config or GuidanceEvidenceConfig()
    cfg.validate()
    input_hashes = {
        "market_state": _sha256_file(market_state_root / "output_manifest.json"),
        "filing_catalog": _sha256_file(filing_catalog_root / "output_manifest.json"),
    }
    tables = calculate_guidance_plan(
        load_catalog_table(market_state_root, "validation_market_state_features.parquet"),
        load_catalog_table(filing_catalog_root, "filing_catalog.parquet"),
        cfg,
    )
    payload = json.dumps(
        {"input_manifest_sha256": input_hashes, "config": _plan_config(cfg)},
        sort_keys=True,
    ).encode()
    dataset_version = cfg.plan_version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return GuidanceEvidenceResult(
            target.resolve(), _read_json(target / "guidance_plan_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete guidance-plan directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "guidance_event_links.parquet": tables.event_links,
        "guidance_document_targets.parquet": tables.document_targets,
        "guidance_sample_funnel.parquet": tables.sample_funnel,
    }
    artifacts = _write_tables(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "guidance_plan_report.json"
    _finalize(staging, target, report_path, report, artifacts, input_hashes)
    return GuidanceEvidenceResult(target.resolve(), report)


def collect_guidance_documents(
    targets: pd.DataFrame,
    client: SecClient,
    *,
    raw_root: Path = Path("data/sec/guidance_evidence"),
    limit: int = 20,
    progress: Callable[[int, int, str], None] | None = None,
) -> GuidanceCollectionResult:
    """Collect exact complete-submission bytes in a resumable content-addressed store."""
    if limit < 1:
        raise ValueError("Guidance collection limit must be positive")
    _require_columns(
        targets,
        {
            "guidance_document_id",
            "complete_submission_url",
            "document_acquisition_order",
        },
        "guidance document targets",
    )
    ordered = targets.sort_values("document_acquisition_order", kind="mergesort")
    pending = [
        row
        for row in ordered.itertuples(index=False)
        if not _valid_completion(raw_root, str(row.guidance_document_id))
    ][:limit]
    completed = 0
    failed = 0
    requests = 0
    for number, row in enumerate(pending, start=1):
        document_id = str(row.guidance_document_id)
        try:
            response = client.get_filing_artifact(str(row.complete_submission_url))
            requests += 1
            _preserve_guidance_response(raw_root, row, response.content, response)
            completed += 1
        except SecRequestError as exc:
            failed += 1
            requests += 1
            _write_failure(raw_root, document_id, str(exc))
        if progress is not None:
            progress(number, len(pending), document_id)
    return GuidanceCollectionResult(len(pending), completed, failed, requests)


def calculate_guidance_evidence(
    event_links: pd.DataFrame,
    document_targets: pd.DataFrame,
    raw_root: Path,
    config: GuidanceEvidenceConfig | None = None,
) -> GuidanceEvidenceTables:
    """Parse preserved EX-99 text and expose auditable event-level guidance features."""
    cfg = config or GuidanceEvidenceConfig()
    cfg.validate()
    audits: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    for target in document_targets.itertuples(index=False):
        audit, evidence = _audit_document(target, raw_root, cfg)
        audits.append(audit)
        sentences.extend(evidence)
    document_audit = pd.DataFrame(audits)
    sentence_evidence = pd.DataFrame(sentences, columns=_sentence_columns())
    event_features = event_links.merge(
        document_audit,
        on=["guidance_document_id", "cik", "prior_accession_number"],
        how="left",
        validate="many_to_one",
    )
    for column in (
        "document_collected",
        "exhibit_text_ready",
        "forward_language_ready",
        "numeric_gross_margin_guidance_ready",
    ):
        event_features[column] = event_features[column].fillna(False).astype(bool)
    candidate = event_features["prior_guidance_candidate_ready"].astype(bool)
    denominator = int(candidate.sum())
    collected = int((candidate & event_features["document_collected"]).sum())
    exhibit = int((candidate & event_features["exhibit_text_ready"]).sum())
    numeric = int((candidate & event_features["numeric_gross_margin_guidance_ready"]).sum())
    any_numeric = int(
        (candidate & event_features["gross_margin_guidance_sentence_count"].gt(0)).sum()
    )
    gates = pd.DataFrame(
        [
            _coverage_gate(
                "complete_submission_collection",
                collected,
                denominator,
                cfg.minimum_document_coverage,
            ),
            _coverage_gate(
                "exhibit_text_extraction",
                exhibit,
                denominator,
                cfg.minimum_exhibit_text_coverage,
            ),
            _coverage_gate(
                "gaap_compatible_numeric_gross_margin_guidance",
                numeric,
                denominator,
                cfg.minimum_numeric_gross_margin_coverage,
            ),
        ]
    )
    report: dict[str, Any] = {
        "contract_version": cfg.evidence_version,
        "guidance_candidate_events": denominator,
        "guidance_candidate_issuers": int(event_features.loc[candidate, "cik"].nunique()),
        "unique_document_targets": len(document_targets),
        "documents_collected": int(document_audit["document_collected"].sum()),
        "documents_with_exhibit_text": int(document_audit["exhibit_text_ready"].sum()),
        "events_with_forward_language": int(
            (candidate & event_features["forward_language_ready"]).sum()
        ),
        "events_with_any_numeric_gross_margin_language": any_numeric,
        "events_with_gaap_compatible_numeric_gross_margin_guidance": numeric,
        "issuers_with_gaap_compatible_numeric_gross_margin_guidance": int(
            event_features.loc[
                candidate & event_features["numeric_gross_margin_guidance_ready"],
                "cik",
            ].nunique()
        ),
        "gaap_compatible_numeric_gross_margin_coverage": (
            numeric / denominator if denominator else 0.0
        ),
        "document_collection_gate": bool(
            gates.loc[gates["gate_id"].eq("complete_submission_collection"), "passed"].iloc[0]
        ),
        "exhibit_text_gate": bool(
            gates.loc[gates["gate_id"].eq("exhibit_text_extraction"), "passed"].iloc[0]
        ),
        "numeric_gross_margin_model_gate": bool(
            gates.loc[
                gates["gate_id"].eq("gaap_compatible_numeric_gross_margin_guidance"),
                "passed",
            ].iloc[0]
        ),
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "current_filing_text_loaded": False,
        "model_run_authorized": bool(gates["passed"].all()),
        "config": asdict(cfg),
    }
    return GuidanceEvidenceTables(document_audit, sentence_evidence, event_features, gates, report)


def build_guidance_evidence_audit(
    plan_root: Path,
    *,
    raw_root: Path = Path("data/sec/guidance_evidence"),
    output_root: Path = Path("data/processed/guidance_evidence"),
    config: GuidanceEvidenceConfig | None = None,
) -> GuidanceEvidenceResult:
    """Persist parsed guidance evidence with hashes tied to the frozen plan."""
    cfg = config or GuidanceEvidenceConfig()
    cfg.validate()
    plan_manifest = _sha256_file(plan_root / "output_manifest.json")
    links = load_catalog_table(plan_root, "guidance_event_links.parquet")
    targets = load_catalog_table(plan_root, "guidance_document_targets.parquet")
    tables = calculate_guidance_evidence(links, targets, raw_root, cfg)
    source_hashes = sorted(
        tables.document_audit.loc[
            tables.document_audit["document_collected"], "source_sha256"
        ].astype(str)
    )
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest,
            "source_hashes": source_hashes,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.evidence_version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return GuidanceEvidenceResult(
            target.resolve(), _read_json(target / "guidance_evidence_report.json")
        )
    if target.exists():
        raise GuidanceEvidenceError(f"Incomplete guidance-evidence directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "guidance_document_audit.parquet": tables.document_audit,
        "guidance_sentence_evidence.parquet": tables.sentence_evidence,
        "guidance_event_features.parquet": tables.event_features,
        "guidance_coverage_gates.parquet": tables.coverage_gates,
    }
    artifacts = _write_tables(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_manifest_sha256": plan_manifest,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "guidance_evidence_report.json"
    _finalize(
        staging,
        target,
        report_path,
        report,
        artifacts,
        {"plan": plan_manifest},
    )
    return GuidanceEvidenceResult(target.resolve(), report)


def parse_complete_submission(content: bytes) -> list[dict[str, Any]]:
    """Extract text-bearing SEC SGML document blocks from exact submission bytes."""
    decoded = content.decode("latin-1", errors="replace")
    rows: list[dict[str, Any]] = []
    for occurrence, match in enumerate(
        re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", decoded, flags=re.I | re.S)
    ):
        block = match.group(1)
        document_type = _sgml_field(block, "TYPE")
        filename = _sgml_field(block, "FILENAME")
        description = _sgml_field(block, "DESCRIPTION")
        text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.I | re.S)
        if text_match is None:
            continue
        text_bytes = text_match.group(1).encode("latin-1", errors="replace")
        if document_type.upper() in {"GRAPHIC", "ZIP", "PDF", "XML"}:
            continue
        visible = extract_visible_filing_text(text_bytes)
        rows.append(
            {
                "source_occurrence": occurrence,
                "document_type": document_type,
                "filename": filename,
                "description": description,
                "document_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "visible_text": visible,
            }
        )
    return rows


def extract_guidance_sentences(
    text: str, maximum_midpoint_spread: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract forward-looking evidence and conservative gross-margin percentages."""
    raw_fragments = [
        re.sub(r"\s+", " ", value).strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]
    fragments = list(dict.fromkeys(raw_fragments))
    document_non_gaap_context = any(
        _GROSS_MARGIN_PATTERN.search(sentence)
        and re.search(r"\bnon-GAAP\b", sentence, flags=re.I)
        and re.search(r"\b(?:guidance|projection|forecast)\b", sentence, flags=re.I)
        for sentence in fragments
    )
    evidence: list[dict[str, Any]] = []
    compatible_ranges: list[tuple[float, float, str]] = []
    numeric_gross_margin_sentences = 0
    direction_counts = {key: 0 for key in _DIRECTION_PATTERNS}
    forward_count = 0
    for index, sentence in enumerate(fragments):
        if not _FORWARD_PATTERN.search(sentence):
            continue
        forward_count += 1
        for key, pattern in _DIRECTION_PATTERNS.items():
            direction_counts[key] += len(pattern.findall(sentence))
        gross = bool(_GROSS_MARGIN_PATTERN.search(sentence))
        percentages = [
            float(value) / 100.0
            for value in _PERCENT_PATTERN.findall(sentence)
            if 0.0 <= float(value) <= 100.0
        ]
        low, high, basis, numeric_candidate = _gross_margin_measurement(
            sentence,
            percentages,
            document_non_gaap_context=document_non_gaap_context,
        )
        if gross and percentages:
            numeric_gross_margin_sentences += 1
        if numeric_candidate and basis in {"gaap", "unspecified"}:
            compatible_ranges.append((low, high, basis))
        evidence.append(
            {
                "sentence_index": index,
                "sentence": sentence,
                "gross_margin_mention": gross,
                "percentages_json": json.dumps(percentages),
                "gross_margin_low": low,
                "gross_margin_high": high,
                "gross_margin_basis": basis,
                "gaap_compatible_numeric_candidate": bool(
                    numeric_candidate and basis in {"gaap", "unspecified"}
                ),
            }
        )
    numeric_ready = False
    extraction_status = "no_numeric_gross_margin_guidance"
    selected_low = np.nan
    selected_high = np.nan
    selected_basis = "not_available"
    if compatible_ranges:
        midpoints = np.asarray([(low + high) / 2.0 for low, high, _ in compatible_ranges])
        if (
            len(midpoints) == 1
            or float(midpoints.max() - midpoints.min()) <= maximum_midpoint_spread
        ):
            selected_low = float(np.mean([value[0] for value in compatible_ranges]))
            selected_high = float(np.mean([value[1] for value in compatible_ranges]))
            selected_basis = (
                "gaap" if all(value[2] == "gaap" for value in compatible_ranges) else "unspecified"
            )
            numeric_ready = True
            extraction_status = "gaap_compatible_numeric_guidance_ready"
        else:
            extraction_status = "ambiguous_multiple_gross_margin_statements"
    elif numeric_gross_margin_sentences:
        extraction_status = "non_gaap_or_ambiguous_numeric_guidance_only"
    summary = {
        "forward_language_sentence_count": forward_count,
        "gross_margin_guidance_sentence_count": numeric_gross_margin_sentences,
        "prior_guidance_gross_margin_low": selected_low,
        "prior_guidance_gross_margin_high": selected_high,
        "prior_guidance_gross_margin_midpoint": (
            (selected_low + selected_high) / 2.0 if numeric_ready else np.nan
        ),
        "prior_guidance_gross_margin_basis": selected_basis,
        "numeric_gross_margin_guidance_ready": numeric_ready,
        "gross_margin_extraction_status": extraction_status,
        **{f"guidance_direction_{key}_count": value for key, value in direction_counts.items()},
    }
    return evidence, summary


def _gross_margin_measurement(
    sentence: str,
    percentages: list[float],
    *,
    document_non_gaap_context: bool,
) -> tuple[float, float, str, bool]:
    """Classify one statement without mixing GAAP and adjusted margin values."""
    if not _GROSS_MARGIN_PATTERN.search(sentence) or not percentages:
        return np.nan, np.nan, "not_available", False
    lowered = sentence.lower()
    if re.search(r"\bgaap\s+and\s+non-gaap\s+gross", lowered):
        value = percentages[0]
        return value, value, "gaap", True
    if re.search(r"\bnon-gaap\s+and\s+gaap\s+gross", lowered):
        if len(percentages) < 2:
            return np.nan, np.nan, "ambiguous", False
        value = percentages[1]
        return value, value, "gaap", True
    if re.search(r"\b(?:non-gaap|adjusted)\s+gross", lowered):
        low, high = _numeric_bounds(lowered, percentages)
        return low, high, "non_gaap", True
    low, high = _numeric_bounds(lowered, percentages)
    if np.isnan(low):
        return low, high, "ambiguous", False
    basis = "gaap" if re.search(r"\bgaap\s+gross", lowered) else "unspecified"
    if basis == "unspecified" and document_non_gaap_context:
        basis = "non_gaap"
    return low, high, basis, True


def _numeric_bounds(sentence: str, percentages: list[float]) -> tuple[float, float]:
    """Treat two values as a range only when the sentence explicitly says so."""
    if len(percentages) == 1:
        return percentages[0], percentages[0]
    if re.search(r"\b(?:between|from|range|low end|high end)\b", sentence):
        return min(percentages[:2]), max(percentages[:2])
    return np.nan, np.nan


def _select_prior_guidance(
    event: Any, filings: pd.DataFrame, config: GuidanceEvidenceConfig
) -> dict[str, Any]:
    accepted = pd.Timestamp(cast(Any, event.accepted_at))
    issuer = filings.loc[filings["cik"].eq(str(event.cik))].copy()
    eligible = issuer.loc[
        issuer["accepted_at"].le(accepted - pd.Timedelta(days=config.minimum_age_days))
        & issuer["accepted_at"].ge(accepted - pd.Timedelta(days=config.maximum_age_days))
    ]
    item_202 = eligible.loc[eligible["item_202"].astype(bool)]
    item_701 = eligible.loc[eligible["item_701"].astype(bool)]
    pool = item_202 if not item_202.empty else item_701
    selected = pool.sort_values("accepted_at", kind="mergesort").tail(1)
    base: dict[str, Any] = {
        "validation_event_id": str(event.validation_event_id),
        "cik": str(event.cik),
        "target_accepted_at": accepted,
        "prior_item_202_ready": not item_202.empty,
        "prior_item_701_ready": not item_701.empty,
        "prior_guidance_candidate_ready": not selected.empty,
        "guidance_selection_status": "selected" if not selected.empty else "no_prior_2_02_or_7_01",
        "guidance_document_id": pd.NA,
        "prior_accession_number": pd.NA,
        "prior_form": pd.NA,
        "prior_accepted_at": pd.NaT,
        "prior_items": pd.NA,
        "primary_document": pd.NA,
        "complete_submission_url": pd.NA,
        "guidance_age_days": pd.NA,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "current_filing_text_loaded": False,
    }
    if selected.empty:
        return base
    row = selected.iloc[0]
    accession = str(row["accession_number"])
    document_id = _stable_hash({"cik": str(event.cik), "accession_number": accession})
    prior_at = pd.Timestamp(row["accepted_at"])
    base.update(
        {
            "guidance_document_id": document_id,
            "prior_accession_number": accession,
            "prior_form": str(row["form"]),
            "prior_accepted_at": prior_at,
            "prior_items": str(row["items"]),
            "primary_document": str(row["primary_document"]),
            "complete_submission_url": _complete_submission_url(str(event.cik), accession),
            "guidance_age_days": (accepted - prior_at).total_seconds() / 86_400.0,
        }
    )
    return base


def _audit_document(
    target: Any, raw_root: Path, config: GuidanceEvidenceConfig
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_id = str(target.guidance_document_id)
    completion = _load_completion(raw_root, document_id)
    base: dict[str, Any] = {
        "guidance_document_id": document_id,
        "cik": str(target.cik),
        "prior_accession_number": str(target.prior_accession_number),
        "document_collected": completion is not None,
        "source_sha256": pd.NA,
        "submission_document_count": 0,
        "exhibit_document_count": 0,
        "exhibit_text_chars": 0,
        "exhibit_text_ready": False,
        "forward_language_ready": False,
        "parser_status": "not_collected",
        "forward_language_sentence_count": 0,
        "gross_margin_guidance_sentence_count": 0,
        "prior_guidance_gross_margin_low": np.nan,
        "prior_guidance_gross_margin_high": np.nan,
        "prior_guidance_gross_margin_midpoint": np.nan,
        "prior_guidance_gross_margin_basis": "not_available",
        "numeric_gross_margin_guidance_ready": False,
        "gross_margin_extraction_status": "not_collected",
        **{f"guidance_direction_{key}_count": 0 for key in _DIRECTION_PATTERNS},
    }
    if completion is None:
        return base, []
    raw_path = raw_root / str(completion["raw_path"])
    content = raw_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != completion.get("sha256"):
        raise GuidanceEvidenceError(f"Guidance source hash mismatch for {document_id}")
    documents = parse_complete_submission(content)
    exhibits = [
        document
        for document in documents
        if re.fullmatch(r"EX-99(?:\.\d+)?", document["document_type"], flags=re.I)
    ]
    selected_exhibits = exhibits or [
        document for document in documents if document["document_type"].upper() in {"8-K", "8-K/A"}
    ]
    combined = "\n".join(str(document["visible_text"]) for document in selected_exhibits)
    evidence, summary = extract_guidance_sentences(combined, config.maximum_numeric_midpoint_spread)
    sentence_rows: list[dict[str, Any]] = []
    for row in evidence:
        sentence_rows.append(
            {
                "guidance_document_id": document_id,
                "cik": str(target.cik),
                "prior_accession_number": str(target.prior_accession_number),
                "source_sha256": digest,
                **row,
            }
        )
    base.update(
        {
            "source_sha256": digest,
            "submission_document_count": len(documents),
            "exhibit_document_count": len(exhibits),
            "exhibit_text_chars": len(combined),
            "exhibit_text_ready": bool(combined.strip()),
            "forward_language_ready": bool(summary["forward_language_sentence_count"] > 0),
            "parser_status": "ready" if combined.strip() else "no_text_exhibit",
            **summary,
        }
    )
    return base, sentence_rows


def _preserve_guidance_response(raw_root: Path, row: Any, content: bytes, response: Any) -> None:
    digest = hashlib.sha256(content).hexdigest()
    raw_directory = raw_root / "raw" / digest[:2]
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_directory / f"{digest}.txt"
    if raw_path.exists() and _sha256_file(raw_path) != digest:
        raise GuidanceEvidenceError(f"Existing guidance bytes changed: {raw_path}")
    if not raw_path.exists():
        temporary = raw_directory / f".{digest}-{uuid.uuid4().hex}.part"
        temporary.write_bytes(content)
        if _sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise GuidanceEvidenceError("Guidance bytes changed before preservation")
        temporary.replace(raw_path)
    document_id = str(row.guidance_document_id)
    completion = {
        "guidance_document_id": document_id,
        "cik": str(row.cik),
        "accession_number": str(row.prior_accession_number),
        "requested_url": str(row.complete_submission_url),
        "response_url": str(response.url),
        "fetched_at": response.fetched_at.isoformat(),
        "status_code": int(response.status_code),
        "sha256": digest,
        "bytes": len(content),
        "raw_path": raw_path.relative_to(raw_root).as_posix(),
    }
    completion_dir = raw_root / "completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    path = completion_dir / f"{document_id}.json"
    temporary_manifest = completion_dir / f".{document_id}-{uuid.uuid4().hex}.part"
    temporary_manifest.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(path)


def _valid_completion(raw_root: Path, document_id: str) -> bool:
    completion = _load_completion(raw_root, document_id)
    if completion is None:
        return False
    path = raw_root / str(completion.get("raw_path", ""))
    return path.is_file() and _sha256_file(path) == completion.get("sha256")


def _load_completion(raw_root: Path, document_id: str) -> dict[str, Any] | None:
    path = raw_root / "completions" / f"{document_id}.json"
    if not path.is_file():
        return None
    return _read_json(path)


def _write_failure(raw_root: Path, document_id: str, detail: str) -> None:
    directory = raw_root / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{document_id}.json").write_text(
        json.dumps(
            {
                "guidance_document_id": document_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "detail": detail,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _complete_submission_url(cik: str, accession: str) -> str:
    if re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) is None:
        raise GuidanceEvidenceError(f"Unsafe accession number: {accession}")
    return COMPLETE_SUBMISSION_URL.format(
        cik=str(int(normalize_cik(cik))),
        accession_dir=accession.replace("-", ""),
        accession=accession,
    )


def _has_item(value: Any, item: str) -> bool:
    if value is None or bool(pd.isna(value)):
        return False
    return item in {part.strip() for part in str(value).split(",")}


def _sgml_field(block: str, field: str) -> str:
    match = re.search(rf"<{field}>\s*([^\r\n<]+)", block, flags=re.I)
    return match.group(1).strip() if match is not None else ""


def _coverage_gate(
    gate_id: str, observed: int, denominator: int, required_coverage: float
) -> dict[str, Any]:
    coverage = observed / denominator if denominator else 0.0
    return {
        "gate_id": gate_id,
        "observed_events": observed,
        "eligible_events": denominator,
        "coverage": coverage,
        "required_coverage": required_coverage,
        "passed": bool(coverage >= required_coverage),
    }


def _sentence_columns() -> list[str]:
    return [
        "guidance_document_id",
        "cik",
        "prior_accession_number",
        "source_sha256",
        "sentence_index",
        "sentence",
        "gross_margin_mention",
        "percentages_json",
        "gross_margin_low",
        "gross_margin_high",
        "gross_margin_basis",
        "gaap_compatible_numeric_candidate",
    ]


def _funnel(stage: str, events: int, issuers: int) -> dict[str, Any]:
    return {"stage": stage, "events": int(events), "issuers": int(issuers)}


def _plan_config(config: GuidanceEvidenceConfig) -> dict[str, Any]:
    return {
        "minimum_age_days": config.minimum_age_days,
        "maximum_age_days": config.maximum_age_days,
        "plan_version": config.plan_version,
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _write_tables(staging: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    return artifacts


def _finalize(
    staging: Path,
    target: Path,
    report_path: Path,
    report: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> None:
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
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


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GuidanceEvidenceError(f"{label} lacks required columns: {missing}")


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise GuidanceEvidenceError(f"{label} contains invalid timestamps")
    return result


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
