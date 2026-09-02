"""Point-in-time coverage audit for numeric company-guidance metrics."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import load_catalog_table
from sec_data.guidance_evidence import GuidanceEvidenceError, parse_complete_submission

GUIDANCE_METRIC_COVERAGE_VERSION: Final[str] = "guidance-metric-coverage-v6"
GUIDANCE_METRICS: Final[tuple[str, ...]] = (
    "revenue",
    "diluted_eps",
    "operating_margin",
    "gross_margin",
)
_METRIC_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "revenue": re.compile(r"\b(?:revenue|revenues|net sales)\b", re.I),
    "diluted_eps": re.compile(
        r"\b(?:EPS|(?:earnings|income|loss)\s+per\s+(?:diluted\s+)?share)\b",
        re.I,
    ),
    "operating_margin": re.compile(r"\boperating\s+margins?\b", re.I),
    "gross_margin": re.compile(r"\bgross\s+(?:profit\s+)?margins?\b", re.I),
}
_GUIDANCE_WORD: Final[re.Pattern[str]] = re.compile(
    r"\b(?:outlook|guidance|financial guidance|business outlook|expectations)\b",
    re.I,
)
_STOP_HEADING: Final[re.Pattern[str]] = re.compile(
    r"\b(?:conference call|webcast|about\s+|forward-looking statements?|"
    r"non-gaap financial (?:measures|information)|reconciliation|supplemental information|"
    r"investor relations|financial results?|results of operations|financial highlights?|"
    r"business highlights?|balance sheet|cash flow)\b",
    re.I,
)
_PERCENT: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent(?:age)?)(?!\w)",
    re.I,
)
_DOLLAR: Final[re.Pattern[str]] = re.compile(
    r"(\()?\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)(\))?",
    re.I,
)


class GuidanceMetricCoverageError(GuidanceEvidenceError):
    """Raised when the multi-metric coverage contract is violated."""


@dataclass(frozen=True)
class GuidanceMetricCoverageConfig:
    """Frozen parsing limits and pilot coverage gates."""

    maximum_section_lines: int = 80
    minimum_pilot_coverage: float = 0.20
    minimum_pilot_issuers: int = 10
    revenue_consistency_tolerance: float = 0.02
    eps_consistency_tolerance: float = 0.05
    margin_consistency_tolerance: float = 0.005
    version: str = GUIDANCE_METRIC_COVERAGE_VERSION

    def validate(self) -> None:
        """Reject permissive or nonsensical audit settings."""
        if not 10 <= self.maximum_section_lines <= 200:
            raise ValueError("Guidance-section line limit must be between 10 and 200")
        if not 0.0 < self.minimum_pilot_coverage <= 1.0:
            raise ValueError("Pilot coverage gate must be in (0, 1]")
        if self.minimum_pilot_issuers < 2:
            raise ValueError("Pilot issuer gate must be at least two")
        if any(
            value <= 0.0
            for value in (
                self.revenue_consistency_tolerance,
                self.eps_consistency_tolerance,
                self.margin_consistency_tolerance,
            )
        ):
            raise ValueError("Guidance consistency tolerances must be positive")


@dataclass(frozen=True)
class GuidanceMetricCoverageTables:
    """Document observations, event matches, and metric-level coverage gates."""

    document_audit: pd.DataFrame
    observations: pd.DataFrame
    event_metric_features: pd.DataFrame
    coverage: pd.DataFrame
    sanity_checks: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class GuidanceMetricCoverageResult:
    """Immutable persisted multi-metric guidance audit."""

    output_root: Path
    report: dict[str, Any]


def extract_guidance_metric_observations(
    text: str,
    *,
    maximum_section_lines: int = 80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract conservative numeric observations from explicit guidance sections."""
    sections = _guidance_sections(text, maximum_section_lines)
    rows: list[dict[str, Any]] = []
    for section_number, section in enumerate(sections):
        heading = section["heading"]
        lines = section["lines"]
        for line_number, line in enumerate(lines):
            metrics = [
                metric for metric, pattern in _METRIC_PATTERNS.items() if pattern.search(line)
            ]
            for metric in metrics:
                local_context = " ".join(lines[max(0, line_number - 3) : line_number])
                combined = _statement_with_value(lines, line_number, line)
                for statement in _metric_statements(metric, combined):
                    observation = _metric_observation(
                        metric,
                        statement,
                        local_context,
                        heading,
                    )
                    if observation is None:
                        continue
                    rows.append(
                        {
                            "section_number": section_number,
                            "section_heading": heading,
                            "line_number": line_number,
                            "statement": statement,
                            **observation,
                        }
                    )
    observations = pd.DataFrame(rows, columns=_observation_columns_without_ids())
    if not observations.empty:
        observations["evidence_id"] = observations.apply(
            lambda row: _stable_hash(
                {
                    "metric": row["metric"],
                    "statement": row["statement"],
                    "horizon_type": row["horizon_type"],
                    "horizon_fiscal_year": row["horizon_fiscal_year"],
                    "horizon_fiscal_quarter": row["horizon_fiscal_quarter"],
                    "numeric_low": row["numeric_low"],
                    "numeric_high": row["numeric_high"],
                    "accounting_basis": row["accounting_basis"],
                }
            ),
            axis=1,
        )
        observations = observations.drop_duplicates("evidence_id").reset_index(drop=True)
    report = {
        "guidance_sections": len(sections),
        "numeric_observations": len(observations),
        "metrics_observed": (
            sorted(observations["metric"].astype(str).unique()) if not observations.empty else []
        ),
    }
    return observations, report


def calculate_guidance_metric_coverage(
    event_links: pd.DataFrame,
    document_targets: pd.DataFrame,
    filing_targets: pd.DataFrame,
    quarterly_history: pd.DataFrame,
    raw_root: Path,
    config: GuidanceMetricCoverageConfig | None = None,
) -> GuidanceMetricCoverageTables:
    """Match prior numeric guidance to target fiscal quarters without reading returns."""
    cfg = config or GuidanceMetricCoverageConfig()
    cfg.validate()
    _require_columns(
        event_links,
        {
            "validation_event_id",
            "cik",
            "guidance_document_id",
            "prior_guidance_candidate_ready",
        },
        "guidance event links",
    )
    _require_columns(
        document_targets,
        {"guidance_document_id", "cik", "prior_accession_number"},
        "guidance document targets",
    )
    _require_columns(
        filing_targets,
        {
            "validation_event_id",
            "filing_accession",
            "fiscal_year",
            "fiscal_quarter",
            "gm_level",
        },
        "target filing features",
    )
    _require_columns(
        quarterly_history,
        {"filing_accession", "revenue", "unit"},
        "quarterly history",
    )
    document_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []
    for target in document_targets.itertuples(index=False):
        audit, observations = _parse_guidance_document(target, raw_root, cfg)
        document_rows.append(audit)
        if not observations.empty:
            observation_frames.append(observations)
    document_audit = pd.DataFrame(document_rows)
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame(columns=_observation_columns())
    )
    target_context = _target_context(filing_targets, quarterly_history)
    events = event_links.merge(
        target_context,
        on="validation_event_id",
        how="left",
        validate="one_to_one",
    )
    metric_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        document_id = str(event.guidance_document_id)
        document_observations = observations.loc[
            observations["guidance_document_id"].astype(str).eq(document_id)
        ]
        for metric in GUIDANCE_METRICS:
            metric_rows.append(_event_metric_row(event, metric, document_observations, cfg))
    event_metric_features = pd.DataFrame(metric_rows)
    coverage = _coverage_table(event_metric_features, cfg)
    sanity_checks = _sanity_check_table(event_metric_features)
    evidence_authorized = (
        coverage.loc[coverage["evidence_coverage_gate"], "metric"].astype(str).tolist()
    )
    model_authorized = coverage.loc[coverage["model_feature_gate"], "metric"].astype(str).tolist()
    recommended = (
        coverage.loc[coverage["model_feature_gate"]]
        .sort_values(
            ["model_ready_events", "model_ready_issuers", "metric"],
            ascending=[False, False, True],
            kind="mergesort",
        )["metric"]
        .iloc[0]
        if model_authorized
        else "none"
    )
    best_evidence_metric = str(
        coverage.sort_values(
            ["exact_quarter_gaap_events", "exact_quarter_gaap_issuers", "metric"],
            ascending=[False, False, True],
            kind="mergesort",
        )["metric"].iloc[0]
    )
    candidate_events = event_links["prior_guidance_candidate_ready"].astype(bool)
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "guidance_candidate_events": int(candidate_events.sum()),
        "guidance_candidate_issuers": int(event_links.loc[candidate_events, "cik"].nunique()),
        "documents_audited": len(document_targets),
        "documents_with_guidance_sections": int(
            document_audit["guidance_section_count"].gt(0).sum()
        ),
        "numeric_observations": len(observations),
        "authorized_evidence_metrics": evidence_authorized,
        "authorized_model_metrics": model_authorized,
        "best_evidence_metric": best_evidence_metric,
        "recommended_next_metric": str(recommended),
        "model_fit_performed": False,
        "sanity_checks_are_selection_inputs": False,
        "return_values_loaded": False,
        "current_filing_text_loaded": False,
        "config": asdict(cfg),
        "metric_results": coverage.to_dict(orient="records"),
        "post_selection_sanity_checks": sanity_checks.to_dict(orient="records"),
    }
    return GuidanceMetricCoverageTables(
        document_audit,
        observations,
        event_metric_features,
        coverage,
        sanity_checks,
        report,
    )


def build_guidance_metric_coverage(
    plan_root: Path,
    filing_feature_root: Path,
    *,
    raw_root: Path = Path("data/sec/guidance_evidence"),
    output_root: Path = Path("data/processed/guidance_metric_coverage"),
    config: GuidanceMetricCoverageConfig | None = None,
) -> GuidanceMetricCoverageResult:
    """Persist the multi-metric coverage audit tied to source manifests and hashes."""
    cfg = config or GuidanceMetricCoverageConfig()
    cfg.validate()
    input_hashes = {
        "guidance_plan": _sha256_file(plan_root / "output_manifest.json"),
        "filing_features": _sha256_file(filing_feature_root / "output_manifest.json"),
    }
    tables = calculate_guidance_metric_coverage(
        load_catalog_table(plan_root, "guidance_event_links.parquet"),
        load_catalog_table(plan_root, "guidance_document_targets.parquet"),
        load_catalog_table(filing_feature_root, "target_filing_features.parquet"),
        load_catalog_table(filing_feature_root, "quarterly_feature_history.parquet"),
        raw_root,
        cfg,
    )
    source_hashes = sorted(
        tables.document_audit.loc[
            tables.document_audit["document_collected"], "source_sha256"
        ].astype(str)
    )
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "source_hashes": source_hashes,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return GuidanceMetricCoverageResult(
            target.resolve(), _read_json(target / "guidance_metric_coverage_report.json")
        )
    if target.exists():
        raise GuidanceMetricCoverageError(f"Incomplete metric-coverage directory: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "guidance_metric_document_audit.parquet": tables.document_audit,
        "guidance_metric_observations.parquet": tables.observations,
        "guidance_event_metric_features.parquet": tables.event_metric_features,
        "guidance_metric_coverage.parquet": tables.coverage,
        "guidance_metric_sanity_checks.parquet": tables.sanity_checks,
    }
    artifacts = _write_tables(staging, outputs)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "guidance_metric_coverage_report.json"
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
    return GuidanceMetricCoverageResult(target.resolve(), report)


def _guidance_sections(text: str, maximum_lines: int) -> list[dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not _is_guidance_heading(line):
            continue
        selected: list[str] = []
        for candidate in lines[index + 1 : index + 1 + maximum_lines]:
            if selected and (_is_guidance_heading(candidate) or _is_stop_heading(candidate)):
                break
            selected.append(candidate)
        sections.append({"heading": line, "lines": selected})
    unique: dict[str, dict[str, Any]] = {}
    for section in sections:
        key = _stable_hash({"heading": section["heading"], "lines": section["lines"]})
        unique.setdefault(key, section)
    return list(unique.values())


def _is_guidance_heading(line: str) -> bool:
    if len(line) > 120 or len(line.split()) > 18 or line.endswith("."):
        return False
    if _STOP_HEADING.search(line):
        return False
    if re.search(
        r"\b(?:reports?|reported|results?|above|high end|preliminary)\b",
        line,
        flags=re.I,
    ):
        return False
    return bool(_GUIDANCE_WORD.search(line))


def _is_stop_heading(line: str) -> bool:
    return len(line) <= 120 and bool(_STOP_HEADING.search(line))


def _statement_with_value(lines: list[str], index: int, line: str) -> str:
    if _PERCENT.search(line) or _DOLLAR.search(line):
        return line
    if index + 1 < len(lines):
        return f"{line} {lines[index + 1]}"
    return line


def _metric_statements(metric: str, statement: str) -> list[str]:
    """Keep numeric clauses attached to their own metric mention."""
    fragments = [
        fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+", statement) if fragment.strip()
    ]
    matched = [fragment for fragment in fragments if _METRIC_PATTERNS[metric].search(fragment)]
    return matched or [statement]


def _metric_observation(
    metric: str,
    statement: str,
    local_context: str,
    section_context: str,
) -> dict[str, Any] | None:
    horizon_type, horizon_year, horizon_quarter = _resolve_horizon(
        statement,
        local_context,
        section_context,
    )
    basis = _accounting_basis(statement)
    if metric in {"operating_margin", "gross_margin"}:
        values = [float(value) / 100.0 for value in _PERCENT.findall(statement)]
        unit = "proportion"
        numeric_kind = "level"
    elif metric == "revenue":
        raw_values = _dollar_values(statement)
        scale = _revenue_scale(statement, local_context)
        values = [value * scale for value in raw_values] if scale is not None else []
        unit = "USD"
        numeric_kind = "amount"
    else:
        values = _dollar_values(statement)
        unit = "USD_per_share"
        numeric_kind = "per_share"
    if not values:
        return None
    low, high, status = _numeric_range(values, statement)
    if np.isnan(low):
        return None
    if re.search(r"\bgaap\s+and\s+non-gaap\b", statement, flags=re.I):
        low = high = values[0]
        basis = "gaap"
        status = "gaap_value_selected_from_dual_disclosure"
    elif re.search(r"\bnon-gaap\s+and\s+gaap\b", statement, flags=re.I):
        if len(values) < 2:
            return None
        low = high = values[1]
        basis = "gaap"
        status = "gaap_value_selected_from_dual_disclosure"
    return {
        "metric": metric,
        "accounting_basis": basis,
        "numeric_kind": numeric_kind,
        "numeric_unit": unit,
        "numeric_low": low,
        "numeric_high": high,
        "numeric_midpoint": (low + high) / 2.0,
        "horizon_type": horizon_type,
        "horizon_fiscal_year": horizon_year,
        "horizon_fiscal_quarter": horizon_quarter,
        "explicit_quarter_horizon_ready": horizon_type == "quarter",
        "extraction_status": status,
    }


def _accounting_basis(statement: str) -> str:
    if re.search(r"\bgaap\s+and\s+non-gaap\b", statement, flags=re.I):
        return "dual_gaap_first"
    if re.search(r"\bnon-gaap\s+and\s+gaap\b", statement, flags=re.I):
        return "dual_non_gaap_first"
    if re.search(r"\b(?:non-gaap|adjusted)\b", statement, flags=re.I):
        return "non_gaap"
    if re.search(r"\bgaap\b", statement, flags=re.I):
        return "gaap"
    return "unspecified"


def _dollar_values(statement: str) -> list[float]:
    values: list[float] = []
    for opening, raw, closing in _DOLLAR.findall(statement):
        value = float(raw.replace(",", ""))
        if opening == "(" and closing == ")":
            value *= -1.0
        values.append(value)
    return values


def _revenue_scale(statement: str, section_context: str) -> float | None:
    explicit: set[float] = set()
    if re.search(r"\bbillions?\b", statement, flags=re.I):
        explicit.add(1_000_000_000.0)
    if re.search(r"\bmillions?\b", statement, flags=re.I):
        explicit.add(1_000_000.0)
    if re.search(r"\bthousands?\b", statement, flags=re.I):
        explicit.add(1_000.0)
    if len(explicit) == 1:
        return next(iter(explicit))
    if len(explicit) > 1:
        return None
    if re.search(r"\bin\s+millions?\b", section_context, flags=re.I):
        return 1_000_000.0
    if re.search(r"\bin\s+thousands?\b", section_context, flags=re.I):
        return 1_000.0
    return None


def _numeric_range(values: list[float], statement: str) -> tuple[float, float, str]:
    if len(values) == 1:
        return values[0], values[0], "point_guidance"
    if re.search(r"\b(?:to|between|from|range)\b|[-–—]", statement, flags=re.I):
        return min(values[:2]), max(values[:2]), "range_guidance"
    return np.nan, np.nan, "ambiguous_multiple_values"


def _parse_horizon(context: str) -> tuple[str, int | None, str]:
    normalized = re.sub(r"\s+", " ", context)
    candidates: set[tuple[int, int]] = set()
    word_quarters = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    for match in re.finditer(
        r"\b(first|second|third|fourth)\s+quarter(?:\s+of)?\s+(?:fiscal\s+)?('?\d{2,4})\b",
        normalized,
        flags=re.I,
    ):
        candidates.add((_normalize_year(match.group(2)), word_quarters[match.group(1).lower()]))
    for match in re.finditer(
        r"\bQ([1-4])\s*(?:FY|fiscal\s+)?('?\d{2,4})\b",
        normalized,
        flags=re.I,
    ):
        candidates.add((_normalize_year(match.group(2)), int(match.group(1))))
    full_years = {
        _normalize_year(value)
        for value in re.findall(
            r"\b(?:full[- ]year|fiscal\s+year|FY)\s*('?\d{2,4})\b",
            normalized,
            flags=re.I,
        )
    }
    if candidates and full_years:
        return "ambiguous", None, ""
    if len(candidates) == 1:
        year, quarter = next(iter(candidates))
        return "quarter", year, f"Q{quarter}"
    if len(candidates) > 1:
        return "ambiguous", None, ""
    if len(full_years) == 1:
        return "full_year", next(iter(full_years)), ""
    return "not_explicit", None, ""


def _resolve_horizon(
    statement: str,
    local_context: str,
    section_context: str,
) -> tuple[str, int | None, str]:
    """Prefer the nearest explicit fiscal horizon and reject mixed quarter/year labels."""
    ambiguous_seen = False
    for context in (statement, local_context, section_context):
        result = _parse_horizon(context)
        if result[0] in {"quarter", "full_year"}:
            return result
        ambiguous_seen = ambiguous_seen or result[0] == "ambiguous"
    if ambiguous_seen:
        return "ambiguous", None, ""
    return "not_explicit", None, ""


def _normalize_year(value: str) -> int:
    number = int(value.replace("'", ""))
    return number + 2000 if number < 100 else number


def _parse_guidance_document(
    target: Any,
    raw_root: Path,
    config: GuidanceMetricCoverageConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    document_id = str(target.guidance_document_id)
    completion_path = raw_root / "completions" / f"{document_id}.json"
    base: dict[str, Any] = {
        "guidance_document_id": document_id,
        "cik": str(target.cik),
        "prior_accession_number": str(target.prior_accession_number),
        "document_collected": completion_path.is_file(),
        "source_sha256": pd.NA,
        "guidance_section_count": 0,
        "numeric_observation_count": 0,
        "parser_status": "not_collected",
    }
    if not completion_path.is_file():
        return base, pd.DataFrame(columns=_observation_columns())
    completion = _read_json(completion_path)
    raw_path = raw_root / str(completion["raw_path"])
    content = raw_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != completion.get("sha256"):
        raise GuidanceMetricCoverageError(f"Guidance source hash mismatch: {document_id}")
    documents = parse_complete_submission(content)
    exhibits = [
        document
        for document in documents
        if str(document["document_type"]).upper().startswith("EX-99")
    ]
    selected = exhibits or [
        document
        for document in documents
        if str(document["document_type"]).upper() in {"8-K", "8-K/A"}
    ]
    text = "\n".join(str(document["visible_text"]) for document in selected)
    observations, extract_report = extract_guidance_metric_observations(
        text,
        maximum_section_lines=config.maximum_section_lines,
    )
    if not observations.empty:
        observations.insert(0, "prior_accession_number", str(target.prior_accession_number))
        observations.insert(0, "cik", str(target.cik))
        observations.insert(0, "guidance_document_id", document_id)
        observations.insert(3, "source_sha256", digest)
    base.update(
        {
            "source_sha256": digest,
            "guidance_section_count": extract_report["guidance_sections"],
            "numeric_observation_count": len(observations),
            "parser_status": "ready" if text.strip() else "no_text_exhibit",
        }
    )
    return base, observations


def _target_context(
    filing_targets: pd.DataFrame,
    quarterly_history: pd.DataFrame,
) -> pd.DataFrame:
    quarterly = quarterly_history[["filing_accession", "revenue", "unit"]].drop_duplicates(
        "filing_accession"
    )
    context = filing_targets[
        [
            "validation_event_id",
            "filing_accession",
            "fiscal_year",
            "fiscal_quarter",
            "gm_level",
        ]
    ].merge(quarterly, on="filing_accession", how="left", validate="many_to_one")
    context["fiscal_year"] = pd.to_numeric(context["fiscal_year"], errors="coerce").astype("Int64")
    context["fiscal_quarter"] = context["fiscal_quarter"].astype(str).str.upper()
    return context


def _event_metric_row(
    event: Any,
    metric: str,
    document_observations: pd.DataFrame,
    config: GuidanceMetricCoverageConfig,
) -> dict[str, Any]:
    metric_observations = document_observations.loc[
        document_observations["metric"].astype(str).eq(metric)
    ].copy()
    candidate_ready = bool(event.prior_guidance_candidate_ready)
    any_numeric = candidate_ready and not metric_observations.empty
    gaap = metric_observations.loc[
        metric_observations["accounting_basis"].isin(["gaap", "unspecified"])
    ]
    target_year = int(event.fiscal_year) if not pd.isna(event.fiscal_year) else None
    target_quarter = str(event.fiscal_quarter)
    exact = gaap.iloc[0:0].copy()
    if target_year is not None:
        exact = gaap.loc[
            gaap["horizon_type"].eq("quarter")
            & gaap["horizon_fiscal_year"].eq(target_year)
            & gaap["horizon_fiscal_quarter"].eq(target_quarter)
        ]
    selected_low = np.nan
    selected_high = np.nan
    selection_status = "no_exact_quarter_gaap_guidance"
    exact_ready = False
    if not exact.empty:
        midpoints = exact["numeric_midpoint"].astype(float).to_numpy()
        tolerance = _consistency_tolerance(metric, config)
        scale = max(abs(float(np.mean(midpoints))), 1.0) if metric == "revenue" else 1.0
        if len(midpoints) == 1 or float(np.ptp(midpoints)) <= tolerance * scale:
            selected_low = float(exact["numeric_low"].astype(float).mean())
            selected_high = float(exact["numeric_high"].astype(float).mean())
            exact_ready = True
            selection_status = "exact_quarter_gaap_guidance_ready"
        else:
            selection_status = "ambiguous_exact_quarter_guidance"
    target_actual_ready = False
    target_actual = np.nan
    target_unit = "not_available"
    if metric == "revenue" and not pd.isna(event.revenue):
        target_actual_ready = str(event.unit).upper() == "USD"
        target_actual = float(event.revenue)
        target_unit = str(event.unit)
    elif metric == "gross_margin" and not pd.isna(event.gm_level):
        target_actual_ready = True
        target_actual = float(event.gm_level)
        target_unit = "proportion"
    model_ready = exact_ready and target_actual_ready
    return {
        "validation_event_id": str(event.validation_event_id),
        "cik": str(event.cik),
        "guidance_document_id": str(event.guidance_document_id),
        "metric": metric,
        "target_fiscal_year": target_year,
        "target_fiscal_quarter": target_quarter,
        "prior_guidance_candidate_ready": candidate_ready,
        "numeric_document_evidence_ready": any_numeric,
        "gaap_compatible_numeric_evidence_ready": not gaap.empty,
        "exact_quarter_candidate_count": len(exact),
        "exact_quarter_gaap_guidance_ready": exact_ready,
        "selected_guidance_low": selected_low,
        "selected_guidance_high": selected_high,
        "selected_guidance_midpoint": (
            (selected_low + selected_high) / 2.0 if exact_ready else np.nan
        ),
        "selection_status": selection_status,
        "target_actual": target_actual,
        "target_actual_unit": target_unit,
        "target_actual_ready": target_actual_ready,
        "model_feature_ready": model_ready,
        "return_values_loaded": False,
        "current_filing_text_loaded": False,
    }


def _consistency_tolerance(
    metric: str,
    config: GuidanceMetricCoverageConfig,
) -> float:
    if metric == "revenue":
        return config.revenue_consistency_tolerance
    if metric == "diluted_eps":
        return config.eps_consistency_tolerance
    return config.margin_consistency_tolerance


def _coverage_table(
    event_metrics: pd.DataFrame,
    config: GuidanceMetricCoverageConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in GUIDANCE_METRICS:
        frame = event_metrics.loc[
            event_metrics["metric"].eq(metric) & event_metrics["prior_guidance_candidate_ready"]
        ]
        denominator = len(frame)
        model_ready = frame["model_feature_ready"].astype(bool)
        exact_ready = frame["exact_quarter_gaap_guidance_ready"].astype(bool)
        exact_events = int(exact_ready.sum())
        exact_issuers = int(frame.loc[exact_ready, "cik"].nunique())
        exact_coverage = exact_events / denominator if denominator else 0.0
        model_events = int(model_ready.sum())
        model_issuers = int(frame.loc[model_ready, "cik"].nunique())
        coverage = model_events / denominator if denominator else 0.0
        rows.append(
            {
                "metric": metric,
                "eligible_events": denominator,
                "numeric_document_events": int(frame["numeric_document_evidence_ready"].sum()),
                "gaap_compatible_numeric_events": int(
                    frame["gaap_compatible_numeric_evidence_ready"].sum()
                ),
                "exact_quarter_gaap_events": exact_events,
                "exact_quarter_gaap_issuers": exact_issuers,
                "exact_quarter_gaap_coverage": exact_coverage,
                "target_actual_ready_events": int(frame["target_actual_ready"].sum()),
                "model_ready_events": model_events,
                "model_ready_issuers": model_issuers,
                "model_ready_coverage": coverage,
                "minimum_pilot_coverage": config.minimum_pilot_coverage,
                "minimum_pilot_issuers": config.minimum_pilot_issuers,
                "evidence_coverage_gate": bool(
                    exact_coverage >= config.minimum_pilot_coverage
                    and exact_issuers >= config.minimum_pilot_issuers
                ),
                "model_feature_gate": bool(
                    coverage >= config.minimum_pilot_coverage
                    and model_issuers >= config.minimum_pilot_issuers
                ),
            }
        )
    return pd.DataFrame(rows)


def _sanity_check_table(event_metrics: pd.DataFrame) -> pd.DataFrame:
    """Describe extracted guidance accuracy after selection; never feed model gates."""
    rows: list[dict[str, Any]] = []
    for metric in GUIDANCE_METRICS:
        frame = event_metrics.loc[
            event_metrics["metric"].eq(metric) & event_metrics["model_feature_ready"]
        ].copy()
        if frame.empty:
            rows.append(
                {
                    "metric": metric,
                    "events": 0,
                    "issuers": 0,
                    "mean_absolute_percentage_error": np.nan,
                    "median_absolute_percentage_error": np.nan,
                    "mean_signed_percentage_error": np.nan,
                    "guidance_range_hit_rate": np.nan,
                    "used_for_selection": False,
                }
            )
            continue
        denominator = frame["selected_guidance_midpoint"].astype(float).abs()
        absolute_error = (
            frame["target_actual"].astype(float) - frame["selected_guidance_midpoint"].astype(float)
        ).abs() / denominator
        signed_error = (
            frame["target_actual"].astype(float) - frame["selected_guidance_midpoint"].astype(float)
        ) / denominator
        range_hit = frame["target_actual"].between(
            frame["selected_guidance_low"],
            frame["selected_guidance_high"],
            inclusive="both",
        )
        rows.append(
            {
                "metric": metric,
                "events": len(frame),
                "issuers": int(frame["cik"].nunique()),
                "mean_absolute_percentage_error": float(absolute_error.mean()),
                "median_absolute_percentage_error": float(absolute_error.median()),
                "mean_signed_percentage_error": float(signed_error.mean()),
                "guidance_range_hit_rate": float(range_hit.mean()),
                "used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _observation_columns_without_ids() -> list[str]:
    return [
        "section_number",
        "section_heading",
        "line_number",
        "statement",
        "metric",
        "accounting_basis",
        "numeric_kind",
        "numeric_unit",
        "numeric_low",
        "numeric_high",
        "numeric_midpoint",
        "horizon_type",
        "horizon_fiscal_year",
        "horizon_fiscal_quarter",
        "explicit_quarter_horizon_ready",
        "extraction_status",
    ]


def _observation_columns() -> list[str]:
    return [
        "guidance_document_id",
        "cik",
        "prior_accession_number",
        "source_sha256",
        *_observation_columns_without_ids(),
        "evidence_id",
    ]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GuidanceMetricCoverageError(f"{label} lacks required columns: {missing}")


def _write_tables(
    staging: Path,
    outputs: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return artifacts


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
