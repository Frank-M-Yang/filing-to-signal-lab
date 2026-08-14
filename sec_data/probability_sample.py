"""Deterministic stratified probability sampling for the historical SEC frame."""

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

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

ASSET_BACKED_SIC: Final[str] = "6189"
BLANK_CHECK_SIC: Final[str] = "6770"
EQUITY_TITLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:common stock|common shares?|ordinary shares?|shares? of beneficial interest|"
    r"common units?|limited partner interests?)\b",
    flags=re.IGNORECASE,
)
NON_EQUITY_TITLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:warrants?|rights?|preferred|notes?|debentures?|bonds?|depositary|units?, each|"
    r"included as part of (?:the )?units?)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ProbabilitySampleConfig:
    """Frozen first-stage sample allocation and deterministic order."""

    sample_per_stratum: int = 10
    version: str = "representative-equity-pilot-v1"
    size_missing_label: str = "size_missing"

    def validate(self) -> None:
        """Reject allocations too small for a useful pilot."""
        if self.sample_per_stratum < 2:
            raise ValueError("Each probability-sample stratum requires at least two units")


@dataclass(frozen=True)
class ProbabilitySampleResult:
    """Immutable probability-sample plan or scope-adjudication asset."""

    output_root: Path
    report: dict[str, Any]


def build_probability_sample_plan(
    universe_root: Path,
    target_plan_root: Path,
    *,
    output_root: Path = Path("data/processed/sec_probability_sample"),
    config: ProbabilitySampleConfig | None = None,
) -> ProbabilitySampleResult:
    """Freeze a year by point-in-time asset-size stratified issuer-year sample.

    The rank uses the already frozen filing-cover sampling hash. Observed identity,
    SIC, security status, exit events, and data-acquisition success never affect
    selection. Units later found outside common-equity scope are retained with
    their original inclusion probability; replacement is forbidden.
    """
    cfg = config or ProbabilitySampleConfig()
    cfg.validate()
    universe_report = json.loads(
        (universe_root / "universe_gate_report.json").read_text(encoding="utf-8")
    )
    target_report = json.loads(
        (target_plan_root / "target_report.json").read_text(encoding="utf-8")
    )
    if target_report.get("catalog_manifest_sha256") != universe_report.get(
        "catalog_manifest_sha256"
    ):
        raise FilingCatalogError("Target plan and universe do not share a catalog manifest")
    universe_manifest_sha256 = _sha256_file(universe_root / "output_manifest.json")
    target_manifest_sha256 = _sha256_file(target_plan_root / "output_manifest.json")
    universe = load_catalog_table(universe_root, "representative_candidates.parquet")
    targets = load_catalog_table(target_plan_root, "filing_cover_targets.parquet")
    target_columns = [
        "target_id",
        "acquisition_order",
        "sampling_hash",
        "cik",
        "calendar_year",
        "filing_id",
        "accession_number",
        "form",
        "report_date",
        "accepted_at",
        "is_xbrl",
        "is_inline_xbrl",
        "primary_document",
        "primary_document_url",
        "index_url",
        "anchor_rule",
        "selection_uses_future_exit_information",
    ]
    frame = universe.merge(
        targets[target_columns],
        on=["cik", "calendar_year"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    if len(frame) != len(universe):
        raise FilingCatalogError(
            "Probability frame and filing-cover target plan are not one-to-one"
        )
    frame["sample_size_bucket"] = frame["size_anchor_stratum"].fillna(cfg.size_missing_label)
    frame["sampling_stratum"] = (
        frame["calendar_year"].astype(str) + "|" + frame["sample_size_bucket"].astype(str)
    )
    frame = frame.sort_values(["sampling_stratum", "sampling_hash", "cik"])
    frame["stratum_population"] = frame.groupby("sampling_stratum")["cik"].transform("size")
    frame["sample_rank_in_stratum"] = frame.groupby("sampling_stratum").cumcount() + 1
    frame["stratum_sample_size"] = frame["stratum_population"].clip(upper=cfg.sample_per_stratum)
    frame["selected_stage1"] = frame["sample_rank_in_stratum"] <= frame["stratum_sample_size"]
    selected = frame.loc[frame["selected_stage1"]].copy()
    selected["stage1_inclusion_probability"] = (
        selected["stratum_sample_size"] / selected["stratum_population"]
    )
    selected["stage1_sampling_weight"] = 1.0 / selected["stage1_inclusion_probability"]
    selected["selection_version"] = cfg.version
    selected["selection_uses_identity_or_industry"] = False
    selected["selection_uses_security_status"] = False
    selected["selection_uses_future_exit_information"] = False
    selected["replacement_after_scope_review_allowed"] = False
    selected = selected.sort_values(
        ["calendar_year", "sample_size_bucket", "sample_rank_in_stratum"]
    ).reset_index(drop=True)
    cover_targets = selected[target_columns].copy()
    strata = (
        selected.groupby(
            ["calendar_year", "sample_size_bucket", "sampling_stratum"], as_index=False
        )
        .agg(
            stratum_population=("stratum_population", "first"),
            selected_issuer_years=("cik", "size"),
            inclusion_probability=("stage1_inclusion_probability", "first"),
            sampling_weight=("stage1_sampling_weight", "first"),
        )
        .sort_values(["calendar_year", "sample_size_bucket"])
    )
    strata["weighted_frame_total"] = strata["selected_issuer_years"] * strata["sampling_weight"]
    weighted_frame_total = float(selected["stage1_sampling_weight"].sum())
    if not np.isclose(weighted_frame_total, len(frame)):
        raise FilingCatalogError("Stage-one weights do not reconstruct the filing frame")

    payload = json.dumps(
        {
            "catalog_manifest_sha256": universe_report["catalog_manifest_sha256"],
            "target_plan_manifest_sha256": target_manifest_sha256,
            "config": asdict(cfg),
            "size_source_version": universe_report.get("size_feature_version"),
            "selection_inputs": ["calendar_year", "point_in_time_asset_size_bucket"],
        },
        sort_keys=True,
    ).encode()
    dataset_version = "sec-probability-sample-plan-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads(
            (target / "sample_plan_report.json").read_text(encoding="utf-8")
        )
        return ProbabilitySampleResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete probability-sample directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "selected_issuer_years.parquet": selected,
        "sample_filing_cover_targets.parquet": cover_targets,
        "sampling_strata.parquet": strata,
    }
    artifacts = _write_artifacts(staging, outputs)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_manifest_sha256": universe_report["catalog_manifest_sha256"],
        "universe_version": universe_report["dataset_version"],
        "universe_manifest_sha256": universe_manifest_sha256,
        "target_plan_version": target_report["dataset_version"],
        "target_plan_manifest_sha256": target_manifest_sha256,
        "config": asdict(cfg),
        "frame_issuer_years": len(frame),
        "frame_ciks": frame["cik"].nunique(),
        "sampling_strata": len(strata),
        "selected_issuer_years": len(selected),
        "selected_ciks": selected["cik"].nunique(),
        "weighted_frame_total": weighted_frame_total,
        "selection_uses_identity_or_industry": False,
        "selection_uses_security_status": False,
        "selection_uses_future_exit_information": False,
        "replacement_after_scope_review_allowed": False,
        "probability_weights_ready": True,
        "equity_scope_ready": False,
        "artifacts": artifacts,
    }
    _finalize_output(staging, target, report, "sample_plan_report.json", artifacts)
    return ProbabilitySampleResult(target.resolve(), report)


def build_equity_scope_audit(
    sample_plan_root: Path,
    cover_root: Path,
    *,
    output_root: Path = Path("data/processed/sec_probability_scope"),
) -> ProbabilitySampleResult:
    """Adjudicate filing-time equity scope without replacing selected units."""
    plan_report = json.loads(
        (sample_plan_root / "sample_plan_report.json").read_text(encoding="utf-8")
    )
    cover_report = json.loads((cover_root / "filing_cover_report.json").read_text(encoding="utf-8"))
    if cover_report.get("catalog_manifest_sha256") != plan_report.get("catalog_manifest_sha256"):
        raise FilingCatalogError("Scope evidence and probability sample use different catalogs")
    plan_manifest_sha256 = _sha256_file(sample_plan_root / "output_manifest.json")
    cover_manifest_sha256 = _sha256_file(cover_root / "output_manifest.json")
    selected = load_catalog_table(sample_plan_root, "selected_issuer_years.parquet")
    cover_anchors = load_catalog_table(cover_root, "issuer_year_identity_industry_anchors.parquet")
    security_observations = load_catalog_table(cover_root, "filing_security_observations.parquet")
    cover_columns = [
        "target_id",
        "index_artifact_preserved",
        "cover_artifact_preserved",
        "filing_identity_ready",
        "security_identity_observed",
        "tradable_security_mapping_ready",
        "explicit_no_trading_symbol",
        "historical_industry_ready",
        "registrant_name",
        "identity_status",
        "sic",
        "sic_description",
        "industry_evidence_status",
    ]
    audit = selected.drop(
        columns=[
            column
            for column in cover_columns
            if column in selected.columns and column != "target_id"
        ]
    ).merge(
        cover_anchors[cover_columns],
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    audit = audit.merge(
        _security_title_summary(security_observations),
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    for column in (
        "index_artifact_preserved",
        "cover_artifact_preserved",
        "filing_identity_ready",
        "security_identity_observed",
        "tradable_security_mapping_ready",
        "explicit_no_trading_symbol",
        "historical_industry_ready",
    ):
        audit[column] = audit[column].fillna(False).astype(bool)
    for column in (
        "equity_like_security_observed",
        "equity_like_exchange_ready",
        "only_non_equity_titles_observed",
    ):
        audit[column] = audit[column].fillna(False).astype(bool)
    asset_backed = audit["sic"].eq(ASSET_BACKED_SIC)
    blank_check = audit["sic"].eq(BLANK_CHECK_SIC)
    audit["equity_scope_status"] = np.select(
        [
            ~audit["index_artifact_preserved"],
            ~audit["filing_identity_ready"],
            asset_backed,
            ~audit["historical_industry_ready"],
            ~audit["cover_artifact_preserved"],
            audit["explicit_no_trading_symbol"] & ~audit["equity_like_security_observed"],
            audit["equity_like_security_observed"] & blank_check,
            audit["equity_like_security_observed"],
            audit["only_non_equity_titles_observed"],
            audit["security_identity_observed"],
        ],
        [
            "awaiting_filing_index",
            "issuer_identity_unresolved",
            "structural_non_common_equity_asset_backed",
            "industry_unresolved",
            "awaiting_filing_cover",
            "explicit_no_trading_symbol_not_price_ready",
            "provisional_blank_check_equity_candidate",
            "provisional_common_equity_candidate",
            "only_non_equity_security_titles_observed",
            "security_title_missing_or_unresolved",
        ],
        default="security_identity_unresolved",
    )
    audit["scope_adjudication_complete"] = asset_backed | audit["equity_scope_status"].isin(
        [
            "explicit_no_trading_symbol_not_price_ready",
            "provisional_blank_check_equity_candidate",
            "provisional_common_equity_candidate",
            "only_non_equity_security_titles_observed",
        ]
    )
    audit["provisional_equity_eligible"] = audit["equity_scope_status"].isin(
        [
            "provisional_blank_check_equity_candidate",
            "provisional_common_equity_candidate",
        ]
    )
    audit["ordinary_operating_equity_candidate"] = audit["equity_scope_status"].eq(
        "provisional_common_equity_candidate"
    )
    audit["blank_check_special_stratum"] = audit["equity_scope_status"].eq(
        "provisional_blank_check_equity_candidate"
    )
    audit["scope_review_replacement_allowed"] = False
    audit["final_price_identity_ready"] = False
    audit["return_study_eligible"] = False
    audit["weighted_scope_indicator"] = audit["stage1_sampling_weight"] * audit[
        "scope_adjudication_complete"
    ].astype(int)
    audit["weighted_provisional_equity_indicator"] = audit["stage1_sampling_weight"] * audit[
        "provisional_equity_eligible"
    ].astype(int)
    strata = (
        audit.groupby(["calendar_year", "sample_size_bucket", "sampling_stratum"], as_index=False)
        .agg(
            stratum_population=("stratum_population", "first"),
            selected_issuer_years=("cik", "size"),
            index_ready=("index_artifact_preserved", "sum"),
            cover_ready=("cover_artifact_preserved", "sum"),
            scope_adjudicated=("scope_adjudication_complete", "sum"),
            provisional_equity_candidates=("provisional_equity_eligible", "sum"),
            structural_asset_backed=(
                "equity_scope_status",
                lambda values: int(values.eq("structural_non_common_equity_asset_backed").sum()),
            ),
            ht_scope_adjudicated=("weighted_scope_indicator", "sum"),
            ht_provisional_equity=("weighted_provisional_equity_indicator", "sum"),
        )
        .sort_values(["calendar_year", "sample_size_bucket"])
    )
    status = (
        audit.groupby("equity_scope_status", as_index=False)
        .agg(
            selected_issuer_years=("cik", "size"),
            distinct_ciks=("cik", "nunique"),
            weighted_frame_estimate=("stage1_sampling_weight", "sum"),
        )
        .sort_values("equity_scope_status")
    )
    payload = json.dumps(
        {
            "plan_manifest_sha256": plan_manifest_sha256,
            "cover_manifest_sha256": cover_manifest_sha256,
            "asset_backed_sic": ASSET_BACKED_SIC,
            "blank_check_sic": BLANK_CHECK_SIC,
            "equity_title_pattern": EQUITY_TITLE_PATTERN.pattern,
            "non_equity_title_pattern": NON_EQUITY_TITLE_PATTERN.pattern,
            "replacement": False,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "sec-equity-scope-audit-v1-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads(
            (target / "scope_audit_report.json").read_text(encoding="utf-8")
        )
        return ProbabilitySampleResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete equity-scope directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "sample_scope_audit.parquet": audit,
        "scope_by_sampling_stratum.parquet": strata,
        "scope_status_summary.parquet": status,
    }
    artifacts = _write_artifacts(staging, outputs)
    report = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_manifest_sha256": plan_report["catalog_manifest_sha256"],
        "sample_plan_version": plan_report["dataset_version"],
        "sample_plan_manifest_sha256": plan_manifest_sha256,
        "cover_version": cover_report["dataset_version"],
        "cover_manifest_sha256": cover_manifest_sha256,
        "selected_issuer_years": len(audit),
        "index_ready": int(audit["index_artifact_preserved"].sum()),
        "cover_ready": int(audit["cover_artifact_preserved"].sum()),
        "scope_adjudicated": int(audit["scope_adjudication_complete"].sum()),
        "provisional_equity_candidates": int(audit["provisional_equity_eligible"].sum()),
        "ordinary_operating_equity_candidates": int(
            audit["ordinary_operating_equity_candidate"].sum()
        ),
        "equity_like_exchange_ready": int(audit["equity_like_exchange_ready"].sum()),
        "only_non_equity_titles_observed": int(audit["only_non_equity_titles_observed"].sum()),
        "blank_check_special_stratum": int(audit["blank_check_special_stratum"].sum()),
        "structural_asset_backed": int(asset_backed.sum()),
        "weighted_frame_total": float(audit["stage1_sampling_weight"].sum()),
        "ht_scope_adjudicated": float(audit["weighted_scope_indicator"].sum()),
        "ht_provisional_equity": float(audit["weighted_provisional_equity_indicator"].sum()),
        "replacement_after_scope_review_allowed": False,
        "scope_gate": bool(audit["scope_adjudication_complete"].all()),
        "price_identity_gate": False,
        "return_study_gate": False,
        "population_estimates_final": False,
        "artifacts": artifacts,
    }
    _finalize_output(staging, target, report, "scope_audit_report.json", artifacts)
    return ProbabilitySampleResult(target.resolve(), report)


def _security_title_summary(observations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_id",
        "equity_like_security_observed",
        "equity_like_exchange_ready",
        "only_non_equity_titles_observed",
        "equity_security_titles",
        "equity_trading_symbols",
        "all_security_titles",
    ]
    if observations.empty:
        return pd.DataFrame(columns=columns)
    scoped = observations.loc[observations["security_subject_cik_match"].astype(bool)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=columns)
    scoped["normalized_security_title"] = scoped["security_title"].map(_normalized_text)
    scoped["equity_like_title"] = scoped["normalized_security_title"].map(_is_equity_like_title)
    scoped["non_equity_title"] = scoped["normalized_security_title"].map(_is_non_equity_title)
    scoped["equity_like_security"] = scoped["equity_like_title"] & scoped[
        "security_identity_observed"
    ].astype(bool)
    scoped["equity_like_exchange"] = scoped["equity_like_title"] & scoped[
        "tradable_security_mapping_ready"
    ].astype(bool)
    rows: list[dict[str, Any]] = []
    for target_id, group in scoped.groupby("target_id"):
        titles = sorted(
            {
                value
                for value in group["normalized_security_title"].astype(str)
                if value and value != "<NA>"
            }
        )
        equity_titles = sorted(
            {
                value
                for value in group.loc[
                    group["equity_like_title"], "normalized_security_title"
                ].astype(str)
                if value and value != "<NA>"
            }
        )
        equity_symbols = sorted(
            {
                str(value)
                for value in group.loc[group["equity_like_security"], "trading_symbol"].dropna()
                if str(value)
            }
        )
        has_any_title = bool(titles)
        has_equity = bool(group["equity_like_security"].any())
        rows.append(
            {
                "target_id": str(target_id),
                "equity_like_security_observed": has_equity,
                "equity_like_exchange_ready": bool(group["equity_like_exchange"].any()),
                "only_non_equity_titles_observed": bool(
                    has_any_title and not has_equity and group["non_equity_title"].all()
                ),
                "equity_security_titles": "|".join(equity_titles),
                "equity_trading_symbols": "|".join(equity_symbols),
                "all_security_titles": "|".join(titles),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _normalized_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _is_equity_like_title(value: str) -> bool:
    return bool(
        value and EQUITY_TITLE_PATTERN.search(value) and not NON_EQUITY_TITLE_PATTERN.search(value)
    )


def _is_non_equity_title(value: str) -> bool:
    return bool(value and NON_EQUITY_TITLE_PATTERN.search(value))


def _write_artifacts(staging: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
    return {
        filename: {
            "rows": len(frame),
            "bytes": (staging / filename).stat().st_size,
            "sha256": _sha256_file(staging / filename),
        }
        for filename, frame in outputs.items()
    }


def _finalize_output(
    staging: Path,
    target: Path,
    report: dict[str, Any],
    report_filename: str,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    (staging / report_filename).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": report["dataset_version"],
                "artifacts": artifacts,
                "report_sha256": _sha256_file(staging / report_filename),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
