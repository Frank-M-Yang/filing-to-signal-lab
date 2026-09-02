"""Historical-SIC fixed-effect diagnostics for filing-event representations.

This module answers a deliberately narrower question than a sector benchmark:
does an exploratory Level/Delta/Innovation rank association remain after removing
coarse, filing-time SIC division means?  It never labels the residual as a
tradable sector excess return.  A point-in-time sector ETF or portfolio return is
still required before the formal sector-benchmark gate can pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sec_data.event_return_diagnostics import REPRESENTATION_FEATURES
from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

SECTOR_NEUTRAL_DIAGNOSTIC_VERSION: Final[str] = "sector-neutral-diagnostics-v1-filing-time-sic"
SIC_DIVISIONS: Final[tuple[tuple[int, int, str, str], ...]] = (
    (100, 999, "A", "Agriculture, forestry, and fishing"),
    (1000, 1499, "B", "Mining"),
    (1500, 1799, "C", "Construction"),
    (2000, 3999, "D", "Manufacturing"),
    (4000, 4999, "E", "Transportation, communications, and utilities"),
    (5000, 5199, "F", "Wholesale trade"),
    (5200, 5999, "G", "Retail trade"),
    (6000, 6799, "H", "Finance, insurance, and real estate"),
    (7000, 8999, "I", "Services"),
    (9100, 9729, "J", "Public administration"),
    (9900, 9999, "K", "Nonclassifiable establishments"),
)


@dataclass(frozen=True)
class SectorNeutralDiagnosticConfig:
    """Frozen gates and uncertainty conventions for the SIC diagnostic."""

    minimum_industry_match_ratio: float = 0.80
    minimum_events_for_ic: int = 20
    minimum_events_per_sic_division: int = 2
    uncertainty_resamples: int = 5000
    uncertainty_seed: int = 20250828
    version: str = SECTOR_NEUTRAL_DIAGNOSTIC_VERSION

    def validate(self) -> None:
        """Reject weak or internally inconsistent diagnostic settings."""
        if not 0.0 < self.minimum_industry_match_ratio <= 1.0:
            raise ValueError("Industry-match ratio must be in (0, 1]")
        if self.minimum_events_for_ic < 10:
            raise ValueError("Sector-adjusted IC requires at least ten events")
        if self.minimum_events_per_sic_division < 2:
            raise ValueError("A SIC fixed effect requires at least two events per division")
        if self.uncertainty_resamples < 1000:
            raise ValueError("Sector-adjusted uncertainty requires at least 1,000 resamples")


@dataclass(frozen=True)
class SectorNeutralDiagnosticTables:
    """Pure historical-industry diagnostic tables before persistence."""

    sample_audit: pd.DataFrame
    sector_composition: pd.DataFrame
    ic_comparison: pd.DataFrame
    ic_uncertainty: pd.DataFrame
    gates: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class SectorNeutralDiagnosticResult:
    """Immutable sector-neutral diagnostic bundle."""

    output_root: Path
    report: dict[str, Any]


def calculate_sector_neutral_diagnostics(
    representation_return_sample: pd.DataFrame,
    filing_industry_observations: pd.DataFrame,
    config: SectorNeutralDiagnosticConfig | None = None,
) -> SectorNeutralDiagnosticTables:
    """Calculate same-sample partial rank ICs using filing-time SIC fixed effects.

    Feature and return ranks are first calculated globally and then residualized
    against SEC SIC division indicators.  Singleton divisions are retained in
    the audit but cannot identify a within-division association and are excluded
    from the fixed-effect sample.
    """
    cfg = config or SectorNeutralDiagnosticConfig()
    cfg.validate()
    sample = _prepare_sample(representation_return_sample)
    audit = _attach_filing_time_industry(sample, filing_industry_observations)
    counts = (
        audit.loc[audit["historical_industry_exact_match"]]
        .groupby(["sic_division", "sic_division_name"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    count_map = counts.set_index("sic_division")["events"] if not counts.empty else pd.Series()
    audit["sic_division_events"] = audit["sic_division"].map(count_map).fillna(0).astype(int)
    audit["sector_fixed_effect_eligible"] = audit["historical_industry_exact_match"] & audit[
        "sic_division_events"
    ].ge(cfg.minimum_events_per_sic_division)
    fixed_effect_sample = audit.loc[audit["sector_fixed_effect_eligible"]].copy()
    exact_matches = int(audit["historical_industry_exact_match"].sum())
    match_ratio = exact_matches / len(audit) if len(audit) else 0.0
    industry_gate = match_ratio >= cfg.minimum_industry_match_ratio
    event_gate = len(fixed_effect_sample) >= cfg.minimum_events_for_ic
    division_gate = fixed_effect_sample["sic_division"].nunique() >= 2
    diagnostic_allowed = industry_gate and event_gate and division_gate
    comparison, uncertainty = _calculate_ic_tables(
        fixed_effect_sample,
        cfg,
        allowed=diagnostic_allowed,
    )
    composition = counts.copy()
    if not composition.empty:
        composition["fixed_effect_eligible"] = composition["events"].ge(
            cfg.minimum_events_per_sic_division
        )
        composition["sample_share"] = composition["events"] / len(audit)
        composition = composition.sort_values(
            ["events", "sic_division"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    gates = pd.DataFrame(
        [
            _gate(
                "filing_time_industry_match",
                exact_matches,
                len(audit),
                industry_gate,
                "Same accession, CIK, accepted_at, subject CIK, and filing-time SIC",
            ),
            _gate(
                "sector_fixed_effect_event_count",
                len(fixed_effect_sample),
                cfg.minimum_events_for_ic,
                event_gate,
                "Events in SIC divisions with at least two complete observations",
            ),
            _gate(
                "sector_fixed_effect_divisions",
                int(fixed_effect_sample["sic_division"].nunique()),
                2,
                division_gate,
                "At least two identifiable SIC divisions",
            ),
            _gate(
                "tradable_sector_benchmark",
                0,
                len(audit),
                False,
                "Point-in-time sector ETF or portfolio returns are not loaded",
            ),
        ]
    )
    level = comparison.loc[comparison["representation"].eq("gm_level")]
    level_sign_preserved = bool(
        diagnostic_allowed
        and len(level) == 1
        and np.sign(float(level.iloc[0]["unadjusted_same_sample_ic"]))
        == np.sign(float(level.iloc[0]["sic_fixed_effect_ic"]))
    )
    report: dict[str, Any] = {
        "contract_version": cfg.version,
        "complete_representation_events": len(audit),
        "historical_industry_exact_matches": exact_matches,
        "historical_industry_match_ratio": match_ratio,
        "sic_divisions_observed": int(audit["sic_division"].nunique(dropna=True)),
        "sic_divisions_fixed_effect_eligible": int(fixed_effect_sample["sic_division"].nunique()),
        "sector_fixed_effect_events": len(fixed_effect_sample),
        "singleton_or_unmapped_events_excluded": len(audit) - len(fixed_effect_sample),
        "sector_fixed_effect_diagnostic_allowed": diagnostic_allowed,
        "historical_sector_mapping_gate": industry_gate,
        "tradable_sector_benchmark_gate": False,
        "level_sign_preserved_after_sic_fixed_effect": level_sign_preserved,
        "sector_adjustment_kind": "partial_spearman_with_filing_time_sic_division_fixed_effects",
        "uncertainty_kind": (
            "division_stratified_residual_pair_bootstrap_and_within_division_permutation"
        ),
        "effectiveness_conclusion_allowed": False,
        "interpretation": (
            "Coarse historical-SIC composition diagnostic only; not a tradable sector "
            "excess return and not evidence of signal effectiveness."
        ),
        "config": asdict(cfg),
    }
    return SectorNeutralDiagnosticTables(
        sample_audit=audit,
        sector_composition=composition,
        ic_comparison=comparison,
        ic_uncertainty=uncertainty,
        gates=gates,
        report=report,
    )


def build_sector_neutral_diagnostics(
    event_diagnostic_root: Path,
    filing_industry_root: Path,
    *,
    validation_plan_root: Path | None = None,
    output_root: Path = Path("data/processed/sector_neutral_diagnostics"),
    config: SectorNeutralDiagnosticConfig | None = None,
) -> SectorNeutralDiagnosticResult:
    """Persist an immutable SIC diagnostic tied to its evidence manifests."""
    cfg = config or SectorNeutralDiagnosticConfig()
    cfg.validate()
    event_report = _read_json(event_diagnostic_root / "event_return_diagnostic_report.json")
    if not event_report.get("exploratory_ic_allowed"):
        raise FilingCatalogError("Sector diagnostic requires an unlocked same-sample IC bundle")
    event_manifest = _sha256_file(event_diagnostic_root / "output_manifest.json")
    industry_manifest = _sha256_file(filing_industry_root / "output_manifest.json")
    sample = load_catalog_table(event_diagnostic_root, "representation_return_sample.parquet")
    industries = load_catalog_table(filing_industry_root, "filing_industry_observations.parquet")
    tables = calculate_sector_neutral_diagnostics(sample, industries, cfg)
    validation_contract = _validation_contract(validation_plan_root)
    validation_manifest = (
        _sha256_file(validation_plan_root / "output_manifest.json")
        if validation_plan_root is not None
        else None
    )
    payload = json.dumps(
        {
            "event_manifest_sha256": event_manifest,
            "industry_manifest_sha256": industry_manifest,
            "validation_manifest_sha256": validation_manifest,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = cfg.version + "-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return SectorNeutralDiagnosticResult(
            target.resolve(), _read_json(target / "sector_neutral_diagnostic_report.json")
        )
    if target.exists():
        raise FilingCatalogError(f"Incomplete sector diagnostic directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    frames = {
        "sector_sample_audit.parquet": tables.sample_audit,
        "sector_composition.parquet": tables.sector_composition,
        "sector_ic_comparison.parquet": tables.ic_comparison,
        "sector_ic_uncertainty.parquet": tables.ic_uncertainty,
        "sector_diagnostic_gates.parquet": tables.gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in frames.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    contract_path = staging / "sector_neutral_diagnostic_contract.json"
    contract_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    if validation_contract is not None:
        validation_path = staging / "independent_time_validation_contract.json"
        validation_path.write_text(
            json.dumps(validation_contract, indent=2, sort_keys=True), encoding="utf-8"
        )
        artifacts[validation_path.name] = _artifact_record(validation_path, 1)
    report = {
        **tables.report,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_diagnostic_version": event_report["dataset_version"],
        "event_manifest_sha256": event_manifest,
        "industry_manifest_sha256": industry_manifest,
        "validation_manifest_sha256": validation_manifest,
        "independent_time_validation": validation_contract,
        "artifacts": artifacts,
    }
    report_path = staging / "sector_neutral_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "event_manifest_sha256": event_manifest,
                "industry_manifest_sha256": industry_manifest,
                "validation_manifest_sha256": validation_manifest,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return SectorNeutralDiagnosticResult(target.resolve(), report)


def _prepare_sample(sample: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_id",
        "cik",
        "accession_number",
        "accepted_at",
        "market_excess_return_arithmetic",
        *REPRESENTATION_FEATURES,
    }
    _require_columns(sample, required, "representation-return sample")
    if sample.empty or sample["target_id"].isna().any() or sample["target_id"].duplicated().any():
        raise FilingCatalogError("Representation-return target IDs must be non-null and unique")
    frame = sample.copy()
    frame["cik"] = frame["cik"].astype(str).str.zfill(10)
    frame["accession_number"] = frame["accession_number"].astype(str)
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True, errors="coerce")
    numeric = [*REPRESENTATION_FEATURES, "market_excess_return_arithmetic"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["accepted_at"].isna().any() or not np.isfinite(frame[numeric]).all().all():
        raise FilingCatalogError("Representation-return sample has invalid clocks or values")
    return frame


def _attach_filing_time_industry(sample: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cik",
        "accession_number",
        "accepted_at",
        "sic",
        "sic_description",
        "index_subject_cik_matched",
        "historical_industry_ready",
        "source_sha256",
        "source_path",
    }
    _require_columns(industries, required, "filing industry observations")
    evidence = industries.copy()
    evidence["cik"] = evidence["cik"].astype(str).str.zfill(10)
    evidence["accession_number"] = evidence["accession_number"].astype(str)
    keys = sample[["cik", "accession_number"]].drop_duplicates()
    evidence = evidence.merge(keys, on=["cik", "accession_number"], how="inner")
    if evidence.duplicated(["cik", "accession_number"]).any():
        raise FilingCatalogError("Filing-time SIC evidence is not unique by CIK and accession")
    evidence["industry_accepted_at"] = pd.to_datetime(
        evidence["accepted_at"], utc=True, errors="coerce"
    )
    evidence = evidence.drop(columns=["accepted_at"])
    frame = sample.merge(
        evidence,
        on=["cik", "accession_number"],
        how="left",
        validate="one_to_one",
    )
    same_clock = frame["accepted_at"].eq(frame["industry_accepted_at"])
    subject_match = frame["index_subject_cik_matched"].fillna(False).astype(bool)
    industry_ready = frame["historical_industry_ready"].fillna(False).astype(bool)
    frame["sic"] = frame["sic"].astype("string")
    mapped = frame["sic"].map(_sic_division)
    frame["sic_division"] = mapped.map(lambda value: value[0] if value else None)
    frame["sic_division_name"] = mapped.map(lambda value: value[1] if value else None)
    frame["historical_industry_exact_match"] = (
        same_clock
        & subject_match
        & industry_ready
        & frame["sic_division"].notna()
        & frame["source_sha256"].notna()
    )
    frame["industry_join_status"] = np.select(
        [
            frame["industry_accepted_at"].isna(),
            ~same_clock,
            ~subject_match,
            ~industry_ready,
            frame["sic_division"].isna(),
        ],
        [
            "no_accession_match",
            "accepted_at_mismatch",
            "subject_cik_mismatch",
            "historical_industry_not_ready",
            "sic_division_unmapped",
        ],
        default="exact_filing_time_sic_match",
    )
    return frame


def _calculate_ic_tables(
    sample: pd.DataFrame,
    config: SectorNeutralDiagnosticConfig,
    *,
    allowed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    returns = sample["market_excess_return_arithmetic"].to_numpy(dtype=float)
    sectors = cast(NDArray[np.str_], sample["sic_division"].astype(str).to_numpy())
    for feature_index, feature in enumerate(REPRESENTATION_FEATURES):
        values = sample[feature].to_numpy(dtype=float)
        unadjusted = _spearman(values, returns) if allowed else math.nan
        adjusted = _partial_rank_ic(values, returns, sectors) if allowed else math.nan
        comparisons.append(
            {
                "representation": feature,
                "events": len(sample),
                "sic_divisions": int(pd.Series(sectors).nunique()),
                "unadjusted_same_sample_ic": unadjusted,
                "sic_fixed_effect_ic": adjusted,
                "absolute_ic_attenuation": abs(unadjusted) - abs(adjusted),
                "sign_preserved": bool(np.sign(unadjusted) == np.sign(adjusted))
                if allowed
                else False,
                "status": "exploratory_only" if allowed else "not_computed_insufficient_gate",
            }
        )
        uncertainty_rows.append(
            _uncertainty_row(
                feature,
                values,
                returns,
                sectors,
                adjusted,
                config,
                feature_index=feature_index,
                allowed=allowed,
            )
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    if allowed:
        uncertainty["permutation_p_holm"] = _holm_adjust(
            uncertainty["permutation_p_two_sided"].to_numpy(dtype=float)
        )
    else:
        uncertainty["permutation_p_holm"] = math.nan
    return pd.DataFrame(comparisons), uncertainty


def _uncertainty_row(
    feature: str,
    values: NDArray[np.float64],
    returns: NDArray[np.float64],
    sectors: NDArray[np.str_],
    observed: float,
    config: SectorNeutralDiagnosticConfig,
    *,
    feature_index: int,
    allowed: bool,
) -> dict[str, Any]:
    bootstrap_low = bootstrap_high = permutation_p = leave_low = leave_high = math.nan
    valid_bootstrap = 0
    if allowed:
        feature_residual, return_residual = _rank_residuals(values, returns, sectors)
        sector_indices = [
            np.flatnonzero(sectors == sector) for sector in sorted(set(sectors.tolist()))
        ]
        rng = np.random.default_rng(config.uncertainty_seed + feature_index)
        bootstrap_values: list[float] = []
        for _ in range(config.uncertainty_resamples):
            draw = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in sector_indices]
            )
            correlation = _pearson(feature_residual[draw], return_residual[draw])
            if np.isfinite(correlation):
                bootstrap_values.append(correlation)
        valid_bootstrap = len(bootstrap_values)
        if bootstrap_values:
            bootstrap_low, bootstrap_high = np.quantile(
                np.asarray(bootstrap_values), [0.025, 0.975]
            ).tolist()
        more_extreme = 0
        valid_permutations = 0
        for _ in range(config.uncertainty_resamples):
            permuted_return_residual = return_residual.copy()
            for indices in sector_indices:
                permuted_return_residual[indices] = rng.permutation(
                    permuted_return_residual[indices]
                )
            correlation = _pearson(feature_residual, permuted_return_residual)
            if np.isfinite(correlation):
                valid_permutations += 1
                more_extreme += int(abs(correlation) >= abs(observed))
        if valid_permutations:
            permutation_p = (1 + more_extreme) / (1 + valid_permutations)
        leave_sector_out = []
        for sector in sorted(set(sectors.tolist())):
            keep = sectors != sector
            if keep.sum() < config.minimum_events_for_ic:
                continue
            correlation = _partial_rank_ic(values[keep], returns[keep], sectors[keep])
            if np.isfinite(correlation):
                leave_sector_out.append(correlation)
        if leave_sector_out:
            leave_low = float(min(leave_sector_out))
            leave_high = float(max(leave_sector_out))
    return {
        "representation": feature,
        "events": len(values),
        "sic_divisions": int(pd.Series(sectors).nunique()),
        "sic_fixed_effect_ic": observed,
        "bootstrap_95_low": bootstrap_low,
        "bootstrap_95_high": bootstrap_high,
        "permutation_p_two_sided": permutation_p,
        "leave_one_division_out_min": leave_low,
        "leave_one_division_out_max": leave_high,
        "resamples_requested": config.uncertainty_resamples,
        "valid_bootstrap_draws": valid_bootstrap,
        "status": "exploratory_only" if allowed else "not_computed_insufficient_gate",
    }


def _rank_residuals(
    values: NDArray[np.float64],
    returns: NDArray[np.float64],
    sectors: NDArray[np.str_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    feature_rank = cast(
        NDArray[np.float64], pd.Series(values).rank(method="average").to_numpy(dtype=float)
    )
    return_rank = cast(
        NDArray[np.float64], pd.Series(returns).rank(method="average").to_numpy(dtype=float)
    )
    sector_codes, _ = pd.factorize(sectors, sort=True)
    return _demean(feature_rank, sector_codes), _demean(return_rank, sector_codes)


def _partial_rank_ic(
    values: NDArray[np.float64],
    returns: NDArray[np.float64],
    sectors: NDArray[np.str_],
) -> float:
    feature_residual, return_residual = _rank_residuals(values, returns, sectors)
    return _pearson(feature_residual, return_residual)


def _demean(values: NDArray[np.float64], group_codes: NDArray[np.intp]) -> NDArray[np.float64]:
    groups = int(group_codes.max()) + 1
    counts = np.bincount(group_codes, minlength=groups)
    sums = np.bincount(group_codes, weights=values, minlength=groups)
    means = sums / counts
    return values - means[group_codes]


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return _pearson(
        cast(
            NDArray[np.float64],
            pd.Series(left).rank(method="average").to_numpy(dtype=float),
        ),
        cast(
            NDArray[np.float64],
            pd.Series(right).rank(method="average").to_numpy(dtype=float),
        ),
    )


def _pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if len(left) < 2 or np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _holm_adjust(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(p_values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _sic_division(value: object) -> tuple[str, str] | None:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    sic = int(text)
    for lower, upper, code, name in SIC_DIVISIONS:
        if lower <= sic <= upper:
            return code, name
    return None


def _validation_contract(validation_plan_root: Path | None) -> dict[str, Any] | None:
    if validation_plan_root is None:
        return None
    report = _read_json(validation_plan_root / "validation_expansion_report.json")
    if report.get("selection_uses_return_values") or report.get("outcome_values_loaded"):
        raise FilingCatalogError("Independent-time validation plan is not outcome blind")
    return {
        "validation_dataset_version": report["dataset_version"],
        "sealed_holdout_start": report["config"]["sealed_holdout_start"],
        "candidate_events": report["candidate_events"],
        "unobserved_validation_candidate_events": report["unobserved_validation_candidate_events"],
        "unobserved_validation_candidate_issuers": report[
            "unobserved_validation_candidate_issuers"
        ],
        "inside_documented_basic_price_window": report["unobserved_inside_documented_basic_window"],
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "model_evaluation_authorized": False,
        "analysis_unlock_rule": (
            "Freeze features and market-data completeness before reading any validation return; "
            "evaluate the unchanged Level/Delta/Innovation definitions on one complete-case sample."
        ),
    }


def _gate(
    gate: str,
    numerator: int,
    denominator: int,
    passed: bool,
    meaning: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "numerator": numerator,
        "denominator": denominator,
        "coverage": numerator / denominator if denominator else 0.0,
        "passed": passed,
        "meaning": meaning,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FilingCatalogError(f"{label} lacks required columns: {missing}")


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
