"""Resolve multiclass validation issuers using filing-time disclosed share counts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table
from sec_data.filing_cover import (
    load_filing_artifact_snapshots,
    parse_share_class_outstanding_facts,
)

VALIDATION_SHARE_CLASS_RESOLUTION_VERSION: Final[str] = (
    "validation-identity-resolution-v3-largest-disclosed-share-class"
)


class ValidationShareClassResolutionError(FilingCatalogError):
    """Raised when filing-time share-class resolution is not uniquely supported."""


class ValidationShareClassResolutionResult:
    """Immutable share-class identity resolution."""

    def __init__(self, output_root: Path, report: dict[str, Any]) -> None:
        self.output_root = output_root
        self.report = report


def calculate_share_class_resolution(
    identity_events: pd.DataFrame,
    identity_components: pd.DataFrame,
    security_evidence: pd.DataFrame,
    primary_contents: dict[str, tuple[bytes, str]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Choose one stable primary class using only SEC shares outstanding facts."""
    for frame, label in (
        (identity_events, "identity events"),
        (identity_components, "identity components"),
        (security_evidence, "security evidence"),
    ):
        if _has_outcome_columns(frame):
            raise ValidationShareClassResolutionError(f"{label} contains outcome columns")
    _require_columns(
        identity_events,
        {
            "market_event_id",
            "cik",
            "event_security_identity_ready",
            "security_component_count",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "identity events",
    )
    _require_columns(
        identity_components,
        {
            "cik",
            "anchor_role",
            "ticker",
            "exchange_mic",
            "security_title",
            "provider_security_id",
            "provider_security_id_kind",
            "identity_key",
            "identity_key_kind",
            "component_identity_ready",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "identity components",
    )
    _require_columns(
        security_evidence,
        {
            "target_id",
            "cik",
            "anchor_role",
            "trading_symbol",
            "security_title",
            "primary_source_sha256",
            "selection_uses_return_values",
            "outcome_values_loaded",
        },
        "security evidence",
    )
    if identity_events["selection_uses_return_values"].astype(bool).any():
        raise ValidationShareClassResolutionError("Identity events used return values")
    if identity_events["outcome_values_loaded"].astype(bool).any():
        raise ValidationShareClassResolutionError("Identity events opened outcomes")
    multiclass_events = identity_events.loc[
        identity_events["security_component_count"].gt(1)
    ].copy()
    if multiclass_events.empty:
        raise ValidationShareClassResolutionError("No multiclass issuer requires resolution")
    multiclass_ciks = set(multiclass_events["cik"].astype(str).str.zfill(10))
    components = identity_components.copy()
    components["cik"] = components["cik"].astype(str).str.zfill(10)
    components = components.loc[components["cik"].isin(multiclass_ciks)].copy()
    evidence = security_evidence.copy()
    evidence["cik"] = evidence["cik"].astype(str).str.zfill(10)
    evidence = evidence.loc[evidence["cik"].isin(multiclass_ciks)].copy()
    if components["selection_uses_return_values"].astype(bool).any():
        raise ValidationShareClassResolutionError("Identity components used return values")
    if components["outcome_values_loaded"].astype(bool).any():
        raise ValidationShareClassResolutionError("Identity components opened outcomes")
    if evidence["selection_uses_return_values"].astype(bool).any():
        raise ValidationShareClassResolutionError("SEC evidence selection used returns")
    if evidence["outcome_values_loaded"].astype(bool).any():
        raise ValidationShareClassResolutionError("SEC evidence opened outcomes")
    fact_rows: list[dict[str, Any]] = []
    for target in (
        evidence[["target_id", "cik", "anchor_role", "primary_source_sha256"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        target_id = str(target.target_id)
        try:
            content, source_sha256 = primary_contents[target_id]
        except KeyError as exc:
            raise ValidationShareClassResolutionError(
                f"Missing preserved primary document for endpoint {target_id}"
            ) from exc
        if source_sha256 != str(target.primary_source_sha256):
            raise ValidationShareClassResolutionError("Primary-document hash does not match audit")
        parsed = parse_share_class_outstanding_facts(content, subject_cik=str(target.cik))
        for fact in parsed:
            class_label = _member_class_label(str(fact["class_member"]))
            if class_label is None:
                continue
            fact_rows.append(
                {
                    "target_id": target_id,
                    "cik": str(target.cik).zfill(10),
                    "anchor_role": str(target.anchor_role),
                    "class_label": class_label,
                    **fact,
                    "primary_source_sha256": source_sha256,
                    "selection_uses_price_coverage": False,
                    "selection_uses_return_values": False,
                    "outcome_values_loaded": False,
                }
            )
    facts = pd.DataFrame(fact_rows)
    if facts.empty:
        raise ValidationShareClassResolutionError("No class-specific shares facts were parsed")
    fact_counts = facts.groupby(["target_id", "class_label"])["shares_outstanding"].nunique()
    if fact_counts.gt(1).any():
        raise ValidationShareClassResolutionError(
            "One endpoint has conflicting shares outstanding for a stock class"
        )
    facts = facts.sort_values(
        ["target_id", "class_label", "instant", "context_ref"], kind="mergesort"
    ).drop_duplicates(["target_id", "class_label"], keep="last")
    candidates = evidence[
        [
            "target_id",
            "cik",
            "anchor_role",
            "trading_symbol",
            "security_title",
            "primary_source_sha256",
        ]
    ].copy()
    candidates["class_label"] = candidates["security_title"].map(_title_class_label)
    if candidates["class_label"].isna().any():
        raise ValidationShareClassResolutionError("A security title lacks an explicit class label")
    candidates = candidates.merge(
        facts[
            [
                "target_id",
                "class_label",
                "context_ref",
                "class_member",
                "instant",
                "shares_outstanding",
            ]
        ],
        on=["target_id", "class_label"],
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.merge(
        components[
            [
                "cik",
                "anchor_role",
                "ticker",
                "exchange_mic",
                "provider_security_id",
                "provider_security_id_kind",
                "identity_key",
                "identity_key_kind",
                "component_identity_ready",
            ]
        ],
        left_on=["cik", "anchor_role", "trading_symbol"],
        right_on=["cik", "anchor_role", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if candidates[["shares_outstanding", "identity_key", "exchange_mic"]].isna().any().any():
        raise ValidationShareClassResolutionError(
            "A stock class lacks shares or stable identity evidence"
        )
    if not candidates["component_identity_ready"].astype(bool).all():
        raise ValidationShareClassResolutionError("A stock class lacks stable identity evidence")
    candidates["shares_rank_desc"] = candidates.groupby("target_id")["shares_outstanding"].rank(
        method="min", ascending=False
    )
    candidates["largest_disclosed_share_class"] = candidates["shares_rank_desc"].eq(1)
    winners = candidates.loc[candidates["largest_disclosed_share_class"]].copy()
    if not winners.groupby("target_id").size().eq(1).all():
        raise ValidationShareClassResolutionError("Largest disclosed share class is not unique")
    decisions: list[dict[str, Any]] = []
    for cik, group in winners.groupby("cik", sort=True):
        if group["target_id"].nunique() != 2:
            raise ValidationShareClassResolutionError(
                "Multiclass issuer does not have two endpoint decisions"
            )
        stable_columns = [
            "trading_symbol",
            "exchange_mic",
            "identity_key",
            "identity_key_kind",
        ]
        if any(group[column].astype(str).nunique() != 1 for column in stable_columns):
            raise ValidationShareClassResolutionError(
                "Largest disclosed share class is not stable across endpoints"
            )
        first = group.sort_values("anchor_role", kind="mergesort").iloc[0]
        decisions.append(
            {
                "cik": str(cik),
                "selected_ticker": str(first["trading_symbol"]),
                "selected_exchange_mic": str(first["exchange_mic"]),
                "selected_provider_security_id": _optional_text(first["provider_security_id"]),
                "selected_provider_security_id_kind": _optional_text(
                    first["provider_security_id_kind"]
                ),
                "selected_identity_key": str(first["identity_key"]),
                "selected_identity_key_kind": str(first["identity_key_kind"]),
                "endpoint_decisions": len(group),
                "selection_rule": "largest_class_specific_shares_outstanding_at_both_endpoints",
                "selection_uses_price_coverage": False,
                "selection_uses_return_values": False,
                "outcome_values_loaded": False,
                "decision_ready": True,
            }
        )
    decision_frame = pd.DataFrame(decisions)
    resolved = identity_events.copy()
    resolved["cik"] = resolved["cik"].astype(str).str.zfill(10)
    decision_lookup = decision_frame.set_index("cik")
    for index in resolved.index[resolved["cik"].isin(decision_lookup.index)]:
        decision = decision_lookup.loc[str(resolved.at[index, "cik"])]
        resolved.at[index, "identity_assignment_method"] = (
            "primary_share_class_by_largest_disclosed_shares"
        )
        resolved.at[index, "assigned_ticker"] = str(decision["selected_ticker"])
        resolved.at[index, "assigned_exchange_mic"] = str(decision["selected_exchange_mic"])
        resolved.at[index, "assigned_provider_security_id"] = _optional_text(
            decision["selected_provider_security_id"]
        )
        resolved.at[index, "assigned_provider_security_id_kind"] = _optional_text(
            decision["selected_provider_security_id_kind"]
        )
        resolved.at[index, "security_component_count"] = 1
        resolved.at[index, "assigned_security_identity_keys"] = str(
            decision["selected_identity_key"]
        )
        resolved.at[index, "issuer_return_construction"] = (
            "single_primary_share_class_by_pre_event_disclosed_shares"
        )
        resolved.at[index, "identity_repair_status"] = (
            "primary_share_class_stable_across_both_endpoints"
        )
    resolved["price_collection_authorized"] = False
    resolved["return_outcome_read_authorized"] = False
    resolved["selection_uses_return_values"] = False
    resolved["outcome_values_loaded"] = False
    gates = pd.DataFrame(
        [
            _gate(
                "class_specific_shares_evidence",
                int(candidates["shares_outstanding"].notna().sum()),
                len(candidates),
                bool(candidates["shares_outstanding"].notna().all()),
                "Every endpoint security class has a subject-attributable SEC shares fact",
            ),
            _gate(
                "unique_largest_class_at_endpoints",
                int(winners["target_id"].nunique()),
                int(candidates["target_id"].nunique()),
                winners["target_id"].nunique() == candidates["target_id"].nunique(),
                "Every endpoint has exactly one largest disclosed stock class",
            ),
            _gate(
                "stable_primary_class_across_endpoints",
                int(decision_frame["decision_ready"].sum()),
                int(multiclass_events["cik"].nunique()),
                bool(decision_frame["decision_ready"].all()),
                "The same largest class and identity persist at both issuer endpoints",
            ),
            _gate(
                "return_outcome_read",
                0,
                len(resolved),
                False,
                "Share-class resolution does not read prices or returns",
            ),
        ]
    )
    report: dict[str, Any] = {
        "price_candidate_events": len(resolved),
        "price_candidate_events_identity_ready": int(
            resolved["event_security_identity_ready"].astype(bool).sum()
        ),
        "multiclass_issuers": int(multiclass_events["cik"].nunique()),
        "multiclass_events_before_resolution": len(multiclass_events),
        "multiclass_events_after_resolution": int(resolved["security_component_count"].gt(1).sum()),
        "share_class_endpoint_candidates": len(candidates),
        "share_class_endpoint_winners": len(winners),
        "issuers_with_stable_primary_class": int(decision_frame["decision_ready"].sum()),
        "selection_rule": "largest_class_specific_shares_outstanding_at_both_endpoints",
        "prior_structural_price_failure_record_preserved": True,
        "prior_price_pilot_used_as_selection_input": False,
        "selection_uses_price_coverage": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
        "price_collection_authorized": False,
        "return_outcome_read_authorized": False,
    }
    return resolved, facts, candidates, decision_frame, gates, report


def build_validation_share_class_resolution(
    identity_resolution_root: Path,
    cover_audit_root: Path,
    *,
    raw_filing_root: Path = Path("data/sec/validation_filing_evidence"),
    output_root: Path = Path("data/processed/validation_identity_repair_resolution"),
) -> ValidationShareClassResolutionResult:
    """Persist the v3 identity asset without using the structural price pilot."""
    identity_manifest = _sha256_file(identity_resolution_root / "output_manifest.json")
    cover_manifest = _sha256_file(cover_audit_root / "output_manifest.json")
    identity_report = _read_json(identity_resolution_root / "identity_repair_report.json")
    events = load_catalog_table(
        identity_resolution_root, "validation_event_identity_readiness_final.parquet"
    )
    components = load_catalog_table(
        identity_resolution_root, "validation_identity_repair_components.parquet"
    )
    security_evidence = load_catalog_table(
        cover_audit_root, "validation_identity_repair_security_evidence.parquet"
    )
    multiclass_ciks = set(
        events.loc[events["security_component_count"].gt(1), "cik"].astype(str).str.zfill(10)
    )
    target_ids = set(
        security_evidence.loc[
            security_evidence["cik"].astype(str).str.zfill(10).isin(multiclass_ciks),
            "target_id",
        ].astype(str)
    )
    primary_contents: dict[str, tuple[bytes, str]] = {}
    for snapshot in load_filing_artifact_snapshots(raw_filing_root):
        if snapshot.target_id in target_ids and snapshot.artifact_kind == "primary":
            primary_contents[snapshot.target_id] = (
                Path(snapshot.path).read_bytes(),
                snapshot.sha256,
            )
    resolved, facts, candidates, decisions, gates, metrics = calculate_share_class_resolution(
        events,
        components,
        security_evidence,
        primary_contents,
    )
    source_hashes = sorted(value[1] for value in primary_contents.values())
    payload = json.dumps(
        {
            "base_identity_manifest_sha256": identity_manifest,
            "cover_audit_manifest_sha256": cover_manifest,
            "primary_source_hashes": source_hashes,
            "version": VALIDATION_SHARE_CLASS_RESOLUTION_VERSION,
            "selection_rule": metrics["selection_rule"],
        },
        sort_keys=True,
    ).encode()
    dataset_version = (
        VALIDATION_SHARE_CLASS_RESOLUTION_VERSION + "-" + hashlib.sha256(payload).hexdigest()[:16]
    )
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        return ValidationShareClassResolutionResult(
            target.resolve(), _read_json(target / "identity_repair_report.json")
        )
    if target.exists():
        raise ValidationShareClassResolutionError(f"Incomplete v3 resolution exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "validation_event_identity_readiness_final.parquet": resolved,
        "validation_identity_repair_components.parquet": components,
        "validation_share_class_outstanding_evidence.parquet": facts,
        "validation_share_class_candidates.parquet": candidates,
        "validation_share_class_decisions.parquet": decisions,
        "validation_share_class_gates.parquet": gates,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = _artifact_record(path, len(frame))
    input_manifests = dict(identity_report.get("input_manifest_sha256", {}))
    input_manifests["base_identity_resolution"] = identity_manifest
    input_manifests["identity_cover_audit"] = cover_manifest
    report: dict[str, Any] = {
        **identity_report,
        **metrics,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_identity_resolution_version": identity_report["dataset_version"],
        "base_identity_resolution_manifest_sha256": identity_manifest,
        "identity_cover_audit_manifest_sha256": cover_manifest,
        "input_manifest_sha256": input_manifests,
        "multiclass_return_rule": (
            "single_primary_share_class_by_largest_pre_event_disclosed_shares"
        ),
        "primary_source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "identity_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "base_identity_resolution_manifest_sha256": identity_manifest,
                "identity_cover_audit_manifest_sha256": cover_manifest,
                "primary_source_hashes": source_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ValidationShareClassResolutionResult(target.resolve(), report)


def _title_class_label(value: Any) -> str | None:
    match = re.search(r"\bclass\s+([a-z0-9]+)\b", str(value), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _member_class_label(value: str) -> str | None:
    local = value.rsplit(":", 1)[-1]
    match = re.search(r"class([a-z0-9]+)member$", local, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _has_outcome_columns(frame: pd.DataFrame) -> bool:
    return any(
        column in frame.columns
        for column in (
            "issuer_total_return",
            "market_total_return",
            "market_excess_return_arithmetic",
            "sector_excess_return",
            "future_return",
            "delisting_return",
        )
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


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationShareClassResolutionError(f"{label} lacks columns: {missing}")


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
