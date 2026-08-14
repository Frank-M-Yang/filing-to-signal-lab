"""Point-in-time security identity and market-return readiness contract."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

PRICE_CONTRACT_VERSION: Final[str] = "point-in-time-price-contract-v1"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_MAPPING_METHODS: Final[frozenset[str]] = frozenset(
    {"current_ticker_join", "current_company_name_join", "forward_filled_ticker"}
)


@dataclass(frozen=True)
class ProviderCapability:
    """Evidence-backed capability assessment for one market-data source."""

    provider: str
    cost_class: str
    permanent_security_id: str
    point_in_time_symbol_history: str
    inactive_and_delisted_coverage: str
    unadjusted_daily_open_close: str
    explicit_corporate_actions: str
    delisting_return_or_terminal_value: str
    historical_exchange_sessions_with_timestamps: str
    exact_response_snapshot_preservable: str
    contract_ready_without_external_data: bool
    primary_documentation_url: str
    audit_note: str


@dataclass(frozen=True)
class PriceReadinessResult:
    """Immutable price-readiness audit for the frozen probability sample."""

    output_root: Path
    report: dict[str, Any]


def provider_capability_matrix() -> pd.DataFrame:
    """Return the conservative provider matrix backed by primary documentation."""
    rows = [
        ProviderCapability(
            provider="CRSP US Stock Databases",
            cost_class="institutional_subscription",
            permanent_security_id="verified_permno",
            point_in_time_symbol_history="verified_name_history",
            inactive_and_delisted_coverage="verified",
            unadjusted_daily_open_close="verified",
            explicit_corporate_actions="verified",
            delisting_return_or_terminal_value="verified_delret",
            historical_exchange_sessions_with_timestamps="external_calendar_required",
            exact_response_snapshot_preservable="delivery_method_dependent",
            contract_ready_without_external_data=False,
            primary_documentation_url=(
                "https://www.crsp.org/wp-content/uploads/guides/"
                "CRSP_US_Stock_%26_Indexes_Database_Guide_Flat_File_Format_2.0.pdf"
            ),
            audit_note=(
                "Research-quality reference candidate. Add a tested historical exchange "
                "calendar and preserve licensed extracts with hashes."
            ),
        ),
        ProviderCapability(
            provider="Massive (formerly Polygon.io)",
            cost_class="free_metadata_plus_paid_history",
            permanent_security_id="figi_fields_available_not_yet_link_audited",
            point_in_time_symbol_history="point_in_time_reference_query_available",
            inactive_and_delisted_coverage="inactive_flag_and_delisted_date_available",
            unadjusted_daily_open_close="verified_day_aggregates",
            explicit_corporate_actions="verified_dividends_and_splits",
            delisting_return_or_terminal_value="not_documented",
            historical_exchange_sessions_with_timestamps="forward_holidays_only",
            exact_response_snapshot_preservable="verified_http_or_flat_file_bytes",
            contract_ready_without_external_data=False,
            primary_documentation_url=("https://massive.com/docs/rest/stocks/tickers/all-tickers"),
            audit_note=(
                "Promising low-cost pilot source, but delisted status is not a delisting "
                "return and the documented holiday endpoint is forward-looking only."
            ),
        ),
        ProviderCapability(
            provider="Alpha Vantage",
            cost_class="free_compact_plus_paid_full_history",
            permanent_security_id="not_documented",
            point_in_time_symbol_history="not_documented",
            inactive_and_delisted_coverage="not_documented",
            unadjusted_daily_open_close="verified",
            explicit_corporate_actions="verified_in_adjusted_endpoint",
            delisting_return_or_terminal_value="not_documented",
            historical_exchange_sessions_with_timestamps="not_documented",
            exact_response_snapshot_preservable="verified_http_bytes",
            contract_ready_without_external_data=False,
            primary_documentation_url="https://www.alphavantage.co/documentation/",
            audit_note=(
                "Useful for spot checks, not accepted as the sole survivor-bias-free "
                "research source under the current documentation."
            ),
        ),
    ]
    return pd.DataFrame(asdict(row) for row in rows)


def point_in_time_price_contract() -> dict[str, Any]:
    """Return the machine-readable schemas and non-negotiable research rules."""
    return {
        "contract_version": PRICE_CONTRACT_VERSION,
        "atomic_unit": "company_x_filing_event_x_information_set_x_security",
        "security_mapping_required_columns": [
            "target_id",
            "cik",
            "security_id",
            "ticker",
            "exchange_mic",
            "valid_from_inclusive",
            "valid_to_exclusive",
            "mapping_method",
            "source_snapshot_sha256",
        ],
        "daily_price_required_columns": [
            "security_id",
            "session_date",
            "open_unadjusted",
            "close_unadjusted",
            "currency",
            "source_snapshot_sha256",
        ],
        "corporate_action_required_columns": [
            "security_id",
            "effective_at",
            "action_type",
            "cash_amount",
            "split_from",
            "split_to",
            "source_snapshot_sha256",
        ],
        "delisting_required_columns": [
            "security_id",
            "delisting_date",
            "delisting_return",
            "terminal_value",
            "missing_return_reason",
            "source_snapshot_sha256",
        ],
        "exchange_session_required_columns": [
            "exchange_mic",
            "session_date",
            "regular_open_at",
            "regular_close_at",
            "source_snapshot_sha256",
        ],
        "rules": {
            "event_security_join": (
                "accepted_at must fall inside the security mapping validity interval"
            ),
            "unbiased_execution": (
                "regular-session open on the second trading session after accepted_at"
            ),
            "reaction_window": "explicit timestamp and price type required",
            "ticker_policy": "never join an historical event to a current ticker",
            "delisting_policy": (
                "a delisting return, terminal value, or explicit missing reason is mandatory"
            ),
            "raw_policy": "preserve exact source bytes and SHA-256 before parsing",
            "coverage_policy": (
                "unmapped selected units remain missing; no replacement and no zero imputation"
            ),
        },
        "forbidden_mapping_methods": sorted(_FORBIDDEN_MAPPING_METHODS),
    }


def validate_security_mapping_frame(mapping: pd.DataFrame) -> None:
    """Reject mappings that violate the historical-identity contract."""
    required = set(point_in_time_price_contract()["security_mapping_required_columns"])
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise FilingCatalogError(f"Security mapping is missing columns: {missing}")
    if mapping.empty:
        raise FilingCatalogError("Security mapping cannot be empty")
    methods = mapping["mapping_method"].astype(str).str.strip()
    forbidden = sorted(set(methods).intersection(_FORBIDDEN_MAPPING_METHODS))
    if forbidden:
        raise FilingCatalogError(f"Forbidden historical mapping methods: {forbidden}")
    valid_from = pd.to_datetime(mapping["valid_from_inclusive"], utc=True, errors="coerce")
    valid_to = pd.to_datetime(mapping["valid_to_exclusive"], utc=True, errors="coerce")
    if valid_from.isna().any() or valid_to.isna().any() or (valid_from >= valid_to).any():
        raise FilingCatalogError("Security mapping contains invalid validity intervals")
    hashes = mapping["source_snapshot_sha256"].astype(str)
    if not hashes.map(lambda value: bool(_SHA256.fullmatch(value))).all():
        raise FilingCatalogError("Security mapping source hashes must be SHA-256 values")


def build_price_readiness_audit(
    scope_root: Path,
    *,
    reference_root: Path | None = None,
    price_root: Path | None = None,
    calendar_root: Path | None = None,
    security_actions_root: Path | None = None,
    delisting_root: Path | None = None,
    output_root: Path = Path("data/processed/sec_price_readiness"),
) -> PriceReadinessResult:
    """Freeze current sample readiness without inventing security or price mappings."""
    scope_report = json.loads((scope_root / "scope_audit_report.json").read_text("utf-8"))
    scope_manifest_sha256 = _sha256_file(scope_root / "output_manifest.json")
    sample = load_catalog_table(scope_root, "sample_scope_audit.parquet").copy()
    sample["price_identity_status"] = "awaiting_scope_adjudication"
    excluded = sample["scope_adjudication_complete"].astype(bool) & ~sample[
        "provisional_equity_eligible"
    ].astype(bool)
    candidates = sample["provisional_equity_eligible"].astype(bool)
    sample.loc[excluded, "price_identity_status"] = "excluded_non_common_equity_scope"
    sample.loc[candidates, "price_identity_status"] = "awaiting_stable_security_mapping"
    sample["stable_security_id"] = pd.NA
    sample["provider_security_id_candidate"] = pd.NA
    sample["event_date_reference_candidate"] = False
    sample["event_date_reference_status"] = "not_collected"
    sample["historical_symbol_interval_ready"] = False
    sample["historical_symbol_interval_status"] = "not_collected"
    sample["provider_price_query_complete"] = False
    sample["daily_price_status"] = "outside_massive_price_plan"
    sample["observed_event_window_candidate"] = False
    sample["candidate_entry_session_date"] = pd.NA
    sample["candidate_entry_open_unadjusted"] = pd.NA
    sample["candidate_exit_session_date"] = pd.NA
    sample["candidate_exit_close_unadjusted"] = pd.NA
    sample["acceptance_session_relation"] = pd.NA
    sample["formal_entry_session_date"] = pd.NA
    sample["formal_entry_open_at"] = pd.NA
    sample["formal_entry_open_unadjusted"] = pd.NA
    sample["formal_exit_session_date"] = pd.NA
    sample["formal_exit_close_at"] = pd.NA
    sample["formal_exit_close_unadjusted"] = pd.NA
    sample["formal_event_window_ready"] = False
    sample["event_calendar_alignment_status"] = "outside_exchange_calendar_plan"
    sample["daily_price_ready"] = False
    sample["corporate_actions_ready"] = False
    sample["corporate_action_status"] = "not_collected"
    sample["split_count"] = 0
    sample["dividend_count"] = 0
    sample["delisting_outcome_ready"] = False
    sample["listing_outcome_status"] = "not_collected"
    sample["active_at_outcome_cutoff"] = False
    sample["later_delisted_after_formal_exit"] = False
    sample["delisted_utc"] = pd.NA
    sample["exchange_session_calendar_ready"] = False
    sample["return_study_eligible"] = False
    reference_report: dict[str, Any] | None = None
    reference_manifest_sha256: str | None = None
    if reference_root is not None:
        reference_report = json.loads(
            (reference_root / "massive_reference_report.json").read_text("utf-8")
        )
        if reference_report.get("scope_manifest_sha256") != scope_manifest_sha256:
            raise FilingCatalogError("Reference audit does not tie to the equity-scope manifest")
        reference_manifest_sha256 = _sha256_file(reference_root / "output_manifest.json")
        reference = load_catalog_table(reference_root, "sample_massive_reference_audit.parquet")
        reference_columns = [
            "target_id",
            "provider_query_complete",
            "event_date_reference_candidate",
            "event_date_reference_status",
            "selected_provider_security_id",
            "selected_ticker",
            "selected_exchange_mic",
        ]
        sample = sample.drop(
            columns=[
                "event_date_reference_candidate",
                "event_date_reference_status",
            ]
        ).merge(
            reference[reference_columns],
            on="target_id",
            how="left",
            validate="one_to_one",
        )
        sample["provider_query_complete"] = sample["provider_query_complete"].fillna(False)
        sample["event_date_reference_candidate"] = sample["event_date_reference_candidate"].fillna(
            False
        )
        sample["event_date_reference_status"] = sample["event_date_reference_status"].fillna(
            "outside_provisional_equity_reference_plan"
        )
        sample["provider_security_id_candidate"] = sample["selected_provider_security_id"]
        sample.loc[
            candidates & sample["event_date_reference_candidate"].astype(bool),
            "price_identity_status",
        ] = "event_date_reference_observed_interval_not_validated"
    price_report: dict[str, Any] | None = None
    price_manifest_sha256: str | None = None
    if price_root is not None:
        if reference_report is None or reference_manifest_sha256 is None:
            raise FilingCatalogError("A Massive price audit requires its reference audit")
        price_report = json.loads(
            (price_root / "massive_price_report.json").read_text(encoding="utf-8")
        )
        if price_report.get("reference_manifest_sha256") != reference_manifest_sha256:
            raise FilingCatalogError("Price audit does not tie to the selected reference manifest")
        price_manifest_sha256 = _sha256_file(price_root / "output_manifest.json")
        price_audit = load_catalog_table(price_root, "sample_massive_price_audit.parquet")
        price_columns = [
            "target_id",
            "provider_price_query_complete",
            "daily_price_status",
            "observed_event_window_candidate",
            "candidate_entry_session_date",
            "candidate_entry_open_unadjusted",
            "candidate_exit_session_date",
            "candidate_exit_close_unadjusted",
        ]
        sample = sample.drop(columns=price_columns[1:]).merge(
            price_audit[price_columns],
            on="target_id",
            how="left",
            validate="one_to_one",
        )
        sample["provider_price_query_complete"] = sample["provider_price_query_complete"].fillna(
            False
        )
        sample["observed_event_window_candidate"] = sample[
            "observed_event_window_candidate"
        ].fillna(False)
        sample["daily_price_status"] = sample["daily_price_status"].fillna(
            "outside_massive_price_plan"
        )
    calendar_report: dict[str, Any] | None = None
    calendar_manifest_sha256: str | None = None
    if calendar_root is not None:
        if price_report is None or price_manifest_sha256 is None:
            raise FilingCatalogError("An exchange calendar audit requires its price audit")
        calendar_report = json.loads(
            (calendar_root / "exchange_calendar_report.json").read_text(encoding="utf-8")
        )
        if calendar_report.get("price_manifest_sha256") != price_manifest_sha256:
            raise FilingCatalogError("Calendar audit does not tie to the selected price manifest")
        calendar_manifest_sha256 = _sha256_file(calendar_root / "output_manifest.json")
        alignment = load_catalog_table(calendar_root, "event_calendar_alignment.parquet")
        calendar_columns = [
            "target_id",
            "acceptance_session_relation",
            "formal_entry_session_date",
            "formal_entry_open_at",
            "formal_entry_open_unadjusted",
            "formal_exit_session_date",
            "formal_exit_close_at",
            "formal_exit_close_unadjusted",
            "formal_event_window_ready",
            "alignment_status",
        ]
        sample = sample.drop(
            columns=[
                "acceptance_session_relation",
                "formal_entry_session_date",
                "formal_entry_open_at",
                "formal_entry_open_unadjusted",
                "formal_exit_session_date",
                "formal_exit_close_at",
                "formal_exit_close_unadjusted",
                "formal_event_window_ready",
                "event_calendar_alignment_status",
                "exchange_session_calendar_ready",
            ]
        ).merge(
            alignment[calendar_columns].rename(
                columns={"alignment_status": "event_calendar_alignment_status"}
            ),
            on="target_id",
            how="left",
            validate="one_to_one",
        )
        sample["formal_event_window_ready"] = sample["formal_event_window_ready"].fillna(False)
        sample["event_calendar_alignment_status"] = sample[
            "event_calendar_alignment_status"
        ].fillna("outside_exchange_calendar_plan")
        sample["exchange_session_calendar_ready"] = sample["formal_event_window_ready"].astype(bool)
        sample["daily_price_ready"] = sample["provider_price_query_complete"].astype(bool) & sample[
            "formal_event_window_ready"
        ].astype(bool)
    security_actions_report: dict[str, Any] | None = None
    security_actions_manifest_sha256: str | None = None
    if security_actions_root is not None:
        if (
            reference_manifest_sha256 is None
            or price_manifest_sha256 is None
            or calendar_manifest_sha256 is None
        ):
            raise FilingCatalogError(
                "A security-actions audit requires reference, price, and calendar audits"
            )
        security_actions_report = json.loads(
            (security_actions_root / "massive_security_actions_report.json").read_text(
                encoding="utf-8"
            )
        )
        expected_manifests = {
            "reference_manifest_sha256": reference_manifest_sha256,
            "price_manifest_sha256": price_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
        }
        if any(
            security_actions_report.get(key) != expected
            for key, expected in expected_manifests.items()
        ):
            raise FilingCatalogError(
                "Security-actions audit does not tie to the selected upstream manifests"
            )
        security_actions_manifest_sha256 = _sha256_file(
            security_actions_root / "output_manifest.json"
        )
        actions_audit = load_catalog_table(
            security_actions_root, "sample_security_actions_audit.parquet"
        )
        action_columns = [
            "target_id",
            "stable_security_id",
            "historical_symbol_interval_status",
            "historical_symbol_interval_ready",
            "corporate_action_status",
            "corporate_actions_ready",
            "split_count",
            "dividend_count",
        ]
        sample = sample.drop(columns=action_columns[1:]).merge(
            actions_audit[action_columns],
            on="target_id",
            how="left",
            validate="one_to_one",
        )
        sample["historical_symbol_interval_ready"] = sample[
            "historical_symbol_interval_ready"
        ].fillna(False)
        sample["historical_symbol_interval_status"] = sample[
            "historical_symbol_interval_status"
        ].fillna("outside_security_actions_plan")
        sample["corporate_actions_ready"] = sample["corporate_actions_ready"].fillna(False)
        sample["corporate_action_status"] = sample["corporate_action_status"].fillna(
            "outside_security_actions_plan"
        )
        sample["split_count"] = sample["split_count"].fillna(0).astype(int)
        sample["dividend_count"] = sample["dividend_count"].fillna(0).astype(int)
        sample.loc[
            sample["historical_symbol_interval_ready"].astype(bool),
            "price_identity_status",
        ] = "historical_security_interval_validated"
    delisting_report: dict[str, Any] | None = None
    delisting_manifest_sha256: str | None = None
    if delisting_root is not None:
        if security_actions_manifest_sha256 is None or calendar_manifest_sha256 is None:
            raise FilingCatalogError(
                "A delisting audit requires security-actions and calendar audits"
            )
        delisting_report = json.loads(
            (delisting_root / "massive_delisting_report.json").read_text(encoding="utf-8")
        )
        if (
            delisting_report.get("security_actions_manifest_sha256")
            != security_actions_manifest_sha256
            or delisting_report.get("calendar_manifest_sha256") != calendar_manifest_sha256
        ):
            raise FilingCatalogError(
                "Delisting audit does not tie to the selected upstream manifests"
            )
        delisting_manifest_sha256 = _sha256_file(delisting_root / "output_manifest.json")
        delisting_audit = load_catalog_table(
            delisting_root, "sample_delisting_outcome_audit.parquet"
        )
        delisting_columns = [
            "target_id",
            "listing_outcome_status",
            "delisted_utc",
            "active_at_outcome_cutoff",
            "later_delisted_after_formal_exit",
            "delisting_outcome_ready",
        ]
        sample = sample.drop(columns=delisting_columns[1:]).merge(
            delisting_audit[delisting_columns],
            on="target_id",
            how="left",
            validate="one_to_one",
        )
        sample["listing_outcome_status"] = sample["listing_outcome_status"].fillna(
            "outside_delisting_outcome_plan"
        )
        sample["active_at_outcome_cutoff"] = sample["active_at_outcome_cutoff"].fillna(False)
        sample["later_delisted_after_formal_exit"] = sample[
            "later_delisted_after_formal_exit"
        ].fillna(False)
        sample["delisting_outcome_ready"] = sample["delisting_outcome_ready"].fillna(False)
    sample["return_study_eligible"] = (
        sample["historical_symbol_interval_ready"].astype(bool)
        & sample["daily_price_ready"].astype(bool)
        & sample["corporate_actions_ready"].astype(bool)
        & sample["delisting_outcome_ready"].astype(bool)
    )
    capabilities = provider_capability_matrix()
    contract = point_in_time_price_contract()
    payload = json.dumps(
        {
            "scope_manifest_sha256": scope_manifest_sha256,
            "contract": contract,
            "provider_matrix": capabilities.to_dict(orient="records"),
            "reference_manifest_sha256": reference_manifest_sha256,
            "price_manifest_sha256": price_manifest_sha256,
            "calendar_manifest_sha256": calendar_manifest_sha256,
            "security_actions_manifest_sha256": security_actions_manifest_sha256,
            "delisting_manifest_sha256": delisting_manifest_sha256,
        },
        sort_keys=True,
    ).encode()
    dataset_version = "sec-price-readiness-v6-" + hashlib.sha256(payload).hexdigest()[:16]
    target = output_root / dataset_version
    if (target / "output_manifest.json").exists():
        existing_report = json.loads((target / "price_readiness_report.json").read_text("utf-8"))
        return PriceReadinessResult(target.resolve(), existing_report)
    if target.exists():
        raise FilingCatalogError(f"Incomplete price-readiness directory exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    outputs = {
        "sample_price_readiness.parquet": sample,
        "price_source_capability_matrix.parquet": capabilities,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        frame.to_parquet(staging / filename, index=False)
        artifacts[filename] = _artifact_record(staging / filename, len(frame))
    contract_path = staging / "point_in_time_price_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    artifacts[contract_path.name] = _artifact_record(contract_path, 1)
    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_version": scope_report["dataset_version"],
        "scope_manifest_sha256": scope_manifest_sha256,
        "reference_version": reference_report["dataset_version"] if reference_report else None,
        "reference_manifest_sha256": reference_manifest_sha256,
        "price_version": price_report["dataset_version"] if price_report else None,
        "price_manifest_sha256": price_manifest_sha256,
        "calendar_version": calendar_report["dataset_version"] if calendar_report else None,
        "calendar_manifest_sha256": calendar_manifest_sha256,
        "security_actions_version": (
            security_actions_report["dataset_version"] if security_actions_report else None
        ),
        "security_actions_manifest_sha256": security_actions_manifest_sha256,
        "delisting_version": delisting_report["dataset_version"] if delisting_report else None,
        "delisting_manifest_sha256": delisting_manifest_sha256,
        "contract_version": PRICE_CONTRACT_VERSION,
        "selected_issuer_years": len(sample),
        "scope_adjudicated": int(sample["scope_adjudication_complete"].sum()),
        "scope_unresolved": int((~sample["scope_adjudication_complete"]).sum()),
        "price_mapping_candidates": int(candidates.sum()),
        "scope_excluded": int(excluded.sum()),
        "provider_queries_complete": int(
            sample.get("provider_query_complete", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "event_date_reference_candidates": int(
            sample["event_date_reference_candidate"].astype(bool).sum()
        ),
        "provider_price_queries_complete": int(
            sample["provider_price_query_complete"].astype(bool).sum()
        ),
        "observed_event_window_candidates": int(
            sample["observed_event_window_candidate"].astype(bool).sum()
        ),
        "formal_event_window_ready": int(sample["formal_event_window_ready"].astype(bool).sum()),
        "exchange_session_calendars_ready": int(
            sample["exchange_session_calendar_ready"].astype(bool).sum()
        ),
        "stable_security_mappings_ready": int(
            sample["historical_symbol_interval_ready"].astype(bool).sum()
        ),
        "corporate_action_windows_ready": int(sample["corporate_actions_ready"].astype(bool).sum()),
        "daily_prices_ready": int(sample["daily_price_ready"].astype(bool).sum()),
        "delisting_outcomes_ready": int(sample["delisting_outcome_ready"].astype(bool).sum()),
        "later_delisted_after_formal_exit": int(
            sample["later_delisted_after_formal_exit"].astype(bool).sum()
        ),
        "return_study_eligible": int(sample["return_study_eligible"].astype(bool).sum()),
        "scope_gate": bool(sample["scope_adjudication_complete"].all()),
        "price_identity_gate": False,
        "daily_price_gate": False,
        "delisting_outcome_gate": False,
        "exchange_calendar_gate": False,
        "pilot_exchange_calendar_gate": bool(
            calendar_report and calendar_report.get("pilot_exchange_calendar_gate")
        ),
        "pilot_price_identity_gate": bool(
            security_actions_report and security_actions_report.get("pilot_identity_gate")
        ),
        "pilot_corporate_action_gate": bool(
            security_actions_report and security_actions_report.get("pilot_corporate_actions_gate")
        ),
        "pilot_delisting_outcome_gate": bool(
            delisting_report and delisting_report.get("pilot_delisting_outcome_gate")
        ),
        "pilot_event_return_gate": bool(
            delisting_report
            and delisting_report.get("pilot_event_return_gate")
            and int(sample["return_study_eligible"].astype(bool).sum())
            == int(delisting_report.get("targets", -1))
        ),
        "historical_constituent_gate": False,
        "return_study_gate": False,
        "current_ticker_join_used": False,
        "unresolved_units_replaced": False,
        "free_or_low_cost_source_meets_full_contract": False,
        "artifacts": artifacts,
    }
    (staging / "price_readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "scope_manifest_sha256": scope_manifest_sha256,
                "reference_manifest_sha256": reference_manifest_sha256,
                "price_manifest_sha256": price_manifest_sha256,
                "calendar_manifest_sha256": calendar_manifest_sha256,
                "security_actions_manifest_sha256": security_actions_manifest_sha256,
                "delisting_manifest_sha256": delisting_manifest_sha256,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(staging / "price_readiness_report.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return PriceReadinessResult(target.resolve(), report)


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
