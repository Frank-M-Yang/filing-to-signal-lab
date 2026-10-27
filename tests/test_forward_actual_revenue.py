"""Tests for strict accession-level forward revenue actuals."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from sec_data.forward_actual_revenue import (
    build_forward_actual_audit,
    build_forward_actual_plan,
    calculate_forward_actual_targets,
    collect_forward_actual_snapshots,
    select_accession_revenue_actual,
)
from sec_data.models import SecResponse


def test_actual_targets_exclude_amendments_and_preserve_clock() -> None:
    candidates = pd.concat(
        [_candidate(), _candidate(form="10-Q/A", accession="0000000001-26-000003")],
        ignore_index=True,
    )
    candidates.loc[1, "amendment_form"] = True

    targets = calculate_forward_actual_targets(candidates)

    assert len(targets) == 1
    assert targets.iloc[0]["form"] == "10-Q"
    assert targets.iloc[0]["actual_availability_clock"] == "sec_acceptance_datetime"
    assert not targets["historical_backfill_admission"].any()
    assert not targets["return_values_loaded"].any()


def test_revenue_selector_requires_exact_accession_and_direct_quarter() -> None:
    target = _target_record()
    facts = _facts_frame(duration_days=91)

    ready = select_accession_revenue_actual(target, facts)
    wrong = select_accession_revenue_actual(
        target,
        facts.assign(accession_number="0000000001-26-999999"),
    )

    assert ready["actual_status"] == "direct_quarterly_fact_ready"
    assert ready["actual_revenue"] == 110.0
    assert ready["actual_available_at"] == pd.Timestamp("2026-10-24T20:00:00Z")
    assert wrong["actual_status"] == "missing_exact_accession_revenue_fact"
    assert wrong["actual_revenue"] is None


def test_ten_k_annual_value_is_not_silently_used_as_quarter() -> None:
    target = _target_record(form="10-K")
    actual = select_accession_revenue_actual(target, _facts_frame(duration_days=365, form="10-K"))

    assert actual["actual_status"] == "annual_only_q4_derivation_required"
    assert actual["actual_revenue"] is None
    assert actual["annual_revenue_value"] == 110.0


def test_ten_k_q4_is_derived_only_from_strict_prior_original_nine_month_fact() -> None:
    target = _target_record(form="10-K")
    target.update(
        {
            "q4_prior_10q_accession": "0000000001-26-000001",
            "q4_prior_10q_accepted_at": pd.Timestamp("2026-10-01T20:00:00Z"),
            "q4_prior_10q_report_date": pd.Timestamp("2026-09-30T00:00:00Z"),
            "report_date": pd.Timestamp("2026-12-31T00:00:00Z"),
            "accepted_at": pd.Timestamp("2027-02-01T20:00:00Z"),
        }
    )
    annual = _facts_frame(duration_days=365, form="10-K").assign(
        value_numeric=500.0,
        period_end=pd.Timestamp("2026-12-31T00:00:00Z"),
        accepted_at=pd.Timestamp("2027-02-01T20:00:00Z"),
    )
    ytd = _facts_frame(duration_days=273).assign(
        fact_id="fact-ytd",
        value_numeric=380.0,
        accession_number="0000000001-26-000001",
        accepted_at=pd.Timestamp("2026-10-01T20:00:00Z"),
    )

    actual = select_accession_revenue_actual(target, pd.concat([annual, ytd]))

    assert actual["actual_status"] == "derived_q4_fy_minus_prior_9m_ready"
    assert actual["actual_revenue"] == 120.0
    assert actual["ytd_revenue_accession"] == "0000000001-26-000001"
    assert actual["actual_available_at"] == pd.Timestamp("2027-02-01T20:00:00Z")


def test_actual_round_trip_preserves_bytes_and_builds_ready_fact(tmp_path: Path) -> None:
    monitor_root = tmp_path / "monitor"
    monitor_root.mkdir()
    _candidate().to_parquet(monitor_root / "forward_filing_candidates.parquet", index=False)
    _candidate().to_parquet(monitor_root / "forward_filing_observations.parquet", index=False)
    (monitor_root / "output_manifest.json").write_text(
        '{"dataset_version":"test-monitor"}', encoding="utf-8"
    )
    plan = build_forward_actual_plan(monitor_root, output_root=tmp_path / "targets")

    class Client:
        def get_companyfacts(self, cik: str) -> SecResponse:
            assert cik == "0000000001"
            return SecResponse(
                url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                content=_companyfacts_payload(),
                fetched_at=datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
                status_code=200,
                headers={"content-type": "application/json"},
            )

    raw_root = tmp_path / "raw"
    collection = collect_forward_actual_snapshots(
        pd.read_parquet(plan.output_root / "forward_actual_revenue_targets.parquet"),
        Client(),  # type: ignore[arg-type]
        raw_root=raw_root,
        capture_date=date(2026, 10, 25),
        limit=1,
    )
    audit = build_forward_actual_audit(
        plan.output_root,
        raw_root=raw_root,
        output_root=tmp_path / "processed",
    )

    assert collection.completed_targets == 1
    assert audit.report["direct_quarterly_actuals_ready"] == 1
    assert audit.report["actual_revenue_join_gate_passed"] is True
    actuals = pd.read_parquet(audit.output_root / "forward_actual_revenue_observations.parquet")
    assert actuals.iloc[0]["actual_revenue"] == 110.0
    assert actuals.iloc[0]["accession_number"] == "0000000001-26-000002"


def _candidate(*, form: str = "10-Q", accession: str = "0000000001-26-000002") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sec_monitor_target_id": "monitor-aaa",
                "live_consensus_target_id": "consensus-aaa",
                "cik": "0000000001",
                "ticker": "AAA",
                "target_period_end": pd.Timestamp("2026-09-30", tz="UTC"),
                "consensus_captured_at": pd.Timestamp("2026-08-29T12:00:00Z"),
                "consensus_revenue_level": 100.0,
                "accession_number": accession,
                "form": form,
                "report_date": pd.Timestamp("2026-09-30", tz="UTC"),
                "accepted_at": pd.Timestamp("2026-10-24T20:00:00Z"),
                "period_match_status": "exact",
                "future_filing_candidate": True,
                "amendment_form": form.endswith("/A"),
                "observed_at": pd.Timestamp("2026-10-25T12:00:00Z"),
                "source_sha256": "submissions-hash",
                "candidate_status": "ready_for_accession_xbrl_actual_collection",
            }
        ]
    )


def _target_record(*, form: str = "10-Q") -> dict[str, object]:
    return {
        "forward_actual_target_id": "actual-aaa",
        "sec_monitor_target_id": "monitor-aaa",
        "live_consensus_target_id": "consensus-aaa",
        "cik": "0000000001",
        "ticker": "AAA",
        "target_period_end": pd.Timestamp("2026-09-30", tz="UTC"),
        "consensus_captured_at": pd.Timestamp("2026-08-29T12:00:00Z"),
        "consensus_revenue_level": 100.0,
        "accession_number": "0000000001-26-000002",
        "form": form,
        "report_date": pd.Timestamp("2026-09-30", tz="UTC"),
        "accepted_at": pd.Timestamp("2026-10-24T20:00:00Z"),
    }


def _facts_frame(*, duration_days: int, form: str = "10-Q") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fact_id": "fact-aaa",
                "taxonomy": "us-gaap",
                "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
                "unit": "USD",
                "value_numeric": 110.0,
                "period_end": pd.Timestamp("2026-09-30", tz="UTC"),
                "duration_days": duration_days,
                "accession_number": "0000000001-26-000002",
                "form": form,
                "accepted_at": pd.Timestamp("2026-10-24T20:00:00Z"),
                "source_snapshot_sha256": "companyfacts-hash",
            }
        ]
    )


def _companyfacts_payload() -> bytes:
    return json.dumps(
        {
            "cik": 1,
            "entityName": "AAA Corp",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "label": "Revenue",
                        "description": "Quarterly revenue",
                        "units": {
                            "USD": [
                                {
                                    "start": "2026-07-01",
                                    "end": "2026-09-30",
                                    "val": 110,
                                    "accn": "0000000001-26-000002",
                                    "fy": 2026,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                    "filed": "2026-10-24",
                                    "frame": "CY2026Q3",
                                },
                                {
                                    "start": "2025-07-01",
                                    "end": "2025-09-30",
                                    "val": 90,
                                    "accn": "0000000001-25-000002",
                                    "fy": 2025,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                    "filed": "2025-10-24",
                                    "frame": "CY2025Q3",
                                },
                            ]
                        },
                    }
                }
            },
        }
    ).encode()


def test_ten_k_prior_ten_q_metadata_accepts_tz_aware_filing_clocks() -> None:
    # Regression: the first live 10-K target crashed because the prior-10-Q clock
    # columns were created tz-naive while SEC acceptance clocks are UTC-aware.
    ten_k = _candidate(form="10-K", accession="0000000001-26-000020")
    ten_k.loc[0, "target_period_end"] = pd.Timestamp("2026-08-31", tz="UTC")
    ten_k.loc[0, "report_date"] = pd.Timestamp("2026-08-30", tz="UTC")
    ten_k.loc[0, "accepted_at"] = pd.Timestamp("2026-10-07T01:13:15Z")
    observations = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "0000000001-26-000010",
                "form": "10-Q",
                "report_date": pd.Timestamp("2026-05-31", tz="UTC"),
                "accepted_at": pd.Timestamp("2026-06-03T20:43:00Z"),
                "amendment_form": False,
            }
        ]
    )
    targets = calculate_forward_actual_targets(ten_k, observations)
    row = targets.iloc[0]
    assert row["q4_derivation_status"] == "prior_9m_filing_candidate_identified"
    assert row["q4_prior_10q_accession"] == "0000000001-26-000010"
    assert pd.Timestamp(row["q4_prior_10q_accepted_at"]) == pd.Timestamp("2026-06-03T20:43:00Z")
    assert pd.Timestamp(row["q4_prior_10q_report_date"]) == pd.Timestamp("2026-05-31", tz="UTC")
    assert str(targets["q4_prior_10q_accepted_at"].dtype) == "datetime64[ns, UTC]"
