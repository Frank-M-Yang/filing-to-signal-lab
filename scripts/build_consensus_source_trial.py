"""Freeze an outcome-free, two-provider sample request, not a trading cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "consensus-source-trial-v1"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verified(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if _digest(content) != expected:
        raise ValueError("Source hash mismatch")
    return content


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp has no timezone")
    return result.astimezone(timezone.utc)


def select_trial_periods(
    document: dict[str, Any], *, cik: str, cutoff_at: str
) -> list[dict[str, str]]:
    """Select up to two distinct original 10-Q periods from pinned metadata only.

    Amendments are not replacement originals. Acceptance is reference metadata,
    not asserted to be the first earnings announcement. Actual values are unused.
    """
    if str(document.get("cik", "")).zfill(10) != cik:
        raise ValueError("SEC metadata issuer mismatch")
    cutoff = _utc(cutoff_at)
    recent = document["filings"]["recent"]
    fields = ("form", "reportDate", "acceptanceDateTime", "accessionNumber")
    if len({len(recent[field]) for field in fields}) != 1:
        raise ValueError("SEC submissions arrays differ in length")
    unique: dict[str, dict[str, str]] = {}
    for i, form in enumerate(recent["form"]):
        if form != "10-Q":
            continue
        accepted = _utc(recent["acceptanceDateTime"][i])
        if accepted >= cutoff:
            continue
        period = date.fromisoformat(recent["reportDate"][i])
        if period > accepted.date():
            raise ValueError("Report period follows its acceptance date")
        key = period.isoformat()
        item = {
            "target_period_end": key,
            "reference_accession": recent["accessionNumber"][i],
            "reference_sec_accepted_at": accepted.isoformat(),
        }
        previous = unique.get(key)
        if previous is None or (item["reference_sec_accepted_at"], item["reference_accession"]) < (
            previous["reference_sec_accepted_at"],
            previous["reference_accession"],
        ):
            unique[key] = item
    return [unique[key] for key in sorted(unique, reverse=True)[:2]]


def build_trial_request(config_path: Path) -> dict[str, Any]:
    """Build the same company/period/perspective requests for both named products."""
    config_content = config_path.read_bytes()
    config = json.loads(config_content)
    issuers = config["issuers"]
    if (
        config["version"] != VERSION
        or config["quarters_per_issuer"] != 2
        or config["perspective_days_before_period_end"] != [30, 7]
        or len(issuers) != 10
        or len({i["cik"] for i in issuers}) != 10
        or len({i["ticker"] for i in issuers}) != 10
        or config["providers"] != ["factset_estimates_pit_consensus", "lseg_ibes_point_in_time"]
    ):
        raise ValueError("Unexpected v1 trial definition")
    plan_root = Path(config["source_plan_root"])
    parent = json.loads(
        _verified(plan_root / "output_manifest.json", config["source_plan_manifest_sha256"])
    )
    target_name = "alpha_vantage_live_targets.parquet"
    _verified(plan_root / target_name, parent["artifacts"][target_name]["sha256"])
    # Only identity/order columns are loaded; no forecasts, actuals or returns.
    original = pd.read_parquet(
        plan_root / target_name, columns=["ticker", "cik", "collection_order"]
    )
    original = original.sort_values("collection_order").head(6)
    if [(i["ticker"], i["cik"]) for i in issuers[:6]] != list(
        zip(original.ticker, original.cik, strict=True)
    ):
        raise ValueError("Baseline six differ from the frozen original plan order")
    if [i["ticker"] for i in issuers[6:]] != ["PSX", "MPC", "SNEX", "DELL"]:
        raise ValueError("Stress cases cannot be replaced after a coverage failure")
    public_sources = []
    for path in config["public_document_manifests"]:
        manifest = json.loads(Path(path).read_bytes())
        content = _verified(Path(manifest["path"]), manifest["sha256"])
        if len(content) != manifest["bytes"]:
            raise ValueError("Public source length mismatch")
        public_sources.append({k: manifest[k] for k in ("url", "sha256", "fetched_at")})
    if len(public_sources) != 4:
        raise ValueError("Missing reviewed public documents")
    requests = []
    coverage = []
    for issuer in issuers:
        cik, digest = issuer["cik"], issuer["submissions_sha256"]
        if re.fullmatch(r"\d{10}", cik) is None or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("Invalid pinned source identity")
        path = Path("data/sec/raw") / cik / digest[:2] / f"submissions-{digest}.json"
        document = json.loads(_verified(path, digest))
        if issuer["ticker"] not in document.get("tickers", []):
            raise ValueError("Ticker hint differs from pinned issuer metadata")
        periods = select_trial_periods(document, cik=cik, cutoff_at=config["selection_cutoff_at"])
        coverage.append(
            {
                **issuer,
                "company_name": document["name"],
                "periods": periods,
                "missing_period_slots": 2 - len(periods),
            }
        )
        for period in periods:
            end = date.fromisoformat(period["target_period_end"])
            for days in config["perspective_days_before_period_end"]:
                perspective = datetime.combine(
                    end - timedelta(days=days), time(23, 59, 59), tzinfo=timezone.utc
                )
                identity = {
                    "cik": cik,
                    "target_period_end": end.isoformat(),
                    "requested_as_of_at": perspective.isoformat(),
                }
                request_id = _digest(json.dumps(identity, sort_keys=True).encode())[:20]
                requests.append(
                    {
                        "request_id": request_id,
                        **identity,
                        **period,
                        "ticker_hint": issuer["ticker"],
                        "metric": "quarterly_revenue",
                        "days_before_period_end": days,
                        "perspective_rule": "latest_native_PIT_snapshot_not_after_requested_time",
                        "reference_is_first_earnings_announcement": False,
                        "provider_identifier_mapping_verified": False,
                    }
                )
    return {
        "trial_version": VERSION,
        "config_sha256": _digest(config_content),
        "selection_rule": config["selection_rule"],
        "selection_cutoff_at": config["selection_cutoff_at"],
        "source_plan_manifest_sha256": config["source_plan_manifest_sha256"],
        "public_sources": public_sources,
        "issuer_count": len(issuers),
        "requested_period_slots": 20,
        "resolved_period_slots": sum(len(r["periods"]) for r in coverage),
        "expected_observations_per_provider": 40,
        "resolved_requests_per_provider": len(requests),
        "issuers": coverage,
        "requests": requests,
        "provider_status": [
            {
                "provider": name,
                "access": "not_verified",
                "requests_sent": 0,
                "financial_sample_rows_received": 0,
                "acceptance": "not_tested",
            }
            for name in config["providers"]
        ],
        "required_delivery_evidence": [
            "stable issuer/provider identifiers and exact quarterly fiscal-period mapping",
            "original provider field names plus field dictionary and extraction query",
            "revenue mean/count and available range, currency, scale and accounting basis",
            "native perspective timestamp, timezone, granularity and PIT-version methodology",
            "availability/correction/activation semantics and snapshot age",
            "same-basis actual definition mapping without sending realized values in phase one",
            "explicit missing/unsupported status for every unfulfilled request ID",
            "license for local research and limits on redistribution and derived results",
        ],
        "optional_delivery_evidence": [
            "contributor IDs if licensed; not required for aggregate PIT validation"
        ],
        "provider_failures_may_be_replaced": False,
        "representative_universe": False,
        "historical_survivorship_coverage_validated": False,
        "return_values_loaded": False,
        "target_actual_values_loaded": False,
        "selection_uses_returns": False,
        "new_cohort_registered": False,
        "formal_model_admission": False,
        "purchase_authorized": False,
    }


def main() -> None:
    """Write a reproducible JSON request pack without network calls or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/consensus_source_trial_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/consensus_source_trial")
    )
    args = parser.parse_args()
    report = build_trial_request(args.config)
    content = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = args.output_root / f"{VERSION}-{_digest(content)[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to overwrite a different trial")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    print(
        f"SOURCE TRIAL READY issuers={report['issuer_count']} "
        f"requests_per_provider={report['resolved_requests_per_provider']}/40 "
        f"financial_samples=0 formal_admission=False output={path}"
    )


if __name__ == "__main__":
    main()
