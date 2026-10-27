"""Freeze, optionally download, and inspect the next twenty SEC sources without returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_sec_clock_layer import _artifact, _encode, _json
from scripts.build_sec_disclosure_audit import write_bundle
from sec_data.client import SecClient
from sec_data.disclosure_audit import (
    METRIC_TAGS,
    audit_financial_fields,
    digest_file,
    load_cached_submissions,
    verify_file,
)
from sec_data.disclosure_review import (
    audit_submission_clock,
    review_document_kinds,
    review_metric_cells,
)
from sec_data.guidance_evidence import collect_guidance_documents, parse_complete_submission
from sec_data.next_disclosure_review import VERSION, freeze_sources, pair_dispositions


def prepare_sources(
    config_path: Path, output_root: Path, client: SecClient | None = None
) -> tuple[Path, dict[str, Any]]:
    """Freeze existing queue before values or network, then preserve exact raw responses."""
    c = json.loads(config_path.read_bytes())
    if c.get("version") != VERSION or any(
        c.get(k) is not False
        for k in (
            "sample_replacement_allowed",
            "model_fitting_allowed",
            "return_values_allowed",
            "trading_admission",
            "overwrite_originals",
        )
    ):
        raise ValueError("Unsupported or weakened source-review contract")
    if client is not None and (
        c["network_allowed"] is not True
        or client.request_interval_seconds < 0.20
        or client.max_retries != 0
    ):
        raise ValueError("Collection exceeds the frozen SEC rate/request contract")
    previous_root, previous_hash = Path(c["source_root"]), c["source_manifest_sha256"]
    queue_path = _artifact(previous_root, previous_hash, "next_review_batch.json")
    queue = json.loads(queue_path.read_bytes())
    old = _json(Path(c["expanded_plan_root"]), c["expanded_plan_manifest_sha256"], "plan.json")
    source = old["financial_sources"]
    history = pd.read_parquet(
        _artifact(
            Path(source["feature_root"]),
            source["feature_manifest_sha256"],
            "quarterly_feature_history.parquet",
        ),
        columns=[
            "company_id",
            "filing_accession",
            "period_start",
            "period_end",
            "filing_form",
            "derivation_method",
        ],
    )
    catalog = pd.read_parquet(
        _artifact(
            Path(source["catalog_root"]),
            source["catalog_manifest_sha256"],
            "filing_catalog.parquet",
        ),
        columns=["cik", "accession_number", "form", "report_date", "accepted_at", "items"],
    )
    events, candidates, targets = freeze_sources(queue, history, catalog)
    plan = {
        "version": VERSION,
        "config": c,
        "config_sha256": digest_file(config_path),
        "queue_sha256": digest_file(queue_path),
        "financial_sources": source,
        "events": events,
        "candidates": candidates,
        "document_targets": targets,
        "selection_uses_values_or_returns": False,
        "original_frozen_issuers_retained": 20,
        "implementation_sha256": {
            name: digest_file(Path(name))
            for name in (
                "scripts/review_next_disclosures.py",
                "sec_data/next_disclosure_review.py",
                "sec_data/disclosure_audit.py",
                "sec_data/disclosure_review.py",
                "sec_data/guidance_evidence.py",
            )
        },
    }
    frozen = write_bundle(output_root / "plans", VERSION + "-plan", {"plan.json": _encode(plan)})
    roots = [Path(p) for p in source["cached_submission_roots"]] + [Path(c["raw_root"])]
    cache = load_cached_submissions(roots)
    missing = [r for r in targets if (r["cik"], r["prior_accession_number"]) not in cache]
    collection: dict[str, Any] = {
        "selected_documents": 0,
        "completed_documents": 0,
        "failed_documents": 0,
        "requests_made": 0,
    }
    if client is not None and missing:
        result = collect_guidance_documents(
            pd.DataFrame(missing),
            client,
            raw_root=Path(c["raw_root"]),
            limit=32,
            progress=lambda done, total, identity: print(
                f"SEC NEXT SOURCE progress={done}/{total} document={identity[:12]}", flush=True
            ),
        )
        collection = vars(result)
        cache = load_cached_submissions(roots)
    # Numeric originals are opened only after selection is frozen. No price/return tables are read.
    verify_file(Path(source["companyfacts_zip"]), source["companyfacts_sha256"])
    fields: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    clocks: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(source["companyfacts_zip"]) as archive:
        for event in events:
            if event["form"] == "10-Q":
                body = archive.read(f"CIK{event['cik']}.json")
                fields.extend(
                    {
                        **r,
                        "source_member_sha256": hashlib.sha256(body).hexdigest(),
                        "source_zip_sha256": source["companyfacts_sha256"],
                    }
                    for r in audit_financial_fields(event, json.loads(body), catalog)
                )
            else:
                fields.extend(
                    {
                        "validation_event_id": event["validation_event_id"],
                        "cik": event["cik"],
                        "metric": metric,
                        "value": None,
                        "status": "annual_quarter_requires_separate_source_proof",
                        "quarter_or_instant_ready": False,
                        "derived_quarter_candidate": None,
                    }
                    for metric in METRIC_TAGS
                )
    event_map = {r["validation_event_id"]: r for r in events}
    for candidate in candidates:
        key = candidate["cik"], candidate["accession_number"]
        if key not in cache:
            unavailable.append({**candidate, "status": "raw_submission_not_collected"})
            continue
        meta = cache[key]
        verify_file(Path(meta["path"]), meta["sha256"])
        if meta["requested_url"] != candidate["url"]:
            raise ValueError("Cached source URL is outside the frozen identity")
        content = Path(meta["path"]).read_bytes()
        clock = audit_submission_clock(content, candidate)
        clocks.append(
            {
                **candidate,
                "source_sha256": meta["sha256"],
                "lab_fetched_at": meta["fetched_at"],
                **clock,
            }
        )
        event = event_map[candidate["validation_event_id"]]
        for document in parse_complete_submission(content):
            if not (
                document["document_type"].upper().startswith("EX-99")
                or (
                    document["document_type"] == candidate["form"]
                    and candidate["source_role"].startswith("periodic_report")
                )
            ):
                continue
            filename = (
                f"{event['ticker']}_{candidate['accession_number']}_"
                f"{document['source_occurrence']}.txt"
            )
            files[filename] = re.sub(r"\s+", " ", document["visible_text"]).encode()
            documents.append(
                {
                    "validation_event_id": event["validation_event_id"],
                    "cik": event["cik"],
                    "ticker": event["ticker"],
                    "period_start": event["period_start"],
                    "period_end": event["period_end"],
                    "source_role": candidate["source_role"],
                    "release_accession": candidate["accession_number"],
                    "submission_sha256": meta["sha256"],
                    "exhibit_filename": document["filename"],
                    "exhibit_sha256": document["document_sha256"],
                    "document_type": document["document_type"],
                    "visible_text_file": filename,
                    "clock": clock,
                }
            )
    summary = {
        "version": VERSION,
        "plan_root": frozen.as_posix(),
        "plan_manifest_sha256": digest_file(frozen / "output_manifest.json"),
        "frozen_issuers": 20,
        "target_documents": len(targets),
        "collected_documents": len(clocks),
        "documents_from_new_batch_cache": sum(
            Path(cache[(r["cik"], r["accession_number"])]["path"])
            .resolve()
            .is_relative_to(Path(c["raw_root"]).resolve())
            for r in clocks
        ),
        "uncollected_documents": len(unavailable),
        "statement_documents_prepared": len(documents),
        "header_catalog_conflicts": sum(
            r["status"] != "consistent_acceptance_proxy" for r in clocks
        ),
        "existing_scope_quarantined_issuers": sum(r["existing_scope_quarantine"] for r in events),
        "collection": collection,
        "first_public_certified": False,
        "models_fitted": 0,
        "return_values_read": False,
        "trading_admission": False,
        "prior_artifacts_overwritten": False,
    }
    files.update(
        {
            "summary.json": _encode(summary),
            "plan.json": _encode(plan),
            "documents.json": _encode(documents),
            "field_audit.json": _encode(fields),
            "header_clock_audit.json": _encode(clocks),
            "uncollected_sources.json": _encode(unavailable),
        }
    )
    return write_bundle(output_root, VERSION + "-prepared", files), summary


def review_sources(
    prepared_root: Path, prepared_hash: str, review_path: Path, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Recheck every hash/GAAP/period/unit/cell assertion and retain unresolved cases."""
    review = json.loads(review_path.read_bytes())
    if review.get("version") != VERSION or any(
        review.get(k) is not False
        for k in (
            "first_public_certified",
            "model_fitting_allowed",
            "return_values_allowed",
            "sample_replacement_allowed",
            "rounding_plugs_allowed",
        )
    ):
        raise ValueError("Unsupported or weakened original-cell review contract")
    plan = _json(prepared_root, prepared_hash, "plan.json")
    prepared_report = _json(prepared_root, prepared_hash, "summary.json")
    documents = _json(prepared_root, prepared_hash, "documents.json")

    def specification(assertion: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
        matching = [
            d
            for d in documents
            if d["ticker"] == assertion["ticker"]
            and d["release_accession"] == assertion["release_accession"]
            and d["exhibit_filename"] == assertion["exhibit_filename"]
            and d["document_type"].upper().startswith("EX-99")
        ]
        if len(matching) != 1:
            raise ValueError("Reviewed exhibit is missing or ambiguous")
        if any(assertion.get(k) != matching[0][k] for k in ("submission_sha256", "exhibit_sha256")):
            raise ValueError("Review is not bound to the exact observed submission/exhibit bytes")
        return {
            **matching[0],
            **assertion,
            "kind": assertion["kind"] if partial else "earnings_release",
            "anchors": [assertion["document_anchor"]],
            "accounting_basis": "GAAP",
        }

    specs = [specification(r) for r in review["reviews"]]
    partial = [specification(r, partial=True) for r in review["partial_candidates"]]
    cache = load_cached_submissions(
        [Path(p) for p in plan["financial_sources"]["cached_submission_roots"]]
        + [Path(plan["config"]["raw_root"])]
    )
    kinds = review_document_kinds(specs + partial, plan["events"], plan["candidates"], cache)
    cells = review_metric_cells(
        specs,
        plan["events"],
        _json(prepared_root, prepared_hash, "field_audit.json"),
        plan["candidates"],
        cache,
    )
    ledger = pair_dispositions(plan["events"], cells, kinds, review["unresolved_reasons"])
    timeline = []
    for event in ledger:
        for pair in event["reviewed_pairs"]:
            if pair["conditional_pair_verified"] and pair["later_than_frozen_earliest_proxy"]:
                early = [
                    d
                    for d in kinds
                    if d["validation_event_id"] == event["validation_event_id"]
                    and d["release_accession"] == event["release_accession_number"]
                    and d["kind"] == "preliminary_results"
                ]
                if len(early) != 1:
                    raise ValueError(
                        "Later complete pair lacks a reviewed earlier partial candidate"
                    )
                timeline.append(
                    {
                        "ticker": event["ticker"],
                        "validation_event_id": event["validation_event_id"],
                        "earlier_candidate_acceptance_proxy_at": event["earliest_disclosure_at"],
                        "earlier_candidate_accession": event["release_accession_number"],
                        "earlier_candidate_submission_sha256": early[0]["submission_sha256"],
                        "complete_pair_acceptance_proxy_at": pair["pair_acceptance_proxy_at"],
                        "complete_pair_accession": pair["release_accession"],
                        "complete_pair_submission_sha256": pair["submission_sha256"],
                        "exact_complete_pair_not_verified_in_earlier_candidate": True,
                        "first_public_certified": False,
                        "backfill_to_earlier_proxy_allowed": False,
                    }
                )
    summary = {
        "version": VERSION + "-reviewed",
        "prepared_root": prepared_root.as_posix(),
        "prepared_manifest_sha256": prepared_hash,
        "review_config_sha256": digest_file(review_path),
        "frozen_issuers_retained": 20,
        "target_documents": prepared_report["target_documents"],
        "collected_documents": prepared_report["collected_documents"],
        "documents_from_new_batch_cache": prepared_report["documents_from_new_batch_cache"],
        "header_catalog_conflicts": prepared_report["header_catalog_conflicts"],
        "existing_scope_quarantined_issuers": prepared_report["existing_scope_quarantined_issuers"],
        "reviewed_pairs": len(specs),
        "conditionally_verified_original_pairs": sum(
            r["conditional_pair_verified"] for r in ledger
        ),
        "later_complete_pairs_not_backfilled": sum(
            any(
                p["conditional_pair_verified"] and p["later_than_frozen_earliest_proxy"]
                for p in r["reviewed_pairs"]
            )
            for r in ledger
        ),
        "partial_candidate_documents_reviewed": len(partial),
        "unresolved_cases_retained": sum(not r["conditional_pair_verified"] for r in ledger),
        "first_public_certified": False,
        "models_fitted": 0,
        "return_values_read": False,
        "trading_admission": False,
        "old_signals_and_returns_recomputed": False,
    }
    return write_bundle(
        output_root,
        VERSION + "-reviewed",
        {
            "summary.json": _encode(summary),
            "case_dispositions.json": _encode(ledger),
            "reviewed_metric_cells.json": _encode(cells),
            "reviewed_document_kinds.json": _encode(kinds),
            "information_set_timeline.json": _encode(timeline),
        },
    ), summary


def main() -> None:
    """Prepare the fixed batch; collecting requires an identified SEC_USER_AGENT, never a key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/next_disclosure_review_v1.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/next_disclosure_review")
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    client = (
        SecClient(
            os.environ.get("SEC_USER_AGENT", ""), request_interval_seconds=0.25, max_retries=0
        )
        if args.collect
        else None
    )
    root, report = prepare_sources(args.config, args.output_root, client)
    if args.reviews:
        root, report = review_sources(
            root, digest_file(root / "output_manifest.json"), args.reviews, args.output_root
        )
    print(
        f"SEC NEXT REVIEW DONE version={report['version']} "
        f"report={json.dumps(report)} output={root}"
    )


if __name__ == "__main__":
    main()
