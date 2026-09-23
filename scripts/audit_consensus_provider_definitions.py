"""Inventory saved response metadata; do not confuse field presence with definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sec_data.alpha_vantage_consensus import alpha_vantage_response_status

VERSION = "consensus-provider-definition-audit-v1"
# These are inspection hints, not provider-supported field names or parsing rules.
METADATA_HINTS = {
    "currency": {"currency", "reportedcurrency", "revenuecurrency"},
    "scale": {"unit", "units", "scale", "revenuescale"},
    "accounting_scope": {"revenuedefinition", "revenuescope", "accountingbasis"},
    "update_time": {"updatedat", "lastupdated", "asof", "timestamp", "lastrefreshed"},
    "fiscal_mapping": {"fiscalperiodend", "fiscalyear", "fiscalquarter"},
    "analyst_identity": {"analystid", "analystids", "analysts", "contributors"},
}


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Capture time requires an explicit timezone")
    return stamp.astimezone(timezone.utc)


def _hint_paths(document: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {name: [] for name in METADATA_HINTS}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}"
                normalized = str(key).replace("_", "").replace(" ", "").lower()
                for name, aliases in METADATA_HINTS.items():
                    if normalized in aliases and child is not None and child != "":
                        result[name].append(path)
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, prefix + "[]")

    visit(document, "payload")
    return {name: sorted(set(paths)) for name, paths in result.items()}


def audit_saved_response(manifest_path: Path, *, raw_root: Path) -> dict[str, Any]:
    """Verify a cached response and inventory its schema without inferring economics.

    Error/redacted responses remain visible, without copying their bodies. The
    inventory is not a continuity check and makes no latest-attempt claim.
    """
    manifest_content = manifest_path.read_bytes()
    manifest = json.loads(manifest_content)
    # Support the documented privacy-redacted filename but never leave the bundle.
    filename = Path(str(manifest["path"])).name
    if filename not in {"response.bin", "response.redacted.bin"}:
        raise ValueError("Unrecognized stored response filename")
    content_path = (manifest_path.parent / filename).resolve()
    if not content_path.is_relative_to(raw_root.resolve()):
        raise ValueError("Response escaped the specified raw root")
    content = content_path.read_bytes()
    if (
        hashlib.sha256(content).hexdigest() != manifest["sha256"]
        or len(content) != manifest["bytes"]
    ):
        raise ValueError("Snapshot integrity mismatch")
    captured = _timestamp(manifest["fetched_at"])
    if captured.date().isoformat() != manifest["capture_date"]:
        raise ValueError("Capture date does not match timestamp")
    status = alpha_vantage_response_status(content, manifest["status_code"])
    if "status" in manifest and manifest["status"] != status:
        raise ValueError("Response status was relabelled")
    result: dict[str, Any] = {
        "ticker": manifest["ticker"],
        "cik": manifest["cik"],
        "captured_at": captured.isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
        "source_sha256": manifest["sha256"],
        "status": status,
        "metadata_inventory_performed": False,
        "top_level_fields": [],
        "estimate_row_fields": [],
        "quarterly_rows": 0,
        "future_quarterly_rows": 0,
        "metadata_hint_paths": {name: [] for name in METADATA_HINTS},
        "http_timestamp_is_estimate_timestamp": False,
        "provider_definition_verified": False,
        "formal_model_admission": False,
    }
    if manifest.get("credentials_redacted", False) or status != "ok":
        result["inventory_exclusion"] = "redacted_or_unsuccessful_response"
        return result
    document = json.loads(content)
    if document.get("symbol") != manifest["ticker"]:
        raise ValueError("Issuer identity mismatch")
    rows = document["estimates"]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Malformed estimate rows")
    quarters = [r for r in rows if str(r.get("horizon", "")).lower() == "fiscal quarter"]
    future = [r for r in quarters if date.fromisoformat(str(r["date"])) >= captured.date()]
    result.update(
        {
            "metadata_inventory_performed": True,
            "top_level_fields": sorted(document),
            "estimate_row_fields": sorted(set().union(*(set(row) for row in rows))),
            "quarterly_rows": len(quarters),
            # Includes zero/invalid financial aggregates: schema coverage, not valid candidates.
            "future_quarterly_rows": len(future),
            "metadata_hint_paths": _hint_paths(document),
        }
    )
    return result


def build_definition_audit(raw_roots: list[Path], *, review_path: Path) -> dict[str, Any]:
    """Tie all saved-response inventories to a hash-verified manual document review."""
    review_content = review_path.read_bytes()
    review = json.loads(review_content)
    if (
        review.get("review_version") != "alpha-vantage-definition-review-v1"
        or review.get("endpoint") != "EARNINGS_ESTIMATES"
        or review.get("formal_model_admission") is not False
        or review.get("vendor_response_received") is not False
        or not review.get("unresolved_definitions")
        or not review.get("sources")
    ):
        raise ValueError("Expected the unresolved v1 review, not an admission authorization")
    for source in review["sources"]:
        content = Path(source["local_path"]).read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != source["sha256"]
            or len(content) != source["bytes"]
        ):
            raise ValueError("Public documentation evidence changed")
    snapshots = []
    seen: set[Path] = set()
    for root in raw_roots:
        for path in sorted(root.glob("snapshots/*/*/*/manifest.json")):
            if path.resolve() in seen:
                raise ValueError("Overlapping raw roots would double count evidence")
            seen.add(path.resolve())
            snapshots.append({"raw_scope": root.name, **audit_saved_response(path, raw_root=root)})
    if not snapshots:
        raise ValueError("No saved responses; no empty success report")
    valid = [r for r in snapshots if r["metadata_inventory_performed"]]
    return {
        "audit_version": VERSION,
        "definition_review_sha256": hashlib.sha256(review_content).hexdigest(),
        "definition_review": review,
        "saved_response_count": len(snapshots),
        "inventoried_response_count": len(valid),
        "excluded_response_count": len(snapshots) - len(valid),
        "issuer_count": len({r["cik"] for r in snapshots}),
        "capture_dates": sorted({r["captured_at"][:10] for r in snapshots}),
        "responses_with_metadata_hints": {
            name: sum(bool(r["metadata_hint_paths"][name]) for r in valid)
            for name in METADATA_HINTS
        },
        "snapshots": snapshots,
        "review_status": "unresolved_definitions_require_vendor_clarification",
        "inventory_is_exhaustive_semantic_validation": False,
        "absence_proves_provider_lacks_information": False,
        "saved_responses_are_independent_events": False,
        "transport_failure_continuity_audited": False,
        "changes_frozen_policy": False,
        "formal_model_admission": False,
        "order_authorized": False,
        "target_actual_values_loaded": False,
        "return_values_loaded": False,
    }


def main() -> None:
    """Generate an offline, immutable audit; no keys, network or cohort changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, action="append")
    parser.add_argument(
        "--definition-review",
        type=Path,
        default=Path("docs/alpha_vantage_definition_review_v1.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/consensus_provider_definitions")
    )
    args = parser.parse_args()
    report = build_definition_audit(
        args.raw_root
        or [
            Path("data/market/alpha_vantage_consensus"),
            Path("data/market/alpha_vantage_coverage_probe"),
        ],
        review_path=args.definition_review,
    )
    content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(content).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{VERSION}-{digest[:16]}.json"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to overwrite different evidence")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    print(
        f"DEFINITION AUDIT DONE responses={len(report['snapshots'])} "
        f"issuers={report['issuer_count']} formal_admission=False output={path}"
    )


if __name__ == "__main__":
    main()
