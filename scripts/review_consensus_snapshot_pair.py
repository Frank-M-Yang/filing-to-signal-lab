"""Offline forensic review of two explicit probe captures; never admits events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from prospective_consensus_review import calculate_consensus_membership_bounds

VERSION = "consensus-snapshot-pair-review-v1"
FIELDS = ("average", "low", "high", "analyst_count")


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Capture timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _load(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest = json.loads(path.read_bytes())
    # Use the adjacent payload, not machine-specific absolute paths in old manifests.
    content = (path.parent / "response.bin").read_bytes()
    if (
        manifest.get("status") != "ok"
        or manifest.get("credentials_redacted", False)
        or hashlib.sha256(content).hexdigest() != manifest.get("sha256")
        or len(content) != manifest.get("bytes")
    ):
        raise ValueError("Only successful, intact, unredacted probe snapshots are reviewable")
    payload = json.loads(content)
    if payload.get("symbol") != manifest.get("ticker"):
        raise ValueError("Payload and manifest issuer disagree")
    if not isinstance(payload.get("estimates"), list) or not payload["estimates"]:
        raise ValueError("No estimate rows")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for occurrence, row in enumerate(payload["estimates"]):
        if not isinstance(row, dict):
            raise ValueError("Malformed estimate row")
        horizon = str(row.get("horizon", "")).strip().lower()
        if horizon not in {"fiscal quarter", "fiscal year"}:
            raise ValueError("Unrecognized horizon; manual review required")
        period = str(row.get("date", ""))
        date.fromisoformat(period)
        key = horizon, period
        if key in indexed:
            raise ValueError("Duplicate period and horizon; do not pick an arbitrary row")
        indexed[key] = {**row, "source_occurrence": occurrence}
    return manifest, indexed


def _aggregate(row: dict[str, Any]) -> dict[str, float]:
    values = {}
    for field in FIELDS:
        raw = row.get(f"revenue_estimate_{field}")
        if raw is None:
            raise ValueError("Missing revenue aggregate")
        if isinstance(raw, bool):
            raise ValueError("Boolean is not a financial aggregate")
        try:
            number = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("Missing or invalid revenue aggregate") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError("Revenue aggregates must be finite and positive")
        values[field] = number
    if not values["analyst_count"].is_integer():
        raise ValueError("Analyst count must be an integer")
    if not values["low"] <= values["average"] <= values["high"]:
        raise ValueError("Mean must lie within the reported range")
    return values


def review_consensus_snapshot_pair(
    previous_manifest: Path, current_manifest: Path, *, period_end: str
) -> dict[str, Any]:
    """Compare one future quarter by semantic key, with conditional membership bounds.

    This is a caller-selected forensic pair, not an adjacent-attempt monitor. It
    must not replace the continuity pipeline (which never bridges failed attempts).
    The provider period remains unmodified and is not a verified fiscal mapping.
    """
    previous_meta, previous_rows = _load(previous_manifest)
    current_meta, current_rows = _load(current_manifest)
    for field in ("ticker", "cik", "coverage_probe_target_id"):
        if not previous_meta.get(field) or previous_meta[field] != current_meta.get(field):
            raise ValueError("Snapshots must belong to the same fixed probe target")
    previous_time = _utc(previous_meta["fetched_at"])
    current_time = _utc(current_meta["fetched_at"])
    if current_time <= previous_time:
        raise ValueError("Capture times must strictly increase")
    if date.fromisoformat(period_end) <= current_time.date():
        raise ValueError("Selected provider quarter must still be in the future")
    key = "fiscal quarter", period_end
    if key not in previous_rows or key not in current_rows:
        raise ValueError("The same quarterly key must exist in both captures")
    previous = _aggregate(previous_rows[key])
    current = _aggregate(current_rows[key])
    bounds = calculate_consensus_membership_bounds(
        int(previous["analyst_count"]),
        int(current["analyst_count"]),
        previous["average"],
        current["average"],
        previous["low"],
        previous["high"],
        current["low"],
        current["high"],
    )
    common_keys = sorted(previous_rows.keys() & current_rows.keys())
    key_rows = [
        {
            "horizon": horizon,
            "provider_period_end": period,
            "previous_occurrence": previous_rows[horizon, period]["source_occurrence"],
            "current_occurrence": current_rows[horizon, period]["source_occurrence"],
            "revenue_aggregate_fields_changed": any(
                previous_rows[horizon, period].get(f"revenue_estimate_{field}")
                != current_rows[horizon, period].get(f"revenue_estimate_{field}")
                for field in FIELDS
            ),
        }
        for horizon, period in common_keys
    ]
    old_width = previous["high"] - previous["low"]
    return {
        "review_version": VERSION,
        "ticker": previous_meta["ticker"],
        "cik": previous_meta["cik"],
        "provider_period_end": period_end,
        "horizon": "fiscal quarter",
        "pair_selection": "explicit_forensic_pair_not_automatic_continuity",
        "previous_captured_at": previous_time.isoformat(),
        "current_captured_at": current_time.isoformat(),
        "previous_source_sha256": previous_meta["sha256"],
        "current_source_sha256": current_meta["sha256"],
        "previous_manifest_sha256": hashlib.sha256(previous_manifest.read_bytes()).hexdigest(),
        "current_manifest_sha256": hashlib.sha256(current_manifest.read_bytes()).hexdigest(),
        "elapsed_hours": (current_time - previous_time).total_seconds() / 3600,
        "previous_aggregate": previous,
        "current_aggregate": current,
        "previous_occurrence": previous_rows[key]["source_occurrence"],
        "current_occurrence": current_rows[key]["source_occurrence"],
        "aggregate_mean_revision_fraction": current["average"] / previous["average"] - 1,
        "range_width_ratio": (current["high"] - current["low"]) / old_width
        if old_width > 0
        else None,
        "new_keys": [list(k) for k in sorted(current_rows.keys() - previous_rows.keys())],
        "removed_keys": [list(k) for k in sorted(previous_rows.keys() - current_rows.keys())],
        "common_key_comparison": key_rows,
        "conditional_membership_bounds": bounds,
        "bounds_assumptions_verified": False,
        "bounds_assumptions": [
            "unweighted arithmetic means over the reported analyst panel",
            "high and low cover that same panel",
            "the smaller panel is a subset of the larger panel",
            "retained analysts have unchanged estimates under the tested null",
        ],
        "provider_currency_unit_scope_verified": False,
        "provider_fiscal_calendar_mapping_verified": False,
        "provider_updated_at": None,
        "market_revision_at": None,
        "new_market_information_verified": False,
        "target_actual_values_loaded": False,
        "return_values_loaded": False,
        "formal_model_admission": False,
        "order_authorized": False,
    }


def main() -> None:
    """Write a deterministic, content-addressed JSON review without any network calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--period-end", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/consensus_snapshot_reviews")
    )
    args = parser.parse_args()
    report = review_consensus_snapshot_pair(
        args.previous_manifest, args.current_manifest, period_end=args.period_end
    )
    content = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(content).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{VERSION}-{digest[:16]}.json"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to overwrite an existing different review")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    print(
        f"SNAPSHOT REVIEW DONE ticker={report['ticker']} "
        f"mean_change={report['aggregate_mean_revision_fraction']:.4%} "
        f"formal_admission=False output={path}"
    )


if __name__ == "__main__":
    main()
