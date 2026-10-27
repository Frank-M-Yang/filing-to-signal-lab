"""Source-preserving, window-specific collection targets and strict price reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sec_data.disclosure_audit import utc_timestamp, verify_file

VERSION = "announcement-market-supplement-v1"
PRICE_TOLERANCE = 1e-8


def validate_contract(config: dict[str, Any]) -> None:
    """Permit only the two documented gaps, a bounded rate and already frozen forecasts."""
    expected = {
        "version": VERSION,
        "expected_tickers": ["EW", "EA"],
        "maximum_collection_targets": 2,
        "request_interval_seconds": 12.1,
        "overlapping_price_absolute_tolerance": PRICE_TOLERANCE,
        "session_identity_fallback_frozen_before_collection": True,
        "reuse_previously_frozen_signals_only": True,
        "all_fourteen_cases_required": True,
        **dict.fromkeys(
            (
                "adjusted",
                "model_refitting_allowed",
                "sample_replacement_allowed",
                "first_public_certified",
                "trading_admission",
                "inferential_tests_allowed",
                "strategy_pnl_allowed",
                "overwrite_originals",
            ),
            False,
        ),
    }
    if any(type(config.get(k)) is not type(v) or config[k] != v for k, v in expected.items()):
        raise ValueError("Unsupported or weakened market supplementation contract")


def collection_targets(
    jobs: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    price_targets: pd.DataFrame,
    action_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clone two parent identities but query their explicitly frozen announcement windows.

    Parent identifiers are retained for same-case joins. The runner MUST isolate
    all completion files under the new content-addressed plan's raw-store root;
    these narrower queries must never reuse the parent's completion directory.
    """
    if [j["ticker"] for j in jobs] != ["EW", "EA"]:
        raise ValueError("Keep precisely EW and EA; no extra or replacement targets")
    lookup = {w["validation_event_id"]: w for w in windows}
    prices, actions = [], []
    for order, job in enumerate(jobs, 1):
        if job["window"] != lookup[job["validation_event_id"]]["window"]:
            raise ValueError("Supplement window changed from the frozen fourteen-case audit")
        p = price_targets.loc[price_targets.price_target_id.eq(job["price_target_id"])]
        a = action_targets.loc[action_targets.action_target_id.eq(job["action_target_id"])]
        if len(p) != 1 or len(a) != 1:
            raise ValueError("Missing or ambiguous parent collection identity")
        pr, ar = p.iloc[0].to_dict(), a.iloc[0].to_dict()
        for target in (pr, ar):
            if (
                any(
                    target[k] != job[v]
                    for k, v in (
                        ("cik", "cik"),
                        ("selected_ticker", "ticker"),
                        ("selected_exchange_mic", "exchange_mic"),
                        ("security_identity_key", "security_id"),
                        ("security_identity_key_kind", "security_id_kind"),
                    )
                )
                or target.get("selection_uses_return_values") is not False
            ):
                raise ValueError("Supplementation changed the outcome-blind parent identity")
            target["price_window_start"] = job["window"]["entry_session_date"]
            target["price_window_end"] = job["window"]["exit_session_date"]
            target["accepted_at"] = job["announcement_at"]
        pr.update(price_acquisition_order=order, adjusted=False)
        ar.update(
            action_acquisition_order=order,
            formal_entry_session_date=pr["price_window_start"],
            formal_exit_session_date=pr["price_window_end"],
            session_grid_fallback_frozen_before_collection=True,
        )
        prices.append(pr)
        actions.append(ar)
    return pd.DataFrame(prices), pd.DataFrame(actions)


def write_typed_bundle(
    output_root: Path,
    kind: str,
    files: dict[str, bytes],
    *,
    report_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Publish immutable typed manifests compatible with legacy collector/audit readers."""
    if not files or any(Path(n).name != n or n == "output_manifest.json" for n in files):
        raise ValueError("Bundle names must be direct artifact files, never paths or manifests")
    artifacts = {n: {"sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()}
    manifest = {**(metadata or {}), "version": kind, "artifacts": artifacts}
    if report_name:
        manifest["report_sha256"] = artifacts[report_name]["sha256"]
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode()
    root = output_root / (kind + "-" + hashlib.sha256(encoded).hexdigest()[:16])
    if root.exists():
        if not (root / "output_manifest.json").is_file() or (
            (root / "output_manifest.json").read_bytes() != encoded
        ):
            raise ValueError("Incomplete or changed immutable supplementation bundle")
        for name, item in artifacts.items():
            verify_file(root / name, item["sha256"])
        return root
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    (root / "output_manifest.json").write_bytes(encoded)
    return root


def replace_target_rows(base: pd.DataFrame, replacement: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return a derived view, retaining every unrelated target and leaving originals unchanged."""
    if key not in base or key not in replacement:
        raise ValueError("Target replacement requires explicit parent identifiers")
    return pd.concat([base.loc[~base[key].isin(replacement[key])], replacement], ignore_index=True)


def reconcile_overlap(
    original: pd.DataFrame,
    supplement: pd.DataFrame,
) -> pd.DataFrame:
    """Reject duplicate/nonfinite bars or material overlap revisions; never silently prefer them."""
    keys = ["price_target_id", "ticker", "session_date"]
    values = ["open_unadjusted", "high_unadjusted", "low_unadjusted", "close_unadjusted", "volume"]
    for frame in (original, supplement):
        if (
            frame.duplicated(keys).any()
            or not np.isfinite(frame[values].to_numpy(dtype=float)).all()
        ):
            raise ValueError("Duplicate or nonfinite price source rows")
    matched = original.merge(
        supplement, on=keys, suffixes=("_original", "_supplement"), validate="one_to_one"
    )
    rows = []
    for record in matched.to_dict("records"):
        gaps = {
            v: abs(float(record[v + "_original"]) - float(record[v + "_supplement"]))
            for v in values
        }
        if any(gap > (0 if v == "volume" else PRICE_TOLERANCE) for v, gap in gaps.items()):
            raise ValueError("Overlapping raw prices differ; do not overwrite or compute returns")
        rows.append(
            {
                **{k: record[k] for k in keys},
                "all_fields_agree": True,
                "original_source_sha256": record["source_sha256_original"],
                "supplement_source_sha256": record["source_sha256_supplement"],
                "maximum_absolute_price_gap": max(gaps[v] for v in values if v != "volume"),
            }
        )
    return pd.DataFrame(rows)


def verify_control_returns(
    controls: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    """Require all six previous controls to retain identical identities, windows and returns."""
    lookup = {r["validation_event_id"]: r for r in results}
    if (
        len(controls) != 6
        or len(results) != 14
        or len(lookup) != 14
        or (len({r["validation_event_id"] for r in controls}) != 6)
    ):
        raise ValueError("Six controls and fourteen unique same-case outcomes required")
    for old in controls:
        if old["validation_event_id"] not in lookup:
            raise ValueError("Previously verified control missing from expanded sample")
        current = lookup[old["validation_event_id"]]
        if any(
            current[k] != old[k]
            for k in (
                "cik",
                "ticker",
                "security_id",
                "price_target_id",
                "entry_price_type",
                "exit_price_type",
            )
        ) or any(
            utc_timestamp(current[k]) != utc_timestamp(old[k])
            for k in ("announcement_at", "entry_at", "exit_at")
        ):
            raise ValueError("Market supplementation changed a previously verified control window")
        for key in ("issuer_total_return", "benchmark_total_return", "market_excess_return"):
            left, right = float(current[key]), float(old[key])
            if not np.isfinite([left, right]).all() or abs(left - right) > 1e-12:
                raise ValueError(
                    "Market supplementation changed a previously verified control return"
                )
