"""Build the frozen point-in-time price-source readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sec_data.price_research import build_price_readiness_audit


def main() -> int:
    """Build a non-imputing price readiness asset for the probability sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope-root",
        type=Path,
        default=Path(
            "data/processed/sec_probability_scope/sec-equity-scope-audit-v1-91be9ca9abf87966"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Massive reference audit; defaults to latest audit matching --scope-root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/sec_price_readiness"),
    )
    parser.add_argument(
        "--price-root",
        type=Path,
        default=None,
        help="Massive price audit; defaults to latest audit matching the reference",
    )
    parser.add_argument(
        "--calendar-root",
        type=Path,
        default=None,
        help="exchange-calendar audit; defaults to latest audit matching the price asset",
    )
    parser.add_argument(
        "--security-actions-root",
        type=Path,
        default=None,
        help="security-actions audit; defaults to latest audit matching price and calendar",
    )
    parser.add_argument(
        "--delisting-root",
        type=Path,
        default=None,
        help="delisting-outcome audit; defaults to latest audit matching security actions",
    )
    args = parser.parse_args()
    reference_root = args.reference_root or _latest_matching_reference(
        Path("data/processed/massive_reference"), args.scope_root
    )
    price_root = args.price_root or _latest_matching_price(
        Path("data/processed/massive_prices"), reference_root
    )
    calendar_root = args.calendar_root or (
        _latest_matching_calendar(Path("data/processed/exchange_calendar"), price_root)
        if price_root is not None
        else None
    )
    security_actions_root = args.security_actions_root or (
        _latest_matching_security_actions(
            Path("data/processed/massive_security_actions"), price_root, calendar_root
        )
        if price_root is not None and calendar_root is not None
        else None
    )
    delisting_root = args.delisting_root or (
        _latest_matching_delisting(
            Path("data/processed/massive_delisting"),
            security_actions_root,
            calendar_root,
        )
        if security_actions_root is not None and calendar_root is not None
        else None
    )
    result = build_price_readiness_audit(
        args.scope_root,
        reference_root=reference_root,
        price_root=price_root,
        calendar_root=calendar_root,
        security_actions_root=security_actions_root,
        delisting_root=delisting_root,
        output_root=args.output_root,
    )
    report = result.report
    print(
        "PRICE READINESS DONE "
        f"version={report['dataset_version']} "
        f"scope={report['scope_adjudicated']:,}/{report['selected_issuer_years']:,} "
        f"mapping_candidates={report['price_mapping_candidates']:,} "
        f"reference={report['reference_version']} "
        f"reference_candidates={report['event_date_reference_candidates']:,} "
        f"price_queries={report['provider_price_queries_complete']:,} "
        f"window_candidates={report['observed_event_window_candidates']:,} "
        f"formal_windows={report['formal_event_window_ready']:,} "
        f"stable_intervals={report['stable_security_mappings_ready']:,} "
        f"actions_ready={report['corporate_action_windows_ready']:,} "
        f"outcomes_ready={report['delisting_outcomes_ready']:,} "
        f"eligible={report['return_study_eligible']:,} "
        f"gate={report['return_study_gate']} "
        f"output={result.output_root}",
        flush=True,
    )
    return 0


def _latest_matching_reference(search_root: Path, scope_root: Path) -> Path:
    scope_manifest_sha256 = hashlib.sha256(
        (scope_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_reference_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("scope_manifest_sha256") == scope_manifest_sha256:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    if not candidates:
        raise FileNotFoundError(
            "No Massive reference audit matches the current scope; run "
            "scripts.build_massive_reference_pilot first"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _latest_matching_price(search_root: Path, reference_root: Path) -> Path | None:
    reference_manifest_sha256 = hashlib.sha256(
        (reference_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_price_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("reference_manifest_sha256") == reference_manifest_sha256:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _latest_matching_calendar(search_root: Path, price_root: Path) -> Path | None:
    price_manifest_sha256 = hashlib.sha256(
        (price_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/exchange_calendar_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("price_manifest_sha256") == price_manifest_sha256:
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _latest_matching_security_actions(
    search_root: Path, price_root: Path, calendar_root: Path
) -> Path | None:
    price_manifest_sha256 = hashlib.sha256(
        (price_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    calendar_manifest_sha256 = hashlib.sha256(
        (calendar_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_security_actions_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("price_manifest_sha256") == price_manifest_sha256
            and report.get("calendar_manifest_sha256") == calendar_manifest_sha256
        ):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _latest_matching_delisting(
    search_root: Path, security_actions_root: Path, calendar_root: Path
) -> Path | None:
    actions_manifest_sha256 = hashlib.sha256(
        (security_actions_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    calendar_manifest_sha256 = hashlib.sha256(
        (calendar_root / "output_manifest.json").read_bytes()
    ).hexdigest()
    candidates: list[tuple[str, Path]] = []
    for report_path in search_root.glob("*/massive_delisting_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("security_actions_manifest_sha256") == actions_manifest_sha256
            and report.get("calendar_manifest_sha256") == calendar_manifest_sha256
        ):
            candidates.append((str(report.get("generated_at", "")), report_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


if __name__ == "__main__":
    sys.exit(main())
