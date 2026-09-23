"""Rule-based earliest SEC disclosure clock for periodic-report financial facts.

The research panel times each quarterly observation at its 10-Q/10-K SEC
acceptance.  Many issuers publish the same period's results earlier, in an
earnings release furnished on Form 8-K under Item 2.02.  This module assigns,
for every original periodic report, the earliest SEC-evidenced disclosure of
that period's results under one explicit rule instead of per-event review:

- candidate releases are original (non-amended) 8-Ks whose item list contains
  Item 2.02;
- a candidate must be accepted strictly after the issuer's previous original
  periodic report, on or after the New York day following the fiscal period
  end, no more than ``max_release_lead_days`` before the report, and no later
  than the report itself;
- the earliest qualifying candidate is the release clock; the number of
  qualifying candidates is kept as an ambiguity measure.

Both clocks remain operational proxies.  Item 2.02 lets an issuer furnish the
8-K up to four business days after a public announcement, so the 8-K
acceptance is not a certified first-public time; the issuer-reported event
date is recorded separately at date precision.  Nothing here loads prices or
returns, and ``first_public_certified`` is always false.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from sec_data.filing_catalog import FilingCatalogError, load_catalog_table

FIRST_PUBLIC_CLOCK_VERSION: Final[str] = "first-public-clock-v1"
CATALOG_COLUMNS: Final[tuple[str, ...]] = (
    "cik",
    "form",
    "base_form",
    "is_amendment",
    "report_date",
    "accepted_at",
    "items",
    "accession_number",
)
PERIODIC_BASE_FORMS: Final[frozenset[str]] = frozenset({"10-Q", "10-K"})
EARNINGS_RELEASE_ITEM: Final[str] = "2.02"
# Items that commonly travel with an earnings release without describing a
# separate, possibly earlier event (Regulation FD furnishing and exhibits).
EARNINGS_COMPANION_ITEMS: Final[frozenset[str]] = frozenset({"2.02", "7.01", "9.01"})
_OFFSET_PATTERN: Final[str] = r"(?:Z|[+-]\d{2}:\d{2})$"
_ITEM_PATTERN: Final[str] = r"\d+\.\d+"
_NANOSECONDS_PER_DAY: Final[float] = 86_400e9


class FirstPublicClockError(FilingCatalogError):
    """Raised when catalog inputs cannot support an unambiguous clock rule."""


@dataclass(frozen=True)
class FirstPublicClockConfig:
    """Explicit matching assumptions; changing one requires a new version."""

    max_release_lead_days: int = 90
    timezone: str = "America/New_York"
    version: str = FIRST_PUBLIC_CLOCK_VERSION

    def validate(self) -> None:
        """Reject matching windows that could reach another fiscal period."""
        if not 1 <= self.max_release_lead_days <= 180:
            raise FirstPublicClockError("max_release_lead_days must lie in [1, 180]")
        if not self.timezone.strip():
            raise FirstPublicClockError("timezone cannot be blank")
        if not self.version.strip():
            raise FirstPublicClockError("version cannot be blank")


@dataclass(frozen=True)
class FirstPublicClockResult:
    """Immutable audit bundle location and its report."""

    output_root: Path
    report: dict[str, Any]


def build_first_public_clock(
    catalog: pd.DataFrame,
    config: FirstPublicClockConfig | None = None,
) -> pd.DataFrame:
    """Assign each original 10-Q/10-K its earliest SEC-evidenced disclosure clock.

    Returns one row per original periodic report with the matched Item 2.02
    release (if any), the candidate count, ``earliest_disclosure_at``, its
    ``clock_source``, and the lead of that clock over the report acceptance.
    """
    cfg = config or FirstPublicClockConfig()
    cfg.validate()
    missing = sorted(set(CATALOG_COLUMNS).difference(catalog.columns))
    if missing:
        raise FirstPublicClockError(f"Catalog lacks columns: {missing}")

    frame = catalog.loc[:, list(CATALOG_COLUMNS)].copy()
    original = ~frame["is_amendment"].astype(bool)
    report_mask = original & frame["base_form"].isin(PERIODIC_BASE_FORMS)
    item_sets = frame["items"].fillna("").astype(str).str.findall(_ITEM_PATTERN)
    release_mask = (
        original
        & frame["base_form"].eq("8-K")
        & item_sets.map(lambda items: EARNINGS_RELEASE_ITEM in items)
    )
    relevant = frame.loc[report_mask | release_mask].copy()
    relevant["item_list"] = item_sets.loc[relevant.index]
    accepted_text = relevant["accepted_at"].astype(str)
    if not accepted_text.str.contains(_OFFSET_PATTERN, regex=True).all():
        raise FirstPublicClockError("Every relevant acceptance time needs an explicit offset")
    relevant["accepted_utc"] = pd.to_datetime(accepted_text, utc=True, errors="coerce")
    if relevant["accepted_utc"].isna().any():
        raise FirstPublicClockError("A relevant acceptance time is unparseable")

    reports = relevant.loc[report_mask.loc[relevant.index]].copy()
    releases = relevant.loc[release_mask.loc[relevant.index]].copy()
    # Combined filings list one accession under every co-registrant CIK, so a
    # periodic report is identified by (cik, accession), never accession alone.
    if reports.duplicated(["cik", "accession_number"]).any():
        raise FirstPublicClockError("Periodic reports must be unique per (cik, accession)")
    reports["_report_key"] = reports["cik"].astype(str) + "|" + reports["accession_number"]
    reports["_period_end"] = pd.to_datetime(
        reports["report_date"], format="%Y-%m-%d", errors="coerce"
    )
    reports["_missing_period_end"] = reports["_period_end"].isna()

    reports = reports.sort_values(["accepted_utc", "accession_number"], kind="mergesort")
    reports["report_ns"] = _epoch_ns(reports["accepted_utc"])
    reports["previous_report_accepted_at"] = reports.groupby("cik", sort=False)[
        "accepted_utc"
    ].shift(1)
    # A report without a period end cannot be tied to a fiscal period; the
    # placeholder only keeps arithmetic defined and is overridden below.
    acceptance_local_day = (
        reports["accepted_utc"].dt.tz_convert(cfg.timezone).dt.tz_localize(None).dt.normalize()
    )
    period_boundary = (
        (reports["_period_end"].fillna(acceptance_local_day) + pd.Timedelta(days=1))
        .dt.tz_localize(cfg.timezone)
        .dt.tz_convert("UTC")
    )
    lead_ns = int(cfg.max_release_lead_days * _NANOSECONDS_PER_DAY)
    no_previous = pd.Timestamp("1900-01-01", tz="UTC")
    # Exclusive lower bound: a release must be strictly later than this value.
    lower_bounds = np.column_stack(
        [
            _epoch_ns(reports["previous_report_accepted_at"].fillna(no_previous)),
            _epoch_ns(period_boundary) - 1,
            reports["report_ns"].to_numpy(dtype=np.int64) - lead_ns - 1,
        ]
    )
    # Missing period end: the bound equals the report itself, so no release can
    # match and the conservative (later) periodic-report clock is kept.
    reports["lower_exclusive_ns"] = np.where(
        reports["_missing_period_end"].to_numpy(dtype=bool),
        reports["report_ns"].to_numpy(dtype=np.int64),
        lower_bounds.max(axis=1),
    )

    releases = releases.sort_values(["accepted_utc", "accession_number"], kind="mergesort")
    releases["release_ns"] = _epoch_ns(releases["accepted_utc"])
    releases["release_rank"] = releases.groupby("cik", sort=False).cumcount() + 1
    release_view = releases.loc[
        :,
        [
            "cik",
            "release_ns",
            "release_rank",
            "accepted_utc",
            "accession_number",
            "report_date",
            "item_list",
        ],
    ].rename(
        columns={
            "accepted_utc": "release_accepted_utc",
            "accession_number": "release_accession_number",
            "report_date": "release_event_date",
            "item_list": "release_item_list",
        }
    )

    first = pd.merge_asof(
        reports.sort_values("lower_exclusive_ns", kind="mergesort"),
        release_view,
        left_on="lower_exclusive_ns",
        right_on="release_ns",
        by="cik",
        direction="forward",
        allow_exact_matches=False,
    )
    rank_at_report = pd.merge_asof(
        reports.loc[:, ["cik", "_report_key", "report_ns"]].sort_values("report_ns"),
        release_view.loc[:, ["cik", "release_ns", "release_rank"]],
        left_on="report_ns",
        right_on="release_ns",
        by="cik",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_report_key")["release_rank"]
    rank_at_lower = pd.merge_asof(
        reports.loc[:, ["cik", "_report_key", "lower_exclusive_ns"]].sort_values(
            "lower_exclusive_ns"
        ),
        release_view.loc[:, ["cik", "release_ns", "release_rank"]],
        left_on="lower_exclusive_ns",
        right_on="release_ns",
        by="cik",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_report_key")["release_rank"]

    first = first.set_index("_report_key")
    matched = first["release_accepted_utc"].notna() & first["release_accepted_utc"].le(
        first["accepted_utc"]
    )
    candidate_count = (
        rank_at_report.reindex(first.index).fillna(0) - rank_at_lower.reindex(first.index).fillna(0)
    ).astype("int64")
    if (matched & candidate_count.lt(1)).any() or (~matched & candidate_count.gt(0)).any():
        raise FirstPublicClockError("Release match and candidate count disagree")

    result = pd.DataFrame(index=first.index)
    result["cik"] = first["cik"]
    result["accession_number"] = first["accession_number"]
    result["form"] = first["form"]
    result["period_end"] = first["report_date"]
    result["report_accepted_at"] = first["accepted_utc"]
    result["previous_report_accepted_at"] = first["previous_report_accepted_at"]
    release_at = first["release_accepted_utc"].where(matched)
    result["release_accession_number"] = first["release_accession_number"].where(matched)
    result["release_accepted_at"] = release_at
    result["release_event_date"] = first["release_event_date"].where(matched)
    result["release_items"] = (
        first["release_item_list"]
        .where(matched)
        .map(lambda items: ",".join(items) if isinstance(items, list) else None)
    )
    result["release_event_date_reliable"] = (
        first["release_item_list"]
        .where(matched)
        .map(lambda items: isinstance(items, list) and set(items) <= EARNINGS_COMPANION_ITEMS)
        .astype(bool)
    )
    result["release_candidate_count"] = candidate_count
    result["clock_source"] = np.select(
        [matched.to_numpy(dtype=bool), first["_missing_period_end"].to_numpy(dtype=bool)],
        ["item_2_02_8k", "periodic_report_missing_period_end"],
        default="periodic_report",
    )
    result["earliest_disclosure_at"] = release_at.where(matched, result["report_accepted_at"])
    result["disclosure_lead_days"] = (
        result["report_accepted_at"] - result["earliest_disclosure_at"]
    ).dt.total_seconds() / 86_400.0
    report_local = result["report_accepted_at"].dt.tz_convert(cfg.timezone).dt.strftime("%Y-%m-%d")
    release_local = release_at.dt.tz_convert(cfg.timezone).dt.strftime("%Y-%m-%d")
    event_date = result["release_event_date"]
    result["same_local_date"] = _masked_boolean(release_local.eq(report_local), matched)
    result["release_filed_after_event_date"] = _masked_boolean(
        release_local.gt(event_date.fillna("")), matched & event_date.notna()
    )
    result["first_public_certified"] = False
    result["clock_version"] = cfg.version
    result = result.reset_index(drop=True)
    return result.sort_values(
        ["cik", "report_accepted_at", "accession_number"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_first_public_clock(clocks: pd.DataFrame) -> dict[str, Any]:
    """Headline shares and lead quantiles overall and by periodic form."""

    def block(frame: pd.DataFrame) -> dict[str, Any]:
        released = frame.loc[frame["clock_source"].eq("item_2_02_8k")]
        lead = released["disclosure_lead_days"]
        count = len(released)

        def share(mask: pd.Series) -> float | None:
            return float(mask.mean()) if count else None

        def quantile(level: float) -> float | None:
            return float(lead.quantile(level)) if count else None

        return {
            "periodic_reports": len(frame),
            "missing_period_end": int(
                frame["clock_source"].eq("periodic_report_missing_period_end").sum()
            ),
            "with_item_2_02_release": count,
            "share_with_release": float(count / len(frame)) if len(frame) else None,
            "share_same_local_date": share(released["same_local_date"].astype(bool)),
            "share_lead_at_least_1_day": share(lead.ge(1.0)),
            "share_lead_at_least_3_days": share(lead.ge(3.0)),
            "median_lead_days": quantile(0.5),
            "p75_lead_days": quantile(0.75),
            "p90_lead_days": quantile(0.9),
            "share_multiple_candidates": share(released["release_candidate_count"].gt(1)),
            "share_release_filed_after_event_date": share(
                released["release_filed_after_event_date"].astype(bool)
            ),
            "share_event_date_reliable": share(released["release_event_date_reliable"]),
        }

    return {
        "all": block(clocks),
        "by_form": {str(form): block(group) for form, group in clocks.groupby("form")},
    }


def build_first_public_clock_audit(
    catalog_root: Path,
    *,
    output_root: Path = Path("data/processed/first_public_clock"),
    config: FirstPublicClockConfig | None = None,
) -> FirstPublicClockResult:
    """Persist a content-addressed clock table from one hash-verified catalog."""
    cfg = config or FirstPublicClockConfig()
    cfg.validate()
    catalog_manifest_sha256 = _sha256_file(catalog_root / "output_manifest.json")
    implementation_sha256 = _sha256_file(Path(__file__))
    payload = json.dumps(
        {
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "config": asdict(cfg),
            "implementation_sha256": implementation_sha256,
        },
        sort_keys=True,
    ).encode()
    dataset_version = f"{cfg.version}-{hashlib.sha256(payload).hexdigest()[:16]}"
    target = output_root / dataset_version
    if (target / "output_manifest.json").is_file():
        report: dict[str, Any] = json.loads(
            (target / "first_public_clock_report.json").read_text(encoding="utf-8")
        )
        return FirstPublicClockResult(target.resolve(), report)
    if target.exists():
        raise FirstPublicClockError(f"Incomplete first-public clock bundle exists: {target}")

    catalog = load_catalog_table(catalog_root, "filing_catalog.parquet")
    clocks = build_first_public_clock(catalog, cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    table_path = staging / "first_public_clock.parquet"
    clocks.to_parquet(table_path, index=False)
    report = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_root": catalog_root.as_posix(),
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "implementation_sha256": implementation_sha256,
        "config": asdict(cfg),
        "summary": summarize_first_public_clock(clocks),
        "first_public_certified": False,
        "prices_loaded": False,
        "returns_computed": False,
    }
    report_path = staging / "first_public_clock_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "artifacts": {
                    "first_public_clock.parquet": {
                        "rows": len(clocks),
                        "sha256": _sha256_file(table_path),
                    }
                },
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return FirstPublicClockResult(target.resolve(), report)


def _epoch_ns(values: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Nanoseconds since the epoch, independent of the parsed datetime resolution."""
    nanoseconds: np.ndarray[Any, np.dtype[np.int64]] = (
        values.dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64)
    )
    return nanoseconds


def _masked_boolean(values: pd.Series, valid: pd.Series) -> pd.Series:
    """Nullable booleans that are NA wherever the comparison is not defined."""
    result = pd.Series(values.to_numpy(dtype=bool), index=values.index, dtype="boolean")
    return result.mask(~valid.to_numpy(dtype=bool))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
