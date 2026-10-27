"""Forward earliest-disclosure clock for the registered prospective events.

The v3 forward pipeline times every registered company-quarter at the SEC
acceptance of its 10-Q/10-K.  Most issuers disclose the same results earlier
in an earnings release furnished on Form 8-K under Item 2.02
(docs/first_public_clock_v1.md).  This module applies that rule *forward*, to
the SEC submissions snapshots the filing monitor already preserves, so that a
registered event receives its earliest SEC-evidenced disclosure as soon as the
8-K appears -- weeks before the periodic report exists.

For each registered (issuer, fiscal period end):

- candidate releases are original 8-Ks whose item list contains Item 2.02,
  accepted strictly after the issuer's previous original periodic report, on
  or after the New York day following the period end, within
  ``max_release_days_after_period_end``, and no later than this period's
  periodic report when that report has been observed;
- the earliest qualifying candidate is the release clock; the matching
  original 10-Q/10-K, when observed, is the periodic clock;
- ``earliest_disclosure_at`` is the earlier of the two.

No new request is made: only preserved, hash-verified snapshot bytes are read.
Both clocks remain SEC proxies.  An 8-K may be furnished up to four business
days after the announcement, so the issuer-reported event date is kept at date
precision and ``first_public_certified`` is always false.  Nothing here loads
an actual value, a price, or a return.
"""

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

FORWARD_DISCLOSURE_CLOCK_VERSION: Final[str] = "forward-disclosure-clock-v1"
EARNINGS_RELEASE_ITEM: Final[str] = "2.02"
EARNINGS_COMPANION_ITEMS: Final[frozenset[str]] = frozenset({"2.02", "7.01", "9.01"})
PERIODIC_FORMS: Final[frozenset[str]] = frozenset({"10-Q", "10-K"})
PERIOD_MATCH_TOLERANCE_DAYS: Final[int] = 7
REGISTRY_COLUMNS: Final[tuple[str, ...]] = (
    "prospective_event_id",
    "cik",
    "ticker",
    "target_period_end",
    "accepted_at",
)
FILING_COLUMNS: Final[tuple[str, ...]] = (
    "cik",
    "ticker",
    "accession_number",
    "form",
    "base_form",
    "is_amendment",
    "items",
    "report_date",
    "accepted_at",
    "observed_at",
    "source_sha256",
)
_ITEM_PATTERN: Final[str] = r"\d+\.\d+"


class ForwardDisclosureClockError(FilingCatalogError):
    """Raised when preserved snapshots or the registry cannot support the clock rule."""


@dataclass(frozen=True)
class ForwardDisclosureClockConfig:
    """Explicit forward matching assumptions; changing one requires a new version."""

    max_release_days_after_period_end: int = 120
    timezone: str = "America/New_York"
    version: str = FORWARD_DISCLOSURE_CLOCK_VERSION

    def validate(self) -> None:
        """Reject windows that could attach a release to the wrong fiscal period."""
        if not 1 <= self.max_release_days_after_period_end <= 180:
            raise ForwardDisclosureClockError(
                "max_release_days_after_period_end must lie in [1, 180]"
            )
        if not self.timezone.strip():
            raise ForwardDisclosureClockError("timezone cannot be blank")
        if not self.version.strip():
            raise ForwardDisclosureClockError("version cannot be blank")


@dataclass(frozen=True)
class ForwardDisclosureClockResult:
    """Immutable bundle location and report."""

    output_root: Path
    report: dict[str, Any]


def parse_disclosure_filings(
    content: bytes,
    *,
    cik: str,
    ticker: str,
    observed_at: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Extract original Item 2.02 8-Ks and original 10-Q/10-Ks from one snapshot."""
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardDisclosureClockError("SEC submissions snapshot is not JSON") from error
    if not isinstance(document, dict) or _cik(document.get("cik")) != _cik(cik):
        raise ForwardDisclosureClockError("SEC submissions snapshot CIK does not match")
    filings = document.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict) or not isinstance(recent.get("accessionNumber"), list):
        raise ForwardDisclosureClockError("SEC submissions snapshot lacks recent filings")
    observed = _utc_timestamp(observed_at, "observed_at")
    rows: list[dict[str, Any]] = []
    for index, accession in enumerate(recent["accessionNumber"]):
        form = str(_value(recent, "form", index) or "").strip().upper()
        base_form = form.removesuffix("/A")
        items = re.findall(_ITEM_PATTERN, str(_value(recent, "items", index) or ""))
        if base_form in PERIODIC_FORMS:
            pass
        elif base_form == "8-K" and EARNINGS_RELEASE_ITEM in items:
            pass
        else:
            continue
        accepted = _optional_utc(_value(recent, "acceptanceDateTime", index))
        if accepted is None:
            continue
        rows.append(
            {
                "cik": _cik(cik),
                "ticker": str(ticker).strip().upper(),
                "accession_number": str(accession).strip(),
                "form": form,
                "base_form": base_form,
                "is_amendment": form.endswith("/A"),
                "items": ",".join(items),
                "report_date": _optional_date(_value(recent, "reportDate", index)),
                "accepted_at": accepted,
                "observed_at": observed,
                "source_sha256": source_sha256,
            }
        )
    return rows


def assign_forward_disclosure_clocks(
    registry: pd.DataFrame,
    filings: pd.DataFrame,
    config: ForwardDisclosureClockConfig | None = None,
) -> pd.DataFrame:
    """Give every registered event its earliest SEC-evidenced disclosure, if any.

    Returns one row per ``prospective_event_id``.  ``clock_source`` is
    ``item_2_02_8k``, ``periodic_report``, or ``none_observed``.
    ``disclosure_first_observed_at`` is the fetch time of the earliest preserved
    snapshot that contained the clock filing: the lab's own knowledge time,
    which is never earlier than SEC acceptance.
    """
    cfg = config or ForwardDisclosureClockConfig()
    cfg.validate()
    _require_columns(registry, REGISTRY_COLUMNS, "registry")
    if registry.empty or registry["prospective_event_id"].duplicated().any():
        raise ForwardDisclosureClockError("The registry is empty or has duplicate event IDs")
    events = registry.loc[:, list(REGISTRY_COLUMNS)].copy()
    events["cik"] = events["cik"].map(_cik)
    events["target_period_end"] = pd.to_datetime(
        events["target_period_end"], utc=True, errors="coerce"
    )
    if events["target_period_end"].isna().any():
        raise ForwardDisclosureClockError("Every registered event needs a target period end")
    events["accepted_at"] = pd.to_datetime(events["accepted_at"], utc=True, errors="coerce")

    frame = _prepare_filings(filings)
    tolerance = pd.Timedelta(days=PERIOD_MATCH_TOLERANCE_DAYS)
    rows: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        period_end = pd.Timestamp(event["target_period_end"]).tz_convert(None).normalize()
        issuer = frame.loc[frame["cik"].eq(event["cik"])]
        periodic = issuer.loc[
            issuer["base_form"].isin(PERIODIC_FORMS)
            & ~issuer["is_amendment"]
            & issuer["report_date"].notna()
        ]
        previous_accepted = pd.NaT
        periodic_row = None
        if not periodic.empty:
            previous = periodic.loc[periodic["report_date"].lt(period_end - tolerance)]
            if not previous.empty:
                previous_accepted = previous["accepted_at"].max()
            current = periodic.loc[(periodic["report_date"] - period_end).abs().le(tolerance)]
            current = current.sort_values(["accepted_at", "accession_number"], kind="mergesort")
            periodic_row = current.iloc[0] if not current.empty else None

        period_boundary = (
            (period_end + pd.Timedelta(days=1)).tz_localize(cfg.timezone).tz_convert("UTC")
        )
        lead_limit = (
            (period_end + pd.Timedelta(days=cfg.max_release_days_after_period_end + 1))
            .tz_localize(cfg.timezone)
            .tz_convert("UTC")
        )
        releases = issuer.loc[
            issuer["base_form"].eq("8-K")
            & ~issuer["is_amendment"]
            & issuer["accepted_at"].ge(period_boundary)
            & issuer["accepted_at"].lt(lead_limit)
        ]
        if pd.notna(previous_accepted):
            releases = releases.loc[releases["accepted_at"].gt(previous_accepted)]
        if periodic_row is not None:
            releases = releases.loc[releases["accepted_at"].le(periodic_row["accepted_at"])]
        releases = releases.sort_values(["accepted_at", "accession_number"], kind="mergesort")
        release_row = releases.iloc[0] if not releases.empty else None

        release_at = release_row["accepted_at"] if release_row is not None else pd.NaT
        periodic_at = periodic_row["accepted_at"] if periodic_row is not None else pd.NaT
        if release_row is not None:
            source, clock_row = "item_2_02_8k", release_row
        elif periodic_row is not None:
            source, clock_row = "periodic_report", periodic_row
        else:
            source, clock_row = "none_observed", None
        earliest = clock_row["accepted_at"] if clock_row is not None else pd.NaT
        release_items = (
            set(str(release_row["items"]).split(",")) - {""} if release_row is not None else set()
        )
        event_date = release_row["report_date"] if release_row is not None else pd.NaT
        release_local_date = (
            pd.Timestamp(release_at).tz_convert(cfg.timezone).normalize().tz_localize(None)
            if pd.notna(release_at)
            else pd.NaT
        )
        rows.append(
            {
                "prospective_event_id": event["prospective_event_id"],
                "cik": event["cik"],
                "ticker": str(event["ticker"]).strip().upper(),
                "target_period_end": event["target_period_end"],
                "previous_periodic_report_accepted_at": previous_accepted,
                "release_accession_number": (
                    release_row["accession_number"] if release_row is not None else None
                ),
                "release_accepted_at": release_at,
                "release_event_date": event_date,
                "release_items": ",".join(sorted(release_items)) if release_items else None,
                "release_event_date_reliable": bool(
                    release_items and release_items <= EARNINGS_COMPANION_ITEMS
                ),
                "release_filed_after_event_date": (
                    bool(release_local_date > event_date)
                    if pd.notna(release_local_date) and pd.notna(event_date)
                    else None
                ),
                "release_candidate_count": len(releases),
                "periodic_accession_number": (
                    periodic_row["accession_number"] if periodic_row is not None else None
                ),
                "periodic_form": periodic_row["form"] if periodic_row is not None else None,
                "periodic_accepted_at": periodic_at,
                "registry_accepted_at": event["accepted_at"],
                "periodic_matches_registry": bool(
                    (pd.isna(periodic_at) and pd.isna(event["accepted_at"]))
                    or (
                        pd.notna(periodic_at)
                        and pd.notna(event["accepted_at"])
                        and periodic_at == event["accepted_at"]
                    )
                ),
                "clock_source": source,
                "earliest_disclosure_at": earliest,
                "disclosure_first_observed_at": (
                    clock_row["observed_at"] if clock_row is not None else pd.NaT
                ),
                "release_lead_days_over_periodic": (
                    (periodic_at - release_at).total_seconds() / 86_400.0
                    if pd.notna(release_at) and pd.notna(periodic_at)
                    else None
                ),
                "first_public_certified": False,
                "clock_version": cfg.version,
            }
        )
    result = pd.DataFrame(rows)
    for column in (
        "previous_periodic_report_accepted_at",
        "release_accepted_at",
        "periodic_accepted_at",
        "registry_accepted_at",
        "earliest_disclosure_at",
        "disclosure_first_observed_at",
    ):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["release_event_date"] = pd.to_datetime(result["release_event_date"])
    return result.sort_values(
        ["target_period_end", "ticker", "prospective_event_id"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_forward_disclosure_clocks(clocks: pd.DataFrame) -> dict[str, Any]:
    """Headline counts; nothing here implies an outcome or an admission."""
    released = clocks.loc[clocks["clock_source"].eq("item_2_02_8k")]
    lead = pd.to_numeric(released["release_lead_days_over_periodic"], errors="coerce")
    by_period: dict[str, dict[str, int]] = {}
    for period, group in clocks.groupby(clocks["target_period_end"].dt.strftime("%Y-%m-%d")):
        by_period[str(period)] = {
            str(k): int(v) for k, v in group["clock_source"].value_counts().items()
        }
    return {
        "events": len(clocks),
        "issuers": int(clocks["cik"].nunique()),
        "clock_source_counts": {
            str(k): int(v) for k, v in clocks["clock_source"].value_counts().items()
        },
        "disclosed_events": int(clocks["earliest_disclosure_at"].notna().sum()),
        "release_before_observed_periodic": int(lead.gt(0).sum()),
        "release_lead_at_least_3_days": int(lead.ge(3.0).sum()),
        "multiple_release_candidates": int(released["release_candidate_count"].gt(1).sum()),
        "release_filed_after_event_date": int(
            released["release_filed_after_event_date"].eq(True).sum()
        ),
        "periodic_registry_mismatches": int((~clocks["periodic_matches_registry"]).sum()),
        "by_target_period_end": by_period,
        "first_public_certified": False,
        "actual_values_loaded": False,
        "return_values_loaded": False,
    }


def build_forward_disclosure_clock(
    readiness_root: Path,
    monitor_audit_root: Path,
    *,
    output_root: Path = Path("data/processed/forward_disclosure_clock"),
    config: ForwardDisclosureClockConfig | None = None,
) -> ForwardDisclosureClockResult:
    """Persist a content-addressed clock table from preserved snapshot bytes only."""
    cfg = config or ForwardDisclosureClockConfig()
    cfg.validate()
    input_hashes = {
        "readiness": _sha256_file(readiness_root / "output_manifest.json"),
        "filing_monitor_audit": _sha256_file(monitor_audit_root / "output_manifest.json"),
    }
    implementation_sha256 = _sha256_file(Path(__file__))
    payload = json.dumps(
        {
            "input_manifest_sha256": input_hashes,
            "implementation_sha256": implementation_sha256,
            "config": asdict(cfg),
        },
        sort_keys=True,
    ).encode()
    dataset_version = f"{cfg.version}-{hashlib.sha256(payload).hexdigest()[:16]}"
    target = output_root / dataset_version
    report_name = "forward_disclosure_clock_report.json"
    if (target / "output_manifest.json").is_file():
        stored: dict[str, Any] = json.loads((target / report_name).read_text(encoding="utf-8"))
        return ForwardDisclosureClockResult(target.resolve(), stored)
    if target.exists():
        raise ForwardDisclosureClockError(f"Incomplete forward disclosure clock bundle: {target}")

    registry = load_catalog_table(
        readiness_root, "prospective_market_innovation_event_readiness.parquet"
    )
    manifests = load_catalog_table(
        monitor_audit_root, "forward_filing_submission_snapshots.parquet"
    )
    _require_columns(manifests, ("cik", "ticker", "fetched_at", "sha256", "path"), "snapshots")
    rows: list[dict[str, Any]] = []
    for manifest in manifests.sort_values(["fetched_at", "cik", "sha256"]).itertuples(index=False):
        raw_path = Path(str(manifest.path))
        if not raw_path.is_file() or _sha256_file(raw_path) != str(manifest.sha256):
            raise ForwardDisclosureClockError(f"Invalid preserved SEC snapshot: {raw_path}")
        rows.extend(
            parse_disclosure_filings(
                raw_path.read_bytes(),
                cik=str(manifest.cik),
                ticker=str(manifest.ticker),
                observed_at=str(manifest.fetched_at),
                source_sha256=str(manifest.sha256),
            )
        )
    filings = pd.DataFrame(rows, columns=list(FILING_COLUMNS))
    clocks = assign_forward_disclosure_clocks(registry, filings, cfg)

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_version}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    for filename, frame in {
        "forward_disclosure_clocks.parquet": clocks,
        "forward_disclosure_filings.parquet": _prepare_filings(filings),
    }.items():
        path = staging / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"rows": len(frame), "sha256": _sha256_file(path)}
    report = {
        "contract_version": cfg.version,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "implementation_sha256": implementation_sha256,
        "config": asdict(cfg),
        "snapshots_read": len(manifests),
        "summary": summarize_forward_disclosure_clocks(clocks),
        "network_requests_made": 0,
        "artifacts": artifacts,
    }
    report_path = staging / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "output_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "input_manifest_sha256": input_hashes,
                "artifacts": artifacts,
                "report_sha256": _sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    staging.replace(target)
    return ForwardDisclosureClockResult(target.resolve(), report)


def _prepare_filings(filings: pd.DataFrame) -> pd.DataFrame:
    if filings.empty:
        # Typed empty frame so date comparisons stay defined for issuers without filings.
        return pd.DataFrame(
            {
                **{column: pd.Series(dtype="object") for column in FILING_COLUMNS},
                "is_amendment": pd.Series(dtype=bool),
                "report_date": pd.Series(dtype="datetime64[ns]"),
                "accepted_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "observed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        ).loc[:, list(FILING_COLUMNS)]
    _require_columns(filings, FILING_COLUMNS, "filings")
    frame = filings.loc[:, list(FILING_COLUMNS)].copy()
    frame["cik"] = frame["cik"].map(_cik)
    frame["is_amendment"] = frame["is_amendment"].astype(bool)
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True, errors="coerce")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    if frame["accepted_at"].isna().any() or frame["observed_at"].isna().any():
        raise ForwardDisclosureClockError("A filing lacks an acceptance or observation time")
    if (frame["observed_at"] < frame["accepted_at"]).any():
        raise ForwardDisclosureClockError("A filing was observed before SEC acceptance")
    # Keep the first time each filing was observed in the supplied snapshots.
    return (
        frame.sort_values(["cik", "accession_number", "observed_at"], kind="mergesort")
        .drop_duplicates(["cik", "accession_number"], keep="first")
        .reset_index(drop=True)
    )


def _value(document: dict[str, Any], key: str, index: int) -> Any:
    values = document.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def _cik(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text.isdigit():
        raise ForwardDisclosureClockError(f"Invalid CIK: {text!r}")
    return text.zfill(10)


def _utc_timestamp(value: Any, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ForwardDisclosureClockError(f"{label} must be an explicit-offset timestamp")
    return stamp.tz_convert("UTC")


def _optional_utc(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    stamp = pd.Timestamp(str(value))
    if pd.isna(stamp) or stamp.tzinfo is None:
        return None
    return stamp.tz_convert("UTC")


def _optional_date(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    stamp = pd.to_datetime(str(value), errors="coerce")
    return None if pd.isna(stamp) else pd.Timestamp(stamp).normalize()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ForwardDisclosureClockError(f"{label} lack columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
