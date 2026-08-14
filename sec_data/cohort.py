"""Build an auditable SEC-available cohort from a Fortune Global 500 ranking.

The ranking is used only as a sampling frame.  Financial statement values come
from SEC EDGAR, and availability is assessed from original 10-Q filings with an
accession-level acceptance timestamp.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

from sec_data.models import SecIngestionResult

FORTUNE_DIRECTORY_URL = (
    "https://fortune.com/api/page/directory/companies/{page}/?rankingId=3&rankingYear={year}"
)
FORTUNE_RANKING_URL = "https://fortune.com/ranking/global500/"
FORTUNE_RANKING_COLUMNS = [
    "fortune_rank",
    "ranking_year",
    "company_name",
    "country",
    "fortune_ticker",
    "stock_quote",
    "fortune_revenue_usd_millions",
    "fortune_database_id",
    "fortune_company_slug",
    "fortune_company_url",
    "fortune_rank_source_url",
    "directory_source_url",
    "directory_snapshot_sha256",
    "company_snapshot_sha256",
    "fetched_at",
]
SEC_MAPPING_COLUMNS = [
    *FORTUNE_RANKING_COLUMNS,
    "sec_cik",
    "sec_ticker",
    "sec_title",
    "mapping_method",
    "mapping_score",
    "mapping_status",
]

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
GROSS_PROFIT_TAGS = ("GrossProfit",)
COST_OF_REVENUE_TAGS = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
)


class HttpGetter(Protocol):
    """Injectable HTTP subset used by the ranking downloader."""

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        """Return an HTTP response for ``url``."""
        ...


@dataclass(frozen=True)
class FortuneResponse:
    """One identified Fortune response and its acquisition timestamp."""

    url: str
    content: bytes
    fetched_at: datetime


class FortuneClient:
    """Small, throttled client for Fortune's public company directory pages."""

    def __init__(
        self,
        user_agent: str,
        *,
        request_interval_seconds: float = 0.20,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sender: HttpGetter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("Fortune User-Agent must not be empty")
        if request_interval_seconds < 0.10:
            raise ValueError("Fortune request interval must be at least 0.10 seconds")
        self.user_agent = user_agent.strip()
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sender = sender or httpx.Client(follow_redirects=True)
        self.sleep = sleep
        self.monotonic = monotonic
        self._last_request_started: float | None = None

    def get(self, url: str) -> FortuneResponse:
        """Fetch a public ranking resource with bounded retry and throttling."""
        last_error = "request did not start"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.sender.get(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                    },
                    timeout=self.timeout_seconds,
                )
            except httpx.RequestError as exc:
                last_error = str(exc)
            else:
                if 200 <= response.status_code < 300:
                    return FortuneResponse(
                        url=str(response.url),
                        content=response.content,
                        fetched_at=datetime.now(timezone.utc),
                    )
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            if attempt < self.max_retries:
                self.sleep(min(0.5 * (2**attempt), 8.0))
        raise RuntimeError(f"Fortune request failed for {url}: {last_error}")

    def _throttle(self) -> None:
        current = self.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (current - self._last_request_started)
            if remaining > 0.0:
                self.sleep(remaining)
                current = self.monotonic()
        self._last_request_started = current


def download_fortune_global500(
    year: int,
    *,
    raw_root: Path,
    interim_root: Path,
    client: FortuneClient,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Download and normalize one Fortune Global 500 membership snapshot.

    Public company pages supply the authoritative rank and ticker.  Extracted
    company-page records are cached by slug so an interrupted 500-page pass can
    resume without repeating completed requests.
    """
    if not 1995 <= year <= datetime.now(timezone.utc).year:
        raise ValueError("Ranking year is outside the supported range")
    raw_root.mkdir(parents=True, exist_ok=True)
    company_cache = interim_root / "company_pages"
    company_cache.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda _message: None)

    first_url = FORTUNE_DIRECTORY_URL.format(page=1, year=year)
    first_response = client.get(first_url)
    first_document = _decode_object(first_response.content, first_response.url)
    directory_documents = [(first_document, first_response)]
    pagination = first_document.get("pagination")
    if not isinstance(pagination, dict):
        raise ValueError("Fortune directory response has no pagination object")
    last_page = int(pagination.get("lastPage", 0))
    total = int(pagination.get("total", 0))
    if total != 500 or last_page <= 0:
        raise ValueError(f"Expected 500 Global 500 members, received {total}")
    for page in range(2, last_page + 1):
        response = client.get(FORTUNE_DIRECTORY_URL.format(page=page, year=year))
        directory_documents.append((_decode_object(response.content, response.url), response))
    emit(f"Fortune directory: {total} members across {last_page} pages")

    members: list[dict[str, Any]] = []
    for document, response in directory_documents:
        digest, _ = preserve_source_bytes(
            raw_root,
            kind="directory",
            content=response.content,
            suffix="json",
        )
        items = document.get("items")
        if not isinstance(items, list):
            raise ValueError("Fortune directory page has no items array")
        for item in items:
            if not isinstance(item, dict):
                continue
            members.append(
                {
                    "company_name": str(item.get("name") or ""),
                    "country": str(item.get("country") or ""),
                    "fortune_database_id": _optional_int(item.get("databaseId")),
                    "fortune_company_slug": str(item.get("slug") or ""),
                    "fortune_company_url": str(item.get("permalink") or ""),
                    "fortune_revenue_usd_millions": _parse_revenue_millions(item.get("revenues")),
                    "directory_source_url": response.url,
                    "directory_snapshot_sha256": digest,
                }
            )
    if len(members) != 500 or len({row["fortune_company_slug"] for row in members}) != 500:
        raise ValueError("Fortune directory did not yield 500 unique company slugs")

    rows: list[dict[str, Any]] = []
    for index, member in enumerate(members, start=1):
        slug = str(member["fortune_company_slug"])
        cache_path = company_cache / f"{slug}.json"
        if cache_path.exists():
            extracted = _decode_object(cache_path.read_bytes(), str(cache_path))
        else:
            company_url = str(member["fortune_company_url"])
            response = client.get(company_url)
            digest, _ = preserve_source_bytes(
                raw_root,
                kind=f"company-{slug}",
                content=response.content,
                suffix="html",
            )
            extracted = extract_fortune_company_page(
                response.content,
                year=year,
                source_url=response.url,
            )
            extracted["company_snapshot_sha256"] = digest
            extracted["fetched_at"] = response.fetched_at.isoformat()
            cache_path.write_text(
                json.dumps(extracted, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
        rows.append(
            {
                "fortune_rank": int(extracted["fortune_rank"]),
                "ranking_year": year,
                **member,
                "fortune_ticker": extracted.get("fortune_ticker"),
                "stock_quote": extracted.get("stock_quote"),
                "fortune_rank_source_url": extracted["fortune_rank_source_url"],
                "company_snapshot_sha256": extracted.get("company_snapshot_sha256"),
                "fetched_at": extracted.get("fetched_at"),
            }
        )
        if index % 25 == 0 or index == len(members):
            emit(f"Fortune company pages: {index}/{len(members)}")
    ranking = pd.DataFrame(rows).reindex(columns=FORTUNE_RANKING_COLUMNS)
    ranking = ranking.sort_values("fortune_rank").reset_index(drop=True)
    expected_ranks = list(range(1, 501))
    if ranking["fortune_rank"].astype(int).tolist() != expected_ranks:
        raise ValueError("Fortune company pages did not yield each rank from 1 through 500")
    return ranking


def extract_fortune_company_page(
    content: bytes,
    *,
    year: int,
    source_url: str,
) -> dict[str, Any]:
    """Extract the current Global 500 rank and market identifiers from a company page."""
    text = content.decode("utf-8")
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Fortune company page has no __NEXT_DATA__: {source_url}")
    document = _decode_object(html.unescape(match.group(1)).encode(), source_url)
    props = document.get("props")
    page_props = props.get("pageProps") if isinstance(props, dict) else None
    company = page_props.get("company") if isinstance(page_props, dict) else None
    if not isinstance(company, dict):
        raise ValueError(f"Fortune company page has no company object: {source_url}")
    ranking_lists = company.get("companyRankingLists")
    if not isinstance(ranking_lists, list):
        raise ValueError(f"Fortune company page has no ranking history: {source_url}")
    rank: int | None = None
    rank_url = FORTUNE_RANKING_URL
    for ranking in ranking_lists:
        if not isinstance(ranking, dict):
            continue
        if ranking.get("slug") == "global500" and _optional_int(ranking.get("year")) == year:
            rank = _optional_int(ranking.get("rank"))
            rank_url = str(ranking.get("permalink") or rank_url)
            break
    if rank is None:
        raise ValueError(f"No {year} Global 500 rank on company page: {source_url}")
    company_info = company.get("companyInfo")
    info = company_info if isinstance(company_info, dict) else {}
    return {
        "fortune_rank": rank,
        "fortune_ticker": _optional_text(info.get("Ticker")),
        "stock_quote": _optional_text(info.get("StockQuote")),
        "fortune_rank_source_url": rank_url,
    }


def map_fortune_members_to_sec(
    ranking: pd.DataFrame,
    ticker_document: Mapping[str, Any],
) -> pd.DataFrame:
    """Map Fortune members to current SEC CIKs with conservative audit labels."""
    sec_records = _sec_ticker_records(ticker_document)
    by_ticker: dict[str, list[dict[str, str]]] = {}
    for record in sec_records:
        by_ticker.setdefault(record["ticker"], []).append(record)

    mapped_rows: list[dict[str, Any]] = []
    for _, fortune in ranking.iterrows():
        fortune_dict: dict[str, Any] = {str(key): value for key, value in fortune.to_dict().items()}
        company_name = str(fortune["company_name"])
        country = str(fortune["country"])
        fortune_ticker = _clean_ticker(fortune.get("fortune_ticker"))
        ticker_candidates = by_ticker.get(fortune_ticker, []) if fortune_ticker else []
        best, method, score = _best_sec_match(
            company_name,
            country=country,
            ticker_candidates=ticker_candidates,
            all_records=sec_records,
        )
        status = "unmapped"
        if best is not None:
            if method == "fortune_ticker" or (
                method == "exact_name" and country in {"U.S.", "United States"}
            ):
                status = "accepted"
            else:
                status = "manual_review"
        mapped_rows.append(
            {
                **fortune_dict,
                "sec_cik": best["cik"] if best is not None else None,
                "sec_ticker": best["ticker"] if best is not None else None,
                "sec_title": best["title"] if best is not None else None,
                "mapping_method": method,
                "mapping_score": score,
                "mapping_status": status,
            }
        )
    return pd.DataFrame(mapped_rows).reindex(columns=SEC_MAPPING_COLUMNS)


def assess_sec_research_completeness(
    result: SecIngestionResult,
    *,
    start_year: int = 2016,
    end_year: int = 2025,
) -> dict[str, Any]:
    """Measure whether original quarterly SEC data can support gross-margin research.

    A ready quarter requires an original 10-Q, a 70--110 day current-period
    revenue fact, and either direct gross profit or a compatible cost-of-revenue
    fact.  Comparative facts repeated in later filings are excluded by matching
    each fact's period end to that filing's report date.
    """
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    filings = result.filings.copy()
    facts = result.facts.copy()
    if filings.empty:
        original_10q = filings
    else:
        report_dates = pd.to_datetime(filings["report_date"], errors="coerce")
        original_10q = filings.loc[
            (filings["form"] == "10-Q") & report_dates.dt.year.between(start_year, end_year)
        ].copy()
    current_quarter_facts = _current_quarter_facts(
        facts,
        original_10q,
        start_year=start_year,
        end_year=end_year,
    )
    revenue_events = _event_keys_for_tags(current_quarter_facts, REVENUE_TAGS)
    gross_profit_events = _event_keys_for_tags(current_quarter_facts, GROSS_PROFIT_TAGS)
    cost_events = _event_keys_for_tags(current_quarter_facts, COST_OF_REVENUE_TAGS)
    direct_events = revenue_events & gross_profit_events
    derived_events = (revenue_events & cost_events) - direct_events
    ready_events = direct_events | derived_events
    ready_period_ends = sorted({event[2] for event in ready_events})
    ready_years = sorted({pd.Timestamp(period_end).year for period_end in ready_period_ends})
    ten_q_count = int(original_10q["accession_number"].nunique()) if not original_10q.empty else 0
    revenue_count = len(revenue_events)
    raw_complete = ten_q_count >= 20 and revenue_count >= 20
    gm_ready = raw_complete and len(ready_events) >= 16 and len(ready_years) >= 6
    changed_amendment_facts = 0
    if "comparison_status" in result.amendment_comparisons.columns:
        changed_amendment_facts = int(
            result.amendment_comparisons["comparison_status"].eq("changed").sum()
        )
    return {
        "sec_cik": result.cik,
        "sec_ticker": result.ticker,
        "sec_company_name": result.company_name,
        "research_start_year": start_year,
        "research_end_year": end_year,
        "original_10q_filings": ten_q_count,
        "quarterly_revenue_events": revenue_count,
        "direct_gross_profit_events": len(direct_events),
        "derived_gross_profit_events": len(derived_events),
        "gross_margin_ready_events": len(ready_events),
        "gross_margin_ready_years": len(ready_years),
        "first_gross_margin_period_end": ready_period_ends[0] if ready_period_ends else None,
        "last_gross_margin_period_end": ready_period_ends[-1] if ready_period_ends else None,
        "amendment_comparisons": len(result.amendment_comparisons),
        "changed_amendment_facts": changed_amendment_facts,
        "rejected_source_records": len(result.rejections),
        "sec_raw_complete": raw_complete,
        "gross_margin_research_ready": gm_ready,
        "dataset_version": result.metadata.get("dataset_version"),
        "normalized_directory": result.metadata.get("normalized_directory"),
    }


def preserve_source_bytes(
    root: Path,
    *,
    kind: str,
    content: bytes,
    suffix: str,
) -> tuple[str, Path]:
    """Write immutable source bytes under a content-addressed filename."""
    digest = hashlib.sha256(content).hexdigest()
    safe_kind = re.sub(r"[^A-Za-z0-9._-]+", "-", kind).strip("-")
    target = root / digest[:2] / f"{safe_kind}-{digest}.{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise RuntimeError(f"Content hash collision at {target}")
    else:
        with target.open("xb") as stream:
            stream.write(content)
    return digest, target.resolve()


def _decode_object(content: bytes, source_url: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON from {source_url}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from {source_url}")
    return value


def _parse_revenue_millions(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digits = re.sub(r"[^0-9-]", "", value)
    return int(digits) if digits and digits != "-" else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _clean_ticker(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^A-Z0-9.-]", "", value.upper().strip())


def _normalize_company_name(value: str) -> str:
    normalized = value.upper().replace("&", " AND ")
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    discard = {
        "THE",
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "COS",
        "GROUP",
        "HOLDING",
        "HOLDINGS",
        "PLC",
        "LTD",
        "LIMITED",
        "LLC",
        "LP",
        "SA",
        "AG",
        "SE",
        "NV",
        "NA",
        "COM",
    }
    tokens = [token for token in normalized.split() if token not in discard]
    return " ".join(tokens)


def _sec_ticker_records(document: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in document.values():
        if not isinstance(value, dict):
            continue
        ticker = _clean_ticker(value.get("ticker"))
        title = str(value.get("title") or "").strip()
        cik_value = _optional_int(value.get("cik_str"))
        if not ticker or not title or cik_value is None:
            continue
        cik = str(cik_value).zfill(10)
        key = (cik, ticker)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "cik": cik,
                "ticker": ticker,
                "title": title,
                "normalized_title": _normalize_company_name(title),
            }
        )
    return records


def _best_sec_match(
    company_name: str,
    *,
    country: str,
    ticker_candidates: list[dict[str, str]],
    all_records: list[dict[str, str]],
) -> tuple[dict[str, str] | None, str, float]:
    target = _normalize_company_name(company_name)
    if ticker_candidates:
        scored_tickers = sorted(
            (
                (_name_score(target, candidate["normalized_title"]), candidate)
                for candidate in ticker_candidates
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        ticker_score, ticker_best = scored_tickers[0]
        if country in {"U.S.", "United States"} or ticker_score >= 0.72:
            return ticker_best, "fortune_ticker", ticker_score

    exact = [record for record in all_records if record["normalized_title"] == target]
    if exact:
        return _preferred_security(exact), "exact_name", 1.0

    target_first = target.split()[0] if target else ""
    blocked = [
        record
        for record in all_records
        if target_first
        and (
            record["normalized_title"].startswith(target_first)
            or target.startswith(record["normalized_title"].split()[0])
        )
    ]
    if not blocked:
        return None, "unmapped", 0.0
    scored = sorted(
        ((_name_score(target, record["normalized_title"]), record) for record in blocked),
        key=lambda pair: pair[0],
        reverse=True,
    )
    score, _best = scored[0]
    if score < 0.80:
        return None, "unmapped", score
    same_cik = [record for candidate_score, record in scored if candidate_score == score]
    return _preferred_security(same_cik), "fuzzy_name", score


def _name_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    sequence_score = SequenceMatcher(None, left, right).ratio()
    containment = min(len(left), len(right)) / max(len(left), len(right))
    if left in right or right in left:
        return max(sequence_score, 0.86 + 0.14 * containment)
    return 0.55 * sequence_score + 0.45 * token_score


def _preferred_security(records: list[dict[str, str]]) -> dict[str, str]:
    return sorted(
        records,
        key=lambda record: (
            "." in record["ticker"] or "-" in record["ticker"],
            len(record["ticker"]),
            record["ticker"],
        ),
    )[0]


def _current_quarter_facts(
    facts: pd.DataFrame,
    original_10q: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if facts.empty or original_10q.empty:
        return facts.iloc[0:0].copy()
    report_end_by_accession = (
        original_10q.assign(
            report_date=pd.to_datetime(original_10q["report_date"], errors="coerce")
        )
        .set_index("accession_number")["report_date"]
        .to_dict()
    )
    filtered = facts.loc[
        (facts["base_form"] == "10-Q")
        & ~facts["is_amendment"].astype(bool)
        & facts["duration_days"].between(70, 110, inclusive="both")
    ].copy()
    if filtered.empty:
        return filtered
    filtered["period_end"] = pd.to_datetime(filtered["period_end"], errors="coerce")
    filtered["filing_report_end"] = filtered["accession_number"].map(report_end_by_accession)
    filtered = filtered.loc[
        filtered["period_end"].dt.year.between(start_year, end_year)
        & (filtered["period_end"] == filtered["filing_report_end"])
    ]
    return pd.DataFrame(filtered)


def _event_keys_for_tags(
    facts: pd.DataFrame,
    tags: tuple[str, ...],
) -> set[tuple[str, str, str]]:
    if facts.empty:
        return set()
    selected = facts.loc[facts["tag"].isin(tags)]
    accessions = selected["accession_number"].astype(str).tolist()
    units = selected["unit"].astype(str).tolist()
    period_ends = pd.to_datetime(selected["period_end"]).dt.strftime("%Y-%m-%d").tolist()
    return set(zip(accessions, units, period_ends, strict=True))
