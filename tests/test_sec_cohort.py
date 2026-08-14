"""Offline ranking, mapping, and point-in-time cohort tests."""

import json
from datetime import datetime, timezone

import pandas as pd

from sec_data.cohort import (
    FORTUNE_RANKING_COLUMNS,
    assess_sec_research_completeness,
    extract_fortune_company_page,
    map_fortune_members_to_sec,
)
from sec_data.models import SecIngestionResult


def test_extract_fortune_company_page_reads_rank_and_ticker() -> None:
    next_data = {
        "props": {
            "pageProps": {
                "company": {
                    "companyInfo": {"Ticker": "ABC", "StockQuote": "NYSE"},
                    "companyRankingLists": [
                        {
                            "slug": "global500",
                            "year": 2026,
                            "rank": 42,
                            "permalink": "https://fortune.test/global500",
                        }
                    ],
                }
            }
        }
    }
    html = (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(next_data)
        + "</script></html>"
    ).encode()

    record = extract_fortune_company_page(html, year=2026, source_url="fixture://company")

    assert record == {
        "fortune_rank": 42,
        "fortune_ticker": "ABC",
        "stock_quote": "NYSE",
        "fortune_rank_source_url": "https://fortune.test/global500",
    }


def test_sec_mapping_prefers_fortune_ticker_and_exact_name() -> None:
    ranking = pd.DataFrame(
        [
            _ranking_row(1, "Amazon", "AMZN"),
            _ranking_row(2, "Example Group", None),
        ],
        columns=FORTUNE_RANKING_COLUMNS,
    )
    ticker_document = {
        "0": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
        "1": {"cik_str": 7, "ticker": "EXM", "title": "EXAMPLE GROUP, INC."},
    }

    mapped = map_fortune_members_to_sec(ranking, ticker_document)

    assert mapped["mapping_status"].tolist() == ["accepted", "accepted"]
    assert mapped["mapping_method"].tolist() == ["fortune_ticker", "exact_name"]
    assert mapped["sec_cik"].tolist() == ["0001018724", "0000000007"]


def test_non_us_name_only_match_requires_manual_review() -> None:
    ranking = pd.DataFrame(
        [_ranking_row(1, "QNB Group", None) | {"country": "Qatar"}],
        columns=FORTUNE_RANKING_COLUMNS,
    )
    ticker_document = {
        "0": {"cik_str": 1, "ticker": "QNBC", "title": "QNB CORP."},
    }

    mapped = map_fortune_members_to_sec(ranking, ticker_document)

    assert mapped.loc[0, "mapping_method"] == "exact_name"
    assert mapped.loc[0, "mapping_status"] == "manual_review"


def test_completeness_uses_only_current_original_quarter_facts() -> None:
    accessions = [f"0000000001-20-{index:06d}" for index in range(1, 25)]
    report_dates = pd.date_range("2016-03-31", periods=24, freq="QE-DEC")
    filings = pd.DataFrame(
        {
            "accession_number": accessions,
            "form": ["10-Q"] * 24,
            "report_date": report_dates,
        }
    )
    fact_rows = []
    for accession, period_end in zip(accessions, report_dates, strict=True):
        start = period_end - pd.Timedelta(days=89)
        for tag, value in (("Revenues", 100.0), ("GrossProfit", 40.0)):
            fact_rows.append(
                {
                    "accession_number": accession,
                    "base_form": "10-Q",
                    "is_amendment": False,
                    "duration_days": 89,
                    "period_start": start,
                    "period_end": period_end,
                    "tag": tag,
                    "unit": "USD",
                    "value_numeric": value,
                }
            )
    facts = pd.DataFrame(fact_rows)
    result = SecIngestionResult(
        cik="0000000001",
        ticker="ABC",
        company_name="ABC Corp",
        filings=filings,
        facts=facts,
        amendment_comparisons=pd.DataFrame(),
        rejections=pd.DataFrame(),
        snapshots=(),
        metadata={"dataset_version": "v1", "normalized_directory": "fixture"},
    )

    assessment = assess_sec_research_completeness(result)

    assert assessment["original_10q_filings"] == 24
    assert assessment["gross_margin_ready_events"] == 24
    assert assessment["gross_margin_research_ready"] is True


def _ranking_row(rank: int, name: str, ticker: str | None) -> dict[str, object]:
    return {
        "fortune_rank": rank,
        "ranking_year": 2026,
        "company_name": name,
        "country": "U.S.",
        "fortune_ticker": ticker,
        "stock_quote": "NASDAQ",
        "fortune_revenue_usd_millions": 100,
        "fortune_database_id": rank,
        "fortune_company_slug": name.lower().replace(" ", "-"),
        "fortune_company_url": f"https://fortune.test/{rank}",
        "fortune_rank_source_url": "https://fortune.test/global500",
        "directory_source_url": "https://fortune.test/directory",
        "directory_snapshot_sha256": "a" * 64,
        "company_snapshot_sha256": "b" * 64,
        "fetched_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    }
