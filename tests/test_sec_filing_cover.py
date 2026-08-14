"""Tests for filing-time identity, security, and SIC evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from sec_data.client import SecClient
from sec_data.filing_cover import (
    build_filing_cover_history,
    build_filing_cover_target_plan,
    collect_filing_cover_artifacts,
    load_filing_cover_targets,
    parse_cover_facts,
    parse_filing_index,
)


class _Sender:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del url, headers, timeout
        return self.responses.pop(0)


def test_cover_parsers_extract_context_security_and_filing_sic() -> None:
    facts = parse_cover_facts(_primary_document())
    index = parse_filing_index(
        _index_document(),
        index_url=(
            "https://www.sec.gov/Archives/edgar/data/1/"
            "000000000120000001/0000000001-20-000001-index.html"
        ),
    )

    assert {fact["concept"] for fact in facts} >= {
        "EntityRegistrantName",
        "EntityCentralIndexKey",
        "TradingSymbol",
        "SecurityExchangeName",
    }
    assert index["sic"] == "3571"
    assert index["sic_description"] == "Electronic Computers"
    assert str(index["instance_document_url"]).endswith("demo-20200331_htm.xml")


def test_collection_preserves_bytes_before_history_build(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _write_catalog(catalog)
    plan = build_filing_cover_target_plan(catalog, output_root=tmp_path / "targets")
    targets = load_filing_cover_targets(plan.output_root)
    sender = _Sender(
        [
            _response(targets.loc[0, "index_url"], _index_document()),
            _response(targets.loc[0, "primary_document_url"], _primary_document()),
        ]
    )
    client = SecClient(
        "Filing Test research@university.edu",
        sender=sender,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    collection = collect_filing_cover_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
        artifact_policy="index-and-cover",
    )
    result = build_filing_cover_history(
        catalog,
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    anchors = pd.read_parquet(result.output_root / "issuer_year_identity_industry_anchors.parquet")
    securities = pd.read_parquet(result.output_root / "filing_security_observations.parquet")

    assert collection.requests_made == 2
    assert collection.targets_completed == 1
    assert anchors.loc[0, "artifact_pair_preserved"]
    assert anchors.loc[0, "historical_identity_ready"]
    assert anchors.loc[0, "historical_industry_ready"]
    assert anchors.loc[0, "identity_industry_ready"]
    assert securities.loc[0, "trading_symbol"] == "DEMO"
    assert securities.loc[0, "security_exchange_name"] == "Nasdaq"
    assert result.report["current_registry_metadata_used_for_history"] is False


def test_index_only_builds_issuer_identity_and_sic_without_security_backfill(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog"
    _write_catalog(catalog)
    plan = build_filing_cover_target_plan(catalog, output_root=tmp_path / "targets")
    targets = load_filing_cover_targets(plan.output_root)
    sender = _Sender([_response(targets.loc[0, "index_url"], _index_document())])
    client = SecClient(
        "Filing Test research@university.edu",
        sender=sender,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
        now=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    collection = collect_filing_cover_artifacts(
        targets,
        client,
        raw_root=tmp_path / "raw",
        limit=1,
        artifact_policy="index-only",
    )
    result = build_filing_cover_history(
        catalog,
        plan.output_root,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
    )
    anchors = pd.read_parquet(result.output_root / "issuer_year_identity_industry_anchors.parquet")

    assert collection.requests_made == 1
    assert anchors.loc[0, "index_artifact_preserved"]
    assert not anchors.loc[0, "cover_artifact_preserved"]
    assert anchors.loc[0, "filing_identity_ready"]
    assert anchors.loc[0, "historical_industry_ready"]
    assert not anchors.loc[0, "security_identity_observed"]
    assert result.report["cover_acquired_issuer_years"] == 0


def _response(url: str, content: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"Content-Type": "text/html"},
        request=httpx.Request("GET", url),
    )


def _primary_document() -> bytes:
    return b"""<html><body>
    <xbrli:context id="c1"><xbrli:entity>
      <xbrli:identifier>0000000001</xbrli:identifier>
    </xbrli:entity></xbrli:context>
    <xbrli:context id="security1"><xbrli:entity>
      <xbrli:identifier>0000000001</xbrli:identifier>
    </xbrli:entity></xbrli:context>
    <ix:nonNumeric name="dei:DocumentType" contextRef="c1">10-Q</ix:nonNumeric>
    <ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c1">
      <span>Demo Corporation</span>
    </ix:nonNumeric>
    <ix:nonNumeric name="dei:EntityCentralIndexKey" contextRef="c1">0000000001</ix:nonNumeric>
    <ix:nonNumeric name="dei:Security12bTitle" contextRef="security1">Common stock</ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol" contextRef="security1">DEMO</ix:nonNumeric>
    <ix:nonNumeric name="dei:SecurityExchangeName" contextRef="security1">Nasdaq</ix:nonNumeric>
    </body></html>"""


def _index_document() -> bytes:
    return b"""<html><body>
    <div class="companyInfo"><span class="companyName">DEMO CORPORATION (Filer)
    <acronym title="Central Index Key">CIK</acronym>:
    <a href="/cgi-bin/browse-edgar?CIK=0000000001&amp;action=getcompany">0000000001</a>
    </span><p>Type: <strong>10-Q</strong><br />
    <acronym title="Standard Industrial Code">SIC</acronym>:
    <b><a href="/cgi-bin/browse-edgar?action=getcompany&amp;SIC=3571">3571</a></b>
    Electronic Computers</p></div>
    <table><tr><td><a href="demo-20200331_htm.xml">demo instance</a></td>
    <td>EX-101.INS</td></tr></table>
    </body></html>"""


def _write_catalog(root: Path) -> None:
    root.mkdir()
    inventory = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "calendar_year": 2020,
                "representative_candidate": True,
            }
        ]
    )
    filings = pd.DataFrame(
        [
            {
                "filing_id": "filing-1",
                "cik": "0000000001",
                "accession_number": "0000000001-20-000001",
                "form": "10-Q",
                "report_date": "2020-03-31",
                "accepted_at": "2020-05-01T20:30:00+00:00",
                "is_xbrl": True,
                "is_inline_xbrl": True,
                "primary_document": "demo-20200331.htm",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1/000000000120000001/demo-20200331.htm"
                ),
            }
        ]
    )
    artifacts: dict[str, dict[str, object]] = {}
    for filename, frame in {
        "issuer_year_inventory.parquet": inventory,
        "filing_catalog.parquet": filings,
    }.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        artifacts[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "catalog_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture-catalog",
                "archive_coverage_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "output_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
