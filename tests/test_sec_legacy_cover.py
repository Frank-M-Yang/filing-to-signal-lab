"""Tests for conservative visible SEC cover-table evidence."""

from __future__ import annotations

from sec_data.legacy_cover import (
    merge_visible_registration_evidence,
    parse_visible_registration_tables,
)


def test_visible_registration_parser_extracts_canonical_three_column_row() -> None:
    content = b"""<html><body><table>
    <tr><th>Title of each class</th><th>Trading Symbol</th>
    <th>Name of each exchange on which registered</th></tr>
    <tr><td>Common Stock, par value $0.01 per share</td><td>DEMO</td>
    <td>The Nasdaq Stock Market LLC</td></tr>
    </table></body></html>"""

    rows = parse_visible_registration_tables(content)

    assert rows == [
        {
            "table_index": 0,
            "header_row_index": 0,
            "value_row_index": 1,
            "security_title": "Common Stock, par value $0.01 per share",
            "trading_symbol": "DEMO",
            "security_exchange_name": "The Nasdaq Stock Market LLC",
            "visible_evidence_kind": ("visible_registration_table_title_and_ticker_and_exchange"),
            "automatic_attribution_requires_single_filer": True,
            "narrative_ticker_parsing_used": False,
        }
    ]


def test_visible_registration_parser_ignores_narrative_ticker_mentions() -> None:
    content = b"""<html><body><p>Our old ticker was OLD and our temporary ticker
    will become NEW. We hope to list on Nasdaq.</p></body></html>"""

    assert parse_visible_registration_tables(content) == []


def test_title_only_visible_row_fills_one_subject_xbrl_ticker() -> None:
    xbrl = [
        {
            "context_ref": "security1",
            "context_entity_cik": "0000000001",
            "security_subject_cik_match": True,
            "trading_symbol": "DEMO",
            "security_exchange_name": None,
            "security_title": None,
            "explicit_no_trading_symbol": False,
            "security_identity_observed": True,
            "tradable_security_mapping_ready": False,
        }
    ]
    visible = [
        {
            "table_index": 2,
            "header_row_index": 0,
            "value_row_index": 1,
            "security_title": "Common Stock",
            "trading_symbol": None,
            "security_exchange_name": "New York Stock Exchange",
            "visible_evidence_kind": "visible_registration_table_title_and_exchange",
            "automatic_attribution_requires_single_filer": True,
            "narrative_ticker_parsing_used": False,
        }
    ]

    rows = merge_visible_registration_evidence(
        xbrl,
        visible,
        subject_cik="0000000001",
        single_filer_subject=True,
    )

    assert rows[0]["security_title"] == "Common Stock"
    assert rows[0]["security_exchange_name"] == "New York Stock Exchange"
    assert rows[0]["tradable_security_mapping_ready"]
    assert rows[0]["security_evidence_kind"] == ("xbrl_ticker_plus_visible_registration_table")


def test_visible_rows_are_not_attributed_in_multi_filer_submission() -> None:
    visible = [
        {
            "table_index": 0,
            "header_row_index": 0,
            "value_row_index": 1,
            "security_title": "Common Stock",
            "trading_symbol": "DEMO",
            "security_exchange_name": "NYSE",
            "visible_evidence_kind": ("visible_registration_table_title_and_ticker_and_exchange"),
            "automatic_attribution_requires_single_filer": True,
            "narrative_ticker_parsing_used": False,
        }
    ]

    rows = merge_visible_registration_evidence(
        [],
        visible,
        subject_cik="0000000001",
        single_filer_subject=False,
    )

    assert rows == []
