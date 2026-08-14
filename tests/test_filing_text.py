"""Visible-text extraction tests for preserved inline-XBRL filings."""

from sec_data.filing_text import extract_visible_filing_text


def test_extract_visible_filing_text_joins_inline_tags_and_skips_xbrl_header() -> None:
    content = b"""
    <html><body>
      <ix:header><ix:hidden>secret context</ix:hidden></ix:header>
      <p>Prior periods were <span>re</span><span>classified</span>.</p>
      <div style="display:none">hidden fact</div>
      <table><tr><td>Revenue</td><td>100</td></tr></table>
    </body></html>
    """

    text = extract_visible_filing_text(content)

    assert "secret context" not in text
    assert "hidden fact" not in text
    assert "Prior periods were re classified." in text
    assert "Revenue\n100" in text
