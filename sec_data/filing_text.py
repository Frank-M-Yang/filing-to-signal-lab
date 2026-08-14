"""Minimal visible-text extraction for preserved SEC inline-XBRL filings."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_TAGS = frozenset(
    {"br", "div", "h1", "h2", "h3", "h4", "hr", "li", "p", "table", "td", "th", "tr"}
)
_SKIP_TAGS = frozenset({"ix:header", "script", "style"})


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        hidden = tag in _SKIP_TAGS or "display:none" in (attributes.get("style") or "").replace(
            " ", ""
        )
        if self.skip_depth > 0 or hidden:
            self.skip_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self.fragments.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth == 0 and tag in _BLOCK_TAGS:
            self.fragments.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self.fragments.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self.fragments.extend((" ", data, " "))


def extract_visible_filing_text(content: bytes) -> str:
    """Return normalized visible text from preserved SEC HTML bytes."""
    parser = _VisibleTextParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    parser.close()
    lines = []
    for line in "".join(parser.fragments).splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        normalized = re.sub(r"\s+([,.;:%)])", r"\1", normalized)
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)
