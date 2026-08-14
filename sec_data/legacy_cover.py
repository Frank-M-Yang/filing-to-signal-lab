"""High-precision parsing of visible SEC cover registration tables."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, Final

_SKIP_TAGS: Final[frozenset[str]] = frozenset({"ix:header", "script", "style"})
_TITLE_HEADER: Final[re.Pattern[str]] = re.compile(r"\btitle\s+of\s+each\s+class\b", re.IGNORECASE)
_TICKER_HEADER: Final[re.Pattern[str]] = re.compile(r"\btrading\s+symbol\b", re.IGNORECASE)
_EXCHANGE_HEADER: Final[re.Pattern[str]] = re.compile(
    r"\bname\s+of\s+each\s+exchange\s+on\s+which\s+registered\b", re.IGNORECASE
)
_TICKER_VALUE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9][A-Z0-9.\-/]{0,14}")
_EXCHANGE_VALUE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:nasdaq|new york stock exchange|nyse|cboe|otc|american stock exchange|"
    r"arca|investors exchange|iex)\b",
    re.IGNORECASE,
)


def parse_visible_registration_tables(content: bytes) -> list[dict[str, Any]]:
    """Extract only canonical Section 12(b)/(g) visible cover-table rows.

    This deliberately ignores ticker-like prose. Narrative filing text can mention
    temporary, historical, or proposed symbols and exchanges, so it is not safe
    point-in-time identity evidence without manual adjudication.
    """
    parser = _VisibleTableParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    parser.close()
    records: list[dict[str, Any]] = []
    for table_index, rows in enumerate(parser.tables):
        records.extend(_registration_rows(rows, table_index=table_index))
    return records


def merge_visible_registration_evidence(
    xbrl_observations: list[dict[str, Any]],
    visible_observations: list[dict[str, Any]],
    *,
    subject_cik: str,
    single_filer_subject: bool,
) -> list[dict[str, Any]]:
    """Conservatively combine XBRL security facts with visible cover rows.

    Visible rows are attributed automatically only when the filing index has one
    filer and that filer is the sampled subject. A title-only legacy row can fill
    one missing XBRL title only when both sides are unambiguous.
    """
    merged = [
        {
            **observation,
            "security_evidence_kind": "xbrl_cover_facts",
            "security_evidence_confidence": "subject_cik_context_verified",
        }
        for observation in xbrl_observations
    ]
    if not single_filer_subject:
        return merged
    subject_ticker_rows = [
        row for row in merged if row.get("security_subject_cik_match") and row.get("trading_symbol")
    ]
    title_only_rows = [row for row in visible_observations if not row.get("trading_symbol")]
    for visible in visible_observations:
        ticker = visible.get("trading_symbol")
        if ticker:
            matches = [row for row in subject_ticker_rows if row.get("trading_symbol") == ticker]
            if len(matches) == 1:
                _fill_from_visible(matches[0], visible)
                continue
            if matches:
                continue
            merged.append(_visible_security_observation(visible, subject_cik=subject_cik))
            continue
        if len(title_only_rows) == 1 and len(subject_ticker_rows) == 1:
            _fill_from_visible(subject_ticker_rows[0], visible)
    return merged


class _VisibleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._cell_depth = 0
        self._cell_fragments: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        attributes = {key.lower(): value or "" for key, value in attrs}
        hidden = (
            lowered in _SKIP_TAGS
            or "display:none" in attributes.get("style", "").replace(" ", "").lower()
        )
        if self._skip_depth > 0 or hidden:
            self._skip_depth += 1
            return
        if lowered == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
            return
        if self._table_depth != 1:
            return
        if lowered == "tr":
            self._current_row = []
        elif lowered in {"td", "th"} and self._current_row is not None:
            self._cell_depth = 1
            self._cell_fragments = []
        elif self._cell_depth > 0:
            self._cell_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered == "table":
            if self._table_depth == 1 and self._current_table:
                self.tables.append(self._current_table)
            self._table_depth = max(0, self._table_depth - 1)
            if self._table_depth == 0:
                self._current_table = None
                self._current_row = None
                self._cell_depth = 0
            return
        if self._table_depth != 1:
            return
        if self._cell_depth > 0:
            self._cell_depth -= 1
            if self._cell_depth == 0 and lowered in {"td", "th"}:
                if self._current_row is not None:
                    self._current_row.append(_normalize("".join(self._cell_fragments)))
                self._cell_fragments = []
                return
        if lowered == "tr" and self._current_row is not None:
            if self._current_table is not None and any(self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._table_depth == 1 and self._cell_depth > 0:
            self._cell_fragments.extend((" ", data, " "))


def _registration_rows(rows: list[list[str]], *, table_index: int) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for header_index, header in enumerate(rows):
        header_map = _header_map(header)
        title_index = header_map.get("title")
        if title_index is None:
            continue
        for row_index in range(header_index + 1, len(rows)):
            values = rows[row_index]
            if _header_map(values):
                break
            title = _cell(values, title_index)
            ticker = _cell(values, header_map.get("ticker"))
            exchange = _cell(values, header_map.get("exchange"))
            if not _valid_title(title):
                continue
            normalized_ticker = _valid_ticker(ticker)
            normalized_exchange = _valid_exchange(exchange)
            if "ticker" in header_map and normalized_ticker is None:
                continue
            if "exchange" in header_map and normalized_exchange is None:
                continue
            observations.append(
                {
                    "table_index": table_index,
                    "header_row_index": header_index,
                    "value_row_index": row_index,
                    "security_title": title,
                    "trading_symbol": normalized_ticker,
                    "security_exchange_name": normalized_exchange,
                    "visible_evidence_kind": _evidence_kind(header_map),
                    "automatic_attribution_requires_single_filer": True,
                    "narrative_ticker_parsing_used": False,
                }
            )
        if observations:
            break
    return observations


def _header_map(row: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(row):
        if _TITLE_HEADER.search(value):
            result["title"] = index
        if _TICKER_HEADER.search(value):
            result["ticker"] = index
        if _EXCHANGE_HEADER.search(value):
            result["exchange"] = index
    return result


def _cell(row: list[str], index: int | None) -> str | None:
    if index is None or index >= len(row):
        return None
    return row[index] or None


def _valid_title(value: str | None) -> bool:
    if value is None or len(value) > 300:
        return False
    lowered = value.casefold()
    return not (
        _TITLE_HEADER.search(value)
        or lowered in {"none", "not applicable", "n/a", "-", "—", "–"}
        or lowered.startswith("indicate by check mark")
        or lowered.startswith("securities registered")
    )


def _valid_ticker(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"NONE", "N/A", "NA", "-", "—", "–"}:
        return None
    return normalized if _TICKER_VALUE.fullmatch(normalized) else None


def _valid_exchange(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize(value)
    if normalized.casefold() in {"none", "not applicable", "n/a", "-", "—", "–"}:
        return None
    return normalized if _EXCHANGE_VALUE.search(normalized) else None


def _evidence_kind(header_map: dict[str, int]) -> str:
    suffix = "_and_".join(key for key in ("title", "ticker", "exchange") if key in header_map)
    return f"visible_registration_table_{suffix}"


def _fill_from_visible(target: dict[str, Any], visible: dict[str, Any]) -> None:
    title = visible.get("security_title")
    exchange = visible.get("security_exchange_name")
    existing_title = target.get("security_title")
    existing_exchange = target.get("security_exchange_name")
    if existing_title and title and _normalize(str(existing_title)) != _normalize(str(title)):
        return
    exchange_conflict = bool(
        existing_exchange
        and exchange
        and _normalize(str(existing_exchange)).casefold() != _normalize(str(exchange)).casefold()
    )
    if exchange_conflict:
        return
    if not existing_title:
        target["security_title"] = title
    if not existing_exchange:
        target["security_exchange_name"] = exchange
    target["tradable_security_mapping_ready"] = bool(
        target.get("security_subject_cik_match")
        and target.get("trading_symbol")
        and target.get("security_exchange_name")
    )
    target["security_evidence_kind"] = "xbrl_ticker_plus_visible_registration_table"
    target["security_evidence_confidence"] = "single_filer_and_subject_context_verified"


def _visible_security_observation(visible: dict[str, Any], *, subject_cik: str) -> dict[str, Any]:
    ticker = visible.get("trading_symbol")
    exchange = visible.get("security_exchange_name")
    return {
        "context_ref": (
            f"visible_registration_table:{visible['table_index']}:{visible['value_row_index']}"
        ),
        "context_entity_cik": subject_cik,
        "security_subject_cik_match": True,
        "trading_symbol": ticker,
        "security_exchange_name": exchange,
        "security_title": visible.get("security_title"),
        "explicit_no_trading_symbol": False,
        "security_identity_observed": bool(ticker),
        "tradable_security_mapping_ready": bool(ticker and exchange),
        "security_evidence_kind": "visible_registration_table",
        "security_evidence_confidence": "single_filer_subject_block_verified",
    }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()
