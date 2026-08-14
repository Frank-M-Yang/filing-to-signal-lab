"""Typed values exchanged by the SEC ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SecResponse:
    """Exact bytes and metadata returned by one SEC request."""

    url: str
    content: bytes
    fetched_at: datetime
    status_code: int
    headers: dict[str, str]


@dataclass(frozen=True)
class RawSnapshot:
    """Content-addressed local copy of a source response."""

    sha256: str
    source_url: str
    fetched_at: datetime
    path: Path
    kind: str


@dataclass(frozen=True)
class ParseResult:
    """Normalized records and structured row-level rejections."""

    data: pd.DataFrame
    rejections: pd.DataFrame


@dataclass(frozen=True)
class SecIngestionResult:
    """Complete audit package for one company download."""

    cik: str
    ticker: str | None
    company_name: str
    filings: pd.DataFrame
    facts: pd.DataFrame
    amendment_comparisons: pd.DataFrame
    rejections: pd.DataFrame
    snapshots: tuple[RawSnapshot, ...]
    metadata: dict[str, Any]
