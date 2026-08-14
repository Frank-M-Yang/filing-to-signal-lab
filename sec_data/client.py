"""Identified, rate-limited HTTP access to public SEC EDGAR data APIs."""

from __future__ import annotations

import email.utils
import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from sec_data.models import SecResponse

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{filename}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class SecConfigurationError(ValueError):
    """Raised when automated access lacks a compliant identity."""


class SecRequestError(RuntimeError):
    """Raised after a public SEC request cannot be completed safely."""


class HttpSender(Protocol):
    """Injectable subset of an HTTP client used in offline tests."""

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        """Return an HTTP response."""
        ...


class SecClient:
    """Download public EDGAR JSON while respecting SEC fair-access guidance."""

    retryable_statuses = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        user_agent: str,
        *,
        request_interval_seconds: float = 0.20,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sender: HttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.user_agent = validate_user_agent(user_agent)
        if request_interval_seconds < 0.10:
            raise SecConfigurationError("Request interval must be at least 0.10 seconds")
        if timeout_seconds <= 0.0 or not 0 <= max_retries <= 8:
            raise SecConfigurationError("Timeout and retry settings are invalid")
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sender = sender or httpx.Client(follow_redirects=True)
        self.sleep = sleep
        self.monotonic = monotonic
        self.now = now
        self._last_request_started: float | None = None

    def get_submissions(self, cik: str) -> SecResponse:
        """Fetch the primary submissions JSON for a zero-padded CIK."""
        return self.get(SUBMISSIONS_URL.format(cik=normalize_cik(cik)))

    def get_submission_archive(self, filename: str) -> SecResponse:
        """Fetch an older submissions-history shard named by the primary payload."""
        if re.fullmatch(r"CIK\d{10}-submissions-\d{3}\.json", filename) is None:
            raise SecConfigurationError(f"Unsafe submissions archive filename: {filename}")
        return self.get(SUBMISSIONS_ARCHIVE_URL.format(filename=filename))

    def get_companyfacts(self, cik: str) -> SecResponse:
        """Fetch all standardized companyfacts for a CIK."""
        return self.get(COMPANYFACTS_URL.format(cik=normalize_cik(cik)))

    def resolve_ticker(self, ticker: str) -> tuple[str, str, str, SecResponse]:
        """Resolve a ticker to CIK, normalized ticker, and SEC company title."""
        normalized = ticker.strip().upper()
        if not normalized or re.fullmatch(r"[A-Z0-9.\-]+", normalized) is None:
            raise SecConfigurationError("Ticker contains unsupported characters")
        response = self.get(TICKER_MAP_URL)
        document = decode_json_object(response.content, source_url=response.url)
        for raw_record in document.values():
            if not isinstance(raw_record, dict):
                continue
            if str(raw_record.get("ticker", "")).upper() == normalized:
                cik = normalize_cik(str(raw_record.get("cik_str", "")))
                return cik, normalized, str(raw_record.get("title", normalized)), response
        raise SecConfigurationError(f"Ticker {normalized} was not found in SEC's current map")

    def get_filing_artifact(self, url: str) -> SecResponse:
        """Fetch one SEC filing HTML/XML artifact with the shared rate limiter."""
        if not _safe_sec_archive_url(url):
            raise SecConfigurationError(f"Unsafe SEC filing artifact URL: {url}")
        return self.get(
            url,
            accept="text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8",
        )

    def get(self, url: str, *, accept: str = "application/json") -> SecResponse:
        """Run one identified request with bounded retry and throttling."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Encoding": "gzip, deflate",
        }
        attempts = self.max_retries + 1
        last_detail = "request did not start"
        for attempt in range(attempts):
            self._throttle()
            try:
                response = self.sender.get(url, headers=headers, timeout=self.timeout_seconds)
            except httpx.RequestError as exc:
                last_detail = str(exc)
                if attempt + 1 == attempts:
                    break
                self.sleep(_backoff_seconds(attempt))
                continue
            if 200 <= response.status_code < 300:
                return SecResponse(
                    url=str(response.url),
                    content=response.content,
                    fetched_at=self.now(),
                    status_code=response.status_code,
                    headers={key.lower(): value for key, value in response.headers.items()},
                )
            last_detail = f"HTTP {response.status_code}"
            if response.status_code not in self.retryable_statuses or attempt + 1 == attempts:
                break
            self.sleep(_retry_delay(response, attempt, self.now()))
        raise SecRequestError(f"SEC request failed for {url}: {last_detail}")

    def _throttle(self) -> None:
        current = self.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (current - self._last_request_started)
            if remaining > 0.0:
                self.sleep(remaining)
                current = self.monotonic()
        self._last_request_started = current


def _safe_sec_archive_url(url: str) -> bool:
    return (
        re.fullmatch(
            r"https://www\.sec\.gov/Archives/edgar/data/\d+/[A-Za-z0-9._/\-]+",
            url,
        )
        is not None
    )


def normalize_cik(value: str) -> str:
    """Return a validated ten-digit SEC Central Index Key."""
    stripped = value.strip()
    if not stripped.isdigit() or len(stripped) > 10:
        raise SecConfigurationError("CIK must contain at most ten digits")
    return stripped.zfill(10)


def validate_user_agent(value: str) -> str:
    """Require the organization/name plus real contact email SEC asks clients to send."""
    normalized = value.strip()
    email_match = re.search(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", normalized)
    if len(normalized) < 10 or email_match is None:
        raise SecConfigurationError(
            "SEC User-Agent must identify a person or organization and include a contact email"
        )
    if email_match.group(0).lower().endswith("@example.com"):
        raise SecConfigurationError("Use a real contact email, not example.com")
    return normalized


def decode_json_object(payload: bytes, *, source_url: str) -> dict[str, Any]:
    """Decode a top-level JSON object with a source-specific error."""
    try:
        decoded: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecRequestError(f"Malformed JSON from {source_url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SecRequestError(f"Expected a JSON object from {source_url}")
    return decoded


def _backoff_seconds(attempt: int) -> float:
    return float(min(0.5 * (2**attempt), 8.0))


def _retry_delay(response: httpx.Response, attempt: int, now: datetime) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return _backoff_seconds(attempt)
    try:
        return max(0.0, min(float(retry_after), 60.0))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return _backoff_seconds(attempt)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return float(max(0.0, min((retry_at - now).total_seconds(), 60.0)))
