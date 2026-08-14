"""Offline SEC client fair-access and retry tests."""

from datetime import datetime, timezone

import httpx
import pytest

from sec_data.client import SecClient, SecConfigurationError, SecRequestError


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds


class FakeSender:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        del timeout
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(
    status: int, content: bytes = b"{}", retry_after: str | None = None
) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(
        status,
        content=content,
        headers=headers,
        request=httpx.Request("GET", "https://data.sec.gov/test"),
    )


def _client(sender: FakeSender, clock: FakeClock, retries: int = 2) -> SecClient:
    return SecClient(
        "Bias Research Lab research@university.edu",
        sender=sender,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
        max_retries=retries,
    )


def test_user_agent_requires_real_contact_email() -> None:
    with pytest.raises(SecConfigurationError):
        SecClient("anonymous bot")
    with pytest.raises(SecConfigurationError, match=r"example\.com"):
        SecClient("Demo demo@example.com")


def test_consecutive_requests_are_identified_and_throttled() -> None:
    clock = FakeClock()
    sender = FakeSender([_response(200), _response(200)])
    client = _client(sender, clock)

    client.get_submissions("1")
    client.get_companyfacts("1")

    assert sender.calls[0][1]["User-Agent"] == "Bias Research Lab research@university.edu"
    assert sender.calls[0][0].endswith("CIK0000000001.json")
    assert clock.sleeps == pytest.approx([0.20])


def test_retry_after_is_honored_for_rate_limit() -> None:
    clock = FakeClock()
    sender = FakeSender([_response(429, retry_after="2"), _response(200)])

    response = _client(sender, clock).get_submissions("1")

    assert response.status_code == 200
    assert clock.sleeps == pytest.approx([2.0])


def test_non_retryable_failure_is_explicit() -> None:
    clock = FakeClock()
    sender = FakeSender([_response(404)])

    with pytest.raises(SecRequestError, match="HTTP 404"):
        _client(sender, clock).get_submissions("1")


def test_filing_artifact_requires_safe_sec_archive_url() -> None:
    clock = FakeClock()
    sender = FakeSender([_response(200, b"<html></html>")])
    client = _client(sender, clock)

    response = client.get_filing_artifact(
        "https://www.sec.gov/Archives/edgar/data/1/0000000001000001/form10q.htm"
    )

    assert response.content == b"<html></html>"
    assert sender.calls[0][1]["Accept"].startswith("text/html")
    with pytest.raises(SecConfigurationError, match="Unsafe"):
        client.get_filing_artifact("https://example.com/filing.htm")
