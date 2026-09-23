"""Repository-local, shared Alpha Vantage request accounting and serialization."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sec_data.filing_catalog import FilingCatalogError


class AlphaVantageBudgetError(FilingCatalogError):
    """A local request allowance or the shared collector lock prevents dispatch."""


class AlphaVantageRequestBudget:
    """Count attempts before dispatch, including legacy responses and transport failures.

    This is repository-local accounting, not an account-wide provider quota API.
    Unrecorded calls elsewhere remain unknown; an explicit external-use allowance
    can conservatively consume some local capacity. No credential is stored.
    """

    def __init__(
        self,
        *,
        state_root: Path = Path("data/market/alpha_vantage_request_budget"),
        legacy_raw_root: Path = Path("data/market/alpha_vantage_consensus"),
        daily_cap: int = 25,
        external_requests_today: int = 0,
        protected_target_ids: tuple[str, ...] = (),
        retry_reserve: int = 0,
        single_attempt_per_ticker_day: bool = False,
        minimum_interval_seconds: float = 12.1,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= daily_cap <= 25 or not 0 <= external_requests_today <= daily_cap:
            raise ValueError("Local daily allowance must remain within 25 requests")
        if retry_reserve < 0 or minimum_interval_seconds < 12:
            raise ValueError("Unsafe retry reserve or request interval")
        self.state_root = state_root
        self.legacy_raw_root = legacy_raw_root
        self.daily_cap = daily_cap
        self.external_requests_today = external_requests_today
        self.protected_target_ids = protected_target_ids
        self.retry_reserve = retry_reserve
        self.single_attempt_per_ticker_day = single_attempt_per_ticker_day
        self.minimum_interval_seconds = minimum_interval_seconds
        self.now = now
        self.sleep = sleep

    @contextmanager
    def request(self, ticker: str) -> Iterator[Path]:
        """Hold one cross-process lock through dispatch; reserve even uncertain attempts."""
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-/]{0,31}", ticker):
            raise ValueError("Invalid budget ticker")
        self.state_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_root / "collector.lock"
        try:
            lock = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise AlphaVantageBudgetError("shared_collector_busy") from exc
        try:
            with lock:
                lock.write("One Alpha Vantage collector owns this lock; no credential stored.\n")
                instant = self._utc_now()
                summary = self.summary(instant)
                if self.single_attempt_per_ticker_day and any(
                    _read_json(p)["ticker"] == ticker
                    for p in (self.state_root / "attempts" / instant.date().isoformat()).glob(
                        "*.json"
                    )
                ):
                    raise AlphaVantageBudgetError("already_attempted_today_by_shared_ledger")
                previous = summary["last_local_request_at"]
                if previous:
                    elapsed = (instant - datetime.fromisoformat(str(previous))).total_seconds()
                    if elapsed < 0:
                        raise AlphaVantageBudgetError("request_clock_moved_backwards")
                    self.sleep(max(0, self.minimum_interval_seconds - elapsed))
                # Recompute the actual UTC day after throttling across midnight.
                instant = self._utc_now()
                summary = self.summary(instant)
                if summary["available_requests"] < 1:
                    raise AlphaVantageBudgetError("local_daily_budget_exhausted_or_protected")
                directory = self.state_root / "attempts" / instant.date().isoformat()
                directory.mkdir(parents=True, exist_ok=True)
                entry = directory / f"{uuid.uuid4().hex}.json"
                with entry.open("x", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "ticker": ticker,
                            "requested_at": instant.isoformat(),
                            "utc_day": instant.date().isoformat(),
                            "dispatch_or_transport_outcome_unknown": True,
                        },
                        stream,
                    )
                yield entry
        finally:
            lock_path.unlink(missing_ok=True)

    def record_response(self, reservation: Path, content: bytes) -> None:
        """Write a separate receipt, never overwrite a pre-dispatch reservation."""
        receipt = reservation.with_suffix(".receipt")
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump({"response_sha256": hashlib.sha256(content).hexdigest()}, stream)

    def summary(self, instant: datetime | None = None) -> dict[str, Any]:
        """Report local capacity, preserving all still-pending original daily targets."""
        timestamp = instant or self._utc_now()
        if timestamp.tzinfo is None:
            raise AlphaVantageBudgetError("Budget clock must be timezone-aware")
        day = timestamp.astimezone(timezone.utc).date().isoformat()
        entries = sorted((self.state_root / "attempts" / day).glob("*.json"))
        matched: set[tuple[str, str]] = set()
        last: datetime | None = None
        for path in entries:
            entry = _read_json(path)
            requested = datetime.fromisoformat(entry["requested_at"])
            if (
                requested.tzinfo is None
                or requested.astimezone(timezone.utc).date().isoformat() != day
            ):
                raise AlphaVantageBudgetError("Invalid request ledger clock")
            last = requested if last is None else max(last, requested)
            receipt = path.with_suffix(".receipt")
            if receipt.exists():
                matched.add((entry["ticker"], _read_json(receipt)["response_sha256"]))
        legacy_signatures: set[tuple[str, str]] = set()
        for path in (self.legacy_raw_root / "snapshots" / day).glob("*/*/manifest.json"):
            manifest = _read_json(path)
            raw = Path(manifest["path"]).resolve()
            if not raw.is_relative_to(self.legacy_raw_root.resolve()) or not raw.is_file():
                raise AlphaVantageBudgetError("Invalid legacy response path")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            if digest != manifest["sha256"]:
                raise AlphaVantageBudgetError("Changed legacy response bytes")
            legacy_signatures.add((manifest["ticker"], digest))
            captured = datetime.fromisoformat(manifest["fetched_at"])
            if (
                captured.tzinfo is None
                or captured.astimezone(timezone.utc).date().isoformat() != day
            ):
                raise AlphaVantageBudgetError("Legacy capture was backdated")
            last = captured if last is None else max(last, captured)
        untracked = len(legacy_signatures - matched)
        protected_pending = sum(
            not (self.legacy_raw_root / "completions" / day / f"{target}.json").is_file()
            for target in self.protected_target_ids
        )
        used = len(entries) + untracked + self.external_requests_today
        return {
            "utc_day": day,
            "local_attempts_recorded": len(entries),
            "legacy_response_attempts_not_in_ledger": untracked,
            "assumed_external_requests_today": self.external_requests_today,
            "account_wide_usage_verified": False,
            "local_daily_cap": self.daily_cap,
            "protected_original_targets_pending": protected_pending,
            "retry_reserve_requests": self.retry_reserve,
            "available_requests": max(
                0, self.daily_cap - used - protected_pending - self.retry_reserve
            ),
            "last_local_request_at": last.isoformat() if last else None,
            "legacy_transport_failures_before_ledger_are_unknown": True,
        }

    def _utc_now(self) -> datetime:
        instant = self.now()
        if instant.tzinfo is None:
            raise AlphaVantageBudgetError("Budget clock must be timezone-aware")
        return instant.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result
