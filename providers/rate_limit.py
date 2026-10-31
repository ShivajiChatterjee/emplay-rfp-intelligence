"""Process-wide, thread-safe provider pacing and bounded retry helpers."""

from __future__ import annotations

import os
import random
import re
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Callable, TypeVar


T = TypeVar("T")


class ProviderQuotaError(RuntimeError):
    def __init__(self, provider: str, message: str, *, retry_after: float | None = None,
                 daily_exhausted: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after
        self.daily_exhausted = daily_exhausted


def retry_after_seconds(exc: Exception) -> float | None:
    """Extract Retry-After/retry-delay without depending on one SDK shape."""
    for owner in (exc, getattr(exc, "response", None)):
        headers = getattr(owner, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value:
                try:
                    return max(0.0, float(value))
                except ValueError:
                    try:
                        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
                    except Exception:
                        pass
    text = str(exc)
    match = re.search(r"(?:retry[- ]after|retryDelay)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*s?", text, re.I)
    return float(match.group(1)) if match else None


def is_daily_exhausted(exc: Exception) -> bool:
    text = str(exc).casefold()
    return any(marker in text for marker in (
        "per day", "daily quota", "requests per day", "rpd", "quota exhausted for the day"
    ))


@dataclass
class RateLimitSettings:
    requests_per_minute: float
    max_concurrency: int
    max_retries: int
    max_backoff_seconds: float


class SharedRateLimiter:
    """Serializes start pacing while allowing bounded in-flight concurrency."""

    def __init__(self, settings: RateLimitSettings, *, clock=time.monotonic, sleeper=time.sleep) -> None:
        self.settings = settings
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(settings.max_concurrency)
        self._next_start = 0.0
        self._daily_exhausted = False

    def acquire(self) -> None:
        if self._daily_exhausted:
            raise ProviderQuotaError("provider", "Daily provider quota is exhausted", daily_exhausted=True)
        self._semaphore.acquire()
        interval = 60.0 / self.settings.requests_per_minute
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + interval
        if wait:
            self._sleep(wait)

    def release(self) -> None:
        self._semaphore.release()

    def mark_daily_exhausted(self) -> None:
        with self._lock:
            self._daily_exhausted = True

    def call(self, provider: str, operation: Callable[[], T], *, retryable: Callable[[Exception], bool]) -> T:
        for attempt in range(self.settings.max_retries + 1):
            self.acquire()
            try:
                return operation()
            except Exception as exc:
                if is_daily_exhausted(exc):
                    self.mark_daily_exhausted()
                    raise ProviderQuotaError(provider, str(exc), daily_exhausted=True) from exc
                if not retryable(exc) or attempt >= self.settings.max_retries:
                    raise
                hinted = retry_after_seconds(exc)
                delay = hinted if hinted is not None else min(
                    self.settings.max_backoff_seconds, (2 ** attempt) + random.random()
                )
            finally:
                self.release()
            self._sleep(min(delay, self.settings.max_backoff_seconds))
        raise AssertionError("unreachable")


def _int(name: str, default: int, minimum: int = 0) -> int:
    return max(minimum, int(os.getenv(name, str(default))))


def _float(name: str, default: float, minimum: float = 0.01) -> float:
    return max(minimum, float(os.getenv(name, str(default))))


@lru_cache(maxsize=1)
def gemini_limiter() -> SharedRateLimiter:
    return SharedRateLimiter(RateLimitSettings(
        requests_per_minute=_float("GEMINI_REQUESTS_PER_MINUTE", 6),
        max_concurrency=_int("GEMINI_MAX_CONCURRENCY", 1, 1),
        # Three total attempts is bounded but gives transient 503/504 failures
        # two paced opportunities to recover. The native SDK is pinned to one
        # attempt, so retries are not multiplied.
        max_retries=_int("GEMINI_PROVIDER_RETRIES", 2),
        max_backoff_seconds=_float("PROVIDER_MAX_BACKOFF_SECONDS", 60),
    ))


@lru_cache(maxsize=1)
def cohere_limiter() -> SharedRateLimiter:
    # Trial keys explicitly permit only 10 calls/minute; default stays below it.
    return SharedRateLimiter(RateLimitSettings(
        requests_per_minute=_float("COHERE_REQUESTS_PER_MINUTE", 8),
        max_concurrency=_int("COHERE_MAX_CONCURRENCY", 1, 1),
        max_retries=_int("COHERE_PROVIDER_RETRIES", 1),
        max_backoff_seconds=_float("PROVIDER_MAX_BACKOFF_SECONDS", 60),
    ))
