"""Thread-safe rolling-window rate limiting for hosted LLM providers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


class RateLimitQueueTimeout(RuntimeError):
    """Raised when a request cannot obtain a permit within its wait budget."""


@dataclass(frozen=True)
class RateLimitPermit:
    waited_seconds: float
    requests_in_window: int
    remaining_requests: int
    limit: int


class SlidingWindowRateLimiter:
    """Enforce both a rolling quota and minimum spacing between requests.

    Every HTTP attempt consumes a permit, including retries and catalog calls.
    A provider 429 can place the shared limiter into a cooldown so concurrent
    Streamlit sessions stop sending requests until the provider window resets.
    """

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float = 60.0,
        min_interval_seconds: float = 0.0,
        max_wait_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.limit = max(1, int(limit))
        self.window_seconds = max(0.01, float(window_seconds))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_wait_seconds = max(0.0, float(max_wait_seconds))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._attempts: deque[float] = deque()
        self._cooldown_until = 0.0
        self._last_request_at: float | None = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._attempts and self._attempts[0] <= cutoff:
            self._attempts.popleft()

    def acquire(self) -> RateLimitPermit:
        started = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                waits = [max(0.0, self._cooldown_until - now)]
                if len(self._attempts) >= self.limit:
                    waits.append(
                        max(0.0, self._attempts[0] + self.window_seconds - now)
                    )
                if self._last_request_at is not None:
                    waits.append(
                        max(
                            0.0,
                            self._last_request_at
                            + self.min_interval_seconds
                            - now,
                        )
                    )
                wait_seconds = max(waits)
                if wait_seconds <= 0:
                    self._attempts.append(now)
                    self._last_request_at = now
                    used = len(self._attempts)
                    return RateLimitPermit(
                        waited_seconds=max(0.0, now - started),
                        requests_in_window=used,
                        remaining_requests=max(0, self.limit - used),
                        limit=self.limit,
                    )

            elapsed = self._clock() - started
            if elapsed + wait_seconds > self.max_wait_seconds:
                raise RateLimitQueueTimeout(
                    f"Provider request queue would wait {wait_seconds:.1f}s, "
                    f"exceeding its {self.max_wait_seconds:.1f}s safety limit."
                )
            self._sleep(max(0.001, wait_seconds))

    def penalize(self, cooldown_seconds: float) -> None:
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until,
                self._clock() + max(0.0, float(cooldown_seconds)),
            )

    def status(self) -> dict[str, float | int]:
        with self._lock:
            now = self._clock()
            self._prune(now)
            used = len(self._attempts)
            waits = [max(0.0, self._cooldown_until - now)]
            if used >= self.limit:
                waits.append(max(0.0, self._attempts[0] + self.window_seconds - now))
            if self._last_request_at is not None:
                waits.append(
                    max(
                        0.0,
                        self._last_request_at + self.min_interval_seconds - now,
                    )
                )
            return {
                "limit": self.limit,
                "requests_in_window": used,
                "remaining_requests": max(0, self.limit - used),
                "window_seconds": self.window_seconds,
                "next_request_in_seconds": max(waits),
                "cooldown_seconds": max(0.0, self._cooldown_until - now),
            }
