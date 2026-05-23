"""Small retry helpers with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _delay(attempt: int, base: float, maximum: float) -> float:
    # Exponential backoff: base, base*2, base*4, capped at maximum.
    return min(maximum, base * (2 ** max(0, attempt - 1)))


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    label: str,
) -> T:
    last_exc: Exception | None = None
    # Attempts are 1-based so log messages match human expectations.
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - retry wrapper is intentionally broad
            last_exc = exc
            if attempt >= attempts:
                break
            wait = _delay(attempt, base_delay, max_delay)
            logger.warning("%s failed on attempt %s/%s: %s", label, attempt, attempts, exc)
            if wait > 0:
                # Async sleep yields control so other fetches can keep running.
                await asyncio.sleep(wait)
    assert last_exc is not None
    # Re-raise the final provider error so callers can decide how to degrade.
    raise last_exc


def retry_sync(
    func: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    label: str,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - retry wrapper is intentionally broad
            last_exc = exc
            if attempt >= attempts:
                break
            wait = _delay(attempt, base_delay, max_delay)
            logger.warning("%s failed on attempt %s/%s: %s", label, attempt, attempts, exc)
            if wait > 0:
                # Synchronous providers block here between retry attempts.
                time.sleep(wait)
    assert last_exc is not None
    # Keep the original exception type/message for clear CLI error output.
    raise last_exc
