"""Source cache service."""

from __future__ import annotations

import logging

from ai import Source
from researcher.storage.cache_store import FileCacheStore
from researcher.validation import canonical_query

logger = logging.getLogger(__name__)


class SourceCache:
    """Cache source fetch results by canonical query and source origin."""

    def __init__(self, store: FileCacheStore, *, ttl_seconds: int) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds

    def get(self, origin: str, question: str) -> list[Source] | None:
        # Normalize question text so spacing/case differences share one cache entry.
        entry = self.store.get(origin, canonical_query(question))
        if entry is None:
            logger.debug("cache miss origin=%s", origin)
            return None
        logger.debug("cache hit origin=%s count=%s", origin, len(entry.sources))
        return entry.sources

    def set(self, origin: str, question: str, sources: list[Source]) -> None:
        # A non-positive TTL is treated as cache disabled.
        if self.ttl_seconds <= 0:
            return
        self.store.set(
            origin,
            canonical_query(question),
            sources,
            ttl_seconds=self.ttl_seconds,
        )
