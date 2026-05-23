"""Concurrent source orchestration with per-source timeouts and degradation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ai import Source
from researcher.config import Settings
from researcher.models import SourceFetchResult, SourceName
from researcher.services.ai_service import AIService
from researcher.services.cache import SourceCache

logger = logging.getLogger(__name__)


class SourceOrchestrator:
    """Fetch sources concurrently while isolating individual failures."""

    def __init__(
        self,
        settings: Settings,
        ai_service: AIService,
        cache: SourceCache,
    ) -> None:
        self.settings = settings
        self.ai_service = ai_service
        self.cache = cache

    async def fetch_all(
        self,
        question: str,
        *,
        sources: tuple[SourceName, ...],
        use_cache: bool,
        max_results: int | None = None,
    ) -> list[SourceFetchResult]:
        max_results = max_results or self.settings.max_sources_per_query
        # One shared HTTP client lets concurrent source fetches reuse connections.
        async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
            # Build every source task before awaiting so Wikipedia/arXiv/web run in parallel.
            tasks = [
                self._fetch_one(
                    origin,
                    question,
                    use_cache=use_cache,
                    max_results=max_results,
                    client=client,
                )
                for origin in sources
            ]
            return await asyncio.gather(*tasks)

    async def _fetch_one(
        self,
        origin: SourceName,
        question: str,
        *,
        use_cache: bool,
        max_results: int,
        client: Any,
    ) -> SourceFetchResult:
        start = time.perf_counter()
        # Try the cache before any network call; canonicalization happens inside SourceCache.
        if use_cache:
            cached = self.cache.get(origin, question)
            if cached is not None:
                return SourceFetchResult(
                    origin=origin,
                    sources=cached[:max_results],
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    cached=True,
                )

        try:
            # A slow provider should not block the entire answer; time out per source.
            async with asyncio.timeout(self.settings.per_source_timeout_seconds):
                fetched = await self.ai_service.fetch_source(
                    origin,
                    question,
                    max_results=max_results,
                    client=client,
                )
        except asyncio.TimeoutError:
            message = f"timed out after {self.settings.per_source_timeout_seconds:g}s"
            logger.warning("%s source %s", origin, message)
            return SourceFetchResult(
                origin=origin,
                sources=[],
                elapsed_ms=(time.perf_counter() - start) * 1000,
                cached=False,
                error=message,
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation is required
            message = str(exc) or exc.__class__.__name__
            logger.warning("%s source failed: %s", origin, message)
            return SourceFetchResult(
                origin=origin,
                sources=[],
                elapsed_ms=(time.perf_counter() - start) * 1000,
                cached=False,
                error=message,
            )

        # Remove duplicate URLs before caching so future reads are already clean.
        unique = dedupe_sources(fetched)
        if use_cache:
            self.cache.set(origin, question, unique)
        return SourceFetchResult(
            origin=origin,
            sources=unique,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            cached=False,
        )


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """Remove duplicate URLs while preserving order."""

    seen: set[str] = set()
    out: list[Source] = []
    # Lowercased URLs are used as stable dedupe keys while preserving first occurrence.
    for source in sources:
        key = source.url.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out


def merge_source_results(results: list[SourceFetchResult]) -> list[Source]:
    """Merge source batches and dedupe globally by URL."""

    merged: list[Source] = []
    # Keep source ordering from fetch_all, then perform one global dedupe pass.
    for result in results:
        merged.extend(result.sources)
    return dedupe_sources(merged)
