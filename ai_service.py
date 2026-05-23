"""Retrying wrappers around the provided `ai` package.

Business logic calls only this service. The service delegates to the public AI
module functions, adding logging and retry behavior without modifying `ai/`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging
from typing import Any

from ai import AnswerWithCitations, Source, fetch_arxiv, fetch_web, fetch_wikipedia, synthesize
from ai.providers.base import LLMProvider
from researcher.config import Settings
from researcher.models import SourceName
from researcher.services.retry import retry_async, retry_sync

logger = logging.getLogger(__name__)
# Fetch functions all share one signature so the service can swap real/fake sources.
FetchFn = Callable[[str, int, Any], Awaitable[list[Source]]]


class AIService:
    """Calls the provided AI functions with retries and instrumentation."""

    def __init__(
        self,
        settings: Settings,
        *,
        fetchers: dict[str, FetchFn] | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        # Injected fetchers make the class easy to test without network access.
        self._fetchers = fetchers or {
            "wikipedia": self._fetch_wikipedia,
            "arxiv": self._fetch_arxiv,
            "web": self._fetch_web,
        }

    async def _fetch_wikipedia(self, question: str, max_results: int, client: Any) -> list[Source]:
        return await fetch_wikipedia(
            question,
            max_results=max_results,
            client=client,
            timeout=self.settings.http_timeout_seconds,
        )

    async def _fetch_arxiv(self, question: str, max_results: int, client: Any) -> list[Source]:
        return await fetch_arxiv(
            question,
            max_results=max_results,
            client=client,
            timeout=self.settings.http_timeout_seconds,
        )

    async def _fetch_web(self, question: str, max_results: int, client: Any) -> list[Source]:
        return await fetch_web(question, max_results=max_results, client=client)

    async def fetch_source(
        self,
        origin: SourceName,
        question: str,
        *,
        max_results: int,
        client: Any = None,
    ) -> list[Source]:
        # Validate the source name before indexing into the fetcher map.
        if origin not in self._fetchers:
            raise ValueError(f"unsupported source: {origin}")
        label = f"fetch_{origin}"

        async def _call() -> list[Source]:
            # Wrap the selected provider so retry_async can call it repeatedly.
            return await self._fetchers[origin](question, max_results, client)

        # Transient provider failures are retried here; orchestration handles hard failures.
        sources = await retry_async(
            _call,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay_seconds,
            max_delay=self.settings.retry_max_delay_seconds,
            label=label,
        )
        logger.info("%s returned %s sources", label, len(sources))
        return sources

    def synthesize_answer(self, question: str, sources: list[Source]) -> AnswerWithCitations:
        # LLM synthesis is synchronous in the provider adapters, so use retry_sync.
        def _call() -> AnswerWithCitations:
            return synthesize(question, sources, llm=self.llm)

        answer = retry_sync(
            _call,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay_seconds,
            max_delay=self.settings.retry_max_delay_seconds,
            label="synthesize",
        )
        logger.info("synthesis completed with %s citations", len(answer.citations))
        return answer
