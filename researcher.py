"""Business logic for answering research questions."""

from __future__ import annotations

import time

from ai import AnswerWithCitations
from researcher.concurrency.orchestrator import SourceOrchestrator, merge_source_results
from researcher.config import Settings, parse_sources
from researcher.models import ResearchResult, SourceName
from researcher.services.ai_service import AIService
from researcher.services.cache import SourceCache
from researcher.storage.cache_store import FileCacheStore
from researcher.validation import ValidationError, sanitize_text, validate_question


class ResearchError(RuntimeError):
    """Base error for research workflow failures."""


class NoSourcesError(ResearchError):
    """Raised when no source returns usable evidence."""


class Researcher:
    """High-level API used by the CLI and tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ai_service: AIService | None = None,
        cache: SourceCache | None = None,
    ) -> None:
        # Dependencies are injectable so tests can use fake services and temp caches.
        self.settings = settings or Settings.from_env()
        self.ai_service = ai_service or AIService(self.settings)
        self.cache = cache or SourceCache(
            FileCacheStore(self.settings.cache_dir),
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        self.orchestrator = SourceOrchestrator(self.settings, self.ai_service, self.cache)

    async def ask(
        self,
        question: str,
        *,
        sources: str | tuple[SourceName, ...] | None = None,
        use_cache: bool | None = None,
        max_results: int | None = None,
    ) -> ResearchResult:
        start = time.perf_counter()
        # Validate early so bad input never reaches cache keys, fetchers, or the LLM.
        cleaned_question = validate_question(question, max_chars=self.settings.max_question_chars)
        chosen_sources = parse_sources(sources)  # type: ignore[arg-type]
        # Per-call use_cache overrides the environment/config default when supplied.
        cache_enabled = self.settings.cache_enabled_by_default if use_cache is None else use_cache

        # Fetch all requested evidence first; failures are captured per source.
        source_results = await self.orchestrator.fetch_all(
            cleaned_question,
            sources=chosen_sources,  # type: ignore[arg-type]
            use_cache=cache_enabled,
            max_results=max_results,
        )
        merged_sources = merge_source_results(source_results)
        # The synthesizer needs at least one real source to avoid unsupported answers.
        if not merged_sources:
            failures = "; ".join(
                f"{result.origin}: {result.error or 'no results'}" for result in source_results
            )
            raise NoSourcesError(f"no usable sources were retrieved ({failures})")

        # Synthesis happens only after all available sources are merged and deduped.
        answer = self.ai_service.synthesize_answer(cleaned_question, merged_sources)
        sanitized_answer = sanitize_text(answer.answer)
        # Keep model output safe for terminals/logs by stripping control characters.
        if sanitized_answer != answer.answer:
            answer = AnswerWithCitations(
                question=answer.question,
                answer=sanitized_answer,
                citations=answer.citations,
            )

        return ResearchResult(
            question=cleaned_question,
            answer=answer,
            source_results=source_results,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            cache_enabled=cache_enabled,
        )


__all__ = ["Researcher", "ResearchError", "NoSourcesError", "ValidationError"]
