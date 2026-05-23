"""Offline tests for the software-engineering layer."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ai import Source
from ai.providers.base import LLMProvider
from researcher.cli import main as cli_main
from researcher.config import Settings, parse_sources
from researcher.core.researcher import NoSourcesError, Researcher
from researcher.services.ai_service import AIService, FetchFn
from researcher.services.cache import SourceCache
from researcher.storage.cache_store import FileCacheStore
from researcher.validation import ValidationError, canonical_query, sanitize_text, validate_question


class TinyLLM(LLMProvider):
    def complete(self, prompt: str, *, json_schema=None, max_tokens: int = 1024) -> str:
        return "A concise answer supported by the retrieved evidence [1]."


def make_source(origin: str, n: int = 1) -> Source:
    return Source(
        title=f"{origin} title {n}",
        url=f"https://example.com/{origin}/{n}",
        snippet=f"{origin} snippet {n}",
        origin=origin,
    )


def test_validate_question_rejects_empty_and_oversized() -> None:
    with pytest.raises(ValidationError):
        validate_question("   ", max_chars=50)
    with pytest.raises(ValidationError):
        validate_question("x" * 51, max_chars=50)


def test_sanitize_and_canonical_query() -> None:
    assert sanitize_text("hello\x00 world") == "hello world"
    assert canonical_query("  WHAT   Is   AI?  ") == "what is ai?"


def test_parse_sources_aliases_and_dedupes() -> None:
    assert parse_sources("wiki,arxiv,wikipedia") == ("wikipedia", "arxiv")
    with pytest.raises(ValueError):
        parse_sources("reddit")


def test_file_cache_canonicalizes_queries(tmp_path) -> None:
    cache = SourceCache(FileCacheStore(tmp_path), ttl_seconds=60)
    cache.set("web", "WHAT   Is AI?", [make_source("web")])

    hit = cache.get("web", "what is ai?")

    assert hit is not None
    assert hit[0].title == "web title 1"


@pytest.mark.asyncio
async def test_researcher_fetches_sources_concurrently(tmp_path) -> None:
    settings = Settings(
        cache_dir=tmp_path,
        retry_attempts=1,
        per_source_timeout_seconds=2.0,
        retry_base_delay_seconds=0,
    )

    starts: list[float] = []

    async def slow(origin: str, question: str, max_results: int, client: Any) -> list[Source]:
        starts.append(time.perf_counter())
        await asyncio.sleep(0.12)
        return [make_source(origin)]

    fetchers: dict[str, FetchFn] = {
        "wikipedia": lambda q, m, c: slow("wikipedia", q, m, c),
        "arxiv": lambda q, m, c: slow("arxiv", q, m, c),
        "web": lambda q, m, c: slow("web", q, m, c),
    }
    researcher = Researcher(settings, ai_service=AIService(settings, fetchers=fetchers, llm=TinyLLM()))

    started = time.perf_counter()
    result = await researcher.ask("What is async orchestration?", use_cache=False)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.34  # sequential would be roughly 0.36s before synthesis
    assert max(starts) - min(starts) < 0.05
    assert len(result.source_results) == 3
    assert all(item.error is None for item in result.source_results)
    assert "[1]" in result.answer.answer


@pytest.mark.asyncio
async def test_researcher_gracefully_degrades_when_one_source_fails(tmp_path) -> None:
    settings = Settings(cache_dir=tmp_path, retry_attempts=1, retry_base_delay_seconds=0)

    async def wiki(question: str, max_results: int, client: Any) -> list[Source]:
        return [make_source("wikipedia")]

    async def arxiv(question: str, max_results: int, client: Any) -> list[Source]:
        raise RuntimeError("arxiv unavailable")

    async def web(question: str, max_results: int, client: Any) -> list[Source]:
        return [make_source("web")]

    fetchers: dict[str, FetchFn] = {"wikipedia": wiki, "arxiv": arxiv, "web": web}
    researcher = Researcher(settings, ai_service=AIService(settings, fetchers=fetchers, llm=TinyLLM()))

    result = await researcher.ask("What is graceful degradation?", use_cache=False)

    assert len(result.failures) == 1
    assert result.failures[0].origin == "arxiv"
    assert len(result.sources) == 2


@pytest.mark.asyncio
async def test_researcher_raises_when_all_sources_fail(tmp_path) -> None:
    settings = Settings(cache_dir=tmp_path, retry_attempts=1, retry_base_delay_seconds=0)

    async def fail(question: str, max_results: int, client: Any) -> list[Source]:
        raise RuntimeError("offline failure")

    fetchers: dict[str, FetchFn] = {"wikipedia": fail, "arxiv": fail, "web": fail}
    researcher = Researcher(settings, ai_service=AIService(settings, fetchers=fetchers, llm=TinyLLM()))

    with pytest.raises(NoSourcesError):
        await researcher.ask("Will this fail?", use_cache=False)


@pytest.mark.asyncio
async def test_cache_prevents_second_fetch(tmp_path) -> None:
    settings = Settings(cache_dir=tmp_path, retry_attempts=1, retry_base_delay_seconds=0)
    calls = {"web": 0}

    async def web(question: str, max_results: int, client: Any) -> list[Source]:
        calls["web"] += 1
        return [make_source("web", calls["web"])]

    fetchers: dict[str, FetchFn] = {"web": web}
    researcher = Researcher(settings, ai_service=AIService(settings, fetchers=fetchers, llm=TinyLLM()))

    first = await researcher.ask("What is caching?", sources="web", use_cache=True)
    second = await researcher.ask("  WHAT is   caching? ", sources="web", use_cache=True)

    assert calls["web"] == 1
    assert first.sources[0].url == second.sources[0].url
    assert second.source_results[0].cached is True


def test_cli_offline_mode_outputs_answer(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    code = cli_main([
        "ask",
        "What is photosynthesis?",
        "--offline",
        "--sources",
        "wiki,web",
        "--no-cache",
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert "Q: What is photosynthesis?" in captured.out
    assert "References:" in captured.out
    assert "Fetch summary:" in captured.out
