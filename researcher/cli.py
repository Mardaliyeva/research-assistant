"""Command line interface for the async research assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any

from ai import AnswerWithCitations, Source
from ai.providers.base import LLMProvider
from researcher.config import Settings, parse_sources
from researcher.core.researcher import NoSourcesError, Researcher
from researcher.logging_utils import configure_logging
from researcher.models import ResearchResult
from researcher.services.ai_service import AIService, FetchFn
from researcher.storage.cache_store import FileCacheStore
from researcher.validation import ValidationError, sanitize_text


# OfflineLLM keeps the CLI demonstrable when the user has not configured real
# API keys. It follows the same interface as a production LLM provider.
class OfflineLLM(LLMProvider):
    """Deterministic LLM for no-key CLI demos and offline tests."""

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        # Sources are numbered in the prompt as [1], [2], etc.; the fake
        # response cites the first few so downstream citation rendering works.
        indices = re.findall(r"^\[(\d+)\]", prompt, flags=re.MULTILINE)
        if not indices:
            return "I cannot answer from the available sources."
        chosen = ", ".join(f"[{idx}]" for idx in indices[: min(3, len(indices))])
        return (
            "The retrieved sources provide enough context for a concise research "
            f"answer, with the strongest evidence coming from {chosen}. In a live "
            "run, the configured LLM will replace this deterministic offline text "
            "with a provider-generated synthesis."
        )


async def _offline_fetch(origin: str, question: str, max_results: int, client: Any) -> list[Source]:
    # The offline fetcher mirrors the real async fetcher signature so tests and
    # demos can swap it in without changing the orchestration layer.
    snippets = {
        "wikipedia": "A concise encyclopedia-style overview of the topic.",
        "arxiv": "A research-paper abstract discussing mechanisms, methods, and evidence.",
        "web": "A web-search result with current explanatory context and examples.",
    }
    labels = {"wikipedia": "Wikipedia", "arxiv": "arXiv", "web": "Web"}
    return [
        Source(
            title=f"Offline {labels[origin]} source for {question[:60]}",
            url=f"https://example.com/offline/{origin}/{abs(hash((origin, question))) % 100000}",
            snippet=snippets[origin],
            origin=origin,
        )
    ][:max_results]


def build_offline_ai_service(settings: Settings) -> AIService:
    # Each source gets a tiny wrapper so the injected fetcher map matches the
    # production source names expected by AIService.
    async def wiki(question: str, max_results: int, client: Any) -> list[Source]:
        return await _offline_fetch("wikipedia", question, max_results, client)

    async def arxiv(question: str, max_results: int, client: Any) -> list[Source]:
        return await _offline_fetch("arxiv", question, max_results, client)

    async def web(question: str, max_results: int, client: Any) -> list[Source]:
        return await _offline_fetch("web", question, max_results, client)

    fetchers: dict[str, FetchFn] = {"wikipedia": wiki, "arxiv": arxiv, "web": web}
    return AIService(settings, fetchers=fetchers, llm=OfflineLLM())


def render_text(result: ResearchResult) -> str:
    # Human-readable CLI output: answer first, references second, diagnostics last.
    answer: AnswerWithCitations = result.answer
    lines = [
        f"Q: {sanitize_text(answer.question)}",
        "",
        f"A: {sanitize_text(answer.answer)}",
        "",
    ]

    if answer.citations:
        lines.append("References:")
        for citation in answer.citations:
            source = citation.source
            lines.append(f"  [{citation.index}] ({source.origin}) {sanitize_text(source.title)}")
            lines.append(f"      {sanitize_text(source.url)}")
        lines.append("")

    if result.failures:
        # Failed sources are reported as notes instead of aborting the whole run.
        lines.append("Source notes:")
        for failure in result.failures:
            lines.append(f"  - {failure.origin}: unavailable ({sanitize_text(failure.error or 'unknown error')})")
        lines.append("")

    lines.append("Fetch summary:")
    for source_result in result.source_results:
        status = "cached" if source_result.cached else "live"
        if source_result.error:
            status = "failed"
        lines.append(
            f"  - {source_result.origin}: {len(source_result.sources)} source(s), "
            f"{source_result.elapsed_ms:.0f} ms, {status}"
        )
    lines.append(f"Total: {result.elapsed_ms:.0f} ms")
    return "\n".join(lines)


def render_json(result: ResearchResult) -> str:
    # JSON mode exposes the same result data for scripts or automated grading.
    payload = {
        "question": result.question,
        "answer": result.answer.to_dict(),
        "source_results": [item.model_dump() for item in result.source_results],
        "elapsed_ms": result.elapsed_ms,
        "cache_enabled": result.cache_enabled,
        "failures": [item.model_dump() for item in result.failures],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def run_ask(args: argparse.Namespace) -> int:
    # Start from environment-backed defaults, then layer CLI overrides on top.
    settings = Settings.from_env()
    overrides = {}
    if args.timeout is not None:
        overrides["per_source_timeout_seconds"] = args.timeout
    if args.max_results is not None:
        overrides["max_sources_per_query"] = args.max_results
    if args.cache_ttl is not None:
        overrides["cache_ttl_seconds"] = args.cache_ttl
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})

    configure_logging(settings.log_level)
    # In normal mode Researcher builds the real AIService; in offline mode we
    # inject deterministic fetchers and a fake LLM for no-network execution.
    ai_service = build_offline_ai_service(settings) if args.offline else None
    researcher = Researcher(settings, ai_service=ai_service)
    try:
        result = await researcher.ask(
            args.question,
            sources=parse_sources(args.sources),
            use_cache=not args.no_cache,
            max_results=args.max_results,
        )
    except (ValidationError, ValueError) as exc:
        print(f"Invalid request: {exc}", file=sys.stderr)
        return 2
    except NoSourcesError as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI should display friendly error
        print(f"Unexpected failure: {exc}", file=sys.stderr)
        return 1

    # Keep rendering separate from business logic so both text and JSON formats
    # use the same ResearchResult object.
    print(render_json(result) if args.json else render_text(result))
    return 0


def run_cache_clear(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    removed = FileCacheStore(settings.cache_dir).clear()
    print(f"Removed {removed} cache file(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # argparse keeps the CLI self-documenting: `python -m researcher ask --help`.
    parser = argparse.ArgumentParser(
        prog="python -m researcher",
        description="Async research assistant with citations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer a research question.")
    ask.add_argument("question", help="Research question to answer.")
    ask.add_argument(
        "--sources",
        default=None,
        help="Comma-separated subset: wiki,wikipedia,arxiv,web. Default: all.",
    )
    ask.add_argument("--no-cache", action="store_true", help="Bypass source cache.")
    ask.add_argument("--offline", action="store_true", help="Use deterministic fake sources and LLM.")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    ask.add_argument("--max-results", type=int, default=None, help="Max results per source.")
    ask.add_argument("--cache-ttl", type=int, default=None, help="Override cache TTL in seconds.")
    ask.add_argument("--timeout", type=float, default=None, help="Per-source timeout in seconds.")
    ask.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR.")
    ask.set_defaults(func=lambda ns: asyncio.run(run_ask(ns)))

    clear = sub.add_parser("cache-clear", help="Delete JSON cache files.")
    clear.set_defaults(func=run_cache_clear)
    return parser


def main(argv: list[str] | None = None) -> int:
    # `argv` is injectable so tests can call main([...]) without touching sys.argv.
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
