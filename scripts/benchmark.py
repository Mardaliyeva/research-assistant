"""Offline parallel-vs-sequential benchmark for the research source layer.

This benchmark intentionally uses fake sources with fixed delays so it is
repeatable and does not need API keys or internet access.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai import Source
from ai.providers.base import LLMProvider
from researcher.config import Settings
from researcher.core.researcher import Researcher
from researcher.services.ai_service import AIService, FetchFn


# A fixed LLM keeps the benchmark focused on source-fetch concurrency.
class BenchmarkLLM(LLMProvider):
    def complete(self, prompt: str, *, json_schema=None, max_tokens: int = 1024) -> str:
        return "Benchmark answer [1]."


def make_fetcher(origin: str, delay: float) -> FetchFn:
    # Each fake fetcher sleeps for the same delay to make speedup easy to see.
    async def fetch(question: str, max_results: int, client: Any) -> list[Source]:
        await asyncio.sleep(delay)
        return [
            Source(
                title=f"{origin} benchmark source",
                url=f"https://example.com/{origin}",
                snippet=f"Benchmark snippet from {origin}.",
                origin=origin,
            )
        ]

    return fetch


async def run_parallel(iterations: int) -> list[float]:
    settings = Settings(retry_attempts=1, retry_base_delay_seconds=0)
    fetchers = {
        "wikipedia": make_fetcher("wikipedia", 0.20),
        "arxiv": make_fetcher("arxiv", 0.20),
        "web": make_fetcher("web", 0.20),
    }
    researcher = Researcher(settings, ai_service=AIService(settings, fetchers=fetchers, llm=BenchmarkLLM()))
    timings: list[float] = []
    for _ in range(iterations):
        # This measures the actual Researcher path, including concurrent orchestration.
        start = time.perf_counter()
        await researcher.ask("benchmark question", use_cache=False)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


async def run_sequential(iterations: int) -> list[float]:
    timings: list[float] = []
    for _ in range(iterations):
        # Baseline: simulate the three sources running one after another.
        start = time.perf_counter()
        for _origin in ("wikipedia", "arxiv", "web"):
            await asyncio.sleep(0.20)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


async def main() -> None:
    iterations = 5
    sequential = await run_sequential(iterations)
    parallel = await run_parallel(iterations)
    seq_avg = statistics.mean(sequential)
    # A healthy async implementation should be close to the longest single fetch.
    par_avg = statistics.mean(parallel)
    print(f"Sequential average: {seq_avg:.0f} ms over {iterations} runs")
    print(f"Parallel average:   {par_avg:.0f} ms over {iterations} runs")
    print(f"Speedup:            {seq_avg / par_avg:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
