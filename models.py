"""Pydantic models owned by the SE layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai import AnswerWithCitations, Source

# A Literal type catches unsupported source names during development/type checking.
SourceName = Literal["wikipedia", "arxiv", "web"]


class SourceFetchResult(BaseModel):
    """Outcome of retrieving one source family for one question."""

    # Reject unexpected fields so API/test contracts stay strict.
    model_config = ConfigDict(extra="forbid")

    origin: SourceName
    sources: list[Source] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        # Convenience flag for UI/CLI code that only needs success/failure.
        return self.error is None


class ResearchResult(BaseModel):
    """Complete answer plus diagnostics for a research run."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: AnswerWithCitations
    source_results: list[SourceFetchResult]
    elapsed_ms: float
    cache_enabled: bool

    @property
    def sources(self) -> list[Source]:
        # Flatten per-origin batches for callers that only need all retrieved sources.
        merged: list[Source] = []
        for result in self.source_results:
            merged.extend(result.sources)
        return merged

    @property
    def failures(self) -> list[SourceFetchResult]:
        # Any SourceFetchResult with an error represents graceful degradation.
        return [result for result in self.source_results if result.error]

    @property
    def used_source_count(self) -> int:
        # Count citations the answer actually used, not every source retrieved.
        return len(self.answer.citations)
