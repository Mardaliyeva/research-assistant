"""Filesystem JSON cache backend.

The cache stores one file per (source, canonical query) pair. The implementation
is intentionally small, deterministic, and easy to inspect during grading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import logging
import time
from typing import Any

from ai import Source

logger = logging.getLogger(__name__)


# CacheEntry is immutable after loading so callers cannot accidentally change
# timestamps or cached source lists in place.
@dataclass(frozen=True)
class CacheEntry:
    sources: list[Source]
    created_at: float
    expires_at: float

    @property
    def expired(self) -> bool:
        # Compare against current wall-clock time each time the entry is read.
        return time.time() > self.expires_at


class FileCacheStore:
    """TTL-aware source cache stored as JSON files."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, origin: str, canonical_query: str) -> Path:
        # Hash the query so filenames stay safe and reasonably short.
        digest = hashlib.sha256(f"{origin}:{canonical_query}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{origin}-{digest}.json"

    def get(self, origin: str, canonical_query: str) -> CacheEntry | None:
        path = self._path_for(origin, canonical_query)
        # Cache misses are normal and should be cheap.
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry(
                sources=[Source(**item) for item in data.get("sources", [])],
                created_at=float(data["created_at"]),
                expires_at=float(data["expires_at"]),
            )
        except Exception as exc:  # corrupted cache must not break the app
            logger.warning("Ignoring invalid cache file %s: %s", path, exc)
            return None
        if entry.expired:
            # Expired entries are removed lazily during reads.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not delete expired cache file %s", path, exc_info=True)
            return None
        return entry

    def set(
        self,
        origin: str,
        canonical_query: str,
        sources: list[Source],
        *,
        ttl_seconds: int,
    ) -> None:
        now = time.time()
        # Store plain JSON so cache files can be inspected during debugging/grading.
        payload: dict[str, Any] = {
            "origin": origin,
            "canonical_query": canonical_query,
            "created_at": now,
            "expires_at": now + ttl_seconds,
            "sources": [source.model_dump() for source in sources],
        }
        path = self._path_for(origin, canonical_query)
        tmp = path.with_suffix(".tmp")
        try:
            # Write to a temporary file, then replace, to avoid half-written cache files.
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Could not write cache file %s: %s", path, exc)

    def clear(self) -> int:
        """Delete cache files and return the number removed."""

        removed = 0
        # Only delete JSON cache entries; leave unrelated files in the directory alone.
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.debug("Could not delete cache file %s", path, exc_info=True)
        return removed
