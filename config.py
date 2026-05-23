"""Typed configuration for the research assistant.

The project deliberately avoids a heavyweight settings dependency. Values are
read from environment variables and, when present, a local `.env` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable


# Environment variables are strings, so these are the accepted truthy values.
TRUE_VALUES = {"1", "true", "yes", "on"}

# Only these source identifiers are accepted by the orchestration layer.
VALID_SOURCES = ("wikipedia", "arxiv", "web")
# User-friendly aliases are normalized before they reach the rest of the app.
SOURCE_ALIASES = {
    "wiki": "wikipedia",
    "wikipedia": "wikipedia",
    "arxiv": "arxiv",
    "web": "web",
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines from a `.env` file if it exists.

    Existing environment variables win. Quoted values are unquoted. Comments and
    blank lines are ignored.
    """

    file_path = Path(path)
    # Missing .env files are fine; deployment can rely entirely on real env vars.
    if not file_path.exists():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Split only on the first equals sign so values may contain '=' later.
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        # Do not overwrite variables already supplied by the shell/host system.
        os.environ.setdefault(key, value)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    # Centralized parsing gives consistent errors for every integer setting.
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    # Timeouts and delays are floats because they may need sub-second precision.
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def normalise_source(name: str) -> str:
    """Return canonical source name for a user-facing alias."""

    key = name.strip().lower()
    if key not in SOURCE_ALIASES:
        valid = ", ".join(SOURCE_ALIASES)
        raise ValueError(f"unknown source {name!r}; expected one of: {valid}")
    return SOURCE_ALIASES[key]


def parse_sources(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated source list into canonical source names."""

    if value is None:
        # No explicit selection means all available sources are enabled.
        return VALID_SOURCES
    if isinstance(value, str):
        pieces = [p for p in value.split(",") if p.strip()]
    else:
        pieces = list(value)
    if not pieces:
        return VALID_SOURCES
    seen: list[str] = []
    # Preserve user order while removing duplicates like 'wiki,wikipedia'.
    for piece in pieces:
        canonical = normalise_source(piece)
        if canonical not in seen:
            seen.append(canonical)
    return tuple(seen)


@dataclass(frozen=True)
class Settings:
    """Runtime settings with safe defaults for local development."""

    log_level: str = "INFO"
    cache_dir: Path = Path(".cache/researcher")
    cache_ttl_seconds: int = 86_400
    per_source_timeout_seconds: float = 10.0
    http_timeout_seconds: float = 15.0
    max_sources_per_query: int = 3
    max_question_chars: int = 1_000
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.25
    retry_max_delay_seconds: float = 2.0
    default_sources: tuple[str, ...] = VALID_SOURCES

    @classmethod
    def from_env(cls, *, dotenv_path: str | Path = ".env") -> "Settings":
        # Loading the .env first lets the helper functions read a complete env.
        load_dotenv(dotenv_path)
        return cls(
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
            cache_dir=Path(_env_str("CACHE_DIR", ".cache/researcher")),
            cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 86_400, min_value=0),
            per_source_timeout_seconds=_env_float(
                "PER_SOURCE_TIMEOUT_SECONDS", 10.0, min_value=0.1
            ),
            http_timeout_seconds=_env_float("HTTP_TIMEOUT_SECONDS", 15.0, min_value=0.1),
            max_sources_per_query=_env_int("MAX_SOURCES_PER_QUERY", 3, min_value=1),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", 1_000, min_value=20),
            retry_attempts=_env_int("RETRY_ATTEMPTS", 3, min_value=1),
            retry_base_delay_seconds=_env_float(
                "RETRY_BASE_DELAY_SECONDS", 0.25, min_value=0.0
            ),
            retry_max_delay_seconds=_env_float(
                "RETRY_MAX_DELAY_SECONDS", 2.0, min_value=0.0
            ),
            default_sources=parse_sources(os.getenv("DEFAULT_SOURCES")),
        )

    @property
    def cache_enabled_by_default(self) -> bool:
        # Evaluated as a property so tests can monkeypatch CACHE_ENABLED later.
        return os.getenv("CACHE_ENABLED", "true").strip().lower() in TRUE_VALUES
