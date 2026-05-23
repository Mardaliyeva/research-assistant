"""Input and output validation helpers."""

from __future__ import annotations

import re

# Remove non-printable characters that can make terminal output messy or unsafe.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Collapse tabs/newlines/repeated spaces to stable single spaces.
_WHITESPACE_RE = re.compile(r"\s+")


class ValidationError(ValueError):
    """Raised when user-provided input fails validation."""


def canonical_query(question: str) -> str:
    """Canonical representation used for cache keys."""

    # Lowercase and normalize whitespace so equivalent questions share cache hits.
    return _WHITESPACE_RE.sub(" ", question.strip().lower())


def sanitize_text(text: str) -> str:
    """Remove unsafe control characters while preserving readable whitespace."""

    return _CONTROL_CHARS_RE.sub("", text).strip()


def validate_question(question: str, *, max_chars: int) -> str:
    """Normalize and validate a research question."""

    # Sanitization runs before length checks so hidden control characters do not count.
    cleaned = sanitize_text(question)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ValidationError("question must not be empty")
    if len(cleaned) > max_chars:
        raise ValidationError(
            f"question is too long ({len(cleaned)} characters); limit is {max_chars}"
        )
    return cleaned
