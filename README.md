# Async Research Assistant — Topic 4

This repository implements the software-engineering layer around the provided `ai/` package for Topic 4. The `ai/` package is intentionally unchanged. The new `researcher/` package adds typed configuration, concurrent orchestration, per-source timeouts, graceful degradation, TTL caching, retries, validation, logging, CLI rendering, offline tests, Docker support, and benchmarking.

## Features

- Concurrent source fetching from Wikipedia, arXiv, and web search with `asyncio.gather`.
- Per-source timeout isolation: one slow or failing source does not block the others.
- Graceful degradation: answers are still synthesized from the sources that succeeded, and failures appear under `Source notes`.
- TTL filesystem cache keyed by `(source, canonicalized query)`.
- `--no-cache` flag for fresh fetches.
- Exponential backoff retries around every provided `ai.*` call.
- Input validation for empty and oversized questions.
- Output sanitization for unsafe control characters.
- Environment-driven logging and settings.
- Offline CLI mode for no-key demonstrations.
- Offline pytest suite with coverage above the required 60% threshold.

## Project layout

```text
.
├── ai/                         # Provided AI module; not modified
├── researcher/                 # Student SE layer
│   ├── config.py               # Typed environment settings
│   ├── models.py               # Pydantic models for SE-layer results
│   ├── cli.py                  # `python -m researcher ask ...`
│   ├── concurrency/            # Async gather, timeouts, degradation
│   ├── core/                   # Research workflow
│   ├── services/               # AI wrappers, cache service, retries
│   └── storage/                # Filesystem JSON cache backend
├── scripts/benchmark.py        # Offline parallel vs sequential benchmark
├── tests/                      # Provided smoke tests + student tests
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill only the providers you plan to use:

```bash
cp .env.example .env
```

For a no-key demo, use `--offline`. For live synthesis and live web search, set values such as:

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
```

DuckDuckGo can be used without an API key, but you must install `duckduckgo-search` if you choose it.

## CLI usage

No-key offline run:

```bash
python -m researcher ask "What is photosynthesis and what are its main stages?" --offline --no-cache
```

Live run using all configured providers:

```bash
python -m researcher ask "How do transformer-based language models handle long context windows?"
```

Restrict sources:

```bash
python -m researcher ask "How does CRISPR-Cas9 work?" --sources wiki,arxiv
```

Bypass cache:

```bash
python -m researcher ask "What is the current state of fusion energy research?" --no-cache
```

JSON output:

```bash
python -m researcher ask "What caused the 2008 financial crisis?" --json
```

Clear cache:

```bash
python -m researcher cache-clear
```

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `CACHE_DIR` | `.cache/researcher` | Filesystem cache directory. |
| `CACHE_TTL_SECONDS` | `86400` | Cache TTL in seconds. Set `0` to avoid writing. |
| `PER_SOURCE_TIMEOUT_SECONDS` | `10` | Timeout for each individual source task. |
| `HTTP_TIMEOUT_SECONDS` | `15` | Shared `httpx.AsyncClient` timeout. |
| `MAX_SOURCES_PER_QUERY` | `3` | Max results requested from each source. |
| `MAX_QUESTION_CHARS` | `1000` | Oversized question rejection limit. |
| `RETRY_ATTEMPTS` | `3` | Attempts for fetch and synthesize calls. |
| `RETRY_BASE_DELAY_SECONDS` | `0.25` | First retry delay. |
| `RETRY_MAX_DELAY_SECONDS` | `2.0` | Maximum retry delay. |
| `DEFAULT_SOURCES` | `wikipedia,arxiv,web` | Default source list. |

## Testing

Run all tests:

```bash
pytest -q
```

Run with coverage:

```bash
pytest --cov=researcher --cov-report=term-missing -q
```

Current local result:

```text
25 passed
researcher coverage: 80%
```

The provided contract tests in `tests/test_ai_smoke.py` are still present and unchanged.

## Parallel vs sequential timing

The benchmark uses deterministic offline fake sources with a fixed 200 ms delay per source. This avoids network variance and demonstrates the concurrency behavior required by the rubric.

```bash
python scripts/benchmark.py
```

Measured result in this environment:

```text
Sequential average: 602 ms over 5 runs
Parallel average:   230 ms over 5 runs
Speedup:            2.61x
```

The speedup is expected because sequential fetching waits for three source delays, while parallel fetching waits mostly for the slowest source plus orchestration overhead.

## Docker

Build:

```bash
docker build -t async-research-assistant .
```

Run offline, no API keys required:

```bash
docker run --rm async-research-assistant
```

Run live with `.env`:

```bash
docker run --rm --env-file .env async-research-assistant \
  python -m researcher ask "What is the current state of fusion energy research?" --no-cache
```

## Design notes

The SE layer never modifies the provided `ai/` package and never calls provider SDKs or source APIs directly. All live source fetching goes through `ai.fetch_wikipedia`, `ai.fetch_arxiv`, `ai.fetch_web`, and synthesis goes through `ai.synthesize`.

The cache key canonicalizes whitespace and casing, so `WHAT IS AI?` and `what is ai?` hit the same cached entry. Cache corruption is treated as a miss, so bad local state does not break a run.
