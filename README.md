# GraphOne / FrontierAtlas: AI Engineer demo task (starter)

Async ingestion pipeline: crawl -> extract (LLM, with fallback) -> validate -> resolve entities -> dedupe -> export.

## Status (read this first)
| Piece | State |
|---|---|
| `schemas.py`: record envelope + validation (STARTUP / PRODUCT / RESEARCH_PAPER / JOB / NEWS) | done, unit-tested |
| `dates.py`: relative/absolute/meta/JSON-LD date parsing + 24h freshness (fresh/stale/uncertain/unknown) | done, unit-tested |
| `entity_resolver.py`: 50-company seed, exact/alias/fuzzy, mapping log | done, unit-tested |
| `llm.py`: fallback chain, 429 backoff+jitter, 413 shrink-and-retry, density-aware truncation, circuit breaker, **grounding check** | done, tested with fake providers |
| `seen_store.py`: URL canonicalisation + atomic claim (SQLite / Redis) | done, SQLite tested |
| `providers.py`, `crawler.py`: Gemini/Groq/DeepSeek adapters, aiohttp fetcher, arXiv parser, GitHub stars | written, **not run against live services** |
| Phase I/II source selection, run script, Google Sheets export | **TODO** (you choose sources, wire the pieces) |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && playwright install chromium   # playwright optional
export GEMINI_API_KEY=... GROQ_API_KEY=... DEEPSEEK_API_KEY=... GITHUB_TOKEN=...
python -m unittest discover -s tests -v                           # stdlib only, no network
```

## Intended flow
1. `crawler.Fetcher` pulls pages/feeds (arXiv API for papers; sitemaps/RSS/listing pages for news & jobs).
2. `seen_store` claims `url_key(url)` first; duplicates are dropped before any LLM spend.
3. `dates.extract_published` -> `classify_freshness`. `fresh` is kept. `uncertain`/`unknown` go through the
   documented heuristic (first-seen-since-last-run, flagged `date_source=first_seen`), never silently labelled fresh.
4. `llm.Orchestrator.extract(...)` with `grounding_validator(source_text)`: every name/title/URL must literally
   appear in the source, which is the guard against the "hallucination = disqualification" rule.
5. `EntityResolver.canonical_or_clean` for company/startup names; the log becomes the *Entity Mapping Log* tab.
6. `schemas.validate`, then export to the six sheet tabs.

Design write-up: `docs/ARCHITECTURE.md`.
