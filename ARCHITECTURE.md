# Architecture (Phase VI)

## 1. Scale strategy: 500k+ records without manual work
- **Frontier, not scripts.** URLs are discovered (sitemaps, directory pagination, arXiv/OAI-PMH, RSS) and pushed to a durable
  queue (Redis Streams -> SQS/Kafka at scale), partitioned by domain. Workers are stateless and pull; scaling = more workers.
- **Per-domain governors.** Token-bucket per host (shared in Redis) + adaptive concurrency (AIMD: halve on 429/403, creep up on success)
  keeps throughput high without tripping blocks.
- **Cheap path first.** Official APIs/feeds/sitemaps and plain HTTP cover most volume; a Playwright pool is a *second tier* used only
  when a page is JS-rendered or returns a challenge. Browsers cost ~10-50x more, so route by observed need, cache the decision per domain.
- **LLM only where parsing fails.** Structured sources (arXiv Atom, JSON-LD, GitHub API) are parsed deterministically. The LLM handles
  unstructured pages; this cuts cost and removes hallucination risk for most fields.
- **Idempotent, resumable.** Every task is keyed by canonical URL; results are upserts; crashes just re-queue. Checkpoints per crawl job.
- **Throughput sketch:** 500k pages at 100 concurrent workers and ~1 s/page is ~1.4 h of fetch time; the LLM stage (not fetching) is the bottleneck, so budget tokens/min, not threads.

## 2. Handling 413s and 429s across thousands of concurrent extractions
- **Prevent 413:** estimate size before sending; `smart_truncate` keeps the head (title/lede) plus the densest paragraphs up to a per-provider budget.
  If a 413/token error still happens, the budget is **halved and retried** (floor at a minimum), then the tier is abandoned for the next provider.
- **Prevent 429:** a *global* per-provider limiter (requests/min and tokens/min in Redis) gates calls, so 429s are the exception, not the control loop.
- **React to 429:** exponential backoff with **full jitter**, honouring `Retry-After`; per-provider circuit breaker (3 failed tiers -> 60 s cooldown) so workers
  stop hammering a degraded provider and fall through the chain (Gemini Flash -> Groq Llama 3 -> DeepSeek).
- **Backpressure:** when all tiers are cooling down, tasks return to the queue with a delay instead of blocking workers.
- **Correctness gate:** output must parse, match the schema, and pass the grounding check (extracted strings must exist in the source), else retry once, then next tier.

## 3. Freshness: never process the same article/job twice across nodes
- **Atomic claim before work:** `SET url:<sha256(canonical_url)> 1 NX EX 7d` in Redis (SQLite `INSERT OR IGNORE` single-node). Only the node that wins proceeds.
- **Canonical URL:** lowercase host, drop fragments and tracking params (utm_*, fbclid, ...), trim trailing slash.
- **Content fingerprint:** a second key over the normalised first 2,000 chars catches the same story re-posted under a new URL.
- **Backstop:** unique constraint on `(source, canonical_url)` in the primary DB, so even a lost Redis key cannot create duplicate rows.
- **24 h guarantee:** dates parsed from JSON-LD/meta/`<time>`/relative text. A day-only date is classified *fresh* only if the whole day is inside the window,
  *uncertain* otherwise. Undated items use the first-seen heuristic and are flagged as such, never presented as verified-fresh.

## 4. Storage strategy
- **System of record: PostgreSQL** (JSONB for the raw canonical record + typed columns for keys, dates, source URL). Transactions, unique constraints, easy backfills, partitioning by `collectedAt`.
- **Vectors: pgvector** in the same database (embeddings for entity/paper similarity). One system until scale proves otherwise; move to a dedicated vector DB if recall/latency demands it.
- **Graph: start as edge tables** (`startup`-`product`, `paper`-`repo`, `founder`-`startup`) in Postgres; add **Neo4j** (or Apache AGE) as a read-optimised projection once multi-hop traversals
  ("founders of startups whose papers cite X") dominate. The graph is derived from Postgres, so it can be rebuilt.
- **Raw HTML/text** in object storage (S3/GCS) keyed by content hash: gives provenance (every record links to source URL + snapshot) and lets us re-extract without re-crawling.

## Anti-bot approach (Phase V)
Prefer legitimate access first: official APIs, feeds, sitemaps, bulk dumps, partnerships. For protected pages: realistic but honest rate limits, session/cookie reuse,
a real browser (Playwright) for JS-rendered content, rotating egress via a proxy provider, and respect for robots.txt/ToS decisions made explicitly per source.
Treat challenge pages as a signal to **slow down or switch tier**, not to loop. Record blocked domains and fall back to APIs or skip.
