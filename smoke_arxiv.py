"""Live smoke test for the paper vertical: arXiv -> GitHub link -> GitHub stars -> validated records.

Usage (from repo root):
    pip install aiohttp
    set GITHUB_TOKEN=ghp_...        (PowerShell: $env:GITHUB_TOKEN="ghp_...")   # optional but strongly advised
    python scripts/smoke_arxiv.py --pages 3 --category cs.AI

Writes papers.jsonl. Every field comes from arXiv / the GitHub API (no LLM, so nothing to hallucinate).
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphone.crawler import Fetcher, arxiv_page, github_stars  # noqa: E402
from graphone.schemas import make_record, validate               # noqa: E402


async def main(pages: int, size: int, category: str, out: str) -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("WARNING: no GITHUB_TOKEN set; GitHub allows only 60 requests/hour unauthenticated.")

    papers: list[dict] = []
    # arXiv asks for roughly 1 request per 3 seconds, so it gets its own polite fetcher.
    async with Fetcher(concurrency=1, per_host_delay=3.0) as arxiv_fetcher:
        for page in range(pages):
            batch = await arxiv_page(arxiv_fetcher, category, start=page * size, size=size)
            print(f"arXiv page {page}: {len(batch)} papers")
            if not batch:
                break
            papers.extend(batch)

    with_repo = [p for p in papers if p["github_url"]]
    print(f"{len(papers)} papers, {len(with_repo)} mention a GitHub repo")

    async with Fetcher(concurrency=10, per_host_delay=0.1) as gh_fetcher:
        stars = await asyncio.gather(*(github_stars(gh_fetcher, p["github_url"], token) for p in with_repo))
    for p, s in zip(with_repo, stars):
        p["github_stars"] = s

    ok = bad = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in papers:
            p.setdefault("github_stars", None)
            rec = make_record("RESEARCH_PAPER", "arXiv", p["paper_url"], p)
            errors = validate(rec)
            if errors:
                bad += 1
                print("invalid:", p["paper_url"], errors)
                continue
            ok += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    got_stars = sum(1 for p in with_repo if p["github_stars"] is not None)
    print(f"wrote {ok} valid records to {out} ({bad} invalid); stars resolved for {got_stars}/{len(with_repo)} repos")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--category", default="cs.AI")
    ap.add_argument("--out", default="papers.jsonl")
    a = ap.parse_args()
    asyncio.run(main(a.pages, a.size, a.category, a.out))
