"""Async crawler building blocks. Needs `aiohttp` (and optionally `playwright`). Not run offline.

Design: one shared session, a global concurrency semaphore, a per-host politeness gate,
bounded retries with jittered backoff, and an escalation path to a real browser when a
page looks like a bot-challenge instead of content."""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional
from urllib.parse import urlsplit

log = logging.getLogger("graphone.crawler")

CHALLENGE_MARKERS = ("just a moment", "cf-chl", "captcha-delivery.com", "datadome", "attention required")
GITHUB_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git|[/#?)\s\"'<]|$)", re.I)


class Fetcher:
    def __init__(self, concurrency: int = 50, per_host_delay: float = 1.0, retries: int = 4,
                 user_agent: str = "GraphOneDemoBot/0.1 (+contact: you@example.com)"):
        self.sem = asyncio.Semaphore(concurrency)
        self.per_host_delay, self.retries, self.ua = per_host_delay, retries, user_agent
        self._host_lock: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._host_last: dict[str, float] = defaultdict(float)
        self._session = None

    async def __aenter__(self):
        import aiohttp
        self._session = aiohttp.ClientSession(headers={"User-Agent": self.ua},
                                              timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    async def _polite(self, host: str) -> None:
        async with self._host_lock[host]:
            wait = self._host_last[host] + self.per_host_delay - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = time.monotonic()

    @staticmethod
    def looks_blocked(status: int, body: str) -> bool:
        return status in (403, 429, 503) and any(m in body[:5000].lower() for m in CHALLENGE_MARKERS)

    async def get(self, url: str, **kw) -> tuple[int, str]:
        host = urlsplit(url).netloc
        for attempt in range(self.retries + 1):
            async with self.sem:
                await self._polite(host)
                try:
                    async with self._session.get(url, **kw) as r:
                        body = await r.text(errors="replace")
                        if r.status < 400:
                            return r.status, body
                        if self.looks_blocked(r.status, body):
                            log.warning("challenge page at %s -> escalate to browser", url)
                            return r.status, body          # caller decides: browser tier / skip / API
                        if r.status not in (429, 500, 502, 503, 504):
                            return r.status, body
                except (asyncio.TimeoutError, OSError) as e:
                    log.warning("fetch error %s: %s", url, e)
            await asyncio.sleep(random.random() * min(30, 2 ** attempt))
        return 599, ""

    async def get_rendered(self, url: str) -> str:
        """Browser tier for JS-heavy or challenge-protected pages (slow/expensive: use sparingly)."""
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=45_000)
            html = await page.content()
            await browser.close()
            return html


# ---------------------------------------------------------------- research papers
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    """Parse export.arxiv.org Atom results. Every field comes straight from the feed (no LLM)."""
    out = []
    for e in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        abs_url = (e.findtext(f"{ATOM}id") or "").strip()
        text_blob = " ".join([e.findtext(f"{ATOM}summary") or "", e.findtext(f"{ARXIV}comment") or ""])
        gh = GITHUB_RE.search(text_blob)
        out.append({
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
            "paper_url": abs_url,
            "published_date": e.findtext(f"{ATOM}published"),
            "github_url": f"https://github.com/{gh.group(1)}/{gh.group(2)}" if gh else None,
        })
    return out


async def arxiv_page(f: Fetcher, category: str = "cs.AI", start: int = 0, size: int = 100) -> list[dict]:
    status, body = await f.get("http://export.arxiv.org/api/query", params={
        "search_query": f"cat:{category}", "start": start, "max_results": size,
        "sortBy": "submittedDate", "sortOrder": "descending"})
    return parse_arxiv_feed(body) if status == 200 else []


async def github_stars(f: Fetcher, github_url: str, token: Optional[str] = None) -> Optional[int]:
    """Current star count via the GitHub REST API (60 req/h unauthenticated; pass a token for 5,000/h).
    Returns None if the repo is missing: never guess."""
    m = GITHUB_RE.match(github_url)
    if not m:
        return None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    status, body = await f.get(f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}", headers=headers)
    if status != 200:
        return None
    import json
    return json.loads(body).get("stargazers_count")
