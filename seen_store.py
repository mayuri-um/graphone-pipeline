"""Cross-run / cross-node dedup so the same article or job is never processed twice."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src$)", re.I)


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k)))
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, query, ""))


def url_key(url: str) -> str:
    return "url:" + hashlib.sha256(canonical_url(url).encode()).hexdigest()


def content_key(text: str) -> str:
    """Fingerprint of the first ~2000 normalised chars; catches re-posts under a new URL."""
    norm = re.sub(r"\W+", " ", text.lower()).strip()[:2000]
    return "txt:" + hashlib.sha256(norm.encode()).hexdigest()


class SeenStore(Protocol):
    def claim(self, key: str) -> bool:
        """Atomically record `key`. True => first time (caller owns processing)."""


class SQLiteSeenStore:
    """Single-node store. INSERT OR IGNORE makes claim() atomic."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, first_seen TEXT)")

    def claim(self, key: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)",
                              (key, datetime.now(timezone.utc).isoformat()))
        self.db.commit()
        return cur.rowcount == 1


class RedisSeenStore:
    """Distributed store: SET key 1 NX EX ttl is atomic across crawler nodes. (Not exercised offline.)"""

    def __init__(self, client, ttl_seconds: int = 7 * 86400):
        self.r, self.ttl = client, ttl_seconds

    def claim(self, key: str) -> bool:
        return bool(self.r.set(key, 1, nx=True, ex=self.ttl))
