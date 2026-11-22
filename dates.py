"""Publication-date extraction, normalisation and 24h-freshness classification."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

UTC = timezone.utc


@dataclass(frozen=True)
class ParsedDate:
    dt: datetime        # timezone-aware, UTC
    precision: str      # "second" or "day"
    source: str         # "relative" | "iso" | "rfc2822" | "text"


_UNIT_SECONDS = {
    "s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60,
    "h": 3600, "hr": 3600, "hour": 3600, "d": 86400, "day": 86400,
    "w": 604800, "week": 604800, "mo": 2592000, "month": 2592000,
    "y": 31536000, "yr": 31536000, "year": 31536000,
}
_REL = re.compile(
    r"\b(?P<n>an?|\d+)\s*(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|"
    r"months?|mo|years?|yrs?|[smhdwy])\.?\s+ago\b", re.I)
_PREFIX = re.compile(r"^\s*(?:posted|published|updated|date)\s*(?:on|at)?\s*:?\s*", re.I)
_ORDINAL = re.compile(r"(\d)(st|nd|rd|th)\b", re.I)
_FORMATS = [
    ("%B %d, %Y", "day"), ("%b %d, %Y", "day"), ("%d %B %Y", "day"), ("%d %b %Y", "day"),
    ("%Y/%m/%d", "day"), ("%B %d, %Y %I:%M %p", "second"), ("%b %d, %Y %I:%M %p", "second"),
]


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _midnight(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_relative(text: str, now: datetime) -> Optional[ParsedDate]:
    t = text.strip().lower()
    if re.search(r"\b(just now|moments? ago|a few seconds ago)\b", t):
        return ParsedDate(now, "second", "relative")
    m = _REL.search(t)
    if m:
        n = 1 if m["n"] in ("a", "an") else int(m["n"])
        u = m["u"].lower()
        u = u.rstrip("s") if len(u) > 1 else u
        return ParsedDate(now - timedelta(seconds=n * _UNIT_SECONDS[u]), "second", "relative")
    if re.search(r"\byesterday\b", t):
        return ParsedDate(_midnight(now) - timedelta(days=1), "day", "relative")
    if re.search(r"\btoday\b", t):
        return ParsedDate(_midnight(now), "day", "relative")
    return None


def parse_absolute(text: str) -> Optional[ParsedDate]:
    t = _ORDINAL.sub(r"\1", _PREFIX.sub("", text.strip()))
    if not t:
        return None
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        precision = "day" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t) else "second"
        return ParsedDate(_utc(dt), precision, "iso")
    except ValueError:
        pass
    try:
        return ParsedDate(_utc(parsedate_to_datetime(t)), "second", "rfc2822")
    except (TypeError, ValueError):
        pass
    for fmt, precision in _FORMATS:
        try:
            return ParsedDate(_utc(datetime.strptime(t, fmt)), precision, "text")
        except ValueError:
            continue
    return None


def parse_date(text: str, now: Optional[datetime] = None) -> Optional[ParsedDate]:
    now = _utc(now or datetime.now(UTC))
    return parse_relative(text, now) or parse_absolute(text)


# ---------------------------------------------------------------- HTML extraction
_META_KEYS = {
    "article:published_time", "og:article:published_time", "datepublished", "date",
    "pubdate", "publishdate", "dc.date.issued", "parsely-pub-date", "sailthru.date",
}
_TAG = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR = re.compile(r"([\w:.-]+)\s*=\s*[\"']([^\"']*)[\"']")
_LD = re.compile(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", re.I | re.S)
_TIME = re.compile(r"<time\b[^>]*\bdatetime=[\"']([^\"']+)[\"']", re.I)


def _walk_ld(node, out: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("datePublished", "dateCreated") and isinstance(v, str):
                out.append(v)
            else:
                _walk_ld(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_ld(v, out)


def date_candidates(html: str) -> list[str]:
    """Candidate date strings in priority order: JSON-LD, meta tags, <time>."""
    out: list[str] = []
    for blob in _LD.findall(html):
        try:
            _walk_ld(json.loads(blob), out)
        except ValueError:
            continue
    for tag in _TAG.findall(html):
        attrs = {k.lower(): v for k, v in _ATTR.findall(tag)}
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        if key in _META_KEYS and attrs.get("content"):
            out.append(attrs["content"])
    out.extend(_TIME.findall(html))
    return out


def extract_published(html: str, visible_text: str = "",
                      now: Optional[datetime] = None) -> Optional[ParsedDate]:
    """Best effort: structured data first, then relative strings like '2 hours ago'."""
    now = _utc(now or datetime.now(UTC))
    for c in date_candidates(html):
        parsed = parse_date(c, now)
        if parsed:
            return parsed
    if visible_text:
        return parse_relative(visible_text[:2000], now)
    return None


# ---------------------------------------------------------------- freshness
def classify_freshness(p: Optional[ParsedDate], now: Optional[datetime] = None,
                       window: timedelta = timedelta(hours=24)) -> str:
    """'fresh' | 'stale' | 'uncertain' | 'unknown'.

    A day-precision date only says *which UTC day*, so it can be provably fresh
    (the whole day lies inside the window), provably stale, or uncertain.
    """
    now = _utc(now or datetime.now(UTC))
    if p is None:
        return "unknown"
    if p.dt > now + timedelta(minutes=10):
        return "unknown"            # future-dated: bad parse or bad source
    cutoff = now - window
    if p.precision == "second":
        return "fresh" if p.dt >= cutoff else "stale"
    if p.dt >= cutoff:
        return "fresh"
    if p.dt + timedelta(days=1) <= cutoff:
        return "stale"
    return "uncertain"
