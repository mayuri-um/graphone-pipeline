"""Multi-tier LLM extraction: fallback chain, 413-aware chunking, 429 backoff, grounding check."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("graphone.llm")


class LLMError(Exception): ...
class RateLimited(LLMError):
    def __init__(self, retry_after: Optional[float] = None):
        super().__init__("429 rate limited")
        self.retry_after = retry_after
class PayloadTooLarge(LLMError): ...
class ProviderUnavailable(LLMError): ...      # 5xx, timeouts, connection resets
class InvalidOutput(LLMError): ...            # bad JSON / schema / ungrounded values
class ExtractionFailed(LLMError): ...         # every tier exhausted


# ------------------------------------------------------------------ chunking
def _density(p: str) -> float:
    """Cheap proxy for 'semantically dense': digits, capitalised words, and length, minus nav-like lines."""
    words = p.split()
    if len(words) < 4:
        return 0.0
    caps = sum(w[:1].isupper() for w in words)
    digits = sum(any(c.isdigit() for c in w) for w in words)
    return (caps + 2 * digits) / len(words) + min(len(words), 60) / 200


def _split_long(p: str, limit: int = 800) -> list[str]:
    if len(p) <= limit:
        return [p]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", p):
        if len(cur) + len(sent) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    out = out + ([cur] if cur else [])
    # a "sentence" with no punctuation (tables, minified text) must still be splittable
    return [c for piece in out for c in (textwrap.wrap(piece, limit) if len(piece) > limit else [piece])]


def smart_truncate(text: str, max_chars: int) -> str:
    """Keep the head (title/lede), then the densest remaining paragraphs, in original order."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    paras = [q for p in text.split("\n") if p.strip() for q in _split_long(p.strip())]
    keep: set[int] = set()
    used, head_budget = 0, int(max_chars * 0.4)
    for i, p in enumerate(paras):
        if used + len(p) + 1 > head_budget:
            break
        keep.add(i)
        used += len(p) + 1
    for i in sorted((i for i in range(len(paras)) if i not in keep), key=lambda i: -_density(paras[i])):
        if used + len(paras[i]) + 1 <= max_chars:
            keep.add(i)
            used += len(paras[i]) + 1
    out = "\n".join(paras[i] for i in sorted(keep))
    return out[:max_chars] if out else text[:max_chars]


# ------------------------------------------------------------------ output checks
def parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise InvalidOutput("no JSON object in response")
    try:
        obj = json.loads(s[start:end + 1])
    except ValueError as e:
        raise InvalidOutput(f"bad JSON: {e}") from e
    if not isinstance(obj, dict):
        raise InvalidOutput("top-level JSON is not an object")
    return obj


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold().strip()


# keys that are legitimately derived/normalised rather than copied from the source
UNGROUNDED_OK = {"pricingModel", "role_family", "recordType", "schemaVersion"}


def grounding_validator(source_text: str,
                        extra: Optional[Callable[[dict], None]] = None) -> Callable[[dict], None]:
    """Reject any extracted string that does not literally appear in the source text.
    This is the anti-hallucination gate: names, titles, authors and URLs must be copied, not invented."""
    hay = _norm(source_text)

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and node.strip():
            if key in UNGROUNDED_OK or key.lower().endswith("date"):
                return
            if _norm(node) not in hay:
                raise InvalidOutput(f"ungrounded value for {key!r}: {node[:60]!r}")

    def check(obj: dict) -> None:
        walk(obj)
        if extra:
            extra(obj)

    return check


# ------------------------------------------------------------------ orchestrator
@dataclass
class Provider:
    name: str
    call: Callable[[str, str], Awaitable[str]]     # (system, user) -> raw text; raises LLMError subclasses
    max_input_chars: int = 24_000
    min_input_chars: int = 1_500
    max_attempts: int = 5


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0


class Orchestrator:
    def __init__(self, providers: list[Provider], *, base_delay: float = 1.0, max_delay: float = 30.0,
                 breaker_threshold: int = 3, breaker_cooldown: float = 60.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 rng: Callable[[], float] = random.random):
        self.providers = providers
        self.base_delay, self.max_delay = base_delay, max_delay
        self.threshold, self.cooldown = breaker_threshold, breaker_cooldown
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self._breakers = {p.name: _Breaker() for p in providers}

    def _delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        cap = min(self.max_delay, self.base_delay * 2 ** attempt)
        jittered = self._rng() * cap                       # "full jitter"
        return max(jittered, retry_after or 0.0)

    async def extract(self, text: str, *, system: str, instruction: str,
                      validate: Callable[[dict], None]) -> tuple[dict, str]:
        """Returns (parsed_object, provider_name_that_succeeded)."""
        for p in self.providers:
            br = self._breakers[p.name]
            if self._clock() < br.open_until:
                log.info("skip %s: circuit open", p.name)
                continue
            budget, invalid, inst = p.max_input_chars, 0, instruction
            for attempt in range(p.max_attempts):
                payload = smart_truncate(text, budget)
                try:
                    raw = await p.call(system, f"{inst}\n\nDOCUMENT:\n{payload}")
                    obj = parse_json(raw)
                    validate(obj)
                    br.failures = 0
                    return obj, p.name
                except RateLimited as e:
                    d = self._delay(attempt, e.retry_after)
                    log.warning("%s 429 (attempt %d) sleeping %.2fs", p.name, attempt, d)
                    await self._sleep(d)
                except PayloadTooLarge:
                    if budget <= p.min_input_chars:
                        log.warning("%s 413 at minimum budget; giving up on tier", p.name)
                        break
                    budget = max(p.min_input_chars, budget // 2)
                    log.warning("%s 413; shrinking input to %d chars", p.name, budget)
                except InvalidOutput as e:
                    invalid += 1
                    log.warning("%s invalid output: %s", p.name, e)
                    if invalid >= 2:
                        break
                    inst = (f"{instruction}\n\nYour previous answer was rejected: {e}. "
                            "Return only values copied verbatim from the document, or null.")
                except (ProviderUnavailable, asyncio.TimeoutError, ConnectionError, OSError) as e:
                    d = self._delay(attempt)
                    log.warning("%s unavailable (%s); sleeping %.2fs", p.name, e, d)
                    await self._sleep(d)
            br.failures += 1
            if br.failures >= self.threshold:
                br.open_until = self._clock() + self.cooldown
            log.error("tier %s exhausted; falling back", p.name)
        raise ExtractionFailed("all LLM tiers exhausted")
