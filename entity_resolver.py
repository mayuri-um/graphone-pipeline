"""Deterministic entity resolution: messy name -> canonical name from a seed list."""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co",
    "company", "gmbh", "plc", "pbc", "sa", "ag", "bv", "pte", "pvt", "private",
}

# canonical name -> known aliases (optional)
SEED_STARTUPS: dict[str, list[str]] = {
    "OpenAI": [], "Anthropic": [], "Mistral AI": ["Mistral"], "Cohere": [],
    "Hugging Face": ["HF"], "Stability AI": [], "Perplexity AI": ["Perplexity"],
    "Scale AI": [], "Databricks": [], "Runway": ["Runway ML", "RunwayML"],
    "Midjourney": [], "Character.AI": ["Character AI"], "Inflection AI": ["Inflection"],
    "Adept": ["Adept AI"], "xAI": ["x.ai"], "Together AI": ["Together Computer"],
    "Anyscale": [], "Weights & Biases": ["W&B", "wandb"], "Pinecone": [],
    "Weaviate": [], "Chroma": ["Chroma DB"], "LangChain": [], "LlamaIndex": ["Llama Index"],
    "Replit": [], "Anysphere": ["Cursor"], "Cognition": ["Cognition AI", "Cognition Labs"],
    "Glean": [], "Jasper": ["Jasper AI"], "Synthesia": [], "ElevenLabs": ["Eleven Labs"],
    "Sierra": ["Sierra AI"], "Harvey": ["Harvey AI"], "Hebbia": [], "Writer": [],
    "AI21 Labs": ["AI21"], "Reka AI": ["Reka"], "Groq": [], "Cerebras": ["Cerebras Systems"],
    "SambaNova": ["SambaNova Systems"], "Lambda": ["Lambda Labs"], "CoreWeave": [],
    "Modal": ["Modal Labs"], "Replicate": [], "Fireworks AI": ["Fireworks"],
    "Baseten": [], "Poolside": ["Poolside AI"], "Magic": ["Magic AI", "Magic.dev"],
    "Luma AI": ["Luma Labs"], "Pika": ["Pika Labs"], "Suno": ["Suno AI"],
    "Black Forest Labs": ["BFL"], "DeepSeek": ["Deep Seek"], "Moonshot AI": ["Moonshot"],
}


def normalize(name: str) -> str:
    """Lowercase, strip accents/punctuation/legal suffixes; '&' -> 'and'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    tokens = s.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    return " ".join(tokens)


def compact(name: str) -> str:
    return normalize(name).replace(" ", "")


@dataclass(frozen=True)
class Resolution:
    raw: str
    canonical: Optional[str]     # None => unresolved
    method: str                  # exact | alias | fuzzy | unresolved
    score: float


class EntityResolver:
    def __init__(self, seed: dict[str, list[str]] = SEED_STARTUPS,
                 fuzzy_threshold: float = 0.92, margin: float = 0.03):
        self.fuzzy_threshold = fuzzy_threshold
        self.margin = margin
        self._canonical: dict[str, str] = {}   # compact(canonical) -> canonical
        self._alias: dict[str, str] = {}       # compact(alias) -> canonical
        for canon, aliases in seed.items():
            self._canonical[compact(canon)] = canon
            for a in aliases:
                self._alias[compact(a)] = canon
        self.log: list[Resolution] = []        # feeds the "Entity Mapping Log" tab

    def resolve(self, raw: str) -> Resolution:
        key = compact(raw)
        if key in self._canonical:
            res = Resolution(raw, self._canonical[key], "exact", 1.0)
        elif key in self._alias:
            res = Resolution(raw, self._alias[key], "alias", 1.0)
        else:
            res = self._fuzzy(raw, key)
        self.log.append(res)
        return res

    def _fuzzy(self, raw: str, key: str) -> Resolution:
        if len(key) < 5:       # short strings fuzzy-match too easily
            return Resolution(raw, None, "unresolved", 0.0)
        scored = sorted(
            ((difflib.SequenceMatcher(None, key, k).ratio(), c)
             for k, c in {**self._canonical, **self._alias}.items()),
            reverse=True)
        best = scored[0]
        second = next((s for s, c in scored[1:] if c != best[1]), 0.0)
        if best[0] >= self.fuzzy_threshold and best[0] - second >= self.margin:
            return Resolution(raw, best[1], "fuzzy", round(best[0], 3))
        return Resolution(raw, None, "unresolved", round(best[0], 3))

    def canonical_or_clean(self, raw: str) -> str:
        """Canonical name if known, else the whitespace-cleaned raw name (never invented)."""
        res = self.resolve(raw)
        return res.canonical or " ".join(raw.split())
