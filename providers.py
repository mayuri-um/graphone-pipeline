"""Real provider adapters (Gemini, Groq, DeepSeek). Untested offline: needs API keys + network.
Model names change often; override via env vars. Status codes are mapped to the orchestrator's exceptions."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from .llm import PayloadTooLarge, Provider, ProviderUnavailable, RateLimited


def _raise_for(status: int, body: str, retry_after: Optional[str]) -> None:
    if status == 429:
        raise RateLimited(float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None)
    if status == 413 or (status == 400 and any(k in body.lower() for k in ("token", "too large", "context length"))):
        raise PayloadTooLarge()
    if status >= 500:
        raise ProviderUnavailable(f"HTTP {status}")
    if status >= 400:
        raise ProviderUnavailable(f"HTTP {status}: {body[:200]}")


def openai_compatible(name: str, base_url: str, model: str, api_key: str,
                      max_input_chars: int, timeout: float = 60) -> Provider:
    async def call(system: str, user: str) -> str:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(f"{base_url}/chat/completions",
                              headers={"Authorization": f"Bearer {api_key}"},
                              json={"model": model, "temperature": 0,
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]}) as r:
                body = await r.text()
                _raise_for(r.status, body, r.headers.get("Retry-After"))
                return (await r.json())["choices"][0]["message"]["content"]
    return Provider(name, call, max_input_chars=max_input_chars)


def gemini(model: str, api_key: str, max_input_chars: int = 60_000, timeout: float = 60) -> Provider:
    async def call(system: str, user: str) -> str:
        import aiohttp
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(url, params={"key": api_key},
                              json={"systemInstruction": {"parts": [{"text": system}]},
                                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                                    "generationConfig": {"temperature": 0,
                                                         "responseMimeType": "application/json"}}) as r:
                body = await r.text()
                _raise_for(r.status, body, r.headers.get("Retry-After"))
                return (await r.json())["candidates"][0]["content"]["parts"][0]["text"]
    return Provider("gemini", call, max_input_chars=max_input_chars)


def default_chain() -> list[Provider]:
    """Gemini Flash -> Groq Llama 3 -> DeepSeek, configured from env."""
    chain = []
    if os.getenv("GEMINI_API_KEY"):
        chain.append(gemini(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), os.environ["GEMINI_API_KEY"]))
    if os.getenv("GROQ_API_KEY"):
        chain.append(openai_compatible("groq", "https://api.groq.com/openai/v1",
                                       os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                       os.environ["GROQ_API_KEY"], max_input_chars=20_000))
    if os.getenv("DEEPSEEK_API_KEY"):
        chain.append(openai_compatible("deepseek", "https://api.deepseek.com",
                                       os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                                       os.environ["DEEPSEEK_API_KEY"], max_input_chars=40_000))
    return chain
