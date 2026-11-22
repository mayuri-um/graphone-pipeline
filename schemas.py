"""Canonical record envelope + validation. Stdlib only."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "1.0"
PRICING_MODELS = {"FREE", "FREEMIUM", "PAID", "ENTERPRISE"}

REQUIRED = {
    "STARTUP": ["source.name", "source.url", "content.entityName", "collectedAt"],
    "PRODUCT": ["source.name", "source.url", "content.startupName",
                "content.pricingModel", "collectedAt"],
    "RESEARCH_PAPER": ["source.name", "source.url", "content.title", "content.authors",
                       "content.paper_url", "content.published_date", "collectedAt"],
    "JOB": ["source.name", "source.url", "content.company", "content.date",
            "content.is_remote", "content.role_family", "collectedAt"],
    "NEWS": ["source.name", "source.url", "content.title", "content.published_date",
             "content.full_text", "collectedAt"],
}

# Field hints that go into the LLM prompt (what the model is asked to return).
CONTENT_PROMPT = {
    "STARTUP": 'entityName (string), data.employeeCount (integer or null)',
    "PRODUCT": 'name (string), startupName (string), pricingModel (FREE|FREEMIUM|PAID|ENTERPRISE or null)',
    "RESEARCH_PAPER": 'title (string), authors (array of strings), published_date (as written in the text, or null)',
    "JOB": 'company (string), title (string), date (as written, or null), is_remote (boolean or null), role_family (string)',
    "NEWS": 'title (string), published_date (as written, or null)',
}


def get_path(obj: dict, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(record_type: str, source_name: str, source_url: str,
                content: dict, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": record_type,
        "source": {"name": source_name, "url": source_url},
        "content": content,
        "collectedAt": iso(now),
    }


def _is_iso(value: Any) -> bool:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate(record: dict) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    errors: list[str] = []
    rtype = record.get("recordType")
    if rtype not in REQUIRED:
        return [f"unknown recordType {rtype!r}"]
    if record.get("schemaVersion") != SCHEMA_VERSION:
        errors.append("bad schemaVersion")
    for path in REQUIRED[rtype]:
        if get_path(record, path) in (None, "", []):
            errors.append(f"missing {path}")
    url = get_path(record, "source.url") or ""
    if not str(url).startswith(("http://", "https://")):
        errors.append("source.url must be an http(s) URL (every record must trace to a source)")
    c = record.get("content", {})
    if "pricingModel" in c and c["pricingModel"] not in PRICING_MODELS:
        errors.append("pricingModel not in enum")
    if record.get("collectedAt") and not _is_iso(record["collectedAt"]):
        errors.append("collectedAt not ISO-8601")
    for key in ("published_date", "date"):
        if c.get(key) and not _is_iso(c[key]):
            errors.append(f"content.{key} not ISO-8601")
    if "authors" in c and not (isinstance(c["authors"], list) and all(isinstance(a, str) for a in c["authors"])):
        errors.append("authors must be a list of strings")
    if c.get("github_stars") is not None and not isinstance(c["github_stars"], int):
        errors.append("github_stars must be an integer")
    if "is_remote" in c and not isinstance(c["is_remote"], bool):
        errors.append("is_remote must be boolean")
    return errors
