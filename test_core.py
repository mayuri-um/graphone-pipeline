import asyncio
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphone import dates, entity_resolver as er, llm, schemas, seen_store  # noqa: E402

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


class DateTests(unittest.TestCase):
    def test_relative(self):
        for text, delta_h in [("2 hours ago", 2), ("Posted 45 minutes ago", 0.75), ("an hour ago", 1),
                              ("5h ago", 5), ("3 days ago", 72)]:
            p = dates.parse_date(text, NOW)
            self.assertAlmostEqual((NOW - p.dt).total_seconds() / 3600, delta_h, places=2, msg=text)

    def test_absolute(self):
        self.assertEqual(dates.parse_date("2026-11-21T09:30:00Z", NOW).dt.hour, 9)
        self.assertEqual(dates.parse_date("Nov 20th, 2026", NOW).dt.day, 20)
        self.assertEqual(dates.parse_date("Fri, 20 Nov 2026 08:00:00 GMT", NOW).dt.hour, 8)
        self.assertIsNone(dates.parse_date("sometime soon", NOW))

    def test_freshness(self):
        cls = lambda s: dates.classify_freshness(dates.parse_date(s, NOW), NOW)
        self.assertEqual(cls("2 hours ago"), "fresh")
        self.assertEqual(cls("3 days ago"), "stale")
        self.assertEqual(cls("2026-11-21"), "fresh")
        self.assertEqual(cls("2026-11-20"), "uncertain")     # day precision straddles the cutoff
        self.assertEqual(cls("2026-11-19"), "stale")
        self.assertEqual(dates.classify_freshness(None, NOW), "unknown")

    def test_html_extraction(self):
        html = ('<meta content="2026-11-21T10:00:00+00:00" property="article:published_time">'
                '<script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-11-20"}</script>')
        self.assertEqual(dates.extract_published(html, now=NOW).dt.date().isoformat(), "2026-11-20")  # JSON-LD first
        self.assertEqual(dates.extract_published("<p>x</p>", "Updated 3 hours ago", NOW).dt.hour, 9)


class EntityTests(unittest.TestCase):
    def setUp(self):
        self.r = er.EntityResolver()

    def test_seed_size(self):
        self.assertGreaterEqual(len(er.SEED_STARTUPS), 50)

    def test_variants(self):
        for raw in ["OpenAI", "OpenAI, Inc.", "Open AI", "open-ai", "OPENAI Inc"]:
            self.assertEqual(self.r.resolve(raw).canonical, "OpenAI", raw)
        self.assertEqual(self.r.resolve("HuggingFace").canonical, "Hugging Face")
        self.assertEqual(self.r.resolve("Weights and Biases").canonical, "Weights & Biases")
        self.assertEqual(self.r.resolve("Cursor").canonical, "Anysphere")
        self.assertEqual(self.r.resolve("Anthropiic").method, "fuzzy")

    def test_unknown_is_not_forced(self):
        res = self.r.resolve("Totally Unknown Startup Ltd")
        self.assertIsNone(res.canonical)
        self.assertEqual(self.r.canonical_or_clean("  Totally   Unknown  "), "Totally Unknown")
        self.assertEqual(len(self.r.log), 2)


class FakeProvider:
    def __init__(self, name, script, max_input_chars=10_000):
        self.script, self.calls, self.sizes = list(script), 0, []
        self.p = llm.Provider(name, self.call, max_input_chars=max_input_chars, min_input_chars=100)

    async def call(self, system, user):
        self.calls += 1
        self.sizes.append(len(user))
        step = self.script.pop(0) if self.script else self.script_last
        if isinstance(step, Exception):
            raise step
        return step


DOC = "Acme Robotics raised funds. " * 5 + "\n" + "filler words here " * 800


async def no_sleep(_):
    return None


class LLMTests(unittest.TestCase):
    def run_orch(self, providers, doc=DOC, validate=None):
        o = llm.Orchestrator([p.p for p in providers], sleep=no_sleep, rng=lambda: 0.5)
        return asyncio.run(o.extract(doc, system="s", instruction="i", validate=validate or (lambda o: None)))

    def test_429_then_success(self):
        a = FakeProvider("a", [llm.RateLimited(2), llm.RateLimited(), '{"x": 1}'])
        obj, who = self.run_orch([a])
        self.assertEqual((obj, who, a.calls), ({"x": 1}, "a", 3))

    def test_413_shrinks_payload(self):
        a = FakeProvider("a", [llm.PayloadTooLarge(), llm.PayloadTooLarge(), '{"ok": true}'])
        self.run_orch([a])
        self.assertLess(a.sizes[2], a.sizes[1])
        self.assertLess(a.sizes[1], a.sizes[0])

    def test_fallback_chain(self):
        a = FakeProvider("a", [llm.ProviderUnavailable("500")] * 10)
        b = FakeProvider("b", ['{"y": 2}'])
        obj, who = self.run_orch([a, b])
        self.assertEqual(who, "b")
        self.assertEqual(a.calls, 5)

    def test_all_fail(self):
        a = FakeProvider("a", [llm.RateLimited()] * 10)
        with self.assertRaises(llm.ExtractionFailed):
            self.run_orch([a])

    def test_hallucination_rejected(self):
        bad = '{"entityName": "Totally Made Up Corp"}'
        good = '{"entityName": "Acme Robotics"}'
        a = FakeProvider("a", [bad, bad])          # retried once, then tier abandoned
        b = FakeProvider("b", [good])
        v = llm.grounding_validator(DOC)
        obj, who = self.run_orch([a, b], validate=v)
        self.assertEqual((who, obj["entityName"]), ("b", "Acme Robotics"))

    def test_truncate_keeps_head_and_respects_limit(self):
        text = "Headline about OpenAI funding\n" + "noise " * 5000 + "\nRaised $500 million in 2026 Series C."
        out = llm.smart_truncate(text, 600)
        self.assertLessEqual(len(out), 600)
        self.assertTrue(out.startswith("Headline"))

    def test_parse_json_fences(self):
        self.assertEqual(llm.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(llm.InvalidOutput):
            llm.parse_json("sorry, I can't")


class MiscTests(unittest.TestCase):
    def test_seen_store(self):
        s = seen_store.SQLiteSeenStore()
        k1 = seen_store.url_key("https://Example.com/post/?utm_source=x#frag")
        k2 = seen_store.url_key("https://example.com/post")
        self.assertEqual(k1, k2)
        self.assertTrue(s.claim(k1))
        self.assertFalse(s.claim(k2))

    def test_schema_validation(self):
        rec = schemas.make_record("PRODUCT", "site", "https://x.io/p", {
            "startupName": "OpenAI", "pricingModel": "FREEMIUM"}, NOW)
        self.assertEqual(schemas.validate(rec), [])
        rec["content"]["pricingModel"] = "CHEAP"
        rec["source"]["url"] = ""
        errs = schemas.validate(rec)
        self.assertTrue(any("enum" in e for e in errs) and any("source.url" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
