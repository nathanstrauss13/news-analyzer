"""_resolve_and_verify_urls submits Gemini redirectors before other URLs, so a
batch deadline falls on pages (kept, fail-open) rather than on redirectors
(dropped). L'Oreal Oct 2026: 735 Gemini citations lost to the deadline.

Run from the repo root:  python3 tests/test_resolver_redirectors_first.py -v
"""
import os
import sys
import threading
import time
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("ANTHROPIC_API_KEY", "offline"); os.environ.setdefault("FLASK_SECRET_KEY", "offline")
import app as A  # noqa: E402

VERTEX = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/TOKEN%d"
PAGE = "https://example.com/page%d"


class RedirectorsFirst(unittest.TestCase):
    def test_redirectors_resolve_before_deadline(self):
        order, lock = [], threading.Lock()
        orig_vertex, orig_get, orig_head, orig_safe = A._resolve_vertex_redirect, A.requests.get, A.requests.head, A._is_safe_url

        class Slow:
            status_code = 200; headers = {}; url = ""
            def __init__(self, u): self.url = u
            def close(self): pass

        def fake_vertex(u, timeout=4):
            with lock: order.append("v")
            return "https://real.example/" + u[-6:]

        def slow_head(u, **k):
            if "example.com" in u:          # a page check; the vertex destination check is not a page submission
                with lock: order.append("p")
            time.sleep(0.5)
            return Slow(u)

        A._resolve_vertex_redirect, A.requests.head, A.requests.get, A._is_safe_url = fake_vertex, slow_head, slow_head, lambda u: True
        try:
            urls = [PAGE % i for i in range(20)] + [VERTEX % i for i in range(5)]
            out = A._resolve_and_verify_urls(urls, timeout=0.2, max_workers=5, batch_deadline=1.0)
        finally:
            A._resolve_vertex_redirect, A.requests.get, A.requests.head, A._is_safe_url = orig_vertex, orig_get, orig_head, orig_safe
        # every redirector resolved; the ones whose destination check stalled
        # fail open to the resolved page instead of being dropped
        self.assertEqual({out[VERTEX % i] for i in range(5)}, {"https://real.example/TOKEN%d" % i for i in range(5)})
        self.assertEqual(order[:5], ["v"] * 5)
        # the deadline fell on pages, which fail open (kept as themselves)
        self.assertTrue(all(out[PAGE % i] == PAGE % i for i in range(20)))


class LandingChallenge(unittest.TestCase):
    def test_known_walls_keep_the_cited_page(self):
        for wall in ("https://www.google.com/sorry/index?continue=https://www.youtube.com/watch%3Fv%3Dx&q=E",
                     "https://www.mercadolibre.com.mx/captcha/wall?go_url=https%3A%2F%2Fwww.mercadolibre.com.mx%2Fblog",
                     "https://www.walmart.com/blocked?url=L2Jyb3dzZS9iZWF1dHk",
                     "https://business.walmart.com/blocked?url=L2Jyb3dzZS9iZWF1dHk",
                     "https://www.ebay.de/splashui/challenge?ru=https%3A%2F%2Fwww.ebay.de%2Fitm%2F1"):
            self.assertEqual(A._redirect_landing_problem("https://example.com/page", wall), "challenge", wall)
        self.assertEqual(A._redirect_landing_problem("https://example.com/page", "https://example.com/page"), "")
        self.assertEqual(A._redirect_landing_problem("https://example.com/page", "https://example.com/404"), "error_page")


if __name__ == "__main__":
    unittest.main()
