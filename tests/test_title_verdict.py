"""A 200 whose title is a wall or a not-found template is not a readable page.

Oct 27, 2026: the Azzaro page pass carried 89 such rows as ok (Macy's and
Bloomingdale's "Access Denied" bodies run past the thin-page floor; the reader
proxy returns an upstream 404's text with a 200). Both the platform fetcher
(app._scrape_cited_page) and the offline tools (page_fetch.judge,
page_check_full.reclassify) now apply the same title verdict.

Run from the repo root:  python3 tests/test_title_verdict.py -v
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "mcp_server"))
os.environ.setdefault("ANTHROPIC_API_KEY", "offline"); os.environ.setdefault("FLASK_SECRET_KEY", "offline")

import page_fetch as PF  # noqa: E402
import page_check_full as PC  # noqa: E402
import app as A  # noqa: E402

WALLS = ["Access Denied - Macy's", "Pardon Our Interruption", "ULTA.com :: Our Apologies", "Robot or human?",
         "Just a moment...", "Too many requests"]
MISSING = ["404 Not Found", "Page not found - Remezcla", "Not Found | CHANEL", "– Page not found",
           "404 - Page Not Found | iScentive", "Oops! Something went wrong"]
CONTENT = ["Azzaro Chrome vs Legend vs United [Comparison 2026]", "Forbidden Smell is Sweet: IFRA and Fragrance",
           "The 404 Best Colognes", "Men", "", None]


class TitleVerdict(unittest.TestCase):
    def test_page_fetch_verdict(self):
        for t in WALLS: self.assertEqual(PF.title_verdict(t), "wall", t)
        for t in MISSING: self.assertEqual(PF.title_verdict(t), "missing", t)
        for t in CONTENT: self.assertIsNone(PF.title_verdict(t), t)

    def test_app_verdict_matches(self):
        for t in WALLS: self.assertEqual(A._unreadable_by_title(t), "wall page served as 200", t)
        for t in MISSING: self.assertEqual(A._unreadable_by_title(t), "page reports not found", t)
        for t in CONTENT: self.assertIsNone(A._unreadable_by_title(t), t)

    def test_judge_long_wall_is_blocked(self):
        body = "x " * 2000
        self.assertEqual(PF.judge(200, f"<html><head><title>Access Denied - Macy's</title></head><body>{body}</body></html>")[0], "blocked")
        self.assertEqual(PF.judge(200, f"<html><head><title>404 Not Found</title></head><body>{body}</body></html>")[0], "http 404")
        self.assertEqual(PF.judge(200, f"<html><head><title>Real page</title></head><body>{body}</body></html>")[0], "ok")

    def test_reclassify_in_place(self):
        checks = {"a": {"status": "ok", "brand_count": 3, "title": "Access Denied - Bloomingdales", "checked_date": "2026-10-27"},
                  "b": {"status": "ok", "brand_count": 0, "title": "404 Not Found", "checked_date": "2026-10-27"},
                  "c": {"status": "ok", "brand_count": 9, "title": "Azzaro", "checked_date": "2026-10-27"},
                  "d": {"status": "blocked", "brand_count": 0, "title": "", "checked_date": "2026-10-27"}}
        n = PC.reclassify(checks)
        self.assertEqual(n, {"wall": 1, "missing": 1})
        self.assertEqual(checks["a"], {"status": "blocked", "brand_count": 0, "title": "Access Denied - Bloomingdales",
                                       "checked_date": "2026-10-27", "note": "wall page served as 200"})
        self.assertEqual(checks["b"]["note"], "page reports not found")
        self.assertEqual(checks["c"]["status"], "ok"); self.assertEqual(checks["c"]["brand_count"], 9)
        self.assertNotIn("note", checks["d"])

    def test_scrape_direct_wall_title(self):
        class R:
            status_code = 200; headers = {}
            text = "<html><head><title>Access Denied - Macy's</title></head><body>" + "Azzaro " * 300 + "</body></html>"
        orig = A.requests.get
        A.requests.get = lambda *a, **k: R()
        try:
            r = A._scrape_cited_page("https://www.macys.com/shop/x", ["Azzaro"], timeout=1)
        finally:
            A.requests.get = orig
        self.assertEqual(r["status"], "blocked"); self.assertEqual(r["note"], "wall page served as 200"); self.assertEqual(r["counts"], {})


if __name__ == "__main__":
    unittest.main()
