"""AI-assistant referrals and non-branded search in the board pull and connector
(Oct 27, 2026). Pure helpers on synthetic rows; no Google call.

Run from the repo root:  python3 tests/test_board_ai_nonbranded.py -v
"""
import importlib.util
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools", "mcp_server"))
_spec = importlib.util.spec_from_file_location("board_fetch_google", os.path.join(_ROOT, "tools", "board", "fetch_google.py"))
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
import board_tools as BT  # noqa: E402


class AiReferrals(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(FG.ai_label("chatgpt.com", "Referral"), "chatgpt.com")
        self.assertEqual(FG.ai_label("chat.openai.com", ""), "chatgpt.com")
        self.assertEqual(FG.ai_label("gemini.google.com", "Referral"), "gemini.google.com")
        self.assertEqual(FG.ai_label("www.perplexity.ai", ""), "perplexity.ai")
        self.assertEqual(FG.ai_label("claude.ai", ""), "claude.ai")
        self.assertEqual(FG.ai_label("copilot.microsoft.com", ""), "copilot")
        self.assertEqual(FG.ai_label("some.newassistant.com", "AI Assistant"), "others")
        self.assertIsNone(FG.ai_label("google", "Organic Search"))
        self.assertIsNone(FG.ai_label("linkedin.com", "Referral"))

    def test_split_sums_and_orders(self):
        rows = [("chatgpt.com", "Referral", 10), ("chat.openai.com", "Referral", 5), ("google", "Organic Search", 500),
                ("perplexity.ai", "AI Assistant", 3), ("x.example", "AI Assistant", 2), ("linkedin.com", "Referral", 40)]
        self.assertEqual(FG.ai_referrals_split(rows), {"chatgpt.com": 15, "perplexity.ai": 3, "others": 2})


class NonBranded(unittest.TestCase):
    def test_classes(self):
        for q, k in {"xsight labs": "brand", "x sight labs": "brand", "x-sight": "brand", "excite labs": "brand", "אקסייט לאבס": "brand",
                     "XsightLabs": "brand", "avigdor willenz": "people", "yossi meyouhas": "people", "x2 switch": "product",
                     "e1 dpu": "product", "e1l": "product", "dpu vs smartnic": None, "excited labs": None, "x20 switch": None,
                     "insight labs": None, "nvme1 drive": None}.items():
            self.assertEqual(FG.classify_query(q), k, q)

    def test_split_with_unattributed_remainder(self):
        rows = [("xsight labs", 50, 500), ("dpu vs smartnic", 30, 900), ("Xsight E1", 10, 100), ("switch silicon 12.8t", 5, 300),
                ("avigdor willenz", 8, 80), ("x2 switch", 7, 70)]
        sp = FG.split_queries(rows, total_clicks=135, total_impressions=2650)
        self.assertEqual(sp["classes"]["brand"], {"clicks": 60, "impressions": 600, "queries": 2})
        self.assertEqual(sp["classes"]["people"], {"clicks": 8, "impressions": 80, "queries": 1})
        self.assertEqual(sp["classes"]["product"], {"clicks": 7, "impressions": 70, "queries": 1})
        # product counts as non-branded (discovery); brand and people are excluded
        self.assertEqual(sp["nonBranded"], {"clicks": 42, "impressions": 1270, "queries": 3})
        self.assertEqual(sp["excluded"], {"clicks": 68, "impressions": 680, "queries": 3})
        self.assertEqual(sp["excludes"], ["brand", "people"])
        self.assertEqual(sp["unattributed"], {"clicks": 25, "impressions": 700})
        self.assertEqual(sp["total"], {"clicks": 135, "impressions": 2650}); self.assertEqual(sp["queries"], 6)
        self.assertEqual(sp["total"]["clicks"], sp["nonBranded"]["clicks"] + sp["excluded"]["clicks"] + sp["unattributed"]["clicks"])
        self.assertTrue(FG.is_excluded("xsight labs")); self.assertTrue(FG.is_excluded("yossi meyouhas"))
        self.assertFalse(FG.is_excluded("x2 switch")); self.assertFalse(FG.is_excluded("dpu vs smartnic"))

    def test_unattributed_never_negative(self):
        sp = FG.split_queries([("a", 10, 10)], 5, 5)
        self.assertEqual(sp["unattributed"], {"clicks": 0, "impressions": 0})


class Connector(unittest.TestCase):
    def setUp(self):
        BT._CFG.profile = {"boards": [{"slug": "b1", "label": "Board"}]}
        self.live = {"pulled": "2026-10-27T08:25",
                     "ga4": {"months": [], "aiReferrals": {"months": [{"month": "Sep 26", "start": "2026-09-01", "end": "2026-09-30", "data_through": "2026-09-30",
                                                                      "partial": False, "bySource": {"chatgpt.com": 4}, "total": 4, "ai_channel_present": False}],
                                                           "basis": FG.AI_REFERRALS_BASIS, "sources": list(FG.AI_ASSISTANT_SOURCES) + ["others"]}},
                     "searchConsole": {"months": [], "nonBranded": {"months": [{"month": "Sep 26", "start": "2026-09-01", "end": "2026-09-30", "data_through": "2026-09-30", "partial": False,
                                                                               "nonBranded": {"clicks": 35, "impressions": 1200, "queries": 2}, "classes": {"brand": {"clicks": 60, "impressions": 600, "queries": 2}},
                                                                               "unattributed": {"clicks": 25, "impressions": 700}, "total": {"clicks": 120, "impressions": 2500}, "queries": 4}],
                                                                    "topQueries90d": [{"query": "dpu vs smartnic", "clicks": 30, "impressions": 900, "position": 4.2}],
                                                                    "brandedShare90dAllQueries": 0.632, "window_90d": {"start": "2026-07-27", "end": "2026-10-25"}, "basis": FG.NON_BRANDED_BASIS}},
                     "pace": {"searchConsole": None, "nonBrandedClicks": None, "ga4": None, "aiReferrals": None, "blogPosts": None, "note": "n"}}
        self._orig = dict(BT.LOADERS)
        BT.LOADERS["live"] = lambda slug: self.live
        BT.LOADERS["data"] = lambda slug: (None, None)

    def tearDown(self):
        BT.LOADERS.update(self._orig); BT._CFG.profile = None

    def test_site_traffic_carries_ai_referrals(self):
        out = BT.get_site_traffic()
        self.assertTrue(out["ai_referrals"]["available"])
        self.assertEqual(out["ai_referrals"]["months"][0]["bySource"], {"chatgpt.com": 4})
        self.assertIn("referrer match", out["ai_referrals"]["basis"])
        self.assertEqual(out["as_of"], "2026-10-27T08:25"); self.assertEqual(out["contract"], BT.CONTRACT)

    def test_search_carries_non_branded(self):
        out = BT.get_search()
        nb = out["non_branded"]
        self.assertTrue(nb["available"])
        self.assertEqual(nb["months"][0]["unattributed"], {"clicks": 25, "impressions": 700})
        self.assertEqual(nb["top_non_branded_queries_90d"][0]["query"], "dpu vs smartnic")
        self.assertIn("unattributed", nb["basis"])

    def test_old_feed_says_unavailable(self):
        del self.live["ga4"]["aiReferrals"]; del self.live["searchConsole"]["nonBranded"]
        self.assertFalse(BT.get_site_traffic()["ai_referrals"]["available"])
        self.assertFalse(BT.get_search()["non_branded"]["available"])

    def test_pace_passes_new_keys(self):
        out = BT.get_pace()
        self.assertIn("nonBrandedClicks", out); self.assertIn("aiReferrals", out)


if __name__ == "__main__":
    unittest.main()
