"""compare_runs: per-class named-in-prose recount and the method caveat.

The August 12, 2026 Xsight baseline is hosted as a converted payload that
keeps each prompt's class (branded / competitor / unbranded) in
`prompt_classes` and a `method_note`. compare_runs against the October run
must report the board's figures (named in prose, per class, both runs) and
carry the caveat without blocking the comparison.

Run from the repo root:  python3 tests/test_compare_runs_classes.py -v
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools", "mcp_server"))

import signal_mcp as SM  # noqa: E402

Q_B = "What is Acme and what does it do?"
Q_C = "Which vendors ship 12.8T switch silicon?"
Q_U = "How do I avoid silicon lock-in?"
AGENTS = ["ChatGPT", "Claude", "Gemini", "Perplexity", "Grok"]


def _rows(named):
    """One answer per agent per prompt; `named` maps (prompt, agent) -> text."""
    out = []
    for q in (Q_B, Q_C, Q_U):
        for a in AGENTS:
            out.append({"llm": a, "prompt": q, "response": named.get((q, a), "nothing here"),
                        "citations": [], "grounded": True})
    return out


AUG = {"brand": "Acme", "brand_aliases": ["Acme"], "slug": "acme-aug",
       "prompt_sets": {"branded": [Q_B, Q_C], "unbranded": [Q_U]},
       "prompt_classes": {Q_B: "branded", Q_C: "competitor", Q_U: "unbranded"},
       "method_note": "earlier method", "comparability": {"caveat": "one pass each"},
       "all_responses": _rows({(Q_B, a): "Acme is a company" for a in AGENTS} |
                              {(Q_C, "ChatGPT"): "Acme ships it", (Q_U, "Grok"): "see https://acme.com/x"})}
OCT = {"brand": "Acme", "brand_aliases": ["Acme"], "slug": "acme-oct",
       "prompt_sets": {"branded": [Q_B, Q_C], "unbranded": [Q_U]},
       "all_responses": _rows({(Q_B, a): "Acme is a company" for a in AGENTS} |
                              {(Q_C, "ChatGPT"): "Acme ships it", (Q_C, "Gemini"): "Acme and others",
                               (Q_U, "Claude"): "Acme avoids it"})}


class CompareRunsClasses(unittest.TestCase):
    def setUp(self):
        SM._CFG_LOCAL.cfg = {"audits": [{"slug": "acme-aug", "label": "Aug", "run_date": "2026-08-12"},
                                        {"slug": "acme-oct", "label": "Oct", "run_date": "2026-10-26"}]}
        self._orig = SM._payload
        SM._payload = lambda slug: {"acme-aug": AUG, "acme-oct": OCT}[slug]
        SM._STALE.update({"checked": True, "status": "test", "warning": None})

    def tearDown(self):
        SM._payload = self._orig
        SM._CFG_LOCAL.cfg = None

    def test_by_class_and_caveat(self):
        out = SM.compare_runs("acme-aug", "acme-oct")
        bc = out["named_in_prose_by_class"]
        self.assertEqual(bc["class_source"], "prompt_classes recorded on run_a")
        self.assertEqual(bc["classes"]["branded"], {"prompts": 1, "run_a": {"named_in_prose": 5, "of": 5},
                                                    "run_b": {"named_in_prose": 5, "of": 5}})
        self.assertEqual(bc["classes"]["competitor"], {"prompts": 1, "run_a": {"named_in_prose": 1, "of": 5},
                                                       "run_b": {"named_in_prose": 2, "of": 5}})
        # URL-only presence (Grok, August) is not named in prose
        self.assertEqual(bc["classes"]["unbranded"], {"prompts": 1, "run_a": {"named_in_prose": 0, "of": 5},
                                                      "run_b": {"named_in_prose": 1, "of": 5}})
        self.assertIn("earlier method", out["method_caveat"])
        self.assertIn("one pass each", out["method_caveat"])
        # the unbranded-scope comparison still runs and still sees the URL-only state
        self.assertEqual(out["shared_prompts"], 1)
        self.assertEqual(out["brand_presence_total"]["run_a"], {"answers": 1, "of": 5})

    def test_without_classes_falls_back_to_prompt_sets(self):
        SM._payload = lambda slug: {"acme-aug": OCT, "acme-oct": OCT}[slug]
        out = SM.compare_runs("acme-aug", "acme-oct")
        self.assertNotIn("method_caveat", out)
        bc = out["named_in_prose_by_class"]
        self.assertEqual(bc["class_source"], "prompt_sets branded/unbranded on run_a")
        self.assertEqual(sorted(bc["classes"]), ["branded", "unbranded"])
        self.assertEqual(bc["classes"]["branded"]["run_b"], {"named_in_prose": 7, "of": 10})


if __name__ == "__main__":
    unittest.main()
