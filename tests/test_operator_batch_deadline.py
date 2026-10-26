"""Regression: operator runs must never be cut by the self-serve batch deadline.

Slug 961b629f74 (Oct 26, 2026): an operator run of 100 prompts x 5 agents went
through /citation-audit as tier 'free', got the fixed 420s deadline, and shipped
123 of 500 answers. Coco Robotics (Oct 19) hit the same wall.

Run from the repo root:  python3 tests/test_operator_batch_deadline.py -v
"""
import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db"))

import app as A  # noqa: E402

OPS_EMAIL = "ops+deadline-test@innatec3.com"
N_AGENTS = len(A.TIER_CONFIG["free"]["llms"])


class DeadlineHelper(unittest.TestCase):
    def setUp(self):
        self._grounded = A.ALL_GROUNDED

    def tearDown(self):
        A.ALL_GROUNDED = self._grounded

    def test_operator_deadline_covers_every_allowed_run(self):
        # The invariant: any run the route accepts gets a deadline sized to drain it.
        for grounded in (True, False):
            A.ALL_GROUNDED = grounded
            for calls in range(N_AGENTS * 5, A.OPERATOR_MAX_CALLS + 1, N_AGENTS):
                d = A._llm_batch_deadline("free", calls, operator=True)
                self.assertGreaterEqual(d, calls * A.LLM_SECONDS_PER_CALL, (grounded, calls))
                self.assertLessEqual(d, A.LLM_BATCH_CEILING_SECONDS)

    def test_guard_never_exceeds_what_the_ceiling_can_drain(self):
        self.assertLessEqual(A.OPERATOR_MAX_CALLS * A.LLM_SECONDS_PER_CALL, A.LLM_BATCH_CEILING_SECONDS)

    def test_self_serve_deadlines_unchanged(self):
        A.ALL_GROUNDED = True
        self.assertEqual(A._llm_batch_deadline("free", 50), 420)
        self.assertEqual(A._llm_batch_deadline("free", 500), 420)
        self.assertEqual(A._llm_batch_deadline("lite", 24), 1200)
        A.ALL_GROUNDED = False
        self.assertEqual(A._llm_batch_deadline("free", 50), 180)
        self.assertEqual(A._llm_batch_deadline("lite", 24), 900)

    def test_the_961b629f74_run_would_not_hit_420s(self):
        A.ALL_GROUNDED = True
        self.assertGreater(A._llm_batch_deadline("free", 250, operator=True), 420)


class RouteGuard(unittest.TestCase):
    def setUp(self):
        self.client = A.app.test_client()
        self.calls = []
        self._orig = (A.run_citation_audit, A._send_audit_failure_email)

        def fake_run(*args, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("stub: no LLM calls in tests")

        A.run_citation_audit = fake_run
        A._send_audit_failure_email = lambda *a, **k: None

    def tearDown(self):
        A.run_citation_audit, A._send_audit_failure_email = self._orig

    def _post(self, n_branded, n_unbranded):
        sets = {"branded": [f"Branded question {i}?" for i in range(n_branded)],
                "unbranded": [f"Unbranded question {i}?" for i in range(n_unbranded)]}
        return self.client.post("/citation-audit", data={
            "problem_statement": "Acme wants to own industrial robotics",
            "email": OPS_EMAIL,
            "prompt_sets": json.dumps(sets),
        })

    def test_oversized_operator_set_is_refused_before_any_llm_call(self):
        r = self._post(40, 60)       # the 961b629f74 shape: 100 prompts, 500 calls
        self.assertEqual(r.status_code, 400)
        body = r.get_json()
        self.assertEqual(body["code"], "operator_run_too_large")
        self.assertEqual(body["max_calls"], A.OPERATOR_MAX_CALLS)
        self.assertEqual(self.calls, [])

    def test_operator_set_at_the_limit_runs_with_operator_deadline(self):
        n = A.OPERATOR_MAX_CALLS // N_AGENTS
        r = self._post(n // 2, n - n // 2)
        self.assertEqual(r.status_code, 200)
        r.get_data()                 # drain the SSE stream so the worker finishes
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0]["operator"])
        self.assertEqual(len(self.calls[0]["prompts_override"]), n)


if __name__ == "__main__":
    unittest.main()
