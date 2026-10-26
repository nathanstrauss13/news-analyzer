"""Panel timing: every run logs how long answer collection took and the seconds
per answer; operator runs also get the line in the progress stream.

The stream carries no timestamps, so the first L'Oreal job (feff6218bc, Oct 26,
2026) could only bound speed per call by whole-job wall time.

Run from the repo root:  python3 tests/test_panel_timing.py -v
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db"))

import app as A  # noqa: E402


class _Stop(Exception):
    pass


class TimingLine(unittest.TestCase):
    def test_line_reports_count_per_answer_and_deadline_share(self):
        self.assertEqual(A._panel_timing_line(1500.0, 250, 250, 3000),
                         "Answer collection: 250/250 in 1500s (6.0s per answer; 50% of the 3000s deadline)")

    def test_zero_answers_does_not_divide_by_zero(self):
        self.assertIn("no answers", A._panel_timing_line(420.0, 0, 50, 420))


class RunEmitsTiming(unittest.TestCase):
    def setUp(self):
        self._orig = (A._call_llm, A._claude_msg)
        A._call_llm = lambda provider, prompt: ("A plain grounded answer about delivery robots.", True)

        def no_claude(*a, **k):
            raise RuntimeError("stub: brand extraction falls back")

        A._claude_msg = no_claude

    def tearDown(self):
        A._call_llm, A._claude_msg = self._orig

    def _run(self, operator):
        events, out = [], io.StringIO()

        def on_progress(step, detail, current, total):
            events.append((step, detail))
            if step == "extract":        # first event after the panel: stop here
                raise _Stop()

        prompts = [f"Question {i}?" for i in range(4)]
        with contextlib.redirect_stdout(out), self.assertRaises(_Stop):
            A.run_citation_audit("Acme wants to own delivery robots", on_progress=on_progress,
                                 tier="free", prompts_override=prompts, operator=operator)
        n = len(prompts) * len(A.TIER_CONFIG["free"]["llms"])
        return events, out.getvalue(), n

    def test_every_run_logs_the_timing_line(self):
        events, log, n = self._run(operator=False)
        self.assertIn(f"[audit] Answer collection: {n}/{n} in ", log)
        self.assertIn("s per answer;", log)
        self.assertFalse(any(d.startswith("Answer collection") for _, d in events))

    def test_operator_runs_get_it_in_the_stream(self):
        events, log, n = self._run(operator=True)
        timing = [d for s, d in events if s == "llm" and d.startswith("Answer collection")]
        self.assertEqual(len(timing), 1)
        self.assertIn(f"{n}/{n} in ", timing[0])


if __name__ == "__main__":
    unittest.main()
