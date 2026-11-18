"""Regression: outreach reminders count from the most recent touch.

Nov 18, 2026: twelve cards (119 to 121, 126 to 132, 134, 135) were followed up
35 days after the pitch. Every step counted from sent_at, so sent_at + 14 was
already past and each card showed "follow-up due" again at once. Nathan closed
them by logging a second follow-up.

Run from the repo root:  python3 tests/test_outreach_due_last_touch.py -v
"""
import os
import sys
import tempfile
import itertools
import unittest
from datetime import datetime, timedelta, date

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db"))
os.environ.setdefault("LINK_STATS_KEY", "test-key")

import app as A  # noqa: E402

TODAY = datetime(2026, 11, 18, 15, 0)
_IDS = itertools.count(1)


class _Clock(datetime):
    """datetime with a settable utcnow, swapped into app for the digest test."""
    now_value = TODAY

    @classmethod
    def utcnow(cls):
        return cls.now_value


def _card(sent_days_ago, cadence="5,14", count=0, last_followup_at=None, status="sent"):
    with A.app.app_context():
        o = A.Outreach(prospect_name=f"Test card {next(_IDS)}", slug="",
                       cadence=cadence, status=status, followup_count=count,
                       sent_at=TODAY - timedelta(days=sent_days_ago),
                       last_activity_at=TODAY - timedelta(days=sent_days_ago),
                       last_followup_at=last_followup_at)
        o.next_followup_due = A._outreach_compute_due(o)
        A.db.session.add(o)
        A.db.session.commit()
        return o.id


class ComputeDue(unittest.TestCase):
    def setUp(self):
        self._dt = A.datetime
        _Clock.now_value = TODAY
        A.datetime = _Clock

    def tearDown(self):
        A.datetime = self._dt

    def _mark(self, oid, action, when=None):
        if when:
            _Clock.now_value = when
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            A._outreach_mark(o, action)
            A.db.session.commit()
            return o.next_followup_due, o.followup_count, o.last_followup_at

    def test_late_followup_comes_due_nine_days_later(self):
        oid = _card(35)                                  # sent Oct 14, 2026
        due, n, at = self._mark(oid, "followup")
        self.assertEqual((n, at), (1, TODAY))
        self.assertEqual(due, date(2026, 11, 27))         # 14 - 5 = 9 days after Nov 18
        self.assertGreater(due, TODAY.date())             # the Nov 18 bug: due Oct 28, already past

    def test_on_time_followup_keeps_the_old_dates(self):
        oid = _card(5)                                   # follow-up logged on day 5
        due, _, _ = self._mark(oid, "followup")
        self.assertEqual(due, (TODAY + timedelta(days=9)).date())   # day 14 from sent, as before

    def test_first_step_still_counts_from_sent(self):
        oid = _card(2)
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            self.assertEqual(o.next_followup_due, (TODAY + timedelta(days=3)).date())

    def test_exhausted_cadence_has_no_reminder(self):
        oid = _card(35)
        self._mark(oid, "followup")
        due, n, _ = self._mark(oid, "followup", TODAY + timedelta(days=9))
        self.assertEqual((due, n), (None, 2))

    def test_dropdown_change_keeps_the_anchor(self):
        oid = _card(35)
        self._mark(oid, "followup")
        _Clock.now_value = TODAY + timedelta(days=4)
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            A._outreach_apply_status(o, "opened")
            self.assertEqual(o.last_followup_at, TODAY)
            self.assertEqual(o.next_followup_due, date(2026, 11, 27))

    def test_reopen_counts_from_last_followup(self):
        oid = _card(35)
        self._mark(oid, "followup")
        self._mark(oid, "cold", TODAY + timedelta(days=2))
        due, _, _ = self._mark(oid, "reopen", TODAY + timedelta(days=3))
        self.assertEqual(due, date(2026, 11, 27))

    def test_mark_sent_clears_the_anchor(self):
        oid = _card(35, count=1, last_followup_at=TODAY)
        _, n, at = self._mark(oid, "sent")
        self.assertEqual((n, at), (0, None))

    def test_row_without_anchor_falls_back_to_sent_schedule(self):
        with A.app.app_context():
            o = A.Outreach(cadence="5,14", followup_count=1, sent_at=TODAY - timedelta(days=35))
            self.assertEqual(A._outreach_compute_due(o), (TODAY - timedelta(days=21)).date())

    def test_new_default_cadence_seven_then_seven(self):
        self.assertEqual(A.OUTREACH_DEFAULT_CADENCE, "7,14")
        self.assertEqual(A._cadence_days(""), [7, 14])
        oid = _card(35, cadence=A.OUTREACH_DEFAULT_CADENCE)
        due, _, _ = self._mark(oid, "followup")
        self.assertEqual(due, date(2026, 11, 25))         # 14 - 7 = 7 days after Nov 18

    def test_non_increasing_cadence_reads_as_intervals(self):
        self.assertEqual(A._cadence_gap([7, 7], 1), 7)
        self.assertEqual(A._cadence_gap([5, 14], 1), 9)

    def test_cadence_label(self):
        self.assertEqual(A._cadence_label("5,14"),
                         "reminders 5 days after sent, then 9 days after the last follow-up")
        self.assertEqual(A._cadence_label("7,14,21"),
                         "reminders 7 days after sent, then 7, then 7 days after the last follow-up")


class DigestAndBoard(unittest.TestCase):
    """The daily digest and the board read next_followup_due; check both."""

    def setUp(self):
        self._dt = A.datetime
        self._send = A._send_outreach_digest_email
        self.sent = []
        A._send_outreach_digest_email = lambda due: self.sent.append([o.id for o in due])
        A.datetime = _Clock
        with A.app.app_context():
            A.Outreach.query.delete()
            A.db.session.commit()

    def tearDown(self):
        A.datetime = self._dt
        A._send_outreach_digest_email = self._send

    def _digest_on(self, day):
        _Clock.now_value = day
        self.sent.clear()
        A._run_outreach_digest()
        return self.sent[0] if self.sent else []

    def test_late_followup_leaves_the_digest_until_day_nine(self):
        _Clock.now_value = TODAY
        oid = _card(35)
        self.assertEqual(self._digest_on(TODAY), [oid])           # due before the follow-up
        _Clock.now_value = TODAY
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            A._outreach_mark(o, "followup")
            A.db.session.commit()
        self.assertEqual(self._digest_on(TODAY), [])
        self.assertEqual(self._digest_on(datetime(2026, 11, 26, 8, 5)), [])
        self.assertEqual(self._digest_on(datetime(2026, 11, 27, 8, 5)), [oid])

    def test_board_shows_no_follow_up_due_after_late_followup(self):
        _Clock.now_value = TODAY
        oid = _card(35)
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            A._outreach_mark(o, "followup")
            A.db.session.commit()
        c = A.app.test_client()
        html = c.get("/outreach?key=test-key").get_data(as_text=True)
        card = html.split(f'id="card-{oid}"', 1)[1].split('<div class="card"', 1)[0]
        self.assertNotIn("follow-up due", card)
        self.assertIn('data-dueflag="0"', html.split(f'id="card-{oid}"', 1)[1][:200])
        self.assertIn("reminders 7 days after sent, then 7 days after the last follow-up", html)


class Backfill(unittest.TestCase):
    def setUp(self):
        with A.app.app_context():
            A.Outreach.query.delete()
            A.db.session.commit()

    def test_legacy_rows_get_an_anchor_and_closed_cycles_stay_closed(self):
        nov18 = datetime(2026, 11, 18, 14, 30)
        with A.app.app_context():
            # A card logged once on Nov 18 under the old rule (due Oct 28, past).
            live = A.Outreach(prospect_name="live", slug="", cadence="5,14", status="followed_up",
                              followup_count=1, sent_at=nov18 - timedelta(days=35),
                              last_activity_at=nov18, next_followup_due=date(2026, 10, 28))
            # One of the twelve cards closed by a second follow-up on Nov 18.
            closed = A.Outreach(prospect_name="closed", slug="", cadence="5,14", status="followed_up",
                                followup_count=2, sent_at=nov18 - timedelta(days=35),
                                last_activity_at=nov18, next_followup_due=None)
            A.db.session.add_all([live, closed])
            A.db.session.commit()
            live_id, closed_id = live.id, closed.id
        A._backfill_outreach_last_followup()
        with A.app.app_context():
            live = A.db.session.get(A.Outreach, live_id)
            closed = A.db.session.get(A.Outreach, closed_id)
            self.assertEqual(live.last_followup_at, nov18)
            self.assertEqual(live.next_followup_due, date(2026, 11, 27))
            self.assertEqual(closed.last_followup_at, nov18)
            self.assertIsNone(closed.next_followup_due)       # not reopened as due
        A._backfill_outreach_last_followup()                  # idempotent second boot
        with A.app.app_context():
            self.assertEqual(A.db.session.get(A.Outreach, live_id).next_followup_due, date(2026, 11, 27))


class SetRoute(unittest.TestCase):
    def setUp(self):
        self.c = A.app.test_client()
        self.c.get("/outreach?key=test-key")              # sets the operator session cookie
        self._dt = A.datetime
        _Clock.now_value = TODAY
        A.datetime = _Clock

    def tearDown(self):
        A.datetime = self._dt

    def test_reopen_one_of_the_twelve_for_a_nov_25_followup(self):
        nov18 = datetime(2026, 11, 18, 14, 30)
        oid = _card(35, count=2, last_followup_at=nov18, status="followed_up")
        r = self.c.post(f"/outreach/{oid}/set", data={"cadence": "7,14", "followup_count": "1", "ajax": "1"})
        self.assertEqual(r.status_code, 204)
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            self.assertEqual((o.cadence, o.followup_count), ("7,14", 1))
            self.assertEqual(o.next_followup_due, date(2026, 11, 25))

    def test_warm_card_takes_three_steps(self):
        oid = _card(3, cadence="7,14")
        r = self.c.post(f"/outreach/{oid}/set", data={"cadence": " 7, 14,21 ", "ajax": "1"})
        self.assertEqual(r.status_code, 204)
        with A.app.app_context():
            self.assertEqual(A.db.session.get(A.Outreach, oid).cadence, "7,14,21")

    def test_bad_values_are_refused_and_change_nothing(self):
        oid = _card(3, cadence="7,14")
        for data in ({"cadence": "soon"}, {"cadence": "7,0"}, {"followup_count": "3"},
                     {"followup_count": "-1"}, {"cadence": "7,14", "followup_count": "x", "notes": "n"}):
            r = self.c.post(f"/outreach/{oid}/set", data=dict(data, ajax="1"))
            self.assertEqual(r.status_code, 400, data)
        with A.app.app_context():
            o = A.db.session.get(A.Outreach, oid)
            self.assertEqual((o.cadence, o.followup_count, o.notes), ("7,14", 0, None))

    def test_reposting_a_card_keeps_its_cadence(self):
        with A.app.app_context():
            o = A._outreach_upsert("Warm Person", "", cadence="7,14,21")
            self.assertEqual(o.cadence, "7,14,21")
            o = A._outreach_upsert("Warm Person", "", insight="new line")
            self.assertEqual(o.cadence, "7,14,21")
            o2 = A._outreach_upsert("Cold Person", "")
            self.assertEqual(o2.cadence, "7,14")


if __name__ == "__main__":
    unittest.main()
