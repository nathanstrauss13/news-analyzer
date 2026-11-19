"""Hosted connector second secret (Nov 19, 2026).

A SIGNAL_MCP_CLIENTS entry with "secret_sha256" requires
"Authorization: Bearer <secret>" on /mcp/<key>. Wrong or missing secret: 401.
Unknown key: 404. Entries without the field are unchanged. The hash never
reaches the profile the tools see or the operator diagnostic.

Run from the repo root:  python3 tests/test_mcp_client_secret.py -v
"""
import os
import sys
import json
import hashlib
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db"))
os.environ.setdefault("LINK_STATS_KEY", "test-key")

SECRET = "correct-horse-battery"
KEY_LOCKED = "locked-key-0123456789"
KEY_OPEN = "open-key-0123456789abc"
KEY_INLINE = "inline-key-0123456789"
os.environ["SIGNAL_MCP_CLIENTS"] = json.dumps({
    KEY_LOCKED: {"profile": "loreal", "secret_sha256": hashlib.sha256(SECRET.encode()).hexdigest()},
    KEY_OPEN: {"profile": "loreal"},
    KEY_INLINE: {"client": "inline-test", "audits": [{"slug": "0a312ac53b", "label": "x", "run_date": "2026-08-27"}],
                 "secret_sha256": hashlib.sha256(SECRET.encode()).hexdigest().upper()},
})

import app as A  # noqa: E402

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


class SecretTests(unittest.TestCase):
    def setUp(self):
        self.c = A.app.test_client()

    def _post(self, key, body, auth=None):
        h = {"Authorization": auth} if auth else {}
        return self.c.post(f"/mcp/{key}", data=json.dumps(body), content_type="application/json", headers=h)

    def test_unknown_key_is_404(self):
        self.assertEqual(self._post("nobody-knows-this-key", INIT).status_code, 404)
        self.assertEqual(self._post("nobody-knows-this-key", INIT, "Bearer " + SECRET).status_code, 404)

    def test_locked_key_without_secret_is_401(self):
        r = self._post(KEY_LOCKED, INIT)
        self.assertEqual(r.status_code, 401)
        self.assertIn("Bearer", r.headers.get("WWW-Authenticate", ""))
        self.assertEqual(self.c.get(f"/mcp/{KEY_LOCKED}").status_code, 401)

    def test_locked_key_with_wrong_secret_is_401(self):
        self.assertEqual(self._post(KEY_LOCKED, INIT, "Bearer " + SECRET + "x").status_code, 401)
        self.assertEqual(self._post(KEY_LOCKED, INIT, "Basic " + SECRET).status_code, 401)
        self.assertEqual(self._post(KEY_LOCKED, INIT, "Bearer ").status_code, 401)

    def test_locked_key_with_secret_serves(self):
        r = self._post(KEY_LOCKED, INIT, "Bearer " + SECRET)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["result"]["serverInfo"]["name"], "signal-audit")
        r = self._post(KEY_LOCKED, LIST, "bearer " + SECRET)
        self.assertEqual(r.status_code, 200)
        names = [t["name"] for t in r.get_json()["result"]["tools"]]
        self.assertIn("list_audits", names)

    def test_open_key_unchanged(self):
        self.assertEqual(self._post(KEY_OPEN, INIT).status_code, 200)
        self.assertEqual(self._post(KEY_OPEN, INIT, "Bearer whatever").status_code, 200)
        self.assertEqual(self.c.get(f"/mcp/{KEY_OPEN}").status_code, 405)

    def test_inline_profile_hash_stripped_and_case_insensitive(self):
        self.assertEqual(self._post(KEY_INLINE, INIT).status_code, 401)
        self.assertEqual(self._post(KEY_INLINE, INIT, "Bearer " + SECRET).status_code, 200)
        entry, prof = A._mcp_resolve_entry(KEY_INLINE)
        self.assertIn("secret_sha256", entry)
        self.assertNotIn("secret_sha256", prof)
        self.assertEqual(prof["client"], "inline-test")

    def test_tool_result_never_carries_hash(self):
        r = self._post(KEY_LOCKED, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                    "params": {"name": "list_audits", "arguments": {}}}, "Bearer " + SECRET)
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("secret_sha256", r.get_data(as_text=True))
        self.assertNotIn(hashlib.sha256(SECRET.encode()).hexdigest(), r.get_data(as_text=True))

    def test_admin_diagnostic_flags_without_hash(self):
        r = self.c.get("/admin/mcp-registry?key=test-key")
        self.assertEqual(r.status_code, 200)
        txt = r.get_data(as_text=True)
        self.assertNotIn(hashlib.sha256(SECRET.encode()).hexdigest(), txt)
        self.assertNotIn(SECRET, txt)
        flags = sorted(e["second_secret_required"] for e in r.get_json()["entries"])
        self.assertEqual(flags, [False, True, True])

    def test_secret_ok_helper(self):
        h = hashlib.sha256(b"s").hexdigest()
        self.assertTrue(A._mcp_secret_ok({}, None))
        self.assertTrue(A._mcp_secret_ok({"secret_sha256": h}, "Bearer s"))
        self.assertFalse(A._mcp_secret_ok({"secret_sha256": h}, "Bearer t"))
        self.assertFalse(A._mcp_secret_ok({"secret_sha256": h}, None))


if __name__ == "__main__":
    unittest.main()
