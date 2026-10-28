"""root_of / host_of: the scheme is stripped before the host is read, so a
doubled slash after the host or a port does not change the root.

Run from the repo root:  python3 tests/test_conventions_root.py -v
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools", "mcp_server"))
from conventions import host_of, root_of  # noqa: E402


class RootOf(unittest.TestCase):
    def test_forms(self):
        cases = {
            "https://www.garmentory.com//sale/24102-polo-ralph-lauren/x": ("garmentory.com", "garmentory.com"),
            "https://www.psychologytoday.com:443/us": ("psychologytoday.com", "psychologytoday.com"),
            "http://fragrantica.com/perfume/Azzaro/x.html": ("fragrantica.com", "fragrantica.com"),
            "https://beta.fragrantica.com/x": ("beta.fragrantica.com", "fragrantica.com"),
            "//cdn.example.co.uk/a": ("cdn.example.co.uk", "example.co.uk"),
            "www.Example.com": ("example.com", "example.com"),
            "https://iren.gcs-web.com/news": ("iren.gcs-web.com", "iren.gcs-web.com"),
            "https://user:pw@host.example.com/": ("host.example.com", "example.com"),
            "": ("", ""),
        }
        for u, (h, r) in cases.items():
            self.assertEqual(host_of(u), h, u); self.assertEqual(root_of(u), r, u)


if __name__ == "__main__":
    unittest.main()
