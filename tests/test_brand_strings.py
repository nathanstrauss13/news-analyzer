"""Brand rule (Nathan): the company is "innate c3", lowercase, plain 3, in every string the app
sends or shows. Outbound email display names and user-facing subjects must not carry the
capitalised form or the retired product name. Comments and internal debug mail are out of scope."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_sender_display_name_is_lowercase_brand():
    src = APP.read_text()
    names = re.findall(r'from_email=\("nstrauss@innatec3\.com",\s*"([^"]+)"\)', src)
    assert names, "no from_email tuples found"
    assert set(names) == {"innate c3"}, sorted(set(names))


def test_no_capitalised_brand_in_code_strings():
    src = APP.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "Innate C3" not in code


def test_magic_link_mail_uses_current_naming():
    src = APP.read_text()
    assert 'subject="Your innate c3 sign-in link"' in src
    assert "sign in to PR Signal Finder" not in src
