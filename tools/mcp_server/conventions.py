"""Shared counting/normalization conventions — the ONE implementation used by
both the Signal MCP server and the raw-run exporter. Any change here changes
both surfaces together; never fork these rules into a second file."""
import re

URLISH_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9.-]*\.(?:com|net|org|io|ai|co)(?:/\S*)?",
    re.I)

SECOND_LEVEL = {"co.uk", "ac.uk", "org.uk", "gov.uk", "com.au", "net.au",
                "org.au", "gov.au", "edu.au", "co.jp", "or.jp", "ne.jp",
                "co.in", "org.in", "com.br", "org.br", "co.nz", "org.nz",
                "com.sg", "com.hk", "com.mx", "com.tw", "co.za", "org.za",
                "com.cn", "org.cn", "co.kr", "or.kr"}

GENERIC_TOKENS = {"company", "companies", "group", "corp", "corporation", "inc",
                  "brands", "foods", "soup", "labs", "systems", "networks",
                  "technologies", "holdings", "global", "international"}


def root_of(host_or_url):
    h = (host_or_url or "").split("//")[-1].split("/")[0].split(":")[0].lower()
    h = h[4:] if h.startswith("www.") else h
    parts = h.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def host_of(url):
    h = (url or "").split("//")[-1].split("/")[0].split(":")[0].lower()
    return h[4:] if h.startswith("www.") else h


def forms_to_pattern(forms):
    """Word-boundary union with apostrophe tolerance, plus the first
    distinctive brand token with optional possessive — gated by the generic
    blocklist (the "Soup" lesson)."""
    parts = []
    for f in forms:
        if not (f or "").strip():
            continue
        parts.append(r"\b" + re.escape(f).replace(r"\'", "'?") + r"\b")
    toks = re.findall(r"[A-Za-z][\w'-]*", forms[0] if forms else "")
    if toks:
        t = toks[0]
        if len(t) >= 5 and t.lower() not in GENERIC_TOKENS:
            parts.append(r"\b" + re.escape(t) + r"(?:'?s)?\b")
    return re.compile("|".join(parts), re.I)


def type_of(host_or_url, cls_map):
    """Host-first classification with root fallback."""
    h = host_of(host_or_url)
    return cls_map.get(h) or cls_map.get(root_of(h))


def split_rows(payload):
    ps = payload.get("prompt_sets") or {}
    branded = set(ps.get("branded") or [])
    rows = payload.get("all_responses") or []
    unb = [r for r in rows if r.get("prompt") not in branded] if branded else rows
    return rows, unb, branded
