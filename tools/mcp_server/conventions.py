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
                "com.cn", "org.cn", "co.kr", "or.kr", "co.id", "or.id",
                "ac.id", "com.my", "co.th", "com.ar", "com.co", "com.pe",
                "com.ph", "com.vn", "com.tr", "com.pl", "com.eg", "co.il",
                "org.il", "com.ng", "co.ke"}

# Curated, not exhaustive — the full Public Suffix List is the paid-exporter
# ticket. Guard for the silent failure mode a partial list creates:
_SUFFIX_FIRST_LABELS = {"co", "or", "ac", "com", "org", "net", "gov", "edu", "ne"}


def is_bare_suffix(root):
    """True when a computed 'root' is likely a public suffix rather than a
    registrable domain (e.g. 'co.id' from a ccTLD missing from SECOND_LEVEL).
    Consumers should warn on these rather than treat them as an outlet —
    unrelated organisations would otherwise silently merge into one row."""
    parts = (root or "").split(".")
    return len(parts) == 2 and parts[0] in _SUFFIX_FIRST_LABELS and len(parts[1]) == 2

GENERIC_TOKENS = {"company", "companies", "group", "corp", "corporation", "inc",
                  "brands", "foods", "soup", "labs", "systems", "networks",
                  "technologies", "holdings", "global", "international"}


# Vendor-hosted investor-relations and newsroom sites: the company is the label
# left of the vendor's domain (iren.gcs-web.com is IREN's IR site, not
# "gcs-web.com"), so each company's site is its own source. Same set as
# app._SHARED_IR_HOSTING_ROOTS.
SHARED_IR_HOSTING = {"gcs-web.com", "q4web.com", "q4ir.com", "investorroom.com",
                     "mediaroom.com", "prezly.com", "pr.co"}


def _host(host_or_url):
    """Host of a URL or bare host, lowercased, www. and port stripped. The
    scheme is removed first: splitting on "//" took the path when a URL had a
    doubled slash after the host (garmentory.com//sale/... -> "sale", Oct 27,
    2026), which left the root untyped."""
    s = (host_or_url or "").strip()
    s = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", s)
    if s.startswith("//"):
        s = s[2:]
    h = s.split("/")[0].split("?")[0].split("#")[0].split("@")[-1].split(":")[0].lower()
    return h[4:] if h.startswith("www.") else h


def root_of(host_or_url):
    h = _host(host_or_url)
    parts = h.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in SHARED_IR_HOSTING:
        return ".".join(parts[-3:])
    if len(parts) >= 3 and ".".join(parts[-2:]) in SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def host_of(url):
    return _host(url)


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
