#!/usr/bin/env python3
"""AI Visibility Intelligence dashboard builder (operator tool, not DIY).

Turns raw citation-audit output into a self-contained, client-facing HTML
dashboard in the current Innate C3 look (Jost/Inter dark premium, gold /
coral / cyan). Successor to the v17/v18 AEO dashboards behind
innatec3.com/pali, /bain and /lululemon, generalized so ONE template serves
any industry and any buyer (GTM, BD, founder, comms) with branded and/or
unbranded prompt sets.

INPUTS (auto-detected, either or both):
  * Raw audit kit JSON  (~/Desktop/Norwest/raw_audit.py output):
      {brand, key, rows:[{query, platform, full_response,
                          citations | citations_resolved, ...}]}
  * Production slug JSON (signal.innatec3.com/signal/<slug>.json):
      {brand, all_responses:[{llm, prompt, response, citations:[{url}]}]}

USAGE
  python3 build_dashboard.py --config phasecraft.config.json
  # config names the branded and/or organic dataset paths + brand facts.
  # Output: <slug>_dashboard.html next to the config (or --out DIR).

DESIGN RULES (house style, enforced in copy)
  * Sober, small-sample framing: directional read, not a verdict.
  * No em dashes in client-facing copy (commas / parens instead).
  * Buyer-neutral language: "AI visibility", never PR jargon by default.

PDF: render with Chrome headless
  chrome --headless=new --print-to-pdf=out.pdf --virtual-time-budget=20000 \
         --run-all-compositor-stages-before-draw --no-pdf-header-footer file.html
"""
import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date

# ---------------------------------------------------------------- constants
PLATFORM_ORDER = ["ChatGPT", "Claude", "Gemini", "Perplexity", "Grok"]

# Names of the assistants themselves: never counted as competitors, and their
# own domains are excluded from "sources" (chatgpt.com in a ChatGPT answer is
# self-reference, not a source AI trusts).
ASSISTANT_NAMES = {"chatgpt", "claude", "gemini", "grok", "perplexity",
                   "copilot", "bard", "openai", "anthropic"}
ASSISTANT_DOMAINS = {"chatgpt.com", "openai.com", "claude.ai", "anthropic.com",
                     "gemini.google.com", "perplexity.ai", "x.ai", "grok.com"}

# Multi-label public suffixes we care about for registrable-root grouping.
_SECOND_LEVEL = {"co.uk", "ac.uk", "org.uk", "gov.uk", "com.au", "net.au",
                 "co.jp", "co.in", "com.br", "co.nz", "com.sg", "com.hk"}

REVIEW_TALENT = {"glassdoor.com", "g2.com", "capterra.com", "trustpilot.com",
                 "indeed.com", "gartner.com", "trustradius.com", "getapp.com",
                 "softwareadvice.com", "producthunt.com", "comparably.com",
                 "yelp.com", "tripadvisor.com"}
COMMUNITY = {"reddit.com", "quora.com", "stackexchange.com", "stackoverflow.com",
             "news.ycombinator.com", "medium.com", "substack.com", "dev.to"}
SOCIAL = {"linkedin.com", "x.com", "twitter.com", "youtube.com", "instagram.com",
          "facebook.com", "tiktok.com", "threads.net"}
REFERENCE = {"wikipedia.org", "wikidata.org", "britannica.com", "crunchbase.com",
             "pitchbook.com", "wellfound.com", "zoominfo.com", "cbinsights.com",
             "tracxn.com", "dnb.com"}

TYPE_LABELS = [
    ("owned",       "Owned",                  "cyan"),
    ("competitor",  "Competitor-owned",       "rival"),
    ("editorial",   "Editorial &amp; media",  "gold"),
    ("institutional","Institutional &amp; academic", "inst"),
    ("reviews",     "Reviews &amp; talent",   "rev"),
    ("community",   "Community",              "comm"),
    ("social",      "Social",                 "soc"),
    ("reference",   "Reference data",         "ref"),
]
TYPE_INDEX = {k: (lbl, cls) for k, lbl, cls in TYPE_LABELS}


# ---------------------------------------------------------------- utilities
def esc(s):
    return html.escape(str(s if s is not None else ""))


def norm_url(u):
    """Display-normalize a citation URL: strip scheme, www, query (utm noise),
    fragment, trailing slash. Preserves path case so links still resolve."""
    u = (u or "").strip()
    if not u:
        return ""
    u = re.sub(r"#.*$", "", u)
    u = re.sub(r"\?.*$", "", u)
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.rstrip("/")


def host_of(disp):
    return disp.split("/", 1)[0].lower()


def root_of(host):
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


_GENERIC_TOKENS = {"the", "and", "for", "inc", "llc", "ltd", "corp", "group",
                   "company", "technologies", "solutions", "systems", "software",
                   "labs", "media", "global", "digital", "quantum", "computing",
                   "financial", "capital", "ventures", "partners", "health"}


def match_forms(name, aliases=None, corpus=None):
    """Word-boundary match forms for an entity: full name + aliases + a
    distinctive token, with the lowercase-dominance guard learned from the
    Signal Finder counting bugs (a token whose corpus usage is mostly
    lowercase is a common word, not the company)."""
    forms = []
    for cand in [name] + list(aliases or []):
        cand = (cand or "").strip()
        if cand and cand.lower() not in {f.lower() for f in forms}:
            forms.append(cand)
    toks = re.findall(r"[A-Za-z][\w&'-]*", name or "")
    distinctive = [t for t in toks
                   if (re.search(r"[a-z][A-Z]", t) or (t.isupper() and len(t) >= 3)
                       or (len(t) >= 5 and t.lower() not in _GENERIC_TOKENS))]
    for t in distinctive[:1]:
        if corpus is not None:
            low = sum(1 for c in corpus
                      if re.search(r"(?<![A-Za-z0-9])" + re.escape(t.lower()) + r"(?![A-Za-z0-9])", c))
            cased = sum(1 for c in corpus
                        if re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", c))
            if low >= 3 and cased < low * 0.5:
                continue  # generic word in this corpus, skip the short form
        if t.lower() not in {f.lower() for f in forms} and t.lower() not in _GENERIC_TOKENS:
            forms.append(t)
    return forms


def present(forms, text):
    t = text or ""
    return any(re.search(r"\b" + re.escape(f) + r"\b", t, re.I) for f in forms)


def page_kind(disp):
    """Owned-URL content kind: is AI citing the homepage (name recognition)
    or deep content (publishing doing the work)? Ported from the owned lens."""
    host, _, path = disp.partition("/")
    p = "/" + path.lower() if path else "/"
    p = re.sub(r"^/(?:[a-z]{2}(?:-[a-z]{2})?)(?=/|$)", "", p)
    p = re.sub(r"/(?:index|home|default)\.\w+$", "", p).rstrip("/")
    if not p:
        return "homepage"
    h = host.lower()
    if h.startswith("ir.") or "investor" in p:
        return "investor relations"
    if h.startswith(("news.", "press.")) or re.search(r"newsroom|/news(?:/|$)|/press|/media(?:/|$)", p):
        return "newsroom"
    if re.search(r"/blog|/insights|/articles|/research|/publications|/perspectives|/resources|/case-stud|/white-?paper", p):
        return "blog / research"
    if h.startswith("docs.") or re.search(r"/docs|/documentation|/developer|/api(?:/|$)", p):
        return "docs / product"
    if re.search(r"/careers|/jobs|/join", p):
        return "careers"
    if re.search(r"/team|/about|/people|/leadership", p):
        return "about / team"
    return "site page"


# ---------------------------------------------------------------- data load
def load_rows(path):
    """Load either input schema, normalize to
    [{query, platform, response, urls:[display-normalized]}]."""
    with open(path) as f:
        d = json.load(f)
    rows = []
    if isinstance(d, dict) and "all_responses" in d:            # production slug JSON
        for r in d.get("all_responses") or []:
            if r.get("error"):
                continue
            urls = [norm_url(c.get("url")) for c in (r.get("citations") or []) if c.get("url")]
            rows.append({"query": r.get("prompt") or "", "platform": r.get("llm") or "?",
                         "response": r.get("response") or "", "urls": [u for u in urls if u]})
    elif isinstance(d, dict) and "rows" in d:                    # raw audit kit JSON
        for r in d["rows"]:
            raw = r.get("citations_resolved") or r.get("citations") or ""
            if isinstance(raw, str):
                urls = [norm_url(u) for u in raw.split("|")]
            else:
                urls = [norm_url(u) for u in raw]
            rows.append({"query": r.get("query") or "", "platform": r.get("platform") or "?",
                         "response": r.get("full_response") or "", "urls": [u for u in urls if u]})
    else:
        sys.exit(f"Unrecognized JSON schema in {path}")
    return rows


# ---------------------------------------------------------------- analysis
def classify_root(root, owned_roots, competitor_roots):
    if root in ASSISTANT_DOMAINS:
        return None                       # self-reference, excluded entirely
    if root in owned_roots:
        return "owned"
    if root in competitor_roots:
        return "competitor"
    if root in REVIEW_TALENT:
        return "reviews"
    if root in COMMUNITY:
        return "community"
    if root in SOCIAL:
        return "social"
    if root in REFERENCE:
        return "reference"
    if root.endswith((".edu", ".gov", ".ac.uk", ".gov.uk")) or root in {"nsf.gov", "nist.gov"}:
        return "institutional"
    return "editorial"


def analyze(rows, cfg):
    """Everything the template needs for ONE dataset (branded or organic)."""
    brand = cfg["brand"]
    exclude = {root_of(host_of(norm_url("https://" + d))) for d in cfg.get("exclude_sources", [])}
    owned_roots = {root_of(host_of(norm_url("https://" + d))) for d in cfg.get("owned_domains", [])}
    comp_cfg = cfg.get("competitors", [])
    competitor_roots = set()
    for c in comp_cfg:
        for d in c.get("domains", []):
            competitor_roots.add(root_of(host_of(norm_url("https://" + d))))

    corpus = [r["response"] for r in rows]
    bforms = match_forms(brand, cfg.get("aliases"), corpus)

    platforms = [p for p in PLATFORM_ORDER if any(r["platform"] == p for r in rows)]
    platforms += sorted({r["platform"] for r in rows} - set(platforms))

    # brand presence per response / per platform
    n = len(rows)
    brand_rows = 0
    per_platform = {p: {"total": 0, "brand": 0, "citations": 0} for p in platforms}
    for r in rows:
        hit = present(bforms, r["response"])
        r["brand_hit"] = hit
        brand_rows += 1 if hit else 0
        pp = per_platform[r["platform"]]
        pp["total"] += 1
        pp["brand"] += 1 if hit else 0
        pp["citations"] += len(r["urls"])

    # competitor presence (word-boundary, alias-aware, assistants excluded)
    comp_counts = []
    for c in comp_cfg:
        nm = c["name"]
        if nm.lower() in ASSISTANT_NAMES:
            continue
        forms = match_forms(nm, c.get("aliases"), corpus)
        cnt = sum(1 for r in rows if present(forms, r["response"]))
        comp_counts.append({"name": nm, "count": cnt})
    comp_counts.sort(key=lambda x: -x["count"])

    # citations: per-URL and per-root aggregation
    url_counts = defaultdict(int)          # display url -> citations
    root_agg = {}                          # root -> {count, answers:set, platforms:set, type}
    total_citations = 0
    for i, r in enumerate(rows):
        seen_roots_this_row = set()
        for u in r["urls"]:
            root = root_of(host_of(u))
            if root in exclude:      # config escape hatch for junk sources
                continue
            typ = classify_root(root, owned_roots, competitor_roots)
            if typ is None:
                continue
            total_citations += 1
            url_counts[(u, typ)] += 1
            a = root_agg.setdefault(root, {"count": 0, "answers": set(), "platforms": set(), "type": typ})
            a["count"] += 1
            a["answers"].add(i)
            a["platforms"].add(r["platform"])
            seen_roots_this_row.add(root)

    sources = [{"root": k, "count": v["count"], "answers": len(v["answers"]),
                "platforms": sorted(v["platforms"], key=lambda p: PLATFORM_ORDER.index(p) if p in PLATFORM_ORDER else 99),
                "type": v["type"]}
               for k, v in root_agg.items()]
    sources.sort(key=lambda s: (-s["count"], s["root"]))

    type_totals = defaultdict(lambda: {"count": 0, "sources": 0})
    for s in sources:
        type_totals[s["type"]]["count"] += s["count"]
        type_totals[s["type"]]["sources"] += 1

    owned_pages = sorted(
        [{"url": u, "count": c, "kind": page_kind(u)}
         for (u, typ), c in url_counts.items() if typ == "owned"],
        key=lambda x: (-x["count"], x["url"]))
    owned_citations = sum(p["count"] for p in owned_pages)
    owned_answers = len(set().union(*[root_agg[r]["answers"] for r in owned_roots if r in root_agg]) if any(r in root_agg for r in owned_roots) else set())
    home_ct = sum(p["count"] for p in owned_pages if p["kind"] == "homepage")

    # opportunity: sources cited in answers where brand is absent but competitors present
    gap_sources = []
    for s in sources:
        if s["type"] in ("owned",):
            continue
        answers = root_agg[s["root"]]["answers"]
        brand_in = sum(1 for i in answers if rows[i]["brand_hit"])
        if s["answers"] >= 2 and brand_in == 0:
            gap_sources.append({**s, "brand_in": brand_in})

    return {
        "rows": rows, "n": n, "platforms": platforms,
        "brand_rows": brand_rows, "per_platform": per_platform,
        "competitors": comp_counts,
        "sources": sources, "type_totals": dict(type_totals),
        "total_citations": total_citations,
        "owned_pages": owned_pages, "owned_citations": owned_citations,
        "owned_answers": owned_answers, "owned_home_citations": home_ct,
        "gap_sources": gap_sources[:8],
        "prompts": list(dict.fromkeys(r["query"] for r in rows)),
    }


# ---------------------------------------------------------------- HTML bits
CSS = """
:root{--ink:#EBECEF;--ink2:#9aa1ad;--muted:#6b7280;--line:rgba(255,255,255,.07);
 --gold:#cbab6d;--gold-g:rgba(203,171,109,.40);--coral:#f0876a;--coral-g:rgba(240,135,106,.34);
 --cyan:#74d0ff;--cyan-g:rgba(116,208,255,.32);--win:#5cf08a;
 --card:linear-gradient(180deg,#12141b,#0d0f14);--cta:linear-gradient(180deg,#f0916f,#d76a4c)}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:radial-gradient(1100px 520px at 80% -6%,rgba(203,171,109,.09),transparent 60%),
 radial-gradient(1000px 560px at 8% 3%,rgba(116,208,255,.06),transparent 55%),#07080b;
 color:var(--ink);font-family:'Inter',-apple-system,sans-serif;line-height:1.55;
 -webkit-font-smoothing:antialiased;padding:0 0 60px}
.wrap{max-width:960px;margin:0 auto;padding:0 28px}
h1,h2,h3,.eyebrow,.wm,.stat-n,.pill,.tag,.bar-name.you,.kindtag,.typechip{font-family:'Jost',sans-serif}
nav{position:sticky;top:0;z-index:50;background:rgba(7,8,11,.86);backdrop-filter:blur(10px);
 border-bottom:1px solid var(--line);padding:13px 0;margin-bottom:44px}
nav .wrap{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}
.wm{font-weight:500;font-size:19px;color:#f4f4f5;text-decoration:none}.wm sup{font-size:.56em;color:var(--gold)}
.navlinks{display:flex;gap:4px;flex-wrap:wrap}
.navlinks a{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;
 font-family:'Jost',sans-serif;color:var(--ink2);text-decoration:none;padding:6px 11px;border-radius:999px}
.navlinks a:hover{color:var(--ink);background:rgba(255,255,255,.05)}
header.hero{padding:26px 0 8px}
.eyebrow{color:var(--gold);letter-spacing:.24em;text-transform:uppercase;font-size:11.5px;font-weight:600;
 text-shadow:0 0 18px var(--gold-g)}
h1{font-weight:600;font-size:38px;line-height:1.12;margin:12px 0 14px;color:#fff;letter-spacing:-.01em}
.cat{display:inline-block;font-size:13.5px;color:#cdd6ea;background:rgba(116,208,255,.08);
 border:1px solid rgba(116,208,255,.18);border-radius:999px;padding:7px 16px;margin-right:8px}
.runmeta{font-size:13px;color:var(--muted);margin-top:12px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0 6px}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 19px;
 box-shadow:0 24px 60px -46px #000}
.stat-n{font-weight:700;font-size:31px;letter-spacing:-.02em;color:#fff}
.stat-n.gold{color:var(--gold);text-shadow:0 0 22px var(--gold-g)}
.stat-n.coral{color:var(--coral);text-shadow:0 0 22px var(--coral-g)}
.stat-n.cyan{color:var(--cyan);text-shadow:0 0 22px var(--cyan-g)}
.stat-l{font-size:12.5px;color:var(--ink2);margin-top:5px;line-height:1.45}
section{margin-top:58px}
.gh{color:var(--gold);letter-spacing:.16em;text-transform:uppercase;font-size:11.5px;font-weight:600}
h2{font-weight:600;font-size:24px;color:#fff;margin:10px 0 8px;letter-spacing:-.01em}
.gsub{font-size:14px;color:var(--ink2);margin:4px 0 22px;max-width:700px;line-height:1.6}
.exec{background:var(--card);border:1px solid var(--line);border-left:2px solid var(--gold);
 border-radius:14px;padding:22px 26px;font-size:15.5px;color:#e8e4da;line-height:1.7}
.exec p+p{margin-top:12px}
.contrast{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.contrast{grid-template-columns:1fr}}
.cpanel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px 26px}
.cpanel.hot{border-color:rgba(116,208,255,.3);box-shadow:0 0 50px -24px var(--cyan-g)}
.cpanel .lbl{font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;
 font-family:'Jost',sans-serif;color:var(--ink2)}
.cpanel .big{font-family:'Jost',sans-serif;font-weight:700;font-size:44px;color:#fff;margin:8px 0 2px}
.cpanel.hot .big{color:var(--cyan);text-shadow:0 0 26px var(--cyan-g)}
.cpanel .note{font-size:13px;color:var(--ink2);line-height:1.55}
.bridge{margin-top:14px;font-size:14.5px;color:#e8e4da;border-left:2px solid var(--coral);
 background:rgba(240,135,106,.06);padding:13px 18px;border-radius:0 10px 10px 0}
.orow{display:grid;grid-template-columns:190px 1fr 92px;align-items:center;gap:16px;margin:10px 0}
@media(max-width:560px){.orow{grid-template-columns:120px 1fr 74px;gap:10px}}
.bar-name{font-size:14px;color:var(--ink2)}.bar-name.you{font-weight:600;color:var(--cyan)}
.ot{height:11px;border-radius:999px;background:rgba(255,255,255,.05);overflow:hidden}
.of{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#d9a24f,#ecca85)}
.of.you{background:linear-gradient(90deg,#5aa8f0,#74d0ff);box-shadow:0 0 12px rgba(116,208,255,.4)}
.ov{font-size:13px;color:var(--ink);text-align:right;font-family:'Jost',sans-serif;font-weight:600}
.plats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
@media(max-width:820px){.plats{grid-template-columns:repeat(2,1fr)}}
.plat{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 16px;text-align:center}
.plat .pn{font-family:'Jost',sans-serif;font-weight:600;font-size:14.5px;color:#fff}
.plat .pv{font-family:'Jost',sans-serif;font-weight:700;font-size:26px;margin:7px 0 2px;color:var(--gold)}
.plat .pv.full{color:var(--win)}
.plat .pl{font-size:11.5px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font-family:'Jost',sans-serif;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--muted);font-weight:600;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
td{padding:10px;border-bottom:1px solid rgba(255,255,255,.05);vertical-align:middle;color:var(--ink2)}
td.num{text-align:right;font-family:'Jost',sans-serif;font-weight:600;color:var(--ink);white-space:nowrap}
tr:hover td{background:rgba(255,255,255,.015)}
.src{color:var(--ink);font-weight:500}
.typechip{display:inline-block;font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;
 border-radius:999px;padding:2.5px 9px;border:1px solid var(--line);color:var(--ink2);background:rgba(255,255,255,.03)}
.typechip.cyan{color:var(--cyan);border-color:rgba(116,208,255,.3);background:rgba(116,208,255,.06)}
.typechip.rival{color:#ecca85;border-color:rgba(217,162,79,.35);background:rgba(217,162,79,.07)}
.typechip.gold{color:var(--gold);border-color:rgba(203,171,109,.3);background:rgba(203,171,109,.06)}
.typechip.inst{color:#b9c7ff;border-color:rgba(147,169,255,.3);background:rgba(147,169,255,.06)}
.typechip.rev{color:#f2b8d0;border-color:rgba(242,184,208,.28);background:rgba(242,184,208,.05)}
.typechip.comm{color:#a8e6c1;border-color:rgba(140,230,170,.25);background:rgba(140,230,170,.05)}
.typechip.soc{color:#d8c6f5;border-color:rgba(200,170,245,.28);background:rgba(200,170,245,.05)}
.typechip.ref{color:#9fd6d0;border-color:rgba(140,214,205,.28);background:rgba(140,214,205,.05)}
.kindtag{display:inline-block;font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;
 color:var(--ink2);border:1px solid var(--line);border-radius:999px;padding:2.5px 9px;background:rgba(255,255,255,.03)}
.kindtag.home{color:var(--gold);border-color:rgba(203,171,109,.35);background:rgba(203,171,109,.07)}
.pdots{white-space:nowrap}
.pdot{display:inline-block;width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.14);margin-right:4px}
.pdot.on{background:var(--cyan);box-shadow:0 0 8px rgba(116,208,255,.5)}
a.upath{color:var(--cyan);text-decoration:none}
a.upath:hover{text-decoration:underline}
.pagenote{margin-top:16px;font-size:13.5px;color:var(--ink2);border-left:2px solid var(--cyan);
 padding:10px 16px;background:rgba(116,208,255,.05);border-radius:0 10px 10px 0}
.pagenote.home{border-left-color:var(--gold);background:rgba(203,171,109,.06)}
.pagenote b{color:var(--ink)}
.opp{background:var(--card);border:1px solid var(--line);border-left:2px solid var(--coral);
 border-radius:13px;padding:16px 20px;margin-top:12px}
.opp h4{font-family:'Jost',sans-serif;font-weight:600;font-size:15.5px;color:#fff;line-height:1.4}
.opp p{font-size:13.5px;color:var(--ink2);margin-top:6px}
.opp b{color:var(--coral)}
.qtools{display:flex;gap:8px;margin:0 0 14px;flex-wrap:wrap}
.qbtn{font-family:'Jost',sans-serif;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
 font-weight:600;color:var(--ink2);background:rgba(255,255,255,.03);border:1px solid var(--line);
 border-radius:999px;padding:7px 14px;cursor:pointer}
.qbtn:hover{color:var(--ink);border-color:rgba(203,171,109,.4)}
.qbtn.on{color:#2a1008;background:var(--cta);border-color:transparent}
.qrow{border:1px solid var(--line);border-radius:12px;background:var(--card);margin-top:10px;overflow:hidden}
.qhead{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;padding:13px 18px;cursor:pointer}
.qhead:hover{background:rgba(255,255,255,.02)}
.qq{font-size:14px;color:var(--ink);line-height:1.45}
.qmarks{white-space:nowrap;font-size:12px;color:var(--muted)}
.qmark{display:inline-block;min-width:20px;text-align:center;margin-left:3px;font-weight:600}
.qmark.hit{color:var(--win)} .qmark.miss{color:rgba(255,255,255,.22)}
.qbody{display:none;border-top:1px solid var(--line);padding:6px 18px 16px}
.qrow.open .qbody{display:block}
.qresp{margin-top:12px}
.qresp .rl{font-family:'Jost',sans-serif;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 font-weight:600;color:var(--gold);margin-bottom:5px}
.qresp .rt{font-size:13px;color:var(--ink2);line-height:1.6;white-space:pre-wrap;max-height:280px;
 overflow:auto;background:rgba(0,0,0,.25);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.qresp .rc{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.7;word-break:break-all}
.qresp .rc a{color:var(--cyan);text-decoration:none}
.meth{font-size:13.5px;color:var(--ink2);line-height:1.75}
.meth b{color:var(--ink)}
.promptlist{margin:10px 0 0 0;padding-left:20px;font-size:13.5px;color:var(--ink2);line-height:1.85}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--line);font-size:13px;color:var(--muted);line-height:1.7}
footer a{color:var(--gold);text-decoration:none}
.modechip{display:inline-block;font-family:'Jost',sans-serif;font-size:10.5px;letter-spacing:.1em;
 text-transform:uppercase;font-weight:600;padding:3px 11px;border-radius:999px;margin-left:8px;vertical-align:middle}
.modechip.br{color:#2a1008;background:var(--cta)}
.modechip.org{color:#08222e;background:linear-gradient(180deg,#7fd4ff,#4fa8dc)}
@media print{
 body{background:#07080b!important;-webkit-print-color-adjust:exact;print-color-adjust:exact;padding:0}
 nav{position:static}
 .qbody{display:block!important}
 .qresp .rt{max-height:none;overflow:visible}
 section{page-break-inside:avoid}
 .qrow{page-break-inside:avoid}
}
"""


def bar_rows(items, you_name=None, denom=None):
    if not items:
        return '<div class="gsub">(none surfaced)</div>'
    mx = max(i["count"] for i in items) or 1
    out = []
    for i in items:
        you = you_name and i["name"].lower() == you_name.lower()
        pct = f'{round(100*i["count"]/denom)}%' if denom else str(i["count"])
        out.append(
            f'<div class="orow"><span class="bar-name{" you" if you else ""}">{esc(i["name"])}'
            f'{" (you)" if you else ""}</span>'
            f'<span class="ot"><span class="of{" you" if you else ""}" style="width:{round(100*i["count"]/mx)}%"></span></span>'
            f'<span class="ov">{i["count"]} &middot; {pct}</span></div>')
    return "".join(out)


def platform_cards(a, branded):
    out = []
    for p in a["platforms"]:
        d = a["per_platform"][p]
        rate = d["brand"] / d["total"] if d["total"] else 0
        cls = " full" if rate >= 0.999 else ""
        sub = f'{d["citations"]} citations' if branded else "of its answers name the brand"
        out.append(f'<div class="plat"><div class="pn">{esc(p)}</div>'
                   f'<div class="pv{cls}">{d["brand"]}/{d["total"]}</div>'
                   f'<div class="pl">{sub}</div></div>')
    return f'<div class="plats">{"".join(out)}</div>'


def sources_table(a, limit=18):
    rows = []
    for s in a["sources"][:limit]:
        lbl, cls = TYPE_INDEX[s["type"]]
        dots = "".join(f'<span class="pdot{" on" if p in s["platforms"] else ""}" title="{esc(p)}"></span>'
                       for p in PLATFORM_ORDER)
        rows.append(f'<tr><td class="src">{esc(s["root"])}</td>'
                    f'<td><span class="typechip {cls}">{lbl}</span></td>'
                    f'<td class="num">{s["count"]}</td>'
                    f'<td class="num">{s["answers"]}/{a["n"]}</td>'
                    f'<td class="pdots">{dots}</td></tr>')
    return ('<table><thead><tr><th>Source</th><th>Type</th><th>Citations</th>'
            '<th>Answers citing</th><th>Assistants</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def owned_pages_block(a, brand):
    if not a["owned_pages"]:
        return (f'<div class="pagenote home">AI cited no {esc(brand)} page in this set, '
                'a common shape when the category conversation runs through third-party '
                'sources, and a clear place where owned publishing can earn a seat.</div>')
    mx = a["owned_pages"][0]["count"] or 1
    rows = []
    for p_ in a["owned_pages"][:10]:
        home = p_["kind"] == "homepage"
        rows.append(f'<tr><td><a class="upath" href="https://{esc(p_["url"])}" target="_blank" rel="noopener">{esc(p_["url"])}</a> '
                    f'<span class="kindtag{" home" if home else ""}">{esc(p_["kind"])}</span></td>'
                    f'<td class="num">{p_["count"]}&times;</td></tr>')
    total = a["owned_citations"]
    home_ct = a["owned_home_citations"]
    deep = total - home_ct
    if total >= 3 and home_ct >= deep:
        note = (f'<div class="pagenote home"><b>{home_ct} of {total}</b> owned citations point at the homepage. '
                'AI recognizes the brand but pulls little deeper content, so citable publishing '
                '(research, blog, newsroom) is the clearest owned opportunity.</div>')
    elif total >= 3:
        note = (f'<div class="pagenote"><b>{deep} of {total}</b> owned citations pull deep content '
                '(research, blog, product pages), meaning the site\'s publishing, not just its '
                'homepage, is doing the work.</div>')
    else:
        note = ""
    return f'<table><thead><tr><th>Page</th><th>Citations</th></tr></thead><tbody>{"".join(rows)}</tbody></table>{note}'


def query_appendix(a, dataset_label, idx_prefix):
    groups = {}
    for r in a["rows"]:
        groups.setdefault(r["query"], {})[r["platform"]] = r
    out = [f'<div class="qtools"><button class="qbtn" onclick="toggleAll(\'{idx_prefix}\',true)">Expand all</button>'
           f'<button class="qbtn" onclick="toggleAll(\'{idx_prefix}\',false)">Collapse all</button></div>']
    for qi, (q, plats) in enumerate(groups.items()):
        marks = []
        for p in PLATFORM_ORDER:
            r = plats.get(p)
            if r is None:
                marks.append(f'<span class="qmark miss" title="{esc(p)}: not run">&ndash;</span>')
            else:
                hit = r.get("brand_hit")
                marks.append(f'<span class="qmark {"hit" if hit else "miss"}" title="{esc(p)}">'
                             f'{"&#10003;" if hit else "&middot;"}</span>')
        bodies = []
        for p in PLATFORM_ORDER:
            r = plats.get(p)
            if r is None:
                continue
            cites = " &middot; ".join(
                f'<a href="https://{esc(u)}" target="_blank" rel="noopener">{esc(u if len(u) <= 70 else u[:67] + "...")}</a>'
                for u in r["urls"][:12]) or '<em>(no citations)</em>'
            bodies.append(f'<div class="qresp"><div class="rl">{esc(p)}</div>'
                          f'<div class="rt">{esc(r["response"])}</div>'
                          f'<div class="rc">{cites}</div></div>')
        out.append(f'<div class="qrow" id="{idx_prefix}q{qi}">'
                   f'<div class="qhead" onclick="this.parentNode.classList.toggle(\'open\')">'
                   f'<span class="qq">{esc(q)}</span><span class="qmarks">{"".join(marks)}</span></div>'
                   f'<div class="qbody">{"".join(bodies)}</div></div>')
    return "".join(out)


# ---------------------------------------------------------------- assembly
def build(cfg, branded, organic, out_path):
    brand = cfg["brand"]
    category = cfg.get("category", "")
    today = cfg.get("date") or date.today().strftime("%B %Y")
    title = cfg.get("title") or f"{brand}: how AI sees you"

    both = branded and organic
    primary = branded or organic

    # ---- hero stat cards (mode-aware)
    cards = []
    if branded:
        b = branded
        cards.append(('cyan', f'{b["brand_rows"]}/{b["n"]}',
                      'branded answers engage with the brand substantively'))
        cards.append(('gold', str(b["owned_citations"]),
                      f'citations of {esc(brand)}\'s own site across branded answers'))
    if organic:
        o = organic
        pct = round(100 * o["brand_rows"] / o["n"]) if o["n"] else 0
        cards.append(('coral', f'{o["brand_rows"]}/{o["n"]}',
                      f'unbranded category answers mention {esc(brand)} ({pct}%)'))
        top = o["competitors"][0] if o["competitors"] else None
        if top:
            cards.append(('', f'{round(100*top["count"]/o["n"])}%',
                          f'the most-cited name ({esc(top["name"])}) appears in this share of category answers'))
    if len(cards) < 4 and primary:
        cards.append(('', str(primary["total_citations"]), 'total verified citations analyzed'))
    cards = cards[:4]
    cards_html = "".join(
        f'<div class="card"><div class="stat-n {c}">{v}</div><div class="stat-l">{l}</div></div>'
        for c, v, l in cards)

    # ---- exec read (config override wins; else sober auto-draft)
    exec_ps = cfg.get("exec_summary")
    if not exec_ps:
        exec_ps = []
        if branded:
            b = branded
            exec_ps.append(
                f'When people ask AI assistants about {brand} directly, the answers are substantive: '
                f'{b["brand_rows"]} of {b["n"]} branded answers in this sample engage with the brand, '
                f'and {brand}\'s own site earns {b["owned_citations"]} citations, '
                f'making it {"the top source AI leans on" if b["sources"] and b["sources"][0]["type"] == "owned" else "a primary source"} for the brand story.')
        if organic:
            o = organic
            leader = o["competitors"][0]["name"] if o["competitors"] else "the category leader"
            exec_ps.append(
                f'On unbranded category questions, {brand} appears in {o["brand_rows"]} of {o["n"]} answers in this sample, '
                f'while {leader} and other established names carry most of the conversation. '
                f'This is a common shape for a specialized company in a category with large incumbents, '
                f'and it maps where discovery can grow.')
        if both:
            exec_ps.append(
                'Read together, the two runs separate destination from discovery: AI already tells the brand\'s story '
                'well when asked by name, and the same machinery can be pointed at the questions buyers ask '
                'before they know the name. This is a small-sample, directional read, not a verdict; '
                'a fuller audit would confirm where the most efficient openings lie.')
        else:
            exec_ps.append('This is a small-sample, directional read, not a verdict; '
                           'a fuller audit across more queries would confirm where the most efficient openings lie.')
    exec_html = "".join(f"<p>{p if cfg.get('exec_summary') else esc(p)}</p>" for p in exec_ps)

    # ---- nav links & sections
    navlinks, sections = [], []

    def add(anchor, label, html_block):
        navlinks.append(f'<a href="#{anchor}">{label}</a>')
        sections.append(html_block)

    # contrast (both datasets)
    if both:
        b, o = branded, organic
        contrast = f'''
<section id="contrast">
  <div class="gh">The headline contrast</div>
  <h2>Known by name, not yet found by category</h2>
  <div class="gsub">The same five assistants, two kinds of questions. Asked about {esc(brand)} directly,
  AI answers fluently and cites the brand\'s own site. Asked category questions with no brand named,
  the brand\'s site {"does not yet appear" if o["owned_citations"] == 0 else "appears far less often"}.</div>
  <div class="contrast">
    <div class="cpanel hot"><div class="lbl">Branded questions <span class="modechip br">by name</span></div>
      <div class="big">{b["owned_citations"]}</div>
      <div class="note">citations of {esc(brand)}\'s own site across {b["n"]} answers
      ({b["owned_answers"]} answers cite it directly)</div></div>
    <div class="cpanel"><div class="lbl">Unbranded questions <span class="modechip org">by category</span></div>
      <div class="big">{o["owned_citations"]}</div>
      <div class="note">citations of the same site when the {esc(o["n"])} answers come from
      category questions with no brand named</div></div>
  </div>
  <div class="bridge">The site already briefs AI well about the brand. The opportunity is pointing that
  same machinery at the questions people ask before they know the name: the content, sources and
  placements that answer category questions.</div>
</section>'''
        add("contrast", "Contrast", contrast)

    # organic competitive landscape
    if organic:
        o = organic
        comp_items = [{"name": brand, "count": o["brand_rows"]}] + o["competitors"][:9]
        comp_items.sort(key=lambda x: -x["count"])
        add("landscape", "Landscape", f'''
<section id="landscape">
  <div class="gh">Unbranded category questions</div>
  <h2>Who AI names when nobody asks for you</h2>
  <div class="gsub">Across {o["n"]} category answers (10 unbranded prompts, five assistants),
  how often each name appears. Counted from the full response text with word-boundary matching.</div>
  {bar_rows(comp_items, you_name=brand, denom=o["n"])}
</section>''')

    # branded per-platform + owned pages
    if branded:
        b = branded
        add("branded", "Branded read", f'''
<section id="branded">
  <div class="gh">Branded questions</div>
  <h2>What happens when people ask about {esc(brand)} by name</h2>
  <div class="gsub">Ten branded prompts (what is it, is it legit, how does it compare, who uses it,
  what does it cost) run on all five assistants. Engagement per assistant, and the specific
  {esc(brand)} pages AI pulls from.</div>
  {platform_cards(b, branded=True)}
  <h2 style="font-size:19px;margin-top:34px">The {esc(brand)} pages AI actually cites</h2>
  {owned_pages_block(b, brand)}
</section>''')

    # organic per-platform (if no branded, show organic platform cards separately)
    if organic and not branded:
        add("platforms", "Assistants", f'''
<section id="platforms">
  <div class="gh">Per-assistant read</div>
  <h2>Where visibility concentrates</h2>
  <div class="gsub">The same questions produce different answers on different assistants;
  concentration on one assistant usually means search-surfaced visibility rather than
  model-embedded knowledge.</div>
  {platform_cards(organic, branded=False)}
</section>''')

    # source intelligence (use primary dataset; both if present -> two tables)
    src_blocks = []
    for label, ds in (("Branded questions", branded), ("Unbranded category questions", organic)):
        if not ds:
            continue
        chip = '<span class="modechip br">by name</span>' if label.startswith("Branded") else '<span class="modechip org">by category</span>'
        src_blocks.append(f'<h2 style="font-size:19px;margin-top:{"34" if src_blocks else "0"}px">{label} {chip}</h2>'
                          + sources_table(ds))
    add("sources", "Sources", f'''
<section id="sources">
  <div class="gh">Source intelligence</div>
  <h2>The sources AI trusts in this category</h2>
  <div class="gsub">Every verified citation, grouped by site and classified. These are the
  publications, references and platforms that shape what AI says here, which makes them
  the working map for content, comms and partnerships.</div>
  {"".join(src_blocks)}
</section>''')

    # opportunities
    opp_items = []
    for rec in (cfg.get("recommendations") or []):
        opp_items.append(f'<div class="opp"><h4>{rec.get("title","")}</h4><p>{rec.get("body","")}</p></div>')
    if not opp_items and organic:
        o = organic
        for g in o["gap_sources"][:5]:
            lbl, _cls = TYPE_INDEX[g["type"]]
            kind = lbl.lower().replace("&amp;", "and")
            article = "An" if kind[0] in "aeiou" else "A"
            opp_items.append(
                f'<div class="opp"><h4>{esc(g["root"])}</h4>'
                f'<p>Cited in <b>{g["answers"]} of {o["n"]}</b> category answers, none of which mention '
                f'{esc(brand)}. {article} {kind} source already shaping this conversation, '
                f'and a natural place to earn presence.</p></div>')
    add("opportunities", "Opportunities", f'''
<section id="opportunities">
  <div class="gh">Where to act</div>
  <h2>Openings this sample suggests</h2>
  <div class="gsub">Directional, drawn from where competitors are cited and the brand is not yet.
  A fuller audit ranks these by effort and expected lift.</div>
  {"".join(opp_items) or '<div class="gsub">(curated recommendations pending)</div>'}
</section>''')

    # query appendices
    appendix_blocks = []
    if branded:
        appendix_blocks.append('<h2 style="font-size:19px">Branded prompt set <span class="modechip br">by name</span></h2>'
                               + query_appendix(branded, "Branded", "b"))
    if organic:
        appendix_blocks.append(f'<h2 style="font-size:19px;margin-top:{"36" if appendix_blocks else "0"}px">'
                               'Unbranded prompt set <span class="modechip org">by category</span></h2>'
                               + query_appendix(organic, "Organic", "o"))
    add("appendix", "Appendix", f'''
<section id="appendix">
  <div class="gh">Full transparency appendix</div>
  <h2>Every question, every answer, every citation</h2>
  <div class="gsub">Nothing summarized away: the complete responses from each assistant and the
  citations behind them, exactly as collected. Click any question to expand.</div>
  {"".join(appendix_blocks)}
</section>''')

    # methodology
    meth_prompts = []
    if branded:
        meth_prompts.append('<div class="gh" style="margin-top:18px">Branded prompts</div><ol class="promptlist">'
                            + "".join(f"<li>{esc(q)}</li>" for q in branded["prompts"]) + "</ol>")
    if organic:
        meth_prompts.append('<div class="gh" style="margin-top:18px">Unbranded prompts</div><ol class="promptlist">'
                            + "".join(f"<li>{esc(q)}</li>" for q in organic["prompts"]) + "</ol>")
    n_desc = " and ".join(filter(None, [
        f'{branded["n"]} branded answers' if branded else None,
        f'{organic["n"]} unbranded answers' if organic else None]))
    add("methodology", "Method", f'''
<section id="methodology">
  <div class="gh">Methodology</div>
  <h2>How this was measured</h2>
  <div class="meth">
    <p><b>Panel:</b> ChatGPT, Claude, Gemini, Perplexity and Grok, each answering with live web
    search enabled, so results reflect what AI finds and cites today rather than static model memory.</p>
    <p style="margin-top:10px"><b>Sample:</b> {n_desc}, collected {esc(today)}. Each prompt runs once
    per assistant; every citation URL is captured, redirect-resolved where applicable, deduplicated
    and classified. Brand and competitor counts use word-boundary matching over the full response text
    and can be reproduced from the appendix above.</p>
    <p style="margin-top:10px"><b>Reading this honestly:</b> this is a small, directional sample
    designed to map the terrain, not a census. Answers vary run to run; the patterns worth acting on
    are the ones that persist across assistants and questions, which a fuller audit confirms.</p>
  </div>
  {"".join(meth_prompts)}
</section>''')

    js = """
function toggleAll(prefix, open){
  document.querySelectorAll('.qrow[id^="'+prefix.charAt(0).toLowerCase()+'q"]').forEach(function(r){
    r.classList.toggle('open', open);
  });
}
"""

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex,nofollow">
<title>{esc(title)} &middot; Innate C3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Jost:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<nav><div class="wrap">
  <a class="wm" href="https://innatec3.com" target="_blank" rel="noopener">innate c<sup>3</sup></a>
  <div class="navlinks">{"".join(navlinks)}</div>
</div></nav>
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">AI Visibility Intelligence &middot; {esc(cfg.get("prepared_for", brand))}</div>
    <h1>{esc(title)}</h1>
    <div><span class="cat">{esc(category)}</span></div>
    <div class="runmeta">{esc(today)} &middot; five AI assistants, live web search &middot; every answer and citation in the appendix</div>
    <div class="cards">{cards_html}</div>
  </header>

  <section id="exec">
    <div class="gh">Executive read</div>
    <h2>What this sample shows</h2>
    <div class="exec">{exec_html}</div>
  </section>

  {"".join(sections)}

  <footer>
    Prepared by <a href="https://innatec3.com" target="_blank" rel="noopener">Innate C3</a> &middot;
    AI Visibility Intelligence &middot; a communications-grade read of how AI assistants describe,
    source and recommend in this category. Confidential, prepared for {esc(cfg.get("prepared_for", brand))}.
  </footer>
</div>
<script>{js}</script>
</body>
</html>'''

    with open(out_path, "w") as f:
        f.write(page)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: alongside config)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(args.config))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    branded = analyze(load_rows(resolve(cfg["branded_data"])), cfg) if cfg.get("branded_data") else None
    organic = analyze(load_rows(resolve(cfg["organic_data"])), cfg) if cfg.get("organic_data") else None
    if not branded and not organic:
        sys.exit("config must name branded_data and/or organic_data")

    out_dir = args.out or base
    out = os.path.join(out_dir, f'{cfg.get("slug", cfg["brand"].lower())}_dashboard.html')
    path = build(cfg, branded, organic, out)
    n = (branded["n"] if branded else 0) + (organic["n"] if organic else 0)
    print(f"built {path}  ({n} answers analyzed)")


if __name__ == "__main__":
    main()
