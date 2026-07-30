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
# Assistant redirect/artifact HOSTS (checked pre-root-grouping: the root of
# vertexaisearch.cloud.google.com is google.com, which would misfile these
# as corporate citations; they are retrieval plumbing, not sources).
ASSISTANT_HOSTS = {"vertexaisearch.cloud.google.com"}

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
# Company/platform sites that are neither the brand nor configured competitors
# (App Store links, payment/commerce platforms): companies, not media. Roots
# only (apps.apple.com groups to apple.com). Extend per client via config
# "corporate_sources". A configured competitor domain always wins over this.
CORPORATE = {"apple.com", "google.com", "shopify.com", "stripe.com", "paypal.com",
             "salesforce.com", "adobe.com", "oracle.com", "sap.com", "microsoft.com",
             "amazon.com", "meta.com", "samsung.com", "intuit.com"}

TYPE_LABELS = [
    ("owned",       "Owned",                  "cyan"),
    ("competitor",  "Competitor-owned",       "rival"),
    ("editorial",   "Editorial &amp; media",  "gold"),
    ("institutional","Institutional &amp; academic", "inst"),
    ("reviews",     "Reviews &amp; talent",   "rev"),
    ("community",   "Community",              "comm"),
    ("social",      "Social",                 "soc"),
    ("reference",   "Reference data",         "ref"),
    ("corporate",   "Company &amp; platform", "corp"),
]
TYPE_INDEX = {k: (lbl, cls) for k, lbl, cls in TYPE_LABELS}


# ---------------------------------------------------------------- utilities
def esc(s):
    return html.escape(str(s if s is not None else ""))


def poss(name):
    """English possessive that survives s-ending brand names."""
    return name + "'" if name.rstrip().endswith("s") else name + "'s"


def with_the(name, cap=False):
    """Prefix an article unless the label already starts with one (portfolio
    labels like 'the PUIG houses' would otherwise render 'the the ...')."""
    if name.lower().startswith(("the ", "a ", "an ")):
        return (name[0].upper() + name[1:]) if cap else (name[0].lower() + name[1:])
    return ("The " if cap else "the ") + name


def bare(name):
    """Strip a leading article from a label ('the PUIG houses' -> 'PUIG houses')."""
    return name[4:] if name.lower().startswith("the ") else name


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
    if "zendesk" in h or h.startswith(("help.", "support.", "faq.")):
        return "help center"
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
            if (r.get("full_response") or "").startswith("ERROR"):
                continue                                          # failed platform call
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
def classify_root(root, owned_roots, competitor_roots, corporate_roots=frozenset()):
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
    if root in CORPORATE or root in corporate_roots:
        return "corporate"
    if root.endswith((".edu", ".gov", ".ac.uk", ".gov.uk")) or root in {"nsf.gov", "nist.gov"}:
        return "institutional"
    return "editorial"


def analyze(rows, cfg):
    """Everything the template needs for ONE dataset (branded or organic)."""
    brand = cfg["brand"]
    exclude = {root_of(host_of(norm_url("https://" + d))) for d in cfg.get("exclude_sources", [])}
    corporate_roots = {root_of(host_of(norm_url("https://" + d))) for d in cfg.get("corporate_sources", [])}
    # Owned domains: a bare registrable domain (babylist.com) matches by root;
    # a platform-hosted subdomain (healthbabylist.zendesk.com) matches by host
    # suffix only, so OTHER brands' zendesk help centers stay third-party.
    owned_roots, owned_hosts = set(), set()
    for d in cfg.get("owned_domains", []):
        h = host_of(norm_url("https://" + d))
        if h == root_of(h):
            owned_roots.add(h)
        else:
            owned_hosts.add(h)
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
    per_platform = {p: {"total": 0, "brand": 0, "citations": 0,
                        "types": defaultdict(int), "top_sources": defaultdict(int)}
                    for p in platforms}
    for r in rows:
        hit = present(bforms, r["response"])
        r["brand_hit"] = hit
        brand_rows += 1 if hit else 0
        pp = per_platform[r["platform"]]
        pp["total"] += 1
        pp["brand"] += 1 if hit else 0
        pp["citations"] += len(r["urls"])

    # competitor presence (word-boundary, alias-aware, assistants excluded),
    # with a per-assistant breakdown for the landscape hover tooltips.
    comp_counts = []
    for c in comp_cfg:
        nm = c["name"]
        if nm.lower() in ASSISTANT_NAMES:
            continue
        forms = match_forms(nm, c.get("aliases"), corpus)
        per_plat = defaultdict(int)
        cnt = 0
        for r in rows:
            if present(forms, r["response"]):
                cnt += 1
                per_plat[r["platform"]] += 1
        comp_counts.append({"name": nm, "count": cnt, "per_platform": dict(per_plat)})
    comp_counts.sort(key=lambda x: -x["count"])
    brand_per_plat = {p: d["brand"] for p, d in per_platform.items()}

    # PROMINENCE: being present is one thing; being named EARLY and FIRST is
    # the leader question. For each answer, find the first-mention offset of
    # the brand and each competitor; whoever appears first among the measured
    # names is "named first" for that answer.
    def first_pos(forms, text):
        best = None
        for f in forms:
            m = re.search(r"\b" + re.escape(f) + r"\b", text or "", re.I)
            if m and (best is None or m.start() < best):
                best = m.start()
        return best
    comp_forms = {c["name"]: match_forms(c["name"],
                  next((cc.get("aliases") for cc in comp_cfg if cc["name"] == c["name"]), None),
                  corpus) for c in comp_counts}
    named_first = defaultdict(int)
    brand_pos_shares = []
    for r in rows:
        text = r["response"]
        entries = []
        bp = first_pos(bforms, text)
        if bp is not None:
            entries.append(("__brand__", bp))
            if len(text) > 0:
                brand_pos_shares.append(bp / len(text))
        for nm, forms in comp_forms.items():
            cp = first_pos(forms, text)
            if cp is not None:
                entries.append((nm, cp))
        if entries:
            named_first[min(entries, key=lambda e: e[1])[0]] += 1
    prominence = {
        "brand_first": named_first.get("__brand__", 0),
        "brand_present": brand_rows,
        "avg_entry": (sum(brand_pos_shares) / len(brand_pos_shares)) if brand_pos_shares else None,
        "leaderboard": sorted(
            [{"name": brand, "count": named_first.get("__brand__", 0)}] +
            [{"name": nm, "count": named_first.get(nm, 0)} for nm in comp_forms],
            key=lambda x: -x["count"]),
    }

    # citations: per-URL and per-root aggregation (+ per-platform counts for
    # the source-table assistant dots and the per-assistant mix bars)
    url_counts = defaultdict(int)          # display url -> citations
    root_agg = {}                          # root -> {count, answers:set, platforms:{p:n}, type}
    total_citations = 0
    for i, r in enumerate(rows):
        for u in r["urls"]:
            host = host_of(u)
            if host in ASSISTANT_HOSTS:
                continue             # retrieval plumbing, not a source
            root = root_of(host)
            if root in exclude:      # config escape hatch for junk sources
                continue
            if any(host == oh or host.endswith("." + oh) for oh in owned_hosts):
                typ = "owned"
                root = host          # aggregate under the brand's help-center host
            else:
                typ = classify_root(root, owned_roots, competitor_roots, corporate_roots)
            if typ is None:
                continue
            total_citations += 1
            url_counts[(u, typ)] += 1
            a = root_agg.setdefault(root, {"count": 0, "answers": set(),
                                           "platforms": defaultdict(int), "type": typ})
            a["count"] += 1
            a["answers"].add(i)
            a["platforms"][r["platform"]] += 1
            pp = per_platform[r["platform"]]
            pp["types"][typ] += 1
            pp["top_sources"][root] += 1

    sources = [{"root": k, "count": v["count"], "answers": len(v["answers"]),
                "platforms": dict(v["platforms"]), "type": v["type"]}
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
    _owned_answer_sets = [v["answers"] for v in root_agg.values() if v["type"] == "owned"]
    owned_answers = len(set().union(*_owned_answer_sets)) if _owned_answer_sets else 0
    home_ct = sum(p["count"] for p in owned_pages if p["kind"] == "homepage")

    # opportunity: sources cited in answers where brand is absent but competitors present
    gap_sources = []
    for s in sources:
        if s["type"] in ("owned", "corporate"):    # can't pitch an app store
            continue
        answers = root_agg[s["root"]]["answers"]
        brand_in = sum(1 for i in answers if rows[i]["brand_hit"])
        if s["answers"] >= 2 and brand_in == 0:
            gap_sources.append({**s, "brand_in": brand_in})

    # PRESENCE / ABSENCE DRIVERS: which sources travel with answers that name
    # the brand, and which dominate the answers that omit it.
    hit_src, miss_src = defaultdict(int), defaultdict(int)
    for i, r in enumerate(rows):
        for u in r["urls"]:
            host = host_of(u)
            if host in ASSISTANT_HOSTS:
                continue
            root = root_of(host)
            if root in exclude or classify_root(root, owned_roots, competitor_roots, corporate_roots) is None:
                continue
            (hit_src if r["brand_hit"] else miss_src)[root] += 1
    presence_drivers = sorted(((k, v) for k, v in hit_src.items()), key=lambda kv: -kv[1])[:4]
    absence_drivers = sorted(((k, v) for k, v in miss_src.items() if k not in owned_roots),
                             key=lambda kv: -kv[1])[:4]

    return {
        "rows": rows, "n": n, "platforms": platforms, "bforms": bforms,
        "prominence": prominence,
        "presence_drivers": presence_drivers, "absence_drivers": absence_drivers,
        "brand_per_platform": brand_per_plat,
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
.typechip.corp{color:#c9cfdb;border-color:rgba(180,190,210,.28);background:rgba(180,190,210,.05)}
.donutwrap{display:grid;grid-template-columns:240px 1fr;gap:26px;align-items:center;
 background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px 26px;margin:6px 0 16px}
@media(max-width:640px){.donutwrap{grid-template-columns:1fr;justify-items:center}}
.donut{width:220px;height:220px}
.donut a path{transition:opacity .2s}
.donut a:hover path{opacity:.75}
.dlegend{list-style:none;width:100%}
.dlegend li{display:grid;grid-template-columns:12px 1fr auto;gap:10px;align-items:center;padding:7px 0;
 border-bottom:1px solid rgba(255,255,255,.05);font-size:13px}
.dlegend li:last-child{border-bottom:none}
.dlegend .sw{width:12px;height:12px;border-radius:3px}
.dlegend a{color:var(--ink);text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dlegend a:hover{color:var(--cyan);text-decoration:underline}
.dlegend .dv{font-family:'Jost',sans-serif;font-weight:600;color:var(--ink);white-space:nowrap}
.dlegend .dk{color:var(--muted);font-size:11px;margin-left:6px}
.srcfilters{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 12px}
.sfbtn{font-family:'Jost',sans-serif;font-size:11px;letter-spacing:.07em;text-transform:uppercase;
 font-weight:600;color:var(--ink2);background:rgba(255,255,255,.03);border:1px solid var(--line);
 border-radius:999px;padding:6px 13px;cursor:pointer}
.sfbtn:hover{color:var(--ink);border-color:rgba(203,171,109,.4)}
.sfbtn.on{color:#2a1008;background:var(--cta);border-color:transparent}
/* v1.2 interactivity */
.navlinks a.on{color:var(--gold);background:rgba(203,171,109,.08)}
.card,.plat{transition:transform .25s cubic-bezier(.2,.7,.2,1),box-shadow .3s}
.card:hover,.plat:hover{transform:translateY(-3px);box-shadow:0 0 40px -22px var(--gold-g),0 24px 60px -44px #000}
.pre{opacity:0;transform:translateY(16px)}
.go{opacity:1;transform:none;transition:opacity .6s ease,transform .6s cubic-bezier(.2,.7,.2,1)}
.of{transition:width 1s cubic-bezier(.2,.7,.2,1)}
[data-tip]{position:relative;cursor:default}
[data-tip]:hover::after{content:attr(data-tip);position:absolute;left:0;bottom:calc(100% + 8px);
 z-index:40;background:#191c24;border:1px solid rgba(255,255,255,.12);border-radius:9px;
 padding:8px 12px;font-size:11.5px;line-height:1.5;color:var(--ink);white-space:pre-line;
 min-width:200px;max-width:320px;box-shadow:0 18px 50px -18px #000;font-family:'Inter',sans-serif;font-weight:400}
.mixbar{display:flex;height:7px;border-radius:999px;overflow:hidden;margin-top:10px;background:rgba(255,255,255,.05)}
.mixbar span{display:block;height:100%}
.mix-owned{background:linear-gradient(90deg,#5aa8f0,#74d0ff)}
.mix-editorial{background:linear-gradient(90deg,#cbab6d,#e2c98f)}
.mix-competitor{background:#d9a24f}
.mix-institutional{background:#93a9ff}
.mix-reviews{background:#f2b8d0}
.mix-community{background:#8ce6aa}
.mix-social{background:#c8aaf5}
.mix-reference{background:#8cd6cd}
.mix-corporate{background:#8a93a6}
.econrow{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 20px;margin:0 0 16px}
.econbar{display:flex;height:16px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.05)}
.econbar span{display:block;height:100%}
.econlegend{display:flex;flex-wrap:wrap;gap:12px;margin-top:11px;font-size:11.5px;color:var(--ink2)}
.econlegend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--gold)}
th.sortable::after{content:' \\2195';opacity:.4}
tr.xrow{display:none}
#appx .showall,.showall{margin-top:12px;display:inline-block}
mark.bm{background:rgba(116,208,255,.18);color:var(--cyan);border-radius:3px;padding:0 2px}
.qsearch{width:100%;max-width:360px;box-sizing:border-box;font:inherit;font-size:13px;color:var(--ink);
 background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:999px;padding:9px 16px;outline:none}
.qsearch:focus{border-color:rgba(203,171,109,.4)}
.dlegend li{cursor:default}
.dlegend li.hl a{color:var(--cyan)}
.donut path{transition:opacity .2s,transform .2s;transform-origin:110px 110px}
.donut path.hl{opacity:.85;transform:scale(1.03)}
@media print{.pre{opacity:1!important;transform:none!important}
 .of{transition:none}
 tr.xrow{display:table-row!important}
 [data-tip]:hover::after{display:none}}
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
    """SoV bars with grow-in animation + hover tooltip carrying the
    per-assistant breakdown when the item provides one."""
    if not items:
        return '<div class="gsub">(none surfaced)</div>'
    mx = max(i["count"] for i in items) or 1
    out = []
    for i in items:
        you = you_name and i["name"].lower() == you_name.lower()
        pct = f'{round(100*i["count"]/denom)}%' if denom else str(i["count"])
        pp = i.get("per_platform") or {}
        tip = ""
        if pp:
            parts = [f'{p} {pp[p]}' for p in PLATFORM_ORDER if pp.get(p)]
            # plain text only: CSS attr() tooltips render characters, not HTML
            tip_txt = f'{i["name"]}, named per assistant:\n' + ", ".join(parts or ["(none)"])
            tip = f' data-tip="{esc(tip_txt)}"'
        w = round(100 * i["count"] / mx)
        out.append(
            f'<div class="orow"{tip}><span class="bar-name{" you" if you else ""}">{esc(i["name"])}'
            f'{" (you)" if you else ""}</span>'
            f'<span class="ot"><span class="of{" you" if you else ""} anim" style="--w:{w}%;width:{w}%"></span></span>'
            f'<span class="ov">{i["count"]} &middot; {pct}</span></div>')
    return "".join(out)


def platform_cards(a, branded):
    """Per-assistant cards, each with a stacked source-mix bar (its sourcing
    personality) and a hover tooltip naming its top sources."""
    out = []
    for p in a["platforms"]:
        d = a["per_platform"][p]
        rate = d["brand"] / d["total"] if d["total"] else 0
        cls = " full" if rate >= 0.999 else ""
        if branded:
            # Presence on branded prompts is a baseline, not an accomplishment:
            # the informative per-assistant number is how much sourcing it does.
            big, sub = str(d["citations"]), f'citations &middot; {d["brand"]}/{d["total"]} answers engaged'
            cls = ""
        else:
            big, sub = f'{d["brand"]}/{d["total"]}', "of its answers name the brand"
        # stacked mix bar of this assistant's citation types
        tot = sum(d["types"].values())
        mix = ""
        if tot:
            segs = "".join(
                f'<span class="mix-{t}" style="width:{100*d["types"][t]/tot:.1f}%" title="{TYPE_INDEX[t][0]}: {d["types"][t]}"></span>'
                for t, _l, _c in TYPE_LABELS if d["types"].get(t))
            mix = f'<div class="mixbar">{segs}</div>'
        top3 = sorted(d["top_sources"].items(), key=lambda kv: -kv[1])[:3]
        tip = ""
        if top3:
            tip_txt = f'{p}, top sources:\n' + "\n".join(f'{r}: {c} citations' for r, c in top3)
            tip = f' data-tip="{esc(tip_txt)}"'
        out.append(f'<div class="plat"{tip}><div class="pn">{esc(p)}</div>'
                   f'<div class="pv{cls}">{big}</div>'
                   f'<div class="pl">{sub}</div>{mix}</div>')
    # one legend for all the mix bars: which color is which source type
    types_present = [t for t, _l, _c in TYPE_LABELS
                     if any(a["per_platform"][p]["types"].get(t) for p in a["platforms"])]
    legend = ""
    if types_present:
        sw = "".join(f'<span><i class="mix-{t}"></i>{TYPE_INDEX[t][0]}</span>' for t in types_present)
        legend = (f'<div class="econlegend" style="margin-top:12px">'
                  f'<span style="color:var(--muted)">Bar under each assistant: its citation mix by source type.</span>'
                  f'{sw}</div>')
    return f'<div class="plats">{"".join(out)}</div>{legend}'


def economy_bar(a):
    """One stacked bar: the dataset's citation share by source type. The
    'source economy' of the category at a glance (editorial-driven vs
    community-driven vs review-driven)."""
    tot = sum(v["count"] for v in a["type_totals"].values()) or 1
    segs, legend = [], []
    for t, lbl, _c in TYPE_LABELS:
        v = a["type_totals"].get(t)
        if not v:
            continue
        pct = 100 * v["count"] / tot
        segs.append(f'<span class="mix-{t}" style="width:{pct:.1f}%" title="{lbl}: {v["count"]} citations ({pct:.0f}%)"></span>')
        legend.append(f'<span><i class="mix-{t}"></i>{lbl} {pct:.0f}%</span>')
    # Four-bucket rollup (matches the roundtable deck's owned/earned/social/
    # reference view) so the two artifacts reconcile at a glance while the
    # finer practitioner taxonomy above stays primary.
    tt = a["type_totals"]
    def _s(*keys):
        return sum(tt.get(k, {}).get("count", 0) for k in keys)
    ro = {"Owned": _s("owned"),
          "Third-party earned": _s("editorial", "institutional", "corporate", "competitor"),
          "Social &amp; community": _s("social", "community"),
          "Reference &amp; reviews": _s("reference", "reviews")}
    rollup = " &middot; ".join(f'{k} {round(100*v/tot)}%' for k, v in ro.items() if v)
    return (f'<div class="econrow"><div class="econbar">{"".join(segs)}</div>'
            f'<div class="econlegend">{"".join(legend)}</div>'
            f'<div style="margin-top:9px;font-size:11px;color:var(--muted)">Rolled up: {rollup}</div></div>')


_DONUT_COLORS = ["#74d0ff", "#cbab6d", "#f0876a", "#5cf08a", "#d8c6f5", "rgba(255,255,255,.22)"]


def owned_donut(a, brand):
    """Clickable SVG donut of the top 5 owned pages (+ aggregated remainder).
    Slices and legend entries link to the live page."""
    pages = a["owned_pages"]
    if not pages:
        return ""
    top = pages[:5]
    rest = pages[5:]
    items = [{"label": p["url"], "count": p["count"], "kind": p["kind"],
              "href": "https://" + p["url"]} for p in top]
    # geometry over the top-5 sum (full ring, no filler slice); legend
    # percentages over ALL owned citations so shares stay honest
    total = sum(i["count"] for i in items) or 1
    owned_total = a["owned_citations"] or 1
    import math
    cx = cy = 110
    r_out, r_in = 104, 64
    ang = -90.0
    paths, legend = [], []
    for idx, it in enumerate(items):
        frac = it["count"] / total
        sweep = frac * 360.0
        a0, a1 = math.radians(ang), math.radians(ang + max(sweep - 0.8, 0.4))
        large = 1 if sweep > 180 else 0
        x0o, y0o = cx + r_out * math.cos(a0), cy + r_out * math.sin(a0)
        x1o, y1o = cx + r_out * math.cos(a1), cy + r_out * math.sin(a1)
        x0i, y0i = cx + r_in * math.cos(a1), cy + r_in * math.sin(a1)
        x1i, y1i = cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)
        d = (f"M{x0o:.1f},{y0o:.1f} A{r_out},{r_out} 0 {large} 1 {x1o:.1f},{y1o:.1f} "
             f"L{x0i:.1f},{y0i:.1f} A{r_in},{r_in} 0 {large} 0 {x1i:.1f},{y1i:.1f} Z")
        color = _DONUT_COLORS[idx % len(_DONUT_COLORS)]
        path = (f'<path d="{d}" fill="{color}" data-di="{idx}">'
                f'<title>{esc(it["label"])}: {it["count"]} citations</title></path>')
        if it["href"]:
            path = f'<a href="{esc(it["href"])}" target="_blank" rel="noopener">{path}</a>'
        paths.append(path)
        short = it["label"] if len(it["label"]) <= 52 else it["label"][:49] + "..."
        name = (f'<a href="{esc(it["href"])}" target="_blank" rel="noopener">{esc(short)}</a>'
                if it["href"] else f'<span style="color:var(--muted)">{esc(short)}</span>')
        kind = f'<span class="dk">{esc(it["kind"])}</span>' if it["kind"] else ""
        legend.append(f'<li data-di="{idx}"><span class="sw" style="background:{color}"></span>'
                      f'<span style="min-width:0">{name}{kind}</span>'
                      f'<span class="dv">{it["count"]}&times; &middot; {round(100*it["count"]/owned_total)}% of owned</span></li>')
        ang += sweep
    svg = (f'<svg class="donut" viewBox="0 0 220 220" role="img" '
           f'aria-label="Top {esc(brand)} pages AI cites">{"".join(paths)}'
           f'<text x="110" y="104" text-anchor="middle" fill="#fff" font-family="Jost,sans-serif" '
           f'font-weight="700" font-size="30">{a["owned_citations"]}</text>'
           f'<text x="110" y="126" text-anchor="middle" fill="#9aa1ad" font-family="Inter,sans-serif" '
           f'font-size="10.5">owned citations</text></svg>')
    cap = (f'<li style="border-bottom:none;color:var(--muted);font-size:11px;grid-column:1/-1;display:block">Top 5 pages shown'
           + (f'; {len(rest)} more in the table below' if rest else '') + '</li>')
    return f'<div class="donutwrap">{svg}<ul class="dlegend">{"".join(legend)}{cap}</ul></div>'


def sources_table(a, table_id, limit=18):
    """Classified source table: per-type filter chips, sortable numeric
    columns, source link-outs, per-assistant citation counts on the dots, and
    a show-all toggle past the fold. Client-side only, print-safe."""
    types_present = []
    for k, lbl, cls in TYPE_LABELS:
        if any(s["type"] == k for s in a["sources"]):
            types_present.append((k, lbl))
    chips = [f'<button class="sfbtn on" data-t="all" onclick="srcFilter(\'{table_id}\',this)">All</button>']
    chips += [f'<button class="sfbtn" data-t="{k}" onclick="srcFilter(\'{table_id}\',this)">{lbl}</button>'
              for k, lbl in types_present]
    rows = []
    for idx, s in enumerate(a["sources"]):
        lbl, cls = TYPE_INDEX[s["type"]]
        dots = "".join(
            f'<span class="pdot{" on" if s["platforms"].get(p) else ""}" '
            f'title="{esc(p)}: {s["platforms"].get(p, 0)} citations"></span>'
            for p in PLATFORM_ORDER)
        hidden = ' class="xrow"' if idx >= limit else ""
        rows.append(f'<tr data-t="{s["type"]}"{hidden}>'
                    f'<td class="src"><a class="upath" href="https://{esc(s["root"])}" target="_blank" '
                    f'rel="noopener">{esc(s["root"])}</a></td>'
                    f'<td><span class="typechip {cls}">{lbl}</span></td>'
                    f'<td class="num" data-v="{s["count"]}">{s["count"]}</td>'
                    f'<td class="num" data-v="{s["answers"]}">{s["answers"]}/{a["n"]}</td>'
                    f'<td class="pdots">{dots}</td></tr>')
    more = ""
    if len(a["sources"]) > limit:
        more = (f'<button class="sfbtn showall" onclick="showAllRows(\'{table_id}\',this)">'
                f'Show all {len(a["sources"])} sources</button>')
    return (f'<div id="{table_id}"><div class="srcfilters">{"".join(chips)}</div>'
            f'<table><thead><tr><th>Source</th><th>Type</th>'
            f'<th class="sortable" onclick="sortRows(\'{table_id}\',2)">Citations</th>'
            f'<th class="sortable" onclick="sortRows(\'{table_id}\',3)">Answers citing</th>'
            f'<th>Assistants</th></tr></thead><tbody>'
            + "".join(rows) + f"</tbody></table>{more}</div>")


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


def _highlight(text_esc, forms):
    """Wrap brand mentions in <mark> inside already-escaped text: the receipts,
    visible at a glance when a response is expanded."""
    if not forms:
        return text_esc
    pat = re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")\b",
                     re.IGNORECASE)
    return pat.sub(lambda m: f'<mark class="bm">{m.group(0)}</mark>', text_esc)


def query_appendix(a, dataset_label, idx_prefix):
    groups = {}
    for r in a["rows"]:
        groups.setdefault(r["query"], {})[r["platform"]] = r
    pid = idx_prefix.lower()
    plat_chips = "".join(
        f'<button class="qbtn" data-p="{esc(p)}" onclick="platFilter(\'{pid}\',this)">{esc(p)}</button>'
        for p in a["platforms"])
    out = [f'<div class="qtools" id="{pid}tools">'
           f'<button class="qbtn" onclick="toggleAll(\'{pid}\',true)">Expand all</button>'
           f'<button class="qbtn" onclick="toggleAll(\'{pid}\',false)">Collapse all</button>'
           f'<button class="qbtn on" data-p="all" onclick="platFilter(\'{pid}\',this)">All assistants</button>'
           f'{plat_chips}'
           f'<input class="qsearch" type="search" placeholder="Filter questions&hellip;" '
           f'oninput="qSearch(\'{pid}\',this.value)"></div>']
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
            bodies.append(f'<div class="qresp" data-p="{esc(p)}"><div class="rl">{esc(p)} '
                          f'&middot; {len(r["urls"])} citations</div>'
                          f'<div class="rt">{_highlight(esc(r["response"]), a.get("bforms"))}</div>'
                          f'<div class="rc">{cites}</div></div>')
        out.append(f'<div class="qrow" id="{pid}q{qi}" data-q="{esc(q.lower())}">'
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

    # ---- hero stat cards (mode-aware). Branded presence is a BASELINE, not an
    # accomplishment (branded prompts trivially engage the brand), so branded
    # cards lead with sourcing: owned citations + owned share of all citations.
    cards = []
    if branded:
        b = branded
        share = round(100 * b["owned_citations"] / b["total_citations"]) if b["total_citations"] else 0
        cards.append(('gold', str(b["owned_citations"]),
                      f'citations of {esc(poss(brand))} own site across branded answers'))
        cards.append(('cyan', f'{share}%',
                      'of all branded-answer citations come from the brand\'s own site'))
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
    if len(cards) < 4 and primary:
        cards.append(('', str(len(primary["sources"])), 'distinct sources cited across the sample'))
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
            share = round(100 * b["owned_citations"] / b["total_citations"]) if b["total_citations"] else 0
            top_ext = next((s for s in b["sources"] if s["type"] != "owned"), None)
            ext_note = (f', with {top_ext["root"]} the most-cited outside source' if top_ext else '')
            exec_ps.append(
                f'Asked about {brand} by name, all {b["brand_rows"]} of {b["n"]} answers engage with the brand, '
                f'which is the expected baseline for branded questions. The informative read is the sourcing: '
                f'{poss(brand)} own site accounts for {b["owned_citations"]} of the citations behind those answers '
                f'({share}%){ext_note}. Who AI trusts to tell the story matters more than whether it answers.')
            # sourcing contributors: assistant asymmetry + owned depth
            pp = b["per_platform"]
            hi = max(b["platforms"], key=lambda p: pp[p]["citations"])
            lo = min(b["platforms"], key=lambda p: pp[p]["citations"])
            deep = b["owned_citations"] - b["owned_home_citations"]
            depth_note = (f'{deep} of the {b["owned_citations"]} owned citations pull pages deeper than the '
                          f'homepage, so the site\'s publishing is carrying the story'
                          if b["owned_citations"] and deep >= b["owned_home_citations"] else
                          f'{b["owned_home_citations"]} of the {b["owned_citations"]} owned citations point at the '
                          f'homepage, so deeper citable pages are the clearest owned opening')
            if pp[hi]["citations"] >= 2 * max(1, pp[lo]["citations"]):
                exec_ps.append(
                    f'Two patterns shape that sourcing. The assistants differ sharply in how much they retrieve: '
                    f'{hi} builds its answers on {pp[hi]["citations"]} citations while {lo} uses '
                    f'{pp[lo]["citations"]} on the same questions, so the brand\'s citable footprint matters most '
                    f'where retrieval is heaviest. And {depth_note}.')
            else:
                exec_ps.append(f'On the owned side, {depth_note}.')
        if organic:
            o = organic
            top = o["competitors"][0] if o["competitors"] else None
            if top and o["brand_rows"] >= top["count"]:
                # leader framing: the brand out-appears every configured competitor
                exec_ps.append(
                    f'On unbranded category questions, {brand} appears in {o["brand_rows"]} of {o["n"]} answers '
                    f'in this sample, the most of any name measured ({top["name"]} follows at {top["count"]}). '
                    f'With presence this established, the working questions shift from being found to being '
                    f'the default recommendation: prominence within answers, and which sources carry the story.')
            else:
                leader = top["name"] if top else "the category leader"
                exec_ps.append(
                    f'On unbranded category questions, {brand} appears in {o["brand_rows"]} of {o["n"]} answers in this sample, '
                    f'while {leader} and other established names carry most of the conversation. '
                    f'This is a common shape for a specialized company in a category with large incumbents, '
                    f'and it maps where discovery can grow.')
            # contributors: which sources travel with presence vs absence
            pd, ad = o.get("presence_drivers") or [], o.get("absence_drivers") or []
            drv = []
            if pd:
                drv.append('presence travels with ' + ' and '.join(f'{r}' for r, _c in pd[:2])
                           + f' ({pd[0][1]}'
                           + (f' and {pd[1][1]}' if len(pd) > 1 else '')
                           + f' citations in the answers that name {brand})')
            if ad:
                drv.append('the answers that omit it lean on ' + ' and '.join(f'{r}' for r, _c in ad[:2]))
            pr = o.get("prominence") or {}
            if pr.get("brand_present"):
                drv.append(f'where {brand} does appear, it is the first name mentioned in '
                           f'{pr.get("brand_first", 0)} of those {pr["brand_present"]} answers')
            if drv:
                exec_ps.append(('Behind the topline: ' + '; '.join(drv) + '.').replace('  ', ' '))
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
      <div class="note">citations of {esc(poss(brand))} own site across {b["n"]} answers
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

    # prominence (organic): present is one thing, named first is another
    if organic and (organic.get("prominence") or {}).get("leaderboard"):
        o = organic
        pr = o["prominence"]
        lead_items = [{"name": x["name"], "count": x["count"]}
                      for x in pr["leaderboard"][:8] if x["count"] > 0]
        entry = pr.get("avg_entry")
        entry_card = ""
        if entry is not None:
            entry_card = (f'<div class="card"><div class="stat-n cyan">top {max(1, round(entry * 100))}%</div>'
                          f'<div class="stat-l">average entry point of the first {esc(bare(brand))} mention '
                          f'within the answer text, when it appears</div></div>')
        first_share = (f'{pr["brand_first"]} of {pr["brand_present"]}'
                       if pr.get("brand_present") else "0")
        prom_html = (
            '<section id="prominence">'
            '<div class="gh">Prominence</div>'
            '<h2>Present is one thing, first is another</h2>'
            '<div class="gsub">AI answers are read top-down and often summarized further, so the first '
            'brand named carries outsized weight. Definition: for each answer, the measured name (the '
            'brand or a configured competitor) whose first word-boundary mention appears earliest in '
            'the answer text is counted as named first. Directional, like everything in this sample.</div>'
            '<div class="cards" style="grid-template-columns:repeat(2,1fr);max-width:640px">'
            f'<div class="card"><div class="stat-n gold">{first_share}</div>'
            f'<div class="stat-l">answers naming {esc(brand)} put the name ahead of every measured '
            'competitor</div></div>'
            f'{entry_card}</div>'
            f'<h2 style="font-size:19px;margin-top:26px">Named first, across all {o["n"]} answers</h2>'
            f'{bar_rows(lead_items, you_name=brand, denom=o["n"])}'
            '</section>')
        add("prominence", "Prominence", prom_html)

    # branded per-platform + owned pages
    if branded:
        b = branded
        add("branded", "Branded read", f'''
<section id="branded">
  <div class="gh">Branded questions</div>
  <h2>Who AI trusts to tell {esc(with_the(brand))} story</h2>
  <div class="gsub">Ten branded prompts (what is it, is it legit, how does it compare, who uses it,
  what does it cost) run on all five assistants. Every assistant engages, which is the baseline
  for branded questions; the read that matters is how much sourcing each does and which
  {esc(brand)} pages AI pulls from.</div>
  {platform_cards(b, branded=True)}
  <h2 style="font-size:19px;margin-top:34px">{esc(with_the(brand, cap=True))} pages AI actually cites</h2>
  {owned_donut(b, brand)}
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
    for label, ds, tid in (("Sources behind the branded answers", branded, "srcB"),
                           ("Sources behind the unbranded category answers", organic, "srcO")):
        if not ds:
            continue
        chip = '<span class="modechip br">by name</span>' if ds is branded else '<span class="modechip org">by category</span>'
        src_blocks.append(f'<h2 style="font-size:19px;margin-top:{"34" if src_blocks else "0"}px">{label} {chip}</h2>'
                          + economy_bar(ds) + sources_table(ds, tid))
    add("sources", "Sources", f'''
<section id="sources">
  <div class="gh">Source intelligence</div>
  <h2>The sources AI trusts in this category</h2>
  <div class="gsub">Every verified citation, grouped by site and classified. These are the
  publications, references and platforms that shape what AI says here, which makes them
  the working map for content, comms and partnerships.</div>
  {"".join(src_blocks)}
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
    <p style="margin-top:10px"><b>Per-assistant differences:</b> the assistants retrieve differently
    by design; some build answers on many citations, others on few. Citation volume is a property of
    each model\'s retrieval behavior, not a verdict on the brand\'s visibility, which is why presence
    and prominence are measured per answer rather than by citation count.</p>
    <p style="margin-top:10px"><b>Reading this honestly:</b> this is a small, directional sample
    designed to map the terrain, not a census. Answers vary run to run; the patterns worth acting on
    are the ones that persist across assistants and questions, which a fuller audit confirms.</p>
    {"".join(f'<p style="margin-top:10px">{note}</p>' for note in cfg.get("method_notes") or [])}
  </div>
  {"".join(meth_prompts)}
</section>''')

    js = """
function toggleAll(pid, open){
  document.querySelectorAll('.qrow[id^="'+pid+'q"]').forEach(function(r){
    r.classList.toggle('open', open);
  });
}
function srcFilter(tableId, btn){
  var box = document.getElementById(tableId);
  var t = btn.getAttribute('data-t');
  box.querySelectorAll('.srcfilters .sfbtn').forEach(function(b){ b.classList.toggle('on', b === btn); });
  var shown = 0;
  box.querySelectorAll('tbody tr').forEach(function(r){
    var match = (t === 'all' || r.getAttribute('data-t') === t);
    // filtering reveals all matching rows, including those past the fold
    r.style.display = match ? '' : 'none';
    if (match && t !== 'all') r.classList.remove('xrow');
  });
}
function showAllRows(tableId, btn){
  document.querySelectorAll('#'+tableId+' tr.xrow').forEach(function(r){ r.classList.remove('xrow'); });
  btn.style.display = 'none';
}
function sortRows(tableId, col){
  var tb = document.querySelector('#'+tableId+' tbody');
  var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'));
  var dir = tb.getAttribute('data-dir') === 'asc' ? -1 : 1;
  tb.setAttribute('data-dir', dir === 1 ? 'asc' : 'desc');
  rows.sort(function(a, b){
    var av = parseFloat(a.cells[col].getAttribute('data-v') || 0);
    var bv = parseFloat(b.cells[col].getAttribute('data-v') || 0);
    return dir * (bv - av);
  });
  rows.forEach(function(r){ tb.appendChild(r); });
}
function platFilter(pid, btn){
  var tools = document.getElementById(pid + 'tools');
  var p = btn.getAttribute('data-p');
  tools.querySelectorAll('.qbtn[data-p]').forEach(function(b){ b.classList.toggle('on', b === btn); });
  document.querySelectorAll('.qrow[id^="'+pid+'q"] .qresp').forEach(function(r){
    r.style.display = (p === 'all' || r.getAttribute('data-p') === p) ? '' : 'none';
  });
}
function qSearch(pid, term){
  term = (term || '').toLowerCase();
  document.querySelectorAll('.qrow[id^="'+pid+'q"]').forEach(function(r){
    r.style.display = (!term || (r.getAttribute('data-q') || '').indexOf(term) >= 0) ? '' : 'none';
  });
}
// donut: legend row <-> slice hover sync
document.querySelectorAll('.donutwrap').forEach(function(w){
  function set(idx, on){
    w.querySelectorAll('[data-di="'+idx+'"]').forEach(function(el){ el.classList.toggle('hl', on); });
  }
  w.querySelectorAll('.dlegend li[data-di]').forEach(function(li){
    li.addEventListener('mouseenter', function(){ set(li.getAttribute('data-di'), true); });
    li.addEventListener('mouseleave', function(){ set(li.getAttribute('data-di'), false); });
  });
  w.querySelectorAll('.donut path[data-di]').forEach(function(pa){
    pa.addEventListener('mouseenter', function(){ set(pa.getAttribute('data-di'), true); });
    pa.addEventListener('mouseleave', function(){ set(pa.getAttribute('data-di'), false); });
  });
});
// scroll-reveal + SoV bar grow-in + hero count-up (all print-safe: final
// values live in the markup; JS only animates toward them)
(function(){
  if (!('IntersectionObserver' in window)) return;
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if (!e.isIntersecting) return;
      e.target.classList.add('go');
      e.target.querySelectorAll('.of.anim').forEach(function(bar){
        var w = bar.style.getPropertyValue('--w');
        bar.style.width = '0%';
        requestAnimationFrame(function(){ requestAnimationFrame(function(){ bar.style.width = w; }); });
        bar.classList.remove('anim');
      });
      io.unobserve(e.target);
    });
  }, {threshold: 0.12});
  document.querySelectorAll('section, header.hero').forEach(function(s){
    var r = s.getBoundingClientRect();
    if (r.top > window.innerHeight) s.classList.add('pre');
    io.observe(s);
  });
  // scrollspy: highlight the nav link for the section in view
  var links = {};
  document.querySelectorAll('.navlinks a[href^="#"]').forEach(function(a){
    links[a.getAttribute('href').slice(1)] = a;
  });
  var spy = new IntersectionObserver(function(es){
    es.forEach(function(e){
      var a = links[e.target.id];
      if (!a) return;
      if (e.isIntersecting){
        Object.keys(links).forEach(function(k){ links[k].classList.remove('on'); });
        a.classList.add('on');
      }
    });
  }, {rootMargin: '-30% 0px -60% 0px'});
  document.querySelectorAll('section[id]').forEach(function(s){ spy.observe(s); });
})();
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
    ap.add_argument("--print-slim", action="store_true",
                    help="also emit <slug>_dashboard_print.html with appendix responses "
                         "clamped (mailable PDF weight; full receipts stay in the "
                         "interactive HTML)")
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
    if args.print_slim:
        page = open(path).read()
        slim_css = ("<style>@media print{.qresp .rt{max-height:150px;overflow:hidden}"
                    ".qbody::after{content:'Full verbatim responses live in the interactive "
                    "version of this dashboard.';display:block;font-size:11px;"
                    "color:#6b7280;margin-top:8px}}</style></head>")
        slim = page.replace("</head>", slim_css, 1)
        spath = path.replace("_dashboard.html", "_dashboard_print.html")
        open(spath, "w").write(slim)
        print(f"built {spath}  (print-slim variant)")


if __name__ == "__main__":
    main()
