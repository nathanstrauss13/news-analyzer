#!/usr/bin/env python3
"""Correction Seeker: fact-consistency audit pipeline (operator tool, MVP).

Where PR Signal Finder measures whether a brand shows up in AI answers,
this verifies whether the FACTS AI states about a company are right, traces
each wrong or stale figure to the source page feeding it, and produces a
prioritized correction queue. Per CORRECTION_SEEKER_BRIEF.md.

Not user-facing. Operator-generated, like tools/audit_dashboard.

Pipeline:
  collect  ~20 fact-bearing prompts x STANDARD_5 grounded collectors -> raw.json
  analyze  claim extraction (Claude) -> ground-truth diff -> trace cited pages
           with evidence snapshots -> analysis.json + evidence/
  report   single sober HTML (P3 look) -> report.html

Usage (from tools/correction_seeker/):
  python3 correction_seeker.py collect --config netflix.config.json [--limit 1]
  python3 correction_seeker.py analyze --config netflix.config.json
  python3 correction_seeker.py report  --config netflix.config.json
  python3 correction_seeker.py all     --config netflix.config.json

Classification taxonomy (sober, defensible):
  current           matches the current primary-source figure
  stale_dated       matches an earlier reported figure and is dated honestly
  stale_as_current  matches an earlier reported figure but presented as current
  incorrect         does not match any figure the company has reported
  unverifiable      cannot be checked against a reported figure

Boundary: facts with a primary-source ground truth only. Interpretation
questions never enter this pipeline.
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# --- env (worktree .env if present, else the canonical repo checkout's) ---
for _env in (os.path.join(REPO, ".env"),
             "/Users/nathanstrauss/Desktop/innate c3/innate apps/ai-citation-audit/.env"):
    if os.path.exists(_env):
        for line in open(_env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break

# .env pins a retired OpenRouter slug; override here only, never in shared .env
os.environ["XAI_OPENROUTER_MODEL"] = "x-ai/grok-4.5"

# platforms/ lives in the main checkout, not the mvp worktree
for _root in (REPO,
              "/Users/nathanstrauss/Desktop/innate c3/innate apps/ai-citation-audit"):
    if os.path.isdir(os.path.join(_root, "platforms")):
        sys.path.insert(0, _root)
        break

FETCH_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
EXTRACT_MODEL = os.environ.get("CS_EXTRACT_MODEL", "claude-sonnet-5")

SEVERITY = {"incorrect": 3, "stale_as_current": 2, "stale_dated": 1,
            "unverifiable": 1, "current": 0, "current_basis": 0}

CLASS_LABEL = {
    "current": "Matches current figure",
    "current_basis": "Matches a current figure on a different basis",
    "stale_dated": "Accurate as of an earlier date, dated honestly",
    "stale_as_current": "Earlier figure presented as current",
    "incorrect": "Does not match any reported figure",
    "unverifiable": "Not verifiable against a reported figure",
}

PUBLISHER_HOSTS = ("nytimes.com", "wsj.com", "reuters.com", "bloomberg.com",
                   "cnbc.com", "forbes.com", "ft.com", "variety.com",
                   "hollywoodreporter.com", "theverge.com", "techcrunch.com",
                   "businessinsider.com", "axios.com", "cnn.com", "bbc.")
AGGREGATOR_HINTS = ("stockanalysis", "macrotrends", "statista",
                    "companiesmarketcap", "wisesheets", "bullfincher",
                    "zippia", "craft.co", "globaldata", "similarweb",
                    "companieshistory", "stockdividendscreener", "wallstreetzen",
                    "simplywall", "marketbeat")


# ---------------------------------------------------------------- utilities

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def host_of(url):
    m = re.match(r"https?://([^/]+)", url or "")
    h = (m.group(1) if m else "").lower()
    return h[4:] if h.startswith("www.") else h


def root_host(host):
    parts = (host or "").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


SCALES = [
    (r"trillion|tn|t", 1e12),
    (r"billion|bn|b", 1e9),
    (r"million|mn|m", 1e6),
    (r"thousand|k", 1e3),
]


def parse_number(s):
    """Parse '$45.18 billion', '45,183 million', '16,000', '1997' -> float.
    Returns None when there is no single unambiguous number."""
    if s is None:
        return None
    t = str(s).lower().strip()
    t = re.sub(r"(approximately|about|around|roughly|nearly|over|more than|"
               r"almost|some|an estimated|estimated|~|\+)", " ", t)
    t = t.replace("us$", "$").replace("usd", " ").strip()
    if re.search(r"\d\s*(-|–|to)\s*\$?\d", t):        # a range, not a value
        return None
    m = re.search(r"\$?\s*([\d][\d,]*\.?\d*)\s*(trillion|billion|million|"
                  r"thousand|tn|bn|mn|[tbmk])?\b", t)
    if not m:
        return None
    if t[m.end(1):m.end(1) + 2].strip().startswith("%"):   # a growth rate
        return None
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2) or ""
    for pat, mult in SCALES:
        if unit and re.fullmatch(pat, unit):
            return num * mult
    return num


def within(a, b, tol_pct):
    if a is None or b is None or b == 0:
        return False
    return abs(a - b) / abs(b) <= tol_pct / 100.0


def sig_key(v):
    """Group variants by 3 significant digits so $45.18B and $45.2B cluster."""
    if v is None or v == 0:
        return "0"
    from math import floor, log10
    d = floor(log10(abs(v)))
    return f"{round(v / 10 ** (d - 2)) * 10 ** (d - 2):.6g}"


def number_page_patterns(value):
    """Regexes that find a numeric value in fetched page text, across the
    common surface forms (scaled words, letter suffixes, comma integers)."""
    pats = []
    for unit_words, mult in (("trillion|tn|t", 1e12), ("billion|bn|b", 1e9),
                             ("million|mn|m", 1e6)):
        if value >= mult * 0.9:
            base = value / mult
            reprs = set()
            for dp in (0, 1, 2):
                r = f"{base:.{dp}f}"
                if "." in r:
                    r = r.rstrip("0").rstrip(".")
                if r and float(r) != 0:
                    reprs.add(re.escape(r))
            alt = "|".join(sorted(reprs))
            pats.append(rf"(?<![\d.])(?:\$\s*)?(?:{alt})\s*(?:{unit_words})\b")
    if value == int(value) and 1000 <= value < 1e9:
        pats.append(rf"(?<![\d.]){int(value):,}(?![\d])".replace(",", ","))
    if value < 1000 and value == int(value):        # years and small counts
        pats.append(rf"(?<!\d){int(value)}(?!\d)")
    return pats


def strip_html(raw):
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw,
               flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t)


# ---------------------------------------------------------------- collect

def collect(cfg, run_dir, limit=None):
    from platforms.chatgpt import ChatGPTPlatform
    from platforms.claude import ClaudePlatform
    from platforms.gemini import GeminiPlatform
    from platforms.grok import GrokPlatform
    from platforms.perplexity import PerplexityPlatform

    plats = [("ChatGPT", ChatGPTPlatform()), ("Claude", ClaudePlatform()),
             ("Gemini", GeminiPlatform()), ("Grok", GrokPlatform()),
             ("Perplexity", PerplexityPlatform())]
    prompts = cfg["prompts"][:limit] if limit else cfg["prompts"]
    total = len(prompts) * len(plats)
    print(f"[collect] {len(prompts)} prompts x {len(plats)} platforms "
          f"= {total} calls", flush=True)

    def run_platform(pname, plat):
        out = []
        for i, prompt in enumerate(prompts):
            try:
                result = plat.get_citations(prompt) or {}
            except Exception as e:
                result = {"response": f"ERROR: {e}", "citations": []}
            cits, seen = [], set()
            for c in result.get("citations") or []:
                u = (c.get("url") or "").strip() if isinstance(c, dict) else str(c)
                if u and u not in seen:
                    seen.add(u)
                    cits.append({"url": u,
                                 "domain": (c.get("domain") if isinstance(c, dict) else "") or host_of(u)})
            out.append({"query": prompt, "platform": pname,
                        "full_response": result.get("response") or "",
                        "citations": cits})
            print(f"  [{pname}] {i + 1}/{len(prompts)} done", flush=True)
            time.sleep(1.0)
        return out

    with ThreadPoolExecutor(max_workers=len(plats)) as ex:
        futs = {pname: ex.submit(run_platform, pname, plat) for pname, plat in plats}
        per_plat = {pname: f.result() for pname, f in futs.items()}
    rows = []
    for i in range(len(prompts)):                    # interleave in prompt order
        for pname, _ in plats:
            rows.append(per_plat[pname][i])

    raw = {"brand": cfg["brand"], "slug": cfg["slug"],
           "collected_at": now_utc(), "rows": rows}
    path = os.path.join(run_dir, "raw.json")
    json.dump(raw, open(path, "w"), indent=1)
    n_err = sum(1 for r in rows if r["full_response"].startswith("ERROR"))
    print(f"[collect] DONE rows={len(rows)} errors={n_err} -> {path}", flush=True)
    return raw


# ---------------------------------------------------------------- extract

EXTRACT_SYSTEM = (
    "You extract factual claims from an AI assistant's answer about a company. "
    "You copy quotes verbatim. You never invent claims. Output JSON only, "
    "no markdown fences.")

EXTRACT_TEMPLATE = """FACTS TO LOOK FOR (only these; ignore all other figures):
{facts_block}

ASSISTANT ANSWER TEXT:
---
{response}
---

Extract every claim the answer makes about the listed facts. Return JSON:
{{"claims": [{{"fact_id": "...", "quote": "...", "stated_value": "...",
"stated_as_of": "..." or null, "presented_as_current": true/false}}]}}

Rules:
- "quote": one sentence copied VERBATIM from the answer containing the claim.
- "stated_value": the exact value phrase as written (e.g. "$39 billion",
  "approximately 14,000", "Ted Sarandos and Greg Peters").
- "stated_as_of": the date or period the answer explicitly attaches to this
  value (e.g. "FY2024", "as of December 31, 2024", "2025"), else null.
- "presented_as_current": true when the answer offers the value as the
  company's current fact; false when it is framed as historical.
- One claim per distinct statement. If the answer states the same fact twice
  with different values, extract both. If nothing matches, return {{"claims": []}}.
"""


def extract_claims(cfg, raw, run_dir):
    import anthropic
    client = anthropic.Anthropic()
    facts_block = "\n".join(
        f"- id={f['id']}: {f['label']}. Look for: {f['hints']}"
        for f in cfg["facts"])
    all_claims, n_resp_with = [], 0
    rows = [r for r in raw["rows"] if not r["full_response"].startswith("ERROR")
            and r["full_response"].strip()]
    print(f"[extract] {len(rows)} usable responses, model={EXTRACT_MODEL}",
          flush=True)
    for idx, row in enumerate(rows):
        prompt = EXTRACT_TEMPLATE.format(facts_block=facts_block,
                                         response=row["full_response"][:12000])
        claims = None
        for attempt in range(3):
            try:
                p = prompt if attempt == 0 else (
                    prompt + "\n\nIMPORTANT: Your previous output was not "
                    "valid JSON. Output STRICT valid JSON only; escape all "
                    "double quotes inside strings.")
                msg = client.messages.create(
                    model=EXTRACT_MODEL, max_tokens=4000,
                    system=EXTRACT_SYSTEM,
                    messages=[{"role": "user", "content": p}])
                text = "".join(b.text for b in msg.content
                               if getattr(b, "type", "") == "text")
                start, end = text.find("{"), text.rfind("}")
                try:
                    claims = json.loads(text[start:end + 1],
                                        strict=False).get("claims", [])
                except Exception:
                    # salvage flat claim objects from truncated output
                    claims = []
                    for frag in re.findall(r"\{[^{}]*\}", text[start + 1:]):
                        try:
                            o = json.loads(frag, strict=False)
                            if o.get("fact_id") and o.get("quote"):
                                claims.append(o)
                        except Exception:
                            pass
                    if not claims:
                        raise
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [extract] row {idx} failed: {e}", flush=True)
                    claims = []
                time.sleep(2)
        norm_resp = re.sub(r"[\s*_`#]+", " ", row["full_response"].lower())
        kept = []
        for c in claims:
            q = re.sub(r"[\s*_`#]+", " ", str(c.get("quote", "")).lower()).strip()
            if not q or q[:200] not in norm_resp:      # verbatim-quote gate
                continue
            c["row_idx"] = raw["rows"].index(row)
            c["platform"] = row["platform"]
            c["query"] = row["query"]
            c["citations"] = [x["url"] for x in row["citations"]]
            kept.append(c)
        if kept:
            n_resp_with += 1
        all_claims.extend(kept)
        if (idx + 1) % 10 == 0:
            print(f"  [extract] {idx + 1}/{len(rows)} responses, "
                  f"{len(all_claims)} claims", flush=True)
        time.sleep(0.25)
    print(f"[extract] DONE claims={len(all_claims)} "
          f"responses_with_claims={n_resp_with}", flush=True)
    return all_claims, len(rows), n_resp_with


# ---------------------------------------------------------------- classify

def classify_claim(fact, claim):
    """Returns (cls, matched_entry) where matched_entry is the ground-truth or
    history entry the value corresponds to (None when nothing matches)."""
    kind = fact.get("kind", "money")
    tol = fact.get("tolerance_pct", 1.5)
    stated_as_of = (claim.get("stated_as_of") or "")
    as_current = bool(claim.get("presented_as_current", True))

    if kind == "text":
        qlow = (claim.get("quote", "") + " " + claim.get("stated_value", "")).lower()
        for tokens in fact.get("accept_tokens", []):
            if all(t.lower() in qlow for t in tokens):
                return "current", fact["truth"]
        for h in fact.get("history", []):
            if all(t.lower() in qlow for t in h.get("tokens", [])):
                if h.get("as_of") and h["as_of"][:4] in stated_as_of:
                    return "stale_dated", h
                return ("stale_as_current" if as_current else "stale_dated"), h
        # no accusation without a match: an unmatched text claim may simply
        # be incomplete (e.g. "led by two co-CEOs" without names)
        return "unverifiable", None

    if kind == "year":
        ym = re.search(r"\b(1[89]\d\d|20\d\d)\b",
                       str(claim.get("stated_value") or ""))
        if ym:
            yv = float(ym.group(1))
            truth_v = float(fact["truth"]["value"])
            return ("current" if yv == truth_v else "incorrect"), (
                fact["truth"] if yv == truth_v else None)
        return "unverifiable", None

    sv0 = str(claim.get("stated_value") or "")
    parts0 = re.split(r"\s*(?:-|–|—|\bto\b)\s*", sv0)
    is_range = len(parts0) == 2 and all(re.search(r"\d", p) for p in parts0)
    val = None if is_range else parse_number(sv0)
    if val is None:
        # a range: when both endpoints classify identically, use that verdict
        sv = str(claim.get("stated_value") or "")
        m = re.split(r"\s*(?:-|–|—|\bto\b)\s*", sv)
        if len(m) == 2:
            unit_re = r"(trillion|billion|million|thousand|tn|bn|mn|[tbmk])\b"
            u = re.search(unit_re, m[1], re.I)
            if u and not re.search(unit_re, m[0], re.I):
                m[0] = m[0] + " " + u.group(1)      # "$45.18-$45.2 billion"
            ends = [parse_number(x) for x in m]
            if all(x is not None for x in ends):
                lo, hi = min(ends), max(ends)
                # containment: a range that brackets a known figure matches it
                cand = ([("current", fact["truth"])] +
                        [("current_basis", ab) for ab in fact.get("alt_bases", [])] +
                        [(None, h) for h in fact.get("history", [])])
                for cls0, entry in cand:
                    ev = float(entry.get("value", 0) or 0)
                    if ev and lo * 0.99 <= ev <= hi * 1.01:
                        if cls0:
                            return cls0, entry
                        vintage = (entry.get("as_of") or "")[:4]
                        if vintage and vintage in stated_as_of:
                            return "stale_dated", entry
                        return ("stale_as_current" if as_current
                                else "stale_dated"), entry
                results = [classify_claim(fact, {**claim, "stated_value": str(x)})
                           for x in ends]
                if results[0][0] == results[1][0]:
                    return results[0]
        return "unverifiable", None

    truth = fact["truth"]
    discontinued = fact.get("discontinued", False)
    if not discontinued and within(val, float(truth["value"]), tol):
        return "current", truth
    for ab in fact.get("alt_bases", []):
        if within(val, float(ab["value"]), ab.get("tolerance_pct", tol)):
            return "current_basis", ab
    entries = ([truth] if discontinued else []) + fact.get("history", [])
    for h in entries:
        if within(val, float(h["value"]), tol):
            vintage_year = (h.get("as_of") or "")[:4]
            if vintage_year and vintage_year in stated_as_of:
                return "stale_dated", h
            return ("stale_as_current" if as_current else "stale_dated"), h
    if discontinued:
        return "unverifiable", None        # metric no longer reported
    return "incorrect", None


def cluster_variants(cfg, claims):
    facts = {f["id"]: f for f in cfg["facts"]}
    variants = {}
    for c in claims:
        fact = facts.get(c.get("fact_id"))
        if not fact:
            continue
        sv = str(c.get("stated_value", ""))
        blob = (str(c.get("quote", "")) + " " + sv).lower()
        if any(re.search(p, sv, re.I)
               for p in fact.get("reject_value_patterns", [])):
            continue                     # off-basis claim (regional subset etc.)
        if any(re.search(p, blob, re.I)
               for p in fact.get("reject_quote_patterns", [])):
            continue                     # off-topic claim (ads revenue etc.)
        cls, matched = classify_claim(fact, c)
        # a numeric claim without the fact's basis qualifier cannot be called
        # incorrect against a basis-specific ground truth; cap at unverifiable
        bt = fact.get("basis_token")
        if bt and cls == "incorrect" and bt.lower() not in blob:
            cls, matched = "unverifiable", None
        c["cls"], c["matched"] = cls, matched
        if fact.get("kind") == "text":
            vkey = (fact["id"], cls, (matched or {}).get("display", "other"))
        else:
            vkey = (fact["id"], cls, sig_key(parse_number(c.get("stated_value"))))
        v = variants.setdefault(vkey, {
            "fact_id": fact["id"], "cls": cls, "claims": [],
            "matched": matched,
            "value": parse_number(c.get("stated_value"))
            if fact.get("kind") != "text" else None})
        v["claims"].append(c)
    out = []
    for v in variants.values():
        surfaces = {}
        for c in v["claims"]:
            s = str(c.get("stated_value", "")).strip()
            surfaces[s] = surfaces.get(s, 0) + 1
        v["surface"] = max(surfaces, key=surfaces.get)
        plats = {}
        for c in v["claims"]:
            plats[c["platform"]] = plats.get(c["platform"], 0) + 1
        v["platforms"] = plats
        v["n"] = len(v["claims"])
        v["severity"] = SEVERITY[v["cls"]]
        v["score"] = v["severity"] * v["n"]
        out.append(v)
    return out


# ---------------------------------------------------------------- trace

def fetch_page(url, cache, evidence_dir):
    if url in cache:
        return cache[url]
    import requests
    rec = {"url": url, "final_url": url, "status": None, "text": "",
           "fetched_at": now_utc(), "ok": False, "reason": ""}
    try:
        r = requests.get(url, headers={"User-Agent": FETCH_UA}, timeout=20,
                         allow_redirects=True)
        rec["status"] = r.status_code
        rec["final_url"] = r.url
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200:
            rec["reason"] = f"HTTP {r.status_code}"
        elif "pdf" in ctype:
            rec["reason"] = "PDF document, verify by hand"
        elif "html" not in ctype and "text" not in ctype:
            rec["reason"] = f"unsupported content-type {ctype.split(';')[0]}"
        else:
            rec["text"] = strip_html(r.text[:600000])
            rec["ok"] = True
    except Exception as e:
        rec["reason"] = f"fetch failed: {type(e).__name__}"
    if rec["ok"]:
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        snap = os.path.join(evidence_dir, f"{h}.txt")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(f"url: {url}\nfinal_url: {rec['final_url']}\n"
                    f"fetched_at: {rec['fetched_at']}\nstatus: {rec['status']}\n"
                    f"---\n{rec['text'][:300000]}")
        rec["snapshot"] = f"evidence/{h}.txt"
    cache[url] = rec
    return rec


def value_in_page(fact, variant, page_text):
    """Find the variant's value in fetched page text; return surrounding quote."""
    tl = page_text
    if fact.get("kind") == "text":
        h = variant.get("matched") or {}
        toks = h.get("tokens") or [variant["surface"]]
        pat = re.escape(toks[0])
    else:
        val = variant.get("value")
        if val is None:
            return None
        pats = number_page_patterns(val)
        pat = None
        for p in pats:
            if re.search(p, tl, re.I):
                pat = p
                break
        if not pat:
            return None
        if fact.get("kind") == "year":
            m = re.search(pat, tl, re.I)
            ctx = tl[max(0, m.start() - 100):m.end() + 100].lower()
            if not re.search(r"found|start|establish|incorporat|launch|began",
                             ctx):
                return None
    m = re.search(pat, tl, re.I)
    if not m:
        return None
    lo, hi = max(0, m.start() - 200), min(len(tl), m.end() + 200)
    return "..." + tl[lo:hi].strip() + "..."


def trace_variants(cfg, variants, evidence_dir):
    cache, manual = {}, []
    flagged = [v for v in variants if v["cls"] in ("stale_as_current", "incorrect")]
    facts = {f["id"]: f for f in cfg["facts"]}
    excluded = set(cfg.get("exclude_sources", []))
    for v in flagged:
        urls, seen = [], set()
        for c in v["claims"]:
            for u in c.get("citations", []):
                if u not in seen and root_host(host_of(u)) not in excluded:
                    seen.add(u)
                    urls.append(u)
        v["origins"], v["origin_urls_checked"] = [], 0
        for u in urls[:8]:
            rec = fetch_page(u, cache, evidence_dir)
            v["origin_urls_checked"] += 1
            if not rec["ok"]:
                manual.append({"url": u, "reason": rec["reason"],
                               "fact_id": v["fact_id"], "surface": v["surface"]})
                continue
            quote = value_in_page(facts[v["fact_id"]], v, rec["text"])
            if quote:
                v["origins"].append({
                    "url": u, "final_url": rec["final_url"],
                    "host": host_of(rec["final_url"]),
                    "quote": quote[:520], "fetched_at": rec["fetched_at"],
                    "snapshot": rec.get("snapshot", "")})
        print(f"[trace] {v['fact_id']} '{v['surface']}': "
              f"{len(v['origins'])} confirmed of {v['origin_urls_checked']} "
              f"checked", flush=True)
    # dedupe manual queue by url
    seen, dedup = set(), []
    for m in manual:
        if m["url"] not in seen:
            seen.add(m["url"])
            dedup.append(m)
    return flagged, dedup


# ---------------------------------------------------------------- routing

def route_for(cfg, host):
    rh = root_host(host)
    if any(rh == root_host(d) or host.endswith("." + d)
           for d in cfg.get("owned_domains", [])):
        return ("owned", "Owned page. Update directly; the durable fix is a "
                "dated canonical fact page.")
    if "wikipedia.org" in host:
        return ("wikipedia", "Wikipedia. A COI-compliant talk page request "
                "with the primary source citation attached. Do not edit the "
                "article directly.")
    if any(k in host for k in AGGREGATOR_HINTS):
        return ("aggregator", "Data aggregator. Correction request via the "
                "site's contact page, primary source attached.")
    if any(host.endswith(p) or p in host for p in PUBLISHER_HOSTS):
        return ("publisher", "Publisher. Corrections desk request citing the "
                "primary source.")
    return ("site", "Site contact or corrections channel, primary source "
            "attached.")


def draft_note(cfg, fact, variant, origin, route_kind):
    brand = cfg["brand"]
    truth = fact["truth"]
    stated = variant["surface"]
    matched = variant.get("matched") or {}
    vintage = matched.get("label") or matched.get("as_of") or ""
    src = fact.get("primary_source", {})
    accurate_line = (f"That figure was accurate as of {vintage}. " if vintage
                     else "")
    if fact.get("discontinued"):
        truth_line = (f"{brand} last reported {truth['display']} as of "
                      f"{pretty_date(truth['as_of'])} and no longer reports "
                      f"this metric on a quarterly basis.")
        ask = ("Could the page be updated to date the figure to "
               f"{pretty_date(truth['as_of'])}, or note that the company no "
               "longer reports it?")
    else:
        truth_line = (f"The current figure is {truth['display']} as of "
                      f"{pretty_date(truth['as_of'])}, per "
                      f"{src.get('label', 'the primary source')}.")
        ask = "Could the page be updated to the current figure?"
    if route_kind == "wikipedia":
        return (f"Talk page request (COI disclosed):\n\n"
                f"I am writing on behalf of {brand} and will not edit the "
                f"article directly. The article currently gives {brand}'s "
                f"{fact['label'].lower()} as {stated}. {accurate_line}"
                f"{truth_line} Primary source: {src.get('url', '')}. "
                f"Requesting an editor review and update the figure with this "
                f"citation.")
    return (f"Subject: Correction request regarding {brand} "
            f"{fact['label'].lower()}\n\n"
            f"Hello, I am reaching out on behalf of {brand}. Your page at "
            f"{origin['final_url']} currently gives {brand}'s "
            f"{fact['label'].lower()} as {stated}. {accurate_line}{truth_line} "
            f"{ask} Primary source for verification: {src.get('url', '')}. "
            f"Happy to provide any further documentation.")


# ---------------------------------------------------------------- analyze

def analyze(cfg, run_dir, reuse_claims=False):
    raw = json.load(open(os.path.join(run_dir, "raw.json")))
    evidence_dir = os.path.join(run_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    prior = os.path.join(run_dir, "analysis.json")
    if reuse_claims and os.path.exists(prior):
        pa = json.load(open(prior))
        claims = pa["claims"]
        n_usable, n_resp_with = pa["n_usable"], pa["n_responses_with_claims"]
        print(f"[analyze] reusing {len(claims)} extracted claims "
              f"(config reclassification only)", flush=True)
    else:
        claims, n_usable, n_resp_with = extract_claims(cfg, raw, run_dir)
    variants = cluster_variants(cfg, claims)
    flagged, manual = trace_variants(cfg, variants, evidence_dir)

    facts = {f["id"]: f for f in cfg["facts"]}
    queue = []
    for v in sorted(flagged, key=lambda x: -x["score"]):
        fact = facts[v["fact_id"]]
        item = {"fact_id": v["fact_id"], "surface": v["surface"],
                "cls": v["cls"], "n": v["n"], "score": v["score"],
                "platforms": v["platforms"],
                "matched": v.get("matched"),
                "origins": []}
        for o in v["origins"]:
            kind, why = route_for(cfg, o["host"])
            item["origins"].append({**o, "route_kind": kind, "route": why,
                                    "note": draft_note(cfg, fact, v, o, kind)})
        queue.append(item)

    n_err = sum(1 for r in raw["rows"] if r["full_response"].startswith("ERROR"))
    analysis = {
        "brand": cfg["brand"], "slug": cfg["slug"],
        "analyzed_at": now_utc(), "collected_at": raw.get("collected_at"),
        "n_rows": len(raw["rows"]), "n_errors": n_err, "n_usable": n_usable,
        "n_responses_with_claims": n_resp_with,
        "n_claims": len(claims), "extract_model": EXTRACT_MODEL,
        "claims": claims,
        "variants": [{k: v for k, v in vv.items() if k != "claims"} |
                     {"n": vv["n"]} for vv in cluster_variants(cfg, claims)],
        "queue": queue, "manual_queue": manual,
    }
    # variants list above re-clusters without claim bodies for compactness;
    # keep the full per-claim record in "claims" for re-derivation.
    path = os.path.join(run_dir, "analysis.json")
    json.dump(analysis, open(path, "w"), indent=1)
    print(f"[analyze] DONE claims={len(claims)} variants="
          f"{len(analysis['variants'])} queue={len(queue)} "
          f"manual={len(manual)} -> {path}", flush=True)
    return analysis


# ---------------------------------------------------------------- report

CSS = """
:root { --bg:#07080b; --panel:#0e1016; --panel2:#12141c; --line:#1e212b;
--ink:#e8e9ee; --dim:#9a9eae; --faint:#6a6e7e; --gold:#cbab6d;
--coral:#f0876a; --cyan:#74d0ff; --red:#ff6b6b; --green:#7ddba3; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--ink); font-family:'Inter',-apple-system,sans-serif;
font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased; }
.wrap { max-width:1060px; margin:0 auto; padding:48px 28px 90px; }
h1,h2,h3 { font-family:'Jost',sans-serif; font-weight:600; }
.eyebrow { font-size:11px; letter-spacing:2.2px; text-transform:uppercase;
color:var(--gold); font-weight:700; margin-bottom:10px; }
h1 { font-size:34px; line-height:1.2; margin-bottom:8px; }
.sub { color:var(--dim); font-size:15px; max-width:760px; }
.section { margin-top:54px; }
h2 { font-size:22px; margin-bottom:6px; }
.section-sub { color:var(--dim); font-size:13.5px; margin-bottom:20px; max-width:760px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px; margin-top:26px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px;
padding:16px 16px 13px; }
.tile .n { font-family:'Jost'; font-size:30px; font-weight:600; }
.tile .l { font-size:11.5px; color:var(--dim); margin-top:2px; line-height:1.4; }
.exec { background:linear-gradient(160deg,var(--panel),var(--panel2));
border:1px solid var(--line); border-left:3px solid var(--gold);
border-radius:12px; padding:22px 24px; font-size:15.5px; max-width:860px; }
.exec p + p { margin-top:12px; }
.fact { background:var(--panel); border:1px solid var(--line); border-radius:14px;
padding:22px 24px; margin-bottom:18px; }
.fact h3 { font-size:17px; }
.truth { font-size:13px; color:var(--dim); margin:6px 0 14px; }
.truth b { color:var(--ink); }
.truth a { color:var(--cyan); text-decoration:none; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.8px;
color:var(--faint); padding:7px 10px; border-bottom:1px solid var(--line); }
td { padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:0; }
.chip { display:inline-block; font-size:10.5px; font-weight:700; letter-spacing:.4px;
padding:2px 9px; border-radius:20px; white-space:nowrap; }
.c-current { color:#07222e; background:var(--cyan); }
.c-current_basis { color:#0a2417; background:var(--green); }
.c-stale_dated { color:#241b06; background:var(--gold); }
.c-stale_as_current { color:#2a1008; background:var(--coral); }
.c-incorrect { color:#fff; background:var(--red); }
.c-unverifiable { color:#c9ccd8; background:#262a36; }
.plats { color:var(--dim); font-size:12.5px; }
.q { color:var(--dim); font-style:italic; font-size:12.5px; }
.item { background:var(--panel); border:1px solid var(--line); border-radius:14px;
padding:22px 24px; margin-bottom:18px; }
.item .rank { font-family:'Jost'; color:var(--gold); font-size:13px;
letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px; }
.item h3 { font-size:16.5px; margin-bottom:4px; }
.fixline { font-size:14px; margin:8px 0 4px; }
.fixline .from { color:var(--coral); font-weight:600; }
.fixline .to { color:var(--green); font-weight:600; }
.origin { background:var(--panel2); border:1px solid var(--line); border-radius:10px;
padding:14px 16px; margin-top:12px; font-size:13.5px; }
.origin a { color:var(--cyan); text-decoration:none; word-break:break-all; }
.origin .meta { color:var(--faint); font-size:11.5px; margin-top:6px; }
.origin blockquote { color:var(--dim); border-left:2px solid var(--line);
padding-left:12px; margin-top:8px; font-size:12.5px; }
details { margin-top:10px; }
summary { cursor:pointer; color:var(--gold); font-size:12.5px; font-weight:600; }
pre.note { white-space:pre-wrap; background:#0a0c11; border:1px solid var(--line);
border-radius:8px; padding:14px; font-size:12.5px; color:var(--dim);
font-family:'Inter'; margin-top:8px; }
.mono { font-family:ui-monospace,monospace; font-size:12px; }
.method { color:var(--dim); font-size:13.5px; max-width:820px; }
.method p + p { margin-top:10px; }
.method b { color:var(--ink); }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }
.footer { margin-top:70px; color:var(--faint); font-size:12px;
border-top:1px solid var(--line); padding-top:18px; }
@media print { body { background:#fff; color:#111; } }
"""


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def pretty_date(s):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(s or ""))
    if not m:
        return s
    return f"{MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}"


def fmt_plats(plats):
    return ", ".join(f"{k} x{v}" if v > 1 else k
                     for k, v in sorted(plats.items(), key=lambda x: -x[1]))


def build_report(cfg, run_dir):
    a = json.load(open(os.path.join(run_dir, "analysis.json")))
    facts = {f["id"]: f for f in cfg["facts"]}
    variants = a["variants"]
    claims = a["claims"]
    e = html.escape

    by_fact = {}
    for v in variants:
        by_fact.setdefault(v["fact_id"], []).append(v)

    n_flagged = sum(1 for v in variants if v["cls"] in
                    ("stale_as_current", "incorrect"))
    n_traced = sum(1 for q in a["queue"] if q["origins"])
    n_assistants = len({c["platform"] for c in claims}) or 5

    # ---- exec read (code-templated so numbers always match the data) ----
    MATCHED = ("current", "current_basis", "stale_dated", "stale_as_current")
    spread_fact, spread_vals = None, []
    for fid, vs in by_fact.items():
        vals = [v for v in vs
                if v.get("value") is not None and v["cls"] in MATCHED]
        numeric = sorted({v["value"] for v in vals})
        if len(numeric) > len(spread_vals):
            spread_fact, spread_vals = fid, numeric
    exec_ps = []
    exec_ps.append(
        f"Across {a['n_usable']} answers from {n_assistants} AI assistants, "
        f"we checked {len(cfg['facts'])} facts about {cfg['brand']} against "
        f"primary sources. {a['n_claims']} individual claims were extracted "
        f"and classified; {n_flagged} claim variants qualify for correction, "
        f"and {n_traced} of those are traced to a fetched source page that "
        f"carries the figure today.")
    if spread_fact and len(spread_vals) > 1:
        f = facts[spread_fact]
        vs = [v for v in by_fact[spread_fact]
              if v.get("value") is not None and v["cls"] in MATCHED]
        def endpoint(v):
            mm = v.get("matched") or {}
            disp = mm.get("display") or v["surface"]
            lbl = (mm.get("label") or "").split(",")[0]
            return f"{disp} ({lbl})" if lbl else disp
        lo = endpoint(min((v for v in vs if v["value"] == spread_vals[0]),
                          key=lambda v: v["surface"]))
        hi = endpoint(min((v for v in vs if v["value"] == spread_vals[-1]),
                          key=lambda v: v["surface"]))
        exec_ps.append(
            f"The widest spread: {f['label'].lower()} appeared as "
            f"{len(spread_vals)} distinct figures, from {lo} to {hi}. "
            f"Every one of them matches something the company has reported "
            f"or forecast at some date. The inaccuracy is vintage, not "
            f"invention: assistants quote sources that were right once and "
            f"were never updated.")
    exec_ps.append(
        "Each queue item below carries the stated figure, the current "
        "primary-source figure, the specific page feeding the error with a "
        "quote fetched at detection time, and a drafted correction note. "
        "This sample is a snapshot of five assistants on one day, not a "
        "census; treat directionally.")

    # ---- fact cards ----
    fact_cards = []
    for fcfg in cfg["facts"]:
        vs = sorted(by_fact.get(fcfg["id"], []),
                    key=lambda v: (-SEVERITY[v["cls"]], -v["n"]))
        truth = fcfg["truth"]
        src = fcfg.get("primary_source", {})
        truth_bits = f"<b>{e(truth['display'])}</b>"
        if truth.get("as_of"):
            truth_bits += f" as of {e(pretty_date(truth['as_of']))}"
        if src.get("url"):
            truth_bits += (f" &nbsp;·&nbsp; <a href='{e(src['url'])}' "
                           f"target='_blank' rel='noopener'>{e(src.get('label', 'primary source'))}</a>")
        if fcfg.get("discontinued"):
            truth_bits += (" &nbsp;·&nbsp; the company no longer reports this "
                           "metric quarterly")
        rows_html = ""
        for v in vs:
            matched = v.get("matched") or {}
            vintage = matched.get("label") or ""
            sample_q = ""
            for c in claims:
                if (c.get("fact_id") == v["fact_id"] and c.get("cls") == v["cls"]
                        and (fcfg.get("kind") == "text"
                             or sig_key(parse_number(c.get("stated_value")))
                             == sig_key(v.get("value") or 0))):
                    sample_q = c.get("quote", "")[:180]
                    break
            rows_html += (
                f"<tr><td><b>{e(v['surface'])}</b></td>"
                f"<td><span class='chip c-{v['cls']}'>"
                f"{e(CLASS_LABEL[v['cls']])}</span>"
                f"{(' <span class=plats>(' + e(vintage) + ')</span>') if vintage and v['cls'] != 'current' else ''}</td>"
                f"<td class='plats'>{e(fmt_plats(v['platforms']))}</td>"
                f"<td class='q'>{e(sample_q)}</td></tr>")
        if not vs:
            rows_html = ("<tr><td colspan='4' class='plats'>No claims about "
                         "this fact appeared in the sample.</td></tr>")
        fact_cards.append(
            f"<div class='fact'><h3>{e(fcfg['label'])}</h3>"
            f"<div class='truth'>Primary source: {truth_bits}</div>"
            f"<table><tr><th>Stated as</th><th>Classification</th>"
            f"<th>Assistants</th><th>Sample quote</th></tr>{rows_html}</table>"
            f"</div>")

    # ---- correction queue ----
    q_html = []
    for i, item in enumerate(a["queue"], 1):
        fcfg = facts[item["fact_id"]]
        truth = fcfg["truth"]
        matched = item.get("matched") or {}
        vintage = matched.get("label") or matched.get("as_of") or ""
        origins_html = ""
        for o in item["origins"]:
            origins_html += (
                f"<div class='origin'><a href='{e(o['final_url'])}' "
                f"target='_blank' rel='noopener'>{e(o['host'])}</a> "
                f"<span class='plats'>· {e(o['route'])}</span>"
                f"<blockquote>{e(o['quote'])}</blockquote>"
                f"<div class='meta'>Fetched {e(o['fetched_at'])} · snapshot "
                f"<span class='mono'>{e(o.get('snapshot', ''))}</span></div>"
                f"<details><summary>Drafted correction note</summary>"
                f"<pre class='note'>{e(o['note'])}</pre></details></div>")
        if not origins_html:
            origins_html = ("<div class='origin plats'>No fetched page in this "
                            "sample's citations carries the figure verbatim. "
                            "The figure may originate in training data or a "
                            "page we could not fetch; see the manual queue."
                            "</div>")
        vintage_line = (f" That figure was accurate as of {e(vintage)}."
                        if vintage else "")
        q_html.append(
            f"<div class='item'><div class='rank'>Queue item {i} · "
            f"{item['n']} answer{'s' if item['n'] != 1 else ''} · severity "
            f"{item['score']}</div>"
            f"<h3>{e(fcfg['label'])}</h3>"
            f"<div class='fixline'><span class='from'>{e(item['surface'])}</span>"
            f" &rarr; <span class='to'>{e(truth['display'])}"
            f"{' (as of ' + e(pretty_date(truth['as_of'])) + ')' if truth.get('as_of') else ''}</span></div>"
            f"<div class='plats'>Stated by {e(fmt_plats(item['platforms']))}."
            f"{vintage_line}</div>{origins_html}</div>")
    if not q_html:
        q_html = ["<div class='item plats'>No claim variants in this sample "
                  "qualified for the correction queue.</div>"]

    manual_html = ""
    for m in a["manual_queue"]:
        manual_html += (f"<tr><td><a href='{e(m['url'])}' target='_blank' "
                        f"rel='noopener' style='color:var(--cyan);text-decoration:none'>"
                        f"{e(host_of(m['url']))}</a></td>"
                        f"<td class='plats'>{e(facts[m['fact_id']]['label'])} · "
                        f"stated as {e(m['surface'])}</td>"
                        f"<td class='plats'>{e(m['reason'])}</td></tr>")

    legend = "".join(f"<span class='chip c-{k}'>{e(v)}</span>"
                     for k, v in CLASS_LABEL.items())

    doc = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<meta name='robots' content='noindex'>
<title>{e(cfg['brand'])}: fact consistency audit</title>
<link rel='preconnect' href='https://fonts.googleapis.com'>
<link href='https://fonts.googleapis.com/css2?family=Jost:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>
<style>{CSS}</style></head><body><div class='wrap'>
<div class='eyebrow'>Innate C3 · Correction Seeker · sample audit</div>
<h1>{e(cfg['brand'])}: what AI gets right, and what it gets stale</h1>
<div class='sub'>A fact consistency audit. {len(cfg['prompts'])} fact-bearing
questions were put to five AI assistants in {e(cfg.get('date', ''))}. Every
factual claim about {e(cfg['brand'])} was extracted, checked against primary
sources, and, where a figure is stale or wrong, traced to the specific page
feeding it.</div>
<div class='tiles'>
<div class='tile'><div class='n'>{a['n_usable']}</div><div class='l'>AI answers analyzed</div></div>
<div class='tile'><div class='n'>{a['n_claims']}</div><div class='l'>factual claims extracted</div></div>
<div class='tile'><div class='n'>{len(cfg['facts'])}</div><div class='l'>facts checked against primary sources</div></div>
<div class='tile'><div class='n'>{n_flagged}</div><div class='l'>claim variants flagged for correction</div></div>
<div class='tile'><div class='n'>{n_traced}</div><div class='l'>traced to a live source page</div></div>
</div>
<div class='section'><h2>Executive read</h2>
<div class='exec'>{''.join(f'<p>{e(p)}</p>' for p in exec_ps)}</div></div>
<div class='section'><h2>Fact by fact</h2>
<div class='section-sub'>Each fact's ground truth comes from a primary source.
Every value stated by an assistant is classified against it. The legend:</div>
<div class='legend'>{legend}</div><div style='height:14px'></div>
{''.join(fact_cards)}</div>
<div class='section'><h2>Correction queue</h2>
<div class='section-sub'>Ranked by how often the figure appears times how far
it is from current. Each item shows the source page that carries the figure
today, with a quote captured at detection time, the route for requesting a
correction, and a drafted note.</div>
{''.join(q_html)}</div>
{f"<div class='section'><h2>Manual verification queue</h2><div class='section-sub'>These cited pages could not be fetched automatically (bot walls, PDFs, non-HTML content). They should be checked by hand before any correction outreach.</div><table><tr><th>Source</th><th>Figure in question</th><th>Why manual</th></tr>{manual_html}</table></div>" if manual_html else ''}
<div class='section'><h2>Method and boundaries</h2><div class='method'>
<p><b>Sample.</b> {len(cfg['prompts'])} fact-bearing prompts x 5 assistants
(ChatGPT, Claude, Gemini, Grok, Perplexity), all with web grounding enabled,
collected {e(a.get('collected_at') or '')}. {a['n_errors']} calls errored and
are excluded. One day, one run: a snapshot, not a census.</p>
<p><b>Ground truth.</b> Every fact is anchored to a primary source (SEC
filings and company disclosures), linked in each card. Facts without a
primary-source ground truth are out of scope by design; contested labels and
interpretation questions never enter this pipeline.</p>
<p><b>Taxonomy.</b> A figure that matches an earlier reported value is stale,
not invented. "Accurate as of an earlier date" is the honest description of
most drift, and the report says so. Only figures matching nothing the company
ever reported are classed as incorrect.</p>
<p><b>Evidence.</b> Every traced origin links a fetched snapshot captured at
detection time (the pages can change any day). Every claim in this report is
re-derivable from stored raw responses and snapshots.</p>
<p><b>Caveat.</b> Which sources an assistant cites is a property of that
model's retrieval on that day, not a verdict on the source. Claim extraction
uses a language model and is quote-gated: a claim only counts when its quote
appears verbatim in the raw answer.</p></div></div>
<div class='footer'>Prepared by Innate C3 · {e(cfg.get('date', ''))} ·
Correction Seeker sample · {e(cfg['brand'])} is used as a public demonstration
subject; figures from public filings.</div>
</div></body></html>"""
    path = os.path.join(run_dir, "report.html")
    open(path, "w", encoding="utf-8").write(doc)
    print(f"[report] DONE -> {path}", flush=True)
    return path


# ---------------------------------------------------------------- scan
# Ground-truth-free spread scan over existing audit datasets (production
# /signal/<slug>.json exports). Finds corporate-fact claims, clusters the
# variants, and writes a per-company digest for operator triage. No verdicts:
# verification against primary sources stays a human step.

SCAN_MODEL = os.environ.get("CS_SCAN_MODEL", "claude-haiku-4-5")

SCAN_FACT_TYPES = ("revenue, aum, employees, founded, ceo, hq, users "
                   "(subscribers/customers/members), stores, valuation, "
                   "funding, market_cap")

SCAN_TEMPLATE = """COMPANY: {brand}

Below are {n} AI assistant answers, each tagged [RESPONSE i]. Extract every
claim any answer makes about {brand}'s corporate facts. Fact types (use these
exact labels): {fact_types}.

{responses_block}

Return JSON only:
{{"claims": [{{"response_idx": 0, "fact_type": "revenue", "quote": "...",
"stated_value": "...", "stated_as_of": "..." or null,
"presented_as_current": true/false}}]}}

Rules:
- Only claims about {brand} itself, never about other companies.
- "quote": one sentence copied VERBATIM from that response.
- "stated_value": the exact value phrase as written.
- Skip vague phrases with no figure or name ("a large company").
- If nothing qualifies, return {{"claims": []}}.
"""


def _scan_extract_batch(client, brand, batch, base_idx):
    responses_block = "\n\n".join(
        f"[RESPONSE {base_idx + i}]\n{(r.get('response') or '')[:6000]}"
        for i, r in enumerate(batch))
    prompt = SCAN_TEMPLATE.format(brand=brand, n=len(batch),
                                  fact_types=SCAN_FACT_TYPES,
                                  responses_block=responses_block)
    for attempt in range(3):
        try:
            p = prompt if attempt == 0 else (
                prompt + "\n\nIMPORTANT: Output STRICT valid JSON only.")
            msg = client.messages.create(
                model=SCAN_MODEL, max_tokens=4000, system=EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": p}])
            text = "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
            start, end = text.find("{"), text.rfind("}")
            try:
                return json.loads(text[start:end + 1],
                                  strict=False).get("claims", [])
            except Exception:
                out = []
                for frag in re.findall(r"\{[^{}]*\}", text[start + 1:]):
                    try:
                        o = json.loads(frag, strict=False)
                        if o.get("fact_type") and o.get("quote"):
                            out.append(o)
                    except Exception:
                        pass
                if out:
                    return out
                raise
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2)


def scan_dataset(path, client):
    d = json.load(open(path))
    brand = d.get("brand") or d.get("brand_name") or "?"
    rows = [r for r in d.get("all_responses", [])
            if (r.get("response") or "").strip() and not r.get("error")]
    claims = []
    BATCH = 6
    for b0 in range(0, len(rows), BATCH):
        batch = rows[b0:b0 + BATCH]
        for c in _scan_extract_batch(client, brand, batch, b0):
            i = c.get("response_idx")
            if not isinstance(i, int) or not (0 <= i < len(rows)):
                continue
            row = rows[i]
            norm_resp = re.sub(r"[\s*_`#]+", " ",
                               (row.get("response") or "").lower())
            q = re.sub(r"[\s*_`#]+", " ", str(c.get("quote", "")).lower()).strip()
            if not q or q[:160] not in norm_resp:
                continue
            c["platform"] = row.get("llm") or row.get("platform") or "?"
            c["prompt"] = row.get("prompt") or row.get("query") or ""
            c["citations"] = [x.get("url") for x in (row.get("citations") or [])
                              if isinstance(x, dict) and x.get("url")][:12]
            claims.append(c)
        time.sleep(0.15)
    # cluster per fact_type by value signature
    clusters = {}
    for c in claims:
        ft = str(c.get("fact_type", "other")).lower().strip()
        val = parse_number(c.get("stated_value"))
        key = (ft, sig_key(val) if val is not None
               else re.sub(r"\W+", " ", str(c.get("stated_value", ""))
                           .lower()).strip()[:40])
        cl = clusters.setdefault(key, {"fact_type": ft, "value": val,
                                       "surfaces": {}, "platforms": {},
                                       "quotes": [], "citations": {},
                                       "as_ofs": {}, "n": 0})
        cl["n"] += 1
        s = str(c.get("stated_value", "")).strip()
        cl["surfaces"][s] = cl["surfaces"].get(s, 0) + 1
        cl["platforms"][c["platform"]] = cl["platforms"].get(c["platform"], 0) + 1
        ao = c.get("stated_as_of")
        if ao:
            cl["as_ofs"][str(ao)] = cl["as_ofs"].get(str(ao), 0) + 1
        if len(cl["quotes"]) < 3:
            cl["quotes"].append(c.get("quote", "")[:220])
        for u in c.get("citations", []):
            cl["citations"][u] = cl["citations"].get(u, 0) + 1
    out = []
    for cl in clusters.values():
        cl["surface"] = max(cl["surfaces"], key=cl["surfaces"].get)
        cl["top_citations"] = sorted(cl["citations"],
                                     key=cl["citations"].get, reverse=True)[:6]
        del cl["citations"]
        out.append(cl)
    # facts with multiple distinct numeric values, then big consensus facts
    from collections import Counter
    per_fact = Counter(cl["fact_type"] for cl in out
                       if cl.get("value") is not None)
    for cl in out:
        spread = per_fact.get(cl["fact_type"], 0)
        cl["fact_variants"] = spread
        cl["score"] = (spread - 1) * 10 + cl["n"]
    out.sort(key=lambda c: (-c["fact_variants"], c["fact_type"], -c["n"]))
    return {"brand": brand, "n_rows": len(rows), "n_claims": len(claims),
            "clusters": out}


def scan(data_dir, out_dir, limit=None):
    import anthropic
    client = anthropic.Anthropic()
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if limit:
        files = files[:limit]
    print(f"[scan] {len(files)} datasets, model={SCAN_MODEL}", flush=True)
    for f in files:
        slug = f[:-5]
        dst = os.path.join(out_dir, f"{slug}.scan.json")
        if os.path.exists(dst):
            print(f"  [scan] {slug}: cached, skipping", flush=True)
            continue
        try:
            res = scan_dataset(os.path.join(data_dir, f), client)
            res["slug"] = slug
            json.dump(res, open(dst, "w"), indent=1)
            multi = sum(1 for c in res["clusters"] if c["fact_variants"] >= 2)
            print(f"  [scan] {slug}: {res['n_claims']} claims, "
                  f"{len(res['clusters'])} clusters, "
                  f"{multi} in multi-variant facts", flush=True)
        except Exception as e:
            print(f"  [scan] {slug} FAILED: {e}", flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "analyze", "report", "all",
                                      "scan"])
    ap.add_argument("--config")
    ap.add_argument("--data-dir", help="scan: directory of raw dataset JSONs")
    ap.add_argument("--out", help="scan: output directory for digests")
    ap.add_argument("--limit-datasets", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="collect only the first N prompts (smoke test)")
    ap.add_argument("--reuse-claims", action="store_true",
                    help="skip re-extraction; reclassify prior claims "
                    "against the current config")
    args = ap.parse_args()
    if args.stage == "scan":
        scan(args.data_dir, args.out or "scan_results",
             limit=args.limit_datasets)
        return
    if not args.config:
        ap.error("--config is required for this stage")
    cfg = json.load(open(os.path.join(HERE, args.config)
                         if not os.path.isabs(args.config) else args.config))
    run_dir = os.path.join(HERE, "runs", cfg["slug"])
    os.makedirs(run_dir, exist_ok=True)
    if args.stage in ("collect", "all"):
        collect(cfg, run_dir, limit=args.limit)
    if args.stage in ("analyze", "all"):
        analyze(cfg, run_dir, reuse_claims=args.reuse_claims)
    if args.stage in ("report", "all"):
        build_report(cfg, run_dir)


if __name__ == "__main__":
    main()
