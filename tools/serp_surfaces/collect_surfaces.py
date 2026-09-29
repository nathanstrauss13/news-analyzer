#!/usr/bin/env python3
"""Collect Google AI Overviews, Google AI Mode and Microsoft Copilot answers
for an audit's prompt set via SerpApi, and merge them into a COPY of the
audit payload as extra answer rows. Standalone: nothing here touches the live
site, the database or the original payload.

Usage:
    python3 collect_surfaces.py <payload.json> <out_dir>
        [--surfaces aio,aimode,copilot] [--max-searches N] [--gl us] [--hl en]
        [--limit N] [--dry-run]

Reads SERPAPI_API_KEY from the environment or the worktree .env.

Output (out_dir):
    surfaces_rows.json    the new answer rows, in the payload's all_responses shape
    merged_payload.json   the input payload + those rows (feed to export_raw_run.py)
    raw/                  every SerpApi response as received
    summary.txt           per-surface counts, printed as well

Conventions (so the rows mean the same thing as the five assistants' rows):
  * citations = the answer's own references list only. Google shopping-panel
    links and google.com links inside the answer text are product widgets,
    not sources, and are excluded.
  * An AI Overview that Google did not show for a query is a finding, not an
    error: the row is kept with an empty answer and error="not_shown".
  * grounding.supports maps each answer sentence to the references it cites
    (Copilot and AI Overviews tag them; AI Mode usually does not).
  * model_id records the collection path ("serpapi:google_ai_mode" etc.);
    SerpApi does not expose which model Google or Microsoft ran.
  * Search budget: an AI Overview costs 2 searches (the Google results page,
    then the overview itself); AI Mode and Copilot cost 1 each. The script
    checks the account balance first and refuses to overrun it.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "mcp_server"))
from conventions import forms_to_pattern  # noqa: E402

API = "https://serpapi.com/search.json"
SURFACE_NAMES = {"aio": "Google AI Overview", "aimode": "Google AI Mode", "copilot": "Copilot"}
COST = {"aio": 2, "aimode": 1, "copilot": 1}
_JUNK = re.compile(r"(Go to product viewer dialog for this item\.?)+", re.I)


def api_key():
    k = os.environ.get("SERPAPI_API_KEY")
    if not k:
        try:
            from dotenv import dotenv_values
            k = dotenv_values(os.path.join(ROOT, ".env")).get("SERPAPI_API_KEY")
        except Exception:
            k = None
    if not k:
        sys.exit("SERPAPI_API_KEY not set (environment or worktree .env)")
    return k


def call(key, params, raw_path, tries=3):
    p = dict(params, api_key=key)
    err = None
    for i in range(tries):
        try:
            r = requests.get(API, params=p, timeout=120)
            d = r.json()
            if r.status_code == 200 and not d.get("error"):
                json.dump(d, open(raw_path, "w"))
                return d, None
            err = d.get("error") or f"HTTP {r.status_code}"
            # "no results" style errors are answers, not failures: don't retry
            if "hasn't returned any results" in str(err) or r.status_code in (400, 401, 403):
                break
        except Exception as e:
            err = type(e).__name__
        time.sleep(3 * (i + 1))
    return None, err


def domain(u):
    try:
        h = urlparse(u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def flatten(blocks, depth=0):
    """text_blocks -> [(line, [reference_index,...])]. List items with a
    shopping_result carry the product name as their text."""
    out = []
    for b in blocks or []:
        t = b.get("type")
        snip = b.get("snippet") or ""
        if not snip and isinstance(b.get("shopping_result"), dict):
            snip = b["shopping_result"].get("title") or ""
        if not snip and b.get("title"):
            snip = b["title"]
        snip = _JUNK.sub("", snip).strip()
        refs = list(b.get("reference_indexes") or [])
        if snip:
            prefix = "#" * min(3, b.get("level") or 2) + " " if t == "heading" else ("  " * depth + "- " if depth else "")
            out.append((prefix + snip, refs))
        if b.get("list"):
            out += flatten(b["list"], depth + 1)
        if b.get("text_blocks"):
            out += flatten(b["text_blocks"], depth + 1)
        if t == "table" and b.get("table"):
            for row in b["table"]:
                out.append((" | ".join(str(c) for c in row), []))
    return out


def to_row(surface, prompt, body, err, engine):
    name = SURFACE_NAMES[surface]
    row = {"llm": name, "prompt": prompt, "response": "", "citations": [], "grounded": True,
           "error": err, "model_id": f"serpapi:{engine}", "surface": surface,
           "grounding": {"source": "serpapi", "queries": [], "retrieved": [], "supports": []}}
    if body is None:
        return row
    refs = body.get("references") or []
    by_idx = {}
    for i, r in enumerate(refs):
        u = r.get("link") or r.get("url")
        if not u or domain(u).endswith("google.com"):
            continue
        by_idx[r.get("index", i)] = u
        row["grounding"]["retrieved"].append({"url": u, "title": r.get("title") or "",
                                              "date": r.get("date") or "", "source": r.get("source") or ""})
    # Copilot writes each source's label inline before the full stop
    # ("...lasting power Bella Vita Organic ." / "...presence 1 ." / "+1 .").
    # Strip them so a label naming the brand's own site can't count as the
    # brand being named in the answer. Labels are the references' own
    # 'source' values, so the strip is exact, not a guess.
    labels = sorted({(r.get("source") or "").strip() for r in refs if (r.get("source") or "").strip()},
                    key=len, reverse=True)
    _alt = "|".join(re.escape(x) for x in labels)
    # numbers only before a SPACED stop ("presence 1 ."), so "costs 95." and
    # "## Top 10" survive; source names also at line end.
    lab_re = re.compile(r"\s+\+?\d+(?=\s+[.,;:])" + (
        r"|\s+(?:" + _alt + r")(?=\s+[.,;:]|\s*$)" if labels else ""))

    def _clean(l):
        prev = None
        while prev != l:
            prev = l
            l = lab_re.sub("", l)
        return re.sub(r"\s+([.,;:])", r"\1", l)
    lines = [(_clean(l), ix) for l, ix in flatten(body.get("text_blocks"))]
    text = "\n".join(l for l, _ in lines)
    if not text and body.get("reconstructed_markdown"):
        text = body["reconstructed_markdown"]
    row["response"] = text
    seen = set()
    for u in by_idx.values():
        if u not in seen:
            seen.add(u)
            row["citations"].append({"url": u, "domain": domain(u)})
    for l, ix in lines:
        us = [by_idx[i] for i in ix if i in by_idx]
        if us:
            row["grounding"]["supports"].append({"claim": re.sub(r"^[#\s-]+", "", l)[:600], "urls": us})
    return row


def collect_one(key, surface, prompt, n, raw_dir, gl, hl):
    tag = f"{surface}_{n:03d}"
    if surface == "aio":
        g, err = call(key, {"engine": "google", "q": prompt, "gl": gl, "hl": hl, "no_cache": "true"},
                      os.path.join(raw_dir, tag + "_google.json"))
        if g is None:
            return to_row(surface, prompt, None, err, "google_ai_overview"), 1
        ao = g.get("ai_overview")
        if not ao:
            return to_row(surface, prompt, None, "not_shown", "google_ai_overview"), 1
        if ao.get("text_blocks"):          # sometimes returned inline, no follow-up needed
            return to_row(surface, prompt, ao, None, "google_ai_overview"), 1
        if not ao.get("page_token"):
            return to_row(surface, prompt, None, "not_shown", "google_ai_overview"), 1
        a, err = call(key, {"engine": "google_ai_overview", "page_token": ao["page_token"]},
                      os.path.join(raw_dir, tag + "_overview.json"))
        return to_row(surface, prompt, (a or {}).get("ai_overview") if a else None, err, "google_ai_overview"), 2
    if surface == "aimode":
        d, err = call(key, {"engine": "google_ai_mode", "q": prompt, "gl": gl, "hl": hl},
                      os.path.join(raw_dir, tag + ".json"))
        return to_row(surface, prompt, d, err, "google_ai_mode"), 1
    d, err = call(key, {"engine": "bing_copilot", "q": prompt}, os.path.join(raw_dir, tag + ".json"))
    return to_row(surface, prompt, d, err, "bing_copilot"), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out_dir")
    ap.add_argument("--surfaces", default="aio,aimode,copilot")
    ap.add_argument("--max-searches", type=int, default=None)
    ap.add_argument("--gl", default="us"); ap.add_argument("--hl", default="en")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    p = json.load(open(a.payload))
    prompts = []
    ps = p.get("prompt_sets") or {}
    for k in ("unbranded", "branded"):
        prompts += [x for x in (ps.get(k) or []) if x not in prompts]
    for r in p.get("all_responses") or []:
        if r.get("prompt") and r["prompt"] not in prompts:
            prompts.append(r["prompt"])
    if a.limit:
        prompts = prompts[:a.limit]
    surfaces = [s.strip() for s in a.surfaces.split(",") if s.strip()]
    bad = [s for s in surfaces if s not in SURFACE_NAMES]
    if bad:
        sys.exit(f"unknown surface(s): {bad}; choose from {list(SURFACE_NAMES)}")

    need = len(prompts) * sum(COST[s] for s in surfaces)
    key = api_key()
    acct = requests.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=30).json()
    left = acct.get("total_searches_left")
    cap = min(x for x in (left, a.max_searches) if x is not None) if (left is not None or a.max_searches) else None
    print(f"{len(prompts)} prompts x {surfaces} -> up to {need} searches "
          f"(AI Overviews not shown cost 1, not 2) | account: {acct.get('plan_name')}, {left} left")
    if cap is not None and need > cap:
        sys.exit(f"refusing: worst case {need} searches exceeds the cap of {cap}. "
                 f"Use --limit, fewer --surfaces, or a larger plan.")
    if a.dry_run:
        return

    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)
    jobs = [(s, q, i) for s in surfaces for i, q in enumerate(prompts)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda j: collect_one(key, j[0], j[1], j[2], os.path.join(a.out_dir, "raw"), a.gl, a.hl), jobs))
    rows = [r for r, _ in res]
    used = sum(c for _, c in res)

    brand = p.get("brand") or ""
    pat = forms_to_pattern([brand] + [x for x in (p.get("brand_aliases") or []) if isinstance(x, str)])
    branded = set(ps.get("branded") or [])
    lines = [f"surfaces collected {time.strftime('%Y-%m-%d %H:%M')} | source payload {p.get('slug')} | "
             f"brand {brand} | searches used ~{used}"]
    for s in surfaces:
        rs = [r for r in rows if r["surface"] == s]
        shown = [r for r in rs if r["response"]]
        failed = [r for r in rs if r["error"] and r["error"] != "not_shown"]
        unb = [r for r in shown if r["prompt"] not in branded]
        unb_all = [r for r in rs if r["prompt"] not in branded]
        cits = sum(len(r["citations"]) for r in shown)
        named_unb = sum(1 for r in unb if pat.search(r["response"]))
        lines.append(f"  {SURFACE_NAMES[s]:<20} answered {len(shown)}/{len(rs)}"
                     f" | not shown {sum(1 for r in rs if r['error']=='not_shown')}"
                     f" | failed {len(failed)} | citations {cits}"
                     f" | {brand} named in {named_unb} of {len(unb_all)} unbranded prompts")
        doms = {}
        for r in shown:
            for c in r["citations"]:
                doms[c["domain"]] = doms.get(c["domain"], 0) + 1
        top = sorted(doms.items(), key=lambda kv: -kv[1])[:6]
        lines.append("      top cited: " + ", ".join(f"{d} {n}" for d, n in top))
        for r in failed[:3]:
            lines.append(f"      failed: {r['prompt'][:60]!r}: {r['error']}")
    summary = "\n".join(lines)
    print(summary)
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(summary + "\n")
    json.dump(rows, open(os.path.join(a.out_dir, "surfaces_rows.json"), "w"), indent=1)

    merged = dict(p)
    merged["all_responses"] = list(p.get("all_responses") or []) + rows
    merged["extra_surfaces"] = {"collected": time.strftime("%Y-%m-%d"), "via": "serpapi",
                                "surfaces": [SURFACE_NAMES[s] for s in surfaces],
                                "note": "Appended by collect_surfaces.py. Headline figures stored in "
                                        "this payload were computed on the original agents only."}
    json.dump(merged, open(os.path.join(a.out_dir, "merged_payload.json"), "w"))
    print(f"wrote {a.out_dir}/surfaces_rows.json, merged_payload.json, summary.txt")


if __name__ == "__main__":
    main()
