#!/usr/bin/env python3
"""Collect Google AI Overviews, Google AI Mode and Microsoft Copilot answers
for an audit's prompt set via SerpApi, and merge them into a COPY of the
audit payload as extra answer rows. Standalone: nothing here touches the live
site, the database or the original payload.

Usage:
    python3 collect_surfaces.py <payload.json> <out_dir>
        [--surfaces aio,aimode,copilot,mistral,deepseek,metaai] [--max-searches N] [--gl us] [--hl en]
        [--limit N] [--dry-run] [--no-resolve]
    python3 collect_surfaces.py <payload.json> <out_dir> --reparse        (no searches)
    python3 collect_surfaces.py <payload.json> <out_dir> --resolve-only   (no API calls)

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
  * Links go through the app's own resolver (2026-10-10), the same pass the
    five assistants get: redirects demasked, dead links dropped and listed in
    dropped_citations, bot walls keep the cited URL, and an identical final
    URL counts once per answer (merged_citation_repeats). --resolve-only
    applies it to an existing out_dir without collecting anything.
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
SURFACE_NAMES = {"aio": "Google AI Overview", "aimode": "Google AI Mode", "copilot": "Copilot",
                 "mistral": "Mistral", "deepseek": "DeepSeek", "metaai": "Meta AI"}
COST = {"aio": 2, "aimode": 1, "copilot": 1, "mistral": 0, "deepseek": 0, "metaai": 0}   # SerpApi searches; the others bill their own APIs
SERP_SURFACES = ("aio", "aimode", "copilot")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-medium-latest")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")   # served as deepseek-v4-flash (9/29)
# Meta AI runs on Muse Spark ("the model behind experiences serving hundreds of
# millions of people in Meta AI", Meta developer blog). Standard tier, never the
# cheaper -contributor tier, so the run matches what Meta AI users get.
META_MODEL = os.environ.get("META_MODEL", "muse-spark-1.3")
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


def _is_google(u):
    d = domain(u)
    return d == "google.com" or d.endswith(".google.com") or d.startswith("google.") \
        or d.endswith("gstatic.com") or d.endswith("googleusercontent.com")


def inline_links(blocks):
    """Every outside link Google or Microsoft wrote INTO the answer (product
    names and phrases linked to a retailer page, an Instagram post, a video),
    in order. The five assistants' in-text links count as citations, so these
    do too. Google-owned links (shopping panels, search redirects, images)
    are widgets, not sources, and are skipped."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("link", "url") and isinstance(v, str) and v.startswith("http") and not _is_google(v):
                    out.append(v)
                elif k not in ("thumbnail", "favicon", "image", "images"):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(blocks)
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
        if not u or _is_google(u):
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
    for u in list(by_idx.values()) + inline_links(body.get("text_blocks")):
        if u not in seen:
            seen.add(u)
            row["citations"].append({"url": u, "domain": domain(u)})
    row["inline_link_count"] = len({u for u in inline_links(body.get("text_blocks"))})
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


def _app_prompting():
    """The product's shared system prompt and date note, read from app.py
    source (not copied), so Mistral gets exactly what the five assistants get."""
    import ast
    from datetime import datetime
    src = open(os.path.join(ROOT, "app.py")).read()
    ns = {"datetime": datetime}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CITATION_SYSTEM_PROMPT":
            ns["SYS"] = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name in ("_today_label", "_time_aware_note"):
            exec(ast.get_source_segment(src, n), ns)
    return ns["SYS"], ns["_time_aware_note"]()


def _app_url_extractor():
    """app.py's own extract_urls (regex, balanced-bracket tail trim, domain
    blacklist), loaded from source so DeepSeek's in-text links are extracted
    exactly as ChatGPT's are."""
    import ast
    src = open(os.path.join(ROOT, "app.py")).read()
    ns = {"re": re}
    want = {"URL_PATTERN", "DOMAIN_BLACKLIST", "_strip_url_tail", "extract_urls"}
    for n in ast.parse(src).body:
        name = getattr(n, "name", None) or (getattr(n.targets[0], "id", None) if isinstance(n, ast.Assign) else None)
        if name in want:
            exec(ast.get_source_segment(src, n), ns)
    return ns["extract_urls"]


def _env_key(name):
    k = os.environ.get(name)
    if not k:
        try:
            from dotenv import dotenv_values
            k = dotenv_values(os.path.join(ROOT, ".env")).get(name)
        except Exception:
            k = None
    if not k:
        sys.exit(f"{name} not set (environment or worktree .env)")
    return k.strip()


def deepseek_balance(key):
    try:
        b = requests.get("https://api.deepseek.com/user/balance",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30).json()
        return float(b["balance_infos"][0]["total_balance"])
    except Exception:
        return None


def collect_deepseek(key, prompt, n, raw_dir, system, note, extract):
    """DeepSeek's own API through its Anthropic-compatible endpoint, with its
    native web_search server tool: DeepSeek's search, run on DeepSeek's
    infrastructure (not OpenRouter's search on a third-party host). DeepSeek
    decides whether to search; it ignores max_uses (6 searches on the probe).
    It cites as markdown links in the text, so citations = the app's
    extract_urls over the answer; retrieved = every search result it read."""
    body = {"model": DEEPSEEK_MODEL, "max_tokens": 4000, "system": system,
            "messages": [{"role": "user", "content": prompt + note}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]}
    hdr = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    d, err = None, None
    for i in range(4):
        try:
            r = requests.post("https://api.deepseek.com/anthropic/v1/messages", headers=hdr, json=body, timeout=300)
            d = r.json() if r.content else {}
            if r.status_code == 200:
                err = None
                break
            err = str((d.get("error") or {}).get("message") or f"HTTP {r.status_code}")[:200]
            if r.status_code in (400, 401, 402, 403):
                break
        except Exception as e:
            err = type(e).__name__
        time.sleep(10 * (i + 1))
    row = {"llm": SURFACE_NAMES["deepseek"], "prompt": prompt, "response": "", "citations": [],
           "grounded": True, "error": err, "model_id": DEEPSEEK_MODEL, "surface": "deepseek",
           "grounding": {"source": "deepseek", "queries": [], "retrieved": [], "supports": []}}
    if err or not d:
        return row
    json.dump(d, open(os.path.join(raw_dir, f"deepseek_{n:03d}.json"), "w"))
    row["model_id"] = d.get("model") or DEEPSEEK_MODEL
    texts = []
    for c in d.get("content") or []:
        t = c.get("type")
        if t == "text":
            texts.append(c.get("text") or "")
        elif t == "server_tool_use":
            q = (c.get("input") or {}).get("query")
            if q:
                row["grounding"]["queries"].append(q)
        elif t == "web_search_tool_result" and isinstance(c.get("content"), list):
            for x in c["content"]:
                if isinstance(x, dict) and x.get("url"):
                    row["grounding"]["retrieved"].append({"url": x["url"], "title": x.get("title") or "",
                                                          "date": x.get("page_age") or ""})
    row["response"] = "\n\n".join(t for t in texts if t.strip())
    row["citations"] = extract(row["response"])
    return row


def meta_key():
    """META_API_KEY (our name) or MODEL_API_KEY (Meta's documented name)."""
    for name in ("META_API_KEY", "MODEL_API_KEY"):
        k = os.environ.get(name)
        if not k:
            try:
                from dotenv import dotenv_values
                k = dotenv_values(os.path.join(ROOT, ".env")).get(name)
            except Exception:
                k = None
        if k:
            return k.strip()
    sys.exit("META_API_KEY not set (environment or worktree .env)")


def collect_metaai(key, prompt, n, raw_dir, system, note, gl="us"):
    """Meta AI through Meta's own Model API (api.meta.ai, Responses API) with
    its built-in web_search tool: Meta's search on Meta's infrastructure, the
    model behind Meta AI. Meta decides whether to search, as Claude, Gemini
    and Mistral do; grounding records whether it did. Citations are the
    answer's url_citation annotations, in order; supports pair each cited span
    with its URL; retrieved = every result the search returned
    (include web_search_call.results). Same system prompt and date note as the
    product; user_location follows --gl, as the SerpApi surfaces do."""
    tool = {"type": "web_search", "user_location": {"type": "approximate", "country": (gl or "us").upper()}}
    base = {"model": META_MODEL, "tools": [tool], "include": ["web_search_call.results"]}
    bodies = [dict(base, instructions=system, input=prompt + note),
              dict(base, input=[{"role": "system", "content": system},
                                {"role": "user", "content": prompt + note}])]
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    d, err, bi = None, None, 0
    for i in range(10):
        try:
            r = requests.post("https://api.meta.ai/v1/responses", headers=hdr, json=bodies[bi], timeout=300)
            if r.status_code == 429:
                err = "rate limited"
                time.sleep(min(60, float(r.headers.get("retry-after") or 15)))
                continue
            d = r.json() if r.content else {}
            if r.status_code == 200:
                err = None
                break
            e_ = d.get("error") if isinstance(d.get("error"), dict) else {}
            err = str(e_.get("message") or d.get("message") or f"HTTP {r.status_code}")[:200]
            if r.status_code == 400 and bi == 0 and "instruction" in err.lower():
                bi = 1      # fall back to a system message if 'instructions' is not accepted
                continue
            if r.status_code in (400, 401, 402, 403, 404, 422):
                break
        except Exception as e:
            err = type(e).__name__
        time.sleep(10)
    row = {"llm": SURFACE_NAMES["metaai"], "prompt": prompt, "response": "", "citations": [],
           "grounded": True, "error": err, "model_id": META_MODEL, "surface": "metaai",
           "grounding": {"source": "meta", "queries": [], "retrieved": [], "supports": []}}
    if err or not d:
        return row
    json.dump(d, open(os.path.join(raw_dir, f"metaai_{n:03d}.json"), "w"))
    row["model_id"] = d.get("model") or META_MODEL
    texts, seen = [], []
    # Muse Spark narrates while it searches ("I've got strong early leads..."):
    # those message items carry phase "commentary". The answer the user gets is
    # the non-commentary message(s); fall back to the last message if every one
    # is commentary. Citations live on the answer (14 of 14 on the probe).
    msgs = [o for o in d.get("output") or [] if o.get("type") == "message"]
    answer_ids = {id(o) for o in ([m for m in msgs if m.get("phase") != "commentary"] or msgs[-1:])}
    row["grounding"]["commentary_messages"] = sum(1 for m in msgs if id(m) not in answer_ids)
    for o in d.get("output") or []:
        if o.get("type") == "web_search_call":
            q = (o.get("action") or {}).get("query") if isinstance(o.get("action"), dict) else None
            if q:
                row["grounding"]["queries"].append(q)
            for x in o.get("results") or []:
                if isinstance(x, dict) and x.get("url"):
                    row["grounding"]["retrieved"].append({"url": x["url"], "title": x.get("title") or "",
                                                          "snippet": (x.get("snippet") or "")[:300]})
            row["grounding"]["searched"] = True
        if o.get("type") != "message" or id(o) not in answer_ids:
            continue
        for blk in o.get("content") or []:
            if blk.get("type") != "output_text":
                continue
            t = blk.get("text") or ""
            texts.append(t)
            for a in blk.get("annotations") or []:
                if a.get("type") != "url_citation" or not a.get("url"):
                    continue
                u = a["url"]
                if u not in seen:
                    seen.append(u)
                s0, e0 = a.get("start_index"), a.get("end_index")
                claim = t[s0:e0] if isinstance(s0, int) and isinstance(e0, int) else ""
                row["grounding"]["supports"].append({"claim": claim.strip()[:600], "urls": [u],
                                                     "title": a.get("title") or ""})
    row["response"] = "\n\n".join(t for t in texts if t.strip())
    row["citations"] = [{"url": u, "domain": domain(u)} for u in seen]
    return row


def mistral_key():
    k = os.environ.get("MISTRAL_API_KEY")
    if not k:
        try:
            from dotenv import dotenv_values
            k = dotenv_values(os.path.join(ROOT, ".env")).get("MISTRAL_API_KEY")
        except Exception:
            k = None
    if not k:
        sys.exit("MISTRAL_API_KEY not set (environment or worktree .env)")
    return k.strip()


def collect_mistral(key, prompt, n, raw_dir, system, note):
    """Mistral's Conversations API with its built-in web_search connector.
    The connector can't be forced (tool_choice 'required' is rejected for
    built-in connectors), so Mistral decides whether to search, as Claude and
    Gemini do; the row's grounding records whether it did. Same system prompt
    and date note as the product. 429s on the per-minute token cap are waited
    out (Retry-After or 20s steps, up to ~4 minutes)."""
    body = {"model": MISTRAL_MODEL, "inputs": prompt + note, "instructions": system,
            "tools": [{"type": "web_search"}], "store": False}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    d, err = None, None
    for i in range(12):
        try:
            r = requests.post("https://api.mistral.ai/v1/conversations", headers=hdr, json=body, timeout=240)
            if r.status_code == 429:
                err = "rate limited"
                time.sleep(min(60, float(r.headers.get("retry-after") or 20)))
                continue
            d = r.json()
            if r.status_code == 200:
                err = None
                break
            err = str(d.get("detail") or d.get("message") or f"HTTP {r.status_code}")[:200]
            if r.status_code in (400, 401, 403, 422):
                break
        except Exception as e:
            err = type(e).__name__
        time.sleep(10)
    row = {"llm": SURFACE_NAMES["mistral"], "prompt": prompt, "response": "", "citations": [],
           "grounded": True, "error": err, "model_id": MISTRAL_MODEL, "surface": "mistral",
           "grounding": {"source": "mistral", "queries": [], "retrieved": [], "supports": []}}
    if d is None or err:
        return row
    json.dump(d, open(os.path.join(raw_dir, f"mistral_{n:03d}.json"), "w"))
    text, cur, seen = [], "", []
    for o in d.get("outputs") or []:
        if o.get("type") == "tool.execution":
            try:
                q = json.loads(o.get("arguments") or "{}").get("query")
                if q:
                    row["grounding"]["queries"].append(q)
            except Exception:
                pass
            try:
                res = json.loads((o.get("info") or {}).get("result") or "{}")
                for v in (res.values() if isinstance(res, dict) else []):
                    if isinstance(v, dict) and v.get("url"):
                        row["grounding"]["retrieved"].append({"url": v["url"], "title": v.get("title") or "",
                                                              "date": v.get("date") or v.get("page_age") or ""})
            except Exception:
                pass
        if o.get("type") != "message.output":
            continue
        row["model_id"] = o.get("model") or row["model_id"]
        c = o.get("content")
        if isinstance(c, str):
            text.append(c)
            continue
        pending = []
        for ch in c or []:
            if ch.get("type") == "text":
                if pending and cur.strip():
                    row["grounding"]["supports"].append({"claim": cur.strip()[-600:], "urls": pending})
                pending = []
                cur = ch.get("text") or ""
                text.append(cur)
            elif ch.get("type") == "tool_reference" and ch.get("url"):
                pending.append(ch["url"])
                if ch["url"] not in seen:
                    seen.append(ch["url"])
        if pending and cur.strip():
            row["grounding"]["supports"].append({"claim": cur.strip()[-600:], "urls": pending})
    row["response"] = "".join(text)
    row["citations"] = [{"url": u, "domain": domain(u)} for u in seen]
    return row


def _app_module():
    """The app itself, for its resolver. Heavy import; DB noise suppressed."""
    import contextlib, io
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    os.environ.setdefault("FLASK_SECRET_KEY", "offline")
    with contextlib.redirect_stdout(io.StringIO()):
        import app as A   # noqa: E402
    return A


def resolve_rows(rows):
    """Run the app's link resolver over these rows in place, so add-on
    platforms follow the five assistants' rules. Returns (dropped, merged)
    record lists in the payload's dropped_citations / merged_citation_repeats
    shapes."""
    urls = sorted({c["url"] for r in rows for c in (r.get("citations") or []) if c.get("url")})
    if not urls:
        return [], []
    A = _app_module()
    url_map = A._resolve_and_verify_urls(urls)
    dropped, merged = [], []
    A._apply_url_resolution(rows, url_map, dropped_out=dropped, merged_out=merged)
    A._remap_grounding_urls(rows, url_map)
    print(f"  links resolved: {len(urls)} distinct | dropped {len(dropped)} | merged repeats {len(merged)}")
    return dropped, merged


def _write_merged(mp, rows, dropped, merged_reps):
    """Swap this out_dir's surface rows into merged_payload.json and record
    the resolver's dropped and merged links next to the original run's."""
    if not os.path.exists(mp):
        return
    m = json.load(open(mp))
    m["all_responses"] = [r for r in m["all_responses"] if not r.get("surface")] + rows
    surf_llms = {r.get("llm") for r in rows}
    m["dropped_citations"] = [d for d in (m.get("dropped_citations") or [])
                              if d.get("llm") not in surf_llms] + dropped
    m["merged_citation_repeats"] = [d for d in (m.get("merged_citation_repeats") or [])
                                    if d.get("llm") not in surf_llms] + merged_reps
    es = m.get("extra_surfaces") or {}
    es["links_resolved"] = {"dropped": len(dropped), "merged_repeats": len(merged_reps),
                            "date": time.strftime("%Y-%m-%d")}
    m["extra_surfaces"] = es
    json.dump(m, open(mp, "w"))


def resolve_only(out_dir):
    rp = os.path.join(out_dir, "surfaces_rows.json")
    rows = json.load(open(rp))
    before = sum(len(r.get("citations") or []) for r in rows)
    dropped, merged_reps = resolve_rows(rows)
    json.dump(rows, open(rp, "w"), indent=1)
    _write_merged(os.path.join(out_dir, "merged_payload.json"), rows, dropped, merged_reps)
    print(f"  citations {before} -> {sum(len(r.get('citations') or []) for r in rows)}")


def reparse(out_dir, prompts, p, resolve=True):
    """Re-derive AI Overview / AI Mode / Copilot rows from the raw responses
    already on disk, so a parsing fix never costs searches. Rows from other
    surfaces (Mistral, DeepSeek) are kept as they are."""
    raw = os.path.join(out_dir, "raw")
    old = json.load(open(os.path.join(out_dir, "surfaces_rows.json")))
    rows = [r for r in old if r.get("surface") not in SERP_SURFACES]
    before = {s: sum(len(r["citations"]) for r in old if r.get("surface") == s) for s in SERP_SURFACES}
    for n, q in enumerate(prompts):
        g, ov = os.path.join(raw, f"aio_{n:03d}_google.json"), os.path.join(raw, f"aio_{n:03d}_overview.json")
        if os.path.exists(g):
            if os.path.exists(ov):
                rows.append(to_row("aio", q, json.load(open(ov)).get("ai_overview"), None, "google_ai_overview"))
            else:
                ao = json.load(open(g)).get("ai_overview") or {}
                rows.append(to_row("aio", q, ao if ao.get("text_blocks") else None,
                                   None if ao.get("text_blocks") else "not_shown", "google_ai_overview"))
        for s, eng in (("aimode", "google_ai_mode"), ("copilot", "bing_copilot")):
            f = os.path.join(raw, f"{s}_{n:03d}.json")
            if os.path.exists(f):
                rows.append(to_row(s, q, json.load(open(f)), None, eng))
    dropped, merged_reps = resolve_rows(rows) if resolve else ([], [])
    json.dump(rows, open(os.path.join(out_dir, "surfaces_rows.json"), "w"), indent=1)
    _write_merged(os.path.join(out_dir, "merged_payload.json"), rows, dropped, merged_reps)
    for s in SERP_SURFACES:
        rs = [r for r in rows if r.get("surface") == s]
        if rs:
            print(f"  {SURFACE_NAMES[s]:<20} citations {before[s]} -> {sum(len(r['citations']) for r in rs)}"
                  f" | answers with any citation {sum(1 for r in rs if r['citations'])}/{len(rs)}"
                  f" | domains {len({c['domain'] for r in rs for c in r['citations']})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out_dir")
    ap.add_argument("--surfaces", default="aio,aimode,copilot")
    ap.add_argument("--max-searches", type=int, default=None)
    ap.add_argument("--gl", default="us"); ap.add_argument("--hl", default="en")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild the SerpApi surfaces' rows from out_dir/raw (no searches), keep other rows")
    ap.add_argument("--resolve-only", action="store_true",
                    help="run the app's link resolver over an existing out_dir's rows (no API calls)")
    ap.add_argument("--no-resolve", action="store_true", help="skip link resolution (offline only)")
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

    if a.resolve_only:
        return resolve_only(a.out_dir)
    if a.reparse:
        return reparse(a.out_dir, prompts, p, resolve=not a.no_resolve)
    need = len(prompts) * sum(COST[s] for s in surfaces)
    serp = [s for s in surfaces if s in SERP_SURFACES]
    key, acct, left = None, {}, None
    if serp:
        key = api_key()
        acct = requests.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=30).json()
        left = acct.get("total_searches_left")
    mkey = mistral_key() if "mistral" in surfaces else None
    dkey = _env_key("DEEPSEEK_API_KEY") if "deepseek" in surfaces else None
    metakey = meta_key() if "metaai" in surfaces else None
    bal0 = deepseek_balance(dkey) if dkey else None
    caps = [x for x in (left, a.max_searches) if x is not None]
    cap = min(caps) if caps and serp else None
    print(f"{len(prompts)} prompts x {surfaces} -> up to {need} SerpApi searches "
          f"(AI Overviews not shown cost 1, not 2) | account: {acct.get('plan_name')}, {left} left")
    if cap is not None and need > cap:
        sys.exit(f"refusing: worst case {need} searches exceeds the cap of {cap}. "
                 f"Use --limit, fewer --surfaces, or a larger plan.")
    if a.dry_run:
        return

    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)
    raw_dir = os.path.join(a.out_dir, "raw")
    jobs = [(s, q, i) for s in serp for i, q in enumerate(prompts)]
    res = []

    def _mistral_all():
        system, note = _app_prompting()
        out = []
        for i, q in enumerate(prompts):
            out.append((collect_mistral(mkey, q, i, raw_dir, system, note), 0))
            print(f"  Mistral {i + 1}/{len(prompts)}", flush=True)
        return out
    def _deepseek_all():
        system, note = _app_prompting()
        extract = _app_url_extractor()
        with ThreadPoolExecutor(3) as dx:
            return list(dx.map(lambda iq: (collect_deepseek(dkey, iq[1], iq[0], raw_dir, system, note, extract), 0),
                               list(enumerate(prompts))))
    def _metaai_all():
        system, note = _app_prompting()
        with ThreadPoolExecutor(3) as mx:
            return list(mx.map(lambda iq: (collect_metaai(metakey, iq[1], iq[0], raw_dir, system, note, a.gl), 0),
                               list(enumerate(prompts))))
    with ThreadPoolExecutor(6) as ex:
        mf = ex.submit(_mistral_all) if mkey else None
        df = ex.submit(_deepseek_all) if dkey else None
        mtf = ex.submit(_metaai_all) if metakey else None
        res = list(ex.map(lambda j: collect_one(key, j[0], j[1], j[2], raw_dir, a.gl, a.hl), jobs))
        if mf:
            res += mf.result()
        if df:
            res += df.result()
        if mtf:
            res += mtf.result()
    rows = [r for r, _ in res]
    used = sum(c for _, c in res)
    dropped, merged_reps = resolve_rows(rows) if not a.no_resolve else ([], [])

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
        if s == "deepseek" and bal0 is not None:
            bal1 = deepseek_balance(dkey)
            if bal1 is not None:
                lines.append(f"  DeepSeek spend (balance delta): ${bal0 - bal1:.4f}")
        if s in ("mistral", "deepseek", "metaai"):
            lines.append(f"  {SURFACE_NAMES[s]} searched the web on {sum(1 for r in shown if r['grounding']['queries'] or r['citations'])}"
                         f" of {len(shown)} answers")
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
    merged["dropped_citations"] = list(p.get("dropped_citations") or []) + dropped
    merged["merged_citation_repeats"] = list(p.get("merged_citation_repeats") or []) + merged_reps
    merged["extra_surfaces"] = {"collected": time.strftime("%Y-%m-%d"), "via": "serpapi",
                                "surfaces": [SURFACE_NAMES[s] for s in surfaces],
                                "links_resolved": None if a.no_resolve else
                                    {"dropped": len(dropped), "merged_repeats": len(merged_reps),
                                     "date": time.strftime("%Y-%m-%d")},
                                "note": "Appended by collect_surfaces.py. Headline figures stored in "
                                        "this payload were computed on the original agents only."}
    json.dump(merged, open(os.path.join(a.out_dir, "merged_payload.json"), "w"))
    print(f"wrote {a.out_dir}/surfaces_rows.json, merged_payload.json, summary.txt")


if __name__ == "__main__":
    main()
