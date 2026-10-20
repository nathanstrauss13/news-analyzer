#!/usr/bin/env python3
"""page_fetch.py: fetch cited pages the way a reader would, and read the byline.

Replaces hand-driving a browser tab for pages that refuse plain HTTP fetches.
Four tiers, tried in order until one returns a real article page:

  direct   requests with browser headers (fast; most pages)
  chrome   headless Google Chrome through Playwright (clears Axios, Block Club,
           Food On Demand, Cloudflare "checking your browser" and most JS walls)
  reader   r.jina.ai plain-text reader (text only, byline usually lost)
  wayback  the Internet Archive's latest snapshot (hard paywalls: NYT, WSJ, WaPo)

Every fetched page is cached in ~/.signal_mcp/page_cache.sqlite (HTML, tier,
final URL, fetched_at) so a re-run or a second audit never refetches.

Extraction (article level, not site level):
  byline        visible byline element, then meta author tags, then JSON-LD
                Article.author. A JSON-LD author list that is really the site's
                staff roster (Restaurant Business) or a date (Gov Tech) is
                detected and demoted; byline_note says why.
  credit        syndication credit when present (Miami Herald / TNS, AP,
                Reuters, Bloomberg, Tribune Content Agency)
  published     JSON-LD datePublished, then the usual meta tags
  counts        optional: --names "Coco Robotics,Serve Robotics" (exact phrase,
                case-sensitive) or --count-module path/to/module.py exposing
                count(text) -> dict (the audit's own conventions)

usage:
  python3 page_fetch.py <urls.txt | pages.json> <out.jsonl>
         [--tiers direct,chrome,reader,wayback] [--workers 12] [--chrome-concurrency 4]
         [--names "A,B"] [--count-module editorial_scan.py] [--refetch]
         [--key walled]   # for a pages.json of {"readable": [[url, meta], ...], "walled": [...]}

Rows carry every input meta field (e.g. cited_unbranded) plus:
  url, final_url, status (ok | blocked | http NNN | thin | error), tier, title,
  byline, byline_source, byline_note, credit, published, chars, <counts>,
  checked_date, cache_hit
"""
import argparse, asyncio, csv, datetime, hashlib, html as H, importlib.util, json, os, re, sqlite3, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse

import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
CACHE_PATH = os.path.expanduser("~/.signal_mcp/page_cache.sqlite")
MIN_ARTICLE_CHARS = 1500
WALL_MARKERS = re.compile(r"checking your browser|verify you are human|access denied|enable javascript and cookies|"
                          r"are you a robot|unusual traffic|please enable cookies|captcha|cloudflare|"
                          r"subscribe to continue|subscription required|this content is for subscribers", re.I)
SITE_HOSTS_SKIP = {"youtube.com", "youtu.be", "x.com", "twitter.com", "reddit.com", "linkedin.com", "facebook.com",
                   "instagram.com", "tiktok.com"}


# ---------- cache ----------
def _db():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, final_url TEXT, tier TEXT, http INTEGER,
                   fetched_at TEXT, html BLOB)""")
    return con


def cache_get(url):
    con = _db(); row = con.execute("SELECT final_url, tier, http, fetched_at, html FROM pages WHERE url=?", (url,)).fetchone(); con.close()
    if not row: return None
    return {"final_url": row[0], "tier": row[1], "http": row[2], "fetched_at": row[3], "html": zlib.decompress(row[4]).decode("utf-8", "replace")}


def cache_put(url, final_url, tier, http, html):
    con = _db()
    con.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?)", (url, final_url, tier, http, datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", zlib.compress(html.encode("utf-8", "ignore"), 6)))
    con.commit(); con.close()


# ---------- extraction ----------
def strip_html(s, trim=False):
    tags = "script|style|noscript|svg|template" + ("|nav|footer|aside|header" if trim else "")
    s = re.sub(r"(?is)<(" + tags + r")[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", H.unescape(s)).strip()


def meta(raw, *names):
    for n in names:
        m = (re.search(r'<meta[^>]+(?:property|name)=["\']' + re.escape(n) + r'["\'][^>]+content=["\']([^"\']{1,300})', raw, re.I) or
             re.search(r'<meta[^>]+content=["\']([^"\']{1,300})["\'][^>]+(?:property|name)=["\']' + re.escape(n) + r'["\']', raw, re.I))
        if m:
            return H.unescape(m.group(1)).strip()
    return ""


def jsonld_article(raw):
    """Authors and dates from JSON-LD objects typed as an article (not WebSite/Organization)."""
    authors, dates = [], []
    for m in re.finditer(r'(?is)<script[^>]+application/ld\+json[^>]*>(.*?)</script>', raw):
        try:
            obj = json.loads(m.group(1).strip())
        except Exception:
            continue
        stack = obj if isinstance(obj, list) else [obj]
        seen = []
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                if "@graph" in o: stack.extend(o["@graph"] if isinstance(o["@graph"], list) else [o["@graph"]])
                t = str(o.get("@type", ""))
                if re.search(r"article|blogposting|report", t, re.I):
                    a = o.get("author")
                    for x in (a if isinstance(a, list) else [a]) if a else []:
                        nm = x.get("name") if isinstance(x, dict) else x
                        if isinstance(nm, str) and nm.strip(): authors.append(nm.strip())
                    for k in ("datePublished", "dateCreated", "uploadDate"):
                        if isinstance(o.get(k), str): dates.append(o[k])
                for v in o.values():
                    if isinstance(v, (dict, list)): stack.append(v)
            elif isinstance(o, list):
                stack.extend(o)
    return list(dict.fromkeys(authors)), dates


DATE_LIKE = re.compile(r"^\s*(?:[A-Z][a-z]+\.? \d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\s*[•·|-]?\s*$")
NOT_A_NAME = re.compile(r"staff|editor|newsroom|contributor|admin|team|desk|wire|press release|sponsored|\bby\b$", re.I)
STOP_WORDS = re.compile(r"^(?:share|follow|email|print|subscribe|comments?|tech|news|business|politics|opinion|sports|more|menu|search|sign in|log in|author|byline|written by|by)$", re.I)
NAME_SHAPE = re.compile(r"^[A-Z][A-Za-z'.-]+(?: (?:and |& |, )?[A-Z][A-Za-z'.-]+){1,5}$")


def visible_byline(raw):
    """The byline the reader sees: an element whose class/rel says byline or author, text starting 'By'."""
    cands = []
    for m in re.finditer(r'(?is)<(a|span|div|p|address)[^>]+(?:class|rel|data-testid|itemprop)=["\'][^"\']*(?:byline|author|contributor|writer)[^"\']*["\'][^>]*>(.*?)</\1>', raw):
        t = re.sub(r"\s+", " ", H.unescape(re.sub(r"(?s)<[^>]+>", " ", m.group(2)))).strip()
        if 2 < len(t) < 160:
            cands.append(t)
    # prefer a candidate that starts with "By"
    for t in cands:
        mm = re.match(r"^(?:By|by|BY)\s+(.{2,120})$", t)
        if mm: return clean_name(mm.group(1))
    for t in cands:
        c = clean_name(t)
        if c and not STOP_WORDS.match(c) and not DATE_LIKE.match(c) and (NAME_SHAPE.match(c) or re.search(r"staff", c, re.I)):
            return c
    # visible "By Name" near the top of the text, as a last resort
    txt = strip_html(raw[:120000])
    mm = re.search(r"\bBy ([A-Z][A-Za-z'.-]+(?: (?:and |& )?[A-Z][A-Za-z'.-]+){1,4})\b", txt)
    return clean_name(mm.group(1)) if mm else ""


MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"


def clean_name(s):
    s = re.sub(r"\s*(?:\||•|·|,)?\s*(?:Staff Writer|Senior Writer|Reporter|Published|Updated|Posted)\b.*$", "", s, flags=re.I)
    s = re.sub(r"\s+(?:on|at|-|\||•|·)\s+" + MONTH + r"\s+\d{1,2},?\s+\d{4}.*$", "", s)      # "Joe Guszkowski on Jun. 12, 2025"
    s = re.sub(r"\s*[|•·-]\s*" + MONTH + r"\s+\d{1,2},?\s+\d{4}.*$", "", s)                   # "PYMNTS | August 26, 2026"
    s = re.sub(r"\s*[|•·-]\s*\d{1,2}/\d{1,2}/\d{2,4}.*$", "", s)
    s = re.split(r"\s+[|•·]\s+", s)[0]                                                        # "Jane Miller | Chicago Tribune", "James Dargan · 19 June 2026"
    s = re.sub(r"\s*(?:You're|You are) currently following.*$", "", s)
    s = re.sub(r"\s+\d{1,2}\s+" + MONTH + r"\s+\d{4}.*$", "", s)                               # day-first dates
    s = re.sub(r",?\s+(?:Senior|Deputy|Managing|Executive|Associate|Assistant|Contributing)?\s*(?:Editor|Writer|Columnist|Correspondent|Reporter|Journalist)(?:-in-Chief)?\b.*$", "", s)
    s = re.sub(r"\s+\d{4}-\d{2}-\d{2}T\S*.*$", "", s)                                          # trailing ISO timestamp
    s = re.sub(r"\s*\[month\].*$", "", s)                                                     # unrendered date template
    s = re.sub(r"\s*[|•·]\s*[^\s@|]+@[^\s@|]+", "", s)                                        # "| jmiller@chicagotribune.com"
    s = re.sub(r"\s+" + MONTH + r"\s+\d{1,2},?\s+\d{4}.*$", "", s)                           # "DC Velocity Staff Jun 08, 2026"
    s = re.sub(r"\s+(?:Tech|Business|Politics|News|Opinion|Sports|Technology|Media)$", "", s)  # "Rachyl Jones Tech" (section label)
    s = re.sub(r"\s+", " ", s).strip(" ,|•·-")
    return s[:120]


def extract(url, raw):
    text = strip_html(raw)
    title = meta(raw, "og:title") or H.unescape((re.search(r"(?is)<title[^>]*>(.*?)</title>", raw) or [None, ""])[1]).strip()
    ld_authors, ld_dates = jsonld_article(raw)
    vis = visible_byline(raw)
    meta_author = meta(raw, "author", "article:author", "parsely-author", "sailthru.author", "dc.creator", "byl")
    if meta_author.startswith("http"): meta_author = ""
    byline, source, note = "", "", ""
    if vis and not DATE_LIKE.match(vis):
        byline, source = vis, "visible byline"
    elif meta_author:
        byline, source = meta_author, "meta author"
    norm = lambda x: re.sub(r"\s+", " ", x or "").strip().lower()
    ld_authors = [re.sub(r"\s+", " ", x) for x in ld_authors]
    if ld_authors:
        first = norm(byline.split(";")[0])
        roster = len(ld_authors) >= 3 and (not byline or first not in [norm(x) for x in ld_authors])
        if roster:
            note = f"JSON-LD author list looks like a staff roster ({len(ld_authors)} names), ignored"
        elif not byline:
            byline, source = "; ".join(ld_authors[:3]), "JSON-LD article author"
        elif byline and first in [norm(x) for x in ld_authors] and 1 < len(ld_authors) <= 3:
            byline, source = "; ".join(ld_authors), source + " + JSON-LD"
    if byline and DATE_LIKE.match(byline):
        note = (note + "; " if note else "") + f"byline slot held a date ({byline}), cleared"; byline, source = "", ""
    if byline and NOT_A_NAME.search(byline) and not re.search(r"staff", byline, re.I):
        note = (note + "; " if note else "") + "byline is not a person"
    credit = ""
    CRED = r"\(TNS\)|Tribune Content Agency|Tribune News Service|\bAssociated Press\b|\(AP\)|\bAP\b(?= —| -)|\bReuters\b(?= —| -|\))|/ Reuters|\bBloomberg\b(?= —| -|\))|\bCNN Newsource\b"
    # a syndication credit sits in the byline/dateline or in the closing rights line, never in the body
    ttext = strip_html(raw, trim=True)
    zone = (byline + " | " + ttext[:1200] + " | " + ttext[-1200:])
    cm = re.search(CRED, zone)
    if cm:
        credit = cm.group(0).strip(" /")
        srcm = re.search(r"([A-Z][A-Za-z.&' -]{3,40}(?:Herald|Times|Tribune|Post|Journal|News|Sun|Star|Bee|Chronicle|Gazette))\b", byline + " " + ttext[:1200])
        if srcm and srcm.group(1).strip().lower() not in title.lower(): credit = srcm.group(1).strip() + " / " + credit
    date = (ld_dates[0] if ld_dates else "") or meta(raw, "article:published_time", "datePublished", "date", "pubdate", "parsely-pub-date", "sailthru.date", "dc.date", "og:updated_time")
    return {"title": title[:200], "byline": byline, "byline_source": source, "byline_note": note, "credit": credit,
            "published": date[:40], "chars": len(text), "_text": text}


def looks_walled(raw, text):
    return len(text) < MIN_ARTICLE_CHARS and bool(WALL_MARKERS.search(raw[:20000] + text[:3000]))


# ---------- tiers ----------
def fetch_direct(url, timeout=25):
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.url, r.text


def fetch_reader(url, timeout=40):
    r = requests.get("https://r.jina.ai/" + url, headers={"User-Agent": UA, "Accept": "text/plain"}, timeout=timeout)
    return r.status_code, url, r.text


def fetch_wayback(url, timeout=40):
    a = requests.get("https://archive.org/wayback/available", params={"url": url}, timeout=timeout).json()
    snap = (a.get("archived_snapshots") or {}).get("closest") or {}
    if not snap.get("available"): return 404, url, ""
    snap_url = snap["url"].replace("http://", "https://", 1)
    if re.match(r"https://web\.archive\.org/web/\d+/", snap_url):
        snap_url = re.sub(r"(/web/\d+)/", r"\1id_/", snap_url, count=1)  # id_ = raw page without the toolbar
    r = requests.get(snap_url, headers=HEADERS, timeout=timeout)
    return r.status_code, snap.get("url", snap_url), r.text


async def fetch_chrome_many(urls, concurrency=4, timeout_ms=35000, channel="chrome"):
    from playwright.async_api import async_playwright
    out = {}
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel=channel, headless=True)
        except Exception:
            browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=UA, locale="en-US", viewport={"width": 1280, "height": 900})
        await ctx.route(re.compile(r"\.(png|jpe?g|gif|webp|svg|woff2?|ttf|otf|mp4|m3u8|mp3)(\?|$)|doubleclick|googlesyndication|adsystem|taboola|outbrain"), lambda r: asyncio.ensure_future(r.abort()))
        sem = asyncio.Semaphore(concurrency)

        async def one(u):
            async with sem:
                page = await ctx.new_page()
                try:
                    resp = await page.goto(u, wait_until="domcontentloaded", timeout=timeout_ms)
                    await page.wait_for_timeout(2500)
                    html = await page.content(); status = resp.status if resp else 0
                    if status in (403, 429, 503) or looks_walled(html, strip_html(html)):
                        await page.wait_for_timeout(6000)                      # let a JS challenge / consent script settle
                        if status == 429:
                            resp = await page.goto(u, wait_until="domcontentloaded", timeout=timeout_ms); await page.wait_for_timeout(3000)
                            status = resp.status if resp else status
                        html = await page.content()
                    out[u] = (status, page.url, html)
                except Exception as e:
                    out[u] = (0, u, f"__error__ {type(e).__name__}: {str(e)[:120]}")
                finally:
                    await page.close()
        await asyncio.gather(*(one(u) for u in urls))
        await browser.close()
    return out


# ---------- orchestration ----------
def judge(http, html):
    """ok | blocked | http NNN | thin | error, plus the extracted text."""
    if isinstance(html, str) and html.startswith("__error__"):
        return "error", ""
    if http and http != 200:
        return (f"http {http}" if http not in (401, 403, 429, 503) else "blocked"), ""
    text = strip_html(html)
    if looks_walled(html, text): return "blocked", text
    if len(text) < MIN_ARTICLE_CHARS: return "thin", text
    return "ok", text


def run(items, tiers, workers, chrome_conc, names, count_fn, refetch, log=print):
    """items: list of (url, meta dict). Returns rows."""
    today = datetime.date.today().isoformat()
    rows = {}; pending = []
    for u, m in items:
        host = (urlparse(u).netloc or "").lower().removeprefix("www.")
        if host in SITE_HOSTS_SKIP:
            rows[u] = {"url": u, "final_url": u, "status": "skipped (social/video host)", "tier": "", **m}; continue
        c = None if refetch else cache_get(u)
        if c:
            st, text = judge(c["http"], c["html"])
            rows[u] = finish(u, c["final_url"], st, c["tier"], c["html"], text, m, names, count_fn, today, cache_hit=True)
            if st == "ok": continue
        pending.append((u, m))
    log(f"{len(items)} pages: {len(items) - len(pending)} settled from cache/skip, {len(pending)} to fetch")

    for tier in tiers:
        if not pending: break
        log(f"tier {tier}: {len(pending)} pages")
        results = {}
        if tier == "chrome":
            urls = [u for u, _ in pending]
            for i in range(0, len(urls), 40):
                results.update(asyncio.run(fetch_chrome_many(urls[i:i + 40], concurrency=chrome_conc)))
                log(f"  chrome {min(i + 40, len(urls))}/{len(urls)}")
        else:
            fn = {"direct": fetch_direct, "reader": fetch_reader, "wayback": fetch_wayback}[tier]
            with ThreadPoolExecutor(max_workers=workers if tier == "direct" else max(2, workers // 3)) as ex:
                futs = {ex.submit(fn, u): u for u, _ in pending}
                for f in as_completed(futs):
                    u = futs[f]
                    try:
                        results[u] = f.result()
                    except Exception as e:
                        results[u] = (0, u, f"__error__ {type(e).__name__}")
        still = []
        for u, m in pending:
            http, final, html = results.get(u, (0, u, "__error__ missing"))
            if tier == "reader" and http == 200 and html and not html.startswith("__error__"):
                pub = re.search(r"^Published Time:\s*(\S+)", html, re.M); ttl = re.search(r"^Title:\s*(.+)$", html, re.M)
                head = ("<title>" + H.escape(ttl.group(1).strip()) + "</title>" if ttl else "") + \
                       ('<meta name="datePublished" content="' + H.escape(pub.group(1)) + '">' if pub else "")
                html = "<html><head>" + head + "</head><body><pre>" + H.escape(html) + "</pre></body></html>"  # reader returns text
            st, text = judge(http, html)
            if st == "ok":
                cache_put(u, final, tier, http, html)
                rows[u] = finish(u, final, st, tier, html, text, m, names, count_fn, today)
            else:
                prev = rows.get(u)
                if not prev or prev.get("status") in ("error", "thin", "blocked") or prev.get("status", "").startswith("http"):
                    rows[u] = finish(u, final, st, tier, html if isinstance(html, str) else "", text, m, names, count_fn, today)
                still.append((u, m))
        ok_n = sum(1 for u, _ in pending if rows[u]["status"] == "ok")
        log(f"  {tier}: {ok_n} ok, {len(still)} remain")
        pending = still
    return [rows[u] for u, _ in items if u in rows]


def finish(u, final, st, tier, html, text, m, names, count_fn, today, cache_hit=False):
    row = {"url": u, "final_url": final, "status": st, "tier": tier, "cache_hit": cache_hit, "checked_date": today, **m}
    if st == "ok":
        ex = extract(u, html); t = ex.pop("_text"); row.update(ex)
        if count_fn: row.update(count_fn(t))
        for n in names: row[n] = len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])", t))
    else:
        row.update({"title": "", "byline": "", "byline_source": "", "byline_note": "", "credit": "", "published": "", "chars": len(text)})
    return row


def load_items(path, key):
    if path.endswith(".json"):
        p = json.load(open(path))
        if isinstance(p, dict) and key in p:
            return [(u, (v if isinstance(v, dict) else {})) for u, v in p[key]]
        if isinstance(p, list):
            return [(x, {}) if isinstance(x, str) else (x[0], x[1] if len(x) > 1 and isinstance(x[1], dict) else {}) for x in p]
    return [(l.strip(), {}) for l in open(path) if l.strip() and not l.startswith("#")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--tiers", default="direct,chrome,reader,wayback")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--chrome-concurrency", type=int, default=4)
    ap.add_argument("--names", default="", help="comma-separated exact phrases to count (case-sensitive)")
    ap.add_argument("--count-module", help="python file exposing count(text) -> dict")
    ap.add_argument("--key", default="walled"); ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--csv", help="also write a CSV with these columns in this order (comma-separated), or 'auto'")
    a = ap.parse_args()
    count_fn = None
    if a.count_module:
        spec = importlib.util.spec_from_file_location("cm", a.count_module); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); count_fn = mod.count
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    items = load_items(a.inp, a.key)
    t0 = time.time()
    rows = run(items, a.tiers.split(","), a.workers, a.chrome_concurrency, names, count_fn, a.refetch)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.csv:
        cols = [c for c in dict.fromkeys(k for r in rows for k in r)] if a.csv == "auto" else a.csv.split(",")
        with open(a.out.rsplit(".", 1)[0] + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rows]
    ok = [r for r in rows if r["status"] == "ok"]
    by_tier = {}
    for r in ok: by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"DONE {len(rows)} pages in {time.time() - t0:.0f}s | ok {len(ok)} {by_tier} | byline on {sum(1 for r in ok if r.get('byline'))} | "
          f"roster/date artifacts caught {sum(1 for r in ok if r.get('byline_note'))} | still blocked {sum(1 for r in rows if r['status'] == 'blocked')} | -> {a.out}")
