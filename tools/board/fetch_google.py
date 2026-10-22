#!/usr/bin/env python3
"""Pull Google Analytics 4 and Search Console figures for the Xsight Signal Board.

One-time setup (Nathan, in a browser):  python3 fetch_google.py --auth
Monthly refresh:                         python3 fetch_google.py            (writes google_live.json)
Merge into data.json:                    python3 fetch_google.py --merge

Uses the OAuth client on the Desktop and stores the refresh token in ~/.config/xsight-signal/google_token.json.
Server use: set GOOGLE_TOKEN_JSON to the contents of that token file and run without --auth; no browser, no client secret needed.
Dependencies: google-auth, google-api-python-client (runtime); google-auth-oauthlib (only for --auth).
Output: google_live.json, or print to stdout with --stdout for a job that stores it elsewhere.
Scopes are read-only. Property: GA4 462267417 (Xsighlab). Site: sc-domain:xsightlabs.com.

Blog (added Oct 22, 2026): the output also carries a "blog" key: the post list
from https://xsightlabs.com/blog/rss.xml (title, url, path, published, newest
first), each joined by path to Search Console clicks/impressions and GA4 page
views over the trailing 12 months and 90 days. xsightlabs.com sits behind a
Vercel checkpoint that rejects bare clients, so the feed is requested with a
browser-shaped header set and retried; if it still fails, the sitemap supplies
the URL list, and if that fails too the job carries the last good post list
forward (BOARD_PRIOR_BLOG env var, JSON, set by the server job). Blog URLs that
Google reports but the feed lacks are appended with GA4's page title and no
date, so the card never misses a post that has traffic.
"""
import json, os, sys, glob, datetime as dt
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")  # Google adds openid/email to the granted scopes
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/analytics.readonly",
          "https://www.googleapis.com/auth/webmasters.readonly"]
GA4_PROPERTY = "properties/462267417"
GSC_SITE = "sc-domain:xsightlabs.com"
TOKEN = Path.home() / ".config/xsight-signal/google_token.json"
HERE = Path(__file__).resolve().parent

def client_secret():
    hits = glob.glob(str(Path.home() / "Desktop/client_secret_*.json"))
    if not hits:
        sys.exit("No client_secret_*.json on the Desktop.")
    return hits[0]

def creds(interactive=False):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    c = None
    env_tok = os.environ.get("GOOGLE_TOKEN_JSON")  # server use: the saved token file's contents, as one env var
    if env_tok:
        c = Credentials.from_authorized_user_info(json.loads(env_tok), SCOPES)
    elif TOKEN.exists():
        c = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if c and c.expired and c.refresh_token:
        c.refresh(Request())
    if not c or not c.valid:
        if not interactive:
            sys.exit("No valid token. Run: python3 fetch_google.py --auth")
        # The Desktop client secret is a "web" client whose registered redirect is
        # http://localhost:8123/auth/google/callback, so run a tiny server on that exact path.
        from google_auth_oauthlib.flow import Flow
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.parse import urlparse, parse_qs
        flow = Flow.from_client_secrets_file(client_secret(), SCOPES, redirect_uri="http://localhost:8123/auth/google/callback")
        url, _state = flow.authorization_url(prompt="consent", access_type="offline")
        print("AUTH_URL:", url, flush=True)
        got = {}
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                q = parse_qs(urlparse(self.path).query)
                got["code"] = q.get("code", [None])[0]; got["error"] = q.get("error", [None])[0]
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(b"<h2>Xsight Signal Board: Google access granted. You can close this tab.</h2>" if got["code"] else b"<h2>No code received.</h2>")
            def log_message(self, *a): pass
        srv = HTTPServer(("localhost", 8123), H)
        while not got.get("code") and not got.get("error"):
            srv.handle_request()
        srv.server_close()
        if not got.get("code"):
            sys.exit("Consent failed: " + str(got.get("error")))
        flow.fetch_token(code=got["code"])
        c = flow.credentials
        TOKEN.parent.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(c.to_json())
        print("Token saved to", TOKEN)
    return c

def month_ranges(n=12):
    today = dt.date.today()
    first = today.replace(day=1)
    out = []
    for i in range(n, 0, -1):
        y, m = first.year, first.month - i
        while m <= 0:
            y, m = y - 1, m + 12
        start = dt.date(y, m, 1)
        end = (dt.date(y + (m // 12), (m % 12) + 1, 1) - dt.timedelta(days=1))
        out.append((start.strftime("%b %y"), start.isoformat(), end.isoformat()))
    out.append(("partial", first.isoformat(), today.isoformat()))
    return out

def ga4(c):
    from googleapiclient.discovery import build
    svc = build("analyticsdata", "v1beta", credentials=c, cache_discovery=False)
    months = []
    for label, s, e in month_ranges():
        body = {"dateRanges": [{"startDate": s, "endDate": e}],
                "dimensions": [{"name": "sessionDefaultChannelGroup"}],
                "metrics": [{"name": "sessions"}]}
        r = svc.properties().runReport(property=GA4_PROPERTY, body=body).execute()
        row = {"month": label, "start": s, "end": e, "byChannel": {}}
        for x in r.get("rows", []):
            row["byChannel"][x["dimensionValues"][0]["value"]] = int(x["metricValues"][0]["value"])
        row["total"] = sum(row["byChannel"].values())
        months.append(row)
    # AI assistant referrals and top sources, trailing 90 days
    end = dt.date.today(); start = end - dt.timedelta(days=90)
    body = {"dateRanges": [{"startDate": start.isoformat(), "endDate": end.isoformat()}],
            "dimensions": [{"name": "sessionSource"}], "metrics": [{"name": "sessions"}],
            "orderBys": [{"metric": {"metricName": "sessions"}, "desc": True}], "limit": 50}
    r = svc.properties().runReport(property=GA4_PROPERTY, body=body).execute()
    sources = {x["dimensionValues"][0]["value"]: int(x["metricValues"][0]["value"]) for x in r.get("rows", [])}
    ai = {k: v for k, v in sources.items() if any(t in k.lower() for t in ("chatgpt", "openai", "gemini", "claude", "anthropic", "perplexity", "copilot", "grok"))}
    body = {"dateRanges": [{"startDate": start.isoformat(), "endDate": end.isoformat()}],
            "dimensions": [{"name": "pagePath"}], "metrics": [{"name": "screenPageViews"}],
            "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}], "limit": 40}
    r = svc.properties().runReport(property=GA4_PROPERTY, body=body).execute()
    pages = [{"path": x["dimensionValues"][0]["value"], "views": int(x["metricValues"][0]["value"])} for x in r.get("rows", [])]
    return {"months": months, "topSources90d": sources, "aiAssistants90d": ai, "topPages90d": pages}

def gsc(c):
    from googleapiclient.discovery import build
    svc = build("searchconsole", "v1", credentials=c, cache_discovery=False)
    months = []
    for label, s, e in month_ranges():
        r = svc.searchanalytics().query(siteUrl=GSC_SITE, body={"startDate": s, "endDate": e, "dimensions": []}).execute()
        row = (r.get("rows") or [{}])[0]
        months.append({"month": label, "start": s, "end": e, "clicks": int(row.get("clicks", 0)),
                       "impressions": int(row.get("impressions", 0)), "ctr": row.get("ctr", 0), "position": row.get("position", 0)})
    end = dt.date.today() - dt.timedelta(days=2); start = end - dt.timedelta(days=90)
    def top(dim, n):
        r = svc.searchanalytics().query(siteUrl=GSC_SITE, body={"startDate": start.isoformat(), "endDate": end.isoformat(), "dimensions": [dim], "rowLimit": n}).execute()
        return [{dim: x["keys"][0], "clicks": int(x["clicks"]), "impressions": int(x["impressions"]), "position": round(x["position"], 1)} for x in r.get("rows", [])]
    queries = top("query", 200)
    brand = sum(q["clicks"] for q in queries if "xsight" in q["query"].lower())
    total = sum(q["clicks"] for q in queries) or 1
    return {"months": months, "topQueries90d": queries[:50], "topPages90d": top("page", 40),
            "brandedShare90d": round(brand / total, 3)}

BLOG_RSS = os.environ.get("BOARD_BLOG_RSS_URL") or "https://xsightlabs.com/blog/rss.xml"          # env override: tests only
BLOG_SITEMAP = os.environ.get("BOARD_BLOG_SITEMAP_URL") or "https://xsightlabs.com/sitemap.xml"
BLOG_PREFIX = "/blog/"
BROWSER_HEADERS = {   # the header shape that clears the Vercel checkpoint without JavaScript (tested Oct 22, 2026)
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="129", "Not=A?Brand";v="8"', "Sec-Ch-Ua-Mobile": "?0", "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none", "Upgrade-Insecure-Requests": "1"}


def _blog_path(url):
    from urllib.parse import urlsplit
    p = (urlsplit(url).path or "/").split("?")[0].rstrip("/")
    return p or "/"


def _get(url, attempts=3):
    import time, requests
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=25)
            if r.status_code == 200 and "Security Checkpoint" not in r.text[:2000]:
                return r.text, None
            last = f"http {r.status_code}" + (" (Vercel checkpoint)" if "Security Checkpoint" in r.text[:2000] else "")
        except Exception as e:
            last = f"{type(e).__name__}"
        time.sleep(3 * (i + 1))
    return None, last


def parse_rss(txt):
    """RSS text -> [{title, url, path, published}] newest first, or (None, error)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    try:
        root = ET.fromstring(txt.encode("utf-8") if isinstance(txt, str) else txt)
    except Exception as e:
        return None, f"rss parse: {type(e).__name__}"
    posts = []
    for it in root.iter("item"):
        link = (it.findtext("link") or "").strip()
        if not link:
            continue
        pub = (it.findtext("pubDate") or "").strip()
        try:
            iso = parsedate_to_datetime(pub).date().isoformat() if pub else None
        except Exception:
            iso = None
        posts.append({"title": (it.findtext("title") or "").strip(), "url": link, "path": _blog_path(link), "published": iso})
    if not posts:
        return None, "rss: no items"
    posts.sort(key=lambda x: x["published"] or "", reverse=True)
    return posts, None


def blog_posts_from_rss():
    """Direct fetch with browser-shaped headers; optionally (BOARD_BLOG_READER_FALLBACK=1)
    the product's reader proxy, which fetches from its own network when the
    direct path is refused. Returns (posts, error, via)."""
    txt, err = _get(BLOG_RSS)
    via = "direct"
    if txt is None and os.environ.get("BOARD_BLOG_READER_FALLBACK") == "1":
        import requests
        try:
            r = requests.get("https://r.jina.ai/" + BLOG_RSS, headers={"User-Agent": BROWSER_HEADERS["User-Agent"], "Accept": "text/plain",
                                                                        "X-Return-Format": "text"}, timeout=40)
            if r.status_code == 200 and "<item>" in r.text:
                txt, err, via = r.text[r.text.find("<"):], None, "reader"
            else:
                err = f"{err}; reader: http {r.status_code}"
        except Exception as e:
            err = f"{err}; reader: {type(e).__name__}"
    if txt is None:
        return None, f"rss: {err}", None
    posts, perr = parse_rss(txt)
    return (posts, None, via) if posts else (None, perr, None)


def blog_posts_from_sitemap():
    import xml.etree.ElementTree as ET
    txt, err = _get(BLOG_SITEMAP)
    if txt is None:
        return None, f"sitemap: {err}"
    try:
        root = ET.fromstring(txt.encode("utf-8"))
    except Exception as e:
        return None, f"sitemap parse: {type(e).__name__}"
    posts = []
    for u in root.iter():
        if u.tag.endswith("url"):
            loc = next((c.text.strip() for c in u if c.tag.endswith("loc") and c.text), "")
            mod = next((c.text.strip()[:10] for c in u if c.tag.endswith("lastmod") and c.text), None)
            if BLOG_PREFIX in loc and _blog_path(loc) not in (BLOG_PREFIX.rstrip("/"), "/blog/rss.xml"):
                posts.append({"title": "", "url": loc, "path": _blog_path(loc), "published": None, "lastmod": mod})
    return (posts, None) if posts else (None, "sitemap: no blog urls")


def _clean_title(t, path):
    """GA4 page titles carry the site suffix; a missing title falls back to the slug."""
    import re
    t = re.sub(r"\s*[|\-\u2013\u2014]\s*Xsight[^|]*$", "", (t or "").strip()).strip()
    if not t:
        slug = path.rstrip("/").rsplit("/", 1)[-1]
        t = " ".join(w.upper() if w in ("dpu", "npb", "xisa", "e1", "e1l", "x2", "tco", "ai") else w.capitalize() for w in slug.split("-"))
    return t


def blog(c):
    """Post list plus per-post Google metrics. Never raises on the feed side: a
    blocked feed yields the last good list (BOARD_PRIOR_BLOG) and an error note."""
    from googleapiclient.discovery import build
    source, err = "rss", None
    posts, err, via = blog_posts_from_rss()
    if via == "reader":
        source = "rss-reader"
    prior = []
    try:
        prior = json.loads(os.environ.get("BOARD_PRIOR_BLOG") or "[]")
    except Exception:
        prior = []
    if posts is None:
        source = "sitemap"
        sm, err2 = blog_posts_from_sitemap()
        if sm is not None:
            byp = {p["path"]: p for p in prior if isinstance(p, dict)}
            for p in sm:                                   # keep titles/dates from the last good list
                q = byp.get(p["path"])
                if q:
                    p["title"], p["published"] = q.get("title", ""), q.get("published")
            posts = sm
        elif prior:
            source, err = "prior", f"{err}; {err2}"
            posts = [{k: p.get(k) for k in ("title", "url", "path", "published")} for p in prior if isinstance(p, dict)]
        else:
            source, err = "none", f"{err}; {err2}"
            posts = []
        if source == "sitemap":
            err = f"{err}; {err2}" if err2 else err
    # Google metrics, joined by path
    today = dt.date.today(); end = today - dt.timedelta(days=2)
    windows = {"12m": ((end - dt.timedelta(days=365)).isoformat(), end.isoformat()),
               "90d": ((end - dt.timedelta(days=90)).isoformat(), end.isoformat())}
    gscv = build("searchconsole", "v1", credentials=c, cache_discovery=False)
    ga = build("analyticsdata", "v1beta", credentials=c, cache_discovery=False)
    metrics = {}
    titles = {}
    for w, (s, e) in windows.items():
        r = gscv.searchanalytics().query(siteUrl=GSC_SITE, body={
            "startDate": s, "endDate": e, "dimensions": ["page"], "rowLimit": 1000,
            "dimensionFilterGroups": [{"filters": [{"dimension": "page", "operator": "contains", "expression": BLOG_PREFIX}]}]}).execute()
        for x in r.get("rows", []):
            m = metrics.setdefault(_blog_path(x["keys"][0]), {})
            m[f"clicks_{w}"] = m.get(f"clicks_{w}", 0) + int(x["clicks"])
            m[f"impressions_{w}"] = m.get(f"impressions_{w}", 0) + int(x["impressions"])
        body = {"dateRanges": [{"startDate": s, "endDate": e}],
                "dimensions": [{"name": "pagePath"}, {"name": "pageTitle"}], "metrics": [{"name": "screenPageViews"}],
                "dimensionFilter": {"filter": {"fieldName": "pagePath", "stringFilter": {"matchType": "BEGINS_WITH", "value": BLOG_PREFIX}}},
                "limit": 1000}
        r = ga.properties().runReport(property=GA4_PROPERTY, body=body).execute()
        for x in r.get("rows", []):
            path = _blog_path(x["dimensionValues"][0]["value"]); m = metrics.setdefault(path, {})
            m[f"views_{w}"] = m.get(f"views_{w}", 0) + int(x["metricValues"][0]["value"])
            titles.setdefault(path, x["dimensionValues"][1]["value"])
    known = {p["path"] for p in posts}
    extra = []
    for path, m in metrics.items():
        if path in known or path in (BLOG_PREFIX.rstrip("/"), "/blog/rss.xml") or not path.startswith(BLOG_PREFIX):
            continue
        if (m.get("clicks_12m", 0) + m.get("views_12m", 0)) == 0:
            continue
        extra.append({"title": _clean_title(titles.get(path, ""), path), "url": "https://xsightlabs.com" + path, "path": path, "published": None, "from_google_only": True})
    out_posts = []
    for p in posts + extra:
        m = metrics.get(p["path"], {})
        row = {"title": p.get("title", ""), "url": p["url"], "path": p["path"], "published": p.get("published"),
               "clicks_12m": m.get("clicks_12m", 0), "impressions_12m": m.get("impressions_12m", 0),
               "clicks_90d": m.get("clicks_90d", 0), "impressions_90d": m.get("impressions_90d", 0),
               "views_12m": m.get("views_12m", 0), "views_90d": m.get("views_90d", 0)}
        if p.get("from_google_only"):
            row["from_google_only"] = True
        out_posts.append(row)
    out_posts.sort(key=lambda x: (x["published"] or "0000", x["clicks_12m"]), reverse=True)
    tot = {k: sum(p[k] for p in out_posts) for k in ("clicks_12m", "impressions_12m", "clicks_90d", "impressions_90d", "views_12m", "views_90d")}
    # Posts published per month, last 12 full months plus the current (partial) month,
    # counted from the feed's pubDates; undated posts are reported, not placed.
    # With no dated source (feed blocked and no seeded list) the counts are
    # unknown, not zero: emit null so a card never reads "nothing published".
    dated_source = any(p["published"] for p in out_posts)
    months = []
    for label, ms, me in month_ranges():
        n = sum(1 for p in out_posts if p["published"] and ms <= p["published"] <= me) if dated_source else None
        months.append({"month": ms[:7], "label": label if label != "partial" else dt.date.today().strftime("%b %y"),
                       "start": ms, "end": me, "posts": n, "partial": label == "partial"})
    return {"source": source, "error": err, "feed_url": BLOG_RSS, "post_count": len(out_posts),
            "undated_count": sum(1 for p in out_posts if not p["published"]), "dated_source": dated_source,
            "windows": {k: {"start": v[0], "end": v[1]} for k, v in windows.items()}, "totals": tot,
            "posts_by_month": months, "posts": out_posts}


def main():
    if "--auth" in sys.argv:
        creds(interactive=True); print("Authorised."); return
    c = creds()
    out = {"pulled": dt.datetime.now().isoformat(timespec="minutes"), "ga4": ga4(c), "searchConsole": gsc(c)}
    try:
        out["blog"] = blog(c)
    except Exception as e:                     # the Google feed must never fail because the blog step did
        out["blog"] = {"source": "none", "error": f"{type(e).__name__}: {str(e)[:200]}", "posts": [], "post_count": 0}
    if "--stdout" in sys.argv:
        print(json.dumps(out)); return
    (HERE / "google_live.json").write_text(json.dumps(out, indent=1))
    print("Wrote google_live.json:", len(out["ga4"]["months"]), "GA months,", len(out["searchConsole"]["months"]), "GSC months")
    if "--merge" in sys.argv:
        d = json.loads((HERE / "data.json").read_text())
        d.setdefault("live", {})["google"] = out
        (HERE / "data.json").write_text(json.dumps(d, indent=1))
        print("Merged into data.json under live.google")

if __name__ == "__main__":
    main()
