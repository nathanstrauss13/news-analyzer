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

def main():
    if "--auth" in sys.argv:
        creds(interactive=True); print("Authorised."); return
    c = creds()
    out = {"pulled": dt.datetime.now().isoformat(timespec="minutes"), "ga4": ga4(c), "searchConsole": gsc(c)}
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
