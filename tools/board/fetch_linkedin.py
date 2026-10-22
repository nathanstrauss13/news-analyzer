#!/usr/bin/env python3
"""LinkedIn Page statistics for a Signal Board (Community Management API).

Pure fetch-and-shape module; the app passes a bearer token and an
organization URN and stores the result under live.json["linkedin"]. Nothing
here logs or returns the token.

Endpoints (LinkedIn Marketing API, versioned; docs read Oct 22, 2026):
  GET /rest/networkSizes/{org}?edgeType=COMPANY_FOLLOWED_BY_MEMBER  -> firstDegreeSize (total followers)
  GET /rest/organizationalEntityFollowerStatistics?q=organizationalEntity&timeIntervals=(...)
        -> followerGains.organicFollowerGain / paidFollowerGain per interval; data 12 months back to 2 days ago
  GET /rest/organizationPageStatistics?q=organization&timeIntervals=(...)
        -> totalPageStatistics.views.allPageViews.pageViews / uniquePageViews per interval
  GET /rest/organizationalEntityShareStatistics?q=organizationalEntity&timeIntervals=(...)
        -> totalShareStatistics: impressionCount, uniqueImpressionsCount, clickCount, likeCount,
           commentCount, shareCount, engagement; ORGANIC ONLY (sponsored needs Ad Analytics)
All need a member with the ADMINISTRATOR role on the page (per the docs) and
the r_organization_admin or rw_organization_admin scope.

Output shape (every date ISO; months oldest first, last entry partial):
  {"fetched": iso, "api_version": "202610", "organization": "urn:li:organization:N",
   "total_followers": int|null, "data_through": {"followers": d, "page_views": d, "posts": d},
   "months": [{"month": "2025-11", "label": "Nov 25", "start", "end", "partial": bool,
               "new_followers_organic", "new_followers_paid", "new_followers",
               "page_views", "unique_visitors",
               "impressions_organic", "unique_impressions_organic", "clicks_organic",
               "reactions_organic", "comments_organic", "reposts_organic", "engagement_rate_organic"}, ...],
   "pace": {"new_followers": {"current": {start, end, value}, "prior": {...}} | null, ...},
   "sponsored": null, "notes": [...]}
"""
import datetime as dt
import calendar
import requests

API = "https://api.linkedin.com/rest"
DEFAULT_VERSION = "202610"


def _hdr(token, version):
    return {"Authorization": f"Bearer {token}", "LinkedIn-Version": version,
            "X-Restli-Protocol-Version": "2.0.0", "Content-Type": "application/json"}


def _ms(d):
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _day(ms):
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).date()


class LinkedInError(Exception):
    """Shape-only error: HTTP status and LinkedIn's error code/message, never the token."""


def _get(url, token, version, params=None, timeout=40):
    r = requests.get(url, headers=_hdr(token, version), params=params, timeout=timeout)
    if r.status_code != 200:
        try:
            j = r.json(); msg = f"{j.get('code') or j.get('serviceErrorCode') or ''} {j.get('message') or ''}".strip()
        except Exception:
            msg = r.text[:160]
        raise LinkedInError(f"http {r.status_code} on {url.split('/rest/')[-1].split('?')[0]}: {msg[:200]}")
    return r.json()


def _interval(start, end_exclusive, gran):
    return f"(timeRange:(start:{_ms(start)},end:{_ms(end_exclusive)}),timeGranularityType:{gran})"


def total_followers(token, org, version):
    j = _get(f"{API}/networkSizes/{org}", token, version, params={"edgeType": "COMPANY_FOLLOWED_BY_MEMBER"})
    return int(j.get("firstDegreeSize")) if j.get("firstDegreeSize") is not None else None


def _series(token, org, version, endpoint, qname, start, end_exclusive, gran):
    url = f"{API}/{endpoint}?q={qname}&{qname}={org}&timeIntervals={_interval(start, end_exclusive, gran)}"
    j = _get(url, token, version)
    return j.get("elements") or []


def _bucket_month(elements, pick):
    """Sum per-day elements into calendar months: {YYYY-MM: {field: value}}."""
    out = {}
    for e in elements:
        tr = e.get("timeRange") or {}
        d = _day(tr.get("start", 0)); key = d.strftime("%Y-%m")
        acc = out.setdefault(key, {})
        for k, v in pick(e).items():
            if v is None: continue
            acc[k] = acc.get(k, 0) + v
    return out


def _pick_followers(e):
    g = e.get("followerGains") or {}
    return {"new_followers_organic": int(g.get("organicFollowerGain") or 0), "new_followers_paid": int(g.get("paidFollowerGain") or 0)}


def _pick_pages(e):
    v = ((e.get("totalPageStatistics") or {}).get("views") or {}).get("allPageViews") or {}
    return {"page_views": int(v.get("pageViews") or 0), "unique_visitors": int(v.get("uniquePageViews") or 0)}


def _pick_shares(e):
    s = e.get("totalShareStatistics") or {}
    return {"impressions_organic": int(s.get("impressionCount") or 0), "unique_impressions_organic": int(s.get("uniqueImpressionsCount") or s.get("uniqueImpressionsCounts") or 0),
            "clicks_organic": int(s.get("clickCount") or 0), "reactions_organic": int(s.get("likeCount") or 0),
            "comments_organic": int(s.get("commentCount") or 0), "reposts_organic": int(s.get("shareCount") or 0)}


def _mtd(last_complete, today):
    cur_start = today.replace(day=1)
    if last_complete < cur_start:
        return None
    py, pm = (cur_start.year - 1, 12) if cur_start.month == 1 else (cur_start.year, cur_start.month - 1)
    plen = calendar.monthrange(py, pm)[1]
    return {"current": (cur_start, last_complete), "prior": (dt.date(py, pm, 1), dt.date(py, pm, min(last_complete.day, plen)))}


def fetch(token, org, version=DEFAULT_VERSION, today=None):
    today = today or dt.date.today()
    notes = []
    first = (today.replace(day=1) - dt.timedelta(days=365)).replace(day=1)   # 12 full months back, 1st of that month
    # LinkedIn's own windows: follower stats run 12 months back to T-2; page and share stats to roughly T-1.
    thru_f = today - dt.timedelta(days=2); thru_p = today - dt.timedelta(days=1)
    daily_f = _series(token, org, version, "organizationalEntityFollowerStatistics", "organizationalEntity", first, thru_f + dt.timedelta(days=1), "DAY")
    daily_p = _series(token, org, version, "organizationPageStatistics", "organization", first, thru_p + dt.timedelta(days=1), "DAY")
    daily_s = _series(token, org, version, "organizationalEntityShareStatistics", "organizationalEntity", first, thru_p + dt.timedelta(days=1), "DAY")
    by_f, by_p, by_s = _bucket_month(daily_f, _pick_followers), _bucket_month(daily_p, _pick_pages), _bucket_month(daily_s, _pick_shares)
    # actual last day with data per source, so data_through is observed, not assumed
    last_f = max((_day((e.get("timeRange") or {}).get("start", 0)) for e in daily_f), default=None)
    last_p = max((_day((e.get("timeRange") or {}).get("start", 0)) for e in daily_p), default=None)
    months = []
    y, m = first.year, first.month
    while (y, m) <= (today.year, today.month):
        key = f"{y:04d}-{m:02d}"; start = dt.date(y, m, 1); last = dt.date(y, m, calendar.monthrange(y, m)[1])
        partial = (y, m) == (today.year, today.month)
        end = min(last, thru_p) if partial else last
        if partial and end < start:
            break
        row = {"month": key, "label": start.strftime("%b %y"), "start": start.isoformat(), "end": end.isoformat(), "partial": partial}
        f = by_f.get(key, {}); p = by_p.get(key, {}); s = by_s.get(key, {})
        row.update({"new_followers_organic": f.get("new_followers_organic"), "new_followers_paid": f.get("new_followers_paid"),
                    "new_followers": (f.get("new_followers_organic", 0) + f.get("new_followers_paid", 0)) if f else None,
                    "page_views": p.get("page_views"), "unique_visitors": p.get("unique_visitors"),
                    "impressions_organic": s.get("impressions_organic"), "unique_impressions_organic": s.get("unique_impressions_organic"),
                    "clicks_organic": s.get("clicks_organic"), "reactions_organic": s.get("reactions_organic"),
                    "comments_organic": s.get("comments_organic"), "reposts_organic": s.get("reposts_organic")})
        imp = s.get("impressions_organic") or 0
        inter = sum(s.get(k, 0) for k in ("clicks_organic", "reactions_organic", "comments_organic", "reposts_organic")) if s else None
        row["engagement_rate_organic"] = round(inter / imp, 4) if (s and imp) else None
        if partial:
            row["data_through"] = {"followers": (min(last_f, end) if last_f else None) and min(last_f, end).isoformat(),
                                   "page_views": (min(last_p, end) if last_p else None) and min(last_p, end).isoformat(), "posts": end.isoformat()}
        months.append(row)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    # pace: month to date against the same days of the prior month, from the daily series
    def pace_for(daily, pick, field, last_complete):
        w = _mtd(last_complete, today) if last_complete else None
        if not w:
            return None
        vals = {}
        for k, (s_, e_) in w.items():
            tot = 0
            for e in daily:
                d = _day((e.get("timeRange") or {}).get("start", 0))
                if s_ <= d <= e_:
                    tot += pick(e).get(field, 0)
            vals[k] = {"start": s_.isoformat(), "end": e_.isoformat(), "value": tot}
        return vals
    pace = {"new_followers": pace_for(daily_f, lambda e: {"v": _pick_followers(e)["new_followers_organic"] + _pick_followers(e)["new_followers_paid"]}, "v", last_f),
            "page_views": pace_for(daily_p, _pick_pages, "page_views", last_p),
            "impressions_organic": pace_for(daily_s, _pick_shares, "impressions_organic", last_p)}
    try:
        total = total_followers(token, org, version)
    except LinkedInError as e:
        total = None; notes.append(f"total followers unavailable: {e}")
    notes.append("impressions and engagement are organic only (organizationalEntityShareStatistics); sponsored activity needs the Ad Analytics API and is not included")
    notes.append("follower statistics lag two days; page views and post statistics about one day; the partial month's data_through records the last day each series returned")
    return {"fetched": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"), "api_version": version, "organization": org,
            "total_followers": total, "months": months, "pace": pace, "sponsored": None, "notes": notes}


def member_roles(token, version=DEFAULT_VERSION):
    """Organizations and roles the authorizing member holds (organizationAcls?q=roleAssignee)."""
    j = _get(f"{API}/organizationAcls?q=roleAssignee&state=APPROVED", token, version)
    return [{"organization": e.get("organization") or e.get("organizationTarget"), "role": e.get("role"), "state": e.get("state")} for e in (j.get("elements") or [])]
