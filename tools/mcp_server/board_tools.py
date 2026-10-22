"""Signal MCP: board tools for a client's Signal Board (first: Xsight, e9ea20b741).

Served ONLY through the hosted connector (app.py /mcp/<key>) for a profile that
lists boards. Facts, never conclusions, the same contract as signal_mcp.py:
every response names the board, the source, and the as-of time, the way the
board's own badges do (Live = the daily Google pull; Reported = figures the
client or the operator supplied in the board's data file).

Two stores, both already on the server:
  live   the BoardLiveFeed row the daily pull writes (live.json): searchConsole,
         ga4, blog, pace
  data   the structured data file uploaded with the board (SharedResult
         payload['board_data']): linkedin, events, earned, uniqueCoverage,
         pickup, aiVisibility, messageTesting, and whatever else the board
         carries, with payload['board_data_as_of']

Allowlist by construction: a tool reads only boards named in the profile's
"boards" list; any other slug is "not in this client's board list", and a
slug that is not configured as a board at all gives the same answer. Nothing
here reads the Google token, the board password hash, operator routes, or
another client's rows: the loaders the app installs return only the two
stores above.
"""
import threading, time, json, os, re

_CFG = threading.local()          # .profile set per request by the hosted dispatcher
LOADERS = {"live": None, "data": None}   # installed by app.py: fn(slug) -> dict|None, fn(slug) -> (dict|None, as_of|None)
REQUESTS_LOG = os.environ.get("SIGNAL_MCP_REQUESTS_LOG") or os.path.join(os.path.expanduser("~"), ".signal_mcp", "board_refresh_requests.log")


def _profile():
    p = getattr(_CFG, "profile", None)
    if not p:
        raise ValueError("no client profile selected")
    return p


def _boards():
    return [b for b in (_profile().get("boards") or []) if isinstance(b, dict) and b.get("slug")]


def _board(board=None):
    """Resolve the board by slug or label (case-insensitive). A profile with one
    board always resolves to it, whatever was passed."""
    bs = _boards()
    if not bs:
        raise ValueError("this client's profile lists no boards")
    if not board or len(bs) == 1:
        return bs[0]
    want = str(board).strip().lower()
    for b in bs:
        if want in (b["slug"].lower(), (b.get("label") or "").lower()):
            return b
    raise ValueError("board not in this client's board list; boards available: " + ", ".join(f"{b['slug']} ({b.get('label')})" for b in bs))


def _live(slug):
    fn = LOADERS.get("live")
    return fn(slug) if fn else None


def _data(slug):
    fn = LOADERS.get("data")
    return fn(slug) if fn else (None, None)


CONTRACT = ("Report the board's figures and windows as given; do not project, extrapolate, or estimate "
            "full-month totals; facts only, no interpretation.")


def _convention():
    """The same build block the audit tools carry (server_build, stale_server
    when applicable), so a client sees one provenance stamp on every response."""
    try:
        import signal_mcp as _sm
        return _sm._build_block()
    except Exception as e:
        return {"server_build": "unavailable", "build_check": f"unavailable: {type(e).__name__}"}


def _hdr(b, source, as_of, badge, note=None):
    out = {"board": b["slug"], "board_label": b.get("label"), "source": source, "as_of": as_of, "badge": badge,
           "contract": CONTRACT, "convention": _convention()}
    if note:
        out["note"] = note
    return out


def _section_meta(data, name, file_as_of):
    """as_of, badge, note, source for a section: _meta.sections.<name> first,
    then the section's own asOf/source fields (the board's cards carry them),
    then the data file's upload time."""
    meta = ((data.get("_meta") or {}).get("sections") or {}).get(name) or {}
    sec = data.get(name)
    own = sec if isinstance(sec, dict) else {}
    as_of = meta.get("asOf") or meta.get("as_of") or own.get("asOf") or own.get("as_of") or file_as_of
    badge = meta.get("badge") or ("Live" if name in ("searchConsole", "analytics", "blog", "pace") else "Reported")
    source = meta.get("source") or own.get("source")
    note = meta.get("note") or own.get("note")
    return as_of, badge, note, source


def _section(b, name):
    data, as_of = _data(b["slug"])
    if not data or name not in data:
        raise ValueError(f"the board's data file has no '{name}' section")
    if name.startswith("_") or not isinstance(data[name], (dict, list)):
        raise ValueError(f"'{name}' is not a section")
    a, badge, note, source = _section_meta(data, name, as_of)
    return data[name], a, badge, (f"source: {source}; {note}" if source and note else (f"source: {source}" if source else note))


def _in_range(s, since, until):
    s = (s or "")[:10]
    return (not since or s >= since) and (not until or s <= until)


def _as_dicts(rows, columns=None):
    """Rows as dicts. Positional rows (lists) are mapped through the section's
    "columns" array when it is present; dict rows pass through."""
    out = []
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, (list, tuple)) and columns:
            out.append({c: (r[i] if i < len(r) else None) for i, c in enumerate(columns)})
    return out


_ISO = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


def _filter_rows(rows, columns=None, **kw):
    """Light, field-tolerant filters: a filter applies only where the row has the
    field. since/until apply only to ISO dates; a free-text date ("Oct 9",
    "since Aug 18") never excludes a row."""
    out = []
    for r in _as_dicts(rows, columns):
        ok = True
        for k, v in kw.items():
            if v in (None, ""):
                continue
            if k in ("since", "until"):
                d = str(r.get("date") or r.get("published") or r.get("month") or "")
                if _ISO.match(d) and not _in_range(d, kw.get("since"), kw.get("until")):
                    ok = False
            elif k in r and str(r.get(k, "")).lower() != str(v).lower() and str(v).lower() not in str(r.get(k, "")).lower():
                ok = False
        if ok:
            out.append(r)
    return out


# ── tools ────────────────────────────────────────────────────────────────

def list_board_sources(board: str = "") -> dict:
    """List the board's data sources with each source's as-of time and badge:
    Live (daily Google pull: Search Console, GA4, blog, pace) and Reported
    (figures supplied with the board: LinkedIn, events, earned coverage,
    pickup, AI visibility, message testing). Start here."""
    b = _board(board)
    live = _live(b["slug"]) or {}
    data, data_as_of = _data(b["slug"])
    sources = []
    for k, label in (("searchConsole", "Google Search Console"), ("ga4", "Google Analytics 4"), ("blog", "Blog posts and per-post metrics"), ("pace", "Month-to-date pace")):
        if k in live:
            blk = live.get(k) or {}
            sources.append({"source": k, "label": label, "badge": "Live", "as_of": live.get("pulled"),
                            **({"post_list_captured": blk.get("captured"), "post_list_source": blk.get("source")} if k == "blog" else {})})
    meta = ((data or {}).get("_meta") or {}).get("sections") or {}
    scalars = {}
    for k, v in (data or {}).items():
        if k.startswith("_"):
            continue
        if not isinstance(v, (dict, list)):
            scalars[k] = v                      # liveNote, period, updated: board-level notes, not sections
            continue
        a, badge, note, src = _section_meta(data, k, data_as_of)
        m = meta.get(k) or {}
        sources.append({"source": k, "label": m.get("label") or k, "badge": badge, "as_of": a,
                        **({"data_source": src} if src else {}), **({"note": note} if note else {})})
    return {**_hdr(b, "board", live.get("pulled") or data_as_of, "mixed"), "sources": sources,
            "board_notes": scalars, "data_file_as_of": data_as_of,
            "boards_in_profile": [{"slug": x["slug"], "label": x.get("label")} for x in _boards()]}


def get_search(board: str = "", months_from: str = "", months_to: str = "", top: int = 20) -> dict:
    """Google Search Console for the board's site: monthly clicks, impressions,
    CTR and position (last 12 months plus the partial current month), top
    queries and top pages over the trailing 90 days, and the branded share of
    clicks. months_from/months_to filter the monthly series by 'YYYY-MM'. Report figures and windows as given; do not project or estimate full-month totals."""
    b = _board(board)
    live = _live(b["slug"]) or {}
    sc = live.get("searchConsole")
    if not sc:
        raise ValueError("no Search Console feed stored for this board yet")
    months = [m for m in sc.get("months", []) if _in_range(m.get("start"), months_from, months_to)]
    top = max(1, min(int(top or 20), 200))
    return {**_hdr(b, "searchConsole", live.get("pulled"), "Live"),
            "months": months, "months_note": "calendar months; an entry with partial=true is the current month to date, and its end and data_through are Search Console's last complete day (two days before the pull), not the pull date",
            "top_queries_90d": (sc.get("topQueries90d") or [])[:top], "top_pages_90d": (sc.get("topPages90d") or [])[:top],
            "window_90d": "trailing 90 days ending two days before the pull (Search Console lag)",
            "branded_share_90d": sc.get("brandedShare90d"),
            "branded_share_90d_basis": "share of clicks among the top 200 queries by clicks over the trailing 90 days, where a branded query contains 'xsight'; not a share of all clicks"}


def get_site_traffic(board: str = "", months_from: str = "", months_to: str = "", top: int = 20) -> dict:
    """Google Analytics 4 for the board's site: monthly sessions by channel
    group (last 12 months plus the partial current month), top sources and
    top pages over the trailing 90 days, and sessions referred by AI
    assistants. months_from/months_to filter the monthly series by 'YYYY-MM'. Report figures and windows as given; do not project or estimate full-month totals."""
    b = _board(board)
    live = _live(b["slug"]) or {}
    ga = live.get("ga4")
    if not ga:
        raise ValueError("no GA4 feed stored for this board yet")
    months = [m for m in ga.get("months", []) if _in_range(m.get("start"), months_from, months_to)]
    top = max(1, min(int(top or 20), 200))
    srcs = ga.get("topSources90d") or {}
    return {**_hdr(b, "ga4", live.get("pulled"), "Live"),
            "months": months, "months_note": "calendar months; an entry with partial=true is the current month to date, and its end and data_through are the day before the pull", "top_sources_90d": dict(list(srcs.items())[:top]),
            "ai_assistant_referrals_90d": ga.get("aiAssistants90d") or {}, "top_pages_90d": (ga.get("topPages90d") or [])[:top]}


def get_blog(board: str = "", sort: str = "published", limit: int = 50, published_since: str = "", min_clicks_12m: int = 0) -> dict:
    """Blog posts with per-post Search Console clicks and impressions and GA4
    page views over 12 months and 90 days, plus posts per month and totals.
    sort: published | clicks_12m | views_12m | impressions_12m. Posts the
    feed lacks but Google reports carry from_google_only and no date. Report figures and windows as given; do not project or estimate full-month totals."""
    b = _board(board)
    live = _live(b["slug"]) or {}
    blog = live.get("blog")
    if not blog:
        raise ValueError("no blog feed stored for this board yet")
    posts = [p for p in blog.get("posts", []) if (not published_since or (p.get("published") or "") >= published_since) and p.get("clicks_12m", 0) >= int(min_clicks_12m or 0)]
    key = sort if sort in ("published", "clicks_12m", "views_12m", "impressions_12m", "clicks_90d", "views_90d") else "published"
    posts.sort(key=lambda p: (p.get(key) or ("" if key == "published" else 0)), reverse=True)
    limit = max(1, min(int(limit or 50), 500))
    note = None
    if blog.get("source") in ("prior", "seed"):
        note = "post list carried from an earlier capture (the publisher's feed refused the server); metrics are current"
    elif blog.get("source") == "none":
        note = "no dated post list available; posts come from Google's page data only"
    return {**_hdr(b, "blog", live.get("pulled"), "Live", note), "post_list_source": blog.get("source"), "post_list_captured": blog.get("captured"),
            "windows": blog.get("windows"), "totals": blog.get("totals"), "posts_by_month": blog.get("posts_by_month"),
            "dated_source": blog.get("dated_source"), "post_count": len(posts), "posts": posts[:limit]}


def get_pace(board: str = "") -> dict:
    """Month-to-date against the same day numbers of the prior month: Search
    Console clicks and impressions, GA4 sessions, blog posts. A source whose
    current month has no complete day yet is null. Report figures and windows as given; do not project or estimate full-month totals."""
    b = _board(board)
    live = _live(b["slug"]) or {}
    pace = live.get("pace")
    if pace is None:
        raise ValueError("no pace block stored for this board yet")
    return {**_hdr(b, "pace", live.get("pulled"), "Live"), **pace}


def _reported(board, section, tool_doc_name, **filters):
    b = _board(board)
    rows, as_of, badge, note = _section(b, section)
    if isinstance(rows, dict) and any(isinstance(v, list) for v in rows.values()):
        cols = rows.get("columns") if isinstance(rows.get("columns"), list) else None
        out = {k: (_filter_rows(v, cols, **filters) if isinstance(v, list) and k != "columns" else v)
               for k, v in rows.items() if not (k in ("asOf", "as_of", "source", "note") )}
        if cols:
            out["columns"] = cols
    elif isinstance(rows, list):
        out = _filter_rows(rows, None, **filters)
    else:
        out = rows
    return {**_hdr(b, section, as_of, badge, note), section: out}


def get_linkedin(board: str = "", since: str = "", until: str = "") -> dict:
    """LinkedIn figures as the board reports them: monthly series (the current
    month partial) and the peer table. since/until filter dated rows ('YYYY-MM')."""
    return _reported(board, "linkedin", "get_linkedin", since=since, until=until)


def get_coverage(board: str = "", announcement: str = "", type: str = "", since: str = "", until: str = "") -> dict:
    """Earned coverage as the board reports it, one row per placement, with
    announcement, outlet, type and date where the board carries them. Filters
    match where the row has the field."""
    return _reported(board, "uniqueCoverage", "get_coverage", announcement=announcement, type=type, since=since, until=until)


def get_pickup(board: str = "", announcement: str = "", since: str = "", until: str = "") -> dict:
    """Release pickup and syndication as the board reports it, by announcement."""
    return _reported(board, "pickup", "get_pickup", announcement=announcement, since=since, until=until)


def get_interviews(board: str = "", status: str = "", since: str = "", until: str = "") -> dict:
    """Interviews and briefings as the board reports them, with status
    (for example scheduled, completed, published) where the board carries it."""
    return _reported(board, "earned", "get_interviews", status=status, since=since, until=until)


def get_events(board: str = "", since: str = "", until: str = "") -> dict:
    """Events as the board reports them."""
    return _reported(board, "events", "get_events", since=since, until=until)


def get_message_testing(board: str = "") -> dict:
    """Message testing as the board's card shows it: verdict counts per message.
    Only the card's figures are served; the underlying runs are not."""
    return _reported(board, "messageTesting", "get_message_testing")


def get_ai_visibility(board: str = "") -> dict:
    """The AI visibility figures the board reports (the baseline card). The
    full audit behind the card is available through the audit tools
    (list_audits, get_audit, query_citations) for the audits in this profile."""
    return _reported(board, "aiVisibility", "get_ai_visibility")


def get_board_section(board: str = "", section: str = "") -> dict:
    """Any other section of the board's data file by name (see
    list_board_sources), returned as the board carries it."""
    if not section or section.startswith("_"):
        raise ValueError("name a section from list_board_sources")
    return _reported(board, section, "get_board_section")


def request_refresh(board: str = "", reason: str = "") -> dict:
    """File a request with the operator to refresh the board's Live sources or
    its reported figures. This tool never runs anything itself; the operator
    reviews the request and the board updates on its next publish or pull."""
    b = _board(board)
    try:
        os.makedirs(os.path.dirname(REQUESTS_LOG), exist_ok=True)
        with open(REQUESTS_LOG, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "client": _profile().get("client"), "board": b["slug"], "reason": (reason or "")[:500]}) + "\n")
    except Exception:
        pass
    return {**_hdr(b, "operator", None, "n/a"), "status": "requested",
            "what_happens_next": "The operator reviews the request. Live sources refresh daily on their own; reported figures update when the board is next published."}


_S = {"type": "string"}
_I = {"type": "integer"}
TOOLS_REG = {
    "list_board_sources": (list_board_sources, {"type": "object", "properties": {"board": _S}, "required": []}),
    "get_search": (get_search, {"type": "object", "properties": {"board": _S, "months_from": _S, "months_to": _S, "top": _I}, "required": []}),
    "get_site_traffic": (get_site_traffic, {"type": "object", "properties": {"board": _S, "months_from": _S, "months_to": _S, "top": _I}, "required": []}),
    "get_blog": (get_blog, {"type": "object", "properties": {"board": _S, "sort": {"type": "string", "enum": ["published", "clicks_12m", "views_12m", "impressions_12m", "clicks_90d", "views_90d"]}, "limit": _I, "published_since": _S, "min_clicks_12m": _I}, "required": []}),
    "get_pace": (get_pace, {"type": "object", "properties": {"board": _S}, "required": []}),
    "get_linkedin": (get_linkedin, {"type": "object", "properties": {"board": _S, "since": _S, "until": _S}, "required": []}),
    "get_coverage": (get_coverage, {"type": "object", "properties": {"board": _S, "announcement": _S, "type": _S, "since": _S, "until": _S}, "required": []}),
    "get_pickup": (get_pickup, {"type": "object", "properties": {"board": _S, "announcement": _S, "since": _S, "until": _S}, "required": []}),
    "get_interviews": (get_interviews, {"type": "object", "properties": {"board": _S, "status": _S, "since": _S, "until": _S}, "required": []}),
    "get_events": (get_events, {"type": "object", "properties": {"board": _S, "since": _S, "until": _S}, "required": []}),
    "get_message_testing": (get_message_testing, {"type": "object", "properties": {"board": _S}, "required": []}),
    "get_ai_visibility": (get_ai_visibility, {"type": "object", "properties": {"board": _S}, "required": []}),
    "get_board_section": (get_board_section, {"type": "object", "properties": {"board": _S, "section": _S}, "required": ["section"]}),
    "request_refresh": (request_refresh, {"type": "object", "properties": {"board": _S, "reason": _S}, "required": []}),
}
TOOL_DESCRIPTIONS = {n: (f.__doc__ or "").strip() for n, (f, _s) in TOOLS_REG.items()}
