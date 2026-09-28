#!/usr/bin/env python3
"""Archive check for citations the pipeline could not verify or read.

For each dropped citation in the payload (payload['dropped_citations'], runs
from 2026-09-28 on) and, optionally, each bot-walled page in a page-check file,
ask the Wayback Machine CDX index whether the URL was ever captured with a 200. A snapshot means
the page existed at some point (dead link or wall, not an invention). No
snapshot is CONSISTENT WITH a fabricated URL but does not prove it: the
archive does not capture everything. Report it that way.

Usage:
    python3 archive_check.py <payload.json> <out.json> [--checks page_checks.json]

Free (public availability API); polite concurrency; read-only.
"""
import argparse, json, time, collections
from concurrent.futures import ThreadPoolExecutor
import requests

CDX = "https://web.archive.org/cdx/search/cdx"
AVAIL = "https://archive.org/wayback/available"   # fallback only: gives false negatives


def lookup(url):
    """Earliest successful capture via the CDX index (reliable), falling back to
    the availability API. The availability API alone reported the Fragrantica
    and NYT homepages as never archived (2026-09-28), so it is not trusted for a
    negative. Returns archived True/False, or None when both lookups failed."""
    hdr = {"User-Agent": "innate-c3-audit/1.0 (archive capture check)"}
    err = None
    for attempt in range(3):
        try:
            r = requests.get(CDX, params={"url": url, "output": "json", "limit": 1,
                                          "filter": "statuscode:200", "fl": "timestamp,original"},
                             timeout=25, headers=hdr)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1)); continue
            if r.status_code == 200:
                rows = r.json() if r.text.strip() else []
                if len(rows) > 1:
                    ts, orig = rows[1][0], rows[1][1]
                    return {"archived": True, "timestamp": ts,
                            "snapshot_url": f"https://web.archive.org/web/{ts}/{orig}", "via": "cdx"}
                return {"archived": False, "via": "cdx"}
            err = f"cdx HTTP {r.status_code}"
        except Exception as e:
            err = type(e).__name__
        time.sleep(2)
    try:
        r = requests.get(AVAIL, params={"url": url}, timeout=20, headers=hdr)
        snap = ((r.json() or {}).get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available"):
            return {"archived": True, "timestamp": snap.get("timestamp"), "snapshot_url": snap.get("url"), "via": "availability"}
    except Exception:
        pass
    return {"archived": None, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out")
    ap.add_argument("--checks", help="page-check JSON {url:{status:...}} to include bot-walled pages")
    ap.add_argument("--workers", type=int, default=3)   # CDX rate-limits aggressively
    a = ap.parse_args()
    p = json.load(open(a.payload))
    targets = {}
    for d in (p.get("dropped_citations") or []):
        if d.get("url"):
            targets.setdefault(d["url"], {"source": "dropped", "reason": d.get("reason"), "llm": d.get("llm")})
    if a.checks:
        for u, v in json.load(open(a.checks)).items():
            if isinstance(v, dict) and v.get("status") in ("blocked", "unchecked"):
                targets.setdefault(u, {"source": "blocked", "reason": v.get("status")})
    urls = sorted(targets)
    print(f"archive check: {len(urls)} URLs ({sum(1 for t in targets.values() if t['source']=='dropped')} dropped, "
          f"{sum(1 for t in targets.values() if t['source']=='blocked')} blocked)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(zip(urls, ex.map(lookup, urls)))
    out = {u: {**targets[u], **res[u]} for u in urls}
    json.dump(out, open(a.out, "w"), indent=1)
    by = collections.Counter((v["source"], v.get("archived")) for v in out.values())
    for src in ("dropped", "blocked"):
        yes, no, err = by.get((src, True), 0), by.get((src, False), 0), by.get((src, None), 0)
        if yes or no or err:
            print(f"  {src:8} archived {yes} | no snapshot {no} | lookup failed {err}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
