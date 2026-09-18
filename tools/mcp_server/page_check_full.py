#!/usr/bin/env python3
"""Full-corpus page check for one audit payload — the offline pass that makes
"every cited page fetched" literally true.

The audit-time check (_citation_checks_for_report) is deadline-bounded
(45s, first 250 URLs in first-seen order), so a big run leaves most pages
unchecked. This script fetches EVERY distinct cited URL with the platform's
own fetcher (_scrape_cited_page: direct GET, then reader-proxy fallback for
bot-walled publishers) and writes a supplement in the evidence-file schema the
MCP server consumes with evidence_mode="replace":

    {url: {"status": "ok"|"blocked", "brand_count": int, "title": str,
           "checked_date": "YYYY-MM-DD"}}

Usage:
    python3 page_check_full.py <payload.json> <out_checks.json> [--workers 40] [--timeout 10]

Prints a summary with denominators. Single-epoch by construction: one
checked_date for the whole corpus. Read-only HTTP; no LLM spend.
"""
import argparse, json, os, sys, time, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root -> import app
os.environ.setdefault("ANTHROPIC_API_KEY", "offline"); os.environ.setdefault("FLASK_SECRET_KEY", "offline")
import app as A   # noqa: E402  (heavy import; prints DB-migration noise on a fresh sqlite — harmless)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--recheck", metavar="PRIOR_CHECKS_JSON",
                    help="re-fetch only URLs whose prior status is in --only-status, merge into prior, write to <out>")
    ap.add_argument("--only-status", default="blocked,unchecked")
    ap.add_argument("--deadline", type=float, default=None, help="override whole-batch deadline seconds")
    a = ap.parse_args()

    d = json.load(open(a.payload))
    rows = d.get("all_responses") or []
    names = [d.get("brand") or ""] + [x for x in (d.get("brand_aliases") or []) if isinstance(x, str)]
    names = [n for n in names if n.strip()]
    urls = sorted({c.get("url") for r in rows for c in (r.get("citations") or [])
                   if isinstance(c, dict) and c.get("url", "").startswith("http")})
    today = datetime.date.today().isoformat()
    prior = {}
    if a.recheck:
        prior = json.load(open(a.recheck))
        want = set(a.only_status.split(","))
        urls = [u for u in urls if prior.get(u, {}).get("status") in want]
        print(f"recheck mode: {len(urls)} URLs with prior status in {sorted(want)}", flush=True)
    print(f"payload {a.payload} | slug {d.get('slug')} | brand forms {names} | distinct URLs {len(urls)}", flush=True)

    out, t0 = {}, time.time()
    # whole-batch deadline, same doctrine as _resolve_and_verify_urls: never hang
    waves = -(-len(urls) // max(1, a.workers))
    deadline = a.deadline or max(120.0, waves * (a.timeout + 10) * 1.5 + 60)
    print(f"whole-batch deadline {deadline:.0f}s ({waves} waves x {a.workers} workers)", flush=True)
    ex = ThreadPoolExecutor(max_workers=a.workers)
    try:
        futs = {ex.submit(A._scrape_cited_page, u, names, a.timeout): u for u in urls}
        done = 0
        try:
            for f in as_completed(futs, timeout=deadline):
                u = futs[f]
                try:
                    r = f.result()
                except Exception:
                    r = {"status": "error", "title": "", "counts": {}}
                status = "ok" if r.get("status") == "ok" else "blocked"
                out[u] = {"status": status,
                          "brand_count": int(sum((r.get("counts") or {}).values())),
                          "title": (r.get("title") or "")[:300],
                          "checked_date": today}
                done += 1
                if done % 50 == 0 or done == len(urls):
                    print(f"  {done}/{len(urls)}  ok={sum(1 for v in out.values() if v['status']=='ok')}  {time.time()-t0:.0f}s", flush=True)
        except FuturesTimeoutError:
            pass
        # Deadline leftovers were never fetched: say so. Marking them "blocked"
        # would claim a bot-wall we did not observe.
        unchecked = 0
        for f, u in futs.items():
            if u not in out:
                out[u] = {"status": "unchecked", "brand_count": 0, "title": "", "checked_date": today}
                unchecked += 1
    finally:
        try: ex.shutdown(wait=False, cancel_futures=True)
        except TypeError: ex.shutdown(wait=False)

    if prior:
        merged = dict(prior); merged.update(out); out = merged
    json.dump(out, open(a.out, "w"), indent=1)
    ok = sum(1 for v in out.values() if v["status"] == "ok")
    unc = sum(1 for v in out.values() if v["status"] == "unchecked")
    bm = sum(1 for v in out.values() if v["brand_count"])
    print(f"\nDONE {len(out)} URLs in file ({len(urls)} fetched this run) on {today} | readable {ok} ({100*ok/max(1,len(out)):.0f}%) | "
          f"bot-walled {len(out)-ok-unc} | UNCHECKED (deadline) {unc} | pages naming the brand {bm} | {time.time()-t0:.0f}s | wrote {a.out}", flush=True)
    os._exit(0)   # don't let wedged pool threads hold the interpreter open


if __name__ == "__main__":
    main()
