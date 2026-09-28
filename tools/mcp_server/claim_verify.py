#!/usr/bin/env python3
"""Claim-versus-page verification (#4a).

For each claim-to-source pair the agents' grounding metadata recorded
(answer row grounding.supports, runs from 2026-09-28 on), fetch the source
page once with the platform fetcher and ask Claude Haiku whether the page
supports the claim, with a verbatim evidence quote. Output per pair:
supported | partially_supported | not_supported | page_unrelated |
page_unavailable (fetch refused; no judge call).

Usage:
    python3 claim_verify.py <payload.json> <out_prefix> [--max 150] [--scope brand|all]

Writes <out_prefix>.json and <out_prefix>.csv (15_claim_verification.csv
columns). --scope brand (default) takes claims naming the brand or a tracked
competitor first, then fills to --max with the rest. Cost at list price:
about $1 per 150 claims (Haiku, ~6k tokens of page text per claim).
"""
import argparse, csv, json, os, re, sys, collections
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("FLASK_SECRET_KEY", "offline")
import app as A  # noqa: E402

JUDGE = os.environ.get("CLAIM_JUDGE_MODEL", A.CLAUDE_HAIKU)
SYSTEM = ("You check whether a web page supports a specific claim. Be strict: 'supported' only if the page "
          "states the claim's substance; 'partially_supported' if it supports part of it; 'not_supported' if the "
          "page is on topic but does not support it or contradicts it; 'page_unrelated' if the page is about "
          "something else. Quote evidence verbatim from the page (max 30 words) or leave it empty. "
          "Reply with JSON only: {\"verdict\": ..., \"evidence_quote\": ..., \"note\": <max 20 words>}.")


def window(text, claim, size=6000):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]{3,}", claim)][:12]
    if not text:
        return ""
    if len(text) <= size:
        return text
    best, best_i = -1, 0
    low = text.lower()
    for i in range(0, len(text) - size + 1, max(1, size // 4)):
        seg = low[i:i + size]
        sc = sum(seg.count(w.lower()) for w in words)
        if sc > best:
            best, best_i = sc, i
    return text[best_i:best_i + size]


def judge(claim, page_text):
    msg = A._claude_msg(timeout=60.0, model=JUDGE, max_tokens=300, system=SYSTEM,
                        messages=[{"role": "user", "content": f"CLAIM:\n{claim}\n\nPAGE TEXT:\n{page_text}"}])
    raw = "".join(getattr(b, "text", "") for b in (msg.content or []))
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    v = d.get("verdict") if d.get("verdict") in ("supported", "partially_supported", "not_supported", "page_unrelated") else "unparseable"
    u = getattr(msg, "usage", None)
    return {"verdict": v, "evidence_quote": (d.get("evidence_quote") or "")[:300], "note": (d.get("note") or "")[:200],
            "in_tok": getattr(u, "input_tokens", 0) or 0, "out_tok": getattr(u, "output_tokens", 0) or 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out_prefix")
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--scope", choices=["brand", "all"], default="brand")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    p = json.load(open(a.payload))
    brand = p.get("brand") or ""
    names = [brand] + [c.get("name") for c in (p.get("competitors") or []) if isinstance(c, dict) and c.get("name")]
    name_re = re.compile("|".join(re.escape(n) for n in names if n), re.I) if any(names) else None
    pairs, seen = [], set()
    for r in p.get("all_responses") or []:
        for sp in ((r.get("grounding") or {}).get("supports") or []):
            claim = (sp.get("claim") or sp.get("text") or "").strip()
            if len(claim) < 20:
                continue
            for u in sp.get("urls") or []:
                if not u or "vertexaisearch" in u or (claim, u) in seen:
                    continue
                seen.add((claim, u))
                pairs.append({"llm": r.get("llm"), "prompt": r.get("prompt"), "claim": claim, "source_url": u,
                              "names_entity": bool(name_re and name_re.search(claim))})
    if not pairs:
        sys.exit("no claim/source pairs: payload has no grounding.supports (runs before 2026-09-28 carry none)")
    if a.scope == "brand":
        pairs.sort(key=lambda x: not x["names_entity"])
    pairs = pairs[:a.max]
    urls = sorted({x["source_url"] for x in pairs})
    print(f"verifying {len(pairs)} claim/source pairs over {len(urls)} pages (judge {JUDGE})", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        pages = dict(zip(urls, ex.map(lambda u: A._scrape_cited_page(u, [brand], keep_text=True), urls)))
    def one(x):
        pg = pages.get(x["source_url"]) or {}
        if pg.get("status") != "ok" or not pg.get("text"):
            return {**x, "page_status": pg.get("status") or "error", "verdict": "page_unavailable",
                    "evidence_quote": "", "note": "fetch refused or empty", "in_tok": 0, "out_tok": 0}
        try:
            return {**x, "page_status": "ok", **judge(x["claim"], window(pg["text"], x["claim"]))}
        except Exception as e:
            return {**x, "page_status": "ok", "verdict": "judge_error", "evidence_quote": "",
                    "note": type(e).__name__, "in_tok": 0, "out_tok": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, pairs))
    json.dump(res, open(a.out_prefix + ".json", "w"), indent=1)
    cols = ["llm", "prompt", "claim", "names_entity", "source_url", "page_status", "verdict",
            "evidence_quote", "note"]
    with open(a.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(res)
    vc = collections.Counter(r["verdict"] for r in res)
    tin, tout = sum(r["in_tok"] for r in res), sum(r["out_tok"] for r in res)
    judged = [r for r in res if r["verdict"] in ("supported", "partially_supported", "not_supported", "page_unrelated")]
    print("verdicts:", dict(vc))
    if judged:
        print(f"supported (incl. partial) among pages that loaded: "
              f"{sum(1 for r in judged if r['verdict'] in ('supported','partially_supported'))}/{len(judged)}")
    by = collections.defaultdict(collections.Counter)
    for r in judged:
        by[r["llm"]][r["verdict"]] += 1
    for k, v in sorted(by.items()):
        print(f"  {k:10} {dict(v)}")
    print(f"judge tokens in {tin:,} out {tout:,} ~ ${tin/1e6*1.0 + tout/1e6*5.0:.2f} at Haiku list price")
    print(f"wrote {a.out_prefix}.json / .csv")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # after flushing: wedged fetch threads must not hold the interpreter open


if __name__ == "__main__":
    main()
