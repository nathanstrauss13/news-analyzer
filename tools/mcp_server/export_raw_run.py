#!/usr/bin/env python3
"""Raw Run exporter v0 — payload in, the six-CSV client set out (+ page-check
files and a manifest). The single implementation of extraction conventions:
imports `conventions.py` (shared with the MCP server); never fork the rules.

Usage:
    python3 export_raw_run.py <payload.json> <out_dir> [host_classes.json]

Scope rules (split audits):
  * 01/02 carry every answer with a prompt_class column (branded/unbranded).
  * 03/04/06 and 05's mention figures are UNBRANDED-scoped per the split-audit
    contract; page-check columns cover the full citation set. Each file's
    scope is stated in MANIFEST.txt.
  * Universe rule: every root present in shipped rows gets a class;
    plumbing-link / name-mention roots carry citations_root = 0.
  * The manifest's vintage is the payload file's sha256 — computed, not typed.
"""
import csv, hashlib, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conventions import (URLISH_RE, forms_to_pattern, host_of, is_bare_suffix,
                         root_of, split_rows, type_of)


def wb(name):
    return re.compile(r"\b" + re.escape(name).replace(r"\'", "'?") + r"\b", re.I)


def export(payload_path, out_dir, host_classes_path=None):
    raw_bytes = open(payload_path, "rb").read()
    payload_sha = hashlib.sha256(raw_bytes).hexdigest()
    p = json.loads(raw_bytes)
    os.makedirs(out_dir, exist_ok=True)

    brand = p.get("brand") or ""
    aliases = [a for a in (p.get("brand_aliases") or []) if isinstance(a, str)]
    pat = forms_to_pattern([brand] + aliases)
    rows, unb, branded = split_rows(p)
    unb_prompts = {r.get("prompt") for r in unb}
    comp_names = [c.get("name") for c in (p.get("competitors") or [])
                  if isinstance(c, dict) and c.get("name")]
    comp_pats = {n: wb(n) for n in comp_names}

    cls_map = dict(p.get("domain_types") or {})
    if host_classes_path and os.path.exists(host_classes_path):
        extra = json.load(open(host_classes_path))
        cls_map.update({k: v for k, v in extra.items() if not k.startswith("_")})
    checks = p.get("citation_checks") or {}
    run_date = p.get("run_date") or ""

    # ---- per-answer + per-citation scaffolding ------------------------------
    answers, citations = [], []
    root_cites_all, root_cites_unb = Counter(), Counter()
    root_answers, root_llms = defaultdict(set), defaultdict(set)
    root_sample = {}
    unrootable = []
    for i, r in enumerate(rows):
        aid = f"{p.get('slug','run')[:6]}-{i:03d}"
        t = r.get("response") or ""
        first = pat.search(t)
        comp_present = []
        comp_first = {}
        for n, cp in comp_pats.items():
            m = cp.search(t)
            if m:
                comp_present.append(n)
                comp_first[n] = m.start()
        before_all = bool(first) and all(first.start() < v for v in comp_first.values())
        cits = [c for c in (r.get("citations") or []) if isinstance(c, dict) and c.get("url")]
        answers.append({
            "answer_id": aid, "llm": r.get("llm"),
            "prompt_class": "branded" if r.get("prompt") in branded else "unbranded",
            "prompt": r.get("prompt"), "grounded": r.get("grounded"),
            "error": r.get("error") or "", "response_chars": len(t),
            "brand_mentioned": bool(first),
            "brand_mention_count": len(pat.findall(t)),
            "brand_first_mention_pct": round(100 * first.start() / max(1, len(t)), 1) if first else "",
            "named_before_all_competitors": before_all if (first and comp_first) else "",
            "competitors_named": "; ".join(comp_present),
            "citation_count": len(cits),
            "response_full_text": t,
        })
        for c in cits:
            u = c["url"]
            root = root_of(u)
            if not root or "." not in root:
                unrootable.append(u)
                root = ""
            if root:
                root_cites_all[root] += 1
                if r.get("prompt") in unb_prompts:
                    root_cites_unb[root] += 1
                root_answers[root].add(aid)
                root_llms[root].add(r.get("llm"))
                root_sample.setdefault(root, u)
            citations.append({
                "answer_id": aid, "llm": r.get("llm"),
                "prompt_class": "branded" if r.get("prompt") in branded else "unbranded",
                "prompt": r.get("prompt"), "citation_url": u,
                "domain": host_of(u), "domain_root": root,
                "source_type": type_of(u, cls_map) or "",
            })

    # ---- page-check rollups (full citation set) -----------------------------
    page_by_root = defaultdict(lambda: [0, 0, 0, 0])   # checked, ok, mentioning, mentions
    per_url = []
    for u, v in checks.items():
        root = root_of(u)
        d = page_by_root[root]
        d[0] += 1
        ok = v.get("status") == "ok"
        if ok:
            d[1] += 1
            if v.get("brand_count"):
                d[2] += 1
                d[3] += v.get("brand_count") or 0
        per_url.append({"url": u, "domain_root": root, "fetch_status": v.get("status"),
                        "fetched_ok": ok, "mentions_brand": bool(ok and v.get("brand_count")),
                        "brand_mentions_on_page": v.get("brand_count") or 0,
                        "page_title": (v.get("title") or "")[:160],
                        "source_type": type_of(u, cls_map) or "",
                        "checked_date": v.get("checked_date") or run_date})
    for c in citations:
        pc = page_by_root.get(c["domain_root"])
        c.update({"domain_pages_checked": pc[0] if pc else 0,
                  "domain_pages_ok": pc[1] if pc else 0,
                  "domain_pages_mentioning_brand": pc[2] if pc else 0})

    # ---- universe rule: classes for every shipped root, zero-count if needed
    universe = set(root_cites_all) | {c["domain_root"] for c in citations if c["domain_root"]}

    # ---- 04 source index (unbranded scope, >=3 unbranded answers citing) ----
    unb_by_aid = {answers[i]["answer_id"]: r for i, r in enumerate(rows)
                  if r.get("prompt") in unb_prompts}
    brand_hits_unb = {aid for aid, r in unb_by_aid.items() if pat.search(r.get("response") or "")}
    baseline = len(brand_hits_unb) / max(1, len(unb_by_aid))
    comp_hits_unb = {n: {aid for aid, r in unb_by_aid.items() if cp.search(r.get("response") or "")}
                     for n, cp in comp_pats.items()}
    src_index = []
    root_unb_answers = defaultdict(set)
    for c in citations:
        if c["prompt_class"] == "unbranded" and c["domain_root"]:
            root_unb_answers[c["domain_root"]].add(c["answer_id"])
    for root, aids in root_unb_answers.items():
        if len(aids) < 3:
            continue
        at_src = len(aids & brand_hits_unb) / len(aids)
        top_n, top_at, top_base = "", 0.0, 0.0
        for n, hits in comp_hits_unb.items():
            v = len(aids & hits) / len(aids)
            if v > top_at:
                top_n, top_at = n, v
                top_base = len(hits) / max(1, len(unb_by_aid))
        pc = page_by_root.get(root, [0, 0, 0, 0])
        src_index.append({
            "domain": root, "answers_citing": len(aids),
            "brand_presence_baseline": round(baseline, 3),
            "brand_presence_at_source": round(at_src, 3),
            "differential": round(at_src - baseline, 3),
            "brand_mentions_at_source": len(aids & brand_hits_unb),
            "top_competitor": top_n,
            "top_competitor_presence_at_source": round(top_at, 3),
            "top_competitor_baseline": round(top_base, 3),
            "top_competitor_differential": round(top_at - top_base, 3),
            "pages_checked": pc[0], "pages_mentioning_brand": pc[2],
        })
    src_index.sort(key=lambda x: -x["answers_citing"])

    # ---- writers ------------------------------------------------------------
    def w(fn, fieldnames, rows_):
        with open(os.path.join(out_dir, fn), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows_)

    w("01_answers.csv",
      ["answer_id", "llm", "prompt_class", "prompt", "grounded", "error", "response_chars",
       "brand_mentioned", "brand_mention_count", "brand_first_mention_pct",
       "named_before_all_competitors", "competitors_named", "citation_count",
       "response_full_text"], answers)
    w("02_citations.csv",
      ["answer_id", "llm", "prompt_class", "prompt", "citation_url", "domain", "domain_root",
       "source_type", "domain_pages_checked", "domain_pages_ok",
       "domain_pages_mentioning_brand"], citations)
    w("03_sources.csv",
      ["domain", "citations_root_all", "citations_root_unbranded", "answers_citing",
       "llms_citing", "source_type", "pages_checked", "pages_ok",
       "pages_mentioning_brand", "brand_mentions_on_pages", "sample_url", "checked_date"],
      [{"domain": root, "citations_root_all": root_cites_all.get(root, 0),
        "citations_root_unbranded": root_cites_unb.get(root, 0),
        "answers_citing": len(root_answers.get(root, ())),
        "llms_citing": "; ".join(sorted(x for x in root_llms.get(root, ()) if x)),
        "source_type": cls_map.get(root) or "",
        "pages_checked": page_by_root.get(root, [0]*4)[0],
        "pages_ok": page_by_root.get(root, [0]*4)[1],
        "pages_mentioning_brand": page_by_root.get(root, [0]*4)[2],
        "brand_mentions_on_pages": page_by_root.get(root, [0]*4)[3],
        "sample_url": root_sample.get(root, ""), "checked_date": run_date}
       for root in sorted(universe, key=lambda r_: -root_cites_all.get(r_, 0))])
    w("04_source_index.csv",
      ["domain", "answers_citing", "brand_presence_baseline", "brand_presence_at_source",
       "differential", "brand_mentions_at_source", "top_competitor",
       "top_competitor_presence_at_source", "top_competitor_baseline",
       "top_competitor_differential", "pages_checked", "pages_mentioning_brand"], src_index)
    per_llm = defaultdict(lambda: [0, 0, 0])
    for aid, r in unb_by_aid.items():
        d = per_llm[r.get("llm")]
        d[1] += 1
        if aid in brand_hits_unb:
            d[0] += 1
        if r.get("grounded"):
            d[2] += 1
    w("05_llm_summary.csv",
      ["llm", "answers_mentioning_brand", "answers_total", "mention_rate", "grounded_answers"],
      [{"llm": k, "answers_mentioning_brand": v[0], "answers_total": v[1],
        "mention_rate": round(v[0] / max(1, v[1]), 3), "grounded_answers": v[2]}
       for k, v in sorted(per_llm.items())])
    w("06_competitors.csv",
      ["name", "answers_mentioning", "cited_by_llms"],
      [{"name": n, "answers_mentioning": len(hits),
        "cited_by_llms": "; ".join(sorted({unb_by_aid[a].get("llm") for a in hits}))}
       for n, hits in sorted(comp_hits_unb.items(), key=lambda kv: -len(kv[1]))])
    per_url.sort(key=lambda x: (-x["brand_mentions_on_page"], x["domain_root"]))
    w("07_page_checks_per_url.csv",
      ["url", "domain_root", "fetch_status", "fetched_ok", "mentions_brand",
       "brand_mentions_on_page", "page_title", "source_type", "checked_date"], per_url)

    # Machine-readable sidecar (biz-dev's builder asserts against row_counts:
    # a mismatch between manifest and files must fail the build loudly).
    import datetime as _dt
    row_counts = {"01_answers.csv": len(answers), "02_citations.csv": len(citations),
                  "03_sources.csv": len(universe), "04_source_index.csv": len(src_index),
                  "05_llm_summary.csv": len(per_llm), "06_competitors.csv": len(comp_hits_unb),
                  "07_page_checks_per_url.csv": len(per_url)}
    json.dump({
        "slug": p.get("slug"), "brand": brand,
        "run_date": run_date or None,
        "payload_sha256": payload_sha,
        "agents": sorted({r.get("llm") for r in rows if r.get("llm")}),
        "answers": {"total": len(rows), "branded": len(rows) - len(unb), "unbranded": len(unb)},
        "citation_rows": len(citations), "rootable_citations": sum(root_cites_all.values()),
        "unrootable_retained": len(unrootable),
        "brand_forms": [brand] + aliases,
        "row_counts": row_counts,
        "scopes": {
            "01_02_07": "all answers; rows carry prompt_class / per-URL grain",
            "03": "citations_root_all = all answers; citations_root_unbranded = unbranded only",
            "04_05_06": "unbranded scope per the split-audit contract",
            "page_check_columns": "full citation set",
            "citation_grain_rule": "distributions denominate on rootable citations, stated with the row count"},
    }, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)

    bare = sorted(r for r in universe if is_bare_suffix(r))
    if bare:
        print(f"WARNING: {len(bare)} computed root(s) look like bare public "
              f"suffixes (ccTLD missing from SECOND_LEVEL?): {bare} — unrelated "
              f"hosts may be merged; extend conventions.SECOND_LEVEL")

    manifest = f"""RAW RUN EXPORT MANIFEST
slug: {p.get('slug')}
brand: {brand}
payload_file: {os.path.basename(payload_path)}
payload_sha256: {payload_sha}
answers: {len(rows)} (branded {len(rows)-len(unb)} / unbranded {len(unb)})
citation_rows: {len(citations)} | rootable: {sum(root_cites_all.values())} | unrootable (retained in 02, root blank): {len(unrootable)}
page_checks: {len(checks)} URLs | classification: {'payload-native domain_types' if p.get('domain_types') else 'NONE in payload'}{' + host-classes supplement' if host_classes_path else ''}
brand_forms: {json.dumps([brand] + aliases)} (+ first distinctive token, apostrophe-tolerant)

SCOPES — read before quoting any number:
  01/02/07: ALL answers; rows carry prompt_class / per-URL grain.
  03: citations_root_all = all answers; citations_root_unbranded = unbranded only. Both labeled.
  04/05/06: UNBRANDED scope per the split-audit contract.
  Page-check columns everywhere: full citation set.
  Universe rule: every shipped root has a class; plumbing/name-mention roots carry 0 counts.
  Citation-grain distributions denominate on rootable citations, stated with the row count.
"""
    open(os.path.join(out_dir, "MANIFEST.txt"), "w").write(manifest)
    return {"sha": payload_sha, "answers": len(rows), "citations": len(citations),
            "unrootable": len(unrootable), "roots": len(universe),
            "unclassified_roots": sum(1 for r_ in universe if not cls_map.get(r_))}


if __name__ == "__main__":
    print(json.dumps(export(*sys.argv[1:]), indent=1))
