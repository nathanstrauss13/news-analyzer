#!/usr/bin/env python3
"""Signal MCP server — a client-side adapter that plugs innate c3 audit data
into the client's own AI assistant.

DESIGN CONTRACT (do not weaken):
  * FACTS, NEVER CONCLUSIONS. Tools whitelist fields. Narrative payload
    fields (executive_summary, headline_move, per_llm_read, verdicts,
    rationales, gap_insights, partnership_play) are never returned. The
    interpretation layer is the consultative engagement, not this pipe.
  * EVERY response carries `convention` metadata: scope, counting method,
    denominators, so an assistant cannot mix grains silently.
  * NO tool returns a whole payload (0.7-1.3MB); everything is sliced here.
  * Per-client allowlist: only slugs in the config are reachable. No
    wildcards. (e.g. Xsight's config carries its branded slug ONLY.)
  * compare_runs recomputes BOTH runs from raw answers under ONE stated
    convention at query time — stored aggregates are era-dependent and are
    never diffed across runs.
  * READ-ONLY: no tool triggers data collection. request_rerun files a
    request for the operator; it never runs anything.

Config (env SIGNAL_MCP_CONFIG, default ~/.signal_mcp/config.json):
  { "base_url": "https://signal.innatec3.com",
    "client": "innate-internal",
    "audits": [ {"slug": "94f86163db", "label": "Campbell's pillar preview",
                 "run_date": "2026-08-25"}, ... ] }
"""
import json, os, re, ssl, sys, time, hashlib
from urllib.request import Request, urlopen

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:                       # pragma: no cover
    _SSL_CTX = ssl.create_default_context()

from mcp.server.mcpserver import MCPServer

CFG_PATH = os.environ.get("SIGNAL_MCP_CONFIG",
                          os.path.expanduser("~/.signal_mcp/config.json"))
CACHE_DIR = os.path.expanduser("~/.signal_mcp/cache")
CACHE_TTL = int(os.environ.get("SIGNAL_MCP_CACHE_TTL", 24 * 3600))
REQUESTS_LOG = os.path.expanduser("~/.signal_mcp/rerun_requests.log")

server = MCPServer("signal-audit")

# ── config / fetch / cache ────────────────────────────────────────────────

def _cfg():
    with open(CFG_PATH) as f:
        c = json.load(f)
    c.setdefault("base_url", "https://signal.innatec3.com")
    c["_by_slug"] = {a["slug"]: a for a in c.get("audits", [])}
    return c


def _payload(slug):
    cfg = _cfg()
    if slug not in cfg["_by_slug"]:
        raise ValueError(f"slug '{slug}' is not in this client's audit list — "
                         f"use list_audits for the slugs available to you")
    os.makedirs(CACHE_DIR, exist_ok=True)
    fp = os.path.join(CACHE_DIR, hashlib.sha1(slug.encode()).hexdigest() + ".json")
    if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < CACHE_TTL:
        with open(fp) as f:
            return json.load(f)
    url = f"{cfg['base_url']}/signal/{slug}.json"
    req = Request(url, headers={"User-Agent": "signal-mcp/1.0"})
    data = json.loads(urlopen(req, timeout=60, context=_SSL_CTX).read().decode("utf-8"))
    with open(fp, "w") as f:
        json.dump(data, f)
    return data


# ── conventions (mirrors the platform's read-time rules) ──────────────────

from conventions import (URLISH_RE, GENERIC_TOKENS as _GENERIC_TOKENS,
                         root_of as _root, forms_to_pattern as _forms_to_pattern,
                         type_of as _type_of, split_rows as _split_rows)


def _convention(payload, counting="stored (full-text: name anywhere in the answer, cited links included)"):
    _, unb, branded = _split_rows(payload)
    return {
        "scope": payload.get("metrics_scope") or "all_responses",
        "counting": counting,
        "denominators": {"all_answers": len(payload.get("all_responses") or []),
                         "unbranded_answers": len(unb),
                         "branded_answers": (len(payload.get("all_responses") or []) - len(unb))},
        "note": ("Landscape/presence metrics are computed on unbranded answers only; "
                 "branded prompts name the brand by construction. Counts of citations "
                 "and counts of answers are different units — each figure is labeled."),
    }


def _evidence(slug, payload):
    """Page-level evidence for a slug: the payload's own citation_checks
    when present, else a config-declared supplement file (an offline
    full-corpus grounding pass, run after collection — e.g. for frozen
    public payloads that predate in-run page checks). Returns
    (checks_dict, source_note). Supplement entries carry checked_date."""
    cc = payload.get("citation_checks") or {}
    if cc:
        return cc, "collected at run time (top-cited subset)"
    meta = _cfg()["_by_slug"].get(slug, {})
    ef = meta.get("evidence_file")
    if ef and os.path.exists(os.path.expanduser(ef)):
        with open(os.path.expanduser(ef)) as f:
            sup = json.load(f)
        note = sup.get("_note") or "offline full-corpus grounding pass"
        return {k: v for k, v in sup.items() if not k.startswith("_")}, note
    return {}, "no page-level evidence available for this audit"


def _classes(slug, payload):
    """Domain-root -> source_type map. From the payload's own domain_types
    when present, else a config-declared classes_file supplement (an offline
    full-coverage pass with the platform's classifier — one classifier, one
    taxonomy). Returns (map, source_note)."""
    dt = payload.get("domain_types") or {}
    if dt:
        return dt, "classified at run time"
    meta = _cfg()["_by_slug"].get(slug, {})
    cf = meta.get("classes_file")
    if cf and os.path.exists(os.path.expanduser(cf)):
        with open(os.path.expanduser(cf)) as f:
            sup = json.load(f)
        return ({k: v for k, v in sup.items() if not k.startswith("_")},
                sup.get("_note") or "offline classification pass")
    return {}, "no source classification available for this audit"


def _label(slug, payload):
    meta = _cfg()["_by_slug"].get(slug, {})
    return {"slug": slug, "brand": payload.get("brand"),
            "label": meta.get("label"), "run_date": meta.get("run_date")}


VERBATIM_NOTE = ("Verbatim AI-agent output collected during the audit. Claims in the text "
                 "are the agent's, not innate c3's; citations are as the agent gave them.")


# ── tools ─────────────────────────────────────────────────────────────────

@server.tool()
def list_audits() -> dict:
    """List the audits available to this client: slug, brand, label, run date,
    answer counts. Facts only; use the slug with the other tools."""
    out = []
    for a in _cfg().get("audits", []):
        try:
            p = _payload(a["slug"])
            rows, unb, _ = _split_rows(p)
            out.append({**_label(a["slug"], p),
                        "answers": len(rows), "unbranded_answers": len(unb),
                        "agents": sorted({r.get("llm") for r in rows if r.get("llm")}),
                        "qa_verified": bool(p.get("qa_verified"))})
        except Exception as e:
            out.append({"slug": a.get("slug"), "error": str(e)[:120]})
    return {"audits": out}


@server.tool()
def get_audit(slug: str) -> dict:
    """Headline facts for one audit, from the platform's stored, QA-gated
    numbers: brand presence (with the named-in-prose vs cited-in-links-only
    split), per-agent visibility, prompt counts, citation totals, and the
    page-fetch ratio. Numbers come with denominators; no interpretation."""
    p = _payload(slug)
    rows, unb, branded = _split_rows(p)
    cc, ev_note = _evidence(slug, p)
    def _urls(rr):
        return {c.get("url") for r in rr for c in (r.get("citations") or [])
                if isinstance(c, dict) and c.get("url")}
    urls, urls_unb = _urls(rows), _urls(unb)
    loaded = sum(1 for v in cc.values() if v.get("status") == "ok")
    per_llm = [{"agent": x.get("llm"),
                "answers_mentioning_brand": x.get("mentions"),
                "of_answers": x.get("total"),
                "named_in_prose": x.get("named_prose"),
                "search_grounded": x.get("grounded")}
               for x in (p.get("per_llm_visibility") or [])]
    return {
        **_label(slug, p),
        "category": p.get("category"),
        "prompts": {"branded": len(branded),
                    "unbranded": len((p.get("prompt_sets") or {}).get("unbranded") or [])},
        "brand_presence": {
            "answers_mentioning_brand": p.get("brand_mention_count"),
            "of_unbranded_answers": len(unb),
            "split": p.get("brand_presence_split"),
            "split_definitions": {
                "named_prose": "the brand name appears in the answer text itself",
                "cited_only": "the name appears only inside cited link URLs — a reader "
                              "sees it only if they open the links"}},
        "per_agent": per_llm,
        "citations": {
            "extracted_unbranded_scope": p.get("total_citations_extracted"),
            "distinct_urls_all_answers": len(urls),
            "distinct_urls_unbranded_answers": len(urls_unb),
            "scope_note": "extracted_unbranded_scope counts citations in unbranded answers "
                          "only (the platform's stored metric); the distinct-URL figures are "
                          "labeled by scope — never compare counts across different scopes"},
        "page_fetch_ratio": {"scope": "all answers (branded + unbranded) — page checks cover "
                             "the full citation set",
                             "distinct_cited_urls": len(urls), "fetched": len(cc),
                             "loaded": loaded, "bot_walled": len(cc) - loaded,
                             "never_fetched": max(0, len(urls) - len(cc)),
                             "evidence_source": ev_note,
                             "note": "Page-level evidence exists only for the fetched subset."},
        "convention": _convention(p),
    }


@server.tool()
def query_citations(slug: str, domain: str = "", agent: str = "",
                    prompt_contains: str = "", limit: int = 40) -> dict:
    """Citations from one audit, filterable by domain root, agent, and prompt
    substring. Returns citation rows (prompt, agent, url, domain root) plus
    totals with denominators. A domain cited 12 times may appear in fewer
    answers — both units are reported."""
    p = _payload(slug)
    rows, unb, branded = _split_rows(p)
    _cls_map, _cls_note = _classes(slug, p)
    hits, answers_with = [], set()
    total = 0
    for r in rows:
        for c in (r.get("citations") or []):
            if not isinstance(c, dict) or not c.get("url"):
                continue
            root = _root(c.get("domain") or c["url"].split("//")[-1])
            total += 1
            if domain and _root(domain) != root:
                continue
            if agent and r.get("llm", "").lower() != agent.lower():
                continue
            if prompt_contains and prompt_contains.lower() not in r.get("prompt", "").lower():
                continue
            answers_with.add((r.get("llm"), r.get("prompt")))
            if len(hits) < max(1, min(limit, 100)):
                hits.append({"prompt": r.get("prompt"),
                             "prompt_class": "branded" if r.get("prompt") in branded else "unbranded",
                             "agent": r.get("llm"), "url": c["url"], "domain_root": root,
                             "source_type": _cls_map.get(root)})
    return {**_label(slug, p),
            "filter": {"domain": domain or None, "agent": agent or None,
                       "prompt_contains": prompt_contains or None},
            "citations_returned": hits,
            "answers_containing_matches": len(answers_with),
            "of_total_citations_all_answers": total,
            "scope_note": "this tool searches ALL answers (branded + unbranded); each row "
                          "carries prompt_class so results can be filtered to one scope",
            "source_type_source": _cls_note,
            "convention": _convention(p)}


@server.tool()
def outlet_profile(slug: str, domain: str) -> dict:
    """One outlet/domain in one audit: citation count, answers citing it,
    which agents cite it, share-of-voice numbers (with denominators), and
    page-level evidence for its fetched URLs. Facts only — no verdicts."""
    p = _payload(slug)
    rows, unb, _ = _split_rows(p)
    root = _root(domain)
    _, unb_, branded_ = _split_rows(p)
    cites, cites_unb, answers, agents = 0, 0, set(), set()
    for r in rows:
        for c in (r.get("citations") or []):
            if isinstance(c, dict) and _root(c.get("domain") or c.get("url", "")) == root:
                cites += 1
                if r.get("prompt") not in branded_:
                    cites_unb += 1
                answers.add((r.get("llm"), r.get("prompt")))
                agents.add(r.get("llm"))
    sov = next((o for o in (p.get("outlet_sov") or []) if _root(o.get("domain", "")) == root), None)
    sov_facts = None
    if sov:
        sov_facts = {k: sov.get(k) for k in
                     ("responses_citing", "brand_mentions_at_outlet", "brand_sov_at_outlet",
                      "brand_overall_sov", "brand_sov_differential") if k in sov}
        comp = sov.get("all_competitors_at_outlet") or []
        sov_facts["competitors_at_outlet"] = [
            {k: c.get(k) for k in ("name", "mentions_at_outlet", "sov_at_outlet", "overall_sov")}
            for c in comp[:6]]
    cc, ev_note = _evidence(slug, p)
    pages = [{"url": u, "fetch_status": v.get("status"),
              "brand_mentions_on_page": v.get("brand_count"),
              "page_title": (v.get("title") or "")[:120],
              **({"checked_date": v["checked_date"]} if v.get("checked_date") else {})}
             for u, v in cc.items() if _root(u.split("//")[-1]) == root]
    _cls_map, _cls_note = _classes(slug, p)
    return {**_label(slug, p), "domain_root": root,
            "source_type": _cls_map.get(root),
            "source_type_source": _cls_note,
            "citations_all_answers": cites,
            "citations_unbranded_answers": cites_unb,
            "answers_citing_all_scopes": len(answers),
            "agents_citing": sorted(a for a in agents if a),
            "share_of_voice": sov_facts,
            "share_of_voice_scope": "unbranded answers only (the platform's stored SoV "
                                    "metric) — compare it with citations_unbranded_answers, "
                                    "never with the all-answers count" if sov_facts else None,
            "page_evidence": pages[:20],
            "page_evidence_source": ev_note,
            "convention": _convention(p)}


@server.tool()
def get_responses(slug: str, prompt_contains: str = "", agent: str = "",
                  max_chars: int = 4000) -> dict:
    """Verbatim agent answers from one audit, filtered by prompt substring
    and/or agent. NOTE: this is raw AI-agent output — the claims are the
    agent's, not innate c3's. Long answers are truncated to max_chars with
    the full length stated."""
    p = _payload(slug)
    rows, _, branded = _split_rows(p)
    out = []
    for r in rows:
        if prompt_contains and prompt_contains.lower() not in r.get("prompt", "").lower():
            continue
        if agent and r.get("llm", "").lower() != agent.lower():
            continue
        t = r.get("response") or ""
        out.append({"prompt": r.get("prompt"),
                    "prompt_class": "branded" if r.get("prompt") in branded else "unbranded",
                    "agent": r.get("llm"), "search_grounded": r.get("grounded"),
                    "response_chars": len(t),
                    "response": t[:max(500, min(max_chars, 12000))],
                    "truncated": len(t) > max_chars,
                    "citations": [c.get("url") for c in (r.get("citations") or [])
                                  if isinstance(c, dict)][:15]})
        if len(out) >= 10:
            break
    return {**_label(slug, p), "note": VERBATIM_NOTE,
            "answers_returned": len(out), "answers": out,
            "convention": _convention(p)}


@server.tool()
def get_page_evidence(slug: str, only_missing_brand: bool = False, limit: int = 50) -> dict:
    """Page-level grounding for one audit: for each fetched cited URL, whether
    it loaded, and how many times the brand appears ON THE PAGE (as opposed
    to in the AI's answer). The fetch ratio states coverage plainly — this
    evidence exists only for the fetched subset, never the whole citation set."""
    p = _payload(slug)
    rows, _, _ = _split_rows(p)
    cc, ev_note = _evidence(slug, p)
    urls = {c.get("url") for r in rows for c in (r.get("citations") or [])
            if isinstance(c, dict) and c.get("url")}
    loaded = [(u, v) for u, v in cc.items() if v.get("status") == "ok"]
    silent = [u for u, v in loaded if not v.get("brand_count")]
    _cls_map, _cls_note = _classes(slug, p)
    # Full-set outlet rollup (never limited by `limit`): the story unit is the
    # OUTLET; pages are the evidence underneath. An outlet counts as
    # never_mentioning only when NONE of its loaded pages mentions the brand.
    _out = {}
    for u, v in cc.items():
        if v.get("status") != "ok":
            continue
        host = u.split("//")[-1].split("/")[0].split(":")[0].lower()
        host = host[4:] if host.startswith("www.") else host
        key = host if host in _cls_map else _root(host)
        t = _type_of(u, _cls_map) or "unclassified"
        d = _out.setdefault((t, key), [0, 0])
        d[0] += 1
        if v.get("brand_count"):
            d[1] += 1
    rollup = {}
    for (t, key), (n_pages, n_ment) in _out.items():
        r = rollup.setdefault(t, {"outlets_with_loaded_pages": 0,
                                  "outlets_never_mentioning_brand": 0,
                                  "pages_loaded": 0, "pages_mentioning_brand": 0})
        r["outlets_with_loaded_pages"] += 1
        r["pages_loaded"] += n_pages
        r["pages_mentioning_brand"] += n_ment
        if n_ment == 0:
            r["outlets_never_mentioning_brand"] += 1
    items = [{"url": u, "fetch_status": v.get("status"),
              "brand_mentions_on_page": v.get("brand_count") or 0,
              "page_title": (v.get("title") or "")[:120],
              "source_type": _type_of(u, _cls_map),
              **({"checked_date": v["checked_date"]} if v.get("checked_date") else {})}
             for u, v in cc.items()
             if not (only_missing_brand and (v.get("status") != "ok" or v.get("brand_count")))]
    return {**_label(slug, p),
            "fetch_ratio": {"scope": "all answers (branded + unbranded)",
                            "distinct_cited_urls": len(urls), "fetched": len(cc),
                            "loaded": len(loaded), "bot_walled": len(cc) - len(loaded),
                            "never_fetched": max(0, len(urls) - len(cc))},
            "loaded_pages_not_mentioning_brand": len(silent),
            "of_loaded_pages": len(loaded),
            "outlet_rollup_by_type": rollup,
            "rollup_scope": "FULL evidence set (never limited by `limit`); classification is "
                            "host-first with root fallback; an outlet is 'never mentioning' "
                            "only if none of its loaded pages mentions the brand",
            "evidence_source": ev_note,
            "source_type_source": _cls_note,
            "pages": items[:max(1, min(limit, 100))],
            "pages_scope": f"first {min(max(1, min(limit, 100)), len(items))} of {len(items)} "
                           "matching evidence rows — do NOT aggregate from this slice; use "
                           "outlet_rollup_by_type and the fetch/loaded totals above, which "
                           "cover the full set",
            "convention": _convention(p)}


@server.tool()
def compare_runs(slug_a: str, slug_b: str) -> dict:
    """Deltas between two runs of the same audit series. Both runs are
    recomputed from raw answers under ONE convention at query time (stored
    aggregates are not comparable across runs). Prompts are matched by exact
    text; unmatched prompts are listed, not silently dropped. Deltas come
    with denominators and a measurement note — no causal language: a
    movement of <=1 agent on a single prompt is within observed run-to-run
    variation on this instrument."""
    pa, pb = _payload(slug_a), _payload(slug_b)
    if (pa.get("brand") or "").strip().lower() != (pb.get("brand") or "").strip().lower():
        raise ValueError("compare_runs is for two runs of the same audit series; "
                         f"these are '{pa.get('brand')}' and '{pb.get('brand')}'")
    # ONE pattern for both runs. Aliases can differ between runs (the pipeline
    # can add one, e.g. a portfolio brand); using each run's own alias list
    # would smuggle convention drift back in — the exact failure this tool
    # exists to prevent. Alias set = the intersection across both runs.
    common_aliases = sorted({a for a in (pa.get("brand_aliases") or []) if isinstance(a, str)} &
                            {a for a in (pb.get("brand_aliases") or []) if isinstance(a, str)})
    forms = [pa.get("brand") or ""] + common_aliases
    pat_a = pat_b = _forms_to_pattern(forms)
    _, unb_a, _ = _split_rows(pa)
    _, unb_b, _ = _split_rows(pb)
    def _ungrounded(rows):
        return sum(1 for r in rows if "grounded" in r and not r.get("grounded"))

    def per_prompt(rows, pat):
        m = {}
        for r in rows:
            d = m.setdefault(r.get("prompt"), {"agents_mentioning": [], "agents_named_prose": [], "n": 0})
            d["n"] += 1
            t = r.get("response") or ""
            if pat.search(t):
                d["agents_mentioning"].append(r.get("llm"))
                if pat.search(URLISH_RE.sub(" ", t)):
                    d["agents_named_prose"].append(r.get("llm"))
        return m

    A, B = per_prompt(unb_a, pat_a), per_prompt(unb_b, pat_b)
    shared = sorted(set(A) & set(B))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    def _state(agent, d):
        if agent in d["agents_named_prose"]:
            return "named_in_prose"
        if agent in d["agents_mentioning"]:
            return "cited_links_only"
        return "absent"

    prompts = []
    for q in shared:
        a, b = A[q], B[q]
        agents = sorted(set(a["agents_mentioning"]) | set(b["agents_mentioning"]) |
                        set(a["agents_named_prose"]) | set(b["agents_named_prose"]))
        transitions = []
        for ag in agents:
            sa, sb = _state(ag, a), _state(ag, b)
            if sa != sb:
                transitions.append({"agent": ag, "run_a": sa, "run_b": sb})
        prompts.append({
            "prompt": q,
            "run_a": {"mentioning": sorted(a["agents_mentioning"]), "of": a["n"],
                      "named_prose": sorted(a["agents_named_prose"])},
            "run_b": {"mentioning": sorted(b["agents_mentioning"]), "of": b["n"],
                      "named_prose": sorted(b["agents_named_prose"])},
            # Three states per agent: named_in_prose > cited_links_only > absent.
            # A transition between ANY two states is a change — including
            # named_in_prose -> cited_links_only, where the mention count holds
            # steady while the name leaves the answer text (presence decaying
            # into the citations). Tracking `mentioning` alone is blind to it.
            "agent_transitions": transitions,
        })
    n_trans = sum(len(p["agent_transitions"]) for p in prompts)
    tot_a = sum(len(A[q]["agents_mentioning"]) for q in shared)
    tot_b = sum(len(B[q]["agents_mentioning"]) for q in shared)
    den_a = sum(A[q]["n"] for q in shared)
    den_b = sum(B[q]["n"] for q in shared)
    return {
        "run_a": {**_label(slug_a, pa), "ungrounded_answers": _ungrounded(unb_a)},
        "run_b": {**_label(slug_b, pb), "ungrounded_answers": _ungrounded(unb_b)},
        "convention": {
            "scope": "unbranded answers, prompts matched by exact text",
            "brand_forms": forms,
            "counting": "recomputed at query time from raw answers: ONE word-boundary "
                        "pattern (brand + aliases common to both runs) applied to both; "
                        "named_prose = same match on the answer text with URLs stripped. "
                        "Answers that ran without live web retrieval are counted but "
                        "reported per run in ungrounded_answers — treat a run with "
                        "nonzero ungrounded answers as only partially comparable",
            "why_recompute": "stored aggregates reflect the counting rules of their era; "
                             "raw answers are stable, so both runs are recounted under this "
                             "one convention"},
        "shared_prompts": len(shared), "prompts_only_in_a": only_a, "prompts_only_in_b": only_b,
        "agents_with_state_changes": n_trans,
        "state_definitions": {"named_in_prose": "brand named in the answer text",
                              "cited_links_only": "brand appears only inside cited link URLs",
                              "absent": "no brand presence in the answer"},
        "brand_presence_total": {"run_a": {"answers": tot_a, "of": den_a},
                                 "run_b": {"answers": tot_b, "of": den_b}},
        "per_prompt": prompts,
        "measurement_note": "A movement of <=1 agent on a single prompt is within observed "
                            "run-to-run variation on this instrument. This tool reports what "
                            "changed, not why.",
    }


@server.tool()
def request_rerun(slug: str, reason: str = "") -> dict:
    """File a request for a fresh data collection with the operator. This tool
    NEVER runs anything itself: re-runs use the same prompt set verbatim, are
    quality-gated, and are reviewed before delivery. Typical turnaround is
    within 2 business days."""
    cfg = _cfg()
    if slug not in cfg["_by_slug"]:
        raise ValueError("slug not in this client's audit list")
    os.makedirs(os.path.dirname(REQUESTS_LOG), exist_ok=True)
    with open(REQUESTS_LOG, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "client": cfg.get("client"), "slug": slug,
                            "reason": reason[:500]}) + "\n")
    return {"status": "requested", "slug": slug,
            "what_happens_next": "The operator reviews the request, runs the same prompt set "
                                 "verbatim, quality-gates the result, and the new run appears "
                                 "in list_audits. Typical turnaround: within 2 business days."}


if __name__ == "__main__":
    server.run()
