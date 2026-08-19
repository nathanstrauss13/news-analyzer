#!/usr/bin/env python3
"""Generic client deck from any audit slug: black/gold house plates, Roboto,
chart-led. Six generic plates computed fresh from the payload's raw answers
(never from stored headline numbers), using the QA-hardened counting rules:
word-boundary brand forms, plumbing hosts excluded at HOST level, metrics
partitioned by prompt_sets.

Usage:
  python3 build_client_deck.py <slug> [--pdf] [--prepared-for "Name · Org"]
  python3 build_client_deck.py /path/to/payload.json [--pdf]

Output: <slug>_deck.html (+ .pdf with --pdf) beside this script, or --out DIR.
Engagement-specific plates (a client's own test re-run, action readouts) are
bespoke by nature: build them on top by importing plate()/finish() from here,
as ~/Desktop/Azzaro/build_azzaro_deck.py did."""
import argparse, json, os, re, subprocess, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charts import GOLD, PALETTE, CSS, hbars_split

INK2, MUTED, ACCENT, GREEN = (PALETTE["ink2"], PALETTE["muted"],
                              PALETTE["accent"], PALETTE["green"])
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ---------------- counting (QA-hardened) ----------------
def host_of(u):
    m = re.match(r"https?://([^/]+)", u or "")
    h = (m.group(1) if m else "").lower()
    return h[4:] if h.startswith("www.") else h

def root_of(h):
    p = h.split(".")
    return ".".join(p[-2:]) if len(p) >= 2 else h

def is_plumbing(u):
    return "vertexaisearch" in host_of(u)   # host level: root collapses to google.com

def brand_pattern(name):
    body = re.escape(name).replace(r"\-", r"[\-‑–]").replace(r"\ ", r"\s+")
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)

def present(text, names):
    return any(brand_pattern(n).search(text or "") for n in names)


def analyze(data):
    brand = (data.get("brand") or "").strip()
    forms = [brand] + [a for a in (data.get("brand_aliases") or []) if len(a) >= 3]
    ps = data.get("prompt_sets") or {}
    bset = set(ps.get("branded") or [])
    rows = [r for r in (data.get("all_responses") or []) if not r.get("error")]
    unb = [r for r in rows if r.get("prompt") not in bset] if bset else rows
    bra = [r for r in rows if r.get("prompt") in bset]

    def cites(rs):
        c = Counter()
        for r in rs:
            for u in (r.get("citations") or []):
                uu = (u.get("url") if isinstance(u, dict) else str(u)) or ""
                if uu and not is_plumbing(uu):
                    c[root_of(host_of(uu))] += 1
        return c

    comp_names = []
    for c in (data.get("competitors") or []):
        n = (c.get("name") if isinstance(c, dict) else str(c)) or ""
        if n and n.lower() != brand.lower():
            comp_names.append(n)
    standings = [(n, sum(1 for r in unb if present(r.get("response"), [n])))
                 for n in comp_names[:11]]
    standings.append((brand, sum(1 for r in unb if present(r.get("response"), forms))))
    standings.sort(key=lambda kv: -kv[1])

    per_q = []
    for q in (ps.get("unbranded") or sorted({r["prompt"] for r in unb})):
        qrows = [r for r in unb if r.get("prompt") == q]
        per_q.append((q, sum(1 for r in qrows if present(r.get("response"), forms)), len(qrows)))
    per_q.sort(key=lambda x: -x[1])

    owned = {root_of(host_of("https://" + d)) for d in (data.get("brand_domains") or []) if d}
    cu, cb = cites(unb), cites(bra)
    return {
        "brand": brand, "forms": forms, "owned": owned,
        "n_unb": len(unb), "n_bra": len(bra),
        "unb_hits": sum(1 for r in unb if present(r.get("response"), forms)),
        "bra_hits": sum(1 for r in bra if present(r.get("response"), forms)),
        "standings": standings, "per_q": per_q,
        "cites_unb": cu, "cites_bra": cb,
        "n_prompts": len((ps.get("branded") or [])) + len((ps.get("unbranded") or [])) or
                     len({r["prompt"] for r in rows}),
        "date": (data.get("collected_at") or "")[:10] or "",
    }


# ---------------- plates ----------------
_plates = []

def deck_css():
    css = (CSS.replace('"Inter",-apple-system,sans-serif', '"Roboto","Helvetica Neue",sans-serif')
              .replace('"Jost","Inter",sans-serif', '"Roboto","Helvetica Neue",sans-serif')
              .replace('font-family:"Jost"', 'font-family:"Roboto"')
              .replace('Jost,Inter,sans-serif', 'Roboto,sans-serif')
              .replace('Inter,sans-serif', 'Roboto,sans-serif'))
    return css + """<style>
@page{size:1180px 880px;margin:0}
@media print{body{padding:0}.plate{margin:0;box-shadow:none;border:none;page-break-after:always}}
.lede{font-size:14.5px !important;max-width:1060px !important}
h2.head{font-size:34px !important}
.eyebrow{font-size:13px !important}
.fact .fv{font-size:27px !important}.fact .fk{font-size:12.5px !important}
.fact{padding:16px 18px !important}
.ev p{font-size:13.5px !important;margin-bottom:11px !important}
.ev .et{font-size:13px !important}
.memoq{font-size:13px !important;padding:13px 18px !important;margin-top:16px !important}
.hbl{font-size:12.5px !important}.hbv{font-size:12px !important}
.hbrow{grid-template-columns:170px 1fr !important;margin-bottom:9px !important}
.grid{gap:26px !important;margin-top:18px !important}
.sig{font-size:12px !important}.band .wm{font-size:16px !important}
</style>"""

def plate(pg, eyebrow, head, lede, body, brand, prepared_for, date_label, total):
    _plates.append(f"""<div class="plate">
<div class="band"><span class="wm">innate c<sup>3</sup></span>
<span class="pg">{brand.upper()} · AI VISIBILITY · {pg} / {total}</span></div>
<div class="eyebrow">{eyebrow}</div>
<h2 class="head">{head}</h2>
<div class="lede">{lede}</div>
{body}
<div class="sig"><span>Prepared for <b>{prepared_for}</b> · {date_label}</span>
<span>Confidential · <b>innate c³</b> · nstrauss@innatec3.com</span></div>
<div class="pageno">{pg}</div></div>""")


def bars(rows, maxv, gold_label=None, h=26):
    return hbars_split(
        [(lbl, v, 0, GOLD if (gold_label and lbl == gold_label) else
          (ACCENT if v == 0 else MUTED)) for lbl, v in rows], maxv=maxv, h=h)


def build(a, prepared_for, date_label, out_html):
    b, total = a["brand"], "05"
    P = lambda *args: plate(*args, brand=b, prepared_for=prepared_for,
                            date_label=date_label, total=total)
    total_cites = sum(a["cites_unb"].values()) + sum(a["cites_bra"].values())
    n_answers = a["n_unb"] + a["n_bra"]

    P("01", "AI VISIBILITY · SAMPLE READ", f"{b} in AI answers",
      f"We put {a['n_prompts']} questions to five AI agents (ChatGPT, Claude, Gemini, Perplexity, "
      f"Grok), each with live web search, and traced every source they cited. Unbranded questions "
      f"never name a vendor: they measure who AI brings up on its own. Every answer is kept "
      f"verbatim; every number here can be traced to a real answer and a real cited page.",
      f"""<div class="factrow">
<div class="fact"><div class="fv">{n_answers}</div><div class="fk">verbatim answers collected</div></div>
<div class="fact"><div class="fv">{total_cites:,}</div><div class="fk">citations extracted, resolved and classified</div></div>
<div class="fact"><div class="fv">{a['unb_hits']}<span class="u"> of {a['n_unb']}</span></div>
<div class="fk"><b>unbranded answers</b> name {b} without being asked</div></div>
<div class="fact"><div class="fv">{a['bra_hits']}<span class="u"> of {a['n_bra']}</span></div>
<div class="fk"><b>branded answers</b> engage when the brand is named (the expected baseline)</div></div></div>""")

    st = a["standings"]
    P("02", "THE DISCOVERY LANDSCAPE", "Who AI names when no one is named",
      f"{a['n_unb']} answers to the unbranded questions, the queries buyers pose before they know "
      f"the vendors. This is the competitive read that matters: presence here is earned, never "
      f"assumed.",
      f"""<div class="grid"><div>{bars(st, maxv=max(v for _, v in st) or 1, gold_label=b)}
<div style="font-size:10px;color:{MUTED};margin-top:6px">Answers naming each brand, of {a['n_unb']} unbranded answers</div>
</div><div><div class="ev"><div class="et">How to read this</div>
<p>A brand counts as present when its name appears in the answer text. Presence when the prompt
names the brand is the baseline; presence here is the earned kind.</p>
<p>Sample reads are directional by design: answers vary run to run, and the patterns worth acting
on are the ones that persist across agents and questions.</p></div></div></div>""")

    qrows = [(q[:52] + ("…" if len(q) > 52 else ""), v) for q, v, n in a["per_q"]]
    P("03", "WHERE THE PRESENCE LIVES", f"Which questions carry {b}'s appearances",
      f"Presence is rarely spread evenly: it clusters where existing coverage supports the answer "
      f"and goes quiet where it does not. The zeros are the amplification map.",
      f"""<div class="grid"><div>{bars(qrows, maxv=max((n for _, _, n in a['per_q']), default=5), h=28)}
<div style="font-size:10px;color:{MUTED};margin-top:6px">Agents naming {b}, of 5, per unbranded question</div>
</div><div><div class="ev"><div class="et">What this says</div>
<p><b>The questions at the top</b> are where AI already believes the story: hold them.</p>
<p><b>The zeros</b> are open lanes. The sources page shows what the answers in those lanes cite,
which is where earned and owned work starts.</p></div></div></div>""")

    def src_rows(c, own):
        rows = [(("%s" % r) + (" (owned)" if r in own else ""), n) for r, n in c.most_common(10)]
        return rows
    cu, cb = a["cites_unb"], a["cites_bra"]
    own_u = sum(n for r, n in cu.items() if r in a["owned"])
    own_b = sum(n for r, n in cb.items() if r in a["owned"])
    P("04", "WHAT THE ANSWERS CITE", "The sources doing the work, by scope",
      f"Discovery answers ({sum(cu.values())} citations) and branded answers ({sum(cb.values())} "
      f"citations) run on different sources. Owned domains draw {own_u} citations at discovery and "
      f"{own_b} when the brand is named: the two scopes need different plays.",
      f"""<div class="grid"><div>
<div class="ev"><div class="et">Unbranded scope · top sources</div></div>
{bars(src_rows(cu, a['owned']), maxv=max(cu.values() or [1]), h=22)}
</div><div>
<div class="ev"><div class="et">Branded scope · top sources</div></div>
{bars(src_rows(cb, a['owned']), maxv=max(cb.values() or [1]), h=22)}
</div></div>
<div class="memoq"><span class="src">The pattern that generalizes</span>
Discovery is carried by third-party sources; owned domains earn their citations once the brand is
named. Earned coverage moves the first number, owned content quality moves the second.</div>""")

    P("05", "METHOD", "Built to be re-measured",
      "The value of a baseline is that the identical questions can be re-asked after the work "
      "ships, so movement is reported question by question rather than asserted.",
      f"""<div class="grid"><div><div class="ev"><div class="et">What was done</div>
<p><b>Collection:</b> five agents (ChatGPT, Claude, Gemini, Perplexity, Grok), live web search,
answers kept verbatim, citations resolved to their real destinations and classified by source.</p>
<p><b>Counting:</b> a brand counts as present when named in the answer text; retrieval-plumbing
links are excluded before counting. Every figure is recomputable from the stored answers.</p></div>
<div class="memoq"><span class="src">Honest limits</span>
A sample read: one day, {a['n_prompts']} questions. Answers vary run to run; a fuller engagement
runs a bespoke set at scale, verifies every cited page by hand, and re-measures after the work
ships.</div></div>
<div><div class="ev"><div class="et">What the full engagement adds</div>
<p><b>Scale:</b> a bespoke question set, hundreds of outputs, re-measured on your calendar.</p>
<p><b>Verification:</b> every cited page fetched and human-read, so what drives the answers is
verified rather than inferred.</p>
<p><b>The read-out:</b> comms and marketing recommendations in your team's language.</p></div>
<div class="factrow" style="grid-template-columns:1fr;margin-top:12px">
<div class="fact"><div class="fv" style="font-size:16px">Suggested next step</div>
<div class="fk">A 30-minute walkthrough of this read with your team.</div></div></div></div></div>""")

    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700'
             '&display=swap" rel="stylesheet">')
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{b} · AI Visibility Sample Read · innate c3</title>"
            + fonts + deck_css() + "</head><body>" + "".join(_plates) + "</body></html>")
    open(out_html, "w").write(html)
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug_or_path")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--prepared-for", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if os.path.exists(args.slug_or_path):
        data = json.load(open(args.slug_or_path))
        slug = data.get("slug") or os.path.basename(args.slug_or_path).split(".")[0]
    else:
        slug = args.slug_or_path
        url = f"https://signal.innatec3.com/signal/{slug}.json"
        # curl instead of urllib: python.org macOS builds ship without local certs
        raw = subprocess.run(["curl", "-s", "--max-time", "60", url],
                             capture_output=True, check=True).stdout
        data = json.loads(raw)

    a = analyze(data)
    out_dir = args.out or os.path.dirname(os.path.abspath(__file__))
    out_html = os.path.join(out_dir, f"{slug}_deck.html")
    from datetime import date
    date_label = a["date"] or date.today().strftime("%B %d, %Y")
    build(a, args.prepared_for or a["brand"], date_label, out_html)
    print(f"built {out_html}")
    print(f"  {a['brand']}: unbranded {a['unb_hits']}/{a['n_unb']}, branded {a['bra_hits']}/{a['n_bra']}")
    print(f"  standings: {a['standings'][:6]}")
    if args.pdf and os.path.exists(CHROME):
        out_pdf = out_html.replace(".html", ".pdf")
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out_pdf}", "--virtual-time-budget=12000",
                        f"file://{out_html}"], capture_output=True)
        print(f"built {out_pdf} ({os.path.getsize(out_pdf)//1024} KB)")


if __name__ == "__main__":
    main()
