#!/usr/bin/env python3
"""innate c3 deck kit — shared chart + card components for client decks.

Extracted from the BlackRock working-session deck (July 2026), which introduced
the shaded/stacked bar charts, clickable tearsheet cards and donut charts that
now read as the house visual language. Import these instead of copy-pasting so
every deck stays consistent.

    import sys; sys.path.insert(0, os.path.expanduser("~/Desktop/innate c3/deck_kit"))
    from charts import hbars_split, paired_bars, donut, legend, butterfly, tear, note, CSS, PALETTE

Design rules that travel with these components:
  * Two-segment bars encode a REAL split (persona, branded vs unbranded, owned vs
    earned). Solid = first segment, hatched = second. Never decorative.
  * Tearsheet cards must link to the actual cited page (href), and carry the
    citation count + how many of the five assistants cited it. That provenance
    is the anti-skeptic feature; do not ship a tearsheet without it.
  * Donuts are for composition of a whole (source mix, platform mix) with the
    total in the center. Max ~5 slices; more than that, use bars.
  * Colors carry meaning per deck (tier or persona); keep one legend per slide.
  * Sober house tone: no superlatives, no em dashes in narrative copy.
"""
import math

# House palette (dark premium). GOLD/PERI are the default two-segment pair.
GOLD, TEAL, TERRA, PERI = "#b8862e", "#0f9c8e", "#bd6247", "#7a7fd9"
C_PUB, C_MED = GOLD, PERI
PALETTE = {"gold": GOLD, "teal": TEAL, "terra": TERRA, "peri": PERI,
           "green": "#c2a36b", "accent": "#c06a52", "ink": "#f1eee8",
           "ink2": "#b7bac1", "muted": "#787c84", "line": "#2b2f37"}
HATCH = "repeating-linear-gradient(45deg, rgba(10,11,13,.55) 0 2.5px, transparent 2.5px 5.5px)"

def hbars_split(rows, maxv=None, h=17):
    """rows: (label, v_pub, v_med, color). Stacked persona segments in the
    outlet's tier color: solid = general public, hatched = media/journalists,
    2px surface gap between segments, total labeled at the bar end."""
    maxv = maxv or max(a + b for _, a, b, _ in rows)
    out = ['<div class="hb">']
    for label, a, b, color in rows:
        total = a + b
        segs = ""
        if a:
            segs += f'<div class="hbbar hbseg" style="width:{max(a / maxv * 100, 1.5):.2f}%;background:{color}"></div>'
        if b:
            segs += (f'<div class="hbbar hbseg" style="width:{max(b / maxv * 100, 1.5):.2f}%;'
                     f'background:{HATCH}, linear-gradient({color},{color})"></div>')
        out.append(
            f'<div class="hbrow" style="height:{h}px">'
            f'<div class="hbl">{label}</div>'
            f'<div class="hbtrack">{segs}<span class="hbv">{total}</span></div></div>')
    out.append('</div>')
    return "".join(out)


def paired_bars(rows, h=15):
    """rows: (label, v_pub, v_med). Two thin bars per row, persona colors."""
    maxv = max(max(a, b) for _, a, b in rows)
    out = ['<div class="hb">']
    for label, a, b in rows:
        wa = max(2.0, a / maxv * 100) if a else 0
        wb = max(2.0, b / maxv * 100) if b else 0
        bars = ""
        for w, v, col in ((wa, a, C_PUB), (wb, b, C_MED)):
            bar = f'<div class="hbbar" style="width:{w}%;background:{col}"></div>' if v else \
                  '<div class="hbbar" style="width:0"></div>'
            bars += f'<div class="hbtrack" style="height:{h-6}px">{bar}<span class="hbv">{v}</span></div>'
        out.append(f'<div class="pbrow"><div class="hbl">{label}</div><div class="pbbars">{bars}</div></div>')
    out.append('</div>')
    return "".join(out)


def donut(slices, size=168, title="", total_label=""):
    """slices: (label, value, color). SVG donut, 3-degree gaps, legend built separately."""
    total = sum(v for _, v, _ in slices)
    cx = cy = size / 2
    r_out, r_in = size / 2 - 2, (size / 2 - 2) * 0.60
    a = -90.0
    pad = 3.0
    paths = []
    for label, v, color in slices:
        sweep = v / total * 360.0
        a0, a1 = a + pad / 2, a + sweep - pad / 2
        if a1 <= a0:
            a = a + sweep
            continue
        large = 1 if (a1 - a0) > 180 else 0
        x0o, y0o = cx + r_out * math.cos(math.radians(a0)), cy + r_out * math.sin(math.radians(a0))
        x1o, y1o = cx + r_out * math.cos(math.radians(a1)), cy + r_out * math.sin(math.radians(a1))
        x1i, y1i = cx + r_in * math.cos(math.radians(a1)), cy + r_in * math.sin(math.radians(a1))
        x0i, y0i = cx + r_in * math.cos(math.radians(a0)), cy + r_in * math.sin(math.radians(a0))
        paths.append(
            f'<path d="M {x0o:.2f} {y0o:.2f} A {r_out:.2f} {r_out:.2f} 0 {large} 1 {x1o:.2f} {y1o:.2f} '
            f'L {x1i:.2f} {y1i:.2f} A {r_in:.2f} {r_in:.2f} 0 {large} 0 {x0i:.2f} {y0i:.2f} Z" fill="{color}"/>')
        a += sweep
    center = (f'<text x="{cx}" y="{cy - 3}" text-anchor="middle" fill="#f1eee8" '
              f'font-family="Jost,Inter,sans-serif" font-weight="700" font-size="26">{total}</text>'
              f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" fill="#787c84" font-size="9.5" '
              f'font-family="Inter,sans-serif" letter-spacing="1">{total_label}</text>')
    return (f'<div class="donutwrap"><div class="donuttitle">{title}</div>'
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{"".join(paths)}{center}</svg></div>')


def legend(items):
    return ('<div class="leg">' +
            "".join(f'<span class="legit"><span class="sw" style="background:{c}"></span>{t}</span>'
                    for t, c in items) + '</div>')


def butterfly(rows, maxv=None):
    """rows: (label, v_pub, v_med). Mirrored bars around a center label column."""
    maxv = maxv or max(max(a, b) for _, a, b in rows)
    out = ['<div class="bf">']
    for label, a, b in rows:
        wa, wb = a / maxv * 100, b / maxv * 100
        out.append(
            '<div class="bfrow">'
            f'<div class="bfside bfleft"><span class="hbv">{a}</span>'
            f'<div class="hbbar" style="width:{max(wa,2):.1f}%;background:{C_PUB}"></div></div>'
            f'<div class="bfl">{label}</div>'
            f'<div class="bfside"><div class="hbbar" style="width:{max(wb,2):.1f}%;background:{C_MED}"></div>'
            f'<span class="hbv">{b}</span></div></div>')
    out.append('</div>')
    return "".join(out)


def tear(src, head, metric, href):
    return (f'<a class="tear" href="{href}"><div class="tsrc">{src}</div>'
            f'<div class="thead">{head}</div><div class="tmetric">{metric} · <span style="color:#cdb87a">view ↗</span></div></a>')


def note(text):
    return f'<div class="tear note"><div class="thead">{text}</div></div>'


# ---------------------------------------------------------------- css
CSS = """<style>
:root{--bg:#0a0b0d;--paper:#141619;--card:#1b1e23;--ink:#f1eee8;--ink2:#b7bac1;
--muted:#787c84;--line:#2b2f37;--green:#c2a36b;--accent:#c06a52;}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{background:var(--bg);color:var(--ink);font-family:"Inter",-apple-system,sans-serif;line-height:1.5;padding:28px 0}
.plate{width:1180px;height:880px;margin:0 auto 28px;background:var(--paper);padding:42px 60px 36px;position:relative;box-shadow:0 8px 34px rgba(0,0,0,.5);overflow:hidden;border:1px solid #20242b}
h1,h2,h3,.jost{font-family:"Jost","Inter",sans-serif}
.band{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--green);padding-bottom:12px;margin-bottom:18px}
.band .wm{font-family:"Jost";font-weight:500;font-size:15px;color:var(--ink)}
.band .pg{font-weight:600;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.eyebrow{font-family:"Jost";font-weight:600;font-size:11.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--green)}
h2.head{font-family:"Jost";font-weight:600;font-size:29px;line-height:1.08;margin:7px 0 6px;color:var(--ink)}
.lede{font-size:12.5px;color:var(--ink2);max-width:1040px;line-height:1.55}
.lede b{color:var(--ink)}
.memoq{background:rgba(194,163,107,.07);border:1px solid var(--line);border-left:3px solid var(--green);padding:9px 16px;margin-top:11px;font-size:11.8px;color:var(--ink2);line-height:1.5}
.memoq b{color:var(--ink)}
.memoq .src{font-family:"Jost";font-weight:600;font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--green);display:block;margin-bottom:2px}
.grid{display:grid;grid-template-columns:1.12fr 1fr;gap:20px;margin-top:12px}
.ev .et,.recs .et{font-family:"Jost";font-weight:600;font-size:11.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.ev p{font-size:12px;color:var(--ink2);line-height:1.55;margin-bottom:8px}
.ev p b{color:var(--ink)}
.rec{display:grid;grid-template-columns:24px 1fr;gap:9px;background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);padding:9px 13px;margin-bottom:8px}
.rec .rn{font-family:"Jost";font-weight:700;font-size:13px;color:var(--accent);opacity:.85}
.rec .rt{font-family:"Jost";font-weight:600;font-size:12.5px;color:var(--ink)}
.rec .rd{font-size:11.2px;color:var(--ink2);line-height:1.48;margin-top:2px}
.factrow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px}
.factrow.f5{grid-template-columns:repeat(5,1fr);gap:12px}
.fact{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--green);padding:12px 16px}
.fact .fv{font-family:"Jost";font-weight:700;font-size:21px;color:var(--green);line-height:1.05}
.fact .fv .u{font-size:11px;font-weight:500;color:var(--muted)}
.fact .fk{font-size:11px;color:var(--ink2);margin-top:5px;line-height:1.45}
.fact .fk b{color:var(--ink)}
.reads{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:11px}
.read{border-left:3px solid var(--accent);background:rgba(192,106,82,.06);padding:8px 12px;font-size:11px;color:var(--ink2);line-height:1.48}
.read b{color:var(--ink)}
/* tearsheet cards */
.tearcols{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px}
.tearcol{display:flex;flex-direction:column;gap:9px}
.tearhd{font-family:"Jost";font-weight:600;font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink);padding-top:8px;border-top:2px solid}
.tearhd .thc{font-size:10px;color:var(--muted);font-weight:500;letter-spacing:.06em;display:block;margin-top:2px;text-transform:none}
.tear{background:var(--card);border:1px solid var(--line);padding:10px 13px;display:flex;flex-direction:column;gap:5px;min-height:80px;text-decoration:none;color:inherit}
a.tear:hover{border-color:var(--green)}
.tear .tsrc{font-family:"Jost";font-weight:700;font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink2);display:flex;align-items:center;gap:7px;min-height:16px}
.tear .tsrc img{display:block}
.tear .thead{font-family:"Jost";font-weight:500;font-size:13px;line-height:1.22;color:var(--ink);flex:1}
.tear .tmetric{font-size:10px;color:var(--muted);border-top:1px solid var(--line);padding-top:6px;font-variant-numeric:tabular-nums}
.tear .tmetric b{color:var(--green);font-weight:600}
.tear.note{background:transparent;border-style:dashed;justify-content:center}
.tear.note .thead{font-size:11px;color:var(--ink2);font-weight:400;flex:none}
.sig{position:absolute;bottom:20px;left:60px;right:96px;display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.sig b{color:var(--ink)}
.sig a{color:#cdb87a;text-decoration:none}
.pageno{position:absolute;bottom:21px;right:26px;font-family:"Jost";font-weight:500;font-size:9.5px;color:var(--muted);letter-spacing:.06em}
/* charts */
.hb{margin-top:4px}
.hbrow{display:grid;grid-template-columns:118px 1fr;gap:10px;align-items:center;margin-bottom:5px}
.hbl{font-size:10.5px;color:var(--ink2);text-align:right;line-height:1.15}
.hbtrack{display:flex;align-items:center;height:11px}
.hbbar{height:100%;border-radius:0 3px 3px 0;min-width:0}
.hbseg{border-radius:0;margin-right:2px}
.hbseg:last-of-type{border-radius:0 3px 3px 0}
.hbv{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;margin-left:6px}
.pbrow{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin-bottom:8px}
.pbbars{display:flex;flex-direction:column;gap:2px}
.leg{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 2px}
.legit{font-size:10.5px;color:var(--ink2);display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.donutrow{display:flex;gap:34px;justify-content:center;margin-top:6px}
.donutwrap{text-align:center}
.donuttitle{font-family:"Jost";font-weight:600;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink2);margin-bottom:7px}
.bf{margin-top:10px}
.bfrow{display:grid;grid-template-columns:1fr 190px 1fr;gap:10px;align-items:center;margin-bottom:12px}
.bfl{font-size:11px;color:var(--ink);text-align:center;line-height:1.25;font-weight:500}
.bfside{display:flex;align-items:center;gap:7px;height:13px}
.bfside .hbbar{border-radius:0 3px 3px 0}
.bfleft{justify-content:flex-end}
.bfleft .hbbar{border-radius:3px 0 0 3px}
/* prompts slide */
.pcols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
.pcol{background:var(--card);border:1px solid var(--line);border-top:3px solid;padding:12px 16px}
.pcol .pt{font-family:"Jost";font-weight:600;font-size:13.5px;color:var(--ink)}
.pcol .ps{font-size:10.8px;color:var(--ink2);line-height:1.45;margin:3px 0 8px}
.pcol ol{margin-left:16px}
.pcol li{font-size:10.6px;color:var(--ink2);line-height:1.42;margin-bottom:2.5px}
/* title slide */
.tslide{display:flex;flex-direction:column;justify-content:center;align-items:flex-start;height:100%;padding-bottom:60px}
.tslide h1{font-family:"Jost";font-weight:600;font-size:46px;line-height:1.12;color:var(--ink);max-width:980px;margin:18px 0 14px}
.tslide h1 .tsecond{font-size:29px;font-weight:500;color:var(--green)}
.tslide .tsub{font-size:15px;color:var(--ink2);max-width:760px;line-height:1.6}
.tslide .tmeta{margin-top:34px;font-size:12px;color:var(--muted)}
.tslide .tmeta b{color:var(--ink2)}
@media print{body{background:var(--bg);padding:0}
.plate{box-shadow:none;margin:0;border:0;page-break-after:always}
@page{size:1180px 880px;margin:0}}
</style>"""
