#!/usr/bin/env python3
"""Client-facing data completeness workbook: the standard internal deliverable
for every paid audit run (Nathan, Oct 20, 2026), for Nathan and between
sessions, next to the Raw Run bundle and the findings deck.

Three tabs. "Read me": definitions, verification scope, how to use. "Answers":
one row per prompt x platform with the response (preview column plus a
collapsible full-response column), search ran, retrieval evidence (agent
searches / pages read / claims supported), citations by delivery channel,
demasked count, duplicates removed, dead links dropped, pages scraped,
pages naming the brand, notes (empty or runaway answers are flagged here).
"Citations": one row per citation kept after dedupe with delivery channel,
demasking, duplicate handling, page scrape, page title, brand-on-page count,
published date, source type and classification basis. Evidence only.

Styling follows the innate c3 tracker look (Roboto, teal header, filters on
every column, frozen panes, coloured dropdown values). Upload the xlsx to
Drive and save as a Google Sheet (see memory reference_client_data_workbook).

usage: python3 build_data_workbook.py <payload.json> <classes.json> <rawrun_dir> <out.xlsx> [--checks checks_full.json] [--client "Cuore"] [--run-date "Oct 19, 2026"]
"""
import json, re, sys, csv, collections
csv.field_size_limit(10**9)
from urllib.parse import urlparse
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

import argparse
ap = argparse.ArgumentParser(description="Build the client-facing data completeness workbook for one audit payload.")
ap.add_argument("payload"); ap.add_argument("classes"); ap.add_argument("rawrun_dir"); ap.add_argument("out")
ap.add_argument("--checks", help="full page-check JSON (default: the payload's citation_checks)")
ap.add_argument("--client", default="", help="who the workbook is prepared for, e.g. 'Cuore'")
ap.add_argument("--run-date", default="", help="collection date shown in the Read me, e.g. 'Oct 19, 2026'")
args = ap.parse_args()
d = json.load(open(args.payload))
cls = json.load(open(args.classes))
ck = json.load(open(args.checks)) if args.checks else (d.get("citation_checks") or {})
rows = d["all_responses"]
BRAND = d.get("brand") or "the brand"
RAW = args.rawrun_dir.rstrip("/") + "/"
CLIENT = args.client; RUN_DATE = args.run_date or d.get("merged_at", "")[:10]
branded = d["prompt_sets"]["branded"]; unb = d["prompt_sets"]["unbranded"]
order = branded + unb
pnum = {p: ("B%02d" % (i + 1) if p in branded else "U%02d" % (unb.index(p) + 1)) for i, p in enumerate(order)}
pc = {r["url"]: r for r in csv.DictReader(open(RAW + "07_page_checks_per_url.csv"))}
retrieved = collections.defaultdict(list)
for r in csv.DictReader(open(RAW + "13_retrieved_pages.csv")):
    retrieved[(r["llm"], r["prompt_class"], r["answer_id"])].append(r)
ans_ids = {(r["llm"], r["prompt"]): r["answer_id"] for r in csv.DictReader(open(RAW + "01_answers.csv"))}

SECOND = {"co.uk", "com.au", "co.jp", "co.in", "com.br", "co.nz", "org.uk", "gov.uk"}
def host(u):
    h = (urlparse(u).netloc or "").lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h
def root(h):
    p = h.split(".")
    if len(p) >= 3 and ".".join(p[-2:]) in SECOND: return ".".join(p[-3:])
    return ".".join(p[-2:]) if len(p) >= 2 else h
def norm(u):
    u = re.sub(r"[?&](utm_[^&#]*|fbclid=[^&#]*|gclid=[^&#]*)", "", u or "")
    return re.sub(r"^https?://(www\.)?", "", u).rstrip("?&/").lower()
WRAP = re.compile(r"vertexaisearch\.cloud\.google\.com|grounding-api-redirect|google\.com/url\?", re.I)
owned = {root(x) for x in d.get("brand_domains") or []}
LABEL = {"editorial": "Earned media (news, trade, business press)", "owned": f"Brand website ({BRAND})",
         "competitor": "Competitor website", "corporate": "Other company website",
         "reference": "Reference source (Wikipedia, encyclopedias)", "social": "Social media",
         "community": "Community / forum", "aggregator": "Aggregator, directory or data site",
         "institutional": "Government, regulator or association", "retail": "Retail / marketplace",
         "reviews": "Review site", "namesake": "Unrelated company with a similar name"}
DECIDED = {e["root"]: e.get("decision") for e in cls.get("_exceptions", [])}
MANUAL = {k: v for k, v in (cls.get("_manual_checks") or {}).items()}   # optional {root: note} block in the classes file

# duplicate counts per answer
dup = collections.Counter()
for m in d.get("merged_citation_repeats") or []: dup[(m["llm"], m["prompt"])] += 1
for m in d.get("tracking_param_repeats_merged") or []: dup[(m["llm"], m["prompt"])] += 1
dup_url = collections.Counter()
for m in (d.get("merged_citation_repeats") or []) + (d.get("tracking_param_repeats_merged") or []):
    dup_url[(m["llm"], m["prompt"], norm(m["url"]))] += 1
dead = collections.Counter()
for x in d.get("dropped_citations") or []: dead[(x["llm"], x["prompt"])] += 1

MODEL = {"ChatGPT": "gpt-5-search-api", "Claude": "claude-sonnet-4-6", "Gemini": "gemini-2.5-flash (Google Search grounding)",
         "Perplexity": "sonar-pro", "Grok": "grok-4.7 (xAI direct)"}
CELL_MAX = 49000

wb = Workbook()
ws = wb.active; ws.title = "Answers"
H1 = ["Prompt #", "Prompt set", "Prompt", "Platform", "Model", "Search ran (grounded)", "Retrieval evidence (agent searches / pages read / claims supported)",
      "Response preview (first 200 characters)", "Full response (collapsible column: use the +/- above the column letters)", "Response characters", "Citations kept (after dedupe)", "Links written in the answer text",
      "Citations supplied through the API citation channel (not in the text)", "Demasked from a Google redirect",
      "Pages read but not cited", "Duplicates removed (identical page in one answer)", "Dead links dropped (404, DNS, refused)",
      "Cited pages scraped: loaded", "Cited pages scraped: refused", f"Cited pages naming {BRAND}", "Notes"]
ws.append(H1)
cit_rows = []
for r in sorted(rows, key=lambda r: (order.index(r["prompt"]), ["ChatGPT", "Claude", "Gemini", "Perplexity", "Grok"].index(r["llm"]))):
    key = (r["llm"], r["prompt"]); pnm = pnum[r["prompt"]]; pcls = "Branded" if r["prompt"] in branded else "Unbranded"
    t = r.get("response") or ""; tn = norm_text = re.sub(r"^https?://(www\.)?", "", t.lower())
    g = r.get("grounding") or {}
    sup = {norm(u) for s in g.get("supports", []) for u in (s.get("urls") or [])}
    ret = {norm(x.get("url")) for x in g.get("retrieved", [])}
    wrappers = len(WRAP.findall(t))
    aid = ans_ids.get(key, "")
    ret_rows = retrieved.get((r["llm"], pcls.lower(), aid), [])
    n_text = n_api = n_demask = 0
    loaded = refused = naming = 0
    for c in r["citations"]:
        u = c["url"]; nu = norm(u)
        in_text = nu in tn
        in_sup, in_ret = nu in sup, nu in ret
        if in_text:
            chan = "Written in the answer text"
        elif wrappers and (in_sup or in_ret):
            chan = "Written in the answer text as a Google redirect, demasked to the final URL"; n_demask += 1
        elif in_sup:
            chan = "API citation channel (structured annotation / grounding support)"
        elif in_ret:
            chan = "API retrieval list (page the agent read and cited through the API)"
        else:
            chan = "Payload citation (form in the answer differs; matched by the resolver)"
        if chan.startswith("Written in the answer text"): n_text += 1
        else: n_api += 1
        p = ck.get(u) or {}
        st = "Loaded" if p.get("status") == "ok" else "Refused (bot wall or paywall)"
        if p.get("status") == "ok": loaded += 1
        else: refused += 1
        bc = int(p.get("brand_count") or 0)
        if bc: naming += 1
        h = host(u); rt = root(h)
        typ = "owned" if rt in owned else (cls.get(h) or cls.get(rt))
        basis = ("Manual decision after review: " + DECIDED[rt]) if rt in DECIDED else ("Manual check: " + MANUAL[rt] if rt in MANUAL else ("Brand domain supplied for the run" if rt in owned else "Classification pass (run-time types plus one model pass), exceptions reviewed"))
        dups = dup_url.get((r["llm"], r["prompt"], nu), 0)
        pcr = pc.get(u, {})
        cit_rows.append([len(cit_rows) + 1, pnm, pcls, r["prompt"], r["llm"], u, rt, chan,
                         "Yes" if chan.startswith("Written in the answer text as a Google redirect") else "No",
                         f"{dups} repeat(s) of this page removed from this answer" if dups else "None",
                         st, (p.get("title") or "")[:200], bc, pcr.get("published") or "", LABEL.get(typ, typ or "Unclassified"), basis, ""])
    notes = []
    if len(t) > CELL_MAX: notes.append(f"Response is {len(t):,} characters; the cell holds the first {CELL_MAX:,}; the full text is in the Raw Run file 01_answers.csv.")
    if len(t) > 100000: notes.append("Runaway repetition in this answer (the agent looped); every count in this workbook is per answer so it does not inflate any figure.")
    if not r["citations"]: notes.append("The agent returned no citations for this answer.")
    ws.append([pnm, pcls, r["prompt"], r["llm"], MODEL.get(r["llm"], r.get("model_id") or ""),
               "Yes" if r.get("grounded") else "No",
               f"{len(g.get('queries', []))} agent searches; {len(g.get('retrieved', []))} pages read; {len(g.get('supports', []))} claims supported",
               (t[:200].replace(chr(10), ' ') + ('…' if len(t) > 200 else '')), t[:CELL_MAX], len(t), len(r["citations"]), n_text, n_api, n_demask,
               sum(1 for x in ret_rows if (x.get("cited_in_answer") or "").lower() != "true"),
               dup.get(key, 0), dead.get(key, 0), loaded, refused, naming, " ".join(notes)])

ws2 = wb.create_sheet("Citations")
H2 = ["Citation #", "Prompt #", "Prompt set", "Prompt", "Platform", "Cited URL (final, after resolution)", "Domain root", "Delivery channel",
      "Demasked from a Google redirect", "Duplicate handling", "Page scrape", "Page title (as scraped)", f"{BRAND} named on page (count)",
      "Published date (where the page carries one)", "Source type", "Classification basis", "Verification notes"]
ws2.append(H2)
for row in cit_rows: ws2.append(row)

ws3 = wb.create_sheet("Read me")
readme = [
    [f"{BRAND} AI Visibility Audit: data completeness and veracity workbook"],
    [f"Prepared by innate c3{(' for ' + CLIENT) if CLIENT else ''}, {RUN_DATE}. Evidence only; the findings are in the separate report."],
    [""],
    ["What was run", f"{len(branded)} branded and {len(unb)} unbranded prompts asked of {', '.join(sorted({r['llm'] for r in rows}))} on {RUN_DATE}; {len(rows)} answers. Every API call carried the same citation request."],
    ["Answers tab", "One row per prompt and platform. The full response is the agent's verbatim output (cells cap at 49,000 characters; the complete text of any longer answer is in the Raw Run file 01_answers.csv)."],
    ["Search ran", "Whether the platform ran a live web search for the answer. 'Retrieval evidence' counts the agent searches it ran, the pages it read and the claims it supported with a source, where the platform exposes them: Claude reports all three; Gemini reports pages read and claims supported; ChatGPT reports claims supported; Perplexity reports pages read and sometimes claims; Grok reports claims supported. A zero means that platform does not expose the figure, not that the step did not happen."],
    ["Delivery channels", "Written in the answer text: a link the agent put in its prose. Google redirect, demasked: Gemini writes links as vertexaisearch redirect URLs; we resolved each to its final page. API citation channel: a source the platform attached as a structured annotation or grounding support rather than in the prose. API retrieval list: a page the platform reports having read, cited through the API. Payload citation: the resolver matched a link whose written form differs from the final URL (tracking parameters, trailing punctuation)."],
    ["Duplicates", "Within one answer, identical final pages count once; this includes the same page cited with and without tracking parameters (utm_*). The Answers tab shows how many repeats were removed per answer; the Citations tab shows it per page."],
    ["Dead links", "Links that returned 404 or 410, failed DNS or refused the connection were dropped and are listed in the Raw Run file 11_dropped_citations.csv."],
    ["Page scrape", f"Every distinct cited page was fetched ({len(ck):,} pages). 'Loaded' means the page text was read and the count of '{BRAND}' on the page recorded. 'Refused' means the site blocked automated reading (bot wall or paywall); a refused page is never counted as not naming Coco."],
    ["Source type", f"Earned media = news, trade and business press. Brand website = {', '.join(d.get('brand_domains') or [])}. Competitor website = the competitors named for the run and the other companies in the category the agents named. Reference, social, community, aggregator, institutional, retail and review sites as labelled. Every cited domain carries a type: {cls.get('_meta', {}).get('classified', '?')} of {cls.get('_meta', {}).get('roots', '?')} domain roots."],
    ["Verification scope", "Types come from the audit's classification pass (the run-time classifier plus one model pass over the remaining roots) with every exception reviewed by an analyst; roots decided by hand or checked live are named in the Classification basis column. Page titles and Coco counts come from the scrape, not from the agents. Counts in this workbook are recomputed from the raw answers, not copied from the platform's summary fields."],
    ["Files behind this workbook", f"{args.payload.split('/')[-1]} (answers and citations), {args.classes.split('/')[-1]} (types), {(args.checks or 'citation_checks in the payload').split('/')[-1]} (page scrape), {args.rawrun_dir.rstrip('/').split('/')[-1]}/ (CSV bundle)."],
]
for r_ in readme: ws3.append(r_)


# ---- styling: the innate c3 tracker look (teal header, Roboto, wrapped, filters, coloured dropdowns) ----
from openpyxl.styles import Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule, FormulaRule
TEAL = "16606E"; GREY = "E8EBEE"; LINE = "D9DEE3"; BAND = "F6F8F9"; INK = "1F2A2E"; MUTED = "5F6368"
F_HEAD = Font(name="Roboto", size=10, bold=True, color="FFFFFF")
F_BODY = Font(name="Roboto", size=10, color=INK)
F_MUTED = Font(name="Roboto", size=9, color=MUTED)
F_TITLE = Font(name="Roboto", size=14, bold=True, color=TEAL)
thin = Side(style="thin", color=LINE); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP_TOP = Alignment(wrap_text=True, vertical="top"); TOP = Alignment(wrap_text=False, vertical="top")
# chip palette: value -> (fill, text)
CHIP = {
    "ChatGPT": ("DCEFE7", "0F5132"), "Claude": ("F4E3D7", "7A3E10"), "Gemini": ("DDE7F7", "1F3F7A"), "Perplexity": ("E3DDF5", "3E2A7A"), "Grok": ("E6E6E6", "333333"),
    "Branded": ("FBE9C7", "6B4A00"), "Unbranded": ("D9ECF2", "0B4F5C"),
    "Yes": ("DCEFE7", "0F5132"), "No": ("F1F3F4", "5F6368"),
    "Loaded": ("DCEFE7", "0F5132"), "Refused (bot wall or paywall)": ("FBE2DD", "8A2A1E"),
    "Written in the answer text": ("DCEFE7", "0F5132"),
    "Written in the answer text as a Google redirect, demasked to the final URL": ("DDE7F7", "1F3F7A"),
    "API citation channel (structured annotation / grounding support)": ("E3DDF5", "3E2A7A"),
    "API retrieval list (page the agent read and cited through the API)": ("FBE9C7", "6B4A00"),
    "Payload citation (form in the answer differs; matched by the resolver)": ("F1F3F4", "5F6368"),
    "Earned media (news, trade, business press)": ("D9ECF2", "0B4F5C"), f"Brand website ({BRAND})": ("FBE9C7", "6B4A00"),
    "Competitor website": ("FBE2DD", "8A2A1E"), "Other company website": ("EFE7DA", "5A4632"),
    "Reference source (Wikipedia, encyclopedias)": ("E6E6E6", "333333"), "Social media": ("E3DDF5", "3E2A7A"),
    "Community / forum": ("F3E0F0", "6B2A62"), "Aggregator, directory or data site": ("E0F0EA", "1F5A44"),
    "Government, regulator or association": ("DDE7F7", "1F3F7A"), "Retail / marketplace": ("FFF2CC", "6B5A00"),
    "Review site": ("FDE9D9", "7A3E10"), "Unrelated company with a similar name": ("F1F3F4", "5F6368"),
}
def style_sheet(sheet, widths, chip_cols, long_cols=(), row_h=None, first_data_row=2):
    for i, w in enumerate(widths, 1): sheet.column_dimensions[get_column_letter(i)].width = w
    for c in sheet[first_data_row - 1]:
        c.font = F_HEAD; c.fill = PatternFill("solid", fgColor=TEAL); c.alignment = WRAP_TOP; c.border = BORDER
    sheet.row_dimensions[first_data_row - 1].height = 42
    last = sheet.max_row; ncol = sheet.max_column
    for r in range(first_data_row, last + 1):
        band = (r - first_data_row) % 2 == 1
        for c in sheet[r]:
            c.font = F_BODY; c.border = BORDER
            c.alignment = TOP if c.column in long_cols else WRAP_TOP
            if band: c.fill = PatternFill("solid", fgColor=BAND)
        if row_h: sheet.row_dimensions[r].height = row_h
    sheet.freeze_panes = sheet.cell(row=first_data_row, column=3)
    sheet.auto_filter.ref = f"A{first_data_row - 1}:{get_column_letter(ncol)}{last}"
    for col, values in chip_cols.items():
        letter = get_column_letter(col); rng = f"{letter}{first_data_row}:{letter}{last}"
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True, showDropDown=False)
        sheet.add_data_validation(dv); dv.add(rng)
        for v in values:
            fill, ink = CHIP.get(v, ("F1F3F4", "5F6368"))
            sheet.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"' + v + '"'], fill=PatternFill("solid", fgColor=fill, bgColor=fill), font=Font(name="Roboto", size=10, bold=True, color=ink)))

PLATFORMS = ["ChatGPT", "Claude", "Gemini", "Perplexity", "Grok"]; SETS = ["Branded", "Unbranded"]; YN = ["Yes", "No"]
CHANNELS = ["Written in the answer text", "Written in the answer text as a Google redirect, demasked to the final URL",
            "API citation channel (structured annotation / grounding support)", "API retrieval list (page the agent read and cited through the API)",
            "Payload citation (form in the answer differs; matched by the resolver)"]
SCRAPE = ["Loaded", "Refused (bot wall or paywall)"]
TYPES = [v for v in LABEL.values()] + ["Unclassified"]
style_sheet(ws, [9, 11, 46, 12, 24, 11, 30, 44, 90, 11, 11, 11, 15, 12, 11, 15, 13, 11, 11, 12, 48],
            {2: SETS, 4: PLATFORMS, 6: YN}, long_cols=(9,), row_h=48)
# the full response sits in a collapsible column group (outline), collapsed by default so the tab opens compact
ws.column_dimensions.group('I', 'I', outline_level=1, hidden=True)
ws.sheet_properties.outlinePr.summaryRight = True
style_sheet(ws2, [9, 9, 11, 44, 12, 64, 22, 46, 12, 28, 24, 44, 11, 16, 34, 48, 22],
            {3: SETS, 5: PLATFORMS, 8: CHANNELS, 9: YN, 11: SCRAPE, 15: TYPES}, long_cols=(6,), row_h=None)
# Read me: title block + two-column definitions
ws3.column_dimensions["A"].width = 26; ws3.column_dimensions["B"].width = 150
ws3["A1"].font = F_TITLE; ws3["A2"].font = F_MUTED
for r in range(4, ws3.max_row + 1):
    ws3.cell(row=r, column=1).font = Font(name="Roboto", size=10, bold=True, color=TEAL); ws3.cell(row=r, column=1).alignment = WRAP_TOP
    ws3.cell(row=r, column=2).font = F_BODY; ws3.cell(row=r, column=2).alignment = WRAP_TOP
    ws3.cell(row=r, column=1).fill = PatternFill("solid", fgColor=GREY)
ws3.append(["Full responses", "The Answers tab opens with the full response column collapsed. Click the + above column I (or the 2 at the top left of the grid) to expand it, and the - to collapse it again; the preview column always shows the first 200 characters. Clicking any cell shows its complete text in the formula bar."])
ws3.append(["How to use", "Every column has a filter (the funnel in the header row): filter the Citations tab by Platform, Source type or Delivery channel, or the Answers tab by Prompt set. Search with Ctrl/Cmd+F. Long cells are wrapped; drag a row border or double-click it to expand. Coloured values are dropdowns, so a re-tag by the client stays within our label set and can be compared with ours."])
r = ws3.max_row; ws3.cell(row=r, column=1).font = Font(name="Roboto", size=10, bold=True, color=TEAL); ws3.cell(row=r, column=1).fill = PatternFill("solid", fgColor=GREY); ws3.cell(row=r, column=1).alignment = WRAP_TOP; ws3.cell(row=r, column=2).font = F_BODY; ws3.cell(row=r, column=2).alignment = WRAP_TOP
wb.move_sheet("Read me", offset=-2)
wb.save(args.out)
print(f"answers {ws.max_row - 1} | citations {ws2.max_row - 1} | written {args.out}")
