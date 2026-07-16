# AI Visibility Intelligence dashboard kit (operator tool)

Successor to the v17/v18 AEO dashboards (innatec3.com/pali, /bain, /lululemon):
ONE template for any industry and any buyer (GTM, BD, founder, comms), with
branded and/or unbranded prompt sets, in the current dark-premium look.
Not DIY: Nathan + Claude generate these together per prospect.

## Inputs (auto-detected)
- Raw audit kit JSON (`~/Desktop/Norwest/raw_audit.py` output: brand/key/rows)
- Production slug JSON (`signal.innatec3.com/signal/<slug>.json`)

## Run
    python3 build_dashboard.py --config <client>.config.json
    # -> <slug>_dashboard.html (self-contained; host or print)

## Config (see phasecraft.config.json for the worked example)
- brand, aliases, owned_domains, category, competitors[{name,aliases,domains}]
- branded_data / organic_data: one or both paths
- exec_summary: null = sober auto-draft with real numbers; or list of
  HTML paragraphs to override (curated voice)
- recommendations: [{title, body}] overrides the deterministic gap list
- exclude_sources: junk domains to drop entirely (e.g. bingx.com)

## What it computes (all deterministic, reproducible from the appendix)
- Brand + competitor presence: word-boundary, alias-aware, lowercase-dominance
  guard, assistant names excluded (the Signal Finder counting lessons)
- Citations: utm/fragment-stripped, redirect-resolved input honored,
  registrable-root grouping, assistant self-references excluded
- Source classification: owned / competitor / editorial / institutional /
  reviews-and-talent / community / social / reference
- Owned page-kinds (homepage vs blog/research vs careers...) + the
  homepage-vs-deep-content insight note
- Branded-vs-organic contrast panel (the "104 vs 0" pitch visual)
- Per-assistant read, opportunity gaps, full interactive query appendix

## PDF
    chrome --headless=new --print-to-pdf=out.pdf --virtual-time-budget=20000 \
      --run-all-compositor-stages-before-draw --no-pdf-header-footer file.html
(One embedded Helvetica subset for exotic glyphs in raw appendix text is
expected and safe; the 0-fallback rule targets UNembedded fonts.)

## House style (enforced in template copy)
Sober, small-sample, directional-not-verdict; no em dashes; buyer-neutral
"AI visibility" language; every number reproducible from the appendix.
