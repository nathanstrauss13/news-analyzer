# Correction Seeker (operator tool, MVP)

Fact-consistency audit pipeline per `CORRECTION_SEEKER_BRIEF.md`. Where PR
Signal Finder measures whether a brand shows up in AI answers, this verifies
whether the **facts** AI states are right, traces each stale or wrong figure
to the source page feeding it, and produces a prioritized correction queue
with drafted notes. Operator-generated, never DIY — same posture as
`tools/audit_dashboard`.

## Run

```bash
cd tools/correction_seeker
python3 correction_seeker.py all --config netflix.config.json          # full
python3 correction_seeker.py all --config netflix.config.json --limit 1  # smoke
# stages also run separately: collect | analyze | report
```

Outputs land in `runs/<slug>/`: `raw.json` (all responses + citations),
`analysis.json` (claims, variants, queue — everything re-derivable),
`evidence/*.txt` (fetched page snapshots, timestamped at detection),
`report.html` (the deliverable).

Collectors come from the main checkout's `platforms/` (STANDARD_5, grounded);
`.env` is read from there too. Claim extraction uses Claude
(`CS_EXTRACT_MODEL`, default `claude-sonnet-5`) and is **quote-gated**: a
claim only counts if its quote appears verbatim in the raw answer.

## Config anatomy (see netflix.config.json)

Per fact: `kind` (money | count | year | text), `truth` (value + display +
as_of + primary source), `history` (earlier reported figures, for vintage
matching), `tolerance_pct`, and optionally:

- `alt_bases` — figures that are **correct on a different basis** (TTM,
  guidance). Without these, a trailing-twelve-month figure would be classed
  incorrect. Populate for any fact with quarterly cadence.
- `discontinued: true` — metric the company stopped reporting (Netflix
  memberships). Any current-sounding figure is at best stale; unmatched
  figures are unverifiable, never "incorrect" (we cannot prove them wrong).
- text facts use `accept_tokens` (OR-list of AND-token sets) and history
  entries with `tokens` (e.g. a former CEO's name).

## Taxonomy (fixed, sober, defensible)

current / current_basis / stale_dated / stale_as_current / incorrect /
unverifiable. Only `stale_as_current` and `incorrect` enter the correction
queue. "Accurate as of an earlier date" beats "the site is wrong" everywhere
in output copy. No em dashes in client-facing copy.

## Ground-truth discipline

Every config value must come from a **fetched primary source** (EDGAR XBRL,
10-K/10-Q text, 8-K exhibits), never from memory. The Netflix config's
provenance: revenue from XBRL `Revenues`, employees from 10-K text,
memberships from the FY2024 10-K table, guidance from the Q2 2026
shareholder letter (Ex-99.1). Rebuild the same way for any client.

## Known limits (MVP)

- Word-unit ranges ("$39 billion to $40 billion") classify by the low
  endpoint; digit-adjacent ranges require both endpoints to agree.
- Bot-walled or PDF sources go to the manual-verify queue, not the trace.
- One company per run; one run at a time (collector rate limits).
- BlackRock never appears in demos (brief rule); Netflix is the public
  sample subject.
