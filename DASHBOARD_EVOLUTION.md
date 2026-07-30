# Signal Finder → AI Citation Audit: evolution plan

Direction (Nathan, July 30 2026): evolve PR Signal Finder into a traditional
GEO dashboard covering **branded and unbranded** queries, in the style of the
PUIG audit. Positioning: a free AI citation audit for communicators that goes
beyond "who to pitch and what to publish." Eventually replaces Signal Finder;
the classic product is archived and revertible.

## Phase 0 — archive (DONE, July 30)

- Tag `signal-finder-v1` at `0c007b62d`, pushed to the remote.
- Template snapshot: `templates/archive/signal_finder_v1/citation_audit.html`.
- `REVERT.md` carries the checkpoint row and restore commands.
- Frozen report slugs are unaffected by any restore: they render from
  persisted JSON, not from live analysis.

## Phase 1 — new dashboard on existing data (DONE, July 30)

`/dashboard/<slug>?key=<OPERATOR_KEY>` renders any stored audit through the
`tools/audit_dashboard` kit. Operator-gated (404 without the key), read-only
over persisted payloads (safe on frozen slugs), no public surface changed.
Rows split branded vs unbranded using the kit's own brand-form matching, so
the partition agrees with how the dashboard counts mentions.

Validated live on lumen, spotify, gap, blackrock, chime (50 answers each,
560-890 KB pages, owned-citation counts reconcile against independent
recounts from the stored JSON: lumen 10, spotify 26, gap 0).

**Known gap, by design:** every existing production audit uses category
(unbranded) prompts only, so previews render organic-only. The branded half
of the dashboard stays empty until Phase 2 generates branded prompts. This is
the single biggest reason to do Phase 2 before any public cutover.

## Phase 2 — branded + unbranded prompt generation (NEXT, needs decisions)

The product currently generates ~10 category prompts per audit. The new
dashboard wants both halves.

Proposed shape:
- Generate two labeled sets per audit: **branded** (name the brand: what is X,
  is X worth it, X vs competitor, reviews of X) and **unbranded** (category
  questions where the brand may or may not appear).
- Store the split on the report payload so the dashboard partition is
  authoritative rather than inferred from prompt text.
- Persist prompt sets exactly as run (already shipped for inbound rows).

Open decisions for Nathan:
1. **Free-tier size and cost.** Today: 10 prompts x 5 assistants = 50 answers.
   A 10 branded + 10 unbranded split doubles runtime and API cost per free
   audit. Options: keep 50 total (5+5 prompts, thinner each half), go to 100
   (fuller picture, ~2x cost and ~2x runtime on the 2 GB instance, which also
   raises the concurrency-guard question), or default 50 with an operator
   flag for 100 on qualified leads.
2. **Domain input.** Branded analysis needs the brand's domain to compute
   owned share. Today the homepage asks brand + what they want to be
   recommended for. Adding a domain field is a UX change but unlocks the
   owned-media half of the dashboard (the PUIG donut, page-kind breakdown).
3. **Competitor input.** The kit shows a competitor set. Options: auto-derive
   from the answers (what the current product does), let the user name 3-5,
   or both (auto-derive, user edits).

## Phase 3 — search-page UX (after Phase 2 decisions)

Revise the homepage for the communicator-framed free audit: brand, domain,
category/positioning, optional competitors, with the promise stated as the
new dashboard delivers (presence, prominence, source mix, owned pages cited)
rather than the PR-pitch framing. Keep the embed autostart contract
(`/?autostart=1&brand=&focus=`) working or version it deliberately with the
marketing lane, since innatec3.com drives that widget.

## Phase 4 — cutover

Only after Phases 2-3 are proven on real runs:
- `/signal/<slug>` keeps rendering existing reports in the classic view
  (permanent, frozen-slug safe).
- New audits render the new dashboard.
- Roll back at any point with `signal-finder-v1` per REVERT.md.
