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

## Phase 2 — branded + unbranded prompt generation (DONE, July 30)

Decisions (Nathan): 10x5 total (5 branded + 5 unbranded — cost unchanged),
domain inferred (existing `_resolve_brand_domains` pipeline), competitors
auto-derived (existing behavior).

Shipped in `59ffdcd3e`:
- `_generate_audit_prompts` emits labeled halves. Branded = how AI describes
  the brand when asked directly; unbranded = unprompted category mindshare
  (original fairness doctrine intact; no competitor names in either half).
- **Scope contract** (`metrics_scope = "unbranded_only"` on the payload):
  every mindshare/SoV/media metric computes over the unbranded half only,
  keeping numbers unprompted and comparable with all past audits. Enforced
  at all four recount surfaces: fresh pipeline, rerender recount, QA gate,
  dashboard partition. Delivery stats (completion_rate) stay full-scope.
- Payload carries both halves (`all_responses`) + labels (`prompt_sets`).
- Curated-prompt (prompts_override) runs are unlabeled and unchanged.

Verified live July 30 on a real production run (slug `387c369bef`,
Patagonia test): 5+5 generated with clean discipline, 50/50 delivered,
completion 1.0, metrics denominator 25, brand recount 25/25 unbranded
(independently reproduced), QA client_ready, classic report renders,
`/dashboard/387c369bef` renders BOTH halves — the first audit born with a
populated branded read.

Note for operators: the audit runs inside the SSE request stream — a
disconnecting client kills the run (janitor flips the row to errored).
Keep the connection open for scripted runs.

## Phase 3 + 4 — search-page UX and cutover (DONE, July 30, `92f152ac7`)

Homepage reframed for the communicator-framed free audit: "Free AI Citation
Audit / Built for communicators" eyebrow, two-question hero promise (does AI
surface you unprompted; how does it describe you when asked), preview card
reworded to the dashboard's read (sources, prominence, owned pages). Form
unchanged (brand + focus; domain inferred, competitors auto-derived). The
"PR Signal Finder" name is retired from the title and hero. Embed autostart
contract (`/?autostart=1&brand=&focus=`) untouched and verified present.

Cutover shipped in the same commit:
- `/signal/<slug>` renders the GEO dashboard (with a report CTA band:
  book 30 min / JSON export / run another) for any payload carrying
  `prompt_sets` — i.e. every new self-serve audit from phase 2 onward.
- Legacy and frozen payloads (no prompt_sets) keep the classic template
  permanently; `?classic=1` shows the classic view of a split audit;
  dashboard render failures fall back to classic automatically.
- Verified live (11-point matrix): new copy + autostart on the homepage;
  `387c369bef` publicly renders the dashboard with CTA; `?classic=1`
  serves the classic template; legacy (`lumen`) and frozen (`gap`) slugs
  unchanged.

Rollback at any point: `signal-finder-v1` tag per REVERT.md.

## Follow-ups (not yet done)

- PDF export of the dashboard view (classic PDF route still renders the
  classic template; fine during transition).
- Completion-email copy still says "report"; consider describing the new
  dashboard sections.
- Marketing lane: embed works unchanged, but innatec3.com copy around the
  widget may want the new two-question framing.
- Consider caching the dashboard render per slug (currently recomputed per
  view; acceptable at current traffic).
