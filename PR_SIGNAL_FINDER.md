# PR Signal Finder — Developer Handoff

Live tool that runs an "AI citation audit" for PR pros: user enters what they want their brand to be known for; the app queries Claude + ChatGPT + Gemini with 10 generated prompts, extracts citation URLs from all 30 responses, and returns three target lists (editorial / partnerships / analyst).

## Live deployment

- **Production URL:** https://signal.innatec3.com (`/` 308-redirects to `/citation-audit`)
- **Backup URL:** https://pr-signal-finder.onrender.com
- **Render service:** `pr-signal-finder` (Starter tier, Oregon, $7/mo) — service ID `srv-d829d6btqb8s73cdmi8g`
- **Render dashboard:** https://dashboard.render.com/web/srv-d829d6btqb8s73cdmi8g
- **GitHub repo:** https://github.com/nathanstrauss13/news-analyzer
- **Deploy branch:** `feat/pr-signal-finder` (open PR: https://github.com/nathanstrauss13/news-analyzer/pull/1 — should be merged into `main`, then Render branch switched)
- **Auto-deploy:** on every push to the configured branch
- **Start command:** `gunicorn app:app --timeout 600 --worker-class gthread --threads 4` (the `--timeout 600` + `gthread` is required for SSE — see "Gotchas" below)

## What this codebase actually is

The local repo `/Users/nathanstrauss/Desktop/innate c3/innate apps/ai-citation-audit/` is historically messy:

- There used to be a separate Flask app at `ai-citation-audit/web_app.py` with a `/typeform-audit` UI, deployed to `ai-citation-audit.onrender.com`. **This is deprecated** — Nathan stopped using its UI and runs citation audits via Python scripts directly.
- PR Signal Finder lives in `app.py` at the repo root, deployed to the `news-analyzer` GitHub repo (NOT the `ai-citation-audit` repo).
- Local git has two remotes: `origin` (ai-citation-audit, deprecated) and `news-analyzer` (active).
- Local `main` and `news-analyzer/main` are different histories — always push to `news-analyzer` for production.

When making changes: work in this directory but **push to `news-analyzer` remote** so the deploy picks them up.

## File map (what to edit)

- **`app.py`** (~3,150 lines) — the entire Flask app. PR Signal Finder code is at the bottom, marked with the comment banner `# PR Signal Finder — citation audit tool`. Includes:
 - `CITATION_SUFFIX`, `CITATION_SYSTEM_PROMPT`, `URL_PATTERN` — citation-forcing prompt fragments
 - `INSTITUTIONAL_TLDS`, `ANALYST_DOMAINS`, `EDITORIAL_ORG_ALLOWLIST`, `NON_EDITORIAL_DOMAINS` — classifier sets
 - `classify_citation_domain(domain)` — returns `'analyst' | 'institutional' | 'non_editorial' | 'editorial'`
 - `extract_urls(text)` — regex URL extraction + cleanup + blacklist filter
 - `aggregate_citations(all_responses)` — counts citations by domain, tracks which LLMs/prompts cited each
 - `run_citation_audit(problem_statement, on_progress)` — the pipeline (4 steps)
 - `@app.route('/citation-audit')` — GET renders template, POST returns SSE stream
 - `@app.route('/citation-audit/request-demo')` — captures bespoke-audit lead, emails Nathan via SendGrid
 - `@app.route('/')` — 308-redirects to `/citation-audit` (the old media-analyzer index is moved to `/legacy-index`)
- **`templates/citation_audit.html`** (~1,000 lines) — single-page UI. Dark theme, IBM Plex Sans. Includes:
 - Input form + countdown progress ring driven by SSE events
 - Three result sections: editorial media targets / institutional partnerships / analyst targets
 - "Bespoke audit" CTA with modal lead-capture form
- **`requirements.txt`** — Python deps. Note: `requests>=2.31.0` is required for `google-genai` compatibility.

## Pipeline (run_citation_audit)

```
Step 1: prompts        — Claude Sonnet 4 generates 10 search prompts from the problem statement
Step 2: llm (× 30)     — each prompt sent to Claude / ChatGPT / Gemini with citation-forcing system prompt and CITATION_SUFFIX
Step 3: extract        — URLs regex-extracted from responses, aggregated by domain
Step 4: analysis       — Claude Sonnet 4 produces final JSON report with three target lists
```

Each step calls `on_progress(step, detail, current, total)`. The route turns those into SSE `data:` events that the frontend consumes to update the percentage ring. **Progress weights** (frontend `STEP_WEIGHTS`): prompts 5%, llm 75%, extract 5%, analysis 15%.

## SSE pattern (critical to understand before touching the route)

The `/citation-audit` POST returns a `text/event-stream`. Implementation:
1. A `queue.Queue` is created per request.
2. `run_citation_audit` runs in a `threading.Thread` worker; its `on_progress` callback `.put()`s events to the queue.
3. The generator yields events as they arrive: `data: {"type":"progress",...}\n\n`, then either `data: {"type":"result",...}\n\n` or `data: {"type":"error",...}\n\n` to close.

**Why gunicorn must use `--timeout 600 --worker-class gthread`:** The SSE connection is held open for the full ~5 minute audit. The default sync gunicorn worker (a) times out after 30s, (b) blocks ALL other concurrent requests. The threaded worker with extended timeout solves both.

## Environment variables (Render dashboard → Environment)

Required:
- `ANTHROPIC_API_KEY` — Claude calls
- `OPENAI_API_KEY` — ChatGPT calls
- `GEMINI_API_KEY` — Gemini calls
- `FLASK_SECRET_KEY` — Flask session signing (use the Render "Generate" wand to create a strong one)
- `PYTHON_VERSION=3.11` — pin Python to avoid `pandas`/`Pillow` build failures on 3.13

Optional but recommended:
- `SENDGRID_API_KEY` — lead-capture emails to `nstrauss@innatec3.com`. Without it, leads still save to DB silently. ⚠️ SendGrid sender identity for `no-reply@innatec3.com` must be verified for emails to actually deliver.
- `NEWS_API_KEY` — referenced by news-analyzer legacy code; can be empty.
- `GA_MEASUREMENT_ID` — Google Analytics 4 tag.
- `DATABASE_URL` — defaults to local SQLite (`sqlite:///waitlist.db`). On Render Starter, SQLite is ephemeral — leads written to it survive restarts only if Render persists the disk. For real persistence, attach a Postgres database.

Legacy/unused (carried over from news-analyzer, safe to leave but unused by PR Signal Finder):
`PERPLEXITY_API_KEY`, `PPLX_API_KEY`, `AZURE_OPENAI_*`, `GOOGLE_API_KEY`, `TOGETHER_*`, `OPENROUTER_API_KEY`, `XAI_OPENROUTER_MODEL`, `FIRECRAWL_API_KEY`, `YOU_API_KEY`, `NYT_API_KEY`, `GUARDIAN_API_KEY`.

## Local development

```bash
cd "/Users/nathanstrauss/Desktop/innate c3/innate apps/ai-citation-audit/.claude/worktrees/loving-thompson-6ac87c"
# .env must contain ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY at minimum
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py # serves on http://localhost:5010 (PORT env var) or 5009 default
```

Flask debug auto-reloads on Python changes; **hard-refresh the browser (Cmd+Shift+R) after template changes** — Flask reloads templates server-side but the browser caches the HTML.

## Deploy flow

1. Make changes locally.
2. `git add ... && git commit -m "..."` on `feat/pr-signal-finder` (or whatever branch Render is tracking).
3. `git push news-analyzer feat/pr-signal-finder`
4. Render auto-deploys. Watch at https://dashboard.render.com/web/srv-d829d6btqb8s73cdmi8g/events
5. Verify at https://signal.innatec3.com

Build typically takes 3-5 min (pandas compilation is the slow step). Subsequent builds are faster due to pip cache.

## Known good architectural decisions

- **Three-tier domain classification** (`classify_citation_domain`): editorial outlets get pitched; institutional/.gov/.edu/non-profit-.orgs require partnerships; analyst firms (Gartner, Forrester, etc.) require analyst relations. Each gets a distinct section with a tailored "play" field. Don't merge these without strong reason.
- **Editorial .org allowlist** (`EDITORIAL_ORG_ALLOWLIST`): NPR, ProPublica, PBS, Consumer Reports, etc. — real news .orgs pass through despite the .org TLD. Add to this list when you see legitimate editorial .orgs being mis-classified as institutional.
- **Real-time SSE progress** beats a fake countdown — the ring reflects actual API call status, which keeps users informed during the 5-min wait.
- **`gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`** are the model choices — Sonnet 4 + GPT-4o for breadth, Flash for speed/cost on Gemini.

## Open TODO / Things future sessions might pick up

### Reliability
- [ ] **Verify SendGrid sender identity** for `no-reply@innatec3.com` (Single Sender route in SendGrid → Settings → Sender Authentication). Without this, lead notification emails are sent by SendGrid but rejected at delivery.
- [ ] **Rotate exposed credentials** — during initial setup, the following keys were exposed in browser screenshots: `PERPLEXITY_API_KEY`, `YOU_API_KEY`, `SENDGRID_API_KEY` (the one created during setup), `FLASK_SECRET_KEY` (was set to weak literal `innate-c3-citation-audit-2025`). Rotate all four.
- [ ] **Database persistence** — currently SQLite. Switch to Render Postgres for production durability so leads/shared results aren't lost on redeploy.
- [ ] **Merge [PR #1](https://github.com/nathanstrauss13/news-analyzer/pull/1)** to `main` and switch Render's deploy branch from `feat/pr-signal-finder` to `main`.

### Cleanup
- [ ] **Retire `insights.innatec3.com`** CNAME (still points to `ai-citation-audit.onrender.com` — the deprecated typeform-audit tool).
- [ ] **Suspend/delete** the 10 other Manually Suspended Render services from the dashboard (they're $0 but clutter the dashboard).
- [ ] **`legacy-index` route in `app.py`** — the old media-analyzer index page is still there at `/legacy-index`. Decide whether to keep or remove.

### Features to consider
- [ ] **Email the report** — let users enter their email and get the report PDF + a teaser of the bespoke offering.
- [ ] **Analyst evaluations field** is in the schema (`evaluations_referenced`) but Claude doesn't always populate it reliably. Worth a prompt iteration.

## Roadmap (planned features, design notes)

These are deliberately not in "open TODO" because they're product directions that need design decisions, not just engineering fixes.

### 1. Parallelize the LLM calls (quick win, ~10x speedup)

Currently the 30 LLM calls (10 prompts × 3 providers) run **sequentially** in `run_citation_audit`. Each takes ~10s, total ~5 min. Firing them concurrently could drop this to ~30-45s end-to-end.

**Recommended approach:** `concurrent.futures.ThreadPoolExecutor`. The Anthropic / OpenAI / google-genai SDKs are all sync, so threading is the right fit (asyncio would require switching to the async SDK variants).

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Build a list of (provider, prompt_index, prompt_text) tasks
tasks = []
for pi, prompt_text in enumerate(prompts):
 tasks.append(("Claude", pi, prompt_text))
 tasks.append(("ChatGPT", pi, prompt_text))
 tasks.append(("Gemini", pi, prompt_text))

def run_one(provider, pi, prompt_text):
 # ... existing per-provider try/except logic
 emit("llm", f"Querying {provider} ({pi+1}/10)...", ..., 30)
 return {"llm": provider, "prompt": prompt_text, "response": ..., "citations": ...}

with ThreadPoolExecutor(max_workers=10) as ex:
 futures = [ex.submit(run_one, *t) for t in tasks]
 for f in as_completed(futures):
 all_responses.append(f.result())
 # update progress
```

**Gotchas:**
- **Rate limits.** Claude has the tightest tier-1 TPM limits. With 10 concurrent Claude calls each ~2K input + 2K output, you'll hit ~40K tokens/burst. Check current org tier; tier 2+ should be fine. Use `max_workers=10` initially, drop to 5 if you see 429s.
- **Progress events become non-sequential** — events arrive in completion order, not prompt order. The frontend's `updateProgress` already handles arbitrary `current` values, but the "Querying Claude (3/10)" label will jump around. Consider switching the detail string to "Completed 12/30" instead of per-LLM/per-prompt labels.
- **Thread-safety of progress callback** — `queue.Queue` is thread-safe, so the existing `on_progress → queue.put` pattern works under threading. ✅

### 2. Paid tier ($25 single audit + $100 5-pack)

Product:
- **Free** (current): 10 prompts × 3 LLMs (Claude/GPT/Gemini), top 5 editorial targets + partnerships + analysts.
- **$25 / single audit**: 100 prompts × 5 LLMs, top 25 media contacts.
- **$100 / 5 credits**: same as $25 audit, but $20/audit when bought in bulk. Credits never expire.
- **Bespoke** (contact form): unchanged — Nathan delivers manually for $5K+ engagements.

**Engineering scope:**

a) **Payment provider — Stripe Checkout** is the standard fit. Two products:
- `price_single_audit` = $25, one-time
- `price_credit_pack_5` = $100, one-time
After payment success, Stripe webhook → grant credits to the user's account.

b) **User accounts.** Simplest path: passwordless magic-link login (e.g. via `flask-login` + SendGrid for the link). Don't roll your own password auth.

c) **New SQLAlchemy models:**
```python
class User(db.Model):
 id, email (unique), name, created_at

class CreditBalance(db.Model):
 user_id, credits_remaining, updated_at

class Purchase(db.Model):
 user_id, stripe_session_id, amount_cents, credits_granted, created_at

class AuditRun(db.Model):
 user_id (nullable for free), slug (FK to SharedResult), tier ('free'|'paid'), credits_consumed, prompt_count, llm_count, created_at
```

d) **Parameterize `run_citation_audit`** to accept `prompt_count` (default 10) and `llms` (default `["claude", "openai", "gemini"]`). Free tier hard-codes the defaults. Paid tier reads from user's tier config.

e) **Which 5 LLMs for paid tier?** Currently 3. Two more candidates:
- **Perplexity** (`sonar-pro` or `sonar-reasoning`) — useful because it grounds in web search citations, so its raw output already includes URLs. Add `PERPLEXITY_API_KEY` (already in env).
- **Grok** (xAI) via `OPENROUTER_API_KEY` (already in env, model `x-ai/grok-4`) — increasingly cited by users; worth including for breadth.
- Alternative 5th: **Mistral Le Chat** or **DeepSeek** if you want non-US perspective. Together AI gives access to both.

f) **UX changes:**
- Free audit → result page shows "Want 5x deeper? Sign up + buy credits" CTA. Don't gate the existing free flow.
- Paid audit → user dashboard with audit history, credit balance, "Run new audit" button. The audit UI is the same, just with a parameterized request.
- Stripe Checkout opens in a hosted Stripe page (don't build your own card form).

g) **Cost math** (for pricing sanity):
- Free audit: ~$0.34 in API costs (calculated earlier in chat).
- Paid audit @ 100 prompts × 5 LLMs = 500 calls. Rough: $3.40 in API costs (linear scale), call it $5 with overhead. **$25 price = ~$20 margin per audit.** Healthy.
- $100 pack = 5 audits × $5 cost = $25 total cost. **$75 margin.**

### 3. Share & export functions

**a) Share link (mostly built — just needs surfacing)**

Every audit already creates a `SharedResult` row with a slug. After audit completes, the JSON response includes `data.slug`. The existing `/results/<slug>` route in `app.py` (from news-analyzer/main) renders shared media analysis reports, but **NOT** PR Signal Finder reports — that route renders the old `result.html` template, not `citation_audit.html`.

To add: a new route `/signal/<slug>` that loads `SharedResult.payload`, parses the JSON, and re-renders `citation_audit.html` in "results-only mode" (skip the input form, just show the saved report). Frontend needs a small flag like `?shared=true` or a render mode to suppress the input area.

UI: after audit completes, show a "Share this report" button that copies `https://signal.innatec3.com/signal/<slug>` to clipboard. Optional: list of recent reports on a user dashboard (paid tier only).

**b) PDF export**

Two approaches, in order of preference:

1. **WeasyPrint** (`pip install weasyprint`) — converts HTML+CSS directly to PDF. Since `citation_audit.html` is already a polished design, this gets you 80% of the way. Build a print-friendly variant (`citation_audit_print.html`) with a lighter color scheme for paper, then route `/signal/<slug>.pdf` to render that template through WeasyPrint and return as `application/pdf`. ~50 lines of code total.

2. **ReportLab** — more control but you rebuild the layout from scratch. Avoid unless WeasyPrint isn't producing acceptable output.

Render's Starter tier has WeasyPrint's system deps (cairo, pango) available. Should "just work" on `pip install weasyprint`.

UI: after audit completes (or on the shared-link page), show a "Download PDF" button next to "Share". For paid tier, this is automatic; for free tier, could be a soft gate ("Sign up to download as PDF").

## Gotchas / things that have already bitten

1. **gunicorn timeout** — must be `--timeout 600 --worker-class gthread --threads 4`. Default `gunicorn app:app` kills SSE streams after 30s. This already broke one deploy.
2. **f-string backslashes in Python 3.11** — Python <3.12 disallows backslashes inside `{expression}` parts of f-strings. The `email_link_html` / `report_link_html` are pre-computed outside the f-string for this reason. Don't inline them back.
3. **`requests` package pin** — `google-genai` needs `requests>=2.28.1`. Don't pin it lower; another deploy died on this.
4. **Two GitHub repos confusion** — `nathanstrauss13/ai-citation-audit` is the deprecated repo; `nathanstrauss13/news-analyzer` is where this code lives. Don't push PR Signal Finder changes to `ai-citation-audit`.
5. **`classify_domain` name collision** — news-analyzer/main already had a `classify_domain` function (for article filtering). Our citation classifier is renamed `classify_citation_domain` to avoid the collision. Don't rename it back.
6. **Browser template caching** — Flask reloads templates on each request in debug mode, but the browser doesn't know they changed. Always Cmd+Shift+R when testing template edits.
7. **The `instance/waitlist.db` SQLite file is gitignored** — don't commit it. Production DB data should NOT round-trip through git.

## Useful commands

```bash
# Tail Render logs from Bash (requires `render` CLI; not strictly necessary, web UI works)
# Web log viewer: https://dashboard.render.com/web/srv-d829d6btqb8s73cdmi8g/logs

# Test the live citation-audit endpoint from CLI (returns SSE stream)
curl -N -X POST https://signal.innatec3.com/citation-audit \
 -d "problem_statement=I want ACME Corp to be known as the premier brand of widgets"

# Quick syntax check before pushing
python3 -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

## Memory / context Nathan has set

- Project direction: ai-citation-audit is being built into a **$5K+ consultative offering**, NOT SaaS. PR Signal Finder is the free lead-magnet that opens the conversation; the real money is in bespoke audits delivered via Claude Code.
- Competitive positioning: differentiate FROM Scrunch (which is full SaaS dashboard). The "radically simple, 5 outlets, no dashboard" framing IS the strategy — preserve it.
- Tone: Nathan wants terse responses, no trailing summaries. Direct.

## When starting a new session

1. `cd "/Users/nathanstrauss/Desktop/innate c3/innate apps/ai-citation-audit/.claude/worktrees/loving-thompson-6ac87c"` (or wherever your worktree is)
2. `git status` to see what branch you're on. PR Signal Finder lives on `feat/pr-signal-finder`.
3. `git log --oneline -5` to see recent commits.
4. Read this doc first if it's been a while.
5. For deploys: push to `news-analyzer` remote, not `origin`.
