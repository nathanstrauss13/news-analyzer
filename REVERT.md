# Emergency revert runbook — PR Signal Finder (signal.innatec3.com)

If a deploy breaks the live tool, here are the rollback paths, fastest first.

Render auto-deploys the `mvp` branch of `nathanstrauss13/news-analyzer`.
Service: `pr-signal-finder` (srv-d829d6btqb8s73cdmi8g).

---

## Option 1 — Render one-click rollback  ⚡ FASTEST (~30s, no rebuild)

Use this when the site is broken and you need it back NOW. Render keeps every
prior deploy as a built image; rolling back just re-points to it — it can't
fail a build.

1. Open https://dashboard.render.com/web/srv-d829d6btqb8s73cdmi8g/deploys
2. Find the last **green / Live** deploy from BEFORE the bad one.
   - The last-stable checkpoint is tagged `v0.2-pre-reposition`
     (commit `1c141bb1`, "free tier 30 → 50 responses").
3. Click the `⋯` menu on that deploy → **Rollback to this deploy**.
4. Confirm. Live in ~30 seconds.

This does NOT change git — the bad commit is still on `mvp`. Follow up with
Option 2 to make the revert permanent, otherwise the next push re-deploys
the broken state.

---

## Option 2 — Clean git revert  ✅ PERMANENT (~3 min rebuild)

Undoes the repositioning commit while KEEPING it in history (so you can retry
later). This is the "considered rollback" — run the helper script:

```bash
cd "<repo>/news-analyzer"        # the loving-thompson worktree
./revert_reposition.sh
```

Or do it by hand:

```bash
git revert --no-edit fda2faddc   # the repositioning commit
git push news-analyzer mvp        # Render auto-deploys the reverted state
```

To revert MORE than just the repositioning (e.g. back to the full
pre-reposition checkpoint), revert the range:

```bash
git revert --no-edit 1c141bb17..HEAD   # undoes everything after v0.2
git push news-analyzer mvp
```

---

## Option 3 — Hard reset to the checkpoint  ⚠️ DESTRUCTIVE (last resort)

Only if history is tangled and you want the branch to exactly match the
checkpoint. This rewrites `mvp` and discards everything after v0.2.

```bash
git reset --hard v0.2-pre-reposition
git push --force news-analyzer mvp
```

Avoid unless Options 1 + 2 won't do — force-push loses the newer commits
from the branch (they survive in the tags / reflog but it's messy).

---

## Checkpoints (git tags)

| Tag | Commit | What it is |
|-----|--------|------------|
| `v0.2-pre-reposition` | `1c141bb1` | Last stable before the AI-mindshare reframe. 5-LLM, flat top-10 list. |
| `v0.1-paid-tier-checkpoint` | (earlier) | Full paid-tier Stripe flow, before the MVP fork. |

After any deploy you've confirmed healthy, tag it so it becomes a future
rollback target:

```bash
git tag -a v0.x-description <commit> -m "what this is + that it's verified healthy"
git push news-analyzer v0.x-description
```

---

## How to confirm the site is healthy after any deploy

```bash
# 1. Page loads
curl -s -o /dev/null -w "%{http_code}\n" https://signal.innatec3.com/citation-audit   # want 200

# 2. An existing report still renders (no 500 from a template change)
curl -s -o /dev/null -w "%{http_code}\n" https://signal.innatec3.com/signal/46ee27c2c4  # want 200

# 3. Render logs show no boot errors / OOM
#    dashboard → Logs, look for "Your service is live" + no Traceback/restart loop
```

If a fresh audit is needed to fully confirm, run one from a
FREE_AUDIT_BYPASS_IPS-allowlisted IP so it doesn't burn the daily cap.
