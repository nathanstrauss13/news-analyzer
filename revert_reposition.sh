#!/usr/bin/env bash
#
# revert_reposition.sh — one-command rollback of the AI-mindshare repositioning.
#
# Cleanly reverts the repositioning commit (keeps it in history so you can
# retry later) and pushes to the Render-watched branch. Render auto-deploys
# the reverted state in ~2-3 min.
#
# For an INSTANT rollback (no rebuild) use Render's dashboard one-click
# instead — see REVERT.md, Option 1.
#
set -euo pipefail

REPOSITION_COMMIT="fda2faddc"   # "reposition as AI mindshare intelligence..."
REMOTE="news-analyzer"
BRANCH="mvp"

echo "==> Reverting the repositioning commit ${REPOSITION_COMMIT} on ${BRANCH}"

# Safety: make sure we're on the right branch + clean tree.
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "$BRANCH" ]; then
  echo "ERROR: on branch '$current_branch', expected '$BRANCH'. Aborting." >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree is dirty. Commit or stash first. Aborting." >&2
  exit 1
fi

# Confirm the commit exists.
if ! git cat-file -e "${REPOSITION_COMMIT}^{commit}" 2>/dev/null; then
  echo "ERROR: commit ${REPOSITION_COMMIT} not found. Aborting." >&2
  exit 1
fi

git revert --no-edit "${REPOSITION_COMMIT}"
echo "==> Revert committed. Pushing to ${REMOTE}/${BRANCH}..."
git push "${REMOTE}" "${BRANCH}"

cat <<'DONE'

==> Done. Render will auto-deploy the reverted state in ~2-3 min.

   Verify health once it lands:
     curl -s -o /dev/null -w "%{http_code}\n" https://signal.innatec3.com/citation-audit

   If you need it back FASTER than a rebuild, use Render's one-click
   rollback instead (REVERT.md, Option 1) — that's instant.

   The repositioning commit is still in history; to re-apply it later:
     git revert --no-edit <this-revert-commit>
DONE
