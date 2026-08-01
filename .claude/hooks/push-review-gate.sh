#!/usr/bin/env bash
# PreToolUse(Bash) gate — require a simplify/self-review pass over the unpushed commits before
# they leave the machine.
#
# Push rather than commit, for two reasons: commits are frequent enough that a judgment gate on
# each one is pure noise, and a per-commit review structurally cannot see across the set — the
# helper added in commit 2 that nothing uses by commit 6 is only visible over the whole range.
#
# Two deliberate exemptions, both fail-closed (a missed signal means an extra review, never a
# skipped one):
#   1. Every new commit carries a `Review-response:` trailer — these answer an automated review
#      that already scrutinised the code, so re-reviewing them is busywork.
#   2. This exact head SHA was already acknowledged. Keyed to the SHA, so appending commits
#      after a review re-arms the gate rather than riding on the old acknowledgement.
set -uo pipefail

# POSIX ERE only — `\b` is a GNU/ugrep extension, and a matcher that quietly fails to match is a
# gate that quietly does not gate. See the fuller note in commit-quality-gate.sh.
readonly PUSH_RE='(^|[;&|])[[:space:]]*git([[:space:]]+[^;&|]*)?[[:space:]]+push([^[:alnum:]_-]|$)'

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0

grep -Eq "$PUSH_RE" <<<"$cmd" || exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0
git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit 0
head=$(git rev-parse HEAD 2>/dev/null) || exit 0
marker="$git_dir/fib-push-review"

deny() {
  jq -n --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

# What this push would publish. No upstream means a first push, so everything since the base
# branch is new.
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null); then
  range="$upstream..HEAD"
else
  range="origin/amd-integration..HEAD"
fi

# Fail closed if the range cannot be resolved. Discarding the error and treating it as an empty
# range would silently allow an unreviewed push whenever the base ref is missing locally — a
# fresh clone that never fetched amd-integration, a renamed remote, a deleted upstream.
if ! rev_out=$(git rev-list "$range" 2>&1); then
  deny "Cannot determine what this push would publish — \`git rev-list $range\` failed:

$rev_out

Refusing rather than guessing: an unresolvable range must not read as \"nothing new\".
Fetch the base (\`git fetch origin amd-integration\`) or set the branch upstream, then retry."
fi

[[ -n "$rev_out" ]] || exit 0   # genuinely nothing new to publish
mapfile -t shas <<<"$rev_out"

# Exemption 1 — the whole range answers an automated review.
for s in "${shas[@]}"; do
  git log -1 --format=%B "$s" 2>/dev/null | grep -qiE '^Review-response:' || { unreviewed=1; break; }
done
[[ -z "${unreviewed:-}" ]] && exit 0

# Exemption 2 — this exact head was already acknowledged.
[[ -f "$marker" && "$(cat "$marker" 2>/dev/null)" == "$head" ]] && exit 0

deny "$(git log --format='  %h %s' "$range" 2>/dev/null)

The ${#shas[@]} commit(s) above would be published without a simplify/self-review pass over
the combined range.

Review \`git diff $range\` as one diff, for:

  1. Simplify — dead code, scratch code, debug-only comments, unused imports. Keep the
     comments that carry the *why*, hidden constraints, or non-obvious invariants.
  2. Correctness.

Fixes produce new commits, which re-arms this gate — that is expected, not a loop. When the
range is clean, record the acknowledgement and retry the push:

    git rev-parse HEAD > $marker

This records acknowledgement, not proof: nothing here can verify the review actually happened.
It exists so the pass lands before the push rather than after.

Exempt: a range where every commit message carries a \`Review-response: #<PR>\` trailer."
