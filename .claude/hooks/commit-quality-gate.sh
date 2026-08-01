#!/usr/bin/env bash
# PreToolUse(Bash) gate — block `git commit` when the mechanical quality checks fail.
#
# Scope: mechanical checks only — lint/format and debug leftovers. The judgment half of the
# gate (simplify, self-review) runs once per push over the whole unpushed range, not here; see
# push-review-gate.sh. Commits are too frequent for a judgment gate to be anything but noise,
# and reviewing one commit at a time cannot see across the set anyway.
#
# Contract: hook input JSON arrives on stdin. Emit a PreToolUse deny to block the commit, or
# exit 0 silently to let it through. Never exit non-zero on our own errors — a broken hook
# must not wedge every commit in the repo.
set -uo pipefail

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0

# Match real commit invocations, including compound ones (`cd x && git commit -m ...`).
grep -Eq '(^|[;&|]) *git\b[^;&|]*\bcommit\b' <<<"$cmd" || exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

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

# `git commit -a` stages tracked changes itself, so the file set differs from `--cached`.
if grep -Eq '\bcommit\b[^;&|]*(-[a-zA-Z]*a|--all)\b' <<<"$cmd"; then
  mapfile -t files < <(git diff --name-only --diff-filter=ACMR HEAD)
  diff_args=(HEAD)
else
  mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
  diff_args=(--cached)
fi

# Nothing to check — let git produce its own "nothing to commit" error.
((${#files[@]})) || exit 0

problems=()

# 1. Lint / format — skipped when the git-level hook is present. `pre-commit install` writes
#    .git/hooks/pre-commit, which already runs on every commit (including ones made outside
#    Claude Code, which this hook cannot see), so running it here too would just double commit
#    latency. Prefer the git hook: install it with `pre-commit install`.
if [[ ! -x "$(git rev-parse --git-common-dir)/hooks/pre-commit" ]] \
  && [[ -f .pre-commit-config.yaml ]] && command -v pre-commit >/dev/null 2>&1; then
  if ! out=$(pre-commit run --files "${files[@]}" 2>&1); then
    problems+=("pre-commit failed:"$'\n'"$out")
  fi
fi

# 2. Debug leftovers in added lines only — deliberately narrow, to stay false-positive free.
#    Bare `print(` is excluded: this repo's scripts/ use it legitimately.
added=$(git diff "${diff_args[@]}" -U0 -- "${files[@]}" 2>/dev/null | grep '^+' | grep -v '^+++')
if hits=$(grep -nE 'breakpoint\(\)|\bpdb\.set_trace\b|console\.log\(|\bdebugger;' <<<"$added"); then
  problems+=("debug leftovers in the staged diff:"$'\n'"$hits")
fi

((${#problems[@]})) || exit 0

# Command substitution strips trailing newlines, so the blank line is written literally here.
deny "$(printf '%s\n\n' "${problems[@]}")

Blocked by the commit quality gate.

Fix the above, then re-stage — pre-commit hooks that reformat leave the fix unstaged — and
commit again."
