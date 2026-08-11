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

# Command-segment matchers, POSIX ERE only. `\b` is a GNU/ugrep extension that POSIX ERE does
# not define; a matcher that quietly fails to match is a gate that quietly does not gate, which
# is the one failure mode this file exists to prevent. Explicit classes behave the same under
# any grep.
#   (^|[;&|])           start of a command segment, so `cd x && git commit` still matches
#   git([[:space:]]+…)? optional global flags, e.g. `git -C sub commit`
#   ([^[:alnum:]_-]|$)  right boundary, so `git commitfoo` does not match
readonly COMMIT_RE='(^|[;&|])[[:space:]]*git([[:space:]]+[^;&|]*)?[[:space:]]+commit([^[:alnum:]_-]|$)'
# `a` may sit anywhere in a short-option cluster: -a, -am, -ma are all "commit all". Two
# subtleties, each learned from a false result:
#   - `commit` must appear before the option, in the same segment. Without it, the `-a` of
#     `git branch -a && git commit -m x` reads as commit-all and the gate scans worktree changes
#     the commit will not include — a false deny, and a gate that cries wolf gets switched off.
#   - The right boundary is `[^[:alnum:]_-]`, not whitespace: a cluster can butt straight up
#     against the message quote, and `git commit -am"msg"` must not read as index-only.
# `--amend`/`--author` stay excluded either way — the alternative needs a space before its `-`.
readonly COMMIT_ALL_RE='(^|[;&|])[[:space:]]*git([[:space:]]+[^;&|]*)?[[:space:]]+commit([[:space:]]+[^;&|]*)?[[:space:]]+(-[[:alnum:]]*a[[:alnum:]]*|--all)([^[:alnum:]_-]|$)'

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0

grep -Eq "$COMMIT_RE" <<<"$cmd" || exit 0

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

# `git commit -a` publishes the union of the index and tracked worktree changes. Neither diff
# alone covers that union: `--cached` misses unstaged edits, and `HEAD` misses a path that was
# staged and then reverted in the worktree — whose staged content still gets committed. Scan both.
if grep -Eq "$COMMIT_ALL_RE" <<<"$cmd"; then
  scopes=(HEAD --cached)
else
  scopes=(--cached)
fi

files=()
for scope in "${scopes[@]}"; do
  while IFS= read -r f; do [[ -n "$f" ]] && files+=("$f"); done \
    < <(git diff --name-only --diff-filter=ACMR "$scope" 2>/dev/null)
done
((${#files[@]})) || exit 0   # nothing to check — let git raise its own "nothing to commit"
mapfile -t files < <(printf '%s\n' "${files[@]}" | sort -u)

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

# 2. Debug leftovers in added lines only — scanning whole files would flag pre-existing code the
#    commit never touched. Deliberately narrow to stay false-positive free: bare `print(` is
#    excluded because scripts/ uses it legitimately, and a gate that cries wolf gets disabled.
added=""
for scope in "${scopes[@]}"; do
  added+=$(git diff "$scope" -U0 -- "${files[@]}" 2>/dev/null | grep '^+' | grep -v '^+++')$'\n'
done
if hits=$(grep -nE 'breakpoint\(\)|(^|[^[:alnum:]_.])pdb\.set_trace|console\.log\(|(^|[^[:alnum:]_])debugger;' <<<"$added" | sort -u -t: -k2); then
  problems+=("debug leftovers in the staged diff:"$'\n'"$hits")
fi

((${#problems[@]})) || exit 0

# Command substitution strips trailing newlines, so the blank line is written literally here.
deny "$(printf '%s\n\n' "${problems[@]}")

Blocked by the commit quality gate.

Fix the above, then re-stage — pre-commit hooks that reformat leave the fix unstaged — and
commit again."
