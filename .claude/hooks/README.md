# Agent hooks

Hooks are run by the Claude Code harness, not by the agent, which is what makes them the only
part of a workflow rule that cannot be forgotten. Skill text applies when the skill happens to be
loaded; a hook fires either way.

| Script | Event | Fires on | Effect |
|---|---|---|---|
| `commit-quality-gate.sh` | `PreToolUse` | `git commit` | **Blocks** if `pre-commit` fails or the staged diff adds debug leftovers. Mechanical only. |
| `push-review-gate.sh` | `PreToolUse` | `git push` | **Blocks** until the unpushed commits have had a simplify/self-review pass |
| `pr-created-review-poller.sh` | `PostToolUse` | `gh pr create` | Injects the instruction to arm the self-deleting Copilot review poller |

## Why the judgment gate is on push, not commit

Simplify and self-review are worth doing over the whole set of new work at once. Per commit they
are noise — and structurally blind: a helper added in commit 2 and orphaned by commit 6 is only
visible across the range. So `commit-quality-gate.sh` keeps the cheap mechanical checks, and
`push-review-gate.sh` carries the judgment pass over `@{upstream}..HEAD` (or
`origin/amd-integration..HEAD` on a first push).

Two exemptions, both **fail-closed** — a missed signal costs an extra review, never a skipped one:

1. **`Review-response: #<PR>` trailer.** Commits answering an automated review have already been
   scrutinised. All-or-nothing over the range: one untagged commit re-arms the gate.
2. **Acknowledged head.** Keyed to the exact SHA in `$GIT_DIR/fib-push-review`, so appending
   commits after a review re-arms the gate instead of riding the stale acknowledgement.

The acknowledgement records *that a pass happened*, not that it was any good — nothing here can
verify that. Its job is to make the pass land before the push rather than after.

## Activation is opt-in, per checkout

The scripts are checked in; the wiring is not. `.claude/settings.local.json` is gitignored, so each
checkout — the main clone **and every worktree** — needs its own copy of the block below. That is a
consequence of how the harness resolves settings (relative to the project root), not a choice.

```jsonc
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/commit-quality-gate.sh",
            "timeout": 120,
            "statusMessage": "Running commit quality gate..."
          },
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/push-review-gate.sh",
            "timeout": 30,
            "statusMessage": "Checking push review gate..."
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/pr-created-review-poller.sh",
          "timeout": 30
        }]
      }
    ]
  }
}
```

Merge this into any existing `settings.local.json` — do not overwrite the file; it usually already
holds `permissions`. After editing, open `/hooks` once or restart: the settings watcher only tracks
directories that had a settings file when the session started, so a newly created
`settings.local.json` is not picked up mid-session.

To move the gate from personal to fork-wide, lift the same block into `.claude/settings.json`
(checked in). Consider whether a hook that blocks `git commit` should apply to contributors who
did not ask for it.

## Design notes

**The scripts filter, not the `matcher`.** Both are wired to the broad `Bash` matcher and decide
internally whether the command is relevant. The narrower `"if": "Bash(git commit*)"` filter would
skip compound commands like `cd sub && git commit -m x`, and a gate with a known hole is worse than
one that costs a few milliseconds per Bash call.

**Invoked as script files, not inline shell.** Hook commands run through `$SHELL`, which is `zsh`
here, and zsh and bash disagree on details — notably backslash handling in the builtin `echo`,
which silently corrupts JSON containing `\n`. A path to a file with a `#!/usr/bin/env bash`
shebang parses identically under either shell and then runs under a known one.

**Matchers are POSIX ERE, with no `\b`.** Word-boundary `\b` is a GNU/ugrep extension that POSIX
ERE does not define. On a grep without it the matcher never fires and the gate silently stops
gating — strictly worse than no gate, since it still looks installed. The patterns use explicit
`[[:space:]]` and `[^[:alnum:]_-]` boundaries instead, and are covered by a case table including
compound commands, global flags (`git -C sub commit`), and near-misses (`git commitfoo`,
`gh pr createx`).

**Failures are silent, never fatal — except when they hide the answer.** The scripts exit 0 on
their own internal errors, because a hook that crashes must not wedge every commit in the repo.
The exception is a range that cannot be resolved: `push-review-gate.sh` denies rather than
treating `git rev-list` failure as "nothing to publish", since that would let an unreviewed push
through precisely when the tooling is confused.

## What these hooks cannot do

Neither gate can verify that a simplify or self-review pass actually happened — only that one was
claimed. The rules themselves live in
[CLAUDE.md](../../CLAUDE.md#quality-gates-commit-push-pr), in context every session, and are
restated in the block messages.

`commit-quality-gate.sh` skips `pre-commit` when `.git/hooks/pre-commit` exists, since
`pre-commit install` already runs it on every commit — including commits made outside Claude Code,
which no hook here can see. Running `pre-commit install` is the better coverage; this hook's
unique contribution is the debug-leftover scan and the agent-facing block message.

A `"type": "agent"` hook could genuinely inspect the transcript for evidence of a review. At
per-commit frequency that was too expensive to justify; at per-**push** frequency it is cheap —
one model call per push. Worth revisiting if the acknowledgement proves too easy to rubber-stamp.
