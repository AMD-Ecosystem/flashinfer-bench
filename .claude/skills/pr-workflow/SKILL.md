---
name: pr-workflow
description: How to create and edit PRs on the AMD-Ecosystem/flashinfer-bench GitHub repo (the ROCm fork) — fail-closed target/base safeguards, the never-push/PR-from-amd-integration rule, the ask-before-push rule, gh CLI quirks, the pre-PR quality gate, the automated-review loop, and PR-description conventions.
---

# PR Workflow (AMD-Ecosystem/flashinfer-bench)

> This repo is the ROCm-only fork of `flashinfer-ai/flashinfer-bench`. The GitHub repo is
> `AMD-Ecosystem/flashinfer-bench`, base branch `amd-integration`. All `gh` commands below target it.

## CRITICAL: PR target safeguard (fail-closed)

`AMD-Ecosystem/flashinfer-bench` is a **GitHub fork** of `flashinfer-ai/flashinfer-bench` (the true
upstream). Because of this, `gh pr create` defaults the **base repository** (the PR target) to the
fork parent `flashinfer-ai/flashinfer-bench` unless explicitly overridden. **A PR must NEVER be
opened against `flashinfer-ai/flashinfer-bench`.**

All PRs go to **`AMD-Ecosystem/flashinfer-bench`**, base branch **`amd-integration`**.

**Before ANY `gh pr create`, run this pre-flight check and ABORT if it fails:**

```bash
gh repo set-default --view   # MUST print exactly: AMD-Ecosystem/flashinfer-bench
```

If it prints anything else (or errors), STOP — do not create the PR. Report the mismatch to the user
instead. Never guess the target.

**Always pass the target and base explicitly** — never rely on gh defaults:

```bash
gh pr create --repo AMD-Ecosystem/flashinfer-bench --base amd-integration \
  --title "<title>" --body "$(cat /tmp/pr_body.md)"
```

If the resolved owner of `--repo` is ever `flashinfer-ai`, abort. It is always better to fail to
raise a PR and explain why than to raise one against upstream.

### One-time setup after a fresh clone

Local config (not checked in), redo per clone:

```bash
git remote -v                                       # origin should be AMD-Ecosystem/flashinfer-bench; there must be NO flashinfer-ai remote
gh repo set-default AMD-Ecosystem/flashinfer-bench  # pin gh base repo so it doesn't fall back to the fork parent
```

If a remote pointing at `flashinfer-ai/flashinfer-bench` exists, remove it: `git remote remove <name>`.

## CRITICAL: never push or PR from `amd-integration` (fail-closed)

`amd-integration` is the **base** branch — it must never be the **head** of a PR, and you must
**never `git push` to the remote `amd-integration`**. To ship any change, create a topic branch off
`origin/amd-integration` (`git checkout -b <topic> origin/amd-integration`) and push/PR that branch.
Before pushing a branch for a PR or running `gh pr create`, check the current branch:

```bash
git branch --show-current   # if "amd-integration" (or empty/detached), do NOT push/PR from it
```

This prints an **empty string** in detached-HEAD state — treat empty output as an abort condition
too; STOP and report so a proper topic branch can be checked out first.

If you are on `amd-integration` with commits to ship, do NOT raise the PR from it. Relocate the
commits to a fresh topic branch, restore `amd-integration` to match the remote, then PR from the
topic branch:

```bash
# 1. Capture the local-only commits onto a new branch at current HEAD
git branch <topic-branch>

# 2. Move amd-integration back to the pristine remote state (no commits lost — they are
#    preserved on <topic-branch>). Verify origin/amd-integration is fetched and current first.
git fetch origin amd-integration
git reset --hard origin/amd-integration

# 3. Switch to the topic branch and proceed with the normal PR flow
git checkout <topic-branch>
```

Only `git reset --hard` here because the commits are already safe on `<topic-branch>` (confirm with
`git log <topic-branch>` before resetting). If anything is ambiguous — uncommitted changes, unclear
which commits are local-only, the topic branch already exists — STOP and report rather than reset. It
is always better to fail to raise a PR than to push to or PR from `amd-integration`.

## Branch naming

Topic branches are created off `origin/amd-integration` and named with **plain hyphenated words**
describing the change:

- ✅ `pr-workflow-push-safety`, `timing-torch-events`, `aiter-solution-generator`
- ❌ **no `rocm/` prefix** — the entire fork is ROCm, so it's redundant noise.
- ❌ **no plan-phase labels** (`p1`, `p2`, …) — name the *change*, not a milestone.

## CRITICAL: ask before pushing to remote (fail-closed)

**Never `git push` (or `gh pr create`, which pushes) without first getting the user's explicit "yes"
for that specific push.** It publishes to a shared repo. This applies to every push — the initial
branch push, force-pushes after a rebase, and follow-up pushes addressing review comments; a prior
"yes" does not authorize a later push. State exactly what will be pushed and where (branch →
`AMD-Ecosystem/flashinfer-bench`) and wait. Local-only work (commits, the quality gate, the local
review) needs no confirmation — only the network push does. When in doubt, hold the push and ask.

## GitHub CLI

`gh pr edit` can fail with a "Projects (classic) is being deprecated" GraphQL error on forks. Use the
REST API instead:

```bash
# Update PR description
gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<number> --method PATCH --field body="<body>"

# Or from a file
gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<number> --method PATCH --field body="$(cat /tmp/pr_body.md)"
```

## Before creating a PR: quality gate

Run this gate on the **branch's full diff** — `git diff origin/amd-integration...HEAD`, the
cumulative diff a reviewer sees — not on the last commit. The two scopes catch different things:
a per-commit pass cannot see a helper added in commit 2 that nothing uses by commit 6, an
abstraction that drifted across commits, or naming that went inconsistent between them. Conversely,
code added *and* removed within the branch never reaches this diff, so it is not this gate's
problem.

This is the last of the three gates in
[CLAUDE.md](../../../CLAUDE.md#quality-gates-commit-push-pr) — commit (mechanical), push
(simplify/self-review over the unpushed range), then this one, which adds the tests. By the time
you reach it the diff has already been reviewed at least once, so what is genuinely new here is
step 3.

In order:

1. **Simplify / make production-ready.** Review all changes on the branch and remove dead code,
   debug/scratch code, debug-only comments, and unused imports. Keep comments that carry real value
   (the *why*, hidden constraints, non-obvious invariants) — do not strip those. (If a `/simplify`
   command is configured in your environment, use it; it is not required.)

2. **Code review.** Self-review the full diff for correctness and quality, then apply the fixes worth
   making. (If a `/code-review` command is configured, use it; it is not required.)

3. **Run the relevant tests.** Run the pytests covering the changed code (see CLAUDE.md for commands,
   e.g. `pytest -n auto --reruns 2 -m "not slow"`) and make sure there are no failures after the
   changes. Docs/skill-only branches that touch no Python have no relevant tests — say so explicitly
   rather than claiming a run.

4. **Commit** the resulting changes.

Only after this gate passes do the pre-flight safeguards and `gh pr create`.

### Stacked PRs

When a PR depends on another branch, still set `--base amd-integration` (per the rules above) and
note the dependency in the body ("Stacked on #N; diff reduces once #N merges"). Don't set the base to
the parent feature branch.

## After creating a PR: handle the automated review

If the repo has an automated reviewer (e.g. GitHub Copilot), run this loop after `gh pr create`
before considering the PR done.

### Arm the poller, and make it delete itself

The review lands minutes after the push, so the loop below is not something to sit and wait on.
Immediately after `gh pr create`, schedule it with `CronCreate`:

- `cron: "3,10,17,24,31,38,45,52,59 * * * *"` — roughly every 7 minutes. Deliberately **off the
  :00 and :30 marks**: every caller who asks for "every N minutes" lands on those, so staying off
  them spreads load.
- `recurring: true`, `durable: true` — a PR review outlives the session that opened it, and durable
  jobs survive a restart.
- The prompt **must name the PR number** and carry its own teardown:

  > Check PR #N on AMD-Ecosystem/flashinfer-bench. Stop when **either** the PR is merged/closed,
  > **or** the review is complete and addressed — every thread resolved, every suppressed finding
  > closed by a top-level comment, and the newest review covering the current branch head. To
  > stop: `CronList`, find the job whose prompt names PR #N, `CronDelete` it. Otherwise run the
  > automated-review loop in the pr-workflow skill.

**Stop when the review is addressed, not when the PR merges.** The review lands within minutes;
once it is closed out, nothing left needs 7-minute granularity. Polling until merge on a PR that
sits open for three days is ~600 invocations, essentially all of them no-ops. The merge condition
stays as the other exit, for a PR merged or abandoned before its review was handled.

Note the third clause — *newest review covers the current head*. Pushing fixes triggers a fresh
review, so "all threads resolved" alone would stop the job one cycle early, right before the
follow-up review arrives.

**Self-deletion is the teardown mechanism — do not add a cleanup hook for it.** A merge performed in
the GitHub web UI fires no local tool call, so nothing on this machine can observe it; a hook on
`gh pr merge` would only catch the minority of merges done from the CLI. Putting both conditions
inside the polled prompt makes them fire on every path. Two backstops sit behind it: recurring jobs
auto-expire after 7 days, and `CronList` makes an orphaned job visible.

`.claude/hooks/pr-created-review-poller.sh` (a `PostToolUse` hook on `gh pr create`) injects this
instruction automatically once the PR exists, so it is not left to memory.

The loop itself:

1. **Wait for all comments to land.** The review is not instant — the reviewer posts a top-level
   review plus inline comments a short while after the PR (and after each later push). Poll until the
   comment set is stable; don't evaluate a half-posted review. If it *auto-pushes* "Potential fix"
   commits to the branch, `git fetch` and integrate them before adding your own (rebase; resolve
   conflicts keeping the more complete version).

2. **Read the review *bodies*, not just the threads — an empty thread list does not mean the review
   is handled.** Copilot files some findings in a collapsed `<details>` block titled "Comments
   suppressed due to low confidence (N)" inside the review body; these never become
   `reviewThread`s, so they are invisible to the GraphQL query below and to the PR's "unresolved
   conversations" count. Fetch every review body after each push and read that section.

3. **Evaluate each comment on its merits.** Decide per comment whether to fix it — the reviewer is
   often right but not always. Use judgement; do not blanket-apply. Apply the same scrutiny to the
   suppressed findings: the low-confidence label describes the reviewer's certainty, not the
   finding's validity.

4. **Address the ones worth fixing**, commit, and push to the PR branch (with consent, per the
   ask-before-push rule above).

   Tag these commits with a `Review-response: #<PR>` trailer:

   ```
   Fix off-by-one in the marker column

   Review-response: #12
   ```

   The trailer exempts the commit from the push-time simplify/self-review gate — these commits
   already came out of a review pass, so re-reviewing them is busywork. The exemption is
   all-or-nothing over the unpushed range: mix in one untagged commit and the gate re-arms, which
   is the correct fail-closed behaviour.

5. **Resolve every thread**, with the right closure for each:
   - *Fixed* → reply citing the commit SHA, then resolve the thread.
   - *Won't fix* → reply with the reason you decided not to address it, then resolve the thread.

   **For findings that have a thread, the threaded reply is the whole closure — do not additionally
   post a top-level comment summarizing them.** The reply sits next to the code it concerns and the
   commit message carries the detail; a summary comment duplicates both and clutters the
   conversation. This prohibition is scoped to threaded findings; step 6 covers the rest.

6. **The suppressed findings from step 2** have no thread to reply to, so record their closure in a
   single top-level comment scoped to just that batch, listing each finding and marking it *Fixed* (with commit SHA) or *Won't fix* (with rationale). This is the **one** case where a top-level
   comment is right.

   Steps 5 and 6 are complementary, not in tension. The rule is one closure per finding, in the only
   place that finding *can* be closed:

   | Finding kind | Closure |
   |---|---|
   | Has an inline thread | Threaded reply (fix + SHA, or won't-fix rationale) + resolve. No top-level comment. |
   | No thread (suppressed, low-confidence) | One top-level comment scoped to just that batch. |

List and resolve threads via GraphQL (thread resolution and the `isResolved` flag are not exposed
over REST; replying to a comment is):

```bash
# Read the review bodies — this is where "suppressed due to low confidence" findings hide.
gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<PR>/reviews \
  -q '.[] | "\(.submitted_at) \(.user.login) id=\(.id)"'
gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<PR>/reviews/<reviewId> -q '.body'

# List threads (id + resolved + comment bodies). Bump first: for large PRs.
gh api graphql -f query='
{ repository(owner:"AMD-Ecosystem", name:"flashinfer-bench") {
    pullRequest(number: <PR>) {
      reviewThreads(first: 50) { nodes {
        id isResolved
        comments(first: 10) { nodes { databaseId author { login } path body } } } } } } }'

# Reply to a comment (use the databaseId from above)
gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<PR>/comments/<commentDatabaseId>/replies \
  --method POST --field body="<reply>"

# Resolve a thread (use the thread node id, e.g. PRRT_...)
gh api graphql -f query='
mutation { resolveReviewThread(input:{threadId:"<threadId>"}) { thread { isResolved } } }'
```

Done = **both** of:

- no unresolved threads remain, each carrying either a fix+SHA reply or a won't-fix rationale; and
- every review body's suppressed-findings section has been read, and any batch found there is closed
  by its own top-level comment.

"Zero unresolved threads" alone is not done — suppressed findings never appear in that count, so a
PR can look clean while real findings sit unaddressed in a collapsed `<details>` block.

## PR Description

**Do not hard-wrap the body to a fixed column.** GitHub renders a single newline inside a paragraph
as a `<br>`, so column-wrapped prose shows up as broken mid-sentence lines. Write each paragraph and
each bullet as **one line** and let the browser soft-wrap; use blank lines only to separate
paragraphs/list items. (Same for issue bodies and PR/issue comments — anything GitHub renders.) This
is the opposite of the repo's `.md`/`.py` source files, which are wrapped normally.

**Body** — include sections that apply, skip the rest:

- `## Summary` — 1–3 sentences on what and why.
- `### What changed` with `####` per component when the PR spans multiple subsystems. Bullet by file:
  ``- **`path`** — one-line purpose``. Call out non-obvious design choices.
- `### Architecture / design notes` — only when there's a real choice to record. Tables for
  routing/dispatch logic; explain *why*.
- `## Benchmark results` — for perf-touching PRs. Shape line + table per entry point + mean
  overhead/speedup row; record `gcnArchName` + `torch.version.hip` (see
  [`rocm-benchmark`](../rocm-benchmark/SKILL.md)).
- `## Test plan` — checklist of what was actually run (not aspirational), ending with
  `pre-commit run -a`.

Don't restate the diff and commits. Explain non-obvious decisions and surprising behaviors.

## See Also

- `CLAUDE.md` → [ROCm / AMD CDNA Fork](../../../CLAUDE.md) — base-branch rule
