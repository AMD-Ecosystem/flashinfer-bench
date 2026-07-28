---
name: rocm-pr-workflow
description: Create and edit PRs on the ROCm fork AMD-Ecosystem/flashinfer-bench with fail-closed safeguards — always base amd-integration, never upstream flashinfer-ai/flashinfer-bench, and never PR from amd-integration itself (with a commit-relocation recovery). Covers the gh pr edit→REST workaround, the pre-PR quality gate, the Copilot-review resolution loop, and PR-description conventions. Use whenever opening or updating a PR from this repo.
---

# ROCm PR Workflow

This repo is the ROCm-only fork. **Every code PR targets `AMD-Ecosystem/flashinfer-bench`, base
`amd-integration`.** The most common mistake is a PR silently opened against upstream
`flashinfer-ai/flashinfer-bench` (the fork parent) — `gh` defaults there. These safeguards are
fail-closed: when the target can't be positively confirmed, **stop and report** rather than risk an
upstream PR. Failing to open a PR is always better than opening it against upstream.

## Hard rules

1. **Base is always `amd-integration`** on `AMD-Ecosystem/flashinfer-bench`. Never `main`, never
   `flashinfer-ai/*`.
2. **Never PR *from* `amd-integration`** — it is the base, never the head (recovery below).
3. **Always pass `--repo AMD-Ecosystem/flashinfer-bench --base amd-integration` explicitly.** Don't
   rely on `gh` defaults.
4. **Exception:** the *dataset* PR (HuggingFace `flashinfer-ai/flashinfer-trace`) is arch-agnostic
   and stays upstream — that's a HuggingFace PR, not a `gh` PR. See
   [`submit-onboarding-prs`](../submit-onboarding-prs/SKILL.md).

## Pre-flight (fail-closed) — run before every PR

```bash
# a) not on the base branch (empty output = detached HEAD → also abort)
b=$(git branch --show-current); [ -n "$b" ] && [ "$b" != "amd-integration" ] \
  || { echo "ABORT: on amd-integration or detached HEAD"; exit 1; }

# b) origin is the AMD fork; there must be NO flashinfer-ai remote
git remote get-url origin | grep -q "AMD-Ecosystem/flashinfer-bench" \
  || { echo "ABORT: origin is not the fork"; exit 1; }
git remote -v | grep -q "flashinfer-ai/flashinfer-bench" \
  && { echo "ABORT: a flashinfer-ai remote exists — remove it"; exit 1; }

# c) if a gh default repo is set, it must be the fork (unset is fine — we pass --repo)
def=$(gh repo set-default --view 2>/dev/null || true)
[ -z "$def" ] || echo "$def" | grep -q "AMD-Ecosystem/flashinfer-bench" \
  || { echo "ABORT: gh default repo is '$def'"; exit 1; }
```

If the resolved owner of `--repo` is ever `flashinfer-ai`, abort and report. Never guess the target.

## Recovery: you're on `amd-integration` with commits to ship

Do **not** PR from it. Relocate the commits to a topic branch, restore the base, then PR from the
topic branch (nothing is lost — the commits are safe on `<topic>` before the reset):

```bash
git branch <topic>                       # 1. capture local-only commits at current HEAD
git fetch origin amd-integration         # 2. restore amd-integration to pristine remote state
git reset --hard origin/amd-integration  #    (commits preserved on <topic>; verify with git log <topic>)
git checkout <topic>                      # 3. proceed with the normal flow
```

Only `git reset --hard` because the commits are already on `<topic>`. If anything is ambiguous
(uncommitted changes, unclear which commits are local-only, `<topic>` already exists), STOP and
report.

## Create the PR

```bash
git push -u origin <feature-branch>
gh pr create --repo AMD-Ecosystem/flashinfer-bench --base amd-integration --head <feature-branch> \
  --title "<type>: <summary>" --body "$(cat /tmp/pr_body.md)"

# verify it landed on the right base
gh pr view <n> --repo AMD-Ecosystem/flashinfer-bench --json baseRefName -q .baseRefName  # MUST be amd-integration
```

**Stacked PRs:** still set `--base amd-integration` (per the hard rules); note the dependency in the
body ("Stacked on #N; diff reduces once #N merges"). Don't set the base to the parent feature branch.

## Editing an open PR

- Push follow-up commits to the same branch — the PR updates in place. Never close/reopen for a
  fixable item; never amend/force-push after review without coordinating.
- **`gh pr edit` can fail** with a "Projects (classic) deprecated" GraphQL error — edit the body via
  REST instead:
  ```bash
  gh api repos/AMD-Ecosystem/flashinfer-bench/pulls/<n> --method PATCH --field body="$(cat /tmp/pr_body.md)"
  ```

## Pre-PR quality gate (in order)

1. **Simplify** — remove dead/scratch/debug code and unused imports; keep comments that carry real
   *why*/constraints.
2. **Code review** — self-review the diff; apply the worthwhile suggestions.
3. **Tests** — run the pytests covering the change (`pytest -n auto --reruns 2 -m "not slow"`) and
   report the real result. Docs/skill-only branches touch no Python — say so explicitly rather than
   claiming a run.
4. **Commit.** Then run the pre-flight safeguards and `gh pr create`.

Pushing/creating PRs publishes to a shared repo — confirm with the user before `git push` / `gh pr
create` unless already authorized.

## After creating: resolve automated review

If the repo has an automated (e.g. Copilot) reviewer: wait for the full comment set to land (it may
also auto-push "Potential fix" commits — `git fetch` and integrate before adding your own), evaluate
each comment on its merits, fix the ones worth fixing and push, then **resolve every thread** with
either a fix+commit-SHA reply or a won't-fix rationale. Thread resolution is GraphQL-only:

```bash
gh api graphql -f query='{ repository(owner:"AMD-Ecosystem", name:"flashinfer-bench") {
  pullRequest(number: <PR>) { reviewThreads(first:50){ nodes {
    id isResolved comments(first:10){ nodes { databaseId path body } } } } } } }'
gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:"<id>"}){ thread { isResolved } } }'
```

## PR description conventions

- `## Summary` — 1–3 sentences: what and why.
- `### What changed` — bullet by file (``- **`path`** — one-line purpose``); `####` per component for
  multi-subsystem PRs. Call out non-obvious choices.
- `## Benchmark results` — for perf-touching PRs: shape line + table + speedup/overhead row (record
  `gcnArchName` + `torch.version.hip`).
- `## Test plan` — what was actually run (not aspirational), ending with `pre-commit run -a`.

Don't restate the diff — explain non-obvious decisions and surprising behavior.

## Maintaining this document

Update if the fork's base branch changes, if the canonical GitHub home moves, or if the dataset-PR
exception changes.

## See Also

- `CLAUDE.md` → [ROCm / AMD CDNA Fork](../../../CLAUDE.md) — base-branch rule
- [submit-onboarding-prs](../submit-onboarding-prs/SKILL.md) — coverage-doc PR (fork) + dataset PR (upstream)
