---
name: rocm-pr-workflow
description: Create pull requests against the ROCm fork AMD-Ecosystem/flashinfer-bench with a fail-closed target check so PRs always go to base amd-integration and never to upstream flashinfer-ai/flashinfer-bench. Use whenever opening a PR from this repo.
---

# ROCm PR Workflow

This repo is the ROCm-only fork. **Every PR targets `AMD-Ecosystem/flashinfer-bench`, base
`amd-integration`.** The single most common mistake is a PR accidentally opened against upstream
`flashinfer-ai/flashinfer-bench` (the fork parent) — `gh` will silently default there. This skill is
the fail-closed procedure that prevents it.

## Hard rules

1. **Base is always `amd-integration`** on `AMD-Ecosystem/flashinfer-bench`. Never `main`; never
   `flashinfer-ai/*`.
2. **Never PR *from* the `amd-integration` branch itself** — always from a feature branch.
3. **Always pass `--repo AMD-Ecosystem/flashinfer-bench --base amd-integration` explicitly.** Do not
   rely on `gh` defaults.
4. The **dataset** PR (HuggingFace `flashinfer-ai/flashinfer-trace`) is the one exception — it is
   arch-agnostic and stays upstream. That is a HuggingFace PR, not a `gh` PR. See
   [`/onboard-model` Phase 4](../onboard-model/SKILL.md#phase-4-submit-prs).

## Pre-flight (fail-closed check)

Run before creating any PR. Abort if any check fails.

```bash
# a) I am NOT on amd-integration
test "$(git branch --show-current)" != "amd-integration" || { echo "ABORT: on base branch"; exit 1; }

# b) origin is the AMD fork
git remote get-url origin | grep -q "AMD-Ecosystem/flashinfer-bench" \
  || { echo "ABORT: origin is not AMD-Ecosystem/flashinfer-bench"; exit 1; }

# c) if a default repo is set, it must be the fork (unset is fine — we pass --repo explicitly)
def=$(gh repo set-default --view 2>/dev/null || true)
[ -z "$def" ] || echo "$def" | grep -q "AMD-Ecosystem/flashinfer-bench" \
  || { echo "ABORT: gh default repo is '$def', not the fork"; exit 1; }
```

## Create the PR

```bash
git push -u origin <feature-branch>

gh pr create \
  --repo AMD-Ecosystem/flashinfer-bench \
  --base amd-integration \
  --head <feature-branch> \
  --title "<type>: <summary>" \
  --body "$(cat <<'EOF'
## Summary
- <what and why>

## Changes
- <files / areas>

## Test plan
- <how validated — e.g. docker/rocm/validate_p0.py 8/8, benchmark loop on gfx942>
EOF
)"
```

Verify the PR landed on the right base:

```bash
gh pr view --repo AMD-Ecosystem/flashinfer-bench --json baseRefName,headRefName,url \
  <pr-number-or-branch>
# baseRefName MUST be "amd-integration"
```

## Stacked PRs

When a PR depends on another branch (e.g. docs stacked on the code port), still set
`--base amd-integration` per the hard rules. Note the dependency in the body ("Stacked on #N; diff
reduces once #N merges") — the PR will show the union of both diffs until the parent merges. Do not
set the base to the parent feature branch.

## Editing an already-open PR

- Push follow-up commits to the same feature branch — the PR updates in place. **Never** close/reopen
  for a fixable item, and **never** amend/force-push a commit after review without coordinating.
- If `gh pr edit` misbehaves on the body, edit via the REST API:
  `gh api -X PATCH repos/AMD-Ecosystem/flashinfer-bench/pulls/<n> -f body="..."`.

## Quality gate (before requesting review)

- `pre-commit run --all-files` clean (do **not** bypass with `--no-verify`).
- Tests / validation appropriate to the change ran and are cited in the body (for ROCm code:
  `docker/rocm/validate_p0.py`, and a benchmark-loop run on gfx942 where relevant).
- Diff is scoped — no stray files, no accidental `flashinfer_trace/` paths.

## Common issues

- **PR opened against `flashinfer-ai/flashinfer-bench`**: you omitted `--repo`. Close it, re-run the
  pre-flight, re-create with the explicit flags.
- **`gh` picks the wrong base**: pass `--base amd-integration` explicitly; don't trust the default.
- **Push rejected**: confirm `origin` points at `AMD-Ecosystem/flashinfer-bench` and you have write
  access.

## Maintaining this document

Update if the fork's base branch changes, if the canonical GitHub home moves, or if the dataset-PR
exception changes.

## See Also

- `CLAUDE.md` → [ROCm / AMD CDNA Fork](../../../CLAUDE.md) — base-branch rule
- [onboard-model](../onboard-model/SKILL.md#phase-4-submit-prs) — Phase 4: coverage-doc PR (fork) + dataset PR (upstream)
