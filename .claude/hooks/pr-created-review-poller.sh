#!/usr/bin/env bash
# PostToolUse(Bash) — after `gh pr create`, arm the Copilot review poller.
#
# Creating the PR and remembering to watch its review are separate acts, and the second one is
# easy to drop once the PR URL is printed and the task feels done. This hook closes that gap by
# injecting the instruction at the moment the PR comes into existence.
#
# Teardown is deliberately NOT hooked: PRs are often merged in the web UI, where no tool call
# exists to hook. The scheduled prompt below deletes itself instead.
set -uo pipefail

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0

grep -Eq 'gh\b[^;&|]*\bpr\b[^;&|]*\bcreate\b' <<<"$cmd" || exit 0

# gh prints the PR URL on success; absence means the create failed and there is nothing to poll.
out=$(printf '%s' "$payload" | jq -r '.tool_response | if type == "object" then ((.stdout // "") + "\n" + (.stderr // "")) else tostring end' 2>/dev/null)
url=$(grep -oE 'https://github\.com/[^ ]+/pull/[0-9]+' <<<"$out" | head -1)
[[ -n "$url" ]] || exit 0
num=${url##*/}

jq -n --arg url "$url" --arg num "$num" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: (
      "PR #" + $num + " was just created (" + $url + ").\n\n" +
      "Per .claude/skills/pr-workflow/, arm the automated-review poller now, before reporting " +
      "the PR as done. Call CronCreate with recurring: true, durable: true, an off-zero " +
      "7-minute schedule (cron \"3,10,17,24,31,38,45,52,59 * * * *\"), and a prompt that tears " +
      "itself down:\n\n" +
      "  Check PR #" + $num + " on AMD-Ecosystem/flashinfer-bench. Stop when EITHER (a) the PR " +
      "is merged or closed, or (b) the review is complete and addressed: every thread resolved, " +
      "every suppressed low-confidence finding closed by a top-level comment, and the newest " +
      "review covers the current branch head, so nothing is awaiting a further review pass. " +
      "To stop: call CronList, find the job whose prompt names PR #" + $num + ", CronDelete it. " +
      "Otherwise run the automated-review loop in the pr-workflow skill: read the review bodies " +
      "(including the collapsed low-confidence section), evaluate each finding, and close each " +
      "one in the only place it can be closed.\n\n" +
      "Condition (b) is the usual exit — the review lands within minutes, and once it is closed " +
      "out there is nothing left that needs 7-minute granularity; waiting for the merge would " +
      "burn hundreds of no-op invocations on a PR that sits open for days. Condition (a) catches " +
      "a PR merged or abandoned before the review was addressed. Both are checked in the prompt " +
      "because a merge done in the GitHub web UI fires no local hook. Recurring jobs also " +
      "auto-expire after 7 days as a backstop."
    )
  }
}'
