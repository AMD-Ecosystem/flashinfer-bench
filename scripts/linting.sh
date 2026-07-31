#!/bin/bash
set -eo pipefail
set -x
echo "Linting..."

# Check if ruff is available, if not install it
if ! command -v ruff &> /dev/null; then
    echo "ruff not found, installing ruff..." && pip install ruff
fi

# --no-fix, NOT --fix. With --fix ruff repairs what it can and exits 0, so in CI the violations
# are silently fixed in the ephemeral checkout, the job passes, and the repairs are discarded with
# the runner — every auto-fixable violation lands on the branch unnoticed. Fail instead; run
# `ruff check . --fix` locally to repair.
#
# Explicit rather than relying on the default: pyproject sets no `[tool.ruff] fix`, so a bare
# `ruff check .` does not fix today — but adding `fix = true` for local convenience would silently
# restore the masking this flag exists to stop.
ruff check . --no-fix
