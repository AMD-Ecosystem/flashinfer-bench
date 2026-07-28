#!/usr/bin/env bash
# Build (if needed) and run the flashinfer-bench ROCm dev container with GPU access.
#
# Usage:
#   docker/rocm/run.sh                 # interactive shell in the container
#   docker/rocm/run.sh <cmd> [args...] # run a one-off command in the container
#
# The repo root is bind-mounted at /workspace and installed editable on entry,
# so source edits on the host are live inside the container.
set -euo pipefail

IMAGE="${FIB_ROCM_IMAGE:-flashinfer-bench:rocm}"
# Resolve repo root (two levels up from this script: docker/rocm/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Build the image if it does not exist yet.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo ">>> Building ${IMAGE} ..."
    docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${SCRIPT_DIR}"
fi

# Editable install of flashinfer-bench (--no-deps: keep the ROCm stack intact,
# never pull the NVIDIA `flashinfer-python`), then exec the requested command.
ENTRY='python3 -m pip install -e /workspace --no-deps -q >/dev/null 2>&1 || python3 -m pip install -e /workspace --no-deps; '
if [ "$#" -eq 0 ]; then
    ENTRY+='exec bash'
else
    ENTRY+="exec $(printf '%q ' "$@")"
fi

exec docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size=16g \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    bash -lc "${ENTRY}"
