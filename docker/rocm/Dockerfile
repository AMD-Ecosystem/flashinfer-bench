# =============================================================================
# FlashInfer-Bench — ROCm / AMD CDNA development image
#
# Inspired by AMD-Ecosystem/flashinfer (.devcontainer/rocm/Dockerfile and
# docker/Dockerfile.rocm_ci). Targets CDNA3 (gfx942, MI300X/MI325X); gfx950
# (CDNA4) is additive via ROCM_ARCH build args.
#
# This image installs all THIRD-PARTY dependencies. The flashinfer-bench source
# itself is expected to be bind-mounted at /workspace at runtime and installed
# editable (`pip install -e . --no-deps`) so edits during the port are live.
# See docker/rocm/run.sh.
#
# Build:
#   docker build -f docker/rocm/Dockerfile -t flashinfer-bench:rocm docker/rocm
# Run (from repo root):
#   bash docker/rocm/run.sh
# =============================================================================

ARG ROCM_VERSION=7.2
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

# ---- Versions (pinned; verified reachable on 2026-07 for ROCm 7.2.0 host) ----
ARG PY_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG TORCH_ROCM_REL=7.2
ARG FLASHINFER_VERSION=0.5.3
ARG FLASHINFER_ROCM_INDEX=7.2.0
ARG AITER_VERSION=0.1.10
ARG AITER_ROCM_INDEX=7.1.1
ARG TVM_FFI_VERSION=0.1.12
# Target GPU architecture(s). gfx942=CDNA3 (MI300X/MI325X). Add gfx950 for CDNA4.
ARG ROCM_ARCH=gfx942

ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
# Consumed by torch, tvm-ffi (TVM_FFI_ROCM_ARCH_LIST) and hipcc builds.
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}
ENV TVM_FFI_ROCM_ARCH_LIST=${ROCM_ARCH}
ENV PATH="${ROCM_PATH}/bin:${PATH}"

# The base rocm/dev-ubuntu-24.04:*-complete image already provides gcc/g++,
# python3, pip3, hipcc, rocprofv3 and ROCm dev libraries. We deliberately do NOT
# run `apt-get` here: the container's Ubuntu archive mirrors (archive.ubuntu.com)
# are slow/unreachable from this network, whereas repo.radeon.com and PyPI are
# fast. Everything we need beyond the base (cmake, ninja for tvm-ffi builds) is
# installed as pip wheels below.
#
# We install into the system interpreter with --break-system-packages (PEP 668).
# This is standard and safe inside a container and avoids needing the
# python3.12-venv apt package (which would require the hanging apt mirror).
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1
# Note: do NOT `pip install --upgrade pip` here — the base image's pip 24.0 is a
# Debian-managed package without a RECORD file and cannot self-uninstall. It is
# new enough. We upgrade setuptools/wheel and add setuptools-scm (build backend
# required to install flashinfer-bench editable).
RUN python3 -m pip install --ignore-installed setuptools wheel setuptools-scm
# Build tools needed by tvm-ffi's HIP build path (ninja) and general C++ builds.
RUN python3 -m pip install cmake ninja

# ---- PyTorch (ROCm build) — pinned first so nothing downgrades/overrides it ----
# find-links (-f) keeps PyPI available for torch's pure-python deps.
RUN python3 -m pip install "torch==${TORCH_VERSION}" \
        -f "https://repo.radeon.com/rocm/manylinux/rocm-rel-${TORCH_ROCM_REL}/"

# ---- AMD FlashInfer (imports as `flashinfer`) + AITER operator library ----
RUN python3 -m pip install "amd-flashinfer==${FLASHINFER_VERSION}" \
        --extra-index-url "https://pypi.amd.com/rocm-${FLASHINFER_ROCM_INDEX}/simple/"
RUN python3 -m pip install "amd_aiter==${AITER_VERSION}" \
        --extra-index-url "https://pypi.amd.com/rocm-${AITER_ROCM_INDEX}/simple/"

# ---- flashinfer-bench runtime deps (explicit; we install fib with --no-deps at
#      runtime so the NVIDIA `flashinfer-python` dep is never pulled in) ----
RUN python3 -m pip install \
        "apache-tvm-ffi==${TVM_FFI_VERSION}" \
        "pydantic>=2.0.0" \
        "safetensors>=0.5.0" \
        "docstring-parser>=0.16" \
        "pyyaml>=6.0" \
        "filelock" \
        "numpy"

# ---- Dev / test / serve extras ----
RUN python3 -m pip install \
        pytest pytest-asyncio pytest-cov pytest-xdist pytest-rerunfailures httpx \
        ruff black isort \
        fastapi "uvicorn[standard]"

# ---- amd-smi Python bindings (clock locking / env snapshot); best-effort ----
RUN python3 -m pip install "${ROCM_PATH}/share/amd_smi" || \
    echo "amd_smi python bindings not installed (optional)"

# ---- git ----
# A real `git` is required at IMPORT time by aiter (get_git_commit_id_short) and
# flashinfer, and by setuptools_scm's file-finder during the editable install.
# The Ubuntu archive is unreachable from this network and the internal mirror
# lacks git, so install it from conda-forge via a standalone micromamba. Exposed
# through a wrapper (not by prepending the env to PATH) so the conda env's own
# python/tools never shadow the system interpreter used everywhere else.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local bin/micromamba \
    && /usr/local/bin/micromamba create -y -q -p /opt/gitenv -c conda-forge git elfutils \
    && printf '#!/bin/sh\nexec /opt/gitenv/bin/git "$@"\n' > /usr/local/bin/git \
    && chmod +x /usr/local/bin/git \
    && git --version \
    # The repo is bind-mounted from the host (owned by the host user) but the
    # container runs as root; allow git to operate on it regardless of owner so
    # setuptools_scm's git introspection works during the editable install.
    && git config --system --add safe.directory /workspace \
    # rocprofv3's tool library needs libdw.so.1 (elfutils), which is absent from the
    # base image and the reachable apt mirrors. Install it from conda-forge (above)
    # and expose ONLY libdw in the system multiarch dir (it resolves against the
    # system libelf/libz/...), so the global env is not polluted with conda libs.
    && cp -L /opt/gitenv/lib/libdw.so.1 /usr/lib/x86_64-linux-gnu/libdw.so.1 \
    && ldconfig \
    && /usr/local/bin/micromamba clean -a -y

# Still pin the version so setuptools_scm produces a stable version without
# needing repo tags in the bind-mounted source.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0

WORKDIR /workspace

# GPU visibility is provided at `docker run` time via --device=/dev/kfd
# --device=/dev/dri --group-add video --group-add render (see run.sh).
CMD ["/bin/bash"]
