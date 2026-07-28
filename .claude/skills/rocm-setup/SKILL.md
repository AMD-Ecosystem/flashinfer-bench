---
name: rocm-setup
description: Stand up the ROCm / AMD-CDNA dev environment for FlashInfer-Bench — build and run the docker/rocm container (ROCm 7.2 + torch-ROCm + amd-flashinfer + AITER + tvm-ffi), install the package editable, and validate the stack on gfx942 with validate_p0.py / validate_aiter_ops.py. Use when preparing a machine to build/benchmark/apply on AMD Instinct GPUs.
---

# ROCm Setup

Bring up the supported ROCm environment and prove it works end-to-end on the GPU. This is the
first thing to run on an AMD host — the ROCm build → benchmark → apply loop depends on it.

The full porting rationale, pinned versions, and on-hardware results live in
[`ROCM_PORT_PLAN.md`](../../../ROCM_PORT_PLAN.md); the container itself is documented in
[`docker/rocm/README.md`](../../../docker/rocm/README.md). This skill is the operational checklist.

## When to use

- First-time setup of an AMD Instinct box (CDNA3 gfx942 first; CDNA4 gfx950 additive).
- Rebuilding the image after a dependency bump (torch-ROCm / amd-flashinfer / amd_aiter).
- Verifying a host before running [`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md) or
  [`generate-aiter-solution`](../generate-aiter-solution/SKILL.md).

## Prerequisites (host)

- AMD Instinct GPU visible: `rocminfo` and `rocm-smi` succeed on the host.
- Docker with GPU device nodes (`/dev/kfd`, `/dev/dri`) accessible.
- The `flashinfer-bench` repo checked out (it is bind-mounted, **not** baked into the image).

## The image

| Component | Pinned | Source |
|---|---|---|
| Base | `rocm/dev-ubuntu-24.04:7.2-complete` | Docker Hub (hipcc, rocprofv3, ROCm libs) |
| PyTorch (ROCm) | 2.9.1+rocm7.2.0 | `repo.radeon.com/rocm/manylinux/rocm-rel-7.2` |
| amd-flashinfer (imports as `flashinfer`) | 0.5.3+amd.1 | `pypi.amd.com/rocm-7.2.0/simple` |
| amd_aiter | 0.1.10 | `pypi.amd.com/rocm-7.1.1/simple` |
| apache-tvm-ffi | 0.1.12 | PyPI |

Arch is pinned via `PYTORCH_ROCM_ARCH` / `TVM_FFI_ROCM_ARCH_LIST` (default `gfx942`).
`flashinfer-bench` is installed editable with `--no-deps` on entry, so host edits are live and the
NVIDIA `flashinfer-python` dependency is never pulled (pending the `pyproject` swap in
`ROCM_PORT_PLAN.md` §3.12).

## Steps

### 1. Build the image

```bash
# From repo root
docker build -f docker/rocm/Dockerfile -t flashinfer-bench:rocm docker/rocm

# Add CDNA4 (or others) at build time — arch is a list, not a constant
docker build -f docker/rocm/Dockerfile --build-arg ROCM_ARCH="gfx942 gfx950" \
    -t flashinfer-bench:rocm docker/rocm
```

If the build fails on registry/apt access, see the "Environment gotchas" baked into the Dockerfile
and enumerated in `ROCM_PORT_PLAN.md` (e.g. `DOCKER_BUILDKIT=0`, `docker logout` before the
anonymous base pull, git from conda-forge because the apt mirror lacks it).

### 2. Enter the container

`docker/rocm/run.sh` auto-builds if the image is missing, bind-mounts the repo at `/workspace`,
does the editable `--no-deps` install, and passes the GPU device flags
(`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
--security-opt seccomp=unconfined --ipc=host`).

```bash
bash docker/rocm/run.sh                 # interactive shell
bash docker/rocm/run.sh python docker/rocm/validate_p0.py   # one-off
bash docker/rocm/run.sh pytest -q       # run the suite
```

Inside the container, confirm GPU access with `rocminfo` / `rocm-smi`.

### 3. Validate the stack (P0)

```bash
bash docker/rocm/run.sh python docker/rocm/validate_p0.py
```

Expect **8/8 PASS** on gfx942: `torch.cuda` on the device, a GPU matmul, `import flashinfer` /
`aiter` / `flashinfer_bench`, tvm-ffi backend = `hip`, a **tvm-ffi HIP kernel compiled + run**, and
the nvtx→roctx mapping. Any failure here means the environment is not ready — fix it before
benchmarking.

### 4. Validate AITER solutions end-to-end (P1)

```bash
bash docker/rocm/run.sh python docker/rocm/validate_aiter_ops.py
```

For each AITER-covered op-type this builds an in-memory definition + workload, asks the §3.9
generator for the AITER-backed solution, and runs the **real Benchmark loop** (build → time →
correctness vs reference → speedup) on the GPU. See
[`generate-aiter-solution`](../generate-aiter-solution/SKILL.md).

### 5. Sanity checks you can run anywhere in the container

```bash
python -c "import torch; assert torch.version.hip, 'not a ROCm torch'; \
  p=torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName)"
python -c "import flashinfer; print('amd-flashinfer', flashinfer.__version__)"
python -c "from flashinfer_bench.integration.aiter import is_aiter_available; print('aiter', is_aiter_available())"
```

`torch.version.hip` must be non-None (fail fast otherwise), and `device.type` is `"cuda"` even on
AMD — that's the HIP-as-CUDA masquerade, not a bug (see `CLAUDE.md`).

## AITER outside the container

AITER is normally installed from `pypi.amd.com` by the image. For a source install (to read/patch
HIP kernels) it must match the image's ROCm version:

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && python3 setup.py develop
```

## Common issues

- **`torch.version.hip is None`** — a CUDA torch got installed. Reinstall torch from
  `repo.radeon.com/rocm/manylinux/rocm-rel-7.2` (use `-f`, not `--index-url`).
- **`import flashinfer` pulls NVIDIA wheel** — you installed `flashinfer-bench` with deps. Use
  `pip install -e . --no-deps` (run.sh already does this).
- **AITER import calls `git` and crashes** — aiter/flashinfer read git at import; ensure `git` is
  present and `git config --global --add safe.directory '*'` for the bind-mounted repo.
- **No GPU in container** — missing device flags; use `run.sh` (don't hand-roll `docker run`).

## Maintaining this document

Update when the pinned versions, base image, build args, or `docker/rocm/` script names change.
Keep the version table in sync with `docker/rocm/Dockerfile` and `docker/rocm/README.md`.

## See Also

- [docker/rocm/README.md](../../../docker/rocm/README.md) — container reference
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) — porting plan + P0/P1 results
- [benchmark-on-rocm](../benchmark-on-rocm/SKILL.md) — timing/profiling once set up
- [generate-aiter-solution](../generate-aiter-solution/SKILL.md) — AITER-backed solutions
- [debug-rocm](../debug-rocm/SKILL.md) — when validation fails
