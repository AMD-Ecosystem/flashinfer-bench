---
name: rocm-setup
description: Stand up and verify a ROCm / AMD-CDNA environment for FlashInfer-Bench — the reproducible docker/rocm container (ROCm 7.2 + torch-ROCm + amd-flashinfer + AITER + tvm-ffi), the bare-metal essentials (torch from AMD's repo, AITER source install, arch pinning), the essential day-to-day commands, and the P0/AITER validation scripts on gfx942. Use when preparing a machine to build/benchmark/apply on AMD Instinct GPUs.
---

# ROCm Setup

Get a working ROCm environment and prove it on the GPU. The reproducible path is the
[`docker/rocm/`](../../../docker/rocm/README.md) container; the bare-metal essentials and the
day-to-day commands below apply either way. Rationale + pinned versions: `ROCM_PORT_PLAN.md`.

## Essential commands

| Task | Command |
|------|---------|
| Install bench editable (no NVIDIA deps) | `pip install -e . --no-deps` |
| Run tests (fast) | `pytest -n auto --reruns 2 -m "not slow"` |
| Run all tests | `pytest -n auto --reruns 2` |
| Clear JIT cache (after any toolchain/flag change) | `rm -rf ~/.cache/flashinfer/` |
| Set target arch(s) | `export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"` |
| Limit parallel build | `export MAX_JOBS=4` |
| Verbose JIT output | `export FLASHINFER_JIT_VERBOSE=1` |
| Lint | `pre-commit run -a` |

`pytest -n auto` halves the physical GPU count to avoid HSA/hipBLAS flakiness under concurrent load;
`--reruns 2` absorbs transient HIP flakiness; the `slow` marker gates heavy sampling / multi-GB tests.

## Bare-metal essentials (what the container encapsulates)

- **Torch must be the AMD ROCm build.** Install from AMD's ROCm repo (`repo.radeon.com/rocm/...`);
  a stray PyPI/CPU wheel breaks everything. Verify: `python -c "import torch; assert torch.version.hip"`.
- **`amd-flashinfer`** (imports as `flashinfer`) from `pypi.amd.com`; install bench with `--no-deps`
  so the NVIDIA `flashinfer-python` dep is never pulled (pending the `pyproject` swap, §3.12).
- **AITER** is a separate install matched to the ROCm version:
  ```bash
  git clone --recursive https://github.com/ROCm/aiter.git
  cd aiter && python3 setup.py develop     # or the pinned wheel from pypi.amd.com (see docker/rocm)
  ```
  Check: `from flashinfer.aiter_utils import is_aiter_supported` /
  `from flashinfer_bench.integration.aiter import is_aiter_available`.
- **Arch** is a list, pinned via `FLASHINFER_ROCM_ARCH_LIST` / `PYTORCH_ROCM_ARCH` /
  `TVM_FFI_ROCM_ARCH_LIST` (default `gfx942`); gfx950/CDNA4 is additive.

## The container (reproducible path)

| Component | Pinned | Source |
|---|---|---|
| Base | `rocm/dev-ubuntu-24.04:7.2-complete` | Docker Hub (hipcc, rocprofv3, ROCm libs) |
| PyTorch (ROCm) | 2.9.1+rocm7.2.0 | `repo.radeon.com/.../rocm-rel-7.2` |
| amd-flashinfer (→ `flashinfer`) | 0.5.3+amd.1 | `pypi.amd.com/rocm-7.2.0/simple` |
| amd_aiter | 0.1.10 | `pypi.amd.com/rocm-7.1.1/simple` |
| apache-tvm-ffi | 0.1.12 | PyPI |

```bash
# Build (arch is a build-arg list)
docker build -f docker/rocm/Dockerfile -t flashinfer-bench:rocm docker/rocm
docker build -f docker/rocm/Dockerfile --build-arg ROCM_ARCH="gfx942 gfx950" -t flashinfer-bench:rocm docker/rocm

# Enter / one-off (bind-mounts the repo, editable --no-deps install, passes GPU device flags)
bash docker/rocm/run.sh
bash docker/rocm/run.sh python docker/rocm/validate_p0.py
bash docker/rocm/run.sh pytest -q
```

`run.sh` passes `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
--security-opt seccomp=unconfined --ipc=host`. Build-environment gotchas (BuildKit off, `docker
logout` for the anonymous base pull, git from conda-forge) are baked into the Dockerfile and listed
in `ROCM_PORT_PLAN.md`.

## Validate

```bash
python docker/rocm/validate_p0.py        # expect 8/8 on gfx942
python docker/rocm/validate_aiter_ops.py # AITER solutions through the real Benchmark loop
```

`validate_p0.py` checks torch.cuda on device, a GPU matmul, `import flashinfer`/`aiter`/
`flashinfer_bench`, tvm-ffi backend = `hip`, a tvm-ffi HIP kernel compile+run, and nvtx→roctx.
Any failure means the env isn't ready — fix before benchmarking.

Quick manual sanity (anywhere with the env active):

```python
import torch; assert torch.version.hip
p = torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName)   # device.type is "cuda" on AMD
```

## Arch ↔ codename

MI300X / MI325X = gfx942 = CDNA3; MI355X = gfx950 = CDNA4.

## Common issues

- **`torch.version.hip is None`** — a CUDA/CPU torch got installed; reinstall from AMD's ROCm repo.
- **`import flashinfer` pulls the NVIDIA wheel** — you installed bench with deps; use `--no-deps`.
- **AITER import calls `git` and crashes** — install git + `git config --global --add safe.directory '*'`.
- **No GPU in container** — use `run.sh` (don't hand-roll `docker run` without the device flags).
- **Stale build after env change** — `rm -rf ~/.cache/flashinfer/` (JIT `build.ninja` is only
  rewritten when missing, so env changes are otherwise silent no-ops).

## Maintaining this document

Update when pinned versions, the base image, build args, or `docker/rocm/` script names change; keep
the version table in sync with `docker/rocm/Dockerfile`.

## See Also

- [docker/rocm/README.md](../../../docker/rocm/README.md) — container reference
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) — porting plan + P0/P1 results
- [rocm-benchmark](../rocm-benchmark/SKILL.md) · [generate-aiter-solution](../generate-aiter-solution/SKILL.md) · [rocm-debug](../rocm-debug/SKILL.md)
