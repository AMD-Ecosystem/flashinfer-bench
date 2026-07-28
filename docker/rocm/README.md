# FlashInfer-Bench on ROCm / AMD CDNA

Development container for running FlashInfer-Bench on AMD Instinct GPUs.
Targets **CDNA3 (gfx942, MI300X/MI325X)**; CDNA4 (gfx950) is additive via build args.

Inspired by [AMD-Ecosystem/flashinfer](https://github.com/AMD-Ecosystem/flashinfer)
(`.devcontainer/rocm/Dockerfile`, `docker/Dockerfile.rocm_ci`).

## What's in the image

| Component | Version | Source |
|---|---|---|
| Base | `rocm/dev-ubuntu-24.04:7.2-complete` | Docker Hub (hipcc, rocprofv3, ROCm libs) |
| PyTorch (ROCm) | 2.9.1 | `repo.radeon.com/rocm/manylinux/rocm-rel-7.2` |
| amd-flashinfer (imports as `flashinfer`) | 0.5.3 | `pypi.amd.com/rocm-7.2.0/simple` |
| amd_aiter | 0.1.10 | `pypi.amd.com/rocm-7.1.1/simple` |
| apache-tvm-ffi (HIP-capable) | 0.1.12 | PyPI |
| flashinfer-bench deps | — | PyPI (pydantic, safetensors, pyyaml, …) |

`flashinfer-bench` itself is **not baked in** — it is bind-mounted from the repo
root at `/workspace` and installed editable with `--no-deps` on entry, so host
edits are live and the NVIDIA `flashinfer-python` dependency is never pulled.

Arch is pinned via `PYTORCH_ROCM_ARCH` / `TVM_FFI_ROCM_ARCH_LIST` = `gfx942`.

## Usage

```bash
# Build the image (from repo root)
docker build -f docker/rocm/Dockerfile -t flashinfer-bench:rocm docker/rocm

# Interactive shell (auto-builds if missing, mounts repo, editable install)
bash docker/rocm/run.sh

# One-off command
bash docker/rocm/run.sh python docker/rocm/validate_p0.py
bash docker/rocm/run.sh pytest -q
```

Override arch (e.g. add CDNA4) at build time:

```bash
docker build -f docker/rocm/Dockerfile --build-arg ROCM_ARCH="gfx942 gfx950" \
    -t flashinfer-bench:rocm docker/rocm
```

## GPU access

`run.sh` passes `--device=/dev/kfd --device=/dev/dri --group-add video
--group-add render --security-opt seccomp=unconfined --ipc=host`. Verify with
`rocminfo` or `rocm-smi` inside the container.

## P0 validation

`docker/rocm/validate_p0.py` checks the whole stack end-to-end: torch.cuda on
gfx942, a GPU matmul, `import flashinfer` / `aiter` / `flashinfer_bench`, and a
**tvm-ffi HIP end-to-end kernel compile+run**.
