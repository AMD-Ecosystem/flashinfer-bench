# FlashInfer-Bench on ROCm

> ROCm-only fork of FlashInfer-Bench for AMD Instinct (CDNA3/gfx942, CDNA4/gfx950). Build →
> benchmark → apply GPU-kernel solutions on AMD, with AITER as a first-class kernel source.

## Quickstart: run a benchmark

The shortest path to benchmarking on AMD Instinct (CDNA3/gfx942). It chains three skills —
[`rocm-setup`](.claude/skills/rocm-setup/SKILL.md) → [`generate-aiter-solution`](.claude/skills/generate-aiter-solution/SKILL.md)
→ [`rocm-benchmark`](.claude/skills/rocm-benchmark/SKILL.md) — with [`rocm-debug`](.claude/skills/rocm-debug/SKILL.md)
as the fallback when something breaks. Full detail lives in those skills; this is the happy path.

**Prerequisites:** an AMD Instinct GPU visible (`rocminfo` / `rocm-smi` work) and Docker with
`/dev/kfd` + `/dev/dri`.

## 1. Set up and verify the environment  (`rocm-setup`)

```bash
# Build + enter the ROCm dev container (ROCm 7.2 + torch-ROCm + amd-flashinfer + AITER + tvm-ffi).
# Bind-mounts the repo and installs it editable --no-deps; passes GPU device flags.
bash docker/rocm/run.sh

# Inside the container, prove the stack works end-to-end on the GPU:
python docker/rocm/validate_p0.py          # expect 8/8 PASS on gfx942
python -c "import torch; assert torch.version.hip; \
  p=torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName)"
```

If `validate_p0.py` isn't green, stop and fix the env before benchmarking (see `rocm-setup`).

## 2. Get a solution to benchmark  (`generate-aiter-solution`)

AITER is AMD's tuned op library — for covered op-types it gives you a fast solution with no compile:

```bash
# End-to-end proof across every AITER-covered op-type (build → time → correctness → speedup):
python docker/rocm/validate_aiter_ops.py
```

Or generate an AITER solution for a specific dataset definition in Python:

```python
import json
from pathlib import Path
from flashinfer_bench.data import Definition
from flashinfer_bench.integration.aiter import generate_aiter_solution, is_aiter_available

assert is_aiter_available()
d = Definition(**json.loads(Path("<trace_dir>/definitions/rmsnorm/rmsnorm_h4096.json").read_text()))
sol = generate_aiter_solution(d)   # None => op/shape not covered; use a Triton/HIP solution instead
```

(You benchmark against definitions + workloads from the arch-agnostic HuggingFace trace dataset —
this fork *consumes* that dataset; it isn't produced here.)

## 3. Run the benchmark  (`rocm-benchmark`)

```bash
# torch/HIP-event timing (default), under gpu-lock so the device is pinned:
tools/gpu-lock --gpus 1 -- \
  python -m flashinfer_bench run --local <trace_dir> --definitions <def> --save-results
```

The loop builds each solution, times it (median over iters, cold-L2 flush), checks correctness vs
the reference, and reports speedup. **A speedup only counts if correctness passes** — on AMD, a
correctness failure is often an fp8 `_fnuz` dtype mismatch or a tolerance issue, not the kernel.
For a high-fidelity pass use the `rocprofv3` backend and pin clocks:

```bash
sudo rocm-smi --setperfdeterminism 1900
FIB_TIMING_BACKEND=rocprof FIB_L2_FLUSH_MB=256 \
  tools/gpu-lock --gpus 1 -- python -m flashinfer_bench run --local <trace_dir> --definitions <def>
sudo rocm-smi --resetclocks
```

Always record `gcnArchName` + `torch.version.hip` with any number — results aren't comparable across
arch/ROCm versions.

## When something breaks  (`rocm-debug`)

```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
  python -m flashinfer_bench run --local <trace_dir> --definitions <def>
```

This localizes HIP faults to the offending kernel. See [`rocm-debug`](.claude/skills/rocm-debug/SKILL.md)
for the per-error recipe table (memory faults, NaN/Inf, HIP OOM, AITER errors, fp8 `_fnuz`).

## Going further

- **Hand-written HIP kernels** (when AITER/Triton don't cover an op): [`add-rocm-kernel`](.claude/skills/add-rocm-kernel/SKILL.md).
- **Contributing changes back**: [`pr-workflow`](.claude/skills/pr-workflow/SKILL.md) (targets
  `AMD-Ecosystem/flashinfer-bench`, base `amd-integration`).
- **Full porting context**: [`ROCM_PORT_PLAN.md`](ROCM_PORT_PLAN.md).
- **Repo-level guidance for agents**: [`CLAUDE.md`](CLAUDE.md).
