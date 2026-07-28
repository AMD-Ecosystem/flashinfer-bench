---
name: debug-rocm
description: Triage HIP crashes, memory faults, NaN/Inf, OOM, FP8 dtype mismatches, and AITER errors that surface when building or benchmarking FlashInfer-Bench solutions on AMD CDNA GPUs. Covers the AMD_SERIALIZE_KERNEL + HIP_LAUNCH_BLOCKING combo, rocgdb, AMD_LOG_LEVEL, dmesg amdgpu faults, and why compute-sanitizer has no ROCm analog. Use when a solution run crashes or produces wrong numbers on ROCm.
---

# Debug ROCm

Fast triage for failures during the bench build → benchmark → apply loop on AMD. Run
[`rocm-setup`](../rocm-setup/SKILL.md) first if the environment itself is suspect (`validate_p0.py`).

## First move: make the crash synchronous and located

HIP kernel launches are async, so the Python traceback usually points at the wrong line. Force
synchronous, serialized execution so the failing kernel is the one reported:

```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
  python -m flashinfer_bench run --local <trace_dir> --definitions <def>
```

`AMD_SERIALIZE_KERNEL=3` waits before *and* after each kernel; `HIP_LAUNCH_BLOCKING=1` makes
launches synchronous. Together they localize memory faults to the offending kernel.

## Turn up logging

| Env | Effect |
|---|---|
| `AMD_LOG_LEVEL=3` (or `4`) | HIP runtime API + kernel logging |
| `HSA_ENABLE_DEBUG=1` | HSA-level debug info |
| `FLASHINFER_JIT_VERBOSE=1` | verbose JIT build (amd-flashinfer) |
| `dmesg | grep -iE "amdgpu|kfd|vm_fault"` | kernel-level GPU faults (page faults, resets) |
| `rocm-smi` | live clocks/power/temp/ECC; confirm the GPU didn't reset |

## Per-symptom recipes

| Symptom | Likely cause → fix |
|---|---|
| **Memory access fault by GPU** (`dmesg` shows `vm_fault`) | OOB indexing / bad pointer in a HIP solution. Re-run with `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`, then step in `rocgdb`. Check index/indptr tensors and grid/block bounds. |
| **NaN / Inf in output** | fast-math (`-ffast-math` re-adds `-fno-finite-math-only`), uninitialized LDS, or a reduction-order difference. Compare against the reference in fp32; disable fast-math to isolate. |
| **HIP OOM** | reduce batch/workload size or free the cache: `rm -rf ~/.cache/flashinfer/` is unrelated (that's JIT); use smaller shapes or fewer parallel workers (test parallelism halves the effective GPU count). |
| **FP8 dtype mismatch** | AMD uses `_fnuz` FP8 (`float8_e4m3fnuz` / `float8_e5m2fnuz`), not NVIDIA `_fn`. Fix the solution's dtypes, not the tolerance. |
| **Correctness fails only on AMD (fp8/bf16)** | rounding/reduction-order difference — tune evaluator tolerances per dtype (see [`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md)), don't loosen blindly. |
| **AITER `ImportError`** | AITER not installed / ROCm-version mismatch → re-run [`rocm-setup`](../rocm-setup/SKILL.md); AITER imports call `git` (ensure git present + `safe.directory '*'`). |
| **AITER `ValueError`/`RuntimeError`** | the op/shape/dtype is outside AITER coverage — the generator should have returned `None`; fall through to Triton/HIP ([`generate-aiter-solution`](../generate-aiter-solution/SKILL.md)). |
| **Build fails with `nvcc`/`libcuda` errors** | a CUDA-only flag leaked into a builder; remove unconditional `-lcuda -lcublas` and let hipcc/tvm-ffi pick HIP libs (see [`author-hip-solution`](../author-hip-solution/SKILL.md)). |
| **Stale build after env change** | JIT `build.ninja` is only rewritten when missing — clear it: `rm -rf ~/.cache/flashinfer/`. |

## Interactive debugging

```bash
# rocgdb is the ROCm gdb; wait-for-debugger lets you attach before the kernel runs
ROCM_DEBUG_WAIT_FOR_DEBUGGER=1 python -m flashinfer_bench run ... &
rocgdb -p <pid>
```

## What ROCm does not have

- **`compute-sanitizer` / `cuda-gdb` have no full ROCm analog.** `memcheck` is partially covered by
  an LLVM AddressSanitizer-instrumented HIP build (`-fsanitize=address`) and `rocgdb`;
  `racecheck` / `synccheck` / `initcheck` are **unsupported** — don't wait for them. This is a known
  gap (`ROCM_PORT_PLAN.md` §3.10, risk #2); the bench sanitizer agent degrades gracefully and returns
  an explicit "unsupported on ROCm" per check.

## Maintaining this document

Update when new AITER error modes appear, when the sanitizer agent (`agents/sanitizer.py`) gains
ROCm coverage (§3.10), or when JIT/cache behavior changes.

## See Also

- [rocm-setup](../rocm-setup/SKILL.md) — validate the environment first
- [benchmark-on-rocm](../benchmark-on-rocm/SKILL.md) — tolerances & timing
- [author-hip-solution](../author-hip-solution/SKILL.md) — build-side issues
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.10 — profiling/debug agents
